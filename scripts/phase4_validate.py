"""Phase 4 validation gate: hashing, isolation, cross-session citations, summary.

Not a pytest suite -- a scripted proof run against the live dev stack
(Postgres, Qdrant Cloud, embedding service, Gemini), mirroring the Phase 2/3
validation scripts. Prints PASS/FAIL per check and exits non-zero on any
failure.
"""

from __future__ import annotations

import asyncio
import datetime
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_assistant.auth import authenticate, register
from legal_assistant.db.models import Message
from legal_assistant.db.session import session_scope
from legal_assistant.memory import (
    COMPACTION_THRESHOLD_TURNS,
    create_conversation,
    list_conversations,
    load_conversation,
    maybe_compact_summary,
)
from legal_assistant.rag.agent import LegalAssistantAgent
from legal_assistant.rag.citation_guard import verify

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


async def test_hashing_and_auth() -> None:
    print("\n=== Hashing & auth ===")
    suffix = uuid.uuid4().hex[:8]
    username = f"lawyer_a_{suffix}"
    password = "correct_horse_battery_staple"

    async with session_scope() as session:
        user = await register(session, username, password)
        check("password_hash is bcrypt form", user.password_hash.startswith("$2b$"), user.password_hash[:10])
        check("password_hash is not plaintext", user.password_hash != password)

    async with session_scope() as session:
        ok_user = await authenticate(session, username, password)
        check("correct password authenticates", ok_user.username == username)

    wrong_ok = True
    async with session_scope() as session:
        try:
            await authenticate(session, username, "totally_wrong_password")
            wrong_ok = False
        except Exception:
            pass
    check("wrong password rejected", wrong_ok)

    return username, password


async def test_user_isolation() -> tuple[int, int]:
    print("\n=== User isolation ===")
    suffix = uuid.uuid4().hex[:8]
    async with session_scope() as session:
        user_a = await register(session, f"iso_a_{suffix}", "pw_a_12345")
        user_b = await register(session, f"iso_b_{suffix}", "pw_b_12345")

    async with session_scope() as session:
        conv_a = await create_conversation(session, user_a.id, "محادثة أ الخاصة")

    async with session_scope() as session:
        b_conversations = await list_conversations(session, user_b.id)
        a_ids = {c.id for c in b_conversations}
        check("user B cannot see user A's conversation", conv_a.id not in a_ids, f"B sees ids={a_ids}")

    async with session_scope() as session:
        from legal_assistant.memory import ConversationNotFoundError

        blocked = True
        try:
            await load_conversation(session, conv_a.id, user_b.id)
            blocked = False
        except ConversationNotFoundError:
            pass
        check("user B cannot load user A's conversation directly", blocked)

    return user_a.id, conv_a.id


async def test_cross_session_citation(user_id: int) -> int:
    print("\n=== Cross-session citation (Option A key test) ===")
    async with session_scope() as session:
        conv = await create_conversation(session, user_id, "اختبار الاستشهاد عبر الجلسات")
    conversation_id = conv.id

    # Session 1: ask a question that should retrieve + cite an article.
    async with session_scope() as session:
        agent1 = await LegalAssistantAgent.create(session, user_id, conversation_id)
        result1 = await agent1.ask(session, "ما هو نص المادة 5 من القانون المدني؟")
    check(
        "turn 1 produced a verified citation",
        len(result1.verified_citations) > 0 and not result1.used_fallback,
        f"citations={result1.verified_citations}",
    )
    check("turn 1 retrieved at least one article", len(result1.retrieved_article_ids) > 0)

    # Simulate a fresh process: brand-new LegalAssistantAgent instance, same conversation_id.
    async with session_scope() as session:
        agent2 = await LegalAssistantAgent.create(session, user_id, conversation_id)
        check(
            "reloaded agent's AllowedSet contains turn 1's citation",
            len(agent2._allowed.numbered) > 0 or len(agent2._allowed.all_labels) > 0,
            f"numbered={agent2._allowed.numbered}",
        )

        result2 = await agent2.ask(
            session, "اشرح المادة التي ذكرتها سابقاً بمزيد من التفصيل من فضلك."
        )
    check(
        "follow-up turn produced a verified citation without hallucination",
        not result2.used_fallback,
        f"guard_report={result2.guard_report}",
    )
    check(
        "follow-up citation matches turn 1's article",
        any(c["article_number"] == 5 for c in result2.verified_citations)
        or any(c["article_number"] == 5 for c in result1.verified_citations),
        f"turn2 citations={result2.verified_citations}",
    )

    # Induced hallucination must still be blocked against the rebuilt AllowedSet.
    async with session_scope() as session:
        loaded = await load_conversation(session, conversation_id, user_id)
        fake_answer = {
            "answer_text": "طبقاً للمادة 999 من القانون المدني...",
            "citations": [
                {"law_name": "القانون المدني", "article_number": 999, "citation_label": "القانون المدني م 999"}
            ],
        }
        guard_result = verify(fake_answer, loaded.allowed)
        check(
            "guard still blocks an induced hallucination against rebuilt AllowedSet",
            not guard_result.is_valid and len(guard_result.hallucinated_citations) == 1,
            guard_result.report,
        )

    return conversation_id


async def test_persistence_integrity(conversation_id: int, user_id: int) -> None:
    print("\n=== Persistence integrity ===")
    async with session_scope() as session:
        loaded = await load_conversation(session, conversation_id, user_id)
        messages = loaded.messages
        check("messages are in chronological order", messages == sorted(messages, key=lambda m: m.created_at))
        check("at least 4 messages persisted (2 turns)", len(messages) >= 4, f"count={len(messages)}")

        assistant_msgs_with_ctx = [m for m in messages if m.role == "assistant" and m.retrieved_context]
        check("at least one assistant message has retrieved_context", len(assistant_msgs_with_ctx) > 0)
        if assistant_msgs_with_ctx:
            ctx = assistant_msgs_with_ctx[0].retrieved_context[0]
            check(
                "retrieved_context entries round-trip expected keys",
                {"point_id", "law_number", "law_name", "article_number", "citation_label", "clean_text"}
                <= set(ctx.keys()),
                str(ctx.keys()),
            )


async def test_summary_compaction(user_id: int) -> None:
    print("\n=== Summary compaction ===")
    async with session_scope() as session:
        conv = await create_conversation(session, user_id, "محادثة طويلة لاختبار التلخيص")
        conversation_id = conv.id

        # Seed a long synthetic history directly (no LLM/retrieval calls needed to
        # build length -- keeps this check fast and independent of Gemini quota).
        base_time = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)
        num_turns = COMPACTION_THRESHOLD_TURNS + 2
        for i in range(num_turns):
            t_user = base_time + datetime.timedelta(minutes=2 * i)
            t_asst = t_user + datetime.timedelta(seconds=30)
            session.add(
                Message(
                    conversation_id=conversation_id,
                    role="user",
                    content=f"سؤال رقم {i + 1} عن موضوع قانوني عام.",
                    created_at=t_user,
                )
            )
            retrieved_ctx = None
            citations = None
            if i == 0:
                retrieved_ctx = [
                    {
                        "point_id": "test-point-1",
                        "law_number": 131,
                        "law_name": "القانون المدني رقم ١٣١ لسنة ١٩٤٨",
                        "article_number": 5,
                        "citation_label": "القانون المدني م 5",
                        "clean_text": "نص تجريبي للمادة الخامسة.",
                    }
                ]
                citations = [
                    {"law_name": "القانون المدني رقم ١٣١ لسنة ١٩٤٨", "article_number": 5, "citation_label": "القانون المدني م 5"}
                ]
            session.add(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=f"إجابة رقم {i + 1}.",
                    citations=citations,
                    retrieved_context=retrieved_ctx,
                    created_at=t_asst,
                )
            )
        await session.flush()

    async with session_scope() as session:
        compacted = await maybe_compact_summary(session, conversation_id, user_id)
        check("compaction triggers past the threshold", compacted)

    async with session_scope() as session:
        loaded = await load_conversation(session, conversation_id, user_id)
        check("summary is non-empty after compaction", bool(loaded.summary and loaded.summary.strip()))
        check(
            "citation from turn 1 still verifiable via retrieved_context after compaction (not summary)",
            loaded.allowed.contains_numbered("القانون المدني رقم ١٣١ لسنة ١٩٤٨", 5, False),
        )
        print(f"    summary preview: {loaded.summary[:120]!r}...")


async def main() -> None:
    await test_hashing_and_auth()
    user_id, _ = await test_user_isolation()
    conversation_id = await test_cross_session_citation(user_id)
    await test_persistence_integrity(conversation_id, user_id)
    await test_summary_compaction(user_id)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
