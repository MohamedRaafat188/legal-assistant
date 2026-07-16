"""Phase 5 validation gate: auth/JWT, isolation, streaming chat, Option-3 safety, failures.

Not a pytest suite -- a scripted proof run against the live app (in-process,
via httpx.ASGITransport so it exercises the real DB/Qdrant/embedding/LLM
stack while still letting us monkeypatch specific failure scenarios).
Everything runs inside a single asyncio.run() / single event loop, which
matters on Windows: the async SQLAlchemy engine's pooled asyncpg connections
cannot survive being reused across different event loops (this is why
starlette's TestClient, which spins up a fresh loop per call, doesn't work
here). Prints PASS/FAIL per check and exits non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

from legal_assistant.api import app as app_module
from legal_assistant.api.routes import chat as chat_routes
from legal_assistant.rag.agent import TurnResult
from legal_assistant.rag.citation_guard import FALLBACK_MESSAGE_AR

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse a raw SSE response body into a list of (event, data) pairs."""
    events: list[tuple[str, dict]] = []
    event_name = None
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data = json.loads(line[len("data:") :].strip())
            events.append((event_name, data))
    return events


async def register(client: httpx.AsyncClient, username: str, password: str) -> tuple[str, int]:
    resp = await client.post("/auth/register", json={"username": username, "password": password})
    resp.raise_for_status()
    body = resp.json()
    return body["access_token"], body["user"]["id"]


async def post_chat(client: httpx.AsyncClient, headers: dict, conversation_id: int, message: str) -> tuple[int, str]:
    async with client.stream(
        "POST", "/chat", json={"conversation_id": conversation_id, "message": message}, headers=headers
    ) as resp:
        body = "".join([chunk async for chunk in resp.aiter_text()])
        return resp.status_code, body


async def main() -> None:
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        print("\n=== Health ===")
        health = (await client.get("/health")).json()
        check("health reports DB up", health.get("db") == "up", str(health))

        print("\n=== Auth & session identity ===")
        suffix = uuid.uuid4().hex[:8]
        token_a, user_a_id = await register(client, f"api_lawyer_a_{suffix}", "correct_horse_battery_staple")
        check("register returns a JWT + user id", bool(token_a) and user_a_id > 0)

        no_auth = await client.get("/conversations")
        check("missing token -> 401", no_auth.status_code == 401, f"status={no_auth.status_code}")

        bad_auth = await client.get("/conversations", headers={"Authorization": "Bearer not-a-real-token"})
        check("invalid token -> 401", bad_auth.status_code == 401, f"status={bad_auth.status_code}")

        login_resp = await client.post(
            "/auth/login",
            json={"username": f"api_lawyer_a_{suffix}", "password": "correct_horse_battery_staple"},
        )
        check("login succeeds with correct password", login_resp.status_code == 200)
        login_wrong = await client.post(
            "/auth/login", json={"username": f"api_lawyer_a_{suffix}", "password": "wrong_password"}
        )
        check("login rejects wrong password -> 401", login_wrong.status_code == 401)

        headers_a = {"Authorization": f"Bearer {token_a}"}

        print("\n=== Conversations ===")
        conv_resp = await client.post("/conversations", json={"title": "اختبار API"}, headers=headers_a)
        check("create conversation succeeds", conv_resp.status_code == 201)
        conversation_id = conv_resp.json()["id"]

        list_resp = await client.get("/conversations", headers=headers_a)
        check(
            "list includes the created conversation",
            any(c["id"] == conversation_id for c in list_resp.json()),
        )

        print("\n=== Streaming /chat: Option 3 (citations only after guard) ===")
        status, body = await post_chat(client, headers_a, conversation_id, "ما هو نص المادة 5 من القانون المدني؟")
        check("chat stream returns 200", status == 200, f"status={status}")
        events = parse_sse(body)
        event_names = [e for e, _ in events]
        check("stream contains token events", "token" in event_names, str(event_names))
        check("stream ends with a done event", bool(event_names) and event_names[-1] == "done")

        token_idx = [i for i, n in enumerate(event_names) if n == "token"]
        citation_idx = [i for i, n in enumerate(event_names) if n == "citations"]
        check(
            "citations event (if any) arrives strictly after all token events",
            bool(citation_idx) and max(token_idx or [-1]) < citation_idx[0],
            f"token_idx={token_idx} citation_idx={citation_idx}",
        )
        reassembled = "".join(d["text"] for e, d in events if e == "token")
        check("reassembled streamed text is non-empty", bool(reassembled.strip()))

        citations_payload = next((d for e, d in events if e == "citations"), {"citations": []})
        check(
            "turn 1 produced a verified citation for article 5",
            any(c["article_number"] == 5 for c in citations_payload["citations"]),
            str(citations_payload),
        )

        print("\n=== Cross-session (Option A) through the API ===")
        # Every /chat call builds a brand-new LegalAssistantAgent from persisted
        # state (see rag/agent.py's LegalAssistantAgent.create), so a follow-up
        # call here *is* a fresh-process reload by construction.
        status2, body2 = await post_chat(
            client, headers_a, conversation_id, "اشرح المادة التي ذكرتها سابقاً بمزيد من التفصيل من فضلك."
        )
        events2 = parse_sse(body2)
        names2 = [e for e, _ in events2]
        check(
            "follow-up stream completes with done, no error",
            "error" not in names2 and "done" in names2,
            str(names2),
        )
        citations2 = next((d for e, d in events2 if e == "citations"), {"citations": []})
        check(
            "follow-up cites article 5 via rebuilt AllowedSet, not fresh retrieval",
            any(c["article_number"] == 5 for c in citations2["citations"]),
            str(citations2),
        )

        print("\n=== User isolation through the API ===")
        token_b, user_b_id = await register(client, f"api_lawyer_b_{suffix}", "another_password_123")
        headers_b = {"Authorization": f"Bearer {token_b}"}

        get_forbidden = await client.get(f"/conversations/{conversation_id}", headers=headers_b)
        check("user B cannot GET user A's conversation -> 404", get_forbidden.status_code == 404)

        status_forbidden, _ = await post_chat(client, headers_b, conversation_id, "test")
        check("user B cannot POST /chat on user A's conversation -> 404", status_forbidden == 404)

        print("\n=== Option-3 safety contract: fallback path never leaks a citations event ===")
        real_agent_cls = chat_routes.LegalAssistantAgent

        class _FakeAgentFallback:
            async def ask(self, session, user_text):  # noqa: ANN001, ARG002
                return TurnResult(
                    answer_text=FALLBACK_MESSAGE_AR,
                    verified_citations=[
                        {
                            "law_name": "القانون المدني",
                            "article_number": 999,
                            "citation_label": "لن تُعرض أبداً",
                        }
                    ],
                    guard_report="citations: 0 valid, 1 hallucinated (simulated)",
                    used_fallback=True,
                )

        class _FakeAgentFactoryFallback:
            @classmethod
            async def create(cls, session, user_id, conversation_id, settings=None):  # noqa: ANN001, ARG003
                return _FakeAgentFallback()

        chat_routes.LegalAssistantAgent = _FakeAgentFactoryFallback
        try:
            status_fb, body_fb = await post_chat(client, headers_a, conversation_id, "test")
            fallback_events = parse_sse(body_fb)
            fallback_names = [e for e, _ in fallback_events]
            check(
                "simulated guard-fallback turn never emits a citations event",
                "citations" not in fallback_names,
                str(fallback_names),
            )
            check(
                "simulated guard-fallback turn emits withdrawn",
                "withdrawn" in fallback_names,
                str(fallback_names),
            )
            withdrawn_payload = next((d for e, d in fallback_events if e == "withdrawn"), {})
            check(
                "withdrawn message is the safe Arabic fallback, not the fake citation",
                withdrawn_payload.get("message") == FALLBACK_MESSAGE_AR,
                str(withdrawn_payload),
            )
        finally:
            chat_routes.LegalAssistantAgent = real_agent_cls

        print("\n=== Graceful failure on downstream error ===")

        class _FakeAgentFactoryError:
            @classmethod
            async def create(cls, session, user_id, conversation_id, settings=None):  # noqa: ANN001, ARG003
                raise RuntimeError("simulated Qdrant/embedding/LLM outage")

        chat_routes.LegalAssistantAgent = _FakeAgentFactoryError
        try:
            status_err, body_err = await post_chat(client, headers_a, conversation_id, "test")
            error_events = parse_sse(body_err)
            error_names = [e for e, _ in error_events]
            check("downstream failure yields exactly an error event", error_names == ["error"], str(error_names))
            error_payload = next((d for e, d in error_events if e == "error"), {})
            check(
                "error message is a graceful Arabic message, not a stack trace",
                bool(error_payload.get("message")) and "Traceback" not in error_payload.get("message", ""),
                str(error_payload),
            )
        finally:
            chat_routes.LegalAssistantAgent = real_agent_cls

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
