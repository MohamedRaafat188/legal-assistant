"""Phase 6 validation gate: Langfuse tracing, guard-verdict scores, feedback, outage resilience.

Reuses phase5_validate's harness (httpx.ASGITransport, single event loop --
Langfuse's batched async export doesn't tolerate a fresh-event-loop-per-call
client any better than asyncpg's connection pool does). Drives the live app
against the real DB/Qdrant/embedding/Gemini/Langfuse Cloud stack. Prints
PASS/FAIL per check and exits non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
from phase5_validate import FAILURES, check, parse_sse, post_chat, register

from legal_assistant import observability
from legal_assistant.api import app as app_module


async def wait_for_trace(client, trace_id: str, min_scores: int = 1, timeout_s: float = 40.0):
    """Poll Langfuse Cloud for a trace's ingested observations/scores (async export)."""
    deadline = time.monotonic() + timeout_s
    last_exc = None
    trace = None
    while time.monotonic() < deadline:
        try:
            trace = client.api.trace.get(trace_id)
            if len(trace.scores) >= min_scores:
                return trace
        except Exception as e:  # noqa: BLE001 -- trace may not be ingested by Cloud yet
            last_exc = e
        await asyncio.sleep(1.0)
    if last_exc:
        print(f"  (last fetch error while polling: {last_exc})")
    return trace


async def main() -> None:
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        suffix = uuid.uuid4().hex[:8]
        token, user_id = await register(client, f"lf_lawyer_{suffix}", "correct_horse_battery_staple")
        headers = {"Authorization": f"Bearer {token}"}

        conv_resp = await client.post(
            "/conversations", json={"title": "اختبار Langfuse"}, headers=headers
        )
        check("create conversation succeeds", conv_resp.status_code == 201)
        conversation_id = conv_resp.json()["id"]

        lf_client = observability.get_client()
        check("Langfuse client initialized from .env credentials", lf_client is not None)

        print("\n=== Normal chat turn: trace + spans + metadata + guard-verdict scores ===")
        status, body = await post_chat(
            client, headers, conversation_id, "ما هو نص المادة 5 من القانون المدني؟"
        )
        check("chat stream returns 200", status == 200, f"status={status}")
        events = parse_sse(body)
        done_payload = next((d for e, d in events if e == "done"), {})
        trace_id = done_payload.get("trace_id")
        check("done event carries a trace_id", bool(trace_id), str(done_payload))

        if lf_client is not None and trace_id:
            lf_client.flush()
            trace = await wait_for_trace(lf_client, trace_id, min_scores=4)
            check("trace is retrievable from Langfuse Cloud", trace is not None, f"trace_id={trace_id}")

            if trace is not None:
                obs_names = {o.name for o in trace.observations}
                check("trace has a chat_turn span", "chat_turn" in obs_names, str(obs_names))
                check("trace has a citation_guard span", "citation_guard" in obs_names, str(obs_names))
                check(
                    "trace has a retrieval span (search_articles or get_article_by_number)",
                    bool({"search_articles", "get_article_by_number"} & obs_names),
                    str(obs_names),
                )

                chat_turn_obs = next((o for o in trace.observations if o.name == "chat_turn"), None)
                turn_meta = (chat_turn_obs.metadata or {}) if chat_turn_obs else {}
                check(
                    "chat_turn span metadata carries conversation_id and user_id",
                    turn_meta.get("conversation_id") == conversation_id
                    and turn_meta.get("user_id") == user_id,
                    str(turn_meta),
                )

                retrieval_obs = next(
                    (o for o in trace.observations if o.name == "search_articles"), None
                )
                if retrieval_obs is not None:
                    check(
                        "search_articles span metadata carries rerank_latency_ms",
                        "rerank_latency_ms" in (retrieval_obs.metadata or {}),
                        str(retrieval_obs.metadata),
                    )

                scores = {s.name: s.value for s in trace.scores}
                check(
                    "guard-verdict score: hallucinated_citations_count == 0 on a clean turn",
                    scores.get("hallucinated_citations_count") == 0,
                    str(scores),
                )
                check(
                    "guard-verdict score: citations_verified_count >= 1 on a clean turn",
                    (scores.get("citations_verified_count") or 0) >= 1,
                    str(scores),
                )
                check(
                    "guard-verdict score: used_fallback == 0 on a clean turn",
                    scores.get("used_fallback") == 0,
                    str(scores),
                )

        print("\n=== Guard-verdict scoring: a forced-fallback verdict scores used_fallback == 1 ===")
        fallback_trace_id = observability.new_trace_id()
        with observability.start_span(
            name="chat_turn",
            trace_id=fallback_trace_id,
            as_type="agent",
            metadata={"simulated_fallback": True},
        ) as span:
            observability.safe_update(span, output={"simulated": True})
        fake_guard = SimpleNamespace(
            hallucinated_citations=[object()], valid_citations=[], unverified_inline_refs=[]
        )
        observability.score_guard_verdict(fallback_trace_id, fake_guard, used_fallback=True)

        if lf_client is not None:
            lf_client.flush()
            fb_trace = await wait_for_trace(lf_client, fallback_trace_id, min_scores=4)
            check("simulated fallback trace is retrievable", fb_trace is not None)
            if fb_trace is not None:
                fb_scores = {s.name: s.value for s in fb_trace.scores}
                check(
                    "used_fallback score == 1 for a simulated fallback verdict",
                    fb_scores.get("used_fallback") == 1,
                    str(fb_scores),
                )
                check(
                    "hallucinated_citations_count == 1 for a simulated fallback verdict",
                    fb_scores.get("hallucinated_citations_count") == 1,
                    str(fb_scores),
                )

        print("\n=== POST /feedback: user-scoped scoring ===")
        fb_resp = await client.post(
            "/feedback",
            json={"trace_id": trace_id, "rating": 1, "comment": "إجابة دقيقة وموثقة"},
            headers=headers,
        )
        check("feedback on own trace succeeds", fb_resp.status_code == 200, f"status={fb_resp.status_code}")

        bad_resp = await client.post(
            "/feedback", json={"trace_id": "not-a-real-trace-id", "rating": 1}, headers=headers
        )
        check("feedback on unknown trace_id -> 404", bad_resp.status_code == 404)

        token_b, _ = await register(client, f"lf_lawyer_b_{suffix}", "another_password_123")
        headers_b = {"Authorization": f"Bearer {token_b}"}
        foreign_resp = await client.post(
            "/feedback", json={"trace_id": trace_id, "rating": 0}, headers=headers_b
        )
        check("feedback on another user's trace -> 404 (isolation)", foreign_resp.status_code == 404)

        if lf_client is not None and trace_id:
            lf_client.flush()
            trace_after_feedback = await wait_for_trace(lf_client, trace_id, min_scores=5)
            if trace_after_feedback is not None:
                scores_after = {s.name: s.value for s in trace_after_feedback.scores}
                check(
                    "user_feedback score lands on the correct (referenced) trace",
                    scores_after.get("user_feedback") == 1,
                    str(scores_after),
                )

        print("\n=== Langfuse outage resilience: chat survives a broken/unreachable client ===")
        original_client = observability._client
        original_attempted = observability._client_init_attempted

        class _BrokenLangfuseClient:
            def __getattr__(self, name):  # noqa: ANN001, ANN204 -- test double
                def _raise(*args, **kwargs):
                    raise RuntimeError("simulated Langfuse outage")

                return _raise

        observability._client = _BrokenLangfuseClient()
        observability._client_init_attempted = True
        try:
            status_outage, body_outage = await post_chat(
                client, headers, conversation_id, "ما هو نص المادة 6 من القانون المدني؟"
            )
            outage_events = parse_sse(body_outage)
            outage_names = [e for e, _ in outage_events]
            check(
                "chat still succeeds end-to-end when Langfuse is broken",
                status_outage == 200 and "error" not in outage_names and "done" in outage_names,
                f"status={status_outage} events={outage_names}",
            )
        finally:
            observability._client = original_client
            observability._client_init_attempted = original_attempted

        print("\n=== Feedback also survives a broken/unreachable Langfuse client ===")
        observability._client = _BrokenLangfuseClient()
        observability._client_init_attempted = True
        try:
            fb_outage_resp = await client.post(
                "/feedback", json={"trace_id": trace_id, "rating": 1}, headers=headers
            )
            check(
                "feedback endpoint doesn't 5xx when Langfuse scoring is broken",
                fb_outage_resp.status_code == 200,
                f"status={fb_outage_resp.status_code}",
            )
        finally:
            observability._client = original_client
            observability._client_init_attempted = original_attempted

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
