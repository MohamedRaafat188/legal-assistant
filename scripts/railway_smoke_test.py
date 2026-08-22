"""Phase 7 post-deploy smoke test: exercises the live Railway URL end to end.

Unlike phase5_validate.py / phase6_validate.py (which drive the app in-process
via httpx.ASGITransport), this hits a real deployed base_url over the network --
it is the actual proof that auth, migrations, CORS, Qdrant Cloud egress, the
Hetzner embedding endpoint, Gemini, Postgres, and Langfuse Cloud are all
reachable *from Railway*, not just from a dev machine. Prints PASS/FAIL per
check and exits non-zero on any failure.

Usage:
    python scripts/railway_smoke_test.py https://<your-app>.up.railway.app

Langfuse trace verification reuses the local LANGFUSE_* credentials from
.env (the same ones set as Railway env vars) to poll Langfuse Cloud directly --
it does not require any access to the Railway deployment itself.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase6_validate import wait_for_trace  # noqa: E402

from legal_assistant import observability  # noqa: E402
from legal_assistant.rag.retrieval import MUKARRAR  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def parse_sse(raw: str) -> list[tuple[str, dict]]:
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


async def post_chat(
    client: httpx.AsyncClient, headers: dict, conversation_id: int, message: str
) -> tuple[int, str]:
    async with client.stream(
        "POST", "/chat", json={"conversation_id": conversation_id, "message": message}, headers=headers
    ) as resp:
        body = "".join([chunk async for chunk in resp.aiter_text()])
        return resp.status_code, body


async def main(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        print(f"\n=== Target: {base_url} ===")

        print("\n=== Health (confirms migrations applied, DB reachable) ===")
        health_resp = await client.get("/health")
        check("health endpoint returns 200", health_resp.status_code == 200, f"status={health_resp.status_code}")
        health = health_resp.json() if health_resp.status_code == 200 else {}
        check("health reports DB up", health.get("db") == "up", str(health))

        print("\n=== Auth: register -> login (JWT) ===")
        suffix = uuid.uuid4().hex[:8]
        username = f"railway_lawyer_{suffix}"
        password = "correct_horse_battery_staple"
        token, user_id = await register(client, username, password)
        check("register returns a JWT + user id", bool(token) and user_id > 0)

        login_resp = await client.post("/auth/login", json={"username": username, "password": password})
        check("login returns 200 with a JWT", login_resp.status_code == 200 and "access_token" in login_resp.json())
        headers = {"Authorization": f"Bearer {token}"}

        no_auth = await client.get("/conversations")
        check("unauthenticated request -> 401", no_auth.status_code == 401)

        print("\n=== Conversation CRUD ===")
        conv_resp = await client.post("/conversations", json={"title": "اختبار Railway"}, headers=headers)
        check("create conversation succeeds", conv_resp.status_code == 201, f"status={conv_resp.status_code}")
        conversation_id = conv_resp.json()["id"]

        print("\n=== Grounded, cited SSE chat (Qdrant Cloud + Hetzner embedding + Gemini) ===")
        status, body = await post_chat(
            client, headers, conversation_id, "ما هو نص المادة 5 من القانون المدني؟"
        )
        check("chat stream returns 200", status == 200, f"status={status}")
        events = parse_sse(body)
        event_names = [e for e, _ in events]
        check("chat produced token events (a grounded answer)", "token" in event_names, str(event_names))
        check("chat produced a citations event", "citations" in event_names, str(event_names))
        check("chat did not error", "error" not in event_names, str(event_names))

        citations_payload = next((d for e, d in events if e == "citations"), {})
        check(
            "citations event carries at least one verified citation",
            len(citations_payload.get("citations", [])) >= 1,
            str(citations_payload),
        )

        done_payload = next((d for e, d in events if e == "done"), {})
        trace_id = done_payload.get("trace_id")
        check("done event carries a trace_id", bool(trace_id), str(done_payload))

        print("\n=== Cross-session memory: second turn in the same conversation ===")
        status2, body2 = await post_chat(client, headers, conversation_id, "وما حكم المادة 6؟")
        events2 = parse_sse(body2)
        check("second turn in same conversation succeeds", status2 == 200 and "error" not in [e for e, _ in events2])

        print("\n=== Law 72/2017 (confirms the deployed code, not just the data, knows it) ===")
        # Law 72's chunks were written straight to Qdrant Cloud, so they go live
        # independently of any deploy. These checks are what distinguish a
        # deployment that actually carries the law-72 code from one still
        # serving the old two-law prompt against the new data.
        conv72 = await client.post(
            "/conversations", json={"title": "اختبار قانون الاستثمار"}, headers=headers
        )
        conv72_id = conv72.json()["id"]

        _, body72 = await post_chat(
            client, headers, conv72_id, "ما نص المادة ١١ مكررًا من قانون الاستثمار؟"
        )
        events72 = parse_sse(body72)
        names72 = [e for e, _ in events72]
        citations72 = next((d for e, d in events72 if e == "citations"), {"citations": []})
        answered = "withdrawn" not in names72 and "error" not in names72
        check(
            "a law-72 مكرر question is answered, not refused as out of scope",
            answered and bool(citations72["citations"]),
            str(names72),
        )
        check(
            "the live citation is marked مكرر, not collapsed onto base article 11",
            any(
                c["article_number"] == 11 and c.get("article_suffix") == MUKARRAR
                for c in citations72["citations"]
            ),
            str(citations72),
        )

        _, body9 = await post_chat(
            client, headers, conv72_id, "ما حكم المادة ٩ من قانون الاستثمار؟"
        )
        answer9 = "".join(d["text"] for e, d in parse_sse(body9) if e == "token")
        check(
            "an amended article's answer names the amending law",
            "١٦٠" in answer9 or "160" in answer9,
            f"answer={answer9[:160]!r}",
        )

        print("\n=== Law 159/1981 (the bis series the suffix widening is for) ===")
        # Same reasoning as law 72: the chunks reach Qdrant Cloud independently
        # of any deploy, so what these checks actually prove is that the running
        # code carries the widened article_suffix. A deployment still on the
        # bis-flag build would answer, but could not tell «١٢٩ مكررًا "٥"» from
        # its nine siblings.
        conv159 = await client.post(
            "/conversations", json={"title": "اختبار قانون الشركات"}, headers=headers
        )
        conv159_id = conv159.json()["id"]

        _, body159 = await post_chat(
            client, headers, conv159_id, 'ما نص المادة ١٢٩ مكررًا "٥" من قانون الشركات؟'
        )
        events159 = parse_sse(body159)
        names159 = [e for e, _ in events159]
        citations159 = next((d for e, d in events159 if e == "citations"), {"citations": []})
        answered159 = "withdrawn" not in names159 and "error" not in names159
        check(
            "a law-159 question is answered, not refused as out of scope",
            answered159 and bool(citations159["citations"]),
            str(names159),
        )
        check(
            "the live citation names the exact sibling, not just «مكرر»",
            any(
                c["article_number"] == 129
                and c.get("article_suffix") not in (None, MUKARRAR)
                and MUKARRAR in c["article_suffix"]
                for c in citations159["citations"]
            ),
            str(citations159),
        )

        _, body_rep = await post_chat(
            client, headers, conv159_id, "ما نص المادة ٢٢ من قانون الشركات؟"
        )
        answer_rep = "".join(d["text"] for e, d in parse_sse(body_rep) if e == "token")
        check(
            "a repealed article is reported as repealed, not served as live law",
            any(w in answer_rep for w in ("ملغا", "ألغيت", "إلغا")),
            f"answer={answer_rep[:160]!r}",
        )

        print("\n=== User isolation ===")
        token_b, _ = await register(client, f"railway_lawyer_b_{suffix}", "another_password_123")
        headers_b = {"Authorization": f"Bearer {token_b}"}
        foreign_get = await client.get(f"/conversations/{conversation_id}", headers=headers_b)
        check("another user cannot read this conversation -> 404", foreign_get.status_code == 404)
        foreign_chat = await client.post(
            "/chat", json={"conversation_id": conversation_id, "message": "hi"}, headers=headers_b
        )
        check("another user cannot post to this conversation -> 404", foreign_chat.status_code == 404)

        print("\n=== Feedback ===")
        if trace_id:
            fb_resp = await client.post(
                "/feedback",
                json={"trace_id": trace_id, "rating": 1, "comment": "دقيق"},
                headers=headers,
            )
            check("feedback on own trace succeeds", fb_resp.status_code == 200, f"status={fb_resp.status_code}")

            foreign_fb = await client.post(
                "/feedback", json={"trace_id": trace_id, "rating": 0}, headers=headers_b
            )
            check("feedback on another user's trace -> 404 (isolation)", foreign_fb.status_code == 404)

        print("\n=== Langfuse Cloud: trace ingested with guard-verdict + feedback scores ===")
        lf_client = observability.get_client()
        if lf_client is not None and trace_id:
            lf_client.flush()
            trace = await wait_for_trace(lf_client, trace_id, min_scores=5, timeout_s=40.0)
            check("trace is retrievable from Langfuse Cloud", trace is not None, f"trace_id={trace_id}")
            if trace is not None:
                scores = {s.name: s.value for s in trace.scores}
                check(
                    "guard-verdict score present (hallucinated_citations_count == 0)",
                    scores.get("hallucinated_citations_count") == 0,
                    str(scores),
                )
                check("user_feedback score landed on the trace", scores.get("user_feedback") == 1, str(scores))
        else:
            print("  (skipped: Langfuse not configured locally, or no trace_id)")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/railway_smoke_test.py https://<your-app>.up.railway.app")
        sys.exit(2)
    asyncio.run(main(sys.argv[1].rstrip("/")))
