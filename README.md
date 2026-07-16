# Legal Assistant

Arabic-language legal research assistant for Egyptian lawyers. Answers legal
questions with article-level citations that must be verifiable against the
source law text — faithfulness to the source is the top priority.

This repository is being built in phases. **Phase 1** covers project
scaffolding and a one-time migration of already-validated law vectors from a
local Docker Qdrant instance to Qdrant Cloud, without re-embedding.

## Project layout

```
src/legal_assistant/
    config.py      # pydantic-settings Settings
    db/qdrant.py   # source + cloud Qdrant client factories
scripts/
    migrate_to_cloud.py   # one-time vector migration (source -> cloud)
    verify_migration.py   # post-migration validation report
tests/
```

## Setup

1. Create/activate the conda environment (`agents_env`) with Python 3.11+.
2. Install the project in editable mode:

   ```
   pip install -e .
   ```

3. Copy `.env.example` to `.env` and fill in the real values:

   ```
   QDRANT_SOURCE_URL=http://localhost:6333
   QDRANT_SOURCE_API_KEY=
   QDRANT_CLOUD_URL=https://<your-cluster>.cloud.qdrant.io
   QDRANT_CLOUD_API_KEY=<your-cloud-api-key>
   QDRANT_COLLECTION_NAME=egyptian_law
   QDRANT_TIMEOUT=60
   ```

   `.env` is git-ignored; only `.env.example` is committed.

## Running the migration

The source Qdrant (local Docker, server mode) must be running and reachable
at `QDRANT_SOURCE_URL`. The source is only ever read from — never written to
or deleted.

```
python scripts/migrate_to_cloud.py
```

- Mirrors the source collection's exact schema (named dense + sparse
  vectors, HNSW config, `on_disk_payload`, quantization) when creating the
  Cloud collection.
- If the Cloud collection already exists with a matching config, creation is
  skipped. Pass `--recreate` to delete and recreate it (destructive; use
  with care).
- Copies every point (vectors + payload verbatim) in batches of
  128-256 with progress logging. Safe to re-run.

## Validating the migration

```
python scripts/verify_migration.py
```

Prints a PASS/FAIL report checking point counts, sampled payloads, and
sampled vector equality between source and Cloud. Exits non-zero on any
failure — do not treat the migration as complete until this passes.

## Phase 5: FastAPI application

Auth, conversations, and streaming chat over the Phase 1-4 core (retrieval,
tool-calling agent, citation guard, Postgres persistence). Langfuse
observability and the frontend are not built yet.

### Running the API

Local Postgres (`docker compose up -d postgres`), Qdrant Cloud, the
embedding service, and Gemini must all be reachable per your `.env`.

```
uvicorn legal_assistant.api.app:app --app-dir src --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`; liveness check at
`GET /health`.

### Endpoints

- `POST /auth/register`, `POST /auth/login` — bcrypt-hashed password auth;
  both return `{access_token, user}`. Send the token as
  `Authorization: Bearer <token>` on every other request. Minimal auth: no
  refresh, reset, or verification.
- `POST /conversations`, `GET /conversations`, `GET /conversations/{id}` —
  scoped to the current user; a foreign or nonexistent `id` 404s.
- `POST /chat` — `{conversation_id, message}`, streamed back over
  Server-Sent Events.
- `POST /feedback` — `{trace_id, rating, comment?}`, scoped to the current
  user; see Phase 6 below.

### `/chat` SSE event protocol (Option 3: prose streams, citations don't)

The citation guard needs the *complete* generated answer to verify it, so
citations are never sent until verification has already happened. Each SSE
event is one `event: <name>` line followed by one JSON `data:` line:

| event        | payload                                     | meaning |
|--------------|----------------------------------------------|---------|
| `token`      | `{"text": "<prose chunk>"}`                  | repeated; streamed prose from the already-guard-verified answer |
| `citations`  | `{"citations": [{"law_name", "article_number", "citation_label"}, ...]}` | sent once, after all `token` events — the only citations the client may ever render |
| `withdrawn`  | `{"message": "<Arabic fallback>"}`           | the guard hard-failed even after one regeneration; no prose was streamed this turn |
| `done`       | `{"conversation_id": <id>, "trace_id": <str\|null>}` | the turn is persisted; safe to finalize the UI. `trace_id` (Langfuse) references this turn for `POST /feedback`; `null` if tracing is disabled |
| `error`      | `{"message": "<Arabic user-safe error>"}`     | a downstream failure (Qdrant, embedding service, Gemini, DB); nothing is persisted |

**Client contract:** never render a citation before its `citations` event
arrives; on `withdrawn`, there is no valid answer for that turn.

Implementation note: this streams via **chunked delivery of an
already-verified answer**, not live token-by-token generation — the agent's
structured `{answer_text, citations}` output and the citation guard both
need the finished answer, so there's no partial-answer state worth showing
early. A turn runs to completion (unchanged `LegalAssistantAgent.ask`,
already guard-verified and already persisted) and *then* its `answer_text`
is streamed in small chunks for a responsive typing effect. This means no
unverified prose is ever sent, and a client disconnect mid-stream loses
nothing, since the turn was already fully persisted before the first chunk
went out.

### Validating

```
python scripts/phase5_validate.py
```

Runs in-process (via `httpx.ASGITransport`, one event loop) against the real
DB/Qdrant/embedding/Gemini stack, and exercises: register/login, 401 on
missing/invalid tokens, conversation CRUD ownership, live streaming chat
with the citations-after-guard ordering, cross-session citation reuse
through the API, user isolation on both `GET /conversations/{id}` and
`POST /chat`, the Option-3 safety contract (a simulated guard-fallback
never leaks a `citations` event), and a simulated downstream outage
producing a graceful Arabic `error` event instead of a crash. Exits
non-zero on any failure.

## Phase 6: Langfuse observability & feedback

Adds Langfuse Cloud tracing over the Phase 1-5 core: every `/chat` request
becomes one trace with spans for the agent turn, retrieval (embedding +
Qdrant + rerank), and the citation guard; the guard's verdict is logged as
scores on every trace (an automated hallucination monitor, no human
required); and `POST /feedback` lets a lawyer attach a thumbs-up/down to the
exact turn they rated. Reuses Phases 1-5 unchanged in behavior -- tracing is
a wrapper: best-effort and non-blocking, so a Langfuse outage or
misconfiguration never affects chat or feedback.

### Config

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and (optionally)
`LANGFUSE_BASE_URL` (default `https://cloud.langfuse.com`) in `.env`. Leave
the keys blank to run with tracing disabled -- the app behaves identically
either way, just without traces/scores.

### Mental model

- **Trace** = one `/chat` request, end to end.
- **Spans** inside it = the meaningful steps: `chat_turn` (the agent's
  tool-choice + generation, via a LangChain/LangGraph callback handler),
  `search_articles` / `get_article_by_number` (retrieval, with
  `rerank_latency_ms` in metadata -- the known slow step, made visible on
  every trace), and `citation_guard`.
- **Scores** attached to a trace = judgments: the guard's own verdict
  (`hallucinated_citations_count`, `citations_verified_count`,
  `inline_unverified_count`, `used_fallback` -- logged automatically on
  every turn) and a lawyer's `user_feedback` (via `POST /feedback`).

### `POST /feedback`

Authenticated, user-scoped: `{trace_id, rating, comment?}` where `rating` is
1 (thumbs-up) or 0 (thumbs-down). Validates that `trace_id` belongs to a
turn in a conversation owned by the caller (a `trace_id` column on the
assistant `Message` row makes this a plain ownership-joined lookup, the same
isolation pattern as everywhere else in the API) before writing a
`user_feedback` score to Langfuse. A bad or foreign `trace_id` -> 404; a
Langfuse-side failure while scoring never turns into a 5xx.

The `/chat` `done` SSE event now carries `trace_id` (`null` if tracing is
disabled) so the client can reference the turn from `/feedback`.

### Validating

```
python scripts/phase6_validate.py
```

Runs against the real DB/Qdrant/embedding/Gemini/Langfuse Cloud stack (same
`httpx.ASGITransport`, single-event-loop approach as Phase 5 -- Langfuse's
batched async export doesn't tolerate a fresh-event-loop-per-call client any
better than asyncpg's pool does). Exercises: a normal chat turn produces a
trace with the expected spans/metadata and a clean guard-verdict score set;
a forced-fallback verdict scores `used_fallback == 1`; `POST /feedback`
attaches `user_feedback` to the right trace and is rejected for an unknown
or foreign `trace_id`; and, with the Langfuse client forced into a broken
state, both `/chat` and `/feedback` still complete correctly end-to-end.
Exits non-zero on any failure.

## Phase 7: Deployment to Railway

Packaging and wiring only -- no changes to the verified Phase 1-6 behavior.
Postgres runs as Railway's managed plugin; the app auto-deploys from this
repo's `main` branch; the URL is Railway's default `*.up.railway.app`
(no custom domain yet).

### What changed for deployment

- The Windows SSL cert-store workaround in `legal_assistant/__init__.py` is
  now gated on `sys.platform == "win32"` -- a no-op on Railway's Linux
  containers.
- `Settings.database_url` (`config.py`) normalizes a `postgres://` or
  `postgresql://` scheme to `postgresql+asyncpg://` automatically, so
  Railway's managed-Postgres connection string works unmodified.
- `railway.json` defines the build (Nixpacks, auto-detects `pyproject.toml`)
  and the start command: `alembic upgrade head && uvicorn ... --host 0.0.0.0
  --port $PORT`. Migrations run as part of every deploy and their failure
  fails the deploy -- the app never silently starts against a stale schema.
  `/health` is the Railway healthcheck path.
- All dependencies in `pyproject.toml` are pinned to exact versions (the
  versions validated throughout Phases 1-6) for reproducible builds.

### Railway dashboard setup (operator steps -- not automatable from here)

1. **New Railway project** -> **Deploy from GitHub repo** -> select this
   repo. Enable auto-deploy on push to `main`.
2. **Add the Postgres plugin** to the project. Railway provisions it and
   exposes a reference variable (e.g. `${{Postgres.DATABASE_URL}}`) you can
   wire directly into the app service's `DATABASE_URL` -- no manual
   connection-string copying, and it stays in sync if Railway ever rotates
   it.
3. **Set every other environment variable** on the app service (Settings ->
   Variables), matching `.env.example`'s names exactly:
   `QDRANT_CLOUD_URL`, `QDRANT_CLOUD_API_KEY`, `QDRANT_COLLECTION_NAME`,
   `QDRANT_TIMEOUT`, `EMBEDDING_SERVICE_URL`, `EMBEDDING_SERVICE_TOKEN`,
   `GOOGLE_API_KEY`, `LLM_MODEL`, `SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES`,
   `CORS_ALLOW_ORIGINS`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
   `LANGFUSE_BASE_URL`. **Every value must be freshly rotated** -- anything
   that ever existed in a local `.env` during development is considered
   burned, per the secret-hygiene gate below. `QDRANT_SOURCE_URL` /
   `QDRANT_SOURCE_API_KEY` are Phase-1-only (one-time local migration) and
   are not needed in Railway.
4. Set `CORS_ALLOW_ORIGINS` to the actual Railway app URL once it's known
   (e.g. `["https://your-app.up.railway.app"]`) -- never a wildcard.
5. Confirm the healthcheck path is `/health` (already set in `railway.json`;
   Railway reads this file automatically).
6. Push to `main` to trigger the first deploy, then run the post-deploy
   smoke test below against the live URL.

### Secret hygiene

This repo was initialized from a clean working tree with `.env` already
git-ignored -- **no real secret was ever committed, so there is no git
history to scrub.** Only `.env.example` (placeholder values) is tracked.
Every secret that was ever used locally during development (Gemini key,
Qdrant Cloud key, Hetzner `EMBEDDING_SERVICE_TOKEN`, JWT `SECRET_KEY`,
Langfuse keys) must still be rotated before going live, since "never
committed" is not the same as "never seen outside Railway" -- Railway's env
vars should hold fresh values, not reused dev-time ones.

### Post-deploy smoke test

```
python scripts/railway_smoke_test.py https://<your-app>.up.railway.app
```

Hits the **live deployed URL** (not in-process) to prove Railway can
actually reach every external dependency: register/login (JWT), conversation
CRUD, a grounded/cited `/chat` turn over SSE (exercising Qdrant Cloud, the
Hetzner embedding+rerank endpoint, and Gemini from Railway's network), a
second turn proving cross-session memory, user isolation on both
`GET /conversations/{id}` and `POST /chat`, `/feedback`, and -- using local
Langfuse credentials to poll Langfuse Cloud directly -- that the trace
ingested with its guard-verdict and feedback scores. Exits non-zero on any
failure. (Simulated downstream-outage resilience is already proven
in-process by `phase5_validate.py` / `phase6_validate.py` and is not
re-tested against production.)
