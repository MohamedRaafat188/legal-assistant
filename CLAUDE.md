# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

An Arabic-language legal research assistant for Egyptian lawyers. `README.md` is the authoritative deep-dive (9 phases from PDF ingestion through deployment); read it for anything not covered here. This file focuses on commands and the invariants that are easy to break.

## Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run the API locally (interactive docs at http://127.0.0.1:8000/docs)
uvicorn legal_assistant.api.app:app --app-dir src --reload

# Local Postgres + migrations
docker compose up -d postgres
alembic upgrade head
alembic revision --autogenerate -m "message"   # new migration

# Lint (ruff: E, F, I, UP, B; line-length 100)
ruff check .
ruff check --fix .

# Offline ingestion CLI (chunks JSON -> embed -> Qdrant upsert)
python -m arabic_ingest.ingest path/to/chunks_law174.json [--recreate]

# End-to-end validation (NOT pytest — these hit the real DB, Qdrant, embedding service, Gemini)
python scripts/phase4_validate.py     # retrieval
python scripts/phase5_validate.py     # agent + citation guard
python scripts/phase6_validate.py     # API (in-process via httpx.ASGITransport)
python scripts/railway_smoke_test.py <live-url>   # post-deploy smoke test
python scripts/ask.py --username U --password P    # interactive CLI chat client
```

**Testing note:** `tests/` is an empty package and `pytest` currently collects nothing. All real validation lives in `scripts/*_validate.py`, which exercise the full live stack rather than mocking. There is no isolated unit-test path — running validation requires a populated `.env` and reachable services.

## Architecture invariants (do not break these)

These are the load-bearing design rules; violating them is how you introduce a hallucinated citation or a schema drift.

- **`body_faithful` is the only text a citation may quote.** Embeddings and ColBERT rerank only decide *which* article surfaces — they never touch citable text. `RetrievedArticle.clean_text` is always `body_faithful`. `arabic_text.py` produces two normalization levels: *faithful* (verbatim, for citation/display) and *embedding* (for vectorization only).
- **The citation guard (`rag/citation_guard.py`) is plain Python, not AI.** It checks every structured citation against the articles actually retrieved this session, plus a regex scan for unverified inline "المادة ن" mentions. On hard failure it regenerates once, then falls back to a fixed Arabic message with zero citations rather than emit an unverified one.
- **Citation correctness survives summarization by replaying persisted `retrieved_context`, not the LLM summary.** `memory.py` folds old turns into a running summary (batched every 6 turns after turn 12), but the summary carries no citation guarantees.
- **`/chat` verifies the full turn (including the guard) BEFORE streaming anything.** Only already-verified prose is streamed as `token` events; `citations` is sent once, after all tokens. A client can never render an unverified citation. See the SSE event table in README Phase 6.
- **Arabic-Indic digits and the مكرر ("bis") distinction must be preserved** anywhere article numbers are parsed or compared. A مكرر article is legally distinct from its base article, so the flag has to survive the *whole* round trip — Qdrant payload (`article_suffix`) → `RetrievedArticle` → `AllowedSet` → the citation JSON contract's `article_suffix` field → `_CitationOut` (the schema the provider structurally enforces) → persisted `citations`/`retrieved_context`. Drop it at any hop and either the guard collapses مكرر onto its base article, or a correct citation gets rejected as hallucinated. `get_article_by_number` returns all matches across laws when `law_number` is ambiguous rather than guessing.
- **Gemini runs at `temperature=0`** — legal claims must be reproducible, not creative.
- **Alembic migrations run on every deploy** (`alembic upgrade head` in the Railway start command) and the deploy fails loudly if a migration fails. Never let the app start against a stale schema.

## Layout notes

- `src/legal_assistant/` is the deployed app. `rag/` = agent, prompts, tools, citation guard, retrieval; `api/routes/` = auth, conversations, chat, feedback; `db/` = Postgres session + Qdrant client factories.
- `arabic_ingest/` is a separate **offline** pipeline (PDF → articles → chunks → Qdrant). It has its own `retrieval.py`/`search.py`/`config.py` used for ingestion-time experiments — the *production* retriever is `src/legal_assistant/rag/retrieval.py`, not the one in `arabic_ingest/`.
- `embedding_service/` is a self-contained FastAPI microservice (BGE-M3 embed + ColBERT rerank) with **no dependency on the main package**; it deploys to Modal serverless GPU via `modal_app.py`. The main app never loads model weights — it calls this service over HTTP (`embedding_client.py`).
- Dependencies in `pyproject.toml` are pinned to exact versions for reproducible builds; keep new deps pinned.
- `Settings.database_url` normalizes `postgres://`/`postgresql://` to `postgresql+asyncpg://` automatically. Secrets live only in `.env` (git-ignored); see `.env.example` for the full variable list.
