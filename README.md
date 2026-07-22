# Legal Assistant

An Arabic-language legal research assistant for Egyptian lawyers. It answers legal questions with **article-level citations that are mechanically verified** against the retrieved law text — every citation is checked in code against what was actually retrieved before it ever reaches the user, so faithfulness to the source is prioritized over fluency.

Currently covers two ingested laws:

- قانون الإجراءات الجنائية رقم ١٧٤ لسنة ٢٠٢٥ (Criminal Procedure Law 174/2025)
- القانون المدني رقم ١٣١ لسنة ١٩٤٨ (Civil Code 131/1948)

**Live demo:** a basic web UI is deployed and open to try at [legal-assistant-frontend.mohamedraafat800.workers.dev](https://legal-assistant-frontend.mohamedraafat800.workers.dev/).

**Pipeline at a glance:**

```
PDF (Arabic law text)
  -> structured articles (arabic_ingest/)
  -> hybrid dense+sparse embeddings (BGE-M3) -> Qdrant Cloud
  -> retrieval + ColBERT rerank (embedding_service/, Modal GPU)
  -> tool-calling agent (Gemini + LangChain) -> JSON-contracted answer
  -> citation guard (verifies every citation against retrieved text)
  -> FastAPI, SSE streaming, Postgres persistence
  -> Langfuse tracing / scoring
```

## Tech stack

| Layer | Technology |
|---|---|
| Ingestion | `pdftotext` (poppler), custom Arabic text normalization |
| Embeddings / rerank | BGE-M3 (hybrid dense+sparse) + ColBERT, via [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) |
| Vector DB | Qdrant Cloud (hybrid RRF search) |
| LLM | Google Gemini via `langchain-google-genai` |
| Agent orchestration | LangChain `create_agent` (tool-calling) |
| API | FastAPI, Server-Sent Events (`sse-starlette`) |
| Persistence | PostgreSQL + SQLAlchemy (async) + Alembic |
| Observability | Langfuse Cloud |
| GPU inference hosting | Modal (serverless, L4 GPU) |
| App hosting | Railway (Nixpacks) |
| Frontend hosting | Cloudflare Workers |

---

## Phase 1 — Ingestion: PDF to structured Arabic law articles

**Location:** `arabic_ingest/` (`pdf_extractor.py`, `arabic_text.py`, `structure.py`, `articles.py`)

Raw law PDFs (`data/law-131-1948.pdf`, `data/قانون الاجراءات الجنائية...pdf`) are turned into clean, structured article records — the single source of truth for everything downstream.

- **Text extraction** (`pdf_extractor.py`) wraps poppler's `pdftotext`, the only engine found to preserve correct logical-order Arabic; the module refuses to run without it.
- **Arabic normalization** (`arabic_text.py`) fixes glyph shapes, Private-Use-Area font glyphs, BiDi spacing artifacts, and clause numbering. It produces **two** normalization levels:
  - *faithful* — verbatim text, used for citation and display (the only text a citation may ever quote).
  - *embedding* — a version used only for vectorization.
- **Structure detection** (`structure.py`) identifies books, parts, chapters, articles, and the "مواد الإصدار" (enacting/issuance articles).
- **Article slicing** (`articles.py`) produces per-article records: citation label, article number, law identity, structural context (book/part/chapter), page range, and body text.
- Inspection tools: `inspect_corpus.py` (structure report) and `preview_articles.py` (per-article JSON for manual review).

---

## Phase 2 — Embedding & vector indexing

**Location:** `arabic_ingest/` (`embeddings.py`, `chunker.py`, `vector_store.py`, `ingest.py`)

Structured articles are chunked and embedded into a searchable hybrid vector index.

- **Model:** BGE-M3 (`BGEM3Embedder` in `embeddings.py`), producing **hybrid dense + sparse** vectors from a single model call.
- **Chunking** (`chunker.py`) turns each article into a vector-DB-ready chunk: a metadata header, the faithful and normalized text, the citation label, and filterable metadata — written to `chunks_law174.json` / `chunks_law131.json`.
- **Vector store** (`vector_store.py`, `LawVectorStore`) sets up a Qdrant collection with named dense + sparse vectors, does idempotent upserts, and runs hybrid RRF (Reciprocal Rank Fusion) search with metadata filters.
- **`ingest.py`** is the CLI that ties it together: chunks JSON → embed → upsert into Qdrant. Supports a local embedded Qdrant (`./qdrant_storage`, no Docker) for development.
- Each stored point's payload includes `body_faithful` (the only text ever eligible for citation), `citation_label`, `header`, and filterable metadata (`law_number`, `article_number`, book/part/chapter, status).

Validated vectors were later migrated **one time**, without re-embedding, from the local Docker Qdrant instance to Qdrant Cloud (`scripts/migrate_to_cloud.py`, verified by `scripts/verify_migration.py`) — this is what production reads from today.

---

## Phase 3 — Embedding & rerank service

**Location:** `embedding_service/` (self-contained FastAPI microservice, no dependency on the main package)

A dedicated service exposes BGE-M3 embedding and ColBERT reranking over HTTP, so the main app never loads model weights itself.

- **`app/model.py`** (`BGEM3Service`) loads `BAAI/bge-m3` once and exposes:
  - `.embed()` — dense + sparse vectors, with Arabic normalization applied first.
  - `.rerank()` — ColBERT late-interaction scoring (`colbert_vecs` / `colbert_score`), computed only at query time and never stored.
- **`app/main.py`** exposes `GET /health` (unauthenticated) plus `POST /embed` and `POST /rerank` (both require a bearer token, `app/auth.py`). Rerank is processed in internal batches to bound peak memory.
- On the CPU deployment, inference was serialized behind a lock (`serialize_inference=True`); the GPU deployment disables this since the GPU can run overlapping requests concurrently.

This service was first tuned and validated on a CPU VPS before being migrated to GPU serverless hosting (see **Phase 8 — Deployment**).

---

## Phase 4 — Retrieval

**Location:** `src/legal_assistant/rag/retrieval.py` (`Retriever`)

Two distinct retrieval paths, chosen by the agent based on the kind of question being asked:

- **`search_articles(query_text, top_k=5, candidate_k=20, law_number=None)`** — conceptual/semantic search. The query is embedded, then Qdrant is queried with both dense and sparse `Prefetch` clauses fused via `FusionQuery(fusion=Fusion.RRF)`. The fused candidates are then reranked through the embedding service's ColBERT `/rerank` endpoint, and the top `k` are returned.
- **`get_article_by_number(article_number, law_number=None)`** — exact, metadata-filtered lookup with no embedding or ranking involved, since a bare article number carries little semantic signal and hybrid search tends to misrank it. Handles Arabic-Indic digits and preserves the مكرر ("bis") distinction; if `law_number` is omitted and the number exists in both laws, all matches are returned rather than guessed.

`RetrievedArticle.clean_text` is always `body_faithful` — embeddings and reranking only decide *which* article surfaces; they never touch the text that ends up quoted in a citation.

---

## Phase 5 — LLM agent & citation guard

**Location:** `src/legal_assistant/rag/` (`agent.py`, `prompts.py`, `tools.py`, `citation_guard.py`), `src/legal_assistant/llm.py`

- **LLM:** Google Gemini via `ChatGoogleGenerativeAI`, `temperature=0` — deterministic, since legal claims and citations must be reproducible, not creative.
- **Orchestration:** `LegalAssistantAgent` wraps a LangChain `create_agent` tool-calling graph bound to two tools (`get_article_by_number`, `search_articles`) and a Pydantic response format (`{answer_text, citations}`) that the model must fill structurally.
- **Prompting** (`prompts.py`): a strict, formal-Arabic system prompt with explicit rules — never cite from memory, how to choose between the two retrieval tools, how to disclose repealed articles, how to handle cross-law ambiguity — paired with a JSON output contract for citations.
- **Citation guard** (`citation_guard.py`) is plain Python, not AI: it parses the model's structured citations and checks each one against the set of articles actually retrieved this session (handling Arabic-Indic digits and the مكرر distinction), plus a regex scan for unverified inline "المادة ن" mentions in the prose. On a hard failure it regenerates once with a corrective instruction; if still invalid, it returns a fixed Arabic fallback message with zero citations rather than risk a hallucinated one.
- **Conversation memory** (`memory.py`): once a conversation reaches 12 turns, the oldest 6 are folded into an LLM-generated running summary, and every 6 turns after that the next full batch is folded in the same way — each compaction only processes the new batch, not the whole history, so the cost stays constant as the conversation grows. Turns not yet folded in are always kept verbatim. The summary carries no citation guarantees — citation correctness is guaranteed purely by replaying the persisted `retrieved_context`, so citations remain valid across sessions even after summarization.

---

## Phase 6 — API

**Location:** `src/legal_assistant/api/`

A FastAPI application (`app.py`, `create_app()`) exposes:

| Endpoint | Purpose |
|---|---|
| `POST /auth/register`, `POST /auth/login` | bcrypt password auth, returns `{access_token, user}`; register is IP-rate-limited |
| `POST /conversations`, `GET /conversations`, `GET /conversations/{id}` | user-scoped conversation CRUD |
| `POST /chat` | streamed chat over Server-Sent Events |
| `POST /feedback` | thumbs-up/down tied to a specific traced turn |
| `GET /health` | liveness check (also the deployment healthcheck path) |

**`/chat` streaming model:** the agent runs a full turn to completion — including citation guard verification — *before* anything is streamed. The verified `answer_text` is then streamed to the client in small chunks for a responsive typing effect. This is intentional: citations are never sent until they've been verified, so a client can never render an unverified citation, and a mid-stream disconnect loses nothing since the turn was already persisted.

| SSE event | Meaning |
|---|---|
| `token` | a chunk of the already-verified prose |
| `citations` | sent once, after all `token` events — the only citations the client may render |
| `withdrawn` | the guard hard-failed even after regeneration; no valid answer this turn |
| `done` | the turn is persisted; carries the Langfuse `trace_id` for `/feedback` |
| `error` | a downstream failure (Qdrant, embedding service, Gemini, DB) |

---

## Phase 7 — Observability

**Location:** `src/legal_assistant/observability.py`

Langfuse tracing wraps the system as a best-effort, non-blocking layer — every call swallows its own exceptions, and a no-op stand-in is used when tracing is disabled, so a Langfuse outage never affects chat behavior.

- **Trace** = one `/chat` request. Session and user IDs are propagated so Langfuse's Session ID / User ID columns populate correctly.
- **Spans:** `chat_turn` (the agent's tool choice and generation), `search_articles` / `get_article_by_number` (retrieval, including `rerank_latency_ms`), and `citation_guard`.
- **Scores:** the citation guard's verdict is logged automatically on every trace — `hallucinated_citations_count`, `citations_verified_count`, `inline_unverified_count`, `used_fallback` — functioning as an automated hallucination monitor with no human review required. `POST /feedback` adds a lawyer's `user_feedback` (1/0) to the same trace.

---

## Phase 8 — Deployment & infrastructure

### Embedding/rerank service — Modal (serverless GPU)

**Location:** `embedding_service/modal_app.py`

- Runs on an **L4 GPU** (deliberately not a larger card — BGE-M3 is a small ~2GB model, not LLM-scale).
- `max_containers=3` as a hard spend cap, up to 8 concurrent requests per container, a 300s scale-down window, and `min_containers=0` — no warm containers are kept, trading occasional 20-30s cold starts for lower idle cost.
- Model weights are baked into the container image at build time and loaded with `HF_HUB_OFFLINE=1`, so a cold start doesn't re-download ~2GB from the Hugging Face Hub.
- Reuses the exact same application code as an earlier CPU/VPS deployment; only inference serialization differs between the two (the CPU deployment serializes requests behind a lock, the GPU deployment does not need to).
- This service was migrated from a CPU VPS to Modal after tuning batch sizes, rerank candidate counts, and truncation lengths under real load — retrieval's rerank candidate pool was lowered to 10 for CPU latency, then raised back to 20 after the GPU migration changed the cost/latency tradeoff.

### Main application — Railway

**Location:** `railway.json`

- **Build:** Nixpacks, auto-detected from `pyproject.toml`.
- **Start command:** `alembic upgrade head && uvicorn legal_assistant.api.app:app --app-dir src --host 0.0.0.0 --port $PORT` — migrations run on every deploy, and the deploy fails loudly if a migration fails, so the app never silently starts against a stale schema.
- **Database:** Railway's managed PostgreSQL plugin in production; `docker-compose.yml` runs a local `postgres:16-alpine` container for development.
- `Settings.database_url` normalizes `postgres://`/`postgresql://` connection strings to `postgresql+asyncpg://` automatically for compatibility with Railway's managed connection string.
- All dependencies are pinned to exact versions in `pyproject.toml` for reproducible builds.
- Secrets live only in `.env` (git-ignored) locally and in Railway's environment variables in production; every secret is rotated before being placed into the production environment, rather than reusing development-time values.

### Environment configuration

Key variables (see `.env.example`): `QDRANT_CLOUD_URL` / `QDRANT_CLOUD_API_KEY` / `QDRANT_COLLECTION_NAME`, `EMBEDDING_SERVICE_URL` / `EMBEDDING_SERVICE_TOKEN`, `GOOGLE_API_KEY` / `LLM_MODEL`, `DATABASE_URL`, `SECRET_KEY` / `ACCESS_TOKEN_EXPIRE_MINUTES`, `CORS_ALLOW_ORIGINS`, `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`.

---

## Phase 9 — Validation & testing

No pytest suite exists yet (`tests/` is an empty package) — validation instead runs as standalone scripts that exercise the real stack end to end:

- `scripts/phase4_validate.py`, `phase5_validate.py`, `phase6_validate.py` — in-process validation (via `httpx.ASGITransport`) against the real database, Qdrant, embedding service, and Gemini: auth, conversation ownership, streaming citation ordering, cross-session citation reuse, user isolation, guard-fallback safety, and simulated downstream outages.
- `scripts/railway_smoke_test.py <url>` — post-deploy smoke test against the **live** Railway URL: auth, a cited chat turn over SSE, cross-session memory, isolation, feedback, and polling Langfuse Cloud to confirm trace ingestion.
- `scripts/check_retrieval.py` — proves the retrieval path (hybrid search → rerank → exact lookup) against Qdrant Cloud with real sample queries.
- `scripts/check_embedding_consistency.py` — proves the deployed embedding service produces vectors numerically consistent with what's already stored in Qdrant Cloud.

---

## Project layout

```
arabic_ingest/            Offline ingestion pipeline: PDF -> articles -> chunks -> Qdrant
  pdf_extractor.py         PDF text extraction (pdftotext)
  arabic_text.py            Arabic normalization (faithful + embedding variants)
  structure.py               Book/part/chapter/article structure detection
  articles.py                 Article record slicing
  embeddings.py                BGE-M3 embedding
  chunker.py                    Article -> chunk transformation
  vector_store.py                 Qdrant collection management + hybrid search
  ingest.py                        CLI: chunks -> embed -> Qdrant upsert

embedding_service/         Standalone FastAPI microservice: BGE-M3 embed + ColBERT rerank
  app/model.py               Model loading, embed(), rerank()
  app/main.py                  FastAPI endpoints + auth
  modal_app.py                  Modal serverless GPU deployment

src/legal_assistant/       Main application package
  rag/                        Agent, prompts, tools, citation guard, retrieval
  api/                          FastAPI app + routes (auth, conversations, chat, feedback)
  db/                            Postgres session + Qdrant client factories
  llm.py                          Gemini client
  auth.py                          Password hashing, JWT
  observability.py                  Langfuse tracing wrapper
  embedding_client.py                HTTP client for embedding_service

scripts/                   Migration, validation, and smoke-test CLIs
alembic/                   Postgres schema migrations
data/                      Source law PDFs
docker-compose.yml         Local Postgres for development
railway.json                Railway deployment config
```

Subfolder details: [`arabic_ingest/README.md`](arabic_ingest/README.md), [`embedding_service/README.md`](embedding_service/README.md).

## Setup (local development)

1. Python 3.11+, install the project in editable mode: `pip install -e .`
2. Copy `.env.example` to `.env` and fill in Qdrant Cloud, embedding service, Gemini, database, and (optionally) Langfuse credentials.
3. Start local Postgres: `docker compose up -d postgres`, then `alembic upgrade head`.
4. Run the API: `uvicorn legal_assistant.api.app:app --app-dir src --reload` — interactive docs at `http://127.0.0.1:8000/docs`.
