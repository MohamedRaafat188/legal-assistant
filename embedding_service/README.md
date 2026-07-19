# Legal Assistant Embedding Service

Standalone BGE-M3 embed + ColBERT rerank service. Self-contained: does not
import the `legal_assistant` package. Deployed on Modal as a serverless GPU
app (`modal_app.py`) — see that file's docstring for deploy/dev commands.

Reproduces the ingestion project's embedding pipeline exactly (same model,
same `normalize_for_embedding`, same encode configuration) so that vectors
produced here are numerically consistent with the vectors already stored in
Qdrant Cloud. See `scripts/check_embedding_consistency.py` in the main
project for the empirical proof.

## Endpoints

- `GET /health` — liveness, no auth required.
- `POST /embed` — `{"texts": ["..."]}` → dense + sparse vectors per text.
  Applies `normalize_for_embedding` internally, so pass raw article text or a
  raw query.
- `POST /rerank` — `{"query": "...", "passages": ["..."]}` → ColBERT scores,
  passages returned sorted by relevance (`index` refers to the original
  input order). Capped at 50 passages per request; encoded in small internal
  batches to bound peak memory.

Both `/embed` and `/rerank` require `Authorization: Bearer <token>`.

## Deploying

```
modal deploy embedding_service/modal_app.py
```

Requires a Modal secret named `embedding-service-secrets` (custom type) with
key `EMBEDDING_SERVICE_TOKEN`. Record the deployed web endpoint URL and the
token into the main project's `.env` and Railway env vars as
`EMBEDDING_SERVICE_URL` / `EMBEDDING_SERVICE_TOKEN`.

For local iteration without a full deploy: `modal serve embedding_service/modal_app.py`.
