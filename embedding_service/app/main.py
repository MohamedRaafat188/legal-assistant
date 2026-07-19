"""FastAPI app exposing BGE-M3 embed + ColBERT rerank endpoints.

Self-contained deployable: does not import the ``legal_assistant`` package.
The model is loaded once at startup (lifespan) and reused across requests.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .auth import require_bearer_token
from .config import get_settings
from .model import get_service

# Hard cap on passages per /rerank call. The internal batching (rerank_batch_size)
# already controls the peak-memory spike; this cap bounds total request latency
# and rejects pathological requests before they can queue up work.
_MAX_RERANK_PASSAGES = 50

# On a GPU-less CPU box, BGE-M3 inference wants every core. FastAPI runs sync
# `def` endpoints in a threadpool, so without this lock two concurrent
# /embed or /rerank calls would run their encode() in parallel threads,
# fighting over the same cores and RAM, slowing both down together rather
# than one queuing behind the other. A GPU parallelizes overlapping
# inference far better, so `serialize_inference=False` (set by the Modal
# deployment) skips the lock.
_inference_lock = threading.Lock() if get_settings().serialize_inference else contextlib.nullcontext()

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the BGE-M3 model once before serving any requests."""
    get_service().load()
    yield


app = FastAPI(title="Legal Assistant Embedding Service", lifespan=lifespan)


# --- request/response models -------------------------------------------------
class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class SparseVectorOut(BaseModel):
    indices: list[int]
    values: list[float]


class EmbedResultOut(BaseModel):
    dense: list[float]
    sparse: SparseVectorOut


class EmbedResponse(BaseModel):
    results: list[EmbedResultOut]


class RerankRequest(BaseModel):
    query: str
    passages: list[str] = Field(..., min_length=1)


class RerankResultOut(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: list[RerankResultOut]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --- endpoints ----------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check: does not require auth."""
    return HealthResponse(status="ok", model_loaded=get_service().is_loaded)


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_bearer_token)])
def embed(request: EmbedRequest) -> EmbedResponse:
    """Normalize + encode one or more texts into dense + sparse vectors."""
    try:
        with _inference_lock:
            vectors = get_service().embed(request.texts)
    except RuntimeError as exc:
        _log.exception("embed failed")
        raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc
    return EmbedResponse(
        results=[
            EmbedResultOut(
                dense=v.dense,
                sparse=SparseVectorOut(indices=v.sparse.indices, values=v.sparse.values),
            )
            for v in vectors
        ]
    )


@app.post("/rerank", response_model=RerankResponse, dependencies=[Depends(require_bearer_token)])
def rerank(request: RerankRequest) -> RerankResponse:
    """ColBERT-rerank passages against a query, returned in original order."""
    if len(request.passages) > _MAX_RERANK_PASSAGES:
        raise HTTPException(
            status_code=413,
            detail=f"too many passages ({len(request.passages)} > {_MAX_RERANK_PASSAGES})",
        )
    try:
        with _inference_lock:
            scores = get_service().rerank(request.query, request.passages)
    except RuntimeError as exc:
        _log.exception("rerank failed")
        raise HTTPException(status_code=500, detail=f"rerank failed: {exc}") from exc
    results = [RerankResultOut(index=i, score=score) for i, score in enumerate(scores)]
    results.sort(key=lambda r: r.score, reverse=True)
    return RerankResponse(results=results)
