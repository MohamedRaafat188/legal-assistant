"""Modal serverless GPU deployment of the embed+rerank service.

This is a *runtime move*, not a re-implementation: the FastAPI app, model
wrapper, Arabic normalization, and bearer-auth in ``app/`` are the exact
same code that runs on the CPU/Hetzner deployment (main.py, model.py,
normalize.py, auth.py, config.py are untouched apart from the
``serialize_inference`` toggle in config.py/main.py, which lets this GPU
deployment skip the CPU-only concurrency lock). Vector consistency with the
corpus already stored in Qdrant Cloud depends on that code staying identical
between deployments -- see scripts/check_embedding_consistency.py.

Deploy:   modal deploy embedding_service/modal_app.py
Dev/test: modal serve embedding_service/modal_app.py

Requires a Modal secret named "embedding-service-secrets" (custom type)
with key EMBEDDING_SERVICE_TOKEN.

No warm containers are kept: min_containers=0 at all times, so every request
after an idle period pays a cold start. This trades latency for cost -- see
the operator's decision in the MAX_CONTAINERS/SCALEDOWN_WINDOW_SECONDS
comments below.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# GPU sizing: BGE-M3 is a ~2GB model; embedding + ColBERT rerank is small-
# model inference, not LLM serving. A T4/L4-class card has huge VRAM
# headroom for this workload. Do NOT upsize to A10G/A100/H100 -- those are
# for training or large-LLM serving and would waste budget on a limited plan.
GPU_TYPE = "L4"

# Hard cap on concurrent GPU containers. Serverless bills per GPU-second
# across all workers; an unbounded burst is the one way this blows the
# budget. A burst beyond this cap queues briefly instead of fanning out.
# Also set/confirm a spend limit at the Modal account level.
MAX_CONTAINERS = 3

# Requests served concurrently per warm container. The GPU parallelizes
# overlapping inference far better than the CPU box's 2 vCPUs did, so this
# is intentionally > 1 (see serialize_inference=false below).
MAX_CONCURRENT_INPUTS = 8

# How long an idle container stays warm before scaling to zero. Keeps a
# container alive briefly between requests within a burst without paying to
# idle indefinitely once traffic actually stops.
SCALEDOWN_WINDOW_SECONDS = 300

# No proactive warm-up: min_containers=0 at all times, day and night.
# Operator chose to eliminate the daytime warm container entirely for cost
# reasons, accepting that the first search_articles call of any burst after
# an idle period (>SCALEDOWN_WINDOW_SECONDS) cold-starts (~20-30s) -- this
# includes the concurrent-second-call scenario from the Phase 9a
# investigation, which is no longer specially handled.
MIN_CONTAINERS = 0

APP_NAME = "legal-assistant-embedding"
app = modal.App(APP_NAME)


def _download_model() -> None:
    """Runs at image-build time (network allowed): bakes model weights into
    the image so cold starts don't re-download ~2GB from Hugging Face."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="BAAI/bge-m3")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "transformers==4.57.6",
        "tokenizers==0.22.2",
        "accelerate==1.13.0",
        "safetensors==0.7.0",
        "sentence-transformers==5.5.0",
        "peft>=0.11",
        "FlagEmbedding==1.2.11",
        "numpy==2.4.4",
        "fastapi==0.115.0",
        "uvicorn[standard]==0.32.0",
        "pydantic-settings==2.14.0",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/model_cache"})
    .run_function(_download_model)
    # Runtime-only: forces the baked-in snapshot, no network calls on cold start.
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("app")
)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    secrets=[modal.Secret.from_name("embedding-service-secrets")],
    env={
        "BGE_USE_FP16": "true",  # real speedup on GPU; was forced False on the CPU box
        "BGE_MAX_LENGTH": "1536",
        "RERANK_MAX_LENGTH": "256",
        "RERANK_BATCH_SIZE": "6",
        "SERIALIZE_INFERENCE": "false",
    },
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class EmbeddingService:
    @modal.enter()
    def load(self) -> None:
        from app.model import get_service

        get_service().load()

    @modal.asgi_app()
    def web(self):
        from app.main import app as fastapi_app

        return fastapi_app
