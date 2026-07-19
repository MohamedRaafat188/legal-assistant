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
          then: modal run embedding_service/modal_app.py::sync_warm_state
Dev/test: modal serve embedding_service/modal_app.py

Requires a Modal secret named "embedding-service-secrets" (custom type)
with key EMBEDDING_SERVICE_TOKEN.

IMPORTANT -- always run sync_warm_state right after modal deploy: Modal
resets a Function's autoscaler to its static decorator config (min_containers=0
here) on every deploy, and the scheduled warm_up/cool_down crons only run at
their fixed times (~07:55/20:00 Africa/Cairo). A deploy done mid-day would
otherwise sit at min_containers=0 -- fully cold -- until the next scheduled
firing, possibly hours away. sync_warm_state applies the correct warm level
for the current moment immediately, closing that gap.
"""

from __future__ import annotations

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

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

# Business-hours warm window (Cairo, Sun-Thu -- Egypt's work week). A
# scheduled min_containers bump just before the first lawyer arrives beats
# both a cold first request and a 24/7 keep-alive poller: natural daytime
# traffic then keeps containers warm, and they scale to zero overnight for
# free. Adjust the day-of-week field if the actual usage pattern differs.
#
# PEAK_MIN_CONTAINERS=1: a single complex query can fire several
# search_articles calls, and a real trace showed the *second* concurrent
# call cold-starting a fresh container (~20-29s) while the first warm
# container was still busy -- see the Phase 9a investigation. Bumping this
# to 2 removes that cold start entirely, but ~doubles the daytime GPU cost
# (~$209/mo -> ~$419/mo at Modal's L4 rate for the Sun-Thu 07:55-20:00
# window). Operator chose to keep the cost down and accept that an
# occasional overlapping second query may cold-start (~20-29s) -- MAX_CONTAINERS
# still caps the worst case at 3 concurrent containers.
PEAK_MIN_CONTAINERS = 1
OFF_HOURS_MIN_CONTAINERS = 0
WARM_WINDOW_TZ = "Africa/Cairo"
WARM_UP_CRON = "55 7 * * 0-4"  # ~07:55, five minutes before the window
COOL_DOWN_CRON = "0 20 * * 0-4"  # 20:00 -- back to scale-to-zero
_BUSINESS_DAYS = {6, 0, 1, 2, 3}  # datetime.weekday(): Sun=6, Mon=0 ... Thu=3
_WINDOW_START = dt_time(7, 55)
_WINDOW_END = dt_time(20, 0)


def _in_business_hours() -> bool:
    now = datetime.now(ZoneInfo(WARM_WINDOW_TZ))
    return now.weekday() in _BUSINESS_DAYS and _WINDOW_START <= now.time() < _WINDOW_END

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
    min_containers=0,  # scale to zero by default; bumped on schedule below
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


def _service_handle() -> "modal.cls.Obj":
    """Handle to the *deployed* class's container pool. Looked up by name
    (not a local `EmbeddingService()` reference) so this works correctly both
    from the scheduled functions running inside the deployed app and from
    `modal run ...::sync_warm_state`, which otherwise spins up its own
    ephemeral app context rather than reaching the persistent deployment."""
    return modal.Cls.from_name(APP_NAME, "EmbeddingService")()


@app.function(schedule=modal.Cron(WARM_UP_CRON, timezone=WARM_WINDOW_TZ))
def warm_up() -> None:
    """Keep PEAK_MIN_CONTAINERS warm through business hours instead of
    cold-starting the first (or second concurrent) request of the day."""
    _service_handle().update_autoscaler(min_containers=PEAK_MIN_CONTAINERS)


@app.function(schedule=modal.Cron(COOL_DOWN_CRON, timezone=WARM_WINDOW_TZ))
def cool_down() -> None:
    """End of business hours: back to scale-to-zero for the overnight idle."""
    _service_handle().update_autoscaler(min_containers=OFF_HOURS_MIN_CONTAINERS)


@app.function()
def sync_warm_state() -> None:
    """Apply the correct warm level for *right now*, regardless of the cron
    schedule. Run this manually after every `modal deploy` -- deploys reset
    the autoscaler to the static decorator default (min_containers=0), and
    without this the service would otherwise stay fully cold until the next
    scheduled warm_up/cool_down firing.

        modal run embedding_service/modal_app.py::sync_warm_state
    """
    target = PEAK_MIN_CONTAINERS if _in_business_hours() else OFF_HOURS_MIN_CONTAINERS
    _service_handle().update_autoscaler(min_containers=target)
    print(f"min_containers set to {target} (business_hours={_in_business_hours()})")
