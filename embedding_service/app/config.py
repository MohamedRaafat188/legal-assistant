"""Configuration for the embedding service, loaded from environment / .env."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Embedding-service settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Must match arabic_ingest/config.py's EMBEDDING_MODEL exactly.
    embedding_model: str = "BAAI/bge-m3"
    # arabic_ingest defaults this to True but auto-forces False on CPU-only
    # boxes inside BGEM3FlagModel.__init__ -- set explicitly here for clarity
    # on a GPU-less Hetzner VPS.
    bge_use_fp16: bool = False
    # Matches FlagEmbedding's own default (and ingestion's un-overridden default)
    # was 8192; lowered to bound worst-case rerank latency on this CPU-only box --
    # see embedding_service latency investigation.
    bge_max_length: int = 1536
    # Separate, shorter cap for ColBERT rerank passages only (embed's
    # bge_max_length must stay large enough to match ingestion-time vectors;
    # rerank vecs are computed fresh per query, so they're free to truncate
    # harder to cut worst-case latency).
    rerank_max_length: int = 256
    # Passages per internal batch during /rerank, to cap the ColBERT memory spike.
    # Tried bumping to 10 (== RERANK_CANDIDATES_DEFAULT, one batch) since
    # measured peak memory (818 MiB) left headroom under the 3800 MiB limit,
    # but it showed no latency win and was measurably slower/heavier in
    # practice -- CPU rerank is compute- not batch-count-bound. Kept at 6.
    rerank_batch_size: int = 6
    # Serialize /embed and /rerank calls through a lock so concurrent requests
    # don't fight over the same CPU cores (see main.py). A GPU parallelizes
    # overlapping inference far better, so the Modal deployment sets this to
    # False; the CPU/Hetzner deployment keeps the default True.
    serialize_inference: bool = True

    # Bearer token required on /embed and /rerank.
    embedding_service_token: str


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance."""
    return Settings()
