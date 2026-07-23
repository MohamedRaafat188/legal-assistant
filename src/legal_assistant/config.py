"""Application configuration loaded from environment variables and .env.

New settings for later phases (LLM keys, embedding-service URL, Langfuse
keys, database URL) should be added here as additional fields.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings, sourced from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Source Qdrant (local Docker, server mode)
    qdrant_source_url: str = "http://localhost:6333"
    qdrant_source_api_key: str | None = None

    # Qdrant Cloud (destination)
    qdrant_cloud_url: str
    qdrant_cloud_api_key: str

    # Shared
    qdrant_collection_name: str
    qdrant_timeout: int = 60

    # Embedding service (BGE-M3 embed + rerank, deployed separately)
    embedding_service_url: str
    embedding_service_token: str

    # LLM (Gemini, via langchain-google-genai)
    google_api_key: str
    llm_model: str
    # Cheaper model for conversation-summary compaction (memory.maybe_compact_summary):
    # that task is prose compression with no tool use or citation formatting, so it
    # doesn't need the main agent model's capability.
    summary_llm_model: str = "gemini-3.5-flash-lite"

    # Postgres (users, conversations, messages) -- async SQLAlchemy + asyncpg.
    # Points at local Docker Postgres in dev; swappable to Railway's managed
    # Postgres later with no code change.
    database_url: str = "postgresql+asyncpg://legal_assistant:legal_assistant_dev@localhost:5432/legal_assistant"

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        """Railway (and most managed-Postgres providers) inject `postgres://`
        or `postgresql://`, but SQLAlchemy's async engine needs the explicit
        `postgresql+asyncpg://` driver scheme."""
        if v.startswith("postgres://"):
            return "postgresql+asyncpg://" + v[len("postgres://") :]
        if v.startswith("postgresql://"):
            return "postgresql+asyncpg://" + v[len("postgresql://") :]
        return v

    # API (FastAPI): session-identity token signing + CORS. Minimal auth --
    # no refresh/reset/verification (deferred). Dev default is NOT for prod;
    # override via .env / real environment before deploying.
    secret_key: str = "dev-only-insecure-secret-change-me"
    access_token_expire_minutes: int = 60 * 24 * 7
    cors_allow_origins: list[str] = ["http://localhost:3000"]

    # Langfuse Cloud (observability): tracing, guard-verdict scores, feedback
    # scores. Optional -- if either key is unset, tracing is disabled and the
    # app runs exactly as before (best-effort, never blocks/breaks the chat
    # or feedback path).
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance."""
    return Settings()
