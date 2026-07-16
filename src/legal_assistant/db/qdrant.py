"""Qdrant client factories.

This is the single place Qdrant clients are constructed, so that the
migration scripts and later phases (retrieval, ingestion) share the same
connection setup.
"""

from qdrant_client import QdrantClient

from legal_assistant.config import Settings, get_settings


def get_source_client(settings: Settings | None = None) -> QdrantClient:
    """Return a client for the source (local Docker) Qdrant instance.

    Always connects in server mode via URL, never via a local file path.
    """
    settings = settings or get_settings()
    return QdrantClient(
        url=settings.qdrant_source_url,
        api_key=settings.qdrant_source_api_key,
        timeout=settings.qdrant_timeout,
    )


def get_cloud_client(settings: Settings | None = None) -> QdrantClient:
    """Return a client for the destination Qdrant Cloud instance."""
    settings = settings or get_settings()
    return QdrantClient(
        url=settings.qdrant_cloud_url,
        api_key=settings.qdrant_cloud_api_key,
        timeout=settings.qdrant_timeout,
    )
