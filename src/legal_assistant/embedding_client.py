"""Thin HTTP client for the standalone embedding service (/embed, /rerank)."""

from __future__ import annotations

# The Windows cert-store SSL workaround (see legal_assistant/__init__.py) is
# applied on package import, before this module runs.

from dataclasses import dataclass

import requests
import urllib3

from legal_assistant.config import Settings, get_settings

# verify=False is intentional: the service currently serves HTTPS via
# Caddy's self-signed internal CA (no public domain yet for this trial
# deployment). Silence the resulting per-request warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass(frozen=True, slots=True)
class SparseVectorOut:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True, slots=True)
class EmbedResult:
    dense: list[float]
    sparse: SparseVectorOut


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    score: float


class EmbeddingClient:
    """Calls the embedding service's /embed and /rerank endpoints.

    The service currently serves HTTPS via Caddy's self-signed internal CA
    (no public domain yet for this trial deployment), so TLS verification is
    disabled here. Switch to verify=True once a real certificate is in place.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._base_url = settings.embedding_service_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {settings.embedding_service_token}"}

    def embed(self, texts: list[str]) -> list[EmbedResult]:
        """Embed one or more raw texts (server applies normalize_for_embedding)."""
        resp = requests.post(
            f"{self._base_url}/embed",
            json={"texts": texts},
            headers=self._headers,
            verify=False,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            EmbedResult(
                dense=r["dense"],
                sparse=SparseVectorOut(indices=r["sparse"]["indices"], values=r["sparse"]["values"]),
            )
            for r in data["results"]
        ]

    def rerank(self, query: str, passages: list[str]) -> list[RerankResult]:
        """ColBERT-rerank passages against a query. Returns results sorted by score desc."""
        resp = requests.post(
            f"{self._base_url}/rerank",
            json={"query": query, "passages": passages},
            headers=self._headers,
            verify=False,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return [RerankResult(index=r["index"], score=r["score"]) for r in data["results"]]

    def health(self) -> bool:
        resp = requests.get(f"{self._base_url}/health", verify=False, timeout=30)
        resp.raise_for_status()
        return resp.json().get("model_loaded", False)
