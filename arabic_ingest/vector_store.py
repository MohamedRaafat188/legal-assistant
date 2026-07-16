# -*- coding: utf-8 -*-
"""Qdrant vector store for the Egyptian-law corpus.

Each point stores TWO named vectors — dense (semantic) and sparse (lexical) —
matching BGE-M3's hybrid output, plus a payload with the faithful text and all
filterable metadata. Search fuses dense + sparse with server-side RRF.

Point ids are deterministic UUIDs derived from the chunk_id, so re-ingesting a
law upserts (updates in place) instead of duplicating.
"""
from __future__ import annotations

import uuid
from typing import Any, Iterable, Optional

from qdrant_client import QdrantClient, models

import config
from embeddings import HybridVec

# Stable namespace so uuid5(chunk_id) is reproducible across runs/machines.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk_id (idempotent upserts)."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def make_client(location: str | None = None) -> QdrantClient:
    """Create a Qdrant client in whichever mode `location` implies:

      * ":memory:"              -> ephemeral, in-process (tests)
      * "http(s)://host:port"   -> a running Qdrant server (production / Railway)
      * any other string        -> a local on-disk path: EMBEDDED Qdrant that
                                   runs inside this process and persists to that
                                   folder. No server, no Docker — ideal for solo
                                   local dev. Note: single-process access only
                                   (you can't ingest and serve at the same time),
                                   so switch to the server URL for deployment.
    """
    loc = location or config.QDRANT_URL
    if loc == ":memory:":
        return QdrantClient(location=":memory:")
    if loc.startswith(("http://", "https://")):
        return QdrantClient(url=loc, api_key=config.QDRANT_API_KEY,
                            timeout=config.QDRANT_TIMEOUT)
    return QdrantClient(path=loc)


class LawVectorStore:
    """Thin, purpose-built wrapper around a Qdrant collection."""

    def __init__(self, client: QdrantClient,
                 collection: str = config.COLLECTION_NAME) -> None:
        self.client = client
        self.collection = collection

    # -- schema ------------------------------------------------------------
    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection (dense + sparse named vectors) if needed."""
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    config.DENSE_VECTOR: models.VectorParams(
                        size=config.DENSE_DIM, distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    config.SPARSE_VECTOR: models.SparseVectorParams()
                },
            )
            # Payload indexes for the fields we filter on most (fast filtering).
            for field, schema in (
                ("law_number", models.PayloadSchemaType.INTEGER),
                ("law_year", models.PayloadSchemaType.INTEGER),
                ("article_number", models.PayloadSchemaType.INTEGER),
                ("book_number", models.PayloadSchemaType.INTEGER),
                ("article_type", models.PayloadSchemaType.KEYWORD),
                # Civil-Code (Law 131) additions -- division ancestry, and
                # repeal status so a query never surfaces a repealed article
                # as if it were active without the caller being able to filter.
                ("division_number", models.PayloadSchemaType.INTEGER),
                ("article_status", models.PayloadSchemaType.KEYWORD),
            ):
                self.client.create_payload_index(self.collection, field, schema)

    # -- writing -----------------------------------------------------------
    @staticmethod
    def _to_point(chunk: dict, vec: HybridVec) -> models.PointStruct:
        """Build a Qdrant point from a chunk dict + its hybrid embedding."""
        payload: dict[str, Any] = {
            "chunk_id": chunk["chunk_id"],
            "citation_label": chunk["citation_label"],
            "header": chunk["header"],
            "body_faithful": chunk["body_faithful"],      # the ONLY citation source
            "text_for_display": chunk["text_for_display"],
            **chunk["metadata"],                           # law/article/book/... fields
        }
        return models.PointStruct(
            id=point_id(chunk["chunk_id"]),
            vector={
                config.DENSE_VECTOR: vec.dense,
                config.SPARSE_VECTOR: models.SparseVector(
                    indices=vec.sparse.indices, values=vec.sparse.values,
                ),
            },
            payload=payload,
        )

    def upsert(self, chunks: list[dict], vectors: list[HybridVec]) -> int:
        """Upsert a batch of chunks with their embeddings. Returns count."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        points = [self._to_point(c, v) for c, v in zip(chunks, vectors)]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    # -- reading -----------------------------------------------------------
    def hybrid_search(
        self,
        query: HybridVec,
        limit: int = 10,
        prefetch_limit: int = 30,
        query_filter: Optional[models.Filter] = None,
    ) -> list[models.ScoredPoint]:
        """Dense + sparse retrieval fused with RRF, with optional metadata filter."""
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=query.dense, using=config.DENSE_VECTOR,
                    limit=prefetch_limit, filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query.sparse.indices, values=query.sparse.values,
                    ),
                    using=config.SPARSE_VECTOR,
                    limit=prefetch_limit, filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points

    def filter_only(
        self,
        query_filter: models.Filter,
        limit: int = 10,
    ) -> list[models.Record]:
        """Fetch points by metadata filter alone, no vector ranking involved.

        For lookups where the caller already knows the exact identifying
        metadata (e.g. a specific article_number), embedding-based ranking
        only adds noise -- filtering is exact where semantic search is not.
        """
        records, _next_offset = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return records


# --- filter helpers (metadata-filtered retrieval) --------------------------
def filter_by(**equals: Any) -> models.Filter:
    """Build an AND filter of exact matches, e.g. filter_by(law_number=174,
    book_number=1). None values are ignored."""
    conditions = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in equals.items() if v is not None
    ]
    return models.Filter(must=conditions)
