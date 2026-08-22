# -*- coding: utf-8 -*-
"""Ingestion pipeline variant: chunks JSON -> embedding SERVICE -> Qdrant.

Same output as `ingest.py`, different source of vectors. `ingest.py` loads
BGE-M3 in-process through FlagEmbedding, which needs torch and ~4 GB of RAM;
this variant calls the deployed embedding service instead, which runs *the same
BAAI/bge-m3 weights through the same FlagEmbedding wrapper* (see
`embedding_service/`) and is also what encodes queries at request time.

Two properties make the two paths interchangeable rather than merely similar:

  * Points are built by `LawVectorStore._to_point`, so the payload and the
    deterministic uuid5 point id are byte-identical either way.
  * The service applies `normalize_for_embedding` server-side, and that
    function is idempotent over the chunk texts (verified across all 219 law-159
    chunks), so sending an already-normalized `text_for_embedding` yields the
    same vector as embedding it locally would.

Usage:
    python ingest_via_service.py chunks_law159.json --cloud
    python ingest_via_service.py chunks_law159.json --qdrant-url http://127.0.0.1:6333

There is deliberately no `--recreate`: this script appends a law to a
collection that already holds others, and dropping them is not something it
should be able to do by accident.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(SP.parent / ".env")

import config  # noqa: E402
from embeddings import HybridVec, SparseVec  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from vector_store import LawVectorStore  # noqa: E402


def _log(message: str) -> None:
    print(message, flush=True)


def _cloud_client() -> tuple[QdrantClient, str]:
    url, key = os.environ["QDRANT_CLOUD_URL"], os.environ["QDRANT_CLOUD_API_KEY"]
    collection = os.environ.get("QDRANT_COLLECTION_NAME", config.COLLECTION_NAME)
    return QdrantClient(url=url, api_key=key, timeout=120), collection


def ingest(chunks_path: Path, client: QdrantClient, collection: str, batch_size: int) -> int:
    from legal_assistant.embedding_client import EmbeddingClient

    chunks: list[dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
    _log(f"[1/4] {len(chunks)} chunks loaded from {chunks_path.name}")
    if not chunks:
        _log("ERROR: no chunks to ingest.")
        raise SystemExit(2)

    embedder = EmbeddingClient()
    if not embedder.health():
        _log("ERROR: embedding service is not healthy.")
        raise SystemExit(2)
    _log("[2/4] embedding service healthy")

    store = LawVectorStore(client, collection)
    before = client.get_collection(collection).points_count
    _log(f"[3/4] collection {collection!r} holds {before} points before ingest")

    total = 0
    started = time.time()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        results = embedder.embed([c["text_for_embedding"] for c in batch])
        vectors = [
            HybridVec(
                dense=r.dense,
                sparse=SparseVec(indices=r.sparse.indices, values=r.sparse.values),
            )
            for r in results
        ]
        total += store.upsert(batch, vectors)
        _log(f"      upserted {total}/{len(chunks)}")

    after = client.get_collection(collection).points_count
    _log(f"[4/4] done in {time.time() - started:.1f}s. "
         f"upserted={total}, points {before} -> {after} (+{after - before})")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed chunks via the service and upsert.")
    parser.add_argument("chunks", type=Path)
    parser.add_argument("--cloud", action="store_true", help="target QDRANT_CLOUD_URL")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.cloud:
        client, collection = _cloud_client()
    elif args.qdrant_url:
        client = QdrantClient(url=args.qdrant_url, timeout=120)
        collection = args.collection or config.COLLECTION_NAME
    else:
        parser.error("pass --cloud or --qdrant-url")

    try:
        ingest(args.chunks, client, args.collection or collection, args.batch_size)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
