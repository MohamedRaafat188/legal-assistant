# -*- coding: utf-8 -*-
"""Ingestion pipeline: chunks JSON -> BGE-M3 embeddings -> Qdrant.

Usage:
    python ingest.py chunks_law174.json --recreate
    QDRANT_URL=http://127.0.0.1:6333 python ingest.py chunks_law174.json

Logs every stage so a failure is visible (a silent exit almost always means the
BGE-M3 model was killed while loading — usually out of memory). Verifies the
final point count against the number of chunks so a partial/empty load can't
pass unnoticed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# --- Windows workarounds, must run before qdrant_client/FlagEmbedding import --
# 1. Some Windows machines have a malformed certificate in the Windows cert
#    store that makes ssl.create_default_context() raise
#    "SSLError: [ASN1: NOT_ENOUGH_DATA]". FlagEmbedding pulls in aiohttp (via
#    its `datasets` dependency), and aiohttp builds a default SSL context at
#    *import time* — so this can abort the import. Route create_default_context
#    through certifi's CA bundle instead of the Windows store to avoid it.
# 2. On the same class of machine, loading pyarrow's native extension (pulled
#    in transitively by FlagEmbedding -> datasets) *after* qdrant_client's
#    native gRPC extension is already loaded crashes the process with a
#    Windows access violation. Importing FlagEmbedding before qdrant_client
#    avoids the conflicting load order.
import ssl
import certifi

_orig_create_default_context = ssl.create_default_context


def _create_default_context_via_certifi(*args, **kwargs):
    kwargs.setdefault("cafile", certifi.where())
    return _orig_create_default_context(*args, **kwargs)


ssl.create_default_context = _create_default_context_via_certifi


def _log(msg: str) -> None:
    print(msg, flush=True)


def _preflight_flagembedding() -> None:
    """Fail early and clearly if the embedding dependency is missing/broken."""
    try:
        import FlagEmbedding  # noqa: F401
    except Exception as exc:  # ImportError, or a broken torch install
        _log("ERROR: could not import FlagEmbedding (the BGE-M3 dependency).")
        _log(f"       {type(exc).__name__}: {exc}")
        _log("Fix: pip install -r requirements.txt")
        raise SystemExit(2)


# Must happen before `from vector_store import ...` below loads qdrant_client's
# native extension (see workaround #2 above).
_preflight_flagembedding()

import config
from vector_store import LawVectorStore, make_client


def ingest(chunks_path: Path, url: str, collection: str,
           batch_size: int, recreate: bool) -> int:
    import json
    _log(f"[1/5] Loading chunks from {chunks_path} ...")
    chunks: list[dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
    _log(f"      {len(chunks)} chunks loaded.")
    if not chunks:
        _log("ERROR: no chunks to ingest. Did chunker.py run?")
        raise SystemExit(2)

    _log(f"[2/5] Connecting to Qdrant at {url!r} ...")
    client = make_client(url)
    store = LawVectorStore(client, collection)
    store.ensure_collection(recreate=recreate)
    _log(f"      collection '{collection}' ready (recreate={recreate}).")

    _log("[3/5] Loading BGE-M3 model "
         "(FIRST run downloads ~2.3 GB and needs ~4 GB RAM) ...")
    from embeddings import BGEM3Embedder
    embedder = BGEM3Embedder()
    t0 = time.time()
    # Force the model to load now (not lazily mid-loop) so this stage is explicit.
    _ = embedder.encode_query("تهيئة")
    _log(f"      model loaded in {time.time() - t0:.1f}s.")

    _log(f"[4/5] Embedding + upserting in batches of {batch_size} ...")
    total = 0
    try:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = embedder.encode_documents(c["text_for_embedding"] for c in batch)
            total += store.upsert(batch, vectors)
            _log(f"      upserted {total}/{len(chunks)}")
    finally:
        # count is read before closing; close releases the connection cleanly
        try:
            count = client.get_collection(collection).points_count
        except Exception:
            count = None
        client.close()

    _log(f"[5/5] Done. upserted={total}, points in collection={count}.")
    if count is not None and count < len(chunks):
        _log("WARNING: fewer points than chunks — some upserts may have failed.")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed chunks and upsert into Qdrant.")
    ap.add_argument("chunks", type=Path, help="chunks_*.json produced by chunker.py")
    ap.add_argument("--qdrant-url", default=config.QDRANT_URL)
    ap.add_argument("--collection", default=config.COLLECTION_NAME)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--recreate", action="store_true",
                    help="drop and recreate the collection first")
    args = ap.parse_args()

    try:
        ingest(args.chunks, args.qdrant_url, args.collection,
               args.batch_size, args.recreate)
    except SystemExit:
        raise
    except Exception as exc:  # surface any error instead of dying silently
        _log(f"\nINGEST FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
