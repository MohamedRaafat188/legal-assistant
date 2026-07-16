# -*- coding: utf-8 -*-
"""Central configuration for the RAG core (embeddings + vector store).

Values are read from environment variables with sensible defaults, so the same
code runs locally (in-memory / localhost Qdrant) and in deployment (Railway)
without edits.
"""
from __future__ import annotations

import os

# --- Embedding model (BGE-M3) ----------------------------------------------
# BGE-M3 is hybrid: one pass yields a dense (semantic) vector AND a sparse
# (lexical) vector. Dense dimension is fixed at 1024.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DENSE_DIM: int = 1024
USE_FP16: bool = os.getenv("BGE_USE_FP16", "1") == "1"  # set 0 on CPU-only boxes

# Named vectors inside each Qdrant point.
DENSE_VECTOR: str = "dense"
SPARSE_VECTOR: str = "sparse"

# --- Qdrant ----------------------------------------------------------------
# For local dev you can also pass ":memory:" as the URL to run fully embedded.
# Default is an EMBEDDED on-disk store (no server/Docker needed for local dev).
# In production set QDRANT_URL to a running server, e.g. http://localhost:6333
# or your Railway Qdrant URL.
QDRANT_URL: str = os.getenv("QDRANT_URL", "./qdrant_storage")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "egyptian_law")
# HTTP timeout (seconds) for server mode. Generous, because a first
# create_collection on a Windows Docker volume can be slow.
QDRANT_TIMEOUT: float = float(os.getenv("QDRANT_TIMEOUT", "60"))

# Ingestion batch size (chunks embedded/upserted per round trip).
BATCH_SIZE: int = int(os.getenv("INGEST_BATCH_SIZE", "32"))
