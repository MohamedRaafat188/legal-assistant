# -*- coding: utf-8 -*-
"""BGE-M3 hybrid embedder.

BGE-M3 returns, in a single forward pass, a dense semantic vector and a sparse
lexical vector (a bag of weighted token ids). We use both: dense for meaning,
sparse for exact legal-term matches (article numbers, fixed phrases like
الحبس الاحتياطي). ColBERT multi-vectors are also available from the model but
are deferred — dense+sparse hybrid is the sweet spot; ColBERT is a future
reranking option.

IMPORTANT — normalization consistency: this class is a *pure encoder*. It does
NOT apply the Arabic normalization. Callers must pass text that has already been
normalized the same way on both sides:
  * documents  -> chunk.text_for_embedding (already normalized by the chunker)
  * queries    -> normalize_for_embedding(query) before calling encode_query
If the two sides are normalized differently, retrieval silently degrades.

The heavy model is loaded lazily on first use, so importing this module is cheap
and does not trigger a model download.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# --- Windows workaround, must run before FlagEmbedding is imported ---------
# Some Windows machines have a malformed certificate in the Windows cert
# store that makes ssl.create_default_context() raise (or, in the worst
# case, crash the process with an access violation) with
# "SSLError: [ASN1: NOT_ENOUGH_DATA]". FlagEmbedding pulls in aiohttp (via
# its `datasets` dependency), and aiohttp builds a default SSL context at
# *import time* -- so this can abort/crash the import. Route
# create_default_context through certifi's CA bundle instead of the Windows
# store to avoid it. Same fix as ingest.py.
import ssl
import certifi

_orig_create_default_context = ssl.create_default_context


def _create_default_context_via_certifi(*args, **kwargs):
    kwargs.setdefault("cafile", certifi.where())
    return _orig_create_default_context(*args, **kwargs)


ssl.create_default_context = _create_default_context_via_certifi

import config


@dataclass(frozen=True, slots=True)
class SparseVec:
    """A sparse embedding as parallel index/value lists (Qdrant's format)."""
    indices: list[int]
    values: list[float]


@dataclass(frozen=True, slots=True)
class HybridVec:
    """One item's hybrid embedding."""
    dense: list[float]
    sparse: SparseVec


class BGEM3Embedder:
    """Thin wrapper over FlagEmbedding's BGE-M3 producing dense + sparse."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 use_fp16: bool = config.USE_FP16) -> None:
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._model = None  # lazy

    # -- model loading -----------------------------------------------------
    def _ensure_model(self):
        if self._model is None:
            # Imported here so the dependency is only needed at encode time.
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self._model_name, use_fp16=self._use_fp16)
        return self._model

    def warmup(self) -> None:
        """Force-load the model now.

        On Windows, FlagEmbedding's import chain pulls in pyarrow (via
        `datasets`), whose bundled native Arrow library conflicts with
        qdrant-client's Rust extension if the Qdrant client is constructed
        first in the same process (crashes with an access violation on
        first query). Call this before constructing the Qdrant client to
        force the safe import order.
        """
        self._ensure_model()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _to_sparse(lexical_weights: dict) -> SparseVec:
        """Convert BGE-M3 lexical weights {token_id: weight} to index/value lists."""
        if not lexical_weights:
            return SparseVec(indices=[], values=[])
        indices = [int(k) for k in lexical_weights.keys()]
        values = [float(v) for v in lexical_weights.values()]
        return SparseVec(indices=indices, values=values)

    def _encode(self, texts: list[str]) -> list[HybridVec]:
        model = self._ensure_model()
        out = model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        sparse = out["lexical_weights"]
        results: list[HybridVec] = []
        for i in range(len(texts)):
            results.append(HybridVec(
                dense=[float(x) for x in dense[i]],
                sparse=self._to_sparse(sparse[i]),
            ))
        return results

    # -- public API --------------------------------------------------------
    def encode_documents(self, texts: Iterable[str]) -> list[HybridVec]:
        """Encode already-normalized document texts (chunk.text_for_embedding)."""
        return self._encode(list(texts))

    def encode_query(self, text: str) -> HybridVec:
        """Encode a single already-normalized query string."""
        return self._encode([text])[0]
