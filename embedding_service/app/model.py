"""BGE-M3 model wrapper: dense+sparse encode and ColBERT rerank.

Replicates the ingestion project's ``embeddings.py`` encode configuration
exactly (same model name, same ``use_fp16``/device auto-detection, same
``encode()`` flags, same sparse lexical-weight -> Qdrant sparse-vector
conversion) so that vectors produced here are numerically consistent with
the vectors already stored in Qdrant Cloud. ColBERT (``return_colbert_vecs``)
is used only for the rerank endpoint; it is never stored, only computed at
query time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import get_settings
from .normalize import normalize_for_embedding

_log = logging.getLogger(__name__)


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


class BGEM3Service:
    """Loads BGE-M3 once and serves embed + rerank requests."""

    def __init__(self) -> None:
        self._model = None
        settings = get_settings()
        self._model_name = settings.embedding_model
        self._use_fp16 = settings.bge_use_fp16
        self._max_length = settings.bge_max_length
        self._rerank_max_length = settings.rerank_max_length
        self._rerank_batch_size = settings.rerank_batch_size

    def load(self) -> None:
        """Load the BGE-M3 model into memory. Call once at app startup."""
        if self._model is not None:
            return
        from FlagEmbedding import BGEM3FlagModel

        _log.info("Loading BGE-M3 model '%s' (use_fp16=%s)...", self._model_name, self._use_fp16)
        self._model = BGEM3FlagModel(self._model_name, use_fp16=self._use_fp16)
        _log.info("BGE-M3 model loaded on device: %s", self._model.device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- sparse conversion, identical to ingestion's embeddings.py ---------
    @staticmethod
    def _to_sparse(lexical_weights: dict) -> SparseVec:
        """Convert BGE-M3 lexical weights {token_id: weight} to index/value lists."""
        if not lexical_weights:
            return SparseVec(indices=[], values=[])
        indices = [int(k) for k in lexical_weights.keys()]
        values = [float(v) for v in lexical_weights.values()]
        return SparseVec(indices=indices, values=values)

    # -- embed ---------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[HybridVec]:
        """Normalize and encode texts into dense + sparse hybrid vectors.

        Applies the same ``normalize_for_embedding`` used at ingestion before
        encoding, so callers may pass either raw article text or a raw query.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        normalized = [normalize_for_embedding(t) for t in texts]
        out = self._model.encode(
            normalized,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        sparse = out["lexical_weights"]
        results: list[HybridVec] = []
        for i in range(len(normalized)):
            results.append(
                HybridVec(
                    dense=[float(x) for x in dense[i]],
                    sparse=self._to_sparse(sparse[i]),
                )
            )
        return results

    # -- rerank (ColBERT, computed at query time only) -----------------------
    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return a ColBERT relevance score for each passage against the query.

        Encodes the query once, then encodes passages in small internal
        batches (``rerank_batch_size``) to cap the peak-memory spike on a
        constrained box. Truncates to ``rerank_max_length`` (shorter than
        embed's ``bge_max_length``), since rerank vecs are computed fresh
        per query and don't need to match ingestion-time vectors.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        normalized_query = normalize_for_embedding(query)
        query_out = self._model.encode(
            [normalized_query],
            max_length=self._rerank_max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        q_reps = query_out["colbert_vecs"][0]

        scores: list[float] = []
        batch_size = self._rerank_batch_size
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            normalized_batch = [normalize_for_embedding(p) for p in batch]
            batch_out = self._model.encode(
                normalized_batch,
                max_length=self._rerank_max_length,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
            for p_reps in batch_out["colbert_vecs"]:
                score = self._model.colbert_score(q_reps, p_reps)
                scores.append(float(score))
            del batch_out
        return scores


_service: BGEM3Service | None = None


def get_service() -> BGEM3Service:
    """Return the process-wide singleton BGEM3Service instance."""
    global _service
    if _service is None:
        _service = BGEM3Service()
    return _service
