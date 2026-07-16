"""Prove the embedding service reproduces the vectors already stored in Cloud.

The entire "no re-embed" strategy rests on the endpoint being numerically
consistent with the corpus. This script samples points from Cloud, rebuilds
their exact ingestion-time embedding input (header + body_faithful, per
chunker.py's `embedding = normalize_for_embedding(f"{header}\\n{rec.body}")`),
sends that raw text to /embed (the service applies normalize_for_embedding
itself), and compares the result to the stored vectors.

Usage:
    python scripts/check_embedding_consistency.py [--sample-size 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_assistant.config import get_settings
from legal_assistant.db.qdrant import get_cloud_client
from legal_assistant.embedding_client import EmbeddingClient
from qdrant_client.http import models

DENSE_PASS_THRESHOLD = 0.999


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sparse_overlap(
    stored_indices: list[int], stored_values: list[float], live_indices: list[int], live_values: list[float]
) -> tuple[float, int, int]:
    """Return (jaccard index overlap, |stored|, |live|) between sparse vectors' index sets."""
    stored_set = set(stored_indices)
    live_set = set(live_indices)
    if not stored_set and not live_set:
        return 1.0, 0, 0
    union = stored_set | live_set
    intersection = stored_set & live_set
    jaccard = len(intersection) / len(union) if union else 1.0
    return jaccard, len(stored_set), len(live_set)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    cloud_client = get_cloud_client(settings)
    client = EmbeddingClient(settings)
    collection_name = settings.qdrant_collection_name

    if not client.health():
        print("FAIL: embedding service /health reports model not loaded.")
        raise SystemExit(1)

    # Sample points that DO have article_number (excludes enacting provisions).
    sample_points, _ = cloud_client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="article_number", range=models.Range(gte=1))]
        ),
        limit=args.sample_size,
        with_vectors=True,
        with_payload=True,
    )

    if not sample_points:
        print("FAIL: no points with article_number found in Cloud.")
        raise SystemExit(1)

    print(f"Sampled {len(sample_points)} points with article_number set.\n")

    results = []
    for point in sample_points:
        payload = point.payload
        header = payload["header"]
        body_faithful = payload["body_faithful"]
        # Exact ingestion-time embedding input, per chunker.py's build_chunks():
        #   embedding = normalize_for_embedding(f"{header}\n{rec.body}")
        # /embed applies normalize_for_embedding itself, so we send the raw
        # concatenation and let the service normalize it identically.
        raw_input_text = f"{header}\n{body_faithful}"

        live = client.embed([raw_input_text])[0]

        stored_dense = point.vector["dense"]
        stored_sparse = point.vector["sparse"]

        dense_cos = cosine_similarity(stored_dense, live.dense)
        sparse_jaccard, n_stored, n_live = sparse_overlap(
            stored_sparse.indices, stored_sparse.values, live.sparse.indices, live.sparse.values
        )

        dense_pass = dense_cos >= DENSE_PASS_THRESHOLD
        sparse_pass = sparse_jaccard >= 0.9  # indices should align closely if model+tokenizer match

        chunk_id = payload.get("chunk_id", point.id)
        status = "PASS" if (dense_pass and sparse_pass) else "FAIL"
        print(
            f"[{status}] {chunk_id}: dense_cosine={dense_cos:.6f} "
            f"sparse_jaccard={sparse_jaccard:.3f} (stored={n_stored} idx, live={n_live} idx)"
        )
        results.append((chunk_id, dense_pass, sparse_pass))

    print()
    n_pass = sum(1 for _, d, s in results if d and s)
    n_total = len(results)
    if n_pass == n_total:
        print(f"RESULT: PASS ({n_pass}/{n_total}) — embedding service is numerically consistent with stored vectors.")
    else:
        print(f"RESULT: FAIL ({n_pass}/{n_total} passed). STOP — reconcile model/normalization before proceeding.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
