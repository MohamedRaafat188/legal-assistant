"""Validate that the Qdrant Cloud migration is a faithful copy of the source.

Checks (all must pass):
  1. Point count in Cloud equals source.
  2. Payload spot-check: sampled points match source exactly.
  3. Vector equality spot-check: sampled points' named vectors match source
     (exact, or cosine similarity ~= 1.0 within floating-point tolerance).

Prints a PASS/FAIL report and exits non-zero on any failure.

Usage:
    python scripts/verify_migration.py [--sample-size 10]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_assistant.config import get_settings
from legal_assistant.db.qdrant import get_cloud_client, get_source_client

COSINE_TOLERANCE = 1e-4


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vectors_match(source_vector, cloud_vector) -> bool:
    """Compare vector payloads, which may be a dict of named dense/sparse vectors."""
    if isinstance(source_vector, dict):
        if set(source_vector.keys()) != set(cloud_vector.keys()):
            return False
        for name, src_v in source_vector.items():
            dst_v = cloud_vector[name]
            if hasattr(src_v, "indices"):  # sparse vector
                if list(src_v.indices) != list(dst_v.indices):
                    return False
                if any(abs(a - b) > COSINE_TOLERANCE for a, b in zip(src_v.values, dst_v.values, strict=True)):
                    return False
            else:  # dense vector
                sim = cosine_similarity(src_v, dst_v)
                if sim < 1.0 - COSINE_TOLERANCE:
                    return False
        return True
    return source_vector == cloud_vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=10, help="Number of points to spot-check (5-10 recommended).")
    args = parser.parse_args()

    settings = get_settings()
    source_client = get_source_client(settings)
    cloud_client = get_cloud_client(settings)
    collection_name = settings.qdrant_collection_name

    failures: list[str] = []

    # 1. Point count check.
    if not cloud_client.collection_exists(collection_name):
        print(f"FAIL: Cloud collection '{collection_name}' does not exist.")
        raise SystemExit(1)

    source_count = source_client.count(collection_name, exact=True).count
    cloud_count = cloud_client.count(collection_name, exact=True).count
    count_ok = source_count == cloud_count
    print(f"[{'PASS' if count_ok else 'FAIL'}] Point count: source={source_count}, cloud={cloud_count}")
    if not count_ok:
        failures.append("point_count_mismatch")

    # Sample point IDs from the source to spot-check.
    sample_points, _ = source_client.scroll(
        collection_name=collection_name,
        limit=args.sample_size,
        with_vectors=False,
        with_payload=False,
    )
    all_ids = [p.id for p in sample_points]
    if len(all_ids) < args.sample_size:
        print(f"NOTE: source only has {len(all_ids)} points available for sampling.")

    sample_ids = random.sample(all_ids, k=min(args.sample_size, len(all_ids))) if all_ids else []

    source_sampled = source_client.retrieve(
        collection_name=collection_name, ids=sample_ids, with_vectors=True, with_payload=True
    )
    cloud_sampled = cloud_client.retrieve(
        collection_name=collection_name, ids=sample_ids, with_vectors=True, with_payload=True
    )
    cloud_by_id = {p.id: p for p in cloud_sampled}

    payload_ok = True
    vector_ok = True
    for src_point in source_sampled:
        cloud_point = cloud_by_id.get(src_point.id)
        if cloud_point is None:
            print(f"  [FAIL] point {src_point.id} missing in Cloud")
            payload_ok = False
            continue

        if src_point.payload != cloud_point.payload:
            print(f"  [FAIL] payload mismatch for point {src_point.id}")
            payload_ok = False

        if not vectors_match(src_point.vector, cloud_point.vector):
            print(f"  [FAIL] vector mismatch for point {src_point.id}")
            vector_ok = False

    print(f"[{'PASS' if payload_ok else 'FAIL'}] Payload spot-check ({len(sample_ids)} points sampled)")
    print(f"[{'PASS' if vector_ok else 'FAIL'}] Vector spot-check ({len(sample_ids)} points sampled)")

    if not payload_ok:
        failures.append("payload_mismatch")
    if not vector_ok:
        failures.append("vector_mismatch")

    print()
    if failures:
        print(f"RESULT: FAIL ({', '.join(failures)})")
        raise SystemExit(1)

    print("RESULT: PASS — Cloud collection is a verified, faithful copy of the source.")


if __name__ == "__main__":
    main()
