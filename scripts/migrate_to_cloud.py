"""Copy the validated `egyptian_law` vectors from the local Docker Qdrant to Qdrant Cloud.

This is a byte-identical copy: vectors and payload are read from the source and
upserted into Cloud verbatim. Nothing is re-embedded. The source is treated as
strictly read-only.

Usage:
    python scripts/migrate_to_cloud.py [--recreate] [--batch-size 256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qdrant_client.http import models
from tqdm import tqdm

from legal_assistant.config import get_settings
from legal_assistant.db.qdrant import get_cloud_client, get_source_client


def build_create_collection_kwargs(source_config: models.CollectionInfo) -> dict:
    """Translate the source collection's live config into create_collection kwargs.

    Replicates named dense vectors, sparse vectors, HNSW config, on_disk_payload,
    and quantization exactly as read from the source. Nothing is "improved".
    """
    params = source_config.config.params

    vectors_config: dict[str, models.VectorParams] = {}
    if params.vectors:
        if isinstance(params.vectors, dict):
            for name, vp in params.vectors.items():
                vectors_config[name] = models.VectorParams(
                    size=vp.size,
                    distance=vp.distance,
                    hnsw_config=vp.hnsw_config,
                    quantization_config=vp.quantization_config,
                    on_disk=vp.on_disk,
                    multivector_config=vp.multivector_config,
                )
        else:
            # Single unnamed vector (not expected for this collection, handled for completeness).
            vectors_config = params.vectors  # type: ignore[assignment]

    sparse_vectors_config: dict[str, models.SparseVectorParams] | None = None
    if params.sparse_vectors:
        sparse_vectors_config = {
            name: models.SparseVectorParams(
                index=svp.index,
                modifier=svp.modifier,
            )
            for name, svp in params.sparse_vectors.items()
        }

    hnsw = source_config.config.hnsw_config
    optimizers = source_config.config.optimizer_config
    wal = source_config.config.wal_config

    return {
        "vectors_config": vectors_config,
        "sparse_vectors_config": sparse_vectors_config,
        "shard_number": params.shard_number,
        "replication_factor": params.replication_factor,
        "write_consistency_factor": params.write_consistency_factor,
        "on_disk_payload": params.on_disk_payload,
        "hnsw_config": models.HnswConfigDiff(**hnsw.model_dump()) if hnsw else None,
        "optimizers_config": models.OptimizersConfigDiff(**optimizers.model_dump()) if optimizers else None,
        "wal_config": models.WalConfigDiff(**wal.model_dump()) if wal else None,
        "quantization_config": source_config.config.quantization_config,
    }


def collection_config_matches(source_config: models.CollectionInfo, cloud_config: models.CollectionInfo) -> bool:
    """Best-effort structural comparison of two collections' vector configs."""
    src_params = source_config.config.params
    dst_params = cloud_config.config.params

    src_vec_names = set(src_params.vectors.keys()) if isinstance(src_params.vectors, dict) else set()
    dst_vec_names = set(dst_params.vectors.keys()) if isinstance(dst_params.vectors, dict) else set()
    if src_vec_names != dst_vec_names:
        return False

    src_sparse_names = set(src_params.sparse_vectors.keys()) if src_params.sparse_vectors else set()
    dst_sparse_names = set(dst_params.sparse_vectors.keys()) if dst_params.sparse_vectors else set()
    if src_sparse_names != dst_sparse_names:
        return False

    for name in src_vec_names:
        if src_params.vectors[name].size != dst_params.vectors[name].size:
            return False
        if src_params.vectors[name].distance != dst_params.vectors[name].distance:
            return False

    return True


def ensure_cloud_collection(
    source_client,
    cloud_client,
    collection_name: str,
    recreate: bool,
) -> None:
    """Create the Cloud collection mirroring the source config, unless it already matches."""
    source_config = source_client.get_collection(collection_name)

    cloud_exists = cloud_client.collection_exists(collection_name)

    if cloud_exists and not recreate:
        cloud_config = cloud_client.get_collection(collection_name)
        if collection_config_matches(source_config, cloud_config):
            print(f"Cloud collection '{collection_name}' already exists with matching config. Skipping creation.")
            return
        raise RuntimeError(
            f"Cloud collection '{collection_name}' exists but its config does not match the source. "
            "Re-run with --recreate to replace it (destructive)."
        )

    if cloud_exists and recreate:
        print(f"--recreate passed: deleting existing Cloud collection '{collection_name}'.")
        cloud_client.delete_collection(collection_name)

    kwargs = build_create_collection_kwargs(source_config)
    print(f"Creating Cloud collection '{collection_name}' with source-mirrored config...")
    cloud_client.create_collection(collection_name=collection_name, **kwargs)


def copy_points(source_client, cloud_client, collection_name: str, batch_size: int) -> int:
    """Scroll all points from source and upsert into Cloud in batches. Returns points copied."""
    total_copied = 0
    next_offset = None

    source_count = source_client.count(collection_name, exact=True).count
    progress = tqdm(total=source_count, desc="Copying points", unit="pt")

    while True:
        points, next_offset = source_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=next_offset,
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            break

        cloud_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )
        total_copied += len(points)
        progress.update(len(points))

        if next_offset is None:
            break

    progress.close()
    return total_copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Cloud collection if it exists (destructive).",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Upsert batch size (128-256 recommended).")
    args = parser.parse_args()

    settings = get_settings()
    source_client = get_source_client(settings)
    cloud_client = get_cloud_client(settings)
    collection_name = settings.qdrant_collection_name

    # Hard gate: verify the source before touching anything.
    if not source_client.collection_exists(collection_name):
        print(f"FATAL: source collection '{collection_name}' does not exist. Aborting.")
        raise SystemExit(1)

    source_count = source_client.count(collection_name, exact=True).count
    if source_count == 0:
        print(f"FATAL: source collection '{collection_name}' has zero points. Aborting.")
        raise SystemExit(1)

    print(f"Source '{collection_name}': {source_count} points confirmed. Proceeding (source is read-only).")

    ensure_cloud_collection(source_client, cloud_client, collection_name, args.recreate)

    copied = copy_points(source_client, cloud_client, collection_name, args.batch_size)
    print(f"Copied {copied} points to Cloud collection '{collection_name}'.")

    cloud_count = cloud_client.count(collection_name, exact=True).count
    print(f"Cloud collection now holds {cloud_count} points.")

    if cloud_count != source_count:
        print("WARNING: cloud point count does not match source. Run scripts/verify_migration.py for details.")
        raise SystemExit(1)

    print("Migration complete. Run scripts/verify_migration.py to validate.")


if __name__ == "__main__":
    main()
