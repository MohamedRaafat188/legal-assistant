"""Recreate payload indexes on Qdrant Cloud to match the source collection.

Phase 1's migration copied points + vectors but not payload indexes. The Cloud
cluster has strict_mode_config.enabled=true, which can reject filtering on
unindexed fields. This script reads the source's indexed fields and creates
matching indexes on Cloud, then proves filtering works under strict mode.

Usage:
    python scripts/fix_payload_indexes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qdrant_client.http import models

from legal_assistant.config import get_settings
from legal_assistant.db.qdrant import get_cloud_client, get_source_client

# Matches vector_store.py's ensure_collection() in the ingestion project exactly.
EXPECTED_INDEXES: dict[str, models.PayloadSchemaType] = {
    "law_number": models.PayloadSchemaType.INTEGER,
    "law_year": models.PayloadSchemaType.INTEGER,
    "article_number": models.PayloadSchemaType.INTEGER,
    "division_number": models.PayloadSchemaType.INTEGER,
    "book_number": models.PayloadSchemaType.INTEGER,
    "article_status": models.PayloadSchemaType.KEYWORD,
    "article_type": models.PayloadSchemaType.KEYWORD,
}


def main() -> None:
    settings = get_settings()
    source_client = get_source_client(settings)
    cloud_client = get_cloud_client(settings)
    collection_name = settings.qdrant_collection_name

    source_info = source_client.get_collection(collection_name)
    source_fields = set(source_info.payload_schema.keys())
    print(f"Source indexed fields: {sorted(source_fields)}")

    expected_fields = set(EXPECTED_INDEXES.keys())
    if source_fields != expected_fields:
        print(f"NOTE: source fields differ from the known-good set. Source only: "
              f"{source_fields - expected_fields}, expected only: {expected_fields - source_fields}")

    cloud_info = cloud_client.get_collection(collection_name)
    cloud_fields = set(cloud_info.payload_schema.keys())
    print(f"Cloud indexed fields (before fix): {sorted(cloud_fields)}")

    missing = expected_fields - cloud_fields
    if not missing:
        print("Cloud already has all expected payload indexes. Nothing to do.")
    else:
        print(f"Creating missing indexes on Cloud: {sorted(missing)}")
        for field in missing:
            cloud_client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=EXPECTED_INDEXES[field],
            )
            print(f"  created index: {field} ({EXPECTED_INDEXES[field]})")

    cloud_info_after = cloud_client.get_collection(collection_name)
    print(f"Cloud indexed fields (after fix): {sorted(cloud_info_after.payload_schema.keys())}")

    # Prove filtering works under strict mode: pick a known point from source.
    sample, _ = source_client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="article_number", match=models.MatchValue(value=221))]
        ),
        limit=1,
        with_payload=True,
    )
    if not sample:
        print("FAIL: could not find a sample point (article_number=221) on source to test with.")
        raise SystemExit(1)

    law_number = sample[0].payload["law_number"]
    article_number = sample[0].payload["article_number"]
    print(f"Testing filtered scroll on Cloud: article_number={article_number}, law_number={law_number}")

    try:
        result, _ = cloud_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="article_number", match=models.MatchValue(value=article_number)),
                    models.FieldCondition(key="law_number", match=models.MatchValue(value=law_number)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
    except Exception as exc:  # noqa: BLE001 - we want to report any strict-mode rejection clearly
        print(f"FAIL: filtered scroll raised an error under strict mode: {exc}")
        raise SystemExit(1) from exc

    if not result or result[0].payload.get("article_number") != article_number:
        print("FAIL: filtered scroll on Cloud did not return the expected point.")
        raise SystemExit(1)

    print(f"PASS: filtered scroll returned the correct point (chunk_id={result[0].payload.get('chunk_id')}).")
    print("Cloud payload indexes fixed and filtering proven under strict mode.")


if __name__ == "__main__":
    main()
