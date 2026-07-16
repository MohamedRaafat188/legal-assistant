"""Prove the full retrieval path against Cloud: hybrid search -> ColBERT rerank,
plus exact-lookup and the مواد الإصدار (enacting-provision) edge case.

For each operator-provided test query: embed via /embed -> Qdrant hybrid
search (dense + sparse prefetch, RRF fusion) for 20 candidates -> /rerank ->
top 5. Reports whether an expected article appears in top-5 and at what rank.

Also captures rerank latency (wall-clock, as the caller would see it) and,
via SSH to the Hetzner VPS, peak container memory during the run -- the
empirical basis for the 4 GB-stay-or-8 GB-upsize decision.

Usage:
    python scripts/check_retrieval.py [--vps-host root@157.180.119.126] [--ssh-key path]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legal_assistant.config import get_settings
from legal_assistant.db.qdrant import get_cloud_client
from legal_assistant.embedding_client import EmbeddingClient
from qdrant_client.http import models

RERANK_CANDIDATES = 20
TOP_K = 5


@dataclass
class TestQuery:
    query: str
    law_number: int
    expected_articles: list[int]
    note: str = ""
    # Matches arabic_ingest/retrieval.py's smart_search() routing: a query
    # that names an explicit article number ("المادة 5 من...", "نص المادة
    # 500...") is routed to Retriever.lookup_article (exact metadata filter),
    # never to ranked search -- query_intent.py's own docstring explains why:
    # a bare article number carries almost no lexical/semantic signal once
    # normalized, so hybrid search routinely misranks these. Only queries
    # with no explicit article reference go through hybrid search + rerank.
    route: str = "semantic"  # "semantic" | "exact"


TEST_QUERIES: list[TestQuery] = [
    TestQuery("المادة 5 من القانون المدني", law_number=131, expected_articles=[5], route="exact"),
    TestQuery(
        "كيف يتعامل القانون المصري مع تنازع القوانين عبر الزمان",
        law_number=131,
        expected_articles=[6, 7, 8, 9],
        route="semantic",
    ),
    TestQuery(
        "ايه الاجراءات الماخوذة في حالة تنازع الاختصاص",
        law_number=174,
        expected_articles=[222, 223, 224, 225, 226, 227],
        route="semantic",
    ),
    TestQuery(
        "ما هو نص المادة 500 من قانون الاجراءات الجنائية؟",
        law_number=174,
        expected_articles=[500],
        route="exact",
    ),
    TestQuery(
        "ما هو نص المادة 70 من القانون المدني؟",
        law_number=131,
        expected_articles=[70],
        note="expected article is repealed (article_status=repealed) -- should still surface",
        route="exact",
    ),
]


@dataclass
class QueryTiming:
    query: str
    rerank_latency_s: float
    found_rank: int | None
    expected_hit: int | None
    route: str = "semantic"


def hybrid_search(cloud_client, collection_name: str, dense: list[float], sparse_indices: list[int],
                   sparse_values: list[float], limit: int) -> list[models.ScoredPoint]:
    """Dense + sparse retrieval fused with RRF (matches vector_store.py's hybrid_search)."""
    response = cloud_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=limit),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return response.points


def run_query_tests(cloud_client, client: EmbeddingClient, collection_name: str) -> list[QueryTiming]:
    timings: list[QueryTiming] = []

    for tq in TEST_QUERIES:
        print(f"\n{'=' * 70}\nQuery: {tq.query}")
        print(f"Expected: law_number={tq.law_number}, articles={tq.expected_articles} {tq.note}")

        if tq.route == "exact":
            # Matches smart_search()'s routing for an explicit article
            # reference: exact metadata filter, no embedding/rerank involved.
            start = time.perf_counter()
            hits, _ = cloud_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="law_number", match=models.MatchValue(value=tq.law_number)),
                        models.FieldCondition(
                            key="article_number", match=models.MatchValue(value=tq.expected_articles[0])
                        ),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            latency = time.perf_counter() - start
            found = bool(hits) and hits[0].payload.get("article_number") in tq.expected_articles
            status = "PASS" if found else "FAIL"
            if hits:
                print(f"  [exact-lookup] found: {hits[0].payload.get('citation_label')} "
                      f"(status={hits[0].payload.get('article_status')})")
            print(f"  [{status}] exact-lookup route, latency={latency:.3f}s")
            timings.append(QueryTiming(tq.query, latency, 1 if found else None, 1 if found else 0, route="exact"))
            continue

        embedded = client.embed([tq.query])[0]
        candidates = hybrid_search(
            cloud_client, collection_name, embedded.dense, embedded.sparse.indices, embedded.sparse.values,
            limit=RERANK_CANDIDATES,
        )
        if not candidates:
            print("  FAIL: hybrid search returned zero candidates.")
            timings.append(QueryTiming(tq.query, 0.0, None, None))
            continue

        candidate_texts = [c.payload.get("text_for_display", c.payload.get("body_faithful", "")) for c in candidates]

        start = time.perf_counter()
        rerank_results = client.rerank(tq.query, candidate_texts)
        latency = time.perf_counter() - start

        top5 = rerank_results[:TOP_K]
        top5_articles = [
            (candidates[r.index].payload.get("article_number"), candidates[r.index].payload.get("law_number"), r.score)
            for r in top5
        ]
        print(f"  Top-{TOP_K} after rerank (article_number, law_number, score):")
        for art, law, score in top5_articles:
            marker = " <-- expected" if art in tq.expected_articles and law == tq.law_number else ""
            print(f"    {art} (law {law}) score={score:.4f}{marker}")

        found_rank = None
        for rank, (art, law, _score) in enumerate(top5_articles, start=1):
            if art in tq.expected_articles and law == tq.law_number:
                found_rank = rank
                break

        expected_hit = 1 if found_rank is not None else 0
        status = "PASS" if found_rank else "FAIL"
        print(f"  [{status}] expected article in top-{TOP_K}: rank={found_rank}")
        print(f"  rerank latency: {latency:.3f}s ({len(candidate_texts)} passages)")

        timings.append(QueryTiming(tq.query, latency, found_rank, expected_hit))

    return timings


def exact_lookup_check(cloud_client, collection_name: str) -> bool:
    print(f"\n{'=' * 70}\nExact-lookup check (article_number + law_number filter)")
    ok = True

    # Known article: law 174, article 500.
    known, _ = cloud_client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="law_number", match=models.MatchValue(value=174)),
                models.FieldCondition(key="article_number", match=models.MatchValue(value=500)),
            ]
        ),
        limit=1,
        with_payload=True,
    )
    if known and known[0].payload.get("article_number") == 500:
        print("  [PASS] known article (law 174, art 500) found via exact filter.")
    else:
        print("  [FAIL] known article (law 174, art 500) NOT found via exact filter.")
        ok = False

    # Non-existent article number: should return empty, not error.
    try:
        missing, _ = cloud_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="law_number", match=models.MatchValue(value=174)),
                    models.FieldCondition(key="article_number", match=models.MatchValue(value=999999)),
                ]
            ),
            limit=1,
        )
        if not missing:
            print("  [PASS] non-existent article number returns empty result, no error.")
        else:
            print("  [FAIL] non-existent article number unexpectedly returned a result.")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] non-existent article number raised an error instead of returning empty: {exc}")
        ok = False

    return ok


def enacting_provision_check(cloud_client, client: EmbeddingClient, collection_name: str) -> bool:
    print(f"\n{'=' * 70}\nمواد الإصدار (enacting-provision) semantic-retrieval check")

    issuance_points, _ = cloud_client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="article_type", match=models.MatchValue(value="issuance"))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not issuance_points:
        print("  FAIL: no issuance-type (مواد الإصدار) points found in Cloud.")
        return False

    target = issuance_points[0]
    target_id = target.id
    body = target.payload.get("body_faithful", "")
    print(f"  Target: {target.payload.get('citation_label')} (article_number={target.payload.get('article_number')})")

    if target.payload.get("article_number") is not None:
        print("  NOTE: this issuance point unexpectedly has an article_number; check is still valid but less illustrative.")

    # Use the article's own faithful body as a pseudo-query proxy for semantic
    # self-retrieval (weaker than an independent lawyer query, but this is
    # only exercising the "findable without a number" property).
    embedded = client.embed([body])[0]
    candidates = hybrid_search(
        cloud_client, collection_name, embedded.dense, embedded.sparse.indices, embedded.sparse.values,
        limit=RERANK_CANDIDATES,
    )
    found = any(c.id == target_id for c in candidates)
    if found:
        print(f"  [PASS] enacting provision retrievable semantically (found in top {RERANK_CANDIDATES} via hybrid search).")
    else:
        print(f"  [FAIL] enacting provision NOT found in top {RERANK_CANDIDATES} via hybrid search.")

    # Exact-lookup by article_number should gracefully find nothing for this
    # provision, since it has no article_number.
    if target.payload.get("article_number") is None:
        by_number, _ = cloud_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="law_number", match=models.MatchValue(value=target.payload["law_number"])),
                    models.FieldCondition(key="article_number", match=models.MatchValue(value=-1)),
                ]
            ),
            limit=1,
        )
        if not by_number:
            print("  [PASS] exact-lookup by article_number gracefully returns nothing for a number-less provision.")
        else:
            print("  [FAIL] exact-lookup unexpectedly returned a result for a sentinel article_number.")
            found = found and False

    return found


def measure_vps_memory(vps_host: str | None, ssh_key: str | None, duration_s: int = 60) -> str | None:
    """Poll `docker stats` on the VPS in the background; return the log path."""
    if not vps_host:
        return None
    log_path = "/tmp/embedding_memlog.txt"
    ssh_cmd = ["ssh"]
    if ssh_key:
        ssh_cmd += ["-i", ssh_key]
    ssh_cmd += [vps_host]
    remote_script = (
        f"rm -f {log_path}; "
        f"for i in $(seq 1 {duration_s * 4}); do "
        f"  date +%s.%N >> {log_path}; "
        f"  docker stats --no-stream --format '{{{{.MemUsage}}}}' legal-assistant-embedding >> {log_path}; "
        f"  sleep 0.25; "
        f"done"
    )
    subprocess.Popen(ssh_cmd + [remote_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return log_path


def fetch_vps_memory_report(vps_host: str | None, ssh_key: str | None, log_path: str | None) -> None:
    if not vps_host or not log_path:
        print("\n(no VPS host provided -- skipping remote memory measurement)")
        return
    ssh_cmd = ["ssh"]
    if ssh_key:
        ssh_cmd += ["-i", ssh_key]
    ssh_cmd += [vps_host, f"cat {log_path}"]
    out = None
    for attempt in range(3):
        try:
            out = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30).stdout
            break
        except subprocess.TimeoutExpired:
            time.sleep(5)  # the background monitor's ssh session can transiently hold the connection
    if out is None:
        print("\nCould not fetch VPS memory log after retries.")
        return

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    mem_values_mib: list[float] = []
    for line in lines:
        if "/" not in line:
            continue
        used = line.split("/")[0].strip()
        try:
            if used.endswith("GiB"):
                mem_values_mib.append(float(used[:-3]) * 1024)
            elif used.endswith("MiB"):
                mem_values_mib.append(float(used[:-3]))
        except ValueError:
            continue

    print(f"\n{'=' * 70}\n4 GB memory verdict")
    if not mem_values_mib:
        print("  No memory samples captured -- cannot report peak RSS.")
        return

    peak = max(mem_values_mib)
    ceiling_mib = 3800  # container mem_limit
    margin_pct = (ceiling_mib - peak) / ceiling_mib * 100
    print(f"  Samples: {len(mem_values_mib)}, peak container memory: {peak:.0f} MiB")
    print(f"  Ceiling (container mem_limit): {ceiling_mib} MiB (margin: {margin_pct:.1f}%)")

    if peak >= ceiling_mib * 0.9:
        print("  VERDICT: peak memory is near the ceiling -- recommend upsizing to 8 GB (CX32/CPX31).")
    else:
        print("  VERDICT: comfortable margin under the 4 GB ceiling -- staying on 4 GB (CX22) is viable.")
    print("  NOTE: check `docker stats` swap/host `free -h` separately to confirm swap was not touched under load.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vps-host", default="root@157.180.119.126", help="SSH target for memory measurement, or '' to skip")
    parser.add_argument("--ssh-key", default=None, help="Path to SSH private key")
    args = parser.parse_args()

    settings = get_settings()
    cloud_client = get_cloud_client(settings)
    client = EmbeddingClient(settings)
    collection_name = settings.qdrant_collection_name

    if not client.health():
        print("FAIL: embedding service /health reports model not loaded.")
        raise SystemExit(1)

    log_path = measure_vps_memory(args.vps_host or None, args.ssh_key, duration_s=60)

    timings = run_query_tests(cloud_client, client, collection_name)
    exact_ok = exact_lookup_check(cloud_client, collection_name)
    issuance_ok = enacting_provision_check(cloud_client, client, collection_name)

    time.sleep(2)  # let the memory poller catch the tail of the last request
    fetch_vps_memory_report(args.vps_host or None, args.ssh_key, log_path)

    print(f"\n{'=' * 70}\nSummary")
    n_hit = sum(1 for t in timings if t.found_rank)
    print(f"  Query retrieval: {n_hit}/{len(timings)} queries found expected article "
          f"(top-{TOP_K} for semantic routes, exact match for exact routes)")
    for t in timings:
        rank_str = f"rank {t.found_rank}" if t.found_rank else "NOT FOUND"
        print(f"    - [{t.route}] {t.query[:50]!r}: {rank_str}, latency={t.rerank_latency_s:.3f}s")

    rerank_latencies = sorted(t.rerank_latency_s for t in timings if t.route == "semantic")
    if rerank_latencies:
        p50 = rerank_latencies[len(rerank_latencies) // 2]
        p95 = rerank_latencies[min(len(rerank_latencies) - 1, int(len(rerank_latencies) * 0.95))]
        print(f"  Rerank latency (semantic-route queries, {RERANK_CANDIDATES} passages each): "
              f"p50={p50:.3f}s p95={p95:.3f}s (n={len(rerank_latencies)})")
    exact_latencies = [t.rerank_latency_s for t in timings if t.route == "exact"]
    if exact_latencies:
        print(f"  Exact-lookup latency (article-reference queries): "
              f"avg={sum(exact_latencies) / len(exact_latencies):.3f}s (n={len(exact_latencies)})")
    print(f"  Exact-lookup path: {'PASS' if exact_ok else 'FAIL'}")
    print(f"  Enacting-provision semantic retrieval: {'PASS' if issuance_ok else 'FAIL'}")

    if n_hit < len(timings) or not exact_ok or not issuance_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
