"""Checks that must hold for chunks_law159.json, and for what landed in Qdrant.

Run with no arguments to check the built chunks only; pass `--cloud` to also
verify every point actually in Qdrant Cloud against what `_to_point` would
build from those chunks.
"""
import json
import sys
from pathlib import Path

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "src"))

chunks = json.loads((SP / "chunks_law159.json").read_text(encoding="utf-8"))
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


meta = {c["chunk_id"]: c["metadata"] for c in chunks}
subs = [c for c in chunks if c["metadata"]["article_type"] == "substantive"]
iss = [c for c in chunks if c["metadata"]["article_type"] == "issuance"]

print("== coverage ==")
plain = sorted(c["metadata"]["article_number"] for c in subs
               if not c["metadata"]["article_suffix"])
check("6 issuance articles", len(iss) == 6, str(len(iss)))
check("articles 1..184 all present", plain == list(range(1, 185)),
      f"missing {sorted(set(range(1, 185)) - set(plain))}")
check("chunk ids unique", len({c["chunk_id"] for c in chunks}) == len(chunks))
# Issuance articles are keyed by citation_label, not by number, so they all
# share (None, None) -- uniqueness only has to hold over the numbered ones.
sub_keys = [(c["metadata"]["article_number"], c["metadata"]["article_suffix"]) for c in subs]
check("(number, suffix) keys unique across substantive articles",
      len(set(sub_keys)) == len(sub_keys), str(len(set(sub_keys))))

print("\n== مكرر series ==")
bis = {c["metadata"]["article_suffix"]: c for c in subs if c["metadata"]["article_suffix"]}
series_129 = sorted(s for s in bis if s.startswith("مكرر") and s != "مكرر"
                    and any(ch.isdigit() or ch in "٠١٢٣٤٥٦٧٨٩" for ch in s))
check("nine numbered siblings under article 129", len(series_129) == 9, str(series_129))
series_135 = sorted(s for s in bis if s in {"مكرر أ", "مكرر ب", "مكرر ج", "مكرر د"})
check("four lettered siblings under article 135", len(series_135) == 4, str(series_135))
check("29 مكرر articles in total",
      sum(1 for c in subs if c["metadata"]["article_suffix"]) == 29)
# The whole point of the suffix widening: eleven separate articles sit on
# article_number 129 (base, مكرر, and مكرر "١".."٩"), and a bis *flag* would
# have merged ten of them into one citation-guard key.
keys_129 = [c["metadata"]["article_suffix"] for c in subs
            if c["metadata"]["article_number"] == 129]
check("article 129's eleven entries all carry distinct suffixes",
      len(keys_129) == 11 and len(set(keys_129)) == 11, str(sorted(keys_129, key=str)))

print("\n== schema parity with laws 131/174 ==")
ref = json.loads((SP / "chunks_law174.json").read_text(encoding="utf-8"))
ref_top, ref_meta = set(ref[0]), set(ref[0]["metadata"])
new_top, new_meta = set(chunks[0]), set(chunks[0]["metadata"])
check("top-level keys identical", ref_top == new_top, str(ref_top ^ new_top))
check("metadata adds only article_suffix + amendments",
      new_meta - ref_meta == {"article_suffix", "amendments"} and not ref_meta - new_meta,
      f"added={new_meta - ref_meta} missing={ref_meta - new_meta}")

print("\n== body integrity ==")
check("no empty bodies", all(len(c["body_faithful"]) > 30 for c in chunks),
      str([c["chunk_id"] for c in chunks if len(c["body_faithful"]) <= 30]))
check("body_faithful is the tail of text_for_display",
      all(c["text_for_display"].endswith(c["body_faithful"]) for c in chunks))
check("header prefixes text_for_display",
      all(c["text_for_display"].startswith(c["header"]) for c in chunks))
check("every citation_label names law 159",
      all("١٥٩" in c["citation_label"] for c in chunks))
check("the explanatory memorandum never entered the corpus",
      not any("مذكرة إيضاحية" in c["body_faithful"] for c in chunks))

print("\n== status ==")
repealed = [c for c in subs if c["metadata"]["article_status"] == "repealed"]
# Nine the source itself marks ملغاة, plus article 94, repealed by law 194/2020
# -- which post-dates the source edition and comes from law159_corrections.
expected = {(21, "مكرر"), (22, None), (23, None), (36, None), (83, None),
            (91, None), (92, None), (93, None), (94, None), (183, None)}
got = {(c["metadata"]["article_number"], c["metadata"]["article_suffix"]) for c in repealed}
check("exactly the ten repealed articles", got == expected, str(got ^ expected))
check("repealed articles are kept, not dropped", len(repealed) == 10)

art94 = next(c for c in subs if c["metadata"]["article_number"] == 94
             and not c["metadata"]["article_suffix"])
check("article 94 is flagged repealed by law 194/2020",
      art94["metadata"]["article_status"] == "repealed"
      and any(r["law_number"] == 194 and r["law_year"] == 2020
              for r in art94["metadata"]["amendments"]),
      str(art94["metadata"]["amendments"]))
check("article 94 keeps its pre-repeal wording, and the note comes first",
      art94["body_faithful"].startswith("ملغاة بالقانون رقم ١٩٤")
      and "لا يجوز لعضو مجلس" in art94["body_faithful"],
      repr(art94["body_faithful"][:90]))
check("article 127 is active (only a phrase in it was struck out)",
      next(c for c in subs if c["metadata"]["article_number"] == 127
           and not c["metadata"]["article_suffix"])["metadata"]["article_status"] == "active")

print("\n== amendments ==")
amended = [c for c in subs if c["metadata"]["amendments"]]
check("amendment notes were parsed", len(amended) >= 75, str(len(amended)))
check("every amendment record names a law and a year",
      all(r["law_number"] and r["law_year"]
          for c in chunks for r in c["metadata"]["amendments"]))
check("amendment notes are also kept in body_faithful",
      all(r["note"] in c["body_faithful"]
          for c in chunks for r in c["metadata"]["amendments"]))

print("\n== suffix normalization round-trip ==")
from legal_assistant.rag.retrieval import normalize_article_suffix  # noqa: E402

normalized = {}
for c in subs:
    key = (c["metadata"]["article_number"],
           normalize_article_suffix(c["metadata"]["article_suffix"]))
    normalized.setdefault(key, []).append(c["chunk_id"])
collisions = {k: v for k, v in normalized.items() if len(v) > 1}
check("no two articles normalize to the same citation-guard key",
      not collisions, str(collisions))

if "--cloud" in sys.argv:
    print("\n== Qdrant Cloud parity ==")
    import os

    from dotenv import load_dotenv
    from embeddings import HybridVec, SparseVec  # noqa: F401
    from qdrant_client import QdrantClient, models
    from vector_store import point_id

    load_dotenv(SP.parent / ".env")
    client = QdrantClient(url=os.environ["QDRANT_CLOUD_URL"],
                          api_key=os.environ["QDRANT_CLOUD_API_KEY"], timeout=120)
    collection = os.environ.get("QDRANT_COLLECTION_NAME", "egyptian_law")

    records, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="law_number", match=models.MatchValue(value=159))
        ]),
        limit=1000, with_payload=True,
    )
    check("all 219 points are in the collection", len(records) == 219, str(len(records)))

    by_id = {str(r.id): r.payload for r in records}
    missing, mismatched = [], []
    for c in chunks:
        expected_payload = {
            "chunk_id": c["chunk_id"], "citation_label": c["citation_label"],
            "header": c["header"], "body_faithful": c["body_faithful"],
            "text_for_display": c["text_for_display"], **c["metadata"],
        }
        actual = by_id.get(point_id(c["chunk_id"]))
        if actual is None:
            missing.append(c["chunk_id"])
        elif actual != expected_payload:
            differing = {k for k in set(actual) | set(expected_payload)
                         if actual.get(k) != expected_payload.get(k)}
            mismatched.append((c["chunk_id"], differing))
    check("no point missing", not missing, str(missing[:5]))
    check("every payload matches what _to_point would build",
          not mismatched, str(mismatched[:3]))
    check("the other three laws are untouched",
          client.get_collection(collection).points_count == 2027,
          str(client.get_collection(collection).points_count))
    client.close()

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
