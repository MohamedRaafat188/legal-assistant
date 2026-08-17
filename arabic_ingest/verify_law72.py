"""Checks that must hold before chunks_law72.json is fit to ingest."""
import json
import re
import sys
from pathlib import Path

SP = Path(__file__).parent
chunks = json.loads((SP / "chunks_law72.json").read_text(encoding="utf-8"))
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


print("== coverage ==")
subs = [c for c in chunks if c["metadata"]["article_type"] == "substantive"]
iss = [c for c in chunks if c["metadata"]["article_type"] == "issuance"]
plain = sorted(c["metadata"]["article_number"] for c in subs
               if not c["metadata"]["article_suffix"])
check("10 issuance articles", len(iss) == 10, str(len(iss)))
check("articles 1..94 all present", plain == list(range(1, 95)),
      f"missing {sorted(set(range(1, 95)) - set(plain))}")
bis = sorted(c["metadata"]["article_number"] for c in subs
             if c["metadata"]["article_suffix"])
check("two مكرر articles (11, 91)", bis == [11, 91], str(bis))
check("chunk ids unique", len({c["chunk_id"] for c in chunks}) == len(chunks))

print("\n== schema parity with laws 131/174 ==")
ref = json.loads(Path(r"E:\DL projects\Legal Assistant\arabic_ingest"
                      r"\chunks_law174.json").read_text(encoding="utf-8"))
ref_top, ref_meta = set(ref[0]), set(ref[0]["metadata"])
new_top, new_meta = set(chunks[0]), set(chunks[0]["metadata"])
check("top-level keys identical", ref_top == new_top, str(ref_top ^ new_top))
check("metadata adds only article_suffix + amendments",
      new_meta - ref_meta == {"article_suffix", "amendments"} and not ref_meta - new_meta,
      f"added={new_meta - ref_meta} missing={ref_meta - new_meta}")

print("\n== body integrity ==")
check("no empty bodies", all(len(c["body_faithful"]) > 30 for c in chunks))
check("body_faithful is the tail of text_for_display",
      all(c["text_for_display"].endswith(c["body_faithful"]) for c in chunks))
check("header prefixes text_for_display",
      all(c["text_for_display"].startswith(c["header"]) for c in chunks))
check("char_count matches body",
      all(c["metadata"]["char_count"] == len(c["body_faithful"]) for c in chunks))
check("no page furniture markers", not any("===== PAGE" in c["body_faithful"] for c in chunks))

print("\n== article 32/33 reconstruction ==")
a32 = next(c for c in subs if c["metadata"]["article_number"] == 32
           and not c["metadata"]["article_suffix"])
a33 = next(c for c in subs if c["metadata"]["article_number"] == 33
           and not c["metadata"]["article_suffix"])
check("art 32 ends with its restored final paragraph",
      "وأسلوب إدارتها" in a32["body_faithful"])
check("art 32 no longer contains art 33's opening",
      "يكون إنشاء المنطقة الحرة" not in a32["body_faithful"])
check("art 33 starts with its own opening",
      a33["body_faithful"].startswith("يكون إنشاء المنطقة الحرة"))
check("art 33 sits in الفصل الرابع",
      a33["metadata"]["chapter_number"] == 4, str(a33["metadata"]["chapter_title"]))

print("\n== amendments ==")
amended = [c for c in chunks if c["metadata"]["amendments"]]
total = sum(len(c["metadata"]["amendments"]) for c in chunks)
check("15 amendments across 14 articles", total == 15 and len(amended) == 14,
      f"{total} amendments / {len(amended)} articles")
expected = {
    (12, "append_part", 141), (48, "append_part", 141), (74, "append_part", 141),
    (91, "add_article", 141),
    (1, "replace_part", 160), (9, "replace_article", 160),
    (11, "replace_part", 160), (12, "replace_part", 160),
    (13, "replace_article", 160), (14, "replace_part", 160),
    (17, "replace_article", 160), (20, "replace_article", 160),
    (34, "replace_article", 160), (40, "replace_part", 160),
    (11, "add_article", 160),
}
actual = {(c["metadata"]["article_number"], a["kind"], a["law_number"])
          for c in chunks for a in c["metadata"]["amendments"]}
check("amendment map matches the Gazette", actual == expected, str(actual ^ expected))

notes_ok = True
for c in amended:
    for a in c["metadata"]["amendments"]:
        if a["note"] not in c["body_faithful"]:
            notes_ok = False
            print(f"      note missing from body: {c['chunk_id']} :: {a['note']}")
check("every amendment note appears in body_faithful", notes_ok)

emb_ok = all(
    all(re.sub(r"[^\u0621-\u064a ]", "", a["note"]).split()[0] in c["text_for_embedding"]
        for a in c["metadata"]["amendments"])
    for c in amended
)
check("amendment notes reach the embedding copy", emb_ok)

print("\n== citations ==")
check("every chunk has a citation label", all(c["citation_label"] for c in chunks))
check("مكرر articles are labelled as such",
      all("مكررًا" in c["citation_label"] for c in chunks
          if c["metadata"]["article_suffix"]))
check("plain articles never say مكرر",
      not any("مكرر" in c["citation_label"] for c in chunks
              if not c["metadata"]["article_suffix"]))

print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
