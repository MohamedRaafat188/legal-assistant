# -*- coding: utf-8 -*-
"""Human-review dump of extracted articles (thin wrapper over articles.py).

Use this to eyeball extraction quality before chunking. For the final
vector-DB-ready chunks (with headers, dual text, citation labels), use
chunker.py instead.

Run:  python preview_articles.py "path/to/law.pdf" --json articles_preview.json
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pdf_extractor import PdfTextExtractor
from articles import iter_articles
from glyph_repair import repair_pages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--law-number", type=int, default=0,
                    help="apply that law's glyph-repair (e.g. 131) before parsing")
    ap.add_argument("--json", type=Path, default=Path("articles_preview.json"))
    a = ap.parse_args()

    pages = PdfTextExtractor().extract_pages(str(a.pdf))
    pages, _ = repair_pages(pages, a.law_number)
    records = iter_articles(pages)
    a.json.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sub = [r for r in records if r.article_type == "substantive" and r.article_number is not None]
    iss = [r for r in records if r.article_type == "issuance"]
    repealed_ranges = [r for r in records if r.repealed_range]
    print(f"Extracted {len(records)} articles -> {a.json}")
    print(f"  issuance: {len(iss)} | substantive: {len(sub)} "
          f"(range {min(r.article_number for r in sub)}..{max(r.article_number for r in sub)})")
    if repealed_ranges:
        print(f"  repealed-range placeholders: {len(repealed_ranges)} "
              f"({', '.join(r.repealed_range for r in repealed_ranges)}) -- "
              "expanded to per-number chunks by chunker.py, not here")
    print(f"  spanning >1 page: {sum(1 for r in records if r.page_end > r.page_start)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
