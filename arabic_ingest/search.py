# -*- coding: utf-8 -*-
"""Query the ingested corpus and print citation-ready results.

Use it to eyeball retrieval quality on real legal questions (the last Phase 2
step) before wiring up the assistant.

Examples:
    python search.py "ما هي شروط التصالح في الجنح؟"
    python search.py "اختصاص محكمة الجنايات" --limit 5
    python search.py "التزوير" --law-number 174 --book-number 2
"""
from __future__ import annotations

import argparse
import textwrap

from retrieval import build_retriever, smart_search


def main() -> int:
    ap = argparse.ArgumentParser(description="Search the Egyptian-law corpus.")
    ap.add_argument("query", help="the legal question / phrase (Arabic)")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--law-number", type=int, default=None)
    ap.add_argument("--law-year", type=int, default=None)
    ap.add_argument("--book-number", type=int, default=None)
    ap.add_argument("--article-number", type=int, default=None)
    ap.add_argument("--article-type", choices=["substantive", "issuance"], default=None)
    ap.add_argument("--full", action="store_true", help="print full article bodies")
    args = ap.parse_args()

    retriever = build_retriever()
    results = smart_search(
        retriever, args.query, limit=args.limit,
        law_number=args.law_number, law_year=args.law_year,
        book_number=args.book_number, article_number=args.article_number,
        article_type=args.article_type,
    )

    print(f"\nQuery: {args.query}")
    print(f"Results: {len(results)}\n" + "=" * 70)
    for i, r in enumerate(results, 1):
        loc = " › ".join(
            x for x in (r.book_title, r.part_title, r.chapter_title) if x
        )
        print(f"\n#{i}  score={r.score:.4f}   {r.citation_label}")
        if loc:
            print(f"    {loc}")
        body = r.body_faithful if args.full else textwrap.shorten(
            r.body_faithful, width=240, placeholder=" …"
        )
        print("    " + body.replace("\n", "\n    "))
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
