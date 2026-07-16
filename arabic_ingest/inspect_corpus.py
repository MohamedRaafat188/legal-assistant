# -*- coding: utf-8 -*-
"""Inspect the structure detected in a law PDF.

Run it from THIS folder:
    python inspect_corpus.py "C:\\path\\to\\law.pdf"
    python inspect_corpus.py "C:\\path\\to\\law.pdf" --law-number 131 --json structure.json

No installation, no subfolders: this file imports its sibling files
(arabic_text.py, pdf_extractor.py, structure.py, articles.py, chunker.py)
that live in the same folder.

Builds the report from `iter_articles`/`build_chunks` directly (division ▸
book ▸ part ▸ chapter ▸ section ▸ sub-section ▸ article), rather than from a
separate structural-marker pass, so ONE traversal is the source of truth for
both the vector-DB chunks and this report -- exactly the corpus that gets
ingested is exactly the corpus this prints. Repealed-range placeholders are
expanded (Option B) before the integrity check, per project spec: the check
must confirm no gaps remain once expansion has run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdf_extractor import PdfTextExtractor
from articles import ParseDiagnostics, iter_articles
from chunker import Chunk, build_chunks, _ISSUANCE_STYLE, _LAWS
from glyph_repair import repair_pages


def build_tree(chunks: list[Chunk]) -> dict:
    """Nest chunk metadata into division ▸ book ▸ part ▸ chapter ▸ section.

    Keyed by ordinal (None-safe: a level absent for a given chunk -- e.g. no
    division under the preamble -- is simply skipped, not synthesized).
    """
    root: dict = {"children": {}, "articles": []}

    def child(container: dict, level: str, number, title) -> dict:
        key = (level, number)
        node = container["children"].get(key)
        if node is None:
            node = {"level": level, "number": number, "title": title,
                     "children": {}, "articles": []}
            container["children"][key] = node
        return node

    for c in chunks:
        m = c.metadata
        node = root
        if m["division_number"]:
            node = child(node, "division", m["division_number"], m["division_title"])
        if m["book_number"]:
            node = child(node, "book", m["book_number"], m["book_title"])
        if m["part_kind"] == "preamble":
            node = child(node, "part", "تمهيدي", m["part_title"])
        elif m["part_number"]:
            node = child(node, "part", m["part_number"], m["part_title"])
        if m["chapter_number"]:
            node = child(node, "chapter", m["chapter_number"], m["chapter_title"])
        if m["section_number"]:
            node = child(node, "section", m["section_number"], m["section_title"])
        node["articles"].append(m)
    return root


def _collect_article_numbers(node: dict) -> list[int]:
    """Recursively gather article numbers at and below this node (any depth)."""
    nums = [n["article_number"] for n in node["articles"] if n["article_number"] is not None]
    for child in node["children"].values():
        nums += _collect_article_numbers(child)
    return nums


def _article_span(node: dict) -> str:
    nums = _collect_article_numbers(node)
    if not nums:
        return "-"
    return f"{min(nums)}-{max(nums)} ({len(nums)} مادة)"


def _print_node(node: dict, indent: int = 0) -> None:
    label = {
        "division": "القسم", "book": "الكتاب", "part": "الباب",
        "chapter": "الفصل", "section": "-",
    }.get(node.get("level"), "")
    if node.get("level"):
        pad = "  " * indent
        num = node["number"]
        print(f"{pad}{label} {num}: {node['title']}  [{_article_span(node)}]")
    for child in node["children"].values():
        _print_node(child, indent + 1)


def article_integrity(numbers: list[int]) -> dict:
    seen: set[int] = set()
    duplicates = sorted({n for n in numbers if n in seen or seen.add(n)})
    lo, hi = (min(numbers), max(numbers)) if numbers else (0, 0)
    missing = sorted(set(range(lo, hi + 1)) - set(numbers)) if numbers else []
    return {
        "total": len(numbers), "unique": len(set(numbers)),
        "minimum": lo, "maximum": hi,
        "missing": missing, "duplicates": duplicates,
        "is_complete": not missing and not duplicates and len(numbers) == len(set(numbers)),
    }


def print_report(chunks: list[Chunk], diagnostics: ParseDiagnostics, tree: dict) -> None:
    substantive = [c for c in chunks if c.metadata["article_type"] == "substantive"]
    issuance = [c for c in chunks if c.metadata["article_type"] == "issuance"]
    repealed = [c for c in chunks if c.metadata["article_status"] == "repealed"]

    divisions = {c.metadata["division_number"] for c in chunks if c.metadata["division_number"]}
    books = {c.metadata["book_number"] for c in chunks if c.metadata["book_number"]}
    parts = {(c.metadata["book_number"], c.metadata["part_number"], c.metadata["part_kind"])
             for c in chunks if c.metadata["part_number"] or c.metadata["part_kind"] == "preamble"}
    chapters = {(c.metadata["book_number"], c.metadata["part_number"], c.metadata["chapter_number"])
                for c in chunks if c.metadata["chapter_number"]}
    sections = {(c.metadata["chapter_number"], c.metadata["section_number"])
                for c in chunks if c.metadata["section_number"]}
    subsections = {s["title"] for s in diagnostics.subsections}

    integ = article_integrity([c.metadata["article_number"] for c in substantive])

    print("=" * 70)
    print("STRUCTURE DETECTED")
    print("=" * 70)
    print(f"  Divisions (أقسام)    : {len(divisions)}")
    print(f"  Books (كتب)          : {len(books)}")
    print(f"  Parts (أبواب)        : {len(parts)}  (incl. باب تمهيدي if present)")
    print(f"  Chapters (فصول)      : {len(chapters)}")
    print(f"  Sections (أقسام فرعية): {len(sections)}")
    print(f"  Sub-sections (فروع)  : {len(subsections)}")
    print(f"  Articles (مواد)      : {len(substantive)} substantive + {len(issuance)} issuance")
    print(f"    of which repealed  : {len(repealed)}")
    print("-" * 70)
    print("ARTICLE INTEGRITY (substantive, after repeal-range expansion)")
    print(f"  range        : {integ['minimum']}..{integ['maximum']}")
    print(f"  unique       : {integ['unique']}/{integ['total']}")
    print(f"  missing      : {integ['missing'] or 'none'}")
    print(f"  duplicates   : {integ['duplicates'] or 'none'}")
    status = "PASS complete, gap-free" if integ["is_complete"] else "FAIL"
    print(f"  completeness : {status}")
    print("-" * 70)
    print("ISSUANCE ARTICLES (مواد الإصدار)")
    for c in sorted(issuance, key=lambda c: c.metadata["issuance_number"]):
        print(f"  {c.citation_label}")
    print("-" * 70)
    print("REPEALED RANGES")
    ranges = sorted({c.metadata["repealed_range"] for c in repealed})
    for rng in ranges:
        count = sum(1 for c in repealed if c.metadata["repealed_range"] == rng)
        print(f"  {rng}: {count} article(s)")
    print("-" * 70)
    print("HIERARCHY")
    _print_node(tree)
    print("=" * 70)
    print("SUB-SECTIONS DETECTED (review for phantoms before trusting)")
    print("=" * 70)
    for s in diagnostics.subsections:
        marker = "" if s["has_colon"] else "  [no colon -- repeal-adjacent exception]"
        print(f"  p{s['page']}: {s['title']!r}{marker}")
    if diagnostics.unclassified:
        print("-" * 70)
        print(f"UNCLASSIFIED LINES ({len(diagnostics.unclassified)}) -- expected to be page "
              "furniture / title echoes; review for anything unexpected")
        for u in diagnostics.unclassified:
            print(f"  p{u['page']}: {u['text']!r}")
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect detected law structure.")
    ap.add_argument("pdf", type=Path, help="Path to the law PDF.")
    ap.add_argument("--law-number", type=int, default=174, choices=sorted(_LAWS),
                    help="which law's metadata/conventions to apply")
    ap.add_argument("--json", type=Path, default=None,
                    help="Optional path to dump the article metadata as JSON.")
    args = ap.parse_args()

    from dataclasses import replace
    law = replace(_LAWS[args.law_number], source_file=args.pdf.name)
    issuance_style = _ISSUANCE_STYLE[args.law_number]

    pages = PdfTextExtractor().extract_pages(args.pdf)
    pages, _glyph_flags = repair_pages(pages, args.law_number)
    diagnostics = ParseDiagnostics()
    records = iter_articles(pages, diagnostics)
    chunks = build_chunks(records, law, issuance_style)
    tree = build_tree(chunks)
    print_report(chunks, diagnostics, tree)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"source": args.pdf.name,
                        "articles": [c.metadata | {"citation_label": c.citation_label}
                                     for c in chunks]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote structure JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
