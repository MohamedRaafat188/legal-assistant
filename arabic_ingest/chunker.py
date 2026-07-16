# -*- coding: utf-8 -*-
"""Article-level chunker — produces vector-DB-ready chunks from a law PDF.

Each chunk is exactly one article (never split mid-article, per the corpus
rules) and carries:

  * ``text_for_embedding`` — the NORMALIZED structural-context header + body.
    This is the only field you embed. Prepending the division/book/part/
    chapter/section/sub-section titles is deliberate "contextual chunking": it
    lets a query about e.g. محاكم الجنايات match articles whose chapter is
    محاكم الجنايات even when the body doesn't repeat the phrase.
  * ``text_for_display`` / ``body_faithful`` — the FAITHFUL text, exactly as
    enacted. This is what the assistant shows and quotes. It must be the ONLY
    text a citation is built from — never the normalized copy, never memory.
  * ``citation_label`` — the canonical Arabic citation string, computed here
    where correctness is guaranteed, so downstream never composes one itself.
  * ``metadata`` — the filterable fields (law number/year, article number,
    division/book/part/chapter/section ancestry, article status, pages) for
    metadata-filtered retrieval.

Repealed-article RANGES (e.g. "المواد من 54 إلى 80 ملغاة") arrive from
`articles.py` as a single placeholder record and are expanded HERE into one
lightweight chunk per article number (Option B — see module docstring in
articles.py), so the by-number lookup router needs zero changes to resolve a
repealed article.

The chunk id is deterministic, so re-ingesting a law upserts (updates) rather
than duplicating.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Literal, Optional

from pdf_extractor import PdfTextExtractor
from arabic_text import ascii_digits_to_arabic, normalize_for_embedding
from structure import INT_TO_ORDINAL, INT_TO_FEM_ORDINAL
from articles import ArticleRecord, LawMetadata, ParseDiagnostics, iter_articles
from glyph_repair import repair_pages

# Rough Arabic token estimate (subword tokenizers run denser on Arabic than
# 4 chars/token). Only used to FLAG articles near an embedding model's limit.
_CHARS_PER_TOKEN = 1.8

IssuanceStyle = Literal["feminine_ordinal", "cardinal"]


def _ar(n: int) -> str:
    """Render an integer in Arabic-Indic digits for display."""
    return ascii_digits_to_arabic(str(n))


def _issuance_label(n: int, style: IssuanceStyle) -> str:
    """Faithful issuance-article label, per the source's own numbering style.

    Law 174 names its enacting provisions with feminine ordinal words
    ("المادة الأولى"); Law 131 numbers them like ordinary articles
    ("مادة 1", "مادة 2"). Both are real source conventions -- render whichever
    the law actually uses rather than forcing one style on both.
    """
    if style == "cardinal":
        return f"مادة {_ar(n)}"
    return f"المادة {INT_TO_FEM_ORDINAL[n]}"


def _ancestry_qualifier(rec: ArticleRecord) -> Optional[str]:
    """A short parenthetical for citations where bare ancestry needs context.

    Continuous article numbering already makes a bare "المادة (N)" citation
    unambiguous, so this is deliberately minimal -- currently only the
    preamble (باب تمهيدي), whose articles would otherwise read as if they sat
    directly under a numbered part.
    """
    if rec.part_kind == "preamble":
        return "الباب التمهيدي"
    return None


def _build_header(rec: ArticleRecord, law: LawMetadata,
                   issuance_style: IssuanceStyle) -> str:
    """Faithful, formal-Arabic structural header prepended to the article.

    Reused for BOTH the faithful display header and (after
    ``normalize_for_embedding``) the embedding-text ancestry prefix -- see
    ``build_chunks`` -- so ancestry is composed in exactly one place.
    """
    lines = [f"{law.law_name} رقم ({_ar(law.law_number)}) لسنة ({_ar(law.law_year)})"]
    if rec.article_type == "issuance":
        lines.append("من مواد الإصدار")
        lines.append(_issuance_label(rec.issuance_number, issuance_style))
        return "\n".join(lines)
    if rec.division_number:
        lines.append(f"القسم {INT_TO_ORDINAL[rec.division_number]}: {rec.division_title}")
    if rec.book_number:
        lines.append(f"الكتاب {INT_TO_ORDINAL[rec.book_number]}: {rec.book_title}")
    if rec.part_kind == "preamble":
        lines.append(f"الباب التمهيدي: {rec.part_title}")
    elif rec.part_number:
        lines.append(f"الباب {INT_TO_ORDINAL[rec.part_number]}: {rec.part_title}")
    if rec.chapter_number:
        lines.append(f"الفصل {INT_TO_ORDINAL[rec.chapter_number]}: {rec.chapter_title}")
    if rec.section_number:
        lines.append(f"{_ar(rec.section_number)}- {rec.section_title}")
    if rec.subsection_title:
        lines.append(rec.subsection_title)
    lines.append(f"مادة ({_ar(rec.article_number)})")
    return "\n".join(lines)


def _build_citation(rec: ArticleRecord, law: LawMetadata,
                     issuance_style: IssuanceStyle) -> str:
    """Canonical Arabic citation label, computed once at ingest time."""
    suffix = f"من {law.law_name} رقم {_ar(law.law_number)} لسنة {_ar(law.law_year)}"
    if rec.article_type == "issuance":
        label = _issuance_label(rec.issuance_number, issuance_style)
        return f"{label} من مواد إصدار {law.law_name} رقم {_ar(law.law_number)} لسنة {_ar(law.law_year)}"
    base = f"المادة ({_ar(rec.article_number)}) {suffix}"
    qualifier = _ancestry_qualifier(rec)
    return f"{base} ({qualifier})" if qualifier else base


def _chunk_id(rec: ArticleRecord, law: LawMetadata) -> str:
    base = f"{law.law_number}-{law.law_year}"
    if rec.article_type == "issuance":
        return f"{base}-issuance-{rec.issuance_number}"
    return f"{base}-art-{rec.article_number}"


@dataclass(slots=True)
class Chunk:
    """A vector-DB-ready, article-level chunk."""
    chunk_id: str
    text_for_embedding: str      # EMBED this (normalized header + body)
    text_for_display: str        # SHOW this (faithful header + body)
    body_faithful: str           # QUOTE citations from this only
    header: str                  # faithful structural header
    citation_label: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _repealed_body(a: int, b: int) -> str:
    """Mandated faithful body for an expanded repealed-article chunk.

    The parenthetical names the collective range in Arabic-Indic digits so
    the chunk stays honest about where the repeal was actually stated --
    never a fabricated standalone assertion about this one article number.
    """
    return f"ملغاة (ضمن المواد من {_ar(a)} إلى {_ar(b)})"


def _expand_repealed_range(rec: ArticleRecord) -> list[ArticleRecord]:
    """Fan a collapsed 'المواد من A إلى B ملغاة' placeholder out to one
    lightweight ArticleRecord per article number in the range (Option B)."""
    a, b = (int(x) for x in rec.repealed_range.split("-"))
    out = []
    for n in range(a, b + 1):
        out.append(replace(
            rec, article_number=n, body=_repealed_body(a, b),
            char_count=len(_repealed_body(a, b)),
        ))
    return out


def build_chunks(records: list[ArticleRecord], law: LawMetadata,
                  issuance_style: IssuanceStyle = "feminine_ordinal") -> list[Chunk]:
    """Turn article records into chunks (header + dual text + metadata + id).

    Records carrying a `repealed_range` are expanded into per-number chunks
    before any other processing, so everything downstream (chunk id,
    citation, header) just sees ordinary numbered ArticleRecords.
    """
    expanded: list[ArticleRecord] = []
    for rec in records:
        if rec.repealed_range:
            expanded.extend(_expand_repealed_range(rec))
        else:
            expanded.append(rec)

    chunks: list[Chunk] = []
    for rec in expanded:
        header = _build_header(rec, law, issuance_style)
        display = f"{header}\n\n{rec.body}"
        # Embed the normalized context header + body (contextual chunking).
        embedding = normalize_for_embedding(f"{header}\n{rec.body}")
        metadata = {
            "law_name": law.law_name,
            "law_number": law.law_number,
            "law_year": law.law_year,
            "jurisdiction": law.jurisdiction,
            "issue_ref": law.issue_ref,
            "source_file": law.source_file,
            "article_type": rec.article_type,
            "article_number": rec.article_number,
            "issuance_number": rec.issuance_number,
            "article_status": rec.article_status,
            "repealed_range": rec.repealed_range,
            "division_number": rec.division_number, "division_title": rec.division_title,
            "book_number": rec.book_number, "book_title": rec.book_title,
            "part_number": rec.part_number, "part_title": rec.part_title,
            "part_kind": rec.part_kind,
            "chapter_number": rec.chapter_number, "chapter_title": rec.chapter_title,
            "section_number": rec.section_number, "section_title": rec.section_title,
            "subsection_title": rec.subsection_title,
            "page_start": rec.page_start, "page_end": rec.page_end,
            "char_count": rec.char_count,
            "approx_tokens": round(rec.char_count / _CHARS_PER_TOKEN),
        }
        chunks.append(Chunk(
            chunk_id=_chunk_id(rec, law),
            text_for_embedding=embedding,
            text_for_display=display,
            body_faithful=rec.body,
            header=header,
            citation_label=_build_citation(rec, law, issuance_style),
            metadata=metadata,
        ))
    return chunks


# --- Per-law config (a registry replaces per-law hard-coding in Phase 2+) ---
LAW_174 = LawMetadata(
    law_name="قانون الإجراءات الجنائية",
    law_number=174,
    law_year=2025,
    issue_ref="الجريدة الرسمية – العدد ٤٥ مكرر (د)",
)
LAW_131 = LawMetadata(
    law_name="القانون المدني",
    law_number=131,
    law_year=1948,
)
# issuance_style is a per-law CONVENTION, not LawMetadata identity -- kept as
# a side table so LawMetadata stays a pure identity record.
_ISSUANCE_STYLE: dict[int, IssuanceStyle] = {
    174: "feminine_ordinal",
    131: "cardinal",
}
_LAWS: dict[int, LawMetadata] = {174: LAW_174, 131: LAW_131}


def main() -> int:
    ap = argparse.ArgumentParser(description="Chunk an Egyptian law PDF at article level.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--law-number", type=int, default=174, choices=sorted(_LAWS),
                    help="which law's metadata/conventions to apply")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    json_path = args.json or Path(f"chunks_law{args.law_number}.json")

    law = replace(_LAWS[args.law_number], source_file=args.pdf.name)
    issuance_style = _ISSUANCE_STYLE[args.law_number]
    diagnostics = ParseDiagnostics()
    pages = PdfTextExtractor().extract_pages(str(args.pdf))
    pages, glyph_flags = repair_pages(pages, args.law_number)
    records = iter_articles(pages, diagnostics)
    chunks = build_chunks(records, law, issuance_style)

    json_path.write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    longest = max(chunks, key=lambda c: c.metadata["approx_tokens"])
    print(f"Built {len(chunks)} chunks -> {json_path}")
    print(f"  substantive: {sum(1 for c in chunks if c.metadata['article_type']=='substantive')}")
    print(f"  issuance   : {sum(1 for c in chunks if c.metadata['article_type']=='issuance')}")
    print(f"  repealed   : {sum(1 for c in chunks if c.metadata['article_status']=='repealed')}")
    print(f"  longest article ~{longest.metadata['approx_tokens']} tokens "
          f"({longest.citation_label})")
    if diagnostics.subsections:
        print(f"  sub-sections detected: {len(diagnostics.subsections)} "
              f"(review before trusting -- see ParseDiagnostics)")
    if diagnostics.unclassified:
        print(f"  unclassified lines: {len(diagnostics.unclassified)} "
              f"(furniture/echoes expected; review for anything unexpected)")
    if glyph_flags:
        total = sum(glyph_flags.values())
        print(f"  glyph-repair: {len(glyph_flags)} UNREVIEWED آ-token(s), "
              f"{total} occ, repaired as آ->ك -- verify: "
              f"{', '.join(t for t, _ in glyph_flags.most_common(10))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
