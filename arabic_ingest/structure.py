# -*- coding: utf-8 -*-
"""Detect the hierarchical skeleton of an Egyptian law from cleaned pages.

Structure of the corpus (validated against Law 174/2025):

    الكتاب (book)  -> الباب (part)  -> الفصل (chapter)  -> مادة (article)

Parts and chapters are optional at any level; the article (مادة) is the only
mandatory leaf. Ordinals RESET inside each parent (every book restarts its
parts at الأول; every part restarts its chapters at الأول), so a level's
number is meaningful only *relative to its parent*. Assembling that parent
chain is done by walking markers in document order (see ``build_hierarchy``).

Detection rules — derived empirically and reconciled to ground truth
(6 books, 23 parts, 46 chapters, 546 articles):

* A **structural header** is the keyword (الكتاب/الباب/الفصل) + an ordinal
  word, and NOTHING else on the line (end-of-line anchor). The anchor is what
  rejects cross-references like "الباب الرابع من الكتاب الثاني من قانون
  العقوبات" that carry trailing text.
* An **article header** is optional 'ال' + مادة + a number, and nothing else.
  The 'ال' is required because exactly one article (134) is written 'المادة'.
  BiDi reordering scatters the parentheses, so they are matched loosely.

All matching happens on ``normalize_for_match`` output so ordinal spelling
variants (الأول/الاول, الثانى/الثاني) collapse. The faithful title text is
still read from the original lines.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from arabic_text import arabic_digits_to_ascii, normalize_for_match
from pdf_extractor import PageText


class Level(str, Enum):
    """Hierarchical level of a structural marker."""

    DIVISION = "division"  # القسم (sits above book; e.g. the Civil Code)
    BOOK = "book"        # الكتاب
    PART = "part"        # الباب
    CHAPTER = "chapter"  # الفصل


# Arabic keyword per level (definite-article form, as it appears in headers).
_LEVEL_KEYWORD: dict[Level, str] = {
    Level.DIVISION: "القسم",
    Level.BOOK: "الكتاب",
    Level.PART: "الباب",
    Level.CHAPTER: "الفصل",
}

# Ordinal vocabulary -> integer. Keys are normalized at import time so lookups
# always compare like-for-like with normalize_for_match. Covers 1..20, which
# comfortably spans Egyptian book/part/chapter numbering (this law peaks at 13).
_RAW_ORDINALS: dict[str, int] = {
    "الأول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5,
    "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9, "العاشر": 10,
    "الحادي عشر": 11, "الثاني عشر": 12, "الثالث عشر": 13, "الرابع عشر": 14,
    "الخامس عشر": 15, "السادس عشر": 16, "السابع عشر": 17, "الثامن عشر": 18,
    "التاسع عشر": 19, "العشرون": 20,
}
ORDINAL_TO_INT: dict[str, int] = {
    normalize_for_match(word): n for word, n in _RAW_ORDINALS.items()
}
# Reverse map (int -> faithful masculine ordinal word) for header display.
INT_TO_ORDINAL: dict[int, str] = {n: w for w, n in _RAW_ORDINALS.items()}

# Longest-first alternation so "الثاني عشر" wins over "الثاني".
_ORDINAL_ALT = "|".join(
    sorted((re.escape(k) for k in ORDINAL_TO_INT), key=len, reverse=True)
)

# One compiled header regex per structural level (matched on normalized text).
_STRUCT_RE: dict[Level, re.Pattern[str]] = {
    level: re.compile(rf"^{kw}\s+(?P<ord>{_ORDINAL_ALT})$")
    for level, kw in _LEVEL_KEYWORD.items()
}

# Article header (matched on normalized + digit-ASCII text). Optional 'ال',
# BiDi-tolerant parentheses, end-of-line anchor.
_ARTICLE_RE = re.compile(r"^(?:ال)?مادة\s*[()]*\s*(?P<num>\d+)\s*[()]*\s*$")

# --- Issuance articles (مواد الإصدار) ---------------------------------------
# The enacting provisions that precede الكتاب الأول are named with FEMININE
# ordinals (مادة is feminine): "(المادة الأولى)" … "(المادة السادسة)". They sit
# outside the book/part/chapter tree and outside the 1..N numbering, but carry
# real legal weight (e.g. المادة الرابعة repeals the prior law), so we capture
# them as a separate, clearly-typed set. Detected only before the first book,
# with the same end-of-line anchor that keeps cross-references out.
_RAW_FEM_ORDINALS: dict[str, int] = {
    "الأولى": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4, "الخامسة": 5,
    "السادسة": 6, "السابعة": 7, "الثامنة": 8, "التاسعة": 9, "العاشرة": 10,
}
FEM_ORDINAL_TO_INT: dict[str, int] = {
    normalize_for_match(word): n for word, n in _RAW_FEM_ORDINALS.items()
}
# Reverse map (int -> canonical faithful ordinal word) for building citations.
INT_TO_FEM_ORDINAL: dict[int, str] = {n: w for w, n in _RAW_FEM_ORDINALS.items()}
_FEM_ORDINAL_ALT = "|".join(
    sorted((re.escape(k) for k in FEM_ORDINAL_TO_INT), key=len, reverse=True)
)
_ISSUANCE_RE = re.compile(
    rf"^[()]*\s*المادة\s+(?P<ord>{_FEM_ORDINAL_ALT})\s*[()]*$"
)

# --- Law 131/1948 (Civil Code) additions ------------------------------------
# This source has a fundamentally different typeset than Law 174: headers routinely
# carry their title INLINE (after a dash, same line) instead of on a following
# line, and article markers are glued directly to their body text rather than
# sitting alone. All patterns below are matched on normalized text and are purely
# ADDITIVE -- Law 174 detection above is untouched.

# Division/book/part header with an OPTIONAL inline title after a dash. Bare
# (nothing after the ordinal) still falls through to `_STRUCT_RE` above/`_title_after`;
# this pattern is tried first so an inline title is captured directly when present.
_INLINE_TITLE_RE: dict[Level, re.Pattern[str]] = {
    level: re.compile(rf"^{kw}\s+(?P<ord>{_ORDINAL_ALT})\s*[-–—]\s*(?P<title>.+)$")
    for level, kw in _LEVEL_KEYWORD.items()
    if level in (Level.DIVISION, Level.BOOK, Level.PART)
}

# The "running banner" line that repeats "Book X - Part Y - Part title" atop
# pages within a part (page-furniture in most cases, but the ONLY source of a
# part's title when no bare/inline part header exists elsewhere -- see
# articles.py's book/part registration logic for how this is deduplicated).
_MERGED_BOOK_PART_RE = re.compile(
    rf"^الكتاب\s+(?P<bord>{_ORDINAL_ALT})\s*[-–—]\s*"
    rf"(?:الباب\s+(?P<pord>{_ORDINAL_ALT})\s*[-–—]\s*)?(?P<title>.+)$"
)

# Chapter header: title is ALWAYS inline in this document (dash separator is
# common but not guaranteed -- "الفصل الأول حق الملكية بوجه العام" has none).
_CHAPTER_INLINE_RE = re.compile(
    rf"^الفصل\s+(?P<ord>{_ORDINAL_ALT})\s*(?:[-–—]\s*)?(?P<title>.*)$"
)

# باب تمهيدي (preamble): an untitled-ordinal part sitting above the division
# tree, covering the articles before القسم الأول begins.
_PREAMBLE_RE = re.compile(r"^باب\s*تمهيدي\s*(?:[-–—]\s*(?P<title>.+))?$")

# Article header glued to its body on the same line ("مادة – 1يلغي...", "مادة
# (1) – 1تسري..."). A dash is required UNLESS a clause marker "(N)" precedes the
# number (empirically the only place the dash is ever dropped, e.g. "مادة (1)
# 523..."). Requiring the dash in the general case is what keeps mid-sentence
# cross-references ("المادة، 864 فإن...", "المادة السابقة...") from matching --
# those never carry a dash.
_ARTICLE_INLINE_RE = re.compile(
    r"^(?:ال)?ماد[ةه]\s*"
    r"(?:\((?P<clause>\d+)\)\s*[-–—]?|[-–—])\s*"
    r"(?P<num>\d+)(?P<rest>.*)$"
)

# Repealed-article-range placeholder ("المواد من 54إلى 80ملغاة..."). Matched on
# normalized + digit-ASCII text; note normalize_for_match folds "إلى" -> "الي".
_REPEALED_RANGE_RE = re.compile(
    r"^المواد\s+من\s*(?P<a>\d+)\s*الي\s*(?P<b>\d+)\s*ملغاة"
)

# Section header ("1-البيع بوجه عام" or its BiDi-flipped sibling "-1البيع...",
# also seen with a stop instead of a dash: ".1نطاقه..."). Resets per chapter.
# Only ever a genuine section when no article is currently open (see
# articles.py) -- the same "N-title" shape recurs as an enumerated list INSIDE
# an article's body (e.g. مادة 52), which the caller must exclude by position.
# The title must not start with a digit -- without this, greedy backtracking
# lets a bare wrapped cross-reference like ".263" spuriously match by
# splitting into n1="26"/title="3" once the "at least one char" title alone
# is required, which would wrongly close whatever article is still open.
_SECTION_RE = re.compile(
    r"^(?:[-.]\s*(?P<n1>\d+)|(?P<n2>\d+)\s*[-.])\s*(?P<title>(?!\d)\S.*)$"
)

# Sub-section label: a short, colon-terminated free-text heading ("الرضاء :",
# "التزامات البائع :"). Guarded further by position in articles.py (must sit
# between a section/chapter marker and the next article, only checked when no
# article is open, and is never itself an article/structural header).
_SUBSECTION_RE = re.compile(r"^(?P<title>[^:：]{1,40})\s*[:：]\s*$")
_SUBSECTION_MAX_WORDS = 6


@dataclass(frozen=True, slots=True)
class StructuralMarker:
    """A detected book/part/chapter header."""

    level: Level
    number: int            # ordinal value, relative to its parent
    ordinal_text: str      # normalized ordinal word (e.g. "الثاني عشر")
    title: str             # faithful title text (line(s) after the header)
    page_number: int
    order: int             # monotonic document-order index (reading order)


@dataclass(frozen=True, slots=True)
class ArticleMarker:
    """A detected article (مادة) header."""

    number: int
    page_number: int
    order: int             # monotonic document-order index (reading order)


@dataclass(slots=True)
class DetectedStructure:
    """Everything the detector found, in document order."""

    structural: list[StructuralMarker] = field(default_factory=list)
    articles: list[ArticleMarker] = field(default_factory=list)

    # -- counts ---------------------------------------------------------
    def count(self, level: Level) -> int:
        return sum(1 for m in self.structural if m.level == level)

    @property
    def n_books(self) -> int:
        return self.count(Level.BOOK)

    @property
    def n_parts(self) -> int:
        return self.count(Level.PART)

    @property
    def n_chapters(self) -> int:
        return self.count(Level.CHAPTER)

    @property
    def n_articles(self) -> int:
        return len(self.articles)

    # -- integrity ------------------------------------------------------
    def article_integrity(self) -> "ArticleIntegrity":
        """Check that article numbers form a complete, unique run.

        This is the corpus-completeness guarantee that underpins verifiable
        citation: if every article 1..max is present exactly once, the
        retriever can never be asked to cite an article the corpus lacks.
        """
        nums = [a.number for a in self.articles]
        seen: set[int] = set()
        duplicates = sorted({n for n in nums if n in seen or seen.add(n)})
        lo, hi = (min(nums), max(nums)) if nums else (0, 0)
        missing = sorted(set(range(lo, hi + 1)) - set(nums)) if nums else []
        return ArticleIntegrity(
            total=len(nums),
            unique=len(set(nums)),
            minimum=lo,
            maximum=hi,
            missing=missing,
            duplicates=duplicates,
        )


@dataclass(frozen=True, slots=True)
class ArticleIntegrity:
    total: int
    unique: int
    minimum: int
    maximum: int
    missing: list[int]
    duplicates: list[int]

    @property
    def is_complete(self) -> bool:
        """True iff articles are a gap-free, duplicate-free run."""
        return not self.missing and not self.duplicates and self.total == self.unique


def _is_running_header(norm_line: str) -> bool:
    """True for the gazette masthead line repeated atop every page."""
    return "الجريدة الرسمية" in norm_line and "العدد" in norm_line


def _is_page_number(line: str) -> bool:
    """True for a line that is nothing but a page number (Arabic or ASCII)."""
    return bool(re.fullmatch(r"\d+", arabic_digits_to_ascii(line).strip()))


def _title_after(lines: list[str], index: int) -> str:
    """Return the faithful title following a structural header.

    A title can span several physical lines (with blank lines between them, as
    pdftotext often inserts). It runs from the line after the header up to — but
    not including — the next structural or article marker. Blank lines, the page
    masthead, and bare page numbers are skipped (not treated as the end), so a
    title that wraps across lines or a page break is captured whole.
    """
    parts: list[str] = []
    for ln in lines[index + 1:]:
        stripped = ln.strip()
        if not stripped:
            continue  # blank line: titles wrap across these — keep going
        norm = normalize_for_match(stripped)
        if _is_running_header(norm) or _is_page_number(stripped):
            continue  # page furniture from a page break — skip, don't stop
        ascii_line = arabic_digits_to_ascii(norm)
        reached_marker = (
            _ARTICLE_RE.match(ascii_line)
            or _ARTICLE_INLINE_RE.match(ascii_line)
            or _PREAMBLE_RE.match(norm)
            or (norm.startswith("الكتاب") and _MERGED_BOOK_PART_RE.match(norm))
            or (norm.startswith("الفصل") and _CHAPTER_INLINE_RE.match(norm))
            or any(rx.match(norm) for rx in _STRUCT_RE.values())
            or any(rx.match(norm) for rx in _INLINE_TITLE_RE.values())
        )
        if reached_marker:
            break
        parts.append(stripped)
    return " ".join(parts).strip()


def detect_structure(pages: list[PageText]) -> DetectedStructure:
    """Scan cleaned pages and return all structural + article markers in order.

    Each marker carries a monotonic ``order`` index reflecting true reading
    order (page, then line). Downstream assembly MUST sort on ``order`` and
    never on marker type — a chapter closing one part can share a page with
    the header opening the next part, and only reading order disambiguates them.
    """
    result = DetectedStructure()
    seq = 0
    for page in pages:
        lines = page.text.split("\n")
        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            norm = normalize_for_match(line)

            # Structural header?
            matched_level = False
            for level, rx in _STRUCT_RE.items():
                m = rx.match(norm)
                if m:
                    ordinal = m.group("ord")
                    result.structural.append(
                        StructuralMarker(
                            level=level,
                            number=ORDINAL_TO_INT[ordinal],
                            ordinal_text=ordinal,
                            title=_title_after(lines, i),
                            page_number=page.page_number,
                            order=seq,
                        )
                    )
                    seq += 1
                    matched_level = True
                    break
            if matched_level:
                continue

            # Article header?
            am = _ARTICLE_RE.match(arabic_digits_to_ascii(norm))
            if am:
                result.articles.append(
                    ArticleMarker(number=int(am.group("num")),
                                  page_number=page.page_number, order=seq)
                )
                seq += 1
    return result
