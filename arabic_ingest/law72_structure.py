"""Slice the corrected Law 72/2017 text into issuance + substantive articles.

Mirrors the role `arabic_ingest/articles.py` plays for laws 131/174, but the
72/2017 source uses its own header conventions, so the patterns live here
rather than being bolted onto the shared module:

    issuance    ( المادة الأولى ) ... ( المادة العاشرة )   -- 10 of them
    substantive مادة "١":  /  مادة (٩):                    -- articles 1..94

Structure headers are الباب (book) and الفصل (chapter), each followed by its
title on the next non-blank line. Sub-headings such as "أولا: الحوافز العامة"
are captured as the section title.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, r"E:\DL projects\Legal Assistant\arabic_ingest")
from arabic_text import format_clause_structure  # noqa: E402

AR_D = "٠-٩"

PAGE_RE = re.compile(r"^===== PAGE (\d+) =====$")
# مادة "١":   /   مادة (٩):   /   مادة"٢٣":
ARTICLE_RE = re.compile(rf'^\s*مادة\s*["(]\s*([{AR_D}]+)\s*[")]\s*:?\s*$')
# ( المادة الأولى )
ISSUANCE_RE = re.compile(r"^\s*\(\s*المادة\s+(\S+?)\s*\)\s*$")
# الباب الأول  /  "الفصل الأول"  /  (الفصل الثالث)
DIVISION_RE = re.compile(r'^\s*["(]?\s*(الباب|الفصل)\s+(\S+?)\s*[")]?\s*$')
# أولا: الحوافز العامة   /   ثالثا الحوافز الإضافية   /   -تعريفات
SECTION_RE = re.compile(
    r"^\s*(أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\s*[:\-]?\s*(.*)$"
)

ORDINALS = {
    "الأولى": 1, "الأول": 1, "الثانية": 2, "الثاني": 2, "الثانى": 2,
    "الثالثة": 3, "الثالث": 3, "الرابعة": 4, "الرابع": 4,
    "الخامسة": 5, "الخامس": 5, "السادسة": 6, "السادس": 6,
    "السابعة": 7, "السابع": 7, "الثامنة": 8, "الثامن": 8,
    "التاسعة": 9, "التاسع": 9, "العاشرة": 10, "العاشر": 10,
}

ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def to_arabic_digits(n: int) -> str:
    return str(n).translate(ARABIC_DIGITS)


def from_arabic_digits(s: str) -> int:
    return int(s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))


@dataclass
class Article:
    kind: str                      # "issuance" | "substantive"
    number: int | None
    suffix: str | None = None      # "مكرر" for the added bis articles
    body_lines: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    book_number: int | None = None
    book_title: str | None = None
    chapter_number: int | None = None
    chapter_title: str | None = None
    section_title: str | None = None

    @property
    def body(self) -> str:
        """Faithful article text, بند-structured.

        Same treatment `arabic_ingest/articles.py` gives laws 131/174: the
        PDF's hard line wrapping carries no meaning, so the body is flattened
        to a single line and then re-broken on بند boundaries by
        `format_clause_structure`.
        """
        flat = " ".join(ln.strip() for ln in self.body_lines if ln.strip())
        # This source also writes بند markers with the dash *after* the digit
        # ("٢ -نسبة"), which `format_clause_structure` does not recognise --
        # it only expects the dash first. Canonicalise to "٢- " so the shared
        # formatter sees a boundary it knows.
        flat = re.sub(rf"(^|[.:؛])\s*([{AR_D}]+)\s+-\s*", "\\1\n\\2- ", flat)
        return format_clause_structure(flat)


def _next_title(lines: list[str], i: int) -> tuple[int, str] | None:
    """The first non-blank, non-structural line after ``i``, as (index, title).

    The index is returned alongside the text because the title is tidied
    (the source decorates some headings with a leading dash or quotes), so it
    can no longer be looked up by value in ``lines``.
    """
    for j in range(i + 1, min(i + 4, len(lines))):
        s = lines[j].strip()
        if not s or PAGE_RE.match(s):
            continue
        if DIVISION_RE.match(s) or ARTICLE_RE.match(s) or ISSUANCE_RE.match(s):
            return None
        title = s.strip('"‘’ -').strip()
        return (j, title) if title else None
    return None


def _heading_precedes_article(lines: list[str], i: int) -> bool:
    """True if the next non-blank line after ``i`` opens an article or division."""
    for j in range(i + 1, min(i + 5, len(lines))):
        s = lines[j].strip()
        if not s or PAGE_RE.match(s):
            continue
        return bool(ARTICLE_RE.match(s) or ISSUANCE_RE.match(s) or DIVISION_RE.match(s))
    return False


def parse(text: str) -> list[Article]:
    lines = text.split("\n")
    articles: list[Article] = []
    cur: Article | None = None
    page = 0
    book_n = book_t = chap_n = chap_t = section_t = None
    consumed_title_at = -1

    for i, line in enumerate(lines):
        s = line.strip()

        m = PAGE_RE.match(s)
        if m:
            page = int(m.group(1))
            if cur:
                cur.page_end = page
            continue

        if i == consumed_title_at:
            continue

        m = DIVISION_RE.match(s)
        if m and s:
            kind, ordinal = m.group(1), m.group(2)
            found = _next_title(lines, i)
            title = None
            if found is not None:
                consumed_title_at, title = found
            if kind == "الباب":
                book_n, book_t = ORDINALS.get(ordinal), title
                chap_n = chap_t = section_t = None
            else:
                chap_n, chap_t = ORDINALS.get(ordinal), title
                section_t = None
            cur = None
            continue

        m = ISSUANCE_RE.match(s)
        if m:
            n = ORDINALS.get(m.group(1))
            cur = Article("issuance", n, page_start=page, page_end=page)
            articles.append(cur)
            continue

        m = ARTICLE_RE.match(s)
        if m:
            cur = Article(
                "substantive", from_arabic_digits(m.group(1)),
                page_start=page, page_end=page,
                book_number=book_n, book_title=book_t,
                chapter_number=chap_n, chapter_title=chap_t,
                section_title=section_t,
            )
            articles.append(cur)
            continue

        # A section heading ("ثالثا الحوافز الإضافية") can appear while an
        # article is still open, so it cannot be gated on `cur is None` --
        # article 12's body would swallow the heading that introduces 13.
        # It is only a heading if the next real line opens an article or a
        # division; an ordinal used as a بند inside a body never does.
        m = SECTION_RE.match(s)
        if m and len(s) < 60 and _heading_precedes_article(lines, i):
            section_t = (m.group(2) or "").strip() or None
            cur = None
            continue

        if cur is not None:
            cur.body_lines.append(line)

    return articles
