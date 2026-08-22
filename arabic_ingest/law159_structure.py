# -*- coding: utf-8 -*-
"""Slice Companies Law 159/1981 into articles carrying their structural context.

The document is a consolidated annotated edition, so an article body is not one
layer of text but several: the original wording, an italic note naming the
amending law, and the wording that replaced it. Those notes stay in the body
(they are part of this official edition, and the system prompt teaches the
model to read them), and are *also* parsed out into `Article.amendments` so the
corpus can be queried by amending law rather than only read.

Heading shapes
--------------
Both tiers of the document are matched, and they are written differently:

    (مادة ١)                 the six issuance articles (مواد الإصدار)
    مادة ١ :                 an ordinary article of the attached law
    مادة ١ مكرراً :          a bis article
    مادة ١٢٩ مكرراً "١" :    one of a bis *series* -- nine on article 129,
    مادة ١٣٥ مكرراً " د":    four on 135, and note the stray space inside the
                             quotes, which the source is not consistent about

Each member of a series is a separate article with its own text, so the suffix
is carried as a full designation string (`مكرر`, `مكرر ١`, `مكرر أ`) rather
than a bis flag -- see `rag/retrieval.normalize_article_suffix` for what
depends on that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

ORDINALS = {
    "الأول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4, "الخامس": 5,
    "السادس": 6, "السابع": 7, "الثامن": 8, "التاسع": 9, "العاشر": 10,
}
INT_TO_ORDINAL = {v: k for k, v in ORDINALS.items()}
ISSUANCE_ORDINALS = {
    1: "المادة الأولى", 2: "المادة الثانية", 3: "المادة الثالثة",
    4: "المادة الرابعة", 5: "المادة الخامسة", 6: "المادة السادسة",
}


def to_int(arabic: str) -> int:
    return int(arabic.translate(_ARABIC_INDIC))


def to_arabic_digits(value: int | str) -> str:
    return str(value).translate(_TO_ARABIC_INDIC)


# -- line classifiers ---------------------------------------------------------

_ISSUANCE_HEAD = re.compile(r"^\(مادة\s*([٠-٩]+)\)\s*$")
# The sub-designation is quoted, and the source sometimes leaves a space inside
# the quotes ("مكرراً \" د\"") -- so whitespace is tolerated on both sides.
_ARTICLE_HEAD = re.compile(
    r"^مادة\s*([٠-٩]+)\s*"
    r"(?:(مكرر\S*)\s*(?:\"\s*([^\"]{1,4}?)\s*\")?)?"
    r"\s*:\s*$"
)
_BOOK_HEAD = re.compile(r"^الباب\s+(\S+)\s*$")
_CHAPTER_HEAD = re.compile(r"^الفصل\s+(\S+)\s*$")
# «أولاً – المؤسسون» / «ثانياً -إجراءات التأسيس»
_SECTION_HEAD = re.compile(r"^((?:أولاً|ثانياً|ثالثاً|رابعاً|خامساً)\s*[-–]\s*\S.*)$")
# «١ –الاندماج» / «-٢ التفتيش»
_SUBSECTION_HEAD = re.compile(r"^[-–]?\s*[٠-٩]+\s*[-–]\s*(\S.*)$")

# An amendment note: «مستبدلة بالقانون رقم ٤ لسنة ٢٠١٨ – الجريدة الرسمية ...»,
# «الفقرة الثالثة مضافة بالقانون ...», «ملغاة بموجب المادة الرابعة من القانون
# رقم ٣ لسنة ١٩٩٨ ...». The verb may be preceded by which part it applies to.
_AMENDMENT_NOTE = re.compile(
    r"(?P<scope>[^\n]{0,60}?)"
    r"(?P<verb>مستبدلة|مستبدل|مضافة|مضاف|إضافتها|ملغاة|ملغى|محذوفة|تم حذفها)"
    r"[^\n]{0,80}?القانون\s*(?:رقم)?\s*(?P<number>[٠-٩]+)\s*لسنة\s*[-–]?\s*(?P<year>[٠-٩]+)"
)
_VERB_KIND = {
    "مستبدلة": "replace", "مستبدل": "replace",
    # «إضافتها» covers the retrospective form a repeal note uses when it also
    # records where the article came from: «ملغاة بموجب ... وكان قد تم إضافتها
    # بالقانون رقم ٢١٢ لسنة ١٩٩٤» -- without it, law 212/1994 is the one
    # amending statute this corpus would never record.
    "مضافة": "add", "مضاف": "add", "إضافتها": "add",
    "ملغاة": "repeal", "ملغى": "repeal",
    "محذوفة": "delete", "تم حذفها": "delete",
}


@dataclass
class Article:
    kind: str  # "issuance" | "substantive"
    number: int
    suffix: str | None  # "مكرر" / "مكرر ١" / "مكرر أ", else None
    body: str = ""
    book_number: int | None = None
    book_title: str | None = None
    chapter_number: int | None = None
    chapter_title: str | None = None
    section_title: str | None = None
    subsection_title: str | None = None
    status: str = "active"
    amendments: list[dict] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0

    @property
    def key(self) -> tuple[int, str | None]:
        return (self.number, self.suffix)


def _canonical_suffix(marker: str | None, sub: str | None) -> str | None:
    """«مكرراً» + «١» -> «مكرر ١»; «مكرر» alone -> «مكرر»; neither -> None."""
    if not marker:
        return None
    sub = (sub or "").strip()
    return f"مكرر {sub}" if sub else "مكرر"


def parse_amendments(body: str) -> list[dict]:
    """Structured record of every amending law named inside an article body."""
    records: list[dict] = []
    for match in _AMENDMENT_NOTE.finditer(body):
        scope = match.group("scope").strip(" .،-–") or None
        records.append({
            "kind": _VERB_KIND[match.group("verb")],
            "law_number": to_int(match.group("number")),
            "law_year": to_int(match.group("year")),
            "scope": scope,
            "note": match.group(0).strip(),
        })
    return records


# A paragraph that is annotation rather than enacted text: it names an
# amendment verb, cites the Gazette, or is the bare date the citation runs on
# to after a line break. The lookbehind matters -- without it «العددية» in
# «بموافقة الأغلبية العددية» reads as a Gazette citation, and مادة ١٢٧ (whose
# operative text survives; only one phrase in it was struck out) is misread as
# repealed in its entirety.
_ANNOTATION_PARAGRAPH = re.compile(
    r"(?<![ء-ي])(?:مستبدل|مضاف|إضافته|ملغ|محذوف|حذفه|الجريدة الرسمية|بتاريخ|تابع)"
)
_DATE_ONLY = re.compile(r"^[\s.()،٠-٩/–-]*$")


def _is_repealed(body: str, amendments: list[dict]) -> bool:
    """True when the article's whole content is a repeal note.

    A repealed article carries no operative text -- only the annotation saying
    it was repealed. An article merely *amended* by a repealing instrument (a
    بند struck out, say) still has a body, and stays active.

    Tested paragraph by paragraph rather than by subtracting the matched notes:
    a note's Gazette citation regularly runs past the match and over a line
    break, so the leftover text is not a reliable measure of what survives.
    """
    if not any(a["kind"] in ("repeal", "delete") for a in amendments):
        return False
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    return all(
        _ANNOTATION_PARAGRAPH.search(p) or _DATE_ONLY.match(p) for p in paragraphs
    )


def parse(pages: list[str]) -> list[Article]:
    """Slice the enacted text into articles, in document order."""
    articles: list[Article] = []
    current: Article | None = None
    buffer: list[str] = []

    book_number = book_title = None
    chapter_number = chapter_title = None
    section_title = subsection_title = None
    pending_book = pending_chapter = False

    def flush(end_page: int) -> None:
        if current is None:
            return
        current.body = re.sub(r"\n{3,}", "\n\n", "\n".join(buffer).strip())
        current.amendments = parse_amendments(current.body)
        if _is_repealed(current.body, current.amendments):
            current.status = "repealed"
        current.page_end = end_page
        articles.append(current)

    for page_number, page in enumerate(pages, start=1):
        for raw_line in page.split("\n"):
            line = raw_line.strip()

            if pending_book and line:
                book_title, pending_book = line, False
                continue
            if pending_chapter and line:
                chapter_title, pending_chapter = line, False
                continue

            if _BOOK_HEAD.match(line):
                flush(page_number)
                current, buffer = None, []
                book_number = ORDINALS.get(_BOOK_HEAD.match(line).group(1))
                pending_book = True
                chapter_number = chapter_title = None
                section_title = subsection_title = None
                continue
            if _CHAPTER_HEAD.match(line):
                flush(page_number)
                current, buffer = None, []
                chapter_number = ORDINALS.get(_CHAPTER_HEAD.match(line).group(1))
                pending_chapter = True
                section_title = subsection_title = None
                continue

            issuance = _ISSUANCE_HEAD.match(line)
            article = _ARTICLE_HEAD.match(line)
            if issuance or article:
                flush(page_number)
                buffer = []
                if issuance:
                    current = Article(
                        kind="issuance", number=to_int(issuance.group(1)), suffix=None,
                        page_start=page_number,
                    )
                else:
                    current = Article(
                        kind="substantive",
                        number=to_int(article.group(1)),
                        suffix=_canonical_suffix(article.group(2), article.group(3)),
                        book_number=book_number, book_title=book_title,
                        chapter_number=chapter_number, chapter_title=chapter_title,
                        section_title=section_title, subsection_title=subsection_title,
                        page_start=page_number,
                    )
                continue

            # Structural dividers only count between articles; the same shape
            # inside a body is a numbered clause («١ – ...» of a list), not a
            # section heading.
            if current is None or not buffer:
                if _SECTION_HEAD.match(line):
                    section_title = _SECTION_HEAD.match(line).group(1)
                    subsection_title = None
                    continue
                if _SUBSECTION_HEAD.match(line) and len(line) < 60:
                    subsection_title = _SUBSECTION_HEAD.match(line).group(1)
                    continue

            if current is not None:
                buffer.append(raw_line)

    flush(len(pages))
    return articles
