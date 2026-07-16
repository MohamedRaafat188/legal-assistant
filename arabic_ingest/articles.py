# -*- coding: utf-8 -*-
"""Article-level extraction — the single source of truth for turning cleaned
pages into structured article records.

Both the human-review tool (`preview_articles.py`) and the production chunker
(`chunker.py`) build on `iter_articles`, so the slicing/boundary logic lives in
exactly one place.

An article record carries the faithful body text plus its full structural
context (division / book / part / chapter / section / subsection) and
law-level identity. It does NOT yet carry the embedding copy, the prepended
header, or the chunk id — those are added by the chunker, which is the layer
that produces vector-DB-ready chunks.

Two source typesets are supported by the same traversal:

* Law 174 style: a structural/article header sits ALONE on its line (title,
  if any, on the following line(s)); article markers are "مادة (N)" with no
  trailing body text.
* Law 131 (Civil Code) style: division/book/part titles are routinely INLINE
  (same line, after a dash); chapter titles are ALWAYS inline; article markers
  are glued directly to their body ("مادة – Nبدأ الجسم..."); book/part headers
  are sometimes duplicated by a "running banner" line that must be
  deduplicated against the real bare/inline header (see `_advance_book_part`).

Both are tried at every position; whichever matches wins, so a single pass
handles either law's PDF without branching on which law is being parsed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from pdf_extractor import PdfTextExtractor, PageText
from arabic_text import (arabic_digits_to_ascii, normalize_for_match,
                         format_clause_structure)
from structure import (_ARTICLE_RE, _ISSUANCE_RE, _STRUCT_RE, _is_running_header,
                       _is_page_number, _title_after, ORDINAL_TO_INT,
                       FEM_ORDINAL_TO_INT, INT_TO_FEM_ORDINAL, Level,
                       _INLINE_TITLE_RE, _MERGED_BOOK_PART_RE, _CHAPTER_INLINE_RE,
                       _PREAMBLE_RE, _ARTICLE_INLINE_RE, _REPEALED_RANGE_RE,
                       _SECTION_RE, _SUBSECTION_RE, _SUBSECTION_MAX_WORDS)

ArticleType = Literal["substantive", "issuance"]
ArticleStatus = Literal["active", "repealed"]
PartKind = Literal["normal", "preamble"]


@dataclass(frozen=True, slots=True)
class LawMetadata:
    """Law-level identity, supplied once per source document.

    Kept separate from article records so the same extraction logic serves any
    Egyptian law; only this object changes per source.
    """
    law_name: str
    law_number: int
    law_year: int
    jurisdiction: str = "مصر"
    issue_ref: Optional[str] = None   # gazette issue, e.g. الجريدة الرسمية العدد ٤٥ مكرر (د)
    source_file: Optional[str] = None


@dataclass(slots=True)
class ArticleRecord:
    """One extracted article with its structural context (faithful text only)."""
    article_type: ArticleType
    article_number: Optional[int]      # 1..N for substantive; None for issuance
    issuance_number: Optional[int]     # 1..N for issuance; None for substantive
    body: str                          # faithful article text, بند-structured
    article_status: ArticleStatus = "active"
    repealed_range: Optional[str] = None   # e.g. "54-80", set only when repealed
    # structural context (all None for issuance articles)
    division_number: Optional[int] = None
    division_title: Optional[str] = None
    book_number: Optional[int] = None
    book_title: Optional[str] = None
    part_number: Optional[int] = None
    part_title: Optional[str] = None
    part_kind: PartKind = "normal"
    chapter_number: Optional[int] = None
    chapter_title: Optional[str] = None
    section_number: Optional[int] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    # provenance
    page_start: int = 0
    page_end: int = 0
    char_count: int = 0


@dataclass(slots=True)
class ParseDiagnostics:
    """Non-fatal findings collected during extraction, for human review."""
    subsections: list[dict] = field(default_factory=list)
    unclassified: list[dict] = field(default_factory=list)


_WS_RE = re.compile(r"\s+")


def _word_count(text: str) -> int:
    return len(_WS_RE.split(text.strip())) if text.strip() else 0


_DASH_RE = re.compile(r"[-–—]")


def _faithful_tail(s: str, dash_count: int, prefix_word_count: int) -> str:
    """Recover a header's faithful (non-normalized) title tail.

    Several header patterns (division/book/part/chapter, and the merged
    book+part banner) are MATCHED against normalized text so ordinal spelling
    variants collapse, but their captured `title` group must never be taken
    from that normalized match -- normalize_for_match folds hamzas, yaa
    variants and strips harakat/tatweel, and the title is faithful,
    citable text. Instead, re-locate the title's boundary directly in the
    original faithful line `s`: split on its Nth dash if the header uses one
    (the normal case), or fall back to skipping `prefix_word_count` words
    (keyword + ordinal) when there is no dash -- safe because tatweel
    stretching only ever occurs INSIDE a word, never introducing a new
    whitespace-separated token.
    """
    if dash_count > 0:
        positions = [m.start() for m in _DASH_RE.finditer(s)]
        if len(positions) >= dash_count:
            return s[positions[dash_count - 1] + 1:].strip()
    words = s.split(maxsplit=prefix_word_count)
    return words[prefix_word_count].strip() if len(words) > prefix_word_count else ""


def iter_articles(
    pages: list[PageText],
    diagnostics: Optional[ParseDiagnostics] = None,
) -> list[ArticleRecord]:
    """Walk cleaned pages in document order and slice article records.

    Tracks division/book/part/chapter/section/subsection as state, captures
    issuance articles (before the first structural marker) and substantive
    articles (inside the tree), expands nothing itself (repealed-range
    expansion is the chunker's job -- this layer emits one placeholder
    ArticleRecord per range, tagged `repealed_range`, that the chunker fans
    out into per-number chunks), strips page furniture, joins wrapped lines,
    and reconstructs بند structure.
    """
    ctx = {
        "division": (None, None),
        "book": (None, None),
        "part": (None, None),
        "part_kind": "normal",
        "chapter": (None, None),
        "section": (None, None),
        "subsection": None,
    }
    out: list[ArticleRecord] = []
    cur: Optional[ArticleRecord] = None
    lines_buf: list[str] = []
    first_structural_seen = False
    page_number = 0
    # True once the open article's body has introduced a colon-terminated
    # list ("... هي :" -> "-1الدولة..." -> "-2الهيئات..."). While true, "N-title"
    # / colon-label lines are enumerated list items belonging to THIS article,
    # not new sections/subsections -- e.g. مادة 52's six-item list uses the
    # exact same "-N title" shape as a genuine top-level section header, and
    # only this content-grounded signal (not "cur is None") tells them apart.
    cur_has_open_list = False
    # True from the moment an article opens until its first body line has been
    # seen. Needed because Law 174's header sits ALONE on its line (seed is
    # always empty; the real opening sentence is the first line appended via
    # case 8), while Law 131's is glued to its seed -- so "is the OPENING
    # sentence colon-terminated" can only be decided uniformly by watching
    # for the first body line appended, whichever path it comes from.
    cur_awaiting_first_line = False

    def close() -> None:
        nonlocal cur, lines_buf, cur_has_open_list, cur_awaiting_first_line
        if cur is not None:
            cur.body = format_clause_structure(" ".join(lines_buf))
            cur.char_count = len(cur.body)
            out.append(cur)
        cur, lines_buf, cur_has_open_list, cur_awaiting_first_line = None, [], False, False

    def reset_below(level: str) -> None:
        """Clear descendant context when a higher level's ordinal changes."""
        order = ["division", "book", "part", "chapter", "section", "subsection"]
        for lvl in order[order.index(level) + 1:]:
            if lvl == "subsection":
                ctx["subsection"] = None
            elif lvl == "part":
                ctx["part"] = (None, None)
                ctx["part_kind"] = "normal"
            else:
                ctx[lvl] = (None, None)

    def append_body(text: str) -> None:
        nonlocal cur_has_open_list, cur_awaiting_first_line
        lines_buf.append(text)
        if cur_awaiting_first_line:
            # The article's OPENING SENTENCE routinely wraps across several
            # physical lines before reaching its terminal punctuation ("مادة
            # 114" wraps 2 lines before its colon) -- so this must keep
            # accumulating until a sentence actually ends, not just look at
            # the literal first line.
            stripped = text.rstrip()
            if stripped.endswith((".", ":", "：", "؟")):
                # A colon-ended opening ONLY needs to suppress section/
                # subsection detection for the rest of this article's body
                # when what follows is actually "-N title"-shaped (e.g. مادة
                # 52's "الأشخاص الاعتبارية هي :" -> "-1الدولة..."), since
                # that's the one shape indistinguishable from a genuine
                # top-level section header. A colon introducing a LETTERED
                # list ("...بأحد التدابير الآتية:" -> "أولاً ...") never
                # collides with that shape, so it's safe to leave detection
                # on -- otherwise a genuine sub-section immediately following
                # such an article (before the next مادة) would be wrongly
                # swallowed into this article's body instead of closing it.
                cur_has_open_list = False
                if stripped.endswith((":", "：")):
                    nxt = peek_next_nonblank(lines, i + 1)
                    if nxt and _SECTION_RE.match(arabic_digits_to_ascii(nxt)):
                        cur_has_open_list = True
                cur_awaiting_first_line = False

    def open_article(article_type: ArticleType, number: Optional[int],
                      issuance_number: Optional[int], seed_text: str) -> None:
        nonlocal cur, lines_buf, cur_awaiting_first_line
        close()
        cur_awaiting_first_line = True
        cur = ArticleRecord(
            article_type=article_type, article_number=number,
            issuance_number=issuance_number, body="",
            division_number=ctx["division"][0], division_title=ctx["division"][1],
            book_number=ctx["book"][0], book_title=ctx["book"][1],
            part_number=ctx["part"][0], part_title=ctx["part"][1],
            part_kind=ctx["part_kind"],
            chapter_number=ctx["chapter"][0], chapter_title=ctx["chapter"][1],
            section_number=ctx["section"][0], section_title=ctx["section"][1],
            subsection_title=ctx["subsection"],
            page_start=page_number, page_end=page_number,
        )
        seed = seed_text.strip()
        if seed:
            append_body(seed)
        # else: Law 174-style header alone on its line -- the first REAL body
        # line arrives later via case 8's append_body call, which is what
        # `cur_awaiting_first_line` is watching for.

    def emit_repealed_range(a: int, b: int) -> None:
        """Placeholder record for a collapsed 'المواد من A إلى B ملغاة' line.

        The chunker expands this into one lightweight chunk per article
        number (Option B, per project spec). `body` here is a bookkeeping
        note only -- the chunker builds the mandated faithful text itself.
        """
        close()
        rec = ArticleRecord(
            article_type="substantive", article_number=None, issuance_number=None,
            body=f"المواد من {a} إلى {b} ملغاة", article_status="repealed",
            repealed_range=f"{a}-{b}",
            division_number=ctx["division"][0], division_title=ctx["division"][1],
            book_number=ctx["book"][0], book_title=ctx["book"][1],
            part_number=ctx["part"][0], part_title=ctx["part"][1],
            part_kind=ctx["part_kind"],
            chapter_number=ctx["chapter"][0], chapter_title=ctx["chapter"][1],
            section_number=ctx["section"][0], section_title=ctx["section"][1],
            subsection_title=ctx["subsection"],
            page_start=page_number, page_end=page_number,
            char_count=0,
        )
        out.append(rec)

    def advance_division(num: int, title: str) -> None:
        if num == ctx["division"][0]:
            if not ctx["division"][1] and title:
                ctx["division"] = (num, title)
            return  # redundant repeat, nothing below changes
        ctx["division"] = (num, title)
        reset_below("division")

    def advance_book(num: int, title: Optional[str]) -> None:
        if num == ctx["book"][0]:
            if not ctx["book"][1] and title:
                ctx["book"] = (num, title)  # finalize a provisional bump
            return
        ctx["book"] = (num, title)
        reset_below("book")

    def advance_part(num: int, title: Optional[str], kind: PartKind = "normal") -> None:
        if num == ctx["part"][0] and kind == ctx["part_kind"]:
            if not ctx["part"][1] and title:
                ctx["part"] = (num, title)
            return
        ctx["part"] = (num, title)
        ctx["part_kind"] = kind
        reset_below("part")

    def advance_chapter(num: int, title: str) -> None:
        if num == ctx["chapter"][0]:
            if not ctx["chapter"][1] and title:
                ctx["chapter"] = (num, title)
            return
        ctx["chapter"] = (num, title)
        reset_below("chapter")

    def advance_section(num: int, title: str) -> None:
        if num == ctx["section"][0]:
            return
        ctx["section"] = (num, title)
        reset_below("section")

    def handle_book_part_banner(m: "re.Match[str]", s: str) -> None:
        """Book(+embedded part) running-banner: 'الكتاب X -[الباب Y -]title'.

        A book+part banner's title always names the PART, never the book
        (empirically confirmed), so the book component is only ever bumped
        with no title -- a subsequent bare/inline book header finalizes it.
        A book-ONLY banner/inline line carries the book's own real title and
        is registered directly.
        """
        bord = ORDINAL_TO_INT[m.group("bord")]
        pord_word = m.group("pord")
        title = _faithful_tail(s, dash_count=2 if pord_word else 1, prefix_word_count=0)
        if pord_word is None:
            advance_book(bord, title)
            return
        if bord != ctx["book"][0]:
            ctx["book"] = (bord, None)
            reset_below("book")
        pord = ORDINAL_TO_INT[pord_word]
        advance_part(pord, title)

    def peek_next_nonblank(lines: list[str], start: int) -> Optional[str]:
        for ln in lines[start:]:
            s = ln.strip()
            if s:
                return s
        return None

    def _next_line_is_article(lines: list[str], start: int) -> bool:
        nxt = peek_next_nonblank(lines, start)
        if nxt is None:
            return False
        nxt_na = arabic_digits_to_ascii(normalize_for_match(nxt))
        return bool(_ARTICLE_RE.match(nxt_na) or _ARTICLE_INLINE_RE.match(nxt_na))

    for page in pages:
        page_number = page.page_number
        lines = page.text.split("\n")
        for i, raw in enumerate(lines):
            s = raw.strip()
            if not s:
                continue
            norm = normalize_for_match(s)
            if _is_running_header(norm) or _is_page_number(s):
                continue

            # 1) division (bare or inline-titled)
            m = _INLINE_TITLE_RE[Level.DIVISION].match(norm)
            if m:
                title = _faithful_tail(s, dash_count=1, prefix_word_count=0)
                advance_division(ORDINAL_TO_INT[m.group("ord")], title)
                first_structural_seen = True
                continue
            m = _STRUCT_RE[Level.DIVISION].match(norm)
            if m:
                advance_division(ORDINAL_TO_INT[m.group("ord")], _title_after(lines, i))
                first_structural_seen = True
                continue

            # 2) book, possibly with an embedded part (running banner)
            m = _MERGED_BOOK_PART_RE.match(norm) if norm.startswith("الكتاب") else None
            if m:
                handle_book_part_banner(m, s)
                first_structural_seen = True
                continue
            m = _STRUCT_RE[Level.BOOK].match(norm)
            if m:
                advance_book(ORDINAL_TO_INT[m.group("ord")], _title_after(lines, i))
                first_structural_seen = True
                continue

            # 3) part (bare or inline-titled)
            m = _INLINE_TITLE_RE[Level.PART].match(norm)
            if m:
                title = _faithful_tail(s, dash_count=1, prefix_word_count=0)
                advance_part(ORDINAL_TO_INT[m.group("ord")], title)
                first_structural_seen = True
                continue
            m = _STRUCT_RE[Level.PART].match(norm)
            if m:
                advance_part(ORDINAL_TO_INT[m.group("ord")], _title_after(lines, i))
                first_structural_seen = True
                continue

            # 4) باب تمهيدي preamble (a part with no ordinal, division=book=None)
            m = _PREAMBLE_RE.match(norm)
            if m:
                title = (_faithful_tail(s, dash_count=1, prefix_word_count=2)
                         if m.group("title") else _title_after(lines, i))
                ctx["division"] = (None, None)
                ctx["book"] = (None, None)
                advance_part(None, title, kind="preamble")
                first_structural_seen = True
                continue

            # 5) chapter (title always inline in Law 131; falls back to next-line
            #    lookup for Law 174, where the title-inline group is empty)
            if norm.startswith("الفصل"):
                m = _CHAPTER_INLINE_RE.match(norm)
                if m:
                    ord_word_count = len(m.group("ord").split())
                    title = (_faithful_tail(s, dash_count=1, prefix_word_count=1 + ord_word_count)
                             if m.group("title").strip() else _title_after(lines, i))
                    advance_chapter(ORDINAL_TO_INT[m.group("ord")], title)
                    first_structural_seen = True
                    continue

            # 6) issuance article (only before the first structural marker) --
            #    Law 174 style (feminine ordinal, alone on its line) or Law 131
            #    style (cardinal digit, glued to its body).
            if not first_structural_seen:
                im = _ISSUANCE_RE.match(norm)
                if im:
                    open_article("issuance", None, FEM_ORDINAL_TO_INT[im.group("ord")], "")
                    continue
                im2 = _ARTICLE_INLINE_RE.match(arabic_digits_to_ascii(norm))
                if im2:
                    n = int(im2.group("num"))
                    seed = _seed_with_clause(im2)
                    open_article("issuance", None, n, seed)
                    continue

            # 7) substantive article -- Law 174 (alone on line) tried first,
            #    then Law 131 (glued to body).
            na = arabic_digits_to_ascii(norm)
            am = _ARTICLE_RE.match(na)
            if am:
                open_article("substantive", int(am.group("num")), None, "")
                continue
            am2 = _ARTICLE_INLINE_RE.match(na)
            if am2:
                n = int(am2.group("num"))
                seed = _seed_with_clause(am2)
                open_article("substantive", n, None, seed)
                continue

            # From here on, a marker is only genuine when it doesn't belong to
            # a colon-introduced enumerated list inside the CURRENTLY OPEN
            # article's own body -- the same "N-title" / colon-label shapes
            # recur there (e.g. مادة 52's six-item list), and are excluded by
            # this content-grounded guard rather than "cur is None" (which
            # would deadlock: these markers are themselves what closes the
            # article, so they must be checked even while cur is still open).
            if not cur_has_open_list:
                rm = _REPEALED_RANGE_RE.match(na)
                if rm:
                    emit_repealed_range(int(rm.group("a")), int(rm.group("b")))
                    continue

                # A leading paren is always a BiDi-glued clause marker
                # (") (2فيكون له :", "(1) ...") -- genuine section/subsection
                # titles never start with one, so this rules out clause
                # continuations of the currently open article that happen to
                # be short and/or colon-terminated.
                starts_with_clause_marker = s.startswith(("(", ")"))

                # _SECTION_RE only keys on digits/dashes/dots, never on Arabic
                # letter shapes -- match against digit-only-converted text
                # (not the fully normalized `na`) so the captured title is the
                # faithful original, no separate faithful-tail lookup needed.
                s_ascii_digits = arabic_digits_to_ascii(s)
                sm = _SECTION_RE.match(s_ascii_digits)
                if sm and len(s_ascii_digits) < 80 and not starts_with_clause_marker:
                    close()
                    num = int(sm.group("n1") or sm.group("n2"))
                    advance_section(num, sm.group("title").strip())
                    continue

                # Per brief §2.5(c), a genuine sub-section always sits directly
                # before the next ARTICLE header -- never before a lettered/
                # numbered clause item introduced by the sentence it ends
                # ("...في الأحوال الآتية:" -> "(أ) ..." is clause prose, not a
                # sub-section, even though it's short and colon-terminated).
                next_is_article = _next_line_is_article(lines, i + 1)

                bm = _SUBSECTION_RE.match(s)
                # A title ending in "الآتية"/"التالية" ("...as follows") is a
                # sentence fragment referring FORWARD to what follows it (a
                # clause list or a cross-referenced group of articles), never
                # a genuine standalone topic label -- e.g. "المقررة فى المواد
                # الآتية" ("...stipulated in the following articles").
                title_words = normalize_for_match(bm.group("title")).split() if bm else []
                ends_with_forward_ref = bool(title_words) and title_words[-1] in (
                    "الاتية", "التالية"
                )

                if (bm and not starts_with_clause_marker and next_is_article
                        and not ends_with_forward_ref
                        and _word_count(bm.group("title")) <= _SUBSECTION_MAX_WORDS):
                    close()
                    title = bm.group("title").strip()
                    ctx["subsection"] = title
                    if diagnostics is not None:
                        diagnostics.subsections.append(
                            {"page": page_number, "text": s, "title": title,
                             "has_colon": True}
                        )
                    continue

                # Narrow exception (see brief §2.5/§2.7): a short bare label
                # (no colon) directly preceding a repealed-range line is also
                # a genuine sub-section heading (e.g. "الجمعيات").
                if (not s.endswith((":", "：")) and _word_count(s) <= _SUBSECTION_MAX_WORDS
                        and len(s) < 60):
                    nxt = peek_next_nonblank(lines, i + 1)
                    if nxt and _REPEALED_RANGE_RE.match(arabic_digits_to_ascii(
                            normalize_for_match(nxt))):
                        close()
                        ctx["subsection"] = s
                        if diagnostics is not None:
                            diagnostics.subsections.append(
                                {"page": page_number, "text": s, "title": s,
                                 "has_colon": False}
                            )
                        continue

            # 8) body line, or genuinely unclassified furniture
            if cur is not None:
                append_body(s)
                cur.page_end = page_number
            elif diagnostics is not None:
                diagnostics.unclassified.append({"page": page_number, "text": s})
    close()
    return out


def _seed_with_clause(m: "re.Match[str]") -> str:
    """Reconstruct the article's first body fragment from an inline match.

    A leading clause marker glued before the article number by BiDi
    reordering ("مادة (1) – 1تسري...") is re-attached to the body text so
    `format_clause_structure` still recognizes it as a بند boundary.
    """
    rest = m.group("rest")
    clause = m.group("clause")
    if clause:
        return f"({clause}) {rest.lstrip()}"
    return rest


def extract_articles_from_pdf(
    pdf_path: str, diagnostics: Optional[ParseDiagnostics] = None
) -> list[ArticleRecord]:
    """Convenience: extract straight from a PDF path."""
    return iter_articles(PdfTextExtractor().extract_pages(pdf_path), diagnostics)
