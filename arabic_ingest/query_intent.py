# -*- coding: utf-8 -*-
"""Detect an explicit article reference in a free-text query.

Lawyer queries that name a specific article ("نص المادة ٢ من قانون
الإجراءات الجنائية؟") want that exact article, not the nearest semantic
neighbor. Hybrid (dense+sparse) search routinely misranks these: once
diacritics and function words are stripped, "المادة ٢" carries almost no
lexical signal to distinguish it from dozens of other articles that are
thematically closer to the rest of the query. Detect the reference here so
the caller can do an exact metadata lookup (Retriever.lookup_article)
instead of ranked search.

Egyptian-law article numbering has two disjoint namespaces (see
structure.py), so the two reference *forms* map directly to
``article_type``:
  * a bare number ("مادة ٢", "مادة (٢)", "مادة رقم 2")   -> substantive article
  * a feminine ordinal word ("المادة الثانية")            -> issuance article
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from arabic_text import arabic_digits_to_ascii, normalize_for_match
from structure import FEM_ORDINAL_TO_INT


@dataclass(frozen=True, slots=True)
class ArticleReference:
    """An explicit article reference parsed out of a query."""
    article_number: int
    article_type: str                  # "substantive" | "issuance"
    law_number: Optional[int] = None
    law_year: Optional[int] = None


_FEM_ORDINAL_ALT = "|".join(
    sorted((re.escape(k) for k in FEM_ORDINAL_TO_INT), key=len, reverse=True)
)
# "(ال)مادة الثانية" (feminine ordinal word) -> issuance article.
_ISSUANCE_REF_RE = re.compile(rf"(?:ال)?ماد[ةه]\s+(?P<ord>{_FEM_ORDINAL_ALT})\b")

# "(ال)مادة ٢" / "مادة (٢)" / "المادة رقم 2" (digits) -> substantive article.
_DIGIT_REF_RE = re.compile(
    r"(?:ال)?ماد[ةه]\s*(?:رقم)?\s*[()]*\s*(?P<num>[0-9]+)\s*[()]*"
)

# "قانون ... رقم 174" / "لسنة 2025" -- scope the lookup to a specific law,
# when the query names one. Bounded gap so "قانون" doesn't pair with an
# unrelated "رقم" much later in the sentence.
_LAW_NUMBER_RE = re.compile(r"قانون[\s\S]{0,60}?رقم\s*[()]*\s*(?P<num>[0-9]+)")
_LAW_YEAR_RE = re.compile(r"لسنة\s*[()]*\s*(?P<num>[0-9]+)")

# Laws are also named, not just numbered ("المادة ٥ من القانون المدني"). Map the
# name -> (number, year) so a by-name query still scopes to one law. Keys are
# normalized through the same pipeline applied to the query text (below), so we
# match on folded forms and don't have to hand-guess the folding. Order matters:
# the most specific name first, so a bare "قانون" can't shadow it.
_NAMED_LAWS_RAW: list[tuple[str, int, int]] = [
    ("قانون الإجراءات الجنائية", 174, 2025),
    ("القانون المدني", 131, 1948),
]
_NAMED_LAWS: list[tuple[str, int, int]] = [
    (arabic_digits_to_ascii(normalize_for_match(name)), num, year)
    for name, num, year in _NAMED_LAWS_RAW
]


def _named_law(text: str) -> tuple[Optional[int], Optional[int]]:
    """Return (law_number, law_year) if the (normalized) text names a known law."""
    for name, num, year in _NAMED_LAWS:
        if name in text:
            return num, year
    return None, None


def extract_article_reference(query: str) -> Optional[ArticleReference]:
    """Parse an explicit article reference out of a free-text query.

    Returns None when the query doesn't name a specific article, so the
    caller should fall back to ranked semantic/lexical search.
    """
    text = arabic_digits_to_ascii(normalize_for_match(query))

    law_number = int(m.group("num")) if (m := _LAW_NUMBER_RE.search(text)) else None
    law_year = int(m.group("num")) if (m := _LAW_YEAR_RE.search(text)) else None
    # Fall back to a named law ("القانون المدني") when no explicit رقم/لسنة given.
    if law_number is None or law_year is None:
        named_num, named_year = _named_law(text)
        law_number = law_number if law_number is not None else named_num
        law_year = law_year if law_year is not None else named_year

    if m := _ISSUANCE_REF_RE.search(text):
        return ArticleReference(
            article_number=FEM_ORDINAL_TO_INT[m.group("ord")],
            article_type="issuance",
            law_number=law_number, law_year=law_year,
        )
    if m := _DIGIT_REF_RE.search(text):
        return ArticleReference(
            article_number=int(m.group("num")),
            article_type="substantive",
            law_number=law_number, law_year=law_year,
        )
    return None
