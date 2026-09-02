"""Split an OCR'd file into the legal instruments it contains.

A GAFI مساهمة file is two documents in one: العقد الابتدائي (مواد ١-١٠) and
then النظام الأساسي (مواد ١-٦٥). Their article numbers collide, and the
محضر مساهمة template amends the النظام الأساسي, so the target series has to
be chosen explicitly. Getting this wrong quotes the wrong "المادة السادسة"
into a filed document.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from legal_assistant.docgen.arabic import normalize_for_match
from legal_assistant.docgen.parsing.articles import ExtractedArticle, segment_articles


class Instrument(enum.StrEnum):
    PRELIMINARY = "preliminary"
    ARTICLES_OF_ASSOCIATION = "articles_of_association"


@dataclass(frozen=True)
class Section:
    instrument: Instrument
    articles: list[ExtractedArticle]


# Company type -> the series its template amends. Straight from the spec's
# "Target document series" table.
TARGET_INSTRUMENT: dict[str, Instrument] = {
    "shakhs_wahed": Instrument.ARTICLES_OF_ASSOCIATION,
    "zmm": Instrument.ARTICLES_OF_ASSOCIATION,
    "masahma": Instrument.ARTICLES_OF_ASSOCIATION,
}

_PRELIMINARY_TITLES = (
    "العقد الابتدائي",
    "عقد ابتدائي",
    # PyMuPDF flattens right-to-left text and transposes the lam-alef
    # ligature in the preliminary instrument's title, so the extracted text
    # layer of the real مساهمة fixture reads "العقد االبتدائي" (doubled
    # alef) instead of "العقد الابتدائي". This is not a typo -- do not
    # "correct" it away. Matching stays literal (no fuzzy/transposition
    # tolerance) per this subsystem's no-fuzzy-matching rule; this is the
    # one observed OCR/extraction variant, added verbatim.
    "العقد االبتدائي",
)


# How far into the tail of the preceding article's body to look for a title.
# `ExtractedArticle.end` is the start of the *next kept heading*, so a new
# group's first heading immediately follows the previous group's last
# article -- there is no gap between them. A mid-document title therefore
# does not sit in the gap; it got swallowed into the tail of that previous
# article's `body`, because `body` runs all the way to the next heading.
_TITLE_TAIL_CHARS = 300


def _looks_preliminary(text: str) -> bool:
    normalized = normalize_for_match(text)
    return any(normalize_for_match(t) in normalized for t in _PRELIMINARY_TITLES)


def split_instruments(text: str) -> list[Section]:
    """Group the document's articles into instruments.

    A new instrument starts wherever the article number fails to increase --
    the restart is the only signal present in every sample, since the titles
    are OCR-fragile. Whichever text precedes each group then decides which
    instrument it is: a group preceded by a title reading العقد الابتدائي is
    PRELIMINARY, and everything else (including a single-instrument
    document) is the ARTICLES_OF_ASSOCIATION. For the first group, that text
    is whatever precedes the first heading in the whole document. For every
    later group, the title lives at the end of the previous group's last
    article's body (see `_TITLE_TAIL_CHARS`), since `ExtractedArticle.end`
    is defined as the start of the next kept heading -- there is no gap
    between one group's last article and the next group's first heading for
    a title to occupy on its own.
    """
    articles = segment_articles(text)
    if not articles:
        return []

    groups: list[list[ExtractedArticle]] = [[articles[0]]]
    for previous, current in zip(articles, articles[1:], strict=False):
        if current.number <= previous.number:
            groups.append([current])
        else:
            groups[-1].append(current)

    sections: list[Section] = []
    for index, group in enumerate(groups):
        if index == 0:
            title_region = text[: group[0].start]
        else:
            preceding_body = groups[index - 1][-1].body
            title_region = preceding_body[-_TITLE_TAIL_CHARS:]
        instrument = (
            Instrument.PRELIMINARY
            if _looks_preliminary(title_region)
            else Instrument.ARTICLES_OF_ASSOCIATION
        )
        sections.append(Section(instrument=instrument, articles=group))

    return sections


def select_target(
    sections: list[Section], company_type: str
) -> tuple[list[ExtractedArticle], str | None]:
    """Return the articles of the series this company type's template amends.

    On a miss, returns ([], warning) rather than falling back to whatever
    else is in the file. Silently segmenting the wrong instrument is worse
    than telling the lawyer nothing was found.
    """
    wanted = TARGET_INSTRUMENT[company_type]
    matching = [s for s in sections if s.instrument is wanted]
    if not matching:
        return [], (
            "لم يتم العثور على النظام الأساسى للشركة داخل الملف المرفوع. "
            "راجع الملف أو أدخل نص المواد يدويا."
        )
    if len(matching) > 1:
        # OCR noise (a misread digit, a duplicated heading) can produce a
        # spurious article-number restart *inside* the target series, which
        # splits it into two groups with no title between them to tell them
        # apart. Do not guess which fragment is correct or merge them --
        # that is guesswork about legal text. Use the last one (the نموذج's
        # ordering puts the operative instrument last) but say so, so a
        # dropped fragment of amendable articles is never silent.
        return matching[-1].articles, (
            f"تم العثور على {len(matching)} أقسام يبدو أنها تنتمي إلى النظام الأساسى "
            "للشركة داخل الملف المرفوع بدلا من قسم واحد. تم استخدام آخر قسم؛ "
            "يرجى مراجعة الملف للتأكد من عدم فقدان مواد."
        )
    # A document can only have one of each series; this is the ordinary,
    # unambiguous case.
    return matching[-1].articles, None
