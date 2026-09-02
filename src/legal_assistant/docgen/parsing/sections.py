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


class Instrument(str, enum.Enum):  # noqa: UP042 -- matches db.models.MessageRole's style
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


def _looks_preliminary(text: str) -> bool:
    normalized = normalize_for_match(text)
    return any(normalize_for_match(t) in normalized for t in _PRELIMINARY_TITLES)


def split_instruments(text: str) -> list[Section]:
    """Group the document's articles into instruments.

    A new instrument starts wherever the article number fails to increase --
    the restart is the only signal present in every sample, since the titles
    are OCR-fragile. The heading text before each group then decides which
    instrument it is: a group titled العقد الابتدائي is PRELIMINARY, and
    everything else (including a single-instrument document) is the
    ARTICLES_OF_ASSOCIATION.
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
        # Look at the text between the previous group's end and this group's
        # first heading -- that is where a document title sits.
        title_start = 0 if index == 0 else groups[index - 1][-1].end
        title_region = text[title_start : group[0].start]
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
    # A document can only have one of each series; if OCR noise produced
    # several, the last one is the operative instrument (the نموذج puts the
    # النظام الأساسي after the العقد الابتدائي).
    return matching[-1].articles, None
