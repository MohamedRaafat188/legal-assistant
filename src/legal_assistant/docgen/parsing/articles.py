"""Split an OCR'd عقد/نظام into its articles.

Segmentation is purely structural: find every article heading, and give each
one the verbatim text running up to the next heading. Nothing here decides
what an article *means* -- that is `signatures.py` -- and nothing here
rewrites a single character of the body.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Sequence
from dataclasses import dataclass

from legal_assistant.docgen.numbering import ArticleRef, parse_article_number

# Heading forms seen across the samples, plus مكرر on either side of the
# closing paren:
#   المادة (٦)   مادة (١)   (مادة13)   المادة  ٢١ :   المادة (٦) مكرر   مادة (٧ مكرر)
ARTICLE_HEADING = re.compile(
    r"^[ \t ]*\(?[ \t]*(?:ال)?ماد[ةه][ \t ]*\)?[ \t]*"
    r"[\(\[]?[ \t]*([0-9٠-٩]{1,3})[ \t]*(مكرر(?:ا|ً|اً)?)?[ \t]*[\)\]]?"
    r"[ \t]*(مكرر(?:ا|ً|اً)?)?[ \t]*[:\-\.]?[ \t]*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ExtractedArticle:
    """One article, verbatim.

    `heading` is the matched heading line, `body` the text after it up to the
    next heading, and `start`/`end` index the text passed to
    `segment_articles` so a caller can re-slice or diff against the source.
    `mukarrar` is True for «المادة (٦) مكرر», which is a different article
    from «المادة (٦)».
    """

    number: int
    heading: str
    body: str
    start: int
    end: int
    mukarrar: bool = False

    @property
    def ref(self) -> ArticleRef:
        return ArticleRef(self.number, self.mukarrar)


def segment_articles(text: str) -> list[ExtractedArticle]:
    """Return every article in `text`, in document order.

    Anything before the first heading (title page, preamble) is discarded.
    Returns [] when no heading is found -- the caller degrades to manual
    entry rather than failing.
    """
    matches = list(ARTICLE_HEADING.finditer(text))
    kept: list[tuple[re.Match[str], int, bool]] = []
    for match in matches:
        number = parse_article_number(match.group(1))
        if number is not None:
            mukarrar = bool(match.group(2) or match.group(3))
            kept.append((match, number, mukarrar))
    articles: list[ExtractedArticle] = []

    for index, (match, number, mukarrar) in enumerate(kept):
        body_start = match.end()
        end = kept[index + 1][0].start() if index + 1 < len(kept) else len(text)
        articles.append(
            ExtractedArticle(
                number=number,
                heading=match.group(0).strip(),
                body=text[body_start:end].strip(),
                start=match.start(),
                end=end,
                mukarrar=mukarrar,
            )
        )

    return articles


def preamble_text(text: str) -> str:
    """Everything before the first article heading (all of `text` when there
    is none), stripped. This is what «التمهيد» names in the page map."""
    first = ARTICLE_HEADING.search(text)
    return (text[: first.start()] if first else text).strip()


class LookupStatus(enum.StrEnum):
    found = "found"
    not_found = "not_found"
    ambiguous = "ambiguous"


@dataclass(frozen=True)
class ArticleLookup:
    status: LookupStatus
    article: ExtractedArticle | None = None
    # The article runs to the end of the span with no heading after it, so it
    # may continue on the next page. Only meaningful when found.
    truncated: bool = False


def find_by_ref(articles: Sequence[ExtractedArticle], ref: ArticleRef) -> ArticleLookup:
    """The one article the lawyer named. Fails closed: a number appearing twice
    (the مساهمة العقد الابتدائي / النظام الأساسي collision inside one span) is
    `ambiguous`, never resolved by position."""
    hits = [i for i, a in enumerate(articles) if a.ref == ref]
    if not hits:
        return ArticleLookup(LookupStatus.not_found)
    if len(hits) > 1:
        return ArticleLookup(LookupStatus.ambiguous)
    index = hits[0]
    return ArticleLookup(
        LookupStatus.found, articles[index], truncated=index == len(articles) - 1
    )
