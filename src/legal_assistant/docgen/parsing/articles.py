"""Split an OCR'd عقد/نظام into its articles.

Segmentation is purely structural: find every article heading, and give each
one the verbatim text running up to the next heading. Nothing here decides
what an article *means* -- that is `signatures.py` -- and nothing here
rewrites a single character of the body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from legal_assistant.docgen.numbering import parse_article_number

# Heading forms seen across the three samples:
#   المادة (٦)      مادة (١)      (مادة13)      المادة  ٢١ :
# Optional ال, optional open paren before or after the word, either digit
# set, and any amount of space (including none) between word and number.
ARTICLE_HEADING = re.compile(
    r"^[ \t ]*\(?[ \t]*(?:ال)?ماد[ةه][ \t ]*\)?[ \t]*"
    r"[\(\[]?[ \t]*([0-9٠-٩]{1,3})[ \t]*[\)\]]?[ \t]*[:\-\.]?[ \t]*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ExtractedArticle:
    """One article, verbatim.

    `heading` is the matched heading line, `body` the text after it up to the
    next heading, and `start`/`end` index the text passed to
    `segment_articles` so a caller can re-slice or diff against the source.
    """

    number: int
    heading: str
    body: str
    start: int
    end: int


def segment_articles(text: str) -> list[ExtractedArticle]:
    """Return every article in `text`, in document order.

    Anything before the first heading (title page, preamble) is discarded.
    Returns [] when no heading is found -- the caller degrades to manual
    entry rather than failing.
    """
    matches = list(ARTICLE_HEADING.finditer(text))
    kept: list[tuple[re.Match[str], int]] = []
    for match in matches:
        number = parse_article_number(match.group(1))
        if number is not None:
            kept.append((match, number))
    articles: list[ExtractedArticle] = []

    for index, (match, number) in enumerate(kept):
        body_start = match.end()
        end = kept[index + 1][0].start() if index + 1 < len(kept) else len(text)
        articles.append(
            ExtractedArticle(
                number=number,
                heading=match.group(0).strip(),
                body=text[body_start:end].strip(),
                start=match.start(),
                end=end,
            )
        )

    return articles
