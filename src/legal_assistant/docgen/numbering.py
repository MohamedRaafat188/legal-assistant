"""Article numbers <-> Arabic ordinal words.

The templates read "قرار بتعديل المادة السادسة" for one article and
"قرار بتعديل المواد السادسة والسابعة والثامنة" for several, so the noun,
the ordinals, and the conjunction are all computed here rather than typed
by a caller. Ordinals are FEMININE throughout, because the noun is مادة.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from legal_assistant.docgen.arabic import to_ascii_digits

_UNITS = {
    1: "الأولى",
    2: "الثانية",
    3: "الثالثة",
    4: "الرابعة",
    5: "الخامسة",
    6: "السادسة",
    7: "السابعة",
    8: "الثامنة",
    9: "التاسعة",
    10: "العاشرة",
}

# 11-19: the feminine unit ordinal followed by عشرة.
_TEEN_UNITS = {
    1: "الحادية",
    2: "الثانية",
    3: "الثالثة",
    4: "الرابعة",
    5: "الخامسة",
    6: "السادسة",
    7: "السابعة",
    8: "الثامنة",
    9: "التاسعة",
}

# 20, 30, ... 90. These keep the -ون form even with مادة --
# "المادة العشرون", never "العشرونة".
_TENS = {
    2: "العشرون",
    3: "الثلاثون",
    4: "الأربعون",
    5: "الخمسون",
    6: "الستون",
    7: "السبعون",
    8: "الثمانون",
    9: "التسعون",
}

_DIGITS = re.compile(r"[0-9]+")


def ordinal_words(n: int) -> str:
    """Feminine Arabic ordinal for 1..99."""
    if not 1 <= n <= 99:
        raise ValueError(f"article ordinal out of supported range 1..99: {n}")
    if n <= 10:
        return _UNITS[n]
    if n <= 19:
        return f"{_TEEN_UNITS[n - 10]} عشرة"
    tens, unit = divmod(n, 10)
    if unit == 0:
        return _TENS[tens]
    return f"{_TEEN_UNITS[unit]} و{_TENS[tens]}"


def article_name(n: int) -> str:
    """Singular reference to one article: "المادة السادسة"."""
    return f"المادة {ordinal_words(n)}"


def articles_title(numbers: Sequence[int]) -> str:
    """Heading fragment naming every amended article.

    One article -> "المادة السادسة".
    Several     -> "المواد السادسة والسابعة والثامنة": و is prefixed to each
                   ordinal after the first, with no serial comma, which is
                   how Egyptian legal drafting writes the list.
    """
    unique = sorted(set(numbers))
    if not unique:
        raise ValueError("articles_title requires at least one article number")
    if len(unique) == 1:
        return article_name(unique[0])
    words = [ordinal_words(n) for n in unique]
    return "المواد " + words[0] + "".join(f" و{w}" for w in words[1:])


def parse_article_number(text: str) -> int | None:
    """First integer in an article heading, in either digit set."""
    match = _DIGITS.search(to_ascii_digits(text))
    return int(match.group(0)) if match else None
