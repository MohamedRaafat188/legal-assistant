"""Canonical Arabic law-identity strings.

There are exactly two ingested laws. Payload carries a bare `law_name`
(e.g. "القانون المدني") plus `law_number`/`law_year`, but citations must use
one canonical full-name string consistently everywhere: the context handed
to the LLM, what the LLM is instructed to cite, and the citation guard's
allowed-set. Consistency here is what makes the guard's verification sound.
"""

from __future__ import annotations

_ASCII_TO_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def _ar(n: int) -> str:
    """Render an integer in Arabic-Indic digits."""
    return str(n).translate(_ASCII_TO_ARABIC_DIGITS)


# Fixed mapping, used as a fallback when payload carries no `law_name`, and
# as the single source of truth for the canonical full citation string.
_LAW_BASE_NAMES: dict[tuple[int, int], str] = {
    (174, 2025): "قانون الإجراءات الجنائية",
    (131, 1948): "القانون المدني",
}


def canonical_law_name(law_number: int, law_year: int, payload_law_name: str | None = None) -> str:
    """Return the canonical Arabic citation string for a law.

    e.g. (174, 2025) -> «قانون الإجراءات الجنائية رقم ١٧٤ لسنة ٢٠٢٥»
         (131, 1948) -> «القانون المدني رقم ١٣١ لسنة ١٩٤٨»
    """
    base_name = _LAW_BASE_NAMES.get((law_number, law_year)) or payload_law_name
    if not base_name:
        raise ValueError(f"No canonical or payload law_name for law_number={law_number}, law_year={law_year}")
    return f"{base_name} رقم {_ar(law_number)} لسنة {_ar(law_year)}"


def known_law_key(law_number: int, law_year: int) -> tuple[int, int] | None:
    """Return (law_number, law_year) if this is one of the two ingested laws, else None."""
    key = (law_number, law_year)
    return key if key in _LAW_BASE_NAMES else None
