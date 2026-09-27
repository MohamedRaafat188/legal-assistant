"""Arabic text helpers for docgen.

Deliberately duplicates ~15 lines of `arabic_ingest/arabic_text.py` rather
than importing it: `pyproject.toml` packages only `src/legal_assistant`, so
`arabic_ingest` is not present in the deployed wheel. Importing it would
work in dev and fail on Railway.
"""

from __future__ import annotations

import re
import unicodedata

_ARABIC_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ASCII_TO_ARABIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# Directional controls and the tatweel/kashida elongation, which carry no
# meaning but break naive equality.
_INVISIBLE = re.compile("[​-‏‪-‮⁦-⁩ـ]")
_MULTISPACE = re.compile(r"\s+")


def to_ascii_digits(text: str) -> str:
    """Map ٠-٩ to 0-9. For *parsing* numbers, never for stored/citable text."""
    return text.translate(_ARABIC_TO_ASCII)


def to_arabic_digits(text: str) -> str:
    """Map 0-9 to ٠-٩. For lawyer-facing display strings."""
    return text.translate(_ASCII_TO_ARABIC)


def digit_variants(text: str) -> list[str]:
    """Both digit renderings of `text`, original first, duplicates removed.

    A value read off the سجل تجاري may use ASCII digits while the same value
    in the عقد uses Arabic-Indic ones. Patching tries every variant before
    declaring a value absent.
    """
    out = [text]
    for variant in (to_ascii_digits(text), to_arabic_digits(text)):
        if variant not in out:
            out.append(variant)
    return out


def normalize_for_match(text: str) -> str:
    """Canonical form for *matching* keywords and signatures.

    Both sides of any comparison MUST pass through this exact function.
    Never use the result as citable or displayable text -- it is lossy.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _INVISIBLE.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
        .replace("ئ", "ي")
        .replace("ؤ", "و")
    )
    text = to_ascii_digits(text)
    return _MULTISPACE.sub(" ", text).strip()
