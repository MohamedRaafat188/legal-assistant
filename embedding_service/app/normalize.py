"""Arabic normalization for the retrieval/embedding text copy.

Byte-for-byte extracted from the ingestion project's arabic_text.py
(normalize_for_embedding and its exact dependencies), via
scripts/extract_normalize.py, to guarantee no transcription drift in the
combining-Arabic-diacritic ranges. Query text must be folded identically to
how document text was folded at ingestion time -- any drift here silently
degrades retrieval, so do not hand-edit this independently of the ingestion
source; re-run the extraction script instead.
"""

from __future__ import annotations

import re

_COMBINING_MARKS = "\u064B-\u0652\u0670\u0653-\u0655"
_MULTISPACE = re.compile(r"[ \t]+")


def normalize_for_embedding(
    text: str,
    *,
    fold_alef: bool = True,
    fold_yaa: bool = True,
    fold_taa_marbuta: bool = False,
    strip_harakat: bool = True,
) -> str:
    """Produce the retrieval/embedding copy from already-cleaned text.

    Defaults follow standard Arabic IR practice and are tuned for recall on
    lawyer queries, which are rarely diacritized or spelled with hamza care:

    * ``fold_alef``       أ إ آ ٱ -> ا      (hamza on alef is often dropped)
    * ``fold_yaa``        ی ى -> ي           (Farsi-yeh is a font artifact here;
                                              final alef-maqsura varies by typist)
    * ``fold_taa_marbuta`` ة -> ه            (OFF by default: it can conflate
                                              distinct legal terms; enable only
                                              if eval shows it helps)
    * ``strip_harakat``   remove diacritics  (queries are undiacritized)

    Always removes tatweel (ـ), which is purely decorative.
    """
    # Farsi -> Arabic letter forms (these appear only because of the masthead
    # font, never as meaningful spelling).
    text = text.replace("\u06CC", "\u064A")  # ARABIC LETTER FARSI YEH -> YEH
    text = text.replace("\u06A9", "\u0643")  # ARABIC LETTER KEHEH -> KAF
    text = text.replace("\u0640", "")        # remove TATWEEL

    if fold_alef:
        for ch in "أإآٱ":
            text = text.replace(ch, "ا")
    if fold_yaa:
        text = text.replace("\u0649", "\u064A")  # ALEF MAKSURA -> YEH
    if fold_taa_marbuta:
        text = text.replace("\u0629", "\u0647")  # TEH MARBUTA -> HEH
    if strip_harakat:
        text = re.sub(rf"[{_COMBINING_MARKS}]", "", text)

    text = _MULTISPACE.sub(" ", text)
    return text.strip()
