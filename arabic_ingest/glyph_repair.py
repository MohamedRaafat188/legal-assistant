# -*- coding: utf-8 -*-
"""Glyph-level repair for a font-mapping defect in specific source PDFs.

The Law 131/1948 (القانون المدني) source PDF ships with an embedded font whose
``ToUnicode``/CID table mis-maps the **initial/isolated-form kaf (ك)** glyph to
U+0622 (ALEF WITH MADDA ABOVE, ``آ``). Every extraction engine (poppler and
PyMuPDF were both cross-checked) faithfully reproduces the wrong codepoint,
because the defect is baked into the PDF, not the extractor. The result is words
like ``كان``/``المحاكم``/``الشركة`` coming out as ``آان``/``المحاآم``/``الشرآة``.

This is a *glyph repair* (restoring the letter the source glyph actually draws),
not a text edit — it is exactly the class of fix the pipeline permits on
faithful text: the shape on the page is a kaf; only its Unicode label is wrong.

Because U+0622 is *also* a legitimate letter (e.g. ``آخر``, ``آثار``, ``الآتية``,
``المنشآت``), a blind ``آ→ك`` replacement is unsafe. Instead we classify every
distinct آ-bearing token observed in the actual extraction (a bounded set — 204
tokens) into three reviewed buckets:

* ``KEEP_MADDA``   — genuine alef-madda words; left untouched.
* ``TO_ALEF``      — a handful where the defect produced آ but the intended
                     letter is plain alef ``ا`` (a rarer second corruption).
* everything else  — the kaf defect; ``آ→ك``.

``REVIEWED_MADDA_TOKENS`` is the full audited universe. Any آ-token seen at
repair time that is *not* in it is surfaced as a warning (still repaired with the
dominant ``آ→ك`` rule, since ~92% of cases are kaf), so a new/unreviewed word in
a future document or re-extraction cannot be silently altered.

The repair is gated to ``AFFECTED_LAWS`` — applying it to a law without the
defect (e.g. Law 174) would itself corrupt that law's legitimate آ words.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Iterable

from pdf_extractor import PageText

# Laws whose source PDF carries the kaf→آ font defect.
AFFECTED_LAWS: frozenset[int] = frozenset({131})

_MADDA = "آ"  # آ
_ALEF = "ا"   # ا
_KAF = "ك"    # ك

# Maximal runs of Arabic letters (matches the tokenisation used to derive the
# reviewed sets below; excludes punctuation and digits so a comma-glued token
# does not escape classification).
_TOKEN_RE = re.compile(r"[ء-غـ-يٱ-ۓ]+")

# --- reviewed classification (derived from the real Law-131 extraction) -------

# Genuine alef-madda words — keep آ exactly as-is.
KEEP_MADDA = frozenset({
    "آثار", "آخر", "الآبار", "الآتي", "الآتية", "الآتيتين", "الآثار",
    "الآخر", "الآخرون", "الآخرين", "الآداب", "الآلات", "المنشآت", "بالآثار",
    "لآخرين", "للآبار", "للآخر", "للآداب", "منشآت", "منشآته", "والآبار",
    "والآثار", "والآلات", "والمنشآت",
})

# Defect produced آ but the intended letter is plain alef ا (user-reviewed).
TO_ALEF = frozenset({"آي", "بآت", "بآي"})

# Full audited universe of آ-tokens seen in the Law-131 extraction. Anything
# outside this set at repair time is unreviewed and gets flagged.
REVIEWED_MADDA_TOKENS = frozenset({
    "آأجرة", "آأن", "آاتب", "آاشف", "آاف", "آافة", "آافيا", "آافية",
    "آالأآشاك", "آالتعويض", "آالكفالة", "آامل", "آاملا", "آاملة", "آان",
    "آانا", "آانت", "آانوا", "آبير", "آبيرا", "آبيرة", "آبيع", "آتاب",
    "آتابة", "آتابها", "آتابي", "آتعيينها", "آثار", "آحادث", "آخر", "آذلك",
    "آذنته", "آسب", "آسبه", "آسبها", "آشف", "آشفه", "آصناعة", "آف", "آفالة",
    "آفايته", "آفل", "آفيل", "آفيلا", "آقانون", "آقبض", "آل", "آلا", "آلات",
    "آلاهما", "آلت", "آلتا", "آلفة", "آلما", "آلن", "آله", "آلها", "آلى",
    "آليها", "آما", "آمالية", "آمل", "آمياه", "آن", "آنص", "آنف", "آهذا",
    "آي", "آيف", "آيفته", "آيفية", "آيل", "آيه", "أآان", "أآانت", "أآبر",
    "أآتوبر", "أآثر", "أآثره", "أآثرها", "أآد", "أآره", "أآمل", "أرآان",
    "أرآانه", "إآراه", "اآتساب", "استهلاآها", "اشتراآات", "الآبار", "الآتي",
    "الآتية", "الآتيتين", "الآثار", "الآخر", "الآخرون", "الآخرين", "الآداب",
    "الآلات", "الأآبر", "الأآثر", "الإآراه", "الترآات", "الترآة", "الترآي",
    "التوآيل", "الذآر", "الراآدة", "الشرآاء", "الشرآات", "الشرآة", "المحاآم",
    "المحرآة", "المذآور", "المذآورة", "المرآبة", "المرآز", "المساآن",
    "المشترآة", "المملوآة", "المنشآت", "الموآل", "الموآلين", "الموآول",
    "الوآالة", "الوآلاء", "الوآيل", "امتلاآه", "بآت", "بآي", "بأآثر",
    "بأآمله", "بإآراه", "بالآثار", "بالإآراه", "بالشرآة", "بالوآالة", "تذآر",
    "ترآة", "ترآت", "ترآيبات", "توآيلا", "ذآر", "ذآره", "ذآرها", "شاآل",
    "شرآاء", "شرآائه", "شرآات", "شرآة", "لآخرين", "لأآثر", "لأملاآهما",
    "لإآراه", "لترآته", "لشرآائه", "للآبار", "للآخر", "للآداب", "للإآراه",
    "للترآة", "للشرآاء", "للشرآة", "للموآل", "للوآالة", "للوآيل", "لمرآز",
    "مأآل", "مرآز", "مرآزها", "مشترآا", "مشترآة", "مملوآا", "مملوآة",
    "مملوآين", "منشآت", "منشآته", "موآله", "موآليهم", "هلاآا", "وآالة",
    "وآالفوائد", "وآان", "وآانت", "وآانوا", "وآذلك", "وآل", "وآما", "وآيلا",
    "والآبار", "والآثار", "والآلات", "والشرآات", "والمشارآة", "والمنشآت",
    "والوآالة", "والوآيل", "وشرآات", "ومملوآا", "ووآلاء", "يترآه", "يذآر",
    "يشترآوا", "يوآله",
})


def repair_text(text: str) -> tuple[str, Counter]:
    """Repair the kaf→آ font defect in ``text``.

    Returns the repaired text and a ``Counter`` of any آ-tokens that were not in
    ``REVIEWED_MADDA_TOKENS`` (repaired anyway with the آ→ك rule, but surfaced so
    a new/unreviewed word cannot pass silently).
    """
    flagged: Counter = Counter()

    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        if _MADDA not in tok:
            return tok
        if tok in KEEP_MADDA:
            return tok
        if tok in TO_ALEF:
            return tok.replace(_MADDA, _ALEF)
        if tok not in REVIEWED_MADDA_TOKENS:
            flagged[tok] += 1
        return tok.replace(_MADDA, _KAF)

    return _TOKEN_RE.sub(_sub, text), flagged


def repair_pages(
    pages: Iterable[PageText], law_number: int
) -> tuple[list[PageText], Counter]:
    """Apply :func:`repair_text` to every page, gated to ``AFFECTED_LAWS``.

    For unaffected laws the pages are returned unchanged. The second return value
    aggregates unreviewed-token warnings across all pages.
    """
    pages = list(pages)
    if law_number not in AFFECTED_LAWS:
        return pages, Counter()
    flagged: Counter = Counter()
    out: list[PageText] = []
    for p in pages:
        fixed, fl = repair_text(p.text)
        flagged.update(fl)
        out.append(replace(p, text=fixed))
    return out, flagged
