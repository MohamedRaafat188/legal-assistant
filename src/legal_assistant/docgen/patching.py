"""Constrained span substitution inside an article's OCR text.

The ONLY transformation allowed on article text between OCR and the lawyer's
review screen. It replaces one verbatim span with another verbatim value, or
it does nothing and says so. There is no fuzzy rewriting, no sentence
reconstruction, and no LLM call anywhere in this module -- a wrong edit here
is a wrong legal document.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from legal_assistant.docgen.arabic import digit_variants


@dataclass(frozen=True)
class Replacement:
    """A request to swap `old` for `new` inside an article.

    `source` records where `new` came from (`cr`, `aoa`, `user`) so the
    review screen can tell the lawyer why the text changed.
    """

    field: str
    old: str
    new: str
    source: str


@dataclass(frozen=True)
class PatchOp:
    """A substitution that actually happened. Offsets index the ORIGINAL text."""

    start: int
    end: int
    old: str
    new: str
    field: str
    source: str


@dataclass(frozen=True)
class PatchResult:
    text: str
    ops: list[PatchOp] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    needs_review: bool = False


def _locate(text: str, old: str) -> tuple[int, int, str] | str:
    """Find `old` (or a digit-variant of it) exactly once.

    Returns (start, end, matched_form) on a unique match, or a note string
    explaining why no substitution may be made.
    """
    for candidate in digit_variants(old):
        count = text.count(candidate)
        if count == 1:
            start = text.index(candidate)
            return start, start + len(candidate), candidate
        if count > 1:
            return (
                f"القيمة «{old}» وردت فى أكثر من موضع داخل المادة، "
                "فلم يتم التعديل تلقائيا (ambiguous match)."
            )
    return f"لم يتم العثور على القيمة «{old}» حرفيا داخل نص المادة"


def patch_article(text: str, replacements: Sequence[Replacement]) -> PatchResult:
    """Apply every replacement whose `old` appears exactly once, verbatim.

    A replacement is skipped -- and the article flagged `needs_review` -- when
    `old` is absent, appears more than once, or overlaps a span already
    claimed by an earlier replacement. Skipping never modifies the text.
    """
    ops: list[PatchOp] = []
    notes: list[str] = []
    claimed: list[tuple[int, int]] = []

    for rep in replacements:
        if not rep.old or not rep.new or rep.old == rep.new:
            # Nothing to do, and nothing suspicious about it.
            continue

        located = _locate(text, rep.old)
        if isinstance(located, str):
            notes.append(f"[{rep.field}] {located}")
            continue

        start, end, matched = located
        if any(start < c_end and c_start < end for c_start, c_end in claimed):
            notes.append(
                f"[{rep.field}] القيمة «{rep.old}» تتداخل مع تعديل آخر، فلم يتم تطبيقها."
            )
            continue

        claimed.append((start, end))
        ops.append(
            PatchOp(
                start=start,
                end=end,
                old=matched,
                new=rep.new,
                field=rep.field,
                source=rep.source,
            )
        )

    # Apply right-to-left so earlier offsets stay valid against the original.
    patched = text
    for op in sorted(ops, key=lambda o: o.start, reverse=True):
        patched = patched[: op.start] + op.new + patched[op.end :]

    ops.sort(key=lambda o: o.start)
    return PatchResult(text=patched, ops=ops, notes=notes, needs_review=bool(notes))
