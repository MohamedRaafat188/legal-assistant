"""Orchestration for docgen. The only module the API routes touch.

Flow: create session -> upload -> classify pages cheaply -> OCR the body
pages -> split instruments -> segment articles -> extract identity fields ->
patch -> lawyer reviews and writes "بعد التعديل" -> render.

Nothing here writes legal prose. Article text reaching the document is
verbatim OCR output, a span substitution recorded in `patch_ops`, or text
the lawyer typed.

This module currently holds only the PURE helpers (no DB, no I/O): the
session-lifecycle orchestration (create_session, get_session, add_upload,
run_ocr_job, update_fields, update_article, update_attendees, render_session,
delete_session, purge_expired) is added on top of this in a follow-up.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from legal_assistant.docgen.arabic import to_ascii_digits
from legal_assistant.docgen.models import SourceMode, UploadKind
from legal_assistant.docgen.parsing.commercial_register import CompanyRecord
from legal_assistant.docgen.parsing.signatures import Concept
from legal_assistant.docgen.patching import Replacement


class SessionNotFoundError(LookupError):
    """No such session for this user, or it has expired."""


class SessionStateError(RuntimeError):
    """The session is not in a state where this operation makes sense."""


class NothingSelectedError(SessionStateError):
    """Render was requested with no article selected."""


# --- pure helpers (unit-tested) ------------------------------------------

# Which extracted field belongs in which article. Only these three are ever
# patched: they are single values stated verbatim in one place. Partner
# tables are structural, so a mismatch there is surfaced, never rewritten.
_CONCEPT_FIELDS: dict[Concept, str] = {
    Concept.HEAD_OFFICE: "company_address",
    Concept.CAPITAL: "capital",
    Concept.COMPANY_NAME: "company_name",
}


def plan_patches(
    record: CompanyRecord, concept: Concept | None, article_text: str
) -> list[Replacement]:
    """The substitutions worth attempting on one article.

    Decides only WHICH field belongs to this article's concept. Whether the
    old value is actually present is `patching.patch_article`'s job, and it
    fails closed when it is not.
    """
    if concept is None:
        return []
    field_name = _CONCEPT_FIELDS.get(concept)
    if field_name is None:
        return []
    new_value = getattr(record, field_name, None)
    if not new_value:
        return []
    # `old` is the article's current value, which we do not know as a span --
    # so the replacement is expressed against the whole article text and the
    # patcher locates it. For a single-valued field the current value is the
    # record's own previous rendering; when the article already says the new
    # value, patch_article no-ops.
    return [
        Replacement(
            field=field_name,
            old=_current_value(field_name, article_text),
            new=new_value,
            source="cr",
        )
    ]


def _current_value(field_name: str, article_text: str) -> str:
    """The article's own rendering of a field, as a verbatim span.

    Returns "" when it cannot be isolated, which makes the replacement a
    silent no-op rather than a guess.
    """
    if field_name == "capital":
        match = re.search(r"[0-9٠-٩][0-9٠-٩,.]*", article_text)
        return match.group(0) if match else ""
    if field_name == "company_address":
        match = re.search(r"(?:الكائن|الكائنة|مقرها)\s*(?:فى|في|ب)?\s*(?P<v>[^.\n]+)", article_text)
        return match.group("v").strip() if match else ""
    if field_name == "company_name":
        match = re.search(r"(?:اسم الشركة|تسمى الشركة)\s*(?:هو|:)?\s*(?P<v>[^.\n]+)", article_text)
        return match.group("v").strip() if match else ""
    return ""


def compute_percentages(attendees: Sequence[dict]) -> tuple[str | None, str | None]:
    """(attendance_percentage, approval_percentage) from the attendee list.

    Attendance is the attending share of the total. Approval defaults to the
    attending share too -- the lawyer overrides it when a partner attended
    but voted against.

    Deliberate decisions on the edge cases (never a fabricated or silently
    dropped number):
    - Empty attendee list -> (None, None): no quorum can be stated.
    - No attendee states a share -> (None, None): nothing to sum.
    - A non-numeric share value -> that attendee's share is EXCLUDED from
      both the total and the attending sum, rather than raising or being
      counted as zero. One garbled OCR value must not sink the whole
      computation, and it must not silently masquerade as "no shares held".
    - Missing "attending" key -> treated as attending=True (present in the
      list without a stated flag means "listed as present", the common case
      when the source only enumerates who showed up).
    - Shares that sum to zero (e.g. every present value is literally "0")
      -> (None, None): a zero total is not a valid quorum denominator and
      must not divide-by-zero or be reported as 0%/undefined.
    - Shares not summing to the company's total capital -- this helper takes
      the attendee list at face value and reports the attending fraction OF
      the shares actually declared here; it does not cross-check against
      `CompanyRecord.issued_capital` (that is a different concern, handled
      by the caller/review screen if at all), so no separate handling is
      needed for the shares-vs-capital-mismatch case.
    """
    total = 0.0
    attending = 0.0
    for person in attendees:
        raw = person.get("shares")
        if not raw:
            continue
        try:
            value = float(to_ascii_digits(str(raw)).replace(",", ""))
        except ValueError:
            continue
        total += value
        if person.get("attending", True):
            attending += value

    if total <= 0:
        return None, None
    percentage = f"{attending / total * 100:g}"
    return percentage, percentage


def validate_source_mode(
    source_mode: str, uploaded_kinds: set[str], typed_cr_no: str | None
) -> None:
    """Assert a session has everything its declared mode needs."""
    if UploadKind.aoa.value not in uploaded_kinds:
        raise SessionStateError("لم يتم رفع عقد التأسيس بعد.")
    if source_mode == SourceMode.aoa_plus_cr.value:
        if UploadKind.commercial_register.value not in uploaded_kinds:
            raise SessionStateError("الوضع المختار يتطلب رفع مستخرج السجل التجارى.")
    elif not typed_cr_no:
        raise SessionStateError("أدخل رقم السجل التجارى وتاريخ القيد.")
