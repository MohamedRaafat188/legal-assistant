"""Orchestration for docgen. The only module the API routes touch.

Flow: create session -> upload -> classify pages cheaply -> OCR the body
pages -> split instruments -> segment articles -> extract identity fields ->
patch -> lawyer reviews and writes "بعد التعديل" -> render.

Nothing here writes legal prose. Article text reaching the document is
verbatim OCR output, a span substitution recorded in `patch_ops`, or text
the lawyer typed.

Every DB-touching function below is reachable only through `get_session`,
which enforces ownership, so an authorization check never has to be repeated
per-call. Every exception this module raises to a caller is built from a
static Arabic message or an exception TYPE NAME ONLY -- never from
interpolating a caught exception's own message -- because uploads, OCR
output, and DB error details (a failing statement's bound parameters) can
all carry a partner's national ID or passport number. See `_fail_stage`,
`_run_stage_sync`/`_run_stage_async`, and `_safe_flush`/`_safe_commit`.
"""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Coroutine, Sequence
from typing import Any, NoReturn, TypeVar

from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from legal_assistant.config import get_settings
from legal_assistant.db.session import get_sessionmaker
from legal_assistant.docgen import pdf, storage
from legal_assistant.docgen.arabic import to_ascii_digits
from legal_assistant.docgen.models import (
    DocgenArticle,
    DocgenFields,
    DocgenSession,
    DocgenUpload,
    SessionStatus,
    SourceMode,
    UploadKind,
    default_expires_at,
)
from legal_assistant.docgen.numbering import article_name, ordinal_words
from legal_assistant.docgen.ocr.base import OcrError, body_pages, get_provider
from legal_assistant.docgen.parsing import sections
from legal_assistant.docgen.parsing.articles import ExtractedArticle
from legal_assistant.docgen.parsing.commercial_register import (
    CR_FIELD_SCHEMA,
    CompanyRecord,
    Party,
    parse_party_table,
    record_from_payload,
    split_capital,
)
from legal_assistant.docgen.parsing.signatures import Concept, classify, find_article
from legal_assistant.docgen.patching import PatchResult, Replacement, patch_article
from legal_assistant.docgen.pdf import InvalidPdfError
from legal_assistant.docgen.render import (
    ArticleBlock,
    Attendee,
    build_context,
    render_document,
)
from legal_assistant.docgen.storage import StorageKeyError
from legal_assistant.docgen.templates.registry import get_template

_log = logging.getLogger(__name__)

_T = TypeVar("_T")


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


def apply_patches(text: str, replacements: Sequence[Replacement]) -> PatchResult:
    """The single, mandatory entry point for turning `plan_patches` output
    into a `PatchResult` -- every production call site goes through this,
    never through `patching.patch_article` directly.

    Why this wrapper exists (the Part 0 fix): `_current_value` cannot always
    isolate an article's current rendering of a field as a verbatim span --
    e.g. "المركز الرئيسى فى الجيزة..." has no "الكائن"/"مقرها" anchor for the
    head-office regex to latch onto. `plan_patches` still has a genuine new
    value for that field, so it returns a `Replacement` with `old=""`.

    `patching.patch_article` treats an empty `old` as a deliberate no-op --
    "nothing to do, and nothing suspicious about it" (see its own docstring
    and `test_empty_old_or_new_is_skipped_silently`) -- and that contract is
    correct and must not change: a caller that explicitly has no old value in
    mind for a field is not doing anything wrong. But `plan_patches` is not
    that caller. It always has a `new` value here (it returns [] otherwise),
    so an empty `old` from _this_ producer specifically means "a value exists
    that should have replaced something, and the something could not be
    found" -- exactly the case `needs_review` exists to catch, and exactly
    the case that was previously vanishing into `patch_article`'s ordinary
    no-op path with no note and `needs_review=False`.

    This function is the one place both facts are visible at once (the
    Replacement's provenance as a `plan_patches` output, and the meaning of
    an empty `old` as "value present, span not located"), so it is where the
    two are reconciled: unlocatable replacements are withheld from
    `patch_article` (so it never has to special-case them) and instead turned
    into an explicit note naming the field, with `needs_review` forced True.
    A future caller cannot bypass this by calling `patch_article` directly
    with a `plan_patches` result, because this function -- not
    `patch_article` -- is what `_build_article_row` (the only place patches
    are actually applied to an article) calls.
    """
    unlocatable = [r for r in replacements if r.new and not r.old]
    locatable = [r for r in replacements if r not in unlocatable]
    result = patch_article(text, locatable)
    if not unlocatable:
        return result

    notes = list(result.notes) + [
        f"[{r.field}] لم يتمكن النظام من تحديد القيمة الحالية لحقل «{r.field}» "
        f"تلقائيا داخل نص المادة؛ راجع المادة يدويا وحدّثها إلى «{r.new}»."
        for r in unlocatable
    ]
    return PatchResult(text=result.text, ops=result.ops, notes=notes, needs_review=True)


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
      A share value is still numeric after stripping a trailing "%"/"٪" (a
      percentage sign copied alongside the number) and the Arabic thousands
      separator "٬" (U+066C) -- those are punctuation around the figure, not
      part of it, and must not make an otherwise-good value look unusable.
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
        cleaned = (
            to_ascii_digits(str(raw))
            .replace(",", "")
            .replace("%", "")
            .replace("٪", "")
            .replace("٬", "")
            .strip()
        )
        try:
            value = float(cleaned)
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


def _reconcile_capital(attendees: Sequence[dict], issued_capital: str | None) -> str | None:
    """Compare the attendee roster's declared shares against the company's
    independently-extracted issued capital. `None` means "reconciles" (or
    the figures are usable and agree); a returned string is an Arabic note
    describing why the roster cannot be trusted as complete.

    `compute_percentages` only ever sees the attendee list itself, so a
    roster missing a partner -- because CR/OCR extraction failed to recover
    them, not because they were genuinely absent -- still looks perfectly
    self-consistent to it: `('100', '100')` for a lone 50-share attendee is
    correct only if that IS the whole company. `issued_capital` comes from a
    *different* extraction (the السجل التجارى Box 9, or the عقد's رأس المال
    المصدر figure via `split_capital`), so comparing the two catches exactly
    this class of silent under-count. This is a best-effort numeric check,
    not proof of completeness -- it cannot catch two OCR errors that happen
    to cancel out -- so callers must still treat "reconciles" as "no
    detected problem", not "verified correct".
    """
    if not issued_capital:
        return "لا يمكن التحقق من اكتمال كشف الحضور: رأس المال المصدر غير معروف."
    try:
        capital_value = float(to_ascii_digits(issued_capital).replace(",", "").replace("٬", ""))
    except ValueError:
        return "لا يمكن التحقق من اكتمال كشف الحضور: قيمة رأس المال المصدر غير صالحة."
    if capital_value <= 0:
        return "لا يمكن التحقق من اكتمال كشف الحضور: قيمة رأس المال المصدر غير صالحة."

    total = 0.0
    any_share = False
    for person in attendees:
        raw = person.get("shares")
        if not raw:
            continue
        cleaned = (
            to_ascii_digits(str(raw))
            .replace(",", "")
            .replace("%", "")
            .replace("٪", "")
            .replace("٬", "")
            .strip()
        )
        try:
            total += float(cleaned)
            any_share = True
        except ValueError:
            continue

    if not any_share:
        return "لا يمكن التحقق من اكتمال كشف الحضور: بيانات الحصص غير معروفة."

    # 1% slack for rounding noise between two independently-OCR'd figures;
    # anything wider than that is treated as a real discrepancy.
    tolerance = max(1.0, capital_value * 0.01)
    if abs(total - capital_value) > tolerance:
        return (
            f"مجموع حصص الشركاء المذكورين فى الكشف ({total:g}) لا يتفق مع رأس المال "
            f"المصدر للشركة ({capital_value:g}). قد يكون كشف الشركاء غير مكتمل أو "
            "غير دقيق؛ راجعه قبل اعتماد نسبة الحضور."
        )
    return None


# --- error handling plumbing ----------------------------------------------
#
# Every exception below can carry OCR'd document content -- InvalidPdfError
# and StorageKeyError messages are built from this codebase's own PDF/storage
# layers (never a raw filesystem path or document text by their own design,
# but treated here as untrusted regardless), OcrError can wrap a transcribed
# page or a سجل تجارى field, FileNotFoundError comes from a missing storage
# key, and SQLAlchemyError's message can include a failing statement's bound
# parameters (this app does not configure `hide_parameters`). None of their
# `str()` output is ever interpolated into a message this module raises,
# logs, or persists -- only the exception's TYPE NAME is used, and the
# replacement exception is always constructed and raised strictly *after*
# its try/except has exited, never from inside the `except` clause: raising
# from inside unconditionally re-attaches the handled exception to the new
# one's `__context__` (surviving even `from None`, which only affects
# display), which would leave the original exception -- and its message --
# reachable to any serializer that reads `__context__` directly. See
# `ocr/gemini.py`'s `_ask`/`_parse_json` for the same pattern, established
# there after two review rounds on exactly this leak class.

_RISKY_ERRORS = (InvalidPdfError, StorageKeyError, FileNotFoundError, OcrError)


class _StageError(RuntimeError):
    """One stage of a docgen pipeline (upload, OCR, render, storage) failed.
    Names the stage and the failing exception's type only -- see the module
    note above. Not specific to OCR: `add_upload`, `render_session`, and
    `delete_session` raise it too, via `_run_stage_sync`."""


def _fail_stage(stage: str, error_type: str) -> NoReturn:
    raise _StageError(f"فشلت مرحلة «{stage}» أثناء معالجة الملف تلقائيا ({error_type}).")


def _run_stage_sync(stage: str, fn, /, *args: Any, **kwargs: Any) -> Any:
    error_type: str | None = None
    try:
        return fn(*args, **kwargs)
    except _RISKY_ERRORS as e:
        error_type = type(e).__name__
    _fail_stage(stage, error_type)  # only reached on failure; always raises


async def _run_stage_async(stage: str, coro: Coroutine[Any, Any, _T]) -> _T:
    error_type: str | None = None
    try:
        return await coro
    except _RISKY_ERRORS as e:
        error_type = type(e).__name__
    _fail_stage(stage, error_type)  # only reached on failure; always raises


async def _safe_flush(db: AsyncSession) -> None:
    """`db.flush()`, converting a DB error into a content-free one."""
    error_type: str | None = None
    try:
        await db.flush()
    except SQLAlchemyError as e:
        error_type = type(e).__name__
    if error_type is not None:
        raise SessionStateError(f"تعذر حفظ التغييرات فى قاعدة البيانات ({error_type}).") from None


async def _safe_commit(db: AsyncSession) -> None:
    """`db.commit()`, converting a DB error into a content-free one."""
    error_type: str | None = None
    try:
        await db.commit()
    except SQLAlchemyError as e:
        error_type = type(e).__name__
    if error_type is not None:
        raise SessionStateError(f"تعذر حفظ التغييرات فى قاعدة البيانات ({error_type}).") from None


# --- session lifecycle ------------------------------------------------------


async def create_session(
    db: AsyncSession, user_id: int, company_type: str, source_mode: str
) -> DocgenSession:
    """Create a draft session.

    Validates the company type and the source mode BEFORE constructing
    anything that depends on them: `validate_source_mode` only checks that a
    session's *uploads* satisfy its declared mode, it never checks that the
    mode string itself is one of the two the app knows about -- that check
    is this function's job, and it runs first.
    """
    get_template(company_type)  # raises KeyError on an unknown company type
    if source_mode not in {m.value for m in SourceMode}:
        raise SessionStateError(f"unknown source mode: {source_mode}")

    settings = get_settings()
    session = DocgenSession(
        user_id=user_id,
        company_type=company_type,
        source_mode=source_mode,
        status=SessionStatus.draft.value,
        # The 2-day retention window is a deliberate, user-mandated ceiling
        # (uploaded documents carry partners' national ID and passport
        # numbers) that a 30-day proposal was explicitly rejected in favour
        # of. `default_expires_at` takes the window as a parameter and the
        # column has no server-side default, so THIS call site is the only
        # place that window is enforced. `docgen_retention_days` defaults to
        # 2 in `Settings` and is documented in `.env.example`; nothing else
        # in this module may compute or override `expires_at`.
        expires_at=default_expires_at(settings.docgen_retention_days),
    )
    db.add(session)
    await _safe_flush(db)
    db.add(DocgenFields(session_id=session.id, data={}))
    await _safe_flush(db)
    return session


def _is_expired(session: DocgenSession, *, now: datetime.datetime | None = None) -> bool:
    """True if `session` is past its retention window, by TIME -- not by
    whatever `status` happens to say.

    `purge_expired` is the only thing that ever sets `status="expired"`, and
    it runs on a schedule: a session's declared 2-day window has nothing to
    do with when that job next ticks. A `status`-only check can therefore
    never substitute for a timestamp check -- a `status="ready"` session
    whose `expires_at` is 5 days in the past is exactly as expired as one
    already marked `status="expired"`, just not yet visited by the purge
    job. Both are treated identically here.

    `expires_at` is stored `DateTime(timezone=True)` and always constructed
    via `default_expires_at` (tz-aware, anchored to `datetime.UTC`), so
    comparing it against an aware `datetime.now(datetime.UTC)` compares two
    aware instants directly -- correct regardless of which tzinfo offset the
    driver hands back, and with no naive/aware mismatch to fail open or
    raise on.
    """
    if session.status == SessionStatus.expired.value:
        return True
    now = now or datetime.datetime.now(datetime.UTC)
    return session.expires_at <= now


def _raise_if_expired(session: DocgenSession) -> None:
    if _is_expired(session):
        raise SessionNotFoundError(f"docgen session {session.id} not found")


async def _load_session_unowned(db: AsyncSession, session_id: int) -> DocgenSession | None:
    """Load a session by id with no `user_id` check, for internal (job)
    callers that are not acting on behalf of a specific caller.

    `run_ocr_job` and `_run_ocr` both need exactly this load -- there is no
    `user_id` here, this is a background job, so `get_session` does not
    apply. This is the one place both go through, so the load itself (and
    any future eager-loading it needs) cannot drift between the two call
    sites. It deliberately does NOT decide what to do about expiry: the two
    callers need different answers (see their own docstrings/comments) for
    what happens once a session has expired mid-job, so that decision stays
    at the call site, made against the shared `_is_expired` predicate
    (`get_session`, `purge_expired`, and this module's docstring already
    treat `_is_expired` as the one place the expiry comparison itself
    lives -- this function does not re-implement it).
    """
    return await db.get(DocgenSession, session_id)


async def get_session(db: AsyncSession, session_id: int, user_id: int) -> DocgenSession:
    """Load a session the caller owns.

    Raises the IDENTICAL `SessionNotFoundError` whether the session does not
    exist, has expired, or belongs to another user -- ownership is enforced
    directly in the query's WHERE clause (not checked afterwards), so a
    caller can never distinguish "no such id" from "not yours" by probing.
    This is the sole authorization gate: every other function in this module
    takes an already-loaded `DocgenSession`, so it is reachable only through
    a session this function returned.

    Both ownership and expiry are filtered in the query itself
    (`user_id == user_id`, `expires_at > func.now()` -- the latter
    consistent with how `purge_expired` compares, and evaluated by the same
    clock as every other row in the database rather than this process's
    wall clock) AND re-checked in Python immediately after load. The second
    layer is deliberate belt-and-suspenders, not redundant ceremony: it is
    what makes this function's authorization/expiry behavior verifiable with
    a stubbed `AsyncSession` that does not actually evaluate SQL predicates
    (see `tests/docgen/test_service.py`), and it is also what protects any
    future caller that constructs the WHERE clause differently or forgets a
    predicate.
    """
    result = await db.execute(
        select(DocgenSession)
        .where(
            DocgenSession.id == session_id,
            DocgenSession.user_id == user_id,
            DocgenSession.expires_at > func.now(),
        )
        .options(
            selectinload(DocgenSession.articles),
            selectinload(DocgenSession.uploads),
            selectinload(DocgenSession.fields),
        )
    )
    session = result.scalar_one_or_none()
    if session is None or session.user_id != user_id:
        raise SessionNotFoundError(f"docgen session {session_id} not found")
    _raise_if_expired(session)
    return session


async def add_upload(
    db: AsyncSession, session: DocgenSession, kind: str, filename: str, data: bytes
) -> DocgenUpload:
    """Store an uploaded PDF and record it. Does not start OCR."""
    if kind not in {k.value for k in UploadKind}:
        raise SessionStateError(f"unknown upload kind: {kind}")
    if session.status == SessionStatus.ocr_running.value:
        raise SessionStateError("جارٍ تحليل الملف الحالى، انتظر حتى ينتهى.")

    count = _run_stage_sync("قراءة عدد صفحات الملف", pdf.page_count, data)
    key = storage.new_key(session.id, kind)
    _run_stage_sync("حفظ الملف المرفوع", storage.write, key, data)

    upload = DocgenUpload(
        session_id=session.id, kind=kind, filename=filename, storage_key=key, page_count=count
    )
    db.add(upload)
    # Any edit invalidates a previously rendered document; a fresh upload is
    # no exception, and also lets the lawyer re-run OCR after replacing a
    # bad scan without a stale .docx staying downloadable.
    session.document_key = None
    if session.status == SessionStatus.rendered.value:
        session.status = SessionStatus.ready.value
    await _safe_flush(db)
    return upload


def _claim_for_ocr_statement(session_id: int):
    """The atomic claim `run_ocr_job` uses to start OCR for `session_id`.

    Pulled out as its own (pure, DB-independent) function so its WHERE
    clause -- the whole re-entrancy/expiry guarantee -- can be asserted on
    directly (compiled to a literal SQL string) without a database
    connection; see `test_claim_for_ocr_statement_*` in
    `tests/docgen/test_service.py`. The statement itself is only ever
    executed from `run_ocr_job`.
    """
    return (
        update(DocgenSession)
        .where(
            DocgenSession.id == session_id,
            DocgenSession.status != SessionStatus.ocr_running.value,
            DocgenSession.expires_at > func.now(),
        )
        .values(status=SessionStatus.ocr_running.value, error=None)
    )


async def run_ocr_job(session_id: int) -> None:
    """Background entry point: classify, OCR, segment, extract, patch.

    Opens its OWN database session -- the request that queued this job has
    already returned and its session is closed. Every failure path lands the
    session in `failed` with a content-free Arabic message (see the module
    note on error handling); uploads are kept so the lawyer can retry
    without re-uploading.

    Starts with a single atomic `UPDATE ... WHERE status != 'ocr_running'
    AND expires_at > now()` rather than a SELECT followed by a separate
    UPDATE. This is not just tidier -- it is the only one of the two that
    actually closes the race: a read-then-write check has a window between
    the read and the write in which two concurrent invocations for the same
    `session_id` can both observe "not running" and both proceed to burn
    billed vision calls and interleave writes to the same rows. A single
    UPDATE has no such window -- Postgres serializes concurrent UPDATEs
    against the same row, so the first to commit is the only one whose WHERE
    clause is evaluated against the pre-update state; the second's WHERE
    clause is evaluated against the row *after* the first UPDATE, sees
    `status = 'ocr_running'`, and matches zero rows. `result.rowcount == 0`
    is therefore a genuine "did not win the race" signal, not merely "was
    unlikely to have raced". The same statement enforces the Fix-1 expiry
    check with the identical guarantee: this is also the only place
    `run_ocr_job` loads a session, and it must not start OCR (more OCR'd
    document content, more billed calls) against a session already past its
    retention window, whether or not `purge_expired` has visited it yet.
    """
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        result = await db.execute(_claim_for_ocr_statement(session_id))
        if result.rowcount == 0:
            _log.warning(
                "docgen OCR job for session %s did not start "
                "(missing, expired, or already running)",
                session_id,
            )
            return
        await _safe_commit(db)

    try:
        await _run_ocr(session_id)
    except Exception as e:  # noqa: BLE001 -- a background job must never escape
        # Deliberately NOT `_log.exception(e)`/`exc_info=True`: the standard
        # traceback formatter renders the exception's own `str()` (and walks
        # its `__context__`/`__cause__` chain), which for an un-wrapped bug
        # could still surface document content. Only the TYPE is logged.
        # `OcrError`/`_StageError`/`SessionStateError` messages raised
        # anywhere in this module are content-free by construction (static
        # Arabic text or a type name only -- never an interpolated message),
        # so they alone are safe to persist verbatim; anything else is a
        # genuinely unexpected failure and gets a type-only message instead.
        if isinstance(e, OcrError | SessionStateError | _StageError):
            message = str(e)
        else:
            message = f"حدث خطأ غير متوقع أثناء معالجة الملف ({type(e).__name__})."
        _log.error(
            "docgen OCR job failed for session %s: %s", session_id, type(e).__name__
        )
        async with sessionmaker() as db:
            try:
                session = await _load_session_unowned(db, session_id)
                expired = session is not None and _is_expired(session)
            except Exception as inner:  # noqa: BLE001 -- must not mask the
                # original failure being handled in this except block; log
                # the TYPE only (same content-free convention as elsewhere
                # in this handler) and treat it as "nothing to write".
                _log.error(
                    "docgen OCR job failure handler could not check session "
                    "%s expiry (%s); leaving its status untouched",
                    session_id,
                    type(inner).__name__,
                )
                session, expired = None, False
            # This write is only a content-free status/error, not new PII,
            # but the session's window may have closed during the run this
            # failure came from -- skip the write entirely rather than
            # touch an expired session at all. Whatever `status` it is left
            # at (almost certainly `ocr_running`, from the Fix-3 claim) is
            # inert: `get_session`'s expiry check is time-based, not
            # status-based, and `purge_expired` will still collect it on
            # its own schedule regardless of `status`.
            if session is not None and not expired:
                session.status = SessionStatus.failed.value
                session.error = message
                await db.commit()


async def _run_ocr(session_id: int) -> None:
    settings = get_settings()
    provider = get_provider(settings)
    sessionmaker = get_sessionmaker()

    async with sessionmaker() as db:
        result = await db.execute(
            select(DocgenSession)
            .where(DocgenSession.id == session_id)
            .options(selectinload(DocgenSession.uploads), selectinload(DocgenSession.fields))
        )
        session = result.scalar_one()
        uploads = {u.kind: u for u in session.uploads}
        company_type = session.company_type
        source_mode = session.source_mode

    aoa = uploads.get(UploadKind.aoa.value)
    if aoa is None:
        raise SessionStateError("لم يتم رفع عقد التأسيس.")

    # 1. Cheap classification pass over every page.
    aoa_bytes = _run_stage_sync("قراءة عقد التأسيس المرفوع", storage.read, aoa.storage_key)
    thumbnails = _run_stage_sync(
        "تجهيز صفحات المعاينة", pdf.render_pages, aoa_bytes, dpi=settings.docgen_classify_dpi
    )
    classifications = await _run_stage_async(
        "تصنيف صفحات الملف", provider.classify_pages(thumbnails)
    )
    wanted = body_pages(classifications)
    if not wanted:
        raise OcrError("لم يتم التعرف على عقد تأسيس داخل هذا الملف.")

    # 2. Full-fidelity OCR on body pages only.
    pages = _run_stage_sync(
        "تجهيز صفحات العقد للتعرف الضوئى",
        pdf.render_pages,
        aoa_bytes,
        dpi=settings.docgen_ocr_dpi,
        pages=wanted,
    )
    page_texts = await _run_stage_async("التعرف الضوئى على نص العقد", provider.extract(pages))
    full_text = "\n".join(p.text for p in page_texts)
    page_confidence = (
        sum(p.confidence for p in page_texts) / len(page_texts) if page_texts else 0.0
    )

    # 3. Instrument split, then article segmentation on the right series.
    instruments = sections.split_instruments(full_text)
    articles, warning = sections.select_target(instruments, company_type)
    if not articles:
        # A session with zero articles can never be rendered (render_session
        # already fails closed via NothingSelectedError once there is at
        # least a `selected` flag to check, but there is nothing here for
        # the lawyer to select at all) -- land the job in `failed`, not
        # `ready`, so this is never mistaken for a completed-but-empty
        # session. `warning` is a static, content-free message authored by
        # `sections.select_target` itself.
        raise OcrError(warning or "لم يتم العثور على مواد قابلة للتعديل داخل الملف.")

    # 4. Identity fields.
    record, provenance = await _run_stage_async(
        "استخراج بيانات هوية الشركة",
        _extract_fields(provider, source_mode, uploads, articles, settings),
    )

    # 5. Patch and persist.
    async with sessionmaker() as db:
        session = await _load_session_unowned(db, session_id)
        if session is None:
            _log.warning(
                "docgen OCR result for session %s discarded: session vanished "
                "during processing",
                session_id,
            )
            return
        if _is_expired(session):
            # The session's window closed somewhere during the classify/OCR/
            # extract calls above -- real external round-trips, not a
            # negligible gap, unlike the claim-to-first-SELECT hop at the top
            # of this function. Abandon the result outright: do not persist
            # the newly OCR'd DocgenArticle rows or DocgenFields.data (both
            # carry identity-document content), and do not flip status to
            # `ready`. Land it in `failed` instead of leaving it stuck in
            # `ocr_running` forever -- a content-free status/error write, not
            # new PII, and `get_session` already blocks this session by
            # `expires_at` regardless of `status`, so this cannot restore
            # read access.
            session.status = SessionStatus.failed.value
            session.error = "انتهت المهلة المتاحة لهذه الجلسة أثناء المعالجة."
            await _safe_commit(db)
            _log.warning(
                "docgen OCR result for session %s discarded: session expired "
                "during processing",
                session_id,
            )
            return
        aoa_row = await db.get(DocgenUpload, aoa.id)
        aoa_row.page_classification = [
            {
                "page": c.page,
                "kind": c.kind.value,
                "starts_article": c.starts_article,
                "confidence": c.confidence,
            }
            for c in classifications
        ]

        await db.execute(
            DocgenArticle.__table__.delete().where(DocgenArticle.session_id == session_id)
        )
        for article in articles:
            db.add(_build_article_row(session_id, article, record, source_mode, page_confidence))

        fields_row = await db.scalar(
            select(DocgenFields).where(DocgenFields.session_id == session_id)
        )
        fields_row.data = provenance

        session.status = SessionStatus.ready.value
        session.error = warning
        await _safe_commit(db)


def _build_article_row(
    session_id: int,
    article: ExtractedArticle,
    record: CompanyRecord,
    source_mode: str,
    confidence: float,
) -> DocgenArticle:
    concept = classify(article)
    if source_mode == SourceMode.aoa_plus_cr.value:
        result = apply_patches(article.body, plan_patches(record, concept, article.body))
    else:
        # aoa_only: the عقد has never been amended, so its text is already
        # current. Patching is a declared no-op, not a silent skip.
        result = apply_patches(article.body, [])

    return DocgenArticle(
        session_id=session_id,
        article_number=article.number,
        ordinal_words=ordinal_words(article.number)
        if 1 <= article.number <= 99
        else str(article.number),
        source_text=article.body,
        patched_text=result.text,
        patch_ops=[
            {
                "start": op.start,
                "end": op.end,
                "old": op.old,
                "new": op.new,
                "field": op.field,
                "source": op.source,
            }
            for op in result.ops
        ],
        confidence=confidence,
        needs_review=result.needs_review,
        selected=False,
    )


async def _extract_fields(
    provider, source_mode: str, uploads: dict, articles: Sequence[ExtractedArticle], settings
) -> tuple[CompanyRecord, dict]:
    """Build the CompanyRecord and its provenance map.

    In `aoa_plus_cr` the سجل wins as the PROPOSED value, but a disagreement
    with the عقد is recorded in `conflict` and surfaced -- never silently
    resolved. The attendee roster built here also gets a capital
    reconciliation note (see `_reconcile_capital`) recorded alongside it, so
    the review screen can show it before the lawyer ever attempts a render;
    `render_session` recomputes the same check against the roster as it
    stands at render time, since edits since extraction can change the
    answer.
    """
    from_aoa = _record_from_articles(articles)

    from_cr = CompanyRecord()
    if source_mode == SourceMode.aoa_plus_cr.value:
        cr = uploads.get(UploadKind.commercial_register.value)
        if cr is not None:
            cr_bytes = _run_stage_sync(
                "قراءة مستخرج السجل التجارى المرفوع", storage.read, cr.storage_key
            )
            images = _run_stage_sync(
                "تجهيز صفحات مستخرج السجل التجارى",
                pdf.render_pages,
                cr_bytes,
                dpi=settings.docgen_ocr_dpi,
            )
            payload = await provider.extract_fields(images, CR_FIELD_SCHEMA)
            from_cr = record_from_payload(payload)

    provenance: dict = {}
    for name in (
        "commercial_registration_no",
        "commercial_registration_date",
        "company_name",
        "law_number",
        "law_year",
        "company_address",
        "capital",
        "issued_capital",
    ):
        cr_value = getattr(from_cr, name)
        aoa_value = getattr(from_aoa, name)
        value = cr_value or aoa_value
        conflict = None
        if cr_value and aoa_value and cr_value != aoa_value:
            conflict = {"cr": cr_value, "aoa": aoa_value}
        provenance[name] = {
            "value": value,
            "source": "cr" if cr_value else ("aoa" if aoa_value else "user"),
            "confidence": 1.0 if value else 0.0,
            "conflict": conflict,
        }

    # Box (9) is authoritative when populated; the عقد's share table is the
    # fallback. Where neither states holdings, names still prefill and the
    # review screen asks the lawyer for the numbers.
    parties = from_cr.parties or from_aoa.parties
    attendees = [
        {
            "name": p.name,
            "shares": p.shares,
            "percentage": p.percentage,
            "attending": True,
            "source": "cr" if from_cr.parties else "aoa",
        }
        for p in parties
    ]
    provenance["attendees"] = attendees
    # Surfaced, never silently trusted: a roster that looks internally
    # consistent (its shares sum to something) is not the same as a roster
    # that is COMPLETE. See `_reconcile_capital`.
    provenance["capital_reconciliation"] = _reconcile_capital(
        attendees, provenance["issued_capital"]["value"]
    )

    merged = CompanyRecord(
        **{
            name: provenance[name]["value"]
            for name in provenance
            if name not in ("attendees", "capital_reconciliation")
        },
        parties=list(parties),
    )
    return merged, provenance


def _record_from_articles(articles: Sequence[ExtractedArticle]) -> CompanyRecord:
    """Everything the عقد itself states, used as fallback and as the conflict side."""
    name_article = find_article(articles, Concept.COMPANY_NAME)
    office = find_article(articles, Concept.HEAD_OFFICE)
    capital = find_article(articles, Concept.CAPITAL)
    parties: list[Party] = parse_party_table(capital.body) if capital else []
    # مساهمة states رأس المال المرخص به and رأس المال المصدر separately; the
    # quorum is computed from المصدر, so the two are kept apart from here on.
    _authorized, issued = split_capital(capital.body) if capital else (None, None)
    return CompanyRecord(
        company_name=_current_value("company_name", name_article.body) or None
        if name_article
        else None,
        company_address=_current_value("company_address", office.body) or None
        if office
        else None,
        capital=_current_value("capital", capital.body) or None if capital else None,
        issued_capital=issued,
        parties=parties,
    )


# --- review edits -----------------------------------------------------------


def _invalidate_document(session: DocgenSession) -> None:
    """Any edit makes a previously rendered document stale."""
    session.document_key = None
    if session.status == SessionStatus.rendered.value:
        session.status = SessionStatus.ready.value


async def update_fields(db: AsyncSession, session: DocgenSession, changes: dict) -> None:
    """Apply lawyer corrections to identity fields, marking them user-sourced."""
    row = await db.scalar(select(DocgenFields).where(DocgenFields.session_id == session.id))
    data = dict(row.data or {})
    for name, value in changes.items():
        entry = dict(data.get(name) or {})
        entry.update({"value": value, "source": "user", "confidence": 1.0, "conflict": None})
        data[name] = entry
    row.data = data
    _invalidate_document(session)
    await _safe_flush(db)


async def update_article(
    db: AsyncSession,
    session: DocgenSession,
    article_number: int,
    *,
    selected: bool | None = None,
    patched_text: str | None = None,
    new_text: str | None = None,
) -> DocgenArticle:
    """Select an article, correct its "قبل التعديل", or write its "بعد التعديل".

    `source_text` is never touched, so the verbatim OCR output stays
    auditable next to whatever the lawyer changed.
    """
    article = await db.scalar(
        select(DocgenArticle).where(
            DocgenArticle.session_id == session.id,
            DocgenArticle.article_number == article_number,
        )
    )
    if article is None:
        raise SessionNotFoundError(f"article {article_number} is not in this session")
    if selected is not None:
        article.selected = selected
    if patched_text is not None:
        article.patched_text = patched_text
        article.needs_review = False
    if new_text is not None:
        article.new_text = new_text
    _invalidate_document(session)
    await _safe_flush(db)
    return article


async def update_attendees(
    db: AsyncSession, session: DocgenSession, attendees: list[dict]
) -> None:
    """Replace the attendee roster and refresh its capital reconciliation.

    The reconciliation note is recomputed here (not just at OCR-extraction
    time) because this is exactly where the roster the lawyer is looking at
    can change -- adding a missed partner, correcting a share figure -- and a
    stale "mismatch" or, worse, a stale "reconciles" left over from before
    the edit would defeat the whole point of the check.
    """
    row = await db.scalar(select(DocgenFields).where(DocgenFields.session_id == session.id))
    data = dict(row.data or {})
    data["attendees"] = attendees
    issued_capital = ((data.get("issued_capital") or {}).get("value")) or None
    data["capital_reconciliation"] = _reconcile_capital(attendees, issued_capital)
    row.data = data
    _invalidate_document(session)
    await _safe_flush(db)


# --- render ------------------------------------------------------------------


async def render_session(db: AsyncSession, session: DocgenSession) -> bytes:
    """Render the document. Idempotent and re-runnable.

    Fails closed rather than emit a partially-complete instrument:
    - no article selected -> `NothingSelectedError`;
    - a selected article with no "بعد التعديل" text yet -> `SessionStateError`
      naming it;
    - for a company type with an attendance table, a roster that does not
      reconcile against the issued capital -> `SessionStateError` (a wrong
      quorum percentage in a محضر جمعية عامة is a legal defect in the filed
      instrument, not a cosmetic one -- see `_reconcile_capital`);
    - any placeholder `render_document` cannot fill -> `MissingContextError`
      (raised by `render_document` itself; not caught or papered over here).
    """
    articles = [a for a in session.articles if a.selected]
    if not articles:
        raise NothingSelectedError("اختر مادة واحدة على الأقل للتعديل.")
    missing = [a.article_number for a in articles if not (a.new_text or "").strip()]
    if missing:
        raise SessionStateError(f"اكتب نص «بعد التعديل» للمواد: {missing}")

    row = await db.scalar(select(DocgenFields).where(DocgenFields.session_id == session.id))
    data = row.data or {}
    scalars = {
        name: (entry or {}).get("value") or ""
        for name, entry in data.items()
        if name not in ("attendees", "capital_reconciliation")
    }

    spec = get_template(session.company_type)
    attendees_data = data.get("attendees", [])
    attendee_rows = [p for p in attendees_data if p.get("attending", True)]
    if spec.attendee_label:
        issued_capital = ((data.get("issued_capital") or {}).get("value")) or None
        mismatch = _reconcile_capital(attendees_data, issued_capital)
        if mismatch is not None:
            raise SessionStateError(mismatch)
        attendance, approval = compute_percentages(attendees_data)
        scalars.setdefault("attendance_percentage", attendance or "")
        scalars.setdefault("approval_percentage", approval or "")

    context = build_context(
        session.company_type,
        scalars=scalars,
        articles=[
            ArticleBlock(
                article_name=article_name(a.article_number),
                article_original_content=a.patched_text,
                article_new_content=a.new_text or "",
            )
            for a in articles
        ],
        article_numbers=[a.article_number for a in articles],
        attendees=[
            Attendee(
                name=p.get("name", ""),
                shares=str(p.get("shares") or ""),
                percentage=str(p.get("percentage") or ""),
            )
            for p in attendee_rows
        ],
    )

    document = render_document(session.company_type, context)
    previous_key = session.document_key
    key = storage.new_key(session.id, "document")
    # Write the NEW object and commit the pointer to it before touching the
    # old one: if deleting the old object then failed partway (e.g. an OSError
    # mid-unlink), the session must still point at a document that exists --
    # never at one it just deleted.
    _run_stage_sync("حفظ المستند الناتج", storage.write, key, document)
    session.document_key = key
    session.status = SessionStatus.rendered.value
    await _safe_flush(db)

    if previous_key and previous_key != key:
        # Best-effort: a session should not accumulate stale renders of a
        # document containing partners' ID numbers, but a failure to remove
        # the superseded one is not fatal -- the new, already-committed
        # document is what matters, and the old object is still purged with
        # the rest of the session's files at expiry/deletion regardless.
        try:
            storage.delete(previous_key)
        except StorageKeyError as e:
            _log.error(
                "docgen render: could not delete the previous rendered "
                "document for session %s: %s",
                session.id,
                type(e).__name__,
            )
    return document


# --- deletion and retention ---------------------------------------------------


async def delete_session(db: AsyncSession, session: DocgenSession) -> None:
    """Delete the session row and every file stored for it, now."""
    _run_stage_sync("حذف ملفات الجلسة", storage.delete_session, session.id)
    await db.delete(session)
    await _safe_flush(db)


async def purge_expired(db: AsyncSession) -> int:
    """Purge every expired session's files and text. Returns how many.

    The session ROW survives in `expired` form for audit; the uploads, the
    OCR text, and the article text -- which carry national ID and passport
    numbers -- do not. Storage deletion is attempted best-effort per session:
    a `StorageKeyError` on one session's files is logged (type only, see the
    module note on error handling) and that session is skipped for this
    pass -- rather than aborting the whole purge, or than marking it purged
    when its files might still be sitting on disk -- and it remains eligible
    to be retried on the next call.
    """
    result = await db.execute(
        select(DocgenSession).where(
            DocgenSession.expires_at < func.now(),
            DocgenSession.status != SessionStatus.expired.value,
        )
    )
    sessions = list(result.scalars())
    purged = 0
    for session in sessions:
        try:
            storage.delete_session(session.id)
        except StorageKeyError as e:
            _log.error(
                "docgen purge: could not delete stored files for session %s: %s",
                session.id,
                type(e).__name__,
            )
            continue
        await db.execute(
            DocgenArticle.__table__.delete().where(DocgenArticle.session_id == session.id)
        )
        await db.execute(
            DocgenUpload.__table__.delete().where(DocgenUpload.session_id == session.id)
        )
        await db.execute(
            DocgenFields.__table__.update()
            .where(DocgenFields.session_id == session.id)
            .values(data={})
        )
        session.status = SessionStatus.expired.value
        session.document_key = None
        purged += 1
    await _safe_flush(db)
    return purged
