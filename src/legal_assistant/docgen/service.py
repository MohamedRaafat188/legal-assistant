"""Orchestration for docgen. The only module the API routes touch.

Flow: create session -> upload -> lawyer submits the page map -> OCR only the
mapped pages (per-page cache) -> scope each entry by span and article ->
extract identity fields -> one row per declared article -> patch -> lawyer
writes "بعد التعديل" -> render.

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
`_run_stage_thread`/`_run_stage_async`, and `_safe_flush`/`_safe_commit`.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import threading
from collections.abc import Coroutine, Sequence
from typing import Any, NoReturn, TypeVar

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from legal_assistant.config import get_settings
from legal_assistant.db.session import get_sessionmaker
from legal_assistant.docgen import pages, pdf, storage
from legal_assistant.docgen.arabic import normalize_for_match, to_ascii_digits
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
from legal_assistant.docgen.numbering import (
    ArticleRef,
    article_label,
    article_name,
    ordinal_words,
)
from legal_assistant.docgen.ocr.base import OcrError, get_provider
from legal_assistant.docgen.pages import (
    PREAMBLE,
    Entry,
    PageMap,
    PageMapError,  # noqa: F401 -- re-exported for the routes
    validate_page_map,
)
from legal_assistant.docgen.parsing.articles import LookupStatus
from legal_assistant.docgen.parsing.commercial_register import (
    CR_FIELD_SCHEMA,
    CompanyRecord,
    record_from_payload,
)
from legal_assistant.docgen.parsing.scoped import AoaExtraction, extract_aoa, scope
from legal_assistant.docgen.parsing.signatures import Concept, classify
from legal_assistant.docgen.parsing.values import current_value as _current_value
from legal_assistant.docgen.patching import PatchResult, Replacement, patch_article
from legal_assistant.docgen.pdf import InvalidPdfError
from legal_assistant.docgen.render import (
    ArticleBlock,
    Attendee,
    MissingContextError,
    build_context,
    render_document,
)
from legal_assistant.docgen.storage import StorageKeyError
from legal_assistant.docgen.templates.registry import CompanyType, get_template

_log = logging.getLogger(__name__)

_T = TypeVar("_T")


class SessionNotFoundError(LookupError):
    """No such session for this user, or it has expired."""


class SessionStateError(RuntimeError):
    """The session is not in a state where this operation makes sense."""


class NothingSelectedError(SessionStateError):
    """Render was requested with no declared article."""


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

    shares = [_figure_value(p.get("shares")) for p in attendees]
    shares = [v for v in shares if v is not None]
    if not shares:
        return "لا يمكن التحقق من اكتمال كشف الحضور: بيانات الحصص غير معروفة."
    total = sum(shares)

    # Two independent ways a complete roster shows itself. Shares summing to
    # the capital holds when a share is worth one pound (and for a سجل box 9
    # listing each partner's amount); a ذ.م.م table of 100 حصص of 1000 each
    # does not, but its «نسبة المشاركة» column must still add up to 100%. A
    # missing partner breaks both. 1% slack absorbs rounding between two
    # independently-OCR'd figures.
    tolerance = max(1.0, capital_value * 0.01)
    if abs(total - capital_value) <= tolerance:
        return None
    percentages = [_figure_value(p.get("percentage")) for p in attendees]
    if percentages and None not in percentages and abs(sum(percentages) - 100) <= 1:
        return None
    return (
        f"مجموع حصص الشركاء المذكورين فى الكشف ({total:g}) لا يتفق مع رأس المال "
        f"المصدر للشركة ({capital_value:g}). قد يكون كشف الشركاء غير مكتمل أو "
        "غير دقيق؛ راجعه قبل اعتماد نسبة الحضور."
    )


def _figure_value(raw: object) -> float | None:
    """A share count or percentage as a number; None when absent or garbled.
    Digits in either set; %, ٪ and thousands separators are punctuation."""
    if raw is None or raw == "":
        return None
    cleaned = (
        to_ascii_digits(str(raw))
        .replace(",", "")
        .replace("%", "")
        .replace("٪", "")
        .replace("٬", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
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
    `delete_session` raise it too, via `_run_stage_thread`."""


def _fail_stage(stage: str, error_type: str) -> NoReturn:
    raise _StageError(f"فشلت مرحلة «{stage}» أثناء معالجة الملف تلقائيا ({error_type}).")


async def _run_stage_thread(stage: str, fn, /, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking stage (PDF rasterizing, file I/O) in a worker thread, so
    the one event loop keeps serving every other request -- /chat streams
    included -- and `asyncio.wait_for` can bound the job awaiting it."""
    error_type: str | None = None
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except _RISKY_ERRORS as e:
        error_type = type(e).__name__
    _fail_stage(stage, error_type)  # only reached on failure; always raises


# PyMuPDF must not run in two threads at once: every call into it holds this
# lock. They still run one at a time, as before, but off the event loop.
_PDF_LOCK = threading.Lock()


def _locked(fn):
    def call(*args: Any, **kwargs: Any) -> Any:
        with _PDF_LOCK:
            return fn(*args, **kwargs)

    return call


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
    if company_type == CompanyType.MASAHMA.value and not settings.docgen_masahma_enabled:
        raise SessionStateError("إعداد محاضر شركات المساهمة غير متاح حاليا.")
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


_ABANDONED_OCR = "توقفت معالجة الملف قبل اكتمالها. أعد إرسال خريطة الصفحات للمحاولة مرة أخرى."
_OCR_TIMED_OUT = "استغرقت معالجة الملف وقتا أطول من المسموح به. أعد المحاولة."
_OCR_BUSY = "جارٍ تحليل الملف الحالى، انتظر حتى ينتهى."


def _ocr_is_live(session: DocgenSession, *, now: datetime.datetime | None = None) -> bool:
    """True while an OCR job may genuinely still be running for `session`.

    OCR runs inside the web process, so a restart kills a job and leaves
    `status == "ocr_running"` behind with nothing left to finish it. Every
    job is cut off after `docgen_ocr_timeout_minutes` (see `run_ocr_job`),
    so a session still marked running `docgen_ocr_stale_minutes` after its
    last update cannot have a live job: it is abandoned, and retrying it can
    never overlap a running one.
    """
    if session.status != SessionStatus.ocr_running.value:
        return False
    if session.updated_at is None:
        return True
    now = now or datetime.datetime.now(datetime.UTC)
    stale_after = datetime.timedelta(minutes=get_settings().docgen_ocr_stale_minutes)
    return session.updated_at > now - stale_after


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


def _list_sessions_statement(user_id: int):
    """The caller's own unexpired sessions, newest first. Pure, so its WHERE
    clause can be asserted on without a database."""
    return (
        select(DocgenSession)
        .where(
            DocgenSession.user_id == user_id,
            DocgenSession.expires_at > func.now(),
            DocgenSession.status != SessionStatus.expired.value,
        )
        .order_by(DocgenSession.created_at.desc())
    )


async def list_sessions(db: AsyncSession, user_id: int) -> list[DocgenSession]:
    """So a lawyer can resume an unfinished session from any device. Rows
    only: no uploads, articles or fields are loaded."""
    return list((await db.execute(_list_sessions_statement(user_id))).scalars())


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


def displayed_status(session: DocgenSession) -> tuple[str, str | None]:
    """(status, error) as the lawyer should see them. A job that died with
    the process that ran it reads as failed with a retry message, instead of
    a spinner that never ends. Only REPORTED, never written: a write here
    could land on top of a fresh claim and let a second job start."""
    if session.status == SessionStatus.ocr_running.value and not _ocr_is_live(session):
        return SessionStatus.failed.value, _ABANDONED_OCR
    return session.status, session.error


async def add_upload(
    db: AsyncSession, session: DocgenSession, kind: str, filename: str, data: bytes
) -> DocgenUpload:
    """Store an uploaded PDF and record it.

    Starts nothing: the lawyer picks pages from on-demand thumbnails, then
    submits the page map.
    """
    if kind not in {k.value for k in UploadKind}:
        raise SessionStateError(f"unknown upload kind: {kind}")
    if _ocr_is_live(session):
        raise SessionStateError(_OCR_BUSY)

    count = await _run_stage_thread("قراءة عدد صفحات الملف", _locked(pdf.page_count), data)
    key = storage.new_key(session.id, kind)
    await _run_stage_thread("حفظ الملف المرفوع", storage.write, key, data)

    for old in [u for u in session.uploads if u.kind == kind]:
        await _run_stage_thread("حذف الملف السابق", storage.delete, old.storage_key)
        await db.delete(old)
    if kind == UploadKind.aoa.value:
        # Page numbers in the old map may point elsewhere in the new file, and
        # the old file's articles and extracted values must never render as
        # this one's. Only what the lawyer typed is kept.
        session.page_map = None
        for article in list(session.articles):
            await db.delete(article)
        if session.fields is not None:
            session.fields.data = {
                name: entry
                for name, entry in (session.fields.data or {}).items()
                if isinstance(entry, dict) and entry.get("source") == "user"
            }
        session.status = SessionStatus.draft.value
        session.error = None

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


async def submit_page_map(db: AsyncSession, session: DocgenSession, raw: dict) -> PageMap:
    """Validate and store the lawyer's page map. The caller queues OCR.

    Needs the عقد uploaded first (and the سجل too, in aoa_plus_cr): the map is
    validated against the عقد's page count.
    """
    if _ocr_is_live(session):
        raise SessionStateError(_OCR_BUSY)
    uploads = {u.kind: u for u in session.uploads}
    aoa = uploads.get(UploadKind.aoa.value)
    if aoa is None:
        raise SessionStateError("لم يتم رفع عقد التأسيس بعد.")
    if (
        session.source_mode == SourceMode.aoa_plus_cr.value
        and UploadKind.commercial_register.value not in uploads
    ):
        raise SessionStateError("الوضع المختار يتطلب رفع مستخرج السجل التجارى.")
    page_map = validate_page_map(raw, session.company_type, aoa.page_count or 0)
    session.page_map = page_map.to_json()
    _invalidate_document(session)
    await _safe_flush(db)
    return page_map


async def thumbnail(session: DocgenSession, upload_id: int, page: int) -> bytes:
    """One page of one of this session's uploads as PNG, rendered now."""
    upload = next((u for u in session.uploads if u.id == upload_id), None)
    if upload is None or not 1 <= page <= (upload.page_count or 0):
        raise SessionNotFoundError("page not found")
    data = await _run_stage_thread("قراءة الملف المرفوع", storage.read, upload.storage_key)
    dpi = get_settings().docgen_thumbnail_dpi
    return await _run_stage_thread("تجهيز صفحة المعاينة", _locked(pages.thumbnail), data, page, dpi)


def _claim_for_ocr_statement(session_id: int):
    """The atomic claim `run_ocr_job` uses to start OCR for `session_id`.

    Pulled out as its own (pure, DB-independent) function so its WHERE
    clause -- the whole re-entrancy/expiry guarantee -- can be asserted on
    directly (compiled to a literal SQL string) without a database
    connection; see `test_claim_for_ocr_statement_*` in
    `tests/docgen/test_service.py`. The statement itself is only ever
    executed from `run_ocr_job`. Without a submitted page map there are no
    pages to OCR.
    """
    stale_minutes = get_settings().docgen_ocr_stale_minutes
    return (
        update(DocgenSession)
        .where(
            DocgenSession.id == session_id,
            or_(
                DocgenSession.status != SessionStatus.ocr_running.value,
                # Abandoned by a restart; see `_ocr_is_live`.
                DocgenSession.updated_at
                < func.now() - func.make_interval(0, 0, 0, 0, 0, stale_minutes),
            ),
            DocgenSession.expires_at > func.now(),
            DocgenSession.page_map.is_not(None),
        )
        # `updated_at` is the job's start time: `_ocr_is_live` measures from it.
        .values(status=SessionStatus.ocr_running.value, error=None, updated_at=func.now())
    )


async def run_ocr_job(session_id: int) -> None:
    """Background entry point: OCR the mapped pages, scope, extract, patch.

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
    async with get_sessionmaker()() as db:
        if not await claim_for_ocr(db, session_id):
            return
        await _safe_commit(db)
    await run_claimed_ocr_job(session_id)


async def claim_for_ocr(db: AsyncSession, session_id: int) -> bool:
    """Atomically mark `session_id` as running; False if it cannot start.
    The caller commits. `PUT /page-map` claims inside the request, so its
    response already reads `ocr_running`, then queues
    `run_claimed_ocr_job`."""
    result = await db.execute(_claim_for_ocr_statement(session_id))
    if result.rowcount == 0:
        _log.warning(
            "docgen OCR job for session %s did not start (missing, expired, or already running)",
            session_id,
        )
        return False
    return True


async def run_claimed_ocr_job(session_id: int) -> None:
    """The OCR job for a session this caller already claimed."""
    sessionmaker = get_sessionmaker()
    timeout = get_settings().docgen_ocr_timeout_minutes * 60
    try:
        # The cap is what lets `_ocr_is_live` call an old `ocr_running` row
        # abandoned: no job outlives it.
        await asyncio.wait_for(_run_ocr(session_id), timeout=timeout)
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
        elif isinstance(e, TimeoutError):
            message = _OCR_TIMED_OUT
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


_SCALAR_FIELDS = (
    "commercial_registration_no", "commercial_registration_date", "company_name",
    "law_number", "law_year", "company_address", "owner_name", "capital", "issued_capital",
)
# field -> the page-map entry it is read from
_FIELD_ENTRY = {
    "company_name": "company_name", "law_number": "law_reference",
    "law_year": "law_reference", "company_address": "company_address",
    "owner_name": "owner_name", "capital": "issued_capital",
    "issued_capital": "issued_capital",
}


def _article_label(entry: Entry) -> str:
    if entry.article is None:
        return PREAMBLE
    return str(entry.article.number) + (" مكرر" if entry.article.mukarrar else "")


def merge_fields(
    aoa: AoaExtraction, cr: CompanyRecord, previous: dict, page_map: PageMap
) -> tuple[CompanyRecord, dict]:
    """Combine عقد and سجل values into the provenance map the review screen shows.

    - The سجل wins as the PROPOSED value; a disagreement is recorded in
      `conflict`, never silently resolved.
    - A field the lawyer already corrected (`source == "user"`) is kept as is:
      re-running extraction after a page-map edit must never undo a correction.
    - Each field records the span and article it was read from, plus the
      entry's flags.
    - A roster the lawyer edited (`attendees_edited`) is kept as is.
    """
    provenance: dict = {}
    for name in _SCALAR_FIELDS:
        old = previous.get(name) or {}
        if old.get("source") == "user":
            provenance[name] = old
            continue
        cr_value = getattr(cr, name, None)
        aoa_value = aoa.values.get(name)
        value = cr_value or aoa_value
        entry = page_map.entries.get(_FIELD_ENTRY.get(name, ""))
        provenance[name] = {
            "value": value,
            # None, not "user", when nothing was found: "user" is kept across re-runs
            # and exempt from the flag warning, so it must mean the lawyer typed it.
            "source": "cr" if cr_value else ("aoa" if aoa_value else None),
            "span": {"from": entry.first, "to": entry.last} if entry else None,
            "article": _article_label(entry) if entry else None,
            "confidence": 1.0 if value else 0.0,
            "flags": list(aoa.flags.get(entry.name, [])) if entry else [],
            "conflict": (
                {"cr": cr_value, "aoa": aoa_value}
                if cr_value and aoa_value and cr_value != aoa_value
                else None
            ),
        }

    parties = cr.parties or aoa.parties
    if previous.get("attendees_edited"):
        attendees = previous.get("attendees", [])
        provenance["attendees_edited"] = True
    else:
        # Box (9) is authoritative when populated; the عقد's table is the fallback.
        attendees = [
            {"name": p.name, "shares": p.shares, "percentage": p.percentage,
             "attending": True, "source": "cr" if cr.parties else "aoa"}
            for p in parties
        ]
    provenance["attendees"] = attendees
    provenance["capital_reconciliation"] = _reconcile_capital(
        attendees, provenance["issued_capital"]["value"]
    )
    record = CompanyRecord(
        **{n: provenance[n]["value"] for n in _SCALAR_FIELDS}, parties=list(parties)
    )
    return record, provenance


def carry_over(
    amended: Sequence[Entry], existing: Sequence[DocgenArticle]
) -> list[tuple[Entry, DocgenArticle | None, str | None]]:
    """For each declared article, either the existing row to KEEP (same article
    and same span: nothing to re-extract, and the lawyer's edits survive), or
    None plus any "بعد التعديل" text to carry into the rebuilt row (same
    article, span changed). مكرر never matches its base article."""
    out = []
    for entry in amended:
        same_ref = [
            r for r in existing
            if ArticleRef(r.article_number, bool(r.is_mukarrar)) == entry.article
        ]
        exact = next(
            (r for r in same_ref if (r.span_first, r.span_last) == (entry.first, entry.last)),
            None,
        )
        if exact is not None:
            out.append((entry, exact, None))
        else:
            out.append((entry, None, same_ref[0].new_text if same_ref else None))
    return out


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
        page_map = PageMap.from_json(session.page_map)
        previous_fields = dict(session.fields.data or {}) if session.fields else {}

    aoa = uploads.get(UploadKind.aoa.value)
    if aoa is None:
        raise SessionStateError("لم يتم رفع عقد التأسيس.")

    # 1. OCR only the mapped pages not already cached for this session.
    cache = {int(k): v for k, v in (aoa.ocr_pages or {}).items()}
    missing = [p for p in page_map.ocr_pages() if p not in cache]
    if missing:
        aoa_bytes = await _run_stage_thread(
            "قراءة عقد التأسيس المرفوع", storage.read, aoa.storage_key
        )
        images = await _run_stage_thread(
            "تجهيز صفحات العقد للتعرف الضوئى",
            _locked(pdf.render_pages),
            aoa_bytes,
            dpi=settings.docgen_ocr_dpi,
            pages=missing,
        )
        texts = await _run_stage_async("التعرف الضوئى على نص العقد", provider.extract(images))
        for text in texts:
            cache[text.page] = {
                "text": text.text, "margins": text.margins, "confidence": text.confidence
            }
    page_texts = {p: v["text"] for p, v in cache.items()}

    # 2. Scoped extraction: each entry sees only its span + article.
    aoa_values = extract_aoa(page_texts, page_map, company_type)
    cr_record = await _run_stage_async(
        "استخراج بيانات السجل التجارى", _extract_cr(provider, source_mode, uploads, settings)
    )
    record, provenance = merge_fields(aoa_values, cr_record, previous_fields, page_map)

    # 3. Persist.
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
        aoa_row.ocr_pages = {str(p): v for p, v in cache.items()}

        existing = list(
            (
                await db.execute(
                    select(DocgenArticle).where(DocgenArticle.session_id == session_id)
                )
            ).scalars()
        )
        plan = carry_over(page_map.amended, existing)
        kept_ids = {row.id for _entry, row, _text in plan if row is not None}
        for row in existing:
            if row.id not in kept_ids:
                await db.delete(row)
        rebuilt_missing = False
        for position, (entry, row, carried_text) in enumerate(plan):
            if row is not None:
                row.position = position
                continue
            scoped = scope(page_texts, entry)
            rebuilt_missing |= scoped.status is not LookupStatus.found
            new_row = _build_article_row(
                session_id, position, entry, scoped, record, source_mode,
                _span_confidence(cache, entry),
            )
            new_row.new_text = carried_text
            db.add(new_row)

        fields_row = await db.scalar(
            select(DocgenFields).where(DocgenFields.session_id == session_id)
        )
        fields_row.data = provenance

        flagged = any(
            isinstance(v, dict) and v.get("flags") and v.get("source") != "user"
            for v in provenance.values()
        )
        session.status = SessionStatus.ready.value
        # Static text only -- never interpolate document content here.
        session.error = (
            "بعض البيانات أو المواد لم يُعثر عليها فى الصفحات المحددة؛ راجع العلامات "
            "قبل الإنشاء."
            if flagged or rebuilt_missing
            else None
        )
        await _safe_commit(db)


def _span_confidence(cache: dict, entry: Entry) -> float:
    values = [cache[p]["confidence"] for p in entry.pages if p in cache]
    return sum(values) / len(values) if values else 0.0


async def _extract_cr(provider, source_mode: str, uploads: dict, settings) -> CompanyRecord:
    """The سجل side: the whole file, unchanged from before."""
    if source_mode != SourceMode.aoa_plus_cr.value:
        return CompanyRecord()
    cr = uploads.get(UploadKind.commercial_register.value)
    if cr is None:
        return CompanyRecord()
    cr_bytes = await _run_stage_thread(
        "قراءة مستخرج السجل التجارى المرفوع", storage.read, cr.storage_key
    )
    images = await _run_stage_thread(
        "تجهيز صفحات مستخرج السجل التجارى", _locked(pdf.render_pages), cr_bytes,
        dpi=settings.docgen_ocr_dpi,
    )
    return record_from_payload(await provider.extract_fields(images, CR_FIELD_SCHEMA))


def _build_article_row(
    session_id: int,
    position: int,
    entry: Entry,
    scoped,
    record: CompanyRecord,
    source_mode: str,
    confidence: float,
) -> DocgenArticle:
    ref = entry.article
    base = dict(
        session_id=session_id,
        position=position,
        article_number=ref.number,
        is_mukarrar=ref.mukarrar,
        ordinal_words=ordinal_words(ref.number) if 1 <= ref.number <= 99 else str(ref.number),
        span_first=entry.first,
        span_last=entry.last,
        confidence=confidence,
    )
    if scoped.status is not LookupStatus.found:
        # Nothing verbatim to show; the lawyer fixes the map or types "قبل".
        return DocgenArticle(
            **base, status=scoped.status.value, source_text="", patched_text="",
            patch_ops=[], needs_review=True, possibly_truncated=False,
        )
    article = scoped.article
    if source_mode == SourceMode.aoa_plus_cr.value:
        replacements = plan_patches(record, classify(article), article.body)
    else:
        # aoa_only: the عقد has never been amended -- a declared no-op.
        replacements = []
    result = apply_patches(article.body, replacements)
    return DocgenArticle(
        **base,
        status=LookupStatus.found.value,
        source_text=article.body,
        patched_text=result.text,
        patch_ops=[
            {"start": op.start, "end": op.end, "old": op.old, "new": op.new,
             "field": op.field, "source": op.source}
            for op in result.ops
        ],
        needs_review=result.needs_review or scoped.truncated,
        possibly_truncated=scoped.truncated,
    )


# --- review edits -----------------------------------------------------------


def _refuse_while_ocr_running(session: DocgenSession) -> None:
    """A running job rewrites fields and articles from a snapshot taken when
    it started, so an edit made meanwhile would be silently lost."""
    if _ocr_is_live(session):
        raise SessionStateError(_OCR_BUSY)


async def read_document(session: DocgenSession) -> bytes:
    """The rendered .docx. A missing file is a content-free `_StageError`
    (a raw FileNotFoundError names the storage key)."""
    return await _run_stage_thread("قراءة المستند الناتج", storage.read, session.document_key)


def _invalidate_document(session: DocgenSession) -> None:
    """Any edit makes a previously rendered document stale."""
    session.document_key = None
    if session.status == SessionStatus.rendered.value:
        session.status = SessionStatus.ready.value


async def update_fields(db: AsyncSession, session: DocgenSession, changes: dict) -> None:
    """Apply lawyer corrections to identity fields, marking them user-sourced."""
    _refuse_while_ocr_running(session)
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
    position: int,
    *,
    patched_text: str | None = None,
    new_text: str | None = None,
) -> DocgenArticle:
    """Correct an article's "قبل التعديل", or write its "بعد التعديل".

    `source_text` is never touched, so the verbatim OCR output stays
    auditable next to whatever the lawyer changed.
    """
    _refuse_while_ocr_running(session)
    article = await db.scalar(
        select(DocgenArticle).where(
            DocgenArticle.session_id == session.id,
            DocgenArticle.position == position,
        )
    )
    if article is None:
        raise SessionNotFoundError(f"article position {position} is not in this session")
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
    _refuse_while_ocr_running(session)
    row = await db.scalar(select(DocgenFields).where(DocgenFields.session_id == session.id))
    data = dict(row.data or {})
    data["attendees"] = attendees
    # See `merge_fields`: this flag is what stops a later OCR re-run from
    # clobbering a roster the lawyer has hand-edited here.
    data["attendees_edited"] = True
    issued_capital = ((data.get("issued_capital") or {}).get("value")) or None
    data["capital_reconciliation"] = _reconcile_capital(attendees, issued_capital)
    row.data = data
    _invalidate_document(session)
    await _safe_flush(db)


# --- render ------------------------------------------------------------------


# Arabic names for the render error that lists empty fields.
_FIELD_LABELS = {
    "company_name": "اسم الشركة",
    "law_number": "رقم القانون",
    "law_year": "سنة القانون",
    "commercial_registration_no": "رقم السجل التجاري",
    "commercial_registration_date": "تاريخ القيد بالسجل التجاري",
    "commercial_registry_office": "مكتب السجل التجاري",
    "day_name": "يوم الاجتماع",
    "day_date": "تاريخ الاجتماع",
    "company_address": "عنوان المركز الرئيسي",
    "owner_name": "اسم مالك الشركة",
    "names_of_commissioners": "أسماء المفوضين",
    "chairman_name": "رئيس الاجتماع",
    "chairman_title": "لقب رئيس الاجتماع",
    "auditor_name": "مراقب الحسابات",
    "secretary_name": "أمين السر",
    "vote_counter_1": "فارز الأصوات الأول",
    "vote_counter_2": "فارز الأصوات الثاني",
    "meeting_time": "وقت بدء الاجتماع",
    "meeting_end_time": "وقت انتهاء الاجتماع",
    "attendance_percentage": "نسبة الحضور",
    "approval_percentage": "نسبة الموافقة",
    "attendees": "جدول الحضور",
    "articles": "المواد المعدلة",
}


def _as_percentage(value: str) -> str:
    """"100" / "100%" / "١٠٠ ٪" -> "100%" (empty stays empty): the templates
    print the figure without a sign of their own."""
    figure = to_ascii_digits(str(value or "")).replace("%", "").replace("٪", "").strip()
    return f"{figure}%" if figure else ""


def attendee_role(name: str, chairman_name: str) -> str:
    """«مدير الشركة» for the partner who chairs the meeting, «شريك» for the
    rest. The ذ.م.م محضر names its chairman as the company's manager, so the
    two are the same person; names compare with spelling variants folded."""
    same = bool(chairman_name.strip()) and normalize_for_match(name) == normalize_for_match(
        chairman_name
    )
    return "مدير الشركة" if same else "شريك"


async def render_session(db: AsyncSession, session: DocgenSession) -> bytes:
    """Render the document. Idempotent and re-runnable.

    Fails closed rather than emit a partially-complete instrument:
    - no declared article -> `NothingSelectedError`;
    - a declared article with no "بعد التعديل" text yet -> `SessionStateError`
      naming it;
    - for a company type with an attendance table, a roster that does not
      reconcile against the issued capital -> `SessionStateError` (a wrong
      quorum percentage in a محضر جمعية عامة is a legal defect in the filed
      instrument, not a cosmetic one -- see `_reconcile_capital`);
    - any placeholder `render_document` cannot fill -> `MissingContextError`
      (raised by `render_document` itself; not caught or papered over here).
    """
    _refuse_while_ocr_running(session)
    articles = list(session.articles)
    if not articles:
        raise NothingSelectedError("لا توجد مواد معلنة للتعديل؛ أرسل خريطة الصفحات أولا.")
    if session.page_map is None or session.status not in (
        SessionStatus.ready.value,
        SessionStatus.rendered.value,
    ):
        raise SessionStateError("أكمل تحليل الملف بنجاح أولا، ثم أنشئ المستند.")
    def labels(rows: Sequence[DocgenArticle]) -> str:
        return "، ".join(article_name(a.article_number, a.is_mukarrar) for a in rows)

    blank = [a for a in articles if not (a.patched_text or "").strip()]
    if blank:
        raise SessionStateError(f"اكتب نص «قبل التعديل» للمواد: {labels(blank)}")
    missing = [a for a in articles if not (a.new_text or "").strip()]
    if missing:
        raise SessionStateError(f"اكتب نص «بعد التعديل» للمواد: {labels(missing)}")

    row = await db.scalar(select(DocgenFields).where(DocgenFields.session_id == session.id))
    data = row.data or {}
    scalars = {
        name: (entry or {}).get("value") or ""
        for name, entry in data.items()
        if name not in ("attendees", "capital_reconciliation", "attendees_edited")
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
        for name in ("attendance_percentage", "approval_percentage"):
            scalars[name] = _as_percentage(scalars[name])
    if "p.title" in spec.attendee_placeholders:
        untitled = [p.get("name", "") for p in attendee_rows if not (p.get("title") or "").strip()]
        if untitled:
            raise SessionStateError(
                "اختر اللقب (السيد / السيدة / السادة) في جدول الشركاء لـ: " + "، ".join(untitled)
            )
    for name in ("law_number", "law_year"):
        scalars[name] = to_ascii_digits(scalars.get(name, ""))

    chairman = scalars.get("chairman_name", "")
    if "chairman_title" in spec.scalar_placeholders and not scalars.get("chairman_title"):
        # The chairman is normally a partner: reuse the title chosen for them.
        scalars["chairman_title"] = next(
            (
                (p.get("title") or "").strip()
                for p in attendee_rows
                if attendee_role(p.get("name", ""), chairman) == "مدير الشركة"
            ),
            "",
        )
    context = build_context(
        session.company_type,
        scalars=scalars,
        articles=[
            ArticleBlock(
                article_name=article_name(a.article_number, a.is_mukarrar),
                article_original_content=a.patched_text,
                article_new_content=a.new_text or "",
                article_label=article_label(a.article_number, a.is_mukarrar),
            )
            for a in articles
        ],
        article_numbers=[ArticleRef(a.article_number, a.is_mukarrar) for a in articles],
        attendees=[
            Attendee(
                name=p.get("name", ""),
                shares=str(p.get("shares") or ""),
                percentage=str(p.get("percentage") or ""),
                title=(p.get("title") or "").strip(),
                role=attendee_role(p.get("name", ""), chairman),
            )
            for p in attendee_rows
        ],
    )

    try:
        document = await asyncio.to_thread(render_document, session.company_type, context)
    except MissingContextError as e:
        names = "، ".join(_FIELD_LABELS.get(n, n) for n in sorted(e.missing))
        raise SessionStateError(f"أكمل البيانات التالية قبل إنشاء المستند: {names}") from e
    previous_key = session.document_key
    key = storage.new_key(session.id, "document")
    # Write the NEW object and commit the pointer to it before touching the
    # old one: if deleting the old object then failed partway (e.g. an OSError
    # mid-unlink), the session must still point at a document that exists --
    # never at one it just deleted.
    await _run_stage_thread("حفظ المستند الناتج", storage.write, key, document)
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
    await _run_stage_thread("حذف ملفات الجلسة", storage.delete_session, session.id)
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
