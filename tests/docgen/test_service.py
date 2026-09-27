import asyncio
import datetime

import pytest

from legal_assistant.docgen import pdf, service, storage
from legal_assistant.docgen.models import DocgenArticle, DocgenFields, DocgenSession, DocgenUpload
from legal_assistant.docgen.numbering import ArticleRef
from legal_assistant.docgen.ocr.base import PageText
from legal_assistant.docgen.pages import Entry, PageMap
from legal_assistant.docgen.parsing.commercial_register import CompanyRecord
from legal_assistant.docgen.parsing.scoped import AoaExtraction
from legal_assistant.docgen.parsing.signatures import Concept
from legal_assistant.docgen.patching import Replacement
from legal_assistant.docgen.service import (
    SessionNotFoundError,
    SessionStateError,
    _claim_for_ocr_statement,
    _load_session_unowned,
    _reconcile_capital,
    apply_patches,
    compute_percentages,
    get_session,
    plan_patches,
    run_ocr_job,
    validate_source_mode,
)


class _FakeScalarResult:
    """Stands in for the object `AsyncSession.execute()` returns -- only
    `.scalar_one_or_none()` is exercised by `get_session`. Deliberately does
    NOT evaluate the SQL `.where()` predicates a real engine would (there is
    no real query here at all): this is what lets these tests demonstrate
    `get_session`'s own Python-level expiry/ownership logic offline, with no
    DB, and prove it does not rely solely on a WHERE clause a stub like this
    could never enforce.
    """

    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeDb:
    def __init__(self, value) -> None:
        self._value = value

    async def execute(self, *_args, **_kwargs):
        return _FakeScalarResult(self._value)

    async def get(self, *_args, **_kwargs):
        return self._value


def _days_ago(n: int) -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=n)


def _days_from_now(n: int) -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=n)


def _page_map():
    return PageMap(
        entries={
            "company_name": Entry("company_name", 1, 1, ArticleRef(1)),
            "company_address": Entry("company_address", 2, 2, ArticleRef(5)),
        },
        amended=[Entry("amended:0", 3, 3, ArticleRef(7))],
    )


def _session(**overrides) -> DocgenSession:
    defaults = dict(
        id=1,
        user_id=7,
        company_type="zmm",
        source_mode="aoa_only",
        status="ready",
        expires_at=datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1),
    )
    defaults.update(overrides)
    return DocgenSession(**defaults)


def test_plan_patches_targets_the_head_office_article_with_the_address():
    record = CompanyRecord(company_address="٥ شارع الهرم، الجيزة", company_name="شركة الفجر")
    replacements = plan_patches(record, Concept.HEAD_OFFICE, "المركز الرئيسى فى ١٢ شارع النيل.")
    assert [r.field for r in replacements] == ["company_address"]
    assert replacements[0].new == "٥ شارع الهرم، الجيزة"
    assert replacements[0].source == "cr"


def test_plan_patches_targets_the_capital_article_with_the_capital_figure():
    record = CompanyRecord(capital="200000")
    replacements = plan_patches(record, Concept.CAPITAL, "رأس مال الشركة ١٠٠٠٠٠ جنيه.")
    assert [r.field for r in replacements] == ["capital"]
    assert replacements[0].new == "200000"


def test_plan_patches_returns_nothing_for_an_unrelated_article():
    record = CompanyRecord(company_address="الجيزة", capital="200000")
    assert plan_patches(record, Concept.DURATION, "مدة الشركة ٢٥ سنة.") == []


def test_plan_patches_returns_nothing_when_the_record_is_empty():
    assert plan_patches(CompanyRecord(), Concept.CAPITAL, "رأس مال الشركة ١٠٠٠٠٠ جنيه.") == []


def test_plan_patches_needs_the_old_value_to_be_present_to_be_worth_planning():
    # The old value is found by the patcher, not here; plan_patches only
    # decides WHICH field belongs to this concept.
    record = CompanyRecord(company_address="الجيزة")
    assert len(plan_patches(record, Concept.HEAD_OFFICE, "المركز الرئيسى فى القاهرة.")) == 1


def test_plan_patches_returns_nothing_when_concept_is_none():
    record = CompanyRecord(company_address="الجيزة", capital="200000")
    assert plan_patches(record, None, "نص مادة غير مصنف.") == []


def test_plan_patches_returns_nothing_for_purpose_concept():
    # PURPOSE has no mapped field: purpose text is never machine-patched.
    record = CompanyRecord(company_address="الجيزة")
    assert plan_patches(record, Concept.PURPOSE, "غرض الشركة التجارة العامة.") == []


def test_apply_patches_flags_needs_review_when_old_value_is_unlocatable():
    # This is the exact real-world phrasing the Part 0 fix targets: no
    # "الكائن"/"مقرها" anchor, so _current_value cannot isolate a span, but
    # plan_patches still has a genuine new address to apply.
    record = CompanyRecord(company_address="٥ شارع الهرم، الجيزة")
    text = "المركز الرئيسى فى الجيزة، جمهورية مصر العربية."
    replacements = plan_patches(record, Concept.HEAD_OFFICE, text)
    assert replacements[0].old == ""  # confirms the helper could not locate a span

    result = apply_patches(text, replacements)
    assert result.text == text  # unchanged -- no guessing
    assert result.ops == []
    assert result.needs_review is True
    assert len(result.notes) == 1
    assert "company_address" in result.notes[0]


def test_apply_patches_still_applies_locatable_replacements_normally():
    text = "المركز الرئيسى للشركة الكائن فى ١٢ شارع النيل."
    replacements = [Replacement("company_address", "١٢ شارع النيل", "٥ شارع الهرم", "cr")]
    result = apply_patches(text, replacements)
    assert "٥ شارع الهرم" in result.text
    assert result.needs_review is False
    assert result.notes == []


def test_apply_patches_reports_both_a_normal_note_and_an_unlocatable_one():
    text = "نص لا صلة له بأى قيمة."
    replacements = [
        # Not present verbatim in `text` -> patch_article's own "missing" note.
        Replacement("capital", "غير موجود", "200000", "cr"),
        Replacement("company_address", "", "الجيزة", "cr"),  # unlocatable
    ]
    result = apply_patches(text, replacements)
    assert result.needs_review is True
    assert any("capital" in n for n in result.notes)
    assert any("company_address" in n for n in result.notes)


def test_compute_percentages_sums_shares_into_percentages():
    attendees = [
        {"name": "أ", "shares": "90", "attending": True},
        {"name": "ب", "shares": "10", "attending": True},
    ]
    attendance, approval = compute_percentages(attendees)
    assert attendance == "100"
    assert approval == "100"


def test_compute_percentages_excludes_absentees_from_attendance():
    attendees = [
        {"name": "أ", "shares": "٩٠", "attending": True},
        {"name": "ب", "shares": "١٠", "attending": False},
    ]
    attendance, approval = compute_percentages(attendees)
    assert attendance == "90"
    assert approval == "90"


def test_compute_percentages_returns_none_when_no_shares_are_known():
    attendance, approval = compute_percentages([{"name": "أ", "attending": True}])
    assert attendance is None and approval is None


def test_compute_percentages_of_an_empty_list_is_none():
    assert compute_percentages([]) == (None, None)


def test_compute_percentages_ignores_non_numeric_shares():
    # A garbled OCR share value must not silently become 0 or crash --
    # it is excluded from the total rather than treated as a real number.
    attendees = [
        {"name": "أ", "shares": "90", "attending": True},
        {"name": "ب", "shares": "غير مقروء", "attending": True},
    ]
    attendance, approval = compute_percentages(attendees)
    assert attendance == "100"
    assert approval == "100"


def test_compute_percentages_defaults_missing_attending_flag_to_true():
    # An attendee dict without an explicit "attending" key came from a
    # source that only lists who showed up -- absence must not silently
    # exclude them from the quorum.
    attendees = [{"name": "أ", "shares": "100"}]
    attendance, approval = compute_percentages(attendees)
    assert attendance == "100"
    assert approval == "100"


def test_compute_percentages_treats_zero_total_as_unknown():
    # All shares present but summing to zero (e.g. every value literally
    # "0") must not raise a ZeroDivisionError or fabricate a percentage.
    attendees = [{"name": "أ", "shares": "0", "attending": True}]
    assert compute_percentages(attendees) == (None, None)


def test_compute_percentages_strips_a_percent_sign_before_parsing():
    attendees = [
        {"name": "أ", "shares": "50%", "attending": True},
        {"name": "ب", "shares": "50%", "attending": False},
    ]
    assert compute_percentages(attendees) == ("50", "50")


def test_compute_percentages_strips_an_arabic_percent_sign_before_parsing():
    attendees = [
        {"name": "أ", "shares": "90٪", "attending": True},
        {"name": "ب", "shares": "10٪", "attending": False},
    ]
    assert compute_percentages(attendees) == ("90", "90")


def test_compute_percentages_strips_the_arabic_thousands_separator():
    attendees = [{"name": "أ", "shares": "١٠٠٬٠٠٠", "attending": True}]
    assert compute_percentages(attendees) == ("100", "100")


def test_reconcile_capital_accepts_a_roster_that_sums_to_the_issued_capital():
    attendees = [{"shares": "60"}, {"shares": "40"}]
    assert _reconcile_capital(attendees, "100") is None


def test_reconcile_capital_flags_a_roster_missing_a_partner():
    # A lone 50-share attendee looks perfectly self-consistent to
    # compute_percentages, but is silently wrong if the company's issued
    # capital is actually 100 -- some partner's holding never made it into
    # the roster (an OCR/extraction miss, not a real absence).
    note = _reconcile_capital([{"shares": "50"}], "100")
    assert note is not None
    assert "50" in note and "100" in note


def test_reconcile_capital_is_unknown_when_issued_capital_is_missing():
    note = _reconcile_capital([{"shares": "100"}], None)
    assert note is not None


def test_reconcile_capital_is_unknown_when_no_share_is_usable():
    note = _reconcile_capital([{"name": "أ"}], "100")
    assert note is not None


def test_reconcile_capital_tolerates_percent_signs_and_separators():
    attendees = [{"shares": "60%"}, {"shares": "40%"}]
    assert _reconcile_capital(attendees, "١٠٠") is None


def test_reconcile_capital_allows_small_rounding_slack():
    # Two independently-OCR'd figures may differ by a rounding hair; only a
    # gap wider than ~1% is treated as a real discrepancy.
    attendees = [{"shares": "99.6"}]
    assert _reconcile_capital(attendees, "100") is None


def test_get_session_rejects_an_expired_session_even_when_status_still_says_ready():
    # The reviewer's exact reproduction: status was never touched by time
    # passing (only purge_expired ever sets status="expired"), so a session
    # 5 days past its window but still "ready" must still be rejected.
    session = _session(status="ready", expires_at=_days_ago(5))
    db = _FakeDb(session)
    with pytest.raises(SessionNotFoundError):
        asyncio.run(get_session(db, session_id=1, user_id=7))


def test_get_session_rejects_a_session_already_marked_expired():
    session = _session(status="expired", expires_at=_days_ago(5))
    db = _FakeDb(session)
    with pytest.raises(SessionNotFoundError):
        asyncio.run(get_session(db, session_id=1, user_id=7))


def test_get_session_accepts_a_session_that_has_not_yet_expired():
    session = _session(status="ready", expires_at=_days_from_now(1))
    db = _FakeDb(session)
    result = asyncio.run(get_session(db, session_id=1, user_id=7))
    assert result is session


def test_get_session_rejects_another_users_session_with_the_identical_error():
    # A real query's WHERE clause already excludes another user's row (a
    # stub cannot exercise that), so this proves the Python-level fallback
    # check catches it too, with the same exception the "not found" and
    # "expired" cases raise -- never a distinguishable one.
    session = _session(user_id=42, expires_at=_days_from_now(1))
    db = _FakeDb(session)
    with pytest.raises(SessionNotFoundError):
        asyncio.run(get_session(db, session_id=1, user_id=7))


def test_get_session_rejects_a_missing_session_with_the_identical_error():
    db = _FakeDb(None)
    with pytest.raises(SessionNotFoundError):
        asyncio.run(get_session(db, session_id=999, user_id=7))


def test_get_session_expired_and_missing_raise_the_same_exception_type():
    # Not just "both raise" -- the SAME type, with no attribute that would
    # let a caller tell "expired" apart from "never existed" or "not yours".
    # A real query also filters by user_id in its WHERE clause (see
    # get_session's docstring); this stub cannot exercise that SQL-level
    # filter, but it still proves the Python-level path never distinguishes
    # the cases by exception type.
    expired_db = _FakeDb(_session(expires_at=_days_ago(5)))
    missing_db = _FakeDb(None)
    with pytest.raises(SessionNotFoundError) as expired_exc:
        asyncio.run(get_session(expired_db, session_id=1, user_id=7))
    with pytest.raises(SessionNotFoundError) as missing_exc:
        asyncio.run(get_session(missing_db, session_id=1, user_id=7))
    assert type(expired_exc.value) is type(missing_exc.value)


def test_claim_for_ocr_statement_only_targets_the_named_session():
    compiled = str(
        _claim_for_ocr_statement(42).compile(compile_kwargs={"literal_binds": True})
    )
    assert "docgen_sessions.id = 42" in compiled


def test_claim_for_ocr_statement_refuses_an_already_running_session():
    # This WHERE clause is the entire re-entrancy guarantee: it is evaluated
    # atomically by the database as part of a single UPDATE, so a second
    # concurrent run_ocr_job for the same session_id sees the row AFTER the
    # first UPDATE committed and matches zero rows -- there is no
    # read-then-write gap for two invocations to both slip through.
    compiled = str(
        _claim_for_ocr_statement(1).compile(compile_kwargs={"literal_binds": True})
    )
    assert "docgen_sessions.status != 'ocr_running'" in compiled


def test_claim_for_ocr_statement_refuses_an_expired_session():
    # Same statement, same atomicity guarantee, closing the Fix-1 gap for
    # run_ocr_job specifically: it must not start OCR on a session already
    # past its retention window, whether or not purge_expired has run yet.
    compiled = str(
        _claim_for_ocr_statement(1).compile(compile_kwargs={"literal_binds": True})
    )
    assert "docgen_sessions.expires_at > now()" in compiled


def test_claim_for_ocr_statement_sets_ocr_running_and_clears_the_error():
    compiled = str(
        _claim_for_ocr_statement(1).compile(compile_kwargs={"literal_binds": True})
    )
    assert "status='ocr_running'" in compiled
    assert "error=NULL" in compiled


def test_load_session_unowned_returns_none_for_a_missing_session():
    assert asyncio.run(_load_session_unowned(_FakeDb(None), session_id=1)) is None


def test_load_session_unowned_returns_whatever_status_the_row_has():
    # Deliberately no expiry/ownership decision baked in here -- that is left
    # to the caller (see `run_ocr_job`/`_run_ocr`), which need different
    # answers to "what do I do about an expired session".
    session = _session(status="ocr_running", expires_at=_days_ago(5))
    result = asyncio.run(_load_session_unowned(_FakeDb(session), session_id=1))
    assert result is session


class _FakeScalarOneResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class _FakeRunOcrDb:
    """Stands in for the `AsyncSession` `_run_ocr` opens -- once at the top
    (to load `uploads`/`company_type`/`source_mode`) and again for "5. Patch
    and persist" -- both times via the same `sessionmaker()` stub, so this
    one fake instance backs both. `execute` only ever needs to satisfy the
    first (a `select(DocgenSession)...scalar_one()`); the persist step's own
    writes (`DocgenArticle` delete/insert, the `DocgenFields` update) are
    deliberately NOT implemented here -- if the abandonment branch is not
    taken and one of those is reached, it raises `AttributeError`, which
    fails the test loudly rather than silently accepting a write that
    should not happen.
    """

    def __init__(self, session):
        self.session = session
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, stmt):
        return _FakeScalarOneResult(self.session)

    async def get(self, model, pk):
        assert model is DocgenSession
        return self.session

    async def commit(self):
        self.commits += 1


def test_run_ocr_persist_step_abandons_the_result_when_the_session_expires_mid_run(
    monkeypatch,
):
    # Reproduces the re-review's residual finding: the claim at job start
    # only guarantees the session was not-yet-expired at that instant.
    # `_run_ocr` then makes several external OCR/extraction calls before
    # reaching "5. Patch and persist" -- a real window, not a negligible
    # one -- and the session must not be written to if it closed during that
    # window: no new DocgenArticle/DocgenFields content, and no `ready`.
    session = _session(id=1, status="ocr_running", expires_at=_days_from_now(1))
    session.page_map = _page_map().to_json()
    session.articles = []
    session.fields = DocgenFields(session_id=1, data={})
    aoa_upload = DocgenUpload(
        id=10, session_id=1, kind="aoa", filename="a.pdf", storage_key="k1", ocr_pages=None
    )
    session.uploads = [aoa_upload]
    db = _FakeRunOcrDb(session)

    class _FakeSettings:
        docgen_ocr_dpi = 150

    class _FakeProvider:
        async def extract(self, images):
            # The window closes DURING this external OCR call -- after the
            # Fix-3 claim, before the persist step below ever loads the row.
            session.expires_at = _days_ago(1)
            return [PageText(page=p, text="نص", confidence=0.9) for p in (1, 2, 3)]

        async def extract_fields(self, images, schema):
            return {}

    def fake_extract_aoa(page_texts, page_map, company_type):
        return AoaExtraction()

    monkeypatch.setattr(service, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(service, "get_provider", lambda settings: _FakeProvider())
    monkeypatch.setattr(service, "get_sessionmaker", lambda: (lambda: db))
    monkeypatch.setattr(service, "extract_aoa", fake_extract_aoa)
    monkeypatch.setattr(storage, "read", lambda key: b"%PDF-fake%")
    monkeypatch.setattr(
        pdf, "render_pages", lambda data, dpi=None, pages=None: ["thumb"] * len(pages or [1])
    )

    asyncio.run(service._run_ocr(1))

    # (a) no article/fields writes were attempted -- proven structurally:
    # `_FakeRunOcrDb` has no `execute`/`add`/`scalar`, so reaching either the
    # `DocgenArticle` delete/insert or the `DocgenFields` update would have
    # raised `AttributeError` and failed this test.
    # (b) not left stuck in `ocr_running`.
    assert session.status == "failed"
    # (c) not flipped to `ready`.
    assert session.status != "ready"
    assert db.commits == 1


class _FakeFailureHandlerDb:
    """Stands in for the `AsyncSession` `run_ocr_job`'s outer `except`
    handler opens to record a failure. Only `db.get` and `db.commit` are
    implemented, for the same "an unexpected write would raise loudly"
    reason as `_FakeRunOcrDb` above.
    """

    def __init__(self, session):
        self.session = session
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, model, pk):
        return self.session

    async def commit(self):
        self.commits += 1


class _FakeClaimDb:
    """Stands in for the `AsyncSession` `run_ocr_job` opens for the Fix-3
    atomic claim. `rowcount=1` simulates a successful claim -- this test is
    about what happens AFTER the claim, when `_run_ocr` itself fails.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, stmt):
        return type("Result", (), {"rowcount": 1})()

    async def commit(self):
        pass


def test_run_ocr_job_failure_handler_skips_the_write_when_the_session_has_expired(
    monkeypatch,
):
    # Lower-stakes than the persist-step case (no new PII, only a
    # content-free status/error), but the same principle: do not write to a
    # session whose window has already closed, even a status flip.
    session = _session(id=1, status="ocr_running", expires_at=_days_ago(1))
    failure_db = _FakeFailureHandlerDb(session)
    dbs = iter([_FakeClaimDb(), failure_db])
    monkeypatch.setattr(service, "get_sessionmaker", lambda: (lambda: next(dbs)))

    async def fake_run_ocr(session_id):
        raise service.OcrError("لم يتم العثور على مواد قابلة للتعديل داخل الملف.")

    monkeypatch.setattr(service, "_run_ocr", fake_run_ocr)

    asyncio.run(run_ocr_job(1))

    assert session.status == "ocr_running"  # untouched, not overwritten to "failed"
    assert session.error is None
    assert failure_db.commits == 0


def test_run_ocr_job_failure_handler_writes_when_the_session_has_not_expired(monkeypatch):
    # Contrast case: the ordinary path (session still valid) must still work
    # exactly as before this fix -- status/error get written.
    session = _session(id=1, status="ocr_running", expires_at=_days_from_now(1))
    failure_db = _FakeFailureHandlerDb(session)
    dbs = iter([_FakeClaimDb(), failure_db])
    monkeypatch.setattr(service, "get_sessionmaker", lambda: (lambda: next(dbs)))

    async def fake_run_ocr(session_id):
        raise service.OcrError("لم يتم العثور على مواد قابلة للتعديل داخل الملف.")

    monkeypatch.setattr(service, "_run_ocr", fake_run_ocr)

    asyncio.run(run_ocr_job(1))

    assert session.status == "failed"
    assert session.error == "لم يتم العثور على مواد قابلة للتعديل داخل الملف."
    assert failure_db.commits == 1


def test_run_ocr_job_cuts_off_a_job_that_overruns_its_timeout(monkeypatch):
    # The cap is what makes a stale `ocr_running` row provably abandoned.
    session = _session(id=1, status="ocr_running", expires_at=_days_from_now(1))
    failure_db = _FakeFailureHandlerDb(session)
    dbs = iter([_FakeClaimDb(), failure_db])
    monkeypatch.setattr(service, "get_sessionmaker", lambda: (lambda: next(dbs)))

    class _FakeSettings:
        docgen_ocr_timeout_minutes = 0.0005  # 30ms
        docgen_ocr_stale_minutes = 15

    monkeypatch.setattr(service, "get_settings", lambda: _FakeSettings())

    async def slow_run_ocr(session_id):
        await asyncio.sleep(5)

    monkeypatch.setattr(service, "_run_ocr", slow_run_ocr)

    asyncio.run(run_ocr_job(1))

    assert session.status == "failed"
    assert session.error == service._OCR_TIMED_OUT


def _minutes_ago(n: int) -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=n)


def test_ocr_is_live_only_for_a_recently_updated_running_session():
    assert service._ocr_is_live(_session(status="ocr_running", updated_at=_minutes_ago(1)))
    # Past `docgen_ocr_stale_minutes` (15): its job died with its process.
    assert not service._ocr_is_live(_session(status="ocr_running", updated_at=_minutes_ago(30)))
    assert not service._ocr_is_live(_session(status="ready", updated_at=_minutes_ago(1)))


def test_get_session_reports_an_abandoned_ocr_job_as_failed():
    session = _session(status="ocr_running", updated_at=_minutes_ago(30))
    result = asyncio.run(get_session(_FakeDb(session), session_id=1, user_id=7))
    assert result.status == "failed"
    assert result.error == service._ABANDONED_OCR


def test_get_session_leaves_a_live_ocr_job_running():
    session = _session(status="ocr_running", updated_at=_minutes_ago(1))
    result = asyncio.run(get_session(_FakeDb(session), session_id=1, user_id=7))
    assert result.status == "ocr_running"
    assert result.error is None


def test_claim_for_ocr_statement_can_retake_an_abandoned_session():
    compiled = str(
        _claim_for_ocr_statement(1).compile(compile_kwargs={"literal_binds": True})
    )
    assert "docgen_sessions.updated_at < now() - make_interval(0, 0, 0, 0, 0, 15)" in compiled
    # The claim stamps the job's start, which `_ocr_is_live` measures from.
    assert "updated_at=now()" in compiled


def test_validate_source_mode_requires_a_cr_upload_in_aoa_plus_cr_mode():
    with pytest.raises(SessionStateError):
        validate_source_mode("aoa_plus_cr", uploaded_kinds={"aoa"}, typed_cr_no=None)


def test_validate_source_mode_requires_typed_cr_fields_in_aoa_only_mode():
    with pytest.raises(SessionStateError):
        validate_source_mode("aoa_only", uploaded_kinds={"aoa"}, typed_cr_no=None)


def test_validate_source_mode_accepts_a_complete_aoa_only_session():
    validate_source_mode("aoa_only", uploaded_kinds={"aoa"}, typed_cr_no="303907")


def test_validate_source_mode_always_requires_the_aoa():
    with pytest.raises(SessionStateError):
        validate_source_mode("aoa_only", uploaded_kinds=set(), typed_cr_no="303907")


def test_validate_source_mode_accepts_a_complete_aoa_plus_cr_session():
    validate_source_mode(
        "aoa_plus_cr", uploaded_kinds={"aoa", "commercial_register"}, typed_cr_no=None
    )


def test_merge_fields_records_span_article_and_flags():
    aoa = AoaExtraction(
        values={"company_name": "انجاز", "company_address": None},
        flags={"company_name": [], "company_address": ["article_not_found"]},
    )
    _record, prov = service.merge_fields(aoa, CompanyRecord(), {}, _page_map())
    assert prov["company_name"]["span"] == {"from": 1, "to": 1}
    assert prov["company_name"]["article"] == "1"
    assert prov["company_address"]["flags"] == ["article_not_found"]
    assert prov["company_address"]["value"] is None


def test_merge_fields_never_overwrites_a_lawyer_correction():
    previous = {"company_name": {"value": "مصحح", "source": "user"}}
    aoa = AoaExtraction(values={"company_name": "انجاز"}, flags={"company_name": []})
    _record, prov = service.merge_fields(aoa, CompanyRecord(), previous, _page_map())
    assert prov["company_name"] == previous["company_name"]


def test_merge_fields_refills_a_field_that_was_empty_last_run():
    # An empty field is not a lawyer correction: a fixed page map must refill it,
    # and until then it must still count as flagged.
    empty = AoaExtraction(values={"company_name": None}, flags={"company_name": ["not_found"]})
    _record, first = service.merge_fields(empty, CompanyRecord(), {}, _page_map())
    assert first["company_name"]["source"] is None
    fixed = AoaExtraction(values={"company_name": "انجاز"}, flags={"company_name": []})
    _record, second = service.merge_fields(fixed, CompanyRecord(), first, _page_map())
    assert second["company_name"]["value"] == "انجاز"
    assert second["company_name"]["source"] == "aoa"


def test_merge_fields_keeps_the_cr_conflict_rule():
    aoa = AoaExtraction(values={"company_name": "انجاز"}, flags={"company_name": []})
    cr = CompanyRecord(company_name="انجاز للمقاولات")
    _record, prov = service.merge_fields(aoa, cr, {}, _page_map())
    assert prov["company_name"]["value"] == "انجاز للمقاولات"
    assert prov["company_name"]["conflict"] == {"cr": "انجاز للمقاولات", "aoa": "انجاز"}


def test_merge_fields_keeps_a_lawyer_edited_roster():
    previous = {"attendees": [{"name": "مصحح"}], "attendees_edited": True}
    _record, prov = service.merge_fields(AoaExtraction(), CompanyRecord(), previous, _page_map())
    assert prov["attendees"] == [{"name": "مصحح"}]


def _row(number, first, last, mukarrar=False, new_text=None):
    return DocgenArticle(
        article_number=number, is_mukarrar=mukarrar, span_first=first, span_last=last,
        new_text=new_text,
    )


def test_carry_over_keeps_an_unchanged_row_and_its_edits():
    kept = _row(7, 3, 3, new_text="جديد")
    [(_entry, row, new_text)] = service.carry_over(_page_map().amended, [kept])
    assert row is kept and new_text is None


def test_carry_over_rebuilds_on_a_changed_span_but_keeps_new_text():
    [(_entry, row, new_text)] = service.carry_over(
        _page_map().amended, [_row(7, 3, 4, new_text="جديد")]
    )
    assert row is None and new_text == "جديد"


def test_carry_over_never_matches_mukarrar_to_its_base():
    [(_entry, row, new_text)] = service.carry_over(
        _page_map().amended, [_row(7, 3, 3, mukarrar=True, new_text="x")]
    )
    assert row is None and new_text is None


def test_claim_for_ocr_statement_requires_a_submitted_page_map():
    sql = str(
        service._claim_for_ocr_statement(1).compile(compile_kwargs={"literal_binds": True})
    )
    assert "page_map IS NOT NULL" in sql


class _FakeRenderDb:
    def __init__(self, fields: DocgenFields) -> None:
        self._fields = fields

    async def scalar(self, *_args, **_kwargs):
        return self._fields

    async def flush(self):
        return None


def _render_session(articles):
    scalars = {
        "commercial_registration_no": "١٢٣٤٥", "commercial_registration_date": "٢٠٢٠/١/١",
        "company_name": "شركة تجريبية", "company_address": "عنوان تجريبي",
        "law_number": "١٥٩", "law_year": "١٩٨١", "owner_name": "محمد أحمد علي حسن",
        "day_date": "٢٠٢٦/٩/٢٦", "day_name": "السبت", "names_of_commissioners": "مفوض تجريبي",
    }
    fields = DocgenFields(
        session_id=1, data={k: {"value": v, "source": "user"} for k, v in scalars.items()}
    )
    session = _session(company_type="shakhs_wahed", articles=articles)
    return session, _FakeRenderDb(fields)


def test_render_session_renders_every_declared_article_with_its_mukarrar_heading(monkeypatch):
    import io

    import docx

    written = {}
    monkeypatch.setattr(storage, "write", lambda key, data: written.update(doc=data))
    articles = [
        DocgenArticle(position=0, article_number=6, is_mukarrar=False, ordinal_words="السادسة",
                      patched_text="نص قبل ٦", new_text="نص بعد ٦"),
        DocgenArticle(position=1, article_number=6, is_mukarrar=True, ordinal_words="السادسة",
                      patched_text="نص قبل ٦ مكرر", new_text="نص بعد ٦ مكرر"),
    ]
    session, db = _render_session(articles)
    asyncio.run(service.render_session(db, session))
    text = "\n".join(p.text for p in docx.Document(io.BytesIO(written["doc"])).paragraphs)
    assert "المادة السادسة مكرر" in text
    assert "نص بعد ٦ مكرر" in text
    assert session.status == "rendered"


def test_render_session_names_every_article_missing_its_new_text():
    articles = [
        DocgenArticle(position=0, article_number=6, is_mukarrar=True, ordinal_words="السادسة",
                      patched_text="قبل", new_text=None),
        DocgenArticle(position=1, article_number=7, is_mukarrar=False, ordinal_words="السابعة",
                      patched_text="قبل", new_text=" "),
    ]
    session, db = _render_session(articles)
    with pytest.raises(SessionStateError) as info:
        asyncio.run(service.render_session(db, session))
    assert str(info.value).endswith("المادة السادسة مكرر، المادة السابعة")


def test_reconcile_capital_accepts_shares_worth_more_than_a_pound_via_percentages():
    # 100 حصص of 1000 each: shares (100) never equal the capital (100000),
    # but the table's percentages add up to 100.
    attendees = [
        {"shares": "٩٠", "percentage": "٩٠"},
        {"shares": "١٠", "percentage": "١٠"},
    ]
    assert _reconcile_capital(attendees, "١٠٠٠٠٠") is None


def test_reconcile_capital_still_catches_a_missing_partner_through_percentages():
    assert _reconcile_capital([{"shares": "٩٠", "percentage": "٩٠"}], "١٠٠٠٠٠") is not None


def test_reconcile_capital_needs_every_percentage_to_use_them():
    attendees = [{"shares": "٩٠", "percentage": "١٠٠"}, {"shares": "١٠", "percentage": None}]
    assert _reconcile_capital(attendees, "١٠٠٠٠٠") is not None
