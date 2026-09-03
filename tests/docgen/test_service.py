import asyncio
import datetime

import pytest

from legal_assistant.docgen.models import DocgenSession
from legal_assistant.docgen.parsing.commercial_register import CompanyRecord
from legal_assistant.docgen.parsing.signatures import Concept
from legal_assistant.docgen.patching import Replacement
from legal_assistant.docgen.service import (
    SessionNotFoundError,
    SessionStateError,
    _claim_for_ocr_statement,
    _reconcile_capital,
    apply_patches,
    compute_percentages,
    get_session,
    plan_patches,
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


def _days_ago(n: int) -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=n)


def _days_from_now(n: int) -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=n)


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
