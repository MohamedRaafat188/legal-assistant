import pytest

from legal_assistant.docgen.parsing.commercial_register import CompanyRecord
from legal_assistant.docgen.parsing.signatures import Concept
from legal_assistant.docgen.patching import Replacement
from legal_assistant.docgen.service import (
    SessionStateError,
    _reconcile_capital,
    apply_patches,
    compute_percentages,
    plan_patches,
    validate_source_mode,
)


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
