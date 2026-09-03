import pytest

from legal_assistant.docgen.parsing.commercial_register import CompanyRecord
from legal_assistant.docgen.parsing.signatures import Concept
from legal_assistant.docgen.service import (
    SessionStateError,
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
