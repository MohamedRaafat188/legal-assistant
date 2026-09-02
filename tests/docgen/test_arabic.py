from legal_assistant.docgen.arabic import (
    digit_variants,
    normalize_for_match,
    to_arabic_digits,
    to_ascii_digits,
)


def test_to_ascii_digits_converts_arabic_indic():
    assert to_ascii_digits("المادة (٦)") == "المادة (6)"


def test_to_arabic_digits_converts_ascii():
    assert to_arabic_digits("100000 جنيه") == "١٠٠٠٠٠ جنيه"


def test_digit_variants_returns_both_forms_deduped():
    assert digit_variants("100000") == ["100000", "١٠٠٠٠٠"]
    assert digit_variants("القاهرة") == ["القاهرة"]


def test_normalize_for_match_folds_alef_hamza_and_tatweel():
    assert normalize_for_match("المــركز الرئيسـى") == normalize_for_match("المركز الرئيسي")
    assert normalize_for_match("رأس المال") == normalize_for_match("راس المال")


def test_normalize_for_match_folds_teh_marbuta_and_collapses_space():
    assert normalize_for_match("الشركة  التجارية") == normalize_for_match("الشركه التجاريه")


def test_normalize_for_match_strips_diacritics_and_directional_marks():
    assert normalize_for_match("مَادَة‏") == normalize_for_match("مادة")
