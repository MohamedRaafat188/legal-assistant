import pathlib

from legal_assistant.docgen.parsing.values import current_value, law_reference, owner_name

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures" / "docgen"


def test_law_reference_across_a_line_break_and_digit_sets():
    assert law_reference("الصادر بالقانون رقم159\n لسنة1981") == ("159", "1981")
    assert law_reference("تخضع لأحكام القانون رقم ١٥٩ لسنة ١٩٨١") == ("١٥٩", "١٩٨١")


def test_law_reference_absent():
    assert law_reference("مدة الشركة خمس وعشرون سنة") is None


def test_owner_name_from_the_synthetic_founder_table():
    text = (FIXTURES / "owner_table_synthetic.txt").read_text(encoding="utf-8")
    assert owner_name(text) == "محمد أحمد علي حسن"


def test_owner_name_absent():
    assert owner_name("رأس مال الشركة ١٠٠٠٠٠ جنيه") is None


def test_current_value_is_the_moved_service_helper():
    assert current_value("capital", "رأس المال ١٠٠٠٠٠ جنيه") == "١٠٠٠٠٠"
