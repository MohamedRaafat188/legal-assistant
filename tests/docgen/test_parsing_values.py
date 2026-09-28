import pathlib

import pytest

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


def test_owner_name_from_a_markdown_pipe_table():
    text = (
        "بيانات مؤسس الشركة:\n"
        "| م | الاسم | الجنسية | إثبات الشخصية |\n"
        "|---|---|---|---|\n"
        "| ١ | محمد أحمد علي حسن | مصري | بطاقة رقم ٢٩٠٠١٠١٠١٠١٠١٠ |\n"
    )
    assert owner_name(text) == "محمد أحمد علي حسن"


def test_owner_name_absent():
    assert owner_name("رأس مال الشركة ١٠٠٠٠٠ جنيه") is None


def test_current_value_is_the_moved_service_helper():
    assert current_value("capital", "رأس المال ١٠٠٠٠٠ جنيه") == "١٠٠٠٠٠"


def test_current_value_reads_an_address_after_al_unwan_al_ati():
    text = (
        "يكون المركز الرئيسي لإدارة الشركة وموطنها القانوني في العنوان الآتي : "
        "١٢ شارع التحرير - الدقي - الجيزة .\nويكون مكان وموقع ممارسة النشاط ..."
    )
    assert current_value("company_address", text) == "١٢ شارع التحرير - الدقي - الجيزة"


def test_owner_name_accepts_a_hamza_spelled_label():
    assert owner_name("بيانات مؤسس الشركة:\nالإسم: محمد أحمد علي حسن\n") == "محمد أحمد علي حسن"


def test_owner_name_from_a_founder_table_with_a_hamza_header():
    text = (
        "بيانات مؤسس الشركة:\n"
        "| م | الإسم | الجنسية |\n"
        "|---|---|---|\n"
        "| ١ | محمد أحمد علي حسن | مصري |\n"
    )
    assert owner_name(text) == "محمد أحمد علي حسن"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The dash before the name and the legal form after it are not the name.
        ("اسم الشركة هو : - النور للتوزيع Nour Trading شركة ذات مسئولية محدودة",
         "النور للتوزيع Nour Trading"),
        ("تسمى الشركة: شركة النور للتوزيع (ش.ذ.م.م).\nالمادة", "النور للتوزيع"),
        ("اسم الشركة هو النور شركة شخص واحد ذات مسئولية محدودة.", "النور"),
        ("اسم الشركة هو: - مصر للتجارة - شركة ذات مسؤولية محدودة", "مصر للتجارة"),
        ("اسم الشركة هو مصر للاستثمار. وتكون", "مصر للاستثمار"),
        # Nothing but the form: keep it rather than return nothing.
        ("اسم الشركة : شركة ذات مسئولية محدودة", "شركة ذات مسئولية محدودة"),
    ],
)
def test_company_name_drops_edge_punctuation_and_legal_form(text, expected):
    value = current_value("company_name", text)
    assert value == expected
    assert value in text  # still a verbatim span: patching replaces it in place
