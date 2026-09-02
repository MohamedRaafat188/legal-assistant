import pytest

from legal_assistant.docgen.parsing.commercial_register import (
    CR_FIELD_SCHEMA,
    CompanyRecord,
    Party,
    parse_party_table,
    record_from_payload,
    split_capital,
)

PAYLOAD = {
    "commercial_registration_no": "303907",
    "commercial_registration_date": "2021/03/14",
    "company_name": "أوربت ديچيتال جروب",
    "law_number": "159",
    "law_year": "1981",
    "company_address": "12 شارع النيل، الجيزة",
    "capital": "200000",
    "parties": [],
}


def test_record_from_payload_maps_every_scalar():
    record = record_from_payload(PAYLOAD)
    assert isinstance(record, CompanyRecord)
    assert record.commercial_registration_no == "303907"
    assert record.commercial_registration_date == "2021/03/14"
    assert record.company_name == "أوربت ديچيتال جروب"
    assert record.law_number == "159"
    assert record.law_year == "1981"
    assert record.company_address == "12 شارع النيل، الجيزة"
    assert record.capital == "200000"
    assert record.parties == []


def test_record_from_payload_tolerates_missing_keys():
    record = record_from_payload({"company_name": "شركة النور"})
    assert record.company_name == "شركة النور"
    assert record.commercial_registration_no is None
    assert record.capital is None
    assert record.parties == []


def test_record_from_payload_blanks_are_none_not_empty_strings():
    record = record_from_payload({"company_name": "  ", "capital": ""})
    assert record.company_name is None
    assert record.capital is None


def test_record_from_payload_builds_parties():
    record = record_from_payload(
        {"parties": [{"name": "مايكل فوزى", "shares": "90", "percentage": "90"}]}
    )
    assert record.parties == [Party(name="مايكل فوزى", shares="90", percentage="90")]


def test_record_from_payload_drops_nameless_parties():
    record = record_from_payload({"parties": [{"name": "", "shares": "90"}]})
    assert record.parties == []


def test_record_from_payload_rejects_a_non_dict():
    with pytest.raises(TypeError):
        record_from_payload(["not", "a", "payload"])


def test_schema_covers_every_record_field():
    properties = set(CR_FIELD_SCHEMA["properties"])
    assert properties == {
        "commercial_registration_no",
        "commercial_registration_date",
        "company_name",
        "law_number",
        "law_year",
        "company_address",
        "capital",
        "issued_capital",
        "parties",
    }
    assert CR_FIELD_SCHEMA["type"] == "object"


def test_parse_party_table_reads_the_zmm_share_rows():
    text = (
        "رأس مال الشركة ١٠٠٠٠٠ جنيه مصرى موزع على ١٠٠ حصة، وزعت على الشركاء كالآتى:\n"
        "مايكل فوزى ٩٠ حصة ٩٠٪\n"
        "مايكل مجدى ١٠ حصة ١٠٪\n"
    )
    assert parse_party_table(text) == [
        Party(name="مايكل فوزى", shares="٩٠", percentage="٩٠"),
        Party(name="مايكل مجدى", shares="١٠", percentage="١٠"),
    ]


def test_parse_party_table_returns_empty_when_there_is_no_table():
    # The blank GAFI مساهمة نموذج's founders table has no share column at all.
    text = "رأس المال المرخص به ١٠٠٠٠٠٠ جنيه ورأس المال المصدر ٢٥٠٠٠٠ جنيه."
    assert parse_party_table(text) == []


def test_split_capital_separates_authorized_from_issued():
    text = (
        "رأس المال المرخص به ١٠٠٠٠٠٠ جنيه مصرى، "
        "ورأس المال المصدر ٢٥٠٠٠٠ جنيه مصرى."
    )
    assert split_capital(text) == ("١٠٠٠٠٠٠", "٢٥٠٠٠٠")


def test_split_capital_of_a_single_figure_article_reports_it_as_issued():
    # ذ.م.م and شخص واحد state one capital figure, which IS the issued capital.
    assert split_capital("رأس مال الشركة ١٠٠٠٠٠ جنيه مصرى.") == (None, "١٠٠٠٠٠")


def test_split_capital_of_an_unrelated_article_is_all_none():
    assert split_capital("مدة الشركة خمس وعشرون سنة.") == (None, None)


def test_record_from_payload_carries_issued_capital():
    record = record_from_payload({"capital": "200000", "issued_capital": "150000"})
    assert record.capital == "200000"
    assert record.issued_capital == "150000"


def test_parse_party_table_never_invents_a_split():
    text = "وزعت على الشركاء كالآتى:\nمايكل فوزى\nمايكل مجدى\n"
    parties = parse_party_table(text)
    assert [p.name for p in parties] == ["مايكل فوزى", "مايكل مجدى"]
    assert all(p.shares is None and p.percentage is None for p in parties)


def test_parse_party_table_keeps_a_row_missing_its_percentage():
    # Shares stated, percentage not -- degrade to None, don't drop the partner
    # and don't derive the missing half.
    text = "وزعت على الشركاء كالآتى:\nمايكل فوزى ٩٠ حصة\n"
    parties = parse_party_table(text)
    assert parties == [Party(name="مايكل فوزى", shares="٩٠", percentage=None)]


def test_parse_party_table_keeps_a_row_missing_its_share_count():
    # Percentage stated, share count not -- mirror case, same rule.
    text = "وزعت على الشركاء كالآتى:\nمايكل فوزى ٩٠٪\n"
    parties = parse_party_table(text)
    assert parties == [Party(name="مايكل فوزى", shares=None, percentage="٩٠")]


def test_parse_party_table_stops_at_trailing_prose():
    text = (
        "وزعت على الشركاء كالآتى:\n"
        "مايكل فوزى ٩٠ حصة ٩٠٪\n"
        "مايكل مجدى ١٠ حصة ١٠٪\n"
        "ويلتزم كل شريك بنسبة حصته فى الأرباح والخسائر.\n"
    )
    parties = parse_party_table(text)
    assert [p.name for p in parties] == ["مايكل فوزى", "مايكل مجدى"]
