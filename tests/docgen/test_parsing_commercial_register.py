import pytest

from legal_assistant.docgen.parsing.commercial_register import (
    CR_FIELD_SCHEMA,
    CompanyRecord,
    Party,
    parse_party_table,
    record_from_payload,
    split_capital,
    table_cells,
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
        "owner_name",
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


def test_parse_party_table_skips_a_blank_line_inside_the_roster():
    # A stray blank line is ordinary in ragged OCR output; it must not
    # silently truncate the roster.
    text = (
        "وزعت على الشركاء كالآتى:\n"
        "مايكل فوزى ٩٠ حصة ٩٠٪\n"
        "\n"
        "مايكل مجدى ١٠ حصة ١٠٪\n"
    )
    parties = parse_party_table(text)
    assert [p.name for p in parties] == ["مايكل فوزى", "مايكل مجدى"]


def test_parse_party_table_skips_a_whitespace_only_line_inside_the_roster():
    text = (
        "وزعت على الشركاء كالآتى:\n"
        "مايكل فوزى ٩٠ حصة ٩٠٪\n"
        "   \t  \n"
        "مايكل مجدى ١٠ حصة ١٠٪\n"
    )
    parties = parse_party_table(text)
    assert [p.name for p in parties] == ["مايكل فوزى", "مايكل مجدى"]


def test_party_table_lead_in_ala_al_wajh_al_ati():
    text = "وزع رأس المال على الشركاء على الوجه الآتي:\nأحمد محمود ٥٠ حصة ٥٠٪"
    assert [p.name for p in parse_party_table(text)] == ["أحمد محمود"]


def test_record_from_payload_reads_owner_name():
    assert record_from_payload({"owner_name": " محمد "}).owner_name == "محمد"


# Invented names and figures; the layout mirrors the ذ.م.م share table seen in
# the scans: a two-row header, a «الاسم والجنسية» column and a total row.
_PIPE_SHARE_TABLE = """وقد تم توزيع هذه الحصص بين الشركاء على الوجه الآتي :
| م | الاسم والجنسية | عدد الحصص | القيمة ب جنيه مصري | نسبة المشاركة | عملة الوفاء |
|---|---|---|---|---|---|
|   |   | نقدي |   |   |   |
| ١ | أحمد محمود سالم / مصر | ٩٠ | ٩٠٠٠٠ | ٩٠ | جنيه مصري |
| ٢ | سارة علي حسن / مصر | ١٠ | ١٠٠٠٠ | ١٠ | جنيه مصري |
|   | الإجمالي | ١٠٠ | ١٠٠٠٠٠ | %١٠٠ |   |

وتبلغ نسبة المشاركة المصرية ١٠٠%"""


def test_parse_party_table_reads_a_markdown_share_table_by_header():
    assert parse_party_table(_PIPE_SHARE_TABLE) == [
        Party(name="أحمد محمود سالم", shares="٩٠", percentage="٩٠"),
        Party(name="سارة علي حسن", shares="١٠", percentage="١٠"),
    ]


def test_parse_party_table_reads_a_wide_spaced_share_table():
    text = (
        "وزعت الحصص على الشركاء كالآتى:\n"
        "م  الاسم  عدد الحصص  نسبة المشاركة\n"
        "١  أحمد محمود سالم  ٦٠  ٦٠\n"
        "٢  سارة علي حسن  ٤٠  ٤٠\n"
    )
    assert [(p.name, p.shares) for p in parse_party_table(text)] == [
        ("أحمد محمود سالم", "٦٠"),
        ("سارة علي حسن", "٤٠"),
    ]


def test_the_ocr_prompt_asks_for_the_table_format_this_parser_reads():
    # parse_party_table and the founder-table reader rely on OCR writing
    # tables as markdown pipe rows; the transcription prompt must ask for it.
    from legal_assistant.docgen.ocr.gemini import _EXTRACT_PROMPT

    assert "|---|" in _EXTRACT_PROMPT
    lines = [line.strip() for line in _EXTRACT_PROMPT.splitlines()]
    example = [line for line in lines if line.startswith("|")]
    assert "الاسم" in table_cells(example[0])


def test_parse_party_table_reads_a_pipe_table_without_any_lead_in():
    # The lead-in sentence can wrap or be worded differently; a markdown
    # table with a «الاسم» header is enough inside the scoped partners article.
    text = (
        "حدد رأس مال الشركة بمبلغ ١٠٠٠٠٠ جنيه مصري وقد تم توزيع هذه الحصص بين الشركاء\n"
        "على الوجه الآتي :\n"
        "| م | الاسم وجنسيته | عدد الحصص نقدي | نسبة المشاركة |\n"
        "|---|---|---|---|\n"
        "| ١ | أحمد محمود سالم / مصر | ٩٠ | ٩٠ |\n"
        "| ٢ | سارة علي حسن / مصر | ١٠ | ١٠ |\n"
        "|  | الإجمالي | ١٠٠ | %١٠٠ |\n"
    )
    assert parse_party_table(text) == [
        Party(name="أحمد محمود سالم", shares="٩٠", percentage="٩٠"),
        Party(name="سارة علي حسن", shares="١٠", percentage="١٠"),
    ]
