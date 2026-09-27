import pytest

from legal_assistant.docgen.numbering import ArticleRef
from legal_assistant.docgen.pages import (
    PREAMBLE,
    PageMap,
    PageMapError,
    required_entries,
    validate_page_map,
)


def _zmm_map(**overrides):
    raw = {
        "entries": {
            "company_name": {"from": 1, "to": 1, "article": "2"},
            "law_reference": {"from": 1, "to": 1, "article": PREAMBLE},
            "company_address": {"from": 2, "to": 2, "article": "5"},
            "partners": {"from": 3, "to": 4, "article": "٦"},
            "issued_capital": {"from": 3, "to": 3, "article": "6"},
        },
        "amended": [
            {"from": 2, "to": 2, "article": "5"},
            {"from": 4, "to": 5, "article": "٧ مكرر"},
        ],
    }
    raw["entries"].update(overrides)
    return raw


def _problems(raw, company_type="zmm", page_count=12):
    with pytest.raises(PageMapError) as info:
        validate_page_map(raw, company_type, page_count)
    return " | ".join(info.value.problems)


def test_required_entries_per_company_type():
    assert set(required_entries("zmm")) == {
        "company_name", "law_reference", "company_address", "partners", "issued_capital",
    }
    assert "owner_name" in required_entries("shakhs_wahed")
    assert {"shareholders", "issued_capital"} <= set(required_entries("masahma"))


def test_a_valid_map_parses_spans_articles_and_preamble():
    page_map = validate_page_map(_zmm_map(), "zmm", 12)
    assert page_map.entries["law_reference"].article is None
    assert page_map.entries["partners"].article == ArticleRef(6)
    assert page_map.entries["partners"].pages == [3, 4]
    assert [e.article for e in page_map.amended] == [ArticleRef(5), ArticleRef(7, True)]


def test_ocr_pages_are_the_deduplicated_union_of_spans():
    assert validate_page_map(_zmm_map(), "zmm", 12).ocr_pages() == [1, 2, 3, 4, 5]


def test_json_round_trip():
    page_map = validate_page_map(_zmm_map(), "zmm", 12)
    assert PageMap.from_json(page_map.to_json()) == page_map


def test_missing_required_entry():
    raw = _zmm_map()
    del raw["entries"]["partners"]
    assert "partners" in _problems(raw)


def test_foreign_entry_is_rejected():
    assert "owner_name" in _problems(_zmm_map(owner_name={"from": 1, "to": 1, "article": "4"}))


def test_span_out_of_range_or_reversed():
    assert "12" in _problems(_zmm_map(company_name={"from": 11, "to": 13, "article": "2"}))
    assert _problems(_zmm_map(company_name={"from": 3, "to": 2, "article": "2"}))
    assert _problems(_zmm_map(company_name={"from": 0, "to": 1, "article": "2"}))


def test_preamble_only_where_allowed():
    assert "company_address" in _problems(
        _zmm_map(company_address={"from": 2, "to": 2, "article": PREAMBLE})
    )


def test_preamble_is_matched_through_spelling_variants():
    raw = _zmm_map(law_reference={"from": 1, "to": 1, "article": "  التمهيد "})
    assert validate_page_map(raw, "zmm", 12).entries["law_reference"].article is None


def test_missing_or_malformed_article():
    assert _problems(_zmm_map(company_name={"from": 1, "to": 1, "article": ""}))
    assert _problems(_zmm_map(company_name={"from": 1, "to": 1, "article": "6/7"}))


def test_amended_rules():
    raw = _zmm_map()
    raw["amended"] = []
    assert _problems(raw)
    raw["amended"] = [{"from": 1, "to": 1, "article": PREAMBLE}]
    assert _problems(raw)
    raw["amended"] = [{"from": 1, "to": 1, "article": "6"}, {"from": 2, "to": 2, "article": "٦"}]
    assert "6" in _problems(raw)


def test_every_problem_is_reported_at_once():
    raw = _zmm_map(company_name={"from": 3, "to": 2, "article": ""})
    raw["amended"] = []
    with pytest.raises(PageMapError) as info:
        validate_page_map(raw, "zmm", 12)
    assert len(info.value.problems) >= 3
