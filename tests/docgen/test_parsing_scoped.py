from legal_assistant.docgen.numbering import ArticleRef
from legal_assistant.docgen.pages import Entry, PageMap
from legal_assistant.docgen.parsing.scoped import extract_aoa, scope

PAGES = {
    1: "عقد تأسيس شركة انجاز\nتخضع لأحكام القانون رقم ١٥٩ لسنة ١٩٨١\nالمادة (١)\n"
    "اسم الشركة هو انجاز للمقاولات.",
    2: "المادة (٥)\nالمركز الرئيسي للشركة الكائن في ١٢ شارع النيل.\nالمادة (٦)\n"
    "رأس مال الشركة ١٠٠٠٠٠ جنيه وزعت على الشركاء كالآتى:\nأحمد محمود ٥٠ حصة ٥٠٪\n"
    "سعيد علي ٥٠ حصة ٥٠٪",
    3: "المادة (٧)\nمدة الشركة خمس وعشرون سنة",
    9: "شهادة بنكية: اسم الشركة هو شركة أخرى تماما",
}


def _map(**entries):
    base = {
        "company_name": Entry("company_name", 1, 1, ArticleRef(1)),
        "law_reference": Entry("law_reference", 1, 1, None),
        "company_address": Entry("company_address", 2, 2, ArticleRef(5)),
        "partners": Entry("partners", 2, 2, ArticleRef(6)),
        "issued_capital": Entry("issued_capital", 2, 2, ArticleRef(6)),
    }
    base.update(entries)
    return PageMap(entries=base, amended=[Entry("amended:0", 3, 3, ArticleRef(7))])


def test_every_zmm_value_comes_from_its_own_entry():
    out = extract_aoa(PAGES, _map(), "zmm")
    assert out.values["company_name"] == "انجاز للمقاولات"
    assert (out.values["law_number"], out.values["law_year"]) == ("١٥٩", "١٩٨١")
    assert out.values["company_address"] == "١٢ شارع النيل"
    assert out.values["issued_capital"] == "١٠٠٠٠٠"
    assert [p.name for p in out.parties] == ["أحمد محمود", "سعيد علي"]
    assert not any(out.flags.values())


def test_a_value_outside_the_span_is_never_picked_up():
    # company_name pointed at page 3 (article 7). The bank certificate on page 9
    # also says "اسم الشركة", but page 9 is not in the span.
    entry = Entry("company_name", 3, 3, ArticleRef(7))
    out = extract_aoa(PAGES, _map(company_name=entry), "zmm")
    assert out.values["company_name"] is None
    assert "not_found" in out.flags["company_name"]


def test_article_not_found_in_span():
    entry = Entry("company_address", 3, 3, ArticleRef(5))
    out = extract_aoa(PAGES, _map(company_address=entry), "zmm")
    assert out.flags["company_address"] == ["article_not_found"]


def test_signature_mismatch_warns_but_keeps_the_lawyers_article():
    entry = Entry("company_address", 2, 2, ArticleRef(6))
    out = extract_aoa(PAGES, _map(company_address=entry), "zmm")
    assert "signature_mismatch" in out.flags["company_address"]


def test_empty_preamble_is_article_not_found():
    entry = Entry("law_reference", 2, 2, None)
    out = extract_aoa(PAGES, _map(law_reference=entry), "zmm")
    assert out.flags["law_reference"] == ["article_not_found"]


def test_scope_carries_the_truncation_flag():
    assert scope(PAGES, Entry("amended:0", 3, 3, ArticleRef(7))).truncated is True
