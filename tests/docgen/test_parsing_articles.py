from legal_assistant.docgen.numbering import ArticleRef
from legal_assistant.docgen.parsing import articles
from legal_assistant.docgen.parsing.articles import (
    LookupStatus,
    find_by_ref,
    preamble_text,
    segment_articles,
)


def test_segments_the_zmm_excerpt_in_order(zmm_text):
    articles = segment_articles(zmm_text)
    assert [a.number for a in articles] == [4, 5, 6, 7]


def test_body_is_verbatim_and_excludes_the_heading(zmm_text):
    articles = segment_articles(zmm_text)
    head_office = next(a for a in articles if a.number == 5)
    assert head_office.body.startswith("المركز الرئيسى")
    assert "المادة (٥)" not in head_office.body
    assert "١٢ شارع النيل، الجيزة" in head_office.body
    # Verbatim means byte-identical to the slice of the source it came from.
    assert head_office.body in zmm_text


def test_offsets_point_back_at_the_source(zmm_text):
    for article in segment_articles(zmm_text):
        assert zmm_text[article.start : article.end].strip().startswith(article.heading.strip())


def test_tolerates_the_no_space_heading_form(shakhs_wahed_text):
    articles = segment_articles(shakhs_wahed_text)
    assert [a.number for a in articles] == [5, 6, 7, 8]
    assert articles[-1].body.startswith("يمثل الشركة")


def test_the_last_article_runs_to_end_of_text(shakhs_wahed_text):
    articles = segment_articles(shakhs_wahed_text)
    assert articles[-1].end == len(shakhs_wahed_text)


def test_preamble_before_the_first_heading_is_dropped(zmm_text):
    articles = segment_articles(zmm_text)
    assert "عقد تأسيس شركة ذات مسئولية محدودة" not in articles[0].body


def test_ascii_digit_headings_are_detected():
    text = "مادة (1)\nاسم الشركة.\nمادة (2)\nغرض الشركة."
    assert [a.number for a in segment_articles(text)] == [1, 2]


def test_text_with_no_headings_returns_empty_list():
    assert segment_articles("لا توجد مواد فى هذا النص على الإطلاق.") == []


def test_masahma_namuzag_yields_a_restarting_sequence(masahma_text):
    numbers = [a.number for a in segment_articles(masahma_text)]
    assert numbers, "the نموذج must yield articles"
    # Two instruments in one file -> the numbering restarts at least once.
    assert any(b <= a for a, b in zip(numbers, numbers[1:], strict=False))


def test_a_skipped_headings_body_merges_into_the_preceding_article(monkeypatch):
    # parse_article_number cannot actually return None through the public
    # regex (its capture group is digits-only), so this drives the branch
    # directly to prove a skipped heading's text is merged, not dropped.
    real_parse = articles.parse_article_number

    def fake_parse(text: str) -> int | None:
        if text == "2":
            return None
        return real_parse(text)

    monkeypatch.setattr(articles, "parse_article_number", fake_parse)

    text = (
        "مادة (1)\n"
        "بند أول.\n"
        "مادة (2)\n"
        "بند مخفى لا يجوز أن يضيع.\n"
        "مادة (3)\n"
        "بند ثالث."
    )

    result = articles.segment_articles(text)

    assert [a.number for a in result] == [1, 3]
    assert "بند مخفى لا يجوز أن يضيع" in result[0].body
    # No character of the input is lost: the kept spans are contiguous and
    # together cover the whole text (there is no preamble here).
    assert result[0].start == 0
    assert result[0].end == result[1].start
    assert result[1].end == len(text)


def test_a_midsentence_cross_reference_does_not_start_a_new_article():
    text = (
        "مادة (1)\n"
        "يجوز للشركة ممارسة أي نشاط آخر وذلك طبقا لأحكام المادة (٣) المنصوص "
        "عليها فى القانون.\n"
        "مادة (2)\n"
        "نص المادة الثانية."
    )

    assert [a.number for a in segment_articles(text)] == [1, 2]


_SPAN = (
    "عقد تأسيس شركة ذات مسئولية محدودة\n"
    "تخضع لأحكام القانون رقم ١٥٩ لسنة ١٩٨١\n"
    "المادة (٦)\n"
    "رأس مال الشركة ١٠٠٠٠٠ جنيه.\n"
    "المادة (٦) مكرر\n"
    "نص المادة السادسة مكرر.\n"
    "المادة (٧)\n"
    "مدة الشركة خمس وعشرون سنة"
)


def test_segment_articles_reads_mukarrar_in_both_heading_positions():
    arts = segment_articles("المادة (٦) مكرر\nأ\nمادة (٧ مكرر)\nب")
    assert [(a.number, a.mukarrar) for a in arts] == [(6, True), (7, True)]


def test_find_by_ref_never_lets_mukarrar_satisfy_the_base_article():
    arts = segment_articles(_SPAN)
    base = find_by_ref(arts, ArticleRef(6))
    bis = find_by_ref(arts, ArticleRef(6, True))
    assert base.status is LookupStatus.found and "رأس مال" in base.article.body
    assert bis.status is LookupStatus.found and "مكرر" in bis.article.body


def test_find_by_ref_reports_not_found():
    assert find_by_ref(segment_articles(_SPAN), ArticleRef(9)).status is LookupStatus.not_found


def test_find_by_ref_reports_ambiguous_on_a_repeated_number():
    text = "مادة (٦)\nالعقد الابتدائي\nالنظام الأساسي\nمادة (٦)\nنص آخر"
    lookup = find_by_ref(segment_articles(text), ArticleRef(6))
    assert lookup.status is LookupStatus.ambiguous
    assert lookup.article is None


def test_find_by_ref_flags_the_last_article_in_the_span_as_possibly_truncated():
    arts = segment_articles(_SPAN)
    assert find_by_ref(arts, ArticleRef(7)).truncated is True
    assert find_by_ref(arts, ArticleRef(6)).truncated is False


def test_preamble_text_is_everything_before_the_first_heading():
    assert preamble_text(_SPAN).endswith("لسنة ١٩٨١")
    assert preamble_text("المادة (١)\nنص") == ""
    assert preamble_text("بلا عناوين") == "بلا عناوين"
