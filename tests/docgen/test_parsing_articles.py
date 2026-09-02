from legal_assistant.docgen.parsing import articles
from legal_assistant.docgen.parsing.articles import segment_articles


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
