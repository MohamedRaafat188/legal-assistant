import pytest

from legal_assistant.docgen.numbering import (
    ArticleRef,
    article_name,
    articles_title,
    ordinal_words,
    parse_article_number,
    parse_article_ref,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, "الأولى"),
        (2, "الثانية"),
        (3, "الثالثة"),
        (6, "السادسة"),
        (10, "العاشرة"),
        (11, "الحادية عشرة"),
        (12, "الثانية عشرة"),
        (19, "التاسعة عشرة"),
        (20, "العشرون"),
        (21, "الحادية والعشرون"),
        (35, "الخامسة والثلاثون"),
        (40, "الأربعون"),
        (65, "الخامسة والستون"),
        (99, "التاسعة والتسعون"),
    ],
)
def test_ordinal_words(n, expected):
    assert ordinal_words(n) == expected


def test_ordinal_words_rejects_out_of_range():
    with pytest.raises(ValueError):
        ordinal_words(0)
    with pytest.raises(ValueError):
        ordinal_words(100)


def test_article_name():
    assert article_name(6) == "المادة السادسة"


def test_articles_title_single_uses_singular_noun():
    assert articles_title([6]) == "المادة السادسة"


def test_articles_title_multiple_uses_plural_and_waw():
    assert articles_title([6, 7, 8]) == "المواد السادسة والسابعة والثامنة"


def test_articles_title_sorts_and_dedupes():
    assert articles_title([8, 6, 6]) == "المواد السادسة والثامنة"


def test_articles_title_empty_raises():
    with pytest.raises(ValueError):
        articles_title([])


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("المادة (٦)", 6),
        ("مادة (1)", 1),
        ("(مادة13)", 13),
        ("المادة  ٢١ :", 21),
        ("لا يوجد رقم", None),
    ],
)
def test_parse_article_number(heading, expected):
    assert parse_article_number(heading) == expected


def test_parse_article_ref_accepts_both_digit_sets():
    assert parse_article_ref("6") == ArticleRef(6)
    assert parse_article_ref("٦") == ArticleRef(6)


def test_parse_article_ref_keeps_mukarrar_distinct():
    assert parse_article_ref("٦ مكرر") == ArticleRef(6, mukarrar=True)
    assert parse_article_ref("6مكرر") == ArticleRef(6, mukarrar=True)
    assert parse_article_ref("٦ مكرراً") == ArticleRef(6, mukarrar=True)
    assert ArticleRef(6, mukarrar=True) != ArticleRef(6)


def test_parse_article_ref_rejects_junk():
    for bad in ("", "abc", "6/7", "المادة 6", "0", "1000", "٦ ب"):
        assert parse_article_ref(bad) is None, bad


def test_article_name_appends_mukarrar():
    assert article_name(6, mukarrar=True) == "المادة السادسة مكرر"


def test_articles_title_orders_mukarrar_after_its_base_article():
    title = articles_title([ArticleRef(7), ArticleRef(6, True), ArticleRef(6)])
    assert title == "المواد السادسة والسادسة مكرر والسابعة"


def test_articles_title_still_accepts_plain_ints():
    assert articles_title([6, 7]) == "المواد السادسة والسابعة"
