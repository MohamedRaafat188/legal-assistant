import pytest

from legal_assistant.docgen.numbering import (
    article_name,
    articles_title,
    ordinal_words,
    parse_article_number,
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
