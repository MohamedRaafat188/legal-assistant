from legal_assistant.docgen.parsing.articles import segment_articles
from legal_assistant.docgen.parsing.signatures import Concept, classify, find_article, states


def test_finds_head_office_at_article_5_in_the_zmm_sample(zmm_text):
    article = find_article(segment_articles(zmm_text), Concept.HEAD_OFFICE)
    assert article is not None
    assert article.number == 5


def test_finds_head_office_at_article_6_in_the_shakhs_wahed_sample(shakhs_wahed_text):
    article = find_article(segment_articles(shakhs_wahed_text), Concept.HEAD_OFFICE)
    assert article is not None
    assert article.number == 6


def test_finds_capital_at_the_right_number_in_each_sample(zmm_text, shakhs_wahed_text):
    assert find_article(segment_articles(zmm_text), Concept.CAPITAL).number == 6
    assert find_article(segment_articles(shakhs_wahed_text), Concept.CAPITAL).number == 7


def test_classify_ignores_hamza_and_ya_spelling_variants():
    from legal_assistant.docgen.parsing.articles import ExtractedArticle

    for body in (
        "المركز الرئيسى للشركة الكائن فى الجيزة.",
        "المركز الرئيسي للشركة الكائن في الجيزة.",
        "المــركز الرئيسـى للشركة.",
    ):
        article = ExtractedArticle(5, "المادة (٥)", body, 0, len(body))
        assert classify(article) is Concept.HEAD_OFFICE


def test_classify_returns_none_for_an_unrecognised_article():
    from legal_assistant.docgen.parsing.articles import ExtractedArticle

    body = "تسرى على هذه الشركة أحكام القانون ولائحته التنفيذية."
    assert classify(ExtractedArticle(20, "المادة (٢٠)", body, 0, len(body))) is None


def test_find_article_returns_none_when_absent():
    assert find_article([], Concept.CAPITAL) is None


def test_find_article_prefers_the_earliest_match(zmm_text):
    articles = segment_articles(zmm_text)
    duration = find_article(articles, Concept.DURATION)
    assert duration is not None and duration.number == 7


def test_states_matches_the_one_word_raasmal_spelling():
    assert states("رأسمال الشركة مائة ألف جنيه", Concept.CAPITAL)


def test_states_matches_mawtinuha_al_qanuni():
    assert states("يكون موطنها القانوني في القاهرة", Concept.HEAD_OFFICE)


def test_states_rejects_an_unrelated_text():
    assert not states("مدة الشركة خمس وعشرون سنة", Concept.HEAD_OFFICE)
