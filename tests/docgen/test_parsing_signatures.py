from legal_assistant.docgen.parsing.signatures import Concept, classify, states


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


def test_states_matches_the_one_word_raasmal_spelling():
    assert states("رأسمال الشركة مائة ألف جنيه", Concept.CAPITAL)


def test_states_matches_mawtinuha_al_qanuni():
    assert states("يكون موطنها القانوني في القاهرة", Concept.HEAD_OFFICE)


def test_states_rejects_an_unrelated_text():
    assert not states("مدة الشركة خمس وعشرون سنة", Concept.HEAD_OFFICE)
