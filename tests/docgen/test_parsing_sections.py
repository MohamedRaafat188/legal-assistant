import pytest

from legal_assistant.docgen.parsing.sections import (
    Instrument,
    select_target,
    split_instruments,
)


def test_single_instrument_document_yields_one_section(zmm_text):
    sections = split_instruments(zmm_text)
    assert len(sections) == 1
    assert sections[0].instrument is Instrument.ARTICLES_OF_ASSOCIATION
    assert [a.number for a in sections[0].articles] == [4, 5, 6, 7]


def test_number_restart_starts_a_new_section():
    text = (
        "العقد الابتدائى\n"
        "مادة (١)\nاسم الشركة.\n"
        "مادة (٢)\nغرض الشركة.\n"
        "النظام الأساسى\n"
        "مادة (١)\nالتأسيس.\n"
        "مادة (٢)\nالاسم.\n"
        "مادة (٣)\nالغرض.\n"
    )
    sections = split_instruments(text)
    assert [s.instrument for s in sections] == [
        Instrument.PRELIMINARY,
        Instrument.ARTICLES_OF_ASSOCIATION,
    ]
    assert [a.number for a in sections[0].articles] == [1, 2]
    assert [a.number for a in sections[1].articles] == [1, 2, 3]


def test_masahma_namuzag_splits_into_two_instruments(masahma_text):
    sections = split_instruments(masahma_text)
    assert len(sections) >= 2
    assert sections[-1].instrument is Instrument.ARTICLES_OF_ASSOCIATION


def test_masahma_namuzag_first_section_is_preliminary(masahma_text):
    # The real fixture's title is PyMuPDF's lam-alef-transposed "العقد
    # االبتدائي" (doubled alef), which sits at offset ~123 -- well before
    # the first article heading at offset ~1494 -- so it falls inside the
    # first group's title region.
    sections = split_instruments(masahma_text)
    assert sections[0].instrument is Instrument.PRELIMINARY


def test_select_target_picks_the_articles_of_association_for_masahma(masahma_text):
    sections = split_instruments(masahma_text)
    articles, warning = select_target(sections, "masahma")
    assert warning is None
    assert articles is sections[-1].articles


def test_select_target_warns_instead_of_guessing_when_the_series_is_missing():
    sections = split_instruments("العقد الابتدائى\nمادة (١)\nاسم الشركة.\n")
    articles, warning = select_target(sections, "masahma")
    assert articles == []
    assert warning is not None
    assert "النظام الأساسى" in warning


def test_select_target_rejects_an_unknown_company_type(zmm_text):
    sections = split_instruments(zmm_text)
    with pytest.raises(KeyError):
        select_target(sections, "tawsiya_bil_ashum")


def test_empty_text_yields_no_sections():
    assert split_instruments("") == []
