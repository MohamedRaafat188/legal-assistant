import pytest

from legal_assistant.docgen.templates.registry import (
    CompanyType,
    TemplateContractError,
    get_template,
    placeholders_in,
    verify_all,
)


def test_every_company_type_has_a_template_file():
    for company_type in CompanyType:
        spec = get_template(company_type.value)
        assert spec.path.exists(), f"missing template for {company_type.value}"
        assert spec.path.suffix == ".docx"


def test_get_template_rejects_an_unknown_company_type():
    with pytest.raises(KeyError):
        get_template("tawsiya_bil_ashum")


def test_verify_all_passes_against_the_shipped_templates():
    verify_all()  # raises TemplateContractError on drift


def test_shakhs_wahed_declares_owner_and_address_but_no_attendees():
    spec = get_template("shakhs_wahed")
    assert "owner_name" in spec.scalar_placeholders
    assert "company_address" in spec.scalar_placeholders
    assert spec.attendee_placeholders == frozenset()
    assert spec.attendee_label is None


def test_zmm_declares_partner_attendees():
    spec = get_template("zmm")
    assert spec.attendee_label == "partner"
    assert "chairman_name" in spec.scalar_placeholders


def test_masahma_declares_shareholder_attendees_and_its_extra_officers():
    spec = get_template("masahma")
    assert spec.attendee_label == "shareholder"
    for name in (
        "secretary_name",
        "vote_collector_name",
        "gafi_representative_name",
        "auditor_name",
        "board_meeting_date",
    ):
        assert name in spec.scalar_placeholders


def test_articles_title_is_shakhs_wahed_only_and_the_article_loop_is_everywhere():
    # Correction 4: only شخص واحد has a document-level line naming the
    # amended article outside the article block. Both محضر templates use
    # `article_name` solely inside the per-article loop, so they have no
    # `articles_title` variable at all.
    for company_type in CompanyType:
        spec = get_template(company_type.value)
        found = placeholders_in(spec.path)
        if company_type == CompanyType.SHAKHS_WAHED:
            assert "articles_title" in found
        else:
            assert "articles_title" not in found
        assert "a.article_original_content" in found
        assert "a.article_new_content" in found


def test_no_literal_old_style_placeholder_survives():
    import re
    import zipfile

    for company_type in CompanyType:
        xml = zipfile.ZipFile(get_template(company_type.value).path).read(
            "word/document.xml"
        ).decode("utf-8")
        assert not re.search(r"(?<!\{)\{[a-z_]+\}", xml)


def test_template_contract_error_names_the_offending_placeholders():
    error = TemplateContractError("zmm", missing={"chairman_name"}, unexpected={"foo"})
    assert "zmm" in str(error)
    assert "chairman_name" in str(error)
    assert "foo" in str(error)
