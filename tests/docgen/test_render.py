import pytest

from legal_assistant.docgen.render import (
    ArticleBlock,
    Attendee,
    MissingContextError,
    build_context,
    document_text,
    render_document,
)

SHAKHS_WAHED_SCALARS = {
    "company_name": "شركة النور للتوزيع",
    "law_number": "159",
    "law_year": "1981",
    "commercial_registration_no": "303907",
    "commercial_registration_date": "2021/03/14",
    "day_name": "الأحد",
    "day_date": "2026/09/06",
    "names_of_commissioners": "أحمد كامل",
    "owner_name": "مايكل فوزى",
    "company_address": "٤٠ شارع الهرم، الجيزة",
}

ZMM_SCALARS = {
    "company_name": "شركة إنجاز",
    "law_number": "159",
    "law_year": "1981",
    "commercial_registration_no": "303907",
    "commercial_registration_date": "2021/03/14",
    "day_name": "الأحد",
    "day_date": "2026/09/06",
    "names_of_commissioners": "أحمد كامل",
    "meeting_time": "الحادية عشرة صباحا",
    "meeting_end_time": "الواحدة ظهرا",
    "meeting_place": "١٢ شارع النيل، الجيزة",
    "chairman_name": "مايكل فوزى",
    "attendance_percentage": "100",
    "approval_percentage": "100",
}


def _articles():
    return [
        ArticleBlock("المادة السادسة", "النص القديم للمادة السادسة.", "النص الجديد."),
        ArticleBlock("المادة السابعة", "النص القديم للمادة السابعة.", "نص جديد آخر."),
    ]


def test_build_context_computes_articles_title_from_the_numbers():
    context = build_context(
        "shakhs_wahed",
        scalars=SHAKHS_WAHED_SCALARS,
        articles=_articles(),
        article_numbers=[6, 7],
        attendees=[],
    )
    assert context["articles_title"] == "المواد السادسة والسابعة"
    assert len(context["articles"]) == 2


def test_render_shakhs_wahed_leaves_no_placeholder_residue():
    context = build_context(
        "shakhs_wahed",
        scalars=SHAKHS_WAHED_SCALARS,
        articles=_articles(),
        article_numbers=[6, 7],
        attendees=[],
    )
    text = document_text(render_document("shakhs_wahed", context))
    assert "{{" not in text and "{%" not in text
    assert "شركة النور للتوزيع" in text
    assert "المواد السادسة والسابعة" in text


def test_render_repeats_the_block_once_per_article():
    context = build_context(
        "shakhs_wahed",
        scalars=SHAKHS_WAHED_SCALARS,
        articles=_articles(),
        article_numbers=[6, 7],
        attendees=[],
    )
    text = document_text(render_document("shakhs_wahed", context))
    assert "النص القديم للمادة السادسة." in text
    assert "النص القديم للمادة السابعة." in text
    assert text.count("قبل التعديل") == 2
    assert text.count("بعد التعديل") == 2


def test_render_a_single_article_uses_the_singular_heading():
    context = build_context(
        "shakhs_wahed",
        scalars=SHAKHS_WAHED_SCALARS,
        articles=_articles()[:1],
        article_numbers=[6],
        attendees=[],
    )
    text = document_text(render_document("shakhs_wahed", context))
    assert "المادة السادسة" in text
    assert "المواد" not in text


def test_render_zmm_emits_one_table_row_per_attendee():
    attendees = [
        Attendee("مايكل فوزى", "٩٠", "٩٠"),
        Attendee("مايكل مجدى", "١٠", "١٠"),
    ]
    context = build_context(
        "zmm",
        scalars=ZMM_SCALARS,
        articles=_articles()[:1],
        article_numbers=[6],
        attendees=attendees,
    )
    text = document_text(render_document("zmm", context))
    assert "مايكل فوزى" in text
    assert "مايكل مجدى" in text
    assert "{{" not in text


def test_render_raises_when_a_declared_scalar_is_missing():
    scalars = dict(SHAKHS_WAHED_SCALARS)
    del scalars["owner_name"]
    context = build_context(
        "shakhs_wahed",
        scalars=scalars,
        articles=_articles()[:1],
        article_numbers=[6],
        attendees=[],
    )
    with pytest.raises(MissingContextError) as excinfo:
        render_document("shakhs_wahed", context)
    assert "owner_name" in str(excinfo.value)


def test_render_raises_for_an_unknown_company_type():
    with pytest.raises(KeyError):
        render_document("tawsiya_bil_ashum", {})


def test_build_context_requires_at_least_one_article():
    with pytest.raises(ValueError):
        build_context(
            "shakhs_wahed",
            scalars=SHAKHS_WAHED_SCALARS,
            articles=[],
            article_numbers=[],
            attendees=[],
        )
