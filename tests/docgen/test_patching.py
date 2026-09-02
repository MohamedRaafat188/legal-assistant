from legal_assistant.docgen.patching import PatchResult, Replacement, patch_article

ADDRESS_ARTICLE = "المركز الرئيسى للشركة الكائن فى ١٢ شارع النيل، الجيزة، جمهورية مصر العربية."


def test_replaces_a_verbatim_span_and_records_the_op():
    result = patch_article(
        ADDRESS_ARTICLE,
        [Replacement("company_address", "١٢ شارع النيل، الجيزة", "٥ شارع الهرم، الجيزة", "cr")],
    )
    assert isinstance(result, PatchResult)
    assert "٥ شارع الهرم، الجيزة" in result.text
    assert "١٢ شارع النيل" not in result.text
    assert len(result.ops) == 1
    op = result.ops[0]
    assert op.field == "company_address"
    assert op.source == "cr"
    assert op.old == "١٢ شارع النيل، الجيزة"
    assert op.new == "٥ شارع الهرم، الجيزة"
    assert ADDRESS_ARTICLE[op.start : op.end] == op.old
    assert result.needs_review is False


def test_missing_old_value_changes_nothing_and_flags_review():
    result = patch_article(
        ADDRESS_ARTICLE,
        [Replacement("company_address", "٩٩ شارع غير موجود", "٥ شارع الهرم", "cr")],
    )
    assert result.text == ADDRESS_ARTICLE
    assert result.ops == []
    assert result.needs_review is True
    assert len(result.notes) == 1
    assert "company_address" in result.notes[0]
    assert "٩٩ شارع غير موجود" in result.notes[0]


def test_ambiguous_old_value_changes_nothing_and_flags_review():
    text = "رأس المال ١٠٠٠٠٠ جنيه، والمدفوع منه ١٠٠٠٠٠ جنيه."
    result = patch_article(text, [Replacement("capital", "١٠٠٠٠٠", "٢٠٠٠٠٠", "cr")])
    assert result.text == text
    assert result.ops == []
    assert result.needs_review is True
    assert "أكثر من موضع" in result.notes[0] or "ambiguous" in result.notes[0]


def test_matches_across_digit_sets_without_rewriting_anything_else():
    text = "رأس مال الشركة ١٠٠٠٠٠ جنيه مصرى."
    # The سجل تجاري gives the figure in ASCII digits; the عقد uses Arabic-Indic.
    result = patch_article(text, [Replacement("capital", "100000", "200000", "cr")])
    assert result.text == "رأس مال الشركة 200000 جنيه مصرى."
    assert result.ops[0].old == "١٠٠٠٠٠"
    assert result.needs_review is False


def test_identical_old_and_new_is_a_no_op_without_review_flag():
    result = patch_article(
        ADDRESS_ARTICLE, [Replacement("company_address", "الجيزة", "الجيزة", "cr")]
    )
    assert result.text == ADDRESS_ARTICLE
    assert result.ops == []
    assert result.needs_review is False
    assert result.notes == []


def test_empty_old_or_new_is_skipped_silently():
    for rep in (
        Replacement("company_address", "", "الجيزة", "cr"),
        Replacement("company_address", "الجيزة", "", "cr"),
    ):
        result = patch_article(ADDRESS_ARTICLE, [rep])
        assert result.text == ADDRESS_ARTICLE
        assert result.ops == []
        assert result.needs_review is False


def test_multiple_replacements_keep_spans_valid():
    text = "شركة النور الكائنة فى الجيزة."
    result = patch_article(
        text,
        [
            Replacement("company_name", "النور", "الفجر", "cr"),
            Replacement("company_address", "الجيزة", "القاهرة الجديدة", "cr"),
        ],
    )
    assert result.text == "شركة الفجر الكائنة فى القاهرة الجديدة."
    assert len(result.ops) == 2
    # Offsets are reported against the ORIGINAL text, so the caller can render
    # a diff against what OCR produced.
    assert all(text[op.start : op.end] == op.old for op in result.ops)


def test_overlapping_replacements_apply_the_first_and_flag_the_second():
    text = "المركز الرئيسى فى القاهرة الجديدة."
    result = patch_article(
        text,
        [
            Replacement("a", "القاهرة الجديدة", "الجيزة", "cr"),
            Replacement("b", "القاهرة", "الاسكندرية", "cr"),
        ],
    )
    assert result.text == "المركز الرئيسى فى الجيزة."
    assert [op.field for op in result.ops] == ["a"]
    assert result.needs_review is True
    assert "b" in result.notes[0]


def test_no_replacements_returns_the_text_unchanged():
    result = patch_article(ADDRESS_ARTICLE, [])
    assert result.text == ADDRESS_ARTICLE
    assert result.ops == []
    assert result.needs_review is False
