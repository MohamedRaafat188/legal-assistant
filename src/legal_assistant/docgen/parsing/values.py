"""Pull a single value out of an already-scoped span of عقد text.

Every function returns a VERBATIM substring (digits untouched) or None. None is
a flag, never a guess: the caller marks the field `not_found`.
"""

from __future__ import annotations

import re

from legal_assistant.docgen.parsing.commercial_register import (
    name_column,
    parse_party_table,
    table_cells,
)

# "بالقانون رقم159\n لسنة1981": OCR splits it across lines and drops spaces.
_LAW = re.compile(r"القانون\s*(?:رقم)?\s*([0-9٠-٩]{1,4})\s*(?:لسنة|لسنه)\s*([0-9٠-٩]{4})")

# «الاسم: ...» / «اسم المؤسس: ...» label lines in a founder table.
_NAME_LABEL = re.compile(
    r"^\s*(?:ال[اإ]سم|اسم المؤسس|اسم مؤسس الشركة|اسم مالك الشركة)\s*[:：/\-]?\s*"
    r"(?P<name>[^\d٠-٩:\n]{3,}?)\s*$",
    re.MULTILINE,
)

# «بيانات مؤسس الشركة» founder table: a heading, a header row naming the
# columns, then one data row in the same column order. The real نور عقد uses
# this shape (م / الاسم / الجنسية / تاريخ الميلاد / إثبات الشخصية / الإقامة)
# rather than an «الاسم:» label line, so the name is read by column position
# instead of a fixed offset -- the column order is not assumed.
_FOUNDER_TABLE_HEADING = re.compile(r"بيانات\s+مؤسس\s+الشركة\s*[:：]?\s*$", re.MULTILINE)


def _founder_table_name(text: str) -> str | None:
    heading = _FOUNDER_TABLE_HEADING.search(text)
    if not heading:
        return None
    # Drop blank lines and |---| separator rows; OCR may render the table with
    # wide spacing, tabs or markdown pipes (see `table_cells`).
    lines = [
        line for line in text[heading.end() :].splitlines() if line.strip(" \t|:-–—_=+")
    ]
    if len(lines) < 2:
        return None
    header_cells = table_cells(lines[0])
    data_cells = table_cells(lines[1])
    name_index = name_column(header_cells)
    if name_index is None or name_index >= len(data_cells):
        return None
    return data_cells[name_index].strip() or None


def law_reference(text: str) -> tuple[str, str] | None:
    match = _LAW.search(text)
    return (match.group(1), match.group(2)) if match else None


def owner_name(text: str) -> str | None:
    """The founder's name from the «بيانات مؤسس الشركة» table."""
    labelled = _NAME_LABEL.search(text)
    if labelled:
        return labelled.group("name").strip()
    table_name = _founder_table_name(text)
    if table_name:
        return table_name
    parties = parse_party_table(text)
    return parties[0].name if parties else None


def current_value(field_name: str, article_text: str) -> str:
    """The article's own rendering of a field, as a verbatim span.

    Returns "" when it cannot be isolated, which makes the replacement a
    silent no-op rather than a guess.
    """
    if field_name == "capital":
        match = re.search(r"[0-9٠-٩][0-9٠-٩,.]*", article_text)
        return match.group(0) if match else ""
    if field_name == "company_address":
        match = re.search(
            r"(?:الكائن|الكائنة|مقرها)\s*(?:فى|في|ب)?\s*(?P<v>[^.\n]+)", article_text
        ) or re.search(
            # «... وموطنها القانوني في العنوان الآتي : <address> .»
            r"(?:فى|في)\s*العنوان\s*(?:الآت[يى]|الات[يى]|التال[يى])?\s*[:：]?\s*(?P<v>[^.\n]+)",
            article_text,
        )
        return match.group("v").strip() if match else ""
    if field_name == "company_name":
        match = re.search(
            r"(?:اسم الشركة|تسمى الشركة)\s*(?:هو|:)?\s*"
            # A «.» ends the name, except inside «ش.ذ.م.م».
            r"(?P<v>(?:[^.\n]|\.(?=\s*[ذم]\s*\.|\s*م\s*\)?))+)",
            article_text,
        )
        return _bare_company_name(match.group("v")) if match else ""
    return ""


# «اسم الشركة هو : - النور ... شركة ذات مسئولية محدودة»: the templates write
# «شركة» and the company's legal form themselves, so the name is read without
# the dash the عقد puts before it, a leading «شركة», and the form after it.
_NAME_EDGE = re.compile(r"^[\s:：\-–—ـ/«\"]+|[\s.،,:：\-–—ـ/»\"]+$")
_LEGAL_FORM = re.compile(
    r"[\s\-–—،,(]*(?:شركة\s+)?(?:"
    r"(?:ذات|ذ)\s+(?:ال)?مس[ئؤ]ولي[ةه]\s+(?:ال)?محدود[ةه]"
    r"|ش\s*\.\s*ذ\s*\.\s*م\s*\.\s*م\s*\.?"
    r"|ذ\s*\.\s*م\s*\.\s*م\s*\.?"
    r"|شخص\s+واحد"
    r")\s*\)?\s*$"
)
# «شركة النور»: every template already writes «شركة» / «لشركة/» before it.
_LEADING_SHARIKA = re.compile(r"^شركة\s+")


def _bare_company_name(value: str) -> str:
    """The name with its edge punctuation and trailing legal form removed; a
    verbatim substring of `value`. Falls back to the edge-trimmed value when
    nothing but the form is left."""
    trimmed = _NAME_EDGE.sub("", value)
    name = trimmed
    while True:
        shorter = _NAME_EDGE.sub("", _LEADING_SHARIKA.sub("", _LEGAL_FORM.sub("", name)))
        if shorter == name:
            break
        name = shorter
    return name or trimmed
