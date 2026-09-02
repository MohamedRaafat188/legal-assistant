"""Convert the three Word templates from {name} to docxtpl syntax.

Idempotent: running it twice is a no-op (it bails out early per file once it
finds docxtpl-style `{{` markers already present).

Word is not available in this environment, so both the textual placeholder
rewrite and the block-level loop insertion (docxtpl's `{%p %}` / `{%tr %}`
tags, which must consume whole paragraphs / table rows) are done here with
python-docx and direct oxml manipulation rather than by hand in Word.

Placeholder rewrite rules (verified against the shipped documents -- every
placeholder sits whole inside a single run, so substituting a run's `.text`
is safe and never splits a placeholder across runs):

  * Plain scalars (company_name, law_number, ...) -> `{{ name }}`, unchanged
    name, wherever they occur.
  * `article_name` / `article_original_content` / `article_new_content`
    inside the قبل/بعد article block -> `{{ a.article_name }}` etc.
  * In the شخص واحد template ONLY, the single `{article_name}` occurrence
    in the document-level recital (outside the article block) becomes
    `{{ articles_title }}` instead -- confirmed by inspection that only this
    template has such a recital; the محضر templates use `article_name`
    solely inside the article block.
  * `partner_name`/`partner_shares`/`partner_percentage` (ذ.م.م) and
    `shareholder_name`/`shareholder_shares`/`shareholder_percentage`
    (مساهمة) -> `{{ p.name }}` / `{{ p.shares }}` / `{{ p.percentage }}`,
    since both templates use the same attendee loop variable name `p`.

Loop markers added (paragraphs/rows are inserted, none of the user's
original text is altered beyond the placeholder rewrite above):

  * Every template: `{%p for a in articles %}` immediately before the
    paragraph that starts the قبل/بعد article block, and `{%p endfor %}`
    immediately after the paragraph that ends it.
  * ذ.م.م and مساهمة only: `{%tr for p in attendees %}` / `{%tr endfor %}`
    as new rows directly above/below the attendance-table data row.
  * ذ.م.م only: the signature table's "الشركاء" cell holds all partners in
    a single cell, so a table-row loop cannot apply there. Its name
    paragraph is wrapped in a docxtpl paragraph loop instead:
    `{%p for p in attendees %}` / `{{ p.name }}` / `{%p endfor %}`. The
    مساهمة signature table lists officers only (chairman/secretary/vote
    collector) and does not repeat per attendee, so it is left untouched.

Usage: python scripts/docgen_convert_templates.py
"""

from __future__ import annotations

import copy
import pathlib
import re

import docx
from docx.oxml.ns import qn
from docx.table import Table, _Cell, _Row
from docx.text.paragraph import Paragraph

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = ROOT / "src" / "legal_assistant" / "docgen" / "templates" / "files"

SHAKHS_WAHED = FILES / "قرار_تعديل_شركة_شخص_واحد_template.docx"
ZMM = FILES / "محضر_تعديل_شركة_ذات_مسئولية_محدودة_template.docx"
MASAHMA = FILES / "محضر_تعديل_شركة_مساهمة_template.docx"

# Scalars that keep their name, rendered once from the top-level context.
# `articles_title` is added by the شخص واحد-specific step below, not here,
# since it does not exist as a literal `{articles_title}` placeholder in any
# source document -- it replaces one particular `{article_name}` occurrence.
_SCALAR_NAMES = {
    "company_name",
    "law_number",
    "law_year",
    "commercial_registration_no",
    "commercial_registration_date",
    "day_name",
    "day_date",
    "owner_name",
    "company_address",
    "names_of_commissioners",
    "meeting_time",
    "meeting_end_time",
    "meeting_place",
    "chairman_name",
    "secretary_name",
    "vote_collector_name",
    "gafi_representative_name",
    "auditor_name",
    "board_meeting_date",
    "attendance_percentage",
    "approval_percentage",
}
_SCALAR_RE = re.compile(
    r"\{(" + "|".join(sorted(_SCALAR_NAMES, key=len, reverse=True)) + r")\}"
)
_ARTICLE_FIELD_RE = re.compile(
    r"\{(article_name|article_original_content|article_new_content)\}"
)


def _sub_scalars(text: str) -> str:
    return _SCALAR_RE.sub(lambda m: "{{ " + m.group(1) + " }}", text)


def _sub_article_fields(text: str) -> str:
    return _ARTICLE_FIELD_RE.sub(lambda m: "{{ a." + m.group(1) + " }}", text)


def _sub_attendee_field(text: str, prefix: str) -> str:
    """`{prefix_name}` -> `{{ p.name }}`, etc., for one attendee prefix."""
    pattern = re.compile(r"\{" + prefix + r"_(name|shares|percentage)\}")
    return pattern.sub(lambda m: "{{ p." + m.group(1) + " }}", text)


def _iter_all_paragraphs(document: docx.Document):
    yield from document.paragraphs
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def _apply_to_runs(paragraph: Paragraph, fn) -> None:
    for run in paragraph.runs:
        if "{" in run.text:
            run.text = fn(run.text)


def insert_paragraph_after(paragraph: Paragraph, text: str) -> Paragraph:
    """A new paragraph immediately after `paragraph`, copying its pPr only."""
    new_p = copy.deepcopy(paragraph._p)
    for child in list(new_p):
        if child.tag != qn("w:pPr"):
            new_p.remove(child)
    paragraph._p.addnext(new_p)
    new_paragraph = Paragraph(new_p, paragraph._parent)
    new_paragraph.add_run(text)
    return new_paragraph


def insert_row_before(table: Table, row: _Row, first_cell_text: str) -> _Row:
    new_tr = copy.deepcopy(row._tr)
    row._tr.addprevious(new_tr)
    new_row = _Row(new_tr, table)
    for i, cell in enumerate(new_row.cells):
        cell.text = first_cell_text if i == 0 else ""
    return new_row


def insert_row_after(table: Table, row: _Row, first_cell_text: str) -> _Row:
    new_tr = copy.deepcopy(row._tr)
    row._tr.addnext(new_tr)
    new_row = _Row(new_tr, table)
    for i, cell in enumerate(new_row.cells):
        cell.text = first_cell_text if i == 0 else ""
    return new_row


def _find_paragraph_by_prefix(paragraphs: list[Paragraph], prefix: str) -> Paragraph:
    for p in paragraphs:
        if p.text.startswith(prefix):
            return p
    raise SystemExit(f"could not find paragraph starting with {prefix!r}")


def _find_paragraph_exact(paragraphs: list[Paragraph], text: str) -> Paragraph:
    for p in paragraphs:
        if p.text == text:
            return p
    raise SystemExit(f"could not find paragraph exactly {text!r}")


def convert_shakhs_wahed(path: pathlib.Path) -> None:
    document = docx.Document(path)
    paragraphs = document.paragraphs

    # p2 (the document-level recital) gets articles_title instead of a.*.
    recital = paragraphs[2]
    _apply_to_runs(
        recital,
        lambda t: re.sub(r"\{article_name\}", "{{ articles_title }}", t),
    )
    _apply_to_runs(recital, _sub_scalars)

    # Everything else: plain scalars + article-block fields.
    for p in paragraphs:
        if p is recital:
            continue
        _apply_to_runs(p, _sub_scalars)
        _apply_to_runs(p, _sub_article_fields)

    # Re-locate the article block by content (indices are stable pre-insert).
    block_start = _find_paragraph_exact(
        document.paragraphs, "تعديل {{ a.article_name }} من النظام الاساسى للشركة"
    )
    block_end = _find_paragraph_exact(document.paragraphs, "{{ a.article_new_content }}")

    block_start.insert_paragraph_before("{%p for a in articles %}")
    insert_paragraph_after(block_end, "{%p endfor %}")

    document.save(path)


def convert_zmm(path: pathlib.Path) -> None:
    document = docx.Document(path)

    for p in document.paragraphs:
        _apply_to_runs(p, _sub_scalars)
        _apply_to_runs(p, _sub_article_fields)

    block_start = _find_paragraph_exact(
        document.paragraphs, "تعديل {{ a.article_name }} من عقد تأسيس الشركة ونظامها الأساسى"
    )
    block_end = _find_paragraph_exact(document.paragraphs, "{{ a.article_new_content }}")
    block_start.insert_paragraph_before("{%p for a in articles %}")
    insert_paragraph_after(block_end, "{%p endfor %}")

    attendance = document.tables[0]
    data_row = attendance.rows[1]
    for cell in data_row.cells:
        for p in cell.paragraphs:
            _apply_to_runs(p, lambda t: _sub_attendee_field(t, "partner"))
    insert_row_before(attendance, data_row, "{%tr for p in attendees %}")
    insert_row_after(attendance, data_row, "{%tr endfor %}")

    signatures = document.tables[1]
    signature_cell: _Cell = signatures.rows[0].cells[1]
    name_paragraph = _find_paragraph_exact(signature_cell.paragraphs, "{partner_name}")
    _apply_to_runs(name_paragraph, lambda t: _sub_attendee_field(t, "partner"))
    name_paragraph.insert_paragraph_before("{%p for p in attendees %}")
    insert_paragraph_after(name_paragraph, "{%p endfor %}")

    # The chairman cell in the same table is a plain scalar; substitute it.
    chairman_cell: _Cell = signatures.rows[0].cells[0]
    for p in chairman_cell.paragraphs:
        _apply_to_runs(p, _sub_scalars)

    document.save(path)


def convert_masahma(path: pathlib.Path) -> None:
    document = docx.Document(path)

    for p in document.paragraphs:
        _apply_to_runs(p, _sub_scalars)
        _apply_to_runs(p, _sub_article_fields)

    block_start = _find_paragraph_exact(
        document.paragraphs, "تعديل {{ a.article_name }} من النظام الأساسى للشركة"
    )
    block_end = _find_paragraph_exact(document.paragraphs, "{{ a.article_new_content }}")
    block_start.insert_paragraph_before("{%p for a in articles %}")
    insert_paragraph_after(block_end, "{%p endfor %}")

    attendance = document.tables[0]
    data_row = attendance.rows[1]
    for cell in data_row.cells:
        for p in cell.paragraphs:
            _apply_to_runs(p, lambda t: _sub_attendee_field(t, "shareholder"))
    insert_row_before(attendance, data_row, "{%tr for p in attendees %}")
    insert_row_after(attendance, data_row, "{%tr endfor %}")

    # table1 (officers) does not repeat per attendee -- scalars only.
    officers = document.tables[1]
    for row in officers.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                _apply_to_runs(p, _sub_scalars)

    document.save(path)


_CONVERTERS = {
    SHAKHS_WAHED: convert_shakhs_wahed,
    ZMM: convert_zmm,
    MASAHMA: convert_masahma,
}


def _already_converted(path: pathlib.Path) -> bool:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    return "{{" in xml or "{%" in xml


def main() -> None:
    for path, converter in _CONVERTERS.items():
        if not path.exists():
            raise SystemExit(f"missing template: {path}")
        if _already_converted(path):
            print(f"skip (already converted): {path.name}")
            continue
        converter(path)
        print(f"converted: {path.name}")


if __name__ == "__main__":
    main()
