"""Fill a company-type template and return .docx bytes.

Nothing in this module writes legal prose. It moves already-approved strings
into a Word template and refuses to emit a file if any declared placeholder
is unfilled -- a document filed at GAFI containing a literal
"{{ owner_name }}" is worse than a 500.
"""

from __future__ import annotations

import copy
import io
import re
import secrets
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docxtpl import DocxTemplate

from legal_assistant.docgen.numbering import ArticleRef, articles_title
from legal_assistant.docgen.parsing.commercial_register import table_cells
from legal_assistant.docgen.templates.registry import get_template


@dataclass(frozen=True)
class ArticleBlock:
    """One قبل/بعد pair as it appears in the document."""

    article_name: str
    article_original_content: str
    article_new_content: str


@dataclass(frozen=True)
class Attendee:
    name: str
    shares: str
    percentage: str


class MissingContextError(RuntimeError):
    """A placeholder the template declares has no value in the context."""

    def __init__(self, company_type: str, missing: set[str]) -> None:
        super().__init__(
            f"cannot render {company_type}: missing context values {sorted(missing)}"
        )
        self.company_type = company_type
        self.missing = missing


def build_context(
    company_type: str,
    scalars: dict[str, str],
    articles: Sequence[ArticleBlock],
    article_numbers: Sequence[ArticleRef | int],
    attendees: Sequence[Attendee],
) -> dict:
    """Assemble the docxtpl context.

    `articles_title` is computed here rather than passed in, so the heading
    can never disagree with the articles actually in the document.
    """
    if not articles:
        raise ValueError("a قرار/محضر التعديل must amend at least one article")
    return {
        **scalars,
        "articles_title": articles_title(article_numbers),
        "articles": [asdict(a) for a in articles],
        "attendees": [asdict(p) for p in attendees],
    }


# --- article text layout ----------------------------------------------------
#
# Article text is stored as OCR wrote it: one line per SCAN line, tables as
# markdown pipe rows. Pasted into Word as-is, every scan line becomes a forced
# break and a table becomes a row of "|" characters. `layout_blocks` turns it
# into paragraphs and tables -- whitespace only; no word is added, dropped or
# changed.


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class TableBlock:
    rows: list[list[str]]


_LIST_ITEM = re.compile(r"^(?:[•●▪◦*\-–]|[0-9٠-٩]{1,2}\s*[-).]|[أ-ي]\s*[-)])\s*")
_PARAGRAPH_END = (".", ":", "؛", "：")


def _is_table_line(line: str) -> bool:
    return line.startswith("|")


def _is_separator(line: str) -> bool:
    return set(line) <= set("|-:–— ")


def layout_blocks(text: str) -> list[TextBlock | TableBlock]:
    """Paragraphs and tables, in order.

    A paragraph ends at a blank line, a table, a list item, or a line ending
    in «.», «:» or «؛». Other line breaks are the scan's line wrapping and
    are joined with a space.
    """
    blocks: list[TextBlock | TableBlock] = []
    paragraph: list[str] = []
    table: list[list[str]] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(TextBlock(" ".join(paragraph)))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            width = max(len(row) for row in table)
            blocks.append(TableBlock([row + [""] * (width - len(row)) for row in table]))
            table.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if _is_table_line(line):
            flush_paragraph()
            if not _is_separator(line):
                table.append(table_cells(line))
            continue
        flush_table()
        if not line:
            flush_paragraph()
            continue
        if _LIST_ITEM.match(line):
            flush_paragraph()
        paragraph.append(line)
        if line.endswith(_PARAGRAPH_END):
            flush_paragraph()
    flush_table()
    flush_paragraph()
    return blocks


_JUSTIFIED = {"both", "distribute", "lowKashida", "mediumKashida", "highKashida", "thaiDistribute"}


def _align_start(element) -> None:
    """Drop justified alignment under `element`. With no `w:jc` a paragraph
    aligns to its start edge -- the right, for Arabic. (An explicit "right"
    would not do: Word reads left/right as start/end in a bidi paragraph.)"""
    for jc in list(element.iter(qn("w:jc"))):
        if jc.get(qn("w:val")) in _JUSTIFIED:
            jc.getparent().remove(jc)


def _new_paragraph(anchor, text: str, bold: bool = False):
    """A paragraph in the anchor paragraph's paragraph and run formatting."""
    paragraph = OxmlElement("w:p")
    ppr = anchor.find(qn("w:pPr"))
    if ppr is not None:
        paragraph.append(copy.deepcopy(ppr))
    first_run = anchor.find(qn("w:r"))
    rpr = first_run.find(qn("w:rPr")) if first_run is not None else None
    rpr = copy.deepcopy(rpr) if rpr is not None else OxmlElement("w:rPr")
    if bold and rpr.find(qn("w:b")) is None:
        rpr.insert(0, OxmlElement("w:bCs"))
        rpr.insert(0, OxmlElement("w:b"))
    run = OxmlElement("w:r")
    run.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    run.append(t)
    paragraph.append(run)
    _align_start(paragraph)
    return paragraph


def _new_table(anchor, rows: list[list[str]]):
    """A bordered, right-to-left, full-width table. The first row is the
    header: bold, and repeated when the table crosses a page."""
    tbl = OxmlElement("w:tbl")
    tblpr = OxmlElement("w:tblPr")
    width = OxmlElement("w:tblW")
    width.set(qn("w:w"), "5000")
    width.set(qn("w:type"), "pct")
    tblpr.append(width)
    tblpr.append(OxmlElement("w:bidiVisual"))  # first column on the right
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "000000")
        borders.append(border)
    tblpr.append(borders)
    tbl.append(tblpr)
    grid = OxmlElement("w:tblGrid")
    for _ in rows[0]:
        grid.append(OxmlElement("w:gridCol"))
    tbl.append(grid)
    for index, cells in enumerate(rows):
        tr = OxmlElement("w:tr")
        if index == 0:
            trpr = OxmlElement("w:trPr")
            trpr.append(OxmlElement("w:tblHeader"))
            tr.append(trpr)
        for value in cells:
            tc = OxmlElement("w:tc")
            tc.append(OxmlElement("w:tcPr"))
            tc.append(_new_paragraph(anchor, value, bold=index == 0))
            tr.append(tc)
        tbl.append(tr)
    return tbl


def _expand_marker(anchor, text: str) -> None:
    """Replace the marker paragraph `anchor` with `text` laid out as
    paragraphs and tables, in the anchor's formatting."""
    for block in layout_blocks(text) or [TextBlock("")]:
        if isinstance(block, TableBlock):
            anchor.addprevious(_new_table(anchor, block.rows))
        else:
            anchor.addprevious(_new_paragraph(anchor, block.text))
    anchor.getparent().remove(anchor)


_LAID_OUT = ("article_original_content", "article_new_content")


def render_document(company_type: str, context: dict) -> bytes:
    """Render the template for `company_type` into .docx bytes.

    Fails closed not only on a missing scalar but also on a template whose
    article/attendee loop would render as silently empty: a قرار تعديل with
    zero amendments, or a محضر جمعية عامة with zero attendees, is not a
    valid instrument even though docxtpl would happily render it with no
    `{{`/`{%` residue at all. This check is enforced here -- not only in
    `build_context` -- because `render_document` is itself a public entry
    point a caller could reach with a hand-built context.
    """
    spec = get_template(company_type)

    missing = {name for name in spec.scalar_placeholders if not context.get(name)}
    if spec.article_placeholders and not context.get("articles"):
        missing.add("articles")
    if spec.attendee_placeholders and not context.get("attendees"):
        missing.add("attendees")
    if missing:
        raise MissingContextError(company_type, missing)

    # Article text goes in as a unique marker; each marker paragraph is then
    # replaced by real paragraphs and tables (see `layout_blocks`).
    token = secrets.token_hex(8)
    texts: dict[str, str] = {}
    articles = []
    for index, article in enumerate(context.get("articles", [])):
        article = dict(article)
        for key in _LAID_OUT:
            marker = f"DOCGEN-{token}-{index}-{key}"
            texts[marker] = article.get(key) or ""
            article[key] = marker
        articles.append(article)
    template = DocxTemplate(str(spec.path))
    template.render({**context, "articles": articles})
    rendered = io.BytesIO()
    template.save(rendered)

    document = docx.Document(io.BytesIO(rendered.getvalue()))
    body = document.element.body
    for paragraph in list(body.iter(qn("w:p"))):
        text = "".join(t.text or "" for t in paragraph.iter(qn("w:t"))).strip()
        if text in texts:
            _expand_marker(paragraph, texts.pop(text))
    if texts or token in "".join(t.text or "" for t in body.iter(qn("w:t"))):
        # The template puts article text inside other text: fail closed
        # rather than ship a marker. Content-free message.
        raise RuntimeError(f"cannot lay out article text in the {company_type} template")
    _align_start(body)
    for part in _header_footer_parts(document):
        _align_start(part._element)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _header_footer_parts(document):
    """Every distinct header/footer object on a document's sections.

    Covers the default, first-page, and even-page variants; a variant that
    is merely linked to the previous section (i.e. not actually defined)
    is skipped so its inherited-empty text is not double-counted.
    """
    parts = []
    for section in document.sections:
        for attr in (
            "header",
            "footer",
            "first_page_header",
            "first_page_footer",
            "even_page_header",
            "even_page_footer",
        ):
            part = getattr(section, attr)
            if not part.is_linked_to_previous:
                parts.append(part)
    return parts


def document_text(docx_bytes: bytes) -> str:
    """All paragraph and table text of a .docx, newline-joined.

    Includes section headers and footers (and any tables nested in them),
    not just the body -- a stray unrendered token left in a footer must be
    just as visible to `"{{" not in text`-style assertions as one in the
    body.

    Used by tests and by the validation script to assert on a rendered
    document without opening Word.
    """
    document = docx.Document(io.BytesIO(docx_bytes))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    for hf in _header_footer_parts(document):
        parts.extend(p.text for p in hf.paragraphs)
        for table in hf.tables:
            for row in table.rows:
                parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)
