"""Fill a company-type template and return .docx bytes.

Nothing in this module writes legal prose. It moves already-approved strings
into a Word template and refuses to emit a file if any declared placeholder
is unfilled -- a document filed at GAFI containing a literal
"{{ owner_name }}" is worse than a 500.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import docx
from docxtpl import DocxTemplate

from legal_assistant.docgen.numbering import articles_title
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
    article_numbers: Sequence[int],
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

    template = DocxTemplate(str(spec.path))
    template.render(context)
    buffer = io.BytesIO()
    template.save(buffer)
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
