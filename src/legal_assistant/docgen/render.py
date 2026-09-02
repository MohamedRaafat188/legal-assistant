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
    """Render the template for `company_type` into .docx bytes."""
    spec = get_template(company_type)

    missing = {name for name in spec.scalar_placeholders if not context.get(name)}
    if missing:
        raise MissingContextError(company_type, missing)

    template = DocxTemplate(str(spec.path))
    template.render(context)
    buffer = io.BytesIO()
    template.save(buffer)
    return buffer.getvalue()


def document_text(docx_bytes: bytes) -> str:
    """All paragraph and table text of a .docx, newline-joined.

    Used by tests and by the validation script to assert on a rendered
    document without opening Word.
    """
    document = docx.Document(io.BytesIO(docx_bytes))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)
