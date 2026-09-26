"""Company type -> template file, plus the placeholder contract for each.

`verify_all()` runs in the app's startup path. A template edited later that
drops or renames a placeholder then fails loudly at boot rather than
producing a filed document with a silent hole in it.
"""

from __future__ import annotations

import enum
import pathlib
import re
import zipfile
from dataclasses import dataclass

FILES = pathlib.Path(__file__).resolve().parent / "files"

_VAR = re.compile(r"\{\{\s*([a-z_][a-z_.]*)\s*\}\}")


class CompanyType(enum.StrEnum):
    SHAKHS_WAHED = "shakhs_wahed"
    ZMM = "zmm"
    MASAHMA = "masahma"


class TemplateContractError(RuntimeError):
    """A template's placeholders no longer match what the code supplies."""

    def __init__(self, company_type: str, missing: set[str], unexpected: set[str]) -> None:
        super().__init__(
            f"template contract broken for {company_type}: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
        self.company_type = company_type
        self.missing = missing
        self.unexpected = unexpected


@dataclass(frozen=True)
class TemplateSpec:
    company_type: CompanyType
    path: pathlib.Path
    scalar_placeholders: frozenset[str]
    article_placeholders: frozenset[str]
    attendee_placeholders: frozenset[str]
    # "partner" for ذ.م.م, "shareholder" for مساهمة, None where the document
    # has no attendance table (شخص واحد has no general assembly).
    attendee_label: str | None


# Shared by all three documents. `articles_title` is NOT here: only the
# شخص واحد template has a document-level line naming the amended article
# outside the article block (its recital, "قرار مالك الشركة بتعديل ..."). In
# both محضر templates every `article_name` occurrence lives inside the
# per-article loop, so they have no use for a standalone title variable.
_COMMON_SCALARS = frozenset(
    {
        "company_name",
        "law_number",
        "law_year",
        "commercial_registration_no",
        "commercial_registration_date",
        "day_name",
        "day_date",
        "names_of_commissioners",
    }
)
_MEETING_SCALARS = frozenset(
    {
        "meeting_time",
        "meeting_end_time",
        "meeting_place",
        "chairman_name",
        "attendance_percentage",
        "approval_percentage",
    }
)
_ARTICLE_FIELDS = frozenset(
    {"a.article_name", "a.article_original_content", "a.article_new_content"}
)
_ATTENDEE_FIELDS = frozenset({"p.name", "p.shares", "p.percentage"})

_SPECS: dict[CompanyType, TemplateSpec] = {
    CompanyType.SHAKHS_WAHED: TemplateSpec(
        company_type=CompanyType.SHAKHS_WAHED,
        path=FILES / "قرار_تعديل_شركة_شخص_واحد_template.docx",
        scalar_placeholders=_COMMON_SCALARS | {"owner_name", "company_address", "articles_title"},
        article_placeholders=_ARTICLE_FIELDS,
        attendee_placeholders=frozenset(),
        attendee_label=None,
    ),
    CompanyType.ZMM: TemplateSpec(
        company_type=CompanyType.ZMM,
        path=FILES / "محضر_تعديل_شركة_ذات_مسئولية_محدودة_template.docx",
        scalar_placeholders=_COMMON_SCALARS | _MEETING_SCALARS | {"company_address"},
        article_placeholders=_ARTICLE_FIELDS,
        attendee_placeholders=_ATTENDEE_FIELDS,
        attendee_label="partner",
    ),
    CompanyType.MASAHMA: TemplateSpec(
        company_type=CompanyType.MASAHMA,
        path=FILES / "محضر_تعديل_شركة_مساهمة_template.docx",
        scalar_placeholders=_COMMON_SCALARS
        | _MEETING_SCALARS
        | {
            "company_address",
            "secretary_name",
            "vote_collector_name",
            "gafi_representative_name",
            "auditor_name",
            "board_meeting_date",
        },
        article_placeholders=_ARTICLE_FIELDS,
        attendee_placeholders=_ATTENDEE_FIELDS,
        attendee_label="shareholder",
    ),
}


def get_template(company_type: str) -> TemplateSpec:
    """The spec for a company type. Raises KeyError on an unknown type.

    Looked up by string rather than by `CompanyType(company_type)` so that an
    unknown type is a KeyError everywhere, instead of a ValueError.
    """
    for spec in _SPECS.values():
        if spec.company_type.value == company_type:
            return spec
    raise KeyError(f"unknown company type: {company_type}")


_SCANNED_PART = re.compile(r"word/(document|header\d*|footer\d*)\.xml")


def placeholders_in(path: pathlib.Path) -> set[str]:
    """Every `{{ name }}` variable in a .docx, dotted names included.

    Scans the document body plus every header/footer part -- default,
    first-page, and even-page variants alike, tables nested inside them
    included -- since OOXML stores each as its own `word/header*.xml` /
    `word/footer*.xml` part regardless of which python-docx property
    exposes it. A scanner limited to `word/document.xml` is blind to a
    stray token left in a header or footer.
    """
    found: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if _SCANNED_PART.fullmatch(name):
                found.update(_VAR.findall(archive.read(name).decode("utf-8")))
    return found


def verify_all() -> None:
    """Assert every shipped template matches its declared placeholder set."""
    for spec in _SPECS.values():
        expected = (
            spec.scalar_placeholders | spec.article_placeholders | spec.attendee_placeholders
        )
        found = placeholders_in(spec.path)
        missing = expected - found
        unexpected = found - expected
        if missing or unexpected:
            raise TemplateContractError(spec.company_type.value, missing, unexpected)
