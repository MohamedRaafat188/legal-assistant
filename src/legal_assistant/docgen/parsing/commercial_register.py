"""Typed company identity data, from the سجل تجاري or from the عقد.

The سجل is a dense 14-box landscape form under a security guilloche; read as
free text it comes back scrambled. So the extractor is handed a JSON schema
with one property per box we need and returns typed fields. This module owns
the schema and the payload -> dataclass conversion; the LLM call that fills
the payload lives in `ocr/gemini.py`, so everything here is offline-testable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Party:
    """A شريك / مساهم and, when the source states it, their holding.

    `shares` and `percentage` are `None` -- never 0, never a guessed split --
    when the source does not state them. The review screen then asks the
    lawyer to fill them in.
    """

    name: str
    shares: str | None = None
    percentage: str | None = None


@dataclass(frozen=True)
class CompanyRecord:
    commercial_registration_no: str | None = None
    commercial_registration_date: str | None = None
    company_name: str | None = None
    law_number: str | None = None
    law_year: str | None = None
    company_address: str | None = None
    capital: str | None = None
    # رأس المال المصدر. Equals `capital` for ذ.م.م and شخص واحد, which state a
    # single figure; differs for مساهمة, whose quorum is computed from المصدر.
    issued_capital: str | None = None
    parties: list[Party] = field(default_factory=list)


# One property per box we read off the مستخرج. Descriptions are in Arabic
# because they are read by the extraction model alongside an Arabic page.
CR_FIELD_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "commercial_registration_no": {
            "type": "string",
            "description": "رقم السجل التجارى كما يظهر فى ترويسة المستخرج",
        },
        "commercial_registration_date": {
            "type": "string",
            "description": "تاريخ القيد من الخانة (1)",
        },
        "company_name": {
            "type": "string",
            "description": "الاسم التجارى من الخانة (2) بدون وصف الشكل القانونى",
        },
        "law_number": {
            "type": "string",
            "description": "رقم القانون الذى تخضع له الشركة، من الخانة (2)",
        },
        "law_year": {"type": "string", "description": "سنة صدور القانون، من الخانة (2)"},
        "company_address": {
            "type": "string",
            "description": "عنوان المحل الرئيسى من الخانة (6)",
        },
        "capital": {"type": "string", "description": "رأس المال من الخانة (9)"},
        "issued_capital": {
            "type": "string",
            "description": "رأس المال المصدر إن ذُكر منفصلا عن رأس المال المرخص به",
        },
        "parties": {
            "type": "array",
            "description": (
                "الشركاء وحصصهم من الخانة (9) إن وُجدت. اتركها فارغة تماما إذا "
                "لم تذكر الخانة أسماء الشركاء -- لا تخمّن أى توزيع."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "shares": {"type": "string"},
                    "percentage": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    "required": [],
}


def _clean(value: object) -> str | None:
    """Trim a payload value; blank and non-string become None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def record_from_payload(payload: Mapping) -> CompanyRecord:
    """Convert a raw extractor payload into a CompanyRecord.

    Missing keys, blanks, and nameless party rows are all tolerated -- the
    lawyer fills the gaps at review. Anything that is not a mapping is a
    programming error and raises.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"CR payload must be a mapping, got {type(payload).__name__}")

    parties: list[Party] = []
    for raw in payload.get("parties") or []:
        if not isinstance(raw, Mapping):
            continue
        name = _clean(raw.get("name"))
        if not name:
            continue
        parties.append(
            Party(
                name=name,
                shares=_clean(raw.get("shares")),
                percentage=_clean(raw.get("percentage")),
            )
        )

    return CompanyRecord(
        commercial_registration_no=_clean(payload.get("commercial_registration_no")),
        commercial_registration_date=_clean(payload.get("commercial_registration_date")),
        company_name=_clean(payload.get("company_name")),
        law_number=_clean(payload.get("law_number")),
        law_year=_clean(payload.get("law_year")),
        company_address=_clean(payload.get("company_address")),
        capital=_clean(payload.get("capital")),
        issued_capital=_clean(payload.get("issued_capital")),
        parties=parties,
    )


# رأس المال المرخص به X ... رأس المال المصدر Y. Either half may be absent.
_AUTHORIZED = re.compile(r"(?:المرخص\s*به)[^0-9٠-٩]{0,20}([0-9٠-٩][0-9٠-٩,.]*)")
_ISSUED = re.compile(r"(?:المصدر)[^0-9٠-٩]{0,20}([0-9٠-٩][0-9٠-٩,.]*)")
_ANY_FIGURE = re.compile(
    r"(?:رأس\s*(?:ال)?مال|راس\s*(?:ال)?مال)[^0-9٠-٩]{0,30}([0-9٠-٩][0-9٠-٩,.]*)"
)


def split_capital(text: str) -> tuple[str | None, str | None]:
    """(authorized, issued) capital figures from a capital article.

    A مساهمة states both and its quorum is computed from المصدر; ذ.م.م and
    شخص واحد state one figure, which IS the issued capital. Reporting a
    single figure as `authorized` would silently feed the wrong number into
    `attendance_percentage`, so a lone figure is always the issued one.
    """
    authorized = _AUTHORIZED.search(text)
    issued = _ISSUED.search(text)
    if authorized or issued:
        return (
            authorized.group(1) if authorized else None,
            issued.group(1) if issued else None,
        )
    single = _ANY_FIGURE.search(text)
    return (None, single.group(1) if single else None)


# A share row is: a name, then a share count, then a percentage. Digits in
# either set; the percentage sign may be ٪ or %. Anything that does not match
# this shape contributes a name-only Party rather than a guessed holding.
_SHARE_ROW = re.compile(
    r"^\s*(?P<name>[^\d٠-٩\n]{3,})?\s*"
    r"(?P<shares>[0-9٠-٩]+)\s*(?:حصة|حصص|سهم|سهما|أسهم)\s*"
    r"(?P<pct>[0-9٠-٩]+(?:[.,][0-9٠-٩]+)?)\s*[٪%]"
)
_NAME_ONLY = re.compile(r"^\s*(?P<name>[^\d٠-٩:،\n]{3,})\s*$")


def parse_party_table(text: str) -> list[Party]:
    """Pull partner/shareholder rows out of a capital article's share table.

    Returns [] when the article states only aggregates -- which is exactly
    what the blank GAFI مساهمة نموذج does. A name with no parseable holding
    yields a Party with `shares=None`, so the name still prefills and the
    review screen asks for the number.
    """
    parties: list[Party] = []
    started = False

    for line in text.splitlines():
        if not started:
            # The roster begins after the "وزعت على الشركاء كالآتى" style lead-in.
            if re.search(r"(الشركاء|المساهمين|المؤسسين)\s*(كالآتى|كالاتى|كالتالى|:)", line):
                started = True
            continue

        row = _SHARE_ROW.match(line)
        if row and row.group("name"):
            parties.append(
                Party(
                    name=row.group("name").strip(),
                    shares=row.group("shares"),
                    percentage=row.group("pct"),
                )
            )
            continue

        name_only = _NAME_ONLY.match(line)
        if name_only:
            parties.append(Party(name=name_only.group("name").strip()))

    return parties
