"""Live end-to-end validation of the docgen pipeline.

Hits the real database, the real OCR provider, and the real templates --
like scripts/phase4_validate.py and friends, and unlike tests/, which are
offline. Requires a populated .env and `alembic upgrade head`.

Usage:
    python scripts/docgen_validate.py "عقد تأسيس انجاز.pdf" --company-type zmm \
        --page-map scripts/docgen_page_maps/injaz.json
    python scripts/docgen_validate.py "عقد تأسيس نور للتوزيع.pdf" \
        --company-type shakhs_wahed --page-map scripts/docgen_page_maps/nour.json \
        --cr-no 303907 --cr-date 2021/03/14
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import sys

from sqlalchemy import select

from legal_assistant.db.models import User
from legal_assistant.db.session import get_engine, get_sessionmaker
from legal_assistant.docgen import service
from legal_assistant.docgen.models import DocgenSession, SessionStatus
from legal_assistant.docgen.pages import PageMap
from legal_assistant.docgen.parsing.commercial_register import parse_party_table
from legal_assistant.docgen.render import document_text
from legal_assistant.docgen.templates.registry import get_template, verify_all

_ADDRESS_RE = re.compile(
    r"(?:الكائن|الكائنة|مقرها|العنوان)\s*(?:التالي|الآتي|الأتى)?"
    r"\s*(?:فى|في|ب)?\s*[:\-]?\s*(?P<v>[^.\n]+)"
)


def _report(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def _find_company_address(articles) -> str | None:
    """Best-effort المركز الرئيسي text, tried against every article's body.

    `service._record_from_articles` only looks at whichever single article
    `find_article(Concept.HEAD_OFFICE)` returns; if that concept's signature
    phrase list doesn't match this particular document's wording, the field
    that production code populates comes back empty. This is the review
    screen's job in the real product (the lawyer types it in via PATCH
    /fields when extraction misses it) -- this function plays that same
    role for the validation script, scanning every article's real OCR'd
    text with the same regex `service._current_value` uses for this field,
    rather than fabricating a value.
    """
    for article in articles:
        match = _ADDRESS_RE.search(article.source_text)
        if match:
            return match.group("v").strip()
    return None


def _find_attendees(articles) -> list[dict]:
    """Real attendee rows via `parse_party_table`, tried against every
    article.

    Both `parsing/` gaps an earlier live run of this script found -- the
    CAPITAL concept classifier's one-word "رأسمال" spelling and
    `parse_party_table`'s "على الوجه الآتي" roster lead-in -- are now fixed
    in `parsing/signatures.py` and `parsing/commercial_register.py`
    respectively, so production extraction should populate both fields
    directly. This function stays as the review screen's fallback (the
    lawyer-correction path this script exercises via PATCH /attendees): a
    document whose real wording still doesn't match either phrase list
    degrades to [] here exactly as it would in production, rather than
    crashing the script. Returns [] when no article's table is recognized;
    the caller substitutes clearly-labeled placeholder attendees rather than
    fabricating real ones, since real partner names are never allowed to
    reach the committed script (see the caller).
    """
    for article in articles:
        parties = parse_party_table(article.source_text)
        if parties:
            return [
                {
                    "name": p.name,
                    "shares": p.shares,
                    "percentage": p.percentage,
                    "attending": True,
                    "source": "aoa",
                }
                for p in parties
            ]
    return []


async def run(
    path: pathlib.Path, company_type: str, page_map_path: pathlib.Path, cr_no: str, cr_date: str
) -> int:
    results: list[bool] = []

    verify_all()
    results.append(_report("template placeholder contract", True))

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as db:
        user = await db.scalar(select(User).limit(1))
        if user is None:
            print("no user in the database; create one via POST /auth/register first")
            return 1

        session = await service.create_session(db, user.id, company_type, "aoa_only")
        await db.commit()
        session_id = session.id
        print(f"session {session_id} created for {company_type}")

    # Every codepath below MUST reach the cleanup at the bottom of this
    # try/finally: this session's storage holds a real partner's national ID
    # and passport numbers from the moment `add_upload` returns, and an
    # uncaught exception from OCR or render (the very thing this script
    # exists to exercise for the first time against a live model) must not
    # leave that content sitting in `var/docgen/` or the DB. The brief's
    # original script had no such guarantee -- a render failure mid-run left
    # exactly this behind in practice during this task's own live run.
    try:
        async with sessionmaker() as db:
            session = await service.get_session(db, session_id, user.id)
            await service.add_upload(db, session, "aoa", path.name, path.read_bytes())
            await db.commit()

        page_map_raw = json.loads(page_map_path.read_text(encoding="utf-8"))
        async with sessionmaker() as db:
            session = await service.get_session(db, session_id, user.id)
            await service.submit_page_map(db, session, page_map_raw)
            await db.commit()

        print("running OCR (this calls the real provider and costs money)...")
        await service.run_ocr_job(session_id)

        async with sessionmaker() as db:
            session = await service.get_session(db, session_id, user.id)
            results.append(
                _report(
                    "session reached ready",
                    session.status == SessionStatus.ready.value,
                    session.error or "",
                )
            )
            results.append(
                _report(
                    "articles extracted", len(session.articles) > 0, f"{len(session.articles)}"
                )
            )

            aoa_upload = next(u for u in session.uploads if u.kind == "aoa")
            ocr_page_count = len((aoa_upload.ocr_pages or {}).keys())
            mapped_page_count = len(PageMap.from_json(session.page_map).ocr_pages())
            results.append(
                _report(
                    "OCR'd only the mapped pages",
                    ocr_page_count == mapped_page_count,
                    f"{ocr_page_count} OCR'd vs {mapped_page_count} mapped",
                )
            )

            for article in session.articles[:5]:
                preview = article.patched_text[:60].replace("\n", " ")
                print(f"    المادة {article.article_number}: {preview}...")

            results.append(
                _report(
                    "no article body is empty",
                    all(a.patched_text.strip() for a in session.articles),
                )
            )
            results.append(
                _report(
                    "every declared article was found",
                    all(a.status == "found" for a in session.articles),
                    ", ".join(f"{a.article_number}:{a.status}" for a in session.articles),
                )
            )

            if not session.articles:
                print("no articles; stopping before render")
                return 0 if all(results) else 1

            target = session.articles[0]

            # Fields no aoa_only extraction path can ever populate, or that
            # this document's wording defeated the concept classifier for
            # (see _find_company_address) -- supplied here exactly as the
            # review screen's lawyer-correction flow (PATCH /fields, PATCH
            # /attendees) is designed to receive them, using values
            # recovered from this document's own real OCR text wherever
            # possible rather than fabricated ones.
            existing = session.fields.data if session.fields else {}

            def _value_of(name: str) -> str:
                return ((existing.get(name) or {}).get("value")) or ""

            extra_fields: dict[str, str] = {}
            if not _value_of("law_number"):
                # aoa_only never extracts this -- it only ever comes from
                # the السجل التجاري in aoa_plus_cr mode (see
                # `service._extract_fields`'s CR-only field list). Egypt's
                # Companies Law 159/1981 is what every one of these company
                # types' عقد recites; a lawyer using aoa_only types it in
                # exactly as done here.
                extra_fields["law_number"] = "159"
            if not _value_of("law_year"):
                extra_fields["law_year"] = "1981"
            if not _value_of("company_address"):
                found_address = _find_company_address(session.articles)
                results.append(
                    _report(
                        "company_address recovered from OCR text (extraction missed it)",
                        found_address is not None,
                    )
                )
                if found_address:
                    extra_fields["company_address"] = found_address
                else:
                    # Same class of miss as `_find_attendees`: neither
                    # `parsing.signatures`'s HEAD_OFFICE phrase list nor this
                    # script's own address regex recognized this document's
                    # actual وموطنها القانوني/العنوان wording. A placeholder
                    # (never real address text, since none was found) keeps
                    # the render from blocking on it entirely; the FAIL above
                    # already records the miss honestly.
                    extra_fields["company_address"] = "(العنوان لم يُستخرج تلقائياً - يُستكمل يدوياً)"

            spec = get_template(company_type)
            if spec.attendee_label:
                issued_capital = _value_of("issued_capital")
                results.append(
                    _report(
                        "issued_capital extracted automatically",
                        bool(issued_capital),
                    )
                )

                attendees = _find_attendees(session.articles)
                results.append(
                    _report(
                        "attendees recovered from OCR text via parse_party_table",
                        bool(attendees),
                    )
                )
                if not attendees and issued_capital:
                    # parse_party_table's lead-in regex did not recognize this
                    # document's real wording (see _find_attendees) -- rather
                    # than commit a real partner's name to this script's git
                    # history, substitute clearly-labeled placeholder rows
                    # whose shares still sum to the real issued capital, so
                    # the reconciliation gate and the render both exercise
                    # real numbers with no real personal names anywhere in
                    # source control.
                    print(
                        "    party table not recognized; substituting "
                        "placeholder attendees for the render (see report)"
                    )
                    attendees = [
                        {
                            "name": "الشريك الأول",
                            "shares": issued_capital,
                            "percentage": "100",
                            "attending": True,
                            "source": "user",
                        }
                    ]
                if attendees:
                    await service.update_attendees(db, session, attendees)

            await service.update_fields(
                db,
                session,
                {
                    "commercial_registration_no": cr_no,
                    "commercial_registration_date": cr_date,
                    "day_name": "الأحد",
                    "day_date": "2026/09/06",
                    "names_of_commissioners": "أحمد كامل",
                    # A generic, fictional name -- NOT a value transcribed
                    # from either real sample. An earlier draft of this
                    # script (matching the task brief's own literal text)
                    # used a name here that turned out to match real name
                    # components of an actual partner named in both samples'
                    # party tables (see this task's report); that value is
                    # deliberately not reproduced or committed here.
                    "owner_name": "محمود عبد الرحمن",
                    "chairman_name": "محمود عبد الرحمن",
                    "meeting_time": "الحادية عشرة صباحا",
                    "meeting_end_time": "الواحدة ظهرا",
                    "meeting_place": "مقر الشركة",
                    "secretary_name": "سارة على",
                    "vote_collector_name": "محمد سمير",
                    "gafi_representative_name": "ممثل الهيئة",
                    "auditor_name": "مراقب الحسابات",
                    "board_meeting_date": "2026/08/20",
                    "attendance_percentage": "100",
                    "approval_percentage": "100",
                    **extra_fields,
                },
            )

            fields_row = session.fields.data if session.fields else {}
            for name, entry in fields_row.items():
                if isinstance(entry, dict):
                    print(
                        f"    field {name}: source={entry.get('source')} "
                        f"flags={entry.get('flags')}"
                    )

            for article in session.articles:
                await service.update_article(
                    db, session, article.position, new_text="النص الجديد للمادة."
                )
            await db.commit()

        async with sessionmaker() as db:
            session = await service.get_session(db, session_id, user.id)
            document = await service.render_session(db, session)
            await db.commit()

        text = document_text(document)
        results.append(_report("no placeholder residue", "{{" not in text and "{%" not in text))
        # Rendering joins the scan's line wraps, so compare whitespace-normalized.
        flat = " ".join(text.split())
        before = " ".join(target.patched_text.split())[:30]
        results.append(_report("قبل التعديل text present", before in flat))
        results.append(_report("no markdown table pipes left", "|" not in text))
        results.append(_report("بعد التعديل text present", "النص الجديد للمادة." in text))

        output = pathlib.Path(f"docgen-validate-{session_id}.docx")
        output.write_bytes(document)
        print(f"wrote {output}")

        return 0 if all(results) else 1
    finally:
        async with sessionmaker() as db:
            session = await db.get(DocgenSession, session_id)
            if session is not None:
                await service.delete_session(db, session)
                await db.commit()
                print(f"cleaned up session {session_id}")
            else:
                print(f"session {session_id} already gone; nothing to clean up")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=pathlib.Path)
    parser.add_argument("--company-type", required=True, choices=["shakhs_wahed", "zmm", "masahma"])
    parser.add_argument("--page-map", type=pathlib.Path, required=True)
    parser.add_argument("--cr-no", default="303907")
    parser.add_argument("--cr-date", default="2021/03/14")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"no such file: {args.pdf}")
        return 1
    if not args.page_map.exists():
        print(f"no such file: {args.page_map}")
        return 1
    try:
        return asyncio.run(
            run(args.pdf, args.company_type, args.page_map, args.cr_no, args.cr_date)
        )
    finally:
        asyncio.run(get_engine().dispose())


if __name__ == "__main__":
    sys.exit(main())
