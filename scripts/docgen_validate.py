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
import sys

from sqlalchemy import select

from legal_assistant.db.models import User
from legal_assistant.db.session import get_engine, get_sessionmaker
from legal_assistant.docgen import service
from legal_assistant.docgen.models import DocgenSession, SessionStatus
from legal_assistant.docgen.pages import PageMap, required_entries
from legal_assistant.docgen.parsing.scoped import ENTRY_FIELDS
from legal_assistant.docgen.render import document_text
from legal_assistant.docgen.templates.registry import get_template, verify_all


def _report(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


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

            # Lengths only: article text can carry national ID numbers.
            for article in session.articles:
                print(
                    f"    article {article.article_number}"
                    f"{' mukarrar' if article.is_mukarrar else ''}: "
                    f"{len(article.patched_text)} chars, status={article.status}"
                )

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

            # Extracted fields are REPORTED as extraction left them and never
            # overwritten here, so this run shows what the pipeline really
            # produced. A missing one gets a labelled placeholder only so the
            # render step still runs; its FAIL is already recorded.
            existing = session.fields.data if session.fields else {}
            placeholders: dict[str, str] = {}
            for entry_name in required_entries(company_type):
                for name in ENTRY_FIELDS[entry_name]:
                    entry = existing.get(name) or {}
                    ok = bool(entry.get("value"))
                    results.append(
                        _report(
                            f"{name} extracted automatically", ok, f"flags={entry.get('flags')}"
                        )
                    )
                    if not ok:
                        placeholders[name] = "(لم يُستخرج تلقائياً)"

            spec = get_template(company_type)
            if spec.attendee_label:
                attendees = existing.get("attendees") or []
                results.append(
                    _report(
                        f"{spec.attendee_label} table extracted automatically",
                        bool(attendees),
                        f"{len(attendees)} rows",
                    )
                )
                issued_capital = (existing.get("issued_capital") or {}).get("value")
                if not attendees and issued_capital:
                    # A labelled placeholder row, never a real name, whose
                    # shares still sum to the real issued capital, so the
                    # capital check and the render run on real numbers.
                    print("    placeholder attendee used for the render (see FAIL above)")
                    await service.update_attendees(
                        db,
                        session,
                        [
                            {
                                "name": "الشريك الأول",
                                "shares": issued_capital,
                                "percentage": "100",
                                "attending": True,
                                "source": "user",
                            }
                        ],
                    )

            # Only what a lawyer always types: nothing here is in the عقد.
            await service.update_fields(
                db,
                session,
                {
                    "commercial_registration_no": cr_no,
                    "commercial_registration_date": cr_date,
                    "day_name": "الأحد",
                    "day_date": "2026/09/06",
                    "names_of_commissioners": "أحمد كامل",
                    # Generic, fictional names -- never values from the samples.
                    "chairman_name": "محمود عبد الرحمن",
                    "meeting_time": "الحادية عشرة صباحا",
                    "meeting_end_time": "الواحدة ظهرا",
                    "secretary_name": "سارة على",
                    "vote_collector_name": "محمد سمير",
                    "gafi_representative_name": "ممثل الهيئة",
                    "auditor_name": "مراقب الحسابات",
                    "board_meeting_date": "2026/08/20",
                    **placeholders,
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
