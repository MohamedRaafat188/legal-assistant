"""HTTP smoke test of every docgen route, in-process via httpx.ASGITransport.

Drives the real app against the local database, but with a FAKE OCR provider
returning fictional text and a synthetic PDF -- so it costs nothing, sends
nothing to Google, and touches no real personal data. It checks what the
service-level `docgen_validate.py` cannot: status codes, the Arabic 422 page-
map problem list, thumbnail headers, article addressing by position,
ownership isolation, render/download, and deletion.

Usage: python scripts/docgen_http_smoke.py   (needs `alembic upgrade head`)
"""

from __future__ import annotations

import asyncio
import datetime
import os
import sys
import tempfile
import uuid
from pathlib import Path

# Before any settings are read: keep this run's files out of var/docgen.
os.environ["DOCGEN_STORAGE_DIR"] = tempfile.mkdtemp(prefix="docgen-smoke-")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
import pymupdf  # noqa: E402
from phase5_validate import register  # noqa: E402
from sqlalchemy import func, update  # noqa: E402

from legal_assistant.api import app as app_module  # noqa: E402
from legal_assistant.db.session import get_sessionmaker  # noqa: E402
from legal_assistant.docgen import service  # noqa: E402
from legal_assistant.docgen.models import DocgenSession  # noqa: E402
from legal_assistant.docgen.ocr.base import PageText  # noqa: E402

FAILURES: list[str] = []

# Fictional company and people.
_PAGES = {
    1: "\n".join(
        [
            "عقد تأسيس شركة ذات مسئولية محدودة",
            "وفقا لأحكام القانون رقم 159 لسنة 1981 ولائحته التنفيذية",
            "المادة (1)",
            "تأسست بين الموقعين على هذا العقد شركة ذات مسئولية محدودة.",
            "المادة (2)",
            "اسم الشركة هو شركة النخيل التجريبية للتجارة ذ.م.م.",
        ]
    ),
    2: "\n".join(
        [
            "المادة (5)",
            "المركز الرئيسي للشركة ومحلها القانوني الكائن في 10 شارع التجربة، القاهرة.",
            "المادة (6)",
            "حدد رأس مال الشركة بمبلغ 100000 جنيه مصري موزعة على الشركاء كالآتي:",
            "| م | الاسم | الجنسية | عدد الحصص | النسبة |",
            "|---|---|---|---|---|",
            "| 1 | شريك تجريبي أول | مصري | 600 | 60% |",
            "| 2 | شريك تجريبي ثان | مصري | 400 | 40% |",
        ]
    ),
}

_PAGE_MAP = {
    "entries": {
        "company_name": {"from": 1, "to": 1, "article": "2"},
        "law_reference": {"from": 1, "to": 1, "article": "التمهيد"},
        "company_address": {"from": 2, "to": 2, "article": "5"},
        "partners": {"from": 2, "to": 2, "article": "6"},
        "issued_capital": {"from": 2, "to": 2, "article": "6"},
    },
    "amended": [
        {"from": 2, "to": 2, "article": "5"},
        {"from": 2, "to": 2, "article": "6"},
    ],
}


class _FakeProvider:
    async def extract(self, images):
        return [PageText(page=i.page, text=_PAGES.get(i.page, ""), confidence=0.99) for i in images]

    async def extract_fields(self, images, schema):
        return {}


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


async def _force_ocr_running(session_id: int, minutes_ago: int) -> None:
    async with get_sessionmaker()() as db:
        await db.execute(
            update(DocgenSession)
            .where(DocgenSession.id == session_id)
            .values(
                status="ocr_running",
                updated_at=func.now() - datetime.timedelta(minutes=minutes_ago),
            )
        )
        await db.commit()


def _synthetic_pdf(pages: int) -> bytes:
    doc = pymupdf.open()
    for n in range(pages):
        doc.new_page().insert_text((72, 72), f"synthetic page {n + 1}")
    return doc.tobytes()


async def main() -> int:
    service.get_provider = lambda settings=None: _FakeProvider()

    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        suffix = uuid.uuid4().hex[:8]
        token, _ = await register(client, f"docgen_smoke_{suffix}", "correct_horse_battery_staple")
        other, _ = await register(client, f"docgen_other_{suffix}", "correct_horse_battery_staple")
        me = {"Authorization": f"Bearer {token}"}
        them = {"Authorization": f"Bearer {other}"}

        r = await client.post(
            "/docgen/sessions", json={"company_type": "masahma", "source_mode": "aoa_only"},
            headers=me,
        )
        check("مساهمة is switched off (400)", r.status_code == 400, str(r.status_code))

        r = await client.post(
            "/docgen/sessions", json={"company_type": "zmm", "source_mode": "aoa_only"},
            headers=me,
        )
        check("create session 201", r.status_code == 201, str(r.status_code))
        sid = r.json()["id"]
        base = f"/docgen/sessions/{sid}"

        try:
            r = await client.get(base, headers=them)
            check("another user's session is 404", r.status_code == 404, str(r.status_code))

            r = await client.post(
                f"{base}/uploads", data={"kind": "aoa"},
                files={"file": ("aoa.pdf", b"not a pdf", "application/pdf")}, headers=me,
            )
            check("a non-PDF upload is 400", r.status_code == 400, str(r.status_code))

            r = await client.post(
                f"{base}/uploads", data={"kind": "aoa"},
                files={"file": ("aoa.pdf", _synthetic_pdf(3), "application/pdf")}, headers=me,
            )
            check(
                "upload 201 with page count",
                r.status_code == 201 and r.json()["page_count"] == 3,
            )
            upload_id = r.json()["id"]

            r = await client.get(f"{base}/uploads/{upload_id}/pages/2", headers=me)
            check(
                "thumbnail is a no-store PNG",
                r.status_code == 200
                and r.headers["content-type"] == "image/png"
                and r.headers.get("cache-control") == "no-store",
            )
            r = await client.get(f"{base}/uploads/{upload_id}/pages/9", headers=me)
            check("thumbnail past the last page is 404", r.status_code == 404, str(r.status_code))
            r = await client.get(f"{base}/uploads/{upload_id}/pages/1", headers=them)
            check("another user's thumbnail is 404", r.status_code == 404, str(r.status_code))

            bad = {
                "entries": {"company_name": {"from": 5, "to": 4, "article": "x"}},
                "amended": [],
            }
            r = await client.put(f"{base}/page-map", json=bad, headers=me)
            problems = r.json().get("detail")
            check(
                "bad page map is 422 with a list of problems",
                r.status_code == 422 and isinstance(problems, list) and len(problems) >= 2,
                f"{r.status_code}, {len(problems) if isinstance(problems, list) else problems!r}",
            )

            r = await client.put(f"{base}/page-map", json=_PAGE_MAP, headers=me)
            check("page map accepted", r.status_code == 200, str(r.status_code))

            # Background OCR has run by the time ASGITransport returns.
            r = await client.get(base, headers=me)
            detail = r.json()
            check("session ready after OCR", detail["status"] == "ready", detail["status"])
            positions = [a["position"] for a in detail["articles"]]
            check("two articles, by position", positions == [0, 1])
            fields = detail["fields"]
            for name in ("company_name", "law_number", "company_address", "issued_capital"):
                check(f"{name} extracted", bool((fields.get(name) or {}).get("value")))
            check("partners table extracted", len(fields.get("attendees") or []) == 2)

            # A job killed by a restart: a live one is left alone, an old one
            # is re-claimable and shows the lawyer a retry message.
            await _force_ocr_running(sid, minutes_ago=1)
            await service.run_ocr_job(sid)
            r = await client.get(base, headers=me)
            check("a live OCR job is not re-claimed", r.json()["status"] == "ocr_running")
            await _force_ocr_running(sid, minutes_ago=60)
            await service.run_ocr_job(sid)
            r = await client.get(base, headers=me)
            check("an abandoned OCR job is re-claimed", r.json()["status"] == "ready")
            await _force_ocr_running(sid, minutes_ago=60)
            r = await client.get(base, headers=me)
            check(
                "an abandoned OCR job reads as failed",
                r.json()["status"] == "failed" and r.json()["error"] == service._ABANDONED_OCR,
            )
            r = await client.put(f"{base}/page-map", json=_PAGE_MAP, headers=me)
            r = await client.get(base, headers=me)
            check("and can be retried", r.json()["status"] == "ready", r.json()["status"])

            r = await client.post(f"{base}/render", headers=me)
            check("render without بعد التعديل is 409", r.status_code == 409, str(r.status_code))

            for position in (0, 1):
                r = await client.patch(
                    f"{base}/articles/{position}", json={"new_text": "نص تجريبي بعد التعديل."},
                    headers=me,
                )
                check(f"patch article {position}", r.status_code == 200, str(r.status_code))
            r = await client.patch(f"{base}/articles/7", json={"new_text": "x"}, headers=me)
            check("patch a missing position is 404", r.status_code == 404, str(r.status_code))

            r = await client.patch(
                f"{base}/fields",
                json={
                    "fields": {
                        "commercial_registration_no": "12345",
                        "commercial_registration_date": "2020/01/01",
                        "day_name": "الأحد",
                        "day_date": "2026/09/06",
                        "names_of_commissioners": "مفوض تجريبي",
                        "chairman_name": "رئيس تجريبي",
                        "meeting_time": "العاشرة صباحا",
                        "meeting_end_time": "الحادية عشرة صباحا",
                    }
                },
                headers=me,
            )
            check("patch fields", r.status_code == 200, str(r.status_code))

            r = await client.post(f"{base}/render", headers=me)
            check("render 200", r.status_code == 200, f"{r.status_code} {r.text[:200]}")

            r = await client.get(f"{base}/document", headers=me)
            check(
                "download is a .docx",
                r.status_code == 200 and r.content[:2] == b"PK"
                and "wordprocessingml" in r.headers["content-type"],
            )
            r = await client.get(f"{base}/document", headers=them)
            check("another user's document is 404", r.status_code == 404, str(r.status_code))

            r = await client.patch(
                f"{base}/articles/0", json={"new_text": "نص معدل."}, headers=me
            )
            r = await client.get(f"{base}/document", headers=me)
            check("an edit invalidates the rendered document", r.status_code == 404)
        finally:
            r = await client.delete(base, headers=me)
            check("delete 204", r.status_code == 204, str(r.status_code))
            r = await client.get(base, headers=me)
            check("deleted session is 404", r.status_code == 404, str(r.status_code))

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main()))
