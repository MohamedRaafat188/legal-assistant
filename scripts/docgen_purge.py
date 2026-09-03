"""Delete expired docgen uploads, OCR text, and article text.

Session rows survive in `expired` form for audit; the documents that carry
national ID and passport numbers do not. Run on a schedule (Railway cron,
daily is ample for a 2-day retention window).

Usage: python scripts/docgen_purge.py
Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from legal_assistant.db.session import get_engine, get_sessionmaker
from legal_assistant.docgen.service import purge_expired

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger("docgen_purge")


async def main() -> int:
    sessionmaker = get_sessionmaker()
    try:
        async with sessionmaker() as db:
            count = await purge_expired(db)
            await db.commit()
        _log.info("purged %d expired docgen session(s)", count)
        return 0
    except Exception:
        # `purge_expired` itself is content-free by construction (see its own
        # docstring): the only unguarded DB calls inside it bind session ids
        # and an empty dict, never document text, so a SQLAlchemyError's
        # bound-parameter repr cannot carry a national ID or passport number
        # here. `logging.exception` below is therefore safe to use as-is.
        _log.exception("docgen purge failed")
        return 1
    finally:
        await get_engine().dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
