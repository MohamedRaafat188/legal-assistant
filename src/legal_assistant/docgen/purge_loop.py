"""Run the retention purge inside the web process, on a timer.

Uploads live on the web service's own volume, and a Railway volume mounts on
exactly one service -- a separate cron service would purge the database rows
but could never reach the files. So the purge runs here: once at startup
(catching up on anything that expired while the app was down), then every
`docgen_purge_interval_minutes`. `scripts/docgen_purge.py` stays for manual
runs against the same storage directory.
"""

from __future__ import annotations

import asyncio
import logging

from legal_assistant.config import get_settings
from legal_assistant.db.session import get_sessionmaker
from legal_assistant.docgen.service import purge_expired

_log = logging.getLogger(__name__)


async def purge_once() -> int:
    async with get_sessionmaker()() as db:
        count = await purge_expired(db)
        await db.commit()
    return count


async def purge_forever() -> None:
    """Never returns; cancelled at shutdown. One failed pass never stops the
    next: the sessions it missed stay eligible."""
    interval = get_settings().docgen_purge_interval_minutes * 60
    while True:
        try:
            count = await purge_once()
            if count:
                _log.info("docgen purge: purged %d expired session(s)", count)
        except Exception as e:  # noqa: BLE001 -- the loop must survive a bad pass
            # Type only, the same content-free convention as `service`.
            _log.error("docgen purge pass failed: %s", type(e).__name__)
        await asyncio.sleep(interval)
