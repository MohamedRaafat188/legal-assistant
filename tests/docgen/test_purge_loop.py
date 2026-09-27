import asyncio

import pytest

from legal_assistant.docgen import purge_loop


def test_purge_forever_keeps_going_after_a_failed_pass(monkeypatch):
    passes = []

    async def flaky_purge_once():
        passes.append(len(passes))
        if len(passes) == 1:
            raise RuntimeError("database unreachable")
        return 0

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise asyncio.CancelledError

    class _FakeSettings:
        docgen_purge_interval_minutes = 60

    monkeypatch.setattr(purge_loop, "purge_once", flaky_purge_once)
    monkeypatch.setattr(purge_loop, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(purge_loop.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(purge_loop.purge_forever())

    assert len(passes) == 2  # the failure did not end the loop
    assert sleeps == [3600, 3600]
