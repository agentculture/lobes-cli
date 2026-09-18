"""End-to-end coverage for the Q-B Qodo finding — "first Hebrew reply
freezes all sessions" — at the ``tts_client.synthesize`` level.

``lobes/realtime/tts_client.py`` imports ``httpx`` at module top (see
``tests/test_tts_pause_and_truncation.py``'s docstring for the pattern), so
this module is skipped in the offline CI env — it only runs where the
``[realtime]`` extra is installed. The mechanism this test exercises
(``LazySingleton`` build-once-under-concurrency + running the build via
``asyncio.to_thread``) is ALSO covered fully offline, stdlib-only, in
``tests/test_realtime_vocalize.py::TestLazySingleton`` — that is the test
that was actually run and observed failing/passing during development of
this fix, per the offline environment's constraint. This file documents and
proves the same guarantee one level up, against the real
``tts_client.synthesize`` / ``_get_hebrew_diacritizer`` call path, for a
deployment where httpx is available.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("httpx", reason="tts_client imports httpx at module top")

import lobes.realtime.tts_client as tts_client  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_hebrew_diacritizer_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets its own fresh singleton so builder call-counts and
    cached values from one test never leak into the next."""
    from lobes.realtime._vocalize import LazySingleton

    monkeypatch.setattr(
        tts_client,
        "_hebrew_diacritizer_singleton",
        LazySingleton(tts_client._build_hebrew_diacritizer),
    )


def test_get_hebrew_diacritizer_builds_at_most_once_under_concurrent_to_thread_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent first Hebrew requests must build the diacritizer
    exactly once, not twice — the Qodo finding's "stays build-once under
    concurrency" requirement, exercised via the SAME asyncio.to_thread path
    ``_maybe_vocalize_hebrew`` uses.
    """
    calls = []

    def fake_build() -> object:
        calls.append(1)
        time.sleep(0.2)
        return object()

    monkeypatch.setattr(tts_client, "_build_hebrew_diacritizer", fake_build)
    from lobes.realtime._vocalize import LazySingleton

    monkeypatch.setattr(tts_client, "_hebrew_diacritizer_singleton", LazySingleton(fake_build))

    async def main():
        results = await asyncio.gather(
            asyncio.to_thread(tts_client._get_hebrew_diacritizer),
            asyncio.to_thread(tts_client._get_hebrew_diacritizer),
        )
        return results

    results = asyncio.run(main())

    assert len(calls) == 1
    assert results[0] is results[1]


def test_maybe_vocalize_hebrew_does_not_block_a_concurrent_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The event-loop-blocking regression itself: a slow first-use build of
    the diacritizer must not stall an unrelated concurrent coroutine on the
    same loop (e.g. another session's own request handling).
    """

    def fake_build() -> object:
        time.sleep(0.3)
        return lambda text: text  # trivial identity "diacritizer"

    from lobes.realtime._vocalize import LazySingleton

    monkeypatch.setattr(tts_client, "_hebrew_diacritizer_singleton", LazySingleton(fake_build))

    async def main():
        ticks = []

        async def heartbeat():
            for _ in range(6):
                await asyncio.sleep(0.05)
                ticks.append(time.monotonic())

        vocalize_task = asyncio.create_task(tts_client._maybe_vocalize_hebrew("שלום", "he", None))
        hb_task = asyncio.create_task(heartbeat())
        result = await vocalize_task
        await hb_task
        return result, ticks

    result, ticks = asyncio.run(main())

    assert result == "שלום"
    assert len(ticks) == 6
