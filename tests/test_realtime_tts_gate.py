"""Offline proof that the TTS voice lane is isolated from the batch lane.

Issue #151 t7: once a /v1/realtime session can speak replies, a shared
concurrency gate means a voice reply can queue behind unrelated batch
``POST /v1/audio/speech`` work — dead air in a spoken turn. The fix in
``lobes/realtime/tts_client.py`` is two independent ``asyncio.Semaphore``
pools instead of one shared/raised one.

``tts_client.py`` imports ``httpx`` at module top, so it only imports inside
the ``realtime`` container (see ``pyproject.toml``'s ``[tool.coverage.run]``
omit list and ``tests/test_realtime_imports.py``) — no CI lane installs the
``[realtime]`` extra, so a test that imports ``tts_client`` directly would
never actually run. This file never imports it either. Instead it exercises
``lobes.realtime._settings.new_tts_lane_semaphores`` — the pure, stdlib-only
(``asyncio`` only; no I/O) semaphore builder that ``tts_client.py``'s own
lazily-built per-lane registry calls into. Testing THIS function tests the
real production isolation guarantee, not a stand-in for it: `asyncio` needs
no running event loop to construct a ``Semaphore`` (true since Python 3.10),
so this is provable fully offline.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

import lobes.realtime._settings as _settings
from lobes.realtime._settings import (
    BATCH_LANE,
    VOICE_LANE,
    build_settings,
    new_tts_lane_semaphores,
)


def test_batch_and_voice_lanes_get_independent_semaphore_objects() -> None:
    s = build_settings({})
    sems = new_tts_lane_semaphores(s)
    assert sems[BATCH_LANE] is not sems[VOICE_LANE]
    assert isinstance(sems[BATCH_LANE], asyncio.Semaphore)
    assert isinstance(sems[VOICE_LANE], asyncio.Semaphore)


def test_voice_lane_synthesis_does_not_queue_behind_a_saturated_batch_lane() -> None:
    """Acceptance criterion (issue #151 t7): the SHIPPED default isolates the lanes.

    Saturates the batch lane's only permit (TTS_CONCURRENCY defaults to 1,
    unchanged by this task — the batch route stays byte-identical) and
    proves a voice-lane acquire still succeeds immediately: it must never
    wait on a permit the batch lane is holding.
    """

    async def _run() -> None:
        s = build_settings({})  # the shipped defaults, nothing overridden
        sems = new_tts_lane_semaphores(s)
        batch_sem = sems[BATCH_LANE]
        voice_sem = sems[VOICE_LANE]

        # Saturate the batch lane exactly as a batch /v1/audio/speech caller
        # holding its only permit would.
        await batch_sem.acquire()
        try:
            # A voice-lane synthesis acquires on a SEPARATE semaphore, so
            # this is uncontended and must return well within any
            # reasonable timeout — never wait on the batch lane at all.
            try:
                await asyncio.wait_for(voice_sem.acquire(), timeout=0.5)
            except asyncio.TimeoutError:
                pytest.fail(
                    "voice-lane semaphore acquire timed out while the batch "
                    "lane was saturated — the two lanes are not isolated"
                )
            else:
                voice_sem.release()
        finally:
            batch_sem.release()

    asyncio.run(_run())


def test_batch_lane_still_serializes_a_second_batch_caller_at_the_default() -> None:
    """Positive control: the batch lane is still a real gate, not a no-op.

    Without this, the isolation test above would trivially pass even if
    BOTH semaphores were unbounded — this proves the batch semaphore still
    blocks a SECOND batch-lane caller at the shipped TTS_CONCURRENCY=1
    default, exactly as before this task (the batch route is unchanged).
    """

    async def _run() -> None:
        s = build_settings({})
        sems = new_tts_lane_semaphores(s)
        batch_sem = sems[BATCH_LANE]

        await batch_sem.acquire()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(batch_sem.acquire(), timeout=0.2)
        finally:
            batch_sem.release()

    asyncio.run(_run())


def test_lane_semaphore_sizes_follow_their_own_env_override() -> None:
    """The two pools size independently — raising one must not affect the other."""

    async def _run() -> None:
        s = build_settings({"TTS_CONCURRENCY": "1", "TTS_VOICE_CONCURRENCY": "2"})
        sems = new_tts_lane_semaphores(s)
        batch_sem = sems[BATCH_LANE]
        voice_sem = sems[VOICE_LANE]

        # Batch: still only 1 permit. The failed/cancelled second acquire
        # never consumed a permit, so only ONE release is correct here.
        await batch_sem.acquire()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(batch_sem.acquire(), timeout=0.2)
        batch_sem.release()

        # Voice: 2 permits — a second concurrent voice acquire must succeed.
        await voice_sem.acquire()
        await asyncio.wait_for(voice_sem.acquire(), timeout=0.2)
        voice_sem.release()
        voice_sem.release()

    asyncio.run(_run())


def test_the_client_pool_is_only_ever_keyed_by_a_normalized_lane() -> None:
    """``_get_client``/``_reset_client`` must normalize before touching ``_clients``.

    ``_get_semaphore`` normalizes its lane, so an unknown value is gated by the
    BATCH semaphore. If the client pool were keyed by the raw string instead,
    that same request would be gated as batch while talking over a third,
    never-closed ``httpx.AsyncClient`` — a leaked connection pool per typo, and
    the exact opposite of the "unknown lane -> batch lane" contract
    ``normalize_tts_lane`` documents.

    Structural (AST) rather than behavioural on purpose: ``tts_client`` imports
    httpx at module top and no CI workflow installs the ``[realtime]`` extra, so
    an ``importorskip`` version of this test would skip in CI and gate nothing
    (the same trap this module's lane-selection logic was moved into
    ``_settings`` to avoid).
    """
    source = Path(_settings.__file__).parent.joinpath("tts_client.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    guarded = {"_get_client", "_reset_client"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in guarded:
            continue
        seen.add(node.name)
        calls = [
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        ]
        assert "normalize_tts_lane" in calls, (
            f"{node.name} indexes _clients without normalizing its lane — an unknown "
            "lane would open a third, never-closed connection pool"
        )

    assert seen == guarded, f"expected to find {guarded} in tts_client.py, found {seen}"
