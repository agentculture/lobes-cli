"""Hidden speculation's pure pieces (hebrew-realtime d9, layer A)."""

from __future__ import annotations

import asyncio

import pytest

import lobes.realtime._session as S
from lobes.realtime._conversation import ConversationBridge, GenerateConfig
from lobes.realtime._speculation import LineBuffer, SegmentAudioCache, can_adopt


def make_bridge(**kwargs) -> ConversationBridge:
    session, _ = S.Session.create(S.parse_session_config({}))
    return ConversationBridge(
        session,
        cancel_generate=lambda: None,
        cancel_tts=lambda: None,
        generate=GenerateConfig(base_url="http://gw:8000", **kwargs),
        clock=lambda: 0,
        chunk_bytes=4800,
    )


# --- the speculative request --------------------------------------------------


def test_no_speculation_for_an_unarmed_session():
    assert make_bridge(stream=True).build_speculative_request("שלום") is None


def test_no_speculation_without_streaming():
    bridge = make_bridge(stream=False)
    bridge.arm()
    assert bridge.build_speculative_request("שלום") is None


def test_no_speculation_for_an_empty_transcript():
    bridge = make_bridge(stream=True)
    bridge.arm()
    assert bridge.build_speculative_request("  ") is None


def test_speculating_leaves_no_trace_and_predicts_the_real_request_exactly():
    bridge = make_bridge(stream=True)
    bridge.arm()
    bridge.on_speech_started()
    bridge.drain()
    history_before = bridge.session.get_history()
    speculative = bridge.build_speculative_request("מה השעה")
    assert speculative is not None
    assert bridge.session.get_history() == history_before
    assert bridge.session.get_history() == []
    assert bridge.drain() == []

    bridge.on_speech_stopped()
    bridge.on_transcript("מה השעה")
    turn_id = bridge.take_pending_response()
    real = bridge.build_generate_request(turn_id)
    assert can_adopt(speculative, real)
    assert speculative.body == real.body
    assert speculative.url == real.url


def test_a_different_final_transcript_is_not_adoptable():
    bridge = make_bridge(stream=True)
    bridge.arm()
    bridge.on_speech_started()
    speculative = bridge.build_speculative_request("מה")
    bridge.on_speech_stopped()
    bridge.on_transcript("מה השעה")
    real = bridge.build_generate_request(bridge.take_pending_response())
    assert not can_adopt(speculative, real)


def test_nothing_is_adoptable_from_nothing():
    assert not can_adopt(None, None)


def test_no_speculation_while_a_tool_result_is_outstanding_or_a_reply_is_running():
    bridge = make_bridge(stream=True)
    bridge.arm()
    bridge.on_speech_started()
    bridge.on_speech_stopped()
    bridge.on_transcript("מה השעה")
    bridge.take_pending_response()
    # the floor is RESPONDING: a pause now belongs to a barge-in, not a new turn
    assert bridge.build_speculative_request("רגע") is None


# --- the line buffer: buffered first, then live -------------------------------


def test_a_late_reader_gets_buffered_lines_then_live_ones_then_the_end():
    async def scenario() -> list[str]:
        buf = LineBuffer()
        buf.put("a")
        buf.put("b")
        got: list[str] = []

        async def read() -> None:
            async for line in buf:
                got.append(line)

        reader = asyncio.create_task(read())
        await asyncio.sleep(0)
        buf.put("c")
        buf.close()
        await reader
        return got

    assert asyncio.run(scenario()) == ["a", "b", "c"]


def test_a_failed_producer_surfaces_in_the_reader():
    async def scenario() -> None:
        buf = LineBuffer()
        buf.put("a")
        buf.close(error=RuntimeError("backend gone"))
        async for _ in buf:
            pass

    coro = scenario()
    with pytest.raises(RuntimeError, match="backend gone"):
        asyncio.run(coro)


# --- the audio cache -----------------------------------------------------------


def test_a_reserved_segment_is_awaited_not_resynthesized():
    async def scenario() -> bytes | None:
        cache = SegmentAudioCache()
        slot = cache.reserve("שלום.")
        assert cache.reserve("שלום.") is None  # already in flight: no second synth
        waiter = asyncio.create_task(cache.get("שלום."))
        await asyncio.sleep(0)
        slot.set_result(b"\x01\x02")
        return await waiter

    assert asyncio.run(scenario()) == b"\x01\x02"


def test_an_unknown_segment_is_a_miss():
    assert asyncio.run(SegmentAudioCache().get("אחר.")) is None


def test_a_failed_or_cancelled_speculative_synth_is_a_miss_not_an_error():
    async def scenario() -> tuple:
        cache = SegmentAudioCache()
        cache.reserve("א.").set_exception(RuntimeError("tts down"))
        cache.reserve("ב.").cancel()
        cache.reserve("ג.").set_result(b"")
        return await cache.get("א."), await cache.get("ב."), await cache.get("ג.")

    assert asyncio.run(scenario()) == (None, None, None)


def test_cancelling_the_waiter_raises_in_it_and_leaves_the_slot_pending():
    # The one CancelledError this cache must never swallow: OUR caller's.
    # A barge-in cancels the real synth worker, and that cancellation has to
    # reach it — while the slot itself stays in flight for anyone else.
    async def scenario() -> None:
        cache = SegmentAudioCache()
        slot = cache.reserve("שלום.")
        waiter = asyncio.create_task(cache.get("שלום."))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not slot.done()

    asyncio.run(scenario())
