"""The bridge's STREAMING surface — deltas in, sentences out, one text.done.

The non-streaming surface (``on_generate_response`` → ``take_pending_synthesis``
→ ``on_tts_audio``) is untouched and is what ``GENERATE_STREAM=false``
selects; this file covers the twin that ships beside it (approved deviation
d7, 2026-09-18):

``begin_generate_stream`` → ``on_generate_delta`` (per text delta) →
``on_generate_stream_end`` (with the accumulator's result), with the route
draining ``take_pending_segment()`` and answering each with
``on_tts_audio(..., segment_index=N)``.

Three properties are worth more than the rest:

- **one ``response.text.done``**, carrying the FULL reply, emitted once at
  stream end — the wire contract does not change because the transport did;
- **a text prefix followed by a tool call is never spoken** — the queued
  segments are cancelled and the tool call proceeds exactly as it does
  without streaming;
- **history records what was HEARD**, across segments, after a barge-in.
"""

from __future__ import annotations

import json

import lobes.realtime._floor as F
import lobes.realtime._session as S
from lobes.realtime._conversation import ConversationBridge, GenerateConfig
from lobes.realtime._turn import StreamAccumulator, ToolCallResult

CHUNK = 4800


def pcm(n: int) -> bytes:
    return bytes((i * 7) % 251 for i in range(n))


class FakeClock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms


def make_bridge(**kwargs) -> tuple[ConversationBridge, FakeClock, dict]:
    clock = FakeClock()
    cancels = {"generate": 0, "tts": 0}
    session, _created = S.Session.create(S.parse_session_config({}))
    bridge = ConversationBridge(
        session,
        cancel_generate=lambda: cancels.__setitem__("generate", cancels["generate"] + 1),
        cancel_tts=lambda: cancels.__setitem__("tts", cancels["tts"] + 1),
        generate=GenerateConfig(base_url="http://gw:8000", **kwargs),
        clock=clock,
        chunk_bytes=CHUNK,
    )
    return bridge, clock, cancels


def open_turn(bridge: ConversationBridge, clock: FakeClock, text: str = "מה השעה") -> int:
    bridge.arm()
    bridge.on_speech_started()
    clock.advance(F.DEFAULT_BARGE_IN_WINDOW_MS)
    bridge.on_speech_stopped()
    bridge.on_transcript(text)
    turn_id = bridge.take_pending_response()
    assert turn_id is not None
    return turn_id


def types_of(bridge: ConversationBridge) -> list[str]:
    return [payload["type"] for payload in bridge.drain()]


# --- the payload knob --------------------------------------------------------


def test_the_generate_request_carries_stream_only_when_configured() -> None:
    bridge, clock, _ = make_bridge()
    turn_id = open_turn(bridge, clock)
    assert bridge.build_generate_request(turn_id).body.get("stream") is None
    assert bridge.streaming_enabled is False

    streamed, clock2, _ = make_bridge(stream=True)
    turn2 = open_turn(streamed, clock2)
    assert streamed.build_generate_request(turn2).body["stream"] is True
    assert streamed.streaming_enabled is True


# --- the happy path ----------------------------------------------------------


def test_sentences_reach_the_synthesis_queue_while_the_stream_is_still_running() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)

    assert bridge.on_generate_delta("השעה עכשיו ארבע וחצי אחר הצהריים.", turn_id=turn_id) is True
    assert bridge.take_pending_segment() is None  # the boundary is not confirmed yet
    bridge.on_generate_delta(" ", turn_id=turn_id)

    pending = bridge.take_pending_segment()
    assert pending == (turn_id, 0, "השעה עכשיו ארבע וחצי אחר הצהריים.")
    assert bridge.take_pending_segment() is None  # taken exactly once
    assert bridge.floor.state is F.FloorState.SPEAKING


def test_one_text_done_carries_the_whole_reply_at_stream_end() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    full = "משפט ראשון ארוך מספיק. משפט שני ארוך מספיק."
    for piece in (full[:20], full[20:]):
        bridge.on_generate_delta(piece, turn_id=turn_id)
    assert bridge.on_generate_stream_end(full, turn_id=turn_id) is True

    payloads = bridge.drain()
    text_done = [p for p in payloads if p["type"] == S.EventType.RESPONSE_TEXT_DONE.value]
    assert len(text_done) == 1
    assert text_done[0]["text"] == full


def test_a_streamed_reply_completes_and_records_the_full_text_in_history() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    full = "משפט ראשון ארוך מספיק. משפט שני ארוך מספיק."
    bridge.on_generate_delta(full, turn_id=turn_id)
    bridge.on_generate_stream_end(full, turn_id=turn_id)

    index = 0
    while True:
        pending = bridge.take_pending_segment()
        if pending is None:
            break
        _turn, segment_index, _text = pending
        assert segment_index == index
        assert bridge.on_tts_audio(pcm(CHUNK), turn_id=turn_id, segment_index=segment_index) is True
        index += 1
    assert index == 2

    while bridge.deliver_next(turn_id=turn_id):
        pass
    assert bridge.session.get_history()[-1] == {"role": "assistant", "content": full}
    assert S.EventType.RESPONSE_DONE.value in types_of(bridge)


def test_the_flushed_tail_becomes_the_final_segment() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    # No terminal punctuation at all: everything arrives through the flush.
    bridge.on_generate_delta("אין כאן שום סימן פיסוק", turn_id=turn_id)
    assert bridge.take_pending_segment() is None
    bridge.on_generate_stream_end("אין כאן שום סימן פיסוק", turn_id=turn_id)
    assert bridge.take_pending_segment() == (turn_id, 0, "אין כאן שום סימן פיסוק")


def test_an_empty_streamed_reply_is_the_named_generate_failure() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    bridge.on_generate_stream_end("", turn_id=turn_id)

    payloads = bridge.drain()
    errors = [p for p in payloads if p["type"] == S.EventType.ERROR.value]
    assert errors
    assert errors[0]["code"] == S.ErrorCode.GENERATE_FAILED.value


def test_deltas_for_a_stale_turn_are_ignored() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    assert bridge.on_generate_delta("משהו", turn_id=turn_id + 5) is False
    assert bridge.on_generate_stream_end("משהו", turn_id=turn_id + 5) is False
    assert bridge.take_pending_segment() is None


# --- text, then a tool call --------------------------------------------------


def test_a_text_prefix_before_a_tool_call_is_cancelled_and_never_spoken() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    bridge.on_generate_delta("רגע אחד בבקשה. ", turn_id=turn_id)
    assert bridge.floor.state is F.FloorState.SPEAKING  # a segment was released

    result = ToolCallResult(call_id="call_1", name="get_time", arguments="{}")
    assert bridge.on_generate_stream_end(result, turn_id=turn_id) is True

    # The queued segment is gone, the floor waits on the client, and the
    # abandoned prefix is nowhere in history.
    assert bridge.take_pending_segment() is None
    assert bridge.take_pending_synthesis() is None
    assert bridge.awaiting_tool_result is True
    assert bridge.floor.state is F.FloorState.TOOL_WAIT
    assert all(msg.get("content") != "רגע אחד בבקשה." for msg in bridge.session.get_history())

    types = types_of(bridge)
    assert S.EventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE.value in types
    assert S.EventType.RESPONSE_AUDIO_DELTA.value not in types
    assert S.EventType.RESPONSE_TEXT_DONE.value not in types


def test_a_tool_call_with_no_text_prefix_takes_exactly_the_non_streaming_path() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    result = ToolCallResult(call_id="call_1", name="get_time", arguments='{"tz":"UTC"}')
    assert bridge.on_generate_stream_end(result, turn_id=turn_id) is True

    assert bridge.outstanding_tool_call.call_id == "call_1"
    assert bridge.session.get_history()[-1]["tool_calls"][0]["id"] == "call_1"


def test_the_tool_result_round_trip_still_closes_the_turn() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    bridge.on_generate_stream_end(
        ToolCallResult(call_id="call_1", name="now", arguments="{}"), turn_id=turn_id
    )
    bridge.drain()

    assert (
        bridge.on_function_call_output(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": json.dumps({"time": "16:30"}),
                },
            }
        )
        is True
    )
    bridge.arm()
    assert bridge.take_pending_response() == turn_id


# --- barge-in mid-stream -----------------------------------------------------


def test_a_barge_in_mid_stream_records_only_the_heard_segments() -> None:
    bridge, clock, cancels = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    bridge.on_generate_delta("משפט ראשון ארוך מספיק. משפט שני ארוך מספיק. ", turn_id=turn_id)

    first = bridge.take_pending_segment()
    second = bridge.take_pending_segment()
    assert first is not None
    assert second is not None
    bridge.on_tts_audio(pcm(CHUNK), turn_id=turn_id, segment_index=0)
    bridge.on_tts_audio(pcm(CHUNK * 4), turn_id=turn_id, segment_index=1)
    assert bridge.deliver_next(turn_id=turn_id) is True  # all of segment 0
    assert bridge.deliver_next(turn_id=turn_id) is True  # a quarter of segment 1

    clock.advance(F.DEFAULT_BARGE_IN_WINDOW_MS)
    bridge.on_speech_started()

    assert cancels["generate"] == 1
    assert cancels["tts"] == 1
    spoken = [m for m in bridge.session.get_history() if m.get("role") == "assistant"]
    assert spoken
    assert spoken[-1]["content"].startswith("משפט ראשון ארוך מספיק.")
    assert "משפט שני ארוך מספיק." not in spoken[-1]["content"]
    assert S.EventType.RESPONSE_INTERRUPTED.value in types_of(bridge)


# --- the two surfaces do not interfere --------------------------------------


def test_the_non_streaming_surface_is_unchanged_when_streaming_is_configured() -> None:
    # GENERATE_STREAM only decides which path the ROUTE drives; the bridge
    # answers a whole-reply response exactly as it always has.
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    body = json.dumps({"choices": [{"message": {"content": "השעה ארבע"}}]}).encode()
    assert bridge.on_generate_response(200, body, turn_id=turn_id) is True
    assert bridge.take_pending_synthesis() == (turn_id, "השעה ארבע")
    assert bridge.take_pending_segment() is None


def test_a_streamed_response_is_still_in_progress_until_it_completes() -> None:
    # The route's delivery pump needs a stdlib-owned answer to "keep going?".
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    assert bridge.response_in_progress(turn_id) is True
    bridge.begin_generate_stream(turn_id)
    bridge.on_generate_delta("משפט ראשון ארוך מספיק. ", turn_id=turn_id)
    assert bridge.response_in_progress(turn_id) is True
    bridge.on_generate_stream_end("משפט ראשון ארוך מספיק.", turn_id=turn_id)
    bridge.take_pending_segment()
    bridge.on_tts_audio(pcm(CHUNK), turn_id=turn_id, segment_index=0)
    while bridge.deliver_next(turn_id=turn_id):
        pass
    assert bridge.response_in_progress(turn_id) is False


def test_a_tool_wait_is_not_a_delivery_pump_the_route_should_keep_running() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    bridge.on_generate_stream_end(
        ToolCallResult(call_id="call_1", name="now", arguments="{}"), turn_id=turn_id
    )
    assert bridge.response_in_progress(turn_id) is False


def test_the_accumulator_and_the_bridge_compose_end_to_end() -> None:
    bridge, clock, _ = make_bridge(stream=True)
    turn_id = open_turn(bridge, clock)
    bridge.begin_generate_stream(turn_id)
    acc = StreamAccumulator()
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]})
        for piece in ("משפט ראשון ארוך מספיק. ", "משפט שני ארוך מספיק.")
    ] + ["data: [DONE]"]
    for line in lines:
        for item in acc.feed_line(line):
            bridge.on_generate_delta(item.text, turn_id=turn_id)
    bridge.on_generate_stream_end(acc.result(), turn_id=turn_id)

    assert bridge.take_pending_segment()[2] == "משפט ראשון ארוך מספיק."
    assert bridge.take_pending_segment()[2] == "משפט שני ארוך מספיק."
    assert bridge.take_pending_segment() is None


def test_generate_config_threads_the_first_clause_threshold_into_the_chunker():
    from lobes.realtime._conversation import GenerateConfig
    from lobes.realtime._sentences import DEFAULT_EAGER_FIRST_MIN_CHARS

    assert GenerateConfig(base_url="http://x").first_clause_min_chars == (
        DEFAULT_EAGER_FIRST_MIN_CHARS
    )
