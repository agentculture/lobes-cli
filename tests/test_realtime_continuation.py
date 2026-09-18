"""Early commit + continuation merge (hebrew-realtime d9, layer B).

With a short confirming silence the machine sometimes starts answering a
speaker who was only pausing. When the speaker RESUMES shortly after such a
commit, that is a continuation of the same turn, not a barge-in on the reply:
the reply stops at once (even inside the barge-in guard window), the aborted
half-turn leaves history entirely, and the route re-transcribes both halves
as one turn.
"""

from __future__ import annotations

import lobes.realtime._floor as F
import lobes.realtime._session as S
from lobes.realtime._conversation import ConversationBridge, GenerateConfig

CHUNK = 4800
WINDOW = 1200


class FakeClock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms


def make_bridge(continuation_window_ms: int = WINDOW):
    clock = FakeClock()
    cancels = {"generate": 0, "tts": 0}
    session, _ = S.Session.create(S.parse_session_config({}))
    bridge = ConversationBridge(
        session,
        cancel_generate=lambda: cancels.__setitem__("generate", cancels["generate"] + 1),
        cancel_tts=lambda: cancels.__setitem__("tts", cancels["tts"] + 1),
        generate=GenerateConfig(base_url="http://gw:8000", stream=True),
        clock=clock,
        chunk_bytes=CHUNK,
        continuation_window_ms=continuation_window_ms,
    )
    bridge.arm()
    return bridge, clock, cancels


def commit(bridge, clock, text="אני רוצה", reason="silence") -> int:
    bridge.on_speech_started()
    clock.advance(900)
    bridge.on_speech_stopped(reason=reason)
    bridge.on_transcript(text)
    turn_id = bridge.take_pending_response()
    assert turn_id is not None
    bridge.begin_generate_stream(turn_id)
    bridge.drain()
    return turn_id


def types_of(bridge) -> list[str]:
    return [p["type"] for p in bridge.drain()]


def test_off_by_default_the_guard_window_still_swallows_an_early_onset():
    bridge, clock, cancels = make_bridge(continuation_window_ms=0)
    commit(bridge, clock)
    clock.advance(300)
    assert bridge.on_speech_started() is False
    assert "response.interrupted" not in types_of(bridge)
    assert bridge.session.get_history() == [{"role": "user", "content": "אני רוצה"}]
    assert cancels == {"generate": 0, "tts": 0}


def test_a_resumed_speaker_stops_the_reply_inside_the_guard_window_and_leaves_no_trace():
    bridge, clock, cancels = make_bridge()
    turn_id = commit(bridge, clock)
    bridge.on_generate_delta("בטח, מה תרצה לדעת? ", turn_id=turn_id)
    clock.advance(300)  # well inside BARGE_IN_WINDOW_MS (750)
    assert bridge.on_speech_started() is True
    assert "response.interrupted" in types_of(bridge)
    assert cancels["generate"] == 1 and cancels["tts"] == 1
    # the aborted half-turn is gone: no user entry, no heard-prefix assistant entry
    assert bridge.session.get_history() == []
    assert bridge.floor.state is F.FloorState.LISTENING


def test_the_merged_turn_is_then_an_ordinary_turn():
    bridge, clock, _ = make_bridge()
    commit(bridge, clock)
    clock.advance(300)
    assert bridge.on_speech_started() is True
    clock.advance(1500)
    bridge.on_speech_stopped(reason="silence")
    bridge.on_transcript("אני רוצה לדעת מה השעה")
    assert bridge.take_pending_response() is not None
    assert bridge.session.get_history() == [{"role": "user", "content": "אני רוצה לדעת מה השעה"}]


def test_heard_audio_is_dropped_too_because_the_whole_exchange_is_being_redone():
    bridge, clock, _ = make_bridge()
    turn_id = commit(bridge, clock)
    bridge.on_generate_delta("בטח, מה תרצה לדעת? ", turn_id=turn_id)
    segment = bridge.take_pending_segment()
    assert segment is not None
    bridge.on_tts_audio(bytes(CHUNK * 4), turn_id=turn_id, segment_index=segment[1])
    assert bridge.deliver_next(turn_id=turn_id)
    clock.advance(400)
    assert bridge.on_speech_started() is True
    assert bridge.session.get_history() == []


def test_after_the_window_it_is_an_ordinary_barge_in():
    bridge, clock, _ = make_bridge()
    commit(bridge, clock)
    clock.advance(WINDOW + 1)
    assert bridge.on_speech_started() is False
    assert "response.interrupted" in types_of(bridge)
    assert bridge.session.get_history() == [{"role": "user", "content": "אני רוצה"}]


def test_a_max_turn_commit_is_never_continued():
    bridge, clock, _ = make_bridge()
    commit(bridge, clock, reason="max_turn")
    clock.advance(300)
    assert bridge.on_speech_started() is False
    assert bridge.session.get_history() == [{"role": "user", "content": "אני רוצה"}]


def test_a_tool_call_already_sent_cannot_be_taken_back():
    from lobes.realtime._turn import ToolCallResult

    bridge, clock, _ = make_bridge()
    turn_id = commit(bridge, clock)
    bridge.on_generate_stream_end(
        ToolCallResult(call_id="c1", name="f", arguments="{}", tool_call_count=1), turn_id=turn_id
    )
    assert bridge.awaiting_tool_result
    history = bridge.session.get_history()
    clock.advance(200)
    assert bridge.on_speech_started() is False
    assert bridge.session.get_history() == history
    assert bridge.awaiting_tool_result


def test_a_reply_that_already_finished_is_not_rewound():
    bridge, clock, _ = make_bridge()
    turn_id = commit(bridge, clock)
    bridge.on_generate_delta("כן. ", turn_id=turn_id)
    bridge.on_generate_stream_end("כן.", turn_id=turn_id)
    while (segment := bridge.take_pending_segment()) is not None:
        bridge.on_tts_audio(bytes(CHUNK), turn_id=turn_id, segment_index=segment[1])
    while bridge.deliver_next(turn_id=turn_id):
        pass
    assert bridge.floor.state is F.FloorState.LISTENING
    history = bridge.session.get_history()
    assert history[-1]["role"] == "assistant"
    clock.advance(100)
    assert bridge.on_speech_started() is False
    assert bridge.session.get_history() == history


def test_an_unarmed_session_never_continues():
    bridge, clock, _ = make_bridge()
    bridge.armed = False
    bridge.on_speech_started()
    bridge.on_speech_stopped(reason="silence")
    bridge.on_transcript("שלום")
    clock.advance(100)
    assert bridge.on_speech_started() is False


def test_session_history_pop_only_removes_an_exact_last_entry():
    session, _ = S.Session.create(S.parse_session_config({}))
    session.append_history("user", "א")
    session.append_history("assistant", "ב")
    assert session.pop_history_if_last("user", "א") is False
    assert session.pop_history_if_last("assistant", "ב") is True
    assert session.get_history() == [{"role": "user", "content": "א"}]
    assert session.pop_history_if_last("user", "x") is False
