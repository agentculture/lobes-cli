"""The floor, generalized from one audio buffer to a QUEUE of segments.

Sentence-level streaming (approved deviation d7, 2026-09-18) means text
segments arrive over TIME while earlier ones are still synthesizing or
delivering. ``_floor.py`` therefore grew:

- :meth:`~lobes.realtime._floor.Floor.on_reply_segment` — append a segment,
  the last one marked ``final``. ``on_reply_text`` is the one-segment
  convenience and keeps its exact pre-streaming behaviour (every test in
  ``tests/test_realtime_floor.py`` still drives it unmodified);
- ``on_audio_ready(..., segment_index=N)`` — attach audio to ITS segment;
- delivery that drains the current segment and advances, strictly IN ORDER;
- ``ResponseDone`` only when the reply was marked final AND every segment is
  delivered.

What this file exists to pin down is the part that is easy to get subtly
wrong: ordering, the deadline that stays armed while generation continues,
and the heard-prefix across segments after a barge-in — because history that
claims the machine said a sentence the user never heard is exactly the lie
``estimate_spoken_prefix`` was written to avoid.
"""

from __future__ import annotations

import pytest

from lobes.realtime._floor import (
    DEFAULT_BARGE_IN_WINDOW_MS,
    FailureReason,
    Floor,
    FloorState,
    ReplySegment,
    ResponseDone,
    ResponseFailed,
    ResponseInterrupted,
    Stage,
)
from lobes.realtime.protocol import BYTES_PER_SAMPLE, TTS_SAMPLE_RATE

CHUNK = 4800
CHUNK_MS = CHUNK * 1000 // (TTS_SAMPLE_RATE * BYTES_PER_SAMPLE)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms


class Recorder:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.chunks: list[bytes] = []
        self.cancelled_generate = 0
        self.cancelled_tts = 0

    def of_type(self, kind) -> list:
        return [e for e in self.events if isinstance(e, kind)]


def make_floor() -> tuple[Floor, Recorder, FakeClock]:
    rec, clock = Recorder(), FakeClock()
    floor = Floor(
        emit_event=rec.events.append,
        send_audio_chunk=rec.chunks.append,
        cancel_generate=lambda: setattr(rec, "cancelled_generate", rec.cancelled_generate + 1),
        cancel_tts=lambda: setattr(rec, "cancelled_tts", rec.cancelled_tts + 1),
        chunk_bytes=CHUNK,
        clock=clock,
    )
    return floor, rec, clock


def pcm(n_bytes: int) -> bytes:
    return bytes((i * 7) % 251 for i in range(n_bytes))


def to_responding(floor: Floor, clock: FakeClock) -> None:
    assert floor.on_turn_committed() is True
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)
    assert floor.on_transcript("מה השעה") is True
    assert floor.state is FloorState.RESPONDING


# --- appending segments ------------------------------------------------------


def test_the_first_segment_takes_the_floor_and_later_ones_append() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)

    assert floor.on_reply_segment("משפט ראשון.", final=False) is True
    assert floor.state is FloorState.SPEAKING
    assert floor.synthesizing is True

    assert floor.on_reply_segment("משפט שני.", final=False) is True
    assert floor.on_reply_segment("משפט שלישי.", final=True) is True
    assert floor.segment_count == 3
    assert [e.index for e in rec.of_type(ReplySegment)] == [0, 1, 2]
    assert [e.final for e in rec.of_type(ReplySegment)] == [False, False, True]


def test_the_generate_deadline_stays_armed_until_the_final_segment() -> None:
    # Generation CONTINUES while the machine speaks — arming the TTS deadline
    # at segment 1 would leave a wedged generate call unbounded.
    floor, _rec, clock = make_floor()
    to_responding(floor, clock)
    assert floor.armed_stage is Stage.GENERATE

    floor.on_reply_segment("משפט ראשון.", final=False)
    assert floor.armed_stage is Stage.GENERATE  # still generating

    floor.on_reply_segment("משפט אחרון.", final=True)
    assert floor.armed_stage is Stage.TTS  # nothing left to generate


def test_a_wedged_stream_expires_on_the_generate_deadline_while_speaking() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("משפט ראשון.", final=False)

    clock.advance(60_000)
    assert floor.tick() is True
    failed = rec.of_type(ResponseFailed)[0]
    assert (failed.stage, failed.reason) == (Stage.GENERATE, FailureReason.GENERATE_TIMEOUT)
    assert floor.state is FloorState.LISTENING


def test_a_tts_failure_on_a_segment_fails_the_response_while_generating() -> None:
    # The TTS stage is not the ARMED one mid-stream (generate is), but a
    # synthesis that failed is still a real failure — silent replies are the
    # one outcome this package refuses.
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("משפט ראשון.", final=False)

    assert floor.fail_stage(FailureReason.TTS_FAILED, "chatterbox said no") is True
    failed = rec.of_type(ResponseFailed)[0]
    assert (failed.stage, failed.reason) == (Stage.TTS, FailureReason.TTS_FAILED)
    assert floor.state is FloorState.LISTENING


def test_an_empty_final_segment_after_real_ones_just_closes_the_reply() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("משפט ראשון.", final=False)
    assert floor.on_reply_segment("", final=True) is True
    assert floor.segment_count == 1
    assert rec.of_type(ResponseFailed) == []


def test_an_empty_reply_with_no_segments_is_still_the_named_generate_failure() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    assert floor.on_reply_segment("   ", final=True) is True
    failed = rec.of_type(ResponseFailed)[0]
    assert failed.reason is FailureReason.GENERATE_FAILED


def test_on_reply_text_is_one_final_segment_and_still_emits_reply_text() -> None:
    # The non-streaming path is the fallback AND what every pre-existing test
    # drives: its event vocabulary must not change.
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    assert floor.on_reply_text("תשובה שלמה") is True
    assert [type(e).__name__ for e in rec.events][-1] == "ReplyText"
    assert rec.of_type(ReplySegment) == []
    assert floor.segment_count == 1


# --- delivery: in order, and only when the audio is there --------------------


def test_delivery_waits_for_the_current_segment_and_then_advances() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("ראשון.", final=False)
    floor.on_reply_segment("שני.", final=True)

    # Segment 1's audio arrives FIRST — delivery must not reorder.
    assert floor.on_audio_ready(pcm(CHUNK), segment_index=1) is True
    assert floor.deliver_next() is False  # segment 0 has nothing yet
    assert rec.chunks == []

    assert floor.on_audio_ready(pcm(CHUNK) + b"\x01\x02", segment_index=0) is True
    assert floor.deliver_next() is True  # segment 0, chunk 1
    assert floor.deliver_next() is True  # segment 0, remainder
    assert floor.deliver_next() is True  # segment 1
    assert floor.deliver_next() is False
    assert rec.chunks == [pcm(CHUNK), b"\x01\x02", pcm(CHUNK)]
    assert floor.state is FloorState.LISTENING


def test_response_done_waits_for_the_final_segment_not_the_last_delivered_one() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("ראשון.", final=False)
    floor.on_audio_ready(pcm(CHUNK), segment_index=0)

    assert floor.deliver_next() is True
    assert rec.of_type(ResponseDone) == []  # more is coming
    assert floor.state is FloorState.SPEAKING

    floor.on_reply_segment("שני.", final=True)
    floor.on_audio_ready(pcm(CHUNK), segment_index=1)
    assert floor.deliver_next() is True
    done = rec.of_type(ResponseDone)[0]
    assert (done.audio_bytes, done.chunks) == (CHUNK * 2, 2)
    assert floor.state is FloorState.LISTENING


def test_a_final_marker_arriving_after_the_last_chunk_still_completes() -> None:
    # THE race the pump can lose: delivery drains the only segment before the
    # stream ends, so there is no later chunk for ResponseDone to ride on.
    # Without this the floor sits in SPEAKING forever and the route's pump
    # spins on a response that can never finish.
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("ראשון.", final=False)
    floor.on_audio_ready(pcm(CHUNK), segment_index=0)
    assert floor.deliver_next() is True
    assert rec.of_type(ResponseDone) == []
    assert floor.state is FloorState.SPEAKING

    assert floor.on_reply_segment("", final=True) is True
    assert len(rec.of_type(ResponseDone)) == 1
    assert floor.state is FloorState.LISTENING


def test_the_last_audio_of_an_already_final_reply_completes_on_delivery() -> None:
    # The mirror image: the final marker arrives FIRST and the audio last.
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("ראשון.", final=True)
    floor.on_audio_ready(pcm(CHUNK), segment_index=0)
    assert rec.of_type(ResponseDone) == []  # nothing has been sent yet
    assert floor.deliver_next() is True
    assert len(rec.of_type(ResponseDone)) == 1


def test_empty_audio_for_a_segment_is_the_named_tts_failure() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("ראשון.", final=True)
    assert floor.on_audio_ready(b"", segment_index=0) is True
    assert rec.of_type(ResponseFailed)[0].reason is FailureReason.TTS_FAILED


def test_audio_for_an_unknown_or_already_filled_segment_is_refused() -> None:
    floor, _rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("ראשון.", final=True)
    assert floor.on_audio_ready(pcm(CHUNK), segment_index=0) is True
    assert floor.on_audio_ready(pcm(CHUNK), segment_index=0) is False  # already filled
    assert floor.on_audio_ready(pcm(CHUNK), segment_index=7) is False  # never announced


def test_a_stale_turns_segment_and_audio_are_both_ignored() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    stale = floor.turn_id
    floor.on_reply_segment("ראשון.", final=False)
    floor.on_speech_started()  # barge-in ends this response

    assert floor.on_reply_segment("שני.", final=True, turn_id=stale) is False
    assert floor.on_audio_ready(pcm(CHUNK), segment_index=0, turn_id=stale) is False
    assert rec.chunks == []


# --- barge-in across segments ------------------------------------------------


def test_a_barge_in_mid_stream_cancels_everything_and_reports_the_heard_prefix() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("משפט ראשון שלם.", final=False)
    floor.on_reply_segment("משפט שני חלקי.", final=False)
    floor.on_audio_ready(pcm(CHUNK), segment_index=0)
    floor.on_audio_ready(pcm(CHUNK * 4), segment_index=1)

    assert floor.deliver_next() is True  # all of segment 0
    assert floor.deliver_next() is True  # a quarter of segment 1
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)
    assert floor.on_speech_started() is True

    event = rec.of_type(ResponseInterrupted)[0]
    assert event.truncated is True
    assert event.delivered_bytes == CHUNK * 2
    assert event.undelivered_bytes == CHUNK * 3
    # Segment 0 was heard WHOLE; only segment 1 is estimated.
    assert event.heard_text.startswith("משפט ראשון שלם.")
    assert "משפט שני חלקי." not in event.heard_text
    assert event.reply_text == "משפט ראשון שלם. משפט שני חלקי."
    assert rec.cancelled_generate == 1  # the stream is told to stop
    assert rec.cancelled_tts == 1
    assert floor.state is FloorState.LISTENING


def test_nothing_delivered_means_nothing_heard() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("משפט ראשון.", final=False)
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)
    floor.on_speech_started()

    event = rec.of_type(ResponseInterrupted)[0]
    assert event.heard_text == ""
    assert event.delivered_bytes == 0


def test_a_single_segment_heard_text_equals_the_classic_estimate() -> None:
    # The non-streaming path must produce exactly what it produced before
    # segments existed — heard_text is additive, never a new answer.
    from lobes.realtime._floor import estimate_spoken_prefix

    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_text("one two three four five six seven eight")
    floor.on_audio_ready(pcm(CHUNK * 4))
    floor.deliver_next()
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)
    floor.on_speech_started()

    event = rec.of_type(ResponseInterrupted)[0]
    assert event.heard_text == estimate_spoken_prefix(
        event.reply_text, event.audio_end_ms, event.audio_total_ms
    )


# --- abandoning a streamed reply for a tool call -----------------------------


def test_discarding_segments_returns_the_floor_to_responding_for_a_tool_call() -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("רגע, אני בודק.", final=False)
    floor.on_audio_ready(pcm(CHUNK), segment_index=0)

    assert floor.discard_reply_segments() is True
    assert floor.state is FloorState.RESPONDING
    assert floor.armed_stage is Stage.GENERATE
    assert floor.segment_count == 0
    assert floor.pending_audio_bytes == 0

    # …and the tool call the stream ended with is accepted normally.
    assert floor.on_tool_call(call_id="call_1", name="now", arguments="{}") is True
    assert floor.state is FloorState.TOOL_WAIT
    assert rec.chunks == []  # the abandoned prefix was never spoken


def test_discarding_is_refused_when_the_machine_is_not_speaking() -> None:
    floor, _rec, clock = make_floor()
    to_responding(floor, clock)
    assert floor.discard_reply_segments() is False
    assert floor.state is FloorState.RESPONDING


@pytest.mark.parametrize("final", (False, True))
def test_close_drops_every_queued_segment(final: bool) -> None:
    floor, rec, clock = make_floor()
    to_responding(floor, clock)
    floor.on_reply_segment("ראשון.", final=final)
    floor.on_audio_ready(pcm(CHUNK * 3), segment_index=0)

    floor.close()
    assert floor.state is FloorState.CLOSED
    assert floor.pending_audio_bytes == 0
    assert floor.segment_count == 0
    assert rec.cancelled_generate == 1
    assert rec.cancelled_tts == 1
    assert floor.deliver_next() is False
