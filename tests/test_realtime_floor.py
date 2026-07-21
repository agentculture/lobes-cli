"""Tests for the floor/turn state machine (stdlib-only; no [realtime] extra).

Every dependency the real floor has at runtime — the LLM call, the TTS
sidecar, the WebSocket, the clock — is injected, so these tests drive
:class:`lobes.realtime._floor.Floor` with plain fakes: a recorder for the
emitted events and the audio sink, counters for the two cancel callables, and
a :class:`FakeClock` that only moves when a test moves it. Nothing here
sleeps, and nothing here reads a real clock — a deadline test that waited on
wall time would be both slow and flaky.

The heart of the file is the interrupt matrix: a speech onset (and a
committed turn) arriving in EVERY state where the machine holds the floor.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import lobes.realtime._floor as _floor
from lobes.realtime._floor import (
    DEFAULT_BARGE_IN_WINDOW_MS,
    DEFAULT_CHUNK_BYTES,
    FailureReason,
    Floor,
    FloorState,
    ReplyText,
    ResponseDone,
    ResponseFailed,
    ResponseInterrupted,
    ResponseStarted,
    Stage,
    estimate_spoken_prefix,
)
from lobes.realtime.protocol import BYTES_PER_SAMPLE, TTS_SAMPLE_RATE

# --- import isolation / parallel-safety guards -------------------------------


def test_module_imports_without_torch_or_the_realtime_extra() -> None:
    # Collecting this file in the offline dev env (no torch, fastapi, httpx,
    # numpy or scipy installed) already proved it; this names the guarantee.
    assert hasattr(_floor, "Floor")


def test_module_source_never_imports_forbidden_deps() -> None:
    # The docstring talks ABOUT the heavy deps and about its sibling modules
    # (documenting what this module must not import), so this scans actual
    # import statements rather than bare substrings.
    src = Path(_floor.__file__).read_text(encoding="utf-8")
    forbidden = (
        # the [realtime] extra — never importable in the offline suite
        "torch",
        "fastapi",
        "httpx",
        "numpy",
        "scipy",
        "silero_vad",
        # sibling realtime modules: everything the floor needs arrives through
        # its constructor, so the route layer (and only the route layer) is
        # what couples this machine to the session schema, the wire codec, the
        # segmenter, the turn builder and env-derived settings.
        "_session",
        "_wire",
        "_turn",
        "_segmenter",
        "_settings",
    )
    offenders = [
        name
        for name in forbidden
        for line in src.splitlines()
        if line.strip().startswith((f"import {name}", f"from {name}", f"from .{name}"))
        or line.strip().startswith(f"from lobes.realtime.{name}")
    ]
    assert not offenders, f"_floor.py imports forbidden deps: {offenders}"


def test_segmenter_stays_floor_agnostic() -> None:
    """The barge-in trigger is consumed ABOVE the segmenter, not inside it.

    The segmenter keeps segmenting whatever audio arrives and has no notion of
    who holds the floor; a ``SpeechStarted`` emitted while the machine speaks
    IS the barge-in trigger, and consuming it is this module's job — which is
    why #151 needs zero segmenter changes. Asserted over the AST, not the raw
    text, so a future cross-reference in its prose is free while a real
    dependency (an import, a parameter, an attribute) is not.
    """
    tree = ast.parse(Path(_floor.__file__).with_name("_segmenter.py").read_text(encoding="utf-8"))

    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not [name for name in imported if "_floor" in name], "the segmenter must not import up"

    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
    offenders = [
        name
        for name in identifiers
        if any(word in name.lower() for word in ("floor", "barge", "interrupt"))
    ]
    assert not offenders, f"_segmenter.py grew floor vocabulary: {offenders}"


# --- fakes -------------------------------------------------------------------


class FakeClock:
    """A monotonic-ms clock that only moves when a test moves it."""

    def __init__(self, start_ms: int = 1_000) -> None:
        self.now_ms = start_ms

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, ms: int) -> int:
        self.now_ms += ms
        return self.now_ms


class Recorder:
    """Captures everything the floor pushes outward."""

    def __init__(self) -> None:
        self.events: list[object] = []
        self.chunks: list[bytes] = []
        self.cancelled_generate = 0
        self.cancelled_tts = 0

    def emit(self, event: object) -> None:
        self.events.append(event)

    def send(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    def cancel_generate(self) -> None:
        self.cancelled_generate += 1

    def cancel_tts(self) -> None:
        self.cancelled_tts += 1

    def of_type(self, kind: type) -> list:
        return [event for event in self.events if isinstance(event, kind)]

    @property
    def delivered(self) -> bytes:
        return b"".join(self.chunks)


# 10 ms of 24 kHz PCM16 = 480 bytes — round numbers in every ms assertion.
CHUNK = 480
CHUNK_MS = 10


def make_floor(**overrides) -> tuple[Floor, Recorder, FakeClock]:
    rec = Recorder()
    clock = FakeClock()
    kwargs: dict[str, object] = {
        "emit_event": rec.emit,
        "send_audio_chunk": rec.send,
        "cancel_generate": rec.cancel_generate,
        "cancel_tts": rec.cancel_tts,
        "clock": clock,
        "chunk_bytes": CHUNK,
    }
    kwargs.update(overrides)
    return Floor(**kwargs), rec, clock  # type: ignore[arg-type]


def pcm(n_bytes: int) -> bytes:
    """Deterministic, position-tagged PCM so a test can prove WHICH bytes went out."""
    return bytes((i * 7) % 251 for i in range(n_bytes))


# Positions in the turn where the machine holds the floor. Every one of these
# must accept an interruption — that is the matrix this file exists for.
MACHINE_HELD_POSITIONS = ("transcribing", "responding", "synthesizing", "delivering")


def drive_to(
    floor: Floor,
    clock: FakeClock,
    position: str,
    *,
    audio: bytes | None = None,
    chunks_delivered: int = 2,
) -> None:
    """Walk the floor to *position*, past the barge-in guard window."""
    assert floor.on_turn_committed() is True
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)  # leave the guard window
    if position == "transcribing":
        return
    assert floor.on_transcript("what time is it") is True
    if position == "responding":
        return
    assert floor.on_reply_text("it is half past four") is True
    if position == "synthesizing":
        return
    assert floor.on_audio_ready(audio if audio is not None else pcm(CHUNK * 5)) is True
    for _ in range(chunks_delivered):
        assert floor.deliver_next() is True


# --- the happy path ----------------------------------------------------------


def test_the_full_turn_walks_listening_transcribing_responding_speaking_listening() -> None:
    floor, rec, clock = make_floor()
    assert floor.state is FloorState.LISTENING

    assert floor.on_turn_committed() is True
    assert floor.state is FloorState.TRANSCRIBING

    assert floor.on_transcript("hello there") is True
    assert floor.state is FloorState.RESPONDING

    assert floor.on_reply_text("general kenobi") is True
    assert floor.state is FloorState.SPEAKING
    assert floor.synthesizing is True and floor.delivering is False

    assert floor.on_audio_ready(pcm(CHUNK * 3)) is True
    assert floor.state is FloorState.SPEAKING
    assert floor.synthesizing is False and floor.delivering is True

    assert [floor.deliver_next() for _ in range(3)] == [True, True, True]
    assert floor.deliver_next() is False  # nothing left to send
    assert floor.state is FloorState.LISTENING  # the floor came back on its own

    assert rec.delivered == pcm(CHUNK * 3)
    assert [type(e) for e in rec.events] == [ResponseStarted, ReplyText, ResponseDone]
    done = rec.of_type(ResponseDone)[0]
    assert (done.audio_ms, done.audio_bytes, done.chunks) == (CHUNK_MS * 3, CHUNK * 3, 3)
    assert rec.cancelled_generate == 0 and rec.cancelled_tts == 0


def test_the_last_chunk_is_the_short_remainder_and_chunking_is_sample_aligned() -> None:
    floor, rec, clock = make_floor()
    audio = pcm(CHUNK * 2 + 16)
    drive_to(floor, clock, "synthesizing")

    floor.on_audio_ready(audio)
    while floor.deliver_next():
        pass

    assert rec.chunks == [audio[:CHUNK], audio[CHUNK : CHUNK * 2], audio[CHUNK * 2 :]]
    assert rec.delivered == audio  # byte-exact, nothing dropped, nothing duplicated


def test_odd_chunk_size_is_rounded_down_to_whole_samples() -> None:
    # A chunk that split a PCM16 sample would desync playback from the first
    # frame; the floor rounds down rather than emitting a half sample.
    floor, _, _ = make_floor(chunk_bytes=481)
    assert floor.chunk_bytes == 480
    floor, _, _ = make_floor(chunk_bytes=1)
    assert floor.chunk_bytes == BYTES_PER_SAMPLE  # never zero — that would never deliver


def test_default_chunk_size_is_derived_from_the_protocol_rate() -> None:
    assert DEFAULT_CHUNK_BYTES % BYTES_PER_SAMPLE == 0
    assert DEFAULT_CHUNK_BYTES == TTS_SAMPLE_RATE * BYTES_PER_SAMPLE * 40 // 1000


def test_the_observation_surface_tracks_the_turn_through_every_stage() -> None:
    # What the route (and the site's event stream) can ask the floor at any
    # moment: which stage is armed, and how much of the reply has gone out.
    floor, _, clock = make_floor()
    assert (floor.armed_stage, floor.delivered_bytes, floor.pending_audio_bytes) == (None, 0, 0)

    floor.on_turn_committed()
    assert floor.armed_stage is Stage.TRANSCRIBE
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)

    floor.on_transcript("hello")
    assert floor.armed_stage is Stage.GENERATE
    floor.on_reply_text("a reply")
    assert floor.armed_stage is Stage.TTS

    floor.on_audio_ready(pcm(CHUNK * 3))
    assert floor.armed_stage is None  # nothing left to time out
    floor.deliver_next()
    assert (floor.delivered_bytes, floor.pending_audio_bytes) == (CHUNK, CHUNK * 2)


def test_an_empty_transcript_returns_the_floor_without_a_response() -> None:
    floor, rec, _ = make_floor()
    floor.on_turn_committed()
    assert floor.on_transcript("   ") is True
    assert floor.state is FloorState.LISTENING
    assert rec.events == []  # silence is not a response, and not an error


def test_an_empty_reply_is_a_named_failure_not_unexplained_silence() -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "responding")
    assert floor.on_reply_text("  ") is True
    assert floor.state is FloorState.LISTENING
    failed = rec.of_type(ResponseFailed)[0]
    assert failed.reason is FailureReason.GENERATE_FAILED
    assert rec.of_type(ReplyText) == []  # nothing was ever going to be spoken


def test_empty_tts_audio_is_a_named_failure_not_a_silent_completion() -> None:
    # tts_client.synthesize() returns b"" on a soft failure — the floor must
    # never render that as a completed spoken reply.
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "synthesizing")
    assert floor.on_audio_ready(b"") is True
    assert floor.state is FloorState.LISTENING
    failed = rec.of_type(ResponseFailed)[0]
    assert failed.reason is FailureReason.TTS_FAILED
    assert failed.stage is Stage.TTS


# --- interrupt matrix: an onset in EVERY machine-held state ------------------


@pytest.mark.parametrize("position", MACHINE_HELD_POSITIONS)
def test_speech_onset_interrupts_from_every_machine_held_state(position: str) -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, position)

    assert floor.on_speech_started() is True  # the onset IS the barge-in trigger
    assert floor.state is FloorState.LISTENING  # the floor went back to the user

    interruptions = rec.of_type(ResponseInterrupted)
    assert len(interruptions) == 1
    assert interruptions[0].truncated is True
    # cancel BOTH, from every state — the floor cannot know whether the route
    # had already handed off to TTS when the onset landed.
    assert rec.cancelled_generate == 1
    assert rec.cancelled_tts == 1


@pytest.mark.parametrize("position", MACHINE_HELD_POSITIONS)
def test_the_interruption_names_the_stage_it_cut_short(position: str) -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, position)
    floor.on_speech_started()

    expected = {
        "transcribing": Stage.TRANSCRIBE,
        "responding": Stage.GENERATE,
        "synthesizing": Stage.TTS,
        "delivering": Stage.TTS,
    }[position]
    assert rec.of_type(ResponseInterrupted)[0].stage is expected


@pytest.mark.parametrize("position", MACHINE_HELD_POSITIONS)
def test_a_committed_turn_also_interrupts_from_every_machine_held_state(position: str) -> None:
    # A committed turn survived the VAD's own silence confirmation, so it is
    # stronger evidence than a bare onset — it is never dropped on the floor.
    floor, rec, clock = make_floor()
    drive_to(floor, clock, position)
    first_turn = floor.turn_id

    assert floor.on_turn_committed() is True
    assert len(rec.of_type(ResponseInterrupted)) == 1
    assert floor.state is FloorState.TRANSCRIBING  # the new turn opened immediately
    assert floor.turn_id != first_turn


def test_speech_onset_while_listening_is_not_an_interruption() -> None:
    floor, rec, _ = make_floor()
    assert floor.on_speech_started() is False
    assert floor.state is FloorState.LISTENING
    assert rec.events == []
    assert rec.cancelled_generate == 0 and rec.cancelled_tts == 0


def test_repeated_onsets_emit_exactly_one_interruption_event() -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "delivering")

    assert floor.on_speech_started() is True
    for _ in range(4):
        assert floor.on_speech_started() is False  # already the user's floor

    assert len(rec.of_type(ResponseInterrupted)) == 1
    assert rec.cancelled_generate == 1 and rec.cancelled_tts == 1


# --- interrupt mid-delivery: the undelivered remainder --------------------


def test_interrupt_mid_delivery_never_sends_the_undelivered_remainder() -> None:
    floor, rec, clock = make_floor()
    audio = pcm(CHUNK * 5)
    drive_to(floor, clock, "delivering", audio=audio, chunks_delivered=2)

    floor.on_speech_started()

    assert rec.delivered == audio[: CHUNK * 2]  # exactly what went out before the cut
    assert floor.deliver_next() is False  # the remainder is gone, not queued
    assert rec.delivered == audio[: CHUNK * 2]  # ... and still gone after pumping
    assert audio[CHUNK * 2 :] not in rec.delivered


def test_the_interruption_event_carries_the_truncation_marker() -> None:
    floor, rec, clock = make_floor()
    audio = pcm(CHUNK * 5)
    drive_to(floor, clock, "delivering", audio=audio, chunks_delivered=2)

    floor.on_speech_started()

    event = rec.of_type(ResponseInterrupted)[0]
    assert event.truncated is True
    assert event.audio_end_ms == CHUNK_MS * 2  # what the client actually heard
    assert event.audio_total_ms == CHUNK_MS * 5  # what the reply would have been
    assert event.delivered_bytes == CHUNK * 2
    assert event.undelivered_bytes == CHUNK * 3
    assert event.chunks_delivered == 2
    assert event.reply_text == "it is half past four"


def test_an_interruption_before_any_audio_marks_a_zero_length_truncation() -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "responding")
    floor.on_speech_started()

    event = rec.of_type(ResponseInterrupted)[0]
    assert event.truncated is True
    assert (event.audio_end_ms, event.delivered_bytes, event.chunks_delivered) == (0, 0, 0)
    assert event.reply_text == ""  # generate never produced one


def test_the_interruption_event_is_ordered_after_the_last_delivered_chunk() -> None:
    # The site stops LOCAL playback on this event, so it must not arrive
    # before audio the client is still meant to play.
    sequence: list[str] = []
    rec = Recorder()
    clock = FakeClock()
    floor = Floor(
        emit_event=lambda e: sequence.append(type(e).__name__),
        send_audio_chunk=lambda b: sequence.append("chunk"),
        cancel_generate=rec.cancel_generate,
        cancel_tts=rec.cancel_tts,
        clock=clock,
        chunk_bytes=CHUNK,
    )
    drive_to(floor, clock, "delivering", audio=pcm(CHUNK * 4), chunks_delivered=2)
    floor.on_speech_started()

    assert sequence == [
        "ResponseStarted",
        "ReplyText",
        "chunk",
        "chunk",
        "ResponseInterrupted",
    ]


# --- the barge-in guard window ----------------------------------------------


@pytest.mark.parametrize("position", MACHINE_HELD_POSITIONS)
def test_an_onset_inside_the_barge_in_window_is_not_honoured(position: str) -> None:
    floor, rec, clock = make_floor()
    # Take the floor, then step to one ms short of the window.
    floor.on_turn_committed()
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS - 1)
    if position != "transcribing":
        floor.on_transcript("hello")
    if position in ("synthesizing", "delivering"):
        floor.on_reply_text("a reply")
    if position == "delivering":
        floor.on_audio_ready(pcm(CHUNK * 3))
        floor.deliver_next()

    assert floor.on_speech_started() is False
    assert floor.state is not FloorState.LISTENING  # the machine kept the floor
    assert rec.of_type(ResponseInterrupted) == []
    assert rec.cancelled_generate == 0 and rec.cancelled_tts == 0


def test_the_window_is_measured_from_the_moment_the_floor_was_taken() -> None:
    floor, rec, clock = make_floor(barge_in_window_ms=500)
    floor.on_turn_committed()

    clock.advance(499)
    assert floor.on_speech_started() is False
    clock.advance(1)  # exactly at the window — honoured from here on
    assert floor.on_speech_started() is True
    assert floor.state is FloorState.LISTENING


def test_a_committed_turn_inside_the_window_is_dropped_like_an_onset() -> None:
    # A 10s guard: drive_to only advances the default 750ms, so the whole
    # response happens inside the window.
    floor, rec, clock = make_floor(barge_in_window_ms=10_000)
    drive_to(floor, clock, "delivering")

    assert floor.on_turn_committed() is False
    assert floor.state is FloorState.SPEAKING
    assert rec.of_type(ResponseInterrupted) == []


def test_a_zero_window_honours_an_immediate_onset() -> None:
    floor, rec, _ = make_floor(barge_in_window_ms=0)
    floor.on_turn_committed()
    floor.on_transcript("hi")
    assert floor.on_speech_started() is True
    assert floor.state is FloorState.LISTENING


# --- per-stage deadlines -----------------------------------------------------

TIMEOUT_CASES = (
    ("transcribing", "transcribe_timeout_ms", Stage.TRANSCRIBE, FailureReason.TRANSCRIBE_TIMEOUT),
    ("responding", "generate_timeout_ms", Stage.GENERATE, FailureReason.GENERATE_TIMEOUT),
    ("synthesizing", "tts_timeout_ms", Stage.TTS, FailureReason.TTS_TIMEOUT),
)


@pytest.mark.parametrize("position,knob,stage,reason", TIMEOUT_CASES)
def test_each_stage_returns_the_floor_to_listening_on_expiry(
    position: str, knob: str, stage: Stage, reason: FailureReason
) -> None:
    bound = 4_000
    floor, rec, clock = make_floor(**{knob: bound})
    drive_to(floor, clock, position)
    assert floor.deadline_ms == floor.stage_started_ms + bound

    clock.now_ms = floor.deadline_ms - 1
    assert floor.tick() is False  # not yet — the bound has not been reached
    assert floor.state is not FloorState.LISTENING

    clock.advance(1)
    assert floor.tick() is True
    assert floor.state is FloorState.LISTENING

    failed = rec.of_type(ResponseFailed)[0]
    assert failed.stage is stage
    assert failed.reason is reason
    assert str(bound) in failed.message  # the bound it actually exceeded
    # A wedged backend must not keep running behind a floor that moved on.
    assert rec.cancelled_generate == 1 and rec.cancelled_tts == 1


@pytest.mark.parametrize("position,knob,stage,reason", TIMEOUT_CASES)
def test_an_expired_stage_emits_exactly_one_error_and_does_not_re_expire(
    position: str, knob: str, stage: Stage, reason: FailureReason
) -> None:
    floor, rec, clock = make_floor(**{knob: 1_000})
    drive_to(floor, clock, position)
    clock.advance(10_000)

    assert floor.tick() is True
    assert floor.tick() is False  # disarmed with the transition
    assert len(rec.of_type(ResponseFailed)) == 1


def test_tick_is_inert_while_listening_and_while_delivering() -> None:
    floor, rec, clock = make_floor()
    assert floor.deadline_ms is None
    clock.advance(10_000_000)
    assert floor.tick() is False  # nothing armed while the user holds the floor

    drive_to(floor, clock, "delivering")
    assert floor.deadline_ms is None  # the TTS deadline was disarmed by the audio
    clock.advance(10_000_000)
    assert floor.tick() is False  # delivery is caller-paced, not deadline-paced
    assert floor.state is FloorState.SPEAKING


@pytest.mark.parametrize(
    "reason,position",
    (
        (FailureReason.TRANSCRIBE_FAILED, "transcribing"),
        (FailureReason.GENERATE_FAILED, "responding"),
        (FailureReason.TTS_FAILED, "synthesizing"),
    ),
)
def test_a_named_backend_failure_returns_the_floor_like_a_timeout(
    reason: FailureReason, position: str
) -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, position)

    assert floor.fail_stage(reason, "backend said no") is True
    assert floor.state is FloorState.LISTENING
    failed = rec.of_type(ResponseFailed)[0]
    assert failed.reason is reason
    assert failed.message == "backend said no"
    assert rec.cancelled_generate == 1 and rec.cancelled_tts == 1


def test_a_failure_carrying_a_stale_turn_id_is_ignored() -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "responding")
    assert floor.fail_stage(FailureReason.GENERATE_FAILED, "turn 1's error", turn_id=99) is False
    assert floor.state is FloorState.RESPONDING
    assert rec.of_type(ResponseFailed) == []


def test_a_failure_for_a_stage_the_floor_has_left_is_ignored() -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "synthesizing")
    # The generate call finally errors out, long after its stage was left.
    assert floor.fail_stage(FailureReason.GENERATE_FAILED, "too late") is False
    assert floor.state is FloorState.SPEAKING
    assert rec.of_type(ResponseFailed) == []


# --- stale completions -------------------------------------------------------


def test_a_late_transcript_never_resurrects_an_abandoned_turn() -> None:
    floor, rec, clock = make_floor(transcribe_timeout_ms=1_000)
    floor.on_turn_committed()
    stale_turn = floor.turn_id
    clock.advance(1_000)
    floor.tick()  # the STT forward wedged; the floor moved on
    assert floor.state is FloorState.LISTENING

    assert floor.on_transcript("...eventually", turn_id=stale_turn) is False
    assert floor.state is FloorState.LISTENING
    assert rec.of_type(ResponseStarted) == []


def test_a_late_reply_never_lands_on_the_next_turn() -> None:
    # The classic stale-response bug: turn 1's generate returns while turn 2
    # is already in flight. Without the turn token, turn 2 would speak turn
    # 1's answer.
    floor, rec, clock = make_floor(generate_timeout_ms=1_000)
    floor.on_turn_committed()
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)
    floor.on_transcript("first question")
    turn_one = floor.turn_id
    clock.advance(1_000)
    floor.tick()  # turn 1's generate timed out

    floor.on_turn_committed()  # turn 2 opens
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)
    floor.on_transcript("second question")
    assert floor.state is FloorState.RESPONDING

    assert floor.on_reply_text("answer to the FIRST question", turn_id=turn_one) is False
    assert floor.state is FloorState.RESPONDING  # still waiting for turn 2's answer
    assert rec.of_type(ReplyText) == []


def test_stale_audio_is_never_delivered() -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "synthesizing")
    stale_turn = floor.turn_id
    floor.on_speech_started()  # barge-in: this response is over

    assert floor.on_audio_ready(pcm(CHUNK * 3), turn_id=stale_turn) is False
    assert floor.deliver_next() is False
    assert rec.chunks == []


def test_completion_inputs_are_ignored_from_the_wrong_state() -> None:
    floor, rec, _ = make_floor()
    # Nothing is in flight — every completion input is a no-op, never a
    # spurious transition and never an exception.
    assert floor.on_transcript("hello") is False
    assert floor.on_reply_text("hi") is False
    assert floor.on_audio_ready(pcm(CHUNK)) is False
    assert floor.deliver_next() is False
    assert floor.state is FloorState.LISTENING
    assert rec.events == []


# --- teardown ----------------------------------------------------------------


@pytest.mark.parametrize("position", ("listening",) + MACHINE_HELD_POSITIONS)
def test_close_is_safe_from_every_state(position: str) -> None:
    floor, rec, clock = make_floor()
    if position != "listening":
        drive_to(floor, clock, position)

    floor.close()
    assert floor.state is FloorState.CLOSED
    assert floor.pending_audio_bytes == 0
    if position != "listening":
        # Whatever was in flight is cancelled — a torn-down session must not
        # leave a generate or a synthesis running.
        assert rec.cancelled_generate == 1 and rec.cancelled_tts == 1
    assert rec.of_type(ResponseInterrupted) == []  # the session layer owns close events


def test_close_is_idempotent_and_later_inputs_are_inert() -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "delivering")
    floor.close()
    floor.close()  # a watchdog tick racing teardown must not double-cancel
    assert rec.cancelled_generate == 1 and rec.cancelled_tts == 1

    before = len(rec.events)
    assert floor.on_speech_started() is False
    assert floor.on_turn_committed() is False
    assert floor.on_transcript("hi") is False
    assert floor.on_reply_text("hi") is False
    assert floor.on_audio_ready(pcm(CHUNK)) is False
    assert floor.deliver_next() is False
    assert floor.tick() is False
    assert floor.fail_stage(FailureReason.TTS_FAILED, "nope") is False
    assert floor.state is FloorState.CLOSED
    assert len(rec.events) == before  # closed floors are silent, never raising


# --- back-to-back turns ------------------------------------------------------


def test_two_turns_run_back_to_back_with_distinct_turn_ids() -> None:
    floor, rec, clock = make_floor()
    for _ in range(2):
        drive_to(floor, clock, "delivering", audio=pcm(CHUNK * 2), chunks_delivered=2)
        assert floor.state is FloorState.LISTENING
    assert len(rec.of_type(ResponseDone)) == 2
    ids = {e.turn_id for e in rec.of_type(ResponseStarted)}
    assert len(ids) == 2


def test_the_user_can_interrupt_and_immediately_take_the_next_turn() -> None:
    floor, rec, clock = make_floor()
    drive_to(floor, clock, "delivering")
    floor.on_speech_started()  # barge-in

    # The interrupting utterance is an ordinary turn from here on.
    assert floor.on_turn_committed() is True
    assert floor.state is FloorState.TRANSCRIBING
    clock.advance(DEFAULT_BARGE_IN_WINDOW_MS)
    assert floor.on_transcript("no, the OTHER one") is True
    assert floor.state is FloorState.RESPONDING
    assert len(rec.of_type(ResponseInterrupted)) == 1


# --- the spoken-prefix helper ------------------------------------------------


def test_estimate_spoken_prefix_cuts_proportionally_on_a_word_boundary() -> None:
    text = "one two three four five six seven eight"
    assert estimate_spoken_prefix(text, 0, 1_000) == ""
    assert estimate_spoken_prefix(text, 1_000, 1_000) == text
    assert estimate_spoken_prefix(text, 2_000, 1_000) == text  # never past the end
    half = estimate_spoken_prefix(text, 500, 1_000)
    assert text.startswith(half) and 0 < len(half) < len(text)
    assert not half.endswith(" ") and " " in half  # cut at a word boundary


def test_estimate_spoken_prefix_handles_a_zero_length_reply() -> None:
    assert estimate_spoken_prefix("", 0, 0) == ""
    assert estimate_spoken_prefix("something", 5, 0) == "something"
