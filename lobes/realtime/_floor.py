"""The conversational floor — a pure state machine over who may speak.

A voice-to-voice session has exactly one floor: either the user holds it (the
server is listening) or the machine holds it (the server is transcribing,
generating, or speaking). Issue #151's central correctness requirement is that
this is **explicit state**, never implied by event ordering — so it lives in
one small class here, the way ``server_vad`` segmentation lives in
:mod:`lobes.realtime._segmenter`. This module is that class's whole world:
no I/O, no sockets, no FastAPI, no httpx, no torch, and — deliberately — no
import of any sibling realtime module either.

Everything the floor needs at runtime arrives through its constructor:

- ``emit_event`` — receives one frozen :data:`FloorEvent` per lifecycle
  moment. The route turns those into wire events; this module never speaks
  JSON, never mints an event/response id, and never touches the session
  schema in :mod:`lobes.realtime._session`.
- ``send_audio_chunk`` — receives one PCM16 chunk of the spoken reply. The
  route base64-encodes it into ``response.audio.delta``; this module never
  encodes anything.
- ``cancel_generate`` / ``cancel_tts`` — the two abandonment hooks. The route
  wires them to a task cancel and to the ``cancel_event`` that
  :func:`lobes.realtime.tts_client.synthesize` already threads through every
  request.
- ``clock`` — returns monotonic milliseconds. Injected for the same reason
  the segmenter counts stream time instead of wall-clock: a deadline test
  that waited on real time would be slow AND flaky. The default is
  :func:`lobes.realtime.protocol.timestamp_ms`.

Why synchronous, when the route is async? Because every one of those hooks
has a *synchronous* implementation on the async side: ``asyncio.Event.set()``,
``asyncio.Queue.put_nowait()``, ``Task.cancel()`` and ``time.monotonic()`` are
all plain calls. Keeping the machine synchronous means it has no await points,
therefore no interleaving, therefore no way for a barge-in to land halfway
through a transition — the property this task exists to guarantee. If a hook
raises (a closed socket, say), the exception propagates unmodified; translating
that into a session-level error is the route's job, exactly as
:mod:`lobes.realtime._segmenter` leaves a raising VAD callable to its caller.

Floor states
------------
``listening`` → ``transcribing`` → ``responding`` → ``speaking`` →
``listening``. The user holds the floor in ``listening``; the machine holds it
in the other three (:data:`MACHINE_HELD_STATES`), and ``closed`` is teardown.

``speaking`` covers TWO sub-stages, because the Chatterbox sidecar has no
streaming route — :func:`~lobes.realtime.tts_client.synthesize` is full-read,
so the bridge holds the *complete* reply PCM before the first byte can go out:

- **synthesizing** — the machine has committed to answering and TTS is in
  flight (:attr:`Floor.synthesizing`). Nothing is audible yet, but the floor
  is genuinely the machine's: the user's turn is over and an answer is coming.
- **delivering** — the PCM arrived and is going out as sequential chunks, one
  per :meth:`Floor.deliver_next` call (:attr:`Floor.delivering`).

That pumped delivery is what makes interruption meaningful at all: it stops the
**undelivered remainder**. A single blocking "send it all" would leave nothing
to interrupt. The floor returns to ``listening`` the moment the final chunk is
handed to ``send_audio_chunk`` — there is no client-side "playback finished"
signal to wait for, and the client stops its own local playback when it sees
the interruption event.

``sample_rate`` is the **output** rate — Chatterbox's 24 kHz, which
``protocol.py`` pins equal to ``CLIENT_SAMPLE_RATE`` so audio-out never
resamples. It is NOT the session's negotiated *input* rate, which may be
16 kHz: feeding that in would misreport every ``audio_end_ms`` by 1.5x and
mis-size every chunk.

Barge-in: cancel both, always
-----------------------------
A speech onset (or a committed turn) arriving while the machine holds the floor
is the barge-in trigger — the segmenter never stops segmenting, and it has no
idea a response is in flight, which is exactly why consuming that trigger is
this module's job and not its. An honoured barge-in:

1. calls ``cancel_generate()`` **and** ``cancel_tts()`` — both, from every
   state. The floor cannot know whether the route had already handed off to
   TTS when the onset landed (there is a real window between
   :meth:`Floor.on_reply_text` and the route launching the synthesis task),
   and both hooks are idempotent, so cancelling both closes that race by
   construction rather than by timing;
2. drops the undelivered remainder — those bytes are never sent;
3. emits exactly ONE :class:`ResponseInterrupted`, carrying the truncation
   marker (``truncated=True`` plus ``audio_end_ms``, the millisecond offset the
   client actually heard — ``0`` when the cut landed before any audio);
4. returns the floor to the user.

The same cancel-both/return-the-floor rule governs every other abandonment
(a stage deadline, a named backend failure, teardown), so there is exactly one
way the machine ever loses the floor.

``barge_in_window_ms`` — a guard, not a delay
---------------------------------------------
The shipped-but-dormant ``BARGE_IN_WINDOW_MS`` knob (default 750, in
:mod:`lobes.realtime._settings`) is armed here as a **guard window**: an onset
landing less than that long after the machine took the floor is ignored — no
event, no cancel — because the likeliest source of speech in that instant is
the tail of the user's own turn or an echo blip as playback starts, not a
deliberate interruption. The window is measured from the **turn commit** (the
moment the floor became the machine's), not from the first audio frame.

This reading is a decision, not a settled requirement: the spec's honesty
condition says injected speech stops playback *within* ``barge_in_window_ms``,
which could equally describe a latency bound. The two readings mostly agree in
practice — full-read synthesis of a spoken reply typically takes longer than
750ms, so by the time audio is audible the guard has long elapsed and a
barge-in is honoured immediately. Window-only barge-in is what ships;
``barge_in_model`` stays declared and unconsumed until a live run shows the
window alone is not enough.

One consequence is deliberate: a **committed turn** arriving while the machine
holds the floor also interrupts (once past the guard), because a turn that
survived the VAD's own silence confirmation is far stronger evidence than a
bare onset — and dropping it would silently discard something the user said.
The interruption and the new turn happen in one step: the floor is released,
then immediately re-taken by the new turn.

Per-stage deadlines, and the stale answers they create
-------------------------------------------------------
Each machine-held stage gets a bounded wait — ``transcribe`` (the Parakeet
forward, mirroring ``app.py``'s ``_STT_FORWARD_TIMEOUT = 60``), ``generate``
(the reason ``scripts/realtime-voice-loop.py`` carries
``PLAYBACK_TIMEOUT_S = 60``: its comment records that a wedged backend can
strand a whole conversation) and ``tts`` (``tts_client``'s own httpx read
timeout is 60s). On expiry the floor returns to ``listening`` with a named
:class:`ResponseFailed`; a session is never left wedged in a responding state.

Deadlines expire only in :meth:`Floor.tick`, which the route calls from a
watchdog — a wedged backend by definition is not calling anything else. An
answer that arrives before its tick therefore wins; that is deterministic and
harmless.

Expiry makes **stale completions** inevitable: the abandoned generate call
eventually returns, possibly while a *later* turn is in flight, and speaking
turn 1's answer during turn 2 is the classic bug here. Every completion input
therefore takes an optional ``turn_id`` (paired with :attr:`Floor.turn_id`,
which advances on every turn); a completion for a turn the floor has left is
ignored and returns ``False``, never a spurious transition.

What this module deliberately does not own
-------------------------------------------
The session event schema and per-session history (:mod:`._session`), the
base64 wire codec (:mod:`._wire`), the chat/completions payload
(:mod:`._turn`), VAD segmentation (:mod:`._segmenter`) and env-derived config
(:mod:`._settings`) all live elsewhere and are all imported by the route, never
by this file. The floor's event dataclasses are floor-local facts; the route
maps them onto the session schema, where the failure reason names line up
one-for-one with the error codes (``transcribe_failed`` is the existing
``stt_forward_failed``).

Per-session isolation
----------------------
All state lives on the :class:`Floor` instance — there is no module-level
mutable state in this file. Two concurrent sessions never observe each other's
floor, audio, or deadlines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .protocol import BYTES_PER_SAMPLE, TTS_SAMPLE_RATE, timestamp_ms

# Mirrors lobes.realtime._settings.build_settings()'s BARGE_IN_WINDOW_MS
# default. Duplicated as a plain literal rather than importing the settings
# singleton so this module stays free of env-derived state at import time —
# the two simply agree on the number, and the route threads the live value in.
DEFAULT_BARGE_IN_WINDOW_MS = 750

# Per-stage bounded waits. Each mirrors an in-tree precedent for the same
# backend call (see the module docstring); each is a constructor parameter so
# the route can thread env-tuned values through without this module reading env.
DEFAULT_TRANSCRIBE_TIMEOUT_MS = 60_000
DEFAULT_GENERATE_TIMEOUT_MS = 60_000
DEFAULT_TTS_TIMEOUT_MS = 60_000

# One audio-out frame. 40ms at 24 kHz PCM16 = 1920 bytes: small enough that an
# interruption truncates within a frame nobody notices, large enough not to
# flood the socket with base64 frames. Derived from protocol.py's rate, never
# an independent magic number. If the wire codec ships its own delta size, the
# route passes it to the constructor — this module never imports it.
DEFAULT_CHUNK_MS = 40
DEFAULT_CHUNK_BYTES = TTS_SAMPLE_RATE * BYTES_PER_SAMPLE * DEFAULT_CHUNK_MS // 1000


class FloorState(str, Enum):
    """Who holds the floor — explicit state, never inferred from event order."""

    LISTENING = "listening"  # the user's floor
    TRANSCRIBING = "transcribing"  # the turn is committed; STT is in flight
    RESPONDING = "responding"  # the generate call is in flight
    SPEAKING = "speaking"  # TTS is in flight, then audio is being delivered
    CLOSED = "closed"


class Stage(str, Enum):
    """A machine-held stage — what a deadline bounds and an interruption cuts."""

    TRANSCRIBE = "transcribe"
    GENERATE = "generate"
    TTS = "tts"


class FailureReason(str, Enum):
    """Named ways a stage ends badly — never a bare exception string.

    The route maps these onto the session schema's error codes;
    ``transcribe_failed`` is the existing ``stt_forward_failed``.
    """

    TRANSCRIBE_TIMEOUT = "transcribe_timeout"
    GENERATE_TIMEOUT = "generate_timeout"
    TTS_TIMEOUT = "tts_timeout"
    TRANSCRIBE_FAILED = "transcribe_failed"
    GENERATE_FAILED = "generate_failed"
    TTS_FAILED = "tts_failed"


MACHINE_HELD_STATES = frozenset(
    {FloorState.TRANSCRIBING, FloorState.RESPONDING, FloorState.SPEAKING}
)

_STAGE_OF_STATE = {
    FloorState.TRANSCRIBING: Stage.TRANSCRIBE,
    FloorState.RESPONDING: Stage.GENERATE,
    FloorState.SPEAKING: Stage.TTS,
}

_TIMEOUT_REASON = {
    Stage.TRANSCRIBE: FailureReason.TRANSCRIBE_TIMEOUT,
    Stage.GENERATE: FailureReason.GENERATE_TIMEOUT,
    Stage.TTS: FailureReason.TTS_TIMEOUT,
}

_STAGE_OF_REASON = {
    FailureReason.TRANSCRIBE_TIMEOUT: Stage.TRANSCRIBE,
    FailureReason.TRANSCRIBE_FAILED: Stage.TRANSCRIBE,
    FailureReason.GENERATE_TIMEOUT: Stage.GENERATE,
    FailureReason.GENERATE_FAILED: Stage.GENERATE,
    FailureReason.TTS_TIMEOUT: Stage.TTS,
    FailureReason.TTS_FAILED: Stage.TTS,
}


# ---------------------------------------------------------------------------
# Events — floor-local facts. The route maps them onto the session schema.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseStarted:
    """The machine took the floor to answer: generate is in flight."""

    at_ms: int
    turn_id: int


@dataclass(frozen=True)
class ReplyText:
    """The generated reply text, before a single byte of it has been spoken."""

    at_ms: int
    turn_id: int
    text: str


@dataclass(frozen=True)
class ResponseDone:
    """The whole reply was delivered; the floor is the user's again."""

    at_ms: int
    turn_id: int
    audio_ms: int
    audio_bytes: int
    chunks: int


@dataclass(frozen=True)
class ResponseInterrupted:
    """Barge-in: the response was cut short and the floor handed back.

    The truncation marker is ``truncated`` plus ``audio_end_ms`` — the offset
    into the reply the client actually heard, and therefore the point history
    should record the reply up to (``0`` when the cut landed before any audio
    went out). ``undelivered_bytes`` is what was dropped rather than sent.
    """

    at_ms: int
    turn_id: int
    stage: Stage
    audio_end_ms: int
    audio_total_ms: int
    delivered_bytes: int
    undelivered_bytes: int
    chunks_delivered: int
    reply_text: str
    truncated: bool = True


@dataclass(frozen=True)
class ResponseFailed:
    """A stage timed out or failed; the floor returned to the user."""

    at_ms: int
    turn_id: int
    stage: Stage
    reason: FailureReason
    message: str


FloorEvent = ResponseStarted | ReplyText | ResponseDone | ResponseInterrupted | ResponseFailed

EmitEvent = Callable[[FloorEvent], None]
SendAudioChunk = Callable[[bytes], None]
Cancel = Callable[[], None]
Clock = Callable[[], int]


def estimate_spoken_prefix(text: str, audio_end_ms: int, audio_total_ms: int) -> str:
    """The part of *text* a listener plausibly heard before a truncation.

    An **estimate**, not an alignment: Chatterbox returns audio with no word
    timings, so this cuts the text proportionally to the audio actually
    delivered and then backs up to the previous word boundary. Offered as a
    named helper rather than a field on :class:`ResponseInterrupted` precisely
    because it is derived, not measured — the event carries only facts. A
    caller writing an interrupted reply into conversation history is the
    intended user: recording the full text as if it had been heard is a worse
    lie than recording a slightly-off prefix.
    """
    if not text:
        return ""
    if audio_total_ms <= 0 or audio_end_ms >= audio_total_ms:
        return text
    if audio_end_ms <= 0:
        return ""
    cut = max(1, len(text) * audio_end_ms // audio_total_ms)
    prefix = text[:cut]
    boundary = prefix.rfind(" ")
    return prefix[:boundary] if boundary > 0 else prefix


class Floor:
    """One session's conversational floor.

    Construct one per realtime session, alongside its
    :class:`~lobes.realtime._segmenter.Segmenter`. All four callables are
    keyword-only on purpose: they share a type, and a positional mix-up
    between ``cancel_generate`` and ``cancel_tts`` would be silent.

    Inputs (the state machine's whole alphabet):
    :meth:`on_speech_started`, :meth:`on_turn_committed`, :meth:`on_transcript`,
    :meth:`on_reply_text`, :meth:`on_audio_ready`, :meth:`deliver_next`,
    :meth:`tick`, :meth:`fail_stage`, :meth:`close`. Every one is total —
    defined from every state, returning ``False`` where it does not apply and
    never raising, because several of them are driven by a watchdog that can
    race a teardown.
    """

    def __init__(
        self,
        *,
        emit_event: EmitEvent,
        send_audio_chunk: SendAudioChunk,
        cancel_generate: Cancel,
        cancel_tts: Cancel,
        clock: Clock = timestamp_ms,
        barge_in_window_ms: int = DEFAULT_BARGE_IN_WINDOW_MS,
        transcribe_timeout_ms: int = DEFAULT_TRANSCRIBE_TIMEOUT_MS,
        generate_timeout_ms: int = DEFAULT_GENERATE_TIMEOUT_MS,
        tts_timeout_ms: int = DEFAULT_TTS_TIMEOUT_MS,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        sample_rate: int = TTS_SAMPLE_RATE,
    ) -> None:
        self._emit = emit_event
        self._send = send_audio_chunk
        self._cancel_generate = cancel_generate
        self._cancel_tts = cancel_tts
        self._clock = clock

        self.barge_in_window_ms = max(0, barge_in_window_ms)
        self._timeouts = {
            Stage.TRANSCRIBE: transcribe_timeout_ms,
            Stage.GENERATE: generate_timeout_ms,
            Stage.TTS: tts_timeout_ms,
        }
        # A chunk that split a PCM16 sample would desync playback from the
        # first frame; round down to whole samples, and never to zero (a
        # zero-byte chunk would deliver nothing, forever).
        self.chunk_bytes = max(
            BYTES_PER_SAMPLE, (chunk_bytes // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE
        )
        self.sample_rate = sample_rate

        self._state = FloorState.LISTENING
        self._turn_id = 0
        self._floor_taken_ms = 0  # when the machine took the floor (guard window)
        self._stage_started_ms = 0  # when the armed stage began (deadline)
        self._deadline_ms: int | None = None
        self._armed_stage: Stage | None = None
        self._reply_text = ""
        self._audio = b""
        self._offset = 0
        self._chunks_sent = 0

    # -- observation ------------------------------------------------------

    @property
    def state(self) -> FloorState:
        return self._state

    @property
    def turn_id(self) -> int:
        """Advances on every turn — pair it with the completion inputs to
        keep a stale answer off a later turn."""
        return self._turn_id

    @property
    def machine_holds_floor(self) -> bool:
        return self._state in MACHINE_HELD_STATES

    @property
    def synthesizing(self) -> bool:
        """Speaking, but the full-read synthesis has not returned yet."""
        return self._state is FloorState.SPEAKING and not self._audio

    @property
    def delivering(self) -> bool:
        """Speaking, with reply audio in hand and chunks going out."""
        return self._state is FloorState.SPEAKING and bool(self._audio)

    @property
    def armed_stage(self) -> Stage | None:
        """The stage whose deadline is currently armed, if any."""
        return self._armed_stage

    @property
    def stage_started_ms(self) -> int:
        return self._stage_started_ms

    @property
    def deadline_ms(self) -> int | None:
        """When :meth:`tick` will abandon the current stage (``None`` = unarmed)."""
        return self._deadline_ms

    @property
    def delivered_bytes(self) -> int:
        return self._offset

    @property
    def pending_audio_bytes(self) -> int:
        return len(self._audio) - self._offset

    # -- inputs -----------------------------------------------------------

    def on_speech_started(self) -> bool:
        """A VAD speech onset arrived. Returns ``True`` if it was a barge-in.

        Takes no timestamp on purpose: the segmenter's ``at_ms`` is quantised
        *audio-stream* time, and mixing it into this machine's clock domain
        would silently skew the guard window. The floor reads its own clock.
        """
        if not self.machine_holds_floor or not self._barge_in_armed():
            return False
        self._interrupt()
        return True

    def on_turn_committed(self) -> bool:
        """A turn was committed (the segmenter's ``SpeechStopped``).

        From ``listening`` this opens the turn. While the machine holds the
        floor it is an interruption first (past the guard window) and the new
        turn second — a committed turn is never discarded on the floor.
        Returns ``False`` only when the guard window swallowed it, or the
        session is closed.
        """
        if self._state is FloorState.CLOSED:
            return False
        if self.machine_holds_floor:
            if not self._barge_in_armed():
                return False
            self._interrupt()
        self._open_turn()
        return True

    def on_transcript(self, text: str, *, turn_id: int | None = None) -> bool:
        """The committed turn's transcript arrived.

        A blank transcript releases the floor without a response — silence is
        not something to answer, and it is not an error either.
        """
        if not self._accepts(FloorState.TRANSCRIBING, turn_id):
            return False
        self._disarm()
        if not text.strip():
            self._release()
            return True
        self._state = FloorState.RESPONDING
        self._arm(Stage.GENERATE)
        self._emit(ResponseStarted(at_ms=self._clock(), turn_id=self._turn_id))
        return True

    def on_reply_text(self, text: str, *, turn_id: int | None = None) -> bool:
        """The generate call returned. An empty reply is a named failure —
        the user gets a rendered error rather than unexplained silence."""
        if not self._accepts(FloorState.RESPONDING, turn_id):
            return False
        self._disarm()
        if not text.strip():
            self._fail(
                Stage.GENERATE,
                FailureReason.GENERATE_FAILED,
                "the generate lane returned an empty reply",
            )
            return True
        self._reply_text = text
        self._state = FloorState.SPEAKING
        self._arm(Stage.TTS)
        self._emit(ReplyText(at_ms=self._clock(), turn_id=self._turn_id, text=text))
        return True

    def on_audio_ready(self, pcm: bytes, *, turn_id: int | None = None) -> bool:
        """The full-read synthesis returned; delivery can begin.

        Empty audio is a named TTS failure, never a silently completed reply:
        :func:`lobes.realtime.tts_client.synthesize` returns ``b""`` on a soft
        failure, and rendering that as "the machine spoke" would be a lie.
        """
        if not self._accepts(FloorState.SPEAKING, turn_id) or self.delivering:
            return False
        self._disarm()
        if not pcm:
            self._fail(Stage.TTS, FailureReason.TTS_FAILED, "the tts lane returned no audio")
            return True
        self._audio = pcm
        self._offset = 0
        self._chunks_sent = 0
        return True

    def deliver_next(self) -> bool:
        """Send the next chunk of the reply. ``True`` if one went out.

        The route pumps this — ``while floor.deliver_next(): await ...`` — so
        the receive side keeps running between chunks and a barge-in can
        actually land mid-reply. On the final chunk the floor emits
        :class:`ResponseDone` and returns to ``listening``.
        """
        if not self.delivering or self._offset >= len(self._audio):
            return False
        chunk = self._audio[self._offset : self._offset + self.chunk_bytes]
        self._send(chunk)
        self._offset += len(chunk)
        self._chunks_sent += 1
        if self._offset >= len(self._audio):
            self._emit(
                ResponseDone(
                    at_ms=self._clock(),
                    turn_id=self._turn_id,
                    audio_ms=self._audio_ms(self._offset),
                    audio_bytes=self._offset,
                    chunks=self._chunks_sent,
                )
            )
            self._release()
        return True

    def tick(self) -> bool:
        """Expire the armed stage's deadline if it is due. ``True`` if it was.

        The route calls this from a watchdog: a wedged backend is, by
        definition, not calling anything else.
        """
        if self._deadline_ms is None or self._armed_stage is None:
            return False
        if self._clock() < self._deadline_ms:
            return False
        stage = self._armed_stage
        self._fail(
            stage,
            _TIMEOUT_REASON[stage],
            f"{stage.value} stage exceeded {self._timeouts[stage]}ms",
        )
        return True

    def fail_stage(
        self, reason: FailureReason, message: str, *, turn_id: int | None = None
    ) -> bool:
        """A backend failed by name (unreachable, non-2xx, ``role_infeasible``).

        Accepted only while that stage is the armed one, so a failure that
        surfaces after the floor moved on is ignored rather than tearing down
        an unrelated turn.
        """
        stage = _STAGE_OF_REASON[reason]
        if self._armed_stage is not stage:
            return False
        if turn_id is not None and turn_id != self._turn_id:
            return False
        self._fail(stage, reason, message)
        return True

    def close(self, reason: str = "client_disconnect") -> None:
        """Tear the floor down from ANY state. Idempotent.

        Cancels whatever was in flight and drops the undelivered audio, but
        emits nothing: session lifecycle events belong to the session engine,
        and a client that is already gone cannot act on an interruption event.
        """
        if self._state is FloorState.CLOSED:
            return
        if self.machine_holds_floor:
            self._cancel_both()
        self._release()
        self._state = FloorState.CLOSED

    # -- internals --------------------------------------------------------

    def _accepts(self, expected: FloorState, turn_id: int | None) -> bool:
        if self._state is not expected:
            return False
        return turn_id is None or turn_id == self._turn_id

    def _barge_in_armed(self) -> bool:
        return self._clock() - self._floor_taken_ms >= self.barge_in_window_ms

    def _open_turn(self) -> None:
        self._turn_id += 1
        self._state = FloorState.TRANSCRIBING
        self._floor_taken_ms = self._clock()
        self._arm(Stage.TRANSCRIBE)

    def _arm(self, stage: Stage) -> None:
        self._armed_stage = stage
        self._stage_started_ms = self._clock()
        self._deadline_ms = self._stage_started_ms + self._timeouts[stage]

    def _disarm(self) -> None:
        self._armed_stage = None
        self._deadline_ms = None

    def _cancel_both(self) -> None:
        # Both hooks, in a fixed order, on every abandonment — see the module
        # docstring's "cancel both, always".
        self._cancel_generate()
        self._cancel_tts()

    def _audio_ms(self, n_bytes: int) -> int:
        return n_bytes * 1000 // (self.sample_rate * BYTES_PER_SAMPLE)

    def _interrupt(self) -> None:
        delivered = self._offset
        total = len(self._audio)
        event = ResponseInterrupted(
            at_ms=self._clock(),
            turn_id=self._turn_id,
            stage=_STAGE_OF_STATE[self._state],
            audio_end_ms=self._audio_ms(delivered),
            audio_total_ms=self._audio_ms(total),
            delivered_bytes=delivered,
            undelivered_bytes=total - delivered,
            chunks_delivered=self._chunks_sent,
            reply_text=self._reply_text,
        )
        self._cancel_both()
        self._release()
        # Emitted last, so the floor is already consistent if the callback
        # re-enters this machine (a route may answer the event synchronously).
        self._emit(event)

    def _fail(self, stage: Stage, reason: FailureReason, message: str) -> None:
        event = ResponseFailed(
            at_ms=self._clock(),
            turn_id=self._turn_id,
            stage=stage,
            reason=reason,
            message=message,
        )
        self._cancel_both()
        self._release()
        self._emit(event)

    def _release(self) -> None:
        """Hand the floor back to the user and drop everything the turn held."""
        self._disarm()
        self._state = FloorState.LISTENING
        self._reply_text = ""
        self._audio = b""
        self._offset = 0
        self._chunks_sent = 0


__all__ = [
    "DEFAULT_BARGE_IN_WINDOW_MS",
    "DEFAULT_TRANSCRIBE_TIMEOUT_MS",
    "DEFAULT_GENERATE_TIMEOUT_MS",
    "DEFAULT_TTS_TIMEOUT_MS",
    "DEFAULT_CHUNK_MS",
    "DEFAULT_CHUNK_BYTES",
    "MACHINE_HELD_STATES",
    "FloorState",
    "Stage",
    "FailureReason",
    "ResponseStarted",
    "ReplyText",
    "ResponseDone",
    "ResponseInterrupted",
    "ResponseFailed",
    "FloorEvent",
    "estimate_spoken_prefix",
    "Floor",
]
