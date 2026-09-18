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

``responding`` has one detour: a generate reply that is a tool call moves to
``tool_wait`` instead of ``speaking`` (:meth:`Floor.on_tool_call`) — a fourth
machine-held state, added for issue #151 t6. The client runs the tool and
sends a result (:meth:`Floor.on_tool_result`), which returns the floor to
``responding`` and re-arms the generate deadline for the bridge's second
generate call. ``tool_wait`` arms its own deadline
(:data:`DEFAULT_TOOL_WAIT_TIMEOUT_MS`, a constructor argument — never
hardcoded policy) rather than the TTS one, and is interrupted by a barge-in
exactly like every other machine-held state, emitting
:class:`ResponseInterrupted` with ``stage=Stage.TOOL``.

``speaking`` covers TWO sub-stages, because the Chatterbox sidecar has no
streaming route — :func:`~lobes.realtime.tts_client.synthesize` is full-read,
so the bridge holds the *complete* segment PCM before the first byte can go out:

- **synthesizing** — the machine has committed to answering and TTS is in
  flight (:attr:`Floor.synthesizing`). Nothing is audible yet, but the floor
  is genuinely the machine's: the user's turn is over and an answer is coming.
- **delivering** — the PCM arrived and is going out as sequential chunks, one
  per :meth:`Floor.deliver_next` call (:attr:`Floor.delivering`).

A QUEUE of segments, not one buffer (deviation d7)
---------------------------------------------------
``speaking`` holds a LIST of segments, because sentence-level streaming
(:mod:`lobes.realtime._sentences`) releases each finished sentence to TTS
while generation continues. :meth:`Floor.on_reply_segment` appends one (the
last carrying ``final=True``); :meth:`Floor.on_reply_text` is the
one-segment convenience the non-streaming path keeps using, unchanged down to
the event it emits. Audio attaches to ITS segment
(``on_audio_ready(..., segment_index=N)``) and may arrive out of order;
delivery is strictly IN ORDER regardless, and :class:`ResponseDone` fires only
once the reply was marked final AND every segment has been delivered.

Deadlines follow the same fact: generation continues DURING ``speaking``, so
the ``generate`` deadline stays armed until the final segment arrives and only
then does the ``tts`` one take over. One consequence is deliberate — while
``generate`` is the armed stage, a TTS failure is still accepted
(:meth:`Floor.fail_stage`), because a synthesis that failed is real whether or
not its stage happens to hold the deadline, and an unspoken reply must never
be silent.

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

# A tool call round-trips through the client (issue #151 t6): the bridge sends
# the call out, the client runs the tool and sends a result back, and only
# then does a second generate call start. That is a genuinely longer wait than
# the other three stages — none of which involve a client round-trip — so it
# gets its own, larger default rather than reusing DEFAULT_GENERATE_TIMEOUT_MS.
# This is plan risk r5: a constructor argument, never hardcoded policy — the
# env knob (TOOL_WAIT_TIMEOUT_MS) is read by _settings.py, not by this module.
DEFAULT_TOOL_WAIT_TIMEOUT_MS = 120_000

# NOTE (issue #151 t6): this module used to define its own
# DEFAULT_CHUNK_MS/DEFAULT_CHUNK_BYTES (40ms / 1920 bytes). It no longer does.
# The wire codec ships the delta size (_wire.DEFAULT_DELTA_CHUNK_BYTES, 100ms /
# 4800 bytes) and chunk size is a WIRE-FRAMING concern, so `chunk_bytes` is now
# a REQUIRED constructor argument here — exactly as this module's docstring
# already anticipated ("If the wire codec ships its own delta size, the route
# passes it to the constructor — this module never imports it"). Requiring it,
# rather than keeping a second default in step with the first, is what makes a
# silent drift between the two impossible.


class FloorState(str, Enum):
    """Who holds the floor — explicit state, never inferred from event order."""

    LISTENING = "listening"  # the user's floor
    TRANSCRIBING = "transcribing"  # the turn is committed; STT is in flight
    RESPONDING = "responding"  # the generate call is in flight
    TOOL_WAIT = "tool_wait"  # the client is running a tool; a result is due
    SPEAKING = "speaking"  # TTS is in flight, then audio is being delivered
    CLOSED = "closed"


class Stage(str, Enum):
    """A machine-held stage — what a deadline bounds and an interruption cuts."""

    TRANSCRIBE = "transcribe"
    GENERATE = "generate"
    TOOL = "tool"
    TTS = "tts"


class FailureReason(str, Enum):
    """Named ways a stage ends badly — never a bare exception string.

    The route maps these onto the session schema's error codes;
    ``transcribe_failed`` is the existing ``stt_forward_failed``.
    """

    TRANSCRIBE_TIMEOUT = "transcribe_timeout"
    GENERATE_TIMEOUT = "generate_timeout"
    TOOL_WAIT_TIMEOUT = "tool_wait_timeout"
    TTS_TIMEOUT = "tts_timeout"
    TRANSCRIBE_FAILED = "transcribe_failed"
    GENERATE_FAILED = "generate_failed"
    TTS_FAILED = "tts_failed"


MACHINE_HELD_STATES = frozenset(
    {
        FloorState.TRANSCRIBING,
        FloorState.RESPONDING,
        FloorState.TOOL_WAIT,
        FloorState.SPEAKING,
    }
)

_STAGE_OF_STATE = {
    FloorState.TRANSCRIBING: Stage.TRANSCRIBE,
    FloorState.RESPONDING: Stage.GENERATE,
    FloorState.TOOL_WAIT: Stage.TOOL,
    FloorState.SPEAKING: Stage.TTS,
}

_TIMEOUT_REASON = {
    Stage.TRANSCRIBE: FailureReason.TRANSCRIBE_TIMEOUT,
    Stage.GENERATE: FailureReason.GENERATE_TIMEOUT,
    Stage.TOOL: FailureReason.TOOL_WAIT_TIMEOUT,
    Stage.TTS: FailureReason.TTS_TIMEOUT,
}

_STAGE_OF_REASON = {
    FailureReason.TRANSCRIBE_TIMEOUT: Stage.TRANSCRIBE,
    FailureReason.TRANSCRIBE_FAILED: Stage.TRANSCRIBE,
    FailureReason.GENERATE_TIMEOUT: Stage.GENERATE,
    FailureReason.GENERATE_FAILED: Stage.GENERATE,
    FailureReason.TOOL_WAIT_TIMEOUT: Stage.TOOL,
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
class ReplySegment:
    """ONE sentence of a streamed reply, ready to synthesize (deviation d7).

    The streaming counterpart of :class:`ReplyText`: emitted once per segment
    as generation releases it, carrying its ``index`` (the order delivery
    must respect) and whether it is the ``final`` one. The non-streaming
    path emits :class:`ReplyText` instead and never this — the two event
    types are what let the bridge tell "this is the whole reply, speak it"
    from "this is a piece, more is coming".
    """

    at_ms: int
    turn_id: int
    index: int
    text: str
    final: bool


@dataclass(frozen=True)
class ToolCallRequested:
    """The generate call returned a tool call instead of a spoken reply.

    Arms the TOOL deadline, never the TTS one — the reply is not yet spoken,
    and may never be spoken at all if the tool result leads to another tool
    call. ``call_id``/``name``/``arguments`` are opaque to this module (it
    never parses ``arguments``); it carries them only so the route can wire
    them onto :class:`~lobes.realtime._session.ResponseFunctionCallArgumentsDoneEvent`
    without this module importing ``_session``.
    """

    at_ms: int
    turn_id: int
    call_id: str
    name: str
    arguments: str


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
    # What the listener plausibly HEARD, computed across segments: the full
    # text of every segment delivered whole, plus the estimated prefix of the
    # one that was cut. For a single-segment (non-streaming) reply this is
    # exactly ``estimate_spoken_prefix(reply_text, audio_end_ms,
    # audio_total_ms)`` — additive, never a different answer. A caller
    # writing history reads THIS rather than re-deriving it, because the
    # proportional estimate is only correct within one segment.
    heard_text: str = ""


@dataclass(frozen=True)
class ResponseFailed:
    """A stage timed out or failed; the floor returned to the user."""

    at_ms: int
    turn_id: int
    stage: Stage
    reason: FailureReason
    message: str


FloorEvent = (
    ResponseStarted
    | ReplyText
    | ReplySegment
    | ToolCallRequested
    | ResponseDone
    | ResponseInterrupted
    | ResponseFailed
)

EmitEvent = Callable[[FloorEvent], None]
SendAudioChunk = Callable[[bytes], None]
Cancel = Callable[[], None]
Clock = Callable[[], int]


@dataclass
class _Segment:
    """One sentence of the reply and the audio that speaks it.

    Mutable and private: the queue's bookkeeping is the floor's own, and
    every fact a caller may need leaves through a frozen event.
    """

    index: int
    text: str
    audio: bytes | None = None
    offset: int = 0

    @property
    def total(self) -> int:
        return len(self.audio or b"")

    @property
    def drained(self) -> bool:
        return self.audio is not None and self.offset >= self.total


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

    ``chunk_bytes`` is REQUIRED, not defaulted (issue #151 t6): the outbound
    delta size belongs to the wire codec
    (:data:`lobes.realtime._wire.DEFAULT_DELTA_CHUNK_BYTES`), and a second
    default here would be a value free to drift out of step with it.
    ``sample_rate`` stays defaulted to the OUTPUT rate
    (:data:`~lobes.realtime.protocol.TTS_SAMPLE_RATE`, 24 kHz) — it is NOT
    the session's negotiated input rate, which may be 16 kHz.

    Inputs (the state machine's whole alphabet):
    :meth:`on_speech_started`, :meth:`on_turn_committed`, :meth:`on_transcript`,
    :meth:`on_reply_text`, :meth:`on_reply_segment`,
    :meth:`discard_reply_segments`, :meth:`on_tool_call`,
    :meth:`on_tool_result`, :meth:`on_audio_ready`, :meth:`deliver_next`,
    :meth:`tick`, :meth:`fail_stage`, :meth:`close`. Every one is total — defined from
    every state, returning ``False`` where it does not apply and never
    raising, because several of them are driven by a watchdog that can race a
    teardown.
    """

    def __init__(
        self,
        *,
        emit_event: EmitEvent,
        send_audio_chunk: SendAudioChunk,
        cancel_generate: Cancel,
        cancel_tts: Cancel,
        chunk_bytes: int,
        clock: Clock = timestamp_ms,
        barge_in_window_ms: int = DEFAULT_BARGE_IN_WINDOW_MS,
        transcribe_timeout_ms: int = DEFAULT_TRANSCRIBE_TIMEOUT_MS,
        generate_timeout_ms: int = DEFAULT_GENERATE_TIMEOUT_MS,
        tool_wait_timeout_ms: int = DEFAULT_TOOL_WAIT_TIMEOUT_MS,
        tts_timeout_ms: int = DEFAULT_TTS_TIMEOUT_MS,
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
            Stage.TOOL: tool_wait_timeout_ms,
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
        self._segments: list[_Segment] = []
        self._current = 0
        self._final_seen = False
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
        """Speaking, but the CURRENT segment's synthesis has not returned yet."""
        return self._state is FloorState.SPEAKING and not self._deliverable

    @property
    def delivering(self) -> bool:
        """Speaking, with the current segment's audio in hand and chunks going out."""
        return self._state is FloorState.SPEAKING and self._deliverable

    @property
    def segment_count(self) -> int:
        """How many reply segments this turn has queued (observation/tests)."""
        return len(self._segments)

    @property
    def _deliverable(self) -> bool:
        """Is there audio ready to send right now, in segment order?"""
        return any(
            segment.audio and not segment.drained
            for segment in self._segments[self._current : self._current + 1]
        )

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
        """Bytes actually sent, across every segment of this reply."""
        return sum(segment.offset for segment in self._segments)

    @property
    def pending_audio_bytes(self) -> int:
        """Bytes synthesized but not yet sent, across every segment."""
        return sum(segment.total - segment.offset for segment in self._segments)

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

    def on_continuation_onset(self) -> bool:
        """The speaker RESUMED right after an early commit (deviation d9, layer B).

        An interruption that deliberately ignores the barge-in guard window:
        the guard exists because an onset that soon after a commit is probably
        the tail of the user's own turn — which is exactly what a continuation
        IS, so here it is the signal rather than the noise. The caller (the
        bridge) owns the decision that this onset is a continuation; the floor
        only refuses what cannot be taken back: a ``tool_wait`` (the call is
        already in the client's hands) and a floor the machine does not hold.
        """
        if not self.machine_holds_floor or self._state is FloorState.TOOL_WAIT:
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
        """The generate call returned the WHOLE reply. An empty reply is a
        named failure — the user gets a rendered error rather than
        unexplained silence.

        The one-segment convenience over :meth:`on_reply_segment`: it accepts
        only from ``responding``, arms the TTS deadline at once (nothing is
        still generating), and emits :class:`ReplyText` rather than
        :class:`ReplySegment`, which is how the bridge tells a whole reply
        from a piece of one.
        """
        if not self._accepts(FloorState.RESPONDING, turn_id):
            return False
        return self._append_segment(text, final=True, streamed=False)

    def on_reply_segment(self, text: str, *, final: bool, turn_id: int | None = None) -> bool:
        """One SENTENCE of a streamed reply (deviation d7). ``True`` if taken.

        The first segment moves ``responding`` to ``speaking``; later ones
        append to the queue while the machine is already speaking. The
        ``generate`` deadline stays armed until *final* arrives, because
        generation is still running — only then does the TTS deadline take
        over, matching :meth:`on_reply_text` exactly.

        An empty *final* segment is how a stream whose flush produced nothing
        closes the reply: with real segments already queued it simply marks
        the end, and with NO segments at all it is the same empty-reply
        failure :meth:`on_reply_text` names.
        """
        if self._state is FloorState.RESPONDING:
            if not self._accepts(FloorState.RESPONDING, turn_id):
                return False
        elif not self._accepts(FloorState.SPEAKING, turn_id):
            return False
        return self._append_segment(text, final=final, streamed=True)

    def _append_segment(self, text: str, *, final: bool, streamed: bool) -> bool:
        """Queue one reply segment and re-point the deadline. Always ``True``."""
        if not text.strip():
            if not final:
                return False
            if not self._segments:
                self._disarm()
                self._fail(
                    Stage.GENERATE,
                    FailureReason.GENERATE_FAILED,
                    "the generate lane returned an empty reply",
                )
                return True
            self._final_seen = True
            self._rearm_speaking()
            self._complete_if_drained()
            return True
        segment = _Segment(index=len(self._segments), text=text)
        self._segments.append(segment)
        self._state = FloorState.SPEAKING
        self._final_seen = self._final_seen or final
        self._rearm_speaking()
        if streamed:
            self._emit(
                ReplySegment(
                    at_ms=self._clock(),
                    turn_id=self._turn_id,
                    index=segment.index,
                    text=text,
                    final=final,
                )
            )
        else:
            self._emit(ReplyText(at_ms=self._clock(), turn_id=self._turn_id, text=text))
        return True

    def _rearm_speaking(self) -> None:
        """Point the single armed deadline at whatever is actually pending.

        Before the final segment the answer is still being GENERATED, so the
        generate deadline stays armed (re-arming it would let a wedged stream
        run forever, one segment at a time). Once the reply is final, the
        wait is on synthesis — the TTS deadline — and once every segment has
        its audio there is nothing left to time out at all: delivery is
        paced by the route, not bounded by a backend.
        """
        if not self._final_seen:
            if self._armed_stage is not Stage.GENERATE:
                self._arm(Stage.GENERATE)
            return
        if any(segment.audio is None for segment in self._segments):
            if self._armed_stage is not Stage.TTS:
                self._arm(Stage.TTS)
            return
        self._disarm()

    def on_tool_call(
        self, *, call_id: str, name: str, arguments: str, turn_id: int | None = None
    ) -> bool:
        """The generate call returned a tool call rather than a spoken reply.

        Moves ``responding`` to ``tool_wait`` and arms the TOOL deadline —
        never the TTS one, since nothing is being synthesized yet and may
        never be, if the tool result leads to another tool call.
        """
        if not self._accepts(FloorState.RESPONDING, turn_id):
            return False
        self._disarm()
        self._state = FloorState.TOOL_WAIT
        self._arm(Stage.TOOL)
        self._emit(
            ToolCallRequested(
                at_ms=self._clock(),
                turn_id=self._turn_id,
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
        )
        return True

    def on_tool_result(self, *, turn_id: int | None = None) -> bool:
        """The client's tool result arrived; the bridge will re-generate.

        Returns the floor to ``responding`` and RE-ARMS the generate
        deadline — the bridge issues a second generate request with the tool
        result folded into history. Accepted only from ``tool_wait``; a
        result arriving in any other state is refused (``False``) without
        changing state, the same idiom every other completion input uses for
        a stale or out-of-order arrival.
        """
        if not self._accepts(FloorState.TOOL_WAIT, turn_id):
            return False
        self._disarm()
        self._state = FloorState.RESPONDING
        self._arm(Stage.GENERATE)
        return True

    def on_audio_ready(
        self, pcm: bytes, *, turn_id: int | None = None, segment_index: int = 0
    ) -> bool:
        """A segment's full-read synthesis returned; its delivery can begin.

        Empty audio is a named TTS failure, never a silently completed reply:
        :func:`lobes.realtime.tts_client.synthesize` returns ``b""`` on a soft
        failure, and rendering that as "the machine spoke" would be a lie.

        *segment_index* defaults to ``0`` — the only segment a non-streaming
        reply has — so every pre-streaming caller is unchanged. Segments may
        finish synthesis OUT OF ORDER (the route is free to parallelize);
        delivery still runs strictly in order. Audio for a segment that was
        never announced, or that already has some, is refused: both mean the
        route and the floor disagree about the reply, and guessing which is
        right would speak the wrong bytes.
        """
        if not self._accepts(FloorState.SPEAKING, turn_id):
            return False
        segment = self._segment(segment_index)
        if segment is None or segment.audio is not None:
            return False
        if not pcm:
            self._disarm()
            self._fail(Stage.TTS, FailureReason.TTS_FAILED, "the tts lane returned no audio")
            return True
        segment.audio = pcm
        self._rearm_speaking()
        return True

    def deliver_next(self) -> bool:
        """Send the next chunk of the reply. ``True`` if one went out.

        The route pumps this — ``while floor.deliver_next(): await ...`` — so
        the receive side keeps running between chunks and a barge-in can
        actually land mid-reply. Segments drain IN ORDER: a later segment
        whose audio arrived first waits its turn, because playback order is
        the reply's meaning. ``False`` means "nothing to send RIGHT NOW",
        which under streaming may simply be "the next segment is still
        synthesizing" — the route keeps pumping until the response ends.

        On the final chunk of the FINAL segment the floor emits
        :class:`ResponseDone` and returns to ``listening``.
        """
        if self._state is not FloorState.SPEAKING:
            return False
        segment = self._segment(self._current)
        if segment is None or segment.audio is None or segment.drained:
            return False
        chunk = segment.audio[segment.offset : segment.offset + self.chunk_bytes]
        self._send(chunk)
        segment.offset += len(chunk)
        self._chunks_sent += 1
        if not segment.drained:
            return True
        self._current += 1
        self._complete_if_drained()
        return True

    def _complete_if_drained(self) -> None:
        """Emit :class:`ResponseDone` once the FINAL segment has been sent.

        Called from both sides of a race the streamed path can genuinely
        lose: delivery may drain the only segment BEFORE the stream ends (no
        later chunk for the completion to ride on), or the final marker may
        land first and the last chunk after. Whichever arrives second
        completes the response; without this the floor sits in ``speaking``
        forever and the route's pump spins on a reply that can never finish.
        """
        if self._state is not FloorState.SPEAKING or not self._final_seen:
            return
        if self._current < len(self._segments):
            return
        delivered = self.delivered_bytes
        self._emit(
            ResponseDone(
                at_ms=self._clock(),
                turn_id=self._turn_id,
                audio_ms=self._audio_ms(delivered),
                audio_bytes=delivered,
                chunks=self._chunks_sent,
            )
        )
        self._release()

    def discard_reply_segments(self, *, turn_id: int | None = None) -> bool:
        """Abandon a streamed reply's queued segments; back to ``responding``.

        The one thing a stream can do that a whole-reply call cannot: emit
        some text and THEN call a tool. The text is not the answer any more,
        so every queued segment — and every byte of audio already synthesized
        for it — is dropped rather than spoken, and the floor returns to
        ``responding`` with the generate deadline re-armed so
        :meth:`on_tool_call` can take it from there.

        Whatever was already DELIVERED is already spoken and cannot be
        unsaid; in practice nothing is, because a tool-call fragment arrives
        long before the first synthesis returns. Refused (``False``) from any
        state but ``speaking``.
        """
        if not self._accepts(FloorState.SPEAKING, turn_id):
            return False
        self._segments = []
        self._current = 0
        self._chunks_sent = 0
        self._final_seen = False
        self._state = FloorState.RESPONDING
        self._arm(Stage.GENERATE)
        return True

    def _segment(self, index: int) -> _Segment | None:
        if 0 <= index < len(self._segments):
            return self._segments[index]
        return None

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

        Accepted while that stage is the armed one, so a failure that
        surfaces after the floor moved on is ignored rather than tearing down
        an unrelated turn — with ONE named exception (deviation d7): while a
        streamed reply is still generating, ``generate`` holds the deadline
        even though TTS is genuinely in flight on an earlier segment, so a
        TTS failure is accepted throughout ``speaking``. Refusing it there
        would leave the response wedged until the generate deadline expired
        and then blame the wrong stage.
        """
        stage = _STAGE_OF_REASON[reason]
        if not self._accepts_failure(stage):
            return False
        if turn_id is not None and turn_id != self._turn_id:
            return False
        self._fail(stage, reason, message)
        return True

    def close(self) -> None:
        """Tear the floor down from ANY state. Idempotent.

        Cancels whatever was in flight and drops the undelivered audio, but
        emits nothing: session lifecycle events belong to the session engine,
        and a client that is already gone cannot act on an interruption event.

        Takes no ``reason``. It used to, mirroring ``Session.teardown``, but the
        floor emits nothing on close so the value was discarded — every caller
        naming one was expressing an intent that went nowhere. The reason still
        travels where it is actually rendered: ``Session.teardown(reason=...)``
        puts it on the ``session.closed`` event.
        """
        if self._state is FloorState.CLOSED:
            return
        if self.machine_holds_floor:
            self._cancel_both()
        self._release()
        self._state = FloorState.CLOSED

    # -- internals --------------------------------------------------------

    def _accepts_failure(self, stage: Stage) -> bool:
        if self._armed_stage is stage:
            return True
        # See fail_stage's docstring: a streamed reply speaks and generates at
        # the same time, and only one stage can hold the deadline.
        return stage is Stage.TTS and self._state is FloorState.SPEAKING

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

    def _reply_text_so_far(self) -> str:
        """Every segment released so far, as one reply string."""
        return " ".join(segment.text for segment in self._segments).strip()

    def _heard_text(self) -> str:
        """What the listener plausibly heard, segment by segment.

        A proportional estimate is only meaningful WITHIN one segment: a
        later segment may have no audio at all yet, so measuring the cut
        against the whole reply's byte total would claim the user heard a
        sentence that was never synthesized. So: every fully-delivered
        segment contributes its text verbatim, and only the segment being
        delivered is estimated (:func:`estimate_spoken_prefix`).
        """
        heard: list[str] = []
        for segment in self._segments:
            if segment.offset <= 0:
                break
            if segment.drained:
                heard.append(segment.text)
                continue
            prefix = estimate_spoken_prefix(
                segment.text,
                self._audio_ms(segment.offset),
                self._audio_ms(segment.total),
            )
            if prefix:
                heard.append(prefix)
            break
        return " ".join(heard).strip()

    def _interrupt(self) -> None:
        delivered = self.delivered_bytes
        total = sum(segment.total for segment in self._segments)
        event = ResponseInterrupted(
            at_ms=self._clock(),
            turn_id=self._turn_id,
            stage=_STAGE_OF_STATE[self._state],
            audio_end_ms=self._audio_ms(delivered),
            audio_total_ms=self._audio_ms(total),
            delivered_bytes=delivered,
            undelivered_bytes=total - delivered,
            chunks_delivered=self._chunks_sent,
            reply_text=self._reply_text_so_far(),
            heard_text=self._heard_text(),
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
        self._segments = []
        self._current = 0
        self._final_seen = False
        self._chunks_sent = 0


__all__ = [
    "DEFAULT_BARGE_IN_WINDOW_MS",
    "DEFAULT_TRANSCRIBE_TIMEOUT_MS",
    "DEFAULT_GENERATE_TIMEOUT_MS",
    "DEFAULT_TOOL_WAIT_TIMEOUT_MS",
    "DEFAULT_TTS_TIMEOUT_MS",
    "MACHINE_HELD_STATES",
    "FloorState",
    "Stage",
    "FailureReason",
    "ResponseStarted",
    "ReplyText",
    "ReplySegment",
    "ToolCallRequested",
    "ResponseDone",
    "ResponseInterrupted",
    "ResponseFailed",
    "FloorEvent",
    "estimate_spoken_prefix",
    "Floor",
]
