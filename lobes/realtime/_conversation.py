"""The conversation bridge — where the five #151 modules become one turn.

Issue #151 t6 is the convergence task: :mod:`._segmenter` finds turn
boundaries, :mod:`._floor` decides who holds the floor, :mod:`._session`
owns the event schema and the session's history, :mod:`._turn` shapes the
generate call, and :mod:`._wire` frames audio in both directions. Each was
built in isolation and none of them imports another. This module is the one
place that imports them all and states, in ordinary Python, how a spoken turn
becomes a spoken reply — so that ``app.py`` stays what it has always been: a
``pragma: no cover`` shell that owns sockets, threads and HTTP, and no
decisions.

Stdlib only. Nothing here opens a socket, awaits anything, or imports
``fastapi``/``httpx``/``torch``/``numpy`` — so the whole convergence is
unit-testable in the offline CI env that never installs the ``[realtime]``
extra, exactly like every module it wires together. It does not import
:mod:`._settings` either: the route resolves env-derived values and passes
them in, mirroring how :mod:`._segmenter` and :mod:`._floor` take their
tuning through their constructors.

Opt-in, and only opt-in
------------------------
:class:`ConversationBridge` starts DISARMED. Until a ``response.create``
event arrives (:func:`is_response_create`), the floor machine is constructed
but never driven: a session gets exactly the #149 transcription-only event
sequence — ``session.created``, boundaries, transcripts, named errors — and
nothing else. That is the ears-only contract the spec pins for
reachy-mini-cli, and it is a structural property here, not a behavioural
promise: every floor call in this module sits behind ``if self.armed``.

Arming is session-level and idempotent. A client may send ``response.create``
once, at connect, and get a reply to every committed turn thereafter; or send
it after each transcript, OpenAI-style. The second shape works because a
transcript the floor did not take is remembered as the *pending* transcript
and answered by the next ``response.create`` — and cleared once answered, so
a duplicate trigger can never produce two replies to one turn.

Two clock domains, kept apart
------------------------------
The segmenter's ``at_ms`` is 32ms-quantised **audio-stream** time. The
floor's clock is monotonic wall-clock milliseconds. They are never mixed:
``at_ms`` goes onto the wire (``SpeechStartedEvent.at_ms`` /
``SpeechStoppedEvent.at_ms``, so an operator can see VAD boundaries and tune
the knobs against a live session) and is NEVER passed into the floor —
:meth:`lobes.realtime._floor.Floor.on_speech_started` takes no timestamp for
exactly this reason.

Error vocabulary — one enumerable list
---------------------------------------
Three modules named failures independently. This module is where they
collapse onto :class:`lobes.realtime._session.ErrorCode`, the single
vocabulary a client renders:

===============================  ==========================  ==============
``_floor.FailureReason``         ``_session.ErrorCode``       stage in text?
===============================  ==========================  ==============
``transcribe_failed``            ``stt_forward_failed``       code says it
``generate_failed``              ``generate_failed``          code says it
``tts_failed``                   ``tts_failed``               code says it
``transcribe_timeout``           ``response_timeout``         **yes**
``generate_timeout``             ``response_timeout``         **yes**
``tts_timeout``                  ``response_timeout``         **yes**
===============================  ==========================  ==============

``transcribe_failed`` reuses the EXISTING ``stt_forward_failed`` code rather
than minting a new one — a committed turn's Parakeet forward failing is the
same event whether or not the session went on to answer it, and it is emitted
through the same :meth:`~lobes.realtime._session.Session.fail_transcription`
call, so the armed and ears-only paths produce an identical error event.

The three timeouts share ONE code, so :func:`describe_failure` guarantees the
stage name is in the message text. Without that an operator reading
``response_timeout`` could not tell which stage wedged — the code alone is
ambiguous by construction, and that ambiguity is the price of not growing the
client-visible vocabulary by three near-identical members.

:data:`WIRE_ERROR_CODES` closes the other half of the same split: before t6,
``app.py`` put a :class:`lobes.realtime._wire.WireErrorCode` value into
``ErrorEvent.code`` verbatim, so the field documented as "always a named
``_session.ErrorCode``" could also carry ``invalid_json`` /
``invalid_append_event`` / ``unsupported_frame_type``. All three now map onto
the single :attr:`~lobes.realtime._session.ErrorCode.INVALID_WIRE_EVENT`,
with the wire reason named in the message text — the same trade the timeouts
make, and the same one ``invalid_session_config`` has always made.

Tools — one dispatch point, one outstanding call
-------------------------------------------------
:meth:`ConversationBridge.on_control_event` is the SINGLE place a non-audio
client event is acted on. It handles exactly three: ``response.create`` (the
opt-in above), ``session.update`` (declare/retract tools, tool_choice or the
language mid-session) and ``conversation.item.create`` carrying a
``function_call_output`` (a tool result). Everything else stays silently
ignored, exactly as before.

A tool turn is the ordinary turn with one extra leg. ``app.py`` drives it
with the SAME four calls it already makes — the difference is only what they
return:

1. ``take_pending_response()`` → ``build_generate_request(turn_id)`` →
   POST → :meth:`~ConversationBridge.on_generate_response`. When the reply
   is a tool call rather than text, the bridge records it, the floor moves to
   ``tool_wait``, and the client gets ONE
   ``response.function_call_arguments.done``. Nothing is synthesized, so
   :meth:`take_pending_synthesis` answers ``None`` — the route must not start
   TTS. :attr:`~ConversationBridge.awaiting_tool_result` says so explicitly,
   for a route that would rather ask than infer it from a ``None``.
2. The client runs the tool and sends ``conversation.item.create``. The
   bridge records the result in history and waits: NO generate goes out yet,
   because the conversation surface is opt-in per response (the same rule
   that makes an ears-only session possible at all).
3. The client's next ``response.create`` releases the floor's tool wait and
   sets a pending response for the SAME turn id, so the route's existing
   poll — ``take_pending_response()`` → ``build_generate_request`` → POST —
   issues the follow-up generate with the tool result folded into history.
   No second ``response.created`` event is emitted: it is one response,
   continued.

Exactly ONE call may be outstanding at a time. A result that answers
anything else is refused with a named ``invalid_wire_event`` error carrying
one of three reason tokens — :data:`TOOL_OUTPUT_UNKNOWN_CALL_ID`,
:data:`TOOL_OUTPUT_CALL_CLOSED`, :data:`TOOL_OUTPUT_DUPLICATE` — and history
is left byte-identical, because a backend cannot be relied on to notice the
mistake: PROBED 2026-09-18, ``associate`` repeats an orphan tool result back
as fact rather than rejecting it.

Whenever a turn holding an unanswered call ends — a barge-in, an expired
tool wait, any other failure — the bridge appends a synthetic
:data:`TOOL_CALL_CANCELLED_OUTPUT` tool message so the assistant's
``tool_calls`` entry is never left dangling in history (a chat-completions
backend rejects that shape outright), and the call id is remembered as
CLOSED so a late result is named ``call_closed`` rather than mistaken for an
orphan.

Per-stage timings (``StageTimings``) are deliberately NOT computed here —
that is a separate task's surface; this module only makes sure the tool wait
is a stage the floor actually arms, so a timing hook has something to read.

What the route still owns
--------------------------
Sockets, threads, HTTP, tasks and time. Concretely: ``app.py`` awaits the
generate POST and the TTS synthesis, hands the raw results back here, and
**pumps** — it calls :meth:`ConversationBridge.deliver_next` with an
``await`` between chunks (so the receive loop keeps running and a barge-in
can actually land mid-reply) and calls :meth:`ConversationBridge.tick` from a
watchdog every :data:`WATCHDOG_INTERVAL_MS` (a wedged backend is, by
definition, not calling anything else, so a deadline that only expires inside
``tick`` needs someone to keep calling it). Wire either of those as a tight
synchronous loop and every guarantee in :mod:`._floor` is inert at runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Callable

from ._floor import (
    DEFAULT_BARGE_IN_WINDOW_MS,
    DEFAULT_GENERATE_TIMEOUT_MS,
    DEFAULT_TOOL_WAIT_TIMEOUT_MS,
    DEFAULT_TRANSCRIBE_TIMEOUT_MS,
    DEFAULT_TTS_TIMEOUT_MS,
    FailureReason,
    Floor,
    FloorState,
    ReplySegment,
    ReplyText,
    ResponseDone,
    ResponseFailed,
    ResponseInterrupted,
    ResponseStarted,
    ToolCallRequested,
    estimate_spoken_prefix,
)
from ._sentences import DEFAULT_EAGER_FIRST_MIN_CHARS, SentenceChunker
from ._session import (
    ErrorCode,
    ErrorEvent,
    Event,
    FunctionCallOutputError,
    Session,
    SessionConfigError,
    StageTimings,
    event_to_dict,
    parse_function_call_output,
)
from ._turn import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    RoleInfeasibleError,
    ToolCallResult,
    TurnRequest,
    TurnRequestError,
    assistant_tool_call_message,
    build_turn_request,
    parse_turn_response,
    tool_result_message,
)
from ._wire import (
    DEFAULT_DELTA_CHUNK_BYTES,
    WireErrorCode,
    WireFormatError,
    encode_audio_chunk,
    is_function_call_output,
    is_session_update,
)
from .protocol import BYTES_PER_SAMPLE, TTS_SAMPLE_RATE, timestamp_ms

# The client's opt-in. OpenAI-Realtime's own event name, adopted for its
# SHAPE only — full parity (session.update semantics, the conversation-item
# schema, tool calls over the session) is an explicitly parked follow-up, so
# nothing here reads the event's body.
RESPONSE_CREATE_EVENT_TYPE = "response.create"

# The voice lane's default generate model when OPENAI_MODEL is unset (spec
# claim c4 / honesty h4). _turn.py is deliberately policy-free about this —
# a falsy model there OMITS the key and the gateway default-routes to the
# primary — so the policy lives here, at the wiring layer, which is the only
# place that knows this is the VOICE lane. Measured in-tree
# (scripts/realtime-voice-loop.py): the Gemma 4 12B multimodal lane answers a
# short spoken turn in ~1s, where the 27B cortex lane spends that on a
# reasoning trace nobody hears.
DEFAULT_VOICE_MODEL = "multimodal"

# How often the route's watchdog should call ConversationBridge.tick().
# 250ms is far below every per-stage deadline (60s) — the interval bounds how
# LATE a timeout fires, not whether it fires — and far above any cost worth
# counting for one sleeping task per armed session. A module constant, not a
# Settings field, on purpose: a new Settings field must be threaded through
# docker-compose.audio.yml AND env.audio.example (the #149 s4 lesson, pinned
# by tests/test_realtime_audio_env_coverage.py), and there is no operator
# question here that a knob would answer.
WATCHDOG_INTERVAL_MS = 250


# ---------------------------------------------------------------------------
# Error vocabulary — see the module docstring's table.
# ---------------------------------------------------------------------------

FAILURE_ERROR_CODES: dict[FailureReason, ErrorCode] = {
    FailureReason.TRANSCRIBE_FAILED: ErrorCode.STT_FORWARD_FAILED,
    FailureReason.GENERATE_FAILED: ErrorCode.GENERATE_FAILED,
    FailureReason.TTS_FAILED: ErrorCode.TTS_FAILED,
    FailureReason.TRANSCRIBE_TIMEOUT: ErrorCode.RESPONSE_TIMEOUT,
    FailureReason.GENERATE_TIMEOUT: ErrorCode.RESPONSE_TIMEOUT,
    FailureReason.TOOL_WAIT_TIMEOUT: ErrorCode.RESPONSE_TIMEOUT,
    FailureReason.TTS_TIMEOUT: ErrorCode.RESPONSE_TIMEOUT,
}

# Reason tokens for a tool result this session cannot attribute. They live in
# the MESSAGE text, never as new ErrorCode/WireErrorCode members: one
# enumerable list of client-visible codes is a contract (see
# ErrorCode.INVALID_WIRE_EVENT's own docstring), so a new failure mode earns a
# reason, not a code — the same trade FunctionCallOutputError already makes.
TOOL_OUTPUT_UNKNOWN_CALL_ID = "unknown_call_id"
TOOL_OUTPUT_CALL_CLOSED = "call_closed"
TOOL_OUTPUT_DUPLICATE = "duplicate_output"

_TOOL_OUTPUT_REJECTIONS = {
    TOOL_OUTPUT_UNKNOWN_CALL_ID: "no tool call with call_id {call_id!r} is outstanding",
    TOOL_OUTPUT_CALL_CLOSED: "the tool call {call_id!r} was closed before this result arrived",
    TOOL_OUTPUT_DUPLICATE: "the tool call {call_id!r} was already answered",
}

# What a tool call that was abandoned before its result arrived records as its
# result. Deliberately a fixed ENGLISH marker, not a translated one: it is
# model-facing context, not a spoken or client-rendered string, and the
# Hebrew voice lane's own replies come from the model, not from here.
TOOL_CALL_CANCELLED_OUTPUT = "cancelled: the tool call was interrupted before a result arrived"

# How many recently-closed call ids to remember, so a late result is named
# `call_closed` rather than `unknown_call_id`. Bounded because a session is
# long-lived and one id per tool call would otherwise grow without limit; a
# result that arrives more than this many calls late is indistinguishable
# from an orphan anyway, and both are refused.
CLOSED_CALL_MEMORY = 16

# What the message says when a retryable status carried no Retry-After. An
# explicit "the gateway named no delay" beats omitting the field, which reads
# as "nobody looked".
RETRY_AFTER_UNSPECIFIED = "unspecified"

# Statuses whose whole point is "come back later", so the absence of a
# Retry-After is itself worth stating.
RETRYABLE_GENERATE_STATUSES = (429, 503)

WIRE_ERROR_CODES: dict[WireErrorCode, ErrorCode] = {
    WireErrorCode.INVALID_JSON: ErrorCode.INVALID_WIRE_EVENT,
    WireErrorCode.INVALID_APPEND_EVENT: ErrorCode.INVALID_WIRE_EVENT,
    WireErrorCode.UNSUPPORTED_FRAME_TYPE: ErrorCode.INVALID_WIRE_EVENT,
}


def describe_failure(event: ResponseFailed) -> str:
    """The message text for one :class:`~lobes.realtime._floor.ResponseFailed`.

    Passes a named failure's message through unchanged — ``generate_failed``,
    ``tts_failed`` and ``stt_forward_failed`` each identify their own stage.
    For the three reasons that collapse onto ``response_timeout``, GUARANTEES
    the stage is named, since the code cannot say it: the floor's own
    tick-generated message already opens with ``"<stage> stage …"`` and is
    left alone, and any other message (a route-observed backend read timeout,
    say) is prefixed. Idempotent either way — calling it twice cannot
    double-prefix.
    """
    message = event.message
    if FAILURE_ERROR_CODES[event.reason] is not ErrorCode.RESPONSE_TIMEOUT:
        return message
    stage = event.stage.value
    return message if message.startswith(f"{stage} stage") else f"{stage} stage: {message}"


def describe_wire_error(exc: WireFormatError) -> str:
    """Message text for a malformed client frame, naming the wire reason.

    All three :class:`~lobes.realtime._wire.WireErrorCode` values map onto
    the single ``invalid_wire_event`` code, so the specific reason has to
    survive in the text or it is lost.
    """
    return f"{exc.code.value}: {exc}"


def describe_role_infeasible(exc: RoleInfeasibleError) -> str:
    """Message text for a gateway ``404 role_infeasible`` on the generate lane.

    Carries the operator-declared ``hosted_by`` peer hint through to the
    client (spec claim c4's instruction) rather than dropping it — the whole
    point of honest referral is that the caller learns WHERE the lane lives.
    Never a fallback to a different lane: :mod:`._turn` raises this exact
    exception type precisely so that "the lane does not exist here" cannot be
    mistaken for "the call failed".
    """
    return f"{exc} (hosted_by={exc.hosted_by})" if exc.hosted_by else str(exc)


def retry_after_from_headers(headers: Mapping[str, str] | None) -> str | None:
    """The response's ``Retry-After`` value, matched case-insensitively.

    ``None`` when there is no such header (or no headers at all) — the route
    may or may not have them, and an absent hint is a fact worth reporting,
    not a reason to fail.
    """
    if not headers:
        return None
    for key, value in headers.items():
        if key.lower() == "retry-after":
            text = str(value).strip()
            return text or None
    return None


def describe_generate_http_failure(
    message: str,
    *,
    status_code: int,
    retry_after: str | None = None,
    hosted_by: str | None = None,
) -> str:
    """Append the machine-readable half of a generate failure to *message*.

    The gateway supplies its own prose for every shed and refusal
    (``"<lane> is under pressure; retry shortly"``), so the numeric status,
    the retry hint and the peer that actually hosts the lane are otherwise
    nowhere on the wire — a client could only recover them by pattern-matching
    English. This suffix carries them as ``key=value`` pairs instead, in the
    same "the code cannot say it, so the text must" idiom
    :func:`describe_failure` uses for the three timeouts.

    ``hosted_by`` is omitted entirely when there is none, so a failure with no
    declared peer never mentions the word — a caller can test for its
    presence.
    """
    details = [f"status={status_code}"]
    if retry_after is not None:
        details.append(f"retry_after={retry_after}")
    elif status_code in RETRYABLE_GENERATE_STATUSES:
        details.append(f"retry_after={RETRY_AFTER_UNSPECIFIED}")
    if hosted_by:
        details.append(f"hosted_by={hosted_by}")
    return f"{message} ({', '.join(details)})"


def resolve_voice_model(configured: str | None) -> str:
    """The generate model for the voice lane: *configured*, else the default.

    *configured* is ``Settings.openai_model`` (``""`` when ``OPENAI_MODEL`` is
    unset). An operator who sets it wins outright — including setting it to a
    lane this box does not host, which surfaces as a NAMED
    ``role_infeasible``-derived error rather than a silent substitution.
    """
    return configured or DEFAULT_VOICE_MODEL


def is_response_create(payload: Mapping[str, object] | None) -> bool:
    """Is this decoded client event the conversation opt-in trigger?

    Takes the payload :func:`lobes.realtime._wire.decide_inbound_message`
    already handed back with an ``IGNORED`` decision, so the frame is parsed
    once. Any other well-formed event stays ignored, exactly as before issue
    #151 t6 — this module adopts the audio-path event SHAPES only.
    """
    return bool(payload) and payload.get("type") == RESPONSE_CREATE_EVENT_TYPE


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------


# How far AHEAD of the playhead the server is allowed to run while delivering
# audio-out. Some lead is required — a client whose buffer runs dry stutters —
# but it is also the barge-in blind spot: an onset arriving inside the lead is
# an interruption the server cannot honour, because those bytes are already
# gone. 400 ms is comfortably above any local socket's jitter and well under
# the ~1 s it takes a human to hear a wrong answer and start objecting.
DELIVERY_LEAD_MS = 400


def delivery_pause_ms(
    *,
    chunks_sent: int,
    chunk_bytes: int,
    sample_rate: int,
    elapsed_ms: int,
    lead_ms: int = DELIVERY_LEAD_MS,
) -> int:
    """Milliseconds to wait before sending the NEXT audio chunk.

    Delivery has to track PLAYBACK, not socket drain. Without this the route
    pumps every chunk as fast as the socket accepts it — MEASURED live at 2-4 ms
    for 7.5-8.5 s of audio (docs/evidence/2026-07-22-accept-realtime-voice-to-
    voice-spark.txt) — and then leaves SPEAKING. The client is still playing for
    seconds afterwards, so a user talking over the reply is, to the server,
    talking while LISTENING: it opens a new turn and `response.interrupted` is
    never emitted. Every barge-in guarantee in :mod:`lobes.realtime._floor` is
    correct and completely inert.

    Pacing also keeps the session's HISTORY honest. The floor trims an
    interrupted reply to the prefix that was plausibly heard; with instant
    delivery nothing is ever undelivered, so the machine records the whole reply
    as spoken and carries on as though the user heard words they never did.

    Returns 0 whenever delivery is already at or behind the playhead (the first
    chunks, or a slow socket) — this only ever *slows* a run-ahead, never adds
    latency to audio the client is waiting on.
    """
    if chunk_bytes <= 0 or sample_rate <= 0:
        return 0
    chunk_ms = (chunk_bytes / BYTES_PER_SAMPLE) * 1000 / sample_rate
    delivered_ms = chunks_sent * chunk_ms
    # We may run `lead_ms` ahead of real time; wait for the excess to elapse.
    return max(0, int(delivered_ms - lead_ms - elapsed_ms))


@dataclass(frozen=True)
class OutstandingToolCall:
    """The ONE tool call this session is waiting on, and whether it is answered.

    ``answered`` is the difference between "the client is still running the
    tool" (a second result would be a duplicate; the turn ending must write a
    synthetic cancellation into history) and "the result is in history,
    waiting for the ``response.create`` that releases the follow-up generate"
    (a second result is still a duplicate, but nothing needs cancelling —
    the real result is already recorded).
    """

    call_id: str
    name: str
    turn_id: int
    answered: bool = False


@dataclass(frozen=True)
class GenerateConfig:
    """Where the voice lane's generate call goes, and how it is shaped.

    These five travel together by construction: every one of them is consumed
    by the SAME :func:`~lobes.realtime._turn.build_turn_request` call and
    nothing else reads them individually, so passing them as five separate
    constructor arguments only spread one decision across five call-site lines.

    ``model`` stays deliberately policy-free here, as in :mod:`_turn`: the
    voice-lane default (``multimodal``) is applied by
    :func:`resolve_voice_model` at the wiring layer, not baked in.
    """

    base_url: str
    api_key: str | None = None
    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    # Ask the backend to STREAM the reply (approved deviation d7). Lives here
    # rather than as a bridge argument for the same reason as the other five:
    # it is consumed by the one build_turn_request call and nothing else.
    # ``False`` — what GENERATE_STREAM=false renders — leaves the request body
    # byte-identical to a pre-streaming deployment.
    stream: bool = False
    # The streamed reply's first-clause threshold (REPLY_FIRST_CLAUSE_MIN_CHARS)
    # — consumed only by this bridge's SentenceChunker.
    first_clause_min_chars: int = DEFAULT_EAGER_FIRST_MIN_CHARS


class ConversationBridge:
    """One session's conversation surface: floor + schema + history + wire.

    Construct one per WebSocket, alongside the session's
    :class:`~lobes.realtime._segmenter.Segmenter`. Every method is
    SYNCHRONOUS and never raises on an out-of-order input — the route drives
    it from a receive loop, a response task and a watchdog, and in a
    single-threaded event loop a synchronous method has no await point for
    those to interleave at, which is what keeps a barge-in from landing
    halfway through a transition.

    Output is an ORDERED OUTBOX of already-serialized wire payloads, not a
    socket: every event this bridge produces — boundary, transcript,
    response-lifecycle, audio delta, named error — lands in one list, and the
    route :meth:`drain`\\ s and sends it. One list means one order, so an
    interruption event can never overtake the last delta that preceded it,
    however many coroutines were producing.
    """

    def __init__(
        self,
        session: Session,
        *,
        cancel_generate: Callable[[], None],
        cancel_tts: Callable[[], None],
        generate: GenerateConfig,
        barge_in_window_ms: int = DEFAULT_BARGE_IN_WINDOW_MS,
        transcribe_timeout_ms: int = DEFAULT_TRANSCRIBE_TIMEOUT_MS,
        generate_timeout_ms: int = DEFAULT_GENERATE_TIMEOUT_MS,
        tool_wait_timeout_ms: int = DEFAULT_TOOL_WAIT_TIMEOUT_MS,
        tts_timeout_ms: int = DEFAULT_TTS_TIMEOUT_MS,
        chunk_bytes: int = DEFAULT_DELTA_CHUNK_BYTES,
        clock: Callable[[], int] = timestamp_ms,
        continuation_window_ms: int = 0,
        continuation_tool_hold_ms: int = 0,
    ) -> None:
        self.session = session
        self.floor = Floor(
            emit_event=self._on_floor_event,
            send_audio_chunk=self._on_audio_chunk,
            cancel_generate=cancel_generate,
            cancel_tts=cancel_tts,
            clock=clock,
            barge_in_window_ms=barge_in_window_ms,
            transcribe_timeout_ms=transcribe_timeout_ms,
            generate_timeout_ms=generate_timeout_ms,
            tool_wait_timeout_ms=tool_wait_timeout_ms,
            tts_timeout_ms=tts_timeout_ms,
            chunk_bytes=chunk_bytes,
            # The OUTPUT rate — Chatterbox's 24 kHz, which protocol.py pins
            # equal to CLIENT_SAMPLE_RATE so audio-out never resamples. NOT
            # the session's negotiated INPUT rate (which may be 16 kHz):
            # passing that would misreport every audio_end_ms by 1.5x and
            # mis-size every chunk. Passed explicitly rather than left to the
            # default so the trap is visible at the wiring site.
            sample_rate=TTS_SAMPLE_RATE,
        )
        self._generate = generate
        # Continuation merge (approved deviation d9, layer B). 0 = off.
        self._clock = clock
        self._continuation_window_ms = max(0, continuation_window_ms)
        self._continuation_tool_hold_ms = max(0, continuation_tool_hold_ms)
        self._commit_at_ms: int | None = None  # the last SILENCE commit, floor clock
        self._commit_text: str | None = None  # the user entry that commit appended
        self._taking_back = False

        self._outbox: list[dict[str, object]] = []
        self.armed = False
        self._turn_open = False
        self._item_id: str | None = None
        self._reply_text = ""
        self._pending_transcript: str | None = None
        self._pending_item_id: str | None = None
        self._pending_response: int | None = None
        self._pending_synthesis: tuple[int, str] | None = None
        # The STREAMING half (deviation d7): one chunker per reply, and an
        # ORDERED queue of (turn_id, segment_index, text) the route's synth
        # worker drains. A deque, not a single slot like _pending_synthesis,
        # because segments pile up while an earlier one is being synthesized.
        self._chunker = SentenceChunker(eager_first_min_chars=generate.first_clause_min_chars)
        self._stream_turn_id: int | None = None
        self._pending_segments: deque[tuple[int, int, str]] = deque()
        self._tool_call: OutstandingToolCall | None = None
        self._closed_call_ids: deque[str] = deque(maxlen=CLOSED_CALL_MEMORY)
        self._timings_provider: Callable[[], StageTimings | None] | None = None

    def set_timings_provider(self, provider: Callable[[], StageTimings | None] | None) -> None:
        """Let the route attach the per-stage stopwatch it reads on ``response.done``.

        The route owns TIME (see the module docstring), so it is the only
        thing that can bracket a POST or a socket write; this is where the
        result of that measuring reaches the wire. Optional and additive: a
        bridge with no provider — every offline test, and any route that does
        not measure — completes a response exactly as before, with no
        ``timings`` key at all.

        Called at ``ResponseDone``, so the provider sees the whole turn,
        including both generate legs of a tool turn.
        """
        self._timings_provider = provider

    def _response_timings(self) -> StageTimings | None:
        """This response's measured stages, or ``None`` when nobody measured."""
        if self._timings_provider is None:
            return None
        return self._timings_provider() or None

    # -- outbound ---------------------------------------------------------

    def drain(self) -> list[dict[str, object]]:
        """Take every wire payload produced since the last drain, in order."""
        payloads, self._outbox = self._outbox, []
        return payloads

    @property
    def pending_payloads(self) -> int:
        """How many payloads are waiting to be drained (observation/tests)."""
        return len(self._outbox)

    def _push(self, event: Event) -> None:
        self._outbox.append(event_to_dict(event))

    # -- inbound: control events -----------------------------------------

    def on_control_event(self, payload: Mapping[str, object] | None) -> bool:
        """Consume one well-formed non-audio client event. ``True`` if acted on.

        The route calls this for every ``IGNORED`` wire decision, and this is
        the SINGLE dispatch point for the three events that mean something
        here — ``response.create``, ``session.update``, and a
        ``conversation.item.create`` carrying a ``function_call_output``.
        Everything else stays silently ignored, exactly as before.

        A ``session.update`` whose ``session`` patch names none of
        :data:`~lobes.realtime._session.SUPPORTED_SESSION_UPDATE_FIELDS`
        changed nothing, so it reports ``False`` and emits nothing: the echo
        body would be empty, and an ack for a no-op is not a fact worth
        putting on the wire. A patch this server cannot parse is a different
        thing entirely — a named error, and ``True``.
        """
        if is_response_create(payload):
            self.arm()
            return True
        if is_session_update(payload):
            return self._apply_session_update(payload or {})
        if is_function_call_output(payload):
            self.on_function_call_output(payload or {})
            return True
        return False

    def _apply_session_update(self, payload: Mapping[str, object]) -> bool:
        """Apply one ``session.update`` and answer it. ``True`` if it acted.

        A malformed patch is the session's own named
        ``invalid_session_config`` error and the session STAYS OPEN with its
        previous config: a bad update is not a turn boundary and not a reason
        to hang up.
        """
        try:
            event = self.session.update_config(payload)
        except SessionConfigError as exc:
            self._push(exc.to_error_event(self.session.session_id))
            return True
        if not event.session:
            return False
        self._push(event)
        return True

    def arm(self) -> None:
        """Opt this session into conversation. Idempotent.

        If a tool result is waiting, this is what releases it: the floor
        leaves ``tool_wait`` and the SAME turn gets a pending response, so the
        route's ordinary poll issues the follow-up generate with the result
        folded into history. While a tool call is still UNanswered the trigger
        does nothing at all — the machine holds the floor, and a trigger that
        arrives while it does has never started a second response.

        Otherwise, if a transcript is already waiting unanswered — the client
        sent ``response.create`` AFTER its turn was transcribed, the
        OpenAI-shaped per-turn flow — it is answered now and cleared, so a
        second trigger cannot answer the same turn twice.
        """
        self.armed = True
        if self._resume_after_tool_result():
            return
        text, item_id = self._pending_transcript, self._pending_item_id
        self._pending_transcript = self._pending_item_id = None
        if text:
            self._open_turn_for(text, item_id)

    # -- inbound: tool results -------------------------------------------

    @property
    def awaiting_tool_result(self) -> bool:
        """Is a tool call outstanding and still unanswered?

        The route's "do NOT start TTS, and do not expect a synthesis" signal.
        It could infer the same thing from ``take_pending_synthesis()``
        answering ``None``, but that conflates "waiting on the client" with
        "the turn failed", which are different things to log and to time.
        """
        return self._tool_call is not None and not self._tool_call.answered

    @property
    def outstanding_tool_call(self) -> OutstandingToolCall | None:
        """The one call this session is waiting on, answered or not."""
        return self._tool_call

    def on_function_call_output(self, payload: Mapping[str, object]) -> bool:
        """A client tool result. ``True`` when it was accepted into history.

        Accepted ONLY for the single outstanding, unanswered call. An orphan,
        a late result for a call the session already closed, and a duplicate
        each produce a named ``invalid_wire_event`` error carrying its own
        reason token, and leave history BYTE-IDENTICAL — nothing downstream
        would catch the miss otherwise: PROBED 2026-09-18, ``associate``
        repeats an orphan tool result back as fact.

        An accepted result does NOT start the follow-up generate. It waits
        for ``response.create``, which is what keeps every generate in this
        module client-triggered.
        """
        try:
            parsed = parse_function_call_output(payload)
        except FunctionCallOutputError as exc:
            self._push(self.session.fail_wire_event(str(exc)))
            return False

        call = self._tool_call
        if call is None or call.call_id != parsed.call_id:
            reason = (
                TOOL_OUTPUT_CALL_CLOSED
                if parsed.call_id in self._closed_call_ids
                else TOOL_OUTPUT_UNKNOWN_CALL_ID
            )
            self._reject_tool_output(reason, parsed.call_id)
            return False
        if call.answered:
            self._reject_tool_output(TOOL_OUTPUT_DUPLICATE, parsed.call_id)
            return False

        self._append_history_message(tool_result_message(parsed.call_id, parsed.output))
        self._tool_call = replace(call, answered=True)
        return True

    def _reject_tool_output(self, reason: str, call_id: str) -> None:
        detail = _TOOL_OUTPUT_REJECTIONS[reason].format(call_id=call_id)
        self._push(self.session.fail_wire_event(f"{reason}: {detail}"))

    def _resume_after_tool_result(self) -> bool:
        """Release an answered tool wait. ``True`` if a call was outstanding.

        Returning ``True`` for an UNanswered call is what makes a trigger
        arriving mid-tool-wait a no-op rather than a second turn: the caller
        stops here instead of falling through to the pending-transcript path.
        """
        call = self._tool_call
        if call is None:
            return False
        if not call.answered:
            return True
        if self.floor.on_tool_result(turn_id=call.turn_id):
            # Same turn, same response — the route's existing poll picks this
            # up and issues the follow-up generate. No second
            # `response.created`: it is one response, continued.
            self._pending_response = self.floor.turn_id
        self._close_outstanding_tool_call()
        return True

    def on_wire_error(self, exc: WireFormatError) -> None:
        """A malformed client frame — the named error, never a silent drop."""
        self._push(self.session.fail_wire_event(describe_wire_error(exc)))

    def fail_vad(self, message: str) -> None:
        """Silero failed mid-session: the named error, and the floor released.

        The floor is closed rather than merely released — a session whose VAD
        is gone can no longer detect the barge-in that makes speaking safe, so
        it must not keep speaking either.
        """
        self._push(self.session.mark_vad_unavailable(message))
        self.floor.close()

    # -- inbound: turn boundaries ----------------------------------------

    def on_speech_started(self, at_ms: int | None = None) -> bool:
        """A VAD speech onset: a barge-in first (when armed), a boundary always.

        The floor runs BEFORE the boundary event so the session's own state
        lands right: an honoured barge-in emits ``response.interrupted``,
        which returns :attr:`~lobes.realtime._session.Session.state` to
        ``idle``, and only then does ``begin_speech`` move it to ``speech``.
        Emitting the boundary first would leave the session reading ``idle``
        while the user is demonstrably speaking.

        *at_ms* goes to the wire only — never into the floor's clock domain.
        """
        continuation = False
        if self.armed:
            continuation = self._take_back_early_commit()
            if not continuation:
                self.floor.on_speech_started()
        self._commit_at_ms = self._commit_text = None
        self._push(self.session.begin_speech(at_ms=at_ms))
        return continuation

    def tool_call_hold_ms(self) -> int:
        """How long the route must still HOLD a tool call before handing it over.

        A spoken reply can be taken back (stop, forget, redo); a tool call the
        client has received cannot — MEASURED live 2026-09-18: with a 500 ms
        commit the model's tool call went out 180 ms after a mid-sentence
        pause, the speaker resumed at 360 ms, and the turn could no longer be
        merged. So while a continuation is still possible, the route sits on
        a finished tool call until ``continuation_tool_hold_ms`` after the
        commit; an onset in that time takes the turn back and the call is
        never sent. Speech is never held. 0 whenever there is nothing to take
        back: merge off, not a silence commit, or history already past the
        half-turn (a tool turn's follow-up leg).
        """
        if not self._continuation_window_ms or self._commit_at_ms is None:
            return 0
        history = self.session.get_history()
        if not history or history[-1] != {"role": "user", "content": self._commit_text}:
            return 0
        elapsed = self._clock() - self._commit_at_ms
        return max(0, self._continuation_tool_hold_ms - elapsed)

    def _take_back_early_commit(self) -> bool:
        """Whether this onset CONTINUES the turn the machine just committed.

        Layer B of approved deviation d9: with a short confirming silence the
        machine sometimes answers a speaker who was only pausing. An onset
        within ``continuation_window_ms`` of such a commit is the speaker
        carrying on, so the reply is stopped at once — inside the barge-in
        guard window too — and the half-turn is REMOVED from history: no user
        entry, no heard-prefix assistant entry. The route then re-transcribes
        both halves as one turn, which becomes an ordinary turn.

        Refused (so the onset is an ordinary barge-in, or nothing) when the
        feature is off, the commit was not a silence commit, the window has
        passed, the reply already finished, a tool call already went out, or
        history moved on past the half-turn's user entry.
        """
        if not self._continuation_window_ms or self._commit_at_ms is None:
            return False
        if self._commit_text is None:
            return False
        if self._clock() - self._commit_at_ms > self._continuation_window_ms:
            return False
        if not self.floor.machine_holds_floor or self.awaiting_tool_result:
            return False
        history = self.session.get_history()
        if not history or history[-1] != {"role": "user", "content": self._commit_text}:
            return False
        self._taking_back = True
        try:
            if not self.floor.on_continuation_onset():
                return False
        finally:
            self._taking_back = False
        self.session.pop_history_if_last("user", self._commit_text)
        return True

    def on_speech_stopped(self, at_ms: int | None = None, reason: str | None = None) -> None:
        """A committed turn: an interruption if the machine held the floor,
        then this turn opens.

        A committed turn that survived the VAD's own silence confirmation is
        far stronger evidence of a real interruption than a bare onset, so the
        floor consumes it as one (past the guard window) and immediately
        re-takes the floor for the new turn. When the guard window swallows it
        the floor stays where it was and this turn simply gets no reply — but
        it is still transcribed and still reported, exactly as an ears-only
        session would (see :meth:`on_transcript`).
        """
        self._turn_open = self.armed and self.floor.on_turn_committed()
        # Only a SILENCE commit can have been premature (layer B); a max_turn
        # or teardown commit is never taken back.
        self._commit_at_ms = self._clock() if self._turn_open and reason == "silence" else None
        self._commit_text = None
        self._push(self.session.end_speech(at_ms=at_ms, reason=reason))

    def on_transcript(self, text: str) -> None:
        """The committed turn's transcript arrived.

        The transcription event is emitted identically whether or not this
        session is armed — the ears-only sequence is never altered by the
        conversation surface, only added to. When the floor holds this turn,
        the transcript also advances it: a blank transcript releases the floor
        without a response (silence is not something to answer, and not an
        error either), and anything else starts one.
        """
        event = self.session.complete_transcription(text)
        self._item_id = event.item_id
        self._push(event)
        if not self._turn_open:
            self._remember_pending(text, event.item_id)
            return
        self._turn_open = False
        turn_id = self.floor.turn_id
        self.floor.on_transcript(text, turn_id=turn_id)
        if self.floor.state is FloorState.RESPONDING:
            self.session.append_history("user", text)
            self._commit_text = text

    def on_transcription_failed(self, message: str) -> None:
        """The committed turn's Parakeet forward failed — never a silent drop.

        Emitted exactly once. When the floor holds the turn, the floor is what
        releases it and the resulting :class:`ResponseFailed` is what produces
        the event (through the very same
        :meth:`~lobes.realtime._session.Session.fail_transcription` call, so
        the payload is identical to the ears-only path). Only if the floor
        refuses the failure — a stage it has already left — does this method
        emit directly.
        """
        if self._turn_open:
            self._turn_open = False
            if self.floor.fail_stage(
                FailureReason.TRANSCRIBE_FAILED, message, turn_id=self.floor.turn_id
            ):
                return
        self._push(self.session.fail_transcription(message))

    def _remember_pending(self, text: str, item_id: str | None) -> None:
        if text.strip():
            self._pending_transcript = text
            self._pending_item_id = item_id

    def _open_turn_for(self, text: str, item_id: str | None) -> None:
        """Answer an already-transcribed turn (the ``response.create``-after-
        transcript flow): open a floor turn and hand it the transcript at once."""
        if not self.floor.on_turn_committed():
            return
        self._item_id = item_id
        self.floor.on_transcript(text, turn_id=self.floor.turn_id)
        if self.floor.state is FloorState.RESPONDING:
            self.session.append_history("user", text)

    # -- the response, driven by the route -------------------------------

    def take_pending_response(self) -> int | None:
        """The turn id of a response the route must now run, or ``None``.

        Set when the floor emits ``ResponseStarted``; taken exactly once, so a
        route that polls after every drive point never launches two tasks for
        one turn.
        """
        turn_id, self._pending_response = self._pending_response, None
        return turn_id

    def build_generate_request(self, turn_id: int) -> TurnRequest | None:
        """The ``/v1/chat/completions`` call for *turn_id* — url, headers, body.

        ``None`` when the turn is stale (interrupted, failed, or already
        overtaken), which is the route's signal to do nothing at all rather
        than issue a request whose answer nobody can use. The system prompt is
        the SESSION's — the connect-config override if the client set one, the
        operator's ``DEFAULT_SYSTEM_PROMPT`` otherwise.
        """
        if turn_id != self.floor.turn_id or self.floor.state is not FloorState.RESPONDING:
            return None
        return build_turn_request(
            self.session.get_history(),
            base_url=self._generate.base_url,
            api_key=self._generate.api_key,
            system_prompt=self.session.system_prompt,
            model=self._generate.model,
            max_tokens=self._generate.max_tokens,
            temperature=self._generate.temperature,
            # The session's own declaration, passed straight through: `None`
            # (never declared) and `()` (declared empty) both leave the
            # payload byte-identical to a call that never mentions tools.
            tools=self.session.config.tools,
            tool_choice=self.session.config.tool_choice,
            # The operator's GENERATE_STREAM, threaded through to the one
            # payload key that changes the transport (deviation d7).
            stream=self._generate.stream,
        )

    def build_speculative_request(self, transcript: str) -> TurnRequest | None:
        """The generate call this turn WOULD make if it ended now with *transcript*.

        Hidden speculation (approved deviation d9, :mod:`._speculation`): pure
        and traceless — no history entry, no event, no floor transition. The
        request is assembled exactly as :meth:`build_generate_request` will
        assemble the real one once the commit has appended the transcript, so
        the two are byte-identical when nothing changed in between; that
        equality (:func:`~._speculation.can_adopt`) is the whole adoption rule.

        ``None`` — do not speculate — for an unarmed session (ears-only must
        stay ears-only), a non-streaming one (there is no stream to adopt), an
        empty transcript, or any floor state other than LISTENING: a pause
        while a reply is running or a tool result is outstanding belongs to a
        barge-in, not to a fresh turn.
        """
        if not self.armed or not self._generate.stream or not transcript.strip():
            return None
        if self.floor.state is not FloorState.LISTENING or self.awaiting_tool_result:
            return None
        history = self.session.get_history() + [{"role": "user", "content": transcript}]
        return build_turn_request(
            history,
            base_url=self._generate.base_url,
            api_key=self._generate.api_key,
            system_prompt=self.session.system_prompt,
            model=self._generate.model,
            max_tokens=self._generate.max_tokens,
            temperature=self._generate.temperature,
            tools=self.session.config.tools,
            tool_choice=self.session.config.tool_choice,
            stream=True,
        )

    def on_generate_response(
        self,
        status_code: int,
        body: bytes,
        *,
        turn_id: int,
        headers: Mapping[str, str] | None = None,
    ) -> bool:
        """Hand the raw generate response back. ``True`` if the turn advanced.

        Every failure shape :mod:`._turn` names — a ``role_infeasible`` 404
        (with its ``hosted_by`` hint preserved), any other non-2xx, a
        malformed body — becomes a named error event, never a placeholder
        reply and never a second attempt against a different lane. *headers*
        is optional and used for exactly one thing: a retryable status's
        ``Retry-After``, which has nowhere else to travel (see
        :func:`describe_generate_http_failure`).

        A reply carrying ``tool_calls`` is NOT text: it becomes one
        ``response.function_call_arguments.done``, the floor waits, and
        nothing is synthesized.
        """
        try:
            result = parse_turn_response(status_code, body)
        except RoleInfeasibleError as exc:
            message = describe_generate_http_failure(
                describe_role_infeasible(exc),
                status_code=status_code,
                retry_after=retry_after_from_headers(headers),
            )
            return self.fail_generate(message, turn_id=turn_id)
        except TurnRequestError as exc:
            message = describe_generate_http_failure(
                str(exc),
                status_code=status_code,
                retry_after=retry_after_from_headers(headers),
                hosted_by=getattr(exc, "hosted_by", None),
            )
            return self.fail_generate(message, turn_id=turn_id)
        if isinstance(result, ToolCallResult):
            return self._request_tool_call(result, turn_id=turn_id)
        return self.floor.on_reply_text(result, turn_id=turn_id)

    def _request_tool_call(self, result: ToolCallResult, *, turn_id: int) -> bool:
        """Record the model's tool call and hand it to the client.

        The floor runs FIRST: a stale turn refuses the call, and then nothing
        at all is recorded — history must not gain an assistant ``tool_calls``
        entry for a turn whose result can never arrive.
        """
        if not self.floor.on_tool_call(
            call_id=result.call_id,
            name=result.name,
            arguments=result.arguments,
            turn_id=turn_id,
        ):
            return False
        self._append_history_message(assistant_tool_call_message(result))
        self._tool_call = OutstandingToolCall(
            call_id=result.call_id, name=result.name, turn_id=self.floor.turn_id
        )
        if result.tool_call_count > 1:
            # One outstanding call at a time is the bookkeeping contract, so
            # the rest are dropped — loudly, because a client that sees one
            # call answered out of three has no way to know that happened.
            self.session.log.info(
                "tool call surfaced 1 of %d requested calls", result.tool_call_count
            )
        return True

    def fail_generate(self, message: str, *, turn_id: int, timed_out: bool = False) -> bool:
        """The generate call failed by name (unreachable, non-2xx, timed out)."""
        reason = FailureReason.GENERATE_TIMEOUT if timed_out else FailureReason.GENERATE_FAILED
        return self.floor.fail_stage(reason, message, turn_id=turn_id)

    def take_pending_synthesis(self) -> tuple[int, str] | None:
        """``(turn_id, reply_text)`` the route must now synthesize, or ``None``.

        The NON-streaming surface, unchanged: one whole reply, taken once.
        The streaming route drains :meth:`take_pending_segment` instead.
        """
        pending, self._pending_synthesis = self._pending_synthesis, None
        return pending

    # -- the streaming surface (approved deviation d7) --------------------

    @property
    def streaming_enabled(self) -> bool:
        """Did the operator ask for streamed generation (``GENERATE_STREAM``)?

        The route reads this to pick which of the two surfaces to drive. The
        bridge itself supports both regardless — this is a wiring fact, not a
        state machine mode.
        """
        return self._generate.stream

    def response_in_progress(self, turn_id: int) -> bool:
        """Is *turn_id*'s reply still being produced or spoken?

        The delivery pump's loop condition. It lives here, not in the route,
        for the usual reason: "which floor states mean audio may still be
        coming" is a decision, and a route that guessed it wrong would either
        spin forever or stop pumping mid-reply. A tool wait answers ``False``
        — nothing will be delivered until the client comes back.
        """
        return turn_id == self.floor.turn_id and self.floor.state in (
            FloorState.RESPONDING,
            FloorState.SPEAKING,
        )

    def begin_generate_stream(self, turn_id: int) -> bool:
        """Open a streamed generate for *turn_id*. ``True`` if it is live.

        Resets the sentence chunker (the eager-first-sentence allowance is
        per reply) and drops any segment left queued by an abandoned turn, so
        a stale sentence can never be synthesized against a new one.
        """
        self._chunker.reset()
        self._pending_segments.clear()
        self._stream_turn_id = turn_id
        return turn_id == self.floor.turn_id and self.floor.state is FloorState.RESPONDING

    def on_generate_delta(self, text_delta: str, *, turn_id: int) -> bool:
        """One streamed text delta. ``True`` if it was consumed.

        Feeds the chunker and hands every COMPLETED sentence to the floor as
        a non-final segment, queuing it for synthesis. Nothing is emitted on
        the wire here: ``response.text.done`` still goes out exactly once, at
        stream end, with the whole reply (see
        :meth:`on_generate_stream_end`).
        """
        if turn_id != self._stream_turn_id or turn_id != self.floor.turn_id:
            return False
        for sentence in self._chunker.feed(text_delta):
            self.floor.on_reply_segment(sentence, final=False, turn_id=turn_id)
        return True

    def on_generate_stream_end(self, result: str | ToolCallResult, *, turn_id: int) -> bool:
        """The stream finished. ``True`` if the turn advanced.

        A TEXT result flushes the chunker's remainder as the FINAL segment,
        emits the single ``response.text.done`` with the full reply, and
        records that full text as what history will carry once the reply is
        delivered.

        A TOOL CALL discards every segment already queued from a text prefix
        — the model abandoned that text, so it must not be spoken — and then
        takes exactly the non-streaming tool path.
        """
        if turn_id != self._stream_turn_id or turn_id != self.floor.turn_id:
            return False
        if isinstance(result, ToolCallResult):
            return self._abandon_stream_for_tool_call(result, turn_id=turn_id)
        tail = self._chunker.flush()
        for sentence in tail[:-1]:
            self.floor.on_reply_segment(sentence, final=False, turn_id=turn_id)
        self._reply_text = result
        if result.strip():
            self._push(self.session.complete_response_text(result))
        return self.floor.on_reply_segment(tail[-1] if tail else "", final=True, turn_id=turn_id)

    def _abandon_stream_for_tool_call(self, result: ToolCallResult, *, turn_id: int) -> bool:
        """Drop a spoken-text prefix and surface the tool call instead."""
        self._pending_segments.clear()
        self._reply_text = ""
        self.floor.discard_reply_segments(turn_id=turn_id)
        return self._request_tool_call(result, turn_id=turn_id)

    def take_pending_segment(self) -> tuple[int, int, str] | None:
        """``(turn_id, segment_index, text)`` to synthesize next, or ``None``.

        In release order, one at a time, taken exactly once — the route's
        synth worker calls this in a loop and answers each with
        :meth:`on_tts_audio`. Order matters twice over: TTS runs serially
        (``TTS_VOICE_CONCURRENCY`` is 1) and the floor delivers segments in
        index order regardless.
        """
        if not self._pending_segments:
            return None
        return self._pending_segments.popleft()

    @property
    def pending_segments(self) -> int:
        """How many segments are waiting to be synthesized (observation/tests)."""
        return len(self._pending_segments)

    def on_tts_audio(self, pcm: bytes, *, turn_id: int, segment_index: int = 0) -> bool:
        """The full-read synthesis returned; delivery can begin.

        Empty audio is a named TTS failure, not a silently completed reply —
        :func:`lobes.realtime.tts_client.synthesize` returns ``b""`` on a soft
        failure, and rendering that as "the machine spoke" would be a lie.

        *segment_index* defaults to ``0`` — the only segment a non-streaming
        reply has — so every pre-streaming caller is unchanged.
        """
        return self.floor.on_audio_ready(pcm, turn_id=turn_id, segment_index=segment_index)

    def fail_tts(self, message: str, *, turn_id: int, timed_out: bool = False) -> bool:
        """The TTS call failed by name."""
        reason = FailureReason.TTS_TIMEOUT if timed_out else FailureReason.TTS_FAILED
        return self.floor.fail_stage(reason, message, turn_id=turn_id)

    def deliver_next(self, *, turn_id: int) -> bool:
        """Queue the next audio chunk of *turn_id*'s reply. ``True`` if one went.

        The route PUMPS this — ``while bridge.deliver_next(turn_id=n): await
        flush()`` — so the receive loop runs between chunks and a barge-in
        can land mid-reply. The *turn_id* guard is what keeps a response task
        that is still unwinding from an interruption out of the NEXT turn's
        audio: without it, a stale pump would happily deliver a reply the user
        never asked for.
        """
        if turn_id != self.floor.turn_id:
            return False
        return self.floor.deliver_next()

    def tick(self) -> bool:
        """Expire the armed stage's deadline if it is due. ``True`` if it was.

        Driven by the route's watchdog. Deadlines expire ONLY here, so a route
        that stops calling this has no timeouts at all — the "floor never
        wedges" property evaporates silently.
        """
        return self.floor.tick()

    def close(self) -> None:
        """Tear the floor down from any state. Idempotent; emits nothing.

        Session lifecycle events belong to :meth:`Session.teardown` — which is
        where a close ``reason`` is actually rendered, onto ``session.closed``.
        This layer used to accept one and discard it; a client that is already
        gone cannot act on an interruption event either way.
        """
        self.floor.close()

    # -- floor callbacks --------------------------------------------------

    def _on_floor_event(self, event: object) -> None:
        """Translate one floor-local fact into the session's schema.

        The floor speaks its own event vocabulary and knows nothing about the
        wire; this is the only place the two meet.
        """
        if isinstance(event, ResponseStarted):
            self._push(self.session.begin_response(item_id=self._item_id))
            self._pending_response = event.turn_id
        elif isinstance(event, ReplyText):
            self._reply_text = event.text
            self._push(self.session.complete_response_text(event.text))
            self._pending_synthesis = (event.turn_id, event.text)
        elif isinstance(event, ReplySegment):
            # A PIECE of a streamed reply: queue it for synthesis and say
            # nothing on the wire. `response.text.done` is emitted once, with
            # the whole reply, by on_generate_stream_end — the wire contract
            # does not change because the transport did.
            self._pending_segments.append((event.turn_id, event.index, event.text))
            # Length only, never the text — the same discipline _turn.py
            # documents for a reply. This line is how an operator SEES
            # streaming working in the bridge log: one per sentence, arriving
            # while the generate call is still open.
            self.session.log.info(
                "reply segment %d released (%d chars, final=%s)",
                event.index,
                len(event.text),
                event.final,
            )
        elif isinstance(event, ToolCallRequested):
            self._push(
                self.session.emit_function_call_arguments_done(
                    call_id=event.call_id,
                    name=event.name,
                    arguments=event.arguments,
                    item_id=self._item_id,
                )
            )
        elif isinstance(event, ResponseDone):
            self.session.append_history("assistant", self._reply_text)
            self._clear_turn()
            self._push(self.session.complete_response(self._response_timings()))
        elif isinstance(event, ResponseInterrupted):
            self._record_interrupted_reply(event)
            self._clear_turn()
            self._push(self.session.interrupt_response())
        elif isinstance(event, ResponseFailed):
            self._clear_turn()
            self._push(self._failure_event(event))

    def _record_interrupted_reply(self, event: ResponseInterrupted) -> None:
        """Write only what the listener plausibly HEARD into history.

        Recording the whole reply as if it had been spoken is a worse lie than
        recording a slightly-off prefix: the next turn's context would claim
        the machine said things the user cut off before hearing. The prefix is
        an estimate, not an alignment — Chatterbox returns audio with no word
        timings (see
        :func:`lobes.realtime._floor.estimate_spoken_prefix`).

        Nothing delivered means nothing heard, and history records nothing.
        The guard is load-bearing rather than an optimisation:
        ``estimate_spoken_prefix`` returns the FULL text when
        ``audio_total_ms`` is zero (its "I have no measurement, assume it all
        played" branch), which is exactly the wrong answer for a cut that
        landed during synthesis — before a single byte existed, let alone
        went out.
        """
        if self._taking_back:
            return  # a continuation: the whole exchange is being redone
        if event.delivered_bytes <= 0:
            return
        # The floor computes the prefix ACROSS segments (a proportional
        # estimate is only meaningful within one); `heard_text` is that
        # answer, and for a single-segment reply it is byte-for-byte what
        # estimate_spoken_prefix returns here. The fallback keeps a caller
        # that constructs the event by hand working.
        heard = event.heard_text or estimate_spoken_prefix(
            event.reply_text, event.audio_end_ms, event.audio_total_ms
        )
        if heard:
            self.session.append_history("assistant", heard)

    def _failure_event(self, event: ResponseFailed) -> ErrorEvent:
        message = describe_failure(event)
        if event.reason is FailureReason.TRANSCRIBE_FAILED:
            # The SAME emitter the ears-only path uses, so an STT forward
            # failure looks identical to a client whether or not the session
            # went on to answer the turn (it also keeps the item_id on the
            # event, which fail_response has no way to set).
            return self.session.fail_transcription(message)
        return self.session.fail_response(FAILURE_ERROR_CODES[event.reason], message)

    def _clear_turn(self) -> None:
        self._close_outstanding_tool_call()
        self._reply_text = ""
        self._pending_response = None
        self._pending_synthesis = None
        self._pending_segments.clear()
        self._stream_turn_id = None

    def _close_outstanding_tool_call(self) -> None:
        """Close the outstanding call, cancelling it in history if unanswered.

        An assistant ``tool_calls`` entry with no matching ``tool`` entry is a
        shape a chat-completions backend rejects outright, so a turn that ends
        mid-wait — barge-in, expired tool wait, any other failure — writes a
        synthetic :data:`TOOL_CALL_CANCELLED_OUTPUT` result rather than
        leaving the pair dangling. An ALREADY-answered call needs none: the
        real result is in history, and inventing a cancellation over it would
        be a lie.

        The id is remembered either way, so a result that arrives afterwards
        is named ``call_closed`` — "you are too late", a different fact from
        "I never asked for that".
        """
        call = self._tool_call
        if call is None:
            return
        self._tool_call = None
        self._closed_call_ids.append(call.call_id)
        if not call.answered:
            self._append_history_message(
                tool_result_message(call.call_id, TOOL_CALL_CANCELLED_OUTPUT)
            )

    def _append_history_message(self, message: dict) -> None:
        """Append one STRUCTURED chat message to the session's history.

        :meth:`~lobes.realtime._session.Session.append_history` takes
        ``(role, content)`` and can express neither an assistant turn carrying
        ``tool_calls`` nor a ``tool`` turn carrying ``tool_call_id``. The
        history list itself is the session's, and it is the list
        ``get_history()`` hands the turn builder, so this reaches it directly
        rather than keeping a second, parallel history here — two lists that
        must interleave correctly is exactly the bug this module exists to not
        have. A ``Session.append_message`` belongs in ``_session.py``; that
        module is owned by the wire-contract task and frozen for this one.
        """
        self.session._history.append(message)

    def _on_audio_chunk(self, chunk: bytes) -> None:
        """One PCM16 chunk of the reply -> one ``response.audio.delta``.

        No resample: Chatterbox emits 24 kHz PCM16 and the client wire format
        IS 24 kHz PCM16 (``protocol.TTS_SAMPLE_RATE == CLIENT_SAMPLE_RATE``),
        so the only transform on the whole audio-out path is this base64
        encode.
        """
        self._push(self.session.emit_audio_delta(encode_audio_chunk(chunk)))


__all__ = [
    "RESPONSE_CREATE_EVENT_TYPE",
    "DEFAULT_VOICE_MODEL",
    "WATCHDOG_INTERVAL_MS",
    "FAILURE_ERROR_CODES",
    "WIRE_ERROR_CODES",
    "TOOL_OUTPUT_UNKNOWN_CALL_ID",
    "TOOL_OUTPUT_CALL_CLOSED",
    "TOOL_OUTPUT_DUPLICATE",
    "TOOL_CALL_CANCELLED_OUTPUT",
    "CLOSED_CALL_MEMORY",
    "RETRY_AFTER_UNSPECIFIED",
    "RETRYABLE_GENERATE_STATUSES",
    "describe_failure",
    "describe_wire_error",
    "describe_role_infeasible",
    "describe_generate_http_failure",
    "retry_after_from_headers",
    "resolve_voice_model",
    "is_response_create",
    "OutstandingToolCall",
    "ConversationBridge",
]
