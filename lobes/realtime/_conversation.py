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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

from ._floor import (
    DEFAULT_BARGE_IN_WINDOW_MS,
    DEFAULT_GENERATE_TIMEOUT_MS,
    DEFAULT_TRANSCRIBE_TIMEOUT_MS,
    DEFAULT_TTS_TIMEOUT_MS,
    FailureReason,
    Floor,
    FloorState,
    ReplyText,
    ResponseDone,
    ResponseFailed,
    ResponseInterrupted,
    ResponseStarted,
    estimate_spoken_prefix,
)
from ._session import ErrorCode, ErrorEvent, Event, Session, event_to_dict
from ._turn import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    RoleInfeasibleError,
    TurnRequest,
    TurnRequestError,
    build_turn_request,
    parse_turn_response,
)
from ._wire import DEFAULT_DELTA_CHUNK_BYTES, WireErrorCode, WireFormatError, encode_audio_chunk
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
    FailureReason.TTS_TIMEOUT: ErrorCode.RESPONSE_TIMEOUT,
}

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
        tts_timeout_ms: int = DEFAULT_TTS_TIMEOUT_MS,
        chunk_bytes: int = DEFAULT_DELTA_CHUNK_BYTES,
        clock: Callable[[], int] = timestamp_ms,
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

        self._outbox: list[dict[str, object]] = []
        self.armed = False
        self._turn_open = False
        self._item_id: str | None = None
        self._reply_text = ""
        self._pending_transcript: str | None = None
        self._pending_item_id: str | None = None
        self._pending_response: int | None = None
        self._pending_synthesis: tuple[int, str] | None = None

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

        The route calls this for every ``IGNORED`` wire decision; only
        ``response.create`` is acted on, and everything else stays ignored.
        """
        if not is_response_create(payload):
            return False
        self.arm()
        return True

    def arm(self) -> None:
        """Opt this session into conversation. Idempotent.

        If a transcript is already waiting unanswered — the client sent
        ``response.create`` AFTER its turn was transcribed, the OpenAI-shaped
        per-turn flow — it is answered now and cleared, so a second trigger
        cannot answer the same turn twice.
        """
        self.armed = True
        text, item_id = self._pending_transcript, self._pending_item_id
        self._pending_transcript = self._pending_item_id = None
        if text:
            self._open_turn_for(text, item_id)

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

    def on_speech_started(self, at_ms: int | None = None) -> None:
        """A VAD speech onset: a barge-in first (when armed), a boundary always.

        The floor runs BEFORE the boundary event so the session's own state
        lands right: an honoured barge-in emits ``response.interrupted``,
        which returns :attr:`~lobes.realtime._session.Session.state` to
        ``idle``, and only then does ``begin_speech`` move it to ``speech``.
        Emitting the boundary first would leave the session reading ``idle``
        while the user is demonstrably speaking.

        *at_ms* goes to the wire only — never into the floor's clock domain.
        """
        if self.armed:
            self.floor.on_speech_started()
        self._push(self.session.begin_speech(at_ms=at_ms))

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
        )

    def on_generate_response(self, status_code: int, body: bytes, *, turn_id: int) -> bool:
        """Hand the raw generate response back. ``True`` if the turn advanced.

        Every failure shape :mod:`._turn` names — a ``role_infeasible`` 404
        (with its ``hosted_by`` hint preserved), any other non-2xx, a
        malformed body — becomes a named error event, never a placeholder
        reply and never a second attempt against a different lane.
        """
        try:
            text = parse_turn_response(status_code, body)
        except RoleInfeasibleError as exc:
            return self.fail_generate(describe_role_infeasible(exc), turn_id=turn_id)
        except TurnRequestError as exc:
            return self.fail_generate(str(exc), turn_id=turn_id)
        return self.floor.on_reply_text(text, turn_id=turn_id)

    def fail_generate(self, message: str, *, turn_id: int, timed_out: bool = False) -> bool:
        """The generate call failed by name (unreachable, non-2xx, timed out)."""
        reason = FailureReason.GENERATE_TIMEOUT if timed_out else FailureReason.GENERATE_FAILED
        return self.floor.fail_stage(reason, message, turn_id=turn_id)

    def take_pending_synthesis(self) -> tuple[int, str] | None:
        """``(turn_id, reply_text)`` the route must now synthesize, or ``None``."""
        pending, self._pending_synthesis = self._pending_synthesis, None
        return pending

    def on_tts_audio(self, pcm: bytes, *, turn_id: int) -> bool:
        """The full-read synthesis returned; delivery can begin.

        Empty audio is a named TTS failure, not a silently completed reply —
        :func:`lobes.realtime.tts_client.synthesize` returns ``b""`` on a soft
        failure, and rendering that as "the machine spoke" would be a lie.
        """
        return self.floor.on_audio_ready(pcm, turn_id=turn_id)

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
        elif isinstance(event, ResponseDone):
            self.session.append_history("assistant", self._reply_text)
            self._clear_turn()
            self._push(self.session.complete_response())
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
        if event.delivered_bytes <= 0:
            return
        heard = estimate_spoken_prefix(event.reply_text, event.audio_end_ms, event.audio_total_ms)
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
        self._reply_text = ""
        self._pending_response = None
        self._pending_synthesis = None

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
    "describe_failure",
    "describe_wire_error",
    "describe_role_infeasible",
    "resolve_voice_model",
    "is_response_create",
    "ConversationBridge",
]
