"""The realtime session engine — event schema, config parsing, teardown.

Stdlib-only (``logging``, ``re``, ``dataclasses``, ``enum``) so this module
imports without the ``[realtime]`` extra, mirroring
:mod:`lobes.realtime.audio_facade` and :mod:`lobes.realtime._settings`. It
owns the session's event schema and per-session bookkeeping; a sibling module,
``_segmenter.py``, owns the VAD state machine, and a later route layer
(``app.py``) wires both to a real WebSocket. This module never imports
``_segmenter`` and never implements VAD segmentation itself.

**Sessions are ephemeral by contract.** There is no resume path and no
on-disk state: teardown releases everything this module allocated, and a
reconnecting client gets a brand-new :class:`Session`. Nothing here opens a
file, a socket, or makes a network call — nothing here is I/O at all.

Event schema (OpenAI-Realtime-flavoured naming, kept deliberately small —
the #149 baseline scope was audio-in, boundaries, and transcription; see
docs/specs/2026-07-21-realtime-ws-server-vad-149.md's Non-goals):

- ``session.created`` / ``session.closed`` — session lifecycle.
- ``input_audio_buffer.speech_started`` / ``...speech_stopped`` — VAD
  boundaries (emitted by the caller once VAD/segmenter logic — owned
  elsewhere — decides a boundary occurred; this module only defines the
  event shape and the session-state transition).
- ``conversation.item.input_audio_transcription.completed`` — a committed
  turn's transcript.
- ``error`` — every failure mode, discriminated by a documented
  :class:`ErrorCode` (never a bare exception string):
  ``invalid_session_config`` (bad session config), ``vad_unavailable``
  (Silero failed to load/run — distinct from ordinary silence, which emits
  no event at all), ``stt_forward_failed`` (a committed turn's Parakeet
  forward failed), ``invalid_wire_event`` (a malformed client frame).

Since issue #151, the engine also owns an OPT-IN, audio-path-only response
lifecycle for server-side voice-to-voice conversation — a committed turn may
trigger a generate call and a synthesized reply streamed back over the SAME
session. A session that never triggers this (the default, ears-only mode)
is byte-identical to the #149 baseline: exactly the event sequence above,
nothing more. Full OpenAI Realtime parity — conversation items, tool calls,
ephemeral tokens — stays a named follow-up (see
docs/specs/2026-07-21-realtime-voice-to-voice-astro-test-site-151.md's
Open/follow-up); this module adopts only the audio-path event shapes:

- ``response.created`` — a committed turn triggered a server-side reply; the
  floor moves from the caller to the assistant (:attr:`SessionState.RESPONDING`).
- ``response.text.done`` — the generate call's full reply text arrived; the
  floor moves to speaking (:attr:`SessionState.SPEAKING`) for TTS delivery.
- ``response.audio.delta`` — one already wire-encoded (base64, owned by
  ``_wire.py``, not this module) chunk of the synthesized reply.
- ``response.done`` — the reply was delivered in full; the floor returns to
  the caller (:attr:`SessionState.IDLE`).
- ``response.interrupted`` — speech arrived while the floor was
  responding/speaking (barge-in); the floor returns to the caller
  immediately and any undelivered audio is truncated (the event's
  ``truncated`` marker) — timing/deadlines/cancellation plumbing is a
  separate floor/turn state machine's job, not this module's.

Four new :class:`ErrorCode` members cover the new failure modes:
``generate_failed`` (the brain call failed), ``tts_failed`` (the mouth call
failed), ``response_timeout`` (a response stage exceeded its deadline — the
floor always returns to the caller with a named error, never stuck
responding/speaking), and ``invalid_wire_event`` (a malformed client frame —
the ONE code the three ``_wire.WireErrorCode`` values all map onto, so this
enum stays the single enumerable error vocabulary on this wire).

The boundary events also carry the segmenter's own audio-stream timing since
issue #151 t6: ``at_ms`` on both, plus ``reason`` on ``speech_stopped``.
Those were computed by ``_segmenter.py`` and dropped at the route before they
reached the wire, which left a client unable to tell a ``max_turn``
force-commit from a silence-confirmed stop, and left VAD-knob effects
observable only in fixture replay.

``SessionState`` gains ``responding``/``speaking`` so the floor holder —
who currently owns the turn — is explicit in the schema itself, not implied
by event ordering.

Per-session, in-memory conversation history and a system prompt back the
opt-in conversation surface (the terminal ``scripts/realtime-voice-loop.py``
proved a coherent reply needs both, client-side; #151 moves both
server-side). Both are ephemeral by the same contract as everything else
here: history lives only on the :class:`Session` object, nothing is ever
written to disk, and :meth:`Session.teardown` drops it.

Since the hebrew-realtime work (t4), the schema also pins the CONTRACT for
client-declared tools, a per-session language, and per-stage timings — shapes
and parsing only, no behaviour: the bridge, floor and turn modules wire them.
Every name below is spelled exactly as OpenAI Realtime spells it, because the
point of the tool wire is that a stock OpenAI Realtime client's tool loop
works unmodified:

- ``session.update`` (client) -> :func:`apply_session_update` /
  :meth:`Session.update_config`, answered by ``session.updated``
  (:class:`SessionUpdatedEvent`) which echoes ONLY the fields that took
  effect (:data:`SUPPORTED_SESSION_UPDATE_FIELDS`) — a field this server does
  not act on is visibly absent from the echo, never silently swallowed.
- ``response.function_call_arguments.done`` (server) ->
  :class:`ResponseFunctionCallArgumentsDoneEvent`: the model's tool call
  handed to the client, whose ``arguments`` is a JSON **string**, as OpenAI's
  is.
- ``conversation.item.create`` with an item of type ``function_call_output``
  (client) -> :func:`parse_function_call_output`: the client's tool RESULT,
  a ``call_id``/``output`` pair.
- ``response.done`` gains an optional, additive
  :class:`StageTimings` mapping (``stt``/``generate``/``tool_wait``/
  ``phonikud``/``tts``/``first_delta``, milliseconds) so the acceptance
  latency table comes off the wire rather than out of a log scrape. An
  unmeasured stage is ABSENT, never zeroed — a zero would read as "instant".

Realtime tools are FLAT (``{"type": "function", "name", "description",
"parameters"}``) where chat-completions' are NESTED (``{"type": "function",
"function": {...}}``); :func:`realtime_tools_to_chat_completions` and
:func:`realtime_tool_choice_to_chat_completions` are the pure translation
between the two, so the generate-request builder never re-derives it.

Two deliberate NON-additions: the error vocabulary stays the seven
:class:`ErrorCode` members it already had — a rejected ``session.update`` is
an ``invalid_session_config`` (the code already covers "a session config
payload was rejected", whether it arrived at connect time or in an update)
and a malformed ``function_call_output`` frame is an ``invalid_wire_event``
with its reason in the message text, exactly as the three
``_wire.WireErrorCode`` reasons are. One enumerable list of codes on this
wire is a contract, not an accident.

Defaults keep the wire byte-identical: an English, tool-free session's
``session.created`` config dict and its ``response.done`` carry not one new
key (:func:`event_to_dict` emits the additive keys only when they are set),
so the ears-only contract reachy-mini-cli depends on is unchanged.

Logging: every helper here logs through :func:`get_session_logger`, which
stamps the session id into the message text itself (not just ``extra``) so
grepping logs for one session id reconstructs its whole lifecycle even under
a bare-bones formatter. :func:`redact_for_log` is the one approved way to log
a client-supplied mapping (e.g. a raw session-config payload) — it masks any
credential-shaped key so an API key or token can never reach a log line.
Transcribed text and reply text are deliberately never logged verbatim
(only their length) — conversation history content follows the same rule.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import NamedTuple

from ._wire import (
    CONVERSATION_ITEM_CREATE_EVENT_TYPE,
    FUNCTION_CALL_OUTPUT_ITEM_TYPE,
    SESSION_UPDATE_EVENT_TYPE,
)
from .protocol import (
    CLIENT_SAMPLE_RATE,
    STT_SAMPLE_RATE,
    AECMode,
    AudioFormat,
    TurnDetectionType,
    gen_event_id,
    gen_item_id,
    gen_response_id,
    gen_session_id,
    timestamp_ms,
)

log = logging.getLogger(__name__)

_SUPPORTED_SAMPLE_RATES = (CLIENT_SAMPLE_RATE, STT_SAMPLE_RATE)  # 24000 (default), 16000

# The STT language every session speaks unless a deployment default or a
# per-session value says otherwise. "en" is what app.py has always sent, so a
# box that never asks for another language keeps sending it byte-for-byte.
DEFAULT_LANGUAGE = "en"

# A short language code: a 2-3 letter primary subtag, optionally with
# region/script subtags ("en", "he", "pt-BR"). Deliberately a SHAPE check, not
# a registry lookup — this module has no business shipping an ISO table, and
# whether a given code is one the STT sidecar loaded is that sidecar's answer.
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")

# The tool "type" OpenAI Realtime declares — the only one either side speaks.
_FUNCTION_TOOL_TYPE = "function"

# The string forms of ``tool_choice`` that pass through untranslated; the
# object form (``{"type": "function", "name": ...}``) is translated instead.
_TOOL_CHOICE_KEYWORDS = ("auto", "none", "required")

# The ONLY ``session.update`` fields this server acts on. Anything else in the
# payload is ignored AND visibly absent from the ``session.updated`` echo —
# the silent-failure rule learned on PR #150/#152: a client must be able to
# see that its field did nothing.
SUPPORTED_SESSION_UPDATE_FIELDS = ("tools", "tool_choice", "language")

# The per-stage timing keys ``response.done`` may carry, in pipeline order.
STAGE_TIMING_KEYS = ("stt", "generate", "tool_wait", "phonikud", "tts", "first_delta")

# ---------------------------------------------------------------------------
# Logging / redaction helpers
# ---------------------------------------------------------------------------

REDACTED_MARKER = "***REDACTED***"

_CREDENTIAL_KEY_RE = re.compile(
    r"(key|token|secret|password|passwd|bearer|authorization|credential)", re.IGNORECASE
)


def redact_for_log(data: Mapping[str, object]) -> dict[str, object]:
    """Shallow-copy *data*, masking any value whose key name looks credential-shaped.

    Session config payloads originate with an untrusted client and may carry
    unrelated junk fields; this is the one approved way to log one (or any
    client-supplied mapping) so an API key/token/secret-shaped field can never
    reach a log line verbatim.
    """
    return {
        key: (REDACTED_MARKER if _CREDENTIAL_KEY_RE.search(str(key)) else value)
        for key, value in data.items()
    }


class _SessionLoggerAdapter(logging.LoggerAdapter):
    """Stamps ``session_id=<id>`` into the message text of every record.

    Embedding it in the text (not just ``extra``) means the id survives
    whatever formatter is configured — including the default one — so a
    plain grep for a session id reconstructs its lifecycle.
    """

    def process(self, msg, kwargs):
        return f"session_id={self.extra['session_id']} {msg}", kwargs


def get_session_logger(session_id: str) -> logging.LoggerAdapter:
    """A logger bound to *session_id* — every record it emits carries the id."""
    return _SessionLoggerAdapter(log, {"session_id": session_id})


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    SESSION_CREATED = "session.created"
    SESSION_CLOSED = "session.closed"
    SPEECH_STARTED = "input_audio_buffer.speech_started"
    SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
    ERROR = "error"
    # --- opt-in response lifecycle + interruption (issue #151) ---
    RESPONSE_CREATED = "response.created"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_DONE = "response.done"
    RESPONSE_INTERRUPTED = "response.interrupted"
    # --- session config updates + client-executed tools (hebrew-realtime) ---
    SESSION_UPDATED = "session.updated"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"


class ErrorCode(str, Enum):
    """Documented, named failure modes — never a bare exception string.

    - ``INVALID_SESSION_CONFIG`` — a session config payload was rejected by
      :func:`parse_session_config` (bad rate/format/channels/turn_detection/aec).
    - ``VAD_UNAVAILABLE`` — Silero failed to load or run; distinguishes VAD-down
      from ordinary silence, which emits no event at all.
    - ``INVALID_WIRE_EVENT`` — a client frame was malformed at the wire-codec
      level: not JSON, not a JSON object, an ``input_audio_buffer.append``
      event with a missing/non-string/non-base64 ``audio`` field, or a raw
      BINARY frame (removed as an accepted input by issue #151). ONE code
      covers all of them, exactly like ``INVALID_SESSION_CONFIG`` covers
      every rejected config shape: the specific wire reason
      (:class:`lobes.realtime._wire.WireErrorCode` —
      ``invalid_json``/``invalid_append_event``/``unsupported_frame_type``)
      is named in the message TEXT, not fragmented across the code
      vocabulary. This exists so there is exactly one enumerable list of
      error codes on this wire: before issue #151 t6 the route put a
      ``WireErrorCode`` value into this field verbatim, which meant a client
      rendering codes had to know two enums (see
      :mod:`lobes.realtime._conversation`'s ``WIRE_ERROR_CODES``).
    - ``STT_FORWARD_FAILED`` — a committed turn's forward to Parakeet failed
      (wired by the route layer; the code is reserved here as part of the
      schema this module owns).
    - ``GENERATE_FAILED`` — a response's forward to the generate lane
      (``/v1/chat/completions``) failed, including a gateway
      ``role_infeasible`` 404 (wired by the route layer; issue #151).
    - ``TTS_FAILED`` — a response's forward to the TTS lane
      (``/v1/audio/speech``) failed (wired by the route layer; issue #151).
    - ``RESPONSE_TIMEOUT`` — a response stage (generate or TTS) exceeded its
      configured deadline; the floor always returns to the caller with this
      named error, never stuck responding/speaking (wired by the floor/turn
      state machine; issue #151).
    """

    INVALID_SESSION_CONFIG = "invalid_session_config"
    VAD_UNAVAILABLE = "vad_unavailable"
    INVALID_WIRE_EVENT = "invalid_wire_event"
    STT_FORWARD_FAILED = "stt_forward_failed"
    GENERATE_FAILED = "generate_failed"
    TTS_FAILED = "tts_failed"
    RESPONSE_TIMEOUT = "response_timeout"


@dataclass(frozen=True)
class SessionConfig:
    """A validated session config — see :func:`parse_session_config`.

    ``system_prompt`` is ``None`` unless the client explicitly overrode it
    (or a caller-supplied settings-style default resolved one) — ``None``
    means :class:`Session` falls back to :data:`DEFAULT_SYSTEM_PROMPT`.

    ``tools``/``tool_choice`` hold the client's declaration in OpenAI
    Realtime's own FLAT shape, validated but otherwise verbatim — this
    module never invents, executes or ships a tool, and
    :func:`realtime_tools_to_chat_completions` is what translates them for a
    generate call. ``tools is None`` means "not declared" (the ears-only
    default); an empty tuple means the client declared none, which is a
    different statement and stays distinguishable.

    ``language`` is the short STT code the session speaks
    (:data:`DEFAULT_LANGUAGE`, ``"en"``, unless a deployment default or a
    ``session.update`` says otherwise).

    Frozen, so a ``session.update`` produces a NEW config
    (:func:`apply_session_update`) rather than mutating one another thread
    may be reading.
    """

    input_audio_format: AudioFormat = AudioFormat.PCM16
    input_sample_rate: int = CLIENT_SAMPLE_RATE
    channels: int = 1
    turn_detection: TurnDetectionType = TurnDetectionType.SERVER_VAD
    aec_mode: AECMode = AECMode.NONE
    system_prompt: str | None = None
    tools: tuple[Mapping[str, object], ...] | None = None
    tool_choice: str | Mapping[str, object] | None = None
    language: str = DEFAULT_LANGUAGE


@dataclass(frozen=True)
class SessionCreatedEvent:
    session_id: str
    event_id: str
    timestamp_ms: int
    config: SessionConfig
    type: EventType = field(default=EventType.SESSION_CREATED, init=False)


@dataclass(frozen=True)
class SessionUpdatedEvent:
    """The server's answer to a ``session.update``.

    ``session`` is the echo of ONLY what took effect — a plain JSON-able dict
    holding a subset of :data:`SUPPORTED_SESSION_UPDATE_FIELDS`, built by
    :func:`apply_session_update`. A field the client sent that this server
    does not act on is absent from it, which is the whole point: an
    unsupported field must be visibly ignored, not silently swallowed. The
    field is named ``session`` (not ``config``) because that is what OpenAI
    Realtime's own ``session.updated`` calls it.
    """

    session_id: str
    event_id: str
    timestamp_ms: int
    session: dict[str, object]
    type: EventType = field(default=EventType.SESSION_UPDATED, init=False)


@dataclass(frozen=True)
class SessionClosedEvent:
    session_id: str
    event_id: str
    timestamp_ms: int
    reason: str
    type: EventType = field(default=EventType.SESSION_CLOSED, init=False)


@dataclass(frozen=True)
class SpeechStartedEvent:
    """A VAD speech onset.

    ``at_ms`` is the segmenter's own ``SpeechStarted.at_ms``: elapsed,
    32ms-quantised **audio-stream** time, NOT wall-clock and NOT the same
    clock domain as ``timestamp_ms`` (a ``time.monotonic()`` process clock).
    Keeping both on the event is the point — ``at_ms`` is what makes
    ``VAD_THRESHOLD``/``VAD_SILENCE_MS``/``VAD_PREFIX_PADDING_MS`` effects
    observable against a LIVE session rather than only in fixture replay
    (issue #151 honesty condition h19). ``None`` when the caller had no
    audio-stream time to report.
    """

    session_id: str
    event_id: str
    timestamp_ms: int
    item_id: str
    at_ms: int | None = None
    type: EventType = field(default=EventType.SPEECH_STARTED, init=False)


@dataclass(frozen=True)
class SpeechStoppedEvent:
    """A committed turn boundary.

    ``at_ms`` is audio-stream time, exactly as on :class:`SpeechStartedEvent`.
    ``reason`` is the segmenter's own ``SpeechStopped.reason`` —
    ``"silence"`` (``vad_silence_ms`` of continuous non-speech confirmed the
    stop), ``"max_turn"`` (the ``vad_max_turn_ms`` force-commit, a normal
    boundary event and never an error), or a caller's ``flush`` reason. Until
    issue #151 t6 threaded it, no client could tell a force-commit from a
    silence-confirmed stop at all.
    """

    session_id: str
    event_id: str
    timestamp_ms: int
    item_id: str
    at_ms: int | None = None
    reason: str | None = None
    type: EventType = field(default=EventType.SPEECH_STOPPED, init=False)


@dataclass(frozen=True)
class TranscriptionCompletedEvent:
    session_id: str
    event_id: str
    timestamp_ms: int
    item_id: str | None
    text: str
    type: EventType = field(default=EventType.TRANSCRIPTION_COMPLETED, init=False)


@dataclass(frozen=True)
class ErrorEvent:
    session_id: str
    event_id: str
    timestamp_ms: int
    code: ErrorCode
    message: str
    item_id: str | None = None
    type: EventType = field(default=EventType.ERROR, init=False)


@dataclass(frozen=True)
class ResponseCreatedEvent:
    """A committed turn triggered a server-side reply; the floor moves from
    the caller to the assistant. ``item_id`` is the transcribed turn this
    response answers (the caller's own bookkeeping — ``None`` when a
    response was not triggered by one specific transcribed item)."""

    session_id: str
    event_id: str
    timestamp_ms: int
    response_id: str
    item_id: str | None = None
    type: EventType = field(default=EventType.RESPONSE_CREATED, init=False)


@dataclass(frozen=True)
class ResponseTextDoneEvent:
    """The generate call's full reply text arrived. ``text`` is the reply
    verbatim — the event schema carries it (a client needs it to render);
    logging it verbatim is what's forbidden (see :meth:`Session.complete_response_text`)."""

    session_id: str
    event_id: str
    timestamp_ms: int
    response_id: str
    text: str
    type: EventType = field(default=EventType.RESPONSE_TEXT_DONE, init=False)


@dataclass(frozen=True)
class ResponseAudioDeltaEvent:
    """One chunk of the synthesized reply. ``delta`` is already wire-encoded
    (base64) by the caller — this module owns the event shape only, never
    audio encoding/chunking (that is ``_wire.py``'s job, a sibling module)."""

    session_id: str
    event_id: str
    timestamp_ms: int
    response_id: str
    delta: str
    type: EventType = field(default=EventType.RESPONSE_AUDIO_DELTA, init=False)


@dataclass(frozen=True)
class ResponseFunctionCallArgumentsDoneEvent:
    """The model asked for a tool; the call goes to the CLIENT, which owns
    execution (nothing under ``lobes/`` runs a tool).

    Field names are OpenAI Realtime's verbatim, including ``arguments`` being
    a JSON **string** rather than a parsed object — this module hands the
    model's own argument text through untouched rather than re-serializing
    it, so a client that validates arguments against its schema sees exactly
    what the model produced. ``call_id`` is the handle a later
    ``function_call_output`` must quote back (see
    :func:`parse_function_call_output`).
    """

    session_id: str
    event_id: str
    timestamp_ms: int
    response_id: str | None
    call_id: str
    name: str
    arguments: str
    item_id: str | None = None
    output_index: int = 0
    type: EventType = field(default=EventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE, init=False)


@dataclass(frozen=True)
class StageTimings:
    """Per-stage milliseconds for one response, carried on ``response.done``.

    Every stage is optional and an unmeasured one is ABSENT from
    :meth:`as_dict`, never zeroed — a ``0`` would read as "instant", which is
    the exact dishonesty this field exists to avoid. A turn with no tool call
    reports no ``tool_wait``; an English deployment reports no ``phonikud``.

    ``first_delta`` is the number this contract exists for: time to the first
    audio chunk actually leaving the server, which the acceptance latency
    table is built from (the alternative was scraping logs).
    """

    stt: int | None = None
    generate: int | None = None
    tool_wait: int | None = None
    phonikud: int | None = None
    tts: int | None = None
    first_delta: int | None = None

    def as_dict(self) -> dict[str, int]:
        """The JSON-able mapping, in pipeline order, omitting absent stages."""
        values = vars(self)
        return {key: values[key] for key in STAGE_TIMING_KEYS if values[key] is not None}

    def __bool__(self) -> bool:
        """False when no stage was measured — so an all-absent timings object
        serializes away entirely rather than as an empty mapping."""
        return bool(self.as_dict())


@dataclass(frozen=True)
class ResponseDoneEvent:
    """The reply was delivered in full; the floor returns to the caller.

    ``timings`` is additive and optional: ``None`` (or an all-absent
    :class:`StageTimings`) serializes to a ``response.done`` byte-identical to
    the pre-hebrew-realtime one, so a client that never asked for timings
    never sees the key.
    """

    session_id: str
    event_id: str
    timestamp_ms: int
    response_id: str
    timings: StageTimings | None = None
    type: EventType = field(default=EventType.RESPONSE_DONE, init=False)


@dataclass(frozen=True)
class ResponseInterruptedEvent:
    """Speech arrived while the floor was responding/speaking (barge-in).
    ``truncated`` is the "truncated marker" the #151 spec requires — it is
    always ``True`` today (a barge-in is the only trigger this module
    defines); the field exists so a future, non-truncating interruption
    reason can be represented without a schema break."""

    session_id: str
    event_id: str
    timestamp_ms: int
    response_id: str
    truncated: bool = True
    type: EventType = field(default=EventType.RESPONSE_INTERRUPTED, init=False)


Event = (
    SessionCreatedEvent
    | SessionUpdatedEvent
    | SessionClosedEvent
    | SpeechStartedEvent
    | SpeechStoppedEvent
    | TranscriptionCompletedEvent
    | ErrorEvent
    | ResponseCreatedEvent
    | ResponseTextDoneEvent
    | ResponseAudioDeltaEvent
    | ResponseDoneEvent
    | ResponseInterruptedEvent
    | ResponseFunctionCallArgumentsDoneEvent
)


def _config_to_dict(config: SessionConfig) -> dict[str, object]:
    """The ``config`` mapping a ``session.created`` event carries.

    The six pre-hebrew-realtime keys are unconditional; the three additive
    ones appear ONLY when set, so a tool-free English session's
    ``session.created`` is byte-identical to the one clients see today. A
    default-valued key that appeared anyway would be a wire change dressed up
    as a no-op.
    """
    out: dict[str, object] = {
        "input_audio_format": config.input_audio_format,
        "input_sample_rate": config.input_sample_rate,
        "channels": config.channels,
        "turn_detection": config.turn_detection,
        "aec_mode": config.aec_mode,
        "system_prompt": config.system_prompt,
    }
    if config.tools is not None:
        out["tools"] = [dict(tool) for tool in config.tools]
    if config.tool_choice is not None:
        out["tool_choice"] = (
            dict(config.tool_choice)
            if isinstance(config.tool_choice, Mapping)
            else config.tool_choice
        )
    if config.language != DEFAULT_LANGUAGE:
        out["language"] = config.language
    return out


def event_to_dict(event: Event) -> dict[str, object]:
    """Flatten any schema event into a plain JSON-able dict.

    Enum fields (``type``, ``code``, and any inside a nested ``config``) are
    ``str``-subclassed, so :func:`json.dumps` serializes them by their
    ``.value`` without further help.

    Two fields are conditional rather than always-emitted, and for the same
    reason: an unset additive field must not change the wire. ``config``'s
    additive keys are handled by :func:`_config_to_dict`, and an absent or
    all-absent ``timings`` drops out of ``response.done`` entirely.
    """
    out: dict[str, object] = {}
    for key, value in vars(event).items():
        if key == "config" and isinstance(value, SessionConfig):
            out[key] = _config_to_dict(value)
        elif key == "timings":
            if isinstance(value, StageTimings):
                if value:
                    out[key] = value.as_dict()
            elif value:
                out[key] = dict(value)
        else:
            out[key] = value
    out["type"] = event.type
    return out


# ---------------------------------------------------------------------------
# Config parsing — criterion 1 (rate/format/channels) + criterion 2 (AEC).
# ---------------------------------------------------------------------------


class SessionConfigError(ValueError):
    """A session config payload was rejected.

    Carries a documented :class:`ErrorCode` (never a bare exception string)
    so a caller can build a proper named :class:`ErrorEvent` from it via
    :meth:`to_error_event`.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code

    def to_error_event(self, session_id: str) -> ErrorEvent:
        return ErrorEvent(
            session_id=session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            code=self.code,
            message=str(self),
        )


def _reject(message: str) -> None:
    raise SessionConfigError(ErrorCode.INVALID_SESSION_CONFIG, message)


def parse_language(value: object) -> str:
    """Validate a short language code, returning it verbatim.

    A SHAPE check only (:data:`_LANGUAGE_RE`) — ``"he"``, ``"en"``,
    ``"pt-BR"`` all pass. Whether the STT sidecar actually serves the code is
    that sidecar's answer, not a table this module has any business shipping,
    and inventing a fallback here would hide a typo behind English audio.
    Raises :class:`SessionConfigError` (:attr:`ErrorCode.INVALID_SESSION_CONFIG`)
    on anything else.
    """
    if not isinstance(value, str) or not _LANGUAGE_RE.match(value):
        _reject(f"language must be a short code like 'en' or 'he', got {value!r}")
    return value


def parse_tools(value: object) -> tuple[dict[str, object], ...]:
    """Validate a client's tool declaration in OpenAI Realtime's FLAT shape.

    Accepts a list/tuple of ``{"type": "function", "name", "description",
    "parameters"}`` mappings and returns them as a tuple of plain dicts —
    validated, but otherwise the client's own content verbatim: this module
    never authors, rewrites or executes a tool. ``type`` may be omitted (the
    only tool kind either wire speaks is ``function``); ``name`` is required
    and must be unique, because a duplicate name makes a returned ``call_id``
    ambiguous about which tool was meant. An empty list is valid and means
    "declared none" — a different statement from not declaring at all.

    Raises :class:`SessionConfigError` (:attr:`ErrorCode.INVALID_SESSION_CONFIG`)
    otherwise, naming the offending entry.
    """
    if not isinstance(value, (list, tuple)):
        _reject(f"tools must be a list of tool declarations, got {value!r}")
    tools: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            _reject(f"tools[{index}] must be an object, got {raw!r}")
        tool_type = raw.get("type", _FUNCTION_TOOL_TYPE)
        if tool_type != _FUNCTION_TOOL_TYPE:
            _reject(f"tools[{index}] has unsupported type {tool_type!r}; only 'function' is")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            _reject(f"tools[{index}] requires a non-empty string 'name', got {name!r}")
        if name in seen:
            _reject(f"tools[{index}] repeats the tool name {name!r}")
        seen.add(name)
        description = raw.get("description")
        if description is not None and not isinstance(description, str):
            _reject(f"tools[{index}] 'description' must be a string, got {description!r}")
        parameters = raw.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            _reject(f"tools[{index}] 'parameters' must be a JSON Schema object, got {parameters!r}")
        tools.append(dict(raw))
    return tuple(tools)


def parse_tool_choice(value: object) -> str | dict[str, object]:
    """Validate a ``tool_choice``, returning it in Realtime's own shape.

    Either one of :data:`_TOOL_CHOICE_KEYWORDS` (``"auto"``/``"none"``/
    ``"required"``) or the FLAT forced-tool object ``{"type": "function",
    "name": ...}``. Translation to chat-completions' nested form is
    :func:`realtime_tool_choice_to_chat_completions`'s job, not this one's —
    parsing keeps the client's vocabulary, so the ``session.updated`` echo
    can quote it back unchanged.
    """
    if not isinstance(value, (str, Mapping)):
        _reject(f"tool_choice must be a keyword or a function object, got {value!r}")
    if isinstance(value, str):
        if value not in _TOOL_CHOICE_KEYWORDS:
            _reject(
                f"unsupported tool_choice {value!r}; accepted keywords are "
                f"{_TOOL_CHOICE_KEYWORDS}, or a "
                "{'type': 'function', 'name': ...} object"
            )
        return value
    if value.get("type") != _FUNCTION_TOOL_TYPE:
        _reject(f"tool_choice object must have type 'function', got {value.get('type')!r}")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        _reject(f"tool_choice object requires a non-empty string 'name', got {name!r}")
    return dict(value)


def parse_session_config(
    payload: Mapping[str, object] | None = None,
    *,
    default_turn_detection: str = "server_vad",
    default_aec_mode: str = "none",
    default_system_prompt: str | None = None,
    default_language: str = DEFAULT_LANGUAGE,
) -> SessionConfig:
    """Validate a client's ``session.update``-style config dict.

    Defaults when a key is omitted: ``input_audio_format=pcm16``,
    ``input_sample_rate=24000`` (:data:`CLIENT_SAMPLE_RATE`),
    ``input_channels=1``, ``turn_detection`` and ``aec_mode`` from the
    *default_turn_detection*/*default_aec_mode* args (a caller threads these
    from :mod:`lobes.realtime._settings`'s ``default_turn_detection``/
    ``default_aec_mode``, which are themselves ``"server_vad"``/``"none"``),
    and ``system_prompt`` from *default_system_prompt* (a caller threads this
    from an operator-set env default; issue #151) — the client's own
    ``system_prompt`` key, if present, always overrides it. ``language``
    follows the same two-level rule (*default_language*, itself
    :data:`DEFAULT_LANGUAGE`, overridden by the payload's own key): a
    per-deployment default AND a per-session override, and with neither set
    the STT request still says ``"en"`` byte-for-byte.

    ``tools``/``tool_choice`` are accepted here too (same validators as
    :func:`apply_session_update`), but the practical path for them is
    ``session.update``: this parser's payload is the connect-time query
    string, where a JSON Schema does not fit.

    PCM16 mono little-endian at 24000 Hz or 16000 Hz is the only accepted
    wire format; AEC stays ``none`` unless the payload explicitly sets
    ``aec_mode``. Anything else raises :class:`SessionConfigError` with
    :attr:`ErrorCode.INVALID_SESSION_CONFIG` — a named error, not a bare
    exception string.
    """
    payload = payload or {}

    fmt = payload.get("input_audio_format", AudioFormat.PCM16.value)
    if fmt != AudioFormat.PCM16.value:
        _reject(
            f"unsupported input_audio_format {fmt!r}; only "
            f"{AudioFormat.PCM16.value!r} (PCM16 mono little-endian) is supported"
        )

    raw_rate = payload.get("input_sample_rate", CLIENT_SAMPLE_RATE)
    try:
        rate = int(raw_rate)
    except (TypeError, ValueError):
        _reject(f"input_sample_rate must be an integer, got {raw_rate!r}")
    if rate not in _SUPPORTED_SAMPLE_RATES:
        _reject(
            f"unsupported input_sample_rate {rate}; accepted rates are {_SUPPORTED_SAMPLE_RATES}"
        )

    raw_channels = payload.get("input_channels", 1)
    try:
        channels = int(raw_channels)
    except (TypeError, ValueError):
        _reject(f"input_channels must be an integer, got {raw_channels!r}")
    if channels != 1:
        _reject(f"unsupported input_channels {channels}; only mono (1) is supported")

    turn_detection = payload.get("turn_detection", default_turn_detection)
    if turn_detection != TurnDetectionType.SERVER_VAD.value:
        _reject(
            f"unsupported turn_detection {turn_detection!r}; only "
            f"{TurnDetectionType.SERVER_VAD.value!r} is supported"
        )

    raw_aec = payload.get("aec_mode", default_aec_mode)
    try:
        aec_mode = AECMode(raw_aec)
    except ValueError:
        _reject(f"unsupported aec_mode {raw_aec!r}")

    system_prompt = payload.get("system_prompt", default_system_prompt)
    if system_prompt is not None and not isinstance(system_prompt, str):
        _reject(f"system_prompt must be a string, got {system_prompt!r}")

    language = parse_language(payload.get("language", default_language))

    raw_tools = payload.get("tools")
    tools = None if raw_tools is None else parse_tools(raw_tools)

    raw_tool_choice = payload.get("tool_choice")
    tool_choice = None if raw_tool_choice is None else parse_tool_choice(raw_tool_choice)

    return SessionConfig(
        input_audio_format=AudioFormat.PCM16,
        input_sample_rate=rate,
        channels=1,
        turn_detection=TurnDetectionType.SERVER_VAD,
        aec_mode=aec_mode,
        system_prompt=system_prompt,
        tools=tools,
        tool_choice=tool_choice,
        language=language,
    )


# ---------------------------------------------------------------------------
# session.update -> session.updated, and the flat/nested tool translation.
# ---------------------------------------------------------------------------


class SessionUpdate(NamedTuple):
    """The pure result of applying one ``session.update``.

    ``config`` is a NEW :class:`SessionConfig` (the old one is frozen and
    untouched); ``applied`` is the echo body for
    :class:`SessionUpdatedEvent`, holding ONLY the fields that took effect.
    """

    config: SessionConfig
    applied: dict[str, object]


def apply_session_update(config: SessionConfig, payload: Mapping[str, object]) -> SessionUpdate:
    """Apply a client ``session.update`` to *config* — pure, no session state.

    *payload* is the whole client event, OpenAI-shaped:
    ``{"type": "session.update", "session": {...}}``. The ``session`` object
    is REQUIRED (a bare patch mapping is not accepted) because that is the
    shape a stock OpenAI Realtime client sends, and quietly accepting a
    second shape would make the contract two contracts.

    Only :data:`SUPPORTED_SESSION_UPDATE_FIELDS` are read. Anything else —
    ``voice``, ``modalities``, ``turn_detection`` retuning, a typo — is
    ignored AND absent from :attr:`SessionUpdate.applied`, so the client can
    SEE that it did nothing. A key present with a malformed value is a named
    :class:`SessionConfigError` instead (:attr:`ErrorCode.INVALID_SESSION_CONFIG`
    — the same code a bad connect-time config gets, since "a session config
    payload was rejected" is exactly what happened); the caller turns it into
    an ``error`` event and keeps the session open.
    """
    session_patch = payload.get("session") if isinstance(payload, Mapping) else None
    if not isinstance(session_patch, Mapping):
        _reject(f"{SESSION_UPDATE_EVENT_TYPE} requires a 'session' object, got {session_patch!r}")

    updates: dict[str, object] = {}
    applied: dict[str, object] = {}
    if "tools" in session_patch:
        tools = parse_tools(session_patch["tools"])
        updates["tools"] = tools
        applied["tools"] = [dict(tool) for tool in tools]
    if "tool_choice" in session_patch:
        tool_choice = parse_tool_choice(session_patch["tool_choice"])
        updates["tool_choice"] = tool_choice
        applied["tool_choice"] = (
            dict(tool_choice) if isinstance(tool_choice, Mapping) else tool_choice
        )
    if "language" in session_patch:
        language = parse_language(session_patch["language"])
        updates["language"] = language
        applied["language"] = language

    return SessionUpdate(config=replace(config, **updates), applied=applied)


def realtime_tools_to_chat_completions(
    tools: Iterable[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Translate FLAT Realtime tools into NESTED chat-completions tools.

    Realtime declares ``{"type": "function", "name", "description",
    "parameters"}``; ``/v1/chat/completions`` wants ``{"type": "function",
    "function": {"name", "description", "parameters"}}``. This is the ONE
    place that difference is encoded, so the generate-request builder never
    re-derives it (and the two can never disagree about, say, whether an
    absent ``description`` becomes ``null``: it stays ABSENT).

    ``None`` yields ``[]`` — an undeclared tool set and an empty one produce
    the same generate request, since a request carrying ``tools: []`` is what
    "no tools" means downstream either way.
    """
    translated: list[dict[str, object]] = []
    for tool in tools or ():
        function: dict[str, object] = {"name": tool["name"]}
        for key in ("description", "parameters"):
            if key in tool:
                function[key] = tool[key]
        translated.append({"type": _FUNCTION_TOOL_TYPE, "function": function})
    return translated


def realtime_tool_choice_to_chat_completions(
    tool_choice: str | Mapping[str, object] | None,
) -> str | dict[str, object] | None:
    """Translate a ``tool_choice`` for the generate call; ``None`` stays ``None``.

    The keywords (``"auto"``/``"none"``/``"required"``) are identical on both
    wires and pass through untouched. The forced-tool OBJECT is not:
    Realtime's ``{"type": "function", "name": "x"}`` nests into
    chat-completions' ``{"type": "function", "function": {"name": "x"}}`` —
    the same flat-vs-nested split as the tools themselves, and the reason
    this is a function rather than a pass-through the caller inlines.
    """
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return {"type": _FUNCTION_TOOL_TYPE, "function": {"name": tool_choice["name"]}}


# ---------------------------------------------------------------------------
# The client's tool RESULT — conversation.item.create(function_call_output).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionCallOutput:
    """A tool result the client returned, already shape-validated.

    ``call_id`` is the handle from the
    :class:`ResponseFunctionCallArgumentsDoneEvent` this answers, and
    ``output`` is the client's result text verbatim (a JSON string by
    convention, but never parsed here — nothing under ``lobes/`` interprets a
    tool's result). Whether this ``call_id`` is the ONE outstanding call of
    the current response is the bridge's bookkeeping, not a shape question.
    """

    call_id: str
    output: str


class FunctionCallOutputError(ValueError):
    """A ``conversation.item.create(function_call_output)`` frame was malformed.

    Carries :attr:`ErrorCode.INVALID_WIRE_EVENT` — the code that already
    means "a malformed client frame" — rather than a new enum member, with
    the specific reason prefixed into the message text exactly as the three
    :class:`lobes.realtime._wire.WireErrorCode` reasons are. One enumerable
    list of error codes on this wire is a contract (see
    :attr:`ErrorCode.INVALID_WIRE_EVENT`'s own docstring), so a new failure
    mode earns a reason token, not a code.
    """

    code = ErrorCode.INVALID_WIRE_EVENT
    reason = "invalid_function_call_output"

    def __init__(self, message: str) -> None:
        super().__init__(f"{self.reason}: {message}")


def parse_function_call_output(payload: Mapping[str, object]) -> FunctionCallOutput:
    """Pull ``call_id``/``output`` out of a client tool-result event.

    *payload* is the whole client event, OpenAI-shaped:
    ``{"type": "conversation.item.create", "item": {"type":
    "function_call_output", "call_id": ..., "output": ...}}``. Both item
    fields are required and must be strings; ``output`` may be empty (a tool
    that produced nothing is not malformed), ``call_id`` may not (an
    un-attributable result is).

    Raises :class:`FunctionCallOutputError`, never a bare ``KeyError`` — the
    caller turns it into a named ``error`` event via
    :meth:`Session.fail_wire_event`, which changes no state, so an
    adversarial or confused client cannot end a session with a bad frame.
    """
    if payload.get("type") != CONVERSATION_ITEM_CREATE_EVENT_TYPE:
        raise FunctionCallOutputError(
            f"expected a {CONVERSATION_ITEM_CREATE_EVENT_TYPE!r} event, "
            f"got {payload.get('type')!r}"
        )
    item = payload.get("item")
    if not isinstance(item, Mapping):
        raise FunctionCallOutputError(f"'item' must be an object, got {item!r}")
    if item.get("type") != FUNCTION_CALL_OUTPUT_ITEM_TYPE:
        raise FunctionCallOutputError(
            f"expected an item of type {FUNCTION_CALL_OUTPUT_ITEM_TYPE!r}, "
            f"got {item.get('type')!r}"
        )
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise FunctionCallOutputError(f"requires a non-empty string 'call_id', got {call_id!r}")
    output = item.get("output")
    if not isinstance(output, str):
        raise FunctionCallOutputError(f"requires a string 'output', got {output!r}")
    return FunctionCallOutput(call_id=call_id, output=output)


# ---------------------------------------------------------------------------
# Session state machine + teardown bookkeeping — criteria 3 and 4.
# ---------------------------------------------------------------------------


class SessionState(str, Enum):
    IDLE = "idle"
    SPEECH = "speech"
    TRANSCRIBING = "transcribing"
    RESPONDING = "responding"
    SPEAKING = "speaking"
    CLOSED = "closed"


# Mirrors scripts/realtime-voice-loop.py's SYSTEM_PROMPT — the client-side
# text that moves server-side per issue #151. An operator-set env default
# (a later task's concern) and a per-session connect-config override both
# flow through :func:`parse_session_config`'s ``system_prompt``/
# ``default_system_prompt``; this is the last-resort fallback when neither
# is set.
DEFAULT_SYSTEM_PROMPT = (
    "You are the voice of this machine. You are being spoken to out loud and "
    "your reply is read back aloud by a text-to-speech voice, so answer in "
    "one or two short spoken sentences. No markdown, no lists, no code "
    "blocks, no emoji — just what you would say."
)


class SessionClosedError(RuntimeError):
    """Raised when a state-changing call arrives after :meth:`Session.teardown`."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session {session_id} is already closed")
        self.session_id = session_id


class Session:
    """A realtime session's state + bookkeeping, bound to one session id.

    Holds only what THIS engine allocates: the current lifecycle state, the
    in-flight conversation item id, the in-flight response id, and (since
    issue #151) the per-session conversation history + system prompt. Audio
    buffers and VAD/segmenter state belong to other modules (the segmenter,
    the route layer); per-stage deadlines/callbacks/cancellation belong to a
    separate floor/turn state machine — this class's job is the schema-level
    bookkeeping and the teardown contract: from ANY state, :meth:`teardown`
    releases it all, including history.
    """

    def __init__(self, config: SessionConfig, session_id: str | None = None) -> None:
        self.session_id = session_id or gen_session_id()
        self.config = config
        self.state = SessionState.IDLE
        self.current_item_id: str | None = None
        self.current_response_id: str | None = None
        self.vad_available = True
        self.system_prompt: str = (
            config.system_prompt if config.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        )
        self._history: list[dict[str, str]] = []
        self._closed = False
        self.log = get_session_logger(self.session_id)

    @property
    def has_open_item(self) -> bool:
        return self.current_item_id is not None

    @classmethod
    def create(
        cls,
        config: SessionConfig,
        raw_payload: Mapping[str, object] | None = None,
        session_id: str | None = None,
    ) -> tuple["Session", SessionCreatedEvent]:
        """Allocate a new session and its ``session.created`` event."""
        session = cls(config, session_id=session_id)
        session.log.info(
            "session created rate=%s channels=%s turn_detection=%s aec_mode=%s",
            config.input_sample_rate,
            config.channels,
            config.turn_detection.value,
            config.aec_mode.value,
        )
        if raw_payload:
            # Debug-only, and only through the redaction helper — a client
            # payload is untrusted and may carry a credential-shaped field.
            session.log.debug("session config payload: %s", redact_for_log(dict(raw_payload)))
        event = SessionCreatedEvent(
            session_id=session.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            config=config,
        )
        return session, event

    def _require_not_closed(self) -> None:
        if self._closed:
            raise SessionClosedError(self.session_id)

    def update_config(self, payload: Mapping[str, object]) -> SessionUpdatedEvent:
        """Apply a client ``session.update`` and answer ``session.updated``.

        Swaps in the new :class:`SessionConfig`
        (:func:`apply_session_update` builds it; the old one is frozen and
        never mutated) and echoes ONLY what took effect. A malformed payload
        raises :class:`SessionConfigError` with the config unchanged — the
        caller answers with ``exc.to_error_event(session.session_id)`` and the
        session stays open and usable, because a bad update is not a turn
        boundary and not a reason to hang up.

        Logs which FIELDS took effect and how many tools were declared, never
        their content: a tool's description and JSON Schema are the client's,
        and this module's logging discipline does not make exceptions for
        structured data.
        """
        self._require_not_closed()
        update = apply_session_update(self.config, payload)
        self.config = update.config
        self.log.info(
            "session updated fields=%s tools=%s language=%s",
            sorted(update.applied),
            len(self.config.tools) if self.config.tools is not None else None,
            self.config.language,
        )
        return SessionUpdatedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            session=update.applied,
        )

    def begin_speech(self, *, at_ms: int | None = None) -> SpeechStartedEvent:
        """Record a VAD-reported speech boundary onset.

        Called by the route layer once the (separately owned) VAD/segmenter
        logic decides speech started — this method only owns the resulting
        state transition and event. *at_ms* is the segmenter's own
        audio-stream timestamp, carried onto the wire verbatim; see
        :class:`SpeechStartedEvent` for the clock-domain caveat that makes it
        worth a separate field from ``timestamp_ms``.
        """
        self._require_not_closed()
        self.current_item_id = gen_item_id()
        self.state = SessionState.SPEECH
        self.log.info("speech started item_id=%s at_ms=%s", self.current_item_id, at_ms)
        return SpeechStartedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            item_id=self.current_item_id,
            at_ms=at_ms,
        )

    def end_speech(
        self, *, at_ms: int | None = None, reason: str | None = None
    ) -> SpeechStoppedEvent:
        """Record a VAD-reported speech boundary offset (turn committed).

        *at_ms* and *reason* are the segmenter's own
        ``SpeechStopped.at_ms``/``.reason``, carried onto the wire verbatim —
        ``reason`` is what lets a client tell a ``"max_turn"`` force-commit
        from a ``"silence"``-confirmed stop.
        """
        self._require_not_closed()
        item_id = self.current_item_id
        self.state = SessionState.TRANSCRIBING
        self.log.info("speech stopped item_id=%s at_ms=%s reason=%s", item_id, at_ms, reason)
        return SpeechStoppedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            item_id=item_id,
            at_ms=at_ms,
            reason=reason,
        )

    def complete_transcription(self, text: str) -> TranscriptionCompletedEvent:
        """A committed turn's Parakeet transcript arrived. Never logs *text*
        verbatim — only its length — so transcript content never appears in a
        log line."""
        self._require_not_closed()
        item_id = self.current_item_id
        self.current_item_id = None
        self.state = SessionState.IDLE
        self.log.info("transcription completed item_id=%s chars=%d", item_id, len(text))
        return TranscriptionCompletedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            item_id=item_id,
            text=text,
        )

    def fail_transcription(self, message: str) -> ErrorEvent:
        """A committed turn's forward to Parakeet failed — a named error
        event, never a silently dropped turn."""
        self._require_not_closed()
        item_id = self.current_item_id
        self.current_item_id = None
        self.state = SessionState.IDLE
        self.log.warning(
            "transcription failed item_id=%s code=%s", item_id, ErrorCode.STT_FORWARD_FAILED.value
        )
        return ErrorEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            code=ErrorCode.STT_FORWARD_FAILED,
            message=message,
            item_id=item_id,
        )

    def fail_wire_event(self, message: str) -> ErrorEvent:
        """A client frame was malformed at the wire-codec level.

        Changes NO state — a bad frame is not a turn boundary, does not open
        or close an item, and the session stays open to keep receiving (the
        #149 contract, unchanged). Deliberately does not call
        :meth:`_require_not_closed` either, mirroring
        :meth:`mark_vad_unavailable`: adversarial client input arriving in a
        teardown race must never turn into an exception.

        *message* is expected to already name the specific wire reason (the
        :class:`lobes.realtime._wire.WireErrorCode` value), because
        :attr:`ErrorCode.INVALID_WIRE_EVENT` alone does not distinguish the
        three — see its docstring and
        :mod:`lobes.realtime._conversation`'s ``WIRE_ERROR_CODES``.
        """
        self.log.warning("wire event rejected code=%s", ErrorCode.INVALID_WIRE_EVENT.value)
        return ErrorEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            code=ErrorCode.INVALID_WIRE_EVENT,
            message=message,
        )

    def mark_vad_unavailable(self, message: str = "server_vad is unavailable") -> ErrorEvent:
        """Silero failed to load/run — the NAMED error distinguishing VAD-down
        from ordinary silence (which emits no event at all)."""
        self.vad_available = False
        self.log.error("vad unavailable code=%s", ErrorCode.VAD_UNAVAILABLE.value)
        return ErrorEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            code=ErrorCode.VAD_UNAVAILABLE,
            message=message,
        )

    # -----------------------------------------------------------------
    # Opt-in response lifecycle + interruption (issue #151).
    #
    # Bookkeeping + event-shape only: per-stage deadlines, cancellation
    # callbacks, and the actual barge-in TRIGGER decision belong to a
    # separate floor/turn state machine, not this class. These methods are
    # what that machine (and the route layer) call to keep Session.state —
    # the schema's floor-holder field — and the response id in sync with
    # whichever event just fired.
    # -----------------------------------------------------------------

    def begin_response(self, item_id: str | None = None) -> ResponseCreatedEvent:
        """A committed turn triggered a server-side reply — the floor moves
        from the caller to the assistant. *item_id* identifies the
        transcribed turn being answered (the caller's own bookkeeping;
        optional since a response need not trace back to one specific item).
        """
        self._require_not_closed()
        self.current_response_id = gen_response_id()
        self.state = SessionState.RESPONDING
        self.log.info(
            "response created response_id=%s item_id=%s", self.current_response_id, item_id
        )
        return ResponseCreatedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            response_id=self.current_response_id,
            item_id=item_id,
        )

    def complete_response_text(self, text: str) -> ResponseTextDoneEvent:
        """The generate call returned the full reply text; the floor moves to
        SPEAKING for TTS delivery. Never logs *text* verbatim — only its
        length. Does not touch history — call :meth:`append_history`
        explicitly; history is opt-in bookkeeping, never an implicit side
        effect of a state transition."""
        self._require_not_closed()
        response_id = self.current_response_id
        self.state = SessionState.SPEAKING
        self.log.info("response text done response_id=%s chars=%d", response_id, len(text))
        return ResponseTextDoneEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            response_id=response_id,
            text=text,
        )

    def emit_function_call_arguments_done(
        self,
        *,
        call_id: str,
        name: str,
        arguments: str,
        item_id: str | None = None,
        output_index: int = 0,
    ) -> ResponseFunctionCallArgumentsDoneEvent:
        """Hand the model's tool call to the client, which owns execution.

        Shape only, exactly like :meth:`emit_audio_delta`: it changes NO
        state. Whether the floor now waits on the client, how long, and what
        happens when the wait expires belong to the floor/turn state machine
        — this method's job is to mint the event. Never logs *arguments*
        verbatim (only its length), matching the transcript/reply-text rule.
        """
        self._require_not_closed()
        self.log.info(
            "function call call_id=%s name=%s arg_chars=%d",
            call_id,
            name,
            len(arguments),
        )
        return ResponseFunctionCallArgumentsDoneEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            response_id=self.current_response_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
            item_id=item_id,
            output_index=output_index,
        )

    def emit_audio_delta(self, delta: str) -> ResponseAudioDeltaEvent:
        """One already wire-encoded (base64) chunk of the synthesized reply.
        Does not change :attr:`state` — a response may emit any number of
        deltas while SPEAKING."""
        self._require_not_closed()
        return ResponseAudioDeltaEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            response_id=self.current_response_id,
            delta=delta,
        )

    def complete_response(self, timings: StageTimings | None = None) -> ResponseDoneEvent:
        """All audio was delivered; the floor returns to the caller.

        *timings* is optional and additive: a caller that measured the stages
        passes a :class:`StageTimings`, and one that did not passes nothing
        and emits the same ``response.done`` as before. Measuring is the
        caller's job — this method never invents a duration.
        """
        self._require_not_closed()
        response_id = self.current_response_id
        self.current_response_id = None
        self.state = SessionState.IDLE
        self.log.info(
            "response done response_id=%s timings=%s",
            response_id,
            timings.as_dict() if timings else None,
        )
        return ResponseDoneEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            response_id=response_id,
            timings=timings,
        )

    def interrupt_response(self, reason: str = "barge_in") -> ResponseInterruptedEvent:
        """Speech arrived while RESPONDING or SPEAKING — the floor returns to
        the caller immediately, from either state, and any undelivered audio
        is truncated (the event's ``truncated`` marker). *reason* documents
        the trigger for logging (default: a barge-in)."""
        self._require_not_closed()
        response_id = self.current_response_id
        self.current_response_id = None
        self.state = SessionState.IDLE
        self.log.info("response interrupted response_id=%s reason=%s", response_id, reason)
        return ResponseInterruptedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            response_id=response_id,
        )

    def fail_response(self, code: ErrorCode, message: str) -> ErrorEvent:
        """A response stage failed or expired (``GENERATE_FAILED``,
        ``TTS_FAILED``, or ``RESPONSE_TIMEOUT``) — the floor returns to the
        caller with a named error, never stuck responding/speaking."""
        self._require_not_closed()
        response_id = self.current_response_id
        self.current_response_id = None
        self.state = SessionState.IDLE
        self.log.warning("response failed response_id=%s code=%s", response_id, code.value)
        return ErrorEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            code=code,
            message=message,
        )

    # -----------------------------------------------------------------
    # Per-session conversation history (issue #151) — ephemeral: lives only
    # on this object, nothing is ever written to disk, teardown drops it.
    # -----------------------------------------------------------------

    def append_history(self, role: str, content: str) -> None:
        """Append one turn to this session's in-memory conversation history.

        Never logs *content* verbatim — only its length — matching the
        module's transcript/reply-text redaction discipline. Purely opt-in:
        no method here calls this implicitly, so an ears-only session that
        never triggers a response never accumulates history."""
        self._require_not_closed()
        self._history.append({"role": role, "content": content})
        self.log.debug(
            "history appended role=%s chars=%d total_turns=%d",
            role,
            len(content),
            len(self._history),
        )

    def get_history(self) -> list[dict[str, str]]:
        """A defensive copy of this session's conversation history so far.

        Lives only on this :class:`Session` object — no disk, no module-level
        state — and is empty again after :meth:`teardown`."""
        return list(self._history)

    def pop_history_if_last(self, role: str, content: str) -> bool:
        """Remove the LAST history entry, only if it is exactly this one.

        The continuation merge (approved deviation d9, layer B) takes back a
        half-turn the machine committed too early. Exact-match-on-the-tail is
        the whole safety rule: if anything was appended after it — an assistant
        reply, a tool call, a tool result — the entry is no longer last, this
        refuses, and history is left alone.
        """
        if self._history and self._history[-1] == {"role": role, "content": content}:
            self._history.pop()
            return True
        return False

    def teardown(self, reason: str = "client_disconnect") -> SessionClosedEvent:
        """Release all session bookkeeping. Safe from ANY state — idle,
        mid-speech, mid-transcription, responding, speaking — and idempotent
        (a second call is a no-op beyond returning a fresh close event)."""
        prior_state = self.state
        already_closed = self._closed
        self.state = SessionState.CLOSED
        self.current_item_id = None
        self.current_response_id = None
        self._history = []
        self.vad_available = False
        self._closed = True
        if not already_closed:
            self.log.info("session closed reason=%s prior_state=%s", reason, prior_state.value)
        return SessionClosedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            reason=reason,
        )


__all__ = [
    "REDACTED_MARKER",
    "redact_for_log",
    "get_session_logger",
    "EventType",
    "ErrorCode",
    "SessionConfig",
    "SessionCreatedEvent",
    "SessionUpdatedEvent",
    "SessionClosedEvent",
    "SpeechStartedEvent",
    "SpeechStoppedEvent",
    "TranscriptionCompletedEvent",
    "ErrorEvent",
    "ResponseCreatedEvent",
    "ResponseTextDoneEvent",
    "ResponseAudioDeltaEvent",
    "ResponseDoneEvent",
    "ResponseInterruptedEvent",
    "ResponseFunctionCallArgumentsDoneEvent",
    "StageTimings",
    "STAGE_TIMING_KEYS",
    "Event",
    "event_to_dict",
    "SessionConfigError",
    "parse_session_config",
    "DEFAULT_LANGUAGE",
    "SUPPORTED_SESSION_UPDATE_FIELDS",
    "parse_language",
    "parse_tools",
    "parse_tool_choice",
    "SessionUpdate",
    "apply_session_update",
    "realtime_tools_to_chat_completions",
    "realtime_tool_choice_to_chat_completions",
    "FunctionCallOutput",
    "FunctionCallOutputError",
    "parse_function_call_output",
    "SessionState",
    "SessionClosedError",
    "Session",
    "DEFAULT_SYSTEM_PROMPT",
    # re-exported from protocol.py — reused, not redefined.
    "AudioFormat",
    "TurnDetectionType",
    "AECMode",
    "gen_session_id",
    "gen_event_id",
    "gen_item_id",
    "gen_response_id",
]
