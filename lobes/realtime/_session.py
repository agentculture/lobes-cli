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
this PR's scope is audio-in, boundaries, and transcription; see
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
  forward failed).

Logging: every helper here logs through :func:`get_session_logger`, which
stamps the session id into the message text itself (not just ``extra``) so
grepping logs for one session id reconstructs its whole lifecycle even under
a bare-bones formatter. :func:`redact_for_log` is the one approved way to log
a client-supplied mapping (e.g. a raw session-config payload) — it masks any
credential-shaped key so an API key or token can never reach a log line.
Transcribed text is deliberately never logged verbatim (only its length).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from .protocol import (
    CLIENT_SAMPLE_RATE,
    STT_SAMPLE_RATE,
    AECMode,
    AudioFormat,
    TurnDetectionType,
    gen_event_id,
    gen_item_id,
    gen_session_id,
    timestamp_ms,
)

log = logging.getLogger(__name__)

_SUPPORTED_SAMPLE_RATES = (CLIENT_SAMPLE_RATE, STT_SAMPLE_RATE)  # 24000 (default), 16000

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


class ErrorCode(str, Enum):
    """Documented, named failure modes — never a bare exception string.

    - ``INVALID_SESSION_CONFIG`` — a session config payload was rejected by
      :func:`parse_session_config` (bad rate/format/channels/turn_detection/aec).
    - ``VAD_UNAVAILABLE`` — Silero failed to load or run; distinguishes VAD-down
      from ordinary silence, which emits no event at all.
    - ``STT_FORWARD_FAILED`` — a committed turn's forward to Parakeet failed
      (wired by the route layer; the code is reserved here as part of the
      schema this module owns).
    """

    INVALID_SESSION_CONFIG = "invalid_session_config"
    VAD_UNAVAILABLE = "vad_unavailable"
    STT_FORWARD_FAILED = "stt_forward_failed"


@dataclass(frozen=True)
class SessionConfig:
    """A validated session config — see :func:`parse_session_config`."""

    input_audio_format: AudioFormat = AudioFormat.PCM16
    input_sample_rate: int = CLIENT_SAMPLE_RATE
    channels: int = 1
    turn_detection: TurnDetectionType = TurnDetectionType.SERVER_VAD
    aec_mode: AECMode = AECMode.NONE


@dataclass(frozen=True)
class SessionCreatedEvent:
    session_id: str
    event_id: str
    timestamp_ms: int
    config: SessionConfig
    type: EventType = field(default=EventType.SESSION_CREATED, init=False)


@dataclass(frozen=True)
class SessionClosedEvent:
    session_id: str
    event_id: str
    timestamp_ms: int
    reason: str
    type: EventType = field(default=EventType.SESSION_CLOSED, init=False)


@dataclass(frozen=True)
class SpeechStartedEvent:
    session_id: str
    event_id: str
    timestamp_ms: int
    item_id: str
    type: EventType = field(default=EventType.SPEECH_STARTED, init=False)


@dataclass(frozen=True)
class SpeechStoppedEvent:
    session_id: str
    event_id: str
    timestamp_ms: int
    item_id: str
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


Event = (
    SessionCreatedEvent
    | SessionClosedEvent
    | SpeechStartedEvent
    | SpeechStoppedEvent
    | TranscriptionCompletedEvent
    | ErrorEvent
)


def event_to_dict(event: Event) -> dict[str, object]:
    """Flatten any schema event into a plain JSON-able dict.

    Enum fields (``type``, ``code``, and any inside a nested ``config``) are
    ``str``-subclassed, so :func:`json.dumps` serializes them by their
    ``.value`` without further help.
    """
    out: dict[str, object] = {}
    for key, value in vars(event).items():
        if key == "config" and isinstance(value, SessionConfig):
            out[key] = {
                "input_audio_format": value.input_audio_format,
                "input_sample_rate": value.input_sample_rate,
                "channels": value.channels,
                "turn_detection": value.turn_detection,
                "aec_mode": value.aec_mode,
            }
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


def parse_session_config(
    payload: Mapping[str, object] | None = None,
    *,
    default_turn_detection: str = "server_vad",
    default_aec_mode: str = "none",
) -> SessionConfig:
    """Validate a client's ``session.update``-style config dict.

    Defaults when a key is omitted: ``input_audio_format=pcm16``,
    ``input_sample_rate=24000`` (:data:`CLIENT_SAMPLE_RATE`),
    ``input_channels=1``, ``turn_detection`` and ``aec_mode`` from the
    *default_turn_detection*/*default_aec_mode* args (a caller threads these
    from :mod:`lobes.realtime._settings`'s ``default_turn_detection``/
    ``default_aec_mode``, which are themselves ``"server_vad"``/``"none"``).

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

    return SessionConfig(
        input_audio_format=AudioFormat.PCM16,
        input_sample_rate=rate,
        channels=1,
        turn_detection=TurnDetectionType.SERVER_VAD,
        aec_mode=aec_mode,
    )


# ---------------------------------------------------------------------------
# Session state machine + teardown bookkeeping — criteria 3 and 4.
# ---------------------------------------------------------------------------


class SessionState(str, Enum):
    IDLE = "idle"
    SPEECH = "speech"
    TRANSCRIBING = "transcribing"
    CLOSED = "closed"


class SessionClosedError(RuntimeError):
    """Raised when a state-changing call arrives after :meth:`Session.teardown`."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session {session_id} is already closed")
        self.session_id = session_id


class Session:
    """A realtime session's state + bookkeeping, bound to one session id.

    Holds only what THIS engine allocates: the current lifecycle state and
    the in-flight conversation item id. Audio buffers and VAD/segmenter state
    belong to other modules (the segmenter, the route layer) — this class's
    job is the schema-level bookkeeping and the teardown contract: from ANY
    state, :meth:`teardown` releases it all.
    """

    def __init__(self, config: SessionConfig, session_id: str | None = None) -> None:
        self.session_id = session_id or gen_session_id()
        self.config = config
        self.state = SessionState.IDLE
        self.current_item_id: str | None = None
        self.vad_available = True
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

    def begin_speech(self) -> SpeechStartedEvent:
        """Record a VAD-reported speech boundary onset.

        Called by the route layer once the (separately owned) VAD/segmenter
        logic decides speech started — this method only owns the resulting
        state transition and event.
        """
        self._require_not_closed()
        self.current_item_id = gen_item_id()
        self.state = SessionState.SPEECH
        self.log.info("speech started item_id=%s", self.current_item_id)
        return SpeechStartedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            item_id=self.current_item_id,
        )

    def end_speech(self) -> SpeechStoppedEvent:
        """Record a VAD-reported speech boundary offset (turn committed)."""
        self._require_not_closed()
        item_id = self.current_item_id
        self.state = SessionState.TRANSCRIBING
        self.log.info("speech stopped item_id=%s", item_id)
        return SpeechStoppedEvent(
            session_id=self.session_id,
            event_id=gen_event_id(),
            timestamp_ms=timestamp_ms(),
            item_id=item_id,
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

    def teardown(self, reason: str = "client_disconnect") -> SessionClosedEvent:
        """Release all session bookkeeping. Safe from ANY state — idle,
        mid-speech, mid-transcription — and idempotent (a second call is a
        no-op beyond returning a fresh close event)."""
        prior_state = self.state
        already_closed = self._closed
        self.state = SessionState.CLOSED
        self.current_item_id = None
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
    "SessionClosedEvent",
    "SpeechStartedEvent",
    "SpeechStoppedEvent",
    "TranscriptionCompletedEvent",
    "ErrorEvent",
    "Event",
    "event_to_dict",
    "SessionConfigError",
    "parse_session_config",
    "SessionState",
    "SessionClosedError",
    "Session",
    # re-exported from protocol.py — reused, not redefined.
    "AudioFormat",
    "TurnDetectionType",
    "AECMode",
    "gen_session_id",
    "gen_event_id",
    "gen_item_id",
]
