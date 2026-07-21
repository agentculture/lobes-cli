"""The base64 event wire — OpenAI-Realtime-shaped JSON, in both directions.

Issue #151 moves the session wire off raw binary WebSocket frames onto
OpenAI-shaped JSON events carrying base64 audio: ``input_audio_buffer.append``
inbound, ``response.audio.delta`` outbound. This module is the *codec* for
that wire — turning one JSON event into raw PCM16 bytes, and raw PCM16 bytes
back into a sequence of JSON events — and nothing else. It does not decide
*when* a delta is sent, does not own session or turn state, and does not
import :mod:`lobes.realtime._session` (a sibling module owns that schema and
is edited in parallel by a different task) — every function here takes and
returns plain ``dict``/``Mapping`` values so it composes with whatever event
members land in the session schema, rather than depending on it.

Stdlib only (``base64``, ``binascii``, ``json``, ``dataclasses``-free plain
dicts, ``enum``) — importable and unit-testable with none of the
``[realtime]`` extra installed (no fastapi, httpx, numpy, scipy, torch
anywhere in this module's import path), mirroring
:mod:`lobes.realtime._segmenter` and :mod:`lobes.realtime._session`'s split:
pure logic here, the FastAPI route shell elsewhere.

Two directions, two entry points
---------------------------------
- **Inbound** (client -> server): a WebSocket text frame arrives as raw JSON.
  :func:`decode_event` turns it into a plain dict (any event ``type`` — this
  function does not care which); :func:`parse_append_event` then pulls the
  base64 ``audio`` field out of an ``input_audio_buffer.append`` payload and
  decodes it to raw PCM16 bytes. Splitting these into two functions keeps the
  generic "is this even valid JSON" concern separate from the
  append-event-specific "does this payload have a usable audio field"
  concern — a caller dispatching on ``payload["type"]`` to other event kinds
  only needs :func:`decode_event`. :func:`decide_inbound_message` (issue
  #151 t4) is the convenience layer on top of both: given one raw WebSocket
  *message* (the ``{"text": ...}``/``{"bytes": ...}`` shape a receive call
  hands back), it decides in one call whether the message is usable audio, an
  ignorable non-append event, or malformed — including a raw BINARY frame,
  which the wire migration removes as an accepted input entirely (a
  deliberate, coordinated break with reachy-mini-cli, tracked in
  reachy-mini-cli#115) rather than a compatibility path this module
  preserves. It exists so ``app.py``'s receive loop is a thin dispatch over
  one pure decision, testable here without the ``[realtime]`` extra —
  mirroring how issue #151 t7 moved ``tts_client.py``'s real concurrency
  decision into stdlib-only ``_settings.py`` for the same reason.
- **Outbound** (server -> client): :func:`serialize_audio_delta` builds one
  ``response.audio.delta`` event from a chunk of PCM16 bytes;
  :func:`iter_audio_deltas` is the convenience wrapper that splits a
  *complete* PCM buffer (the TTS sidecar has no streaming route — see
  :mod:`lobes.realtime.tts_client`'s own docstring — so the bridge always
  holds the whole reply before the first out-frame) into a lazy sequence of
  such events, so a caller can start sending frame 1 before frame N is even
  produced.

Errors never escape — :class:`WireFormatError`
-------------------------------------------------
Every malformed-input path here raises :class:`WireFormatError`, a
``ValueError`` subclass carrying a documented :class:`WireErrorCode` — never
a bare ``json.JSONDecodeError``/``binascii.Error``/``KeyError`` escaping to
the caller, and never a silently-wrong result (e.g. treating a missing
``audio`` field as empty audio). This mirrors how :mod:`lobes.realtime._session`'s
``SessionConfigError`` carries a documented ``ErrorCode`` — the same shape,
defined **locally** in this module rather than imported, per this module's
own no-``_session``-dependency rule above. A client-controlled wire event is
adversarial input by construction (a hostile or buggy client, a version
mismatch, a truncated frame); this module's job is to turn every one of
those into a value the caller can turn into a named ``error`` event, not an
exception that unwinds the WebSocket handler.

Delta chunk sizing
-------------------
:data:`DEFAULT_DELTA_CHUNK_BYTES` chunks outbound audio into
:data:`DELTA_CHUNK_MS` (100ms) frames at :data:`TTS_SAMPLE_RATE` — small
enough that the first frame of a reply reaches the client quickly and an
interruption only ever discards a bounded remainder (see the #151 spec's
truncation-on-barge-in requirement, wired by a later task), large enough that
JSON/base64 framing overhead (33% larger than raw PCM, plus one event
envelope) does not dominate. Every chunk boundary keeps whole PCM16 samples
(:data:`~lobes.realtime.protocol.BYTES_PER_SAMPLE`-aligned) so a chunk is
never split mid-sample; :func:`iter_audio_deltas` validates any
caller-supplied ``chunk_bytes`` the same way.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import NamedTuple

from .protocol import BYTES_PER_SAMPLE, TTS_SAMPLE_RATE, gen_event_id

# ---------------------------------------------------------------------------
# Event type names (OpenAI-Realtime-flavoured, audio-path subset only — see
# the #151 spec's "full OpenAI Realtime API parity" non-goal).
# ---------------------------------------------------------------------------

APPEND_EVENT_TYPE = "input_audio_buffer.append"
AUDIO_DELTA_EVENT_TYPE = "response.audio.delta"

# ---------------------------------------------------------------------------
# Delta chunk sizing — see the module docstring's "Delta chunk sizing".
# ---------------------------------------------------------------------------

DELTA_CHUNK_MS = 100
# 24000 Hz * 0.1s * 2 bytes/sample = 4800 bytes. Derived from protocol.py's
# constants, not an independently-chosen magic number.
DEFAULT_DELTA_CHUNK_BYTES = int(TTS_SAMPLE_RATE * DELTA_CHUNK_MS / 1000) * BYTES_PER_SAMPLE


# ---------------------------------------------------------------------------
# Named errors — mirrors _session.py's SessionConfigError/ErrorCode pairing,
# defined locally per this module's no-_session-import rule.
# ---------------------------------------------------------------------------


class WireErrorCode(str, Enum):
    """Documented, named wire-codec failure modes — never a bare exception.

    - ``INVALID_JSON`` — :func:`decode_event`'s raw text/bytes did not parse
      as JSON, or parsed to something other than a JSON object (e.g. a bare
      array, string, or number at the top level).
    - ``INVALID_APPEND_EVENT`` — :func:`parse_append_event`'s payload was
      missing an ``audio`` field, the field was not a string, or the string
      failed base64 decoding. One code covers all three — like
      ``_session.py``'s single ``INVALID_SESSION_CONFIG`` code covering every
      rejected config shape — with the specific reason carried in the
      exception's message text, not fragmented across many codes.
    - ``UNSUPPORTED_FRAME_TYPE`` — :func:`decide_inbound_message`'s message
      was not a JSON text frame: a raw BINARY WebSocket frame (accepted as
      audio before issue #151's wire migration, rejected outright now — see
      the module docstring), or a message carrying neither a ``"text"`` nor
      a ``"bytes"`` payload at all (defensive; a well-formed ASGI data
      message always carries one or the other).
    """

    INVALID_JSON = "invalid_json"
    INVALID_APPEND_EVENT = "invalid_append_event"
    UNSUPPORTED_FRAME_TYPE = "unsupported_frame_type"


class WireFormatError(ValueError):
    """A client-supplied wire event was malformed.

    Carries a documented :class:`WireErrorCode` so a caller can build a
    named ``error`` event from it (the session-schema module, not this one,
    owns that event's exact shape) instead of ever letting a raw
    ``json.JSONDecodeError``/``binascii.Error``/``KeyError`` escape.
    """

    def __init__(self, code: WireErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Inbound: raw text -> dict -> PCM16 bytes.
# ---------------------------------------------------------------------------


def decode_event(raw: str | bytes) -> dict[str, object]:
    """Parse one raw WebSocket text frame as a JSON object.

    Generic across every event ``type`` — this function does not inspect or
    care what kind of event it is; a caller dispatches on
    ``payload["type"]`` afterward. Raises :class:`WireFormatError` with
    :attr:`WireErrorCode.INVALID_JSON`, never a bare exception, when *raw*
    is not valid JSON or its top-level value is not a JSON object (e.g. a
    bare array/string/number/``null``/``true``).
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        raise WireFormatError(WireErrorCode.INVALID_JSON, f"malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise WireFormatError(
            WireErrorCode.INVALID_JSON,
            f"top-level JSON value must be an object, got {type(parsed).__name__}",
        )
    return parsed


def parse_append_event(payload: Mapping[str, object]) -> bytes:
    """Pull the base64 ``audio`` field out of an append event payload.

    Returns the exact decoded PCM16 bytes. Does not inspect ``payload["type"]``
    — a caller that has already routed on the event type (via
    :func:`decode_event`'s return value) calls this only once it knows the
    payload is an ``input_audio_buffer.append`` event; this function's own
    contract is purely "does this mapping have a usable base64 audio field."

    Raises :class:`WireFormatError` with :attr:`WireErrorCode.INVALID_APPEND_EVENT`,
    never a bare exception, when the ``audio`` field is missing, is not a
    string, or fails base64 decoding (invalid characters or padding). An
    empty string is valid (decodes to ``b""``) — zero bytes of audio is not
    malformed, just empty.
    """
    audio = payload.get("audio")
    if not isinstance(audio, str):
        raise WireFormatError(
            WireErrorCode.INVALID_APPEND_EVENT,
            f"{APPEND_EVENT_TYPE} requires a base64 string 'audio' field, got {audio!r}",
        )
    try:
        return base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise WireFormatError(
            WireErrorCode.INVALID_APPEND_EVENT,
            f"'audio' field is not valid base64: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Inbound message classification — issue #151 t4.
#
# One decision point for app.py's WebSocket receive loop, extracted here so
# it is exhaustively unit-testable without the [realtime] extra (app.py
# itself is fastapi/torch-only and pragma-no-cover/coverage-omitted — see
# its own module docstring). This is the migration's single point of truth
# for "no code path accepts a raw binary audio frame anymore": a message
# carrying ``bytes`` is classified as a named error, never as audio.
# ---------------------------------------------------------------------------


class InboundKind(str, Enum):
    """What :func:`decide_inbound_message` decided about one received
    WebSocket message.

    - ``AUDIO`` — a valid ``input_audio_buffer.append`` event; the
      returned :class:`InboundDecision`'s ``audio`` field carries the
      decoded PCM16 bytes (possibly empty — see :func:`parse_append_event`).
    - ``IGNORED`` — well-formed JSON whose ``"type"`` is not
      :data:`APPEND_EVENT_TYPE` (e.g. a future ``response.create`` — out of
      scope for the #151 t4 input-framing migration, wired by a later task).
      Not an error: valid input, just not audio this route acts on yet —
      mirrors how the pre-migration route silently ignored every
      text/control frame, just now discriminated by JSON event ``type``
      instead of by WebSocket frame kind.
    - ``ERROR`` — malformed input; the decision's ``error`` field carries
      the :class:`WireFormatError` a caller turns into a named ``error``
      session event. Covers everything :func:`decode_event`/
      :func:`parse_append_event` already reject, PLUS a raw BINARY
      WebSocket frame (:attr:`WireErrorCode.UNSUPPORTED_FRAME_TYPE`) —
      accepted as audio before issue #151, rejected outright now: the wire
      is base64 JSON text in both directions, a deliberate, coordinated
      break with reachy-mini-cli (reachy-mini-cli#115), not a compatibility
      path this module preserves.
    """

    AUDIO = "audio"
    IGNORED = "ignored"
    ERROR = "error"


class InboundDecision(NamedTuple):
    """The result of classifying one received WebSocket message.

    Exactly one of ``audio``/``error`` is populated, matching ``kind`` — see
    :class:`InboundKind` for what each combination means. A plain
    ``NamedTuple`` (not a dataclass): this is an internal decision value
    consumed only by the route layer, never itself serialized to the wire —
    unlike this module's actual wire *events*, which stay plain dicts (see
    the module docstring) precisely because those ARE serialized.
    """

    kind: InboundKind
    audio: bytes | None = None
    error: WireFormatError | None = None


def decide_inbound_message(message: Mapping[str, object]) -> InboundDecision:
    """Classify one received WebSocket message: audio, ignorable, or malformed.

    *message* is the plain mapping a WebSocket receive call hands back —
    this function only reads the ``"text"``/``"bytes"`` keys (the ASGI/
    Starlette receive-message shape: exactly one of the two is populated for
    a data frame), so it composes with any WebSocket layer without importing
    one. A caller (``app.py``'s ``_pump_session``) is expected to have
    already handled ``message["type"] == "websocket.disconnect"`` itself —
    this function only classifies a frame that carries data.

    Decision order:

    1. A ``"bytes"`` key present (a raw BINARY frame) -> :attr:`InboundKind.ERROR`
       with :attr:`WireErrorCode.UNSUPPORTED_FRAME_TYPE` — removed by issue
       #151; the wire is base64 JSON text now, in both directions.
    2. Neither ``"bytes"`` nor ``"text"`` present (no data at all) -> the
       same :attr:`WireErrorCode.UNSUPPORTED_FRAME_TYPE` error — defensive;
       a well-formed ASGI data message always carries one or the other.
    3. ``"text"`` present but not valid JSON, or not a JSON object ->
       :attr:`InboundKind.ERROR` with whatever :func:`decode_event` raised
       (:attr:`WireErrorCode.INVALID_JSON`).
    4. Valid JSON whose ``"type"`` is not :data:`APPEND_EVENT_TYPE` ->
       :attr:`InboundKind.IGNORED`.
    5. An ``input_audio_buffer.append`` event with a missing/non-string/
       non-base64 ``"audio"`` field -> :attr:`InboundKind.ERROR` with
       whatever :func:`parse_append_event` raised
       (:attr:`WireErrorCode.INVALID_APPEND_EVENT`).
    6. A valid append event -> :attr:`InboundKind.AUDIO` carrying the exact
       decoded PCM16 bytes.

    Never raises :class:`WireFormatError` itself — every rejection path
    above is returned as an ``ERROR`` decision, matching this module's
    "errors never escape" contract for adversarial wire input.
    """
    if message.get("bytes") is not None:
        return InboundDecision(
            kind=InboundKind.ERROR,
            error=WireFormatError(
                WireErrorCode.UNSUPPORTED_FRAME_TYPE,
                "binary WebSocket frames are no longer accepted; send "
                f"{APPEND_EVENT_TYPE!r} JSON text events with a base64 "
                "'audio' field instead",
            ),
        )
    raw_text = message.get("text")
    if raw_text is None:
        return InboundDecision(
            kind=InboundKind.ERROR,
            error=WireFormatError(
                WireErrorCode.UNSUPPORTED_FRAME_TYPE,
                "WebSocket message carried neither 'text' nor 'bytes'",
            ),
        )
    try:
        payload = decode_event(raw_text)
    except WireFormatError as exc:
        return InboundDecision(kind=InboundKind.ERROR, error=exc)
    if payload.get("type") != APPEND_EVENT_TYPE:
        return InboundDecision(kind=InboundKind.IGNORED)
    try:
        audio = parse_append_event(payload)
    except WireFormatError as exc:
        return InboundDecision(kind=InboundKind.ERROR, error=exc)
    return InboundDecision(kind=InboundKind.AUDIO, audio=audio)


# ---------------------------------------------------------------------------
# Outbound: PCM16 bytes -> dict event(s).
# ---------------------------------------------------------------------------


def serialize_audio_delta(
    pcm: bytes,
    *,
    response_id: str,
    item_id: str,
    event_id: str | None = None,
    output_index: int = 0,
    content_index: int = 0,
) -> dict[str, object]:
    """Build one ``response.audio.delta`` event carrying *pcm* as base64.

    ``response_id``/``item_id`` identify which in-progress response and
    conversation item this chunk belongs to — owned and threaded by the
    caller (the turn/floor machine, a later task), not generated here: every
    chunk of one reply must share the same ids, so this function never
    invents them. ``event_id`` defaults to a fresh
    :func:`~lobes.realtime.protocol.gen_event_id` when not supplied (every
    event needs a unique id; most callers have no reason to pick one), but
    accepts an explicit value for deterministic tests or replay.
    """
    return {
        "type": AUDIO_DELTA_EVENT_TYPE,
        "event_id": event_id or gen_event_id(),
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "delta": base64.b64encode(pcm).decode("ascii"),
    }


def iter_audio_deltas(
    pcm: bytes,
    chunk_bytes: int = DEFAULT_DELTA_CHUNK_BYTES,
    *,
    response_id: str,
    item_id: str,
    output_index: int = 0,
    content_index: int = 0,
) -> Iterator[dict[str, object]]:
    """Split a complete PCM16 buffer into sequential ``response.audio.delta`` events.

    The TTS sidecar is full-read (no streaming route), so the caller holds
    the whole reply and hands it here in one call; this function is what
    turns that single buffer into the sequence of frames actually sent over
    the WebSocket, in order, each :data:`chunk_bytes`-sized except a
    possibly-shorter final chunk. An empty *pcm* yields no events at all.

    A lazy generator, not a pre-built list, so a caller can start sending
    frame 1 before frame N is even produced.

    Raises ``ValueError`` (a programmer-input problem, not adversarial wire
    data — hence not :class:`WireFormatError`) if *chunk_bytes* is not a
    positive, whole number of PCM16 samples
    (:data:`~lobes.realtime.protocol.BYTES_PER_SAMPLE`-aligned): a
    non-aligned chunk size would split one 16-bit sample's bytes across two
    frames.
    """
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    if chunk_bytes % BYTES_PER_SAMPLE != 0:
        raise ValueError(
            f"chunk_bytes must be a whole number of {BYTES_PER_SAMPLE}-byte "
            f"PCM16 samples, got {chunk_bytes}"
        )
    for start in range(0, len(pcm), chunk_bytes):
        yield serialize_audio_delta(
            pcm[start : start + chunk_bytes],
            response_id=response_id,
            item_id=item_id,
            output_index=output_index,
            content_index=content_index,
        )


__all__ = [
    "APPEND_EVENT_TYPE",
    "AUDIO_DELTA_EVENT_TYPE",
    "DELTA_CHUNK_MS",
    "DEFAULT_DELTA_CHUNK_BYTES",
    "WireErrorCode",
    "WireFormatError",
    "decode_event",
    "parse_append_event",
    "InboundKind",
    "InboundDecision",
    "decide_inbound_message",
    "serialize_audio_delta",
    "iter_audio_deltas",
]
