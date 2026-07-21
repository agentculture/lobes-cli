"""Server-side VAD segmentation — a pure state machine over 16 kHz PCM16 chunks.

This module turns a raw PCM16 mono, 16 kHz audio stream into ``speech_started``
/ ``speech_stopped`` turn-boundary events — OpenAI Realtime's ``server_vad``
turn-detection strategy (see
:class:`lobes.realtime.protocol.TurnDetectionType`). It is a **pure state
machine**: no I/O, no sockets, no FastAPI, no torch. The Silero VAD model
itself is never imported here — the caller injects it as a plain callable,
``vad_probability: Callable[[bytes], float]``, taking exactly one
:data:`CHUNK_BYTES`-sized (512-sample / 32 ms) PCM16 chunk and returning a
speech probability in ``[0, 1]``. Wiring the *real* Silero model behind that
callable is a later task (#149 t6) — tests here drive a scripted fake.

Stdlib only (``collections``, ``dataclasses``, ``typing``) — importable and
unit-testable with neither torch nor the ``[realtime]`` extra installed, and
without a GPU (mirrors :mod:`lobes.realtime.audio_facade`'s split: pure logic
here, the FastAPI shell elsewhere). Reuses
:mod:`lobes.realtime.protocol`'s audio constants (``VAD_CHUNK_SAMPLES``,
``VAD_CHUNK_MS``, ``BYTES_PER_SAMPLE``) rather than redefining them.

Framing
-------
Bytes handed to :meth:`Segmenter.feed` are buffered until a full
:data:`CHUNK_BYTES` (1024-byte / 512-sample) frame is available; a short
trailing remainder is held over to the next call and is never handed to
``vad_probability``. Callers may feed any number of bytes per call — a
fraction of a sample, one chunk, or many chunks at once — the segmenter
reassembles the 512-sample framing regardless of how the caller happened to
chunk its reads (or writes).

Boundary events
----------------
:meth:`Segmenter.feed` returns a list of zero or more events, each either a
:class:`SpeechStarted` or :class:`SpeechStopped`:

- ``SpeechStarted`` fires the instant a chunk's probability crosses
  ``vad_threshold`` while idle. Its ``audio`` is the **padded onset**: up to
  ``vad_prefix_padding_ms`` of the immediately preceding non-speech audio
  (rounded down to whole 32 ms chunks — e.g. 300ms -> 9 chunks -> ~288ms)
  followed by the chunk that crossed the threshold, so a consumer never
  loses the syllable that preceded detection.
- ``SpeechStopped`` fires once ``vad_silence_ms`` of continuous non-speech
  has been confirmed (``reason="silence"``), OR once the in-progress turn's
  audio reaches ``max_turn_ms`` first (``reason="max_turn"`` — see "Max-turn
  cap" below). Its ``audio`` is the complete turn: the padded onset through
  the chunk that confirmed the stop. The confirming silence itself is kept,
  not trimmed — this stays a single forward pass with no second
  buffer-trim step.

Max-turn cap: force-commit, not error
--------------------------------------
A stream that never falls silent (a stuck mic, an uninterrupted monologue)
must not grow a turn's buffered audio without bound. ``max_turn_ms``
(default 30000ms; the intended env key — wired in a later task, not yet a
:mod:`lobes.realtime._settings` field — is ``VAD_MAX_TURN_MS``) bounds it:
when a turn's accumulated audio reaches ``max_turn_ms``, the segmenter
**force-commits** it — emits ``SpeechStopped(reason="max_turn")`` with
whatever audio has accumulated — and immediately starts listening for the
next turn from the very next chunk (with empty padding, since there is no
preceding silence to pad from — the stream is still mid-speech). The
segmenter itself never raises on this path. A caller that wants
error-like semantics can treat ``reason="max_turn"`` as one by inspecting
the field; this module's own choice is to always yield a usable, bounded
turn rather than lose audio to an exception.

VAD failures propagate — this module does not translate them
---------------------------------------------------------------
If ``vad_probability`` raises, :meth:`Segmenter.feed` lets the exception
propagate unmodified. This module has no opinion on what "VAD unavailable"
means at the session level; translating a raising/unavailable VAD into a
named, silence-distinct session error event is the session engine's job
(#149 t2), which wraps calls into this module accordingly.

Per-session isolation
----------------------
All state lives on the :class:`Segmenter` instance (an input buffer, a
pre-roll ring buffer, and the in-progress turn's accumulated chunks) — there
is no module-level mutable state anywhere in this file. Two instances fed
interleaved, independently-scripted streams never observe each other's
audio or emit each other's events.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from .protocol import BYTES_PER_SAMPLE, VAD_CHUNK_MS, VAD_CHUNK_SAMPLES

# 512 samples * 2 bytes/sample = 1024 bytes — derived from protocol.py's
# declared Silero framing, never redefined independently.
CHUNK_BYTES = VAD_CHUNK_SAMPLES * BYTES_PER_SAMPLE

# Mirror lobes.realtime._settings.build_settings()'s VAD defaults
# (vad_threshold, vad_silence_ms, vad_prefix_padding_ms). Duplicated here as
# plain literals rather than importing `_settings.settings` so this module
# stays free of any dependency on env-derived state at import time — the two
# modules simply agree on the same numbers; a caller (the session engine,
# #149 t2) is expected to pass the live Settings values through explicitly.
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_SILENCE_MS = 600
DEFAULT_VAD_PREFIX_PADDING_MS = 300

# Not yet a _settings.py field (issue #149 t1) — a later task wires the env
# name VAD_MAX_TURN_MS through to this constructor parameter.
DEFAULT_MAX_TURN_MS = 30_000

VadProbability = Callable[[bytes], float]


@dataclass(frozen=True)
class SpeechStarted:
    """Turn boundary: speech onset detected, at the padded onset.

    ``at_ms`` is elapsed audio-stream time (not wall-clock) at the chunk that
    triggered onset — a pure function of how many 32ms chunks have been
    processed so far, kept deterministic and independent of real time so
    offline tests never need to mock the clock.

    ``audio`` is up to ``vad_prefix_padding_ms`` of pre-roll (the
    immediately preceding non-speech chunks) followed by the chunk whose
    probability crossed ``vad_threshold``.
    """

    at_ms: int
    audio: bytes


@dataclass(frozen=True)
class SpeechStopped:
    """Turn boundary: the turn ended; ``audio`` is the complete turn to hand off.

    ``reason`` is ``"silence"`` (``vad_silence_ms`` of continuous non-speech
    confirmed the stop), ``"max_turn"`` (the turn's audio reached
    ``max_turn_ms`` before silence did — a force-commit, never an error; see
    the module docstring), or whatever string a caller passes to
    :meth:`Segmenter.flush` (e.g. ``"closed"`` on session teardown).
    """

    at_ms: int
    audio: bytes
    reason: str


Event = SpeechStarted | SpeechStopped


class Segmenter:
    """Per-session ``server_vad`` state machine over 16 kHz/32 ms PCM16 chunks.

    One instance per realtime session — construct a fresh :class:`Segmenter`
    per WebSocket connection. All state (the pending-bytes buffer, the
    pre-roll ring buffer, the in-progress turn) is instance-local; nothing
    is shared across instances, so concurrent sessions never corrupt each
    other's boundaries (see the module docstring's "Per-session isolation").

    Parameters mirror :mod:`lobes.realtime._settings`'s VAD/turn-detection
    fields — pass the live :class:`~lobes.realtime._settings.Settings`
    values through explicitly; this module never reads them itself.
    """

    def __init__(
        self,
        vad_probability: VadProbability,
        *,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
        vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
        vad_prefix_padding_ms: int = DEFAULT_VAD_PREFIX_PADDING_MS,
        max_turn_ms: int = DEFAULT_MAX_TURN_MS,
    ) -> None:
        self._vad_probability = vad_probability
        self.vad_threshold = vad_threshold
        self.vad_silence_ms = vad_silence_ms
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.max_turn_ms = max_turn_ms

        # Padding rounds DOWN to whole 32ms chunks (documented above).
        padding_chunks = max(0, vad_prefix_padding_ms // VAD_CHUNK_MS)
        self._preroll: "deque[bytes]" = deque(maxlen=padding_chunks)
        self._pending = bytearray()  # bytes not yet forming a full chunk
        self._speaking = False
        self._turn_chunks: list[bytes] = []
        self._silence_run_ms = 0
        self._turn_ms = 0
        self._stream_ms = 0  # elapsed audio-stream time; see SpeechStarted.at_ms

    @property
    def speaking(self) -> bool:
        """Whether a turn is currently in progress (for callers/tests)."""
        return self._speaking

    def feed(self, pcm: bytes) -> list[Event]:
        """Feed raw PCM16 mono LE bytes; returns zero or more boundary events.

        Any number of bytes may be passed per call. Bytes are buffered until
        a full :data:`CHUNK_BYTES` frame is available; a short trailing
        remainder is held for the next call and is never handed to
        ``vad_probability`` (never feed a partial frame to the VAD).

        If the injected ``vad_probability`` callable raises, the exception
        propagates unmodified — see the module docstring.
        """
        events: list[Event] = []
        self._pending.extend(pcm)
        while len(self._pending) >= CHUNK_BYTES:
            chunk = bytes(self._pending[:CHUNK_BYTES])
            del self._pending[:CHUNK_BYTES]
            event = self._process_chunk(chunk)
            if event is not None:
                events.append(event)
        return events

    def flush(self, reason: str = "closed") -> Event | None:
        """Force-commit an in-progress turn (e.g., on session teardown).

        Returns the resulting :class:`SpeechStopped` (carrying ``reason``)
        if a turn was in progress, else ``None`` when idle. Any buffered
        partial-chunk bytes (fewer than :data:`CHUNK_BYTES`) are discarded —
        they were never handed to ``vad_probability`` and never will be.
        """
        if not self._speaking:
            return None
        return self._commit(reason)

    def _process_chunk(self, chunk: bytes) -> Event | None:
        self._stream_ms += VAD_CHUNK_MS
        probability = self._vad_probability(chunk)
        is_speech = probability >= self.vad_threshold

        if not self._speaking:
            if is_speech:
                turn_chunks = list(self._preroll) + [chunk]
                self._preroll.clear()
                self._turn_chunks = turn_chunks
                self._turn_ms = len(turn_chunks) * VAD_CHUNK_MS
                self._speaking = True
                self._silence_run_ms = 0
                return SpeechStarted(at_ms=self._stream_ms, audio=b"".join(turn_chunks))
            self._preroll.append(chunk)
            return None

        self._turn_chunks.append(chunk)
        self._turn_ms += VAD_CHUNK_MS
        if is_speech:
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += VAD_CHUNK_MS

        if self._silence_run_ms >= self.vad_silence_ms:
            return self._commit("silence")
        if self._turn_ms >= self.max_turn_ms:
            return self._commit("max_turn")
        return None

    def _commit(self, reason: str) -> SpeechStopped:
        event = SpeechStopped(
            at_ms=self._stream_ms,
            audio=b"".join(self._turn_chunks),
            reason=reason,
        )
        self._speaking = False
        self._turn_chunks = []
        self._silence_run_ms = 0
        self._turn_ms = 0
        self._preroll.clear()
        return event
