"""The realtime container's FastAPI app — OpenAI ``/v1/audio/*`` surface plus
the ``/v1/realtime`` WebSocket session route.

Runs ONLY in the ``realtime`` fleet container (the ``[realtime]`` extra: fastapi,
uvicorn, httpx, numpy, scipy, silero-vad, torch). The offline dev/CI env has
none of those, so this module is never imported by the unit suite — its logic
lives in stdlib-only modules that ARE tested directly:
:mod:`lobes.realtime.audio_facade` (codec + request parsing),
:mod:`lobes.realtime.tts_client`, :mod:`lobes.realtime._segmenter` (the
server_vad state machine), :mod:`lobes.realtime._session` (event schema +
session bookkeeping), :mod:`lobes.realtime._pcm` (resample-decision +
frame-alignment arithmetic for ``/v1/realtime``), :mod:`lobes.realtime._wire`
(the base64 event codec), :mod:`lobes.realtime._floor` (who holds the
conversational floor), :mod:`lobes.realtime._turn` (the generate call's
shape), and :mod:`lobes.realtime._conversation` (the bridge that wires those
five into one turn). The routes here are thin shells (``# pragma: no cover``)
that wire those modules to a real WebSocket, real Silero VAD, real scipy
resampling, and real HTTP; the live stack is exercised by the curl/WS smoke
tests documented in ``docs/realtime-pipeline.md``.

What this file is allowed to own (issue #151 t6): sockets, threads, HTTP,
asyncio tasks, and time. Not decisions. Every error code, every default,
every state transition and every piece of turn bookkeeping lives in the
stdlib modules above, so that the parts of voice-to-voice that can be wrong
are the parts CI can prove right.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import anyio
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from scipy.signal import resample as scipy_resample

from ._conversation import WATCHDOG_INTERVAL_MS, ConversationBridge, resolve_voice_model
from ._pcm import needs_resample, resampled_frame_count, take_aligned_samples
from ._segmenter import Segmenter, SpeechStarted, SpeechStopped
from ._session import Session, SessionConfigError, event_to_dict, parse_session_config
from ._settings import VOICE_LANE, settings
from ._wire import DEFAULT_DELTA_CHUNK_BYTES, InboundKind, decide_inbound_message
from .audio_facade import (
    SpeechRequestError,
    aggregate_audio_ready,
    parse_speech_request,
    pcm_to_container,
)
from .protocol import BYTES_PER_SAMPLE, VAD_SAMPLE_RATE, gen_session_id
from .tts_client import synthesize

log = logging.getLogger(__name__)

app = FastAPI(title="lobes realtime", version="1")


@app.get("/health")  # pragma: no cover - exercised live, not in the offline suite
async def health() -> dict:
    return {"status": "ok", "service": "model-gear-realtime"}


@app.get("/v1/health/ready")  # pragma: no cover - exercised live, not in the offline suite
async def ready() -> Response:
    """Aggregate readiness over the TTS (Chatterbox) + STT (Parakeet) backends.

    The gateway probes this so it advertises stt/tts as consumable in
    GET /capabilities only when an audio round-trip would actually succeed
    (issue #89). 200 iff both backends report ready; else 503.
    """
    tts_ok = await _probe_backend_ready(settings.tts_url)
    stt_ok = await _probe_backend_ready(settings.stt_url)
    code, body = aggregate_audio_ready(tts_ok, stt_ok)
    return JSONResponse(status_code=code, content=body)


async def _probe_backend_ready(base_url: str) -> bool:  # pragma: no cover
    url = base_url.rstrip("/") + "/v1/health/ready"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(url)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


@app.post("/v1/audio/speech")  # pragma: no cover
async def speech(request: Request) -> Response:
    """OpenAI text-to-speech → Chatterbox. Returns a wav/pcm audio body."""
    try:
        body = await request.json()
    except ValueError:
        return _error(400, "request body must be valid JSON")
    try:
        params = parse_speech_request(body)
    except SpeechRequestError as exc:
        return _error(400, str(exc))

    pcm = await synthesize(
        params.input,
        voice=params.voice or settings.default_voice,
        speed=params.speed,
        tts_url=settings.tts_url,
    )
    if not pcm:
        return _error(502, "TTS backend returned no audio")
    data, media_type = pcm_to_container(pcm, params.response_format)
    return Response(content=data, media_type=media_type)


@app.post("/v1/audio/transcriptions")  # pragma: no cover
async def transcriptions(
    file: UploadFile = File(...),
    language: str = Form("en"),
    model: str = Form(None),
) -> Response:
    """OpenAI speech-to-text → forwards the upload to Parakeet (already OpenAI-shaped)."""
    content = await file.read()
    url = settings.stt_url.rstrip("/") + "/v1/audio/transcriptions"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                data={"language": language},
                files={
                    "file": (
                        file.filename or "audio.wav",
                        content,
                        file.content_type or "audio/wav",
                    )
                },
            )
    except httpx.HTTPError as exc:
        return _error(502, f"STT backend unreachable: {exc}")
    try:
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except ValueError:
        return _error(502, "STT backend returned a non-JSON response")


def _error(status: int, message: str) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=status, content={"error": {"message": message}})


# ---------------------------------------------------------------------------
# /v1/realtime — one server_vad session per WebSocket (issue #149 t6), plus
# the opt-in voice-to-voice conversation surface (issue #151 t6).
#
# All decisions live in stdlib modules tested offline: _session.py owns the
# event schema/config parsing/teardown, _segmenter.py owns VAD segmentation,
# _pcm.py owns the resample-needed decision and frame alignment, _wire.py owns
# the base64 codec and delta sizing, _floor.py owns the floor state machine,
# _turn.py owns the generate call's shape, and _conversation.py owns the
# translation between all of them. This block is the thin, pragma-no-cover
# glue that wires them to a real WebSocket, a real Silero model, real scipy
# resampling, and real HTTP calls.
# ---------------------------------------------------------------------------

_STT_FORWARD_TIMEOUT = 60.0
# Mirrors _STT_FORWARD_TIMEOUT, and is threaded into the floor's own generate
# deadline below so the two agree by construction rather than by coincidence.
_GENERATE_FORWARD_TIMEOUT = 60.0


class _STTForwardError(RuntimeError):
    """A committed turn's forward to Parakeet failed — never a silent drop.

    Raised by :func:`_forward_turn_to_stt` for every failure mode (unreachable
    backend, non-2xx, non-JSON, or a JSON body missing ``text``); the route
    catches it and turns it into the session's named
    ``error``/``stt_forward_failed`` event via :meth:`Session.fail_transcription`.
    """


async def _forward_turn_to_stt(pcm16k: bytes) -> str:  # pragma: no cover
    """Forward one committed turn's 16 kHz PCM16 audio to Parakeet.

    Wraps the raw PCM in a WAV container (:func:`audio_facade.pcm_to_container`
    — never a hand-rolled header) and POSTs it exactly like the batch
    ``/v1/audio/transcriptions`` route above, so a WebSocket turn and a batch
    upload hit the identical backend contract. Raises :class:`_STTForwardError`
    on any failure instead of returning a sentinel, so a caller cannot
    forget to check for one.
    """
    wav_bytes, _media_type = pcm_to_container(pcm16k, "wav", rate=VAD_SAMPLE_RATE)
    url = settings.stt_url.rstrip("/") + "/v1/audio/transcriptions"
    try:
        async with httpx.AsyncClient(timeout=_STT_FORWARD_TIMEOUT) as client:
            resp = await client.post(
                url,
                data={"language": "en"},
                files={"file": ("turn.wav", wav_bytes, "audio/wav")},
            )
    except httpx.HTTPError as exc:
        raise _STTForwardError(f"STT backend unreachable: {exc}") from exc
    if resp.status_code != 200:
        raise _STTForwardError(f"STT backend returned HTTP {resp.status_code}")
    try:
        body = resp.json()
    except ValueError as exc:
        raise _STTForwardError("STT backend returned a non-JSON response") from exc
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str):
        raise _STTForwardError("STT backend response missing a 'text' field")
    return text


def _load_vad_probability():  # pragma: no cover
    """Load a FRESH Silero VAD model for one session and wrap it as the
    ``Callable[[bytes], float]`` :class:`~lobes.realtime._segmenter.Segmenter`
    expects.

    Loaded per-session, never shared, and never module-level: Silero's loaded
    model object carries its own recurrent (hidden-state) attributes that
    every inference call mutates in place — sharing one instance across
    concurrent sessions would let one session's chunks corrupt another's VAD
    state (the concurrency hazard docs/specs/2026-07-21-...#149.md's s14 flags).
    A fresh load is cheap: the silero-vad wheel bundles its weights
    (``silero_vad/data/*.jit``/``.onnx``) — this makes no network request, so
    a session opens with no egress at VAD-arm time either.

    Imports torch/silero_vad lazily (inside this function, not at module
    import time) so importing ``app.py`` itself never pays torch's import/init
    cost until a session actually needs VAD; the caller runs this in a worker
    thread (``anyio.to_thread.run_sync``) so the event loop is never blocked
    by the load.
    """
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad()

    def _probability(chunk: bytes) -> float:
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
        tensor = torch.from_numpy(samples.copy())
        with torch.no_grad():
            return float(model(tensor, VAD_SAMPLE_RATE).item())

    return _probability


def _resample_to_16k(pcm: bytes, input_rate: int) -> bytes:  # pragma: no cover
    """Resample PCM16 mono LE *pcm* from *input_rate* to 16 kHz (Silero/Parakeet's
    native rate).

    A genuine no-op passthrough when *input_rate* is already 16000 Hz — see
    :func:`lobes.realtime._pcm.needs_resample` — never a resample-to-itself
    round trip through floats. Otherwise uses scipy (already in the
    ``[realtime]`` extra), sized via
    :func:`lobes.realtime._pcm.resampled_frame_count` so the arithmetic itself
    stays covered by the offline suite.
    """
    if not needs_resample(input_rate, VAD_SAMPLE_RATE):
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return b""
    out_len = resampled_frame_count(samples.size, input_rate, VAD_SAMPLE_RATE)
    resampled = scipy_resample(samples, out_len)
    return np.clip(resampled, -32768, 32767).astype("<i2").tobytes()


@app.websocket("/v1/realtime")  # pragma: no cover - exercised live, not in the offline suite
async def realtime(websocket: WebSocket) -> None:
    """One ``server_vad`` session per WebSocket connection.

    Wire format
    -----------
    - **Session config** is passed as QUERY PARAMETERS on the connect URL —
      the same keys :func:`lobes.realtime._session.parse_session_config`
      accepts (``input_audio_format``, ``input_sample_rate``,
      ``input_channels``, ``turn_detection``, ``aec_mode``), e.g.
      ``wss://.../v1/realtime?input_sample_rate=16000``. An invalid config is
      rejected with the named ``error``/``invalid_session_config`` event and
      the socket is closed before any audio is accepted — no session is
      allocated for a rejected config.
    - **Input audio**: streamed as ``input_audio_buffer.append`` JSON TEXT
      events carrying base64-encoded PCM16 mono little-endian audio (see
      :mod:`lobes.realtime._wire` — :func:`~lobes.realtime._wire.decode_event`
      parses the frame, :func:`~lobes.realtime._wire.parse_append_event`
      decodes the base64 ``audio`` field to exact PCM bytes;
      :func:`~lobes.realtime._wire.decide_inbound_message` is the single
      classification point this route calls into for every received
      message). Any chunking granularity is accepted — an append event's
      audio need not align to a whole sample, let alone a whole VAD chunk;
      the server reassembles it. Raw BINARY WebSocket frames are no longer
      accepted (issue #151 — a deliberate, coordinated break with
      reachy-mini-cli, tracked in reachy-mini-cli#115, not a compatibility
      path this route preserves): one arriving now yields the named
      ``error``/``invalid_wire_event`` event instead of being read as
      audio. A well-formed JSON event whose ``type`` is neither
      ``input_audio_buffer.append`` nor ``response.create`` is silently
      ignored — this route adopts the audio-path event shapes only — never
      mistaken for a boundary/audio frame. A malformed frame (invalid JSON,
      or a missing/invalid base64 ``audio`` field) is never a silent drop
      either: it becomes the named ``error``/``invalid_wire_event`` event,
      whose message names which wire-level reason applied
      (``invalid_json`` / ``invalid_append_event`` / ``unsupported_frame_type``,
      per :class:`~lobes.realtime._wire.WireErrorCode`), and the session
      stays open to keep receiving. ``input_sample_rate`` defaults to
      **24000 Hz** (OpenAI-Realtime-compatible) — **16000 Hz is also
      accepted** (Parakeet/Silero's native rate; the server skips resampling
      entirely in that case). Any other rate is rejected as an invalid
      session config. The server resamples 24 kHz to 16 kHz itself
      (server-side, via scipy) — the client never resamples.
    - **Events** (session/boundary/transcription/response/error) are sent
      back as JSON TEXT frames using the schema in
      :mod:`lobes.realtime._session` (:func:`event_to_dict`). Boundary events
      carry the segmenter's own ``at_ms`` (32ms-quantised audio-stream time,
      a different clock from ``timestamp_ms``) and ``speech_stopped`` also
      carries ``reason`` — ``"silence"`` or the ``"max_turn"`` force-commit.
    - The server sends ``session.created`` immediately after the handshake,
      confirming the negotiated config (including the resolved
      ``input_sample_rate``) — a client that omitted every config query
      param can read the effective defaults off this event.

    Conversation is OPT-IN (issue #151)
    ------------------------------------
    A session is ears-only until the client sends a ``response.create``
    event, and a session that never sends one gets exactly the #149
    transcription-only sequence — that is the contract reachy-mini-cli
    depends on. After the trigger (idempotent; send it once at connect, or
    once per turn OpenAI-style), each committed turn's transcript is
    answered: the generate lane is called through
    ``settings.openai_base_url``, the reply text comes back as
    ``response.created``/``response.text.done``, Chatterbox synthesizes it,
    and the audio streams back on THIS connection as sequential
    ``response.audio.delta`` events (base64 PCM16 at 24 kHz — the same rate
    the client sends, so audio-out never resamples), ending in
    ``response.done``.

    Speaking during playback interrupts it: the undelivered remainder is
    never sent, the generate/TTS calls in flight are both cancelled, and
    ``response.interrupted`` goes out with its truncation marker. Every
    response stage has a deadline; on expiry the floor returns to the caller
    with a named error rather than wedging. All of that is decided in
    :mod:`lobes.realtime._floor`/:mod:`lobes.realtime._conversation` — this
    route only pumps it.

    Turn flow
    ---------
    Audio is fed through :class:`~lobes.realtime._segmenter.Segmenter` — real
    Silero VAD injected as the callable it expects (see
    :func:`_load_vad_probability`) — after resampling to 16 kHz when needed.
    A ``SpeechStarted``/``SpeechStopped`` pair becomes
    ``input_audio_buffer.speech_started``/``...speech_stopped`` on the SAME
    connection. A committed turn (``SpeechStopped``, whether reason
    ``"silence"`` or the ``vad_max_turn_ms`` force-commit) is forwarded to
    ``settings.stt_url`` (Parakeet) and completes as
    ``conversation.item.input_audio_transcription.completed`` on the SAME
    connection — or the named ``error``/``stt_forward_failed`` event on any
    forward failure (unreachable backend, non-2xx, non-JSON). A turn is
    never silently dropped.

    If Silero fails to load — or a later VAD call raises mid-session — the
    session receives the named ``error``/``vad_unavailable`` event (never an
    ordinary boundary event) and the connection is then closed: a consumer
    can always distinguish VAD-down from ordinary silence by event type
    alone (silence alone emits nothing).

    Sessions are ephemeral: disconnecting for ANY reason (idle, mid-speech,
    mid-transcription) tears down the session and releases the segmenter via
    :meth:`Session.teardown`; there is no resume — a reconnecting client
    starts a brand-new session with a brand-new id.
    """
    await websocket.accept()
    session: Session | None = None
    segmenter: Segmenter | None = None
    bridge: ConversationBridge | None = None
    tasks = _SessionTasks()
    try:
        opened = await _open_session(websocket)
        if opened is None:
            return  # the config was rejected; the named error is already sent
        session, input_rate = opened
        cancels = _ResponseCancels()
        bridge = _build_bridge(session, cancels)
        sender = _Sender(websocket, bridge)
        segmenter = await _arm_segmenter(websocket, session)
        if segmenter is None:
            return  # VAD is down; the named error is already sent
        await _pump_session(websocket, bridge, sender, cancels, segmenter, input_rate, tasks)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("unhandled error in /v1/realtime session")
    finally:
        await tasks.cancel_all()
        if bridge is not None:
            # Cancels whatever generate/TTS call was in flight and drops the
            # undelivered audio. Emits nothing: a client that is already gone
            # cannot act on an interruption event.
            bridge.close(reason="client_disconnect")
        if segmenter is not None and segmenter.speaking:
            # Best-effort: the client is already gone by the time we get
            # here on most paths, so the flushed turn is discarded rather
            # than forwarded to STT — nothing could ever deliver its
            # transcript back. teardown() below is what actually matters.
            segmenter.flush(reason="closed")
        if session is not None:
            session.teardown(reason="client_disconnect")


# ---------------------------------------------------------------------------
# Conversation wiring (issue #151 t6): sockets, tasks, HTTP. No decisions.
# ---------------------------------------------------------------------------


def _build_bridge(  # pragma: no cover
    session: Session, cancels: "_ResponseCancels"
) -> ConversationBridge:
    """Construct this session's conversation bridge from the live settings.

    Every value here is either read straight off ``settings`` or derived by a
    stdlib module — ``resolve_voice_model`` applies the voice lane's
    ``multimodal`` default policy (which ``_turn.py`` deliberately refuses to
    hardcode), ``DEFAULT_DELTA_CHUNK_BYTES`` is the wire codec's own delta
    size (the single source of truth for outbound chunking), and the two
    stage deadlines mirror this module's own forward timeouts so the floor
    and the HTTP client can never disagree about how long is too long. The
    TTS deadline is left at the floor's default (60s), which is
    ``tts_client``'s own httpx read timeout.
    """
    return ConversationBridge(
        session,
        cancel_generate=cancels.generate,
        cancel_tts=cancels.tts,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=resolve_voice_model(settings.openai_model),
        barge_in_window_ms=settings.barge_in_window_ms,
        transcribe_timeout_ms=int(_STT_FORWARD_TIMEOUT * 1000),
        generate_timeout_ms=int(_GENERATE_FORWARD_TIMEOUT * 1000),
        chunk_bytes=DEFAULT_DELTA_CHUNK_BYTES,
    )


def _self_cancelled() -> bool:  # pragma: no cover
    """True when THIS task is the one being cancelled, not just its child.

    ``_drive_response`` awaits two child tasks that barge-in deliberately
    cancels, so a :class:`asyncio.CancelledError` there is usually an ordinary,
    already-handled interruption. But the coroutine also runs inside a tracked
    ``_run_response`` task that session teardown cancels — and at teardown BOTH
    are true at once, because the floor's ``close()`` fires the same cancel
    hooks barge-in does.

    Without this distinction the teardown case silently loses: the driver would
    catch its own cancellation, return normally, and carry on to flush a socket
    that is going away — and the task would report itself completed rather than
    cancelled to whoever cancelled it. ``Task.cancelling()`` is what separates
    the two (non-zero only when cancellation was requested on this task);
    ``requires-python = ">=3.12"`` guarantees it exists.
    """
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class _ActiveResponse:  # pragma: no cover
    """The two in-flight calls of ONE response, and the hooks that kill them.

    Per-response, not per-session: the floor's ``cancel_generate``/
    ``cancel_tts`` hooks are fixed at construction, so they route through
    :class:`_ResponseCancels` to whichever response is currently live. A
    response that is still unwinding after an interruption therefore cannot
    have the NEXT turn's calls cancelled out from under it.

    ``cancel_tts`` both sets ``tts_client``'s own ``cancel_event`` (checked
    before each chunk request, so a queued retry stops) AND cancels the task
    (so a synthesis already blocked on the socket stops too). Both, because
    the floor cancels both hooks from every state and cannot know which
    applies.
    """

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self.cancelled = False
        self.generate_task: asyncio.Task | None = None
        self.tts_task: asyncio.Task | None = None
        self.tts_cancel = asyncio.Event()

    def cancel_generate(self) -> None:
        self.cancelled = True
        if self.generate_task is not None:
            self.generate_task.cancel()

    def cancel_tts(self) -> None:
        self.cancelled = True
        self.tts_cancel.set()
        if self.tts_task is not None:
            self.tts_task.cancel()


class _ResponseCancels:  # pragma: no cover
    """Indirection from the floor's fixed cancel hooks to the live response.

    The floor takes its two cancel callables once, at construction, but the
    thing they must cancel changes every turn. One of these per SESSION (a
    local, not a registry keyed by session id) holds the indirection: no
    module-level mutable state, so concurrent sessions cannot reach each
    other's calls and an abandoned entry cannot outlive its socket.
    """

    def __init__(self) -> None:
        self.active: _ActiveResponse | None = None

    def clear(self, response: _ActiveResponse) -> None:
        if self.active is response:
            self.active = None

    def generate(self) -> None:
        if self.active is not None:
            self.active.cancel_generate()

    def tts(self) -> None:
        if self.active is not None:
            self.active.cancel_tts()


class _Sender:  # pragma: no cover
    """Drains the bridge's outbox onto the socket, under one lock.

    The receive loop, the response task and the watchdog all produce events.
    Draining INSIDE the lock (not just sending inside it) is what keeps their
    payloads globally ordered: a second flusher waits, then drains what is
    there, so an interruption event can never overtake the delta that
    preceded it.
    """

    def __init__(self, websocket: WebSocket, bridge: ConversationBridge) -> None:
        self._websocket = websocket
        self._bridge = bridge
        self._lock = asyncio.Lock()

    async def flush(self) -> None:
        async with self._lock:
            for payload in self._bridge.drain():
                await self._websocket.send_json(payload)


class _SessionTasks:  # pragma: no cover
    """This session's background tasks — the watchdog and any live response."""

    def __init__(self) -> None:
        self.watchdog: asyncio.Task | None = None
        self.responses: set[asyncio.Task] = set()

    def ensure_watchdog(self, bridge: ConversationBridge, sender: _Sender) -> None:
        """Start the deadline watchdog once, when the session first arms.

        Lazy on purpose: an ears-only session that never opts into
        conversation spawns no extra task at all.
        """
        if self.watchdog is None:
            self.watchdog = asyncio.create_task(_watchdog(bridge, sender))

    def track(self, task: asyncio.Task) -> None:
        self.responses.add(task)
        task.add_done_callback(self.responses.discard)

    async def cancel_all(self) -> None:
        pending = [task for task in (self.watchdog, *self.responses) if task is not None]
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def _watchdog(bridge: ConversationBridge, sender: _Sender) -> None:  # pragma: no cover
    """Drive ``bridge.tick()`` so per-stage deadlines can actually expire.

    Deadlines expire ONLY inside ``tick``; a wedged backend is by definition
    not calling anything else, so without this loop a stuck generate or TTS
    call would strand the floor forever and the named-timeout guarantee would
    be inert.
    """
    interval_s = WATCHDOG_INTERVAL_MS / 1000
    while True:
        await asyncio.sleep(interval_s)
        if bridge.tick():
            await sender.flush()


async def _post_generate(request) -> tuple[int, bytes]:  # pragma: no cover
    """The one HTTP call of the generate stage. Shape decided by ``_turn.py``."""
    async with httpx.AsyncClient(timeout=_GENERATE_FORWARD_TIMEOUT) as client:
        resp = await client.post(request.url, headers=request.headers, json=request.body)
    return resp.status_code, resp.content


def _start_pending_response(  # pragma: no cover
    bridge: ConversationBridge,
    sender: _Sender,
    cancels: _ResponseCancels,
    tasks: _SessionTasks,
) -> None:
    """Launch a response task if the bridge says one is due.

    A background task, not an inline await, precisely so the receive loop
    keeps running while the machine thinks and speaks — that is what makes a
    barge-in possible at all.
    """
    turn_id = bridge.take_pending_response()
    if turn_id is None:
        return
    active = _ActiveResponse(turn_id)
    cancels.active = active
    tasks.track(asyncio.create_task(_run_response(bridge, sender, cancels, active)))


async def _run_response(  # pragma: no cover
    bridge: ConversationBridge,
    sender: _Sender,
    cancels: _ResponseCancels,
    active: _ActiveResponse,
) -> None:
    """Run one response to completion, interruption, or a named failure."""
    try:
        await _drive_response(bridge, sender, active)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("unhandled error in /v1/realtime response")
    finally:
        cancels.clear(active)
    await sender.flush()


async def _drive_response(  # pragma: no cover
    bridge: ConversationBridge, sender: _Sender, active: _ActiveResponse
) -> None:
    """generate -> TTS -> pumped audio-out, each stage handed back to the bridge.

    Delivery is a PUMP: one chunk per iteration with an ``await`` on the
    socket between them, so the receive loop gets a turn and an interruption
    lands on the undelivered remainder instead of arriving after every byte
    is already gone.
    """
    turn_id = active.turn_id
    request = bridge.build_generate_request(turn_id)
    if request is None:
        return  # the turn was interrupted or failed before we got here
    active.generate_task = asyncio.create_task(_post_generate(request))
    try:
        status_code, body = await active.generate_task
    except asyncio.CancelledError:
        # Barge-in cancelled the CHILD and the floor already emitted the
        # interruption, so there is nothing left to do — but only swallow that
        # when this task is not itself being cancelled (see _self_cancelled).
        if active.cancelled and not _self_cancelled():
            return
        raise
    except httpx.TimeoutException as exc:
        bridge.fail_generate(f"{type(exc).__name__}: {exc}", turn_id=turn_id, timed_out=True)
        return
    except httpx.HTTPError as exc:
        bridge.fail_generate(f"generate backend unreachable: {exc}", turn_id=turn_id)
        return
    bridge.on_generate_response(status_code, body, turn_id=turn_id)
    await sender.flush()

    pending = bridge.take_pending_synthesis()
    if pending is None:
        return  # empty reply, stale turn, or a named failure — already emitted
    _, reply_text = pending
    active.tts_task = asyncio.create_task(
        synthesize(
            reply_text,
            voice=settings.default_voice,
            tts_url=settings.tts_url,
            cancel_event=active.tts_cancel,
            lane=VOICE_LANE,
        )
    )
    try:
        pcm = await active.tts_task
    except asyncio.CancelledError:
        # Same rule as the generate stage above: a cancelled synthesis is
        # barge-in's business, our own cancellation is teardown's.
        if active.cancelled and not _self_cancelled():
            return
        raise
    except httpx.TimeoutException as exc:
        bridge.fail_tts(f"{type(exc).__name__}: {exc}", turn_id=turn_id, timed_out=True)
        return
    # No resample: Chatterbox's PCM16 is already the client's 24 kHz wire
    # rate, so these bytes reach the socket exactly as synthesize() returned
    # them, base64 and nothing else.
    bridge.on_tts_audio(pcm, turn_id=turn_id)
    await sender.flush()
    while bridge.deliver_next(turn_id=turn_id):
        await sender.flush()


async def _open_session(websocket: WebSocket) -> tuple[Session, int] | None:  # pragma: no cover
    """Parse the connect-URL config and open a session.

    ``None`` means the config was rejected — the named
    ``invalid_session_config`` error has been sent and the socket closed with
    1008 (policy violation), so the caller just returns.
    """
    raw_config = dict(websocket.query_params)
    try:
        config = parse_session_config(
            raw_config,
            default_turn_detection=settings.default_turn_detection,
            default_aec_mode=settings.default_aec_mode,
            # The operator half of the system-prompt contract; a client's own
            # `system_prompt` connect-config key always overrides it.
            default_system_prompt=settings.default_system_prompt,
        )
    except SessionConfigError as exc:
        await websocket.send_json(event_to_dict(exc.to_error_event(gen_session_id())))
        await websocket.close(code=1008)
        return None
    session, created_event = Session.create(config, raw_payload=raw_config)
    await websocket.send_json(event_to_dict(created_event))
    return session, config.input_sample_rate


async def _arm_segmenter(  # pragma: no cover
    websocket: WebSocket, session: Session
) -> Segmenter | None:
    """Load Silero and build this session's segmenter.

    ``None`` means the model could not be loaded — the named
    ``vad_unavailable`` error has been sent, which is what lets a consumer
    tell "VAD is down" from "nobody spoke".
    """
    try:
        vad_probability = await anyio.to_thread.run_sync(_load_vad_probability)
    except Exception as exc:
        await websocket.send_json(
            event_to_dict(session.mark_vad_unavailable(f"{type(exc).__name__}: {exc}"))
        )
        return None
    return Segmenter(
        vad_probability,
        vad_threshold=settings.vad_threshold,
        vad_silence_ms=settings.vad_silence_ms,
        vad_prefix_padding_ms=settings.vad_prefix_padding_ms,
        max_turn_ms=settings.vad_max_turn_ms,
    )


async def _to_pcm16k(aligned: bytes, input_rate: int) -> bytes:  # pragma: no cover
    """Bring a chunk to Silero/Parakeet's 16 kHz, off the event loop.

    scipy is SYNCHRONOUS CPU work and the bridge runs a single uvicorn loop
    shared by every realtime session and every batch ``/v1/audio/*`` request,
    so a resample runs in a worker thread — the same way
    ``chatterbox_server.py`` offloads its own model work. At 16 kHz there is
    nothing to do and a thread hop would cost more than the passthrough.
    """
    if not needs_resample(input_rate, VAD_SAMPLE_RATE):
        return aligned
    return await anyio.to_thread.run_sync(_resample_to_16k, aligned, input_rate)


async def _emit_turn_events(  # pragma: no cover
    bridge: ConversationBridge, sender: _Sender, events: list
) -> None:
    """Relay the segmenter's boundaries, transcribing each committed turn.

    A committed turn goes to Parakeet on this same connection; a failed
    forward becomes the named ``stt_forward_failed`` error, never a silently
    dropped turn. The STT forward stays INLINE (not a task): it is the
    #149 behaviour verbatim, and nothing is speaking yet for a barge-in to
    interrupt. Flushing after every boundary keeps the client's event stream
    live across a slow forward.

    Both segmenter timings reach the wire here — ``at_ms`` on each boundary
    and ``reason`` on the commit — and neither is passed into the floor's
    clock: they are audio-stream time, not wall-clock.
    """
    for event in events:
        if isinstance(event, SpeechStarted):
            bridge.on_speech_started(at_ms=event.at_ms)
            await sender.flush()
        elif isinstance(event, SpeechStopped):
            bridge.on_speech_stopped(at_ms=event.at_ms, reason=event.reason)
            await sender.flush()
            await _transcribe_turn(bridge, sender, event.audio)


async def _transcribe_turn(  # pragma: no cover
    bridge: ConversationBridge, sender: _Sender, audio: bytes
) -> None:
    try:
        text = await _forward_turn_to_stt(audio)
    except _STTForwardError as exc:
        bridge.on_transcription_failed(str(exc))
    else:
        bridge.on_transcript(text)
    await sender.flush()


async def _pump_session(  # pragma: no cover
    websocket: WebSocket,
    bridge: ConversationBridge,
    sender: _Sender,
    cancels: _ResponseCancels,
    segmenter: Segmenter,
    input_rate: int,
    tasks: _SessionTasks,
) -> None:
    """Receive audio until the client goes away, segmenting as it arrives.

    Every received message is classified by
    :func:`~lobes.realtime._wire.decide_inbound_message` — audio (a valid
    ``input_audio_buffer.append`` event), ignorable (a well-formed event
    that is not audio), or malformed (a named ``error`` event, never a silent
    drop). An ignorable event's decoded payload is offered to
    :meth:`~lobes.realtime._conversation.ConversationBridge.on_control_event`,
    which is where ``response.create`` — and only ``response.create`` — is
    acted on. This loop is a thin dispatch over those two pure decisions;
    both are tested offline in ``tests/test_realtime_wire.py`` and
    ``tests/test_realtime_conversation.py``, since this route (like every
    route in this module) is never imported by the offline suite.
    """
    pending = bytearray()
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        decision = decide_inbound_message(message)
        if decision.kind is InboundKind.IGNORED:
            if bridge.on_control_event(decision.payload):
                tasks.ensure_watchdog(bridge, sender)
                await sender.flush()
                _start_pending_response(bridge, sender, cancels, tasks)
            continue
        if decision.kind is InboundKind.ERROR:
            bridge.on_wire_error(decision.error)
            await sender.flush()
            continue
        audio = decision.audio

        pending.extend(audio)
        aligned = take_aligned_samples(pending, BYTES_PER_SAMPLE)
        if not aligned:
            continue
        pcm16k = await _to_pcm16k(aligned, input_rate)

        try:
            # torch inference, one call per 32 ms chunk — off the loop for the
            # same reason as the resample above.
            events = await anyio.to_thread.run_sync(segmenter.feed, pcm16k)
        except Exception as exc:
            # _segmenter.py deliberately lets a raising VAD callable
            # propagate (see its module docstring) — translating that into
            # the named session error is this route's job.
            bridge.fail_vad(f"{type(exc).__name__}: {exc}")
            await sender.flush()
            return
        await _emit_turn_events(bridge, sender, events)
        _start_pending_response(bridge, sender, cancels, tasks)
        await sender.flush()


def main() -> None:  # pragma: no cover - process entrypoint
    logging.basicConfig(level=logging.INFO)
    log.info("starting lobes realtime on %s:%d", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
