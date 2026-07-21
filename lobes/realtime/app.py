"""The realtime container's FastAPI app — OpenAI ``/v1/audio/*`` surface plus
the ``/v1/realtime`` WebSocket session route.

Runs ONLY in the ``realtime`` fleet container (the ``[realtime]`` extra: fastapi,
uvicorn, httpx, numpy, scipy, silero-vad, torch). The offline dev/CI env has
none of those, so this module is never imported by the unit suite — its logic
lives in stdlib-only modules that ARE tested directly:
:mod:`lobes.realtime.audio_facade` (codec + request parsing),
:mod:`lobes.realtime.tts_client`, :mod:`lobes.realtime._segmenter` (the
server_vad state machine), :mod:`lobes.realtime._session` (event schema +
session bookkeeping), and :mod:`lobes.realtime._pcm` (resample-decision +
frame-alignment arithmetic for ``/v1/realtime``). The routes here are thin
shells (``# pragma: no cover``) that wire those modules to a real WebSocket,
real Silero VAD, and real scipy resampling; the live stack is exercised by
the curl/WS smoke tests documented in ``docs/realtime-pipeline.md``.
"""

from __future__ import annotations

import logging

import anyio
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from scipy.signal import resample as scipy_resample

from ._pcm import needs_resample, resampled_frame_count, take_aligned_samples
from ._segmenter import Segmenter, SpeechStarted, SpeechStopped
from ._session import Session, SessionConfigError, event_to_dict, parse_session_config
from ._settings import settings
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
# /v1/realtime — one server_vad session per WebSocket (issue #149 t6).
#
# All decisions live in stdlib modules tested offline: _session.py owns the
# event schema/config parsing/teardown, _segmenter.py owns VAD segmentation,
# _pcm.py owns the resample-needed decision and frame alignment. This block
# is the thin, pragma-no-cover glue that wires them to a real WebSocket, a
# real Silero model, and real scipy resampling.
# ---------------------------------------------------------------------------

_STT_FORWARD_TIMEOUT = 60.0


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
    - **Input audio**: PCM16 mono little-endian, streamed as BINARY WebSocket
      frames at whatever chunking granularity the client happens to send (the
      server reassembles it — a WS frame need not align to a whole sample,
      let alone a whole VAD chunk). ``input_sample_rate`` defaults to
      **24000 Hz** (OpenAI-Realtime-compatible) — **16000 Hz is also
      accepted** (Parakeet/Silero's native rate; the server skips resampling
      entirely in that case). Any other rate is rejected as an invalid
      session config. The server resamples 24 kHz to 16 kHz itself
      (server-side, via scipy) — the client never resamples.
    - **Events** (session/boundary/transcription/error) are sent back as JSON
      TEXT frames using the schema in :mod:`lobes.realtime._session`
      (:func:`event_to_dict`). This route never sends audio back on this
      connection — audio-in only; no TTS-out over ``/v1/realtime`` (see the
      spec's Non-goals).
    - The server sends ``session.created`` immediately after the handshake,
      confirming the negotiated config (including the resolved
      ``input_sample_rate``) — a client that omitted every config query
      param can read the effective defaults off this event.

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
    try:
        opened = await _open_session(websocket)
        if opened is None:
            return  # the config was rejected; the named error is already sent
        session, input_rate = opened
        segmenter = await _arm_segmenter(websocket, session)
        if segmenter is None:
            return  # VAD is down; the named error is already sent
        await _pump_session(websocket, session, segmenter, input_rate)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("unhandled error in /v1/realtime session")
    finally:
        if segmenter is not None and segmenter.speaking:
            # Best-effort: the client is already gone by the time we get
            # here on most paths, so the flushed turn is discarded rather
            # than forwarded to STT — nothing could ever deliver its
            # transcript back. teardown() below is what actually matters.
            segmenter.flush(reason="closed")
        if session is not None:
            session.teardown(reason="client_disconnect")


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
    websocket: WebSocket, session: Session, events: list
) -> None:
    """Relay the segmenter's boundaries, transcribing each committed turn.

    A committed turn goes to Parakeet on this same connection; a failed
    forward becomes the named ``stt_forward_failed`` error, never a silently
    dropped turn.
    """
    for event in events:
        if isinstance(event, SpeechStarted):
            await websocket.send_json(event_to_dict(session.begin_speech()))
        elif isinstance(event, SpeechStopped):
            await websocket.send_json(event_to_dict(session.end_speech()))
            await _transcribe_turn(websocket, session, event.audio)


async def _transcribe_turn(  # pragma: no cover
    websocket: WebSocket, session: Session, audio: bytes
) -> None:
    try:
        text = await _forward_turn_to_stt(audio)
    except _STTForwardError as exc:
        await websocket.send_json(event_to_dict(session.fail_transcription(str(exc))))
    else:
        await websocket.send_json(event_to_dict(session.complete_transcription(text)))


async def _pump_session(  # pragma: no cover
    websocket: WebSocket, session: Session, segmenter: Segmenter, input_rate: int
) -> None:
    """Receive audio until the client goes away, segmenting as it arrives."""
    pending = bytearray()
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        audio = message.get("bytes")
        if audio is None:
            # A text/control frame. Out of scope (spec Non-goals: no
            # response.create, no mid-session session.update) — ignored,
            # never mistaken for a boundary/audio frame.
            continue

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
            await websocket.send_json(
                event_to_dict(session.mark_vad_unavailable(f"{type(exc).__name__}: {exc}"))
            )
            return
        await _emit_turn_events(websocket, session, events)


def main() -> None:  # pragma: no cover - process entrypoint
    logging.basicConfig(level=logging.INFO)
    log.info("starting lobes realtime on %s:%d", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
