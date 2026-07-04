"""The realtime container's FastAPI app — OpenAI ``/v1/audio/*`` surface.

Runs ONLY in the ``realtime`` fleet container (the ``[realtime]`` extra: fastapi,
uvicorn, httpx). The offline dev/CI env has none of those, so this module is never
imported by the unit suite — its logic lives in the stdlib-only
:mod:`lobes.realtime.audio_facade` (codec + request parsing) and
:mod:`lobes.realtime.tts_client`, which are tested directly. The routes here
are thin shells (``# pragma: no cover``); the live stack is exercised by the
curl/WS smoke tests documented in ``docs/realtime-pipeline.md``.

PR2 adds the ``/v1/realtime`` WebSocket route to this same app.
"""

from __future__ import annotations

import logging

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ._settings import settings
from .audio_facade import (
    SpeechRequestError,
    aggregate_audio_ready,
    parse_speech_request,
    pcm_to_container,
)
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


def main() -> None:  # pragma: no cover - process entrypoint
    logging.basicConfig(level=logging.INFO)
    log.info("starting lobes realtime on %s:%d", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
