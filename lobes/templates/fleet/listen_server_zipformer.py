#!/usr/bin/env python3
"""The ``stt-he`` lane's engine: ivrit.ai Zipformer2 (66M, streaming) on sherpa-onnx, CPU.

Serves ``ivrit-ai/zipformer2-66m-streaming-he`` (pinned revision, the int8
``sherpa-onnx/`` export) as an OpenAI-compatible transcription server for the
gateway's ``stt-he`` lane (``lobes/stt_lanes.py``, ``docs/stt-lanes.md``).
CPU-only: no CUDA, no GPU reservation; two decode workers by default.

The weights are **CC-BY-4.0** (ivrit.ai): attribution is required wherever
their output is published — see ``docs/stt-lanes.md``.

Endpoints:
    POST /v1/audio/transcriptions  multipart: file, model?, language?,
                                   response_format? (json | text | verbose_json)
    GET  /v1/health/ready          200 once the model is loaded AND a warm-up
                                   decode succeeded (the gateway's lane probe)
    GET  /health                   liveness only
    GET  /v1/models                the one served id
    WS   /v1/audio/stream          EXPERIMENTAL, not fronted by the gateway:
                                   PCM16LE 16 kHz mono binary frames in,
                                   {"type": "partial"|"final", "text"} out;
                                   a text frame {"type": "end"} flushes.

Decode: any container ffmpeg can read is transcoded to 16 kHz mono PCM16 and
fed incrementally; endpoint detection is OFF and 1.5 s of silence is appended
before the stream is finished, so the last word is flushed. The model is
Hebrew-only: a caller's ``language`` field is accepted and ignored (English
speech comes out as Hebrew-script phonetics — a known limitation).

Adapted from the 2026-10-23 experiment server (measured on a Thor's CPU: RTF
~0.05-0.07, RSS ~280 MB); that run is an experiment, not an acceptance.

Structure (mirrors listen_server_whisper.py): every pure helper imports with
the standard library only, and numpy / sherpa-onnx / fastapi are imported
lazily or behind ``_FASTAPI_AVAILABLE``, so this file stays importable in the
offline test environment.
"""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404 — fixed ffmpeg argv, no shell
import threading
from typing import Optional

# The readiness decision from its single source of truth, exactly as
# listen_server.py / listen_server_whisper.py do: the Dockerfile COPYs
# _readiness.py next to this file; the wheel path is a dev fallback.
try:
    from _readiness import evaluate_readiness  # container-local copy (top-level)
except ImportError:
    from lobes.realtime._readiness import evaluate_readiness  # wheel install

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "ivrit-ai/zipformer2-66m-streaming-he"
MODEL_REVISION = os.environ.get("STT_HE_REVISION", "ee769a2a13b887f91572eec349458bfb0fe79e7a")
MODEL_SUBDIR = "sherpa-onnx"
RUNTIME = "sherpa-onnx"
LANGUAGE = "he"
SAMPLE_RATE = 16000
TAIL_PAD_SECONDS = 1.5
# 2 s of s16le per ffmpeg read.
CHUNK_BYTES = SAMPLE_RATE * 2 * 2
RESPONSE_FORMATS = ("json", "text", "verbose_json")
MODEL_FILES = {
    "tokens": "tokens.txt",
    "encoder": "encoder.int8.onnx",
    "decoder": "decoder.int8.onnx",
    "joiner": "joiner.int8.onnx",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


MAX_UPLOAD_BYTES = _env_int("STT_HE_MAX_UPLOAD_MB", 100) * 1024 * 1024
MAX_AUDIO_SECONDS = _env_float("STT_HE_MAX_AUDIO_SECONDS", 2 * 3600.0)
WORKERS = max(1, _env_int("STT_HE_WORKERS", 2))


# --------------------------------------------------------------------------
# Pure helpers — standard library only.
# --------------------------------------------------------------------------


def model_paths(model_dir: str) -> dict[str, str]:
    """The four sherpa-onnx transducer files inside *model_dir*."""
    return {key: os.path.join(model_dir, name) for key, name in MODEL_FILES.items()}


def missing_model_files(model_dir: str) -> list[str]:
    return [p for p in model_paths(model_dir).values() if not os.path.isfile(p)]


def ffmpeg_command(path: str) -> list[str]:
    """Transcode any input to raw 16 kHz mono PCM16 on stdout."""
    return [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]


def split_even(buf: bytes) -> tuple[bytes, bytes]:
    """``(whole 16-bit samples, leftover odd byte)`` — a read may end mid-sample."""
    cut = len(buf) - (len(buf) % 2)
    return buf[:cut], buf[cut:]


def response_format_error(fmt: Optional[str]) -> Optional[str]:
    """An error message for an unsupported ``response_format``, else ``None``."""
    if fmt in RESPONSE_FORMATS:
        return None
    return f"unsupported response_format {fmt!r} (json | text | verbose_json)"


def shape_response(text: str, fmt: str, duration: float) -> tuple[str, object]:
    """``("text", str)`` or ``("json", dict)`` — the OpenAI transcription shapes."""
    if fmt == "text":
        return "text", text
    if fmt == "verbose_json":
        return "json", {
            "text": text,
            "language": LANGUAGE,
            "duration": duration,
            "model": MODEL_ID,
        }
    return "json", {"text": text}


def error_body(message: str, code: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


def readiness(model_loaded: bool, warmed: bool) -> tuple[int, dict]:
    """``evaluate_readiness`` for a CPU engine: no CUDA check, but the warm-up
    decode must have succeeded, not merely the model object exist."""
    status, body = evaluate_readiness(model_loaded, True)
    if status == 200 and not warmed:
        status, body = 503, {"status": "not_ready", "reason": "warm-up decode pending"}
    return status, {**body, "model": MODEL_ID, "revision": MODEL_REVISION, "runtime": RUNTIME}


class AudioTooLong(Exception):
    pass


# --------------------------------------------------------------------------
# The engine (numpy + sherpa-onnx, imported lazily).
# --------------------------------------------------------------------------


class Engine:
    """One shared recognizer; each request decodes on its own stream."""

    def __init__(self) -> None:
        self.recognizer = None
        self.warmed = False
        self.error: Optional[str] = None

    def resolve_model_dir(self) -> str:
        local = os.environ.get("STT_HE_MODEL_DIR", "").strip()
        if local:
            return local
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=[f"{MODEL_SUBDIR}/*"],
        )
        return os.path.join(root, MODEL_SUBDIR)

    def load(self) -> None:
        try:
            import sherpa_onnx

            model_dir = self.resolve_model_dir()
            missing = missing_model_files(model_dir)
            if missing:
                raise FileNotFoundError(f"model files missing: {missing}")
            paths = model_paths(model_dir)
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=paths["tokens"],
                encoder=paths["encoder"],
                decoder=paths["decoder"],
                joiner=paths["joiner"],
                num_threads=1,
                sample_rate=SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=False,
                provider="cpu",
                model_type="zipformer2",
            )
            self.warm_up()
            logger.info("stt-he ready model=%s revision=%s", MODEL_ID, MODEL_REVISION)
        except Exception as exc:  # pragma: no cover - exercised only in the container
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("stt-he model load failed: %s", self.error)

    def warm_up(self) -> None:
        import numpy as np

        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.finish(stream)
        self.warmed = True

    def text(self, stream) -> str:
        result = self.recognizer.get_result(stream)
        return (result if isinstance(result, str) else getattr(result, "text", str(result))).strip()

    def drain(self, stream) -> None:
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

    def feed(self, stream, buf: bytes) -> int:
        import numpy as np

        pcm = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
        stream.accept_waveform(SAMPLE_RATE, pcm)
        self.drain(stream)
        return int(pcm.size)

    def finish(self, stream) -> str:
        import numpy as np

        pad = np.zeros(int(TAIL_PAD_SECONDS * SAMPLE_RATE), dtype=np.float32)
        stream.accept_waveform(SAMPLE_RATE, pad)
        stream.input_finished()
        self.drain(stream)
        return self.text(stream)

    def transcribe_file(self, path: str) -> tuple[str, float]:
        """Transcode with ffmpeg and decode incrementally: ``(text, seconds)``."""
        stream = self.recognizer.create_stream()
        proc = subprocess.Popen(  # nosec B603 — fixed argv, no shell
            ffmpeg_command(path), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        samples = 0
        rem = b""
        try:
            while True:
                buf = proc.stdout.read(CHUNK_BYTES)
                if not buf:
                    break
                whole, rem = split_even(rem + buf)
                samples += self.feed(stream, whole)
                if samples > MAX_AUDIO_SECONDS * SAMPLE_RATE:
                    raise AudioTooLong()
        finally:
            if proc.poll() is None:
                proc.kill()
            err = proc.stderr.read().decode("utf-8", "replace")
            proc.wait()
        if samples == 0:
            raise ValueError(err.strip()[-500:] or "no audio decoded")
        return self.finish(stream), samples / SAMPLE_RATE


ENGINE = Engine()


# --------------------------------------------------------------------------
# HTTP app (fastapi, guarded so the pure helpers import without it).
# --------------------------------------------------------------------------

try:
    import asyncio
    import json
    import tempfile
    import time
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, PlainTextResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - the offline test environment
    _FASTAPI_AVAILABLE = False


if _FASTAPI_AVAILABLE:  # pragma: no cover - exercised only in the container
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    app = FastAPI(title="lobes stt-he (zipformer2, sherpa-onnx)")

    def _err(status: int, message: str, code: str):
        return JSONResponse(error_body(message, code), status_code=status)

    @app.on_event("startup")
    async def startup():
        # Load off the event loop so /health answers while the weights fetch.
        threading.Thread(target=ENGINE.load, name="stt-he-load", daemon=True).start()

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/v1/health/ready")
    async def ready():
        status, body = readiness(ENGINE.recognizer is not None, ENGINE.warmed)
        if ENGINE.error:
            body["error"] = ENGINE.error
        return JSONResponse(body, status_code=status)

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "ivrit-ai"}],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(MODEL_ID),
        language: Optional[str] = Form(None),
        response_format: str = Form("json"),
    ):
        del model, language  # one Hebrew-only model; both are accepted, never required
        problem = response_format_error(response_format)
        if problem:
            return _err(400, problem, "bad_response_format")
        if not ENGINE.warmed:
            return _err(503, "stt-he model not ready yet — retry shortly", "not_ready")
        suffix = os.path.splitext(file.filename or "")[1][:10]
        fd, path = tempfile.mkstemp(prefix="stt-he-", suffix=suffix)
        try:
            size = 0
            with os.fdopen(fd, "wb") as out:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        return _err(413, "upload too large", "upload_too_large")
                    out.write(chunk)
            started = time.monotonic()
            try:
                text, duration = await asyncio.get_running_loop().run_in_executor(
                    pool, ENGINE.transcribe_file, path
                )
            except AudioTooLong:
                return _err(413, f"audio exceeds {MAX_AUDIO_SECONDS:.0f} s", "audio_too_long")
            except ValueError as exc:
                return _err(415, f"could not decode audio: {exc}", "unsupported_media")
            elapsed_ms = (time.monotonic() - started) * 1000.0
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        headers = {"X-Transcribe-Ms": f"{elapsed_ms:.1f}", "X-Audio-Seconds": f"{duration:.3f}"}
        logger.info("stt_request seconds=%.2f ms=%.1f chars=%d", duration, elapsed_ms, len(text))
        kind, payload = shape_response(text, response_format, duration)
        if kind == "text":
            return PlainTextResponse(payload, headers=headers)
        return JSONResponse(payload, headers=headers)

    @app.websocket("/v1/audio/stream")
    async def stream_ws(ws: WebSocket):
        """EXPERIMENTAL streaming (not fronted by the gateway)."""
        await ws.accept()
        if not ENGINE.warmed:
            await ws.send_json({"type": "error", "message": "not_ready"})
            await ws.close()
            return
        loop = asyncio.get_running_loop()
        lock = threading.Lock()
        state = {"stream": ENGINE.recognizer.create_stream(), "last": "", "samples": 0}

        def feed(buf: bytes) -> str:
            with lock:
                state["samples"] += ENGINE.feed(state["stream"], buf)
                return ENGINE.text(state["stream"])

        def finish() -> str:
            with lock:
                text = ENGINE.finish(state["stream"])
                state.update(stream=ENGINE.recognizer.create_stream(), last="", samples=0)
                return text

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    whole, _odd = split_even(msg["bytes"])
                    text = await loop.run_in_executor(pool, feed, whole)
                    if state["samples"] > MAX_AUDIO_SECONDS * SAMPLE_RATE:
                        await ws.send_json({"type": "error", "message": "audio_too_long"})
                        break
                    if text != state["last"]:
                        state["last"] = text
                        await ws.send_json({"type": "partial", "text": text})
                elif msg.get("text"):
                    try:
                        event = json.loads(msg["text"])
                    except ValueError:
                        event = {}
                    if event.get("type") == "end":
                        text = await loop.run_in_executor(pool, finish)
                        await ws.send_json({"type": "final", "text": text})
        except WebSocketDisconnect:
            pass


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",  # nosec B104 — inside the container, port-less on the compose network
        port=_env_int("STT_HE_PORT", 9002),
        workers=1,
        log_level="info",
    )
