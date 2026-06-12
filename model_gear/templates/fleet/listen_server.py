#!/usr/bin/env python3
"""Parakeet ASR HTTP server (vendored from the qwen3-tts sibling).

Serves NVIDIA Parakeet TDT 0.6B as a simple HTTP API compatible with the OpenAI
/ Riva ASR transcription endpoint. Runs inside the `stt` fleet container (built
from Dockerfile.parakeet), which has nemo / soundfile / fastapi installed — it is
a scaffolded template, not part of the model_gear package's runtime imports.

Endpoints:
    POST /v1/audio/transcriptions  - Transcribe an uploaded audio file
    GET  /v1/health/ready          - Readiness check (model loaded + CUDA live)
"""

import io
import logging
import os

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

# Import the readiness decision logic.  The Dockerfile COPYs _readiness.py
# next to listen_server.py (as a top-level module in /app), so this import
# works whether or not the model_gear wheel is installed in the container.
# Falls back gracefully if neither is present (shouldn't happen, but defensive).
try:
    from _readiness import evaluate_readiness  # container-local copy (top-level)
except ImportError:
    try:
        from model_gear.realtime._readiness import evaluate_readiness  # wheel install
    except ImportError:
        # Last-resort inline fallback — keeps the container functional even if
        # both import paths fail (e.g. a dev build without the COPY step).
        def evaluate_readiness(model_loaded: bool, cuda_ok: bool):  # type: ignore[misc]
            if not model_loaded:
                return 503, {"status": "not_ready", "reason": "model not loaded"}
            if not cuda_ok:
                return 503, {"status": "not_ready", "reason": "CUDA not available"}
            return 200, {"status": "ready"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Parakeet ASR")

SAMPLE_RATE = 16000
MODEL_NAME = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")

_model = None


def get_model():
    global _model
    if _model is None:
        import nemo.collections.asr as nemo_asr

        logging.getLogger("nemo").setLevel(logging.WARNING)
        logger.info("Loading model %s...", MODEL_NAME)
        _model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
        _model.eval()
        logger.info("Model loaded.")
    return _model


@app.on_event("startup")
async def startup():
    get_model()


@app.get("/v1/health/ready")
async def health():
    """Cheap readiness probe (issue #39, decision c16).

    Reports ready ONLY when the NeMo model is loaded AND a trivial CUDA tensor
    op succeeds — never unconditionally.  Docker healthcheck treats non-2xx as
    failing, so 503 keeps the container unhealthy until it can actually serve.
    """
    model_loaded = _model is not None

    # Cheap CUDA liveness: allocate a 1-element tensor and synchronise.
    # Any CUDA error (unknown error, driver not ready, OOM, …) → not ready.
    try:
        import torch

        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        cuda_ok = True
    except Exception:
        cuda_ok = False

    status_code, body = evaluate_readiness(model_loaded, cuda_ok)
    return JSONResponse(status_code=status_code, content=body)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    """Transcribe an uploaded audio file."""
    content = await file.read()

    # Load audio
    audio, sr = sf.read(io.BytesIO(content), dtype="float32")

    # Resample if needed
    if sr != SAMPLE_RATE:
        import scipy.signal

        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = scipy.signal.resample(audio, num_samples)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Transcribe
    model = get_model()
    results = model.transcribe([audio], verbose=False)
    r = results[0]
    text = r.text if hasattr(r, "text") else str(r)

    return {"text": text}


if __name__ == "__main__":
    port = int(os.environ.get("PARAKEET_PORT", "9002"))
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104 — bind all inside the container
