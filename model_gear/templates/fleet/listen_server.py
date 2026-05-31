#!/usr/bin/env python3
"""Parakeet ASR HTTP server (vendored from the qwen3-tts sibling).

Serves NVIDIA Parakeet TDT 0.6B as a simple HTTP API compatible with the OpenAI
/ Riva ASR transcription endpoint. Runs inside the `stt` fleet container (built
from Dockerfile.parakeet), which has nemo / soundfile / fastapi installed — it is
a scaffolded template, not part of the model_gear package's runtime imports.

Endpoints:
    POST /v1/audio/transcriptions  - Transcribe an uploaded audio file
    GET  /v1/health/ready          - Health check
"""

import io
import logging
import os

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

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
    return {"status": "ready"}


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
