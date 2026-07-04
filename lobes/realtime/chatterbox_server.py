"""Chatterbox TTS sidecar — FastAPI HTTP server for the fleet audio overlay.

Runs ONLY in the ``chatterbox`` fleet container (the ``[chatterbox]`` extra:
fastapi, uvicorn). The offline dev/CI env has none of those **and** no GPU, so
this module is never imported by the unit suite — its routes are thin shells
(``# pragma: no cover``). The stdlib-only PCM16 conversion helper at the bottom
**is** tested directly.

HTTP contract
-------------
GET  /v1/health/ready            → 503 ``{"status":"loading"}`` until the model
                                   is loaded; 200 ``{"status":"ok"}`` once ready.
POST /v1/audio/synthesize        → JSON body ``{"text": str, "voice": str|null}``
                                   Response: raw PCM16 mono 24 kHz bytes
                                   Content-Type: audio/pcm

Voice semantics: if ``voice`` ends with ``.wav`` (case-insensitive) it is passed
to Chatterbox as ``audio_prompt_path`` (zero-shot cloning); any other value (or
null/empty) uses the model's single built-in default voice.
"""

from __future__ import annotations

import logging
import os
import struct
import threading

log = logging.getLogger(__name__)

try:
    import numpy as _np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pure stdlib helper — unit-testable without fastapi/torch/numpy
# ---------------------------------------------------------------------------


def float_tensor_to_pcm16(tensor) -> bytes:  # type: ignore[type-arg]
    """Convert a 1-D float tensor (or array-like) to raw PCM16 bytes.

    Works with torch.Tensor, numpy arrays, and plain Python lists/iterables.
    Clamps the float values to [-1, 1] before converting to signed 16-bit
    little-endian integers.

    Uses a numpy fast path when numpy is available (typical in the GPU container
    and the dev env); falls back to a pure-Python/stdlib path otherwise (offline
    CI).  Both paths produce byte-identical output.

    Asymmetric int16 scaling so both peaks map to the full int16 range:
        +1.0 → 32767 (max int16)   -1.0 → -32768 (min int16)

    Args:
        tensor: A 1-D (or squeezable to 1-D) float tensor/array/list with
            values nominally in [-1, 1].

    Returns:
        Raw PCM16 bytes (little-endian, mono).
    """
    if _NUMPY_AVAILABLE:
        # --- numpy fast path ---------------------------------------------------
        arr = _np.asarray(tensor, dtype=_np.float32).reshape(-1)
        if arr.size == 0:
            return b""
        arr = _np.clip(arr, -1.0, 1.0)
        # Asymmetric scaling: positive values → *32767, negative → *32768
        scaled = _np.where(arr >= 0, arr * 32767.0, arr * 32768.0)
        return scaled.astype("<i2").tobytes()

    # --- pure-Python / stdlib fallback (numpy absent) --------------------------
    # Flatten to 1-D.  torch.Tensor and numpy.ndarray expose .squeeze() and
    # .tolist(); plain Python lists/iterables do not — handle all three cases.
    if hasattr(tensor, "squeeze"):
        # torch.Tensor or numpy.ndarray — squeeze then convert to a Python list.
        squeezed = tensor.squeeze()
        if hasattr(squeezed, "tolist"):
            samples: list = squeezed.tolist()
        else:
            samples = list(squeezed)
    else:
        # Plain Python list or other iterable — use as-is.
        samples = list(tensor)

    if not samples:
        return b""

    clamped = [max(-1.0, min(1.0, float(s))) for s in samples]

    def _scale(s: float) -> int:
        if s >= 0.0:
            return int(s * 32767)
        return int(s * 32768)

    packed = struct.pack(f"<{len(clamped)}h", *[_scale(s) for s in clamped])
    return packed


# ---------------------------------------------------------------------------
# FastAPI sidecar (imported only inside the chatterbox container)
# ---------------------------------------------------------------------------

# fastapi / uvicorn / anyio are provided by the [chatterbox] extra and the
# container's Dockerfile.chatterbox install recipe — not in the offline CI env.
# All route bodies are marked pragma: no cover.

try:
    import anyio
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False

_model = None  # ChatterboxTTS singleton
_model_lock = threading.Lock()
_cuda_poisoned: bool = False


def readiness_status(model_loaded: bool, cuda_poisoned: bool) -> tuple[int, dict]:
    """Return an ``(http_status, body)`` pair reflecting real TTS readiness.

    Pure function — no torch / fastapi imports — so it is unit-testable in the
    offline CI environment (no GPU, no container deps).
    """
    if not model_loaded:
        return 503, {"status": "loading"}
    if cuda_poisoned:
        return 503, {"status": "unavailable", "reason": "cuda_context_poisoned"}
    return 200, {"status": "ready"}


def _get_model():  # pragma: no cover
    """Lazy-load the ChatterboxTTS model once per process (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from chatterbox.tts import ChatterboxTTS  # type: ignore[import]

                log.info("[Chatterbox] loading model (cold-load ~19 s) …")
                _model = ChatterboxTTS.from_pretrained(device="cuda")
                log.info("[Chatterbox] model ready — sample rate: %d Hz", _model.sr)
    return _model


if _FASTAPI_AVAILABLE:
    app = FastAPI(title="lobes chatterbox-tts", version="1")

    @app.on_event("startup")  # pragma: no cover
    async def _warm_model() -> None:
        threading.Thread(target=_get_model, daemon=True).start()

    @app.get("/v1/health/ready")  # pragma: no cover
    async def health() -> Response:
        code, body = readiness_status(_model is not None, _cuda_poisoned)
        return JSONResponse(status_code=code, content=body)

    @app.post("/v1/audio/synthesize")  # pragma: no cover
    async def synthesize(request_body: dict) -> Response:
        """Synthesize text to raw PCM16 mono 24 kHz bytes.

        Request JSON:
            {"text": "<utterance>", "voice": "<optional .wav path or null>"}

        Returns:
            audio/pcm — raw PCM16 little-endian mono 24 kHz.
        """
        global _cuda_poisoned
        text = (request_body.get("text") or "").strip()
        if not text:
            return JSONResponse(
                status_code=400, content={"error": {"message": "text must be non-empty"}}
            )

        voice = request_body.get("voice") or ""

        def _generate() -> bytes:
            mdl = _get_model()
            kwargs: dict = {"exaggeration": 0.5, "cfg_weight": 0.5}
            if voice.lower().endswith(".wav"):
                kwargs["audio_prompt_path"] = voice
            wav_tensor = mdl.generate(text, **kwargs)
            # Chatterbox emits 24 kHz mono (mdl.sr); the facade wraps the bare
            # PCM16 in a container, so return raw bytes per the audio/pcm contract.
            return float_tensor_to_pcm16(wav_tensor)

        try:
            pcm = await anyio.to_thread.run_sync(_generate)
        except Exception as exc:
            exc_msg = f"{type(exc).__name__}: {exc}"
            if "cuda" in exc_msg.lower() or "accelerator" in exc_msg.lower():
                _cuda_poisoned = True
                log.warning("[Chatterbox] CUDA/accelerator error — marking poisoned: %s", exc_msg)
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": "CUDA context error", "detail": exc_msg}},
                )
            raise
        _cuda_poisoned = False
        return Response(content=pcm, media_type="audio/pcm")


def main() -> None:  # pragma: no cover
    """Process entrypoint — ``python -m lobes.realtime.chatterbox_server``."""
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("CHATTERBOX_HOST", "0.0.0.0")  # nosec B104
    port = int(os.environ.get("CHATTERBOX_PORT", "9000"))
    log.info("starting lobes chatterbox-tts on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
