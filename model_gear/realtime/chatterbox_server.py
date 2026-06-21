"""Chatterbox TTS sidecar — FastAPI HTTP server for the fleet audio overlay.

Runs ONLY in the ``chatterbox`` fleet container (the ``[chatterbox]`` extra:
fastapi, uvicorn). The offline dev/CI env has none of those **and** no GPU, so
this module is never imported by the unit suite — its routes are thin shells
(``# pragma: no cover``). The stdlib-only PCM16 conversion helper at the bottom
**is** tested directly.

HTTP contract
-------------
GET  /v1/health/ready            → 200 ``{"status":"ok"}``
POST /v1/audio/synthesize        → JSON body ``{"text": str, "voice": str|null}``
                                   Response: raw PCM16 mono 24 kHz bytes
                                   Content-Type: audio/pcm

Voice semantics: if ``voice`` ends with ``.wav`` it is passed to Chatterbox as
``audio_prompt_path`` (zero-shot cloning); any other value (or null/empty) uses
the model's single built-in default voice.
"""

from __future__ import annotations

import logging
import os
import struct

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure stdlib helper — unit-testable without fastapi/torch/numpy
# ---------------------------------------------------------------------------


def float_tensor_to_pcm16(tensor) -> bytes:  # type: ignore[type-arg]
    """Convert a 1-D float tensor (or array-like) to raw PCM16 bytes.

    Works with torch.Tensor, numpy arrays, and plain Python lists/iterables.
    Clamps the float values to [-1, 1] before converting to signed 16-bit
    little-endian integers.

    This function is **stdlib-only** at the call site — the tensor may be a
    torch.Tensor but no torch import is required *here*.  The caller (inside the
    GPU container) passes the tensor; the helper just iterates and packs.

    Args:
        tensor: A 1-D (or squeezable to 1-D) float tensor/array/list with
            values nominally in [-1, 1].

    Returns:
        Raw PCM16 bytes (little-endian, mono).
    """
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

    # Asymmetric int16 scaling so both peaks map to the full range:
    #   +1.0 * 32767 → 32767  (max int16);  -1.0 * 32768 → -32768 (min int16).
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


def _get_model():  # pragma: no cover
    """Lazy-load the ChatterboxTTS model once per process."""
    global _model
    if _model is None:
        # Import chatterbox lazily so the module can be imported (e.g. for the
        # PCM16 helper) even when chatterbox-tts is not installed.
        from chatterbox.tts import ChatterboxTTS  # type: ignore[import]

        log.info("[Chatterbox] loading model (first request; cold-load ~19 s) …")
        _model = ChatterboxTTS.from_pretrained(device="cuda")
        log.info("[Chatterbox] model ready — sample rate: %d Hz", _model.sr)
    return _model


if _FASTAPI_AVAILABLE:
    app = FastAPI(title="model-gear chatterbox-tts", version="1")

    @app.get("/v1/health/ready")  # pragma: no cover
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/audio/synthesize")  # pragma: no cover
    async def synthesize(request_body: dict) -> Response:
        """Synthesize text to raw PCM16 mono 24 kHz bytes.

        Request JSON:
            {"text": "<utterance>", "voice": "<optional .wav path or null>"}

        Returns:
            audio/pcm — raw PCM16 little-endian mono 24 kHz.
        """
        text = (request_body.get("text") or "").strip()
        if not text:
            return JSONResponse(
                status_code=400, content={"error": {"message": "text must be non-empty"}}
            )

        voice = request_body.get("voice") or ""

        def _generate() -> bytes:
            mdl = _get_model()
            kwargs: dict = {"exaggeration": 0.5, "cfg_weight": 0.5}
            if voice.endswith(".wav"):
                kwargs["audio_prompt_path"] = voice
            wav_tensor = mdl.generate(text, **kwargs)
            # Chatterbox emits 24 kHz mono (mdl.sr); the facade wraps the bare
            # PCM16 in a container, so return raw bytes per the audio/pcm contract.
            return float_tensor_to_pcm16(wav_tensor)

        pcm = await anyio.to_thread.run_sync(_generate)
        return Response(content=pcm, media_type="audio/pcm")


def main() -> None:  # pragma: no cover
    """Process entrypoint — ``python -m model_gear.realtime.chatterbox_server``."""
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("CHATTERBOX_HOST", "0.0.0.0")  # nosec B104
    port = int(os.environ.get("CHATTERBOX_PORT", "9000"))
    log.info("starting model-gear chatterbox-tts on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
