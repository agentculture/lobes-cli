"""BlueTTS sidecar — the Hebrew voice behind ``POST /v1/audio/synthesize``.

hebrew-realtime plan, approved deviation **d8** (operator, 2026-09-18:
"Sounds perfect!"). A drop-in for the ``chatterbox`` service: same internal
port keys (``CHATTERBOX_HOST`` / ``CHATTERBOX_PORT``), same two routes, same
response body — raw PCM16 mono little-endian at 24 kHz — so the realtime
bridge's ``TTS_URL`` and ``tts_client.synthesize`` need no change at all.

Why it replaces Chatterbox Multilingual as the Hebrew voice
----------------------------------------------------------
Measured on the DGX Spark, 2026-09-18
(``docs/evidence/2026-09-hebrew-tts-ab-spark.txt``): BlueTTS (a Supertonic-
style flow-matching ONNX stack, https://github.com/maxmelichov/BlueTTS, MIT
code) synthesizes one Hebrew sentence in **160-325 ms on CPU** with a 0 %
round-trip WER, where Chatterbox Multilingual took seconds per sentence, needs
external niqqud, and samples non-deterministically (run-ons up to 24 s, hence
that sidecar's runaway guard). BlueTTS is deterministic enough to need no
guard, so none is carried here.

What differs from the Chatterbox sidecars
-----------------------------------------
* **CPU, not GPU.** onnxruntime's CPU provider; no CUDA context, so no
  "poisoned" state and no GPU budget to fund.
* **Its own G2P.** BlueTTS phonemizes Hebrew with RenikudPlus internally. It
  wants PLAIN Hebrew: niqqud from another vocalizer is a second authority, so
  :func:`strip_niqqud` removes any that arrives (the bridge's phonikud hook is
  switched off on a BlueTTS deployment by leaving ``PHONIKUD_MODEL_PATH``
  unset — the strip is the belt to that brace).
* **44.1 kHz native.** Resampled to the bridge's 24 kHz contract here
  (:func:`resample_ratio` → 80/147 polyphase), never in the bridge.
* **Readiness includes the G2P.** RenikudPlus loads (and, unless
  ``BLUETTS_RENIKUD_PATH`` names a local file, downloads) on FIRST use —
  measured in seconds. ``/v1/health/ready`` stays 503 until a warm-up
  synthesis has run, so the first spoken turn never pays for it.

Weights: deliberately NO default repo
-------------------------------------
The BlueTTS *code* is MIT; its HuggingFace *weights* repo declared no licence
when this was written (2026-09-18; the operator is asking the author). So this
module never downloads weights and names no weights repo: it reads a LOCAL
directory the operator mounted (``BLUETTS_ONNX_DIR``). Chatterbox Multilingual
stays the Hebrew overlay's shipped default until a licence is declared.
"""

from __future__ import annotations

import logging
import math
import os
import re
import struct
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from .protocol import TTS_SAMPLE_RATE

log = logging.getLogger(__name__)

__all__ = [
    "TTS_SAMPLE_RATE",
    "BlueSettings",
    "build_blue_settings",
    "floats_to_pcm16",
    "readiness_status",
    "resample_ratio",
    "resolve_voice_path",
    "strip_niqqud",
]

# ---------------------------------------------------------------------------
# Pure stdlib helpers — unit-testable without numpy/onnxruntime/fastapi
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 9000
_DEFAULT_STEPS = 5
_DEFAULT_SPEED = 1.0
_MIN_SPEED = 0.5
_MAX_SPEED = 2.0


@dataclass(frozen=True)
class BlueSettings:
    host: str
    port: int
    language: str
    onnx_dir: str
    voices_dir: str
    config_path: str
    voice: str
    total_step: int
    speed: float
    threads: int
    renikud_path: str | None


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, "").strip() or default)
    except ValueError:
        return default


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        value = float(env.get(key, "").strip() or default)
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def build_blue_settings(env: Mapping[str, str] | None = None) -> BlueSettings:
    """Read the sidecar's knobs. A typo falls back to the default (never a
    crash-looping container); steps and speed are clamped to a sane range."""
    env = os.environ if env is None else env
    return BlueSettings(
        host=env.get("CHATTERBOX_HOST", "").strip() or "0.0.0.0",  # nosec B104
        port=_as_int(env, "CHATTERBOX_PORT", _DEFAULT_PORT),
        language=env.get("TTS_LANGUAGE", "").strip() or "he",
        onnx_dir=env.get("BLUETTS_ONNX_DIR", "").strip() or "/models/bluetts/onnx_models",
        voices_dir=env.get("BLUETTS_VOICES_DIR", "").strip() or "/opt/bluetts/voices",
        config_path=env.get("BLUETTS_CONFIG_PATH", "").strip() or "/opt/bluetts/config/tts.json",
        voice=env.get("BLUETTS_VOICE", "").strip() or "noa",
        total_step=max(1, _as_int(env, "BLUETTS_STEPS", _DEFAULT_STEPS)),
        speed=min(_MAX_SPEED, max(_MIN_SPEED, _as_float(env, "BLUETTS_SPEED", _DEFAULT_SPEED))),
        threads=max(0, _as_int(env, "BLUETTS_THREADS", 0)),
        renikud_path=env.get("BLUETTS_RENIKUD_PATH", "").strip() or None,
    )


# U+0591-U+05AF cantillation + U+05B0-U+05C7 niqqud, minus the four code
# points in that span that are PUNCTUATION (maqaf, paseq, sof pasuq, nun
# hafukha) — removing a maqaf would glue two words together.
_HEBREW_PUNCTUATION = frozenset({0x05BE, 0x05C0, 0x05C3, 0x05C6})
# phonikud's vocal-shva / prefix bar, which is not a Hebrew code point at all.
_PHONIKUD_BAR = "|"


def strip_niqqud(text: str) -> str:
    """Remove vocalization marks so RenikudPlus is the only G2P authority."""
    return "".join(
        ch
        for ch in text
        if ch != _PHONIKUD_BAR
        and not (0x0591 <= ord(ch) <= 0x05C7 and ord(ch) not in _HEBREW_PUNCTUATION)
    )


_SAFE_VOICE = re.compile(r"^[A-Za-z0-9_-]+$")


def resolve_voice_path(
    voice: str | None,
    voices_dir: str,
    default_voice: str,
    *,
    exists: Callable[[str], bool] = os.path.exists,
) -> str:
    """Map a request's ``voice`` to a style JSON inside ``voices_dir``.

    An unknown voice — or anything that is not a bare name, so a request can
    never walk out of the directory — falls back to the default voice rather
    than failing the spoken turn (the English sidecar ignores an unknown voice
    the same way).
    """
    if voice and _SAFE_VOICE.match(voice):
        candidate = os.path.join(voices_dir, f"{voice}.json")
        if exists(candidate):
            return candidate
    return os.path.join(voices_dir, f"{default_voice}.json")


def resample_ratio(src_rate: int, dst_rate: int) -> tuple[int, int]:
    """``(up, down)`` for a polyphase resample — (80, 147) for 44100 → 24000."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {src_rate} -> {dst_rate}")
    g = math.gcd(src_rate, dst_rate)
    return dst_rate // g, src_rate // g


def floats_to_pcm16(samples: Iterable[float]) -> bytes:
    """Clip to [-1, 1] and pack as PCM16 little-endian (symmetric ±32767)."""
    ints = [int(max(-1.0, min(1.0, float(s))) * 32767) for s in samples]
    return struct.pack(f"<{len(ints)}h", *ints)


def readiness_status(model_loaded: bool, warmup_ok: bool) -> tuple[int, dict]:
    if not model_loaded:
        return 503, {"status": "loading"}
    if not warmup_ok:
        return 503, {"status": "warming"}
    return 200, {"status": "ready"}


# ---------------------------------------------------------------------------
# Engine + FastAPI shell — needs numpy/scipy/onnxruntime/blue_onnx; container-only
# ---------------------------------------------------------------------------

try:  # pragma: no cover - the offline suite has none of these
    import anyio
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False

_settings = build_blue_settings()
_engines: dict[str, object] = {}
_engine_lock = threading.Lock()
_synth_lock = threading.Lock()
_warmup_ok = False

_WARMUP_TEXT = {"he": "שלום, אני כאן."}


def _get_engine(voice_path: str):  # pragma: no cover
    """One BlueTTS per voice style, built lazily; the ONNX graphs dominate the
    cost, so extra voices are cheap but not free — only requested ones load."""
    with _engine_lock:
        engine = _engines.get(voice_path)
        if engine is None:
            if _settings.threads:
                os.environ.setdefault("OMP_NUM_THREADS", str(_settings.threads))
            from blue_onnx import BlueTTS

            started = time.monotonic()
            engine = BlueTTS(
                onnx_dir=_settings.onnx_dir,
                style_json=voice_path,
                renikud_path=_settings.renikud_path,
                config_path=_settings.config_path,
            )
            _engines[voice_path] = engine
            log.info("[BlueTTS] loaded %s in %.2fs", voice_path, time.monotonic() - started)
        return engine


def _synthesize_pcm(text: str, voice: str | None) -> bytes:  # pragma: no cover
    import numpy as np
    from scipy.signal import resample_poly

    engine = _get_engine(resolve_voice_path(voice, _settings.voices_dir, _settings.voice))
    plain = strip_niqqud(text)
    started = time.monotonic()
    with _synth_lock:  # one onnxruntime run at a time; the bridge sends sentences serially
        audio, rate = engine.synthesize(
            plain,
            lang=_settings.language,
            total_step=_settings.total_step,
            speed=_settings.speed,
        )
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    up, down = resample_ratio(int(rate), TTS_SAMPLE_RATE)
    if (up, down) != (1, 1):
        audio = resample_poly(audio, up, down)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    log.info(
        "[BlueTTS] %d chars -> %.2fs audio in %d ms",
        len(plain),
        len(pcm) / 2 / TTS_SAMPLE_RATE,
        int((time.monotonic() - started) * 1000),
    )
    return pcm


if _FASTAPI_AVAILABLE:  # pragma: no cover
    app = FastAPI(title="lobes bluetts", version="1")

    @app.on_event("startup")
    async def _warm_model() -> None:
        def _warm() -> None:
            global _warmup_ok
            try:
                _synthesize_pcm(_WARMUP_TEXT.get(_settings.language, "hello."), None)
                _warmup_ok = True
                log.info("[BlueTTS] warm-up synthesis succeeded (G2P loaded) — ready")
            except Exception:  # noqa: BLE001 - readiness stays 503; never crash startup
                log.exception("[BlueTTS] warm-up synthesis failed — staying not-ready")

        threading.Thread(target=_warm, daemon=True).start()

    @app.get("/v1/health/ready")
    async def health() -> Response:
        code, body = readiness_status(bool(_engines), _warmup_ok)
        return JSONResponse(status_code=code, content=body)

    @app.post("/v1/audio/synthesize")
    async def synthesize(request_body: dict) -> Response:
        """Synthesize text to raw PCM16 mono 24 kHz bytes."""
        text = (request_body.get("text") or "").strip()
        if not text:
            return JSONResponse(
                status_code=400, content={"error": {"message": "text must be non-empty"}}
            )
        voice = request_body.get("voice") or None
        pcm = await anyio.to_thread.run_sync(_synthesize_pcm, text, voice)
        return Response(content=pcm, media_type="audio/pcm")


def main() -> None:  # pragma: no cover
    """Process entrypoint — ``python -m lobes.realtime.bluetts_server``."""
    logging.basicConfig(level=logging.INFO)
    log.info(
        "starting lobes bluetts on %s:%d (language=%s, voice=%s, steps=%d)",
        _settings.host,
        _settings.port,
        _settings.language,
        _settings.voice,
        _settings.total_step,
    )
    uvicorn.run(app, host=_settings.host, port=_settings.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
