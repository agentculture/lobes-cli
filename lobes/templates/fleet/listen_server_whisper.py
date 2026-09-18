#!/usr/bin/env python3
"""Whisper ASR HTTP server for the Hebrew realtime overlay (hebrew-realtime t10).

Serves ``STT_MODEL`` (default ``ivrit-ai/whisper-large-v3-turbo``) as an ASR
API with the SAME response shape and route surface as
``listen_server.py`` (the Parakeet server), so the realtime bridge and the
gateway need no change: ``STT_URL=http://stt:${PARAKEET_PORT:-9002}`` keeps
working regardless of which server the ``stt`` container runs.

The runtime here is **transformers + PyTorch**, not NeMo — measured live on
the DGX Spark GB10 (docs/evidence/2026-09-hebrew-stt-spike-spark.txt): 1.56
GiB peak CUDA, ~0.14s per 9.35s clip, transformers 5.12.1 / torch
2.11.0+cu130 inside the same base image ``Dockerfile.whisper-stt`` reuses
from ``Dockerfile.parakeet``.

Endpoints:
    POST /v1/audio/transcriptions  - Transcribe an uploaded WAV file
    GET  /v1/health/ready          - Readiness (model loaded + CUDA live +
                                      a warm-up transcription succeeded)

Structure (mirrors listen_server.py): every PURE helper below (WAV decode,
resample, channel selection, clip-duration validation, language resolution,
the non-speech tag filter, and request/response shaping) imports with ZERO
third-party dependencies — stdlib only (``wave``, ``struct``, ``io``, ``re``,
``math``). torch / transformers / fastapi / uvicorn are imported lazily
(inside functions, or behind the ``_FASTAPI_AVAILABLE`` guard below) so this
module stays importable in the offline test environment where none of those
packages are installed — the same reason ``lobes/templates/*`` is excluded
from the coverage run (see pyproject.toml).
"""

from __future__ import annotations

import io
import logging
import math
import os
import re
import struct
import wave
from typing import Optional, Sequence

# Import the readiness decision from the single source of truth — the exact
# convention listen_server.py uses. The Dockerfile COPYs _readiness.py next
# to this file (top-level module in /app); the wheel path is a dev fallback.
try:
    from _readiness import evaluate_readiness  # container-local copy (top-level)
except ImportError:
    from lobes.realtime._readiness import evaluate_readiness  # wheel install


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Whisper's attention window is 30s; clips longer than that are REFUSED (a
# clear 4xx) rather than chunked-and-joined. Two reasons: (1) chunk-joining
# introduces boundary artifacts/duplicated words at the seam, and (2) the
# realtime bridge's own VAD_MAX_TURN_MS defaults to exactly 30000ms
# (CLAUDE.md's #151 material) — a well-behaved caller never sends a clip
# longer than this window in the first place, so refusing costs nothing in
# practice and avoids a whole class of hallucination risk at the seam.
MAX_CLIP_SECONDS = 30.0
MODEL_NAME = os.environ.get("STT_MODEL", "ivrit-ai/whisper-large-v3-turbo")
DEFAULT_LANGUAGE = "he"

# Short BCP-47-ish language code: 2-3 letters, optional "-REGION"/"-Script"
# subtag. Deliberately permissive (this gates against garbage, not a full
# BCP-47 validator) — "he", "en", "en-US" all pass.
_LANGUAGE_CODE_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$")

# A "tag" is a bracketed/parenthesised aside: (...) or [...], no nesting.
_BRACKET_TAG_RE = re.compile(r"[\(\[][^\(\)\[\]]*[\)\]]")
# Whitespace and light punctuation a caller might leave between/around tags
# ("(laughter) (coughing)", "[music]... [applause]") — stripped when deciding
# whether anything OTHER than tags remains.
_FILLER_RE = re.compile(r"[\s.,;:!?\-–—־׃]+")


# --------------------------------------------------------------------------
# Pure helpers — no torch/transformers/fastapi import anywhere below this
# line until the ``if _FASTAPI_AVAILABLE:`` app-building block.
# --------------------------------------------------------------------------


def is_valid_language_code(code: Optional[str]) -> bool:
    """True iff *code* is a short, plausible language code (not empty/garbage)."""
    return bool(code) and bool(_LANGUAGE_CODE_RE.fullmatch(code))


def resolve_language(
    form_language: Optional[str],
    env_language: Optional[str],
    default: str = DEFAULT_LANGUAGE,
) -> str:
    """Resolve the transcription language: form field wins, then ``STT_LANGUAGE``,
    then *default*. A candidate that fails :func:`is_valid_language_code` is
    skipped rather than raising — a stray/garbage value falls through to the
    next source instead of failing the whole request. *default* itself must
    be a valid code (``"he"`` is)."""
    for candidate in (form_language, env_language, default):
        if is_valid_language_code(candidate):
            return candidate  # type: ignore[return-value]
    return default


# Unicode bidirectional CONTROL characters (LRM/RLM/ALM, the embedding/override
# set, the isolate set). Measured live 2026-09-18: a Hebrew transcript came
# back as ' \u202bתודה רבה.' - invisible, but it breaks string equality, tool
# arguments and the TTS text path downstream.
_BIDI_CONTROLS = dict.fromkeys(
    [0x200E, 0x200F, 0x061C, *range(0x202A, 0x202F), *range(0x2066, 0x206A)]
)


# Confidence gate against Whisper's plain-text hallucinations on non-speech.
# MEASURED on the DGX Spark 2026-09-18 with the operator's recordings
# (docs/evidence/2026-09-hebrew-stt-spike-spark.txt): every real utterance
# scored an average token log-probability between -0.00 and -0.12, every
# hallucination -0.58 or lower (the famous plain-text 'תודה רבה' at -0.60 and
# -0.58; bracketed tags at -0.61 and -1.50). This checkpoint's <|nospeech|>
# probability read 0.000 on EVERY clip, so it is useless as a gate here. The
# default sits midway; n = 16 clips, one speaker — it is a knob, not a law.
DEFAULT_MIN_AVG_LOGPROB = -0.35


def parse_min_avg_logprob(raw: str | None) -> float | None:
    """``STT_MIN_AVG_LOGPROB``: a float threshold; ``off`` / ``none`` / empty
    string disables the gate; anything unparseable falls back to the default
    (a typo must not silently disable a safety gate)."""
    if raw is None:
        return DEFAULT_MIN_AVG_LOGPROB
    value = raw.strip().lower()
    if value in ("", "off", "none", "disabled"):
        return None
    try:
        return float(value)
    except ValueError:
        return DEFAULT_MIN_AVG_LOGPROB


def average_logprob(token_logprobs: list[float]) -> float | None:
    """Mean log-probability of the generated TEXT tokens; ``None`` when there
    were none (an empty transcript has no confidence to judge)."""
    if not token_logprobs:
        return None
    return sum(token_logprobs) / len(token_logprobs)


def is_low_confidence(avg_logprob: float | None, threshold: float | None) -> bool:
    """True when the transcript should be dropped as a likely hallucination.
    A disabled gate (``threshold is None``) or an unknown confidence never
    drops anything."""
    if threshold is None or avg_logprob is None:
        return False
    return avg_logprob < threshold


MIN_AVG_LOGPROB = parse_min_avg_logprob(os.environ.get("STT_MIN_AVG_LOGPROB"))


def strip_bidi_controls(text: str) -> str:
    """Remove invisible bidi control characters; every visible character,
    Hebrew or not, is left exactly as it was."""
    return text.translate(_BIDI_CONTROLS)


def filter_non_speech_only(text: str) -> str:
    """Return ``""`` iff *text* consists ONLY of one or more bracketed/
    parenthesised tags (round or square brackets, no nesting), with only
    whitespace/light punctuation between or around them — the shape
    transformers-Whisper emits for a non-silent "silent" clip (measured:
    ``'(צחוק)'``, ``'(הורות)'`` where vLLM returned empty text). A bracketed
    aside INSIDE real speech (``'שלום (צחוק) עולם'``) is left completely
    alone — this only fires when tags are the WHOLE transcript."""
    stripped = text.strip()
    if not stripped:
        return text
    tags = _BRACKET_TAG_RE.findall(stripped)
    if not tags:
        return text
    remainder = _FILLER_RE.sub("", _BRACKET_TAG_RE.sub("", stripped))
    return "" if remainder == "" else text


def decode_wav_pcm16(raw_bytes: bytes) -> tuple[list[int], int, int]:
    """Decode 16-bit PCM WAV *raw_bytes* into ``(samples, sample_rate,
    channels)``. *samples* is interleaved (frame 0 ch0, frame 0 ch1, ...).
    Raises :class:`ValueError` for a non-16-bit-PCM file."""
    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"unsupported WAV sample width: {sampwidth * 8}-bit (need 16-bit PCM)")
    count = len(frames) // 2
    samples = list(struct.unpack(f"<{count}h", frames[: count * 2]))
    return samples, sample_rate, channels


def first_channel(samples: Sequence[int], channels: int) -> list[int]:
    """De-interleave *samples* and return channel 0 only.

    Deliberately NOT an average-of-channels downmix: for the reSpeaker mic
    array this repo targets, channel 0 is the measured-good channel and
    averaging mixes in the others' AEC residuals (see CLAUDE.md's audio
    hardware-check material). The CLIENT is responsible for channel choice —
    this function only implements "take the first one"."""
    if channels <= 1:
        return list(samples)
    return list(samples[0::channels])


def linear_resample(samples: Sequence[float], src_rate: int, dst_rate: int) -> list[float]:
    """A small linear resampler (no numpy/torchaudio dependency).

    Used only when torchaudio is not importable in the container; when it
    is, the heavy server code prefers it. Pure and exactly reproducible —
    deliberately simple, not a windowed-sinc resampler."""
    if src_rate == dst_rate or not samples:
        return list(samples)
    n_in = len(samples)
    duration = (n_in - 1) / src_rate if n_in > 1 else 0.0
    n_out = max(1, round(duration * dst_rate) + 1)
    out: list[float] = []
    for i in range(n_out):
        src_pos = (i / dst_rate) * src_rate if n_out > 1 else 0.0
        idx0 = min(int(math.floor(src_pos)), n_in - 1)
        idx1 = min(idx0 + 1, n_in - 1)
        frac = src_pos - idx0
        out.append(samples[idx0] * (1 - frac) + samples[idx1] * frac)
    return out


def _clip_too_long_message(duration: float, max_seconds: float) -> str:
    """Shared wording for a refused clip, whichever check found it too long
    (the post-decode sample-count check, or the pre-decode WAV-header peek)."""
    return (
        f"clip too long: {duration:.1f}s exceeds the {max_seconds:.0f}s "
        "Whisper window (refused, not chunked — see MAX_CLIP_SECONDS)"
    )


def validate_clip_duration(
    num_samples: int, sample_rate: int, max_seconds: float = MAX_CLIP_SECONDS
) -> Optional[str]:
    """Return an error message iff the clip exceeds *max_seconds*, else
    ``None``. See the module docstring / ``MAX_CLIP_SECONDS`` comment for why
    this refuses instead of chunking."""
    if sample_rate <= 0:
        return "invalid sample rate"
    duration = num_samples / sample_rate
    if duration > max_seconds:
        return _clip_too_long_message(duration, max_seconds)
    return None


# Cheap upload-size cap, checked BEFORE any decoding (Qodo finding: reading
# the whole multipart body and decoding every frame into Python objects
# before the 30s duration limit is applied lets an oversized upload exhaust a
# speech worker). 30s of 48kHz stereo PCM16 is ~5.8MB, so 16 MiB is generous
# headroom above any legitimate clip while still bounding memory; the
# realtime bridge's own turn audio (max 30s at 16kHz mono, ~1MB) is well
# under it either way.
DEFAULT_MAX_UPLOAD_BYTES = 16 * 1024 * 1024


def parse_max_upload_bytes(raw: str | None) -> int:
    """``STT_MAX_UPLOAD_BYTES``: a positive integer byte cap; empty/unset or
    an unparseable/non-positive value falls back to the default (a typo must
    not silently disable the cap)."""
    if raw is None:
        return DEFAULT_MAX_UPLOAD_BYTES
    value = raw.strip()
    if not value:
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_MAX_UPLOAD_BYTES
    return parsed if parsed > 0 else DEFAULT_MAX_UPLOAD_BYTES


def check_upload_size(num_bytes: int, max_bytes: int) -> Optional[str]:
    """Return an error message iff *num_bytes* exceeds *max_bytes*, else
    ``None``. Cheap: a plain integer comparison, called before any WAV
    parsing or frame decoding."""
    if num_bytes > max_bytes:
        return (
            f"upload too large: {num_bytes} bytes exceeds the {max_bytes} byte "
            "cap (STT_MAX_UPLOAD_BYTES)"
        )
    return None


MAX_UPLOAD_BYTES = parse_max_upload_bytes(os.environ.get("STT_MAX_UPLOAD_BYTES"))


def peek_wav_duration_seconds(raw_bytes: bytes) -> Optional[float]:
    """Clip duration (seconds) read from the WAV HEADER alone — ``wave``'s
    ``getnframes()``/``getframerate()`` are header reads, not a frame decode
    (:func:`decode_wav_pcm16` does the actual ``readframes()``). Lets the
    duration limit reject a pathological header (a huge declared frame count)
    before any frame is read into Python objects. Returns ``None`` when the
    header can't be parsed — the caller falls through to the full decode,
    which raises its own, more specific error."""
    try:
        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
    except (wave.Error, EOFError):
        return None
    if rate <= 0:
        return None
    return frames / rate


def build_success_response(text: str) -> dict:
    """Same top-level shape as ``listen_server.py``'s ``return {"text": text}``."""
    return {"text": text}


def build_error_body(message: str, code: str) -> dict:
    """Error body shape for this server's own request-validation failures
    (invalid WAV, clip too long). ``listen_server.py`` has no equivalent
    validation path — this shape is new, not a parity requirement."""
    return {"error": {"message": message, "code": code}}


def build_readiness_body(
    base_body: dict, *, model_loaded: bool, cuda_ok: bool, model_name: str
) -> dict:
    """Layer ``model_loaded``/``cuda_ok``/``model`` onto ``evaluate_readiness``'s
    ``base_body`` WITHOUT editing ``_readiness.py`` — its ``{"status": ...,
    ["reason"]: ...}`` shape has no field for the model id, so this function
    adds it here instead, keeping ``_readiness.py``'s decision logic (and its
    parity with the canonical ``lobes/realtime/_readiness.py``) untouched."""
    return {**base_body, "model_loaded": model_loaded, "cuda_ok": cuda_ok, "model": model_name}


# --------------------------------------------------------------------------
# Heavy app — fastapi/torch/transformers. Guarded so the pure helpers above
# stay importable offline (no fastapi/torch/transformers in the dev/test
# environment; see pyproject.toml's coverage omit list for lobes/templates/*).
# --------------------------------------------------------------------------

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in the container
    _FASTAPI_AVAILABLE = False

if _FASTAPI_AVAILABLE:
    app = FastAPI(title="Whisper ASR (Hebrew overlay)")

    _model = None
    _processor = None
    _warmup_ok = False

    def get_model_and_processor():
        """Lazily load the Whisper model + processor onto CUDA, fp16."""
        global _model, _processor
        if _model is None:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            logger.info("Loading model %s...", MODEL_NAME)
            # nosec B615 — MODEL_NAME is STT_MODEL, an operator-set deployment
            # env var (docker-compose.audio-he.yml), not user/request input;
            # no revision pin was specified for this checkpoint by the task
            # brief, matching Dockerfile.parakeet's own unpinned
            # from_pretrained() call for the Parakeet checkpoint.
            _processor = WhisperProcessor.from_pretrained(MODEL_NAME)  # nosec B615
            _model = (
                WhisperForConditionalGeneration.from_pretrained(  # nosec B615
                    MODEL_NAME, torch_dtype=torch.float16
                )
                .to("cuda")
                .eval()
            )
            logger.info("Model loaded.")
        return _model, _processor

    def _run_warmup() -> bool:
        """A tiny (1s of zeros) transcription on CUDA — readiness stays
        non-200 until this succeeds, not merely until the model object
        exists (issue t10 criterion 1: "non-200 until a transcription can
        succeed")."""
        global _warmup_ok
        try:
            import numpy as np
            import torch

            model, processor = get_model_and_processor()
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            inputs = processor(silence, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_features = inputs.input_features.to("cuda", dtype=torch.float16)
            with torch.no_grad():
                model.generate(
                    input_features,
                    language=DEFAULT_LANGUAGE,
                    task="transcribe",
                    max_new_tokens=8,
                )
            _warmup_ok = True
        except Exception as exc:  # pragma: no cover - exercised only in the container
            logger.warning("Warm-up transcription failed: %s: %s", type(exc).__name__, exc)
            _warmup_ok = False
        return _warmup_ok

    @app.on_event("startup")
    async def startup():
        get_model_and_processor()
        _run_warmup()

    @app.get("/v1/health/ready")
    async def health():
        """Readiness: model loaded AND CUDA live AND the warm-up transcription
        succeeded. Reuses ``evaluate_readiness``'s decision logic exactly (its
        two checks) by folding the warm-up result into the ``model_loaded``
        flag it receives — a "loaded but never actually transcribed" model
        does not count as loaded for this endpoint."""
        model_ready = _model is not None and _warmup_ok

        try:
            import torch

            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            cuda_ok = True
        except Exception as exc:
            logger.warning("CUDA readiness probe failed: %s: %s", type(exc).__name__, exc)
            cuda_ok = False

        status_code, base_body = evaluate_readiness(model_ready, cuda_ok)
        body = build_readiness_body(
            base_body, model_loaded=model_ready, cuda_ok=cuda_ok, model_name=MODEL_NAME
        )
        return JSONResponse(status_code=status_code, content=body)

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        language: str = Form(None),
    ):
        """Transcribe an uploaded WAV file (PCM16 mono/stereo 16kHz — the
        realtime bridge's own output shape). See the module docstring for the
        channel/resample/duration-limit contract."""
        content = await file.read()

        size_error = check_upload_size(len(content), MAX_UPLOAD_BYTES)
        if size_error is not None:
            return JSONResponse(
                status_code=413, content=build_error_body(size_error, "upload_too_large")
            )

        header_duration = peek_wav_duration_seconds(content)
        if header_duration is not None and header_duration > MAX_CLIP_SECONDS:
            return JSONResponse(
                status_code=413,
                content=build_error_body(
                    _clip_too_long_message(header_duration, MAX_CLIP_SECONDS), "clip_too_long"
                ),
            )

        try:
            samples, sample_rate, channels = decode_wav_pcm16(content)
        except (ValueError, wave.Error) as exc:
            return JSONResponse(status_code=400, content=build_error_body(str(exc), "invalid_wav"))

        samples = first_channel(samples, channels)

        duration_error = validate_clip_duration(len(samples), sample_rate)
        if duration_error is not None:
            return JSONResponse(
                status_code=413, content=build_error_body(duration_error, "clip_too_long")
            )

        import numpy as np
        import torch

        floats = [s / 32768.0 for s in samples]
        if sample_rate != SAMPLE_RATE:
            try:
                import torchaudio

                waveform = torch.tensor([floats], dtype=torch.float32)
                resampled = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
                floats = resampled[0].tolist()
            except ImportError:
                floats = linear_resample(floats, sample_rate, SAMPLE_RATE)

        lang = resolve_language(language, os.environ.get("STT_LANGUAGE"), DEFAULT_LANGUAGE)

        model, processor = get_model_and_processor()
        audio = np.array(floats, dtype=np.float32)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to("cuda", dtype=torch.float16)
        with torch.no_grad():
            out = model.generate(
                input_features,
                language=lang,
                task="transcribe",
                max_new_tokens=128,
                return_dict_in_generate=True,
                output_scores=True,
            )
        n_new = len(out.scores)
        new_ids = out.sequences[0][-n_new:] if n_new else out.sequences[0][:0]
        eot = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        token_logprobs = [
            torch.log_softmax(step[0].float(), dim=-1)[tok].item()
            for step, tok in zip(out.scores, new_ids)
            if tok.item() < eot  # text tokens only — specials sit at/after <|endoftext|>
        ]
        text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
        # Whisper's decode leaves a leading space (measured live: ' מה מזג ...');
        # listen_server.py's callers never see one, so strip before filtering.
        text = filter_non_speech_only(strip_bidi_controls(text).strip())
        confidence = average_logprob(token_logprobs)
        if text and is_low_confidence(confidence, MIN_AVG_LOGPROB):
            logger.info(
                "dropping low-confidence transcript (avg_logprob %.2f < %.2f): %r",
                confidence,
                MIN_AVG_LOGPROB,
                text,
            )
            text = ""

        return build_success_response(text)


if __name__ == "__main__":  # pragma: no cover - exercised only in the container
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "fastapi is not installed — this entrypoint must run inside the container"
        )
    import uvicorn

    port = int(os.environ.get("PARAKEET_PORT", "9002"))
    uvicorn.run(app, host="0.0.0.0", port=port)  # nosec B104 — bind all inside the container
