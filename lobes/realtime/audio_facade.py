"""Pure helpers for the OpenAI ``/v1/audio/*`` facade — stdlib only.

The FastAPI routes live in :mod:`lobes.realtime.app` (which needs the
``[realtime]`` extra). The *logic* — parsing a ``/v1/audio/speech`` request and
wrapping raw PCM16 in an audio container — lives here so it is importable and
unit-testable without fastapi/httpx (the offline dev env has neither).

Chatterbox returns raw PCM16 mono @ 24000 Hz. We expose two OpenAI ``response_format``
values: ``wav`` (default; a self-describing container) and ``pcm`` (raw bytes,
24000 Hz). ``mp3``/``opus``/``aac``/``flac`` need an encoder (ffmpeg) and are a
documented follow-up — they return HTTP 400 for now.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from .protocol import TTS_SAMPLE_RATE

SUPPORTED_FORMATS = ("wav", "pcm")
_MEDIA_TYPE = {"wav": "audio/wav", "pcm": "audio/pcm"}

# OpenAI's documented /v1/audio/speech ``speed`` multiplier range. Out-of-range
# values are clamped (not rejected) so naive callers keep working — mirroring the
# response_format-defaults-to-wav philosophy above.
_OPENAI_SPEED_MIN = 0.25
_OPENAI_SPEED_MAX = 4.0


class SpeechRequestError(ValueError):
    """The ``/v1/audio/speech`` request was invalid or unsupported (→ HTTP 400)."""


@dataclass(frozen=True)
class SpeechParams:
    """A validated ``/v1/audio/speech`` request (voice/speed still optional)."""

    input: str
    voice: str | None
    response_format: str
    speed: int | None  # speed percentage (100 = normal); None → service default


def parse_speech_request(body: object) -> SpeechParams:
    """Validate an OpenAI ``/v1/audio/speech`` JSON body. Raises on bad input.

    ``input`` is required. ``response_format`` defaults to ``wav`` (OpenAI's
    default is mp3, which we cannot encode yet — defaulting to wav keeps naive
    callers working instead of 400ing them). OpenAI ``speed`` is a 0.25–4.0
    multiplier (clamped to that range here); converted to a percentage so
    ``1.0 → 100`` (accepted for API compatibility; Chatterbox ignores it).
    """
    if not isinstance(body, dict):
        raise SpeechRequestError("request body must be a JSON object")
    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        raise SpeechRequestError("'input' is required and must be a non-empty string")
    voice = body.get("voice")
    if voice is not None and not isinstance(voice, str):
        raise SpeechRequestError("'voice' must be a string")
    fmt = body.get("response_format") or "wav"
    if not isinstance(fmt, str):
        raise SpeechRequestError("'response_format' must be a string")
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise SpeechRequestError(
            f"unsupported response_format {fmt!r}; supported: {', '.join(SUPPORTED_FORMATS)} "
            "(mp3/opus/aac/flac need an encoder — not yet available)"
        )
    raw_speed = body.get("speed")
    speed: int | None = None
    if raw_speed is not None:
        try:
            multiplier = float(raw_speed)
        except (TypeError, ValueError):
            raise SpeechRequestError("'speed' must be a number") from None
        # Clamp to OpenAI's 0.25–4.0 range before converting to a percentage.
        multiplier = max(_OPENAI_SPEED_MIN, min(_OPENAI_SPEED_MAX, multiplier))
        speed = int(round(multiplier * 100))
    return SpeechParams(input=text, voice=voice, response_format=fmt, speed=speed)


def pcm_to_container(
    pcm: bytes, response_format: str, rate: int = TTS_SAMPLE_RATE
) -> tuple[bytes, str]:
    """Wrap raw PCM16 mono bytes in the requested container.

    Returns ``(bytes, media_type)``. ``wav`` builds a self-describing RIFF/WAVE
    container (stdlib :mod:`wave`); ``pcm`` returns the raw bytes untouched.
    """
    fmt = response_format.lower()
    if fmt == "pcm":
        return pcm, _MEDIA_TYPE["pcm"]
    if fmt == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return buf.getvalue(), _MEDIA_TYPE["wav"]
    raise SpeechRequestError(
        f"unsupported response_format {fmt!r}; supported: {', '.join(SUPPORTED_FORMATS)}"
    )
