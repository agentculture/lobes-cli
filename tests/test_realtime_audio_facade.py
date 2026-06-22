"""Tests for the OpenAI /v1/audio/* pure helpers (stdlib-only; no [realtime] extra).

The FastAPI routes (lobes/realtime/app.py) need fastapi/httpx and are not
installed offline; the logic they delegate to lives here and is tested directly.
"""

from __future__ import annotations

import io
import wave

import pytest

from lobes.realtime.audio_facade import (
    SUPPORTED_FORMATS,
    SpeechRequestError,
    parse_speech_request,
    pcm_to_container,
)
from lobes.realtime.protocol import TTS_SAMPLE_RATE

# --- pcm_to_container -----------------------------------------------------


def test_pcm_passthrough_is_untouched() -> None:
    data, media_type = pcm_to_container(b"\x01\x02\x03\x04", "pcm")
    assert data == b"\x01\x02\x03\x04"
    assert media_type == "audio/pcm"


def test_wav_wraps_pcm_in_a_self_describing_container() -> None:
    pcm = b"\x00\x00\x10\x20" * 50
    data, media_type = pcm_to_container(pcm, "wav", rate=24000)
    assert media_type == "audio/wav"
    with wave.open(io.BytesIO(data)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 24000
        assert wf.readframes(wf.getnframes()) == pcm


def test_wav_default_rate_uses_tts_sample_rate() -> None:
    """pcm_to_container with no explicit rate must use TTS_SAMPLE_RATE (24000)."""
    pcm = b"\x00\x00" * 100
    data, _ = pcm_to_container(pcm, "wav")
    with wave.open(io.BytesIO(data)) as wf:
        assert wf.getframerate() == TTS_SAMPLE_RATE
        assert TTS_SAMPLE_RATE == 24000


def test_unsupported_container_format_raises() -> None:
    with pytest.raises(SpeechRequestError):
        pcm_to_container(b"\x00\x00", "mp3")


def test_supported_formats_are_wav_and_pcm() -> None:
    assert set(SUPPORTED_FORMATS) == {"wav", "pcm"}


# --- parse_speech_request -------------------------------------------------


def test_minimal_request_defaults_to_wav() -> None:
    p = parse_speech_request({"input": "Reachy is online."})
    assert p.input == "Reachy is online."
    assert p.response_format == "wav"  # default (OpenAI default mp3 isn't encodable yet)
    assert p.voice is None
    assert p.speed is None


def test_voice_format_and_speed_multiplier() -> None:
    p = parse_speech_request(
        {"input": "hi", "voice": "alloy", "response_format": "PCM", "speed": 1.25}
    )
    assert p.voice == "alloy"
    assert p.response_format == "pcm"  # lower-cased
    assert p.speed == 125  # OpenAI 1.25x → speed percentage


def test_missing_input_is_rejected() -> None:
    with pytest.raises(SpeechRequestError):
        parse_speech_request({"voice": "alloy"})


def test_blank_input_is_rejected() -> None:
    with pytest.raises(SpeechRequestError):
        parse_speech_request({"input": "   "})


def test_unsupported_response_format_is_rejected_early() -> None:
    with pytest.raises(SpeechRequestError):
        parse_speech_request({"input": "hi", "response_format": "mp3"})


def test_non_object_body_is_rejected() -> None:
    with pytest.raises(SpeechRequestError):
        parse_speech_request(["not", "a", "dict"])


def test_non_string_voice_is_rejected() -> None:
    with pytest.raises(SpeechRequestError):
        parse_speech_request({"input": "hi", "voice": 123})


def test_non_string_response_format_is_rejected() -> None:
    with pytest.raises(SpeechRequestError):
        parse_speech_request({"input": "hi", "response_format": 7})


def test_non_numeric_speed_is_rejected() -> None:
    with pytest.raises(SpeechRequestError):
        parse_speech_request({"input": "hi", "speed": "fast"})


def test_speed_within_range_is_converted_unchanged() -> None:
    assert parse_speech_request({"input": "hi", "speed": 0.25}).speed == 25
    assert parse_speech_request({"input": "hi", "speed": 4.0}).speed == 400


def test_too_fast_speed_is_clamped_to_the_max() -> None:
    # OpenAI's max is 4.0 (→ 400%); a huge value must be clamped before forwarding.
    assert parse_speech_request({"input": "hi", "speed": 100}).speed == 400


def test_negative_or_tiny_speed_is_clamped_to_the_min() -> None:
    # Below 0.25 (→ 25%); negatives would otherwise emit SSML rate="-…%".
    assert parse_speech_request({"input": "hi", "speed": -3}).speed == 25
    assert parse_speech_request({"input": "hi", "speed": 0.01}).speed == 25
