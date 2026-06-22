"""Tests for the realtime settings builder (stdlib-only; no [realtime] extra)."""

from __future__ import annotations

from lobes.realtime._settings import build_settings


def test_defaults_point_at_the_fleet_compose_network() -> None:
    s = build_settings({})
    assert s.tts_url == "http://chatterbox:9000"
    assert s.stt_url == "http://stt:9002"
    assert s.openai_base_url == "http://gateway:8000"
    assert s.openai_model == ""  # empty → the gateway default-routes the LLM
    assert s.openai_api_key == "EMPTY"
    assert s.port == 8080
    assert s.tts_speed == 125
    assert s.tts_concurrency == 1
    assert s.default_voice == ""  # Chatterbox default voice (empty = built-in)


def test_overrides_and_trailing_slash_stripped() -> None:
    s = build_settings(
        {
            "TTS_URL": "http://chatterbox:1/",
            "STT_URL": "http://stt:2/",
            "OPENAI_BASE_URL": "http://gw:8000/",
            "OPENAI_MODEL": "mmangkad/Qwen3.6-27B-NVFP4",
            "REALTIME_PORT": "9090",
            "TTS_SPEED": "100",
            "DEFAULT_VOICE": "/data/voices/ref.wav",
        }
    )
    assert s.tts_url == "http://chatterbox:1"
    assert s.stt_url == "http://stt:2"
    assert s.openai_base_url == "http://gw:8000"
    assert s.openai_model == "mmangkad/Qwen3.6-27B-NVFP4"
    assert s.port == 9090
    assert s.tts_speed == 100
    assert s.default_voice == "/data/voices/ref.wav"


def test_bad_numbers_fall_back_to_defaults() -> None:
    s = build_settings({"REALTIME_PORT": "notanint", "TTS_SPEED": "", "VAD_THRESHOLD": "x"})
    assert s.port == 8080
    assert s.tts_speed == 125
    assert s.vad_threshold == 0.5


def test_tts_concurrency_is_clamped_to_at_least_one() -> None:
    # Semaphore(0)/Semaphore(<0) would block every TTS request forever.
    assert build_settings({"TTS_CONCURRENCY": "0"}).tts_concurrency == 1
    assert build_settings({"TTS_CONCURRENCY": "-5"}).tts_concurrency == 1
    assert build_settings({"TTS_CONCURRENCY": "4"}).tts_concurrency == 4


def test_tts_speed_is_clamped_to_at_least_one() -> None:
    # A 0/negative percentage would emit nonsensical values; clamp to 1.
    assert build_settings({"TTS_SPEED": "0"}).tts_speed == 1
    assert build_settings({"TTS_SPEED": "-100"}).tts_speed == 1
    assert build_settings({"TTS_SPEED": "150"}).tts_speed == 150
