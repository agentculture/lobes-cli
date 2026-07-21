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
    assert s.vad_max_turn_ms == 30_000


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


def test_vad_max_turn_ms_default_and_override() -> None:
    # Default mirrors _segmenter.py's DEFAULT_MAX_TURN_MS (30_000) — the two
    # modules agree on the same number without importing each other (#149 t6).
    assert build_settings({}).vad_max_turn_ms == 30_000
    assert build_settings({"VAD_MAX_TURN_MS": "15000"}).vad_max_turn_ms == 15000


def test_vad_max_turn_ms_bad_value_falls_back_to_default() -> None:
    assert build_settings({"VAD_MAX_TURN_MS": "not-a-number"}).vad_max_turn_ms == 30_000


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


def test_vad_max_turn_ms_is_clamped_to_a_sane_floor() -> None:
    """A non-positive cap would force-commit every chunk forever.

    The segmenter commits once a turn reaches ``max_turn_ms``, so 0 or a
    negative value turns every single 32 ms chunk into a boundary pair plus an
    STT forward — an event storm from one typo in `.env`.
    """
    for bad in ("0", "-1", "-30000"):
        assert build_settings({"VAD_MAX_TURN_MS": bad}).vad_max_turn_ms == 1_000
    # Garbage falls back to the default, then clamps like anything else.
    assert build_settings({"VAD_MAX_TURN_MS": "not-a-number"}).vad_max_turn_ms == 30_000
    # A legitimate operator value passes through untouched.
    assert build_settings({"VAD_MAX_TURN_MS": "45000"}).vad_max_turn_ms == 45_000
