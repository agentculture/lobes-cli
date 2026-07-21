"""Tests for the realtime settings builder (stdlib-only; no [realtime] extra)."""

from __future__ import annotations

from lobes.realtime._session import DEFAULT_SYSTEM_PROMPT
from lobes.realtime._settings import (
    BATCH_LANE,
    VOICE_LANE,
    build_settings,
    normalize_tts_lane,
    tts_concurrency_for_lane,
)


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
    assert s.tts_voice_concurrency == 1
    assert s.default_voice == ""  # Chatterbox default voice (empty = built-in)
    assert s.vad_max_turn_ms == 30_000
    assert s.default_system_prompt == DEFAULT_SYSTEM_PROMPT


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


# --- TTS concurrency lanes (issue #151 t7) ----------------------------------
#
# tts_client.py's _tts_semaphore used to be ONE module-global gate shared by
# every /v1/realtime voice session AND the batch POST /v1/audio/speech route
# — a voice reply could queue behind unrelated batch TTS work, and in a
# spoken turn that queueing IS dead air. The fix is two independent
# asyncio.Semaphore pools ("batch" and "voice"), not a single raised number:
# raising a SHARED ceiling only reduces how often batch traffic blocks a
# voice reply, it cannot guarantee a voice reply never waits behind a batch
# caller that fully saturates the pool. See test_realtime_tts_gate.py for the
# proof that the two built semaphores are actually independent.


def test_tts_voice_concurrency_default_and_override() -> None:
    assert build_settings({}).tts_voice_concurrency == 1
    assert build_settings({"TTS_VOICE_CONCURRENCY": "3"}).tts_voice_concurrency == 3


def test_tts_voice_concurrency_is_clamped_to_at_least_one() -> None:
    # Same danger as tts_concurrency/tts_speed above: Semaphore(0) (or
    # negative) blocks every voice-lane TTS request forever.
    assert build_settings({"TTS_VOICE_CONCURRENCY": "0"}).tts_voice_concurrency == 1
    assert build_settings({"TTS_VOICE_CONCURRENCY": "-5"}).tts_voice_concurrency == 1


def test_normalize_tts_lane_only_the_voice_literal_opts_in() -> None:
    # An existing caller that passes nothing, or any future typo, MUST
    # degrade to the long-serving batch behavior — never silently open a
    # brand-new, unbounded pool. Only the exact string "voice" opts in.
    assert normalize_tts_lane("voice") == VOICE_LANE
    assert normalize_tts_lane("batch") == BATCH_LANE
    assert normalize_tts_lane(None) == BATCH_LANE
    assert normalize_tts_lane("") == BATCH_LANE
    assert normalize_tts_lane("VOICE") == BATCH_LANE  # exact match only
    assert normalize_tts_lane("typo") == BATCH_LANE


def test_tts_concurrency_for_lane_reads_the_matching_settings_field() -> None:
    s = build_settings({"TTS_CONCURRENCY": "3", "TTS_VOICE_CONCURRENCY": "5"})
    assert tts_concurrency_for_lane(s, BATCH_LANE) == 3
    assert tts_concurrency_for_lane(s, VOICE_LANE) == 5
    assert tts_concurrency_for_lane(s, None) == 3  # unknown/default lane → batch
    assert tts_concurrency_for_lane(s, "typo") == 3


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


# --- default_system_prompt (issue #151 t8) ----------------------------------
#
# The OPERATOR half of "an operator-set default system prompt via env and a
# per-session override in the connect config" (spec c34) — the per-session
# override half (SessionConfig.system_prompt / parse_session_config's
# default_system_prompt parameter) already existed from #151 t3. This field
# is what a caller (app.py, a later task) threads into that parameter.


def test_default_system_prompt_mirrors_the_session_default_when_unset() -> None:
    # An operator who never sets DEFAULT_SYSTEM_PROMPT must get exactly the
    # pre-#151 in-code fallback back — not a blank prompt, which would let a
    # reply revert to the model's normal WRITTEN register (markdown, bullet
    # lists, code fences) that a TTS voice then reads aloud verbatim.
    assert build_settings({}).default_system_prompt == DEFAULT_SYSTEM_PROMPT
    assert build_settings({}).default_system_prompt != ""


def test_default_system_prompt_operator_override() -> None:
    s = build_settings({"DEFAULT_SYSTEM_PROMPT": "You are terse. One word answers only."})
    assert s.default_system_prompt == "You are terse. One word answers only."


def test_default_system_prompt_blank_env_falls_back_to_the_mirrored_default() -> None:
    # Same "empty string in .env means unset" idiom as every other string
    # field in this module (e.g. OPENAI_MODEL, DEFAULT_VOICE) — a blank line
    # left in .env must not silently ship a blank system prompt.
    assert build_settings({"DEFAULT_SYSTEM_PROMPT": ""}).default_system_prompt == (
        DEFAULT_SYSTEM_PROMPT
    )
