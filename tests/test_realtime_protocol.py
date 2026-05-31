"""Tests for the realtime protocol helpers (stdlib-only; no [realtime] extra).

IDs, audio constants, and the OpenAI→Magpie voice mapping import without
fastapi/httpx, so they are unit-tested here directly.
"""

from __future__ import annotations

from model_gear.realtime import protocol as P

# --- ID generation --------------------------------------------------------


def test_id_generators_carry_their_prefix_and_are_unique() -> None:
    cases = {
        "event_": P.gen_event_id,
        "item_": P.gen_item_id,
        "resp_": P.gen_response_id,
        "sess_": P.gen_session_id,
        "part_": P.gen_content_part_id,
    }
    for prefix, gen in cases.items():
        a, b = gen(), gen()
        assert a.startswith(prefix) and b.startswith(prefix)
        assert a != b  # uuid4-backed → distinct per call
        # 24 hex chars after the prefix
        assert len(a) == len(prefix) + 24


def test_timestamp_ms_is_monotonic_non_negative_int() -> None:
    t1 = P.timestamp_ms()
    t2 = P.timestamp_ms()
    assert isinstance(t1, int) and t1 >= 0
    assert t2 >= t1  # time.monotonic never goes backwards


# --- audio constants ------------------------------------------------------


def test_audio_constants_match_the_backends() -> None:
    assert P.CLIENT_SAMPLE_RATE == 24000  # OpenAI Realtime PCM16
    assert P.TTS_SAMPLE_RATE == 22050  # Magpie output
    assert P.STT_SAMPLE_RATE == 16000  # Parakeet input
    assert P.VAD_SAMPLE_RATE == 16000
    assert P.BYTES_PER_SAMPLE == 2
    assert P.VAD_CHUNK_SAMPLES == 512 and P.VAD_CHUNK_MS == 32


def test_enums_expose_the_expected_members() -> None:
    assert P.AudioFormat.PCM16.value == "pcm16"
    assert P.TurnDetectionType.SERVER_VAD.value == "server_vad"
    assert P.AECMode.NONE.value == "none" and P.AECMode.AEC.value == "aec"


# --- voice mapping --------------------------------------------------------


def test_openai_voice_names_map_to_magpie_voices() -> None:
    # An OpenAI name → its mapped Magpie voice, prefixed with the EN-US bank.
    assert P.resolve_voice("alloy") == f"{P.VOICE_PREFIX}Mia.Calm"
    assert P.resolve_voice("nova") == f"{P.VOICE_PREFIX}Mia.Happy"


def test_voice_name_lookup_is_case_insensitive() -> None:
    assert P.resolve_voice("ALLOY") == P.resolve_voice("alloy")


def test_unmapped_short_name_gets_the_voice_prefix() -> None:
    # A Magpie short name not in VOICE_MAP is still prefixed.
    assert P.resolve_voice("Leo.Calm") == f"{P.VOICE_PREFIX}Leo.Calm"


def test_already_full_magpie_voice_is_returned_verbatim() -> None:
    full = "Magpie-Multilingual.EN-US.Aria.Calm"
    assert P.resolve_voice(full) == full
