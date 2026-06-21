"""Tests for the realtime protocol helpers (stdlib-only; no [realtime] extra).

IDs, audio constants, and the Chatterbox voice resolver import without
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
    assert P.TTS_SAMPLE_RATE == 24000  # Chatterbox output — matches CLIENT_SAMPLE_RATE
    assert P.STT_SAMPLE_RATE == 16000  # Parakeet input
    assert P.VAD_SAMPLE_RATE == 16000
    assert P.BYTES_PER_SAMPLE == 2
    assert P.VAD_CHUNK_SAMPLES == 512 and P.VAD_CHUNK_MS == 32


def test_enums_expose_the_expected_members() -> None:
    assert P.AudioFormat.PCM16.value == "pcm16"
    assert P.TurnDetectionType.SERVER_VAD.value == "server_vad"
    assert P.AECMode.NONE.value == "none" and P.AECMode.AEC.value == "aec"


# --- Chatterbox voice resolution ------------------------------------------


def test_wav_path_is_returned_verbatim_for_cloning() -> None:
    # A .wav path → zero-shot cloning reference; returned unchanged.
    assert P.resolve_voice("/data/voices/speaker.wav") == "/data/voices/speaker.wav"
    assert P.resolve_voice("reference.wav") == "reference.wav"


def test_non_wav_names_resolve_to_default_voice() -> None:
    # OpenAI names, arbitrary strings, and empty string all map to the default.
    assert P.resolve_voice("alloy") == ""
    assert P.resolve_voice("nova") == ""
    assert P.resolve_voice("") == ""
    assert P.resolve_voice("some-voice") == ""


def test_only_wav_suffix_triggers_cloning() -> None:
    # A path ending with .WAV (upper-case) is NOT a .wav ending — no special casing.
    assert P.resolve_voice("clip.WAV") == ""
    # A path ending with .wav is always treated as a cloning reference.
    assert P.resolve_voice("clip.wav") == "clip.wav"
