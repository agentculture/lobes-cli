"""Offline tests for the BlueTTS sidecar's pure helpers (hebrew-realtime d8).

Everything here is stdlib-only: the offline suite has no numpy, onnxruntime
or fastapi, exactly like the Chatterbox sidecars' own tests.
"""

from __future__ import annotations

import struct

import pytest

from lobes.realtime import bluetts_server as srv

# --- settings ---------------------------------------------------------------


def test_defaults_are_a_drop_in_for_the_chatterbox_service():
    s = srv.build_blue_settings({})
    assert s.host == "0.0.0.0"  # nosec B104 - container-internal bind, as the siblings
    assert s.port == 9000
    assert s.language == "he"
    assert s.voice == "noa"
    assert s.total_step == 5
    assert s.speed == 1.0


def test_no_default_names_a_weights_repo():
    """The weights' HF repo declares no licence (operator, 2026-09-18): the
    sidecar only ever reads a LOCAL directory the operator mounted."""
    s = srv.build_blue_settings({})
    for value in (s.onnx_dir, s.voices_dir, s.config_path):
        assert value.startswith("/")
    assert "notmax123" not in repr(s)


def test_settings_read_the_shared_port_keys_and_own_knobs():
    s = srv.build_blue_settings(
        {
            "CHATTERBOX_HOST": "127.0.0.1",
            "CHATTERBOX_PORT": "9100",
            "TTS_LANGUAGE": "en",
            "BLUETTS_ONNX_DIR": "/w",
            "BLUETTS_VOICES_DIR": "/v",
            "BLUETTS_CONFIG_PATH": "/c/tts.json",
            "BLUETTS_VOICE": "adam",
            "BLUETTS_STEPS": "8",
            "BLUETTS_SPEED": "1.15",
            "BLUETTS_THREADS": "6",
            "BLUETTS_RENIKUD_PATH": "/g2p/model.onnx",
        }
    )
    assert (s.host, s.port, s.language) == ("127.0.0.1", 9100, "en")
    assert (s.onnx_dir, s.voices_dir, s.config_path) == ("/w", "/v", "/c/tts.json")
    assert (s.voice, s.total_step, s.speed, s.threads) == ("adam", 8, 1.15, 6)
    assert s.renikud_path == "/g2p/model.onnx"


@pytest.mark.parametrize("key", ["CHATTERBOX_PORT", "BLUETTS_STEPS", "BLUETTS_SPEED"])
def test_a_typo_falls_back_to_the_default_rather_than_crashing(key):
    assert srv.build_blue_settings({key: "fast"}) == srv.build_blue_settings({})


def test_out_of_range_knobs_are_clamped():
    s = srv.build_blue_settings({"BLUETTS_STEPS": "0", "BLUETTS_SPEED": "9"})
    assert s.total_step == 1
    assert s.speed == 2.0
    assert srv.build_blue_settings({"BLUETTS_SPEED": "0.01"}).speed == 0.5


def test_empty_renikud_path_means_auto():
    assert srv.build_blue_settings({"BLUETTS_RENIKUD_PATH": " "}).renikud_path is None


# --- text -------------------------------------------------------------------


def test_niqqud_is_stripped_because_bluetts_runs_its_own_g2p():
    assert srv.strip_niqqud("שָׁלוֹם עוֹלָם") == "שלום עולם"


def test_strip_niqqud_keeps_maqaf_punctuation_and_latin():
    text = 'בית־ספר, צה״ל: "ok" 14:30?'
    assert srv.strip_niqqud(text) == text


def test_strip_niqqud_drops_phonikud_stress_and_vocal_shva_marks():
    assert srv.strip_niqqud("שָׁל֫וֹם") == "שלום"


# --- voice ------------------------------------------------------------------


def _exists(*present):
    return lambda p: p in present


def test_requested_voice_resolves_inside_the_voices_dir():
    got = srv.resolve_voice_path("adam", "/v", "noa", exists=_exists("/v/adam.json"))
    assert got == "/v/adam.json"


@pytest.mark.parametrize("voice", ["", None, "ghost", "../../etc/passwd", "a/b", "noa.json"])
def test_unknown_or_unsafe_voice_falls_back_to_the_default(voice):
    got = srv.resolve_voice_path(voice, "/v", "noa", exists=_exists("/v/noa.json"))
    assert got == "/v/noa.json"


# --- audio ------------------------------------------------------------------


def test_resample_ratio_for_the_measured_44100_output():
    assert srv.resample_ratio(44100, 24000) == (80, 147)


def test_resample_ratio_is_identity_at_24k():
    assert srv.resample_ratio(24000, 24000) == (1, 1)


@pytest.mark.parametrize("rates", [(0, 24000), (44100, 0), (-1, 24000)])
def test_resample_ratio_refuses_nonsense(rates):
    with pytest.raises(ValueError):
        srv.resample_ratio(*rates)


def test_floats_to_pcm16_is_little_endian_and_clips():
    pcm = srv.floats_to_pcm16([0.0, 1.0, -1.0, 2.0, -2.0, 0.5])
    assert struct.unpack("<6h", pcm) == (0, 32767, -32767, 32767, -32767, 16383)


def test_floats_to_pcm16_of_nothing_is_empty():
    assert srv.floats_to_pcm16([]) == b""


# --- readiness --------------------------------------------------------------


def test_not_ready_until_the_model_is_loaded():
    code, body = srv.readiness_status(model_loaded=False, warmup_ok=False)
    assert code == 503 and body["status"] == "loading"


def test_not_ready_until_the_g2p_warm_up_synthesis_has_run():
    """RenikudPlus fetches/loads on FIRST use (measured: seconds) — a sidecar
    that reports ready before that makes the first spoken turn pay for it."""
    code, body = srv.readiness_status(model_loaded=True, warmup_ok=False)
    assert code == 503 and body["status"] == "warming"


def test_ready_after_warm_up():
    assert srv.readiness_status(model_loaded=True, warmup_ok=True) == (200, {"status": "ready"})


def test_contract_constants_match_the_bridge():
    assert srv.TTS_SAMPLE_RATE == 24000
