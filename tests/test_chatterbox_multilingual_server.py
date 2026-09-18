"""Tests for lobes/realtime/chatterbox_multilingual_server.py (hebrew-realtime
plan, task t11 — the Hebrew TTS sidecar behind /v1/audio/synthesize).

Pure-function tests only, stdlib + pytest, no fastapi/torch/chatterbox — this
module guards those imports exactly like chatterbox_server.py (see
tests/test_realtime_imports.py's sibling assertion for that module), so it
imports fine offline; only the FastAPI route bodies are `pragma: no cover`.

The runaway-guard constants are checked against the actual measured sentences
and durations from docs/evidence/2026-09-hebrew-tts-ab-spark.txt.
"""

from __future__ import annotations

import struct

import pytest

from lobes.realtime import chatterbox_multilingual_server as m
from lobes.realtime import chatterbox_server

# ---------------------------------------------------------------------------
# has_niqqud
# ---------------------------------------------------------------------------


class TestHasNiqqud:
    def test_plain_hebrew_has_no_niqqud(self) -> None:
        assert m.has_niqqud("שלום, אני העוזר הקולי שלך") is False

    def test_vocalized_hebrew_has_niqqud(self) -> None:
        assert m.has_niqqud("שָׁלוֹם") is True

    def test_empty_string(self) -> None:
        assert m.has_niqqud("") is False

    def test_english_text_has_no_niqqud(self) -> None:
        assert m.has_niqqud("hello world") is False

    def test_single_dagesh_counts_as_niqqud(self) -> None:
        # U+05BC DAGESH — a real vocalization mark, inside the checked range.
        assert m.has_niqqud("בּ") is True

    @pytest.mark.parametrize(
        "ch",
        [
            "־",  # MAQAF (Hebrew hyphen)
            "׀",  # PASEQ
            "׃",  # SOF PASUQ
            "׆",  # NUN HAFUKHA
        ],
    )
    def test_punctuation_like_marks_do_not_count(self, ch: str) -> None:
        assert m.has_niqqud(f"שלום{ch}") is False

    def test_mixed_hebrew_and_latin_with_niqqud(self) -> None:
        assert m.has_niqqud("path/to/file.txt שָׁלוֹם") is True


# ---------------------------------------------------------------------------
# strip_phonikud_invented_marks
# ---------------------------------------------------------------------------


class TestStripPhonikudInventedMarks:
    def test_strips_ole_accent(self) -> None:
        assert m.strip_phonikud_invented_marks("שָׁ֫לוֹם") == "שָׁלוֹם"

    def test_strips_meteg(self) -> None:
        assert m.strip_phonikud_invented_marks("שָֽׁלוֹם") == "שָׁלוֹם"

    def test_strips_pipe(self) -> None:
        assert m.strip_phonikud_invented_marks("שָׁלוֹ|ם") == "שָׁלוֹם"

    def test_strips_all_three_together(self) -> None:
        assert m.strip_phonikud_invented_marks("֫שָֽׁלוֹ|ם") == "שָׁלוֹם"

    def test_idempotent_on_text_without_invented_marks(self) -> None:
        text = "שָׁלוֹם, אני העוזר הקולי שלך."
        assert m.strip_phonikud_invented_marks(text) == text

    def test_empty_string(self) -> None:
        assert m.strip_phonikud_invented_marks("") == ""

    def test_does_not_touch_standard_niqqud(self) -> None:
        # Standard niqqud (e.g. U+05B8 QAMATS) must survive stripping.
        text = "שָׁלוֹם"
        assert m.strip_phonikud_invented_marks(text) == text


# ---------------------------------------------------------------------------
# Runaway-synthesis guard — max_plausible_duration_s / is_runaway
# ---------------------------------------------------------------------------

# The three A/B sentences, verbatim, from
# docs/evidence/2026-09-hebrew-tts-ab-spark.txt.
_HE1 = "שלום, אני העוזר הקולי שלך. במה אפשר לעזור?"
_HE2 = "בתיקיית המסמכים יש שלושה קבצים, והאחרון עודכן אתמול בערב."
_HE3 = "מה ההבדל בין מחשב נייד למחשב שולחני?"


class TestRunawayGuardMeasuredCases:
    """The two cases named in this task's brief: an 8-word sentence's 24.5s
    runaway must be CAUGHT; normal 2.9s/3.4s clips must PASS."""

    def test_he1_stable_clip_2_92s_is_not_runaway(self) -> None:
        assert m.is_runaway(_HE1, 2.92) is False

    def test_he1_runaway_clip_24_5s_is_caught(self) -> None:
        # he1: 8 words, ~42 base chars -> threshold ~11.4s. 24.5s is >2x over.
        assert m.is_runaway(_HE1, 24.5) is True

    def test_he3_stable_clip_2_84s_is_not_runaway(self) -> None:
        assert m.is_runaway(_HE3, 2.84) is False

    def test_a_3_4s_clip_on_he1_sized_text_is_not_runaway(self) -> None:
        assert m.is_runaway(_HE1, 3.4) is False

    def test_he2_11_56s_clip_is_not_flagged(self) -> None:
        """he2's phonikud-arm clip (11.56s for 9 correct words) is noted in
        the evidence file as having "ran long" but was NOT garbage — the
        chosen constants keep it under threshold (no spurious retry) while
        still catching the much larger 24.5s garbage case above."""
        assert m.is_runaway(_HE2, 11.56) is False

    def test_threshold_grows_with_text_length(self) -> None:
        short = m.max_plausible_duration_s("שלום")
        long = m.max_plausible_duration_s(_HE2)
        assert long > short

    def test_niqqud_marks_do_not_inflate_the_threshold(self) -> None:
        plain = m.max_plausible_duration_s("שלום")
        vocalized = m.max_plausible_duration_s("שָׁלוֹם")
        assert plain == vocalized

    def test_empty_text_has_a_positive_floor(self) -> None:
        assert m.max_plausible_duration_s("") == pytest.approx(m._DEFAULT_MAX_DURATION_SLACK_S)


# ---------------------------------------------------------------------------
# PCM16 duration / truncation
# ---------------------------------------------------------------------------


def _silence_pcm(num_samples: int) -> bytes:
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


def _tone_pcm(num_samples: int, amplitude: int = 20000) -> bytes:
    return struct.pack(f"<{num_samples}h", *([amplitude, -amplitude] * (num_samples // 2)))


class TestPcm16Duration:
    def test_duration_of_one_second_at_24khz(self) -> None:
        pcm = _silence_pcm(24000)
        assert m.pcm16_duration_s(pcm, 24000) == pytest.approx(1.0)

    def test_empty_pcm_has_zero_duration(self) -> None:
        assert m.pcm16_duration_s(b"", 24000) == 0.0

    def test_zero_sample_rate_never_raises(self) -> None:
        assert m.pcm16_duration_s(_silence_pcm(100), 0) == 0.0


class TestFindTruncationSample:
    def test_finds_silence_before_the_ceiling(self) -> None:
        sample_rate = 1000
        # 0.5s of tone, 0.5s of silence, 1.0s of tone — a runaway tail.
        pcm = _tone_pcm(500) + _silence_pcm(500) + _tone_pcm(1000)
        max_samples = 1800  # inside the trailing tone region
        cut = m.find_truncation_sample(pcm, sample_rate, max_samples, window_ms=50.0)
        # The cut point should land inside/at the end of the silence gap
        # (500-1000), not mid-tone.
        assert 500 <= cut <= 1000

    def test_no_quiet_window_hard_cuts_at_max_samples(self) -> None:
        sample_rate = 1000
        pcm = _tone_pcm(2000)
        cut = m.find_truncation_sample(pcm, sample_rate, 1500, window_ms=50.0)
        assert cut == 1500

    def test_max_samples_beyond_clip_length_clamps(self) -> None:
        sample_rate = 1000
        pcm = _silence_pcm(500)
        cut = m.find_truncation_sample(pcm, sample_rate, 10_000, window_ms=50.0)
        assert cut == 500

    def test_silent_clip_returns_max_samples(self) -> None:
        sample_rate = 1000
        pcm = _silence_pcm(2000)
        cut = m.find_truncation_sample(pcm, sample_rate, 1500, window_ms=50.0)
        assert cut == 1500

    def test_zero_max_samples_returns_zero(self) -> None:
        assert m.find_truncation_sample(_tone_pcm(2000), 1000, 0) == 0


class TestTruncatePcm16:
    def test_result_never_exceeds_max_duration_plus_one_window(self) -> None:
        sample_rate = 1000
        pcm = _tone_pcm(500) + _silence_pcm(200) + _tone_pcm(3000)
        truncated = m.truncate_pcm16(pcm, sample_rate, max_duration_s=1.0, window_ms=50.0)
        assert m.pcm16_duration_s(truncated, sample_rate) <= 1.0 + 0.05

    def test_short_clip_under_ceiling_is_unaffected_in_length_terms(self) -> None:
        sample_rate = 1000
        pcm = _tone_pcm(500)
        # Ceiling far beyond the clip — find_truncation_sample clamps to the
        # clip's own length, so nothing is cut.
        truncated = m.truncate_pcm16(pcm, sample_rate, max_duration_s=10.0)
        assert truncated == pcm

    def test_returns_bytes(self) -> None:
        out = m.truncate_pcm16(_tone_pcm(2000), 1000, max_duration_s=1.0)
        assert isinstance(out, bytes)


# ---------------------------------------------------------------------------
# Settings parsing
# ---------------------------------------------------------------------------


class TestBuildMlSettings:
    def test_defaults(self) -> None:
        s = m.build_ml_settings({})
        assert s.host == "0.0.0.0"
        assert s.port == 9000
        assert s.language == "he"
        assert s.diacritize == "auto"
        assert s.max_retries == 2
        assert s.phonikud_model_path == ""

    def test_diacritize_off_case_insensitive(self) -> None:
        assert m.build_ml_settings({"TTS_DIACRITIZE": "OFF"}).diacritize == "off"
        assert m.build_ml_settings({"TTS_DIACRITIZE": "Off"}).diacritize == "off"

    def test_diacritize_unknown_value_falls_back_to_auto(self) -> None:
        assert m.build_ml_settings({"TTS_DIACRITIZE": "bogus"}).diacritize == "auto"
        assert m.build_ml_settings({"TTS_DIACRITIZE": ""}).diacritize == "auto"

    def test_language_override(self) -> None:
        assert m.build_ml_settings({"TTS_LANGUAGE": "en"}).language == "en"

    def test_max_retries_override_and_floor(self) -> None:
        assert m.build_ml_settings({"TTS_MAX_RETRIES": "5"}).max_retries == 5
        # Never negative — a negative retry count is meaningless.
        assert m.build_ml_settings({"TTS_MAX_RETRIES": "-3"}).max_retries == 0

    def test_max_retries_non_numeric_falls_back_to_default(self) -> None:
        assert m.build_ml_settings({"TTS_MAX_RETRIES": "nope"}).max_retries == 2

    def test_phonikud_model_path_passthrough(self) -> None:
        s = m.build_ml_settings({"PHONIKUD_MODEL_PATH": "/models/phonikud-1.0.int8.onnx"})
        assert s.phonikud_model_path == "/models/phonikud-1.0.int8.onnx"

    def test_runaway_knobs_are_tunable(self) -> None:
        s = m.build_ml_settings(
            {
                "TTS_MAX_SECONDS_PER_BASE_CHAR": "0.5",
                "TTS_MAX_DURATION_SLACK_S": "1.0",
                "TTS_TRUNCATE_WINDOW_MS": "30",
                "TTS_TRUNCATE_SILENCE_RATIO": "0.2",
            }
        )
        assert s.max_seconds_per_base_char == 0.5
        assert s.max_duration_slack_s == 1.0
        assert s.truncate_window_ms == 30.0
        assert s.truncate_silence_ratio == 0.2

    def test_truncate_silence_ratio_is_clamped_to_0_1(self) -> None:
        assert (
            m.build_ml_settings({"TTS_TRUNCATE_SILENCE_RATIO": "5"}).truncate_silence_ratio == 1.0
        )
        assert (
            m.build_ml_settings({"TTS_TRUNCATE_SILENCE_RATIO": "-1"}).truncate_silence_ratio == 0.0
        )

    def test_module_level_env_reads_from_os_environ_by_default(self, monkeypatch) -> None:
        monkeypatch.setenv("TTS_LANGUAGE", "he")
        s = m.build_ml_settings()
        assert s.language == "he"


# ---------------------------------------------------------------------------
# Readiness shape parity with chatterbox_server.readiness_status
# ---------------------------------------------------------------------------


class TestReadinessStatus:
    def test_model_not_loaded_returns_503_loading(self) -> None:
        code, body = m.readiness_status(False, False, False)
        assert code == 503
        assert body == {"status": "loading"}

    def test_model_loaded_but_warmup_pending_returns_503(self) -> None:
        code, body = m.readiness_status(True, False, False)
        assert code == 503
        assert body["status"] == "loading"
        assert body["reason"] == "warmup_pending"

    def test_model_loaded_cuda_poisoned_returns_503_unavailable(self) -> None:
        code, body = m.readiness_status(True, False, True)
        assert code == 503
        assert body == {"status": "unavailable", "reason": "cuda_context_poisoned"}

    def test_cuda_poisoned_checked_before_warmup(self) -> None:
        # Both conditions true at once: poisoned CUDA is the more specific
        # failure, and must win over the generic "still warming up" state.
        code, body = m.readiness_status(True, False, True)
        assert body["reason"] == "cuda_context_poisoned"

    def test_model_loaded_and_warmed_up_returns_200_ready(self) -> None:
        code, body = m.readiness_status(True, True, False)
        assert code == 200
        assert body == {"status": "ready"}

    def test_response_shape_matches_chatterbox_server_english_states(self) -> None:
        """Same {"status": ...} JSON shape as chatterbox_server.py's states,
        for every state the two modules share (loading / ready)."""
        en_loading = chatterbox_server.readiness_status(False, False)
        ml_loading = m.readiness_status(False, False, False)
        assert set(en_loading[1].keys()) == set(ml_loading[1].keys())
        assert en_loading[1]["status"] == ml_loading[1]["status"] == "loading"

        en_ready = chatterbox_server.readiness_status(True, False)
        ml_ready = m.readiness_status(True, True, False)
        assert en_ready == (200, {"status": "ready"})
        assert ml_ready == (200, {"status": "ready"})


# ---------------------------------------------------------------------------
# float_tensor_to_pcm16 reuse — imported, not re-implemented
# ---------------------------------------------------------------------------


class TestFloatTensorToPcm16Reuse:
    def test_reused_directly_from_chatterbox_server(self) -> None:
        assert m.float_tensor_to_pcm16 is chatterbox_server.float_tensor_to_pcm16

    def test_reused_function_still_works(self) -> None:
        pcm = m.float_tensor_to_pcm16([0.0, 1.0, -1.0])
        assert pcm == chatterbox_server.float_tensor_to_pcm16([0.0, 1.0, -1.0])


# ---------------------------------------------------------------------------
# Import isolation — mirrors tests/test_realtime_imports.py's
# test_chatterbox_server_imports_without_fastapi for this sibling module.
# ---------------------------------------------------------------------------


class TestImportIsolation:
    def test_module_imports_without_fastapi_extra(self) -> None:
        # If we got this far, the import at module top already succeeded in
        # this (fastapi-less) offline test env — this assertion just names
        # the property explicitly for a reader of this test file.
        assert callable(m.float_tensor_to_pcm16)
        assert callable(m.has_niqqud)
