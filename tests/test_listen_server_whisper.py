"""Tests for lobes/templates/fleet/listen_server_whisper.py's pure helpers
(hebrew-realtime plan, task t10).

No torch / transformers / fastapi imports — the module guards all of those
behind ``_FASTAPI_AVAILABLE`` / lazy in-function imports, exactly the
convention ``lobes/realtime/_readiness.py``'s stdlib-only design and
``pyproject.toml``'s ``lobes/templates/*`` coverage-omit comment describe:
this file is not installed with those extras in the offline CI env. Loaded
via ``importlib`` from its packaged path (mirrors
``tests/test_realtime_readiness.py``'s ``_load_vendored_evaluate_readiness``
pattern) rather than a normal import, since ``lobes/templates`` is not a
Python package on ``sys.path``.

Criterion 3 (shape parity with ``listen_server.py``): ``listen_server.py``
itself imports ``fastapi``/``soundfile``/``uvicorn`` unconditionally at
module level, so it cannot be imported in this offline environment either
(coverage.omit already excludes lobes/templates/* for exactly this reason).
Its success-response shape is instead pinned from its own source
(``return {"text": text}``, read literally below) — this test fails loudly
if that literal ever changes without a matching update here. Its readiness
error/success shape comes from the SAME canonical
``lobes.realtime._readiness.evaluate_readiness`` both servers call, so that
half of parity is structural, not literal-string-pinned.
"""

from __future__ import annotations

import importlib.util
import io
import struct
import wave
from pathlib import Path

import pytest

from lobes.realtime._readiness import evaluate_readiness

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet"
_WHISPER_SERVER = _TEMPLATES / "listen_server_whisper.py"
_PARAKEET_SERVER = _TEMPLATES / "listen_server.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lsw():
    return _load(_WHISPER_SERVER)


def _make_wav_bytes(samples: list[int], *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


# --------------------------------------------------------------------------
# Module import / offline-safety
# --------------------------------------------------------------------------


def test_module_imports_offline_without_fastapi_torch_transformers(lsw) -> None:
    assert lsw._FASTAPI_AVAILABLE is False


def test_parakeet_server_is_not_importable_offline_either() -> None:
    """Confirms *why* this test compares against listen_server.py's SOURCE
    (below) rather than importing it: it unconditionally imports fastapi/
    soundfile/uvicorn at module level, so it is not importable in this
    dependency-free test environment."""
    with pytest.raises(ModuleNotFoundError):
        _load(_PARAKEET_SERVER)


# --------------------------------------------------------------------------
# Criterion 1: language resolution
# --------------------------------------------------------------------------


class TestResolveLanguage:
    def test_form_field_wins(self, lsw) -> None:
        assert lsw.resolve_language("en", "he", "he") == "en"

    def test_env_wins_when_form_absent(self, lsw) -> None:
        assert lsw.resolve_language(None, "en", "he") == "en"

    def test_default_wins_when_both_absent(self, lsw) -> None:
        assert lsw.resolve_language(None, None, "he") == "he"

    def test_invalid_form_falls_back_to_env(self, lsw) -> None:
        assert lsw.resolve_language("not-a-lang-code!", "en", "he") == "en"

    def test_invalid_form_and_env_fall_back_to_default(self, lsw) -> None:
        assert lsw.resolve_language("???", "???", "he") == "he"

    def test_empty_string_form_is_treated_as_absent(self, lsw) -> None:
        assert lsw.resolve_language("", "en", "he") == "en"

    @pytest.mark.parametrize("code", ["he", "en", "en-US", "heb", "zh-Hans"])
    def test_valid_codes(self, lsw, code: str) -> None:
        assert lsw.is_valid_language_code(code)

    @pytest.mark.parametrize("code", [None, "", "1", "toolongcodehere", "he_IL", "-"])
    def test_invalid_codes(self, lsw, code) -> None:
        assert not lsw.is_valid_language_code(code)


# --------------------------------------------------------------------------
# Non-speech bracketed-tag filter (measured need — see module docstring)
# --------------------------------------------------------------------------


class TestFilterNonSpeechOnly:
    @pytest.mark.parametrize(
        "text",
        [
            "(צחוק)",
            "[מוזיקה]",
            "(laughter)",
            "[MUSIC]",
            "(צחוק) (הורות)",
            "[applause] [music]",
            "  (coughing)  ",
            "(תווית)...",
            "[tag], [tag2]",
        ],
    )
    def test_pure_tag_transcripts_become_empty(self, lsw, text: str) -> None:
        assert lsw.filter_non_speech_only(text) == ""

    @pytest.mark.parametrize(
        "text",
        [
            "שלום (צחוק) עולם",
            "hello (laughter) world",
            "(צחוק) שלום",
            "שלום (צחוק)",
            "hello world",
            "שלום עולם",
            "",
            "   ",
        ],
    )
    def test_real_speech_or_empty_is_left_alone(self, lsw, text: str) -> None:
        assert lsw.filter_non_speech_only(text) == text


# --------------------------------------------------------------------------
# WAV decode / channel selection / resample / duration validation
# --------------------------------------------------------------------------


class TestDecodeWavPcm16:
    def test_round_trips_mono(self, lsw) -> None:
        samples = [0, 100, -100, 32767, -32768, 5]
        raw = _make_wav_bytes(samples, sample_rate=16000, channels=1)
        decoded, sr, ch = lsw.decode_wav_pcm16(raw)
        assert decoded == samples
        assert sr == 16000
        assert ch == 1

    def test_round_trips_stereo(self, lsw) -> None:
        # Interleaved: ch0, ch1, ch0, ch1, ...
        samples = [1, -1, 2, -2, 3, -3]
        raw = _make_wav_bytes(samples, sample_rate=24000, channels=2)
        decoded, sr, ch = lsw.decode_wav_pcm16(raw)
        assert decoded == samples
        assert sr == 24000
        assert ch == 2

    def test_rejects_non_16_bit(self, lsw) -> None:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)  # 8-bit
            wf.setframerate(16000)
            wf.writeframes(bytes([0, 1, 2, 3]))
        with pytest.raises(ValueError, match="16-bit"):
            lsw.decode_wav_pcm16(buf.getvalue())


class TestFirstChannel:
    def test_mono_passthrough(self, lsw) -> None:
        assert lsw.first_channel([1, 2, 3], 1) == [1, 2, 3]

    def test_stereo_takes_first_channel_only(self, lsw) -> None:
        # ch0, ch1, ch0, ch1, ch0, ch1
        assert lsw.first_channel([10, -10, 20, -20, 30, -30], 2) == [10, 20, 30]

    def test_does_not_average_channels(self, lsw) -> None:
        """Documents the deliberate choice: NOT (ch0+ch1)/2."""
        result = lsw.first_channel([100, 0, 100, 0], 2)
        assert result == [100, 100]
        assert result != [50, 50]


class TestLinearResample:
    def test_same_rate_is_passthrough(self, lsw) -> None:
        assert lsw.linear_resample([1.0, 2.0, 3.0], 16000, 16000) == [1.0, 2.0, 3.0]

    def test_empty_input(self, lsw) -> None:
        assert lsw.linear_resample([], 8000, 16000) == []

    def test_upsample_doubles_length_roughly(self, lsw) -> None:
        out = lsw.linear_resample([0.0, 1.0, 0.0, -1.0], 8000, 16000)
        # ~2x the samples for 2x the rate over the same duration.
        assert 6 <= len(out) <= 8

    def test_downsample_endpoints_preserved(self, lsw) -> None:
        out = lsw.linear_resample([0.0, 1.0, 2.0, 3.0, 4.0], 8000, 4000)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(4.0)


class TestValidateClipDuration:
    def test_under_limit_is_ok(self, lsw) -> None:
        assert lsw.validate_clip_duration(16000 * 29, 16000) is None

    def test_exactly_at_limit_is_ok(self, lsw) -> None:
        assert lsw.validate_clip_duration(16000 * 30, 16000) is None

    def test_over_limit_is_refused(self, lsw) -> None:
        msg = lsw.validate_clip_duration(16000 * 31, 16000)
        assert msg is not None
        assert "30" in msg

    def test_zero_sample_rate_is_refused(self, lsw) -> None:
        assert lsw.validate_clip_duration(100, 0) is not None


# --------------------------------------------------------------------------
# Criterion 3: response/error shape parity
# --------------------------------------------------------------------------


class TestResponseShapeParity:
    def test_success_shape_matches_listen_server_source(self, lsw) -> None:
        """listen_server.py's transcribe() literally ``return {"text": text}``
        (read from its own source below, since it cannot be imported
        offline — see test_parakeet_server_is_not_importable_offline_either).
        """
        parakeet_source = _PARAKEET_SERVER.read_text(encoding="utf-8")
        assert 'return {"text": text}' in parakeet_source

        body = lsw.build_success_response("hello")
        assert set(body.keys()) == {"text"}
        assert isinstance(body["text"], str)

    def test_readiness_body_shares_the_canonical_decision(self, lsw) -> None:
        """Both servers call the SAME lobes.realtime._readiness.evaluate_readiness
        — the readiness shape is therefore structurally shared, not merely
        literal-matched."""
        status_code, base_body = evaluate_readiness(model_loaded=True, cuda_ok=True)
        assert status_code == 200
        body = lsw.build_readiness_body(
            base_body, model_loaded=True, cuda_ok=True, model_name="ivrit-ai/whisper-large-v3-turbo"
        )
        assert body["status"] == "ready"
        assert body["model_loaded"] is True
        assert body["cuda_ok"] is True
        assert body["model"] == "ivrit-ai/whisper-large-v3-turbo"

    def test_readiness_body_not_ready_shape(self, lsw) -> None:
        status_code, base_body = evaluate_readiness(model_loaded=False, cuda_ok=True)
        assert status_code == 503
        body = lsw.build_readiness_body(base_body, model_loaded=False, cuda_ok=True, model_name="m")
        assert body["status"] == "not_ready"
        assert "reason" in body
        assert isinstance(body["reason"], str)
        assert body["model_loaded"] is False

    def test_error_body_shape_is_consistent_across_error_kinds(self, lsw) -> None:
        invalid_wav = lsw.build_error_body("bad wav", "invalid_wav")
        too_long = lsw.build_error_body("too long", "clip_too_long")
        for body in (invalid_wav, too_long):
            assert set(body.keys()) == {"error"}
            assert set(body["error"].keys()) == {"message", "code"}
            assert isinstance(body["error"]["message"], str)
            assert isinstance(body["error"]["code"], str)


class TestStripBidiControls:
    """Measured live 2026-09-18: a transcript arrived as ' \\u202b<hebrew>.'."""

    def test_the_measured_case(self, lsw) -> None:
        mod = lsw
        assert (
            mod.strip_bidi_controls(" \u202b\u05ea\u05d5\u05d3\u05d4 \u05e8\u05d1\u05d4.").strip()
            == "\u05ea\u05d5\u05d3\u05d4 \u05e8\u05d1\u05d4."
        )

    def test_every_control_is_removed_and_visible_text_is_untouched(self, lsw) -> None:
        mod = lsw
        controls = "\u200e\u200f\u061c\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
        assert mod.strip_bidi_controls(controls) == ""
        plain = "\u05e9\u05dc\u05d5\u05dd hello 123 (\u05e6\u05d7\u05d5\u05e7)"
        assert mod.strip_bidi_controls(plain) == plain


class TestConfidenceGate:
    """Thresholds measured on the DGX Spark 2026-09-18: speech >= -0.12,
    hallucinations <= -0.58 (see the module comment)."""

    def test_default_sits_between_the_measured_groups(self, lsw) -> None:
        assert -0.58 < lsw.DEFAULT_MIN_AVG_LOGPROB < -0.12

    @pytest.mark.parametrize("measured", [-0.00, -0.01, -0.04, -0.06, -0.08, -0.12])
    def test_every_measured_real_utterance_is_kept(self, lsw, measured) -> None:
        assert not lsw.is_low_confidence(measured, lsw.DEFAULT_MIN_AVG_LOGPROB)

    @pytest.mark.parametrize("measured", [-0.58, -0.60, -0.61, -1.50])
    def test_every_measured_hallucination_is_dropped(self, lsw, measured) -> None:
        assert lsw.is_low_confidence(measured, lsw.DEFAULT_MIN_AVG_LOGPROB)

    def test_a_disabled_gate_or_unknown_confidence_never_drops(self, lsw) -> None:
        assert not lsw.is_low_confidence(-9.0, None)
        assert not lsw.is_low_confidence(None, -0.35)

    def test_average_logprob(self, lsw) -> None:
        assert lsw.average_logprob([]) is None
        assert lsw.average_logprob([-0.2, -0.4]) == pytest.approx(-0.3)

    @pytest.mark.parametrize("raw", ["off", "OFF", "none", "disabled", "", "  "])
    def test_the_gate_can_be_switched_off_explicitly(self, lsw, raw) -> None:
        assert lsw.parse_min_avg_logprob(raw) is None

    def test_a_typo_never_silently_disables_the_gate(self, lsw) -> None:
        assert lsw.parse_min_avg_logprob("-0,35") == lsw.DEFAULT_MIN_AVG_LOGPROB
        assert lsw.parse_min_avg_logprob(None) == lsw.DEFAULT_MIN_AVG_LOGPROB
        assert lsw.parse_min_avg_logprob("-0.5") == -0.5
