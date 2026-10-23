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
        raw = buf.getvalue()
        with pytest.raises(ValueError, match="16-bit"):
            lsw.decode_wav_pcm16(raw)


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
        assert lsw.validate_clip_duration(16000 * 29, 16000, 30) is None

    def test_exactly_at_limit_is_ok(self, lsw) -> None:
        assert lsw.validate_clip_duration(16000 * 30, 16000, 30) is None

    def test_over_limit_is_refused(self, lsw) -> None:
        msg = lsw.validate_clip_duration(16000 * 31, 16000, 30)
        assert msg is not None
        assert "30" in msg

    def test_longer_than_one_window_is_not_refused_by_default(self, lsw) -> None:
        # Past one Whisper window is chunked now; only the total cap refuses.
        assert lsw.validate_clip_duration(16000 * 31, 16000) is None
        assert lsw.validate_clip_duration(16000 * 601, 16000) is not None

    def test_zero_sample_rate_is_refused(self, lsw) -> None:
        assert lsw.validate_clip_duration(100, 0) is not None


# --------------------------------------------------------------------------
# Qodo finding: a cheap byte cap BEFORE decoding, so an oversized multipart
# upload cannot exhaust a speech worker by being fully decoded into Python
# objects before the (post-decode) duration limit ever runs.
# --------------------------------------------------------------------------


class TestCheckUploadSize:
    def test_under_limit_is_ok(self, lsw) -> None:
        assert lsw.check_upload_size(1000, 2000) is None

    def test_exactly_at_limit_is_ok(self, lsw) -> None:
        assert lsw.check_upload_size(2000, 2000) is None

    def test_over_limit_is_refused(self, lsw) -> None:
        msg = lsw.check_upload_size(2001, 2000)
        assert msg is not None
        assert "2001" in msg
        assert "2000" in msg

    def test_default_cap_comfortably_covers_the_realtime_bridge_turn_audio(self, lsw) -> None:
        # The realtime bridge's own turn audio: max 30s at 16kHz mono PCM16,
        # ~1MB — this must never be rejected by the default cap.
        bridge_turn_bytes = 30 * 16000 * 2
        assert lsw.check_upload_size(bridge_turn_bytes, lsw.DEFAULT_MAX_UPLOAD_BYTES) is None

    def test_default_cap_is_generous_but_bounded(self, lsw) -> None:
        # ~5.8MB is 30s of 48kHz stereo PCM16 (the module comment's own
        # figure) — comfortably under the 16 MiB default.
        assert lsw.DEFAULT_MAX_UPLOAD_BYTES >= 6 * 1024 * 1024


class TestParseMaxUploadBytes:
    def test_unset_is_the_default(self, lsw) -> None:
        assert lsw.parse_max_upload_bytes(None) == lsw.DEFAULT_MAX_UPLOAD_BYTES

    def test_empty_string_is_the_default(self, lsw) -> None:
        assert lsw.parse_max_upload_bytes("  ") == lsw.DEFAULT_MAX_UPLOAD_BYTES

    def test_valid_override_is_honoured(self, lsw) -> None:
        assert lsw.parse_max_upload_bytes("1048576") == 1048576

    def test_a_typo_never_silently_disables_the_cap(self, lsw) -> None:
        assert lsw.parse_max_upload_bytes("not-a-number") == lsw.DEFAULT_MAX_UPLOAD_BYTES

    def test_non_positive_falls_back_to_the_default(self, lsw) -> None:
        assert lsw.parse_max_upload_bytes("0") == lsw.DEFAULT_MAX_UPLOAD_BYTES
        assert lsw.parse_max_upload_bytes("-5") == lsw.DEFAULT_MAX_UPLOAD_BYTES


class TestPeekWavDurationSeconds:
    def test_matches_the_full_decode_for_a_valid_header(self, lsw) -> None:
        raw = _make_wav_bytes([0] * 16000, sample_rate=16000, channels=1)
        assert lsw.peek_wav_duration_seconds(raw) == pytest.approx(1.0)

    def test_does_not_require_reading_any_audio_frames(self, lsw) -> None:
        # A 31s clip header — large enough that a full decode would be the
        # thing this check exists to avoid doing first.
        raw = _make_wav_bytes([0] * (16000 * 31), sample_rate=16000, channels=1)
        assert lsw.peek_wav_duration_seconds(raw) == pytest.approx(31.0)

    def test_malformed_bytes_return_none(self, lsw) -> None:
        assert lsw.peek_wav_duration_seconds(b"not a wav file") is None

    def test_empty_bytes_return_none(self, lsw) -> None:
        assert lsw.peek_wav_duration_seconds(b"") is None


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


# --------------------------------------------------------------------------
# Non-WAV uploads (m4a, mp3, ogg, webm, ...) are transcoded by the image's
# ffmpeg; a RIFF/WAVE upload never touches it.
# --------------------------------------------------------------------------


class TestIsRiffWave:
    def test_wav_is_detected(self, lsw) -> None:
        assert lsw.is_riff_wave(_make_wav_bytes([0, 1, 2]))

    def test_m4a_is_not_wav(self, lsw) -> None:
        assert not lsw.is_riff_wave(b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00")

    def test_short_or_empty_is_not_wav(self, lsw) -> None:
        assert not lsw.is_riff_wave(b"")
        assert not lsw.is_riff_wave(b"RIFF")


class TestFfmpegTranscodeCommand:
    def test_targets_16k_mono_pcm16_wav_file(self, lsw) -> None:
        cmd = lsw.ffmpeg_transcode_command("/t/in", "/t/out.wav", 600)
        assert cmd[0] == "ffmpeg"
        assert cmd[cmd.index("-i") + 1] == "/t/in"
        assert cmd[cmd.index("-ac") + 1] == "1"
        assert cmd[cmd.index("-ar") + 1] == "16000"
        assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
        assert cmd[-1] == "/t/out.wav"  # a seekable FILE, never pipe:1

    def test_decode_stops_one_second_past_the_cap(self, lsw) -> None:
        cmd = lsw.ffmpeg_transcode_command("/t/in", "/t/out.wav", 600)
        assert float(cmd[cmd.index("-t") + 1]) == pytest.approx(601)


class _FakeProc:
    def __init__(self, returncode: int, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr


class TestTranscodeToWav:
    def test_success_returns_ffmpegs_output_file(self, lsw) -> None:
        wav = _make_wav_bytes([5, -5, 7])
        seen = {}

        def runner(cmd, **kwargs):
            src, dst = cmd[cmd.index("-i") + 1], cmd[-1]
            seen["input"] = Path(src).read_bytes()
            seen["timeout"] = kwargs.get("timeout")
            Path(dst).write_bytes(wav)
            return _FakeProc(0)

        assert lsw.transcode_to_wav(b"m4a-bytes", runner=runner) == wav
        assert seen["input"] == b"m4a-bytes"
        assert seen["timeout"] == lsw.FFMPEG_TIMEOUT_SECONDS

    def test_ffmpeg_failure_raises_with_its_last_stderr_line(self, lsw) -> None:
        def runner(cmd, **kwargs):
            return _FakeProc(1, b"noise\nin: Invalid data found when processing input\n")

        with pytest.raises(ValueError, match="Invalid data found"):
            lsw.transcode_to_wav(b"garbage", runner=runner)

    def test_missing_ffmpeg_is_a_value_error(self, lsw) -> None:
        def runner(cmd, **kwargs):
            raise FileNotFoundError("ffmpeg")

        with pytest.raises(ValueError, match="ffmpeg is not installed"):
            lsw.transcode_to_wav(b"x", runner=runner)

    def test_zero_exit_without_output_is_a_failure(self, lsw) -> None:
        with pytest.raises(ValueError, match="could not decode audio"):
            lsw.transcode_to_wav(b"x", runner=lambda cmd, **kw: _FakeProc(0))


class TestParseMaxAudioSeconds:
    def test_default_when_unset_empty_or_bad(self, lsw) -> None:
        for raw in (None, "", "  ", "ten", "0", "-5", "inf", "nan"):
            assert lsw.parse_max_audio_seconds(raw) == lsw.DEFAULT_MAX_AUDIO_SECONDS

    def test_positive_value_is_used(self, lsw) -> None:
        assert lsw.parse_max_audio_seconds("120") == 120.0


# --------------------------------------------------------------------------
# Chunking: longer than one Whisper window is split at the quietest point
# near each window's end, never refused.
# --------------------------------------------------------------------------


class TestPlanChunks:
    RATE = 100  # 100 Hz keeps the fixtures tiny; frame = 2 samples at 20 ms

    def _check_cover(self, chunks, n, window) -> None:
        assert chunks[0][0] == 0
        assert chunks[-1][1] == n
        for (a, b), (c, _) in zip(chunks, chunks[1:]):
            assert b == c  # contiguous, no overlap, no gap
        for a, b in chunks:
            assert 0 < b - a <= window

    def test_one_window_is_one_chunk(self, lsw) -> None:
        assert lsw.plan_chunks([1.0] * 2000, self.RATE, 30) == [(0, 2000)]

    def test_exactly_one_window_is_one_chunk(self, lsw) -> None:
        assert lsw.plan_chunks([1.0] * 3000, self.RATE, 30) == [(0, 3000)]

    def test_cut_lands_in_the_pause(self, lsw) -> None:
        samples = [1.0] * 4500
        samples[2700:2720] = [0.0] * 20  # a 0.2 s pause 27 s in
        chunks = lsw.plan_chunks(samples, self.RATE, 30, 5)
        self._check_cover(chunks, 4500, 3000)
        assert 2700 <= chunks[0][1] <= 2720

    def test_long_clip_covers_every_sample_within_windows(self, lsw) -> None:
        samples = [((i * 7919) % 13) / 13.0 for i in range(100 * 200)]  # 200 s
        chunks = lsw.plan_chunks(samples, self.RATE, 30, 5)
        self._check_cover(chunks, len(samples), 3000)
        assert len(chunks) >= 7

    def test_no_pause_still_makes_progress(self, lsw) -> None:
        chunks = lsw.plan_chunks([1.0] * 9000, self.RATE, 30, 5)
        self._check_cover(chunks, 9000, 3000)


class TestJoinChunkTexts:
    def test_skips_empty_windows(self, lsw) -> None:
        assert lsw.join_chunk_texts(["שלום", "", "  ", "עולם "]) == "שלום עולם"

    def test_all_empty_is_empty(self, lsw) -> None:
        assert lsw.join_chunk_texts(["", ""]) == ""


# --------------------------------------------------------------------------
# Issue #329: decode-time loop prevention, compression-ratio gate, retry
# before dropping, Silero silence skipping, and drop/retry observability.
# Every model call below is a FAKE — no torch, no transformers, no weights.
# --------------------------------------------------------------------------


_SPEECH = "שלום לכולם, היום אנחנו נדבר על הפרויקט החדש שלנו ועל הדרך שבה אנחנו עובדים."
_LOOP = "נפשט קרוב. -" * 30  # the shape of the issue's loop, not its recording


class TestGenerateKwargs:
    def test_all_knobs_off_is_exactly_todays_call(self, lsw) -> None:
        settings = lsw.DecodeSettings(no_repeat_ngram_size=0, repetition_penalty=1.0)
        assert lsw.build_generate_kwargs("he", 0.0, settings) == {
            "language": "he",
            "task": "transcribe",
            "max_new_tokens": lsw.MAX_NEW_TOKENS,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

    def test_loop_prevention_knobs_reach_generate(self, lsw) -> None:
        settings = lsw.DecodeSettings(no_repeat_ngram_size=6, repetition_penalty=1.1)
        kwargs = lsw.build_generate_kwargs("he", 0.0, settings)
        assert kwargs["no_repeat_ngram_size"] == 6
        assert kwargs["repetition_penalty"] == pytest.approx(1.1)
        assert "do_sample" not in kwargs  # attempt 0 stays greedy

    def test_a_fallback_attempt_samples_from_the_full_distribution(self, lsw) -> None:
        kwargs = lsw.build_generate_kwargs("he", 0.4, lsw.DecodeSettings())
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == pytest.approx(0.4)
        # top-k/top-p off, so scores * temperature are the processed logits
        # and the confidence gate reads the same quantity on every attempt.
        assert kwargs["top_k"] == 0
        assert kwargs["top_p"] == pytest.approx(1.0)

    def test_defaults(self, lsw) -> None:
        settings = lsw.DecodeSettings()
        assert settings.no_repeat_ngram_size == lsw.DEFAULT_NO_REPEAT_NGRAM_SIZE == 6
        assert settings.repetition_penalty == lsw.DEFAULT_REPETITION_PENALTY == 1.0


class TestDecodeKnobParsing:
    @pytest.mark.parametrize("raw", ["0", "off", "OFF", "none", "", " "])
    def test_ngram_off(self, lsw, raw) -> None:
        assert lsw.parse_no_repeat_ngram_size(raw) == 0

    def test_ngram_value_and_typo(self, lsw) -> None:
        assert lsw.parse_no_repeat_ngram_size(None) == 6
        assert lsw.parse_no_repeat_ngram_size("4") == 4
        assert lsw.parse_no_repeat_ngram_size("four") == 6
        assert lsw.parse_no_repeat_ngram_size("-3") == 6

    @pytest.mark.parametrize("raw", ["1", "1.0", "off", "none", ""])
    def test_repetition_penalty_off(self, lsw, raw) -> None:
        assert lsw.parse_repetition_penalty(raw) == 1.0

    def test_repetition_penalty_value_and_typo(self, lsw) -> None:
        assert lsw.parse_repetition_penalty(None) == 1.0
        assert lsw.parse_repetition_penalty("1.1") == pytest.approx(1.1)
        assert lsw.parse_repetition_penalty("1,1") == 1.0
        assert lsw.parse_repetition_penalty("0.5") == 1.0  # < 1 would REWARD repeats

    @pytest.mark.parametrize("raw", ["off", "none", "", "0"])
    def test_compression_ratio_gate_off(self, lsw, raw) -> None:
        assert lsw.parse_max_compression_ratio(raw) is None

    def test_compression_ratio_value_and_typo(self, lsw) -> None:
        assert lsw.parse_max_compression_ratio(None) == pytest.approx(2.4)
        assert lsw.parse_max_compression_ratio("3") == pytest.approx(3.0)
        assert lsw.parse_max_compression_ratio("two") == pytest.approx(2.4)

    @pytest.mark.parametrize("raw", ["off", "none", "", "  "])
    def test_fallback_temperatures_off(self, lsw, raw) -> None:
        assert lsw.parse_fallback_temperatures(raw) == ()

    def test_fallback_temperatures_value_and_typo(self, lsw) -> None:
        assert lsw.parse_fallback_temperatures(None) == (0.2, 0.4)
        assert lsw.parse_fallback_temperatures("0.2, 0.4,0.6") == (0.2, 0.4, 0.6)
        assert lsw.parse_fallback_temperatures("0.2,hot") == (0.2, 0.4)
        assert lsw.parse_fallback_temperatures("0,0.2") == (0.2, 0.4)  # 0 is attempt 0
        assert lsw.parse_fallback_temperatures("0.2,1.5") == (0.2, 0.4)

    def test_logprob_stat_defaults_to_median(self, lsw) -> None:
        """#329 follow-up: the median gate is the default (live A/B on the
        Spark: 39/215 decoded chunks dropped vs 118/215 under the mean)."""
        assert lsw.parse_logprob_stat(None) == "median"
        assert lsw.parse_logprob_stat("") == "median"
        assert lsw.parse_logprob_stat("mode") == "median"
        assert lsw.DEFAULT_LOGPROB_STAT == "median"

    def test_logprob_stat_mean_is_the_legacy_value(self, lsw) -> None:
        assert lsw.parse_logprob_stat("mean") == "mean"
        assert lsw.parse_logprob_stat(" MEAN ") == "mean"
        assert lsw.parse_logprob_stat(" MEDIAN ") == "median"

    def test_the_policy_default_is_the_median_gate(self, lsw) -> None:
        assert lsw.DecodePolicy().logprob_stat == "median"

    @pytest.mark.parametrize("raw", ["0", "off", "OFF", "none", "", " "])
    def test_max_word_run_off(self, lsw, raw) -> None:
        assert lsw.parse_max_word_run(raw) == 0

    def test_max_word_run_value_and_typo(self, lsw) -> None:
        assert lsw.parse_max_word_run(None) == 4
        assert lsw.parse_max_word_run("6") == 6
        assert lsw.parse_max_word_run("2") == 2
        assert lsw.parse_max_word_run("four") == 4
        assert lsw.parse_max_word_run("-3") == 4
        assert lsw.parse_max_word_run("1") == 4  # 1 would fail every chunk

    def test_vad_mode(self, lsw) -> None:
        assert lsw.parse_vad_mode(None) == "auto"
        assert lsw.parse_vad_mode("off") == "off"
        assert lsw.parse_vad_mode("false") == "off"
        assert lsw.parse_vad_mode("0") == "off"
        assert lsw.parse_vad_mode("silero") == "auto"


class TestCompressionRatio:
    def test_empty_is_zero(self, lsw) -> None:
        assert lsw.compression_ratio("") == 0.0

    def test_a_loop_is_far_above_the_gate(self, lsw) -> None:
        assert lsw.compression_ratio(_LOOP) > 10

    def test_ordinary_hebrew_is_comparable_to_english(self, lsw) -> None:
        """UTF-8 gives every Hebrew letter the same lead byte, which alone
        pushes ~575 chars of ordinary Hebrew prose to ~2.3 — a hair under
        OpenAI's English-calibrated 2.4. The ratio is therefore taken over
        one byte per character, where the same prose scores ~1.7."""
        prose = (_SPEECH + " ") * 1 + (
            "אז מה שקרה זה שהלכתי לחנות ולא היה להם את מה שחיפשתי, אז חזרתי "
            "הביתה והזמנתי באינטרנט. זה הגיע אחרי שבוע, וכשפתחתי את החבילה "
            "ראיתי שזה בכלל לא הצבע שהזמנתי. התקשרתי לשירות לקוחות והם אמרו "
            "שיחליפו לי, אבל עד היום לא קיבלתי כלום."
        )
        assert lsw.compression_ratio(prose) < 2.0

    def test_speech_followed_by_a_loop_trips_the_gate(self, lsw) -> None:
        assert lsw.compression_ratio(_SPEECH + " " + "נפשט קרוב. -" * 20) > 2.4


_WORD_RUN_SPEECH = "בוקר טוב לכולם, אני רוצה להתחיל מהנקודה האחרונה שעצרנו בה בפגישה הקודמת."


class TestLongestWordRun:
    def test_empty_is_zero(self, lsw) -> None:
        assert lsw.longest_word_run("") == 0
        assert lsw.longest_word_run(" ,.- ") == 0

    def test_ordinary_speech_has_no_run(self, lsw) -> None:
        assert lsw.longest_word_run(_SPEECH) == 1
        assert lsw.longest_word_run(_WORD_RUN_SPEECH) == 1

    def test_a_word_repeated_with_punctuation_between_is_one_run(self, lsw) -> None:
        assert lsw.longest_word_run("זה, זה, זה. זה! זה") == 5

    def test_the_longest_run_inside_speech_is_found(self, lsw) -> None:
        text = _SPEECH + " " + "זה, " * 11 + _WORD_RUN_SPEECH
        assert lsw.longest_word_run(text) == 11

    def test_a_two_word_phrase_run_counts(self, lsw) -> None:
        assert lsw.longest_word_run("תודה רבה, תודה רבה, תודה רבה. תודה רבה, תודה רבה") == 5

    def test_a_two_word_phrase_said_twice_is_two(self, lsw) -> None:
        assert lsw.longest_word_run("שלום לך, שלום לך. מה שלומך?") == 2

    def test_case_and_niqqud_do_not_split_a_run(self, lsw) -> None:
        assert lsw.longest_word_run("No no NO nO") == 4
        # the same word with and without niqqud (U+05B8 qamats) is one word
        assert lsw.longest_word_run("\u05d6\u05b8\u05d4 \u05d6\u05d4 \u05d6\u05d4") == 3

    def test_a_word_loop_is_long(self, lsw) -> None:
        assert lsw.longest_word_run(_LOOP) == 30


class TestConfidenceStat:
    def test_mean_is_todays_statistic(self, lsw) -> None:
        assert lsw.confidence([-0.1, -0.1, -2.5], "mean") == pytest.approx(-0.9)

    def test_median_ignores_one_rare_token(self, lsw) -> None:
        assert lsw.confidence([-0.1, -0.1, -2.5], "median") == pytest.approx(-0.1)

    def test_none_without_tokens(self, lsw) -> None:
        assert lsw.confidence([], "median") is None


class TestChunkHasSpeech:
    def test_silence_has_no_speech(self, lsw) -> None:
        calls = []

        def prob(window):
            calls.append(len(window))
            return 0.01

        assert not lsw.chunk_has_speech([0.0] * 16000, 16000, prob, 0.5, 250)
        assert calls
        assert all(n == 512 for n in calls)

    def test_speech_stops_early(self, lsw) -> None:
        calls = []

        def prob(window):
            calls.append(1)
            return 0.9

        assert lsw.chunk_has_speech([0.0] * 16000 * 30, 16000, prob, 0.5, 250)
        # 250 ms at 32 ms per window -> 8 windows, never the whole 30 s.
        assert len(calls) == 8

    def test_a_cough_shorter_than_the_minimum_is_not_speech(self, lsw) -> None:
        seq = iter([0.9, 0.9] + [0.0] * 1000)
        assert not lsw.chunk_has_speech([0.0] * 16000 * 5, 16000, lambda w: next(seq), 0.5, 250)

    def test_a_vad_error_counts_as_speech(self, lsw) -> None:
        def boom(window):
            raise RuntimeError("vad died")

        assert lsw.chunk_has_speech([0.0] * 16000, 16000, boom, 0.5, 250)

    def test_reset_is_called_once_per_chunk(self, lsw) -> None:
        resets = []

        def prob(window):
            return 0.0

        prob.reset = lambda: resets.append(1)
        lsw.chunk_has_speech([0.0] * 4000, 16000, prob, 0.5, 250)
        assert resets == [1]


class _Decoder:
    """A fake Whisper: returns a scripted (text, token_logprobs) per call and
    records the temperature of every attempt."""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.temperatures: list[float] = []

    def __call__(self, samples, temperature):
        self.temperatures.append(temperature)
        return self.results[min(len(self.temperatures) - 1, len(self.results) - 1)]


def _policy(lsw, **over):
    base = dict(
        min_logprob=-0.35,
        logprob_stat="mean",
        max_compression_ratio=2.4,
        fallback_temperatures=(0.2, 0.4),
        max_word_run=4,
    )
    base.update(over)
    return lsw.DecodePolicy(**base)


class TestDecodeChunk:
    def test_confident_speech_is_kept_on_the_first_attempt(self, lsw) -> None:
        dec = _Decoder((_SPEECH, [-0.05] * 20))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw))
        assert outcome.text == _SPEECH
        assert outcome.attempts == 1
        assert not outcome.dropped
        assert dec.temperatures == [0.0]

    def test_a_loop_is_re_decoded_with_temperature_fallback(self, lsw) -> None:
        dec = _Decoder((_LOOP, [-0.01] * 440), (_SPEECH, [-0.1] * 20))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw))
        assert outcome.text == _SPEECH
        assert dec.temperatures == [0.0, 0.2]
        assert outcome.attempts == 2
        assert outcome.retried
        assert not outcome.dropped

    def test_low_confidence_is_retried_before_dropping(self, lsw) -> None:
        dec = _Decoder((_SPEECH, [-0.8] * 20), (_SPEECH, [-0.2] * 20))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw))
        assert outcome.text == _SPEECH
        assert dec.temperatures == [0.0, 0.2]

    def test_every_attempt_failing_drops_the_chunk(self, lsw) -> None:
        dec = _Decoder((_SPEECH, [-0.9] * 20))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw))
        assert outcome.text == ""
        assert outcome.dropped
        assert outcome.reason == "low_logprob"
        assert dec.temperatures == [0.0, 0.2, 0.4]

    def test_a_loop_that_never_clears_is_dropped(self, lsw) -> None:
        dec = _Decoder((_LOOP, [-0.01] * 440))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw, max_word_run=0))
        assert outcome.text == ""
        assert outcome.reason == "compression_ratio"

    def test_silence_is_never_retried(self, lsw) -> None:
        dec = _Decoder(("", []))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw))
        assert outcome.text == ""
        assert not outcome.dropped
        assert dec.temperatures == [0.0]

    def test_a_tags_only_transcript_is_silence_not_a_retry(self, lsw) -> None:
        dec = _Decoder((" \u202b(צחוק)", [-1.5] * 4))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw))
        assert outcome.text == ""
        assert not outcome.dropped
        assert dec.temperatures == [0.0]

    def test_text_is_cleaned_like_today(self, lsw) -> None:
        dec = _Decoder((" \u202bתודה רבה.", [-0.01] * 4))
        assert lsw.decode_chunk([0.0], dec, _policy(lsw)).text == "תודה רבה."

    def test_all_knobs_off_is_todays_single_greedy_gate(self, lsw) -> None:
        off = _policy(lsw, max_compression_ratio=None, fallback_temperatures=(), max_word_run=0)
        loop = _Decoder((_LOOP, [-0.01] * 440))
        assert lsw.decode_chunk([0.0], loop, off).text == _LOOP.strip()  # today's bug, restored
        low = _Decoder((_SPEECH, [-0.9] * 20))
        assert lsw.decode_chunk([0.0], low, off).text == ""
        assert low.temperatures == [0.0]

    def test_the_gate_off_keeps_everything(self, lsw) -> None:
        off = _policy(lsw, min_logprob=None, max_compression_ratio=None)
        dec = _Decoder((_SPEECH, [-3.0] * 20))
        assert lsw.decode_chunk([0.0], dec, off).text == _SPEECH

    def test_median_gate_keeps_speech_with_one_rare_token(self, lsw) -> None:
        lps = [-0.05] * 6 + [-6.0]  # mean -0.90, median -0.05
        mean = lsw.decode_chunk([0.0], _Decoder((_SPEECH, lps)), _policy(lsw))
        median = lsw.decode_chunk(
            [0.0], _Decoder((_SPEECH, lps)), _policy(lsw, logprob_stat="median")
        )
        assert mean.dropped
        assert median.text == _SPEECH

    def test_outcome_carries_both_statistics_and_the_ratio(self, lsw) -> None:
        lps = [-0.05] * 6 + [-6.0]
        outcome = lsw.decode_chunk([0.0], _Decoder((_SPEECH, lps)), _policy(lsw))
        assert outcome.avg_logprob == pytest.approx(sum(lps) / len(lps))
        assert outcome.median_logprob == pytest.approx(-0.05)
        assert outcome.compression_ratio == pytest.approx(lsw.compression_ratio(_SPEECH))


# A confident hallucination: a short word repeated 11 times, punctuation between,
# inside ordinary speech. The n-gram block and the compression ratio miss it.
_WORD_RUN_HALLUCINATION = _SPEECH + " " + "זה, " * 11 + _WORD_RUN_SPEECH


class TestWordRunGate:
    def test_the_fixture_slips_past_the_other_gates(self, lsw) -> None:
        assert lsw.compression_ratio(_WORD_RUN_HALLUCINATION) < 2.4
        assert lsw.longest_word_run(_WORD_RUN_HALLUCINATION) == 11

    def test_a_word_run_is_re_decoded_with_temperature_fallback(self, lsw) -> None:
        dec = _Decoder((_WORD_RUN_HALLUCINATION, [-0.05] * 60), (_SPEECH, [-0.1] * 20))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw))
        assert outcome.text == _SPEECH
        assert dec.temperatures == [0.0, 0.2]
        assert outcome.retried
        assert not outcome.dropped

    def test_a_word_run_that_never_clears_is_dropped_as_word_run(self, lsw) -> None:
        dec = _Decoder((_WORD_RUN_HALLUCINATION, [-0.05] * 60))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw))
        assert outcome.text == ""
        assert outcome.dropped
        assert outcome.reason == "word_run"
        assert outcome.word_run == 11
        assert dec.temperatures == [0.0, 0.2, 0.4]

    def test_the_threshold_is_inclusive(self, lsw) -> None:
        four = _Decoder(("כן, כן, כן, כן.", [-0.05] * 8))
        assert lsw.decode_chunk([0.0], four, _policy(lsw, fallback_temperatures=())).dropped
        three = _Decoder(("כן, כן, כן.", [-0.05] * 6))
        kept = lsw.decode_chunk([0.0], three, _policy(lsw))
        assert kept.text == "כן, כן, כן."
        assert kept.word_run == 3
        assert three.temperatures == [0.0]

    def test_a_two_word_phrase_run_is_a_failure_too(self, lsw) -> None:
        text = _SPEECH + " " + "תודה רבה, " * 5
        dec = _Decoder((text, [-0.05] * 40))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw, fallback_temperatures=()))
        assert outcome.dropped
        assert "word_run" in outcome.reason.split(",")

    def test_off_keeps_the_run(self, lsw) -> None:
        dec = _Decoder((_WORD_RUN_HALLUCINATION, [-0.05] * 60))
        outcome = lsw.decode_chunk([0.0], dec, _policy(lsw, max_word_run=0))
        assert outcome.text == _WORD_RUN_HALLUCINATION
        assert dec.temperatures == [0.0]

    def test_the_log_lines_carry_the_run_not_the_text(self, lsw, caplog) -> None:
        dec = _Decoder((_WORD_RUN_HALLUCINATION, [-0.05] * 60))
        with caplog.at_level("INFO", logger=lsw.logger.name):
            lsw.decode_chunk([0.0], dec, _policy(lsw, fallback_temperatures=(0.2,)))
        lines = [r.getMessage() for r in caplog.records]
        retry = next(ln for ln in lines if ln.startswith("stt_chunk event=retry"))
        drop = next(ln for ln in lines if ln.startswith("stt_chunk event=dropped"))
        for line in (retry, drop):
            assert "reason=word_run" in line
            assert "word_run=11" in line
            assert "max_word_run=4" in line
        assert _WORD_RUN_HALLUCINATION not in retry
        assert _WORD_RUN_HALLUCINATION not in drop  # only the truncated text= prefix


class TestTranscribeChunks:
    def test_vad_skips_silence_without_decoding_it(self, lsw) -> None:
        dec = _Decoder((_SPEECH, [-0.05] * 20))
        windows = [[0.0] * 10, [1.0] * 10]
        texts, summary = lsw.transcribe_chunks(
            windows, dec, _policy(lsw), has_speech=lambda w: w[0] > 0
        )
        assert texts == ["", _SPEECH]
        assert dec.temperatures == [0.0]  # the silent window never reached Whisper
        assert summary.chunks == 2
        assert summary.vad_skipped == 1
        assert summary.decoded == 1

    def test_a_single_window_request_is_never_vad_skipped(self, lsw) -> None:
        """A realtime turn is one window the bridge's own Silero already
        opened on a single above-threshold chunk; a short word ('כן') may hold
        less than STT_VAD_MIN_SPEECH_MS of speech, so it is not re-judged."""
        dec = _Decoder((_SPEECH, [-0.05] * 20))
        asked = []
        texts, summary = lsw.transcribe_chunks(
            [[0.0]], dec, _policy(lsw), has_speech=lambda w: asked.append(1) or False
        )
        assert texts == [_SPEECH]
        assert asked == []
        assert summary.vad_skipped == 0

    def test_without_vad_every_chunk_is_decoded(self, lsw) -> None:
        dec = _Decoder((_SPEECH, [-0.05] * 20))
        texts, summary = lsw.transcribe_chunks([[0.0], [0.0]], dec, _policy(lsw), None)
        assert texts == [_SPEECH, _SPEECH]
        assert summary.vad_skipped == 0

    def test_summary_counts_retries_recoveries_and_drops(self, lsw) -> None:
        dec = _Decoder(
            (_LOOP, [-0.01] * 440),  # chunk 1, attempt 0: loop
            (_SPEECH, [-0.1] * 20),  # chunk 1, attempt 1: recovered
            (_SPEECH, [-0.9] * 20),  # chunk 2: low every time -> dropped
        )
        texts, summary = lsw.transcribe_chunks([[0.0], [0.0]], dec, _policy(lsw), None)
        assert texts == [_SPEECH, ""]
        assert (summary.retried, summary.recovered, summary.dropped) == (2, 1, 1)
        assert summary.decodes == 5

    def test_one_log_line_per_retry_and_drop_plus_a_summary(self, lsw, caplog) -> None:
        dec = _Decoder(
            (_LOOP, [-0.01] * 440),  # chunk 1, attempt 0: loop -> retry
            (_SPEECH, [-0.1] * 20),  # chunk 1, attempt 1: recovered
            (_SPEECH, [-0.9] * 20),  # chunk 2: low -> retry, then dropped
        )
        with caplog.at_level("INFO", logger=lsw.logger.name):
            lsw.transcribe_chunks(
                [[0.0], [0.0]], dec, _policy(lsw, fallback_temperatures=(0.2,)), None
            )
        lines = [r.getMessage() for r in caplog.records]
        retries = [ln for ln in lines if ln.startswith("stt_chunk event=retry")]
        drops = [ln for ln in lines if ln.startswith("stt_chunk event=dropped")]
        summaries = [ln for ln in lines if ln.startswith("stt_request ")]
        assert len(retries) == 2  # chunk 1 attempt 0 (loop), chunk 2 attempt 0 (low)
        assert len(drops) == 1
        assert len(summaries) == 1
        assert "reason=compression_ratio" in retries[0]
        for field in (
            "chunk=1/2",
            "attempt=0",
            "temperature=0.0",
            "avg_logprob=",
            "median_logprob=",
        ):
            assert field in retries[0]
        assert "compression_ratio=" in drops[0]
        assert "reason=low_logprob" in drops[0]
        for field in ("chunks=2", "dropped=1", "retried=2", "recovered=1", "vad_skipped=0"):
            assert field in summaries[0]
