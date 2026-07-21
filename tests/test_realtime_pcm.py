"""Tests for the /v1/realtime PCM accounting helpers (stdlib-only; no
numpy/scipy/torch, no [realtime] extra).

Covers the pure pieces app.py's WS route (#149 t6) leans on before it ever
reaches for scipy: the resample-needed decision, scipy's own output-length
convention, and WebSocket-frame byte alignment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import lobes.realtime._pcm as _pcm
from lobes.realtime._pcm import needs_resample, resampled_frame_count, take_aligned_samples
from lobes.realtime.protocol import BYTES_PER_SAMPLE, VAD_SAMPLE_RATE

# --- import isolation -------------------------------------------------------


def test_module_imports_without_the_realtime_extra() -> None:
    # If this test file collects at all in the offline dev env (no numpy,
    # scipy, torch, fastapi, httpx installed), the import above already
    # proved it. This test just makes the guarantee explicit and named.
    assert callable(needs_resample)


def test_module_source_never_imports_forbidden_deps() -> None:
    src = Path(_pcm.__file__).read_text(encoding="utf-8")
    forbidden = ("torch", "fastapi", "httpx", "numpy", "scipy", "silero_vad")
    offenders = [
        name
        for name in forbidden
        for line in src.splitlines()
        if line.strip().startswith((f"import {name}", f"from {name}"))
    ]
    assert not offenders, f"_pcm.py imports forbidden deps: {offenders}"


# --- needs_resample ----------------------------------------------------------


def test_16000_input_never_needs_resample() -> None:
    assert needs_resample(16000) is False
    assert needs_resample(16000, target_rate=16000) is False


def test_24000_input_needs_resample() -> None:
    assert needs_resample(24000) is True
    assert needs_resample(VAD_SAMPLE_RATE + 1, target_rate=VAD_SAMPLE_RATE) is True


def test_matching_rates_never_need_resample_regardless_of_value() -> None:
    assert needs_resample(48000, target_rate=48000) is False


# --- resampled_frame_count ----------------------------------------------------


def test_resampled_frame_count_24k_to_16k_matches_scipy_convention() -> None:
    # scipy.signal.resample(x, num) sizing rule: num = round(len(x) * target/input).
    # 768 samples (32ms @ 24kHz) -> 512 samples (32ms @ 16kHz), the exact
    # Silero chunk size this route resamples INTO.
    assert resampled_frame_count(768, 24000, 16000) == 512


def test_resampled_frame_count_is_a_pure_rounding_computation() -> None:
    assert resampled_frame_count(3, 24000, 16000) == round(3 * 16000 / 24000)
    assert resampled_frame_count(1000, 44100, 16000) == round(1000 * 16000 / 44100)


def test_resampled_frame_count_zero_samples_is_zero() -> None:
    assert resampled_frame_count(0, 24000, 16000) == 0


def test_resampled_frame_count_rejects_non_positive_input_rate() -> None:
    with pytest.raises(ValueError):
        resampled_frame_count(100, 0, 16000)
    with pytest.raises(ValueError):
        resampled_frame_count(100, -8000, 16000)


# --- take_aligned_samples ------------------------------------------------------


def test_take_aligned_samples_drains_a_fully_aligned_buffer() -> None:
    buf = bytearray(b"\x01\x02\x03\x04")
    out = take_aligned_samples(buf, BYTES_PER_SAMPLE)
    assert out == b"\x01\x02\x03\x04"
    assert buf == bytearray()


def test_take_aligned_samples_holds_back_a_trailing_odd_byte() -> None:
    buf = bytearray(b"\x01\x02\x03")  # 1.5 samples at 2 bytes/sample
    out = take_aligned_samples(buf, BYTES_PER_SAMPLE)
    assert out == b"\x01\x02"
    assert buf == bytearray(b"\x03")  # held over, not handed downstream


def test_take_aligned_samples_accumulates_the_held_byte_across_calls() -> None:
    buf = bytearray(b"\x01\x02\x03")
    first = take_aligned_samples(buf, BYTES_PER_SAMPLE)
    assert first == b"\x01\x02"
    assert buf == bytearray(b"\x03")

    buf.extend(b"\x04")  # next frame arrives; buffer now holds a full sample
    second = take_aligned_samples(buf, BYTES_PER_SAMPLE)
    assert second == b"\x03\x04"
    assert buf == bytearray()


def test_take_aligned_samples_returns_empty_for_a_buffer_shorter_than_one_sample() -> None:
    buf = bytearray(b"\x01")
    out = take_aligned_samples(buf, BYTES_PER_SAMPLE)
    assert out == b""
    assert buf == bytearray(b"\x01")  # untouched


def test_take_aligned_samples_returns_empty_for_an_empty_buffer() -> None:
    buf = bytearray()
    out = take_aligned_samples(buf, BYTES_PER_SAMPLE)
    assert out == b""
    assert buf == bytearray()


def test_take_aligned_samples_defaults_to_the_protocol_bytes_per_sample() -> None:
    buf = bytearray(b"\x01\x02\x03")
    out = take_aligned_samples(buf)  # no explicit bytes_per_sample
    assert out == b"\x01\x02"
    assert buf == bytearray(b"\x03")
