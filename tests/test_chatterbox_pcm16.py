"""Unit tests for the stdlib-only float→PCM16 conversion helper.

``float_tensor_to_pcm16`` lives in ``chatterbox_server`` but has no GPU/fastapi
deps — it works with plain Python lists, so these tests run in the offline CI
env without any extras installed.
"""

from __future__ import annotations

import struct

import pytest

from model_gear.realtime.chatterbox_server import float_tensor_to_pcm16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack_pcm16(data: bytes) -> list[int]:
    """Unpack raw PCM16 little-endian bytes into a list of signed int16 values."""
    n = len(data) // 2
    return list(struct.unpack(f"<{n}h", data))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_silence_produces_zero_bytes() -> None:
    samples = [0.0] * 8
    result = float_tensor_to_pcm16(samples)
    assert result == b"\x00\x00" * 8


def test_positive_peak_maps_to_max_int16() -> None:
    result = float_tensor_to_pcm16([1.0])
    values = _unpack_pcm16(result)
    assert values == [32767]


def test_negative_peak_maps_to_min_int16() -> None:
    result = float_tensor_to_pcm16([-1.0])
    values = _unpack_pcm16(result)
    assert values == [-32768]


def test_clamping_above_one() -> None:
    # Values > 1.0 are clamped to 1.0 → 32767.
    result = float_tensor_to_pcm16([2.0, 100.0])
    values = _unpack_pcm16(result)
    assert values == [32767, 32767]


def test_clamping_below_negative_one() -> None:
    # Values < -1.0 are clamped to -1.0 → -32768.
    result = float_tensor_to_pcm16([-2.0, -99.9])
    values = _unpack_pcm16(result)
    assert values == [-32768, -32768]


def test_midpoint_half() -> None:
    result = float_tensor_to_pcm16([0.5])
    values = _unpack_pcm16(result)
    # int(0.5 * 32767) = 16383
    assert values == [16383]


def test_output_length_matches_input() -> None:
    n = 100
    result = float_tensor_to_pcm16([0.1] * n)
    assert len(result) == n * 2  # 2 bytes per sample


def test_list_input() -> None:
    """Plain Python list (no torch/numpy) must work."""
    data = float_tensor_to_pcm16([0.0, 0.25, -0.25, 1.0])
    assert len(data) == 8


def test_numpy_array_input() -> None:
    """numpy array must work (numpy is available in the dev env)."""
    np = pytest.importorskip("numpy")
    arr = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    result = float_tensor_to_pcm16(arr)
    values = _unpack_pcm16(result)
    assert values[0] == 0
    assert values[1] == 16383
    assert values[2] == -16384
    assert values[3] == 32767


def test_2d_array_is_squeezed() -> None:
    """A [1, N] shaped array (e.g. model output) must be squeezed to 1-D."""
    np = pytest.importorskip("numpy")
    arr = np.array([[0.0, 1.0]], dtype=np.float32)  # shape (1, 2)
    result = float_tensor_to_pcm16(arr)
    assert len(result) == 4  # 2 samples × 2 bytes


def test_empty_input_returns_empty_bytes() -> None:
    assert float_tensor_to_pcm16([]) == b""
