"""Tests for the host memory-pressure sampler (``lobes.runtime._pressure``).

The sampler reads ``/proc/meminfo`` and ``/proc/stat`` (two snapshots for
iowait).  All tests use fixture strings so they run without a real ``/proc``
or any timing dependency.
"""

from __future__ import annotations

import io
import time

import pytest

from lobes.runtime import _pressure

# ---------------------------------------------------------------------------
# Fixture text fragments
# ---------------------------------------------------------------------------

MEMINFO_NORMAL = """\
MemTotal:       65536000 kB
MemFree:        32768000 kB
SwapTotal:       8388608 kB
SwapFree:        4194304 kB
SwapCached:            0 kB
"""

# SwapTotal == SwapFree → 0% used
MEMINFO_ZERO_SWAP_USED = """\
MemTotal:       65536000 kB
SwapTotal:       4096000 kB
SwapFree:        4096000 kB
"""

# SwapTotal == 0 → no-swap system (divide-by-zero guard)
MEMINFO_NO_SWAP = """\
MemTotal:       65536000 kB
SwapTotal:              0 kB
SwapFree:              0 kB
"""

# /proc/stat baseline snapshot
STAT_BEFORE = """\
cpu  100 20 30 800 50 0 0 0 0 0
cpu0 50 10 15 400 25 0 0 0 0 0
intr 1234 0 0 0
"""

# /proc/stat after snapshot: iowait rose by 10, other fields unchanged
# Before totals: 100+20+30+800+50+0+0+0+0+0 = 1000
# After  totals: 100+20+30+800+60+0+0+0+0+0 = 1010  delta_total=10
# iowait delta = 60-50 = 10  → 10/10 * 100 = 100%
STAT_AFTER_IOWAIT = """\
cpu  100 20 30 800 60 0 0 0 0 0
cpu0 50 10 15 400 30 0 0 0 0 0
intr 1234 0 0 0
"""

# /proc/stat identical to STAT_BEFORE → total_delta == 0 (guard)
STAT_IDENTICAL = STAT_BEFORE


# ---------------------------------------------------------------------------
# parse_swap_percent
# ---------------------------------------------------------------------------


def test_swap_normal() -> None:
    """Half of swap is used → 50.0%."""
    # SwapTotal=8388608, SwapFree=4194304 → used=4194304 → 50%
    result = _pressure.parse_swap_percent(MEMINFO_NORMAL)
    assert result == pytest.approx(50.0)


def test_swap_zero_used() -> None:
    """SwapFree == SwapTotal → 0.0% (no division error)."""
    result = _pressure.parse_swap_percent(MEMINFO_ZERO_SWAP_USED)
    assert result == pytest.approx(0.0)


def test_swap_no_swap_device() -> None:
    """SwapTotal == 0 → 0.0% (divide-by-zero guard)."""
    result = _pressure.parse_swap_percent(MEMINFO_NO_SWAP)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# parse_iowait_percent
# ---------------------------------------------------------------------------


def test_iowait_delta() -> None:
    """iowait rose by 10 out of total-delta 10 → 100.0%."""
    result = _pressure.parse_iowait_percent(STAT_BEFORE, STAT_AFTER_IOWAIT)
    assert result == pytest.approx(100.0)


def test_iowait_no_delta() -> None:
    """Identical snapshots → total_delta == 0 → 0.0% (divide-by-zero guard)."""
    result = _pressure.parse_iowait_percent(STAT_IDENTICAL, STAT_IDENTICAL)
    assert result == pytest.approx(0.0)


def test_iowait_partial() -> None:
    """Custom snapshot where iowait is 5 out of a total delta of 20 → 25.0%."""
    before = "cpu  1000 0 0 0 0 0 0 0 0 0\n"
    after = "cpu  1010 0 0 0 5 0 0 5 0 0\n"  # total_delta=20, iowait_delta=5
    result = _pressure.parse_iowait_percent(before, after)
    assert result == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# sample_pressure — side-effect-free assertion
# ---------------------------------------------------------------------------


def test_sample_pressure_reads_only_proc_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample_pressure() must only open /proc/meminfo and /proc/stat for reading."""
    opened_files: list[tuple[str, str]] = []

    real_open = open  # keep a reference before patching

    def tracking_open(path, mode="r", **kwargs):
        opened_files.append((str(path), mode))
        # Return appropriate fixture content
        if str(path) == "/proc/meminfo":
            return io.StringIO(MEMINFO_NORMAL)
        if str(path) == "/proc/stat":
            return io.StringIO(STAT_BEFORE)
        return real_open(path, mode, **kwargs)

    # Also patch sleep so the test is instant
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.open", tracking_open)

    _pressure.sample_pressure()

    # Only /proc paths should have been opened
    for path, mode in opened_files:
        assert path in ("/proc/meminfo", "/proc/stat"), f"Unexpected file opened: {path!r}"
        assert "w" not in mode, f"File opened for writing: {path!r} (mode={mode!r})"


def test_sample_pressure_returns_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample_pressure() must return a dict with the two documented keys."""
    call_count = 0

    def patched_open(path, mode="r", **kwargs):
        nonlocal call_count
        if str(path) == "/proc/meminfo":
            return io.StringIO(MEMINFO_NORMAL)
        if str(path) == "/proc/stat":
            call_count += 1
            # Return BEFORE on first call, AFTER on second call
            if call_count == 1:
                return io.StringIO(STAT_BEFORE)
            return io.StringIO(STAT_AFTER_IOWAIT)
        raise FileNotFoundError(path)

    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr("builtins.open", patched_open)

    result = _pressure.sample_pressure()

    assert isinstance(result, dict)
    assert set(result.keys()) == {"swap_used_percent", "iowait_percent"}
    assert isinstance(result["swap_used_percent"], float)
    assert isinstance(result["iowait_percent"], float)
