"""Unit tests for :func:`lobes.assess.calibration_knee` (capacity-relative pool
routing, t2 — https://docs/specs/2026-08-27-capacity-relative-pool-routing.md).

All tests are pure: no HTTP, no clock, no live engine. `calibration_knee` takes
a plain list of ``(concurrency, aggregate_tok_s, ttft_s)`` samples and returns a
:class:`~lobes.assess.CalibrationKnee`.
"""

from __future__ import annotations

import lobes.assess as A

# A generous default bound so tests that don't care about TTFT never trip it.
_NO_TTFT_PRESSURE = 60.0


# ---------------------------------------------------------------------------
# Acceptance 1 — plateau: knee is the highest level that still rose meaningfully
# and whose TTFT is under the bound.
# ---------------------------------------------------------------------------


def test_knee_at_throughput_plateau() -> None:
    samples = [
        (1, 10.0, 1.0),
        (2, 19.0, 1.1),  # gain 0.90 -> continue
        (4, 21.0, 1.2),  # gain 0.105 -> continue
        (8, 21.5, 1.3),  # gain 0.024 -> plateau, stop BEFORE this level
    ]
    result = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE)
    assert result.concurrency == 4
    assert result.plateaued is True
    assert result.stopped_by == "plateau"
    # samples returned are the accepted prefix, not including the plateau step
    assert [s[0] for s in result.samples] == [1, 2, 4]


def test_knee_exact_gain_threshold_boundary_does_not_stop() -> None:
    """Gain exactly equal to the threshold counts as still rising (strict <)."""
    samples = [
        (1, 10.0, 1.0),
        (2, 11.0, 1.0),  # gain == 0.10 exactly -> NOT a plateau, continue
        (4, 11.05, 1.0),  # gain ~0.0045 -> plateau, stop
    ]
    result = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE, min_relative_gain=0.1)
    assert result.concurrency == 2
    assert result.stopped_by == "plateau"


def test_custom_min_relative_gain_threshold() -> None:
    samples = [
        (1, 10.0, 1.0),
        (2, 12.0, 1.0),  # gain 0.2
        (4, 13.0, 1.0),  # gain ~0.083 -- plateau only if threshold > 0.083
    ]
    result = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE, min_relative_gain=0.05)
    # 0.083 >= 0.05, so the ramp keeps climbing all the way to the top.
    assert result.concurrency == 4
    assert result.stopped_by == "top_of_ramp"
    assert result.plateaued is False


# ---------------------------------------------------------------------------
# Acceptance 2 — a ramp that never plateaus returns the top level tried, and is
# reported as un-plateaued rather than silently treated as a real knee.
# ---------------------------------------------------------------------------


def test_ramp_that_never_plateaus_reports_top_of_ramp_unplateaued() -> None:
    samples = [
        (1, 10.0, 1.0),
        (2, 20.0, 1.0),  # gain 1.0
        (4, 40.0, 1.0),  # gain 1.0
        (8, 80.0, 1.0),  # gain 1.0 -- still rising at the top of the ramp
    ]
    result = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE)
    assert result.concurrency == 8
    assert result.plateaued is False
    assert result.stopped_by == "top_of_ramp"


# ---------------------------------------------------------------------------
# Acceptance 3 — TTFT crosses the bound before any plateau: return the last
# level under the bound.
# ---------------------------------------------------------------------------


def test_ttft_bound_crossed_before_plateau() -> None:
    samples = [
        (1, 10.0, 0.5),
        (2, 20.0, 0.6),  # still rising, still under bound
        (4, 40.0, 0.9),  # still rising, but TTFT crosses the 0.8s bound here
        (8, 80.0, 1.5),  # never reached -- would have kept rising too
    ]
    result = A.calibration_knee(samples, ttft_bound_s=0.8)
    assert result.concurrency == 2
    assert result.plateaued is False
    assert result.stopped_by == "ttft_bound"
    assert [s[0] for s in result.samples] == [1, 2]


def test_ttft_exactly_at_bound_is_not_a_violation() -> None:
    """TTFT == bound is allowed; only strictly-over trips the guard."""
    samples = [
        (1, 10.0, 0.8),
        (2, 20.0, 0.8),
    ]
    result = A.calibration_knee(samples, ttft_bound_s=0.8)
    assert result.concurrency == 2
    assert result.stopped_by == "top_of_ramp"


def test_ttft_violates_bound_at_the_very_first_sample() -> None:
    """Even the lowest concurrency tried breaches the bound: no admissible level."""
    samples = [
        (1, 10.0, 5.0),
        (2, 20.0, 6.0),
    ]
    result = A.calibration_knee(samples, ttft_bound_s=1.0)
    assert result.concurrency == 0
    assert result.plateaued is False
    assert result.stopped_by == "ttft_bound"
    assert result.samples == ()


# ---------------------------------------------------------------------------
# Acceptance 4 — purity: same input always yields the same output, no I/O.
# ---------------------------------------------------------------------------


def test_calibration_knee_is_pure_and_deterministic() -> None:
    samples = [(1, 10.0, 1.0), (2, 19.0, 1.1), (4, 21.0, 1.2), (8, 21.5, 1.3)]
    r1 = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE)
    r2 = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Edge cases named in the task: empty, single-sample, non-monotonic, out-of-order.
# ---------------------------------------------------------------------------


def test_empty_samples_returns_neutral_empty_result() -> None:
    result = A.calibration_knee([], ttft_bound_s=_NO_TTFT_PRESSURE)
    assert result.concurrency == 0
    assert result.plateaued is False
    assert result.stopped_by == "empty"
    assert result.samples == ()


def test_single_sample_under_bound_is_top_of_ramp_not_a_plateau() -> None:
    """One data point can't demonstrate a plateau -- it's just the top tried."""
    result = A.calibration_knee([(4, 30.0, 1.0)], ttft_bound_s=_NO_TTFT_PRESSURE)
    assert result.concurrency == 4
    assert result.plateaued is False
    assert result.stopped_by == "top_of_ramp"
    assert [s[0] for s in result.samples] == [4]


def test_single_sample_over_bound_is_inadmissible() -> None:
    result = A.calibration_knee([(4, 30.0, 5.0)], ttft_bound_s=1.0)
    assert result.concurrency == 0
    assert result.stopped_by == "ttft_bound"
    assert result.samples == ()


def test_out_of_order_samples_are_sorted_by_concurrency() -> None:
    samples = [
        (4, 21.0, 1.0),
        (1, 10.0, 1.0),
        (8, 21.5, 1.0),  # gain from 4->8 is 0.024 -> plateau
        (2, 19.0, 1.0),
    ]
    result = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE)
    assert result.concurrency == 4
    assert result.stopped_by == "plateau"
    assert [s[0] for s in result.samples] == [1, 2, 4]


def test_non_monotonic_noise_is_treated_as_failing_to_rise_meaningfully() -> None:
    """A throughput dip fails the 'rises meaningfully' test exactly like a plateau."""
    samples = [
        (1, 10.0, 1.0),
        (2, 20.0, 1.0),  # gain 1.0
        (4, 15.0, 1.0),  # dip -- negative gain, well under threshold -> stop
    ]
    result = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE)
    assert result.concurrency == 2
    assert result.stopped_by == "plateau"
    assert result.plateaued is True


def test_zero_throughput_baseline_is_guarded_against_division_by_zero() -> None:
    """A degenerate zero-throughput sample must not raise ZeroDivisionError."""
    samples = [(1, 0.0, 1.0), (2, 5.0, 1.0)]
    result = A.calibration_knee(samples, ttft_bound_s=_NO_TTFT_PRESSURE)
    # Should not raise, and should treat the zero baseline as a degenerate
    # case to skip past rather than crash on.
    assert result.concurrency in (1, 2)
