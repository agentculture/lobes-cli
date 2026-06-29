"""Tests for lobes/gateway/_pressure_policy.py — pure pressure policy / state machine.

All tests are pure: they call decide() with explicit numeric inputs and assert
the returned dict.  No I/O, no /proc, no sampler imports.

Threshold defaults (module-level constants, strict >):
    SWAP_NO_HARD_THRESHOLD      = 50.0   swap > 50 → max normal
    SWAP_PREFER_CHEAP_THRESHOLD = 65.0   swap > 65 → still max normal (stronger warn)
    SWAP_DEGRADED_THRESHOLD     = 75.0   swap > 75 → degraded, max cheap
    IOWAIT_NO_HARD_THRESHOLD    = 25.0   iowait > 25 → max normal
    IOWAIT_DEGRADED_THRESHOLD   = 50.0   iowait > 50 → degraded, max cheap
"""

from __future__ import annotations

import pytest

from lobes.gateway._pressure_policy import (
    IOWAIT_DEGRADED_THRESHOLD,
    IOWAIT_NO_HARD_THRESHOLD,
    SWAP_DEGRADED_THRESHOLD,
    SWAP_NO_HARD_THRESHOLD,
    SWAP_PREFER_CHEAP_THRESHOLD,
    decide,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_TIERS = ("cheap", "normal", "hard")


def _decide(swap: float, iowait: float, tier: str) -> dict:
    """Thin wrapper so call sites are a bit shorter."""
    return decide(swap_used_percent=swap, iowait_percent=iowait, requested_tier=tier)


# ---------------------------------------------------------------------------
# 1. No-pressure baseline
# ---------------------------------------------------------------------------


class TestNoPressure:
    """swap=10, iowait=5 — everything should be warm and unconstrained."""

    def test_hard_request_stays_hard(self):
        r = _decide(10.0, 5.0, "hard")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "hard"
        assert r["allowed_tier"] == "hard"
        assert r["reason"] == "default"

    def test_normal_request_stays_normal(self):
        r = _decide(10.0, 5.0, "normal")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "hard"
        assert r["allowed_tier"] == "normal"
        assert r["reason"] == "default"

    def test_cheap_request_stays_cheap(self):
        r = _decide(10.0, 5.0, "cheap")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "hard"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 2. swap > 50 band (no-hard, max = normal, mode = warm)
# ---------------------------------------------------------------------------


class TestSwap55:
    """swap=55 triggers the no-hard band (max_allowed_tier=normal, mode=warm)."""

    def test_hard_request_downgraded_to_normal(self):
        r = _decide(55.0, 5.0, "hard")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "normal"
        assert r["allowed_tier"] == "normal"
        assert r["reason"] == "pressure"

    def test_normal_request_stays_normal(self):
        r = _decide(55.0, 5.0, "normal")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "normal"
        assert r["allowed_tier"] == "normal"
        assert r["reason"] == "default"

    def test_cheap_request_stays_cheap_reason_default(self):
        """cheap is below the max — not constrained — so reason is default."""
        r = _decide(55.0, 5.0, "cheap")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "normal"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 3. swap > 65 band (stronger warn, still max = normal)
# ---------------------------------------------------------------------------


class TestSwap70:
    """swap=70 enters the stronger-warn band — still max=normal, mode=warm."""

    def test_hard_request_downgraded_to_normal(self):
        r = _decide(70.0, 5.0, "hard")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "normal"
        assert r["allowed_tier"] == "normal"
        assert r["reason"] == "pressure"

    def test_cheap_request_stays_cheap(self):
        r = _decide(70.0, 5.0, "cheap")
        assert r["mode"] == "warm"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 4. swap > 75 band — degraded, max = cheap
# ---------------------------------------------------------------------------


class TestSwap80Degraded:
    """swap=80 triggers degraded mode.  Every request resolves to cheap."""

    def test_hard_degraded_to_cheap(self):
        r = _decide(80.0, 5.0, "hard")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "cheap"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "pressure"

    def test_normal_degraded_to_cheap(self):
        r = _decide(80.0, 5.0, "normal")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "cheap"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "pressure"

    def test_cheap_stays_cheap_but_reason_pressure(self):
        """cheap is the only permitted tier; even though allowed==requested,
        the reason is still 'pressure' because we are in degraded mode."""
        r = _decide(80.0, 5.0, "cheap")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "cheap"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 5. iowait > 25 band (max = normal, mode = warm)
# ---------------------------------------------------------------------------


class TestIowait30:
    """iowait=30 triggers the no-hard band via iowait."""

    def test_hard_downgraded_to_normal(self):
        r = _decide(10.0, 30.0, "hard")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "normal"
        assert r["allowed_tier"] == "normal"
        assert r["reason"] == "pressure"

    def test_cheap_stays_cheap(self):
        r = _decide(10.0, 30.0, "cheap")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "normal"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 6. iowait > 50 band — emergency degraded, max = cheap
# ---------------------------------------------------------------------------


class TestIowait60Degraded:
    """iowait=60 triggers emergency-degraded mode."""

    def test_hard_degraded_to_cheap(self):
        r = _decide(10.0, 60.0, "hard")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "cheap"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "pressure"

    def test_normal_degraded_to_cheap(self):
        r = _decide(10.0, 60.0, "normal")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "cheap"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "pressure"

    def test_cheap_stays_cheap_reason_pressure(self):
        r = _decide(10.0, 60.0, "cheap")
        assert r["mode"] == "degraded"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 7. Combined pressure — most restrictive wins
# ---------------------------------------------------------------------------


class TestCombinedPressure:
    """swap=55 (max=normal) + iowait=60 (max=cheap,degraded) → cheap + degraded."""

    def test_most_restrictive_across_signals(self):
        r = _decide(55.0, 60.0, "hard")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "cheap"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "pressure"

    def test_combined_swap_degraded_and_iowait_no_hard(self):
        """swap=80 (degraded/cheap) + iowait=30 (normal) → cheap + degraded."""
        r = _decide(80.0, 30.0, "normal")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "cheap"
        assert r["allowed_tier"] == "cheap"
        assert r["reason"] == "pressure"

    def test_two_independent_no_hard_signals_give_normal(self):
        """swap=55 (normal) + iowait=30 (normal) → max=normal, mode=warm."""
        r = _decide(55.0, 30.0, "hard")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "normal"
        assert r["allowed_tier"] == "normal"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 8. Boundary values — strict > (threshold value itself is NOT triggered)
# ---------------------------------------------------------------------------


class TestBoundaries:
    """Strict greater-than: the threshold value itself does NOT trigger the band.

    E.g. swap_used_percent == 50.0 does NOT enter the no-hard band;
    50.0001 does.  Same logic for all five thresholds.
    """

    def test_swap_exactly_at_no_hard_threshold_not_triggered(self):
        """swap == SWAP_NO_HARD_THRESHOLD (50) → still max_allowed_tier = hard."""
        r = _decide(SWAP_NO_HARD_THRESHOLD, 0.0, "hard")
        assert r["max_allowed_tier"] == "hard"
        assert r["mode"] == "warm"
        assert r["reason"] == "default"

    def test_swap_just_above_no_hard_threshold_triggered(self):
        """swap == 50 + ε → max = normal."""
        r = _decide(SWAP_NO_HARD_THRESHOLD + 0.001, 0.0, "hard")
        assert r["max_allowed_tier"] == "normal"

    def test_swap_exactly_at_prefer_cheap_threshold_still_normal(self):
        """swap == SWAP_PREFER_CHEAP_THRESHOLD (65) → the no-hard band (>50) IS
        triggered, so max=normal; the prefer-cheap band itself is not (65 > 65
        is False) but that band also gives max=normal, so the result is normal."""
        r = _decide(SWAP_PREFER_CHEAP_THRESHOLD, 0.0, "hard")
        assert r["max_allowed_tier"] == "normal"
        assert r["mode"] == "warm"

    def test_swap_exactly_at_degraded_threshold_not_degraded(self):
        """swap == SWAP_DEGRADED_THRESHOLD (75) → the no-hard / prefer-cheap
        bands fire (>50, >65) giving max=normal; the degraded band (>75) does
        NOT fire (75 > 75 is False) so mode stays warm."""
        r = _decide(SWAP_DEGRADED_THRESHOLD, 0.0, "hard")
        assert r["max_allowed_tier"] == "normal"
        assert r["mode"] == "warm"

    def test_swap_just_above_degraded_threshold_triggers_degraded(self):
        """swap == 75 + ε → max = cheap, mode = degraded."""
        r = _decide(SWAP_DEGRADED_THRESHOLD + 0.001, 0.0, "hard")
        assert r["max_allowed_tier"] == "cheap"
        assert r["mode"] == "degraded"

    def test_iowait_exactly_at_no_hard_threshold_not_triggered(self):
        """iowait == IOWAIT_NO_HARD_THRESHOLD (25) → max = hard (not triggered)."""
        r = _decide(0.0, IOWAIT_NO_HARD_THRESHOLD, "hard")
        assert r["max_allowed_tier"] == "hard"
        assert r["mode"] == "warm"
        assert r["reason"] == "default"

    def test_iowait_just_above_no_hard_threshold_triggered(self):
        """iowait == 25 + ε → max = normal."""
        r = _decide(0.0, IOWAIT_NO_HARD_THRESHOLD + 0.001, "hard")
        assert r["max_allowed_tier"] == "normal"

    def test_iowait_exactly_at_degraded_threshold_not_degraded(self):
        """iowait == IOWAIT_DEGRADED_THRESHOLD (50) → the no-hard band (>25) IS
        triggered (max=normal); the degraded band (>50) is NOT (50 > 50 is False)
        so mode=warm."""
        r = _decide(0.0, IOWAIT_DEGRADED_THRESHOLD, "hard")
        assert r["max_allowed_tier"] == "normal"
        assert r["mode"] == "warm"

    def test_iowait_just_above_degraded_threshold_triggers_degraded(self):
        """iowait == 50 + ε → max = cheap, mode = degraded."""
        r = _decide(0.0, IOWAIT_DEGRADED_THRESHOLD + 0.001, "hard")
        assert r["max_allowed_tier"] == "cheap"
        assert r["mode"] == "degraded"


# ---------------------------------------------------------------------------
# 9. Return-dict shape
# ---------------------------------------------------------------------------


class TestReturnShape:
    """decide() always returns exactly the four required keys."""

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_required_keys_present(self, tier: str):
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert set(r) == {"mode", "max_allowed_tier", "reason", "allowed_tier"}

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_mode_values_valid(self, tier: str):
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert r["mode"] in ("warm", "degraded")

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_reason_values_valid(self, tier: str):
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert r["reason"] in ("default", "pressure")

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_tier_values_valid(self, tier: str):
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert r["max_allowed_tier"] in ("cheap", "normal", "hard")
        assert r["allowed_tier"] in ("cheap", "normal", "hard")


# ---------------------------------------------------------------------------
# 10. Invalid tier raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidTier:
    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            decide(swap_used_percent=10.0, iowait_percent=5.0, requested_tier="ultra")

    def test_empty_string_tier_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            decide(swap_used_percent=10.0, iowait_percent=5.0, requested_tier="")


# ---------------------------------------------------------------------------
# 11. Pure: no I/O side effects
# ---------------------------------------------------------------------------


class TestPurity:
    """decide() must be importable and callable without any /proc access."""

    def test_callable_without_proc(self):
        """If this import+call succeeds, the module has no implicit I/O at import."""
        # The import already happened at the top of the file; this just documents
        # that calling decide() in a sandbox with no /proc is fine.
        r = decide(swap_used_percent=0.0, iowait_percent=0.0, requested_tier="hard")
        assert isinstance(r, dict)
