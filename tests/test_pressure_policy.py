"""Tests for lobes/gateway/_pressure_policy.py — pure pressure policy / state machine.

All tests are pure: they call decide() with explicit numeric inputs and assert
the returned dict.  No I/O, no /proc, no sampler imports.

Vocabulary reframed to **main / minor / multimodal** (issue #69) and the
pressure seam resolved (t6): ``multimodal`` is a *different capability*
(vision+audio), NOT a cheaper rung below ``main``.  The capability ladder is
therefore not linear, so "downgrade under pressure" cannot walk
``main -> multimodal -> minor``.  The only cheaper target is ``minor``:

* **degraded** (swap > 75 OR iowait > 50) → ``max_allowed_tier = minor``;
  every request (``main`` AND ``multimodal``) downgrades to ``minor``.
* **otherwise** → the ceiling is the full tier (``main``/``multimodal`` allowed);
  nothing is downgraded.

Back-compat input tiers (``cheap``/``normal``/``hard``) are still accepted and
normalize to the new vocabulary (cheap→minor, normal→multimodal, hard→main).

Threshold defaults (module-level constants, strict >):
    SWAP_DEGRADED_THRESHOLD     = 75.0   swap > 75 → degraded, ceiling = minor
    IOWAIT_DEGRADED_THRESHOLD   = 50.0   iowait > 50 → degraded, ceiling = minor
    SWAP_NO_HARD_THRESHOLD      = 50.0   retained constant (advisory; no rung)
    SWAP_PREFER_CHEAP_THRESHOLD = 65.0   retained constant (advisory; no rung)
    IOWAIT_NO_HARD_THRESHOLD    = 25.0   retained constant (advisory; no rung)
"""

from __future__ import annotations

import pytest

from lobes.catalog import TIER_ROLE as CATALOG_TIER_ROLE
from lobes.gateway._pressure_policy import (
    _TIER_ROLE,
    IOWAIT_DEGRADED_THRESHOLD,
    IOWAIT_NO_HARD_THRESHOLD,
    SWAP_DEGRADED_THRESHOLD,
    SWAP_NO_HARD_THRESHOLD,
    SWAP_PREFER_CHEAP_THRESHOLD,
    _env_float,
    decide,
)


def test_pressure_tier_role_mirror_matches_catalog() -> None:
    """The local _TIER_ROLE mirror must not drift from catalog.TIER_ROLE.

    _pressure_policy.py keeps a local copy of the tier->role map so the module
    stays pure stdlib (no catalog import in the policy path). This test is the
    guard that the two never diverge — a rename in catalog without updating the
    mirror would silently break pressure-aware routing (colleague review, #69).
    """
    assert _TIER_ROLE == dict(CATALOG_TIER_ROLE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Primary-vocabulary tiers.
_NEW_TIERS = ("main", "minor", "multimodal")
# Back-compat aliases that still reach decide() from older callers / clients.
_BACK_COMPAT_TIERS = ("cheap", "normal", "hard")
_ALL_TIERS = _NEW_TIERS + _BACK_COMPAT_TIERS


def _decide(swap: float, iowait: float, tier: str) -> dict:
    """Thin wrapper so call sites are a bit shorter."""
    return decide(swap_used_percent=swap, iowait_percent=iowait, requested_tier=tier)


# ---------------------------------------------------------------------------
# 1. No-pressure baseline — full tiers granted as requested, nothing downgraded
# ---------------------------------------------------------------------------


class TestNoPressure:
    """swap=0, iowait=0 — everything warm and unconstrained (no downgrade)."""

    def test_main_request_stays_main(self):
        r = _decide(0.0, 0.0, "main")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "main"
        assert r["allowed_tier"] == "main"
        assert r["reason"] == "default"

    def test_multimodal_request_stays_multimodal(self):
        """multimodal is a sibling full tier, NOT a downgrade target — it stays."""
        r = _decide(0.0, 0.0, "multimodal")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "main"
        assert r["allowed_tier"] == "multimodal"
        assert r["reason"] == "default"

    def test_minor_request_stays_minor(self):
        r = _decide(0.0, 0.0, "minor")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "main"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "default"

    def test_mild_pressure_below_degraded_does_not_downgrade(self):
        """swap=60, iowait=30 — below the degraded floor → still full tier.

        The OLD linear policy capped this band to ``normal``; the seam
        resolution collapses that band because ``multimodal`` is not a cheaper
        rung, so a ``main`` request is NOT downgraded until degraded.
        """
        for tier in ("main", "multimodal"):
            r = _decide(60.0, 30.0, tier)
            assert r["mode"] == "warm", tier
            assert r["max_allowed_tier"] == "main", tier
            assert r["reason"] == "default", tier


# ---------------------------------------------------------------------------
# 2. Degraded via swap (> 75) — the seam: main AND multimodal → minor
# ---------------------------------------------------------------------------


class TestDegradedSwap:
    """swap=80 → degraded; every request resolves to the minor floor."""

    def test_main_degraded_to_minor(self):
        r = _decide(80.0, 5.0, "main")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "minor"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"

    def test_multimodal_degraded_to_minor(self):
        """The seam: multimodal is a capability, not a rung — it degrades to
        minor like main does (not to some 'cheaper multimodal')."""
        r = _decide(80.0, 5.0, "multimodal")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "minor"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"

    def test_minor_stays_minor_but_reason_pressure(self):
        """minor is already the floor; allowed==requested but degraded mode still
        marks reason='pressure'."""
        r = _decide(80.0, 5.0, "minor")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "minor"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 3. Degraded via iowait (> 50) — same seam behaviour
# ---------------------------------------------------------------------------


class TestDegradedIowait:
    """iowait=60 → emergency-degraded; main AND multimodal → minor."""

    def test_main_degraded_to_minor(self):
        r = _decide(5.0, 60.0, "main")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "minor"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"

    def test_multimodal_degraded_to_minor(self):
        r = _decide(5.0, 60.0, "multimodal")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "minor"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 4. Back-compat input vocabulary (cheap/normal/hard) normalizes to new vocab
# ---------------------------------------------------------------------------


class TestBackCompatVocabulary:
    """Old-vocabulary requests still work and normalize to main/minor/multimodal."""

    def test_hard_normalizes_to_main_no_pressure(self):
        r = _decide(0.0, 0.0, "hard")
        assert r["allowed_tier"] == "main"
        assert r["max_allowed_tier"] == "main"
        assert r["reason"] == "default"

    def test_normal_normalizes_to_multimodal_no_pressure(self):
        r = _decide(0.0, 0.0, "normal")
        assert r["allowed_tier"] == "multimodal"
        assert r["reason"] == "default"

    def test_cheap_normalizes_to_minor_no_pressure(self):
        r = _decide(0.0, 0.0, "cheap")
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "default"

    def test_hard_degraded_to_minor(self):
        r = _decide(80.0, 0.0, "hard")
        assert r["mode"] == "degraded"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"

    def test_normal_degraded_to_minor(self):
        r = _decide(80.0, 0.0, "normal")
        assert r["mode"] == "degraded"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 5. Combined pressure — either degraded signal triggers the floor
# ---------------------------------------------------------------------------


class TestCombinedPressure:
    def test_swap_degraded_dominates(self):
        """swap=80 (degraded) + iowait=30 (mild) → minor + degraded."""
        r = _decide(80.0, 30.0, "main")
        assert r["mode"] == "degraded"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"

    def test_iowait_degraded_dominates(self):
        """swap=60 (mild) + iowait=60 (degraded) → minor + degraded."""
        r = _decide(60.0, 60.0, "multimodal")
        assert r["mode"] == "degraded"
        assert r["allowed_tier"] == "minor"
        assert r["reason"] == "pressure"

    def test_two_mild_signals_do_not_degrade(self):
        """swap=60 + iowait=30 — both below the degraded floor → full tier."""
        r = _decide(60.0, 30.0, "main")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "main"
        assert r["allowed_tier"] == "main"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 6. Boundary values — strict > on the degraded thresholds
# ---------------------------------------------------------------------------


class TestBoundaries:
    """Strict greater-than on the degraded thresholds (the floor triggers)."""

    def test_swap_exactly_at_degraded_threshold_not_degraded(self):
        """swap == 75 → NOT degraded (75 > 75 is False) → full tier, warm."""
        r = _decide(SWAP_DEGRADED_THRESHOLD, 0.0, "main")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "main"
        assert r["allowed_tier"] == "main"
        assert r["reason"] == "default"

    def test_swap_just_above_degraded_threshold_triggers_minor(self):
        """swap == 75 + ε → degraded, ceiling = minor."""
        r = _decide(SWAP_DEGRADED_THRESHOLD + 0.001, 0.0, "main")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "minor"
        assert r["allowed_tier"] == "minor"

    def test_iowait_exactly_at_degraded_threshold_not_degraded(self):
        """iowait == 50 → NOT degraded (50 > 50 is False) → full tier, warm."""
        r = _decide(0.0, IOWAIT_DEGRADED_THRESHOLD, "main")
        assert r["mode"] == "warm"
        assert r["max_allowed_tier"] == "main"
        assert r["reason"] == "default"

    def test_iowait_just_above_degraded_threshold_triggers_minor(self):
        """iowait == 50 + ε → degraded, ceiling = minor."""
        r = _decide(0.0, IOWAIT_DEGRADED_THRESHOLD + 0.001, "main")
        assert r["mode"] == "degraded"
        assert r["max_allowed_tier"] == "minor"


# ---------------------------------------------------------------------------
# 7. No-hard / prefer-cheap thresholds are retained but advisory (no rung)
# ---------------------------------------------------------------------------


class TestRetainedAdvisoryThresholds:
    """The no-hard / prefer-cheap thresholds are kept (env-overridable) but no
    longer impose a tier ceiling — there is no intermediate rung to drop to."""

    def test_constants_still_exist(self):
        assert SWAP_NO_HARD_THRESHOLD == 50.0
        assert SWAP_PREFER_CHEAP_THRESHOLD == 65.0
        assert IOWAIT_NO_HARD_THRESHOLD == 25.0

    def test_above_no_hard_band_does_not_downgrade(self):
        """swap just over the (advisory) no-hard threshold → still full tier."""
        r = _decide(SWAP_NO_HARD_THRESHOLD + 1.0, 0.0, "main")
        assert r["max_allowed_tier"] == "main"
        assert r["allowed_tier"] == "main"
        assert r["mode"] == "warm"
        assert r["reason"] == "default"

    def test_above_iowait_no_hard_band_does_not_downgrade(self):
        r = _decide(0.0, IOWAIT_NO_HARD_THRESHOLD + 1.0, "main")
        assert r["max_allowed_tier"] == "main"
        assert r["mode"] == "warm"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 8. Return-dict shape
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
    def test_tier_values_in_new_vocab(self, tier: str):
        """max_allowed_tier / allowed_tier are always emitted in the new vocab."""
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert r["max_allowed_tier"] in ("main", "minor", "multimodal")
        assert r["allowed_tier"] in ("main", "minor", "multimodal")


# ---------------------------------------------------------------------------
# 9. Invalid tier raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidTier:
    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            decide(swap_used_percent=10.0, iowait_percent=5.0, requested_tier="ultra")

    def test_empty_string_tier_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            decide(swap_used_percent=10.0, iowait_percent=5.0, requested_tier="")


# ---------------------------------------------------------------------------
# 10. Pure: no I/O side effects
# ---------------------------------------------------------------------------


class TestPurity:
    """decide() must be importable and callable without any /proc access."""

    def test_callable_without_proc(self):
        r = decide(swap_used_percent=0.0, iowait_percent=0.0, requested_tier="main")
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 11. _env_float rejects non-finite values (nan / inf / -inf)
# ---------------------------------------------------------------------------


class TestEnvFloatNonFinite:
    """_env_float must fall back to *default* when the env var parses to a
    non-finite float.  A nan/inf threshold silently breaks every ``>``
    comparison so the pressure downgrade never fires — we treat it like a
    parse failure instead.
    """

    def test_nan_returns_default(self, monkeypatch):
        monkeypatch.setenv("LOBES_TEST_THRESHOLD", "nan")
        assert _env_float("LOBES_TEST_THRESHOLD", 75.0) == 75.0

    def test_inf_returns_default(self, monkeypatch):
        monkeypatch.setenv("LOBES_TEST_THRESHOLD", "inf")
        assert _env_float("LOBES_TEST_THRESHOLD", 50.0) == 50.0

    def test_negative_inf_returns_default(self, monkeypatch):
        monkeypatch.setenv("LOBES_TEST_THRESHOLD", "-inf")
        assert _env_float("LOBES_TEST_THRESHOLD", 25.0) == 25.0

    def test_valid_value_still_accepted(self, monkeypatch):
        """Sanity: a normal finite value is returned unchanged."""
        monkeypatch.setenv("LOBES_TEST_THRESHOLD", "42.5")
        assert _env_float("LOBES_TEST_THRESHOLD", 75.0) == 42.5

    def test_missing_key_returns_default(self, monkeypatch):
        """Key not in env → default returned (existing behaviour, not broken)."""
        monkeypatch.delenv("LOBES_TEST_THRESHOLD", raising=False)
        assert _env_float("LOBES_TEST_THRESHOLD", 99.0) == 99.0
