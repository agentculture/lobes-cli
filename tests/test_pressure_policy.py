"""Tests for lobes/gateway/_pressure_policy.py — busy/shed pressure policy.

All tests are pure: they call decide() with explicit numeric inputs and assert
the returned dict.  No I/O, no /proc, no sampler imports.

Busy/shed semantics (t1)
-------------------------
Under swap/iowait pressure the gateway **sheds** full-tier requests with HTTP
429 + Retry-After ("busy, retry shortly").  The degrade-to-minor path is
**removed** — no model substitution occurs.  ``minor`` is the floor: an
explicit minor request is served even under pressure, never shed.

Return keys:
    mode: "warm" | "busy"
    shed: bool (True → shed with 429)
    reason: "pressure" | "default"
    servable_tier: str ("minor" under pressure, else normalized requested tier)
    requested_tier: str (normalized requested tier)

Threshold defaults (module-level constants, strict >):
    SWAP_DEGRADED_THRESHOLD     = 75.0   swap > 75 → busy
    IOWAIT_DEGRADED_THRESHOLD   = 50.0   iowait > 50 → busy
"""

from __future__ import annotations

import pytest

from lobes.catalog import TIER_ROLE as CATALOG_TIER_ROLE
from lobes.gateway._pressure_policy import (
    _TIER_ROLE,
    BUSY_RETRY_AFTER_SECONDS,
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
# 1. No-pressure baseline — full tiers granted as requested, nothing shed
# ---------------------------------------------------------------------------


class TestNoPressure:
    """swap=0, iowait=0 — everything warm and unconstrained (no shed)."""

    def test_main_request_stays_main(self):
        r = _decide(0.0, 0.0, "main")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["servable_tier"] == "main"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "default"

    def test_multimodal_request_stays_multimodal(self):
        """multimodal is a sibling full tier — it stays."""
        r = _decide(0.0, 0.0, "multimodal")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["servable_tier"] == "multimodal"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "default"

    def test_minor_request_stays_minor(self):
        r = _decide(0.0, 0.0, "minor")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "minor"
        assert r["reason"] == "default"

    def test_mild_pressure_below_degraded_does_not_shed(self):
        """swap=60, iowait=30 — below the busy floor → still full tier."""
        for tier in ("main", "multimodal"):
            r = _decide(60.0, 30.0, tier)
            assert r["mode"] == "warm", tier
            assert r["shed"] is False, tier
            assert r["reason"] == "default", tier


# ---------------------------------------------------------------------------
# 2. Busy via swap (> 75) — shed main and multimodal, serve minor
# ---------------------------------------------------------------------------


class TestBusySwap:
    """swap=80 → busy; main and multimodal are shed (429), minor is served."""

    def test_main_shed_under_swap_pressure(self):
        r = _decide(80.0, 5.0, "main")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "pressure"

    def test_multimodal_shed_under_swap_pressure(self):
        """multimodal is a capability, not a rung — shed like main."""
        r = _decide(80.0, 5.0, "multimodal")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "pressure"

    def test_minor_never_shed_under_swap_pressure(self):
        """minor is the floor — served even under pressure, never shed."""
        r = _decide(80.0, 5.0, "minor")
        assert r["mode"] == "busy"
        assert r["shed"] is False
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "minor"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 3. Busy via iowait (> 50) — same shed behaviour
# ---------------------------------------------------------------------------


class TestBusyIowait:
    """iowait=60 → busy; main and multimodal shed, minor served."""

    def test_main_shed_under_iowait_pressure(self):
        r = _decide(5.0, 60.0, "main")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "pressure"

    def test_multimodal_shed_under_iowait_pressure(self):
        r = _decide(5.0, 60.0, "multimodal")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "pressure"

    def test_minor_never_shed_under_iowait_pressure(self):
        r = _decide(5.0, 60.0, "minor")
        assert r["mode"] == "busy"
        assert r["shed"] is False
        assert r["servable_tier"] == "minor"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 4. Back-compat input vocabulary (cheap/normal/hard) normalizes to new vocab
# ---------------------------------------------------------------------------


class TestBackCompatVocabulary:
    """Old-vocabulary requests still work and normalize to main/minor/multimodal."""

    def test_hard_normalizes_to_main_no_pressure(self):
        r = _decide(0.0, 0.0, "hard")
        assert r["servable_tier"] == "main"
        assert r["requested_tier"] == "main"
        assert r["shed"] is False
        assert r["reason"] == "default"

    def test_normal_normalizes_to_multimodal_no_pressure(self):
        r = _decide(0.0, 0.0, "normal")
        assert r["servable_tier"] == "multimodal"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "default"

    def test_cheap_normalizes_to_minor_no_pressure(self):
        r = _decide(0.0, 0.0, "cheap")
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "minor"
        assert r["reason"] == "default"

    def test_hard_shed_under_pressure(self):
        r = _decide(80.0, 0.0, "hard")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "pressure"

    def test_normal_shed_under_pressure(self):
        r = _decide(80.0, 0.0, "normal")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 4b. Capability-ROLE vocabulary (cortex/senses) over primary/multimodal roles
# ---------------------------------------------------------------------------


class TestCortexSensesVocabulary:
    """cortex maps onto the primary role (like main/hard); senses onto the
    multimodal role (like multimodal/normal). Both shed under pressure."""

    def test_cortex_normalizes_to_main_no_pressure(self):
        r = _decide(0.0, 0.0, "cortex")
        assert r["servable_tier"] == "main"
        assert r["requested_tier"] == "main"
        assert r["shed"] is False
        assert r["reason"] == "default"

    def test_senses_normalizes_to_multimodal_no_pressure(self):
        r = _decide(0.0, 0.0, "senses")
        assert r["servable_tier"] == "multimodal"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "default"

    def test_cortex_shed_under_pressure(self):
        r = _decide(80.0, 0.0, "cortex")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "pressure"

    def test_senses_shed_under_pressure(self):
        """The seam: senses collapses to shed (a distinct capability,
        not an intermediate rung), exactly like multimodal."""
        r = _decide(80.0, 0.0, "senses")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "pressure"

    def test_senses_shed_via_iowait(self):
        r = _decide(0.0, 60.0, "senses")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 5. Combined pressure — either signal triggers busy
# ---------------------------------------------------------------------------


class TestCombinedPressure:
    def test_swap_busy_dominates(self):
        """swap=80 (busy) + iowait=30 (mild) → busy, shed main."""
        r = _decide(80.0, 30.0, "main")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["reason"] == "pressure"

    def test_iowait_busy_dominates(self):
        """swap=60 (mild) + iowait=60 (busy) → busy, shed multimodal."""
        r = _decide(60.0, 60.0, "multimodal")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["reason"] == "pressure"

    def test_two_mild_signals_do_not_trigger_busy(self):
        """swap=60 + iowait=30 — both below the busy floor → warm."""
        r = _decide(60.0, 30.0, "main")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["servable_tier"] == "main"
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 6. Boundary values — strict > on the thresholds
# ---------------------------------------------------------------------------


class TestBoundaries:
    """Strict greater-than on the thresholds (busy triggers only above)."""

    def test_swap_exactly_at_threshold_not_busy(self):
        """swap == 75 → NOT busy (75 > 75 is False) → warm."""
        r = _decide(SWAP_DEGRADED_THRESHOLD, 0.0, "main")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["servable_tier"] == "main"
        assert r["reason"] == "default"

    def test_swap_just_above_threshold_triggers_busy(self):
        """swap == 75 + ε → busy, shed main."""
        r = _decide(SWAP_DEGRADED_THRESHOLD + 0.001, 0.0, "main")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"

    def test_iowait_exactly_at_threshold_not_busy(self):
        """iowait == 50 → NOT busy (50 > 50 is False) → warm."""
        r = _decide(0.0, IOWAIT_DEGRADED_THRESHOLD, "main")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["reason"] == "default"

    def test_iowait_just_above_threshold_triggers_busy(self):
        """iowait == 50 + ε → busy, shed main."""
        r = _decide(0.0, IOWAIT_DEGRADED_THRESHOLD + 0.001, "main")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "minor"


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

    def test_above_no_hard_band_does_not_shed(self):
        """swap just over the (advisory) no-hard threshold → still warm."""
        r = _decide(SWAP_NO_HARD_THRESHOLD + 1.0, 0.0, "main")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["servable_tier"] == "main"
        assert r["reason"] == "default"

    def test_above_iowait_no_hard_band_does_not_shed(self):
        r = _decide(0.0, IOWAIT_NO_HARD_THRESHOLD + 1.0, "main")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["reason"] == "default"


# ---------------------------------------------------------------------------
# 8. Return-dict shape
# ---------------------------------------------------------------------------


class TestReturnShape:
    """decide() always returns exactly the five required keys."""

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_required_keys_present(self, tier: str):
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert set(r) == {"mode", "shed", "reason", "servable_tier", "requested_tier"}

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_mode_values_valid(self, tier: str):
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert r["mode"] in ("warm", "busy")

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_reason_values_valid(self, tier: str):
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert r["reason"] in ("default", "pressure")

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_tier_values_in_new_vocab(self, tier: str):
        """servable_tier / requested_tier are always emitted in the new vocab."""
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert r["servable_tier"] in ("main", "minor", "multimodal")
        assert r["requested_tier"] in ("main", "minor", "multimodal")

    @pytest.mark.parametrize("tier", _ALL_TIERS)
    def test_shed_is_bool(self, tier: str):
        r = decide(swap_used_percent=20.0, iowait_percent=10.0, requested_tier=tier)
        assert isinstance(r["shed"], bool)


# ---------------------------------------------------------------------------
# 9. BUSY_RETRY_AFTER_SECONDS constant
# ---------------------------------------------------------------------------


class TestBusyRetryAfter:
    """BUSY_RETRY_AFTER_SECONDS is a module-level int constant."""

    def test_constant_exists_and_is_int(self):
        assert isinstance(BUSY_RETRY_AFTER_SECONDS, int)

    def test_constant_value(self):
        assert BUSY_RETRY_AFTER_SECONDS == 5


# ---------------------------------------------------------------------------
# 10. No degrade path — no LOBES_PRESSURE_POLICY toggle
# ---------------------------------------------------------------------------


class TestNoDegradePath:
    """Verify the degrade-to-minor substitution path is gone."""

    def test_no_degraded_mode(self):
        """mode is never 'degraded' — only 'warm' or 'busy'."""
        r = _decide(80.0, 5.0, "main")
        assert r["mode"] == "busy"
        assert r["mode"] != "degraded"

    def test_no_max_allowed_tier_key(self):
        """The old 'max_allowed_tier' key must not exist."""
        r = _decide(80.0, 5.0, "main")
        assert "max_allowed_tier" not in r

    def test_no_allowed_tier_key(self):
        """The old 'allowed_tier' key must not exist."""
        r = _decide(80.0, 5.0, "main")
        assert "allowed_tier" not in r


# ---------------------------------------------------------------------------
# 11. Invalid tier raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidTier:
    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            decide(swap_used_percent=10.0, iowait_percent=5.0, requested_tier="ultra")

    def test_empty_string_tier_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            decide(swap_used_percent=10.0, iowait_percent=5.0, requested_tier="")


# ---------------------------------------------------------------------------
# 12. Pure: no I/O side effects
# ---------------------------------------------------------------------------


class TestPurity:
    """decide() must be importable and callable without any /proc access."""

    def test_callable_without_proc(self):
        r = decide(swap_used_percent=0.0, iowait_percent=0.0, requested_tier="main")
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# 13. _env_float rejects non-finite values (nan / inf / -inf)
# ---------------------------------------------------------------------------


class TestEnvFloatNonFinite:
    """_env_float must fall back to *default* when the env var parses to a
    non-finite float.  A nan/inf threshold silently breaks every ``>``
    comparison so the pressure shed never fires — we treat it like a
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
