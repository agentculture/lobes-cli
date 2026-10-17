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
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "hand"
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
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "pressure"

    def test_multimodal_shed_under_swap_pressure(self):
        """multimodal is a capability, not a rung — shed like main."""
        r = _decide(80.0, 5.0, "multimodal")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "pressure"

    def test_minor_never_shed_under_swap_pressure(self):
        """minor is the floor — served even under pressure, never shed."""
        r = _decide(80.0, 5.0, "minor")
        assert r["mode"] == "busy"
        assert r["shed"] is False
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "hand"
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
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "pressure"

    def test_multimodal_shed_under_iowait_pressure(self):
        r = _decide(5.0, 60.0, "multimodal")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "pressure"

    def test_minor_never_shed_under_iowait_pressure(self):
        r = _decide(5.0, 60.0, "minor")
        assert r["mode"] == "busy"
        assert r["shed"] is False
        assert r["servable_tier"] == "hand"
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
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "hand"
        assert r["reason"] == "default"

    def test_hard_shed_under_pressure(self):
        r = _decide(80.0, 0.0, "hard")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "pressure"

    def test_normal_shed_under_pressure(self):
        r = _decide(80.0, 0.0, "normal")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "hand"
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
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "main"
        assert r["reason"] == "pressure"

    def test_senses_shed_under_pressure(self):
        """The seam: senses collapses to shed (a distinct capability,
        not an intermediate rung), exactly like multimodal."""
        r = _decide(80.0, 0.0, "senses")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "multimodal"
        assert r["reason"] == "pressure"

    def test_senses_shed_via_iowait(self):
        r = _decide(0.0, 60.0, "senses")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "hand"
        assert r["reason"] == "pressure"


# ---------------------------------------------------------------------------
# 4c. `worker` capability-ROLE (thor-worker-lobe plan, t1) — its role name IS
# its tier, exactly like muse; sheds to minor under pressure like every other
# non-minor tier.
# ---------------------------------------------------------------------------


class TestWorkerVocabulary:
    """worker is its own backend (mirrors muse) — no pressure means it stays
    'worker'; pressure sheds it to minor exactly like main/multimodal/muse."""

    def test_worker_normalizes_to_worker_no_pressure(self):
        r = _decide(0.0, 0.0, "worker")
        assert r["mode"] == "warm"
        assert r["shed"] is False
        assert r["servable_tier"] == "worker"
        assert r["requested_tier"] == "worker"
        assert r["reason"] == "default"

    def test_worker_shed_under_swap_pressure(self):
        r = _decide(80.0, 5.0, "worker")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "worker"
        assert r["reason"] == "pressure"

    def test_worker_shed_under_iowait_pressure(self):
        r = _decide(5.0, 60.0, "worker")
        assert r["mode"] == "busy"
        assert r["shed"] is True
        assert r["servable_tier"] == "hand"
        assert r["requested_tier"] == "worker"
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
        assert r["servable_tier"] == "hand"

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
        assert r["servable_tier"] == "hand"


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
        # `shed_signal` is ADDITIVE (deviation d1): it names which signal
        # justified a shed, so the line between "genuine exhaustion" and "a
        # host-level reading" is auditable from the decision itself.
        assert set(r) == {
            "mode",
            "shed",
            "reason",
            "servable_tier",
            "requested_tier",
            "shed_signal",
        }

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
        # The OUTPUT vocabulary is the canonical role-derived tier names
        # (_ROLE_TO_TIER). `minor`/`cheap` are accepted INPUTS that normalize to
        # `hand` — the floor's role name — so they never appear on the way out.
        assert r["servable_tier"] in ("main", "hand", "multimodal")
        assert r["requested_tier"] in ("main", "hand", "multimodal")

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


# ---------------------------------------------------------------------------
# 10. Deviation d1 — the pooled carve-out and the engine-state gate
# ---------------------------------------------------------------------------


class TestPooledCarveOut:
    """A POOLED request is not shed on a host-level verdict alone.

    The split, and why it lands where it does:

    * ``swap`` over threshold is first-party evidence the box is PAGING —
      weights and KV live in the memory being paged — so it still sheds
      whether the role is pooled or not.
    * the ENGINE at capacity is the direct serving signal, and sheds too.
    * ``iowait`` over threshold is a whole-host CPU-time statistic charged for
      any process's block wait, including cgroups entirely outside docker. It
      still reports ``mode: busy`` (an honest observation about the host) but
      it no longer sheds a POOLED request on its own.

    Unpooled behaviour is untouched: with ``pooled=False`` (the default, and
    every pre-d1 call site) every row below decides exactly as it did before.
    """

    def test_iowait_alone_does_not_shed_a_pooled_request(self):
        r = decide(0.0, 90.0, "main", pooled=True)
        assert r["shed"] is False
        assert r["shed_signal"] == ""
        assert r["servable_tier"] == "main"
        # The host observation is still published honestly.
        assert r["mode"] == "busy"
        assert r["reason"] == "default"

    def test_iowait_alone_still_sheds_an_unpooled_request(self):
        r = decide(0.0, 90.0, "main")
        assert r["shed"] is True
        assert r["shed_signal"] == "iowait"
        assert r["servable_tier"] == "hand"

    def test_swap_thrash_sheds_a_pooled_request(self):
        r = decide(90.0, 0.0, "main", pooled=True)
        assert r["shed"] is True
        assert r["shed_signal"] == "swap"
        assert r["reason"] == "pressure"

    def test_a_full_engine_sheds_a_pooled_request(self):
        r = decide(0.0, 90.0, "main", pooled=True, engine_active=4, engine_capacity=4)
        assert r["shed"] is True
        assert r["shed_signal"] == "engine"

    def test_an_engine_with_headroom_does_not_shed(self):
        r = decide(0.0, 90.0, "main", pooled=True, engine_active=3, engine_capacity=4)
        assert r["shed"] is False

    def test_an_unpublished_capacity_never_saturates(self):
        # h3: "no capacity published" must never read as "capacity of zero".
        r = decide(0.0, 90.0, "main", pooled=True, engine_active=99, engine_capacity=None)
        assert r["shed"] is False

    def test_a_non_positive_capacity_is_not_a_capacity(self):
        for capacity in (0.0, -4.0):
            r = decide(0.0, 90.0, "main", pooled=True, engine_active=1, engine_capacity=capacity)
            assert r["shed"] is False

    def test_a_non_finite_capacity_is_not_a_capacity(self):
        r = decide(0.0, 90.0, "main", pooled=True, engine_active=1, engine_capacity=float("nan"))
        assert r["shed"] is False

    def test_swap_outranks_the_engine_signal_in_reporting(self):
        r = decide(90.0, 90.0, "main", pooled=True, engine_active=4, engine_capacity=4)
        assert r["shed_signal"] == "swap"

    def test_the_floor_is_never_shed_by_any_signal(self):
        for kwargs in (
            {"pooled": True},
            {"pooled": False},
            {"pooled": True, "engine_active": 8, "engine_capacity": 4},
        ):
            r = decide(90.0, 90.0, "hand", **kwargs)
            assert r["shed"] is False
            assert r["shed_signal"] == ""

    def test_no_pressure_at_all_is_unchanged_by_the_new_arguments(self):
        plain = decide(0.0, 0.0, "main")
        pooled = decide(0.0, 0.0, "main", pooled=True, engine_active=0, engine_capacity=4)
        assert plain == pooled
        assert plain["mode"] == "warm"
        assert plain["shed"] is False


# --- NEVER_SHED_ROLES is load-bearing (code review, finding 10) -------------


@pytest.mark.parametrize("swap,iowait", [(95.0, 0.0), (0.0, 95.0), (99.0, 99.0)])
def test_a_never_shed_role_is_not_shed_under_any_pressure(swap: float, iowait: float) -> None:
    """``decide`` itself honours NEVER_SHED_ROLES — the guarantee does not rest
    on where the gateway's /v1/systemone branch sits in do_POST."""
    from lobes.gateway._pressure_policy import NEVER_SHED_ROLES

    assert NEVER_SHED_ROLES  # the set is not vacuous
    for role in NEVER_SHED_ROLES:
        d = decide(swap, iowait, role)
        assert d["shed"] is False
        assert d["reason"] == "default"
        assert d["shed_signal"] == ""
        assert d["mode"] == "busy"  # the host verdict stays honest
        assert d["requested_tier"] == role
        assert d["servable_tier"] == role


# --- thermal: the fourth shed signal (lane-safety t14, contract §1/§7) -------


class TestThermalShed:
    """``thermal_c`` above the card's shed band sheds every full tier, with
    hysteresis down to the resume band; ``None`` (no reading) never sheds."""

    @pytest.fixture(autouse=True)
    def _spark_band(self, monkeypatch):
        import lobes.gateway._pressure_policy as P

        monkeypatch.setattr(P, "THERMAL_SHED_C", 91.0)
        monkeypatch.setattr(P, "THERMAL_RESUME_C", 86.0)

    @pytest.mark.parametrize("tier", ["main", "cortex", "multimodal", "senses", "worker", "muse"])
    def test_above_shed_band_sheds_full_tiers_with_thermal_signal(self, tier):
        from lobes.gateway._pressure_policy import SHED_SIGNAL_THERMAL

        d = decide(0.0, 0.0, tier, thermal_c=91.5)
        assert d["shed"] is True
        assert d["reason"] == "pressure"
        assert d["shed_signal"] == SHED_SIGNAL_THERMAL == "thermal"
        assert d["mode"] == "busy"
        assert d["servable_tier"] == "hand"

    def test_thermal_sheds_pooled_roles_too(self):
        # Like swap, heat is first-party exhaustion: pooled or not.
        assert decide(0.0, 0.0, "main", pooled=True, thermal_c=95.0)["shed_signal"] == "thermal"

    def test_exactly_at_shed_band_does_not_shed(self):
        d = decide(0.0, 0.0, "main", thermal_c=91.0)
        assert d["shed"] is False
        assert d["mode"] == "warm"

    @pytest.mark.parametrize("tier", ["hand", "minor", "cheap"])
    def test_hand_floor_is_served_when_hot(self, tier):
        d = decide(0.0, 0.0, tier, thermal_c=99.0)
        assert d["shed"] is False
        assert d["shed_signal"] == ""
        assert d["mode"] == "busy"

    def test_never_shed_roles_are_served_when_hot(self):
        from lobes.gateway._pressure_policy import NEVER_SHED_ROLES

        for role in NEVER_SHED_ROLES:
            d = decide(0.0, 0.0, role, thermal_c=99.0, thermal_latched=True)
            assert d["shed"] is False

    def test_none_never_sheds_even_when_latched(self):
        d = decide(0.0, 0.0, "main", thermal_c=None, thermal_latched=True)
        assert d["shed"] is False
        assert d["mode"] == "warm"

    def test_hysteresis_keeps_shedding_between_resume_and_shed(self):
        assert decide(0.0, 0.0, "main", thermal_c=88.0)["shed"] is False  # not latched
        latched = decide(0.0, 0.0, "main", thermal_c=88.0, thermal_latched=True)
        assert latched["shed"] is True
        assert latched["shed_signal"] == "thermal"
        assert decide(0.0, 0.0, "main", thermal_c=86.0, thermal_latched=True)["shed"] is True
        assert decide(0.0, 0.0, "main", thermal_c=85.9, thermal_latched=True)["shed"] is False

    def test_swap_outranks_thermal_in_the_reported_signal(self):
        assert decide(90.0, 0.0, "main", thermal_c=95.0)["shed_signal"] == "swap"

    def test_band_off_never_sheds(self, monkeypatch):
        import lobes.gateway._pressure_policy as P

        monkeypatch.setattr(P, "THERMAL_SHED_C", None)
        monkeypatch.setattr(P, "THERMAL_RESUME_C", None)
        d = decide(0.0, 0.0, "main", thermal_c=120.0, thermal_latched=True)
        assert d["shed"] is False
        assert d["mode"] == "warm"

    def test_latch_follows_the_hysteresis(self):
        from lobes.gateway._pressure_policy import ThermalLatch

        latch = ThermalLatch()
        seq = [80.0, 90.0, 92.0, 89.0, 86.0, 85.0, 90.0, 91.1, None, 90.0]
        got = [latch.update(t) for t in seq]
        assert got == [False, False, True, True, True, False, False, True, False, False]
        assert latch.shedding is False

    def test_passing_the_current_latch_state_is_idempotent(self):
        from lobes.gateway._pressure_policy import ThermalLatch

        latch = ThermalLatch()
        for t in (92.0, 88.0, 85.0):
            hot = latch.update(t)
            assert decide(0.0, 0.0, "main", thermal_c=t, thermal_latched=hot)["shed"] is hot

    def test_override_style_default_args_keep_the_old_contract(self):
        assert decide(0.0, 0.0, "main") == decide(0.0, 0.0, "main", thermal_c=None)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, (None, None)),
        ({"LOBES_THERMAL_SHED_C": ""}, (None, None)),
        ({"LOBES_THERMAL_SHED_C": "hot"}, (None, None)),
        ({"LOBES_THERMAL_SHED_C": "nan"}, (None, None)),
        ({"LOBES_THERMAL_SHED_C": "91", "LOBES_THERMAL_RESUME_C": "86"}, (91.0, 86.0)),
        ({"LOBES_THERMAL_SHED_C": "91"}, (91.0, 91.0)),
        ({"LOBES_THERMAL_SHED_C": "91", "LOBES_THERMAL_RESUME_C": ""}, (91.0, 91.0)),
        # A resume band ABOVE the shed band is nonsense; it clamps to the shed band.
        ({"LOBES_THERMAL_SHED_C": "91", "LOBES_THERMAL_RESUME_C": "95"}, (91.0, 91.0)),
        # RESUME alone, with no SHED, is off.
        ({"LOBES_THERMAL_RESUME_C": "86"}, (None, None)),
    ],
)
def test_thermal_bands_from_env(env, expected) -> None:
    from lobes.gateway._pressure_policy import thermal_bands_from_env

    assert thermal_bands_from_env(env) == expected


# --- contract §8: no lobes code pauses or `docker update`s a container --------

_DOCKER_PAUSE_UPDATE = [
    # shell / string form: `docker pause`, `docker update`, `docker compose pause`
    r"docker\s+(compose\s+)?(un)?(pause|update)\b",
    # argv form: ["docker", ..., "pause"] / ["docker", "update", ...] on one line
    r"""\[\s*["']docker["'][^\]\n]*["'](un)?(pause|update)["']""",
]


def test_no_docker_pause_or_update_anywhere_in_lobes() -> None:
    """Thermal shedding answers 429s; it never throttles a container (§8)."""
    import re
    from pathlib import Path

    import lobes

    root = Path(lobes.__file__).parent
    patterns = [re.compile(p) for p in _DOCKER_PAUSE_UPDATE]
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".png", ".jpg", ".wav"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(p.search(line) for p in patterns):
                hits.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    assert hits == []


@pytest.mark.parametrize(
    "line",
    [
        "docker pause vllm-primary",
        "docker update --cpus 2 vllm-primary",
        "docker compose pause",
        '_run(["docker", "pause", name])',
        "subprocess.run(['docker', 'update', '--memory', '1g', c])",
    ],
)
def test_the_docker_pause_update_grep_is_not_vacuous(line: str) -> None:
    import re

    assert any(re.search(p, line) for p in _DOCKER_PAUSE_UPDATE)


# --- review3 #6: one float parser for the gateway constants and the .env read --


@pytest.mark.parametrize("raw", [None, "", "   ", " 80 ", "80", "1e2", "abc", "nan", "inf", "-inf"])
def test_env_float_and_degraded_thresholds_parse_identically(raw):
    from lobes.gateway import _pressure_policy as pp

    environ = {} if raw is None else {pp.SWAP_THRESHOLD_ENV: raw}
    via_env_float = pp._env_float(pp.SWAP_THRESHOLD_ENV, pp.SWAP_DEGRADED_THRESHOLD, environ)
    assert pp.degraded_thresholds_from_env(environ)[0] == via_env_float
    if raw is not None and raw.strip() in ("80", "1e2"):
        assert via_env_float == float(raw)
    else:
        assert via_env_float == pp.SWAP_DEGRADED_THRESHOLD
