"""Pressure-aware busy/shed backpressure at the gateway (t2, #68/#69/#85).

Two pure, fully-testable pieces live in :mod:`lobes.gateway._tier_request`:

* :func:`resolve_tier_request` — the pure decision function. Given a requested
  tier (or a plain model id), the sampled pressure dict, an override flag and
  the routing table, it returns ``{"busy", "served_name", "served_tier",
  "reason", "requested_tier"}``.  When the pressure verdict is ``shed=True``
  and no override is set, it returns a **busy marker** (``busy=True``,
  ``served_name=None``) instead of resolving the tier through routing.  This
  severs issue #85 (silent upward-fallback substitution).
* :class:`PressureCache` — a non-blocking provider. The 150 ms
  :func:`sample_pressure` never runs on the request path; a background daemon
  thread refreshes a cached value every ``interval`` seconds and ``.current()``
  just returns the cached dict.

Vocabulary: requests/served tiers are reported in the **main/minor/multimodal**
vocabulary (issue #69).

These tests use injected samplers only — no ``/proc`` reads, no real timing on
the assertion path.
"""

from __future__ import annotations

import threading

from lobes.gateway._config import build_config
from lobes.gateway._tier_request import (
    PressureCache,
    is_tier_alias,
    resolve_tier_request,
)

# --- Fleet fixtures ---------------------------------------------------------


# A full three-tier generate fleet with identifiable served names.
def _full_fleet():
    table, _ = build_config(
        {
            "PRIMARY_SERVED_NAME": "PRIMARY",
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MINOR_SERVED_NAME": "MINOR",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
            "MULTIMODAL_SERVED_NAME": "MULTIMODAL",
        }
    )
    return table


def _primary_only():
    table, _ = build_config({"PRIMARY_SERVED_NAME": "PRIMARY"})
    return table


# Primary + multimodal wired, but MINOR is UNWIRED.
# This is the DEFAULT fleet shape (no minor profile activated).
# Regression fixture for issue #85: under pressure, a cortex/main request
# must NOT silently fall back to the multimodal/gemma served name.
def _primary_plus_multimodal():
    table, _ = build_config(
        {
            "PRIMARY_SERVED_NAME": "PRIMARY",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
            "MULTIMODAL_SERVED_NAME": "MULTIMODAL",
        }
    )
    return table


_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}
_HIGH_SWAP = {"swap_used_percent": 80.0, "iowait_percent": 0.0}  # > 75 → busy/shed


# --- is_tier_alias ------------------------------------------------------------


def test_is_tier_alias_recognises_the_tiers_both_vocabularies() -> None:
    # Primary vocabulary.
    assert is_tier_alias("main")
    assert is_tier_alias("minor")
    assert is_tier_alias("multimodal")
    # Back-compat aliases still recognised.
    assert is_tier_alias("cheap")
    assert is_tier_alias("normal")
    assert is_tier_alias("hard")
    # Plain model ids / None are not tiers.
    assert not is_tier_alias("PRIMARY")
    assert not is_tier_alias("Qwen/Qwen3-14B-NVFP4")
    assert not is_tier_alias(None)


def test_is_tier_alias_recognises_cortex_and_senses() -> None:
    """The new capability-ROLE names are tiers too (single source: catalog.TIER_ROLE)."""
    assert is_tier_alias("cortex")
    assert is_tier_alias("senses")


# --- resolve_tier_request: busy/shed contract ---------------------------------


def test_cortex_routes_to_primary_backend_reason_default() -> None:
    """model=cortex routes to the primary backend (served as the 'main' tier)."""
    table = _full_fleet()
    out = resolve_tier_request("cortex", _NO_PRESSURE, override=False, table=table)
    assert out == {
        "busy": False,
        "served_name": "PRIMARY",
        "served_tier": "main",
        "reason": "default",
        "requested_tier": "main",
    }


def test_senses_routes_to_multimodal_backend_reason_default() -> None:
    """model=senses routes to the multimodal backend (served as the 'multimodal' tier)."""
    table = _full_fleet()
    out = resolve_tier_request("senses", _NO_PRESSURE, override=False, table=table)
    assert out == {
        "busy": False,
        "served_name": "MULTIMODAL",
        "served_tier": "multimodal",
        "reason": "default",
        "requested_tier": "multimodal",
    }


def test_no_pressure_main_stays_main_reason_default() -> None:
    table = _full_fleet()
    out = resolve_tier_request("main", _NO_PRESSURE, override=False, table=table)
    assert out == {
        "busy": False,
        "served_name": "PRIMARY",
        "served_tier": "main",
        "reason": "default",
        "requested_tier": "main",
    }


def test_no_pressure_multimodal_stays_multimodal_reason_default() -> None:
    table = _full_fleet()
    out = resolve_tier_request("multimodal", _NO_PRESSURE, override=False, table=table)
    assert out == {
        "busy": False,
        "served_name": "MULTIMODAL",
        "served_tier": "multimodal",
        "reason": "default",
        "requested_tier": "multimodal",
    }


# --- Busy/shed: non-minor tiers under high pressure ---------------------------


def test_high_pressure_sheds_main_with_busy_marker() -> None:
    """Under high pressure, a main request is shed (busy=True, served_name=None)."""
    table = _full_fleet()
    out = resolve_tier_request("main", _HIGH_SWAP, override=False, table=table)
    assert out == {
        "busy": True,
        "served_name": None,
        "served_tier": None,
        "reason": "pressure",
        "requested_tier": "main",
    }


def test_high_pressure_sheds_multimodal_with_busy_marker() -> None:
    """Under high pressure, a multimodal request is shed (busy=True, served_name=None)."""
    table = _full_fleet()
    out = resolve_tier_request("multimodal", _HIGH_SWAP, override=False, table=table)
    assert out == {
        "busy": True,
        "served_name": None,
        "served_tier": None,
        "reason": "pressure",
        "requested_tier": "multimodal",
    }


def test_high_pressure_sheds_cortex_with_busy_marker() -> None:
    """cortex normalizes to main, so it is shed under pressure."""
    table = _full_fleet()
    out = resolve_tier_request("cortex", _HIGH_SWAP, override=False, table=table)
    assert out == {
        "busy": True,
        "served_name": None,
        "served_tier": None,
        "reason": "pressure",
        "requested_tier": "main",
    }


def test_high_pressure_sheds_senses_with_busy_marker() -> None:
    """senses normalizes to multimodal, so it is shed under pressure."""
    table = _full_fleet()
    out = resolve_tier_request("senses", _HIGH_SWAP, override=False, table=table)
    assert out == {
        "busy": True,
        "served_name": None,
        "served_tier": None,
        "reason": "pressure",
        "requested_tier": "multimodal",
    }


# --- Regression for issue #85 ------------------------------------------------


def test_issue_85_regression_cortex_under_pressure_no_upward_fallback() -> None:
    """Regression for #85: a cortex/main request under high pressure on a fleet
    with primary+multimodal wired but minor UNWIRED must return busy=True with
    served_name=None — it must NEVER resolve to the multimodal/gemma served name.

    Before the fix, the upward-fallback in _served_name_for() would resolve
    minor (the decided servable_tier) upward to multimodal since minor was
    absent, silently serving Gemma instead of shedding the request.
    """
    table = _primary_plus_multimodal()
    out = resolve_tier_request("cortex", _HIGH_SWAP, override=False, table=table)
    assert out["busy"] is True
    assert out["served_name"] is None
    assert out["served_tier"] is None
    assert out["reason"] == "pressure"
    assert out["requested_tier"] == "main"
    # The served name must NOT be the multimodal/gemma served name.
    assert out["served_name"] != "MULTIMODAL"


def test_issue_85_regression_senses_under_pressure_no_upward_fallback() -> None:
    """A senses/multimodal request under high pressure on a fleet with primary+
    multimodal wired but minor UNWIRED also returns busy=True, served_name=None."""
    table = _primary_plus_multimodal()
    out = resolve_tier_request("senses", _HIGH_SWAP, override=False, table=table)
    assert out["busy"] is True
    assert out["served_name"] is None
    assert out["served_tier"] is None
    assert out["reason"] == "pressure"
    assert out["requested_tier"] == "multimodal"


# --- Minor is the floor: served even under pressure ---------------------------


def test_minor_request_under_high_pressure_is_served() -> None:
    """An explicit minor request under high pressure is served (not shed)."""
    table = _full_fleet()
    out = resolve_tier_request("minor", _HIGH_SWAP, override=False, table=table)
    assert out == {
        "busy": False,
        "served_name": "MINOR",
        "served_tier": "minor",
        "reason": "default",
        "requested_tier": "minor",
    }


# --- Override forces the requested tier --------------------------------------


def test_override_forces_main_under_high_pressure() -> None:
    table = _full_fleet()
    out = resolve_tier_request("main", _HIGH_SWAP, override=True, table=table)
    assert out == {
        "busy": False,
        "served_name": "PRIMARY",
        "served_tier": "main",
        "reason": "manual_override",
        "requested_tier": "main",
    }


def test_override_forces_multimodal_under_high_pressure() -> None:
    table = _full_fleet()
    out = resolve_tier_request("multimodal", _HIGH_SWAP, override=True, table=table)
    assert out == {
        "busy": False,
        "served_name": "MULTIMODAL",
        "served_tier": "multimodal",
        "reason": "manual_override",
        "requested_tier": "multimodal",
    }


# --- Back-compat aliases -----------------------------------------------------


def test_back_compat_hard_request_normalizes_and_sheds() -> None:
    """A legacy ``hard`` request still works: normalizes to main, and under
    high pressure is shed with busy=True."""
    table = _full_fleet()
    warm = resolve_tier_request("hard", _NO_PRESSURE, override=False, table=table)
    assert warm == {
        "busy": False,
        "served_name": "PRIMARY",
        "served_tier": "main",
        "reason": "default",
        "requested_tier": "main",
    }
    hot = resolve_tier_request("hard", _HIGH_SWAP, override=False, table=table)
    assert hot["busy"] is True
    assert hot["served_name"] is None
    assert hot["reason"] == "pressure"
    assert hot["requested_tier"] == "main"


def test_legacy_keyed_operator_override_applies_on_normalized_tier_path() -> None:
    """A ``GATEWAY_ALIASES`` override keyed by a *legacy* alias (``hard``) still
    takes effect even though the request normalizes to the new vocab (main).
    """
    table, _ = build_config(
        {
            "PRIMARY_SERVED_NAME": "PRIMARY",
            "GATEWAY_ALIASES": "hard=CUSTOM-27B",
        }
    )
    out = resolve_tier_request("hard", _NO_PRESSURE, override=False, table=table)
    assert out["served_name"] == "CUSTOM-27B"
    # The canonical-vocabulary request honours the same override...
    main_out = resolve_tier_request("main", _NO_PRESSURE, override=False, table=table)
    assert main_out["served_name"] == "CUSTOM-27B"
    # ...and the override survives a manual override too (forced, no pressure path).
    forced = resolve_tier_request("hard", _HIGH_SWAP, override=True, table=table)
    assert forced["served_name"] == "CUSTOM-27B"


# --- Pass-through: plain model ids -------------------------------------------


def test_plain_model_id_passes_through_unchanged() -> None:
    table = _full_fleet()
    # A concrete model id (not a tier) is returned verbatim, no downgrade.
    out = resolve_tier_request("PRIMARY", _HIGH_SWAP, override=False, table=table)
    assert out == {
        "busy": False,
        "served_name": "PRIMARY",
        "served_tier": None,
        "reason": "default",
        "requested_tier": None,
    }
    # Even with override set, a plain id is untouched (override only forces tiers).
    out2 = resolve_tier_request("PRIMARY", _HIGH_SWAP, override=True, table=table)
    assert out2 == {
        "busy": False,
        "served_name": "PRIMARY",
        "served_tier": None,
        "reason": "default",
        "requested_tier": None,
    }


# --- Iowait pressure ---------------------------------------------------------


def test_multimodal_under_iowait_pressure_is_shed() -> None:
    """iowait > 50 → busy mode, multimodal is shed."""
    table = _full_fleet()
    pressure = {"swap_used_percent": 0.0, "iowait_percent": 60.0}
    out = resolve_tier_request("multimodal", pressure, override=False, table=table)
    assert out["busy"] is True
    assert out["served_name"] is None
    assert out["reason"] == "pressure"


def test_main_request_below_degraded_is_not_shed() -> None:
    """A mid-band swap (60) does not trigger busy — main is served."""
    table = _full_fleet()
    pressure = {"swap_used_percent": 60.0, "iowait_percent": 30.0}
    out = resolve_tier_request("main", pressure, override=False, table=table)
    assert out["busy"] is False
    assert out["served_name"] == "PRIMARY"
    assert out["served_tier"] == "main"
    assert out["reason"] == "default"


def test_minor_request_below_degraded_is_not_marked_pressure() -> None:
    """swap=60 is below the degraded floor (75): not busy, minor served normally."""
    table = _full_fleet()
    pressure = {"swap_used_percent": 60.0, "iowait_percent": 0.0}
    out = resolve_tier_request("minor", pressure, override=False, table=table)
    assert out["busy"] is False
    assert out["served_name"] == "MINOR"
    assert out["served_tier"] == "minor"
    assert out["reason"] == "default"


# --- PressureCache: non-blocking provider -----------------------------------


def test_current_returns_seeded_sampler_value() -> None:
    sample = {"swap_used_percent": 12.0, "iowait_percent": 3.0}
    cache = PressureCache(sampler=lambda: dict(sample), interval=1000, start=False)
    try:
        assert cache.current() == sample
    finally:
        cache.stop()


def test_current_never_calls_the_sampler_on_the_request_path() -> None:
    # The sampler raises on any call after the constructor seed; .current() must
    # never invoke it (that is what would cost 150 ms per request on the real one).
    calls = {"n": 0}

    def sampler() -> dict:
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("current() must not sample on the request path")
        return {"swap_used_percent": 7.0, "iowait_percent": 1.0}

    cache = PressureCache(sampler=sampler, interval=1000, start=False)
    try:
        for _ in range(2000):
            assert cache.current() == {"swap_used_percent": 7.0, "iowait_percent": 1.0}
    finally:
        cache.stop()
    assert calls["n"] == 1  # only the constructor seed, never on .current()


def test_current_is_isolated_from_caller_mutation() -> None:
    # .current() hands back a copy: a caller mutating it must not corrupt the cache.
    cache = PressureCache(
        sampler=lambda: {"swap_used_percent": 5.0, "iowait_percent": 2.0},
        interval=1000,
        start=False,
    )
    try:
        snapshot = cache.current()
        snapshot["swap_used_percent"] = 999.0
        assert cache.current()["swap_used_percent"] == 5.0
    finally:
        cache.stop()


def test_sampler_failure_seeds_zeros_not_crash() -> None:
    # A sampler that explodes (e.g. /proc unreadable) must degrade to zeros, not
    # take down the gateway thread.
    def boom() -> dict:
        raise OSError("no /proc")

    cache = PressureCache(sampler=boom, interval=1000, start=False)
    try:
        assert cache.current() == {"swap_used_percent": 0.0, "iowait_percent": 0.0}
    finally:
        cache.stop()


def test_background_thread_refreshes_the_cached_value() -> None:
    # The daemon thread re-samples every ``interval``; verify it fires more than
    # once (so the cache tracks live pressure, not just the seed).
    counter = {"n": 0}
    refreshed = threading.Event()

    def sampler() -> dict:
        counter["n"] += 1
        if counter["n"] >= 2:
            refreshed.set()
        return {"swap_used_percent": float(counter["n"]), "iowait_percent": 0.0}

    cache = PressureCache(sampler=sampler, interval=0.01, start=True)
    try:
        assert refreshed.wait(2.0), "background thread never re-sampled"
        assert cache.current()["swap_used_percent"] >= 1.0
    finally:
        cache.stop()
