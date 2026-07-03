"""Pressure-aware tier downgrade + manual override at the gateway (t6, #68/#69).

Two pure, fully-testable pieces live in :mod:`lobes.gateway._tier_request`:

* :func:`resolve_tier_request` — the pure decision function. Given a requested
  tier (or a plain model id), the sampled pressure dict, an override flag and
  the routing table, it returns ``{"served_name", "served_tier", "reason"}``.
  It calls :func:`lobes.gateway._pressure_policy.decide` for the tier ceiling.
* :class:`PressureCache` — a non-blocking provider. The 150 ms
  :func:`sample_pressure` never runs on the request path; a background daemon
  thread refreshes a cached value every ``interval`` seconds and ``.current()``
  just returns the cached dict.

Vocabulary: requests/served tiers are reported in the **main/minor/multimodal**
vocabulary (issue #69). The seam (t6): ``multimodal`` is a *different
capability*, not a cheaper rung — so under degraded pressure both ``main`` and
``multimodal`` downgrade to ``minor`` (the only cheaper target).

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


_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}
_HIGH_SWAP = {"swap_used_percent": 80.0, "iowait_percent": 0.0}  # > 75 → degraded/minor


# --- resolve_tier_request: the pure decision function -----------------------


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


def test_cortex_routes_to_primary_backend_reason_default() -> None:
    """model=cortex routes to the primary backend (served as the 'main' tier)."""
    table = _full_fleet()
    out = resolve_tier_request("cortex", _NO_PRESSURE, override=False, table=table)
    assert out == {"served_name": "PRIMARY", "served_tier": "main", "reason": "default"}


def test_senses_routes_to_multimodal_backend_reason_default() -> None:
    """model=senses routes to the multimodal backend (served as the 'multimodal' tier)."""
    table = _full_fleet()
    out = resolve_tier_request("senses", _NO_PRESSURE, override=False, table=table)
    assert out == {
        "served_name": "MULTIMODAL",
        "served_tier": "multimodal",
        "reason": "default",
    }


def test_cortex_degrades_to_minor_under_high_pressure() -> None:
    """cortex normalizes to the primary role, so it degrades to minor like main."""
    table = _full_fleet()
    out = resolve_tier_request("cortex", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "minor"
    assert out["served_name"] == "MINOR"
    assert out["reason"] == "pressure"


def test_senses_degrades_to_minor_under_high_pressure() -> None:
    """The seam: senses is the multimodal capability — under pressure it collapses
    straight to minor (not an intermediate downgrade rung)."""
    table = _full_fleet()
    out = resolve_tier_request("senses", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "minor"
    assert out["served_name"] == "MINOR"
    assert out["reason"] == "pressure"


def test_no_pressure_main_stays_main_reason_default() -> None:
    table = _full_fleet()
    out = resolve_tier_request("main", _NO_PRESSURE, override=False, table=table)
    assert out == {"served_name": "PRIMARY", "served_tier": "main", "reason": "default"}


def test_no_pressure_multimodal_stays_multimodal_reason_default() -> None:
    table = _full_fleet()
    out = resolve_tier_request("multimodal", _NO_PRESSURE, override=False, table=table)
    assert out == {
        "served_name": "MULTIMODAL",
        "served_tier": "multimodal",
        "reason": "default",
    }


def test_high_swap_downgrades_main_to_minor_reason_pressure() -> None:
    table = _full_fleet()
    out = resolve_tier_request("main", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "minor"
    assert out["served_name"] == "MINOR"  # minor → the minor gear's served name
    assert out["reason"] == "pressure"


def test_high_swap_downgrades_multimodal_to_minor_reason_pressure() -> None:
    """The seam: multimodal is a capability, not a rung — it degrades to minor
    just like main (not to a 'cheaper multimodal', which does not exist)."""
    table = _full_fleet()
    out = resolve_tier_request("multimodal", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "minor"
    assert out["served_name"] == "MINOR"
    assert out["reason"] == "pressure"


def test_override_forces_main_under_high_pressure() -> None:
    table = _full_fleet()
    out = resolve_tier_request("main", _HIGH_SWAP, override=True, table=table)
    assert out == {
        "served_name": "PRIMARY",
        "served_tier": "main",
        "reason": "manual_override",
    }


def test_override_forces_multimodal_under_high_pressure() -> None:
    table = _full_fleet()
    out = resolve_tier_request("multimodal", _HIGH_SWAP, override=True, table=table)
    assert out == {
        "served_name": "MULTIMODAL",
        "served_tier": "multimodal",
        "reason": "manual_override",
    }


def test_back_compat_hard_request_normalizes_and_downgrades() -> None:
    """A legacy ``hard`` request still works: normalizes to main, and under
    high pressure downgrades to minor with reason=pressure."""
    table = _full_fleet()
    warm = resolve_tier_request("hard", _NO_PRESSURE, override=False, table=table)
    assert warm == {"served_name": "PRIMARY", "served_tier": "main", "reason": "default"}
    hot = resolve_tier_request("hard", _HIGH_SWAP, override=False, table=table)
    assert hot["served_tier"] == "minor"
    assert hot["reason"] == "pressure"


def test_legacy_keyed_operator_override_applies_on_normalized_tier_path() -> None:
    """A ``GATEWAY_ALIASES`` override keyed by a *legacy* alias (``hard``) still
    takes effect even though the request normalizes to the new vocab (main).

    Regression for the tier-normalization seam: ``resolve_tier_request`` resolves
    the served name from the *normalized* tier (``hard``→``main``), so an override
    keyed only by ``hard`` would be bypassed without the synonym expansion in
    ``build_config`` — the request would silently fall back to the primary.
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


def test_plain_model_id_passes_through_unchanged() -> None:
    table = _full_fleet()
    # A concrete model id (not a tier) is returned verbatim, no downgrade.
    out = resolve_tier_request("PRIMARY", _HIGH_SWAP, override=False, table=table)
    assert out == {"served_name": "PRIMARY", "served_tier": None, "reason": "default"}
    # Even with override set, a plain id is untouched (override only forces tiers).
    out2 = resolve_tier_request("PRIMARY", _HIGH_SWAP, override=True, table=table)
    assert out2 == {"served_name": "PRIMARY", "served_tier": None, "reason": "default"}


def test_multimodal_under_iowait_pressure_downgrades_to_minor() -> None:
    table = _full_fleet()
    # iowait > 50 → degraded, minor ceiling. multimodal requested → served minor.
    pressure = {"swap_used_percent": 0.0, "iowait_percent": 60.0}
    out = resolve_tier_request("multimodal", pressure, override=False, table=table)
    assert out["served_tier"] == "minor"
    assert out["served_name"] == "MINOR"
    assert out["reason"] == "pressure"


def test_minor_request_below_degraded_is_not_marked_pressure() -> None:
    table = _full_fleet()
    # swap=60 is below the degraded floor (75): not degraded, and minor is not
    # downgraded → reason default.
    pressure = {"swap_used_percent": 60.0, "iowait_percent": 0.0}
    out = resolve_tier_request("minor", pressure, override=False, table=table)
    assert out["served_tier"] == "minor"
    assert out["served_name"] == "MINOR"
    assert out["reason"] == "default"


def test_main_request_below_degraded_is_not_downgraded() -> None:
    """The seam collapse: a mid-band swap (60) no longer caps main — there is no
    intermediate rung, so main is served until the degraded floor."""
    table = _full_fleet()
    pressure = {"swap_used_percent": 60.0, "iowait_percent": 30.0}
    out = resolve_tier_request("main", pressure, override=False, table=table)
    assert out["served_tier"] == "main"
    assert out["served_name"] == "PRIMARY"
    assert out["reason"] == "default"


def test_minor_request_in_degraded_mode_is_marked_pressure() -> None:
    table = _full_fleet()
    # swap=80 > 75 → degraded mode. minor can't drop further but mode==degraded
    # → reason is pressure (the system is under emergency pressure).
    out = resolve_tier_request("minor", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "minor"
    assert out["reason"] == "pressure"


def test_downgrade_honours_upward_fallback_when_gear_absent() -> None:
    # Only the primary is wired → minor falls back UPWARD to primary. A main
    # request under high pressure is "downgraded" to minor, but minor resolves to
    # the only wired gear (primary). served_tier reports the decided tier (minor);
    # served_name honours the upward fallback.
    table = _primary_only()
    out = resolve_tier_request("main", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "minor"
    assert out["served_name"] == "PRIMARY"  # minor → primary (no minor gear wired)
    assert out["reason"] == "pressure"


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
