"""Pressure-aware tier downgrade + manual override at the gateway (t6, #68).

Two pure, fully-testable pieces live in :mod:`lobes.gateway._tier_request`:

* :func:`resolve_tier_request` — the pure decision function. Given a requested
  tier (or a plain model id), the sampled pressure dict, an override flag and
  the routing table, it returns ``{"served_name", "served_tier", "reason"}``.
  It calls :func:`lobes.gateway._pressure_policy.decide` for the tier ceiling.
* :class:`PressureCache` — a non-blocking provider. The 150 ms
  :func:`sample_pressure` never runs on the request path; a background daemon
  thread refreshes a cached value every ``interval`` seconds and ``.current()``
  just returns the cached dict.

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
            "MIDDLE_BASE_URL": "http://vllm-middle:8000",
            "MIDDLE_SERVED_NAME": "MIDDLE",
        }
    )
    return table


def _primary_only():
    table, _ = build_config({"PRIMARY_SERVED_NAME": "PRIMARY"})
    return table


_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}
_HIGH_SWAP = {"swap_used_percent": 80.0, "iowait_percent": 0.0}  # > 75 → degraded/cheap


# --- resolve_tier_request: the pure decision function -----------------------


def test_is_tier_alias_recognises_only_the_three_tiers() -> None:
    assert is_tier_alias("cheap")
    assert is_tier_alias("normal")
    assert is_tier_alias("hard")
    assert not is_tier_alias("PRIMARY")
    assert not is_tier_alias("Qwen/Qwen3-14B-NVFP4")
    assert not is_tier_alias(None)


def test_no_pressure_hard_stays_hard_reason_default() -> None:
    table = _full_fleet()
    out = resolve_tier_request("hard", _NO_PRESSURE, override=False, table=table)
    assert out == {"served_name": "PRIMARY", "served_tier": "hard", "reason": "default"}


def test_high_swap_downgrades_hard_to_cheap_reason_pressure() -> None:
    table = _full_fleet()
    out = resolve_tier_request("hard", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "cheap"
    assert out["served_name"] == "MINOR"  # cheap → the minor gear's served name
    assert out["reason"] == "pressure"


def test_override_forces_hard_under_high_pressure() -> None:
    table = _full_fleet()
    out = resolve_tier_request("hard", _HIGH_SWAP, override=True, table=table)
    assert out == {
        "served_name": "PRIMARY",
        "served_tier": "hard",
        "reason": "manual_override",
    }


def test_plain_model_id_passes_through_unchanged() -> None:
    table = _full_fleet()
    # A concrete model id (not a tier) is returned verbatim, no downgrade.
    out = resolve_tier_request("PRIMARY", _HIGH_SWAP, override=False, table=table)
    assert out == {"served_name": "PRIMARY", "served_tier": None, "reason": "default"}
    # Even with override set, a plain id is untouched (override only forces tiers).
    out2 = resolve_tier_request("PRIMARY", _HIGH_SWAP, override=True, table=table)
    assert out2 == {"served_name": "PRIMARY", "served_tier": None, "reason": "default"}


def test_normal_under_iowait_pressure_downgrades_to_cheap() -> None:
    table = _full_fleet()
    # iowait > 50 → degraded, cheap ceiling. normal requested → served cheap.
    pressure = {"swap_used_percent": 0.0, "iowait_percent": 60.0}
    out = resolve_tier_request("normal", pressure, override=False, table=table)
    assert out["served_tier"] == "cheap"
    assert out["served_name"] == "MINOR"
    assert out["reason"] == "pressure"


def test_cheap_request_under_pressure_is_not_marked_pressure() -> None:
    table = _full_fleet()
    # cheap is already the floor; high swap can't downgrade it further, and it is
    # not in degraded mode for swap>50/<75... but swap=80 IS degraded. Use the
    # mid band (swap 60: >50 normal-cap, not degraded) so cheap is unconstrained.
    pressure = {"swap_used_percent": 60.0, "iowait_percent": 0.0}
    out = resolve_tier_request("cheap", pressure, override=False, table=table)
    assert out["served_tier"] == "cheap"
    assert out["served_name"] == "MINOR"
    # Not downgraded (already cheap) and not degraded (60 < 75) → default.
    assert out["reason"] == "default"


def test_cheap_request_in_degraded_mode_is_marked_pressure() -> None:
    table = _full_fleet()
    # swap=80 > 75 → degraded mode. cheap can't drop further but mode==degraded
    # → reason is pressure (the system is under emergency pressure).
    out = resolve_tier_request("cheap", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "cheap"
    assert out["reason"] == "pressure"


def test_downgrade_honours_upward_fallback_when_gear_absent() -> None:
    # Only the primary is wired → cheap falls back UPWARD to primary. A hard
    # request under high pressure is "downgraded" to cheap, but cheap resolves to
    # the only wired gear (primary). served_tier reports the decided tier (cheap);
    # served_name honours the upward fallback.
    table = _primary_only()
    out = resolve_tier_request("hard", _HIGH_SWAP, override=False, table=table)
    assert out["served_tier"] == "cheap"
    assert out["served_name"] == "PRIMARY"  # cheap → primary (no minor gear wired)
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
