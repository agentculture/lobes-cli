"""Pure pressure policy — maps (swap%, iowait%) + requested_tier to a decision dict.

This module is **side-effect-free**: no I/O, no ``/proc`` reads, no subprocess
calls.  It accepts numeric inputs and returns a plain dict.  The sampler
(:mod:`lobes.runtime._pressure`) is the companion that produces those numbers
from the live host; ``_pressure_policy`` deliberately does not import it.

Tier ordering (cheap < normal < hard)
--------------------------------------
The three capability tiers mirror ``catalog.TIER_ROLE`` keys:

    cheap   → 4B minor  (fast, low memory)
    normal  → 14B middle (balanced)
    hard    → 27B primary (full capability)

Threshold table (#68)
---------------------
All comparisons are **strictly greater than** (``>``).  A value *exactly equal*
to a threshold does NOT trigger that band.  The effective ``max_allowed_tier``
is the MOST restrictive across both the swap and iowait bands.

+---------------------------+---------------------+-------+--------+
| Condition                 | max_allowed_tier    | mode  | notes  |
+===========================+=====================+=======+========+
| swap > 50 %               | normal              | warm  | band 1 |
+---------------------------+---------------------+-------+--------+
| swap > 65 %               | normal              | warm  | band 2 |
+---------------------------+---------------------+       | (stronger warn; |
|                           |                     |       | same tier cap as band 1) |
+---------------------------+---------------------+-------+--------+
| swap > 75 %               | cheap               | degraded       |
+---------------------------+---------------------+-------+--------+
| iowait > 25 %             | normal              | warm  | band 4 |
+---------------------------+---------------------+-------+--------+
| iowait > 50 %             | cheap               | degraded       |
+---------------------------+---------------------+-------+--------+

When no threshold fires, ``max_allowed_tier = hard`` and ``mode = warm``.

Env overrides
-------------
Each threshold constant is readable from a corresponding environment variable at
module import time.  The variable name and default are documented alongside each
constant.  Override example::

    LOBES_SWAP_DEGRADED_THRESHOLD=70 uv run lobes ...

Public API
----------
:func:`decide` is the single entry point.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Tier ordering (cheap < normal < hard)
# Strings mirror catalog.TIER_ROLE keys — do not rename without updating both.
# ---------------------------------------------------------------------------

_TIER_ORDER: tuple[str, ...] = ("cheap", "normal", "hard")
_KNOWN_TIERS: frozenset[str] = frozenset(_TIER_ORDER)


def _env_float(key: str, default: float) -> float:
    """Read a float from *key* in ``os.environ``; fall back to *default*."""
    try:
        raw = os.environ.get(key)
        return float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


# ---------------------------------------------------------------------------
# Threshold constants (config-driven; each has a named env override)
# ---------------------------------------------------------------------------

#: swap_used_percent **above** this value: no new *hard* (27 B) jobs allowed.
#: Env override: ``LOBES_SWAP_NO_HARD_THRESHOLD`` (default 50.0).
SWAP_NO_HARD_THRESHOLD: float = _env_float("LOBES_SWAP_NO_HARD_THRESHOLD", 50.0)

#: swap_used_percent **above** this value: stronger warning band — prefer
#: cheap/middle only.  The tier ceiling is still *normal* (same as
#: ``SWAP_NO_HARD_THRESHOLD``); this band does not further restrict the cap
#: but is documented and exported for observability / tuning.
#: Env override: ``LOBES_SWAP_PREFER_CHEAP_THRESHOLD`` (default 65.0).
SWAP_PREFER_CHEAP_THRESHOLD: float = _env_float("LOBES_SWAP_PREFER_CHEAP_THRESHOLD", 65.0)

#: swap_used_percent **above** this value: degraded mode, cheap tier only.
#: Env override: ``LOBES_SWAP_DEGRADED_THRESHOLD`` (default 75.0).
SWAP_DEGRADED_THRESHOLD: float = _env_float("LOBES_SWAP_DEGRADED_THRESHOLD", 75.0)

#: iowait_percent **above** this value: no new *hard* (27 B) jobs allowed.
#: Env override: ``LOBES_IOWAIT_NO_HARD_THRESHOLD`` (default 25.0).
IOWAIT_NO_HARD_THRESHOLD: float = _env_float("LOBES_IOWAIT_NO_HARD_THRESHOLD", 25.0)

#: iowait_percent **above** this value: emergency degraded mode, cheap tier only.
#: Env override: ``LOBES_IOWAIT_DEGRADED_THRESHOLD`` (default 50.0).
IOWAIT_DEGRADED_THRESHOLD: float = _env_float("LOBES_IOWAIT_DEGRADED_THRESHOLD", 50.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tier_index(tier: str) -> int:
    """Return the ordinal position of *tier* in ``_TIER_ORDER`` (0 = most restricted)."""
    return _TIER_ORDER.index(tier)


def _min_tier(a: str, b: str) -> str:
    """Return the more restrictive of two tier strings (lower index in _TIER_ORDER)."""
    return a if _tier_index(a) <= _tier_index(b) else b


def _max_allowed_for_swap(swap: float) -> str:
    """Compute the max-allowed tier imposed by swap pressure alone.

    Bands are checked from most to least restrictive so the first match wins.
    All comparisons are strictly ``>`` (threshold value itself is NOT triggered).
    """
    if swap > SWAP_DEGRADED_THRESHOLD:
        return "cheap"
    if swap > SWAP_NO_HARD_THRESHOLD:  # covers both the 50 % and 65 % bands (both → normal)
        return "normal"
    return "hard"


def _max_allowed_for_iowait(iowait: float) -> str:
    """Compute the max-allowed tier imposed by iowait pressure alone.

    All comparisons are strictly ``>`` (threshold value itself is NOT triggered).
    """
    if iowait > IOWAIT_DEGRADED_THRESHOLD:
        return "cheap"
    if iowait > IOWAIT_NO_HARD_THRESHOLD:
        return "normal"
    return "hard"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide(
    swap_used_percent: float,
    iowait_percent: float,
    requested_tier: str,
) -> dict:
    """Map system pressure + a requested tier to a concrete routing decision.

    Parameters
    ----------
    swap_used_percent:
        Fraction of the swap device in use, 0–100.  Typically from
        :func:`lobes.runtime._pressure.sample_pressure`.
    iowait_percent:
        CPU iowait percentage over the last sample interval, 0–100.
    requested_tier:
        The capability tier the caller asked for.  Must be one of
        ``"cheap"``, ``"normal"``, ``"hard"``.

    Returns
    -------
    dict with four keys:

    ``mode``
        ``"warm"`` under normal operation; ``"degraded"`` when
        ``swap_used_percent > SWAP_DEGRADED_THRESHOLD`` or
        ``iowait_percent > IOWAIT_DEGRADED_THRESHOLD``.
    ``max_allowed_tier``
        The highest capability tier permitted under current pressure.
        This is the **most restrictive** cap across both the swap and iowait
        bands (computed as ``min(swap_cap, iowait_cap)`` by tier ordering).
    ``allowed_tier``
        The tier actually granted for *requested_tier*, i.e.
        ``min(requested_tier, max_allowed_tier)`` by tier ordering.
    ``reason``
        ``"pressure"`` when the system is in degraded mode **or** when the
        request was constrained below what was asked (``allowed_tier !=
        requested_tier``); ``"default"`` otherwise.

    Raises
    ------
    ValueError
        If *requested_tier* is not one of ``"cheap"``, ``"normal"``, ``"hard"``.

    Boundary behaviour
    ------------------
    All threshold comparisons are **strictly greater than** (``>``).
    A value exactly equal to a threshold does **not** trigger that band.
    For example, ``swap_used_percent == 50.0`` does NOT enter the no-hard band;
    ``50.001`` does.  This holds for every threshold.
    """
    if requested_tier not in _KNOWN_TIERS:
        known = ", ".join(_TIER_ORDER)
        raise ValueError(f"unknown tier {requested_tier!r} — must be one of: {known}")

    # 1. Compute the tier ceiling imposed by each signal independently.
    swap_cap = _max_allowed_for_swap(swap_used_percent)
    iowait_cap = _max_allowed_for_iowait(iowait_percent)

    # 2. Most restrictive cap wins (min by tier ordering).
    max_allowed_tier = _min_tier(swap_cap, iowait_cap)

    # 3. Mode: degraded when either signal exceeds its emergency threshold.
    degraded = (
        swap_used_percent > SWAP_DEGRADED_THRESHOLD or iowait_percent > IOWAIT_DEGRADED_THRESHOLD
    )
    mode = "degraded" if degraded else "warm"

    # 4. Granted tier: the lower of what was requested and what is allowed.
    allowed_tier = _min_tier(requested_tier, max_allowed_tier)

    # 5. Reason: pressure whenever we are degraded OR the request was constrained.
    constrained = allowed_tier != requested_tier
    reason = "pressure" if (degraded or constrained) else "default"

    return {
        "mode": mode,
        "max_allowed_tier": max_allowed_tier,
        "reason": reason,
        "allowed_tier": allowed_tier,
    }
