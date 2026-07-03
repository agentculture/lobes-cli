"""Pure pressure policy — maps (swap%, iowait%) + requested_tier to a decision dict.

This module is **side-effect-free**: no I/O, no ``/proc`` reads, no subprocess
calls.  It accepts numeric inputs and returns a plain dict.  The sampler
(:mod:`lobes.runtime._pressure`) is the companion that produces those numbers
from the live host; ``_pressure_policy`` deliberately does not import it.

Tier vocabulary (main / minor / multimodal) and the pressure seam
-----------------------------------------------------------------
Issue #69 reframed the generate-lane capability tiers to **main / minor /
multimodal** (mirroring ``catalog.TIER_ROLE``):

    main        → 27B primary   (full text capability, the former "hard" tier)
    minor       → 4B minor       (fast, low memory, the former "cheap" tier)
    multimodal  → 12B multimodal (text+image+audio — a *different* capability)

**The seam (t6).** ``multimodal`` is NOT a cheaper rung below ``main`` on a
linear capability ladder — it is a *different capability* (vision+audio).  So
"downgrade under pressure" cannot simply walk ``main -> multimodal -> minor``:
swapping a text ``main`` request onto the multimodal gear is not a graceful
degradation, and a ``multimodal`` request cannot be satisfied by ``main`` at
all.  The resolution: **the only cheaper target is ``minor``.**  Under degraded
pressure, a ``main`` *or* a ``multimodal`` request both downgrade to ``minor``
(``reason="pressure"``) — multimodal degrades to minor just like main does,
because it is a capability and not a cheaper tier.  This collapses the old
linear intermediate band: there is no rung between the full tiers and ``minor``.

Back-compat input tiers (``cheap`` / ``normal`` / ``hard``) are still accepted
and normalize to the new vocabulary on output (``cheap``→``minor``,
``normal``→``multimodal``, ``hard``→``main``).

Decision table (#68/#69)
-------------------------
Comparisons are **strictly greater than** (``>``); a value *exactly equal* to a
threshold does NOT trigger the band.  Two degraded signals gate the policy:

+----------------------------------------+--------------------+----------+
| Condition                              | max_allowed_tier   | mode     |
+========================================+====================+==========+
| swap > 75 %  OR  iowait > 50 %         | minor              | degraded |
+----------------------------------------+--------------------+----------+
| otherwise                              | main (full tier)   | warm     |
+----------------------------------------+--------------------+----------+

Under ``warm`` the full tier is granted as requested (``main`` *and*
``multimodal`` allowed — ``max_allowed_tier`` reports ``main`` as the apex of
the ceiling, but ``multimodal`` is a permitted sibling, not a downgrade).  Under
``degraded`` every request resolves to ``minor``.

Retained-but-advisory thresholds
---------------------------------
The pre-#69 no-hard / prefer-cheap thresholds (``SWAP_NO_HARD_THRESHOLD``,
``SWAP_PREFER_CHEAP_THRESHOLD``, ``IOWAIT_NO_HARD_THRESHOLD``) are **kept** as
named, env-overridable constants for observability/tuning and back-compat, but
they no longer impose a separate tier ceiling — under the seam resolution there
is no intermediate rung for them to cap to.  Only the two *degraded* thresholds
participate in :func:`decide`.

Env overrides
-------------
Each threshold constant is readable from a corresponding environment variable at
module import time.  Override example::

    LOBES_SWAP_DEGRADED_THRESHOLD=70 uv run lobes ...

Public API
----------
:func:`decide` is the single entry point; :func:`normalize_tier` exposes the
back-compat → new-vocabulary normalization used by the request layer.
"""

from __future__ import annotations

import math
import os

# ---------------------------------------------------------------------------
# Tier vocabulary — mirror of catalog.TIER_ROLE (kept local so this module stays
# pure stdlib; do not rename a tier without updating catalog.TIER_ROLE too).
# ---------------------------------------------------------------------------

#: Tier alias (both vocabularies) → backend role. Mirrors ``catalog.TIER_ROLE``.
#: The capability-ROLE names (``cortex``/``senses``) alias the same backends as
#: ``main``/``multimodal``, so they normalize (and degrade) identically.
_TIER_ROLE: dict[str, str] = {
    # Primary vocabulary.
    "main": "primary",
    "minor": "minor",
    "multimodal": "multimodal",
    # Back-compat aliases.
    "cheap": "minor",
    "normal": "multimodal",
    "hard": "primary",
    # Capability-ROLE names (alias the same backends as main / multimodal). Kept
    # in the same order as catalog.TIER_ROLE (senses before cortex) so the two
    # dicts stay identical — the mirror guard test asserts equality.
    "senses": "multimodal",
    "cortex": "primary",
}

#: Backend role → canonical new-vocabulary tier name (the inverse of the primary
#: vocabulary rows above).
_ROLE_TO_TIER: dict[str, str] = {
    "primary": "main",
    "minor": "minor",
    "multimodal": "multimodal",
}

_KNOWN_TIERS: frozenset[str] = frozenset(_TIER_ROLE)

#: The full (non-minor) tiers. Under pressure these all collapse to ``minor`` —
#: ``main`` and ``multimodal`` are siblings, not rungs on a linear ladder.
_FULL_CEILING: str = "main"
#: The single cheaper target under degraded pressure.
_DEGRADED_FLOOR: str = "minor"


def _env_float(key: str, default: float) -> float:
    """Read a float from *key* in ``os.environ``; fall back to *default*.

    Non-finite values (``nan``, ``inf``, ``-inf``) are treated as parse
    failures and return *default* — a non-finite threshold silently breaks
    every ``>`` comparison (``nan > x`` is always ``False``).
    """
    try:
        raw = os.environ.get(key)
        value = float(raw) if raw is not None else float(default)
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


# ---------------------------------------------------------------------------
# Threshold constants (config-driven; each has a named env override)
# ---------------------------------------------------------------------------

#: swap_used_percent **above** this value: degraded mode, minor tier only.
#: Env override: ``LOBES_SWAP_DEGRADED_THRESHOLD`` (default 75.0).
SWAP_DEGRADED_THRESHOLD: float = _env_float("LOBES_SWAP_DEGRADED_THRESHOLD", 75.0)

#: iowait_percent **above** this value: emergency degraded mode, minor tier only.
#: Env override: ``LOBES_IOWAIT_DEGRADED_THRESHOLD`` (default 50.0).
IOWAIT_DEGRADED_THRESHOLD: float = _env_float("LOBES_IOWAIT_DEGRADED_THRESHOLD", 50.0)

# --- Retained-but-advisory thresholds (no longer cap the tier; see module doc) ---

#: swap_used_percent advisory warning threshold (pre-#69 "no new hard jobs").
#: Retained for observability/tuning and env-override stability; it does NOT
#: impose a tier ceiling under the seam resolution (no intermediate rung).
#: Env override: ``LOBES_SWAP_NO_HARD_THRESHOLD`` (default 50.0).
SWAP_NO_HARD_THRESHOLD: float = _env_float("LOBES_SWAP_NO_HARD_THRESHOLD", 50.0)

#: swap_used_percent advisory stronger-warning threshold (pre-#69 "prefer cheap").
#: Retained, advisory only (does not cap the tier).
#: Env override: ``LOBES_SWAP_PREFER_CHEAP_THRESHOLD`` (default 65.0).
SWAP_PREFER_CHEAP_THRESHOLD: float = _env_float("LOBES_SWAP_PREFER_CHEAP_THRESHOLD", 65.0)

#: iowait_percent advisory warning threshold (pre-#69 "no new hard jobs").
#: Retained, advisory only (does not cap the tier).
#: Env override: ``LOBES_IOWAIT_NO_HARD_THRESHOLD`` (default 25.0).
IOWAIT_NO_HARD_THRESHOLD: float = _env_float("LOBES_IOWAIT_NO_HARD_THRESHOLD", 25.0)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_tier(tier: str) -> str:
    """Normalize a tier alias (either vocabulary) to its new-vocabulary name.

    ``main``/``hard`` → ``"main"``; ``minor``/``cheap`` → ``"minor"``;
    ``multimodal``/``normal`` → ``"multimodal"``.

    Raises
    ------
    ValueError
        If *tier* is not a known tier alias.
    """
    role = _TIER_ROLE.get(tier)
    if role is None:
        known = ", ".join(sorted(_KNOWN_TIERS))
        raise ValueError(f"unknown tier {tier!r} — must be one of: {known}")
    return _ROLE_TO_TIER[role]


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
        The capability tier the caller asked for.  One of the new-vocabulary
        tiers (``"main"`` / ``"minor"`` / ``"multimodal"``) or a back-compat
        alias (``"cheap"`` / ``"normal"`` / ``"hard"``).

    Returns
    -------
    dict with four keys (tiers are always emitted in the new vocabulary):

    ``mode``
        ``"warm"`` under normal operation; ``"degraded"`` when
        ``swap_used_percent > SWAP_DEGRADED_THRESHOLD`` or
        ``iowait_percent > IOWAIT_DEGRADED_THRESHOLD``.
    ``max_allowed_tier``
        The capability ceiling under current pressure.  ``"minor"`` when
        degraded (the only cheaper target), otherwise ``"main"`` — the apex of
        the full ceiling (``multimodal`` is a permitted sibling, not above
        ``main``).
    ``allowed_tier``
        The tier actually granted for *requested_tier*.  Under ``warm`` this is
        the requested tier (normalized to the new vocabulary, never downgraded —
        a ``multimodal`` request stays ``multimodal``).  Under ``degraded`` it
        is always ``"minor"`` (the seam: both ``main`` and ``multimodal``
        collapse to ``minor``).
    ``reason``
        ``"pressure"`` when the system is in degraded mode **or** the request
        was constrained below what was asked; ``"default"`` otherwise.

    Raises
    ------
    ValueError
        If *requested_tier* is not a known tier alias.

    Boundary behaviour
    ------------------
    Both degraded comparisons are **strictly greater than** (``>``).  A value
    exactly equal to a threshold does **not** trigger degraded mode.  For
    example, ``swap_used_percent == 75.0`` stays ``warm``; ``75.001`` degrades.
    """
    normalized = normalize_tier(requested_tier)  # validates + maps to new vocab

    # Degraded when either signal exceeds its emergency threshold. The seam: the
    # only cheaper target is ``minor`` — there is no intermediate rung, because
    # ``multimodal`` is a different capability, not a cheaper version of ``main``.
    degraded = (
        swap_used_percent > SWAP_DEGRADED_THRESHOLD or iowait_percent > IOWAIT_DEGRADED_THRESHOLD
    )
    mode = "degraded" if degraded else "warm"

    if degraded:
        max_allowed_tier = _DEGRADED_FLOOR  # minor — the only cheaper target
        allowed_tier = _DEGRADED_FLOOR  # main AND multimodal collapse to minor
    else:
        max_allowed_tier = _FULL_CEILING  # main apex; multimodal also permitted
        allowed_tier = normalized  # granted as requested (no downgrade)

    # Reason: pressure whenever degraded OR the request was constrained below
    # what was asked (a full-tier request served as minor).
    constrained = allowed_tier != normalized
    reason = "pressure" if (degraded or constrained) else "default"

    return {
        "mode": mode,
        "max_allowed_tier": max_allowed_tier,
        "reason": reason,
        "allowed_tier": allowed_tier,
    }
