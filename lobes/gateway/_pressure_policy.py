"""Pure pressure policy — busy/shed semantics under swap/iowait pressure.

This module is **side-effect-free**: no I/O, no ``/proc`` reads, no subprocess
calls.  It accepts numeric inputs and returns a plain dict.  The sampler
(:mod:`lobes.runtime._pressure`) is the companion that produces those numbers
from the live host; ``_pressure_policy`` deliberately does not import it.

Busy/shed semantics (t1)
-------------------------
Under swap/iowait pressure the gateway **stops substituting** a cheaper or
different model.  Instead it sheds full-tier requests with HTTP 429 +
Retry-After ("busy, retry shortly").  The degrade-to-minor path is **removed**
outright — no model substitution occurs.

``minor`` is the floor: an explicit minor request is served even under pressure,
never shed.  There is no ``LOBES_PRESSURE_POLICY`` toggle.

Return keys from :func:`decide`:

    mode: "warm" | "busy"          — box-level pressure state
    shed: bool                       — True → shed this request (429)
    reason: "pressure" | "default"  — "pressure" when shed, else "default"
    servable_tier: str               — "minor" under pressure, else normalized tier
    requested_tier: str              — normalize_tier(requested_tier)

Tier vocabulary (main / minor / multimodal)
-------------------------------------------
Issue #69 reframed the generate-lane capability tiers to **main / minor /
multimodal** (mirroring ``catalog.TIER_ROLE``):

    main        → 27B primary   (full text capability, the former "hard" tier)
    minor       → 4B minor       (fast, low memory, the former "cheap" tier)
    multimodal  → 12B multimodal (text+image+audio — a *different* capability)

Back-compat input tiers (``cheap`` / ``normal`` / ``hard``) are still accepted
and normalize to the new vocabulary on output (``cheap``→``minor``,
``normal``→``multimodal``, ``hard``→``main``).

Decision rules (#68/#69)
-------------------------
Comparisons are **strictly greater than** (``>``); a value *exactly equal* to a
threshold does NOT trigger the busy band.  Two pressure signals gate the policy:

+----------------------------------------+------+-------------------+
| Condition                              | mode | shed (non-minor)  |
+========================================+======+===================+
| swap > 75 %  OR  iowait > 50 %         | busy | True (429)        |
+----------------------------------------+------+-------------------+
| otherwise                              | warm | False             |
+----------------------------------------+------+-------------------+

Under ``warm`` the full tier is granted as requested.  Under ``busy`` every
non-minor request is shed (429); ``minor`` is the floor and is always served.

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
#: ``main``/``multimodal``, so they normalize (and shed) identically.
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

#: swap_used_percent **above** this value: busy mode, shed non-minor requests.
#: Env override: ``LOBES_SWAP_DEGRADED_THRESHOLD`` (default 75.0).
SWAP_DEGRADED_THRESHOLD: float = _env_float("LOBES_SWAP_DEGRADED_THRESHOLD", 75.0)

#: iowait_percent **above** this value: busy mode, shed non-minor requests.
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
# Busy retry-after default (used by the gateway for HTTP 429 Retry-After header)
# ---------------------------------------------------------------------------

#: Static default seconds for the Retry-After header on busy (429) responses.
BUSY_RETRY_AFTER_SECONDS: int = 5


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
    """Map system pressure + a requested tier to a busy/shed routing decision.

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
    dict with five keys (tiers are always emitted in the new vocabulary):

    ``mode``
        ``"warm"`` under normal operation; ``"busy"`` when
        ``swap_used_percent > SWAP_DEGRADED_THRESHOLD`` or
        ``iowait_percent > IOWAIT_DEGRADED_THRESHOLD``.
    ``shed``
        ``True`` when the request must be shed (HTTP 429).  Under ``busy``
        mode, all non-minor requests are shed; ``minor`` is the floor and is
        never shed.
    ``reason``
        ``"pressure"`` when shed; ``"default"`` otherwise.
    ``servable_tier``
        ``"minor"`` under pressure (the only servable tier); otherwise the
        normalized requested tier.
    ``requested_tier``
        The normalized tier name for the input *requested_tier*.

    Raises
    ------
    ValueError
        If *requested_tier* is not a known tier alias.

    Boundary behaviour
    ------------------
    Both busy comparisons are **strictly greater than** (``>``).  A value
    exactly equal to a threshold does **not** trigger busy mode.  For
    example, ``swap_used_percent == 75.0`` stays ``warm``; ``75.001`` is busy.
    """
    normalized = normalize_tier(requested_tier)  # validates + maps to new vocab

    under_pressure = (
        swap_used_percent > SWAP_DEGRADED_THRESHOLD or iowait_percent > IOWAIT_DEGRADED_THRESHOLD
    )
    mode = "busy" if under_pressure else "warm"

    # minor is the floor: never shed even under pressure
    shed = under_pressure and normalized != "minor"
    reason = "pressure" if shed else "default"
    servable_tier = "minor" if under_pressure else normalized

    return {
        "mode": mode,
        "shed": shed,
        "reason": reason,
        "servable_tier": servable_tier,
        "requested_tier": normalized,
    }
