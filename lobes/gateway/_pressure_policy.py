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
    servable_tier: str               — the floor when shedding, else normalized tier
    requested_tier: str              — normalize_tier(requested_tier)
    shed_signal: str                 — "swap" | "engine" | "iowait" | "" (d1)

Tier vocabulary (main / minor / multimodal / worker / muse)
-------------------------------------------------------------
Issue #69 reframed the generate-lane capability tiers to **main / minor /
multimodal**, the seventh Colleague role added **muse**, and the
thor-worker-lobe plan added **worker** (mirroring ``catalog.TIER_ROLE``;
capability order ``minor < multimodal < worker < muse < main``):

    main        → 27B primary   (full text capability, the former "hard" tier)
    minor       → 4B minor       (fast, low memory, the former "cheap" tier)
    multimodal  → 12B multimodal (text+image+audio — a *different* capability)
    worker      → 35B-A3B worker (opt-in MoE lobe — its role name IS its
                                  tier; sheds to ``minor`` under pressure
                                  exactly like main/multimodal/muse)
    muse        → 31B muse       (opt-in creative/ideation lobe — its role name
                                  IS its tier; sheds to ``minor`` under
                                  pressure exactly like main/multimodal)

Back-compat input tiers (``cheap`` / ``normal`` / ``hard``) are still accepted
and normalize to the new vocabulary on output (``cheap``→``minor``,
``normal``→``multimodal``, ``hard``→``main``).

Decision rules (#68/#69, amended by deviation ``d1`` of
docs/plans/2026-08-27-capacity-relative-pool-routing.md)
-------------------------------------------------------
Comparisons are **strictly greater than** (``>``); a value *exactly equal* to a
threshold does NOT trigger the busy band.

``mode`` is UNCHANGED — it is this box's honest report of what the host looks
like, and it is what ``/status``, ``lobes status --pressure`` and every peer
probe read:

+----------------------------------------+------+
| Condition                              | mode |
+========================================+======+
| swap > 75 %  OR  iowait > 50 %         | busy |
+----------------------------------------+------+
| otherwise                              | warm |
+----------------------------------------+------+

``shed`` is where ``d1`` moved the line.  Reporting a busy HOST and refusing to
SERVE are different claims, and only some signals support the second one:

+--------------------------------+---------------------+---------------------+
| Signal                         | unpooled role       | pooled role         |
+================================+=====================+=====================+
| swap > 75 %                    | shed (``"swap"``)   | shed (``"swap"``)   |
+--------------------------------+---------------------+---------------------+
| engine_active >= capacity      | shed (``"engine"``) | shed (``"engine"``) |
+--------------------------------+---------------------+---------------------+
| iowait > 50 % (alone)          | shed (``"iowait"``) | **served**          |
+--------------------------------+---------------------+---------------------+

Where that line sits, and why:

* **swap** is first-party evidence the box is PAGING.  A serving box holds its
  weights and KV cache in the memory being paged, so a thrashing box serves at
  a fraction of its rate and taking on more work cannot rescue it.  Verifiable
  exhaustion — it still sheds, pooled or not.
* **the engine at capacity** is the DIRECT signal: the serving lane itself is
  full.  It is the same ``active >= capacity`` fact
  :func:`lobes.gateway._selection.is_full` gates pool candidacy on, so a box
  cannot select itself and refuse itself on contradictory evidence.  An
  UNPUBLISHED capacity (``None``, non-positive, non-finite) is never read as a
  capacity of zero — an uncalibrated box is never "full".
* **iowait** is a whole-host CPU-time statistic charged for *any* process's
  block wait.  Measured live 2026-08-27: a DGX Spark reading ~60 % iowait
  across five samples while its engine reported ``running=0``, traced to one
  sleeping desktop terminal in ``user.slice`` — outside the docker/vLLM
  cgroups entirely, with an EMPTY ``io.stat``.  The *sampler* is correct
  (:mod:`lobes.runtime._pressure` is deliberately untouched by ``d1``); the
  ROUTING INFERENCE from it was wrong.  It is not evidence about serving
  capacity, so on its own it no longer refuses POOLED work.

The carve-out is **pooled-only**, deliberately.  For a single-owner role a 429
is honest backpressure — there genuinely is nowhere else for the request to go
— and every deployment with no ``*_PEER_ORIGINS`` declared therefore decides
exactly as it did before ``d1``.  For a pooled role the same 429 is a lie
whenever a replica has room, and the round trip that produced it (box A
forwards, box B refuses on its own iowait reading, the 429 relays back) cost
the caller a hop to arrive at the same refusal.

Under ``warm`` the full tier is granted as requested.  When a request IS shed,
``shed_signal`` names which of the three signals justified it, so the line is
auditable from the decision rather than re-derived from thresholds; ``minor``
(the ``hand`` floor) is never shed by any of them.

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
    # Primary vocabulary. `minor`/`cheap` name the `hand` BACKEND since the
    # hand lobe replaced Qwen3.5-4B in that slot — the tier spellings survive
    # for back-compat, the `minor` backend role does not.
    "main": "primary",
    "minor": "hand",
    "multimodal": "multimodal",
    # Back-compat aliases.
    "cheap": "hand",
    "normal": "multimodal",
    "hard": "primary",
    # Capability-ROLE names (alias the same backends as main / multimodal;
    # hand/muse/worker are their own backends). Kept in the same order as
    # catalog.TIER_ROLE (hand, senses, worker, muse, cortex) so the two dicts
    # stay identical — the mirror guard test asserts equality.
    "hand": "hand",
    "senses": "multimodal",
    "worker": "worker",
    "muse": "muse",
    # `associate` (lightning-on-orin plan, t6) — the tenth role, its own
    # backend, the highest non-cortex rung. A FULL tier: it sheds under
    # pressure exactly like cortex/senses/worker/muse. It is NOT a servable
    # floor — `hand` remains the only one (see _FLOOR_TIER below).
    "associate": "associate",
    "cortex": "primary",
}

#: Backend role → canonical new-vocabulary tier name (the inverse of the primary
#: vocabulary rows above; hand's/muse's/worker's role IS its tier name).
_ROLE_TO_TIER: dict[str, str] = {
    "primary": "main",
    "hand": "hand",
    "multimodal": "multimodal",
    "worker": "worker",
    "muse": "muse",
    "associate": "associate",
}

#: The SERVABLE FLOOR tier — the one tier never shed, whatever the pressure.
#: Named once here rather than spelled inline at each comparison so the floor
#: cannot drift between the shed test and the ``servable_tier`` it reports.
#: It moved from ``minor`` to ``hand`` with the tier repoint; a caller still
#: sending ``model=minor``/``model=cheap`` normalizes to ``hand`` and is served
#: exactly as before, so the floor's PROMISE is unchanged — only its name.
_FLOOR_TIER = "hand"

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
# Shed signals (deviation d1) — WHICH fact justified refusing to serve
# ---------------------------------------------------------------------------
#
# Named rather than spelled inline so the "genuine exhaustion" line is one
# vocabulary a reader (or a trace, once t5 surfaces it) can check against the
# table in the module docstring.

#: The box is paging — verifiable resource exhaustion. Sheds any non-floor tier.
SHED_SIGNAL_SWAP = "swap"

#: The serving lane itself is at its calibrated capacity. The direct signal.
SHED_SIGNAL_ENGINE = "engine"

#: Host iowait alone. Sheds an UNPOOLED request only; see the module docstring.
SHED_SIGNAL_IOWAIT = "iowait"

#: Nothing justified a shed.
SHED_SIGNAL_NONE = ""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _engine_saturated(
    engine_active: float | None,
    engine_capacity: float | None,
) -> bool:
    """True when the serving lane has REACHED its own published capacity.

    Mirrors :func:`lobes.gateway._selection.is_full` so the gate that keeps a
    replica out of the pool and the gate that makes a box refuse a request
    cannot disagree. Anything that is not a usable positive finite capacity —
    ``None``, zero, negative, ``nan``/``inf``, an unparseable value — means "no
    capacity published for this box" and is NEVER read as a capacity of zero:
    an uncalibrated box must not start refusing work the moment it has one
    request in flight (h3).
    """
    if engine_active is None or engine_capacity is None:
        return False
    try:
        active = float(engine_active)
        capacity = float(engine_capacity)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(active) and math.isfinite(capacity)):
        return False
    if capacity <= 0:
        return False
    return active >= capacity


def normalize_tier(tier: str) -> str:
    """Normalize a tier alias (either vocabulary) to its new-vocabulary name.

    ``main``/``hard`` → ``"main"``; ``minor``/``cheap`` → ``"minor"``;
    ``multimodal``/``normal`` → ``"multimodal"``; ``muse`` → ``"muse"``;
    ``worker`` → ``"worker"`` (muse's and worker's role names ARE their
    tiers — no back-compat alias for either).

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
    *,
    pooled: bool = False,
    engine_active: float | None = None,
    engine_capacity: float | None = None,
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
        tiers (``"main"`` / ``"minor"`` / ``"multimodal"`` / ``"muse"``) or a
        back-compat alias (``"cheap"`` / ``"normal"`` / ``"hard"``).
    pooled:
        ``True`` when this role has replicas on other boxes AND a live snapshot
        of them (deviation ``d1``).  A pooled request is not shed on host
        ``iowait`` alone — see the module docstring's table.  Defaults to
        ``False``, which is every pre-``d1`` call site and every single-box
        deployment, and decides exactly as before.
    engine_active:
        This box's current active request count for the role
        (``running + waiting``), or ``None`` when unknown.
    engine_capacity:
        This box's published capacity for the role (its calibrated max active
        requests), or ``None`` when nothing has been published.  ``None`` /
        non-positive / non-finite all mean "uncalibrated" and never saturate.

    Returns
    -------
    dict with six keys (tiers are always emitted in the new vocabulary):

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
    ``shed_signal``
        Which fact justified the shed: ``"swap"`` (paging), ``"engine"`` (the
        serving lane is at capacity), ``"iowait"`` (host iowait, unpooled roles
        only), or ``""`` when nothing was shed.  Additive in ``d1``.

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

    swap_thrash = swap_used_percent > SWAP_DEGRADED_THRESHOLD
    iowait_high = iowait_percent > IOWAIT_DEGRADED_THRESHOLD

    # `mode` is UNCHANGED by d1: it reports the HOST, and both signals are
    # honest observations of it. Only the shed band below narrowed.
    under_pressure = swap_thrash or iowait_high
    mode = "busy" if under_pressure else "warm"

    # The shed band, most-authoritative signal first (the order is also the
    # reporting priority, so a paging box says "swap" even when its engine is
    # full too — the more fundamental fact is the one worth naming).
    if swap_thrash:
        signal = SHED_SIGNAL_SWAP
    elif _engine_saturated(engine_active, engine_capacity):
        signal = SHED_SIGNAL_ENGINE
    elif iowait_high and not pooled:
        # d1: host iowait alone refuses only where a 429 is honest — a role
        # with nowhere else to go. See the module docstring.
        signal = SHED_SIGNAL_IOWAIT
    else:
        signal = SHED_SIGNAL_NONE

    exhausted = signal != SHED_SIGNAL_NONE

    # `hand` is the floor: never shed even under pressure. A `minor`/`cheap`
    # request normalizes to `hand` above, so the back-compat spellings keep the
    # floor's protection unchanged.
    shed = exhausted and normalized != _FLOOR_TIER
    reason = "pressure" if shed else "default"
    servable_tier = _FLOOR_TIER if exhausted else normalized

    return {
        "mode": mode,
        "shed": shed,
        "reason": reason,
        "servable_tier": servable_tier,
        "requested_tier": normalized,
        "shed_signal": signal if shed else SHED_SIGNAL_NONE,
    }
