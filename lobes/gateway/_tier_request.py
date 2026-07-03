"""Request-layer busy/shed backpressure for the gateway (t2, #68/#69/#85).

Two pieces, both kept out of :mod:`lobes.gateway.server` so the gateway's
decision-making core stays unit-testable offline:

* :func:`resolve_tier_request` — the **pure** decision function. It maps a
  requested tier (or a plain model id), the sampled host pressure, an override
  flag and the routing table to a result dict with keys ``busy``, ``served_name``,
  ``served_tier``, ``reason``, and ``requested_tier``.  When the pressure verdict
  is ``shed=True`` and no override is set, it returns a **busy marker**
  (``busy=True``, ``served_name=None``) instead of resolving the tier through
  routing.  This severs issue #85 (silent upward-fallback substitution).

* :class:`PressureCache` — a non-blocking provider for the request path.
  :func:`lobes.runtime._pressure.sample_pressure` sleeps ~150 ms (it takes two
  ``/proc/stat`` snapshots), so it must **never** run inline per request. The
  cache seeds a value once at construction and refreshes it on a background
  daemon thread every ``interval`` seconds; ``.current()`` only ever returns the
  cached dict — it never samples — so a request reads pressure in O(1) with no
  blocking.
"""

from __future__ import annotations

import threading
from typing import Callable

from lobes.catalog import TIER_ROLE
from lobes.gateway import _pressure_policy
from lobes.gateway._routing import RoutingTable, tier_aliases
from lobes.runtime._pressure import sample_pressure

# Every capability-tier alias (primary vocabulary ``main``/``minor``/
# ``multimodal`` plus the back-compat ``cheap``/``normal``/``hard``). Single
# source of truth is catalog.TIER_ROLE — the same map the routing layer keys
# its alias table by.
_KNOWN_TIERS: frozenset[str] = frozenset(TIER_ROLE)

_ZERO_PRESSURE: dict[str, float] = {"swap_used_percent": 0.0, "iowait_percent": 0.0}

Sampler = Callable[[], "dict[str, float]"]


def is_tier_alias(name: object) -> bool:
    """True when *name* is a capability tier (``main``/``minor``/``multimodal``
    or a back-compat ``cheap``/``normal``/``hard`` alias).

    A plain model id (or ``None``) is not a tier and must pass through the normal
    routing path untouched.
    """
    return name in _KNOWN_TIERS


def _served_name_for(table: RoutingTable, tier: str) -> str:
    """The served name a *tier* resolves to, honouring the upward-fallback.

    Prefers the live routing table (the same map :func:`resolve_model` reads, so
    an operator ``GATEWAY_ALIASES`` override on a tier still wins); falls back to
    recomputing :func:`tier_aliases` from the wired backends for a table built
    without the tier-alias layer, and finally to the default model.
    """
    served = table.aliases.get(tier)
    if served is not None:
        return served
    computed = tier_aliases(table.backends, TIER_ROLE)
    return computed.get(tier, table.default_model)


def resolve_tier_request(
    requested_tier: str,
    pressure: "dict[str, float]",
    override: bool,
    table: RoutingTable,
) -> dict:
    """Decide the served name/tier/reason for a (possibly tier) generate request.

    Parameters
    ----------
    requested_tier:
        The incoming model field. Either a capability tier (``main`` / ``minor``
        / ``multimodal``, or a back-compat ``cheap`` / ``normal`` / ``hard``
        alias) or a concrete model id.
    pressure:
        A sampled-pressure dict with ``swap_used_percent`` / ``iowait_percent``
        (missing keys default to ``0.0``). Typically :meth:`PressureCache.current`.
    override:
        When truthy, force the requested tier despite pressure
        (``reason="manual_override"``). The ``X-Lobes-Override`` header.
    table:
        The gateway routing table (read-only; never mutated here).

    Returns
    -------
    dict with five keys:

    ``busy``
        ``True`` when the request must be shed (HTTP 429).  ``False`` when the
        request is served normally.
    ``served_name``
        The served model id to forward to (after rewrite), or ``None`` when the
        request is shed (busy).  For a plain model id this is the id itself.
    ``served_tier``
        The capability tier actually served, in the **new vocabulary**
        (``main`` / ``minor`` / ``multimodal``), or ``None`` when the request was
        a plain model id (pass-through) or when the request is shed (busy).
    ``reason``
        ``"default"`` (served the requested tier, no pressure), ``"pressure"``
        (shed under pressure), or ``"manual_override"`` (override forced the
        requested tier).
    ``requested_tier``
        The normalized tier name for the input, or ``None`` for a plain model id.

    Pass-through: a non-tier model id is returned verbatim with
    ``busy=False``, ``served_tier=None``, ``requested_tier=None``,
    ``reason="default"`` (override is ignored — it only forces tier requests).
    The caller then routes it via the normal path.
    """
    if not is_tier_alias(requested_tier):
        return {
            "busy": False,
            "served_name": requested_tier,
            "served_tier": None,
            "reason": "default",
            "requested_tier": None,
        }

    normalized = _pressure_policy.normalize_tier(requested_tier)

    decision = _pressure_policy.decide(
        pressure.get("swap_used_percent", 0.0),
        pressure.get("iowait_percent", 0.0),
        requested_tier,
    )

    if override:
        # Force the requested tier despite pressure.
        return {
            "busy": False,
            "served_name": _served_name_for(table, normalized),
            "served_tier": normalized,
            "reason": "manual_override",
            "requested_tier": normalized,
        }

    if decision["shed"]:
        # Shed: return busy marker. Do NOT call _served_name_for — that would
        # trigger the upward-fallback (issue #85: minor absent → resolves up to
        # multimodal/gemma). A shed request resolves to NO served name.
        return {
            "busy": True,
            "served_name": None,
            "served_tier": None,
            "reason": "pressure",
            "requested_tier": normalized,
        }

    # Warm or minor-under-pressure: serve normally.
    served_tier = decision["servable_tier"]
    return {
        "busy": False,
        "served_name": _served_name_for(table, served_tier),
        "served_tier": served_tier,
        "reason": decision["reason"],
        "requested_tier": normalized,
    }


class PressureCache:
    """A non-blocking host-pressure provider for the gateway request path.

    :func:`sample_pressure` sleeps ~150 ms per call, so it must never run inline
    on a request. This cache samples once at construction (a one-time cost off
    the request path) and then on a background **daemon** thread every
    ``interval`` seconds. :meth:`current` only ever returns the cached dict — it
    never samples — so a request reads pressure in O(1), no blocking.

    The sampler is injectable (``sampler=...``) so tests can drive it with fixed
    values and never touch ``/proc`` or real timing. A sampler that raises is
    swallowed (the last good value, or zeros, is kept) so a transient ``/proc``
    read error can never kill the gateway.
    """

    def __init__(
        self,
        sampler: Sampler | None = None,
        interval: float = 2.0,
        *,
        start: bool = True,
    ) -> None:
        self._sampler: Sampler = sampler or sample_pressure
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Seed synchronously so the very first request never blocks on a sample.
        self._value: dict[str, float] = self._read()
        if start:
            self.start()

    def _read(self) -> dict[str, float]:
        """Sample once, degrading to the zero baseline if the sampler raises."""
        try:
            sampled = self._sampler()
            return {
                "swap_used_percent": float(sampled.get("swap_used_percent", 0.0)),
                "iowait_percent": float(sampled.get("iowait_percent", 0.0)),
            }
        except Exception:  # nosec B110 — pressure is best-effort; never crash the gateway
            return dict(_ZERO_PRESSURE)

    def current(self) -> dict[str, float]:
        """Return a copy of the cached pressure dict. Never samples, never blocks."""
        with self._lock:
            return dict(self._value)

    def start(self) -> None:
        """Start the background refresh thread (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="lobes-pressure-cache", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # Event.wait(interval) returns True only when stop() is set, so this both
        # paces the refresh and exits promptly on shutdown.
        while not self._stop.wait(self._interval):
            value = self._read()
            with self._lock:
                self._value = value

    def stop(self) -> None:
        """Signal the background thread to exit (best-effort; daemon anyway)."""
        self._stop.set()
