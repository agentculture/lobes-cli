"""The per-chip strategy registry — an explicit, import-driven catalog.

No plugin/entry-point machinery on purpose: the built-in strategies are
*registered by importing them* (see :mod:`lobes.machines`), and third parties (or
tests) extend the fleet with a single :func:`register` call. Registration order
is detection precedence — the GB10 ``spark`` is registered before the discrete
``blackwell`` so a Grace-Blackwell part never trips the discrete profile.

Adding a chip is therefore one new module plus one :func:`register` line, with no
edits to :mod:`lobes.profiles` or to this module: :func:`detect`, the name lookup
and the profile derivation all read the live registry.
"""

from __future__ import annotations

from ._strategy import CardStrategy

# Insertion-ordered; order is detection precedence (spark before blackwell).
_REGISTRY: dict[str, CardStrategy] = {}


def register(strategy: CardStrategy, *, replace: bool = False) -> CardStrategy:
    """Add a strategy to the registry (idempotent only with ``replace=True``).

    Returns the strategy so a module can ``STRATEGY = register(CardStrategy(...))``
    in one line. Raises :class:`ValueError` on a duplicate name unless ``replace``.
    """
    if strategy.name in _REGISTRY and not replace:
        raise ValueError(f"machine strategy {strategy.name!r} already registered")
    _REGISTRY[strategy.name] = strategy
    return strategy


def unregister(name: str) -> None:
    """Remove a strategy by name; no-op if absent (test/teardown convenience)."""
    _REGISTRY.pop(name, None)


def strategies() -> tuple[CardStrategy, ...]:
    """All registered strategies, in registration (detection-precedence) order."""
    return tuple(_REGISTRY.values())


def names() -> tuple[str, ...]:
    """The canonical names of all registered strategies, in order."""
    return tuple(_REGISTRY.keys())


def get(name: str) -> CardStrategy | None:
    """The strategy for ``name``, or ``None`` if no such chip is registered.

    Honest by design: it does *not* fall back to ``generic``. Callers that need
    the legacy silent fallback do it themselves (see
    :func:`lobes.profiles.detect_machine`).
    """
    return _REGISTRY.get((name or "").strip().lower())


def detect(
    gpu_name: str | None = None,
    hostname: str | None = None,
    compute_capability: str | None = None,
    total_memory_gb: float | None = None,
) -> CardStrategy | None:
    """First strategy whose signature matches the supplied facts, else ``None``.

    ``compute_capability`` / ``total_memory_gb`` are optional refinements: when
    given, a strategy that *declares* the corresponding trait must agree with the
    probed value or it is skipped. Omitting them reproduces the legacy
    name-substring behaviour exactly, so a caller that cannot probe still
    resolves a card rather than falling to UNKNOWN.

    Returns ``None`` — not ``generic`` — when nothing matches: the honest
    UNKNOWN resolution the future detector wants. Legacy callers keep their silent
    ``generic`` fallback by wrapping this.
    """
    for strategy in _REGISTRY.values():
        if strategy.signature.matches(gpu_name, hostname, compute_capability, total_memory_gb):
            return strategy
    return None
