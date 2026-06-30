"""Pure routing / failover logic for the gateway — no sockets, no I/O.

Kept isolated from :mod:`lobes.gateway.server` so the gateway's
decision-making core is fully unit-testable offline. ``server`` is the only
module that touches ``http.client`` / sockets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """One upstream vLLM server in the fleet."""

    name: str  # logical role: "primary" / "fallback"
    base_url: str  # e.g. "http://vllm-primary:8000"
    served_name: str  # the OpenAI model id this backend serves
    task: str = "generate"  # task family: "generate" | "embed" | "score"


@dataclass(frozen=True)
class RoutingTable:
    """How the gateway maps a requested model to a backend (frozen → thread-safe)."""

    backends: tuple[Backend, ...]
    default_model: str  # served_name used for a missing/unknown request model
    aliases: dict[str, str]  # alias -> served_name


def is_audio_path(path: str) -> bool:
    """True for the OpenAI audio endpoints (``/v1/audio/...``).

    These are *path*-routed to the single audio backend, not *model*-routed like
    chat/completions — the bodies are multipart or plain TTS JSON, never a model
    the routing table knows about.
    """
    return path.split("?", 1)[0].startswith("/v1/audio/")


def tier_aliases(
    backends: Iterable[Backend],
    tier_role: Mapping[str, str],
) -> dict[str, str]:
    """Map each capability tier alias to a wired generate backend's served name.

    ``tier_role`` is :data:`lobes.catalog.TIER_ROLE` — a map of tier alias →
    backend role. The primary vocabulary is ``main``→``primary`` /
    ``minor``→``minor`` / ``multimodal``→``multimodal``; back-compat aliases
    ``cheap``→``minor`` / ``normal``→``multimodal`` / ``hard``→``primary``
    resolve identically to their primary-vocabulary counterparts.

    A backend's role is its :attr:`Backend.name` (``"primary"`` / ``"minor"``
    / ``"multimodal"`` / …), so a tier resolves to the served name of the
    *generate* backend whose ``name`` equals the tier's role.

    Fallback contract: when a tier's own backend is not wired, the alias falls
    back **upward** to the nearest available higher-capability tier — ultimately
    the always-present ``primary`` (so ``multimodal``/``normal``→primary when
    the multimodal gear is absent; ``minor``/``cheap``→multimodal, else
    primary, when the minor gear is absent). ``main``/``hard`` therefore always
    resolve to primary. Pooling backends (embed/score) are ignored: tier
    aliases are a *generate-only* layer on top of the task-family routing.

    Capability order is derived by sorting the unique role values from
    ``tier_role`` by their **last occurrence position** in the values sequence.
    The back-compat aliases (``cheap`` / ``normal`` / ``hard``) appear last in
    ``TIER_ROLE`` in ascending capability order, so last-position sort yields
    the correct ascending sequence ``[minor, multimodal, primary]`` regardless
    of where the primary-vocabulary keys appear.
    """
    served_by_role = {b.name: b.served_name for b in backends if b.task == "generate"}
    # Determine ascending capability order for unique roles by sorting on their
    # last occurrence in tier_role.values(). The back-compat aliases
    # (cheap/normal/hard) appear last in ascending order, anchoring the sort.
    last_pos: dict[str, int] = {}
    for i, role in enumerate(tier_role.values()):
        last_pos[role] = i
    roles_asc = sorted(last_pos, key=last_pos.__getitem__)
    # For each unique role in ascending capability, walk upward to find the
    # nearest wired backend. Primary is always present, so every tier resolves.
    role_served: dict[str, str] = {}
    for i, role in enumerate(roles_asc):
        for higher_role in roles_asc[i:]:
            served = served_by_role.get(higher_role)
            if served is not None:
                role_served[role] = served
                break
    # Map every tier alias (primary vocabulary and back-compat) to its resolved
    # served name.
    out: dict[str, str] = {}
    for tier, role in tier_role.items():
        if role in role_served:
            out[tier] = role_served[role]
    return out


def resolve_model(table: RoutingTable, requested: str | None) -> str:
    """Map a requested model name to a served model name.

    An alias resolves to its target; a name some backend already serves resolves
    to itself; anything else (``None`` or unknown) resolves to ``default_model``.
    """
    if requested:
        if requested in table.aliases:
            return table.aliases[requested]
        for backend in table.backends:
            if backend.served_name == requested:
                return requested
    return table.default_model


def _backend_for(table: RoutingTable, served_name: str) -> Backend | None:
    for backend in table.backends:
        if backend.served_name == served_name:
            return backend
    return None


def order_backends(table: RoutingTable, served_name: str) -> list[Backend]:
    """Attempt order for ``served_name``: its owner first, then same-task failovers.

    The owner is tried first; failover candidates are restricted to backends
    with the same ``task`` as the owner. This prevents an embed request from
    falling over to a generate backend (which would return a confusing 400 for
    ``/v1/embeddings``). An unmatched ``served_name`` falls back to the default
    model's owner (a generate backend) with same-task (generate) failovers.
    """
    owner = _backend_for(table, served_name) or _backend_for(table, table.default_model)
    ordered: list[Backend] = []
    # Invariant: a built table always has a primary backend and default_model
    # resolves to it, so owner is non-None in practice. We degrade gracefully (an
    # empty list → handle_post returns a 502) rather than assert, so a malformed
    # table can never crash the long-lived gateway process.
    if owner is not None:
        ordered.append(owner)
        ordered.extend(b for b in table.backends if b is not owner and b.task == owner.task)
    return ordered


def list_models_payload(table: RoutingTable) -> dict:
    """OpenAI ``/v1/models`` shape listing every backend's served model."""
    return {
        "object": "list",
        "data": [
            {"id": backend.served_name, "object": "model", "owned_by": "lobes"}
            for backend in table.backends
        ],
    }


def supported_models_payload(table: RoutingTable, catalog) -> dict:
    """The full supported-model catalog annotated with current fleet state.

    A lobes-specific (non-OpenAI) shape — ``object`` is
    ``"lobes.supported_models"`` so a client never mistakes it for the
    standard ``/v1/models`` list. Each catalog entry (a dict; see
    :mod:`lobes.catalog`) is returned as-is plus two flags computed against
    the live routing table:

    * ``loaded`` — this model's id is the ``served_name`` of a current backend
      (so a request naming it routes to a warm engine right now);
    * ``default`` — it is the gateway's default model (where unknown/missing
      names route).

    Matching is by served name (the truth of what the gateway will accept),
    independent of the catalog's ``role_hint``. Pure: the catalog is injected so
    this is unit-testable without sockets or the package import.
    """
    loaded = {backend.served_name for backend in table.backends}
    return {
        "object": "lobes.supported_models",
        "default_model": table.default_model,
        "data": [
            {
                **entry,
                "loaded": entry["id"] in loaded,
                "default": entry["id"] == table.default_model,
            }
            for entry in catalog
        ],
    }
