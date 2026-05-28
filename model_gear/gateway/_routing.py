"""Pure routing / failover logic for the gateway — no sockets, no I/O.

Kept isolated from :mod:`model_gear.gateway.server` so the gateway's
decision-making core is fully unit-testable offline. ``server`` is the only
module that touches ``http.client`` / sockets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """One upstream vLLM server in the fleet."""

    name: str  # logical role: "primary" / "fallback"
    base_url: str  # e.g. "http://vllm-primary:8000"
    served_name: str  # the OpenAI model id this backend serves


@dataclass(frozen=True)
class RoutingTable:
    """How the gateway maps a requested model to a backend (frozen → thread-safe)."""

    backends: tuple[Backend, ...]
    default_model: str  # served_name used for a missing/unknown request model
    aliases: dict[str, str]  # alias -> served_name


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
    """Attempt order for ``served_name``: its owner first, then the rest.

    The owner is tried first; the remaining backends are failover targets. An
    unmatched ``served_name`` falls back to the default model's owner first.
    """
    owner = _backend_for(table, served_name) or _backend_for(table, table.default_model)
    ordered: list[Backend] = []
    if owner is not None:
        ordered.append(owner)
    ordered.extend(b for b in table.backends if b is not owner)
    return ordered


def list_models_payload(table: RoutingTable) -> dict:
    """OpenAI ``/v1/models`` shape listing every backend's served model."""
    return {
        "object": "list",
        "data": [
            {"id": backend.served_name, "object": "model", "owned_by": "model-gear"}
            for backend in table.backends
        ],
    }
