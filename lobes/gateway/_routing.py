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
    """Resolve ``served_name`` to its single owning backend — never a failover chain.

    Returns a list of length 0 or 1. **No cross-backend failover, ever** (issue
    #91, "advertised implies reachable"): a request that resolves to one model
    is attempted at that model's owner only, never retried against a different
    backend serving a different model.

    This used to walk every other backend that shared the owner's ``task`` as a
    failover chain — e.g. cortex (primary) falling over to the multimodal
    (Gemma) backend when the vLLM engine died. That is unsound: the retry still
    carries the *original* body, which still names the original model (cortex's
    Qwen id). A backend that does not serve that model has exactly one honest
    answer — an OpenAI-shaped 404 ``model does not exist`` — and that 404 is
    **indistinguishable to the caller** from "this model id was never valid".
    ``handle_post``'s own rule ("2xx or 4xx → commit to this backend; 4xx is a
    client error, no failover") then relays that 404 as terminal, silently
    killing the request instead of surfacing the real problem (the owner's
    engine crashed). Worse, if the other backend's model *did* happen to exist,
    the caller would get a real answer from the wrong model — a `final_authority`
    role-contract violation (issue #81): a caller who asked for cortex must never
    silently receive a Gemma answer.
    So: one served name resolves to exactly one backend, tried once. If that
    backend is unreachable or errors, the caller gets an honest failure instead
    of an answer from a model they did not ask for. (The *static* tier-alias
    upward fallback in :func:`tier_aliases` is unrelated and unaffected — that
    resolves an unwired capability tier to a different served name at
    table-build time, before ``order_backends`` ever runs; it is config-time
    resolution, not a runtime retry against a mismatched body.)

    An unmatched ``served_name`` still falls back to the ``default_model``'s
    owner (preserves the existing "unknown model routes to default" behaviour)
    — that remains a single backend, not a chain.
    """
    owner = _backend_for(table, served_name) or _backend_for(table, table.default_model)
    # Invariant: a built table always has a primary backend and default_model
    # resolves to it, so owner is non-None in practice. We degrade gracefully (an
    # empty list → handle_post returns a 502) rather than assert, so a malformed
    # table can never crash the long-lived gateway process.
    return [owner] if owner is not None else []


def list_models_payload(
    table: RoutingTable, ready: Mapping[str, "bool | None"] | None = None
) -> dict:
    """OpenAI ``/v1/models`` shape listing the fleet's served models.

    When ``ready`` is supplied — the gateway's live readiness snapshot, keyed by
    backend **name** (exactly what
    :meth:`lobes.gateway._readiness.ReadinessCache.current` returns) — only
    backends whose signal ``is True`` are listed. This is the core of "advertised
    implies reachable" (issue #92): a backend that is wired but dead/missing
    (``None``) or reached-but-unhealthy (``False``) must NOT be advertised, so a
    client can trust that a model id appearing here will reach a live engine.
    ``None`` (*unknown*) and ``False`` are BOTH treated as not-ready — only an
    affirmative ``True`` advertises; treating ``None`` as "list it anyway" is the
    exact defect #92 fixes (a wired-but-dead backend probes ``None``, not
    ``False``). ``ready=None`` (the default) lists every wired backend unchanged —
    the offline/CLI path and any caller without a live signal.
    """
    backends = table.backends
    if ready is not None:
        backends = tuple(b for b in backends if ready.get(b.name) is True)
    return {
        "object": "list",
        "data": [
            {"id": backend.served_name, "object": "model", "owned_by": "lobes"}
            for backend in backends
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
