"""Pure routing / failover logic for the gateway — no sockets, no I/O.

Kept isolated from :mod:`lobes.gateway.server` so the gateway's
decision-making core is fully unit-testable offline. ``server`` is the only
module that touches ``http.client`` / sockets.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from lobes.catalog import TIER_ROLE


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
    # Backend NAMES this machine's per-machine profile declared it CANNOT serve
    # AT ALL (issue #92's "advertised implies reachable" extended to the
    # HARDWARE dimension — plan "per-machine profiles", task t6). Populated by
    # :func:`lobes.gateway._config.build_config` from ``<PREFIX>_FEASIBLE=false``
    # env vars (:data:`lobes.gateway._config.FEASIBLE_ENV`) — the SAME
    # per-backend-name env convention the served-context overlay already uses.
    # Independent of wiring: a backend can be BOTH present in ``backends`` (the
    # primary is unconditionally wired) AND infeasible — this field is what
    # lets :func:`infeasible_owner` / :func:`list_models_payload` /
    # :mod:`lobes.roles` reject/hide it anyway. Defaults to empty so every
    # existing caller/table construction (this module's own tests included) is
    # completely unaffected.
    infeasible: frozenset[str] = frozenset()
    # Backend NAME -> the OPERATOR-DECLARED origin of the peer box that hosts
    # that backend's role (mesh-brain t3, issue #112's "direct + honest
    # referral" decision). Populated by :func:`lobes.gateway._config.
    # build_config` from ``<PREFIX>_PEER_ORIGIN`` env vars
    # (:data:`lobes.gateway._config.PEER_ORIGIN_ENV`) — the SAME
    # per-backend-name env convention ``infeasible`` above already uses.
    # Consulted ONLY to ANNOTATE honesty surfaces (/capabilities and the 404
    # ``role_infeasible`` body) for a role in ``infeasible``; it is NEVER
    # dialed — the gateway does no data-plane proxying to peers (proxy-lobes
    # is deferred, issue #115). Per the #92 lesson an origin here is always
    # operator-declared, never derived from hostnames/interfaces. Defaults to
    # empty so a deployment with no peer config is byte-identical to the
    # pre-referral contract on every surface.
    peer_origins: Mapping[str, str] = field(default_factory=dict)
    # Backend NAMES whose dropped role this box has opted in to PROXY to its
    # declared peer (proxy-lobes t1, issues #115/#127 — the follow-up
    # ``peer_origins`` above explicitly deferred). Populated by
    # :func:`lobes.gateway._config.build_config` from ``<PREFIX>_PEER_PROXY``
    # truthy env vars (:data:`lobes.gateway._config.PEER_PROXY_ENV`) — the
    # SAME per-backend-name env convention ``infeasible``/``peer_origins``
    # already use — and ONLY for a name that ALSO has a declared peer origin
    # AND is in ``infeasible`` (a knob without an origin has nothing to dial;
    # a knob on a locally-feasible role is ignored — the local engine serves
    # it). Consumed by the proxy data plane (t6): a request resolving to a
    # name here is FORWARDED to its declared peer instead of taking the
    # referral 404 (see ``lobes.gateway.server._proxy_to_peer``), and its
    # served id is advertised on /v1/models while the live peer probe verifies
    # it. Defaults to empty so every existing table construction is completely
    # unaffected, and so an origin declared WITHOUT the knob stays
    # annotation-only referral — the issue #112 contract is preserved
    # byte-for-byte.
    peer_proxied: frozenset[str] = frozenset()
    # Backend NAME -> the OUTBOUND API key this box presents when dialing
    # that role's declared peer (proxy-lobes t1, issues #115/#127 — the
    # pairwise-auth half). Populated by :func:`lobes.gateway._config.
    # build_config` from ``<PREFIX>_PEER_API_KEY`` env vars
    # (:data:`lobes.gateway._config.PEER_API_KEY_ENV`), verbatim (stripped),
    # and ONLY for names with a declared peer origin (a key without an
    # origin is inert — there is no peer to authenticate to). Attached by the
    # proxy data plane (t6) as the OUTBOUND ``Authorization: Bearer`` on a
    # forwarded request — replacing, never accompanying, the caller's own
    # credential — and by the peer-readiness probe (t4).
    # ``repr=False`` because the values are SECRETS: they must NEVER appear
    # in ``repr``/``str`` of the table (logs, tracebacks, --json debug
    # output). Defaults to empty so every existing construction is
    # unaffected.
    peer_api_keys: Mapping[str, str] = field(default_factory=dict, repr=False)


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

    This never distinguishes *unspecified* from *unknown* — it always returns a
    concrete served name so :func:`order_backends` always has an owner to try.
    That UNSPECIFIED-vs-UNKNOWN policy lives one level up, in
    :func:`is_unknown_model` + :func:`lobes.gateway.server.handle_post`: an
    unknown non-empty id is rejected (404) *before* ``handle_post`` ever calls
    this, so ``resolve_model``'s unknown→default fall-back is now a pure-routing
    safety net (e.g. for an internal caller passing a stale name), not the path a
    client's unknown model id takes. Kept unchanged so its many callers (the tier
    tests, ``order_backends``) are unaffected.
    """
    if requested:
        if requested in table.aliases:
            return table.aliases[requested]
        for backend in table.backends:
            if backend.served_name == requested:
                return requested
    return table.default_model


def is_unknown_model(table: RoutingTable, requested: str | None) -> bool:
    """True when ``requested`` is a NON-EMPTY id that was NEVER advertised —
    neither an alias nor any wired backend's ``served_name`` — so it must not be
    silently served under the default backend's weights (honesty h23, issue #91).

    Distinguishes UNKNOWN from UNSPECIFIED — the distinction :func:`resolve_model`
    deliberately does not make:

    * **Unspecified** — ``requested`` is ``None`` or ``""`` (a missing/blank
      ``model`` field). This is NOT unknown: it intentionally routes to
      ``default_model`` (see :func:`resolve_model`) and is served. Returns
      ``False``.
    * **Known** — ``requested`` is an alias or a wired backend's served name.
      Returns ``False``.
    * **Unknown** — a non-empty id that is neither. Returns ``True``; the caller
      (:func:`lobes.gateway.server.handle_post`) turns this into a 404
      ``model_not_found`` rather than routing it to the default owner.

    **Decided against the ROUTING TABLE, never the readiness-filtered
    ``/v1/models`` list.** A backend that is wired but dead/warming is dropped
    from ``/v1/models`` (see :func:`list_models_payload`) yet its ``served_name``
    is still in ``table.backends`` — so it is *known*, and a request naming it
    routes to its owner and yields a retryable **503** (owner down), NOT a 404.
    Deciding unknown-ness against the readiness list instead would 404 a merely
    warming/transiently-dead backend and reintroduce issue #91. Unknown-ness is a
    question about *wiring* ("is this id in the table at all"), not *liveness*.

    ``default_model`` is always treated as KNOWN, even in the degenerate case of a
    malformed table where no backend actually serves it: naming the deployment's
    declared default identity explicitly is equivalent to leaving ``model``
    unspecified — both route to ``default_model`` — so that path stays a terminal
    **502** ``upstream_unavailable`` (the malformed-table signal), not a 404. In a
    well-formed table ``default_model`` is some wired backend's served name, so
    this clause is redundant there; it only matters for the pathological table.
    """
    if not requested:
        return False  # unspecified (missing/blank) → routes to default, not unknown
    if requested == table.default_model:
        return False  # the declared default identity is known (see docstring)
    if requested in table.aliases:
        return False
    return not any(backend.served_name == requested for backend in table.backends)


def _backend_for(table: RoutingTable, served_name: str) -> Backend | None:
    for backend in table.backends:
        if backend.served_name == served_name:
            return backend
    return None


def infeasible_owner(table: RoutingTable, requested: str | None) -> str | None:
    """The infeasible backend NAME ``requested`` resolves to, else ``None``.

    Resolves ``requested`` EXACTLY the way :func:`resolve_model` would — a
    capability-tier or role-identity alias (``cortex``/``main``/``hard``/
    ``senses``/``multimodal``/…), a custom operator alias, a concrete served
    model id, or an unspecified/unknown id (both of which fall back to
    ``table.default_model`` — see :func:`resolve_model`) — then checks whether
    the OWNING backend's name is in :attr:`RoutingTable.infeasible`.

    This deliberately reuses ``resolve_model`` rather than re-deriving tier
    semantics: ``table.aliases`` already carries every tier/role alias
    (``tier_aliases`` computed it in :func:`~lobes.gateway._config.build_config`,
    independent of feasibility), so a role-identity request like ``cortex``
    resolves to the SAME served name a pressure-aware
    :func:`~lobes.gateway._tier_request.resolve_tier_request` would use for a
    WIRED backend (it only diverges via the upward-fallback substitution when a
    tier's own backend is unwired — a case this function does not need to
    special-case, because an infeasible-but-unwired backend never owns
    anything a request could resolve to in the first place).

    Callers decide WHEN to consult this relative to their own precedence
    rules (e.g. :func:`~lobes.gateway.server.handle_post` runs it before
    pressure-shedding for a tier request — feasibility is a hardware fact,
    not a load condition — but after the ``is_unknown_model`` 404 for a plain
    id, so a genuinely never-advertised id still gets ``model_not_found``,
    not ``role_infeasible``).

    **Two lookups, because a DROPPED role's backend may be UNWIRED**
    (brain-shapes t5, issue #113). A mesh-brain deployment shape that drops a
    role (spark-lobe drops ``senses``, thor-lobe drops ``cortex``) renders the
    drop as ``<PREFIX>_FEASIBLE=false`` AND, realistically, simply does not run
    that role's container — so no ``*_BASE_URL`` is set and the backend is
    absent from ``table.backends``. In that shape :func:`resolve_model` is a
    TRAP: the dropped role's capability-tier aliases (``senses`` / ``multimodal``
    / ``normal`` → the ``multimodal`` role) upward-fall-back in
    :func:`tier_aliases` to the always-present primary's served name, so a
    ``resolve_model``-only check would see the OWNER as the (feasible) primary
    and wave the request through — silently answering a dropped ``senses``
    request with ``cortex``, the exact "never silently rerouted" violation this
    gate exists to prevent. So we FIRST map the *literal* requested alias to its
    role's backend NAME via :data:`lobes.catalog.TIER_ROLE` — the same
    alias→backend map the routing layer keys its alias table by — and reject it
    if that backend is infeasible, WITHOUT resolving through the wiring-dependent
    upward-fallback. Only a request that is not a tier/role alias (a concrete
    served id, or an unspecified/unknown id) falls through to the
    ``resolve_model`` owner check below, which still catches a wired-but-
    infeasible backend (thor-lobe's cortex is unconditionally wired) and the
    unspecified→``default_model`` case.
    """
    if not table.infeasible:
        return None
    # A capability-tier / role-identity alias (main/hard/cortex, multimodal/
    # normal/senses, minor/cheap) resolves to a backend NAME independent of
    # whether that backend is wired — so a dropped-but-unwired role is caught
    # here before the upward-fallback in resolve_model could mask it.
    role = TIER_ROLE.get(requested) if isinstance(requested, str) else None
    if role is not None and role in table.infeasible:
        return role
    owner = _backend_for(table, resolve_model(table, requested))
    return owner.name if owner is not None and owner.name in table.infeasible else None


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
    table: RoutingTable,
    ready: Mapping[str, "bool | None"] | None = None,
    peer_served: Mapping[str, str] | None = None,
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

    ``table.infeasible`` (task t6, the HARDWARE dimension of the same
    invariant) is applied UNCONDITIONALLY, regardless of whether ``ready`` was
    supplied — a backend this machine's profile declared infeasible is never
    listed, even when a live readiness probe reports it healthy (``ready=True``
    is not evidence of hardware capability, only of "the process answered").

    **Proxied roles** (proxy-lobes t6, issues #115/#127): ``peer_served`` maps
    a proxied backend name to the served id this box forwards for it — a
    dropped role is realistically UNWIRED (no local :class:`Backend` exists),
    so the id cannot come from ``table.backends``; the caller supplies it from
    the same :class:`~lobes.gateway._readiness.PeerSpec` the peer probe
    verifies (see ``lobes.gateway.server.peer_specs_from_table`` for the
    resolution order). A proxied id is listed IFF ALL of: its name is in
    ``table.peer_proxied`` (the routing table's opt-in is the only gate —
    stray ``peer_served`` entries are ignored), a live ``ready`` snapshot was
    supplied, and that snapshot's verdict for the name ``is True`` — which for
    a proxied name is the PEER probe's verdict ("the peer answered 200 AND its
    own ``/v1/models`` lists exactly this id"). Peer down, unprobed (``None``),
    missing, or ``ready`` omitted entirely ⇒ the id drops, exactly like a dead
    local backend — #92 extended across the box boundary, never a hardcoded
    reachability claim (h2). ``peer_served=None`` (the default) changes
    nothing for every existing caller.
    """
    backends = table.backends
    if ready is not None:
        backends = tuple(b for b in backends if ready.get(b.name) is True)
    if table.infeasible:
        backends = tuple(b for b in backends if b.name not in table.infeasible)
    data = [
        {"id": backend.served_name, "object": "model", "owned_by": "lobes"} for backend in backends
    ]
    if peer_served and ready is not None:
        listed = {entry["id"] for entry in data}
        for name in sorted(table.peer_proxied):
            served = peer_served.get(name)
            if served and served not in listed and ready.get(name) is True:
                data.append({"id": served, "object": "model", "owned_by": "lobes"})
                listed.add(served)
    return {"object": "list", "data": data}


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
