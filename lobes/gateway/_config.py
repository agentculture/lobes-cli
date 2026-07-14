"""Build the gateway's :class:`RoutingTable` + :class:`ServerConfig` from env vars.

Reads a mapping (``os.environ`` by default) and constructs frozen config objects.
No sockets — pass a plain ``dict`` to unit-test it offline. The env keys mirror
the ``gateway`` service's ``environment:`` block in the fleet compose template.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from lobes.catalog import TIER_ROLE
from lobes.gateway._routing import Backend, RoutingTable, tier_aliases

_DEFAULT_PRIMARY = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_DEFAULT_FALLBACK = "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4"
_DEFAULT_EMBED = "Qwen/Qwen3-Embedding-0.6B"
_DEFAULT_RERANK = "Qwen/Qwen3-Reranker-0.6B"
_DEFAULT_MINOR = "Qwen/Qwen3.5-4B"
# "support both" (docs/vllm-nightly-migration.md §7, 2026-07-02): the NVFP4 base +
# native-MTP gear is the new default "multimodal" gear (28.6 tok/s, 57.9% draft
# acceptance — the fastest measured Gemma config). The coder fine-tune (kept, opt-in
# below as _DEFAULT_MULTIMODAL_CODER) is coding-strong but its MTP acceptance is only
# 30.8%, not worth wiring/defaulting.
_DEFAULT_MULTIMODAL = "coolthor/gemma-4-12B-it-NVFP4A16"
# Opt-in coder gear (demoted from default; catalog role_hint="candidate"). Reachable
# via the "multimodal-coder" alias when its own backend is wired — see
# _optional_backend(name="multimodal-coder", ...) below.
_DEFAULT_MULTIMODAL_CODER = "sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4"
_DEFAULT_MIDDLE = "nvidia/Qwen3-14B-NVFP4"

# Per-backend "this machine's per-machine profile declares it CANNOT be served
# AT ALL" signal (issue #92's "advertised implies reachable" extended to the
# HARDWARE dimension — plan "per-machine profiles", task t6). The ONE channel
# a rendered Profile's ``RoleProfile.feasible=False`` composes down to: an
# operator (or ``lobes init``, rendering a per-machine Profile) sets
# ``<PREFIX>_FEASIBLE=false`` in the deployment's ``.env``. Named after the
# SAME backend-name prefixes the served-context overlay already uses
# (``PRIMARY_MAX_MODEL_LEN`` etc., see ``lobes.roles.ROLE_MAX_MODEL_LEN_ENV``)
# so there is exactly one "<PREFIX>_<KNOB>" env convention to learn. Scoped to
# the four backends the per-machine Profile schema covers
# (:data:`lobes.profiles.schema.ROLES`) — the opt-in fallback/minor/middle/
# multimodal-coder backends are out of that schema's scope and have no entry
# here (never infeasible via this channel).
FEASIBLE_ENV: dict[str, str] = {
    "primary": "PRIMARY_FEASIBLE",
    "multimodal": "MULTIMODAL_FEASIBLE",
    "embed": "EMBED_FEASIBLE",
    "rerank": "RERANK_FEASIBLE",
}

_FALSY_FEASIBLE = frozenset({"false", "0", "no"})

# Per-backend "the peer box at THIS origin hosts the role I dropped" channel
# (mesh-brain t3, issue #112's confirmed cross-box decision: direct + honest
# referral). DESIGN DECISION, made within the #92 lesson: the referral origin
# is a full, OPERATOR-DECLARED origin (e.g. ``http://spark.local:8001``) set
# per peer in the deployment's ``.env`` — NEVER fabricated or inferred from
# hostnames/interfaces (deriving a URL from the local box's own view of the
# network is exactly what #92 forbade). It uses the SAME
# ``<PREFIX>_<KNOB>`` backend-name prefixes as :data:`FEASIBLE_ENV` /
# ``ROLE_MAX_MODEL_LEN_ENV`` so there is still exactly one env convention to
# learn, and it is scoped to the same four Profile-schema backends —
# referral, like feasibility, is a core-role fact (the audio overlay is
# outside the Profile schema and carries no referral channel).
#
# A declared origin is CONTROL-PLANE metadata only: it annotates
# ``/capabilities`` and the 404 ``role_infeasible`` body for a role this box
# does not host. The gateway NEVER dials it — no data-plane proxying exists
# (proxy-lobes is deferred to issue #115). Unset everywhere (the default) ⇒
# every response is byte-identical to the pre-referral contract.
PEER_ORIGIN_ENV: dict[str, str] = {
    "primary": "PRIMARY_PEER_ORIGIN",
    "multimodal": "MULTIMODAL_PEER_ORIGIN",
    "embed": "EMBED_PEER_ORIGIN",
    "rerank": "RERANK_PEER_ORIGIN",
}


def _peer_origins(env: Mapping[str, str]) -> dict[str, str]:
    """The declared peer origins, keyed by backend name; blank/unset omitted.

    Values are taken VERBATIM from the operator's env (trailing slash
    trimmed, matching every other URL knob here) — nothing is derived,
    validated against DNS, or probed. An empty mapping (no ``*_PEER_ORIGIN``
    set anywhere) is the default and leaves every response surface
    byte-identical to the pre-referral contract.
    """
    out: dict[str, str] = {}
    for name, key in PEER_ORIGIN_ENV.items():
        origin = (env.get(key) or "").strip().rstrip("/")
        if origin:
            out[name] = origin
    return out


def _is_feasible(env: Mapping[str, str], backend_name: str) -> bool:
    """True unless ``backend_name``'s ``<PREFIX>_FEASIBLE`` env var (see
    :data:`FEASIBLE_ENV`) holds an explicit falsy token.

    Absent/blank/anything-but-a-recognised-falsy-token → feasible — an
    untouched deployment (no FEASIBLE var set anywhere) is completely
    unaffected, matching every other knob's ``${VAR:-default}`` convention.
    A backend with no entry in :data:`FEASIBLE_ENV` is always feasible here
    (out of the per-machine Profile schema's four-role scope).
    """
    key = FEASIBLE_ENV.get(backend_name)
    if key is None:
        return True
    return (env.get(key) or "").strip().lower() not in _FALSY_FEASIBLE


@dataclass(frozen=True)
class ServerConfig:
    """Where the gateway listens and how patient it is with backends."""

    host: str
    port: int
    connect_timeout: float  # short: a refused/down backend fails over fast
    read_timeout: float  # long: a reasoning model's first token is slow
    # The audio/realtime backend that serves /v1/audio/* (+ /v1/realtime in PR2).
    # None on a text-only fleet → those paths 404. Set by the --audio overlay.
    audio_url: str | None = None
    # Optional client-reachable origin the gateway advertises for every role in
    # GET /capabilities (issue #87). None → the route derives it from the
    # incoming request Host header (correct for a normal published host port);
    # set GATEWAY_PUBLIC_URL to override for a tunnel / Host-rewriting proxy.
    public_url: str | None = None


def _parse_aliases(raw: str | None) -> dict[str, str]:
    """Parse ``alias=served,other=served`` into a dict; skip blank/malformed pairs."""
    out: dict[str, str] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        alias, _, target = pair.partition("=")
        alias, target = alias.strip(), target.strip()
        if alias and target:
            out[alias] = target
    return out


def _expand_tier_alias_synonyms(operator: dict[str, str]) -> dict[str, str]:
    """Mirror a tier-keyed operator override onto its vocabulary synonyms.

    Tier requests are normalized to the new vocabulary (``hard``→``main``,
    ``cheap``→``minor``, ``normal``→``multimodal``) *before* the alias table is
    consulted (see :func:`lobes.gateway._tier_request.resolve_tier_request`), so
    an operator ``GATEWAY_ALIASES`` override keyed only by a legacy alias would
    otherwise be silently bypassed. For each tier-keyed override, also set every
    other alias sharing its capability role (the new-vocab name for a legacy key
    and vice versa) so the override applies regardless of which vocabulary the
    operator used. An explicit key for a synonym always wins (never clobbered);
    non-tier custom aliases (e.g. ``fast=...``) pass through untouched.
    """
    out = dict(operator)
    for alias, target in operator.items():
        role = TIER_ROLE.get(alias)
        if role is None:
            continue  # a non-tier custom alias — leave it alone
        for synonym, synonym_role in TIER_ROLE.items():
            if synonym_role == role and synonym not in operator:
                out[synonym] = target
    return out


def _as_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key) or default)
    except (TypeError, ValueError):
        return float(default)


def _as_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key) or default)
    except (TypeError, ValueError):
        return int(default)


def _optional_backend(
    env: Mapping[str, str],
    *,
    name: str,
    url_key: str,
    name_key: str,
    default_url: str,
    default_name: str,
    task: str = "generate",
) -> Backend | None:
    """A fleet backend wired only when its ``url_key`` env var is non-empty.

    ``name_key`` alone is NOT enough — a served name with no URL describes a
    model, not a reachable backend, and wiring one anyway invents a "phantom"
    backend whose ``base_url`` falls back to a hardcoded ``default_url``
    naming a compose service that need not exist (advertised on
    ``GET /v1/models`` yet unreachable: every request to it fails to
    connect). This mirrors the contract the fleet already documents for
    ``MINOR_BASE_URL`` — empty ⇒ silently unwired.

    Returns ``None`` when ``url_key`` is absent/empty — so the default
    gateway serves the primary alone, and each extra backend (fallback /
    embed / rerank / …) opts in independently via its own ``*_BASE_URL``.
    """
    if not env.get(url_key):
        return None
    return Backend(
        name=name,
        base_url=(env.get(url_key) or default_url).rstrip("/"),
        served_name=env.get(name_key) or default_name,
        task=task,
    )


def build_config(env: Mapping[str, str] | None = None) -> tuple[RoutingTable, ServerConfig]:
    """Construct the routing table and server config from environment variables."""
    env = os.environ if env is None else env

    primary = Backend(
        name="primary",
        base_url=(env.get("PRIMARY_URL") or "http://vllm-primary:8000").rstrip("/"),
        served_name=env.get("PRIMARY_SERVED_NAME") or _DEFAULT_PRIMARY,
    )
    # The primary is always present; fallback / embed / rerank are each wired only
    # when their own env pair is set (so the default gateway serves the primary
    # alone, and a pooling/fallback backend opts in independently).
    optional = (
        _optional_backend(
            env,
            name="fallback",
            url_key="FALLBACK_URL",
            name_key="FALLBACK_SERVED_NAME",
            default_url="http://vllm-fallback:8000",
            default_name=_DEFAULT_FALLBACK,
        ),
        # The minor co-resident generate backend (Qwen/Qwen3.5-4B, bf16).
        # Wired only when MINOR_BASE_URL or MINOR_SERVED_NAME is present in
        # the environment — i.e. when the operator has activated the compose
        # "minor" profile and set these vars (they are absent by default so
        # the routing table is unchanged on a standard fleet startup).
        _optional_backend(
            env,
            name="minor",
            url_key="MINOR_BASE_URL",
            name_key="MINOR_SERVED_NAME",
            default_url="http://vllm-minor:8000",
            default_name=_DEFAULT_MINOR,
        ),
        # The multimodal co-resident generate backend (Gemma 4 12B unified
        # text+image+audio, the "normal"/"multimodal" tier). Wired only when
        # MULTIMODAL_BASE_URL or MULTIMODAL_SERVED_NAME is present — i.e. when
        # the operator has activated the compose "multimodal" profile and set
        # these vars (absent by default, so the routing table is unchanged on a
        # standard fleet startup). The 14B Qwen3 "middle" gear is LEGACY and is
        # no longer a tier backend; address it explicitly by model id (see the
        # middle backend wired below).
        _optional_backend(
            env,
            name="multimodal",
            url_key="MULTIMODAL_BASE_URL",
            name_key="MULTIMODAL_SERVED_NAME",
            default_url="http://vllm-multimodal:8000",
            default_name=_DEFAULT_MULTIMODAL,
        ),
        # The opt-in coder gear (Gemma 4 12B coder fine-tune, catalog
        # role_hint="candidate" since the "support both" demotion — see
        # docs/vllm-nightly-migration.md §7). Wired only when
        # MULTIMODAL_CODER_BASE_URL or MULTIMODAL_CODER_SERVED_NAME is present (the
        # compose "multimodal-coder" profile sets them). Its backend name
        # "multimodal-coder" is NOT a TIER_ROLE role, so it gets no tier alias — but
        # a dedicated "multimodal-coder" alias is added below (once wired) so callers
        # can reach it without hardcoding the served model id, mirroring the tier
        # alias ergonomics without making it a capability tier of its own.
        _optional_backend(
            env,
            name="multimodal-coder",
            url_key="MULTIMODAL_CODER_BASE_URL",
            name_key="MULTIMODAL_CODER_SERVED_NAME",
            default_url="http://vllm-multimodal-coder:8000",
            default_name=_DEFAULT_MULTIMODAL_CODER,
        ),
        # The legacy 14B Qwen3-NVFP4 "middle" gear. Demoted in #69 from the
        # "normal" tier (now the Gemma multimodal gear) to an opt-in legacy
        # candidate: wired only when MIDDLE_BASE_URL or MIDDLE_SERVED_NAME is
        # present (the compose "middle"/"legacy" profile sets them). Because its
        # backend name "middle" is NOT a TIER_ROLE role, it gets no tier alias —
        # it is reachable by its explicit served name only (resolve_model matches
        # backend.served_name), exactly as the compose template documents. Kept
        # so enabling the profile actually routes to the 14B instead of silently
        # falling back to the primary.
        _optional_backend(
            env,
            name="middle",
            url_key="MIDDLE_BASE_URL",
            name_key="MIDDLE_SERVED_NAME",
            default_url="http://vllm-middle:8000",
            default_name=_DEFAULT_MIDDLE,
        ),
        _optional_backend(
            env,
            name="embed",
            url_key="EMBED_URL",
            name_key="EMBED_SERVED_NAME",
            default_url="http://vllm-embed:8000",
            default_name=_DEFAULT_EMBED,
            task="embed",
        ),
        _optional_backend(
            env,
            name="rerank",
            url_key="RERANK_URL",
            name_key="RERANK_SERVED_NAME",
            default_url="http://vllm-rerank:8000",
            default_name=_DEFAULT_RERANK,
            task="score",
        ),
    )
    backends = [primary, *(b for b in optional if b is not None)]
    # The capability-tier layer: main/minor/multimodal (and back-compat
    # cheap/normal/hard) resolve to the served name of the wired minor /
    # multimodal / primary *generate* gear, on top of the task-family routing.
    # Computed from the wired generate backends using catalog.TIER_ROLE (no
    # parallel tier map). A tier whose gear is absent falls back upward to the
    # nearest higher tier (ultimately the always-present primary). Explicit
    # GATEWAY_ALIASES are merged last so an operator override wins over a
    # computed tier alias.
    aliases = tier_aliases(backends, TIER_ROLE)
    # The opt-in coder alias: only added once its own backend is wired (mirrors the
    # tier-fallback contract — an alias never points at a served name nothing
    # actually serves). Computed before the GATEWAY_ALIASES merge so an operator
    # override still wins if they explicitly set "multimodal-coder=..." themselves.
    coder_backend = next((b for b in backends if b.name == "multimodal-coder"), None)
    if coder_backend is not None:
        aliases["multimodal-coder"] = coder_backend.served_name
    aliases.update(_expand_tier_alias_synonyms(_parse_aliases(env.get("GATEWAY_ALIASES"))))
    # Hardware feasibility (task t6): computed over the FOUR canonical backend
    # names FEASIBLE_ENV knows about — independent of whether each is actually
    # WIRED in this table, so a role declared infeasible with no *_BASE_URL set
    # at all still lands in `infeasible` (a config/display fact, not contingent
    # on wiring). See RoutingTable.infeasible / infeasible_owner.
    infeasible = frozenset(name for name in FEASIBLE_ENV if not _is_feasible(env, name))
    table = RoutingTable(
        backends=tuple(backends),
        default_model=env.get("GATEWAY_DEFAULT_MODEL") or primary.served_name,
        aliases=aliases,
        infeasible=infeasible,
        # Opt-in honest referral (mesh-brain t3): the operator-declared peer
        # origins, empty by default — see PEER_ORIGIN_ENV above.
        peer_origins=_peer_origins(env),
    )
    server = ServerConfig(
        host=env.get("GATEWAY_HOST") or "0.0.0.0",  # nosec B104 — bind all inside the container
        port=_as_int(env, "GATEWAY_PORT", 8000),
        connect_timeout=_as_float(env, "GATEWAY_CONNECT_TIMEOUT", 5.0),
        read_timeout=_as_float(env, "GATEWAY_READ_TIMEOUT", 600.0),
        audio_url=(env.get("AUDIO_URL") or "").rstrip("/") or None,
        public_url=(env.get("GATEWAY_PUBLIC_URL") or "").rstrip("/") or None,
    )
    return table, server
