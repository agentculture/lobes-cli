"""Build the gateway's :class:`RoutingTable` + :class:`ServerConfig` from env vars.

Reads a mapping (``os.environ`` by default) and constructs frozen config objects.
No sockets — pass a plain ``dict`` to unit-test it offline. The env keys mirror
the ``gateway`` service's ``environment:`` block in the fleet compose template.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

from lobes.catalog import TIER_ROLE
from lobes.gateway._routing import Backend, RoutingTable, tier_aliases

_DEFAULT_PRIMARY = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_DEFAULT_FALLBACK = "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4"
_DEFAULT_EMBED = "Qwen/Qwen3-Embedding-0.6B"
# The opt-in "deep" embedding slot — the higher-fidelity companion to _DEFAULT_EMBED,
# reachable via the "embed-deep" alias when its own backend is wired (mirrors the
# multimodal-coder opt-in alias below). Deliberately a SECOND embed-task backend
# rather than a replacement: the 0.6B keeps the latency-sensitive hot path.
# The slot is named for its job, not its model — swapping this default to a larger
# checkpoint later keeps the alias stable (but invalidates any index built with the
# previous one; the two vector spaces are not interoperable).
_DEFAULT_EMBED_DEEP = "Qwen/Qwen3-Embedding-4B"
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
# The opt-in muse gear (Gemma 4 31B IT, NVIDIA NVFP4) — the seventh Colleague
# role's backend. Hosted only by a muse-hosting deployment shape (never
# machine-as-brain), so its backend is wired only when MUSE_BASE_URL is set —
# and, uniquely, it is INFEASIBLE by default when unwired (see OPT_IN_BACKENDS).
_DEFAULT_MUSE = "nvidia/Gemma-4-31B-IT-NVFP4"

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
    "muse": "MUSE_FEASIBLE",
    "embed": "EMBED_FEASIBLE",
    "rerank": "RERANK_FEASIBLE",
    # First-class audio roles (issue #129): stt/tts joined the same channel so
    # a box that deliberately does not serve one audio lane (e.g. Parakeet
    # local, Chatterbox on a peer) declares it exactly like a dropped core
    # role. ABSENT/blank stays feasible — the operator direction that dropped/
    # undeployed models remain DECLARED sleeping lobes (feasible:true,
    # ready:false) holds, so every existing deployment renders byte-identically
    # until an operator explicitly sets STT_FEASIBLE/TTS_FEASIBLE=false.
    "stt": "STT_FEASIBLE",
    "tts": "TTS_FEASIBLE",
}

_FALSY_FEASIBLE = frozenset({"false", "0", "no"})

# Backend names that are OPT-IN heavy lobes: hosted only by an explicit
# muse-hosting deployment shape, never by the default machine-as-brain (see
# lobes.profiles.shapes.OPT_IN_CORE_ROLES). Their feasibility DEFAULT is
# inverted: with no explicit ``<PREFIX>_FEASIBLE`` value in the env, an
# opt-in name is feasible only when its backend is actually WIRED
# (``*_BASE_URL`` set). This keeps a pre-muse ``.env`` honest without a
# re-init — ``model=muse`` on such a box 404s ``role_infeasible`` (referable /
# proxyable via the peer channels) instead of silently upward-falling-back to
# the primary, the exact half-honest posture #92 forbids. An explicit
# truthy/falsy ``MUSE_FEASIBLE`` always wins over this default.
OPT_IN_BACKENDS: frozenset[str] = frozenset({"muse"})

# Generic truthy-token set for opt-in boolean env knobs (mirrors
# lobes.gateway.server._OVERRIDE_TRUTHY, which does the same job for the
# X-Lobes-Override HEADER — this is the env-var counterpart). Kept local to
# this module rather than imported from server.py to avoid the reverse
# import (server.py imports THIS module, not the other way around).
_TRUTHY = frozenset({"1", "true", "yes"})


def _as_bool(env: Mapping[str, str], key: str) -> bool:
    """True iff ``env[key]`` holds a truthy token (``1``/``true``/``yes``,
    case-insensitive). Absent/blank/anything else -> False, so an untouched
    deployment is unaffected — every opt-in boolean knob built on this
    (e.g. ``GATEWAY_FORCE_STRICT_TOOLS``) is default-off.
    """
    return (env.get(key) or "").strip().lower() in _TRUTHY


# Per-backend "the peer box at THIS origin hosts the role I dropped" channel
# (mesh-brain t3, issue #112's confirmed cross-box decision: direct + honest
# referral). DESIGN DECISION, made within the #92 lesson: the referral origin
# is a full, OPERATOR-DECLARED origin (e.g. ``http://spark.local:8001``) set
# per peer in the deployment's ``.env`` — NEVER fabricated or inferred from
# hostnames/interfaces (deriving a URL from the local box's own view of the
# network is exactly what #92 forbade). It uses the SAME
# ``<PREFIX>_<KNOB>`` backend-name prefixes as :data:`FEASIBLE_ENV` /
# ``ROLE_MAX_MODEL_LEN_ENV`` so there is still exactly one env convention to
# learn. Scoped to the five Profile-schema backends PLUS the two first-class
# audio roles (stt/tts — issue #129; they stay outside the Profile TUNING
# schema but ride the same referral/feasibility/proxy channels).
#
# A declared origin is CONTROL-PLANE metadata by default: it annotates
# ``/capabilities`` and the 404 ``role_infeasible`` body for a role this box
# does not host, and the gateway does NOT dial it on its own — origin alone
# stays referral-only (the issue #112 contract, preserved byte-for-byte). A
# box CAN be opted into actually dialing it — the data-plane proxy branch
# (:data:`PEER_PROXY_ENV` below, proxy-lobes t6, issues #115/#127) — but only
# for a name that ALSO carries the truthy ``<PREFIX>_PEER_PROXY`` knob; origin
# without that knob never gets dialed. Unset everywhere (the default) ⇒ every
# response is byte-identical to the pre-referral contract.
PEER_ORIGIN_ENV: dict[str, str] = {
    "primary": "PRIMARY_PEER_ORIGIN",
    "multimodal": "MULTIMODAL_PEER_ORIGIN",
    "muse": "MUSE_PEER_ORIGIN",
    "embed": "EMBED_PEER_ORIGIN",
    "rerank": "RERANK_PEER_ORIGIN",
    # First-class audio roles (issue #129 item 3): the referral/proxy channels
    # now cover stt/tts with the same one env convention — the trigger was a
    # real deployment (Spark GB10) wanting Chatterbox served from the Thor
    # while Parakeet stays local, which AUDIO_URL alone cannot express.
    "stt": "STT_PEER_ORIGIN",
    "tts": "TTS_PEER_ORIGIN",
}

# Per-backend "PROXY my dropped role to its declared peer" opt-in knob
# (proxy-lobes t1, issues #115/#127 — the follow-up :data:`PEER_ORIGIN_ENV`
# above explicitly deferred). Same ``<PREFIX>_<KNOB>`` backend-name prefixes
# as :data:`FEASIBLE_ENV` / :data:`PEER_ORIGIN_ENV` — still exactly one env
# convention to learn — over the same seven-name scope (the five core
# backends + the first-class stt/tts audio roles, issue #129).
#
# A truthy token (``1``/``true``/``yes``, case-insensitive — the same
# :func:`_as_bool` contract every opt-in boolean knob here uses) arms the
# knob, but it composes into :attr:`RoutingTable.peer_proxied` ONLY when
# that backend ALSO has a declared peer origin AND is in the infeasible
# set. The two ignored combinations are deliberate:
#
# * **origin without the knob** stays annotation-only referral — the issue
#   #112 contract is preserved byte-for-byte (an operator who declared a
#   peer for honesty's sake is never silently upgraded to proxying);
# * **knob without an origin** has nothing to dial — a proxy target is
#   always OPERATOR-DECLARED (the #92 lesson), never derived, so an armed
#   knob with no origin is inert, and a knob on a locally-FEASIBLE role is
#   equally inert (the local engine serves it — hosted behaviour unchanged).
#
# The data-plane branch that actually forwards a request using this knob is
# :func:`lobes.gateway.server._proxy_to_peer` (proxy-lobes t6, issues
# #115/#127) — this module only parses the knob into the routing table.
PEER_PROXY_ENV: dict[str, str] = {
    "primary": "PRIMARY_PEER_PROXY",
    "multimodal": "MULTIMODAL_PEER_PROXY",
    "muse": "MUSE_PEER_PROXY",
    "embed": "EMBED_PEER_PROXY",
    "rerank": "RERANK_PEER_PROXY",
    # Audio roles (issue #129): same three-condition arming as every other
    # name — truthy knob + declared origin + declared infeasible here.
    "stt": "STT_PEER_PROXY",
    "tts": "TTS_PEER_PROXY",
}

# Per-backend OUTBOUND credential for the declared peer (proxy-lobes t1,
# issues #115/#127 — the pairwise-auth half). Same prefixes/scope as the
# other three channels above. The value is the API key this box will
# present when dialing that role's peer origin — taken VERBATIM (stripped)
# from the operator's env, never transformed. Parsed into
# :attr:`RoutingTable.peer_api_keys` ONLY for a backend that ALSO has a
# declared peer origin (a key without an origin is inert — there is no
# peer to authenticate to); blank/unset omitted. Deliberately NOT gated on
# :data:`PEER_PROXY_ENV`: the credential rides the origin declaration, so
# a referral-only peer may already carry its key (harmless until the later
# data-plane task dials it). SECRET — it must never appear in repr/str of
# the config objects (see the ``repr=False`` on the RoutingTable field).
PEER_API_KEY_ENV: dict[str, str] = {
    "primary": "PRIMARY_PEER_API_KEY",
    "multimodal": "MULTIMODAL_PEER_API_KEY",
    "muse": "MUSE_PEER_API_KEY",
    "embed": "EMBED_PEER_API_KEY",
    "rerank": "RERANK_PEER_API_KEY",
    # Audio roles (issue #129): the O(machines) rule holds — the value is a
    # copy of the peer box's own inbound GATEWAY_API_KEY, never minted per
    # pairing.
    "stt": "STT_PEER_API_KEY",
    "tts": "TTS_PEER_API_KEY",
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


def _peer_proxied(
    env: Mapping[str, str],
    peer_origins: Mapping[str, str],
    infeasible: frozenset[str],
) -> frozenset[str]:
    """Backend names whose dropped role is opted in to peer proxying.

    A name lands here only when ALL THREE hold: its ``<PREFIX>_PEER_PROXY``
    env var (see :data:`PEER_PROXY_ENV`) is truthy, it has a declared peer
    origin, and it is infeasible on this box. Origin without the knob stays
    referral-only (the issue #112 contract preserved); knob without an
    origin has nothing to dial; knob on a feasible role is ignored (hosted
    behaviour unchanged). Empty (the default) everywhere no knob is set, so a
    deployment that never sets ``<PREFIX>_PEER_PROXY`` is unaffected. A name
    that DOES land here is dialed by the data-plane proxy branch
    (:func:`lobes.gateway.server._proxy_to_peer`, proxy-lobes t6, issues
    #115/#127) — this function only computes the set; it dials nothing
    itself.
    """
    return frozenset(
        name
        for name, key in PEER_PROXY_ENV.items()
        if _as_bool(env, key) and name in peer_origins and name in infeasible
    )


def _peer_api_keys(env: Mapping[str, str], peer_origins: Mapping[str, str]) -> dict[str, str]:
    """Outbound per-peer API keys, keyed by backend name; blank/unset omitted.

    Values are taken VERBATIM (stripped) from ``<PREFIX>_PEER_API_KEY``
    (see :data:`PEER_API_KEY_ENV`), and kept only for names that ALSO have
    a declared peer origin — a key without an origin is inert (no peer to
    authenticate to). Not gated on the proxy knob: the credential rides
    the origin declaration. The values are SECRETS — they flow into the
    ``repr=False`` :attr:`RoutingTable.peer_api_keys` field and must never
    be logged or echoed.
    """
    out: dict[str, str] = {}
    for name, key in PEER_API_KEY_ENV.items():
        value = (env.get(key) or "").strip()
        if value and name in peer_origins:
            out[name] = value
    return out


def _gateway_api_key(env: Mapping[str, str]) -> str | None:
    """The inbound gateway API key: ``GATEWAY_API_KEY`` → ``CULTURE_VLLM_API_KEY`` → None.

    Resolution order (first non-blank wins, whitespace stripped):

    1. ``GATEWAY_API_KEY`` — the explicit, gateway-scoped knob;
    2. ``CULTURE_VLLM_API_KEY`` — the key Culture-mesh operators ALREADY
       distribute to callers of this endpoint, so an operator whose exposed
       deployment runs on that existing key gets gateway auth without
       minting/redistributing a second secret;
    3. ``None`` — both unset/blank ⇒ auth disabled, byte-identical to
       today's no-auth behaviour (an untouched deployment is unaffected).

    The inbound auth check that enforces this key is
    :meth:`lobes.gateway.server._Handler._authorized` (proxy-lobes t2,
    issues #115/#127) — this function only resolves the key's value.
    """
    for key in ("GATEWAY_API_KEY", "CULTURE_VLLM_API_KEY"):
        value = (env.get(key) or "").strip()
        if value:
            return value
    return None


def _is_feasible(env: Mapping[str, str], backend_name: str, *, wired: bool = True) -> bool:
    """True unless ``backend_name``'s ``<PREFIX>_FEASIBLE`` env var (see
    :data:`FEASIBLE_ENV`) holds an explicit falsy token.

    Absent/blank/anything-but-a-recognised-falsy-token → feasible — an
    untouched deployment (no FEASIBLE var set anywhere) is completely
    unaffected, matching every other knob's ``${VAR:-default}`` convention.
    A backend with no entry in :data:`FEASIBLE_ENV` is always feasible here
    (out of the per-machine Profile schema's core-role scope).

    ONE exception (see :data:`OPT_IN_BACKENDS`): an opt-in heavy lobe whose
    ``<PREFIX>_FEASIBLE`` is absent/blank defaults to the ``wired`` fact
    instead of ``True`` — an unwired opt-in lobe is honestly infeasible, so a
    request for it 404s ``role_infeasible`` rather than upward-falling-back
    to the primary. An explicit truthy/falsy value always wins.
    """
    key = FEASIBLE_ENV.get(backend_name)
    if key is None:
        return True
    raw = (env.get(key) or "").strip().lower()
    if raw in _FALSY_FEASIBLE:
        return False
    if not raw and backend_name in OPT_IN_BACKENDS:
        return wired
    return True


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
    # Opt-in (colleague#320): force `"strict": true` onto every tool schema of
    # a chat-completions request routed to a backend in
    # lobes.gateway.server._STRICT_TOOL_LANES — currently `primary` (cortex)
    # ONLY — so xgrammar's structural-tag constrained decoding makes a
    # malformed tool call impossible. False (default) is a byte-identical
    # passthrough — this knob touches NOTHING unless explicitly turned on. See
    # lobes.gateway.server.inject_strict_tools / handle_post for the
    # injection + retry-without-strict-on-compile-failure behaviour.
    # `muse` is DELIBERATELY excluded despite serving tool calls: measured live
    # on the 31B, strict never engages xgrammar on that lane at all, so arming
    # it would advertise a grammar-constrained lane that isn't one. That lane
    # set is the single authority — see _STRICT_TOOL_LANES for the evidence.
    force_strict_tools: bool = False
    # The INBOUND gateway API key (proxy-lobes t1, issues #115/#127 — the
    # pairwise-auth half). Resolved by :func:`_gateway_api_key`:
    # ``GATEWAY_API_KEY`` if non-blank, else ``CULTURE_VLLM_API_KEY`` if
    # non-blank (the key Culture-mesh operators already hand to callers of
    # this endpoint keeps working — no second secret to mint), else ``None``
    # ⇒ auth disabled, today's exact no-auth behaviour. Enforced by
    # :meth:`lobes.gateway.server._Handler._authorized` (t2) on every
    # data-plane route. ``repr=False`` because the value is a SECRET: it must
    # never appear in repr/str of this object (logs, tracebacks, debug
    # output).
    api_key: str | None = field(default=None, repr=False)


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


def _warn_on_served_name_collisions(backends: list[Backend]) -> None:
    """Emit a stderr warning for any served name claimed by more than one backend."""
    by_name: dict[str, list[Backend]] = {}
    for backend in backends:
        by_name.setdefault(backend.served_name, []).append(backend)
    for served, owners in sorted(by_name.items()):
        if len(owners) < 2:
            continue
        names = ", ".join(sorted(b.name for b in owners))
        tasks = {b.task for b in owners}
        detail = (
            " Both serve task=embed, so requests may be answered from the WRONG "
            "VECTOR SPACE — embeddings from different models are not comparable."
            if tasks == {"embed"}
            else ""
        )
        sys.stderr.write(
            f"[gateway] WARNING: served name {served!r} is claimed by {len(owners)} "
            f"backends ({names}); routing resolves it to the first match, so "
            f"ownership is order-dependent.{detail} Give each backend a distinct "
            f"*_SERVED_NAME.\n"
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
        # The opt-in muse generate backend (Gemma 4 31B IT, NVIDIA NVFP4 — the
        # seventh Colleague role, the creative/ideation lobe). Wired only when
        # MUSE_BASE_URL is present — i.e. when a muse-hosting deployment shape
        # (thor-muse) rendered its activation env (COMPOSE_PROFILES=muse +
        # MUSE_BASE_URL, see lobes.profiles.shape_render). Absent by default,
        # so the routing table is unchanged on every pre-muse deployment; the
        # unwired backend is also INFEASIBLE by default (OPT_IN_BACKENDS above)
        # so `model=muse` 404s role_infeasible instead of falling back upward.
        _optional_backend(
            env,
            name="muse",
            url_key="MUSE_BASE_URL",
            name_key="MUSE_SERVED_NAME",
            default_url="http://vllm-muse:8000",
            default_name=_DEFAULT_MUSE,
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
        # The opt-in "deep" embedding gear — a SECOND task="embed" backend beside
        # the 0.6B one above. Wired only when EMBED_DEEP_BASE_URL is set (the
        # *_BASE_URL convention every opt-in backend uses; only the original
        # primary/embed/rerank trio uses the older *_URL spelling), so an
        # existing deployment renders byte-identically until an operator opts in.
        # Task-family routing is already generic over N backends — resolve_model /
        # order_backends match on served_name, not on a one-per-task assumption.
        _optional_backend(
            env,
            name="embed-deep",
            url_key="EMBED_DEEP_BASE_URL",
            name_key="EMBED_DEEP_SERVED_NAME",
            default_url="http://vllm-embed-deep:8000",
            default_name=_DEFAULT_EMBED_DEEP,
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
    # Opt-in backends whose alias is simply their own backend name. "embed-deep"
    # joins on the same contract: it is NOT a generate-lane capability tier (it
    # serves task="embed", and tier_aliases is generate-only), so it gets no
    # upward fallback — an absent deep gear means the alias is absent, never a
    # silent downgrade to the 0.6B, which would answer in the WRONG VECTOR SPACE.
    for _opt_in in ("multimodal-coder", "embed-deep"):
        _opt_in_backend = next((b for b in backends if b.name == _opt_in), None)
        if _opt_in_backend is not None:
            aliases[_opt_in] = _opt_in_backend.served_name
    aliases.update(_expand_tier_alias_synonyms(_parse_aliases(env.get("GATEWAY_ALIASES"))))
    # Hardware feasibility (task t6): computed over the FIVE canonical backend
    # names FEASIBLE_ENV knows about — independent of whether each is actually
    # WIRED in this table, so a role declared infeasible with no *_BASE_URL set
    # at all still lands in `infeasible` (a config/display fact, not contingent
    # on wiring). See RoutingTable.infeasible / infeasible_owner. The `wired`
    # fact is passed through for the OPT_IN_BACKENDS default (muse: unwired and
    # unflagged ⇒ infeasible — see _is_feasible).
    # Served-name collision guard. resolve_model / order_backends match on
    # served_name and return the FIRST hit, so two wired backends sharing one
    # served name make ownership silently order-dependent. Harmless for a
    # duplicated generate gear (same family, same answer shape); NOT harmless on
    # the embed lane, where the two gears occupy different VECTOR SPACES — the
    # wrong owner returns confident, meaningless similarity instead of an error,
    # defeating the whole reason embed-deep has no fallback. Only an operator can
    # cause this (by pointing EMBED_DEEP_SERVED_NAME at another gear's id), so we
    # do not refuse to start — taking the fleet down over a name clash is worse
    # than serving it — but it must never be SILENT.
    _warn_on_served_name_collisions(backends)
    wired_names = frozenset(b.name for b in backends)
    infeasible = frozenset(
        name for name in FEASIBLE_ENV if not _is_feasible(env, name, wired=name in wired_names)
    )
    # Opt-in honest referral (mesh-brain t3): the operator-declared peer
    # origins, empty by default — see PEER_ORIGIN_ENV above. Computed once
    # here because the proxy-lobes channels below both gate on it.
    peer_origins = _peer_origins(env)
    table = RoutingTable(
        backends=tuple(backends),
        default_model=env.get("GATEWAY_DEFAULT_MODEL") or primary.served_name,
        aliases=aliases,
        infeasible=infeasible,
        peer_origins=peer_origins,
        # Proxy-lobes config channels (t1, #115/#127) — parsed only, nothing
        # dials them in this task; see PEER_PROXY_ENV / PEER_API_KEY_ENV above.
        peer_proxied=_peer_proxied(env, peer_origins, infeasible),
        peer_api_keys=_peer_api_keys(env, peer_origins),
    )
    server = ServerConfig(
        host=env.get("GATEWAY_HOST") or "0.0.0.0",  # nosec B104 — bind all inside the container
        port=_as_int(env, "GATEWAY_PORT", 8000),
        connect_timeout=_as_float(env, "GATEWAY_CONNECT_TIMEOUT", 5.0),
        read_timeout=_as_float(env, "GATEWAY_READ_TIMEOUT", 600.0),
        audio_url=(env.get("AUDIO_URL") or "").rstrip("/") or None,
        public_url=(env.get("GATEWAY_PUBLIC_URL") or "").rstrip("/") or None,
        force_strict_tools=_as_bool(env, "GATEWAY_FORCE_STRICT_TOOLS"),
        api_key=_gateway_api_key(env),
    )
    return table, server
