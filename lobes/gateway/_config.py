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

from lobes.catalog import TIER_ROLE, resolve_tier
from lobes.gateway._routing import Backend, RoutingTable, tier_aliases

# The multimodal cortex (promoted 2026-07-31, replacing the text-only
# sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP). NOTE this is the served-name a
# deployment falls back to when PRIMARY_SERVED_NAME is unset — changing it
# changes the id callers must name, and NO consumer in this mesh addresses by
# role name (they all send the raw id), so a swap 404s them until they migrate
# to the stable `cortex`/`main` aliases. See docs/model-switch-playbook.md §2.
_DEFAULT_PRIMARY = "unsloth/Qwen3.8-27B-NVFP4"
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
# and, like the worker gear below, it is INFEASIBLE by default when unwired
# (see OPT_IN_BACKENDS).
_DEFAULT_MUSE = "nvidia/Gemma-4-31B-IT-NVFP4"
# The opt-in worker gear — the eighth Colleague role's backend
# (thor-worker-lobe plan, t1/t3). Hosted only by a worker-hosting deployment
# shape (never machine-as-brain), so its backend is wired only when
# WORKER_BASE_URL is set — and, like muse above, it is INFEASIBLE by default
# when unwired (see OPT_IN_BACKENDS).
#
# DERIVED from the catalog's own role_hint="worker" entry (#244 t6) rather
# than a literal id — the previous literal here
# (``unsloth/Qwen3.6-35B-A3B-NVFP4``) went stale the moment deviation d1
# re-checkpointed `worker` to Lightning
# (``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4``, see CLAUDE.md and
# catalog.py's role_hint="worker" entry): an unset WORKER_SERVED_NAME was
# advertising a checkpoint no box in the mesh serves any more. Deriving
# instead of hardcoding means the next checkpoint swap only has to update
# the catalog, not this file too — see catalog.resolve_tier.
_DEFAULT_WORKER = resolve_tier("worker").id
# The opt-in associate gear (lightning-on-orin plan, t6) — the TENTH Colleague
# role's backend. It shares its checkpoint HISTORY with `worker` (associate is
# worker MINUS repo_action, a different AUTHORITY, not a different model) but
# NOT its catalog role_hint any more: issue #244 t2 gave the Lightning entry
# its own role_hint="associate", precisely so a `worker` checkpoint swap
# cannot silently move this default (the shared hint WAS the defect — #244
# t1 repointed `worker` to a different checkpoint, and this literal, unlike
# _DEFAULT_WORKER above, would otherwise never have noticed).
#
# DERIVED from the catalog's own role_hint="associate" entry, the same
# pattern _DEFAULT_WORKER uses, rather than a literal id — see that comment
# for the failure mode a hardcoded literal repeats. Hosted only by an
# associate-hosting deployment shape (never machine-as-brain), so its backend
# is wired only when ASSOCIATE_BASE_URL is set — and, like muse/worker, it is
# INFEASIBLE by default when unwired (see OPT_IN_BACKENDS). The compose
# lane's own default (docker-compose.yml's vllm-associate) is this same id.
_DEFAULT_ASSOCIATE = resolve_tier("associate").id
# The `hand` gear (LiquidAI LFM2.5-1.2B-Instruct) — the NINTH Colleague role's
# backend and the fleet's designated fine-tuning base. Unlike muse/worker this
# one is DEFAULT-HOSTED on every card (~2.4 GiB bf16 is cheap enough to always
# co-reside), so it is deliberately NOT in OPT_IN_BACKENDS: an unwired hand is
# the sleeping-lobe posture (feasible:true / ready:false), not infeasible.
# It also took over the `minor`/`cheap` capability tier from Qwen/Qwen3.5-4B —
# see lobes.catalog.TIER_ROLE.
_DEFAULT_HAND = "LiquidAI/LFM2.5-1.2B-Instruct"

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
    # The opt-in worker role (thor-worker-lobe plan, t3) rides the same
    # channel as muse — see OPT_IN_BACKENDS below.
    "worker": "WORKER_FEASIBLE",
    # The opt-in associate role (lightning-on-orin plan, t6) rides the same
    # channel as muse/worker — see OPT_IN_BACKENDS below.
    "associate": "ASSOCIATE_FEASIBLE",
    # The `hand` role (hand-lobe plan, t4) rides the same channel, but is NOT
    # in OPT_IN_BACKENDS: it is default-hosted, so an ABSENT HAND_FEASIBLE
    # means feasible. That is the deliberate sleeping-lobe posture for a wheel
    # upgrade — a pre-hand `.env` reads hand as feasible:true / ready:false
    # until the operator re-inits and the lane actually comes up, rather than
    # advertising a lane that is running (it isn't) or denying a lane the card
    # can obviously serve (it can).
    "hand": "HAND_FEASIBLE",
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
    # The opt-in innereye role (issue #82, t5) — the ComfyUI render tenant,
    # the eleventh Colleague role. Rides the same channel as
    # muse/worker/associate (see OPT_IN_BACKENDS below), NOT the stt/tts
    # sleeping-lobe default: an unwired-and-unflagged innereye is honestly
    # infeasible, so `model=innereye` 404s role_infeasible instead of
    # silently reading as a not-yet-ready generate lane.
    "innereye": "INNEREYE_FEASIBLE",
}

_FALSY_FEASIBLE = frozenset({"false", "0", "no"})

# Backend names that are OPT-IN heavy lobes: hosted only by an explicit
# muse-hosting or worker-hosting deployment shape, never by the default
# machine-as-brain (see lobes.profiles.shapes.OPT_IN_CORE_ROLES). Their
# feasibility DEFAULT is inverted: with no explicit ``<PREFIX>_FEASIBLE``
# value in the env, an opt-in name is feasible only when its backend is
# actually WIRED (``*_BASE_URL`` set). This keeps a pre-muse/pre-worker
# ``.env`` honest without a re-init — ``model=muse`` / ``model=worker`` on
# such a box 404s ``role_infeasible`` (referable / proxyable via the peer
# channels) instead of silently upward-falling-back to the primary, the exact
# half-honest posture #92 forbids. An explicit truthy/falsy
# ``MUSE_FEASIBLE``/``WORKER_FEASIBLE`` always wins over this default. worker
# joined muse on this channel via the thor-worker-lobe plan (t3) — the second
# opt-in-core role, same honesty contract.
# `innereye` joined this set with issue #82's t5: the ComfyUI render tenant is
# opt-in-hosted like muse/worker/associate, never default-hosted like the
# audio overlay — see FEASIBLE_ENV above.
OPT_IN_BACKENDS: frozenset[str] = frozenset({"muse", "worker", "associate", "innereye"})

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


# Retired (t14, issue mesh-vs-peer plan): the env peer family — the singular
# referral/proxy/credential channel (PEER_ORIGIN/PEER_PROXY/PEER_API_KEY) and
# the plural replica-pool channel (PEER_ORIGINS/PEER_API_KEYS) — used to be
# parsed here into RoutingTable.peer_origins/peer_proxied/peer_api_keys/
# replica_origins/replica_api_keys. t13 made the mesh RoutingSnapshot the
# candidate source for both pool placement and forwarding, so no behaviour
# depended on these env vars any more; this task deletes the parsing. The
# RoutingTable fields themselves are UNCHANGED (still consumed by
# lobes.gateway.server's pool/proxy/referral code) but now always resolve to
# their empty defaults — nothing in this module populates them from env.
# A leftover ``<PREFIX>_PEER_*`` key in a deployment's .env is now silently
# inert; ``lobes doctor``'s ``peer_family_retired`` finding is what catches it
# (lobes/cli/_commands/doctor.py). See docs/gateway-fleet.md's mesh section
# and CLAUDE.md for the mesh-routing replacement.
NEVER_PROXIED_BACKENDS: frozenset[str] = frozenset()


class ReplicaConfigError(ValueError):
    """A replica-pool configuration is unsafe to arm.

    Retired (t14): this used to also cover a ``<PREFIX>_PEER_API_KEYS``
    length mismatch against its origins list — that parsing is gone with the
    rest of the env peer family. The class survives because
    :func:`lobes.gateway.server._check_pool_arming` still raises it for a
    hand-built :class:`~lobes.gateway._routing.RoutingTable` that declares a
    plural replica pool for a dropped role with no singular referral origin
    — a shape a mesh-sourced or directly-constructed table can still take,
    even though no env var populates it any more.
    """


class CapacityConfigError(ValueError):
    """A ``<PREFIX>_MAX_ACTIVE`` value could not be parsed as a number.

    Raised by :func:`_local_capacities` (capacity-relative-pool-routing,
    issue #199, t1). A malformed capacity is worse than a missing one — an
    absent value falls back cleanly to a neutral default a later task
    chooses, but a value that silently parsed to something other than what
    the operator typed would poison ranking without any visible error. So
    this fails loudly at config-build time instead, naming the offending
    backend and knob.
    """


# Per-backend "this box's own declared max ACTIVE requests for this role"
# channel (capacity-relative-pool-routing, issue #199, t1). Same nine
# ``<PREFIX>_<KNOB>`` backend-name keys as every other channel in this module
# (mirrored the retired plural peer-origins channel's naming; PEER_ORIGINS_ENV
# is gone — see the "Retired" comment above). This is deliberately NOT
# ``<PREFIX>_MAX_NUM_SEQS`` — that knob already exists (rendered per-card by
# the profile system) but names vLLM's OOM safety cap, not a measured
# throughput capacity; conflating the two would let an OOM guard silently
# double as a routing weight it was never validated to be (see docs/plans/
# 2026-08-27-capacity-relative-pool-routing.md, s3/c4). The MEASURED knee a
# calibration routine produces (a later task) is what an operator is meant to
# write here — this task only parses whatever number is present.
MAX_ACTIVE_ENV: dict[str, str] = {
    "primary": "PRIMARY_MAX_ACTIVE",
    "multimodal": "MULTIMODAL_MAX_ACTIVE",
    "muse": "MUSE_MAX_ACTIVE",
    "worker": "WORKER_MAX_ACTIVE",
    "associate": "ASSOCIATE_MAX_ACTIVE",
    "hand": "HAND_MAX_ACTIVE",
    "embed": "EMBED_MAX_ACTIVE",
    "rerank": "RERANK_MAX_ACTIVE",
    "stt": "STT_MAX_ACTIVE",
    "tts": "TTS_MAX_ACTIVE",
    "innereye": "INNEREYE_MAX_ACTIVE",
}

# The sentinel every replica ranks at today (weight hardcoded 1.0 everywhere
# — see lobes/gateway/_selection.py's own docstring). Named here, not
# inlined, so the kill switch below and a later ranking task both cite the
# same constant rather than two independently-typed "1.0"s drifting apart.
CAPACITY_SENTINEL: float = 1.0

# A single global env knob (deliberately NOT per-backend, unlike every other
# channel in this module) that pins every resolved capacity — local AND
# peer alike — back to :data:`CAPACITY_SENTINEL`, reproducing today's
# ranking exactly while leaving the replica pool itself armed. Capacity-
# relative routing's own kill switch (capacity-relative-pool-routing, issue
# #199, t1, q4): the pool keeps routing and forwarding, only the capacity
# INPUT reverts to 1.0. Read once at gateway start by :func:`_local_capacities`;
# a later task (t4) applies the same switch to a probed PEER capacity before
# it reaches `_selection.py`, so "local and peer alike" holds end to end.
CAPACITY_KILL_SWITCH_ENV = "GATEWAY_CAPACITY_KILL_SWITCH"


def _local_capacities(env: Mapping[str, str]) -> dict[str, float]:
    """This box's own declared max-active-requests capacity, per backend name.

    Read from ``<PREFIX>_MAX_ACTIVE`` (see :data:`MAX_ACTIVE_ENV`). Absent or
    blank for a name yields NO entry — never a fabricated default and never
    an error: the neutral fallback a caller should use for an undeclared
    capacity is a ranking-policy decision made downstream (issue #199, t3),
    not baked into this config layer. A present-but-unparseable value (e.g.
    ``PRIMARY_MAX_ACTIVE=lots``) raises :class:`CapacityConfigError` naming
    the backend and the offending value, rather than silently falling back
    as if it had been absent.

    When :data:`CAPACITY_KILL_SWITCH_ENV` is truthy (see :func:`_as_bool`),
    every one of the nine role names resolves to :data:`CAPACITY_SENTINEL`
    instead, and any declared ``<PREFIX>_MAX_ACTIVE`` values are ignored —
    the kill switch always wins. A single-box deployment that sets neither
    the per-role knobs nor the kill switch gets an empty mapping, byte-
    identical to today's absence of any capacity signal.
    """
    if _as_bool(env, CAPACITY_KILL_SWITCH_ENV):
        return dict.fromkeys(MAX_ACTIVE_ENV, CAPACITY_SENTINEL)
    out: dict[str, float] = {}
    for name, key in MAX_ACTIVE_ENV.items():
        raw = (env.get(key) or "").strip()
        if not raw:
            continue
        try:
            out[name] = float(raw)
        except ValueError as exc:
            raise CapacityConfigError(
                f"{name.upper()}: {key}={raw!r} is not a valid number — "
                "a declared capacity must be a plain float (this box's own "
                "measured max active requests for that role)."
            ) from exc
    return out


def _self_origin(env: Mapping[str, str]) -> str:
    """This box's own OPERATOR-DECLARED origin, from ``GATEWAY_SELF_ORIGIN``.

    Stripped and trailing-slash-trimmed, matching every other origin value
    in this module. Per the #92 lesson this is NEVER derived from the local
    box's own view of its network (hostname, bound interface, container
    name) — an operator types it, exactly like a peer's origin. Default
    ``""`` (undeclared); a later cortex-replica-pool task uses this to
    recognise (and refuse) a replica pool that names this box as one of its
    own peers.
    """
    return (env.get("GATEWAY_SELF_ORIGIN") or "").strip().rstrip("/")


# The engine knobs a caller might reasonably expect to be IDENTICAL across
# every replica in a role's pool (cortex-replica-pool, issue #199, t2) — two
# "cortex" replicas silently disagreeing on, say, TOOL_CALL_PARSER would answer a
# strict-tools request differently depending on which replica happened to
# serve it. Read per backend name as ``<PREFIX>_<SUFFIX>`` — the same
# ``<PREFIX>_<KNOB>`` convention every other per-backend channel in this
# module uses — by :func:`_lane_fingerprints`. Exported so a later task
# (cross-replica consistency checking) reuses the exact same suffix list
# rather than re-typing it.
LANE_FINGERPRINT_SUFFIXES: tuple[str, ...] = (
    "QUANTIZATION",
    "KV_CACHE_DTYPE",
    "REASONING_PARSER",
    "TOOL_CALL_PARSER",
    "SPECULATIVE_CONFIG",
)


def _lane_fingerprints(env: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Declared per-backend engine knobs, keyed by backend name then suffix.

    Iterates the same ten backend names :data:`FEASIBLE_ENV` knows about
    (``<PREFIX> = name.upper()``) and, for each of
    :data:`LANE_FINGERPRINT_SUFFIXES`, reads ``<PREFIX>_<SUFFIX>`` verbatim
    (stripped) when non-blank. Only SET knobs appear — a backend with none of
    the five set gets no entry in the outer mapping at all, and a backend
    with some-but-not-all set carries only the ones it declared. Parsing
    only in this task: nothing yet compares one backend's fingerprint
    against another's (a later cortex-replica-pool task does that
    cross-replica consistency check). Empty mapping (the default) when no
    such knob is set anywhere, leaving every existing deployment unaffected.
    """
    out: dict[str, dict[str, str]] = {}
    for name in FEASIBLE_ENV:
        prefix = name.upper()
        lane: dict[str, str] = {}
        for suffix in LANE_FINGERPRINT_SUFFIXES:
            value = (env.get(f"{prefix}_{suffix}") or "").strip()
            if value:
                lane[suffix] = value
        if lane:
            out[name] = lane
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
    # This box's own declared max-active-requests capacity per backend name
    # (capacity-relative-pool-routing, issue #199, t1). Populated by
    # :func:`_local_capacities` from ``<PREFIX>_MAX_ACTIVE`` (see
    # :data:`MAX_ACTIVE_ENV`) — absent/blank names are simply missing from
    # this mapping, never defaulted here; the neutral fallback for a missing
    # name is a ranking-policy decision a later task (t3) makes at the point
    # of use. Defaults to empty so an untouched deployment is unaffected.
    local_capacities: Mapping[str, float] = field(default_factory=dict)
    # The capacity kill switch (see :data:`CAPACITY_KILL_SWITCH_ENV`):
    # when True, `local_capacities` above already holds
    # :data:`CAPACITY_SENTINEL` for every role name — this field is the
    # flag itself, carried through so a later task (t4) can apply the SAME
    # switch to a probed PEER capacity before it reaches
    # `lobes.gateway._selection`, satisfying "local and peer alike" end to
    # end. Default False (off) — the pool's capacity signal is live by
    # default wherever a capacity is declared.
    capacity_kill_switch: bool = False


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


# The `hand` lobe's LoRA adapter inventory, declared once at boot (hand-lobe
# plan t4). Operator-typed as a comma-separated ``name=path`` list — the SAME
# string vLLM's own ``--lora-modules`` consumes verbatim on the vllm-hand lane,
# so there is exactly one place an adapter is declared and the gateway can
# never disagree with the engine about which names exist.
#
# There is deliberately NO runtime hot-load: adding an adapter is a lane
# restart. A mutable-adapter API would put changeable state on the one lane
# whose value is being cheap and predictable.
HAND_LORA_MODULES_ENV = "HAND_LORA_MODULES"

# The separator between the role name and the adapter domain in the
# caller-facing spelling (``hand:legal``). Colon, not slash or dot: every
# existing model id in the catalog is ``org/name``, so a slash would be
# ambiguous with an HF repo path, and dots appear inside version numbers
# (``Qwen3.6``). Nothing in the gateway, roles.py or capabilities.py parses a
# model id by delimiter, so this shape passes through the whole stack unharmed.
HAND_ADAPTER_SEP = ":"


def _hand_adapter_names(env: Mapping[str, str]) -> tuple[str, ...]:
    """Adapter NAMES declared in ``HAND_LORA_MODULES``, in declaration order.

    The value is a comma-separated ``name=path`` list. Only the names are read
    here — the paths are vLLM's business, and the gateway deliberately never
    stats them: an adapter path is mounted into the ``vllm-hand`` container, not
    into the gateway's, so a filesystem check here would false-negative every
    correctly-configured adapter. Whether an adapter actually LOADED is
    answered by the live probe against the hand backend's own ``/v1/models``
    (see :func:`lobes.gateway._readiness.probe_backend_adapters`), which is the
    engine's own evidence rather than the gateway's guess.

    Malformed entries are skipped rather than raising: a blank segment, a
    segment with no ``=``, or one with an empty name cannot name a servable
    adapter, and a typo in one entry must not take down a gateway that would
    otherwise serve the base model and every other adapter fine. Duplicate
    names collapse to the first occurrence, preserving order.

    ``partition("=")`` (not ``split("=")``) so an adapter path containing an
    ``=`` — a query string, a padded base64 segment — keeps its full value; the
    same convention this module already uses for its other list-valued knobs.
    """
    raw = (env.get(HAND_LORA_MODULES_ENV) or "").strip()
    if not raw:
        return ()
    names: list[str] = []
    for segment in raw.split(","):
        name, sep, path = segment.strip().partition("=")
        name = name.strip()
        if not sep or not name or not path.strip():
            continue
        if name not in names:
            names.append(name)
    return tuple(names)


def _optional_backend(
    env: Mapping[str, str],
    *,
    name: str,
    url_key: str,
    name_key: str,
    default_url: str,
    default_name: str,
    task: str = "generate",
    adapters: tuple[str, ...] = (),
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
        adapters=adapters,
    )


def _claimed_model_ids(backend: Backend) -> tuple[str, ...]:
    """Every model id ``backend`` answers to: its served name AND its adapters.

    Both are routable. :func:`~lobes.gateway._routing._backend_for` matches
    ``served_name == requested or requested in backend.adapters``, so an adapter
    name is an ownership claim exactly like a served name is, and a collision
    between the two kinds is as order-dependent as a collision between two
    served names.
    """
    return (backend.served_name, *backend.adapters)


def _warn_on_served_name_collisions(backends: list[Backend]) -> None:
    """Emit a stderr warning for any model id claimed by more than one backend.

    Covers served names *and* LoRA adapter names — see
    :func:`_claimed_model_ids` for why the two are the same kind of claim. A
    backend claiming the same id twice (an adapter named after its own base) is
    also reported, since that too resolves by first match.
    """
    by_name: dict[str, list[Backend]] = {}
    for backend in backends:
        for claimed in _claimed_model_ids(backend):
            by_name.setdefault(claimed, []).append(backend)
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
        # A LoRA adapter name is an ownership claim too, so a collision can be
        # adapter-vs-served-name or even a single backend claiming an id twice.
        # Name the remedy for whichever kind this is: *_SERVED_NAME is useless
        # advice when the duplicate came from HAND_LORA_MODULES.
        remedy = (
            "Give each backend a distinct *_SERVED_NAME."
            if all(served == b.served_name for b in owners)
            else "Rename the colliding LoRA adapter in HAND_LORA_MODULES, or give "
            "the backend a distinct *_SERVED_NAME."
        )
        sys.stderr.write(
            f"[gateway] WARNING: model id {served!r} is claimed by {len(owners)} "
            f"backend claim(s) ({names}); routing resolves it to the first match, "
            f"so ownership is order-dependent.{detail} {remedy}\n"
        )


def _hand_adapter_aliases(backends: list[Backend]) -> dict[str, str]:
    """``{"hand:<domain>": "<domain>"}`` for a WIRED hand backend's adapters.

    Extracted from :func:`build_config` so the alias derivation is testable on
    its own and does not add branching to an already-large builder.

    Note what is NOT here: ``hand`` itself. That comes from ``tier_aliases`` as
    a capability tier, so the bare role name resolves to the BASE checkpoint and
    never 404s just because the adapter inventory is empty — an armed-but-empty
    lane is a working lane. An UNdeclared ``hand:<domain>`` gets no alias, is
    not any backend's served name or adapter, and therefore takes the
    :func:`~lobes.gateway._routing.is_unknown_model` 404 ``model_not_found`` —
    never a silent fall-back to the base weights or to another lane, which
    would hand a caller who asked for the legal specialist a generalist answer
    and call it success.
    """
    hand = next((b for b in backends if b.name == "hand"), None)
    if hand is None:
        return {}
    return {f"hand{HAND_ADAPTER_SEP}{adapter}": adapter for adapter in hand.adapters}


def _warn_on_adapter_alias_shadowing(backends: list[Backend], aliases: Mapping[str, str]) -> None:
    """Warn when an adapter name is also an alias key — the alias WINS.

    :func:`~lobes.gateway._routing.resolve_model` checks ``table.aliases``
    first, so an adapter named after a tier or role (``cortex``, ``main``,
    ``senses``, an operator's ``GATEWAY_ALIASES`` key) is unreachable by its own
    name: the request silently routes to the alias target instead. Unlike the
    served-name collision above this is not order-dependent — it is a total
    shadow — which makes it worth its own message.
    """
    for backend in backends:
        for adapter in backend.adapters:
            if adapter in aliases and aliases[adapter] != adapter:
                sys.stderr.write(
                    f"[gateway] WARNING: LoRA adapter {adapter!r} on backend "
                    f"{backend.name!r} is ALSO an alias for "
                    f"{aliases[adapter]!r}; the alias wins in resolve_model, so "
                    f"this adapter is unreachable by name. Rename it in "
                    f"HAND_LORA_MODULES.\n"
                )


def _add_self_named_opt_in_aliases(aliases: dict[str, str], backends: list[Backend]) -> None:
    """Alias each wired opt-in backend to its own name.

    ``multimodal-coder`` and ``embed-deep`` are addressed by their backend name
    rather than a capability tier. ``embed-deep`` in particular gets **no**
    upward fallback: it serves ``task="embed"`` and ``tier_aliases`` is
    generate-only, so an absent deep gear means the alias is simply absent —
    never a silent downgrade to the 0.6B, which would answer in the WRONG
    VECTOR SPACE.

    Only a WIRED backend earns an alias; an alias must never point at a served
    name nothing actually serves. Extracted from :func:`build_config` to keep
    that builder's branching down.
    """
    for name in ("multimodal-coder", "embed-deep"):
        backend = next((b for b in backends if b.name == name), None)
        if backend is not None:
            aliases[name] = backend.served_name


def _add_pooling_role_aliases(aliases: dict[str, str], backends: list[Backend]) -> None:
    """Give the embed/rerank lanes the stable role names the generate lane has.

    A caller that names a role survives a checkpoint swap; a caller that
    hardcodes a served id does not. Before this, ``embedder``/``reranker`` were
    not addressable at all — the only working address was the raw served id,
    because ``tier_aliases`` is generate-only. Both the Colleague-facing ROLE
    name and the internal BACKEND name are accepted, exactly as the generate
    lane takes ``senses``/``multimodal``.

    ``setdefault``, so an operator alias already in ``aliases`` still wins. An
    unwired lane gets nothing — absent (404) rather than substituted, since an
    embedding from a different model answers in the WRONG VECTOR SPACE and a
    rerank from the wrong head returns meaningless orderings. Extracted from
    :func:`build_config` to keep that builder's branching down.
    """
    for role_name, backend_name in (("embedder", "embed"), ("reranker", "rerank")):
        backend = next((b for b in backends if b.name == backend_name), None)
        if backend is not None:
            aliases.setdefault(role_name, backend.served_name)
            aliases.setdefault(backend_name, backend.served_name)


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
        # The `hand` co-resident generate backend (LiquidAI LFM2.5-1.2B, bf16)
        # — the ninth Colleague role, the fleet's fine-tuning base, and the
        # gear the minor/cheap capability tier resolves to since it replaced
        # Qwen3.5-4B in that slot (lobes.catalog.TIER_ROLE).
        #
        # Wired when HAND_BASE_URL is present — that env var ALONE, per
        # _optional_backend's contract: HAND_SERVED_NAME only names the
        # backend once a URL has wired it, and never activates it on its own
        # (a served name with no URL describes a model, not a reachable
        # backend). It is default-HOSTED (every rendered card profile declares
        # it), so on a freshly-inited deployment the URL is always set; it
        # stays an _optional_backend anyway so a pre-hand `.env` — which has
        # neither — simply renders no hand backend rather than pointing at a
        # container that isn't running. `hand` is NOT in OPT_IN_BACKENDS, so
        # that unwired state reads feasible:true / ready:false (the sleeping
        # lobe), not role_infeasible.
        _optional_backend(
            env,
            name="hand",
            url_key="HAND_BASE_URL",
            name_key="HAND_SERVED_NAME",
            default_url="http://vllm-hand:8000",
            default_name=_DEFAULT_HAND,
            adapters=_hand_adapter_names(env),
        ),
        # The LEGACY minor co-resident generate backend (Qwen/Qwen3.5-4B, bf16).
        # KEPT (cite-don't-delete) but no longer a TIER backend: the minor/cheap
        # tiers now resolve to `hand` above. Like the 14B "middle" gear below,
        # it stays addressable by explicit model id when its own env pair is
        # set — `COMPOSE_PROFILES=minor` still works, it is simply no longer
        # what `model=minor` selects.
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
        # The opt-in worker generate backend (unsloth Qwen3.6-35B-A3B-NVFP4,
        # MoE with a self-hosted MTP draft — the eighth Colleague role, the
        # ground-work execution lobe; thor-worker-lobe plan t1/t3). Wired only
        # when WORKER_BASE_URL is present — i.e. when a worker-hosting
        # deployment shape (thor-worker) rendered its activation env
        # (COMPOSE_PROFILES=worker + WORKER_BASE_URL, see
        # lobes.profiles.shape_render). Absent by default, so the routing
        # table is unchanged on every pre-worker deployment; the unwired
        # backend is also INFEASIBLE by default (OPT_IN_BACKENDS above) so
        # `model=worker` 404s role_infeasible instead of falling back upward.
        _optional_backend(
            env,
            name="worker",
            url_key="WORKER_BASE_URL",
            name_key="WORKER_SERVED_NAME",
            default_url="http://vllm-worker:8000",
            default_name=_DEFAULT_WORKER,
        ),
        # The opt-in associate generate backend (Nemotron 3.5 Lightning — the
        # TENTH Colleague role, the doer that does NOT act; lightning-on-orin
        # plan t6). Wired only when ASSOCIATE_BASE_URL is present — i.e. when
        # an associate-hosting deployment shape rendered its activation env
        # (COMPOSE_PROFILES=associate + ASSOCIATE_BASE_URL, see
        # lobes.profiles.shape_render). Absent by default, so the routing
        # table is unchanged on every pre-associate deployment; the unwired
        # backend is also INFEASIBLE by default (OPT_IN_BACKENDS above) so
        # `model=associate` 404s role_infeasible instead of falling back
        # upward to cortex.
        _optional_backend(
            env,
            name="associate",
            url_key="ASSOCIATE_BASE_URL",
            name_key="ASSOCIATE_SERVED_NAME",
            default_url="http://vllm-associate:8000",
            default_name=_DEFAULT_ASSOCIATE,
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
    _add_self_named_opt_in_aliases(aliases, backends)
    # HAND ADAPTER aliases (hand-lobe plan t4): the caller-facing
    # ``hand:<domain>`` spelling for each declared LoRA adapter, resolving to
    # the bare name vLLM itself serves the adapter under (its
    # ``--lora-modules <name>=<path>`` key). ``handle_post`` already rewrites
    # the forwarded body's ``model`` field to the resolved served name, so the
    # engine receives a name it knows without any adapter-specific code on the
    # data path.
    #
    # Added only for a WIRED hand backend, mirroring the opt-in-alias contract
    # directly above: an alias must never point at a served name nothing
    # actually serves. Declared BEFORE the GATEWAY_ALIASES merge so an explicit
    # operator override still wins. See _hand_adapter_aliases for what is
    # deliberately NOT derived here (the bare ``hand``, and any UNdeclared
    # ``hand:<domain>``).
    aliases.update(_hand_adapter_aliases(backends))
    # POOLING ROLE IDENTITY aliases — the stable address for the embed/rerank
    # lanes, mirroring what `cortex`/`senses` already give the generate lane.
    #
    # Why this exists: a caller that names a role survives a checkpoint swap; a
    # caller that hardcodes a served id does not. Before this, `embedder` and
    # `reranker` were NOT addressable at all — the only working address was the
    # raw served id (`Qwen/Qwen3-Embedding-0.6B`), because tier_aliases is
    # generate-only. So every embed consumer (eidetic among them) had to pin a
    # concrete checkpoint, with no stable name to migrate to, and an embed-model
    # swap would 404 all of them. The 2026-07-31 cortex swap demonstrated that
    # failure mode on the generate lane, where role aliases at least existed as
    # an escape hatch; the pooling lanes had none.
    #
    # Same no-fallback contract as `embed-deep` directly above: an alias is added
    # ONLY when its own backend is wired. An absent gear means the alias is
    # absent (404 role_infeasible / model_not_found), NEVER a silent substitution
    # — an embedding served from a different model answers in the WRONG VECTOR
    # SPACE, and a rerank from the wrong head returns meaningless orderings.
    # Both the Colleague-facing ROLE name and the internal BACKEND name are
    # accepted, exactly as the generate lane takes `senses` and `multimodal`.
    _add_pooling_role_aliases(aliases, backends)
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
    # After the GATEWAY_ALIASES merge above, so an operator alias that shadows
    # an adapter is caught too — not just the built-in tier/role ones.
    _warn_on_adapter_alias_shadowing(backends, aliases)
    wired_names = frozenset(b.name for b in backends)
    infeasible = frozenset(
        name for name in FEASIBLE_ENV if not _is_feasible(env, name, wired=name in wired_names)
    )
    # Retired (t14): the env peer family used to be parsed here into
    # peer_origins/peer_proxied/peer_api_keys/replica_origins/replica_api_keys
    # — see the "Retired" comment above _config.NEVER_PROXIED_BACKENDS. The
    # mesh RoutingSnapshot (t13) is the candidate source now, so the table
    # simply takes its dataclass defaults (all empty) for those five fields.
    table = RoutingTable(
        backends=tuple(backends),
        default_model=env.get("GATEWAY_DEFAULT_MODEL") or primary.served_name,
        aliases=aliases,
        infeasible=infeasible,
        self_origin=_self_origin(env),
        lane_fingerprints=_lane_fingerprints(env),
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
        local_capacities=_local_capacities(env),
        capacity_kill_switch=_as_bool(env, CAPACITY_KILL_SWITCH_ENV),
    )
    return table, server
