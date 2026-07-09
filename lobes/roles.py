"""The role registry — the six first-class, Colleague-facing lobes (issue #81).

lobes exposes the fleet not as a bag of model ids but as SIX discoverable
*roles*, each resolved to a live endpoint + metadata so a caller (Colleague)
can address a capability by role — ``cortex``, ``senses``, ``embedder``,
``reranker``, ``stt``, ``tts`` — without hardcoding any single model endpoint:

* ``cortex``   → the ``primary`` generate backend (Qwen 3.6 27B NVFP4 MTP).
  The authoritative reasoning/action/decision layer — the final authority.
* ``senses``   → the ``multimodal`` generate backend (Gemma 4 12B). The
  user-facing intake/perception/speak-back layer; it does NOT decide or act.
* ``embedder`` → the ``embed`` pooling backend (Qwen3-Embedding-0.6B) →
  ``POST /v1/embeddings``.
* ``reranker`` → the ``score``/rerank backend (Qwen3-Reranker-0.6B) →
  ``POST /v1/rerank`` (+ ``/v1/score``).
* ``stt``      → the Parakeet sidecar behind the audio overlay →
  ``POST /v1/audio/transcriptions``. Opt-in (``lobes init --fleet --audio``).
* ``tts``      → the Chatterbox sidecar behind the audio overlay →
  ``POST /v1/audio/speech``. Opt-in.

This module is the SHARED core the CLI (``lobes capabilities``, t5) and the
gateway (``GET /capabilities``, t6) both consume, so the role→endpoint contract
has exactly one source of truth. It is pure/offline: it reads the same config
the gateway builds (a :class:`~lobes.gateway._routing.RoutingTable` +
:class:`~lobes.gateway._config.ServerConfig`) plus the static
:mod:`lobes.catalog`, and touches no sockets.

**Provisional wording (plan risk r2, issue #81):** the ``responsibilities`` /
``forbidden_responsibilities`` token lists below are issue #81's worked
examples. They describe the intended DIVISION OF LABOUR between the lobes; they
are *not* claims about answer correctness or task success — lobes emits a
runtime-only contract. The exact vocabulary is a build-time call and may be
refined without breaking the machine-readable shape.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from lobes.catalog import SUPPORTED_MODELS, SupportedModel
from lobes.gateway._config import ServerConfig, build_config
from lobes.gateway._routing import RoutingTable

# The six first-class roles, in canonical order: generate lane, pooling lane,
# then the opt-in audio overlay. Downstream (CLI/gateway) iterate this for a
# stable ordering.
ROLES: tuple[str, ...] = ("cortex", "senses", "embedder", "reranker", "stt", "tts")

# role → the internal gateway :attr:`Backend.name` that serves it. Only the four
# gateway-fronted roles appear here; ``stt``/``tts`` are audio-overlay sidecars,
# not gateway backends (they are resolved from ``ServerConfig.audio_url`` below).
# NOTE the name↔role_hint mismatch for the pooling lane: the *backend* is named
# ``embed``/``rerank`` while the *catalog* role_hint is ``embedding``/``reranker``.
ROLE_BACKEND: dict[str, str] = {
    "cortex": "primary",
    "senses": "multimodal",
    "embedder": "embed",
    "reranker": "rerank",
}

# role → the catalog ``role_hint`` of its canonical model. Used to (a) look up
# context/quant/mtp for that role, and (b) name the model a role WOULD serve
# when its backend is not wired in this deployment (loaded=False but still named).
ROLE_ROLE_HINT: dict[str, str] = {
    "cortex": "primary",
    "senses": "multimodal",
    "embedder": "embedding",
    "reranker": "reranker",
}

# role → the OpenAI path a caller hits. The reranker exposes both /v1/rerank and
# /v1/score; /v1/rerank is the canonical path advertised here.
ROLE_PATH: dict[str, str] = {
    "cortex": "/v1/chat/completions",
    "senses": "/v1/chat/completions",
    "embedder": "/v1/embeddings",
    "reranker": "/v1/rerank",
    "stt": "/v1/audio/transcriptions",
    "tts": "/v1/audio/speech",
}

# The two audio-overlay sidecars — hardcoded here (as in the gateway/realtime
# code) because they are NOT in the switchable catalog (lobes/catalog.py): they
# are fixed GPU sidecars behind the /v1/audio/* facade, activated together by
# ``lobes init --fleet --audio``.
_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"  # Parakeet TDT 0.6B, NeMo ASR
_STT_RUNTIME = "parakeet"
_TTS_MODEL = "ResembleAI/chatterbox"  # Chatterbox, Resemble AI 0.5B, Apache-2.0
_TTS_RUNTIME = "chatterbox"
_VLLM_RUNTIME = "vllm"  # the four gateway-fronted roles all serve on vLLM

# Canonical responsibilities per role (issue #81 worked examples — PROVISIONAL,
# see the module docstring). A role's responsibilities are what it is EXPECTED to
# own in the division of labour, never a correctness/success claim.
ROLE_RESPONSIBILITIES: dict[str, tuple[str, ...]] = {
    "cortex": (
        "reasoning",
        "deciding",
        "planning",
        "tool_use",
        "code_repo_actions",
        "validation",
        "final_authority",
    ),
    "senses": (
        "intake",
        "normalize_input",
        "classify_intent",
        "prepare_context_packet",
        "speak_back",
    ),
    "embedder": ("vectorization", "memory_retrieval_input"),
    "reranker": ("retrieval_ordering", "relevance_refinement"),
    "stt": ("transcribe", "audio_input_to_text"),
    "tts": ("speech_output", "synthesize"),
}

# What each role must NOT do. cortex is the final authority (nothing forbidden);
# senses is intake/perception only — it must not decide, act on the repo, or make
# security calls. The service roles carry no forbidden list of their own.
ROLE_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "cortex": (),
    "senses": ("final_decision", "repo_action", "security_decision"),
    "embedder": (),
    "reranker": (),
    "stt": (),
    "tts": (),
}

# role → the deployment env var that carries the SERVED ``--max-model-len`` for
# that role's backend (issue #81, t5). Mirrors the fleet compose template's
# `--max-model-len=${...}` flags (see docs/gateway-fleet.md / the fleet
# env.example). Only the four gateway-fronted roles carry one — stt/tts have no
# token context (see :func:`_audio_role`), so they are deliberately absent here.
ROLE_MAX_MODEL_LEN_ENV: dict[str, str] = {
    "cortex": "PRIMARY_MAX_MODEL_LEN",
    "senses": "MULTIMODAL_MAX_MODEL_LEN",
    "embedder": "EMBED_MAX_MODEL_LEN",
    "reranker": "RERANK_MAX_MODEL_LEN",
}


@dataclass(frozen=True)
class RoleInfo:
    """Live metadata for one first-class role (a Colleague-facing lobe).

    Frozen so it is safe to share across gateway threads. JSON-serialisable with
    :func:`dataclasses.asdict` (tuples become arrays) — the CLI ``--json`` (t5)
    and the gateway ``GET /capabilities`` (t6) build their payloads from this.
    """

    role: str
    model: str  # the served model id this role resolves to (never hardcoded blank)
    runtime: str  # the serving stack: "vllm" | "parakeet" | "chatterbox"
    endpoint: str  # base URL of the service the caller hits ("" when not wired)
    path: str  # the OpenAI path, e.g. "/v1/chat/completions"
    # The SERVED context (tokens): the deployment's `--max-model-len` override
    # (ROLE_MAX_MODEL_LEN_ENV) when the env sets one, else the catalog native
    # (`SupportedModel.native_max_model_len`) — issue #81 t5. 0 for audio roles.
    context: int
    quant: str  # vLLM quantization for the model; "" when n/a (pooling/audio)
    mtp: bool  # speculative decoding (MTP draft head) active for this model
    responsibilities: tuple[str, ...]
    forbidden_responsibilities: tuple[str, ...]
    # Runtime readiness — a caller-supplied LIVE signal, folded in by
    # build_role_registry: `backend_ready` (keyed by the ROLE_BACKEND name)
    # for the four gateway-fronted roles, `audio_ready` for stt/tts (issue
    # #89). Generalised from the stt/tts-only split (issue #89/#90) to all six
    # roles (issue #81 t5) — `ready` is no longer a bare alias of `loaded`.
    # When a caller supplies no signal (the parameter is `None`, the default),
    # `ready` falls back to the coarse `loaded` "configured/wired" proxy — the
    # original t4 behaviour, still exercised by every non-HTTP caller (the
    # CLI's non-live paths, most of this module's own test suite).
    # Structurally CLAMPED either way: a role whose backend is not wired
    # (`loaded is False`) or whose `endpoint` is empty can never report
    # `ready=True`, no matter what signal a caller passes in. This mirrors —
    # and is enforced by the same code path as — the stt/tts clamp on
    # `audio_configured` (issue #89/#90 review finding), now applied to all
    # six roles by build_role_registry itself, not left to caller discipline.
    ready: bool = False
    # Is this role's backend/service wired/present in THIS deployment? An
    # unconfigured/opt-in role is still returned, with loaded=False.
    loaded: bool = False


def _catalog_by_id(model_id: str) -> SupportedModel | None:
    """The catalog entry whose ``id`` == ``model_id`` (an operator's served name)."""
    return next((m for m in SUPPORTED_MODELS if m.id == model_id), None)


def _catalog_by_role_hint(role_hint: str) -> SupportedModel | None:
    """The canonical catalog entry for a role_hint (each is unique in the catalog)."""
    return next((m for m in SUPPORTED_MODELS if m.role_hint == role_hint), None)


def _gateway_base_url(server: ServerConfig) -> str:
    """The gateway's caller-facing base URL — NEVER fabricated from host:port.

    ``ServerConfig.host``/``.port`` (``GATEWAY_HOST``/``GATEWAY_PORT``) are the
    gateway process's own INTERNAL listen config — where it binds inside its
    container — not necessarily where a caller can reach it from outside. On
    the reference rig the gateway listens on internal container port 8000 but
    is PUBLISHED on host port 8001, and host port 8000 belongs to a wholly
    unrelated daemon (a stray uvicorn service). A URL built from
    ``host:port`` would therefore silently advertise that foreign daemon as
    if it were the gateway — a caller dialing it gets whatever happens to be
    listening there, not a 404 from the gateway, which is worse than an
    honest "unknown". This function must never do that.

    Returns ``server.public_url`` (the operator-declared, caller-reachable
    origin — ``GATEWAY_PUBLIC_URL``), rstripped of a trailing slash, when it
    is set; otherwise ``""``. An empty return here is not a degraded case to
    special-case downstream — :func:`build_role_registry` already treats an
    empty ``endpoint`` as a hard "never advertise ready=True" signal, so a
    caller either gets a real, dialable endpoint or an honest absence of one.

    Callers that know the real reachable address from elsewhere (a published
    host port, a tunnel URL, or — as the gateway's own HTTP route does,
    issue #87 — the request's own ``Host`` header) must pass it explicitly as
    ``gateway_url`` to :func:`build_role_registry`; that explicit value always
    wins over this fallback.
    """
    return (server.public_url or "").rstrip("/")


def _served_context(role: str, env: Mapping[str, str], native: int) -> int:
    """The SERVED context for ``role`` — issue #81 t5.

    Reads the deployment's ``--max-model-len`` override
    (:data:`ROLE_MAX_MODEL_LEN_ENV`) from ``env`` when present and numeric;
    falls back to the catalog ``native`` context otherwise (unset key, blank
    value, or a malformed override — never raises). ``role`` values with no
    entry in :data:`ROLE_MAX_MODEL_LEN_ENV` (the audio roles) always fall back
    to ``native`` (which :func:`_audio_role` always passes as ``0``).
    """
    key = ROLE_MAX_MODEL_LEN_ENV.get(role)
    if key is None:
        return native
    raw = env.get(key)
    if not raw:
        return native
    try:
        return int(raw)
    except (TypeError, ValueError):
        return native


def _gateway_role(
    role: str,
    table: RoutingTable,
    gateway: str,
    env: Mapping[str, str],
    ready_signal: bool | None,
) -> RoleInfo:
    """Resolve a gateway-fronted role (cortex/senses/embedder/reranker).

    ``ready_signal`` is the caller's live-readiness tri-state for THIS role's
    backend (the value :func:`build_role_registry` looked up in its
    ``backend_ready`` mapping by :data:`ROLE_BACKEND` name) — ``True``/
    ``False`` reflect an actual probe result; ``None`` means no live signal is
    available (no ``backend_ready`` mapping was supplied, or it had no entry
    for this backend), in which case ``ready`` falls back to the coarse
    ``loaded`` proxy, matching the original t4 behaviour.

    Either way, ``ready`` is CLAMPED to ``False`` whenever the backend is not
    wired (``loaded is False``) or the resolved ``endpoint`` is empty (see
    :func:`_gateway_base_url`) — enforced HERE, structurally, so a caller
    passing a stale/wrong ``ready_signal`` for an unwired or undialable role
    can never fabricate ``ready=True``. This generalises, to all four
    gateway-fronted roles, the same clamp issue #89/#90 established for
    stt/tts (a caller-supplied signal can never override "nothing is wired"
    or "nothing to dial").
    """
    backend = next((b for b in table.backends if b.name == ROLE_BACKEND[role]), None)
    loaded = backend is not None
    if backend is not None:
        model_id = backend.served_name
    else:
        # Not wired: still name the model this role WOULD serve (catalog default).
        canonical = _catalog_by_role_hint(ROLE_ROLE_HINT[role])
        model_id = canonical.id if canonical else ""
    # Metadata: prefer the entry matching the served id; fall back to the role's
    # canonical entry when the operator serves a non-catalog name.
    entry = _catalog_by_id(model_id) or _catalog_by_role_hint(ROLE_ROLE_HINT[role])
    native_context = entry.native_max_model_len if entry else 0
    endpoint = gateway
    if loaded and endpoint:
        ready = ready_signal if ready_signal is not None else loaded
    else:
        ready = False
    return RoleInfo(
        role=role,
        model=model_id,
        runtime=_VLLM_RUNTIME,
        endpoint=endpoint,
        path=ROLE_PATH[role],
        context=_served_context(role, env, native_context),
        quant=entry.quantization if entry else "",
        mtp=bool(entry.speculative_config) if entry else False,
        responsibilities=ROLE_RESPONSIBILITIES[role],
        forbidden_responsibilities=ROLE_FORBIDDEN[role],
        ready=ready,
        loaded=loaded,
    )


def _audio_role(
    role: str,
    model: str,
    runtime: str,
    endpoint: str,
    loaded: bool,
    *,
    ready: bool | None = None,
) -> RoleInfo:
    """Resolve an audio-overlay role (stt/tts). No catalog entry → 0/""/False."""
    if ready is None:
        ready = loaded
    return RoleInfo(
        role=role,
        model=model,
        runtime=runtime,
        endpoint=endpoint,
        path=ROLE_PATH[role],
        context=0,
        quant="",
        mtp=False,
        responsibilities=ROLE_RESPONSIBILITIES[role],
        forbidden_responsibilities=ROLE_FORBIDDEN[role],
        ready=ready,
        loaded=loaded,
    )


def build_role_registry(
    table: RoutingTable,
    server: ServerConfig,
    *,
    env: Mapping[str, str] | None = None,
    gateway_url: str | None = None,
    audio_ready: bool | None = None,
    backend_ready: Mapping[str, bool | None] | None = None,
) -> dict[str, RoleInfo]:
    """Resolve the six first-class roles to live metadata — the #81 contract.

    This is the ONE canonical builder both the CLI (t5) and gateway (t6) call.
    Its inputs are exactly what :func:`lobes.gateway._config.build_config`
    returns (``table``, ``server``), plus the raw ``env`` mapping for the
    served-context overlay below — no new config source is invented.

    :param table: the gateway routing table — its wired :class:`Backend` objects
        tell us which roles are ``loaded`` and each role's served model id.
    :param server: the gateway server config — supplies the audio overlay URL
        (``audio_url``) for stt/tts and, absent ``gateway_url``, the (very
        narrow) ``public_url`` fallback for the four gateway-fronted roles —
        see :func:`_gateway_base_url`.
    :param env: the deployment's environment mapping, consulted ONLY for the
        served ``--max-model-len`` overlay (:data:`ROLE_MAX_MODEL_LEN_ENV`) —
        so ``RoleInfo.context`` reports what the deployment actually SERVES
        (e.g. ``PRIMARY_MAX_MODEL_LEN``), not just the catalog native. ``None``
        (the default) or a mapping missing the relevant key falls back to the
        catalog native — the t4 behaviour is unchanged when ``env`` is omitted.
        Kept separate from ``table``/``server`` (typically built from the SAME
        env) so a caller assembling those by hand isn't forced to also pass it.
    :param gateway_url: the caller-facing gateway base URL for cortex / senses /
        embedder / reranker. When ``None``, it falls back to
        ``server.public_url`` (an operator-declared ``GATEWAY_PUBLIC_URL``) and,
        failing that, to ``""`` — it is NEVER fabricated from
        ``server.host``/``server.port`` (issue #81 t5; those are the gateway's
        INTERNAL listen config, not a client-reachable address — see
        :func:`_gateway_base_url`). Audio roles also use this origin as their
        endpoint when the overlay is wired (issue #87).
    :param audio_ready: optional live-readiness signal for stt/tts (issue #89).
        When not ``None`` it sets the audio roles' ``ready`` (the runtime signal)
        — ``loaded`` stays the config fact ``bool(audio_url)``. When ``None``,
        ``ready`` falls back to ``bool(audio_url)`` (the CLI/back-compat path).
    :param backend_ready: optional live-readiness signal for the four
        gateway-fronted roles (issue #81 t5), keyed by the internal
        :class:`~lobes.gateway._routing.Backend` name (:data:`ROLE_BACKEND`'s
        values — ``"primary"``/``"multimodal"``/``"embed"``/``"rerank"``), one
        tri-state value (``True``/``False``/``None``) per backend — exactly
        the shape :meth:`lobes.gateway._readiness.ReadinessCache.current`
        returns. Mirrors ``audio_ready``'s shape and defaulting: when a role's
        backend has an entry, that entry sets ``ready`` (the runtime signal);
        ``loaded`` stays the config fact "is this backend wired". When
        ``backend_ready`` is ``None`` (the default) — OR a role's backend has
        no entry in it — ``ready`` falls back to ``loaded``, the original t4
        behaviour, so every existing non-HTTP caller (the CLI, this module's
        own offline test suite) is unchanged. ``roles.py`` itself never probes
        anything to produce this signal — it is computed elsewhere (t3's
        :class:`~lobes.gateway._readiness.ReadinessCache`, socket-free to read)
        and handed in, exactly like ``audio_ready``.
    :returns: an ordered ``dict`` keyed by role name with EXACTLY the six roles.
        Every role is always present — an unconfigured/opt-in role (stt/tts with
        ``audio_url`` unset, or an unwired embed/rerank/multimodal backend) is
        returned with ``loaded=False``, never omitted and never raising.

    Readiness (``RoleInfo.ready``) is no longer a bare alias of ``loaded``
    (issue #81 t5 — generalising the stt/tts split from issue #89/#90 to all
    six roles): it reflects ``backend_ready``/``audio_ready`` when the caller
    supplies a live signal, else it falls back to the coarse "configured/wired"
    ``loaded`` proxy. Either way it is CLAMPED, here, to ``False`` whenever a
    role's backend is not wired OR its resolved ``endpoint`` is empty — a
    caller can never fabricate ``ready=True`` for a role with nothing to dial,
    regardless of what signal it passes in. ``roles.py`` stays pure/offline
    either way — it opens no socket to produce or consume this signal; true
    liveness is probed elsewhere (t3's ``ReadinessCache`` /
    ``probe_audio_ready``, issue #89) and handed in as a plain value.
    """
    resolved_env: Mapping[str, str] = env if env is not None else {}
    gateway = (gateway_url or _gateway_base_url(server)).rstrip("/")
    registry: dict[str, RoleInfo] = {}

    for role in ("cortex", "senses", "embedder", "reranker"):
        signal = backend_ready.get(ROLE_BACKEND[role]) if backend_ready is not None else None
        registry[role] = _gateway_role(role, table, gateway, resolved_env, signal)

    audio_url = (server.audio_url or "").rstrip("/")
    audio_configured = bool(audio_url)
    # Audio roles use the gateway origin when the overlay is wired (issue #87),
    # but fall back to empty endpoint when it is not wired.
    audio_endpoint = gateway if audio_configured else ""
    # `loaded` is a config fact — is the audio overlay wired in THIS deployment —
    # kept SEPARATE from `ready`, the runtime signal. `ready` is the gateway's
    # live probe (`audio_ready`) when it supplied one, else it falls back to the
    # configured signal. Keeping them apart means a warming backend reports
    # loaded=True/ready=False (deployed but not yet consumable) instead of
    # masquerading as not-deployed, and an unconfigured overlay never reports a
    # ready role with an empty endpoint.
    #
    # Clamp on `audio_configured` AND `audio_endpoint` so that last invariant
    # holds STRUCTURALLY, not merely by caller discipline: an unconfigured
    # overlay, or one whose endpoint came back empty because no gateway_url/
    # public_url was known (issue #81 t5, criterion 3), is never ready, no
    # matter what `audio_ready` a caller passes. When configured AND dialable,
    # use the live probe signal if one was supplied, else fall back to the
    # configured fact.
    audio_ready_signal = (
        audio_configured
        and bool(audio_endpoint)
        and (audio_ready if audio_ready is not None else True)
    )
    registry["stt"] = _audio_role(
        "stt", _STT_MODEL, _STT_RUNTIME, audio_endpoint, audio_configured, ready=audio_ready_signal
    )
    registry["tts"] = _audio_role(
        "tts", _TTS_MODEL, _TTS_RUNTIME, audio_endpoint, audio_configured, ready=audio_ready_signal
    )
    return registry


def role_registry_from_env(
    env: Mapping[str, str] | None = None,
    *,
    gateway_url: str | None = None,
    audio_ready: bool | None = None,
    backend_ready: Mapping[str, bool | None] | None = None,
) -> dict[str, RoleInfo]:
    """Build the role registry straight from an env mapping.

    A thin convenience over the one canonical builder: it runs the same
    :func:`lobes.gateway._config.build_config` the gateway uses, then delegates
    to :func:`build_role_registry` — threading the SAME resolved env through as
    ``env`` too, so the served-context overlay (``PRIMARY_MAX_MODEL_LEN`` and
    friends, t5) is applied automatically. Lets a host-side caller (the CLI, t5)
    build the registry from a deployment's ``.env`` without assembling a
    ``RoutingTable``/``ServerConfig`` pair by hand. ``env`` defaults to
    ``os.environ`` when omitted (matching :func:`build_config`'s default).
    ``audio_ready``/``backend_ready`` pass straight through to
    :func:`build_role_registry` (both default ``None`` — this offline
    convenience never probes anything itself; a caller with a live signal in
    hand supplies it here exactly as it would to the canonical builder).
    """
    resolved_env = os.environ if env is None else env
    table, server = build_config(resolved_env)
    return build_role_registry(
        table,
        server,
        env=resolved_env,
        gateway_url=gateway_url,
        audio_ready=audio_ready,
        backend_ready=backend_ready,
    )
