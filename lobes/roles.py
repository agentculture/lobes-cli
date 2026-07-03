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
    # Coarse "configured/wired" readiness — always == `loaded` today. This is
    # the same proxy the gateway's GET /capabilities uses (issue #81), so the
    # CLI and gateway agree on one boolean. It is NOT a live health probe;
    # true liveness (did the backend answer a request just now) is
    # `lobes measure`'s job (t8), not this dataclass.
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
    """The gateway's caller-facing base URL derived from its listen config.

    ``ServerConfig.host`` defaults to the wildcard bind ``0.0.0.0`` (usable for
    binding, not as a client target), so it is normalized to ``localhost``.
    Callers that know the real reachable address (a published host port, a tunnel
    URL) should pass an explicit ``gateway_url`` to :func:`build_role_registry`.

    An IPv6 literal host (e.g. ``GATEWAY_HOST=::1``) is bracketed per RFC 3986
    (``http://[::1]:8000``) — an unbracketed IPv6 literal ahead of ``:<port>``
    is not a valid URL authority (the address's own colons collide with the
    port separator). IPv4 literals and hostnames carry no colon and pass
    through unchanged.
    """
    host = server.host
    if host in ("0.0.0.0", "::", ""):  # nosec B104 — a comparison, not a bind
        host = "localhost"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{server.port}"


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
) -> RoleInfo:
    """Resolve a gateway-fronted role (cortex/senses/embedder/reranker)."""
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
    return RoleInfo(
        role=role,
        model=model_id,
        runtime=_VLLM_RUNTIME,
        endpoint=gateway,
        path=ROLE_PATH[role],
        context=_served_context(role, env, native_context),
        quant=entry.quantization if entry else "",
        mtp=bool(entry.speculative_config) if entry else False,
        responsibilities=ROLE_RESPONSIBILITIES[role],
        forbidden_responsibilities=ROLE_FORBIDDEN[role],
        ready=loaded,
        loaded=loaded,
    )


def _audio_role(role: str, model: str, runtime: str, endpoint: str, loaded: bool) -> RoleInfo:
    """Resolve an audio-overlay role (stt/tts). No catalog entry → 0/""/False."""
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
        ready=loaded,
        loaded=loaded,
    )


def build_role_registry(
    table: RoutingTable,
    server: ServerConfig,
    *,
    env: Mapping[str, str] | None = None,
    gateway_url: str | None = None,
) -> dict[str, RoleInfo]:
    """Resolve the six first-class roles to live metadata — the #81 contract.

    This is the ONE canonical builder both the CLI (t5) and gateway (t6) call.
    Its inputs are exactly what :func:`lobes.gateway._config.build_config`
    returns (``table``, ``server``), plus the raw ``env`` mapping for the
    served-context overlay below — no new config source is invented.

    :param table: the gateway routing table — its wired :class:`Backend` objects
        tell us which roles are ``loaded`` and each role's served model id.
    :param server: the gateway server config — supplies the audio overlay URL
        (``audio_url``) for stt/tts and, absent ``gateway_url``, the derived
        gateway base URL for the four gateway-fronted roles.
    :param env: the deployment's environment mapping, consulted ONLY for the
        served ``--max-model-len`` overlay (:data:`ROLE_MAX_MODEL_LEN_ENV`) —
        so ``RoleInfo.context`` reports what the deployment actually SERVES
        (e.g. ``PRIMARY_MAX_MODEL_LEN``), not just the catalog native. ``None``
        (the default) or a mapping missing the relevant key falls back to the
        catalog native — the t4 behaviour is unchanged when ``env`` is omitted.
        Kept separate from ``table``/``server`` (typically built from the SAME
        env) so a caller assembling those by hand isn't forced to also pass it.
    :param gateway_url: the caller-facing gateway base URL for cortex / senses /
        embedder / reranker. When ``None`` it is derived from ``server`` (host
        ``0.0.0.0`` → ``localhost``). Audio roles ignore this — they resolve to
        ``server.audio_url`` (the /v1/audio/* overlay).
    :returns: an ordered ``dict`` keyed by role name with EXACTLY the six roles.
        Every role is always present — an unconfigured/opt-in role (stt/tts with
        ``audio_url`` unset, or an unwired embed/rerank/multimodal backend) is
        returned with ``loaded=False``, never omitted and never raising.

    Readiness (``RoleInfo.ready``) is set to the same coarse "configured/wired"
    proxy as ``loaded`` — the CLI (t5) and the gateway's ``GET /capabilities``
    (t6) must agree on one boolean, so it is computed here, once, for both.
    This is NOT a live health probe (it opens no socket); true liveness is a
    later task's concern (``lobes measure``, t8).
    """
    resolved_env: Mapping[str, str] = env if env is not None else {}
    gateway = (gateway_url or _gateway_base_url(server)).rstrip("/")
    registry: dict[str, RoleInfo] = {}

    for role in ("cortex", "senses", "embedder", "reranker"):
        registry[role] = _gateway_role(role, table, gateway, resolved_env)

    audio_url = (server.audio_url or "").rstrip("/")
    audio_loaded = bool(audio_url)
    registry["stt"] = _audio_role("stt", _STT_MODEL, _STT_RUNTIME, audio_url, audio_loaded)
    registry["tts"] = _audio_role("tts", _TTS_MODEL, _TTS_RUNTIME, audio_url, audio_loaded)
    return registry


def role_registry_from_env(
    env: Mapping[str, str] | None = None,
    *,
    gateway_url: str | None = None,
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
    """
    resolved_env = os.environ if env is None else env
    table, server = build_config(resolved_env)
    return build_role_registry(table, server, env=resolved_env, gateway_url=gateway_url)
