"""Tests for the role registry + capability-metadata core (issue #81, task t4).

``lobes.roles`` defines the SIX first-class Colleague-facing roles and resolves
each to live metadata from the gateway config + the catalog:

    cortex   → primary generate backend   (Qwen 27B MTP) — reasoning/authority
    senses   → multimodal generate backend (Gemma 4 12B)  — intake/perception
    embedder → embed pooling backend       (Qwen3-Embedding-0.6B, /v1/embeddings)
    reranker → score/rerank backend        (Qwen3-Reranker-0.6B, /v1/rerank)
    stt      → Parakeet audio sidecar       (/v1/audio/transcriptions) — opt-in
    tts      → Chatterbox audio sidecar     (/v1/audio/speech) — opt-in

The registry is built from what the gateway would read (a ``RoutingTable`` +
``ServerConfig`` from :func:`lobes.gateway._config.build_config`), so the same
one builder feeds both the CLI (t5) and the gateway ``GET /capabilities`` (t6).
"""

from __future__ import annotations

from lobes.catalog import SUPPORTED_MODELS, SupportedModel
from lobes.gateway._config import build_config
from lobes.roles import (
    ROLE_FORBIDDEN,
    ROLE_MAX_MODEL_LEN_ENV,
    ROLE_RESPONSIBILITIES,
    ROLES,
    RoleInfo,
    build_role_registry,
    role_registry_from_env,
)

# ---------------------------------------------------------------------------
# Fixtures — deployment env mappings the gateway would read
# ---------------------------------------------------------------------------

_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_MULTIMODAL_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"

_EXPECTED_ROLES = {"cortex", "senses", "embedder", "reranker", "stt", "tts"}


def _full_env() -> dict[str, str]:
    """A fully-wired six-role fleet: primary + multimodal + embed + rerank + audio."""
    return {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _PRIMARY_ID,
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _MULTIMODAL_ID,
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": _RERANK_ID,
        "AUDIO_URL": "http://realtime:8080",
    }


def _primary_only_env() -> dict[str, str]:
    """A minimal text-only fleet: just the always-present primary, no overlays."""
    return {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _PRIMARY_ID,
    }


def _catalog(model_id: str) -> SupportedModel:
    return next(m for m in SUPPORTED_MODELS if m.id == model_id)


def _registry(env: dict[str, str], *, audio_ready: bool | None = None, **kw) -> dict[str, RoleInfo]:
    table, server = build_config(env)
    # Thread the SAME env through for the served-context overlay (t5) — a test
    # helper omitting this would silently mask the overlay and always assert
    # against the catalog native, which is exactly the t4-vs-t5 regression this
    # module now guards against.
    kw.setdefault("env", env)
    return build_role_registry(table, server, audio_ready=audio_ready, **kw)


# ---------------------------------------------------------------------------
# Acceptance 1 — exactly the six roles, each with the full metadata block
# ---------------------------------------------------------------------------


def test_registry_returns_exactly_the_six_roles() -> None:
    registry = _registry(_full_env())
    assert set(registry) == _EXPECTED_ROLES
    assert len(registry) == 6
    assert set(ROLES) == _EXPECTED_ROLES


def test_every_role_carries_the_full_metadata_block() -> None:
    registry = _registry(_full_env())
    for name, info in registry.items():
        assert isinstance(info, RoleInfo)
        assert info.role == name
        assert isinstance(info.model, str) and info.model  # never empty
        assert isinstance(info.runtime, str) and info.runtime
        assert isinstance(info.endpoint, str)
        assert info.path.startswith("/v1/")
        assert isinstance(info.context, int) and info.context >= 0
        assert isinstance(info.quant, str)
        assert isinstance(info.mtp, bool)
        assert isinstance(info.responsibilities, tuple) and info.responsibilities
        assert isinstance(info.forbidden_responsibilities, tuple)
        assert isinstance(info.loaded, bool)
        # Coarse "configured/wired" readiness — always a present boolean, equal
        # to `loaded` (the CLI and gateway must agree, issue #81). Live health
        # is a later task's concern (t8, `lobes measure`) and lives elsewhere.
        assert isinstance(info.ready, bool)
        assert info.ready == info.loaded


def test_role_info_is_frozen() -> None:
    info = _registry(_full_env())["cortex"]
    try:
        info.model = "nope"  # type: ignore[misc]
    except Exception:  # FrozenInstanceError (a dataclasses error subclass)
        return
    raise AssertionError("RoleInfo must be frozen/immutable")


# ---------------------------------------------------------------------------
# Acceptance 2 — role → backend model + context/quant/mtp from the catalog
# ---------------------------------------------------------------------------


def test_cortex_resolves_to_primary_model_with_catalog_metadata() -> None:
    registry = _registry(_full_env())
    cortex = registry["cortex"]
    entry = _catalog(_PRIMARY_ID)
    assert cortex.model == _PRIMARY_ID
    assert cortex.context == entry.native_max_model_len
    assert cortex.quant == entry.quantization
    assert cortex.mtp is bool(entry.speculative_config)
    assert cortex.mtp is True  # the primary carries an MTP draft head
    assert cortex.path == "/v1/chat/completions"
    assert cortex.loaded is True


def test_senses_resolves_to_multimodal_model_with_catalog_metadata() -> None:
    registry = _registry(_full_env())
    senses = registry["senses"]
    entry = _catalog(_MULTIMODAL_ID)
    assert senses.model == _MULTIMODAL_ID
    assert senses.context == entry.native_max_model_len
    assert senses.quant == entry.quantization
    assert senses.mtp is bool(entry.speculative_config)
    assert senses.path == "/v1/chat/completions"
    assert senses.loaded is True


def test_embedder_and_reranker_resolve_to_pooling_models() -> None:
    registry = _registry(_full_env())
    embedder, reranker = registry["embedder"], registry["reranker"]
    assert embedder.model == _EMBED_ID
    assert embedder.path == "/v1/embeddings"
    assert embedder.context == _catalog(_EMBED_ID).native_max_model_len
    assert embedder.quant == ""  # pooling models carry no quantization
    assert embedder.mtp is False
    assert embedder.loaded is True
    assert reranker.model == _RERANK_ID
    assert reranker.path == "/v1/rerank"
    assert reranker.context == _catalog(_RERANK_ID).native_max_model_len
    assert reranker.mtp is False
    assert reranker.loaded is True


def test_no_hardcoded_model_id_when_operator_renames_served_name() -> None:
    """cortex reports the operator's served name, not a hardcoded catalog id."""
    env = _full_env()
    env["PRIMARY_SERVED_NAME"] = "acme/custom-27b"
    registry = _registry(env)
    assert registry["cortex"].model == "acme/custom-27b"
    assert registry["cortex"].loaded is True


# ---------------------------------------------------------------------------
# Acceptance 3 — unconfigured/opt-in roles are PRESENT with loaded=False
# ---------------------------------------------------------------------------


def test_audio_roles_present_but_unloaded_when_audio_url_unset() -> None:
    registry = _registry(_primary_only_env())
    assert {"stt", "tts"} <= set(registry)  # present, not omitted
    for name in ("stt", "tts"):
        info = registry[name]
        assert info.loaded is False
        assert info.endpoint == ""  # no audio overlay wired
        assert info.context == 0  # audio roles have no token context
        assert info.quant == ""
        assert info.mtp is False
    assert registry["stt"].path == "/v1/audio/transcriptions"
    assert registry["tts"].path == "/v1/audio/speech"


def test_absent_pooling_and_multimodal_roles_present_but_unloaded() -> None:
    registry = _registry(_primary_only_env())
    # All six still present even though only the primary is wired.
    assert set(registry) == _EXPECTED_ROLES
    for name in ("senses", "embedder", "reranker"):
        assert registry[name].loaded is False
    # An unloaded role still names the model it WOULD serve (the catalog default),
    # with that model's catalog metadata — never blank, never an error.
    assert registry["senses"].model == _MULTIMODAL_ID
    assert registry["senses"].context == _catalog(_MULTIMODAL_ID).native_max_model_len
    assert registry["embedder"].model == _EMBED_ID
    assert registry["reranker"].model == _RERANK_ID
    # The primary is always wired → cortex is loaded even on a minimal fleet.
    assert registry["cortex"].loaded is True


def test_audio_roles_loaded_when_audio_url_set() -> None:
    registry = _registry(_full_env())
    for name in ("stt", "tts"):
        info = registry[name]
        assert info.loaded is True
        # Audio roles report the SAME gateway origin as the other roles (issue #87).
        assert info.endpoint == "http://localhost:8000"
        assert info.runtime  # a named audio runtime


# ---------------------------------------------------------------------------
# Acceptance 4 — canonical responsibilities / forbidden lists
# ---------------------------------------------------------------------------


def test_senses_forbidden_responsibilities() -> None:
    senses = _registry(_full_env())["senses"]
    assert senses.forbidden_responsibilities == (
        "final_decision",
        "repo_action",
        "security_decision",
    )


def test_cortex_carries_authoritative_responsibilities_and_no_forbidden() -> None:
    cortex = _registry(_full_env())["cortex"]
    assert "final_authority" in cortex.responsibilities
    assert "reasoning" in cortex.responsibilities
    assert "code_repo_actions" in cortex.responsibilities
    assert cortex.forbidden_responsibilities == ()  # the final authority


def test_static_responsibility_maps_cover_all_six_roles() -> None:
    assert set(ROLE_RESPONSIBILITIES) == _EXPECTED_ROLES
    assert set(ROLE_FORBIDDEN) == _EXPECTED_ROLES
    assert ROLE_RESPONSIBILITIES["stt"] == ("transcribe", "audio_input_to_text")
    assert ROLE_RESPONSIBILITIES["tts"] == ("speech_output", "synthesize")
    assert ROLE_RESPONSIBILITIES["embedder"] == ("vectorization", "memory_retrieval_input")
    assert ROLE_RESPONSIBILITIES["reranker"] == ("retrieval_ordering", "relevance_refinement")


# ---------------------------------------------------------------------------
# Endpoint resolution + builder ergonomics
# ---------------------------------------------------------------------------


def test_gateway_roles_endpoint_defaults_to_derived_gateway_url() -> None:
    """host 0.0.0.0 (bind wildcard) is normalized to a caller-usable localhost."""
    registry = _registry(_full_env())
    for name in ("cortex", "senses", "embedder", "reranker"):
        assert registry[name].endpoint == "http://localhost:8000"


def test_gateway_roles_endpoint_brackets_ipv6_host() -> None:
    """An IPv6 literal GATEWAY_HOST must be bracketed per RFC 3986 — an
    unbracketed 'http://::1:8000' is not a valid/parseable URL authority
    (the address's own colons collide with the ':<port>' separator)."""
    env = _full_env()
    env["GATEWAY_HOST"] = "::1"
    registry = _registry(env)
    for name in ("cortex", "senses", "embedder", "reranker"):
        assert registry[name].endpoint == "http://[::1]:8000"


def test_gateway_roles_endpoint_leaves_ipv4_and_hostnames_unbracketed() -> None:
    """IPv4 literals and hostnames carry no colon — bracketing must be scoped
    to IPv6 only, never applied to these."""
    for host, expected in (
        ("127.0.0.1", "http://127.0.0.1:8000"),
        ("gateway.internal", "http://gateway.internal:8000"),
    ):
        env = _full_env()
        env["GATEWAY_HOST"] = host
        registry = _registry(env)
        assert registry["cortex"].endpoint == expected


def test_explicit_gateway_url_applies_to_all_roles_including_audio() -> None:
    """Audio roles now also use the gateway origin (issue #87)."""
    registry = _registry(_full_env(), gateway_url="https://tunnel.example/")
    assert registry["cortex"].endpoint == "https://tunnel.example"  # trailing slash trimmed
    assert registry["embedder"].endpoint == "https://tunnel.example"
    # Audio roles also use the gateway origin, not the internal audio_url.
    assert registry["stt"].endpoint == "https://tunnel.example"
    assert registry["tts"].endpoint == "https://tunnel.example"


def test_role_registry_from_env_matches_manual_build() -> None:
    # Include a served-context override so this also proves role_registry_from_env
    # THREADS the same env through to build_role_registry's overlay (t5) — not
    # just to build_config.
    env = _full_env()
    env["PRIMARY_MAX_MODEL_LEN"] = "131072"
    from_env = role_registry_from_env(env)
    table, server = build_config(env)
    manual = build_role_registry(table, server, env=env)
    assert from_env == manual
    assert set(from_env) == _EXPECTED_ROLES
    assert from_env["cortex"].context == 131072


# ---------------------------------------------------------------------------
# Served-context overlay (issue #81, task t5) — context reports what the
# deployment ACTUALLY SERVES (--max-model-len), not just the catalog native.
# ---------------------------------------------------------------------------


def test_served_context_overlay_from_deployment_env() -> None:
    """The fleet env's own defaults (env.example) diverge from catalog native for
    every gateway-fronted role — a real regression test, not a contrived one."""
    env = _full_env()
    env.update(
        {
            "PRIMARY_MAX_MODEL_LEN": "131072",
            "MULTIMODAL_MAX_MODEL_LEN": "32768",
            "EMBED_MAX_MODEL_LEN": "8192",
            "RERANK_MAX_MODEL_LEN": "8192",
        }
    )
    registry = _registry(env)
    assert registry["cortex"].context == 131072
    assert registry["senses"].context == 32768
    assert registry["embedder"].context == 8192
    assert registry["reranker"].context == 8192
    # Catalog native would have been different for cortex/senses — prove the
    # overlay actually overrides it, not just coincides with it.
    assert registry["cortex"].context != _catalog(_PRIMARY_ID).native_max_model_len
    assert registry["senses"].context != _catalog(_MULTIMODAL_ID).native_max_model_len
    # Audio roles carry no token context, overlay or not.
    assert registry["stt"].context == 0
    assert registry["tts"].context == 0


def test_served_context_falls_back_to_catalog_native_when_unset() -> None:
    """No *_MAX_MODEL_LEN in the env → context is the catalog native — the t4
    contract is preserved when a deployment doesn't set the overlay."""
    registry = _registry(_full_env())
    assert registry["cortex"].context == _catalog(_PRIMARY_ID).native_max_model_len
    assert registry["senses"].context == _catalog(_MULTIMODAL_ID).native_max_model_len
    assert registry["embedder"].context == _catalog(_EMBED_ID).native_max_model_len
    assert registry["reranker"].context == _catalog(_RERANK_ID).native_max_model_len


def test_served_context_ignores_malformed_override() -> None:
    """A non-numeric override degrades to the catalog native rather than raising."""
    env = _full_env()
    env["PRIMARY_MAX_MODEL_LEN"] = "not-a-number"
    registry = _registry(env)
    assert registry["cortex"].context == _catalog(_PRIMARY_ID).native_max_model_len


def test_served_context_ignores_blank_override() -> None:
    """An empty override (KEY=) falls back to native, mirroring _env.read_env's
    ``${v:-default}`` semantics elsewhere in this CLI."""
    env = _full_env()
    env["MULTIMODAL_MAX_MODEL_LEN"] = ""
    registry = _registry(env)
    assert registry["senses"].context == _catalog(_MULTIMODAL_ID).native_max_model_len


def test_served_context_env_map_covers_only_gateway_fronted_roles() -> None:
    """Audio roles (stt/tts) have no *_MAX_MODEL_LEN entry — they carry no token
    context regardless of any deployment env (see _audio_role)."""
    assert set(ROLE_MAX_MODEL_LEN_ENV) == {"cortex", "senses", "embedder", "reranker"}
    assert "stt" not in ROLE_MAX_MODEL_LEN_ENV
    assert "tts" not in ROLE_MAX_MODEL_LEN_ENV


def test_role_registry_from_env_applies_overlay_via_os_environ(monkeypatch) -> None:
    """role_registry_from_env(None) resolves os.environ for BOTH build_config and
    the overlay — a deployment's real process env (e.g. inside the gateway
    container, t6) drives context without the caller assembling a dict by hand."""
    monkeypatch.setenv("PRIMARY_URL", "http://vllm-primary:8000")
    monkeypatch.setenv("PRIMARY_SERVED_NAME", _PRIMARY_ID)
    monkeypatch.setenv("PRIMARY_MAX_MODEL_LEN", "131072")
    registry = role_registry_from_env()
    assert registry["cortex"].context == 131072


# ---------------------------------------------------------------------------
# Issue #87 — audio roles report the gateway origin, not the internal audio_url
# ---------------------------------------------------------------------------


def test_audio_roles_use_gateway_origin_when_audio_configured() -> None:
    """Audio roles (stt/tts) report the same gateway origin as the other roles
    when the audio overlay is wired (issue #87)."""
    registry = _registry(_full_env())
    # Gateway-fronted roles use the derived gateway URL.
    assert registry["cortex"].endpoint == "http://localhost:8000"
    # Audio roles use the SAME origin.
    assert registry["stt"].endpoint == "http://localhost:8000"
    assert registry["tts"].endpoint == "http://localhost:8000"


def test_audio_roles_empty_endpoint_when_audio_not_configured() -> None:
    """When AUDIO_URL is not set, audio roles stay unloaded with empty endpoint."""
    registry = _registry(_primary_only_env())
    for name in ("stt", "tts"):
        info = registry[name]
        assert info.loaded is False
        assert info.endpoint == ""
        assert info.ready is False


# ---------------------------------------------------------------------------
# Issue #89 — audio_ready override
# ---------------------------------------------------------------------------


def test_audio_ready_false_overrides_ready_not_loaded() -> None:
    """audio_ready=False sets ready=False but keeps loaded=True — the overlay IS
    wired, it is just warming. `loaded` is a config fact, separate from the
    runtime `ready` signal (issue #89)."""
    registry = _registry(_full_env(), audio_ready=False)
    for name in ("stt", "tts"):
        info = registry[name]
        assert info.loaded is True  # configured/deployed...
        assert info.ready is False  # ...but not consumable right now
        assert info.endpoint  # still advertises a dial-able endpoint


def test_audio_ready_never_fabricates_loaded_when_unconfigured() -> None:
    """audio_ready must NOT force loaded/ready True when AUDIO_URL is unset:
    `loaded` is a config fact and `ready` is clamped on it, so an unwired overlay
    never reports a role with an empty endpoint that a client would try to dial,
    and never reports ready=True with no backend (issue #89 review findings)."""
    registry = _registry(_primary_only_env(), audio_ready=True)
    for name in ("stt", "tts"):
        info = registry[name]
        assert info.loaded is False  # not configured in this deployment
        assert info.ready is False  # unconfigured ⇒ never ready, even if a caller
        #                             passes audio_ready=True (Qodo #90 finding)
        assert info.endpoint == ""  # nothing to dial — never ready+empty-endpoint
        # Invariant: a role is never advertised loaded/ready with no endpoint.
        assert not (info.loaded and info.endpoint == "")
        assert not (info.ready and info.endpoint == "")


def test_audio_ready_none_falls_back_to_audio_url() -> None:
    """When audio_ready is None (default), readiness follows bool(audio_url) —
    the back-compat behaviour (issue #89)."""
    # With AUDIO_URL set → loaded/ready True.
    registry = _registry(_full_env())
    for name in ("stt", "tts"):
        assert registry[name].loaded is True
        assert registry[name].ready is True
    # Without AUDIO_URL → loaded/ready False.
    registry = _registry(_primary_only_env())
    for name in ("stt", "tts"):
        assert registry[name].loaded is False
        assert registry[name].ready is False


# ---------------------------------------------------------------------------
# All six roles expose the identical key set
# ---------------------------------------------------------------------------


def test_all_six_roles_expose_identical_key_set() -> None:
    """Every role's asdict keys are identical — no role has extra or missing fields."""
    import dataclasses

    registry = _registry(_full_env())
    first_keys = set(dataclasses.asdict(registry["cortex"]).keys())
    for name in ("senses", "embedder", "reranker", "stt", "tts"):
        assert set(dataclasses.asdict(registry[name]).keys()) == first_keys
