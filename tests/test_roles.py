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


def _registry(env: dict[str, str], **kw) -> dict[str, RoleInfo]:
    table, server = build_config(env)
    # Thread the SAME env through for the served-context overlay (t5) — a test
    # helper omitting this would silently mask the overlay and always assert
    # against the catalog native, which is exactly the t4-vs-t5 regression this
    # module now guards against.
    kw.setdefault("env", env)
    return build_role_registry(table, server, **kw)


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
        # Live health is a later task's concern (t8) — unknown without a probe.
        assert info.ready is None


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
        assert info.endpoint == "http://realtime:8080"
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


def test_explicit_gateway_url_overrides_and_audio_uses_audio_url() -> None:
    registry = _registry(_full_env(), gateway_url="https://tunnel.example/")
    assert registry["cortex"].endpoint == "https://tunnel.example"  # trailing slash trimmed
    assert registry["embedder"].endpoint == "https://tunnel.example"
    # Audio roles ignore gateway_url — they hit the audio overlay directly.
    assert registry["stt"].endpoint == "http://realtime:8080"


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
