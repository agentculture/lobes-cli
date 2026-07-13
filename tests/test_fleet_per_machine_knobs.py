"""Per-machine profile knobs (devague plan task t3, issue #109's r6 risk).

Every machine-dependent vLLM flag that used to be hardcoded in the shipped
compose templates must now be env-substituted, with today's exact rendered
behavior as the default (GB10 deployments see zero behavior change). This
guards the four divergences between the packaged fleet template and the
hand-edited /home/thor/.lobes/docker-compose.yml ground truth (a Jetson AGX
Thor, sm_110):

  1. vllm-primary  --kv-cache-dtype       fp8 (shipped default) / auto (Thor)
  2. vllm-embed     --attention-config    auto (shipped default) / TRITON_ATTN (Thor)
  3. vllm-rerank    --attention-config    auto (shipped default) / TRITON_ATTN (Thor)
  4. vllm-rerank    enforce-eager toggle  --no-enforce-eager (shipped) / --enforce-eager (Thor)

"auto" is not a placeholder — vLLM's AttentionConfig.backend field_validator
treats the string "auto" as None, its own automatic-selection sentinel
(vllm/config/attention.py), so defaulting to "auto" is byte-equivalent in
effect to omitting --attention-config entirely (today's shipped behavior).
Likewise --no-enforce-eager is an explicit no-op: vLLM's `enforce_eager: bool`
field gets argparse.BooleanOptionalAction (--enforce-eager / --no-enforce-eager
are both always-valid tokens), and ModelConfig.enforce_eager already defaults
to False, so shipping --no-enforce-eager explicitly changes nothing.

Also guards the explicit DEVIATION from the exported plan: MULTIMODAL_ATTENTION_
BACKEND (the senses/vllm-multimodal gear) must NOT be touched — its removal is
gated on a GB10 live-verification tracked in agentculture/lobes-cli#109.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"
_FLEET_ENV_EXAMPLE = _TEMPLATES / "fleet" / "env.example"
_SINGLE_COMPOSE = _TEMPLATES / "docker-compose.yml"
_SINGLE_ENV_EXAMPLE = _TEMPLATES / "env.example"


def _fleet_compose() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _command(compose: dict, service: str) -> list[str]:
    return compose["services"][service]["command"]


class TestPrimaryKvCacheDtypeIsParameterised:
    def test_kv_cache_dtype_item_uses_env_var_with_fp8_default(self) -> None:
        command = _command(_fleet_compose(), "vllm-primary")
        assert "--kv-cache-dtype=${PRIMARY_KV_CACHE_DTYPE:-fp8}" in command, (
            "vllm-primary --kv-cache-dtype must be env-substituted with the "
            "shipped default fp8 (today's exact behavior) — see #109"
        )

    def test_no_bare_hardcoded_fp8_kv_cache_dtype_remains(self) -> None:
        command = _command(_fleet_compose(), "vllm-primary")
        assert "--kv-cache-dtype=fp8" not in command


class TestEmbedAttentionConfigIsParameterised:
    def test_attention_config_item_defaults_to_auto(self) -> None:
        command = _command(_fleet_compose(), "vllm-embed")
        assert '--attention-config={"backend": "${EMBED_ATTENTION_BACKEND:-auto}"}' in command, (
            "vllm-embed must gain an env-substituted --attention-config "
            "defaulting to 'auto' — byte-equivalent in effect to today's "
            "flagless render (vLLM treats the string 'auto' as None, its own "
            "automatic-selection sentinel)"
        )


class TestRerankAttentionConfigAndEnforceEagerAreParameterised:
    def test_attention_config_item_defaults_to_auto(self) -> None:
        command = _command(_fleet_compose(), "vllm-rerank")
        assert '--attention-config={"backend": "${RERANK_ATTENTION_BACKEND:-auto}"}' in command

    def test_enforce_eager_toggle_defaults_to_explicit_no_op(self) -> None:
        command = _command(_fleet_compose(), "vllm-rerank")
        assert "${RERANK_ENFORCE_EAGER:---no-enforce-eager}" in command, (
            "the enforce-eager toggle must default to the explicit "
            "--no-enforce-eager no-op (BooleanOptionalAction; "
            "ModelConfig.enforce_eager already defaults to False)"
        )

    def test_no_bare_hardcoded_enforce_eager_remains(self) -> None:
        command = _command(_fleet_compose(), "vllm-rerank")
        assert "--enforce-eager" not in command


class TestMultimodalAttentionBackendDeviationIsPreserved:
    """DEVIATION (binding, per the task brief): MULTIMODAL_ATTENTION_BACKEND
    stays exactly as-is — its removal is gated on GB10 live-verification
    (agentculture/lobes-cli#109, risk r6), not this task."""

    def test_multimodal_attention_backend_env_is_unchanged(self) -> None:
        compose = _fleet_compose()
        env = compose["services"]["vllm-multimodal"]["environment"]
        assert "VLLM_ATTENTION_BACKEND=${MULTIMODAL_ATTENTION_BACKEND:-TRITON_ATTN}" in env

    def test_deviation_comment_points_at_109(self) -> None:
        text = _FLEET_COMPOSE.read_text(encoding="utf-8")
        assert "#109" in text


class TestThorReproducingEnvSetIsDocumented:
    """The four new vars must appear in env.example with the shipped default
    baked in, so `lobes init --fleet` scaffolds a .env an operator can then
    edit to reproduce the Thor box."""

    def test_fleet_env_example_documents_all_four_new_vars(self) -> None:
        text = _FLEET_ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "PRIMARY_KV_CACHE_DTYPE=fp8" in text
        assert "EMBED_ATTENTION_BACKEND=auto" in text
        assert "RERANK_ATTENTION_BACKEND=auto" in text
        assert "RERANK_ENFORCE_EAGER=--no-enforce-eager" in text


class TestSingleModelTemplateKvCacheDtypeIsParameterised:
    """The legacy single-model scaffold carries the same hardcoded
    --kv-cache-dtype=fp8 knob; it gets the same treatment."""

    def test_kv_cache_dtype_item_uses_env_var_with_fp8_default(self) -> None:
        compose = yaml.safe_load(_SINGLE_COMPOSE.read_text(encoding="utf-8"))
        command = _command(compose, "vllm")
        assert any(
            item == "--kv-cache-dtype=${VLLM_KV_CACHE_DTYPE:-fp8}" for item in command
        ), "single-model vllm --kv-cache-dtype must be env-substituted with default fp8"

    def test_single_env_example_documents_the_var(self) -> None:
        text = _SINGLE_ENV_EXAMPLE.read_text(encoding="utf-8")
        assert "VLLM_KV_CACHE_DTYPE=fp8" in text
