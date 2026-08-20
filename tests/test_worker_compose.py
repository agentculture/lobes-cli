"""Tests for the opt-in-core ``vllm-worker`` fleet service (thor-worker-lobe plan, t4).

``worker`` is the eighth Colleague role: ``unsloth/Qwen3.6-35B-A3B-NVFP4`` — a
Qwen3.5 multimodal (image+video, NO audio) MoE with a self-hosted MTP draft.
It mirrors the opt-in-core ``vllm-muse`` precedent (profile-gated so a plain
``docker compose up`` never starts it; gateway backend wired only when
``WORKER_BASE_URL`` is set) but rides the SAME Qwen/vLLM nightly lane the
primary/embed/rerank gears use — NOT the Gemma ``lobes/vllm-gemma4:local``
image — and serves MULTIMODAL (so NO ``--language-model-only``).

These assertions read the SHIPPED template (PyYAML expands nothing, so the raw
``${VAR}`` placeholders stay visible), plus an opt-in ``docker compose config``
render behind an availability skip.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"
_FLEET_ENV = _TEMPLATES / "fleet" / "env.example"

_NIGHTLY_VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:" "8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695"
)
_GEMMA_LOCAL_TAG = "lobes/vllm-gemma4:local"


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _worker_command() -> list[str]:
    """The RAW (unsubstituted) vllm-worker command, tokenized.

    ``command:`` is now a shell-lexed STRING (spec-knobs task, the same
    off-switch mechanism as the senses lane), not a YAML list — see
    ``tests/test_senses_speculative_config.py`` for the substitution +
    shell-lexing proof. ``shlex.split`` on the raw, unsubstituted text still
    tokenizes correctly here: none of ``$``, ``{``, ``}`` are shell metacharacters,
    so a bare ``${VAR:-default}`` stays one token, and the only quoted segment
    (the speculative-config default) round-trips because its wrapping single
    quotes are real quote characters in the source.
    """
    svc = _load_fleet()["services"]["vllm-worker"]
    command = svc["command"]
    if isinstance(command, str):
        return shlex.split(command)
    return [str(tok) for tok in command]


class TestWorkerServiceExists:
    def test_vllm_worker_service_present(self) -> None:
        assert "vllm-worker" in _load_fleet()["services"]

    def test_container_name(self) -> None:
        svc = _load_fleet()["services"]["vllm-worker"]
        assert svc.get("container_name") == "model-gear-vllm-worker"

    def test_profile_gated_on_worker(self) -> None:
        svc = _load_fleet()["services"]["vllm-worker"]
        assert svc.get("profiles") == [
            "worker"
        ], "vllm-worker must be gated behind the 'worker' profile"

    def test_no_host_port_only_expose(self) -> None:
        svc = _load_fleet()["services"]["vllm-worker"]
        assert "ports" not in svc, "vllm-worker must NOT publish a host port"
        assert "8000" in [str(p) for p in svc.get("expose", [])]

    def test_depends_on_pooling_gears_healthy(self) -> None:
        # Mirror vllm-muse's Thor first-boot ordering mitigation.
        svc = _load_fleet()["services"]["vllm-worker"]
        deps = svc.get("depends_on", {})
        assert "vllm-embed" in deps
        assert "vllm-rerank" in deps
        assert deps["vllm-embed"]["condition"] == "service_healthy"
        assert deps["vllm-rerank"]["condition"] == "service_healthy"


class TestWorkerImageIsQwenLane:
    def test_image_uses_worker_image_var(self) -> None:
        text = _FLEET_COMPOSE.read_text(encoding="utf-8")
        assert "${WORKER_IMAGE" in text, "WORKER_IMAGE override var not found"

    def test_image_defaults_to_nightly_digest_not_gemma(self) -> None:
        svc = _load_fleet()["services"]["vllm-worker"]
        image: str = svc["image"]
        assert image.startswith("${WORKER_IMAGE")
        # Worker is a Qwen3.5 MoE, not a Gemma gear -> rides the Qwen nightly lane.
        assert _NIGHTLY_VLLM_IMAGE in image
        assert _GEMMA_LOCAL_TAG not in image

    def test_no_build_block(self) -> None:
        # Rides the pre-built nightly digest directly (like primary/embed/rerank),
        # not a custom Dockerfile.vllm-gemma4 build.
        svc = _load_fleet()["services"]["vllm-worker"]
        assert "build" not in svc


class TestWorkerCommand:
    def test_serves_worker_model_default(self) -> None:
        # Default is the catalog's role_hint=worker model since 2026-08-20 (d1):
        # Lightning, validated on the Spark; the demoted Qwen candidate is an
        # override documented in env.example.
        cmd = _worker_command()
        assert "${WORKER_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}" in cmd

    def test_served_model_name(self) -> None:
        cmd = _worker_command()
        assert any(c.startswith("--served-model-name=${WORKER_SERVED_NAME:-") for c in cmd)

    def test_quantization_modelopt_default(self) -> None:
        # Lightning's own quant_method (config.json 2026-08-20); the Qwen
        # candidate overrides to compressed-tensors.
        cmd = _worker_command()
        assert "--quantization=${WORKER_QUANTIZATION:-modelopt}" in cmd

    def test_max_model_len_knob(self) -> None:
        cmd = _worker_command()
        assert any(c.startswith("--max-model-len=${WORKER_MAX_MODEL_LEN:-") for c in cmd)

    def test_gpu_mem_util_knob(self) -> None:
        cmd = _worker_command()
        assert any(c.startswith("--gpu-memory-utilization=${WORKER_GPU_MEM_UTIL:-") for c in cmd)

    def test_moe_backend_not_forced_so_vllm_auto_selects(self) -> None:
        # --moe-backend is DELIBERATELY not forced. Measured on Thor sm_110
        # (docs/evidence/2026-07-31-accept-worker-thor.txt): every forced NVFP4
        # MoE backend was refused (flashinfer_* lack sm_110 kernels;
        # marlin/triton reject the mixed quantized-main/unquantized-MTP experts).
        # vLLM auto-selects a working kernel per path when the flag is absent.
        cmd = _worker_command()
        assert not any(c.startswith("--moe-backend") for c in cmd)

    def test_speculative_default_off(self) -> None:
        # DEFAULT-OFF since 2026-08-20: the Lightning worker's MTP/DSpark is
        # card-declared but unmeasured on this fleet — the Spark validation ran
        # plain decode per #187. The knob's default carries NO flag; enabling
        # is an explicit env.example-documented override.
        cmd = _worker_command()
        assert "${WORKER_SPECULATIVE_CONFIG-}" in cmd
        assert not any("--speculative-config" in c for c in cmd)

    def test_parser_pair_default(self) -> None:
        # --tool-call-parser stays hardcoded qwen3_coder: BOTH the shipped
        # WORKER_MODEL default (the demoted unsloth/Qwen3.6-35B-A3B-NVFP4
        # candidate) and the catalog's current role_hint=worker model
        # (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) agree on it —
        # only the reasoning parser differs between the two (WORKER_REASONING_PARSER).
        cmd = _worker_command()
        assert "--enable-auto-tool-choice" in cmd
        assert "--tool-call-parser=qwen3_coder" in cmd
        assert "--reasoning-parser=${WORKER_REASONING_PARSER:-nemotron_v3}" in cmd

    def test_trust_remote_code(self) -> None:
        assert "--trust-remote-code" in _worker_command()

    def test_no_language_model_only_worker_serves_vision(self) -> None:
        # Worker keeps its ViT vision tower (image+video). Must NOT drop it.
        assert "--language-model-only" not in _worker_command()


class TestWorkerReasoningParserKnob:
    """WORKER_REASONING_PARSER (spec-knobs task): the worker lane's
    --reasoning-parser is now env-parameterized, defaulting to nemotron_v3 —
    the CURRENT catalog role_hint=worker model
    (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, validated live on the
    Spark 2026-08-20 — docs/evidence/2026-08-20-accept-worker-hand-spark.txt).
    """

    def test_default_is_nemotron_v3(self) -> None:
        cmd = _worker_command()
        assert "--reasoning-parser=${WORKER_REASONING_PARSER:-nemotron_v3}" in cmd

    def test_no_bare_hardcoded_qwen3_reasoning_parser_remains(self) -> None:
        cmd = _worker_command()
        assert "--reasoning-parser=qwen3" not in cmd

    @pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
    def test_override_qwen3_for_the_demoted_qwen_candidate(self) -> None:
        # The shipped WORKER_MODEL default is still the demoted Qwen candidate,
        # which needs qwen3, not nemotron_v3 — env.example documents this override.
        out = _compose_config({"COMPOSE_PROFILES": "worker", "WORKER_REASONING_PARSER": "qwen3"})
        rendered = yaml.safe_load(out)
        cmd = [str(c) for c in rendered["services"]["vllm-worker"]["command"]]
        assert "--reasoning-parser=qwen3" in cmd


class TestWorkerSpeculativeConfigKnob:
    """WORKER_SPECULATIVE_CONFIG (spec-knobs task): the same MTP off-switch
    mechanism as MULTIMODAL_SPECULATIVE_CONFIG / PRIMARY_SPECULATIVE_CONFIG —
    a set-but-empty value drops the flag entirely rather than blanking it. The
    recorded need: the Spark box serving Lightning as worker removed the MTP
    line by hand (plain decode first, #187) before this knob existed — see
    docs/evidence/2026-08-20-accept-worker-hand-spark.txt.
    """

    def test_knob_uses_the_unset_only_default_operator(self) -> None:
        text = _FLEET_COMPOSE.read_text(encoding="utf-8")
        assert "${WORKER_SPECULATIVE_CONFIG-" in text
        assert "${WORKER_SPECULATIVE_CONFIG:-" not in text, (
            "WORKER_SPECULATIVE_CONFIG must NOT use ${VAR:-default}: an empty "
            "value would fall back to the default and the off-switch would "
            "never engage"
        )

    @pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
    def test_empty_value_removes_the_flag_entirely(self) -> None:
        out = _compose_config({"COMPOSE_PROFILES": "worker", "WORKER_SPECULATIVE_CONFIG": ""})
        rendered = yaml.safe_load(out)
        cmd = [str(c) for c in rendered["services"]["vllm-worker"]["command"]]
        assert not any(c.startswith("--speculative-config") for c in cmd)
        assert "" not in cmd

    @pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
    def test_unset_renders_no_speculative_flag(self) -> None:
        # Default-off (2026-08-20): unset renders NO --speculative-config in
        # the argv at all — the same absence set-but-empty produces.
        out = _compose_config({"COMPOSE_PROFILES": "worker"})
        rendered = yaml.safe_load(out)
        cmd = [str(c) for c in rendered["services"]["vllm-worker"]["command"]]
        assert not any(c.startswith("--speculative-config") for c in cmd)


class TestGatewayWiresWorker:
    def test_gateway_environment_passes_worker_keys(self) -> None:
        svc = _load_fleet()["services"]["gateway"]
        env: list[str] = svc["environment"]
        keys = {e.split("=", 1)[0] for e in env if "=" in e}
        for k in (
            "WORKER_BASE_URL",
            "WORKER_SERVED_NAME",
            "WORKER_MAX_MODEL_LEN",
            "WORKER_FEASIBLE",
            "WORKER_PEER_ORIGIN",
            "WORKER_PEER_PROXY",
            "WORKER_PEER_API_KEY",
        ):
            assert k in keys, f"gateway environment must pass through {k}"

    def test_gateway_environment_passes_hand_peer_keys(self) -> None:
        # hand rides the same feasibility/peer channels since 2026-08-20 (the
        # d1 reversal of NEVER_PROXIED_BACKENDS) — the gateway container must
        # actually RECEIVE the knobs or the reversal is inert in deployments
        # (qodo PR #190 finding 2).
        svc = _load_fleet()["services"]["gateway"]
        env: list[str] = svc["environment"]
        keys = {e.split("=", 1)[0] for e in env if "=" in e}
        for k in (
            "HAND_FEASIBLE",
            "HAND_PEER_ORIGIN",
            "HAND_PEER_PROXY",
            "HAND_PEER_API_KEY",
        ):
            assert k in keys, f"gateway environment must pass {k}"


class TestEnvExampleDocumentsWorkerKnobs:
    def test_every_worker_knob_is_documented(self) -> None:
        text = _FLEET_ENV.read_text(encoding="utf-8")
        for knob in (
            "WORKER_MODEL",
            "WORKER_SERVED_NAME",
            "WORKER_BASE_URL",
            "WORKER_MAX_MODEL_LEN",
            "WORKER_GPU_MEM_UTIL",
            "WORKER_QUANTIZATION",
            "WORKER_MOE_BACKEND",
            "WORKER_ATTENTION_BACKEND",
            "WORKER_IMAGE",
            "WORKER_SPECULATIVE_CONFIG",
            "WORKER_REASONING_PARSER",
            "WORKER_PEER_ORIGIN",
            "WORKER_PEER_PROXY",
            "WORKER_PEER_API_KEY",
        ):
            assert knob in text, f"env.example must document {knob}"


def _compose_config(env_extra: dict[str, str]) -> str:
    env = {"PATH": __import__("os").environ.get("PATH", "")}
    env.update(env_extra)
    proc = subprocess.run(
        ["docker", "compose", "-f", str(_FLEET_COMPOSE), "config"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_FLEET_COMPOSE.parent),
    )
    assert proc.returncode == 0, f"docker compose config failed:\n{proc.stderr}"
    return proc.stdout


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
class TestWorkerComposeConfigRender:
    def test_with_profile_worker_service_resolves_with_flags(self) -> None:
        out = _compose_config({"COMPOSE_PROFILES": "worker"})
        rendered = yaml.safe_load(out)
        assert "vllm-worker" in rendered["services"], "worker profile must resolve vllm-worker"
        svc = rendered["services"]["vllm-worker"]
        cmd = [str(c) for c in svc["command"]]
        assert "--tool-call-parser=qwen3_coder" in cmd
        assert "--reasoning-parser=nemotron_v3" in cmd
        assert "--quantization=modelopt" in cmd
        assert not any("--speculative-config" in c for c in cmd)
        assert "--language-model-only" not in cmd
        assert _NIGHTLY_VLLM_IMAGE in svc["image"]

    def test_without_profile_worker_service_absent(self) -> None:
        out = _compose_config({})
        rendered = yaml.safe_load(out)
        assert (
            "vllm-worker" not in rendered["services"]
        ), "vllm-worker must NOT start without COMPOSE_PROFILES=worker"
