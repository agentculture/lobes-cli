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

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"
_FLEET_ENV = _TEMPLATES / "fleet" / "env.example"

_NIGHTLY_VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:" "7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9"
)
_GEMMA_LOCAL_TAG = "lobes/vllm-gemma4:local"


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _worker_command() -> list[str]:
    svc = _load_fleet()["services"]["vllm-worker"]
    return [str(tok) for tok in svc["command"]]


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
        assert "vllm-embed" in deps and "vllm-rerank" in deps
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
        cmd = _worker_command()
        assert "${WORKER_MODEL:-unsloth/Qwen3.6-35B-A3B-NVFP4}" in cmd

    def test_served_model_name(self) -> None:
        cmd = _worker_command()
        assert any(c.startswith("--served-model-name=${WORKER_SERVED_NAME:-") for c in cmd)

    def test_quantization_compressed_tensors(self) -> None:
        cmd = _worker_command()
        assert "--quantization=${WORKER_QUANTIZATION:-compressed-tensors}" in cmd

    def test_max_model_len_knob(self) -> None:
        cmd = _worker_command()
        assert any(c.startswith("--max-model-len=${WORKER_MAX_MODEL_LEN:-") for c in cmd)

    def test_gpu_mem_util_knob(self) -> None:
        cmd = _worker_command()
        assert any(c.startswith("--gpu-memory-utilization=${WORKER_GPU_MEM_UTIL:-") for c in cmd)

    def test_moe_backend_is_overridable(self) -> None:
        cmd = _worker_command()
        assert any(c.startswith("--moe-backend=${WORKER_MOE_BACKEND:-") for c in cmd)

    def test_self_draft_mtp_no_external_draft_model(self) -> None:
        cmd = _worker_command()
        spec = [c for c in cmd if "--speculative-config" in c]
        assert spec, "worker must declare a --speculative-config (self-draft MTP)"
        blob = spec[0]
        assert '"method": "mtp"' in blob
        assert '"num_speculative_tokens": 2' in blob
        # Self-draft: NO external draft-model key (unlike the Gemma gears).
        assert '"model"' not in blob and "draft_model" not in blob

    def test_parser_pair_qwen(self) -> None:
        cmd = _worker_command()
        assert "--enable-auto-tool-choice" in cmd
        assert "--tool-call-parser=qwen3_coder" in cmd
        assert "--reasoning-parser=qwen3" in cmd

    def test_trust_remote_code(self) -> None:
        assert "--trust-remote-code" in _worker_command()

    def test_no_language_model_only_worker_serves_vision(self) -> None:
        # Worker keeps its ViT vision tower (image+video). Must NOT drop it.
        assert "--language-model-only" not in _worker_command()


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
        assert "--reasoning-parser=qwen3" in cmd
        assert "--quantization=compressed-tensors" in cmd
        assert any("--speculative-config" in c and '"method": "mtp"' in c for c in cmd)
        assert "--language-model-only" not in cmd
        assert _NIGHTLY_VLLM_IMAGE in svc["image"]

    def test_without_profile_worker_service_absent(self) -> None:
        out = _compose_config({})
        rendered = yaml.safe_load(out)
        assert (
            "vllm-worker" not in rendered["services"]
        ), "vllm-worker must NOT start without COMPOSE_PROFILES=worker"
