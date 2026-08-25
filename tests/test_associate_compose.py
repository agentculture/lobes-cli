"""Tests for the opt-in ``vllm-associate`` fleet lane (lightning-on-orin plan, t7).

This lane exists for ONE reason: NVIDIA's published Jetson serve recipe for
``nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`` uses eight ``vllm
serve`` flags this repo could not previously express (five Mamba-cache
flags, ``--enable-prefix-caching``, ``--max-num-batched-tokens`` and
``--trust-remote-code``). Per issue #92 (a knob declared but not wired to a
real flag is a dead declaration and is forbidden), every declared knob here
must be proven to reach the rendered ``vllm serve`` argv.

The lane is DELIBERATELY not a Colleague role: no ``lobes/roles.py`` entry,
no ``catalog.py`` tier placement, no gateway wiring. These tests only prove
the compose-level expressiveness, mirroring ``tests/test_worker_compose.py``.
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

# The eight flags this task exists to give a real home. Each must appear,
# rendered, in the vllm-associate command — either as an env-parameterized
# knob (default value baked in) or hardcoded outright (--trust-remote-code).
_EIGHT_FLAGS = (
    "--mamba-backend",
    "--mamba-ssm-cache-dtype",
    "--enable-mamba-cache-stochastic-rounding",
    "--mamba-cache-philox-rounds",
    "--mamba-cache-mode",
    "--enable-prefix-caching",
    "--max-num-batched-tokens",
    "--trust-remote-code",
)

# The knobs actually declared for this lane (name -> the raw ${VAR...}
# substitution text that must appear in the unsubstituted command). Every one
# of these must ALSO be proven to render a real flag via `docker compose
# config` below — that is the "no dead declarations" acceptance bar.
_DECLARED_KNOBS = {
    "ASSOCIATE_IMAGE": "${ASSOCIATE_IMAGE:-",
    "ASSOCIATE_MAMBA_BACKEND": "${ASSOCIATE_MAMBA_BACKEND:-flashinfer}",
    "ASSOCIATE_MAMBA_SSM_CACHE_DTYPE": "${ASSOCIATE_MAMBA_SSM_CACHE_DTYPE:-float16}",
    "ASSOCIATE_MAMBA_CACHE_STOCHASTIC_ROUNDING": (
        "${ASSOCIATE_MAMBA_CACHE_STOCHASTIC_ROUNDING:---enable-mamba-cache-stochastic-rounding}"
    ),
    "ASSOCIATE_MAMBA_CACHE_PHILOX_ROUNDS": "${ASSOCIATE_MAMBA_CACHE_PHILOX_ROUNDS:-5}",
    "ASSOCIATE_MAMBA_CACHE_MODE": "${ASSOCIATE_MAMBA_CACHE_MODE:-align}",
    "ASSOCIATE_PREFIX_CACHING": "${ASSOCIATE_PREFIX_CACHING:---enable-prefix-caching}",
    "ASSOCIATE_MAX_NUM_BATCHED_TOKENS": "${ASSOCIATE_MAX_NUM_BATCHED_TOKENS:-16384}",
}


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


def _associate_command() -> list[str]:
    """The RAW (unsubstituted) vllm-associate command, tokenized.

    ``command:`` is a shell-lexed STRING (same mechanism as vllm-worker /
    vllm-primary / vllm-multimodal — see tests/test_senses_speculative_config.py
    for the substitution + shell-lexing proof): none of ``$``, ``{``, ``}``
    are shell metacharacters, so ``shlex.split`` on the raw text still
    tokenizes each ``${VAR:-default}`` as one token.
    """
    svc = _load_fleet()["services"]["vllm-associate"]
    command = svc["command"]
    if isinstance(command, str):
        return shlex.split(command)
    return [str(tok) for tok in command]


class TestAssociateServiceExists:
    def test_vllm_associate_service_present(self) -> None:
        assert "vllm-associate" in _load_fleet()["services"]

    def test_container_name(self) -> None:
        svc = _load_fleet()["services"]["vllm-associate"]
        assert svc.get("container_name") == "model-gear-vllm-associate"

    def test_profile_gated_on_associate(self) -> None:
        svc = _load_fleet()["services"]["vllm-associate"]
        assert svc.get("profiles") == [
            "associate"
        ], "vllm-associate must be gated behind the 'associate' profile"

    def test_no_host_port_only_expose(self) -> None:
        svc = _load_fleet()["services"]["vllm-associate"]
        assert "ports" not in svc, "vllm-associate must NOT publish a host port"
        assert "8000" in [str(p) for p in svc.get("expose", [])]

    def test_command_is_a_shell_lexed_string_not_a_list(self) -> None:
        # A YAML list command cannot conditionally omit an empty-substituted
        # token (see test_senses_speculative_config.py's docstring) — the
        # ${ASSOCIATE_SPECULATIVE_CONFIG-} off-switch and the boolean-flag
        # knobs below both need the string form.
        svc = _load_fleet()["services"]["vllm-associate"]
        assert isinstance(svc["command"], str)


class TestAssociateImageOverride:
    def test_image_uses_associate_image_var(self) -> None:
        text = _FLEET_COMPOSE.read_text(encoding="utf-8")
        assert "${ASSOCIATE_IMAGE" in text, "ASSOCIATE_IMAGE override var not found"

    def test_image_defaults_to_nightly_digest(self) -> None:
        svc = _load_fleet()["services"]["vllm-associate"]
        image: str = svc["image"]
        assert image.startswith("${ASSOCIATE_IMAGE")
        assert _NIGHTLY_VLLM_IMAGE in image


class TestEightFlagsReachTheCommand:
    """Every one of the eight previously-unexpressible flags must appear,
    with its default value, in the UNSUBSTITUTED command text."""

    @pytest.mark.parametrize("flag", _EIGHT_FLAGS)
    def test_flag_present_in_raw_command(self, flag: str) -> None:
        text = _FLEET_COMPOSE.read_text(encoding="utf-8")
        # Scope the search to the vllm-associate block only, so a coincidental
        # substring match in an unrelated lane can't pass this test.
        start = text.index("vllm-associate:")
        end = text.index("\n  gateway:", start)
        block = text[start:end]
        assert flag in block, f"{flag} does not appear in the vllm-associate service block"

    def test_trust_remote_code_is_hardcoded_not_a_knob(self) -> None:
        # Deliberately NOT a knob (recorded reason): every other generate lane
        # in this file hardcodes --trust-remote-code unconditionally as a
        # fixed security posture, not a per-deployment tuning choice.
        cmd = _associate_command()
        assert "--trust-remote-code" in cmd
        assert not any("ASSOCIATE" in c and "TRUST" in c for c in cmd)


class TestDeclaredKnobsRenderNoDeadDeclarations:
    """#92: a declared knob must reach a real flag. Prove each one renders,
    both at its default and when overridden, via `docker compose config`."""

    @pytest.mark.parametrize("knob, raw_text", _DECLARED_KNOBS.items())
    def test_knob_appears_in_raw_command(self, knob: str, raw_text: str) -> None:
        text = _FLEET_COMPOSE.read_text(encoding="utf-8")
        start = text.index("vllm-associate:")
        end = text.index("\n  gateway:", start)
        block = text[start:end]
        assert raw_text in block or f"${{{knob}" in block, (
            f"{knob} is declared but its substitution text was not found in "
            "the vllm-associate block — dead declaration"
        )

    @pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
    def test_defaults_render_via_docker_compose_config(self) -> None:
        out = _compose_config({"COMPOSE_PROFILES": "associate"})
        rendered = yaml.safe_load(out)
        cmd = [str(c) for c in rendered["services"]["vllm-associate"]["command"]]
        assert "--mamba-backend=flashinfer" in cmd
        assert "--mamba-ssm-cache-dtype=float16" in cmd
        assert "--enable-mamba-cache-stochastic-rounding" in cmd
        assert "--mamba-cache-philox-rounds=5" in cmd
        assert "--mamba-cache-mode=align" in cmd
        assert "--enable-prefix-caching" in cmd
        assert "--max-num-batched-tokens=16384" in cmd
        assert "--trust-remote-code" in cmd

    @pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
    def test_associate_image_override_reaches_the_image_field(self) -> None:
        out = _compose_config(
            {"COMPOSE_PROFILES": "associate", "ASSOCIATE_IMAGE": "myrepo/lightning:pinned"}
        )
        rendered = yaml.safe_load(out)
        assert rendered["services"]["vllm-associate"]["image"] == "myrepo/lightning:pinned"

    @pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
    def test_overrides_reach_the_rendered_argv(self) -> None:
        out = _compose_config(
            {
                "COMPOSE_PROFILES": "associate",
                "ASSOCIATE_MAMBA_BACKEND": "triton",
                "ASSOCIATE_MAMBA_SSM_CACHE_DTYPE": "bfloat16",
                "ASSOCIATE_MAMBA_CACHE_STOCHASTIC_ROUNDING": (
                    "--no-enable-mamba-cache-stochastic-rounding"
                ),
                "ASSOCIATE_MAMBA_CACHE_PHILOX_ROUNDS": "7",
                "ASSOCIATE_MAMBA_CACHE_MODE": "pad",
                "ASSOCIATE_PREFIX_CACHING": "--no-enable-prefix-caching",
                "ASSOCIATE_MAX_NUM_BATCHED_TOKENS": "8192",
            }
        )
        rendered = yaml.safe_load(out)
        cmd = [str(c) for c in rendered["services"]["vllm-associate"]["command"]]
        assert "--mamba-backend=triton" in cmd
        assert "--mamba-ssm-cache-dtype=bfloat16" in cmd
        assert "--no-enable-mamba-cache-stochastic-rounding" in cmd
        assert "--enable-mamba-cache-stochastic-rounding" not in cmd
        assert "--mamba-cache-philox-rounds=7" in cmd
        assert "--mamba-cache-mode=pad" in cmd
        assert "--no-enable-prefix-caching" in cmd
        assert "--enable-prefix-caching" not in cmd
        assert "--max-num-batched-tokens=8192" in cmd

    @pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
    def test_without_profile_service_absent(self) -> None:
        out = _compose_config({})
        rendered = yaml.safe_load(out)
        assert (
            "vllm-associate" not in rendered["services"]
        ), "vllm-associate must NOT start without COMPOSE_PROFILES=associate"


class TestAssociateLaneIsWiredToTheRoleSystem:
    """t7 shipped the LANE with the role system deliberately untouched; t6
    (this plan's tenth-role task) connected the two. These assertions are the
    INVERSE of t7's scope guard, and they are what keeps the lane from being a
    container nothing can address."""

    def test_gateway_environment_passes_the_associate_prefix_through(self) -> None:
        svc = _load_fleet()["services"]["gateway"]
        env: list[str] = svc.get("environment", [])
        keys = {e.split("=", 1)[0] for e in env if "=" in e}
        # The wiring key plus every channel the other nine role prefixes carry.
        for key in (
            "ASSOCIATE_BASE_URL",
            "ASSOCIATE_SERVED_NAME",
            "ASSOCIATE_FEASIBLE",
            "ASSOCIATE_MAX_MODEL_LEN",
            "ASSOCIATE_PEER_ORIGIN",
            "ASSOCIATE_PEER_PROXY",
            "ASSOCIATE_PEER_API_KEY",
            "ASSOCIATE_PEER_ORIGINS",
            "ASSOCIATE_PEER_API_KEYS",
        ):
            assert key in keys, f"gateway must pass {key} through"

    def test_associate_base_url_defaults_empty_so_the_backend_stays_unwired(self) -> None:
        # The opt-in contract: no ASSOCIATE_BASE_URL in .env => no backend =>
        # `model=associate` 404s role_infeasible. A non-empty default here
        # would silently wire a lane no card has budgeted.
        svc = _load_fleet()["services"]["gateway"]
        env: list[str] = svc.get("environment", [])
        line = next(e for e in env if e.startswith("ASSOCIATE_BASE_URL="))
        assert line == "ASSOCIATE_BASE_URL=${ASSOCIATE_BASE_URL:-}"

    def test_roles_module_declares_the_associate_role(self) -> None:
        from lobes.roles import ROLE_BACKEND, ROLES

        assert "associate" in ROLES
        assert ROLE_BACKEND["associate"] == "associate"


class TestEnvExampleDocumentsAssociateKnobs:
    def test_every_associate_knob_is_documented(self) -> None:
        text = _FLEET_ENV.read_text(encoding="utf-8")
        for knob in (
            "ASSOCIATE_IMAGE",
            "ASSOCIATE_MODEL",
            "ASSOCIATE_KV_CACHE_DTYPE",
            "ASSOCIATE_MAMBA_BACKEND",
            "ASSOCIATE_MAMBA_SSM_CACHE_DTYPE",
            "ASSOCIATE_MAMBA_CACHE_STOCHASTIC_ROUNDING",
            "ASSOCIATE_MAMBA_CACHE_PHILOX_ROUNDS",
            "ASSOCIATE_MAMBA_CACHE_MODE",
            "ASSOCIATE_PREFIX_CACHING",
            "ASSOCIATE_MAX_NUM_BATCHED_TOKENS",
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
