"""The recipe knobs the ``vllm-worker`` lane can express (worker-recipe-knobs, t3).

Before this task the worker lane emitted exactly five tunable flags
(``--quantization`` / ``--max-model-len`` / ``--gpu-memory-utilization`` /
``${WORKER_SPECULATIVE_CONFIG-}`` / ``--reasoning-parser``) and HARDCODED
``--tool-call-parser=qwen3_coder``. Three knobs the profile layer already
RENDERED — ``WORKER_KV_CACHE_DTYPE`` / ``WORKER_ATTENTION_BACKEND`` /
``WORKER_MAX_NUM_SEQS`` — reached ``.env`` and were read by NOTHING on the
lane; six more (MoE backend, batched-token budget, chunked prefill, async
scheduling, prefix caching, load format) could not be expressed at all, so an
arm that needed one meant hand-editing the packaged compose file (the same
class of drift ``deployment.lock.toml`` exists to catch — see CLAUDE.md's
2026-08-25 Spark incident).

THE MECHANISM (unchanged, and load-bearing): ``command:`` stays ONE shell-lexed
STRING. A YAML ``command:`` LIST cannot omit an item — a variable that
substitutes to "" renders as an empty argv element and ``vllm serve`` exits 2
on it (proved in ``tests/test_senses_speculative_config.py``'s docstring). A
STRING is shell-lexed by Compose AFTER substitution, so an unset variable
leaves NO token. Two idioms carry that here:

* ``${WORKER_X:+--flag=${WORKER_X}}`` — the value-carrying knobs. Unset (or
  empty) renders nothing at all; set renders exactly one argv token. Verified
  against real ``docker compose config`` below, not only simulated.
* ``${WORKER_X-}`` — the boolean-ish knobs, whose ``.env`` value is the FULL
  flag text (``--enable-prefix-caching`` / ``--no-enable-prefix-caching``),
  the same idiom ``RERANK_ENFORCE_EAGER`` and ``ASSOCIATE_PREFIX_CACHING``
  already use.

The acceptance contract these tests pin:

1. every knob added is declared in ``schema.KNOB_NAMES``, mapped in
   ``render._KNOB_ENV_SUFFIX``, substituted in the ``vllm-worker`` command and
   documented in ``env.example`` — no knob renders a key nothing reads;
2. with NONE of the new knobs set the rendered argv is byte-identical to the
   pre-task one (frozen below, captured from ``docker compose config`` on the
   unmodified template);
3. the worker command does NOT pass ``--language-model-only``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from lobes.cli._errors import ModelGearError
from lobes.profiles.render import _KNOB_ENV_SUFFIX, profile_env
from lobes.profiles.schema import KNOB_NAMES, Profile, RoleProfile
from lobes.runtime._lock import lock_keys

_REPO = Path(__file__).resolve().parents[1]
_FLEET = _REPO / "lobes" / "templates" / "fleet"
_FLEET_COMPOSE = _FLEET / "docker-compose.yml"
_FLEET_ENV = _FLEET / "env.example"

#: knob field name -> the ``WORKER_*`` key it renders, for every knob this task
#: made live on the lane (three pre-existing, seven new).
WORKER_RECIPE_KNOBS: dict[str, str] = {
    "kv_cache_dtype": "WORKER_KV_CACHE_DTYPE",
    "attention_backend": "WORKER_ATTENTION_BACKEND",
    "max_num_seqs": "WORKER_MAX_NUM_SEQS",
    "moe_backend": "WORKER_MOE_BACKEND",
    "max_num_batched_tokens": "WORKER_MAX_NUM_BATCHED_TOKENS",
    "chunked_prefill": "WORKER_CHUNKED_PREFILL",
    "async_scheduling": "WORKER_ASYNC_SCHEDULING",
    "prefix_caching": "WORKER_PREFIX_CACHING",
    "load_format": "WORKER_LOAD_FORMAT",
    "tool_call_parser": "WORKER_TOOL_CALL_PARSER",
}

#: The knobs this task ADDS to the vocabulary (the rest already existed).
NEW_KNOBS = (
    "moe_backend",
    "max_num_batched_tokens",
    "chunked_prefill",
    "async_scheduling",
    "prefix_caching",
    "load_format",
    "tool_call_parser",
)

#: The worker lane's argv on the UNMODIFIED template, captured from
#: ``docker compose config --format json`` with ``COMPOSE_PROFILES=worker`` and
#: nothing else set (Compose v2.40.3). Criterion 2's frozen baseline.
BASELINE_ARGV = [
    "vllm",
    "serve",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "--served-model-name=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "--host=0.0.0.0",
    "--port=8000",
    "--quantization=modelopt",
    "--max-model-len=65536",
    "--gpu-memory-utilization=0.30",
    "--enable-auto-tool-choice",
    "--tool-call-parser=qwen3_coder",
    "--reasoning-parser=nemotron_v3",
    "--trust-remote-code",
]


def _compose_argv(env_extra: dict[str, str]) -> list[str]:
    env = {"PATH": os.environ.get("PATH", "")}
    env.update(env_extra)
    proc = subprocess.run(
        ["docker", "compose", "-f", str(_FLEET_COMPOSE), "config"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_FLEET),
    )
    assert proc.returncode == 0, f"docker compose config failed:\n{proc.stderr}"
    rendered = yaml.safe_load(proc.stdout)
    return [str(tok) for tok in rendered["services"]["vllm-worker"]["command"]]


# --- 1. the knob vocabulary -------------------------------------------------


class TestKnobVocabulary:
    def test_every_new_knob_is_declared(self) -> None:
        for knob in NEW_KNOBS:
            assert knob in KNOB_NAMES, f"{knob} must be declared in schema.KNOB_NAMES"

    def test_every_new_knob_is_a_role_profile_field(self) -> None:
        rp = RoleProfile()
        for knob in NEW_KNOBS:
            assert getattr(rp, knob) is None, f"{knob} must default to None (no opinion)"

    def test_every_knob_maps_to_its_env_suffix(self) -> None:
        for knob, key in WORKER_RECIPE_KNOBS.items():
            assert knob in _KNOB_ENV_SUFFIX, f"{knob} must be mapped in render._KNOB_ENV_SUFFIX"
            assert key == f"WORKER_{_KNOB_ENV_SUFFIX[knob]}"

    def test_wrong_type_is_a_load_error(self) -> None:
        for knob, bad in (
            ("moe_backend", 7),
            ("max_num_batched_tokens", "16384"),
            ("chunked_prefill", "true"),
            ("load_format", 1.5),
            ("tool_call_parser", True),
        ):
            with pytest.raises(ModelGearError):
                RoleProfile.from_dict("worker", {knob: bad})


# --- 2. a knob may only be declared for a lane that expands it ---------------


class TestLaneGate:
    """No knob may render a key nothing reads — the SPECULATIVE_CONFIG_ROLES rule."""

    def test_worker_only_knobs_are_refused_elsewhere(self) -> None:
        worker_only = [k for k in NEW_KNOBS if k != "tool_call_parser"]
        for role in ("cortex", "senses", "muse", "hand", "embedder", "reranker"):
            for knob in worker_only:
                value = _sample(knob)
                with pytest.raises(ModelGearError) as excinfo:
                    RoleProfile.from_dict(role, {knob: value})
                assert knob in str(excinfo.value)

    def test_worker_accepts_all_of_them(self) -> None:
        for knob in NEW_KNOBS:
            rp = RoleProfile.from_dict("worker", {knob: _sample(knob)})
            assert getattr(rp, knob) == _sample(knob)

    def test_tool_call_parser_is_also_allowed_on_associate(self) -> None:
        # vllm-associate reads ASSOCIATE_TOOL_CALL_PARSER on its own command.
        assert RoleProfile.from_dict("associate", {"tool_call_parser": "qwen3_xml"})


def _sample(knob: str):
    return {
        "moe_backend": "triton",
        "max_num_batched_tokens": 16384,
        "chunked_prefill": True,
        "async_scheduling": True,
        "prefix_caching": False,
        "load_format": "runai_streamer",
        "tool_call_parser": "qwen3_xml",
    }[knob]


# --- 3. rendering -----------------------------------------------------------


class TestRendering:
    def test_declared_knobs_render_worker_keys(self) -> None:
        profile = Profile(
            name="t",
            roles={
                "worker": RoleProfile(
                    model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                    kv_cache_dtype="fp8",
                    attention_backend="TRITON_ATTN",
                    max_num_seqs=8,
                    moe_backend="triton",
                    max_num_batched_tokens=16384,
                    chunked_prefill=True,
                    async_scheduling=True,
                    prefix_caching=False,
                    load_format="runai_streamer",
                    tool_call_parser="qwen3_xml",
                )
            },
        )
        env = profile_env(profile)
        assert env["WORKER_KV_CACHE_DTYPE"] == "fp8"
        assert env["WORKER_ATTENTION_BACKEND"] == "TRITON_ATTN"
        assert env["WORKER_MAX_NUM_SEQS"] == "8"
        assert env["WORKER_MOE_BACKEND"] == "triton"
        assert env["WORKER_MAX_NUM_BATCHED_TOKENS"] == "16384"
        assert env["WORKER_LOAD_FORMAT"] == "runai_streamer"
        assert env["WORKER_TOOL_CALL_PARSER"] == "qwen3_xml"
        # boolean-ish knobs carry the FULL flag text, like ENFORCE_EAGER.
        assert env["WORKER_CHUNKED_PREFILL"] == "--enable-chunked-prefill"
        assert env["WORKER_ASYNC_SCHEDULING"] == "--async-scheduling"
        assert env["WORKER_PREFIX_CACHING"] == "--no-enable-prefix-caching"

    def test_enforce_eager_token_rendering_is_unchanged(self) -> None:
        env = profile_env(
            Profile(name="t", roles={"reranker": RoleProfile(model="m", enforce_eager=True)})
        )
        assert env["RERANK_ENFORCE_EAGER"] == "--enforce-eager"

    def test_new_keys_join_the_deployment_lock_allowlist(self) -> None:
        # The lock allowlist DERIVES from render's tables — a new knob joins it
        # automatically, and that derivation is the thing being verified.
        for key in WORKER_RECIPE_KNOBS.values():
            assert key in lock_keys(), f"{key} must be lockable"


# --- 4. the compose lane ----------------------------------------------------


class TestComposeLane:
    def test_command_is_a_shell_lexed_string(self) -> None:
        svc = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))["services"]["vllm-worker"]
        assert isinstance(svc["command"], str)

    def test_every_key_is_substituted_in_the_command(self) -> None:
        svc = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))["services"]["vllm-worker"]
        command: str = svc["command"]
        for key in WORKER_RECIPE_KNOBS.values():
            assert f"${{{key}" in command, f"vllm-worker command must expand {key}"

    def test_no_language_model_only(self) -> None:
        svc = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))["services"]["vllm-worker"]
        assert "--language-model-only" not in svc["command"]

    def test_env_example_documents_every_key(self) -> None:
        text = _FLEET_ENV.read_text(encoding="utf-8")
        for key in WORKER_RECIPE_KNOBS.values():
            assert key in text, f"env.example must document {key}"

    def test_env_example_no_longer_claims_attention_backend_is_unwired(self) -> None:
        text = _FLEET_ENV.read_text(encoding="utf-8")
        assert "WORKER_ATTENTION_BACKEND — deliberately NOT wired" not in text


# --- 5. the live render (criteria 2 and 3) ----------------------------------


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available")
class TestLiveComposeRender:
    def test_unset_is_byte_identical_to_the_pre_task_argv(self) -> None:
        assert _compose_argv({"COMPOSE_PROFILES": "worker"}) == BASELINE_ARGV

    def test_empty_values_are_also_omitted(self) -> None:
        argv = _compose_argv(
            {"COMPOSE_PROFILES": "worker", **{key: "" for key in WORKER_RECIPE_KNOBS.values()}}
        )
        # An empty WORKER_TOOL_CALL_PARSER falls back to the shipped default
        # (`:-`), which is what keeps today's hardcoded value the default.
        assert argv == BASELINE_ARGV
        assert "" not in argv

    def test_each_knob_reaches_the_argv_when_set(self) -> None:
        argv = _compose_argv(
            {
                "COMPOSE_PROFILES": "worker",
                "WORKER_KV_CACHE_DTYPE": "fp8",
                "WORKER_ATTENTION_BACKEND": "TRITON_ATTN",
                "WORKER_MAX_NUM_SEQS": "8",
                "WORKER_MOE_BACKEND": "triton",
                "WORKER_MAX_NUM_BATCHED_TOKENS": "16384",
                "WORKER_CHUNKED_PREFILL": "--enable-chunked-prefill",
                "WORKER_ASYNC_SCHEDULING": "--async-scheduling",
                "WORKER_PREFIX_CACHING": "--no-enable-prefix-caching",
                "WORKER_LOAD_FORMAT": "runai_streamer",
                "WORKER_TOOL_CALL_PARSER": "qwen3_xml",
            }
        )
        for token in (
            "--kv-cache-dtype=fp8",
            '--attention-config={"backend": "TRITON_ATTN"}',
            "--max-num-seqs=8",
            "--moe-backend=triton",
            "--max-num-batched-tokens=16384",
            "--enable-chunked-prefill",
            "--async-scheduling",
            "--no-enable-prefix-caching",
            "--load-format=runai_streamer",
            "--tool-call-parser=qwen3_xml",
        ):
            assert token in argv, f"{token} missing from {argv}"
        assert "--tool-call-parser=qwen3_coder" not in argv
        assert "--language-model-only" not in argv
        assert "" not in argv
