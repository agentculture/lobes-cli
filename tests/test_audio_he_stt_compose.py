"""docker-compose.audio-he.yml / env.audio-he.example: the Hebrew STT sidecar
wiring (hebrew-realtime plan, task t10).

Complements tests/test_whisper_stt_dockerfile.py (Dockerfile structure) and
tests/test_realtime_audio_env_coverage.py (the `realtime` service's own
settings-key coverage, which this task does not touch). This file checks:

  1. The `stt` service in docker-compose.audio-he.yml overrides ONLY
     `build.dockerfile` and `environment` — every other key (container_name,
     healthcheck, volumes, expose, PARAKEET_PORT) is inherited unchanged
     from docker-compose.audio.yml via compose's deep merge, so the bridge's
     STT_URL=http://stt:${PARAKEET_PORT:-9002} needs no change.
  2. STT_MODEL/STT_RUNTIME/STT_LANGUAGE reach the `gateway` service too
     (approved deviation d3) so GET /capabilities can advertise the engine
     actually served.
  3. Every STT_* key documented with a default in env.audio-he.example
     matches the `${KEY:-default}` compose declares, for both the `stt` and
     `gateway` blocks — the same "doctor --fix has a default to heal with"
     contract test_realtime_audio_env_coverage.py enforces for `realtime`.
  4. docker-compose.audio.yml, env.audio.example, Dockerfile.parakeet and
     listen_server.py (the English overlay) are untouched by this task.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES = _REPO_ROOT / "lobes" / "templates" / "fleet"
_AUDIO_COMPOSE = _TEMPLATES / "docker-compose.audio.yml"
_AUDIO_HE_COMPOSE = _TEMPLATES / "docker-compose.audio-he.yml"
_AUDIO_HE_ENV_EXAMPLE = _TEMPLATES / "env.audio-he.example"

_STT_KEYS = ("STT_MODEL", "STT_RUNTIME", "STT_LANGUAGE")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _env_map(service: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in service.get("environment", []):
        key, _, value = entry.partition("=")
        out[key] = value
    return out


def _env_example_defaults() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _AUDIO_HE_ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        out[key.strip()] = value.strip()
    return out


def _is_self_referencing(key: str, value: str) -> bool:
    return bool(re.fullmatch(rf"\$\{{{re.escape(key)}(:-[^}}]*)?\}}", value))


class TestSttServiceOverride:
    def test_stt_service_present_in_hebrew_overlay(self) -> None:
        compose = _load(_AUDIO_HE_COMPOSE)
        assert "stt" in compose["services"]

    def test_only_build_and_environment_are_overridden(self) -> None:
        stt = _load(_AUDIO_HE_COMPOSE)["services"]["stt"]
        assert set(stt.keys()) <= {"build", "environment"}, (
            f"stt override should only touch build/environment, got keys: "
            f"{sorted(stt.keys())} — container_name/healthcheck/volumes/expose "
            "must stay inherited from docker-compose.audio.yml"
        )

    def test_build_dockerfile_points_at_whisper_stt(self) -> None:
        stt = _load(_AUDIO_HE_COMPOSE)["services"]["stt"]
        assert stt["build"]["dockerfile"] == "Dockerfile.whisper-stt"
        # context is inherited (not repeated) from docker-compose.audio.yml.
        assert "context" not in stt["build"]

    def test_environment_carries_all_three_stt_keys(self) -> None:
        stt = _load(_AUDIO_HE_COMPOSE)["services"]["stt"]
        env = _env_map(stt)
        for key in _STT_KEYS:
            assert key in env, f"{key} missing from the stt service environment"

    def test_base_stt_service_keeps_parakeet_port_and_context(self) -> None:
        """Confirms what "inherited unchanged" means concretely: the base
        overlay's stt service still declares PARAKEET_PORT and the same
        build context — nothing here needs to repeat them."""
        base_stt = _load(_AUDIO_COMPOSE)["services"]["stt"]
        base_env = _env_map(base_stt)
        assert "PARAKEET_PORT" in base_env
        assert base_stt["build"]["context"] == "."
        assert base_stt["container_name"] == "model-gear-stt"
        assert "healthcheck" in base_stt


class TestGatewayPassthrough:
    def test_gateway_environment_carries_all_three_stt_keys(self) -> None:
        gateway = _load(_AUDIO_HE_COMPOSE)["services"]["gateway"]
        env = _env_map(gateway)
        for key in _STT_KEYS:
            assert key in env, f"{key} missing from the gateway passthrough (deviation d3)"


class TestEnvExampleDocumentsDefaults:
    def test_all_three_keys_documented(self) -> None:
        example = _env_example_defaults()
        for key in _STT_KEYS:
            assert key in example, f"{key} undocumented in env.audio-he.example"

    def test_documented_defaults_match_compose_stt_service(self) -> None:
        stt_env = _env_map(_load(_AUDIO_HE_COMPOSE)["services"]["stt"])
        example = _env_example_defaults()
        for key in ("STT_MODEL", "STT_LANGUAGE"):
            value = stt_env[key]
            assert _is_self_referencing(key, value), f"{key} is not operator-tunable: {value!r}"
            default = value[len(f"${{{key}:-") : -1]
            assert example[key] == default, (
                f"{key} default drift: compose says {default!r}, "
                f"env.audio-he.example says {example[key]!r}"
            )

    def test_gateway_stt_runtime_default_matches_example(self) -> None:
        gateway_env = _env_map(_load(_AUDIO_HE_COMPOSE)["services"]["gateway"])
        example = _env_example_defaults()
        value = gateway_env["STT_RUNTIME"]
        assert _is_self_referencing("STT_RUNTIME", value)
        default = value[len("${STT_RUNTIME:-") : -1]
        assert example["STT_RUNTIME"] == default


class TestEnglishOverlayUntouched:
    """docker-compose.audio.yml, env.audio.example, Dockerfile.parakeet and
    listen_server.py must stay byte-identical to what wave 2's other tasks
    left them at — a Hebrew deployment is opt-in, layered on top."""

    def test_english_files_match_git_head(self) -> None:
        for rel in (
            "lobes/templates/fleet/docker-compose.audio.yml",
            "lobes/templates/fleet/env.audio.example",
            "lobes/templates/fleet/Dockerfile.parakeet",
            "lobes/templates/fleet/listen_server.py",
        ):
            result = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", rel],
                cwd=_REPO_ROOT,
                check=False,
            )
            assert result.returncode == 0, f"{rel} differs from git HEAD — must stay untouched"
