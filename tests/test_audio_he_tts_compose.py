"""docker-compose.audio-he.yml / env.audio-he.example: the Hebrew TTS
sidecar wiring (hebrew-realtime plan, task t11).

Complements tests/test_chatterbox_ml_dockerfile.py (Dockerfile structure)
and tests/test_audio_he_stt_compose.py (the sibling STT compose test this
file mirrors for TTS). This file checks:

  1. The `chatterbox` service in docker-compose.audio-he.yml overrides ONLY
     `build.dockerfile`, `environment` and `healthcheck.start_period` — every
     other key (container_name, volumes, expose, deploy, ipc,
     CHATTERBOX_PORT) is inherited unchanged from docker-compose.audio.yml.
  2. TTS_MODEL/TTS_RUNTIME/TTS_LANGUAGE reach the `gateway` service too
     (deviation d3, extended from STT to TTS by this task).
  3. Every TTS_* key documented with a default in env.audio-he.example
     matches the `${KEY:-default}` compose declares.
  4. No shipped template anywhere under lobes/templates/ references a
     non-commercially-licensed phonikud-tts engine or checkpoint (criterion
     3 / plan risk r3, q4 deferred).
  5. docker-compose.audio.yml, env.audio.example and chatterbox_server.py
     (the English overlay) are untouched by this task.
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

_TTS_KEYS = ("TTS_MODEL", "TTS_RUNTIME", "TTS_LANGUAGE")
_CHATTERBOX_ENV_KEYS = ("TTS_LANGUAGE", "TTS_DIACRITIZE", "TTS_MAX_RETRIES", "PHONIKUD_MODEL_PATH")


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


class TestChatterboxServiceOverride:
    def test_chatterbox_service_present_in_hebrew_overlay(self) -> None:
        compose = _load(_AUDIO_HE_COMPOSE)
        assert "chatterbox" in compose["services"]

    def test_only_build_environment_and_healthcheck_are_overridden(self) -> None:
        chatterbox = _load(_AUDIO_HE_COMPOSE)["services"]["chatterbox"]
        assert set(chatterbox.keys()) <= {"build", "environment", "healthcheck"}, (
            f"chatterbox override should only touch build/environment/healthcheck, "
            f"got keys: {sorted(chatterbox.keys())} — container_name/volumes/expose/"
            "deploy/ipc must stay inherited from docker-compose.audio.yml"
        )

    def test_build_dockerfile_points_at_chatterbox_ml(self) -> None:
        chatterbox = _load(_AUDIO_HE_COMPOSE)["services"]["chatterbox"]
        assert chatterbox["build"]["dockerfile"] == "Dockerfile.chatterbox-ml"
        # context is inherited (not repeated) from docker-compose.audio.yml.
        assert "context" not in chatterbox["build"]

    def test_healthcheck_only_overrides_start_period(self) -> None:
        chatterbox = _load(_AUDIO_HE_COMPOSE)["services"]["chatterbox"]
        assert set(chatterbox["healthcheck"].keys()) == {"start_period"}

    def test_start_period_is_at_least_300s(self) -> None:
        chatterbox = _load(_AUDIO_HE_COMPOSE)["services"]["chatterbox"]
        value = chatterbox["healthcheck"]["start_period"]
        assert value.endswith("s")
        assert int(value[:-1]) >= 300

    def test_environment_carries_all_four_chatterbox_keys(self) -> None:
        chatterbox = _load(_AUDIO_HE_COMPOSE)["services"]["chatterbox"]
        env = _env_map(chatterbox)
        for key in _CHATTERBOX_ENV_KEYS:
            assert key in env, f"{key} missing from the chatterbox service environment"

    def test_base_chatterbox_service_keeps_port_and_context(self) -> None:
        """Confirms what "inherited unchanged" means concretely: the base
        overlay's chatterbox service still declares CHATTERBOX_PORT and the
        same build context — nothing here needs to repeat them."""
        base_chatterbox = _load(_AUDIO_COMPOSE)["services"]["chatterbox"]
        base_env = _env_map(base_chatterbox)
        assert "CHATTERBOX_PORT" in base_env
        assert base_chatterbox["build"]["context"] == "."
        assert base_chatterbox["container_name"] == "model-gear-chatterbox"
        assert "healthcheck" in base_chatterbox
        assert "volumes" in base_chatterbox


class TestRealtimeVolumeMount:
    def test_realtime_gains_the_hf_cache_mount(self) -> None:
        """The bridge's own vocalize-before-chunking hook (t9) and this
        sidecar's TTS_DIACRITIZE=auto fallback path need the SAME mounted
        phonikud model file — the English overlay's `realtime` service
        declares no volumes at all, so the Hebrew overlay must add one."""
        realtime = _load(_AUDIO_HE_COMPOSE)["services"]["realtime"]
        assert "volumes" in realtime
        assert any("huggingface" in v for v in realtime["volumes"])

    def test_english_realtime_service_has_no_volumes(self) -> None:
        realtime = _load(_AUDIO_COMPOSE)["services"]["realtime"]
        assert "volumes" not in realtime


class TestGatewayPassthrough:
    def test_gateway_environment_carries_all_three_tts_keys(self) -> None:
        gateway = _load(_AUDIO_HE_COMPOSE)["services"]["gateway"]
        env = _env_map(gateway)
        for key in _TTS_KEYS:
            assert key in env, f"{key} missing from the gateway passthrough (deviation d3)"

    def test_tts_model_default_is_the_shared_repo_id(self) -> None:
        """TTS_MODEL deliberately keeps the SAME value as the English
        default — the installed package's chatterbox.mtl_tts.REPO_ID really
        is the same HuggingFace repo (confirmed 2026-09-18)."""
        gateway = _load(_AUDIO_HE_COMPOSE)["services"]["gateway"]
        env = _env_map(gateway)
        assert env["TTS_MODEL"] == "${TTS_MODEL:-ResembleAI/chatterbox}"

    def test_tts_runtime_default_distinguishes_the_engine(self) -> None:
        gateway = _load(_AUDIO_HE_COMPOSE)["services"]["gateway"]
        env = _env_map(gateway)
        assert env["TTS_RUNTIME"] == "${TTS_RUNTIME:-chatterbox-multilingual}"


class TestEnvExampleDocumentsDefaults:
    def test_all_chatterbox_and_gateway_keys_documented(self) -> None:
        example = _env_example_defaults()
        for key in set(_CHATTERBOX_ENV_KEYS) | set(_TTS_KEYS):
            assert key in example, f"{key} undocumented in env.audio-he.example"

    def test_documented_defaults_match_compose_chatterbox_service(self) -> None:
        chatterbox_env = _env_map(_load(_AUDIO_HE_COMPOSE)["services"]["chatterbox"])
        example = _env_example_defaults()
        for key in _CHATTERBOX_ENV_KEYS:
            value = chatterbox_env[key]
            assert _is_self_referencing(key, value), f"{key} is not operator-tunable: {value!r}"
            default = value[len(f"${{{key}:-") : -1]
            assert example[key] == default, (
                f"{key} default drift: compose says {default!r}, "
                f"env.audio-he.example says {example[key]!r}"
            )

    def test_documented_defaults_match_compose_gateway_service(self) -> None:
        gateway_env = _env_map(_load(_AUDIO_HE_COMPOSE)["services"]["gateway"])
        example = _env_example_defaults()
        for key in _TTS_KEYS:
            value = gateway_env[key]
            assert _is_self_referencing(key, value), f"{key} is not operator-tunable: {value!r}"
            default = value[len(f"${{{key}:-") : -1]
            assert example[key] == default, (
                f"{key} default drift: compose says {default!r}, "
                f"env.audio-he.example says {example[key]!r}"
            )


class TestNoNonCommercialWeightReferenced:
    """Criterion 3 / plan risk r3: no non-commercially-licensed weight is
    referenced by any SHIPPED template — q4 (whether phonikud-tts may ever
    appear as an opt-in engine) stays deferred; this task uses Chatterbox
    Multilingual (MIT) + phonikud-onnx (CC BY 4.0) only."""

    _FORBIDDEN_STRINGS = ("phonikud-tts", "thewh1teagle/phonikud-tts")

    def test_no_template_file_references_phonikud_tts(self) -> None:
        offenders: list[str] = []
        for path in _TEMPLATES.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for forbidden in self._FORBIDDEN_STRINGS:
                if forbidden in text:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}: {forbidden}")
        assert not offenders, f"non-commercial engine reference(s) found: {offenders}"

    def test_no_checkpoints_repo_referenced(self) -> None:
        """A "*-checkpoints" style repo id is the shape phonikud-tts's own
        non-commercial model releases use — belt-and-suspenders on top of
        the exact-name check above."""
        offenders: list[str] = []
        for path in _TEMPLATES.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line in text.splitlines():
                if "-checkpoints" in line and "phonikud" in line.lower():
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}: {line.strip()}")
        assert not offenders, f"non-commercial checkpoints reference(s) found: {offenders}"


class TestEnglishOverlayUntouched:
    """docker-compose.audio.yml, env.audio.example and chatterbox_server.py
    must stay byte-identical to what earlier tasks left them at — a Hebrew
    deployment is opt-in, layered on top."""

    def test_english_files_match_git_head(self) -> None:
        for rel in (
            "lobes/templates/fleet/docker-compose.audio.yml",
            "lobes/templates/fleet/env.audio.example",
            "lobes/templates/fleet/Dockerfile.chatterbox",
            "lobes/realtime/chatterbox_server.py",
        ):
            result = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", rel],
                cwd=_REPO_ROOT,
                check=False,
            )
            assert result.returncode == 0, f"{rel} differs from git HEAD — must stay untouched"
