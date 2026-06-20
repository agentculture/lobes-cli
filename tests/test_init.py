"""Tests for ``model init`` — scaffold a deployment dir."""

from __future__ import annotations

import json
import stat

from model_gear.cli import main
from model_gear.runtime import _compose


def test_init_dry_run_writes_nothing(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", str(target)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert not target.exists()


def test_init_apply_writes_both_files(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply"])
    assert rc == 0
    assert (target / "docker-compose.yml").is_file()
    assert (target / ".env").is_file()
    # the compose template carries the renamed container
    compose = (target / "docker-compose.yml").read_text()
    assert "model-gear-vllm" in compose
    # OpenAI tool/function calling is enabled out of the box (issue #9); the
    # parser is env-driven so a switched model can override it (default qwen3_coder
    # for the Qwen3.6-27B primary).
    assert "--enable-auto-tool-choice" in compose
    assert "--tool-call-parser=${VLLM_TOOL_CALL_PARSER:-qwen3_coder}" in compose
    assert "VLLM_TOOL_CALL_PARSER=qwen3_coder" in (target / ".env").read_text()
    # Durable logs (issue #50): the wrapper is scaffolded, the log dir is pre-created
    # (user-owned), and the vllm service runs the wrapper as its entrypoint + tees to
    # a host-mounted log dir.
    assert (target / "mg-logwrap.sh").is_file()
    assert (target / "logs").is_dir()
    assert 'entrypoint: ["bash", "/usr/local/bin/mg-logwrap"]' in compose
    assert "MG_LOG_NAME=vllm" in compose
    assert "/logs/model-gear" in compose


def test_init_apply_json(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scaffolded"] == str(target)
    assert set(payload["files"]) == {
        "docker-compose.yml",
        ".env",
        "mg-logwrap.sh",
        "cf-tunnel.env.example",
    }


def test_init_refuses_overwrite_without_force(tmp_path) -> None:
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    rc = main(["init", str(target), "--apply"])
    assert rc == 1  # exists; needs --force


def test_init_force_overwrites(tmp_path) -> None:
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    assert main(["init", str(target), "--apply", "--force"]) == 0


def test_init_default_target(capsys) -> None:
    # The autouse fixture points default_deployment_dir at an empty tmp dir.
    default = _compose.default_deployment_dir()
    rc = main(["init", "--apply"])
    assert rc == 0
    assert (default / "docker-compose.yml").is_file()
    assert (default / ".env").is_file()


def test_init_env_is_owner_only(tmp_path) -> None:
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    mode = stat.S_IMODE((target / ".env").stat().st_mode)
    assert mode == 0o600  # .env may hold HF_TOKEN — not world-readable


def test_init_local_folder(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["init", ".", "--apply"])
    assert rc == 0
    assert (tmp_path / "docker-compose.yml").is_file()


# --- fleet scaffold -------------------------------------------------------


def test_init_fleet_apply_writes_three_files(tmp_path) -> None:
    from model_gear import __version__

    target = tmp_path / "fleet"
    rc = main(["init", "--fleet", str(target), "--apply"])
    assert rc == 0
    assert (target / "docker-compose.yml").is_file()
    assert (target / ".env").is_file()
    assert (target / "Dockerfile.gateway").is_file()
    compose = (target / "docker-compose.yml").read_text()
    assert "vllm-primary" in compose
    assert "model-gear-gateway" in compose
    # Single-backend by default: no fallback service is scaffolded (the compose
    # may mention vllm-fallback in "how to add one" comments, so check the
    # service's container_name, which only appears when the service is defined).
    assert "model-gear-vllm-fallback" not in compose
    # Durable logs (issue #50): wrapper scaffolded + each vLLM gear runs it + names
    # its own per-boot log file (primary/embed/rerank).
    assert (target / "mg-logwrap.sh").is_file()
    assert (target / "logs").is_dir()
    assert 'entrypoint: ["bash", "/usr/local/bin/mg-logwrap"]' in compose
    for svc in ("primary", "embed", "rerank"):
        assert f"MG_LOG_NAME={svc}" in compose
    env = (target / ".env").read_text()
    assert "PRIMARY_MODEL=sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" in env
    assert "FALLBACK_MODEL=" not in env
    # The primary is restored to its solo headroom (util 0.6, full 256K).
    assert "PRIMARY_GPU_MEM_UTIL=0.6" in env
    assert "PRIMARY_MAX_MODEL_LEN=262144" in env
    # init --fleet pins the gateway image to the running model-gear version.
    assert f"MODEL_GEAR_VERSION={__version__}" in env
    # coherence mirror keeps the single-model read-only verbs sensible.
    assert "VLLM_SERVED_NAME=sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" in env


def test_init_fleet_dry_run_json(tmp_path, capsys) -> None:
    target = tmp_path / "fleet"
    rc = main(["init", "--fleet", str(target), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fleet"] is True
    names = {f["name"] for f in payload["files"]}
    assert names == {
        "docker-compose.yml",
        ".env",
        "Dockerfile.gateway",
        "mg-logwrap.sh",
        "cf-tunnel.env.example",
    }
    assert not target.exists()


# --- audio overlay (--fleet --audio) --------------------------------------


def test_init_audio_requires_fleet(capsys) -> None:
    rc = main(
        ["init", "--audio", "/tmp/nope", "--json"]
    )  # nosec B108 - never written (errors first)
    assert rc == 1  # EXIT_USER_ERROR
    assert "--audio requires --fleet" in capsys.readouterr().err


def test_init_fleet_audio_apply_writes_overlay_and_appends_env(tmp_path) -> None:
    target = tmp_path / "fa"
    rc = main(["init", "--fleet", "--audio", str(target), "--apply"])
    assert rc == 0
    # fleet files + the audio overlay files. _readiness.py MUST be scaffolded:
    # Dockerfile.parakeet COPYs it, so a missing scaffold breaks `build stt`.
    for name in (
        "docker-compose.yml",
        "Dockerfile.gateway",
        "docker-compose.audio.yml",
        "Dockerfile.realtime",
        "Dockerfile.parakeet",
        "listen_server.py",
        "_readiness.py",
    ):
        assert (target / name).is_file(), name
    env = (target / ".env").read_text()
    # fleet keys still present, audio keys appended (not clobbered).
    assert "PRIMARY_MODEL=sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" in env
    assert "NGC_API_KEY=" in env
    assert "AUDIO_URL=http://realtime:8080" in env
    assert "MAGPIE_TTS_PORT=9000" in env


def test_init_fleet_audio_dry_run_json_lists_overlay(tmp_path, capsys) -> None:
    target = tmp_path / "fa"
    rc = main(["init", "--fleet", "--audio", str(target), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fleet"] is True and payload["audio"] is True
    names = {f["name"] for f in payload["files"]}
    assert {"docker-compose.audio.yml", "Dockerfile.realtime", "listen_server.py"} <= names
    assert not target.exists()


def test_init_fleet_audio_dry_run_text_mentions_appended_env(tmp_path, capsys) -> None:
    target = tmp_path / "fa"
    rc = main(["init", "--fleet", "--audio", str(target)])  # text mode, no --apply
    assert rc == 0
    out = capsys.readouterr().out
    assert ".env (+ audio keys appended)" in out
    assert "Re-run with --apply to write." in out
    assert not target.exists()
