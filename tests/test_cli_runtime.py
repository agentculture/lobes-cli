"""Tests for the model-ops runtime: .env r/w, dir resolution, switch/serve/stop/status."""

from __future__ import annotations

import json
import types

import pytest

from model_gear.cli import main
from model_gear.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from model_gear.runtime import _compose, _env, _health


def _ok() -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _scaffold(path):
    _compose.write_scaffold(path, force=True)
    return path


# --- _env -----------------------------------------------------------------


def test_env_read_write(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("VLLM_PORT=8000\nHF_TOKEN=\n", encoding="utf-8")
    assert _env.read_env(env, "VLLM_PORT") == "8000"
    # empty value (KEY=) reads as the caller's default
    assert _env.read_env(env, "HF_TOKEN", "fallback") == "fallback"
    # absent key reads as default
    assert _env.read_env(env, "NOPE", "x") == "x"
    # rewrite-if-present
    _env.set_env(env, "VLLM_PORT", "9001")
    assert _env.read_env(env, "VLLM_PORT") == "9001"
    # append-if-absent
    _env.set_env(env, "VLLM_MODEL", "foo/bar")
    assert _env.read_env(env, "VLLM_MODEL") == "foo/bar"


def test_read_env_missing_file_returns_default(tmp_path) -> None:
    assert _env.read_env(tmp_path / "nope.env", "K", "default") == "default"


def test_set_env_missing_file_raises(tmp_path) -> None:
    with pytest.raises(ModelGearError) as exc:
        _env.set_env(tmp_path / "nope.env", "K", "V")
    assert exc.value.code == EXIT_ENV_ERROR


# --- resolve_deployment_dir ----------------------------------------------


def test_resolve_explicit(tmp_path) -> None:
    _scaffold(tmp_path)
    assert _compose.resolve_deployment_dir(str(tmp_path)) == tmp_path


def test_resolve_explicit_missing_raises_user_error(tmp_path) -> None:
    with pytest.raises(ModelGearError) as exc:
        _compose.resolve_deployment_dir(str(tmp_path / "empty"))
    assert exc.value.code == EXIT_USER_ERROR


def test_resolve_env_var(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setenv("MODEL_GEAR_DIR", str(tmp_path))
    assert _compose.resolve_deployment_dir(None) == tmp_path


def test_resolve_default_missing_raises_env_error() -> None:
    # The autouse fixture points the default home at an empty tmp dir.
    with pytest.raises(ModelGearError) as exc:
        _compose.resolve_deployment_dir(None)
    assert exc.value.code == EXIT_ENV_ERROR


# --- port parsing ---------------------------------------------------------


def test_parse_port_invalid_raises_env_error() -> None:
    with pytest.raises(ModelGearError) as exc:
        _env.parse_port("not-a-number", "VLLM_PORT")
    assert exc.value.code == EXIT_ENV_ERROR


def test_invalid_env_port_gives_clean_error(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _env.set_env(tmp_path / ".env", "VLLM_PORT", "abc")
    rc = main(["status", "--compose-dir", str(tmp_path)])
    assert rc == EXIT_ENV_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err  # structured, not a generic "unexpected: ValueError"


# --- switch ---------------------------------------------------------------


def test_switch_dry_run_changes_nothing(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "VLLM_MODEL=foo/bar" in out
    # .env untouched
    assert _env.read_env(tmp_path / ".env", "VLLM_MODEL") == "nvidia/Qwen3-32B-NVFP4"


def test_switch_apply_recreates_and_writes_env(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_down", lambda d: (calls.append("down"), _ok())[1])
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: (calls.append("up"), _ok())[1])
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)

    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert calls == ["down", "up"]  # frees prior model before starting new one
    env = tmp_path / ".env"
    assert _env.read_env(env, "VLLM_MODEL") == "foo/bar"
    assert _env.read_env(env, "VLLM_SERVED_NAME") == "foo/bar"


def test_switch_writes_tool_call_parser_when_given(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_compose, "compose_down", lambda d: _ok())
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)

    rc = main(
        [
            "switch",
            "mmangkad/Qwen3.6-27B-NVFP4",
            "--tool-call-parser",
            "qwen3_coder",
            "--compose-dir",
            str(tmp_path),
            "--apply",
        ]
    )
    assert rc == 0
    assert _env.read_env(tmp_path / ".env", "VLLM_TOOL_CALL_PARSER") == "qwen3_coder"


def test_switch_leaves_tool_call_parser_when_absent(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    # the scaffolded .env carries the default; a switch without the flag must
    # neither plan nor write VLLM_TOOL_CALL_PARSER.
    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "VLLM_TOOL_CALL_PARSER" not in capsys.readouterr().out
    assert _env.read_env(tmp_path / ".env", "VLLM_TOOL_CALL_PARSER") == "hermes"


def test_switch_apply_surfaces_compose_failure(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(
        _compose,
        "compose_down",
        lambda d: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == EXIT_ENV_ERROR


# --- serve / stop ---------------------------------------------------------


def test_serve_dry_run(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["serve", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_serve_apply(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: (calls.append("up"), _ok())[1])
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    rc = main(["serve", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert calls == ["up"]


def test_start_is_serve_alias(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["start", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_stop_dry_run(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["stop", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_stop_apply(tmp_path, monkeypatch) -> None:
    _scaffold(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_down", lambda d: (calls.append("down"), _ok())[1])
    rc = main(["stop", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert calls == ["down"]


# --- status ---------------------------------------------------------------


def test_status_json(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    rc = main(["status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["container"] == "model-gear-vllm"
    assert payload["state"] == "not created"  # offline _probe → None
    assert payload["health"] == "not responding"  # offline is_healthy → False
    assert payload["model"] == "nvidia/Qwen3-32B-NVFP4"
