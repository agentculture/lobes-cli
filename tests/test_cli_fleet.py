"""Tests for the ``model fleet`` verbs (up / down / status) and ``init --fleet``."""

from __future__ import annotations

import json
import types

from model_gear.cli import main
from model_gear.runtime import _compose, _health


def _ok() -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


def _scaffold_fleet_audio(path):
    templates = {**_compose.FLEET_TEMPLATES, **_compose.AUDIO_TEMPLATES}
    _compose.write_scaffold(path, force=True, templates=templates)
    return path


# --- fleet up -------------------------------------------------------------


def test_fleet_up_dry_run_changes_nothing(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)

    def boom(*a, **k):
        raise AssertionError("compose ran during dry-run")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    rc = main(["fleet", "up", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_fleet_up_apply_builds_and_waits(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        _compose, "compose_up_build", lambda d: (calls.append("up-build"), _ok())[1]
    )
    waited: dict = {}

    def fake_wait(port, **kw):
        waited["port"] = port
        waited["container"] = kw.get("container")

    monkeypatch.setattr(_health, "wait_health", fake_wait)
    rc = main(["fleet", "up", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    assert calls == ["up-build"]
    assert waited["container"] == _compose.FLEET_GATEWAY  # waits on the gateway front


# --- fleet down -----------------------------------------------------------


def test_fleet_down_dry_run(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["fleet", "down", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_fleet_down_apply(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_down", lambda d: (calls.append("down"), _ok())[1])
    rc = main(["fleet", "down", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert calls == ["down"]


# --- fleet status ---------------------------------------------------------


def test_fleet_status_json_reports_default_containers(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["fleet", "status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    names = [c["name"] for c in payload["containers"]]
    # The default fleet runs four containers: generate primary, embedding +
    # reranker gears, and the gateway (the generate fallback is opt-in, excluded).
    assert names == list(_compose.FLEET_CONTAINERS)
    assert names == [
        "model-gear-vllm-primary",
        "model-gear-vllm-embed",
        "model-gear-vllm-rerank",
        "model-gear-gateway",
    ]
    # offline fixture: _probe → None (state "not created"), is_healthy → False.
    assert all(c["state"] == "not created" for c in payload["containers"])
    assert payload["gateway_health"] == "not responding"
    assert payload["models"] is None  # not healthy → no /v1/models fetch
    assert payload["port"] == 8000


def test_bare_fleet_defaults_to_status(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["fleet", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gateway:" in out
    assert _compose.FLEET_GATEWAY in out


def test_fleet_status_fetches_models_when_healthy(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(_health, "is_healthy", lambda *a, **k: True)
    from model_gear import assess

    monkeypatch.setattr(
        assess, "_get", lambda url, path, timeout=10: (200, {"data": [{"id": "P"}, {"id": "F"}]})
    )
    rc = main(["fleet", "status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gateway_health"] == "ok"
    assert payload["models"] == ["P", "F"]


def test_fleet_status_unscaffolded_errors(capsys) -> None:
    # No deployment scaffolded (autouse fixture points the home at an empty dir).
    rc = main(["fleet", "status"])
    assert rc == 2  # EXIT_ENV_ERROR
    assert "hint:" in capsys.readouterr().err


# --- audio overlay awareness ----------------------------------------------


def test_compose_files_only_adds_overlay_when_present(tmp_path) -> None:
    _scaffold_fleet(tmp_path)  # no audio overlay
    assert _compose._compose_files(tmp_path) == []
    assert _compose.audio_overlay_present(tmp_path) is False
    _scaffold_fleet_audio(tmp_path)  # now with the overlay
    assert _compose.audio_overlay_present(tmp_path) is True
    assert _compose._compose_files(tmp_path) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.AUDIO_OVERLAY,
    ]


def test_compose_up_build_includes_overlay_argv(tmp_path, monkeypatch) -> None:
    _scaffold_fleet_audio(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        _compose, "_run", lambda argv, **kw: captured.setdefault("argv", argv) or _ok()
    )
    _compose.compose_up_build(tmp_path)
    assert captured["argv"] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.audio.yml",
        "up",
        "-d",
        "--build",
    ]


def test_fleet_status_includes_audio_containers_with_overlay(tmp_path, capsys) -> None:
    _scaffold_fleet_audio(tmp_path)
    rc = main(["fleet", "status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    names = [c["name"] for c in payload["containers"]]
    assert names == list(_compose.FLEET_CONTAINERS) + list(_compose.FLEET_AUDIO_CONTAINERS)


def test_fleet_up_reports_audio_containers_with_overlay(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet_audio(tmp_path)
    monkeypatch.setattr(_compose, "compose_up_build", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    rc = main(["fleet", "up", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["containers"] == (
        list(_compose.FLEET_CONTAINERS) + list(_compose.FLEET_AUDIO_CONTAINERS)
    )
