"""Tests for ``lobes doctor`` — real docker/compose/.env/health checks."""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.runtime import _compose, _env


def test_doctor_offline_is_unhealthy(capsys) -> None:
    # Offline fixture: docker unavailable + no deployment scaffolded.
    rc = main(["doctor", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["docker_available"]["passed"] is False
    assert ids["compose_present"]["passed"] is False
    assert payload["healthy"] is False


def test_doctor_healthy_when_docker_and_scaffold_present(tmp_path, monkeypatch, capsys) -> None:
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)

    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["docker_available"]["passed"] is True
    assert ids["compose_present"]["passed"] is True
    assert "env_coherence" in ids
    # The scaffolded VLLM_SERVED_NAME matches culture.yaml in this repo.
    assert ids["env_coherence"]["passed"] is True
    # /health is down (info severity only) → run is still healthy overall.
    assert payload["healthy"] is True
    assert rc == 0


def test_doctor_compose_dir_flag(tmp_path, monkeypatch, capsys) -> None:
    # Default home is the empty tmp dir (autouse); --compose-dir points elsewhere.
    _compose.write_scaffold(tmp_path, force=True)
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    rc = main(["doctor", "--compose-dir", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["compose_present"]["passed"] is True
    assert "env_coherence" in ids  # only reachable once a dir resolves
    assert rc == 0


def test_doctor_env_mismatch_warns_but_passes(tmp_path, monkeypatch, capsys) -> None:
    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / ".env", "VLLM_SERVED_NAME", "other/model")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)

    rc = main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["env_coherence"]["passed"] is False
    assert ids["env_coherence"]["severity"] == "warn"
    # A warn does not fail the run.
    assert payload["healthy"] is True
    assert rc == 0
