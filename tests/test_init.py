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
    assert "model-gear-vllm" in (target / "docker-compose.yml").read_text()


def test_init_apply_json(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scaffolded"] == str(target)
    assert set(payload["files"]) == {"docker-compose.yml", ".env"}


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
