"""Tests for the lobes-deploy skill's ``lobes-compose.sh`` wrapper.

Dry-run only: a stub ``lobes`` on PATH answers ``fleet files``, and the wrapper
prints the compose command it would run. No docker is invoked.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[1] / (
    ".claude/skills/lobes-deploy/scripts/lobes-compose.sh"
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    deploy = tmp_path / "deploy"
    deploy.mkdir(exist_ok=True)
    (deploy / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "lobes"
    stub.write_text("#!/usr/bin/env bash\nprintf -- '-f\\ndocker-compose.yml\\n'\n")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(  # nosec B603 - fixed local script, test-only args
        ["bash", str(WRAPPER), "--compose-dir", str(deploy), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_profile_after_the_subcommand_is_moved_to_the_front(tmp_path) -> None:
    """``up -d --no-deps --profile innereye comfyui`` is what an agent types;
    compose only accepts --profile before the subcommand."""
    out = _run(tmp_path, "up", "-d", "--no-deps", "--profile", "innereye", "comfyui")
    assert out.returncode == 0, out.stderr
    assert "-f docker-compose.yml --profile innereye up -d --no-deps comfyui" in out.stdout
    assert "DRY-RUN" in out.stdout


def test_profile_equals_form_is_moved_too(tmp_path) -> None:
    out = _run(tmp_path, "build", "--profile=innereye", "comfyui")
    assert out.returncode == 0, out.stderr
    assert "--profile innereye build comfyui" in out.stdout


def test_no_profile_leaves_the_command_alone(tmp_path) -> None:
    out = _run(tmp_path, "up", "-d", "--no-deps", "gateway")
    assert out.returncode == 0, out.stderr
    assert "-f docker-compose.yml up -d --no-deps gateway" in out.stdout


def test_profile_after_the_service_belongs_to_the_command_inside(tmp_path) -> None:
    """In `exec comfyui python main.py --profile x`, --profile is an argument
    of the in-container command; it must stay where it is."""
    out = _run(tmp_path, "exec", "comfyui", "python", "main.py", "--profile", "x")
    assert out.returncode == 0, out.stderr
    assert "-f docker-compose.yml exec comfyui python main.py --profile x" in out.stdout


def test_the_template_folder_is_refused(tmp_path) -> None:
    templates = tmp_path / "lobes" / "templates" / "fleet"
    templates.mkdir(parents=True)
    out = subprocess.run(  # nosec B603 - fixed local script, test-only args
        ["bash", str(WRAPPER), "--compose-dir", str(templates), "ps"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 2
    assert "REFUSED" in out.stderr
