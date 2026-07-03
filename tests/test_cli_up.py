"""Tests for ``lobes up <role>`` (t7, issue #81) — role-based serving.

No docker: every test asserts the dry-run PLAN, or that the (monkeypatched)
compose runner is / isn't invoked. Mirrors ``tests/test_cli_fleet.py``'s offline
style. Covers the four t7 acceptance criteria:

1. ``lobes up colleague-stack`` brings up all six roles across the fleet + audio
   compose files; ``--apply`` is what runs it; colleague-stack is a real target.
2. ``lobes up <role>`` targets ONLY that role's service.
3. Mutation safety: dry-run never invokes the runner.
4. ``lobes up bogus`` → EXIT_USER_ERROR listing the valid targets.
"""

from __future__ import annotations

import json
import types

import pytest

from lobes import roles
from lobes.cli import _build_parser, main
from lobes.cli._commands import up as up_cmd
from lobes.runtime import _compose

_SIX = ["vllm-primary", "vllm-multimodal", "vllm-embed", "vllm-rerank", "stt", "chatterbox"]


def _ok() -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


def _scaffold_fleet_audio(path):
    templates = {**_compose.FLEET_TEMPLATES, **_compose.AUDIO_TEMPLATES}
    _compose.write_scaffold(path, force=True, templates=templates)
    return path


# --- registration + role map (r3) -----------------------------------------


def test_up_is_registered_as_top_level_verb() -> None:
    """r3: ``up`` is a NEW top-level verb dispatching to cmd_up."""
    parser = _build_parser()
    args = parser.parse_args(["up", "cortex"])
    assert args.func is up_cmd.cmd_up


def test_role_service_map_covers_exactly_the_six_roles() -> None:
    """ROLE_SERVICE must stay in lockstep with lobes.roles.ROLES (single source)."""
    assert set(up_cmd.ROLE_SERVICE) == set(roles.ROLES)
    assert up_cmd.TARGETS == roles.ROLES + ("colleague-stack",)


# --- acceptance 2: a single role targets ONLY its service ------------------


def test_up_cortex_dry_run_targets_only_primary(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["up", "cortex", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "docker compose up -d vllm-primary" in out
    # ONLY cortex's service — never the multimodal (senses) or pooling gears.
    assert "vllm-multimodal" not in out
    assert "vllm-embed" not in out
    assert "vllm-rerank" not in out


@pytest.mark.parametrize(
    "role,service",
    [
        ("cortex", "vllm-primary"),
        ("senses", "vllm-multimodal"),
        ("embedder", "vllm-embed"),
        ("reranker", "vllm-rerank"),
    ],
)
def test_up_each_fleet_role_targets_its_service(tmp_path, capsys, role, service) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["up", role, "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["services"] == [service]
    assert payload["command"] == f"docker compose up -d {service}"


def test_up_fleet_role_ignores_audio_overlay_even_when_present(tmp_path, capsys) -> None:
    """A fleet-only role uses the base compose only — no ``-f`` audio overlay,
    even when it happens to be scaffolded (targets ONLY that service)."""
    _scaffold_fleet_audio(tmp_path)
    rc = main(["up", "cortex", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "docker compose up -d vllm-primary"
    assert "docker-compose.audio.yml" not in payload["command"]


# --- acceptance 1: colleague-stack = all six across both compose files -----


def test_up_colleague_stack_dry_run_covers_all_six(tmp_path, capsys) -> None:
    _scaffold_fleet_audio(tmp_path)
    rc = main(["up", "colleague-stack", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["target"] == "colleague-stack"
    # cortex+senses+embedder+reranker AND stt+tts (r4), canonical role order.
    assert payload["services"] == _SIX
    # ...across the fleet + audio compose files.
    assert payload["command"] == (
        "docker compose -f docker-compose.yml -f docker-compose.audio.yml up -d "
        "vllm-primary vllm-multimodal vllm-embed vllm-rerank stt chatterbox"
    )


def test_up_colleague_stack_text_plan_names_the_six(tmp_path, capsys) -> None:
    _scaffold_fleet_audio(tmp_path)
    rc = main(["up", "colleague-stack", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    for service in _SIX:
        assert service in out


# r4: colleague-stack REQUIRES the audio overlay; explain when it is missing.


def test_up_colleague_stack_without_overlay_errors_with_hint(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)  # base fleet, NO audio overlay
    rc = main(["up", "colleague-stack", "--compose-dir", str(tmp_path)])
    assert rc == 1  # EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "audio" in err.lower()
    assert "--audio" in err  # points at `lobes init --fleet --audio`


def test_up_tts_without_overlay_errors(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)  # no overlay
    rc = main(["up", "tts", "--compose-dir", str(tmp_path)])
    assert rc == 1
    assert "--audio" in capsys.readouterr().err


def test_up_stt_dry_run_includes_audio_overlay(tmp_path, capsys) -> None:
    _scaffold_fleet_audio(tmp_path)
    rc = main(["up", "stt", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["services"] == ["stt"]
    assert payload["command"] == (
        "docker compose -f docker-compose.yml -f docker-compose.audio.yml up -d stt"
    )


def test_up_tts_dry_run_targets_chatterbox_service(tmp_path, capsys) -> None:
    _scaffold_fleet_audio(tmp_path)
    rc = main(["up", "tts", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # tts → the `chatterbox` compose service (NOT the container name).
    assert payload["services"] == ["chatterbox"]


# --- acceptance 3: mutation safety -----------------------------------------


def test_up_dry_run_does_not_invoke_runner(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)

    def boom(*a, **k):
        raise AssertionError("compose ran during dry-run")

    monkeypatch.setattr(_compose, "run_compose", boom)
    rc = main(["up", "cortex", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_up_apply_invokes_runner_with_only_the_target(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    captured: dict = {}

    def fake_run(deploy_dir, argv):
        captured["argv"] = argv
        return _ok()

    monkeypatch.setattr(_compose, "run_compose", fake_run)
    rc = main(["up", "cortex", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    assert captured["argv"] == ["docker", "compose", "up", "-d", "vllm-primary"]


def test_up_apply_colleague_stack_runs_full_argv(tmp_path, monkeypatch) -> None:
    _scaffold_fleet_audio(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        _compose, "run_compose", lambda d, argv: (captured.setdefault("argv", argv), _ok())[1]
    )
    rc = main(["up", "colleague-stack", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert captured["argv"] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.audio.yml",
        "up",
        "-d",
        *_SIX,
    ]


# --- --down (scoped stop, never a project-wide `down`) ---------------------


def test_up_down_dry_run_plans_scoped_stop(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["up", "cortex", "--down", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "stop"
    assert payload["command"] == "docker compose stop vllm-primary"


def test_up_down_apply_runs_scoped_stop(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        _compose, "run_compose", lambda d, argv: (captured.setdefault("argv", argv), _ok())[1]
    )
    rc = main(["up", "cortex", "--down", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    # scoped `stop` — never a project-wide `down` that removes every container.
    assert captured["argv"] == ["docker", "compose", "stop", "vllm-primary"]


# --- acceptance 4: bogus role ----------------------------------------------


def test_up_bogus_role_is_user_error_listing_targets(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["up", "bogus", "--compose-dir", str(tmp_path)])
    assert rc == 1  # EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "hint:" in err
    for target in ("cortex", "senses", "embedder", "reranker", "stt", "tts", "colleague-stack"):
        assert target in err


def test_up_bogus_role_errors_before_scaffold_resolution(capsys) -> None:
    """Target validation happens before deployment-dir resolution, so a bogus
    role errors on the name (USER_ERROR) even with nothing scaffolded."""
    rc = main(["up", "bogus"])
    assert rc == 1
    assert "colleague-stack" in capsys.readouterr().err
