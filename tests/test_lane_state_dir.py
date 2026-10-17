"""review-b #1 / #7: host bind sources the lanes write into, created as the operator.

#1: ``logs/lane-state/`` used to be created by the first ROOT container that
wrote a start counter (root 0755), so the operator's ``lobes up <role> --apply``
could not unlink ``<lane>.starts`` and ``except OSError: pass`` hid it — the lane
stayed held while the verb reported success.

#7: the compile-cache bind sources were created only by ``lobes init`` and
``doctor --fix``; every lifecycle verb that starts containers must create them
too, or Docker creates ``compile-cache/<lane>`` root-owned on first mount.
"""

from __future__ import annotations

import os
import stat
import types
from pathlib import Path

import pytest

from lobes.cli import main
from lobes.cli._commands import _doctor_lane_safety as ls
from lobes.runtime import _compose, _machine

_COMPOSE = """\
services:
  vllm-primary:
    environment:
      - MG_LOG_NAME=primary
    volumes:
      - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/primary:/cache/vllm
      - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/primary-flashinfer:/cache/flashinfer
  vllm-hand:
    environment:
      - MG_LOG_NAME=hand
"""


@pytest.fixture
def no_docker(monkeypatch):
    events: list[str] = []

    def fake_run(argv, **kw):
        events.append(" ".join(argv[2:]))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_compose, "_run", fake_run)
    monkeypatch.setattr(_compose, "_probe", lambda *a, **k: None)
    monkeypatch.setattr(_compose, "_cleared_after_down", False)
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: None)
    return events


def _deploy(tmp_path: Path) -> Path:
    d = tmp_path / "dep"
    d.mkdir()
    (d / "docker-compose.yml").write_text(_COMPOSE, encoding="utf-8")
    return d


# --- #1: ensure_log_dir creates lane-state/ as the operator -------------------


def test_ensure_log_dir_creates_lane_state_group_writable(tmp_path):
    d = _deploy(tmp_path)
    log_dir = _compose.ensure_log_dir(d)
    state = log_dir / _compose.LANE_STATE_DIRNAME
    assert state.is_dir()
    assert state.stat().st_uid == os.getuid()
    assert stat.S_IMODE(state.stat().st_mode) & 0o070 == 0o070  # group rwx


def test_ensure_log_dir_honours_configured_log_dir(tmp_path):
    d = _deploy(tmp_path)
    _compose.ensure_log_dir(d, "elsewhere")
    assert (d / "elsewhere" / "lane-state").is_dir()


def test_ensure_log_dir_never_rechmods_an_existing_lane_state(tmp_path):
    d = _deploy(tmp_path)
    state = d / "logs" / "lane-state"
    state.mkdir(parents=True)
    state.chmod(0o700)
    _compose.ensure_log_dir(d)
    assert stat.S_IMODE(state.stat().st_mode) == 0o700


def test_ensure_log_dir_chowns_created_dirs_to_the_sudo_user(tmp_path, monkeypatch):
    d = _deploy(tmp_path)
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_GID", "5678")
    monkeypatch.setattr(os, "chown", lambda p, u, g: calls.append((str(p), u, g)))
    _compose.ensure_log_dir(d)
    chowned = {p for p, _, _ in calls}
    assert str(d / "logs") in chowned
    assert str(d / "logs" / "lane-state") in chowned
    assert {(u, g) for _, u, g in calls} == {(1234, 5678)}


def test_ensure_log_dir_never_chowns_as_a_user(tmp_path, monkeypatch):
    d = _deploy(tmp_path)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(os, "chown", lambda *a: pytest.fail("chown as non-root"))
    _compose.ensure_log_dir(d)


# --- #1: a failed counter reset is reported, never silent ----------------------


def _counters(d: Path, *names: str) -> Path:
    state = d / "logs" / "lane-state"
    state.mkdir(parents=True, exist_ok=True)
    for n in names:
        (state / f"{n}.starts").write_text("1\n", encoding="utf-8")
    return state


def _deny_unlink(monkeypatch, state: Path):
    real = Path.unlink

    def unlink(self, *a, **k):
        if self.parent == state:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", unlink)


def test_failed_reset_prints_one_line_with_remediation(tmp_path, monkeypatch, capsys):
    d = _deploy(tmp_path)
    state = _counters(d, "primary", "hand")
    _deny_unlink(monkeypatch, state)
    assert _compose.reset_start_counters(d) == []
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "primary.starts" in err and "hand.starts" in err
    assert f"sudo rm {state / 'hand.starts'}" in err
    assert "lobes doctor" in err
    assert "held" in err


def test_missing_counter_is_not_a_failure(tmp_path, capsys):
    d = _deploy(tmp_path)
    _counters(d)
    assert _compose.reset_start_counters(d, ["vllm-hand"]) == []
    assert capsys.readouterr().err == ""


def test_targeted_up_reports_a_root_owned_counter(tmp_path, monkeypatch, capsys, no_docker):
    d = _deploy(tmp_path)
    state = _counters(d, "hand")
    _deny_unlink(monkeypatch, state)
    _compose.run_compose(d, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    err = capsys.readouterr().err
    assert "could not reset" in err and "hand.starts" in err
    assert no_docker == ["up -d --no-deps vllm-hand"]  # the verb still runs


# --- #1: doctor lane_state_writable ---------------------------------------------


def test_doctor_lane_state_writable_passes(tmp_path):
    d = _deploy(tmp_path)
    _counters(d)
    c = ls.lane_state_writable_check(d)
    assert c["id"] == "lane_state_writable" and c["passed"] is True


def test_doctor_lane_state_writable_skips_when_absent(tmp_path):
    assert ls.lane_state_writable_check(_deploy(tmp_path)) is None


def test_doctor_lane_state_writable_warns_when_not_writable(tmp_path, monkeypatch):
    d = _deploy(tmp_path)
    state = _counters(d)
    real = os.access
    monkeypatch.setattr(
        os, "access", lambda p, m, *a, **k: False if Path(p) == state else real(p, m, *a, **k)
    )
    c = ls.lane_state_writable_check(d)
    assert c["passed"] is False and c["severity"] == "warn"
    assert str(state) in c["message"]
    assert "sudo chown" in c["remediation"]


def test_doctor_registers_lane_state_writable(tmp_path):
    d = _deploy(tmp_path)
    _counters(d)
    assert "lane_state_writable" in {c["id"] for c in ls.lane_safety_checks(d)}


# --- #7: every lifecycle start creates the bind sources -------------------------


def test_compose_up_creates_compile_cache_and_lane_state(tmp_path, no_docker):
    d = _deploy(tmp_path)
    _compose.compose_up_detached(d)
    assert (d / "compile-cache" / "primary").is_dir()
    assert (d / "compile-cache" / "primary-flashinfer").is_dir()
    assert (d / "logs" / "lane-state").is_dir()


def test_compose_up_build_creates_compile_cache(tmp_path, no_docker):
    d = _deploy(tmp_path)
    _compose.compose_up_build(d)
    assert (d / "compile-cache" / "primary-flashinfer").is_dir()


def test_run_compose_up_creates_compile_cache(tmp_path, no_docker):
    d = _deploy(tmp_path)
    _compose.run_compose(d, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    assert (d / "compile-cache" / "primary").is_dir()


def test_run_compose_stop_creates_nothing(tmp_path, no_docker):
    d = _deploy(tmp_path)
    _compose.run_compose(d, ["docker", "compose", "stop", "vllm-hand"])
    assert not (d / "compile-cache").exists()


@pytest.mark.parametrize(
    "argv",
    [["serve"], ["fleet", "up"], ["switch", "Qwen/Qwen3.5-4B"], ["up", "hand"]],
)
def test_lifecycle_dry_run_creates_no_bind_source(tmp_path, capsys, no_docker, argv):
    d = tmp_path / "dep"
    _compose.write_scaffold(d, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    main([*argv, "--compose-dir", str(d)])
    capsys.readouterr()
    assert not (d / "compile-cache").exists()
    assert not (d / "logs" / "lane-state").exists()
    assert no_docker == []


def test_switch_apply_creates_compile_cache(tmp_path, monkeypatch, capsys, no_docker):
    from lobes.cli._commands import switch
    from lobes.runtime import _health

    d = tmp_path / "dep"
    _compose.write_scaffold(d, force=True)
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    monkeypatch.setattr(switch._runtime_ops, "probe_tool_calling", lambda *a, **k: None)
    main(["switch", "Qwen/Qwen3.5-4B", "--apply", "--force", "--compose-dir", str(d)])
    capsys.readouterr()
    assert (d / "compile-cache").is_dir()
    assert (d / "logs" / "lane-state").is_dir()


# --- review2-b #8: _before_up is the ONLY bind-source chokepoint --------------


@pytest.mark.parametrize(
    "argv,stub",
    [
        (["serve", "--no-probe"], "compose_up_detached"),
        (["fleet", "up"], "compose_up_build"),
        (["up", "hand"], "run_compose"),
    ],
)
def test_verbs_leave_bind_sources_to_the_chokepoint(tmp_path, monkeypatch, capsys, argv, stub):
    """No verb calls ensure_log_dir itself: the compose helper's _before_up does."""
    from lobes.runtime import _health

    d = tmp_path / "dep"
    _compose.write_scaffold(d, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    monkeypatch.setattr(
        _compose, "ensure_log_dir", lambda *a, **k: pytest.fail("verb bypassed _before_up")
    )
    ok = types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(_compose, stub, lambda *a, **k: ok)
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    monkeypatch.setattr("lobes.cli._commands.up.trigger_reannounce", lambda *a, **k: None)
    assert main([*argv, "--apply", "--compose-dir", str(d)]) == 0
    capsys.readouterr()
