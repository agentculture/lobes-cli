"""Tests for the lobes host watcher (lane-safety contract section 5) and
``lobes watcher install|uninstall|status|run``.

Nothing here touches real systemd, sudo or docker: the watcher takes an
injectable clear/clock/log, and the installer takes an injectable root path
and command runner.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from lobes.cli import main
from lobes.runtime import _machine
from lobes.runtime._machine import MachineResult
from lobes.watcher import _install, _run

# --------------------------------------------------------------------------
# helpers


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _marker(state_dir: Path, lane: str, since: int = 999_990, **extra) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    p = state_dir / f"{lane}.gated"
    body = {"lane": lane, "since": since, "need_kib": 57671680, "avail_kib": 1048576}
    body.update(extra)
    p.write_text(json.dumps(body))
    return p


def _watcher(tmp_path: Path, clear, clock: Clock, logs: list[str], **kw):
    return _run.Watcher(
        state_dir=tmp_path / "logs" / "lane-state",
        state_file=tmp_path / ".lobes-watcher-state.json",
        min_interval_s=kw.pop("min_interval_s", 300),
        stale_after_s=kw.pop("stale_after_s", 1800),
        clear=clear,
        clock=clock,
        log=logs.append,
    )


class CountingClear:
    def __init__(self, result: MachineResult | None = None) -> None:
        self.calls = 0
        self.result = result or MachineResult(True, None, {"cleared": True}, "machine-cli")

    def __call__(self) -> MachineResult:
        self.calls += 1
        return self.result


# --------------------------------------------------------------------------
# the watcher loop


def test_no_marker_no_clear(tmp_path):
    clear, logs = CountingClear(), []
    w = _watcher(tmp_path, clear, Clock(), logs)
    assert w.tick() == []
    assert clear.calls == 0


def test_marker_triggers_one_clear_per_window(tmp_path):
    clock, clear, logs = Clock(), CountingClear(), []
    _marker(tmp_path / "logs" / "lane-state", "worker")
    w = _watcher(tmp_path, clear, clock, logs)

    actions = w.tick()
    assert clear.calls == 1
    assert actions[0]["action"] == "cleared"
    assert actions[0]["lane"] == "worker"
    assert any("worker" in line and "cleared" in line and "57671680" in line for line in logs)

    # still gated, inside the window: no second clear
    for dt in (10, 100, 299):
        clock.t = 1_000_000.0 + dt
        acts = w.tick()
        assert acts[0]["action"] == "rate-limited"
    assert clear.calls == 1

    # window elapsed: one more clear
    clock.t = 1_000_000.0 + 300
    assert w.tick()[0]["action"] == "cleared"
    assert clear.calls == 2


def test_rate_limit_survives_restart(tmp_path):
    clock, clear, logs = Clock(), CountingClear(), []
    _marker(tmp_path / "logs" / "lane-state", "worker")
    _watcher(tmp_path, clear, clock, logs).tick()
    assert clear.calls == 1
    clock.t += 50
    # a fresh watcher (process restart) reads the persisted last attempt
    _watcher(tmp_path, clear, clock, logs).tick()
    assert clear.calls == 1


def test_two_markers_share_one_clear(tmp_path):
    clear, logs = CountingClear(), []
    sd = tmp_path / "logs" / "lane-state"
    _marker(sd, "worker")
    _marker(sd, "primary")
    actions = _watcher(tmp_path, clear, Clock(), logs).tick()
    assert clear.calls == 1
    assert sorted(a["lane"] for a in actions) == ["primary", "worker"]


def test_no_grant_logs_only(tmp_path):
    clear = CountingClear(MachineResult(False, "not_granted"))
    logs: list[str] = []
    _marker(tmp_path / "logs" / "lane-state", "worker")
    actions = _watcher(tmp_path, clear, Clock(), logs).tick()
    assert actions[0]["action"] == "skipped:not_granted"
    assert any("skipped:not_granted" in line for line in logs)
    state = json.loads((tmp_path / ".lobes-watcher-state.json").read_text())
    assert state["last_clear"] is None


def test_stale_marker_is_ignored_and_logged_once(tmp_path):
    """review-b #3b: a marker older than 2x the gate timeout is not acted on."""
    clear, logs = CountingClear(), []
    _marker(tmp_path / "logs" / "lane-state", "worker", since=1)
    w = _watcher(tmp_path, clear, Clock(), logs, min_interval_s=0)
    for _ in range(3):
        actions = w.tick()
    assert actions[0]["stale"] is True and actions[0]["action"] == "ignored:stale"
    assert clear.calls == 0
    assert sum("stale" in line for line in logs) == 1


def test_stale_marker_beside_a_fresh_one_still_clears_once(tmp_path):
    clear, logs = CountingClear(), []
    sd = tmp_path / "logs" / "lane-state"
    _marker(sd, "old", since=1)
    _marker(sd, "worker")
    acts = {a["lane"]: a["action"] for a in _watcher(tmp_path, clear, Clock(), logs).tick()}
    assert acts == {"old": "ignored:stale", "worker": "cleared"}
    assert clear.calls == 1


def test_malformed_marker_ages_by_mtime(tmp_path):
    sd = tmp_path / "logs" / "lane-state"
    sd.mkdir(parents=True)
    bad = sd / "bad.gated"
    bad.write_text("{not json")
    import os as _os

    _os.utime(bad, (1, 1))
    clear, logs = CountingClear(), []
    actions = _watcher(tmp_path, clear, Clock(), logs).tick()
    assert actions[0]["action"] == "ignored:stale" and clear.calls == 0


def test_marker_whose_lane_has_no_container_is_stale(tmp_path):
    clear, logs = CountingClear(), []
    _marker(tmp_path / "logs" / "lane-state", "worker")
    w = _watcher(tmp_path, clear, Clock(), logs)
    w.health = lambda: {"primary": "running"}  # worker's container is gone
    actions = w.tick()
    assert actions[0]["action"] == "ignored:stale" and actions[0]["stale"] is True
    assert clear.calls == 0
    assert any("no running container" in line for line in logs)


def test_clear_deferred_while_another_lane_is_starting(tmp_path):
    """review-b #3a: dropping the cache mid-boot can trip vLLM's profiling."""
    clock, clear, logs = Clock(), CountingClear(), []
    _marker(tmp_path / "logs" / "lane-state", "worker")
    w = _watcher(tmp_path, clear, clock, logs)
    w.health = lambda: {"worker": "starting", "primary": "starting"}
    for _ in range(3):
        actions = w.tick()
    assert actions[0]["action"] == "deferred:lane-starting"
    assert clear.calls == 0
    assert sum("primary" in line and "deferred" in line for line in logs) == 1
    state = tmp_path / ".lobes-watcher-state.json"
    assert not state.exists()  # a deferral is not an attempt: no rate-limit spent
    w.health = lambda: {"worker": "starting", "primary": "running"}
    assert w.tick()[0]["action"] == "cleared"
    assert clear.calls == 1


def test_gated_lane_own_starting_status_does_not_defer(tmp_path):
    clear, logs = CountingClear(), []
    _marker(tmp_path / "logs" / "lane-state", "worker")
    w = _watcher(tmp_path, clear, Clock(), logs)
    w.health = lambda: {"worker": "starting", "embed": "running"}
    assert w.tick()[0]["action"] == "cleared"


def test_unreadable_docker_falls_back_to_marker_age_only(tmp_path):
    clear, logs = CountingClear(), []
    _marker(tmp_path / "logs" / "lane-state", "worker")
    w = _watcher(tmp_path, clear, Clock(), logs)
    w.health = lambda: None
    assert w.tick()[0]["action"] == "cleared"


def test_malformed_and_symlinked_markers_are_ignored(tmp_path):
    sd = tmp_path / "logs" / "lane-state"
    sd.mkdir(parents=True)
    (sd / "bad.gated").write_text("{not json")
    target = tmp_path / "elsewhere.json"
    target.write_text("{}")
    (sd / "link.gated").symlink_to(target)
    markers = _run.scan_markers(sd)
    assert [m["lane"] for m in markers] == ["bad"]
    assert markers[0]["malformed"] is True


def test_watcher_runs_no_mutating_docker_command(tmp_path, monkeypatch):
    """Drive the REAL drop_caches adapter + lane-health read with every subprocess
    recorded. The only docker argv allowed is the fixed, read-only ``docker ps``
    (review-b #3 narrowed the old "never runs docker" invariant to this)."""
    from lobes.runtime import _compose

    seen: list[list[str]] = []

    def fake_machine_run(argv, *, stdin=None):
        seen.append(list(argv))
        return _machine._Proc(1, "", "a password is required")

    def fake_subprocess_run(argv, *a, **k):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 1, "", "")

    def fake_probe(argv, **k):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "vllm-worker\tmodel-gear-vllm-worker\tUp\n", "")

    monkeypatch.setattr(_machine, "_run", fake_machine_run)
    monkeypatch.setattr(_machine, "_current_cli", lambda: None)
    monkeypatch.setattr(_compose, "_probe", fake_probe)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(subprocess, "Popen", lambda argv, *a, **k: seen.append(list(argv)))

    deploy = tmp_path
    (deploy / "docker-compose.yml").write_text(
        "services:\n  vllm-worker:\n    environment:\n      - MG_LOG_NAME=worker\n"
    )
    _marker(deploy / "logs" / "lane-state", "worker", since=int(Clock()()))
    w = _run.watcher_for(deploy, clock=Clock(), log=[].append)
    w.min_interval_s = 0
    for _ in range(3):
        w.tick()
    assert seen, "the adapter should have probed the sudo grant"
    docker = [a for a in seen if a and Path(a[0]).name == "docker"]
    assert docker, "the lane-health read should have run"
    assert all(a == _compose.LANE_HEALTH_ARGV for a in docker)
    assert _compose.LANE_HEALTH_ARGV[:2] == ["docker", "ps"]
    assert not any(
        verb in _compose.LANE_HEALTH_ARGV
        for verb in ("stop", "start", "restart", "kill", "rm", "update", "pause", "up", "down")
    )
    assert not any("docker" in " ".join(a) for a in seen if a not in docker)


def test_run_once_cli(tmp_path, monkeypatch, capsys):
    deploy = tmp_path / "dep"
    deploy.mkdir()
    (deploy / "docker-compose.yml").write_text("services: {}\n")
    _marker(deploy / "logs" / "lane-state", "worker", since=int(time.time()))
    clear = CountingClear()
    monkeypatch.setattr(_run, "drop_caches", clear)
    rc = main(["watcher", "run", "--compose-dir", str(deploy), "--once", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["actions"][0]["action"] == "cleared"
    assert clear.calls == 1


def test_min_interval_knob(tmp_path, monkeypatch):
    deploy = tmp_path / "dep"
    deploy.mkdir()
    (deploy / ".env").write_text("LOBES_WATCHER_MIN_INTERVAL_S=42\nLOBES_MEM_GATE_TIMEOUT_S=60\n")
    monkeypatch.delenv("LOBES_WATCHER_MIN_INTERVAL_S", raising=False)
    w = _run.watcher_for(deploy)
    assert w.min_interval_s == 42
    assert w.stale_after_s == 120
    monkeypatch.setenv("LOBES_WATCHER_MIN_INTERVAL_S", "7")
    assert _run.watcher_for(deploy).min_interval_s == 7
    monkeypatch.setenv("LOBES_WATCHER_MIN_INTERVAL_S", "junk")
    assert _run.watcher_for(deploy).min_interval_s == 300


# --------------------------------------------------------------------------
# install / uninstall / status on a fake root


class FakeRunner:
    def __init__(self, *, linger: str = "yes", sudo_ok: bool = True, active: str = "active"):
        self.calls: list[list[str]] = []
        self.linger = linger
        self.sudo_ok = sudo_ok
        self.active = active

    def __call__(self, argv: list[str]) -> tuple[int, str, str]:
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] == ["loginctl", "show-user"]:
            return 0, f"Linger={self.linger}\n", ""
        if argv[:2] == ["sudo", "-n"]:
            if not self.sudo_ok:
                return 1, "", "sudo: a password is required"
            inner = argv[2:]
            if inner[:1] == ["install"]:
                src, dst = inner[-2], inner[-1]
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text(Path(src).read_text())
            elif inner[:2] == ["rm", "-f"]:
                Path(inner[2]).unlink(missing_ok=True)
            elif inner[:1] == ["systemctl"] and "is-active" in inner:
                return 0, f"{self.active}\n", ""
            return 0, "", ""
        if "is-active" in argv:
            return (0 if self.active == "active" else 3), f"{self.active}\n", ""
        if "is-enabled" in argv:
            return 0, "enabled\n", ""
        return 0, "", ""


def _deploy(tmp_path: Path) -> Path:
    d = tmp_path / "home" / "u" / ".lobes"
    d.mkdir(parents=True)
    (d / "docker-compose.yml").write_text("services: {}\n")
    return d


def _ctx(tmp_path, runner, **kw):
    return _install.InstallContext(
        root=tmp_path / "root",
        home=Path("/home/u"),
        user="u",
        runner=runner,
        exec_argv=["/opt/bin/lobes"],
        path_env="/home/u/.local/bin:/usr/bin",
        **kw,
    )


def test_install_plan_is_dry_run(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    plan = _install.plan_install(deploy, _ctx(tmp_path, runner))
    assert plan.unit_type == "user"
    assert plan.unit_path == tmp_path / "root" / "home/u/.config/systemd/user/lobes-watcher.service"
    assert "ExecStart=/opt/bin/lobes watcher run --compose-dir" in plan.content
    assert str(deploy) in plan.content
    assert "Restart=on-failure" in plan.content
    assert "Environment=PATH=/home/u/.local/bin:/usr/bin" in plan.content
    assert "@" not in plan.content.split("[Unit]", 1)[1]
    assert "WantedBy=default.target" in plan.content
    assert plan.blockers == []
    assert not plan.unit_path.exists()
    assert not (deploy / _install.MANIFEST).exists()
    # only the read-only linger probe ran
    assert all(c[0] == "loginctl" for c in runner.calls)


def test_install_apply_uninstall_round_trip(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    ctx = _ctx(tmp_path, runner)
    plan = _install.plan_install(deploy, ctx)
    result = _install.apply_install(plan, ctx)
    unit = plan.unit_path
    assert unit.is_file()
    manifest = json.loads((deploy / _install.MANIFEST).read_text())
    assert manifest["unit_type"] == "user"
    assert manifest["files"] == [str(unit)]
    assert result["files"] == [str(unit)]
    assert ["systemctl", "--user", "daemon-reload"] in runner.calls
    assert ["systemctl", "--user", "enable", "--now", "lobes-watcher.service"] in runner.calls

    # an unrelated file in the same dir must survive uninstall
    neighbour = unit.parent / "other.service"
    neighbour.write_text("x")

    st = _install.status(deploy, ctx)
    assert st["installed"] is True and st["active"] == "active" and st["linger"] is True

    uplan = _install.plan_uninstall(deploy, ctx)
    assert uplan["files"] == [str(unit)]
    assert unit.exists()  # plan only
    _install.apply_uninstall(deploy, ctx)
    assert not unit.exists()
    assert neighbour.exists()
    assert not (deploy / _install.MANIFEST).exists()
    assert ["systemctl", "--user", "disable", "--now", "lobes-watcher.service"] in runner.calls
    assert _install.status(deploy, ctx)["installed"] is False
    assert not any(c and c[0] == "docker" for c in runner.calls)


def test_install_refuses_without_linger(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner(linger="no")
    ctx = _ctx(tmp_path, runner)
    plan = _install.plan_install(deploy, ctx)
    assert plan.blockers and "enable-linger u" in plan.blockers[0]
    with pytest.raises(Exception) as exc:
        _install.apply_install(plan, ctx)
    assert "linger" in str(exc.value).lower()
    assert not plan.unit_path.exists()


def test_system_install_via_sudo_n(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner(linger="no")
    ctx = _ctx(tmp_path, runner, system=True)
    plan = _install.plan_install(deploy, ctx)
    assert plan.unit_type == "system"
    assert plan.blockers == []
    assert "User=u" in plan.content and "WantedBy=multi-user.target" in plan.content
    _install.apply_install(plan, ctx)
    assert plan.unit_path == tmp_path / "root" / "etc/systemd/system/lobes-watcher.service"
    assert plan.unit_path.is_file()
    installs = [c for c in runner.calls if c[:3] == ["sudo", "-n", "install"]]
    assert installs and installs[0][-1] == str(plan.unit_path)
    assert ["sudo", "-n", "systemctl", "enable", "--now", "lobes-watcher.service"] in runner.calls
    _install.apply_uninstall(deploy, ctx)
    assert not plan.unit_path.exists()
    assert ["sudo", "-n", "rm", "-f", str(plan.unit_path)] in runner.calls


def test_system_install_refuses_without_passwordless_sudo(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner(sudo_ok=False)
    ctx = _ctx(tmp_path, runner, system=True)
    plan = _install.plan_install(deploy, ctx)
    with pytest.raises(Exception) as exc:
        _install.apply_install(plan, ctx)
    assert "sudo" in str(exc.value)
    assert not (deploy / _install.MANIFEST).exists()


def test_uninstall_only_removes_unit_named_files(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    victim = tmp_path / "precious.txt"
    victim.write_text("keep")
    (deploy / _install.MANIFEST).write_text(
        json.dumps({"unit_type": "user", "unit": "lobes-watcher.service", "files": [str(victim)]})
    )
    _install.apply_uninstall(deploy, _ctx(tmp_path, runner))
    assert victim.exists()


def test_install_already_installed_refuses(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    ctx = _ctx(tmp_path, runner)
    _install.apply_install(_install.plan_install(deploy, ctx), ctx)
    plan = _install.plan_install(deploy, ctx)
    assert any("already installed" in b for b in plan.blockers)


def test_template_is_not_scaffolded_by_init():
    from lobes.runtime import _compose

    for tmpl in (_compose.SINGLE_TEMPLATES, _compose.FLEET_TEMPLATES, _compose.AUDIO_TEMPLATES):
        assert _install.UNIT_TEMPLATE not in tmpl
        assert _install.UNIT_TEMPLATE not in tmpl.values()


def test_cli_install_dry_run_default(tmp_path, monkeypatch, capsys):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    monkeypatch.setattr(
        _install, "default_context", lambda system=False: _ctx(tmp_path, runner, system=system)
    )
    rc = main(["watcher", "install", "--compose-dir", str(deploy), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert not (deploy / _install.MANIFEST).exists()

    rc = main(["watcher", "install", "--compose-dir", str(deploy), "--apply", "--json"])
    assert rc == 0
    assert (deploy / _install.MANIFEST).exists()

    rc = main(["watcher", "status", "--compose-dir", str(deploy), "--json"])
    st = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert st["installed"] is True

    rc = main(["watcher", "uninstall", "--compose-dir", str(deploy), "--apply", "--json"])
    assert rc == 0
    assert not (deploy / _install.MANIFEST).exists()


def test_cli_install_no_linger_apply_errors(tmp_path, monkeypatch, capsys):
    deploy, runner = _deploy(tmp_path), FakeRunner(linger="no")
    monkeypatch.setattr(
        _install, "default_context", lambda system=False: _ctx(tmp_path, runner, system=system)
    )
    rc = main(["watcher", "install", "--compose-dir", str(deploy), "--apply"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "enable-linger" in err
