"""``lobes site install|uninstall|status`` — the reboot-surviving user unit.

Everything runs against a fake root and a fake runner; real systemd is never
touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from lobes.cli._commands import site as site_cmd
from lobes.cli._errors import ModelGearError
from lobes.site import _service
from lobes.watcher._install import InstallContext


class FakeRunner:
    def __init__(self, linger: str = "yes", active: str = "active"):
        self.linger = linger
        self.active = active
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if argv[:2] == ["loginctl", "show-user"]:
            return 0, f"Linger={self.linger}\n", ""
        if "is-active" in argv:
            return 0, f"{self.active}\n", ""
        if "is-enabled" in argv:
            return 0, "enabled\n", ""
        return 0, "", ""


def _deploy(tmp_path: Path) -> Path:
    d = tmp_path / "deploy"
    d.mkdir()
    (d / "docker-compose.yml").write_text("services: {}\n")
    return d


def _ctx(tmp_path, runner) -> InstallContext:
    return InstallContext(
        root=tmp_path / "root",
        home=Path("/home/u"),
        user="u",
        runner=runner,
        exec_argv=["/opt/bin/lobes"],
        path_env="/home/u/.local/bin:/usr/bin",
    )


def test_plan_is_read_only_and_renders_restart_always(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    plan = _service.plan_install(deploy, _ctx(tmp_path, runner), host="100.1.2.3", port=9490)
    assert not plan.unit_path.exists()
    assert not (deploy / _service.MANIFEST).exists()
    assert all(c[0] == "loginctl" for c in runner.calls)
    assert plan.blockers == []
    assert plan.unit_path == tmp_path / "root/home/u/.config/systemd/user/lobes-site.service"
    assert "Restart=always" in plan.content
    assert "WantedBy=default.target" in plan.content
    assert (
        f"ExecStart=/opt/bin/lobes site --compose-dir {deploy} "
        "--host 100.1.2.3 --port 9490 --apply"
    ) in plan.content
    assert 'Environment="PATH=' not in plan.content  # no whitespace -> unquoted
    assert "Environment=PATH=/home/u/.local/bin:/usr/bin" in plan.content


def test_gateway_is_carried_into_the_unit(tmp_path):
    plan = _service.plan_install(
        _deploy(tmp_path),
        _ctx(tmp_path, FakeRunner()),
        host="127.0.0.1",
        port=9490,
        gateway="http://127.0.0.1:8000",
    )
    assert "--gateway http://127.0.0.1:8000 --apply" in plan.content


@pytest.mark.parametrize("linger", ["no", "garbage"])
def test_no_linger_refuses(tmp_path, linger):
    ctx = _ctx(tmp_path, FakeRunner(linger=linger))
    plan = _service.plan_install(_deploy(tmp_path), ctx, host="127.0.0.1", port=9490)
    assert any("enable-linger u" in b for b in plan.blockers)
    with pytest.raises(ModelGearError):
        _service.apply_install(plan, ctx)
    assert not plan.unit_path.exists()


def test_port_zero_refuses(tmp_path):
    plan = _service.plan_install(
        _deploy(tmp_path), _ctx(tmp_path, FakeRunner()), host="127.0.0.1", port=0
    )
    assert any("--port 0" in b for b in plan.blockers)


def test_apply_writes_unit_manifest_and_enables(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    ctx = _ctx(tmp_path, runner)
    plan = _service.plan_install(deploy, ctx, host="127.0.0.1", port=9490)
    result = _service.apply_install(plan, ctx)
    assert plan.unit_path.read_text() == plan.content
    manifest = json.loads((deploy / _service.MANIFEST).read_text())
    assert manifest["files"] == [str(plan.unit_path)]
    assert result["unit"] == "lobes-site.service"
    sysctl = [c for c in runner.calls if c[0] == "systemctl"]
    assert sysctl == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "lobes-site.service"],
        ["systemctl", "--user", "restart", "lobes-site.service"],
    ]


def test_reinstall_rewrites_with_new_flags(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    ctx = _ctx(tmp_path, runner)
    _service.apply_install(_service.plan_install(deploy, ctx, host="127.0.0.1", port=9490), ctx)
    plan2 = _service.plan_install(deploy, ctx, host="100.1.2.3", port=9491)
    assert plan2.replaces and plan2.blockers == []
    _service.apply_install(plan2, ctx)
    assert "--host 100.1.2.3 --port 9491" in plan2.unit_path.read_text()


def test_uninstall_removes_only_listed_unit(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    ctx = _ctx(tmp_path, runner)
    plan = _service.plan_install(deploy, ctx, host="127.0.0.1", port=9490)
    _service.apply_install(plan, ctx)
    stray = tmp_path / "other.service"
    stray.write_text("keep")
    m = json.loads((deploy / _service.MANIFEST).read_text())
    m["files"].append(str(stray))
    (deploy / _service.MANIFEST).write_text(json.dumps(m))
    out = _service.apply_uninstall(deploy, ctx)
    assert out["removed"] == [str(plan.unit_path)]
    assert not plan.unit_path.exists() and stray.exists()
    assert not (deploy / _service.MANIFEST).exists()
    assert ["systemctl", "--user", "disable", "--now", "lobes-site.service"] in runner.calls


def test_uninstall_when_absent_is_noop(tmp_path):
    out = _service.apply_uninstall(_deploy(tmp_path), _ctx(tmp_path, FakeRunner()))
    assert out == {"installed": False, "files": [], "commands": []}


def test_status_reports_active_enabled(tmp_path):
    deploy, runner = _deploy(tmp_path), FakeRunner()
    ctx = _ctx(tmp_path, runner)
    assert _service.status(deploy, ctx)["installed"] is False
    _service.apply_install(_service.plan_install(deploy, ctx, host="127.0.0.1", port=9490), ctx)
    st = _service.status(deploy, ctx)
    assert st["installed"] and st["active"] == "active" and st["enabled"] == "enabled"
    assert st["files_present"] and st["linger"] is True


def _args(tmp_path, **kw):
    base = dict(
        action="serve",
        host="127.0.0.1",
        port=9490,
        gateway=None,
        compose_dir=str(_deploy(tmp_path)),
        apply=False,
        json=True,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_install_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    runner = FakeRunner()
    monkeypatch.setattr(_service, "default_context", lambda: _ctx(tmp_path, runner))
    args = _args(tmp_path, action="install")
    assert site_cmd.cmd_site(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["unit_type"] == "user"
    assert not Path(out["unit_path"]).exists()
    assert not any(c[0] == "systemctl" for c in runner.calls)


def test_cli_install_apply_then_status(tmp_path, monkeypatch, capsys):
    runner = FakeRunner()
    monkeypatch.setattr(_service, "default_context", lambda: _ctx(tmp_path, runner))
    args = _args(tmp_path, action="install", apply=True)
    assert site_cmd.cmd_site(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is False and Path(out["files"][0]).exists()
    args.action, args.apply = "status", False
    assert site_cmd.cmd_site(args) == 0
    assert json.loads(capsys.readouterr().out)["active"] == "active"


def test_bare_site_still_parses_as_serve():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    site_cmd.register(sub)
    assert parser.parse_args(["site", "--apply"]).action == "serve"
    assert parser.parse_args(["site", "install", "--port", "9490"]).action == "install"
