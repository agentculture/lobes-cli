"""Tests for the graded thermal governor (``lobes governor run``).

Nothing here touches real docker or real thermal zones: the governor takes an
injectable docker runner, thermal reader, log sink and clock, and the CLI test
points LOBES_SYS_ROOT at a fake sysfs and puts a stub ``docker`` on PATH.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from lobes.cli import main
from lobes.runtime import _governor as g

BANDS = g.Bands()


# --------------------------------------------------------------------------
# the pure band function


@pytest.mark.parametrize(
    "temp,cpus",
    [
        (50.0, 16),
        (83.9, 16),
        (84.0, 12),
        (87.9, 12),
        (88.0, 8),
        (90.9, 8),
        (91.0, 4),
        (93.4, 4),
    ],
)
def test_band_boundaries_running(temp, cpus):
    d = g.decide(temp, paused=False, bands=BANDS)
    assert (d.kind, d.cpus) == ("cpus", cpus)


def test_pause_at_935():
    assert g.decide(93.5, paused=False, bands=BANDS).kind == "pause"
    assert g.decide(99.0, paused=False, bands=BANDS).kind == "pause"


def test_paused_hysteresis():
    for t in (93.5, 90.0, 86.0, 86.1):
        assert g.decide(t, paused=True, bands=BANDS).kind == "hold"
    d = g.decide(85.9, paused=True, bands=BANDS)
    assert (d.kind, d.cpus) == ("resume", 4)


def test_custom_bands_parse():
    b = g.parse_bands("70:8,80:4", pause_c=90.0, resume_c=75.0, resume_cpus=2, top_cpus=16)
    assert g.decide(69.9, paused=False, bands=b).cpus == 16
    assert g.decide(70.0, paused=False, bands=b).cpus == 8
    assert g.decide(80.0, paused=False, bands=b).cpus == 4
    assert g.decide(90.0, paused=False, bands=b).kind == "pause"
    assert g.decide(74.0, paused=True, bands=b).cpus == 2


def test_parse_bands_rejects_garbage():
    with pytest.raises(ValueError):
        g.parse_bands("80:4,70:8", pause_c=90, resume_c=75, resume_cpus=2, top_cpus=16)
    with pytest.raises(ValueError):
        g.parse_bands("abc", pause_c=90, resume_c=75, resume_cpus=2, top_cpus=16)


# --------------------------------------------------------------------------
# the stateful governor against a fake docker


class FakeDocker:
    def __init__(self, status="running"):
        self.status = status
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        verb = argv[1]
        if verb == "inspect":
            return 0, self.status + "\n"
        if verb == "pause":
            self.status = "paused"
        if verb == "unpause":
            self.status = "running"
        return 0, ""

    def mutations(self):
        return [c for c in self.calls if c[1] != "inspect"]


def _gov(docker, temps, apply=True):
    it = iter(temps)
    lines: list[str] = []
    gv = g.Governor(
        "ctr",
        BANDS,
        docker=docker,
        read_temp=lambda: next(it),
        log=lines.append,
        apply=apply,
        now=lambda: "T",
    )
    return gv, lines


def test_steps_through_bands_and_logs_each_change():
    d = FakeDocker()
    gv, lines = _gov(d, [70, 71, 85, 89, 92, 92.5, 60])
    for _ in range(7):
        gv.tick()
    assert [c[3] for c in d.mutations()] == ["16", "12", "8", "4", "16"]
    assert len(lines) == 5  # unchanged ticks log nothing
    assert "cpus=12" in lines[1]
    assert "85.0" in lines[1]


def test_pause_then_resume_only_own_pause():
    d = FakeDocker()
    gv, lines = _gov(d, [94, 90, 85])
    gv.tick()
    assert d.status == "paused"
    gv.tick()
    assert d.status == "paused"
    gv.tick()
    assert d.status == "running"
    assert ["docker", "update", "--cpus", "4", "ctr"] in d.calls
    assert any("resume" in ln for ln in lines)


def test_foreign_pause_is_never_resumed():
    d = FakeDocker(status="paused")
    gv, lines = _gov(d, [50, 50, 50])
    for _ in range(3):
        gv.tick()
    assert d.status == "paused"
    assert d.mutations() == []
    assert sum("foreign" in ln for ln in lines) == 1  # noted once


def test_dry_run_changes_nothing():
    d = FakeDocker()
    gv, lines = _gov(d, [70, 94, 80], apply=False)
    for _ in range(3):
        gv.tick()
    assert d.mutations() == []
    assert lines
    assert all("dry-run" in ln for ln in lines)
    assert any("pause" in ln for ln in lines)
    assert any("resume" in ln for ln in lines)


def test_unknown_temperature_does_nothing():
    d = FakeDocker()
    gv, lines = _gov(d, [None])
    gv.tick()
    assert d.mutations() == []
    assert lines == []


def test_exited_container_ends_loop():
    d = FakeDocker(status="exited")
    gv, _ = _gov(d, [70])
    assert gv.tick() is False


def test_cleanup_restores_cap_and_unpauses_own_pause():
    d = FakeDocker()
    gv, lines = _gov(d, [94])
    gv.tick()
    gv.cleanup()
    assert d.status == "running"
    assert d.mutations() == [["docker", "pause", "ctr"], ["docker", "unpause", "ctr"]]
    assert any("unpaused (our own pause)" in ln for ln in lines)


def test_cleanup_restores_top_cap_after_a_throttle():
    d = FakeDocker()
    gv, lines = _gov(d, [92])
    gv.tick()
    gv.cleanup()
    assert d.mutations()[-1] == ["docker", "update", "--cpus", "16", "ctr"]
    assert any("restor" in ln for ln in lines)


def test_cleanup_leaves_foreign_pause():
    d = FakeDocker(status="paused")
    gv, _ = _gov(d, [50])
    gv.tick()
    gv.cleanup()
    assert d.status == "paused"
    assert not any(c[1] == "unpause" for c in d.calls)


def test_cleanup_noop_when_never_acted():
    d = FakeDocker()
    gv, _ = _gov(d, [])
    gv.cleanup()
    assert d.mutations() == []


# --------------------------------------------------------------------------
# thermal reading + CLI


def _fake_sys(root: Path, temps_mc: list[int]) -> None:
    for i, t in enumerate(temps_mc):
        z = root / "class" / "thermal" / f"thermal_zone{i}"
        z.mkdir(parents=True)
        (z / "type").write_text(f"zone{i}\n")
        (z / "temp").write_text(f"{t}\n")


def test_hottest_zone_reader(tmp_path, monkeypatch):
    _fake_sys(tmp_path, [55000, 88200, 70000])
    monkeypatch.setenv("LOBES_SYS_ROOT", str(tmp_path))
    monkeypatch.delenv("LOBES_THERMAL_ZONES", raising=False)
    assert g.read_hottest_c() == 88.2


def test_cli_once_dry_run(tmp_path, monkeypatch, capsys):
    _fake_sys(tmp_path / "sys", [89000])
    monkeypatch.setenv("LOBES_SYS_ROOT", str(tmp_path / "sys"))
    monkeypatch.delenv("LOBES_THERMAL_ZONES", raising=False)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "docker"
    stub.write_text(
        f'#!/bin/sh\necho "$@" >> {tmp_path}/calls\n[ "$1" = inspect ] && echo running\nexit 0\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    deploy = tmp_path / "deploy"
    (deploy / "logs").mkdir(parents=True)
    (deploy / "docker-compose.yml").write_text("services: {}\n")
    rc = main(
        ["governor", "run", "--container", "c1", "--once", "--json", "--compose-dir", str(deploy)]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["temp_c"] == 89.0
    assert out["action"] == "cpus=8"
    calls = (tmp_path / "calls").read_text().splitlines()
    assert all(c.startswith("inspect") for c in calls)
    assert not (deploy / "logs" / "governor.log").exists()


def test_cli_bad_bands_is_a_user_error(tmp_path, capsys):
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / "docker-compose.yml").write_text("services: {}\n")
    rc = main(
        ["governor", "run", "--container", "c", "--bands", "9:1,3:2", "--compose-dir", str(deploy)]
    )
    assert rc != 0
