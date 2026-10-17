"""Machine-CLI adapter (lane-safety t10): fake executables on a tmp PATH."""

from __future__ import annotations

import json
import os
import stat

import pytest

from lobes.runtime import _machine
from lobes.runtime._detect import DetectedCard


def _card(name):
    return DetectedCard(name, None, None, None, None, None)


@pytest.fixture
def shim(tmp_path, monkeypatch):
    """Create fake executables; each records argv(+stdin) to <name>.log."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "cat").symlink_to("/usr/bin/cat")  # shims need cat; nothing else
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(_machine.os, "sync", lambda: None)

    def make(name, stdout="", rc=0):
        p = bindir / name
        out = tmp_path / f"{name}.out"
        out.write_text(stdout)
        p.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "$*" >> "{tmp_path}/{name}.log"\n'
            f'cat >> "{tmp_path}/{name}.stdin"\n'
            f'cat "{out}"\n'
            f"exit {rc}\n"
        )
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return tmp_path / f"{name}.log"

    return make


def _set_card(monkeypatch, name):
    monkeypatch.setattr(_machine, "detect_card", lambda: _card(name))


@pytest.mark.parametrize("card", ["spark", "thor", "orin"])
def test_window_calls_the_card_cli(card, shim, monkeypatch, tmp_path):
    _set_card(monkeypatch, card)
    log = shim(card, json.dumps({"ok": True, "reason": None, "active": False}))
    r = _machine.window("show")
    assert r.ok and r.source == "machine-cli" and r.payload["active"] is False
    assert log.read_text().split() == ["window", "show", "--json"]
    assert _machine.machine_cli_for(_card(card).resolved) == card


def test_unknown_card_is_tool_missing(shim, monkeypatch):
    _set_card(monkeypatch, "unknown")
    assert _machine.machine_cli_for("unknown") is None
    r = _machine.window("show")
    assert not r.ok and r.reason == "tool_missing" and r.source is None


def test_missing_cli_is_tool_missing(shim, monkeypatch):
    _set_card(monkeypatch, "thor")
    r = _machine.window("lock")
    assert not r.ok and r.reason == "tool_missing"


def test_bad_action_rejected(shim, monkeypatch):
    _set_card(monkeypatch, "thor")
    shim("thor", "{}")
    assert _machine.window("rm -rf").reason == "bad_action"


def test_malformed_json_is_typed_failure(shim, monkeypatch):
    _set_card(monkeypatch, "orin")
    shim("orin", "not json")
    r = _machine.window("show")
    assert not r.ok and r.reason == "malformed_output"


def test_cli_failure_carries_reason_from_json(shim, monkeypatch):
    _set_card(monkeypatch, "thor")
    shim("thor", json.dumps({"ok": False, "reason": "not_granted"}), rc=2)
    r = _machine.window("lock")
    assert not r.ok and r.reason == "not_granted"


def test_cli_failure_without_json_reason(shim, monkeypatch):
    _set_card(monkeypatch, "thor")
    shim("thor", "", rc=2)
    r = _machine.window("lock")
    assert not r.ok and r.reason == "failed"


def test_drop_caches_prefers_machine_cli(shim, monkeypatch):
    _set_card(monkeypatch, "thor")
    log = shim("thor", json.dumps({"ok": True, "reason": None, "cleared": True}))
    sudo = shim("sudo", "")
    r = _machine.drop_caches()
    assert r.ok and r.source == "machine-cli"
    assert log.read_text().split() == ["drop-caches", "--json"]
    assert not sudo.exists()


def test_fallback_sysctl_rule(shim, monkeypatch):
    _set_card(monkeypatch, "spark")  # no spark binary on PATH
    listing = "    (root) NOPASSWD: /usr/sbin/sysctl -w vm.drop_caches\\=3\n"
    sudo = shim("sudo", listing)
    r = _machine.drop_caches()
    assert r.ok and r.source == "sudo-fallback" and r.payload["rule"] == "sysctl"
    calls = sudo.read_text().splitlines()
    assert calls[0] == "-n -l"
    assert calls[1] == "-n /usr/sbin/sysctl -w vm.drop_caches=3"


def test_fallback_tee_rule(shim, monkeypatch, tmp_path):
    _set_card(monkeypatch, "spark")
    sudo = shim("sudo", "    (root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches\n")
    r = _machine.drop_caches()
    assert r.ok and r.payload["rule"] == "tee"
    assert sudo.read_text().splitlines()[1] == "-n tee /proc/sys/vm/drop_caches"
    assert (tmp_path / "sudo.stdin").read_text().strip() == "3"


def test_fallback_nopasswd_all(shim, monkeypatch):
    _set_card(monkeypatch, "spark")
    sudo = shim("sudo", "    (ALL : ALL) NOPASSWD: ALL\n")
    r = _machine.drop_caches()
    assert r.ok and r.payload["rule"] == "all"
    assert sudo.read_text().splitlines()[1] == "-n /usr/sbin/sysctl -w vm.drop_caches=3"


def test_no_grant_never_runs_the_clear(shim, monkeypatch):
    _set_card(monkeypatch, "spark")
    sudo = shim("sudo", "    (root) NOPASSWD: /usr/bin/apt\n")
    r = _machine.drop_caches()
    assert not r.ok and r.reason == "not_granted" and r.source is None
    assert sudo.read_text().splitlines() == ["-n -l"]


def test_sudo_listing_fails_is_not_granted(shim, monkeypatch):
    _set_card(monkeypatch, "spark")
    shim("sudo", "", rc=1)
    assert _machine.drop_caches().reason == "not_granted"
    assert _machine.grant_status().granted is False


def test_no_sudo_binary_is_not_granted(shim, monkeypatch):
    _set_card(monkeypatch, "spark")
    r = _machine.drop_caches()
    assert not r.ok and r.reason == "not_granted"


def test_fallback_run_failure(shim, monkeypatch):
    _set_card(monkeypatch, "spark")

    def fake(argv, **k):
        if argv[-1] == "-l":
            return _machine._Proc(0, "    (root) NOPASSWD: ALL\n", "")
        return _machine._Proc(1, "", "boom")

    monkeypatch.setattr(_machine, "_run", fake)
    r = _machine.drop_caches()
    assert not r.ok and r.reason == "failed"


def test_grant_status_reports_rule(shim, monkeypatch):
    shim("sudo", "    (root) NOPASSWD: /usr/sbin/sysctl -w vm.drop_caches\\=3\n")
    g = _machine.grant_status()
    assert g.granted and g.rule == "sysctl"


def test_sudo_always_non_interactive(shim, monkeypatch):
    _set_card(monkeypatch, "spark")
    sudo = shim("sudo", "    (root) NOPASSWD: ALL\n")
    _machine.drop_caches()
    assert all(line.startswith("-n ") for line in sudo.read_text().splitlines())
    assert os.environ["PATH"]
