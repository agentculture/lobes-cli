"""Tests for ``model tunnel`` — offline; never spawns cloudflared.

The ``--apply`` start path is covered with ``_tunnel.start_tunnel`` mocked. The
pidfile/lifecycle helpers are exercised against a harmless ``python -c sleep``
process (not cloudflared) so the real Popen/killpg path is still covered.
"""

from __future__ import annotations

import json
import sys

import pytest

from model_gear.cli import main
from model_gear.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR
from model_gear.runtime import _compose, _health, _tunnel


def _scaffold(path):
    _compose.write_scaffold(path, force=True)
    return path


def _write_cf(path, *, hostname="host.example", shushu="sealed-name", plain=None):
    lines = []
    if hostname is not None:
        lines.append(f"{_tunnel.HOSTNAME_KEY}={hostname}")
    if shushu is not None:
        lines.append(f"{_tunnel.TOKEN_SHUSHU_KEY}={shushu}")
    if plain is not None:
        lines.append(f"{_tunnel.TOKEN_PLAIN_KEY}={plain}")
    (path / _tunnel.TUNNEL_ENV_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _clear_tunnel_env(monkeypatch):
    # Don't let a real $CULTURE_VLLM_PUBLIC_HOSTNAME on the host leak into tests.
    monkeypatch.delenv(_tunnel.HOSTNAME_KEY, raising=False)


# --- pure helpers ----------------------------------------------------------


def test_build_command_shushu() -> None:
    assert _tunnel.build_command("shushu", "sealed") == [
        "shushu",
        "run",
        "--inject",
        "TOKEN=sealed",
        "--",
        "cloudflared",
        "tunnel",
        "run",
        "--token",
        "$TOKEN",
    ]


def test_build_command_plain() -> None:
    assert _tunnel.build_command("plain", "tok") == [
        "cloudflared",
        "tunnel",
        "run",
        "--token",
        "tok",
    ]


def test_redacted_masks_plaintext_only() -> None:
    assert _tunnel.redacted(["cloudflared", "tunnel", "run", "--token", "tok"]) == [
        "cloudflared",
        "tunnel",
        "run",
        "--token",
        "***",
    ]
    sealed = _tunnel.build_command("shushu", "x")
    assert _tunnel.redacted(sealed) == sealed  # $TOKEN is not a secret


# --- dry-run ---------------------------------------------------------------


def test_tunnel_dry_run(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path)
    rc = main(["tunnel", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "cloudflared" in out
    assert "https://host.example/v1" in out


def test_tunnel_dry_run_redacts_plaintext_token(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu=None, plain="SUPER-SECRET-TOKEN")
    rc = main(["tunnel", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUPER-SECRET-TOKEN" not in out
    assert "***" in out


def test_tunnel_dry_run_json(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu="my-secret")
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["url"] == "https://host.example/v1"
    assert payload["token_source"] == "shushu"
    assert "$TOKEN" in payload["command"]


# --- hostname resolution ---------------------------------------------------


def test_hostname_flag_wins(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, hostname="from-file.example")
    monkeypatch.setenv(_tunnel.HOSTNAME_KEY, "from-env.example")
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--hostname", "from-flag.example"])
    assert rc == 0
    assert "https://from-flag.example/v1" in capsys.readouterr().out


def test_hostname_env_over_file(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, hostname="from-file.example")
    monkeypatch.setenv(_tunnel.HOSTNAME_KEY, "from-env.example")
    rc = main(["tunnel", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "https://from-env.example/v1" in capsys.readouterr().out


def test_hostname_from_file(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, hostname="from-file.example")
    rc = main(["tunnel", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "https://from-file.example/v1" in capsys.readouterr().out


def test_hostname_missing_is_user_error(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, hostname=None)  # token present, no hostname
    rc = main(["tunnel", "--compose-dir", str(tmp_path)])
    assert rc == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


# --- token resolution ------------------------------------------------------


def test_token_missing_is_user_error(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu=None, plain=None)  # hostname only
    rc = main(["tunnel", "--compose-dir", str(tmp_path)])
    assert rc == EXIT_USER_ERROR
    assert "run-token" in capsys.readouterr().err


def test_token_plain_fallback(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu=None, plain="tok")
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["token_source"] == "plain"
    assert payload["command"][:3] == ["cloudflared", "tunnel", "run"]


# --- preflight (--apply) ---------------------------------------------------


def test_apply_requires_cloudflared(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path)
    monkeypatch.setattr(_tunnel, "cloudflared_present", lambda: False)
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == EXIT_ENV_ERROR
    err = capsys.readouterr().err
    assert "cloudflared" in err
    assert "hint:" in err


def test_apply_requires_shushu_for_sealed_token(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu="my-secret")
    monkeypatch.setattr(_tunnel, "cloudflared_present", lambda: True)
    monkeypatch.setattr(_tunnel, "shushu_present", lambda: False)
    monkeypatch.setattr(_health, "is_healthy", lambda *a, **k: True)
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == EXIT_ENV_ERROR
    assert "shushu" in capsys.readouterr().err


def test_apply_requires_healthy_server(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path)
    monkeypatch.setattr(_tunnel, "cloudflared_present", lambda: True)
    monkeypatch.setattr(_tunnel, "shushu_present", lambda: True)
    # the autouse offline fixture already makes is_healthy() return False
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == EXIT_ENV_ERROR
    assert "not healthy" in capsys.readouterr().err


def test_apply_starts_tunnel(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu="my-secret")
    monkeypatch.setattr(_tunnel, "cloudflared_present", lambda: True)
    monkeypatch.setattr(_tunnel, "shushu_present", lambda: True)
    monkeypatch.setattr(_health, "is_healthy", lambda *a, **k: True)
    calls: dict = {}
    monkeypatch.setattr(
        _tunnel, "start_tunnel", lambda d, cmd: (calls.update(dir=str(d), cmd=cmd), 4242)[1]
    )
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tunneling"] is True
    assert payload["pid"] == 4242
    assert payload["url"] == "https://host.example/v1"
    assert calls["cmd"][0] == "shushu"  # the resolved command was passed through


# --- stop ------------------------------------------------------------------


def test_stop_dry_run(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_tunnel, "tunnel_pid", lambda d: 999)
    rc = main(["tunnel", "--stop", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "999" in out


def test_stop_apply(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_tunnel, "stop_tunnel", lambda d: 999)
    rc = main(["tunnel", "--stop", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stopped"] is True
    assert payload["pid"] == 999


def test_stop_apply_no_running(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_tunnel, "stop_tunnel", lambda d: None)
    rc = main(["tunnel", "--stop", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert "no running tunnel" in capsys.readouterr().out


# --- lifecycle helpers (real process, NOT cloudflared) ---------------------


def test_tunnel_pid_states(tmp_path) -> None:
    import os

    # missing pidfile → None
    assert _tunnel.tunnel_pid(tmp_path) is None
    # a live pid (this test process) → returned
    _tunnel.pid_path(tmp_path).write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert _tunnel.tunnel_pid(tmp_path) == os.getpid()
    # a bogus/dead pid → None
    _tunnel.pid_path(tmp_path).write_text("2147480000\n", encoding="utf-8")
    assert _tunnel.tunnel_pid(tmp_path) is None


def test_stop_tunnel_clears_stale_pidfile(tmp_path) -> None:
    _tunnel.pid_path(tmp_path).write_text("2147480000\n", encoding="utf-8")
    assert _tunnel.stop_tunnel(tmp_path) is None
    assert not _tunnel.pid_path(tmp_path).exists()


def test_start_and_stop_tunnel_roundtrip(tmp_path) -> None:
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    pid = _tunnel.start_tunnel(tmp_path, cmd)
    try:
        assert _tunnel.pid_path(tmp_path).is_file()
        assert _tunnel.log_path(tmp_path).is_file()
        assert _tunnel.tunnel_pid(tmp_path) == pid
    finally:
        stopped = _tunnel.stop_tunnel(tmp_path)
    assert stopped == pid
    assert not _tunnel.pid_path(tmp_path).exists()
