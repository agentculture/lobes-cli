"""Tests for ``model tunnel`` — offline; never spawns cloudflared.

The ``--apply`` start path is covered with ``_tunnel.start_tunnel`` mocked. The
pidfile/lifecycle helpers are exercised against a harmless ``python -c sleep``
process (not cloudflared) so the real Popen/killpg path is still covered.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from lobes.cli import main
from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR
from lobes.runtime import _compose, _health, _tunnel


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
    # The token never rides on argv: shushu injects it as $TUNNEL_TOKEN, which
    # cloudflared reads natively (no literal $TOKEN, no --token, no shell).
    assert _tunnel.build_command("shushu", "sealed") == [
        "shushu",
        "run",
        "--inject",
        "TUNNEL_TOKEN=sealed",
        "--",
        "cloudflared",
        "tunnel",
        "run",
    ]


def test_build_command_plain() -> None:
    # Plain mode carries no token on argv either — it goes via token_env().
    assert _tunnel.build_command("plain", "tok") == ["cloudflared", "tunnel", "run"]


def test_token_env() -> None:
    assert _tunnel.token_env("plain", "tok") == {"TUNNEL_TOKEN": "tok"}
    assert _tunnel.token_env("shushu", "sealed") == {}  # shushu injects it itself


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


def test_tunnel_dry_run_keeps_plaintext_token_off_argv(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu=None, plain="SUPER-SECRET-TOKEN")
    rc = main(["tunnel", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    # the plaintext token is passed via the environment, so it never appears on argv
    assert "SUPER-SECRET-TOKEN" not in out
    assert "cloudflared tunnel run" in out


def test_tunnel_dry_run_json(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu="my-secret")
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["url"] == "https://host.example/v1"
    assert payload["token_source"] == "shushu"
    assert payload["command"][-3:] == ["cloudflared", "tunnel", "run"]
    assert "$TOKEN" not in payload["command"]


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


def test_invalid_hostname_is_user_error(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path)
    # `=` form so argparse takes the leading-dash value rather than reading it as a flag
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--hostname=bad host"])
    assert rc == EXIT_USER_ERROR
    assert "invalid public hostname" in capsys.readouterr().err


def test_invalid_shushu_name_is_user_error(tmp_path, capsys) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path, shushu="--evil")  # would be argument injection on the argv
    rc = main(["tunnel", "--compose-dir", str(tmp_path)])
    assert rc == EXIT_USER_ERROR
    assert "invalid shushu secret name" in capsys.readouterr().err


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
        _tunnel,
        "start_tunnel",
        lambda d, cmd, env=None: (calls.update(dir=str(d), cmd=cmd, env=env), 4242)[1],
    )
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tunneling"] is True
    assert payload["pid"] == 4242
    assert payload["url"] == "https://host.example/v1"
    assert calls["cmd"][0] == "shushu"  # the resolved command was passed through
    assert calls["env"] == {}  # shushu injects the token itself — none added here


def test_apply_refuses_when_already_running(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    _write_cf(tmp_path)
    monkeypatch.setattr(_tunnel, "tunnel_pid", lambda d: 4321)
    monkeypatch.setattr(_tunnel, "cloudflared_present", lambda: True)
    monkeypatch.setattr(_tunnel, "shushu_present", lambda: True)
    monkeypatch.setattr(_health, "is_healthy", lambda *a, **k: True)
    rc = main(["tunnel", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "already running" in err
    assert "4321" in err


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
    monkeypatch.setattr(_tunnel, "stop_tunnel", lambda d: ("stopped", 999))
    rc = main(["tunnel", "--stop", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stopped"] is True
    assert payload["status"] == "stopped"
    assert payload["pid"] == 999


def test_stop_apply_no_running(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_tunnel, "stop_tunnel", lambda d: ("idle", None))
    rc = main(["tunnel", "--stop", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert "no running tunnel" in capsys.readouterr().out


def test_stop_apply_failed_is_env_error(tmp_path, capsys, monkeypatch) -> None:
    _scaffold(tmp_path)
    monkeypatch.setattr(_tunnel, "stop_tunnel", lambda d: ("failed", 999))
    rc = main(["tunnel", "--stop", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == EXIT_ENV_ERROR
    err = capsys.readouterr().err
    assert "did not exit" in err
    assert "999" in err


# --- lifecycle helpers (real process, NOT cloudflared) ---------------------


def test_tunnel_pid_states(tmp_path) -> None:
    # missing pidfile → None
    assert _tunnel.tunnel_pid(tmp_path) is None
    # a live pid (this test process) → returned
    _tunnel.pid_path(tmp_path).write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert _tunnel.tunnel_pid(tmp_path) == os.getpid()
    # a bogus/dead pid → None
    _tunnel.pid_path(tmp_path).write_text("2147480000\n", encoding="utf-8")
    assert _tunnel.tunnel_pid(tmp_path) is None


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="identity guard needs Linux procfs")
def test_tunnel_pid_rejects_mismatched_identity(tmp_path) -> None:
    # A live pid (this test process) recorded as some *other* program: the PID-reuse
    # guard must reject it so stop never signals an unrelated process.
    _tunnel.pid_path(tmp_path).write_text(
        json.dumps({"pid": os.getpid(), "pgid": os.getpgid(0), "argv0": "cloudflared"}),
        encoding="utf-8",
    )
    assert _tunnel.tunnel_pid(tmp_path) is None


def test_stop_tunnel_clears_stale_pidfile(tmp_path) -> None:
    _tunnel.pid_path(tmp_path).write_text("2147480000\n", encoding="utf-8")
    assert _tunnel.stop_tunnel(tmp_path) == ("idle", None)
    assert not _tunnel.pid_path(tmp_path).exists()


def test_start_and_stop_tunnel_roundtrip(tmp_path) -> None:
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    pid = _tunnel.start_tunnel(tmp_path, cmd)
    try:
        assert _tunnel.pid_path(tmp_path).is_file()
        assert _tunnel.log_path(tmp_path).is_file()
        assert _tunnel.tunnel_pid(tmp_path) == pid
    finally:
        status, stopped = _tunnel.stop_tunnel(tmp_path)
    assert (status, stopped) == ("stopped", pid)
    assert not _tunnel.pid_path(tmp_path).exists()
