"""Boot hold + staggered release in mg-logwrap.sh (lane-safety contract section 2).

The wrapper is driven as a real bash process with a fake proc root and a fake
``sleep`` shim on PATH that records its argument and returns at once.
"""

from __future__ import annotations

import os
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest

SCRIPT = Path(str(files("lobes.templates") / "mg-logwrap.sh"))


def _run(tmp_path: Path, uptime: str, **env: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    (proc / "uptime").write_text(f"{uptime} 100.00\n")
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    rec = tmp_path / "sleeps"
    shim = bindir / "sleep"
    shim.write_text(f'#!/bin/sh\necho "$1" >> "{rec}"\n')
    shim.chmod(0o755)
    full = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "LOBES_PROC_ROOT": str(proc),
        "MG_LOG_DIR": str(tmp_path / "logs"),
        "MG_LOG_NAME": env.pop("MG_LOG_NAME", "primary"),
        **env,
    }
    res = subprocess.run(
        ["bash", str(SCRIPT), "echo", "served", "--empty="],
        env=full,
        capture_output=True,
        text=True,
        timeout=30,
    )
    sleeps = rec.read_text().split() if rec.exists() else []
    return res, sleeps


def test_holds_exact_remainder_default_delay(tmp_path):
    res, sleeps = _run(tmp_path, "30.50")
    assert res.returncode == 0
    assert "served" in res.stdout and "--empty" not in res.stdout
    assert sleeps == ["90"]
    assert "lobes: host up 30s, holding 90s (boot delay 120s + stagger 0s)" in res.stdout
    log = next((tmp_path / "logs").glob("primary-2*.log")).read_text()
    assert "holding 90s" in log


def test_stagger_offset_by_position(tmp_path):
    res, sleeps = _run(
        tmp_path,
        "10.0",
        LOBES_BOOT_ORDER="embedder,primary,reranker",
        LOBES_BOOT_STAGGER_S="20",
    )
    assert sleeps == ["130"]  # 120 + 1*20 - 10
    assert "stagger 20s" in res.stdout


def test_unlisted_lane_releases_last(tmp_path):
    _, sleeps = _run(
        tmp_path,
        "0.0",
        MG_LOG_NAME="other",
        LOBES_BOOT_ORDER="a,b,c",
        LOBES_BOOT_STAGGER_S="20",
    )
    assert sleeps == ["180"]  # 120 + 3*20


def test_uptime_past_release_does_not_sleep(tmp_path):
    res, sleeps = _run(tmp_path, "500.0")
    assert sleeps == []
    assert "holding" not in res.stdout
    assert "served" in res.stdout


def test_uptime_past_delay_but_inside_stagger_sleeps_remainder(tmp_path):
    _, sleeps = _run(tmp_path, "125.9", LOBES_BOOT_ORDER="primary", LOBES_BOOT_STAGGER_S="0")
    assert sleeps == []
    _, sleeps = _run(tmp_path, "125.9", LOBES_BOOT_ORDER="a,primary", LOBES_BOOT_STAGGER_S="20")
    assert sleeps == ["15"]  # 140 - 125


def test_delay_zero_disables_hold_and_stagger(tmp_path):
    res, sleeps = _run(
        tmp_path,
        "1.0",
        LOBES_BOOT_DELAY_S="0",
        LOBES_BOOT_ORDER="a,primary",
        LOBES_BOOT_STAGGER_S="20",
    )
    assert sleeps == []
    assert "served" in res.stdout


@pytest.mark.parametrize("bad", ["abc", "-5", "1.5", ""])
def test_invalid_knobs_fall_back_to_defaults(tmp_path, bad):
    res, sleeps = _run(tmp_path, "100.0", LOBES_BOOT_DELAY_S=bad, LOBES_BOOT_STAGGER_S=bad)
    assert res.returncode == 0
    assert "served" in res.stdout
    assert sleeps == ["20"]  # default 120 - 100


def test_missing_or_garbled_uptime_never_blocks_serving(tmp_path):
    res, sleeps = _run(tmp_path, "garbage")
    assert res.returncode == 0
    assert sleeps == []
    assert "served" in res.stdout


def test_sigterm_during_boot_hold_exits_143_promptly(tmp_path):
    """A lane in its boot hold stops on TERM (2026-10-17 hardware finding).

    Uses the REAL sleep: the hold is a 120 s wait, and a foreground `sleep`
    would defer the trap until it returned.
    """
    import signal
    import time

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    (proc_root / "uptime").write_text("10.00 1.00\n")
    p = subprocess.Popen(
        ["bash", str(SCRIPT), "echo", "served"],
        env={
            "PATH": os.environ["PATH"],
            "LOBES_PROC_ROOT": str(proc_root),
            "MG_LOG_DIR": str(tmp_path / "logs"),
            "MG_LOG_NAME": "primary",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Startup only (seconds under a full parallel suite); the exit is timed below.
    end = time.monotonic() + 15.0
    while time.monotonic() < end:
        logs = list((tmp_path / "logs").glob("primary-2*.log"))
        if logs and "holding" in logs[0].read_text():
            break
        time.sleep(0.02)
    else:
        p.kill()
        raise AssertionError("wrapper never entered the boot hold")
    time.sleep(0.1)
    start = time.monotonic()
    p.send_signal(signal.SIGTERM)
    try:
        rc = p.wait(timeout=2.5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        raise AssertionError("wrapper ignored SIGTERM during the boot hold")
    assert rc == 143
    assert time.monotonic() - start < 2.0
    assert not (tmp_path / "logs" / "lane-state" / "primary.starts").exists()
