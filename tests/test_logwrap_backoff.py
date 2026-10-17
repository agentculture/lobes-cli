"""Crash backoff in mg-logwrap.sh (lane-safety contract sections 3 and 4).

The start counter (``${MG_LOG_DIR}/lane-state/<name>.starts``) and the pre-exec
memory gate (``--gpu-memory-utilization`` x ``MemTotal`` against
``MemAvailable``) are driven through a real bash process. ``sleep`` is a PATH
shim that records its argument and runs an optional per-test hook, so a test
can mutate the fake world (age the counter, free memory) between polls. The
boot hold is switched off (``LOBES_BOOT_DELAY_S=0``) so only backoff sleeps
are recorded.

Counter rule: a start is held when the file ALREADY holds ``LOBES_START_MAX``
or more starts inside the window, i.e. with ``LOBES_START_MAX=3`` the 4th start
inside 600 s holds. A start is recorded when the lane proceeds past the hold.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from importlib.resources import files
from pathlib import Path

import pytest

SCRIPT = Path(str(files("lobes.templates") / "mg-logwrap.sh"))
GIB_KIB = 1024 * 1024
TOTAL_KIB = 100 * GIB_KIB  # 100 GiB host


def _world(tmp_path: Path, avail_kib: int = TOTAL_KIB, hook: str = "") -> dict[str, str]:
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    (proc / "uptime").write_text("100000.0 1.0\n")
    _meminfo(tmp_path, avail_kib)
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    shim = bindir / "sleep"
    shim.write_text(f'#!/bin/bash\necho "$1" >> "{tmp_path}/sleeps"\n{hook}\n')
    shim.chmod(0o755)
    return {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "LOBES_PROC_ROOT": str(proc),
        "MG_LOG_DIR": str(tmp_path / "logs"),
        "MG_LOG_NAME": "worker",
        "LOBES_BOOT_DELAY_S": "0",
    }


def _meminfo(tmp_path: Path, avail_kib: int) -> None:
    (tmp_path / "proc" / "meminfo").write_text(
        f"MemTotal:       {TOTAL_KIB} kB\nMemFree:         1024 kB\n"
        f"MemAvailable:   {avail_kib} kB\nBuffers:            0 kB\n"
    )


def _run(tmp_path: Path, env: dict[str, str], *args: str, **extra: str):
    res = subprocess.run(
        ["bash", str(SCRIPT), "echo", "served", *args],
        env={**env, **extra},
        capture_output=True,
        text=True,
        timeout=30,
    )
    rec = tmp_path / "sleeps"
    return res, (rec.read_text().split() if rec.exists() else [])


def _starts(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "lane-state" / "worker.starts"


def _gated(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "lane-state" / "worker.gated"


# --- start counter -----------------------------------------------------------


def test_each_start_appends_an_epoch(tmp_path):
    env = _world(tmp_path)
    before = int(time.time())
    for _ in range(3):
        res, sleeps = _run(tmp_path, env)
        assert res.returncode == 0 and "served" in res.stdout
    lines = _starts(tmp_path).read_text().split()
    assert len(lines) == 3
    assert all(before - 1 <= int(x) <= int(time.time()) + 1 for x in lines)
    assert sleeps == []


def test_fourth_start_inside_window_holds_until_file_deleted(tmp_path):
    starts = _starts(tmp_path)
    env = _world(tmp_path, hook=f'[ "$1" = 30 ] && rm -f "{starts}"')
    now = int(time.time())
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now - 100}\n{now - 50}\n{now - 10}\n")
    res, sleeps = _run(tmp_path, env)
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == ["30"]
    msg = (
        f"lobes: 3 starts in 600s — holding; clear with 'lobes up worker --apply' "
        f"or delete {starts}"
    )
    assert msg in res.stdout
    log = next((tmp_path / "logs").glob("worker-2*.log")).read_text()
    assert msg in log
    # the file was deleted, so this start begins a fresh count
    assert len(starts.read_text().split()) == 1


def test_hold_releases_when_oldest_start_ages_out(tmp_path):
    starts = _starts(tmp_path)
    now = int(time.time())
    # after two polls, rewrite the file as if the oldest start had aged out
    hook = (
        f'n=$(wc -l < "{tmp_path}/sleeps"); '
        f'[ "$n" -ge 2 ] && printf "%s\\n" {now - 900} {now - 50} {now - 10} > "{starts}"'
    )
    env = _world(tmp_path, hook=hook)
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now - 100}\n{now - 50}\n{now - 10}\n")
    res, sleeps = _run(tmp_path, env)
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == ["30", "30"]
    kept = [int(x) for x in starts.read_text().split()]
    assert now - 900 not in kept  # pruned on rewrite
    assert len(kept) == 3


def test_clean_window_resets(tmp_path):
    env = _world(tmp_path)
    old = int(time.time()) - 601
    starts = _starts(tmp_path)
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{old}\n{old}\n{old}\n{old}\ngarbage\n")
    res, sleeps = _run(tmp_path, env)
    assert res.returncode == 0 and sleeps == []
    assert len(starts.read_text().split()) == 1  # old + garbage pruned


def test_custom_max_and_window(tmp_path):
    env = _world(tmp_path, hook=f'rm -f "{_starts(tmp_path)}"')
    now = int(time.time())
    starts = _starts(tmp_path)
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now - 50}\n")
    _, sleeps = _run(tmp_path, env, LOBES_START_MAX="1", LOBES_START_WINDOW_S="60")
    assert sleeps == ["30"]
    starts.write_text(f"{now - 70}\n")
    (tmp_path / "sleeps").unlink()
    _, sleeps = _run(tmp_path, env, LOBES_START_MAX="1", LOBES_START_WINDOW_S="60")
    assert sleeps == []


def test_start_max_zero_turns_counter_off(tmp_path):
    env = _world(tmp_path)
    now = int(time.time())
    starts = _starts(tmp_path)
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now}\n" * 10)
    res, sleeps = _run(tmp_path, env, LOBES_START_MAX="0")
    assert res.returncode == 0 and sleeps == []
    assert len(starts.read_text().split()) == 10  # not touched


@pytest.mark.parametrize("bad", ["abc", "-1", "2.5", ""])
def test_invalid_counter_knobs_fall_back_to_defaults(tmp_path, bad):
    env = _world(tmp_path, hook=f'rm -f "{_starts(tmp_path)}"')
    now = int(time.time())
    starts = _starts(tmp_path)
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now - 500}\n{now - 5}\n{now - 5}\n")
    res, sleeps = _run(tmp_path, env, LOBES_START_MAX=bad, LOBES_START_WINDOW_S=bad)
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == ["30"]  # default 3 in 600 s


def test_unwritable_log_dir_skips_counter_but_serves(tmp_path):
    env = _world(tmp_path)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    res, sleeps = _run(tmp_path, env, MG_LOG_DIR=str(blocker / "sub"))
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == []
    assert res.stdout.count("lane-state") == 1  # one skip line


# --- memory gate -------------------------------------------------------------


def test_gate_passes_immediately_when_memory_is_free(tmp_path):
    env = _world(tmp_path, avail_kib=50 * GIB_KIB)
    res, sleeps = _run(tmp_path, env, "--gpu-memory-utilization=0.45")
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == []
    assert not _gated(tmp_path).exists()


def test_gate_waits_writes_marker_then_passes(tmp_path):
    gated = _gated(tmp_path)
    hook = (
        f'cp "{gated}" "{tmp_path}/marker-seen" 2>/dev/null; '
        f'n=$(wc -l < "{tmp_path}/sleeps"); [ "$n" -ge 2 ] && '
        f'sed -i "s/^MemAvailable:.*/MemAvailable:   {60 * GIB_KIB} kB/" '
        f'"{tmp_path}/proc/meminfo"'
    )
    env = _world(tmp_path, avail_kib=GIB_KIB, hook=hook)
    res, sleeps = _run(tmp_path, env, "--gpu-memory-utilization", "0.45")
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == ["5", "5"]
    seen = json.loads((tmp_path / "marker-seen").read_text())
    assert seen["lane"] == "worker"
    assert seen["need_kib"] == 47185920  # ceil(0.45 * 104857600)
    assert seen["avail_kib"] == GIB_KIB
    assert abs(seen["since"] - time.time()) < 60
    assert not gated.exists()  # removed on pass
    assert "0.45" in res.stdout  # argv passed through intact


def test_gate_times_out_exits_75_and_removes_marker(tmp_path):
    gated = _gated(tmp_path)
    hook = f'cp "{gated}" "{tmp_path}/marker-seen" 2>/dev/null'
    env = _world(tmp_path, avail_kib=GIB_KIB, hook=hook)
    res, sleeps = _run(
        tmp_path, env, "--gpu-memory-utilization=0.45", LOBES_MEM_GATE_TIMEOUT_S="15"
    )
    assert res.returncode == 75
    assert "served" not in res.stdout
    assert sleeps == ["5", "5", "5"]
    assert (
        f"lobes: memory gate timed out (need 47185920 KiB, have {GIB_KIB} KiB) — "
        "run 'lobes watcher install' or the machine CLI's drop-caches"
    ) in res.stdout
    assert (tmp_path / "marker-seen").exists()
    assert not gated.exists()


@pytest.mark.parametrize(
    "util,need",
    [
        ("0.45", 47185920),
        (".5", 52428800),
        ("1", 104857600),
        ("0.058", 6081741),  # ceil(6081740.8)
    ],
)
def test_need_is_ceil_of_util_times_memtotal(tmp_path, util, need):
    env = _world(tmp_path, avail_kib=0)
    res, _ = _run(tmp_path, env, f"--gpu-memory-utilization={util}", LOBES_MEM_GATE_TIMEOUT_S="0")
    assert res.returncode == 75
    assert f"need {need} KiB, have 0 KiB" in res.stdout


def test_need_boundary_passes_at_exact_ceil(tmp_path):
    env = _world(tmp_path, avail_kib=6081741)
    res, sleeps = _run(tmp_path, env, "--gpu-memory-utilization=0.058")
    assert res.returncode == 0 and sleeps == []


@pytest.mark.parametrize("util", ["abc", "0.4.5", ".", "-0.3"])
def test_garbage_util_skips_gate_with_log_line(tmp_path, util):
    env = _world(tmp_path, avail_kib=0)
    res, sleeps = _run(tmp_path, env, "--gpu-memory-utilization", util)
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == []
    assert "memory gate skipped" in res.stdout


def test_no_util_flag_skips_gate(tmp_path):
    env = _world(tmp_path, avail_kib=0)
    res, sleeps = _run(tmp_path, env, "--ctx-size", "4096")
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == []
    assert "memory gate" not in res.stdout
    assert not _gated(tmp_path).exists()


def test_mem_gate_zero_turns_gate_off(tmp_path):
    env = _world(tmp_path, avail_kib=0)
    res, sleeps = _run(tmp_path, env, "--gpu-memory-utilization=0.45", LOBES_MEM_GATE="0")
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == []


def test_unreadable_meminfo_never_blocks_serving(tmp_path):
    env = _world(tmp_path)
    (tmp_path / "proc" / "meminfo").unlink()
    res, sleeps = _run(tmp_path, env, "--gpu-memory-utilization=0.45")
    assert res.returncode == 0 and "served" in res.stdout
    assert sleeps == []


def test_gate_runs_without_marker_when_log_dir_unwritable(tmp_path):
    env = _world(tmp_path, avail_kib=0)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    res, _ = _run(
        tmp_path,
        env,
        "--gpu-memory-utilization=0.45",
        MG_LOG_DIR=str(blocker / "sub"),
        LOBES_MEM_GATE_TIMEOUT_S="5",
    )
    assert res.returncode == 75  # the gate still runs on /proc alone


def test_empty_flag_filter_still_applies_with_gate(tmp_path):
    env = _world(tmp_path)
    res, _ = _run(tmp_path, env, "--lora-modules=", "--gpu-memory-utilization=0.1")
    assert res.returncode == 0
    assert "--lora-modules" not in res.stdout.splitlines()[-1]


def test_gate_runs_after_boot_hold(tmp_path):
    gated = _gated(tmp_path)
    hook = (
        f'[ "$1" = 5 ] && sed -i "s/^MemAvailable:.*/MemAvailable:   {60 * GIB_KIB} kB/" '
        f'"{tmp_path}/proc/meminfo"'
    )
    env = _world(tmp_path, avail_kib=GIB_KIB, hook=hook)
    (tmp_path / "proc" / "uptime").write_text("100.0 1.0\n")
    res, sleeps = _run(tmp_path, env, "--gpu-memory-utilization=0.45", LOBES_BOOT_DELAY_S="120")
    assert res.returncode == 0
    assert sleeps == ["20", "5"]  # boot hold first, then the gate (c42)
    assert not gated.exists()


# --- SIGTERM while held (2026-10-17 hardware finding) ------------------------
# A lane held by the start counter or the memory gate must stop on `docker
# stop` / a TERM to its client. A foreground `sleep` defers bash traps until it
# returns (and bash as PID 1 ignores TERM with no handler), so the wrapper
# waits with `sleep N & wait $!` under a TERM/INT handler. These tests use the
# REAL sleep: the shim records instantly and would hide the deferral.


def _spawn_held(tmp_path: Path, env: dict[str, str], *args: str, **extra: str):
    real = {**env, **extra, "PATH": os.environ["PATH"]}  # real sleep, not the shim
    return subprocess.Popen(
        ["bash", str(SCRIPT), "echo", "served", *args],
        env=real,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_log(tmp_path: Path, needle: str, deadline_s: float = 15.0) -> None:
    # Startup only: under a full parallel suite bash can take seconds to reach the
    # hold. The prompt-exit promise is timed separately in _term_and_time.
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        logs = list((tmp_path / "logs").glob("worker-2*.log"))
        if logs and needle in logs[0].read_text():
            time.sleep(0.1)  # let bash reach the wait
            return
        time.sleep(0.02)
    raise AssertionError(f"wrapper never logged {needle!r}")


def _term_and_time(proc: subprocess.Popen, sig=None) -> tuple[int, float]:
    import signal

    start = time.monotonic()
    proc.send_signal(sig or signal.SIGTERM)
    try:
        rc = proc.wait(timeout=2.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise AssertionError("wrapper did not exit within 2.5 s of the signal")
    return rc, time.monotonic() - start


def test_sigterm_while_start_counter_holds_exits_143_promptly(tmp_path):
    env = _world(tmp_path)
    now = int(time.time())
    starts = _starts(tmp_path)
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now - 100}\n{now - 50}\n{now - 10}\n")
    proc = _spawn_held(tmp_path, env)
    _wait_for_log(tmp_path, "holding")
    rc, took = _term_and_time(proc)
    assert rc == 143
    assert took < 2.0
    assert len(starts.read_text().split()) == 3  # the aborted start is not recorded


def test_sigint_while_start_counter_holds_exits_130(tmp_path):
    import signal

    env = _world(tmp_path)
    now = int(time.time())
    starts = _starts(tmp_path)
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now - 100}\n{now - 50}\n{now - 10}\n")
    proc = _spawn_held(tmp_path, env)
    _wait_for_log(tmp_path, "holding")
    rc, took = _term_and_time(proc, signal.SIGINT)
    assert rc == 130
    assert took < 2.0


def test_sigterm_while_memory_gate_waits_exits_143_and_removes_marker(tmp_path):
    env = _world(tmp_path, avail_kib=GIB_KIB)
    proc = _spawn_held(tmp_path, env, "--gpu-memory-utilization=0.45")
    gated = _gated(tmp_path)
    end = time.monotonic() + 2.0
    while not gated.exists() and time.monotonic() < end:
        time.sleep(0.02)
    assert gated.exists(), "gate marker never written"
    time.sleep(0.1)  # let bash reach the wait
    rc, took = _term_and_time(proc)
    assert rc == 143
    assert took < 2.0
    assert not gated.exists()
