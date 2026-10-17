"""review2-b #2 / #3: the wrapper's ``.held`` marker and lane-state ownership.

#2: a lane in the boot hold/stagger or a start-counter hold is held BEFORE
``exec`` (no GPU use) yet shows docker ``health: starting``, so it used to
defer every page-cache clear for minutes. The wrapper now keeps
``lane-state/<name>.held`` (JSON ``{"lane", "since", "reason"}``) for exactly
the duration of each hold, and ``lanes_starting()`` excludes those lanes the
same way it excludes ``.gated`` ones (contract sections 2, 3 and 5).

#3: a ``lane-state/`` a root container created before the operator-side fix
stays root-owned forever. The wrapper (root inside the container) aligns the
dir and the files it writes to the parent log dir's owner on every start,
best-effort. Tests cannot chown as a user, so ``chown``/``stat`` are PATH
shims that record what was attempted.

``sleep`` is a PATH shim that records its argument plus whether the ``.held``
marker exists at that moment (and its reason), then runs an optional hook.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from importlib.resources import files
from pathlib import Path

SCRIPT = Path(str(files("lobes.templates") / "mg-logwrap.sh"))
GIB_KIB = 1024 * 1024


def _world(tmp_path: Path, uptime: str = "100000.0", hook: str = "") -> dict[str, str]:
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    (proc / "uptime").write_text(f"{uptime} 1.0\n")
    (proc / "meminfo").write_text(
        f"MemTotal: {100 * GIB_KIB} kB\nMemAvailable: {100 * GIB_KIB} kB\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    held = _held(tmp_path)
    shim = bindir / "sleep"
    shim.write_text(
        "#!/bin/bash\n"
        f'if [ -f "{held}" ]; then r=$(cat "{held}"); else r=none; fi\n'
        f'printf "%s\\t%s\\n" "$1" "$r" >> "{tmp_path}/sleeps"\n'
        f"{hook}\n"
    )
    shim.chmod(0o755)
    return {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "LOBES_PROC_ROOT": str(proc),
        "MG_LOG_DIR": str(tmp_path / "logs"),
        "MG_LOG_NAME": "worker",
    }


def _held(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "lane-state" / "worker.held"


def _starts(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "lane-state" / "worker.starts"


def _run(tmp_path: Path, env: dict[str, str], **extra: str):
    res = subprocess.run(
        ["bash", str(SCRIPT), "echo", "served"],
        env={**env, **extra},
        capture_output=True,
        text=True,
        timeout=30,
    )
    rec = tmp_path / "sleeps"
    rows = [ln.split("\t", 1) for ln in rec.read_text().splitlines()] if rec.exists() else []
    return res, rows


# --- #2: the .held marker ----------------------------------------------------


def test_boot_hold_keeps_a_held_marker_then_removes_it(tmp_path):
    env = _world(tmp_path, uptime="30.0")
    before = int(time.time())
    res, rows = _run(tmp_path, env)
    assert res.returncode == 0
    assert "served" in res.stdout
    assert [r[0] for r in rows] == ["90"]
    body = json.loads(rows[0][1])
    assert body["lane"] == "worker"
    assert body["reason"] == "boot_hold"
    assert before - 1 <= body["since"] <= int(time.time()) + 1
    assert not _held(tmp_path).exists()


def test_no_hold_writes_no_held_marker(tmp_path):
    res, rows = _run(tmp_path, _world(tmp_path))
    assert res.returncode == 0
    assert rows == []
    assert not _held(tmp_path).exists()


def test_start_counter_hold_keeps_a_held_marker_then_removes_it(tmp_path):
    starts = _starts(tmp_path)
    env = _world(tmp_path, hook=f'rm -f "{starts}"')
    now = int(time.time())
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now - 100}\n{now - 50}\n{now - 10}\n")
    res, rows = _run(tmp_path, env)
    assert res.returncode == 0
    assert "served" in res.stdout
    assert [r[0] for r in rows] == ["30"]
    body = json.loads(rows[0][1])
    assert body == {"lane": "worker", "since": body["since"], "reason": "start_counter"}
    assert not _held(tmp_path).exists()


def test_memory_gate_wait_is_gated_not_held(tmp_path):
    """The gate keeps its own .gated marker; .held is for the pre-gate holds."""
    gated = tmp_path / "logs" / "lane-state" / "worker.gated"
    meminfo = tmp_path / "proc" / "meminfo"
    env = _world(
        tmp_path,
        hook=f'printf "MemTotal: {100 * GIB_KIB} kB\\nMemAvailable: {100 * GIB_KIB} kB\\n"'
        f' > "{meminfo}"',
    )
    meminfo.write_text(f"MemTotal: {100 * GIB_KIB} kB\nMemAvailable: {GIB_KIB} kB\n")
    res = subprocess.run(
        ["bash", str(SCRIPT), "echo", "served", "--gpu-memory-utilization=0.45"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0
    rows = (tmp_path / "sleeps").read_text().splitlines()
    assert rows == ["5\tnone"]
    assert not gated.exists()
    assert not _held(tmp_path).exists()


def _wait_for(path: Path, deadline_s: float = 15.0) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if path.exists():
            time.sleep(0.1)  # let bash reach the wait
            return
        time.sleep(0.02)
    raise AssertionError(f"{path} never appeared")


def _term(proc: subprocess.Popen, sig=signal.SIGTERM) -> int:
    proc.send_signal(sig)
    try:
        return proc.wait(timeout=2.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise AssertionError("wrapper ignored the signal")


def _spawn(tmp_path: Path, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        ["bash", str(SCRIPT), "echo", "served"],
        env={**env, "PATH": os.environ["PATH"]},  # real sleep
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_sigterm_during_boot_hold_removes_the_held_marker(tmp_path):
    env = _world(tmp_path, uptime="10.0")
    proc = _spawn(tmp_path, env)
    _wait_for(_held(tmp_path))
    assert _term(proc) == 143
    assert not _held(tmp_path).exists()


def test_sigint_during_counter_hold_removes_the_held_marker(tmp_path):
    env = _world(tmp_path)
    now = int(time.time())
    starts = _starts(tmp_path)
    starts.parent.mkdir(parents=True)
    starts.write_text(f"{now - 100}\n{now - 50}\n{now - 10}\n")
    proc = _spawn(tmp_path, env)
    _wait_for(_held(tmp_path))
    assert json.loads(_held(tmp_path).read_text())["reason"] == "start_counter"
    assert _term(proc, signal.SIGINT) == 130
    assert not _held(tmp_path).exists()


def test_unwritable_lane_state_holds_without_a_marker(tmp_path):
    env = _world(tmp_path, uptime="30.0")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    res, rows = _run(tmp_path, env, MG_LOG_DIR=str(blocker / "sub"))
    assert res.returncode == 0
    assert "served" in res.stdout
    assert rows == [["90", "none"]]


# --- #3: lane-state ownership self-heal ---------------------------------------


def _owner_shims(tmp_path: Path, env: dict[str, str], reference_ok: bool = True) -> Path:
    rec = tmp_path / "chowns"
    bindir = Path(env["PATH"].split(":", 1)[0])
    chown = bindir / "chown"
    fail = "" if reference_ok else 'case "$1" in --reference=*) exit 1 ;; esac\n'
    chown.write_text(f'#!/bin/bash\n{fail}echo "$*" >> "{rec}"\n')
    chown.chmod(0o755)
    stat_shim = bindir / "stat"
    stat_shim.write_text(f'#!/bin/bash\necho "stat $*" >> "{rec}"\necho 1234:5678\n')
    stat_shim.chmod(0o755)
    return rec


def test_lane_state_and_written_files_are_aligned_to_the_log_dir(tmp_path):
    starts = _starts(tmp_path)
    env = _world(tmp_path, uptime="30.0", hook=f'rm -f "{starts}"')
    rec = _owner_shims(tmp_path, env)
    res, _ = _run(tmp_path, env)
    assert res.returncode == 0
    assert "served" in res.stdout
    logs = tmp_path / "logs"
    calls = rec.read_text().splitlines()
    assert f"--reference={logs} {logs / 'lane-state'}" in calls
    assert f"--reference={logs} {_held(tmp_path)}" in calls
    assert f"--reference={logs} {starts}" in calls


def test_ownership_falls_back_to_stat_when_reference_is_unsupported(tmp_path):
    env = _world(tmp_path)
    rec = _owner_shims(tmp_path, env, reference_ok=False)
    res, _ = _run(tmp_path, env)
    assert res.returncode == 0
    assert "served" in res.stdout
    calls = rec.read_text().splitlines()
    assert f"stat -c %u:%g {tmp_path / 'logs'}" in calls
    assert f"1234:5678 {tmp_path / 'logs' / 'lane-state'}" in calls
    assert f"1234:5678 {_starts(tmp_path)}" in calls


def test_a_failing_chown_never_blocks_the_lane(tmp_path):
    env = _world(tmp_path)
    bindir = Path(env["PATH"].split(":", 1)[0])
    for name in ("chown", "stat"):
        (bindir / name).write_text("#!/bin/bash\necho denied >&2\nexit 1\n")
        (bindir / name).chmod(0o755)
    res, _ = _run(tmp_path, env)
    assert res.returncode == 0
    assert "served" in res.stdout
    assert "denied" not in res.stdout + res.stderr  # silent
    assert len(_starts(tmp_path).read_text().split()) == 1


def test_a_stale_held_marker_from_a_killed_run_is_dropped_at_start(tmp_path):
    held = _held(tmp_path)
    held.parent.mkdir(parents=True)
    held.write_text('{"lane": "worker", "since": 1, "reason": "boot_hold"}\n')
    res, rows = _run(tmp_path, _world(tmp_path))
    assert res.returncode == 0
    assert rows == []
    assert not held.exists()
