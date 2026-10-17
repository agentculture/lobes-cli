"""Benchmark window — an exclusive, locked-clocks window around a load run.

Only ``lobes benchmark --lock`` enters this window; ``init``/``serve``/
``switch``/``up``/``fleet`` never do (guarded by ``tests/test_benchmark_window.py``).
Locking is OPT-IN (operator decision d8): ``lobes benchmark`` is a read-only
verb, so without ``--lock`` it takes no host lock and changes no clock — it only
records the start/end ``window show`` conditions, with ``locked: false`` and
reason ``not_requested``.

Sequence: take a per-host file lock -> ``window show`` (start conditions) ->
``store`` -> ``lock`` -> run -> ``restore`` exactly once (only if ``store``
succeeded) -> ``show`` (end conditions) -> release the lock. SIGINT/SIGTERM are
turned into exceptions so the ``finally`` restore always runs. If the clocks
cannot be locked (no CLI, no privilege, unsupported) the run proceeds
UNLOCKED and the conditions say so, with the reason.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import tempfile
from pathlib import Path
from typing import Any, Iterator

from lobes.cli._errors import EXIT_ENV_ERROR, ModelGearError
from lobes.runtime import _machine

_SHOW_KEYS = ("power_mode", "clocks", "fan_profile", "temps_c")


def lock_path() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return Path(base) / "lobes-benchmark-window.lock"


def _snapshot(res: Any) -> dict[str, Any]:
    if not res.ok:
        return {"available": False, "reason": res.reason}
    return {k: res.payload.get(k) for k in _SHOW_KEYS}


class Conditions:
    """Recorder yielded by :func:`benchmark_window`; ``as_dict`` is the JSON block."""

    def __init__(self) -> None:
        self.locked = False
        self.reason: str | None = None
        self.start: dict[str, Any] = {}
        self.end: dict[str, Any] = {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "locked": self.locked,
            "reason": None if self.locked else self.reason,
            "start": self.start,
            "end": self.end,
        }


def _raise_interrupt(signum, _frame):  # pragma: no cover - trivial
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def _host_lock() -> Iterator[None]:
    path = lock_path()
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = ""
            try:
                holder = os.pread(fd, 32, 0).decode().strip()
            except OSError:
                pass
            who = f" (held by pid {holder})" if holder else ""
            raise ModelGearError(
                code=EXIT_ENV_ERROR,
                message=f"another benchmark window is already active on this host{who}",
                remediation="wait for it to finish; one locked-clocks window per host",
            ) from None
        os.ftruncate(fd, 0)
        os.pwrite(fd, str(os.getpid()).encode(), 0)
        yield
    finally:
        os.close(fd)  # closing releases the flock


@contextlib.contextmanager
def _observe_only() -> Iterator[Conditions]:
    """No window (d8): record the conditions, take no lock, change nothing."""
    cond = Conditions()
    cond.reason = "not_requested"
    cond.start = _snapshot(_machine.window("show"))
    try:
        yield cond
    finally:
        cond.end = _snapshot(_machine.window("show"))


@contextlib.contextmanager
def benchmark_window(*, lock_clocks: bool = True) -> Iterator[Conditions]:
    """The locked window, or (``lock_clocks=False``) an observe-only recorder."""
    if not lock_clocks:
        with _observe_only() as cond:
            yield cond
        return
    cond = Conditions()
    with _host_lock():
        cond.start = _snapshot(_machine.window("show"))
        stored = False
        st = _machine.window("store")
        if not st.ok:
            cond.reason = st.reason or "failed"
        else:
            stored = True
            lk = _machine.window("lock")
            if lk.ok:
                cond.locked = True
            else:
                cond.reason = lk.reason or "failed"
        prev: dict[int, Any] = {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    prev[sig] = signal.signal(sig, _raise_interrupt)
                except ValueError:  # not the main thread
                    break
            yield cond
        finally:
            try:
                if stored:
                    _machine.window("restore")
                cond.end = _snapshot(_machine.window("show"))
            finally:
                for sig, handler in prev.items():
                    signal.signal(sig, handler)
