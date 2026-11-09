"""Per-member rollout lock: an atomic remote ``mkdir`` of ~/.lobes/.rollout.lock."""

from __future__ import annotations

import os
import socket
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from lobes.rollout._transport import Runner, default_runner, run

LOCK_REL = ".lobes/.rollout.lock"
HELD_EXIT = 17

_ACQUIRE = (
    'd="$HOME/' + LOCK_REL + '"; mkdir -p "$HOME/.lobes" || exit 2; '
    'if mkdir "$d" 2>/dev/null; then printf \'%s\\n\' "$1" > "$d/holder"; '
    'else echo HELD; cat "$d/holder" 2>/dev/null; exit ' + str(HELD_EXIT) + "; fi"
)
_RELEASE = 'rm -rf "$HOME/' + LOCK_REL + '"'


class LockHeld(RuntimeError):
    """The member's rollout lock is held by someone else."""

    def __init__(self, holder: str):
        self.holder = holder
        super().__init__(f"rollout lock held on member by: {holder or 'unknown holder'}")


class MemberLock:
    def __init__(
        self,
        resident: Any,
        *,
        runner: Runner = default_runner,
        host: Optional[str] = None,
        pid: Optional[int] = None,
        now: Optional[Callable[[], str]] = None,
        timeout: float = 30.0,
    ):
        self.resident = resident
        self.runner = runner
        self.timeout = timeout
        self.holder = "host={} pid={} started={}".format(
            host or socket.gethostname(),
            pid if pid is not None else os.getpid(),
            (now or (lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z")))(),
        ).replace("'", "")
        self.held = False

    def acquire(self) -> None:
        r = run(
            self.resident,
            ["sh", "-c", _ACQUIRE, "sh", self.holder],
            timeout=self.timeout,
            runner=self.runner,
        )
        if r.returncode == HELD_EXIT:
            lines = r.stdout.strip().splitlines()
            raise LockHeld(" ".join(lines[1:]).strip() if lines else "")
        if not r.ok:
            raise RuntimeError(
                f"could not take rollout lock (exit {r.returncode}): {r.stderr.strip()}"
            )
        self.held = True

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        run(self.resident, ["sh", "-c", _RELEASE], timeout=self.timeout, runner=self.runner)


@contextmanager
def member_lock(resident: Any, **kw: Any) -> Iterator[MemberLock]:
    lock = MemberLock(resident, **kw)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
