"""SSH (or local) command transport for rollout. Tests inject the runner."""

from __future__ import annotations

import shlex
import subprocess  # nosec B404
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from lobes.rollout._redact import redact

DEFAULT_TIMEOUT = 30.0
# argv, timeout -> (returncode, stdout, stderr)
Runner = Callable[[Sequence[str], float], "tuple[int, str, str]"]


@dataclass(frozen=True)
class Target:
    """Where to run: ssh target/user, absolute lobes path, or local."""

    ssh_target: str = ""
    user: str = ""
    lobes_path: str = ""
    local: bool = False


@dataclass(frozen=True)
class CommandResult:
    argv: tuple
    returncode: int
    stdout: str  # redacted
    stderr: str  # redacted
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def as_target(resident: Any) -> Target:
    """Coerce a mapping or attribute-bearing object into a Target."""
    if isinstance(resident, Target):
        return resident

    def get(*names: str, default: Any = "") -> Any:
        for n in names:
            v = resident.get(n) if isinstance(resident, Mapping) else getattr(resident, n, None)
            if v not in (None, ""):
                return v
        return default

    return Target(
        ssh_target=str(get("ssh_target", "ssh_host", "host")),
        user=str(get("user", "ssh_user")),
        lobes_path=str(get("lobes_path", "lobes_bin")),
        local=bool(get("local", "is_local", default=False)),
    )


def default_runner(argv: Sequence[str], timeout: float) -> "tuple[int, str, str]":
    try:
        p = subprocess.run(  # nosec B603
            list(argv), capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    return p.returncode, p.stdout, p.stderr


def build_argv(target: Target, remote_argv: Sequence[str], timeout: float) -> list:
    if target.local:
        return list(remote_argv)
    dest = f"{target.user}@{target.ssh_target}" if target.user else target.ssh_target
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        dest,
        " ".join(shlex.quote(a) for a in remote_argv),
    ]


def run(
    resident: Any,
    remote_argv: Sequence[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    runner: Runner = default_runner,
    secrets: Sequence[str] = (),
) -> CommandResult:
    """Run a command on the member; output is redacted before return."""
    t = as_target(resident)
    argv = build_argv(t, remote_argv, timeout)
    rc, out, err = runner(argv, timeout)
    return CommandResult(
        tuple(argv), rc, redact(out, secrets), redact(err, secrets), timed_out=(rc == 124)
    )


def run_lobes(
    resident: Any,
    args: Sequence[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    runner: Runner = default_runner,
    secrets: Sequence[str] = (),
) -> CommandResult:
    """Run lobes by its absolute path (non-login ssh PATH lacks ~/.local/bin)."""
    t = as_target(resident)
    if not t.lobes_path.startswith("/"):
        raise ValueError("lobes_path must be absolute")
    return run(resident, [t.lobes_path, *args], timeout=timeout, runner=runner, secrets=secrets)


def read_remote_value(
    resident: Any,
    remote_argv: Sequence[str],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    runner: Runner = default_runner,
) -> str:
    """Return a raw remote value (e.g. a gateway key) to memory only.

    Never logged or redacted-for-display here; callers pass it as `secrets`
    to later run() calls so it is scrubbed from any captured output.
    """
    t = as_target(resident)
    rc, out, _ = runner(build_argv(t, remote_argv, timeout), timeout)
    if rc != 0:
        raise RuntimeError(f"remote read failed (exit {rc})")
    return out.strip()
