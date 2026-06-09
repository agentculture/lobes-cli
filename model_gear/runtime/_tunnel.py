"""Run the cloudflared tunnel that exposes the local vLLM API (stdlib only).

``model tunnel`` starts a standalone ``cloudflared tunnel run`` process (NOT a
compose service) that proxies the owner-chosen public hostname to the local
OpenAI-compatible server. The Cloudflare side (tunnel + ingress + DNS, run-token
sealed in ``shushu``) is provisioned once by ``cultureflare remote-login
--no-access``; this module owns the local lifecycle: resolve the hostname + token
source, build the command, and start/stop the background process via a pidfile.

The run-token never reaches the process argv (so it can't leak via ``ps`` or the
log) — cloudflared reads it from the ``TUNNEL_TOKEN`` environment variable, which
``shushu`` injects (sealed mode) or we set directly (plaintext fallback). All
subprocess calls use fixed argv lists (no shell) — the bandit ``B404``/``B603``
skips in ``pyproject.toml`` cover them. The hostname and the sealed-secret name
never live in committed config: they come from ``$CULTURE_VLLM_PUBLIC_HOSTNAME`` /
a gitignored ``.cf-tunnel.env`` in the deployment dir, and both are validated
against a conservative charset before they reach the argv.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess  # fixed argv lists only, never shell=True
import time
from pathlib import Path

from model_gear.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from model_gear.runtime import _env

# Gitignored config in the deployment dir; pid + log are *.pid / *.log (also ignored).
TUNNEL_ENV_FILE = ".cf-tunnel.env"
PID_FILE = ".cloudflared.pid"
LOG_FILE = "cloudflared.log"

# Keys read from .cf-tunnel.env (or the process environment, for the hostname).
# The TOKEN_* names are env-var KEYS, not secret values (the secret lives in the
# gitignored .cf-tunnel.env) — nosec silences bandit's hardcoded-password heuristic.
HOSTNAME_KEY = "CULTURE_VLLM_PUBLIC_HOSTNAME"
TOKEN_SHUSHU_KEY = "CULTURE_CF_TUNNEL_TOKEN_SHUSHU"  # nosec B105
TOKEN_PLAIN_KEY = "CULTURE_CF_TUNNEL_TOKEN"  # nosec B105

# cloudflared reads its connector token from this env var natively (the documented
# equivalent of ``--token``). Passing the token here instead of on argv means it
# never appears in ``ps``/the log and there is no shell expansion to rely on.
TOKEN_ENV_VAR = "TUNNEL_TOKEN"  # nosec B105

# Values sourced from .cf-tunnel.env that reach the cloudflared argv (the sealed
# secret name) or a printed URL (the hostname) must match this conservative shape:
# alphanumeric ends, with ``. _ : / -`` interior. This rejects argument-injection
# payloads (e.g. a value starting with ``-``) before they feed the subprocess. The
# plaintext token is passed via the environment, never argv, so it is not constrained.
_SAFE_VALUE = re.compile(r"\A[A-Za-z0-9](?:[A-Za-z0-9._:/-]*[A-Za-z0-9])?\Z")

# Labels for the validated values (named so the error wording lives in one place).
_HOSTNAME_LABEL = "public hostname"
_SHUSHU_LABEL = "shushu secret name"

INSTALL_HINT = (
    "install cloudflared and put it on PATH: https://developers.cloudflare.com/"
    "cloudflare-one/connections/connect-networks/downloads/"
)


def cloudflared_present() -> bool:
    """True when ``cloudflared`` resolves on PATH."""
    return shutil.which("cloudflared") is not None


def shushu_present() -> bool:
    """True when ``shushu`` resolves on PATH (needed only for the sealed-token source)."""
    return shutil.which("shushu") is not None


def tunnel_env_path(deploy_dir: os.PathLike | str) -> Path:
    return Path(deploy_dir) / TUNNEL_ENV_FILE


def pid_path(deploy_dir: os.PathLike | str) -> Path:
    return Path(deploy_dir) / PID_FILE


def log_path(deploy_dir: os.PathLike | str) -> Path:
    return Path(deploy_dir) / LOG_FILE


def _require_safe(value: str, *, what: str) -> str:
    """Return ``value`` if it matches :data:`_SAFE_VALUE`, else raise a USER_ERROR.

    A sanitizer for the externally-sourced values that flow into the cloudflared
    argv — it blocks argument injection and keeps the subprocess input well-formed.
    """
    if not _SAFE_VALUE.match(value):
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"invalid {what}: {value!r}",
            remediation=(
                f"{what} may contain only letters, digits and '. _ : / -' "
                "and must start and end with a letter or digit"
            ),
        )
    return value


def resolve_hostname(explicit: str | None, deploy_dir: os.PathLike | str) -> str:
    """Public hostname: ``--hostname`` → ``$CULTURE_VLLM_PUBLIC_HOSTNAME`` → ``.cf-tunnel.env``.

    Never committed — raises a USER_ERROR (with a hint) when none of the three is set,
    and validates the resolved value against the safe charset.
    """
    if explicit:
        return _require_safe(explicit, what=_HOSTNAME_LABEL)
    env_val = os.environ.get(HOSTNAME_KEY)
    if env_val:
        return _require_safe(env_val, what=_HOSTNAME_LABEL)
    file_val = _env.read_env(tunnel_env_path(deploy_dir), HOSTNAME_KEY)
    if file_val:
        return _require_safe(file_val, what=_HOSTNAME_LABEL)
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message="no public hostname set",
        remediation=(
            f"pass --hostname, export ${HOSTNAME_KEY}, or set {HOSTNAME_KEY} in "
            f"{tunnel_env_path(deploy_dir)} (copy the scaffolded cf-tunnel.env.example)"
        ),
    )


def resolve_token(deploy_dir: os.PathLike | str) -> tuple[str, str]:
    """Token source from ``.cf-tunnel.env`` as ``(mode, value)``.

    ``("shushu", name)`` (preferred, the sealed-secret name) if set, else
    ``("plain", token)``. The sealed-secret *name* reaches the argv, so it is
    validated; the plaintext token goes via the environment and is not. Raises a
    USER_ERROR when neither key is set.
    """
    env_path = tunnel_env_path(deploy_dir)
    shushu_name = _env.read_env(env_path, TOKEN_SHUSHU_KEY)
    if shushu_name:
        return ("shushu", _require_safe(shushu_name, what=_SHUSHU_LABEL))
    plain = _env.read_env(env_path, TOKEN_PLAIN_KEY)
    if plain:
        return ("plain", plain)
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message="no cloudflared run-token configured",
        remediation=(
            f"set {TOKEN_SHUSHU_KEY} (preferred) or {TOKEN_PLAIN_KEY} in {env_path}; "
            "provision it with 'cultureflare remote-login setup --no-access --shushu --apply'"
        ),
    )


def build_command(mode: str, value: str) -> list[str]:
    """The cloudflared argv for the resolved token source (fixed argv, no shell).

    The token never appears here. ``shushu``: run cloudflared under ``shushu run``,
    which injects the sealed secret as ``$TUNNEL_TOKEN`` (cloudflared reads it
    natively). ``plain``: bare ``cloudflared tunnel run`` — the launcher sets
    ``TUNNEL_TOKEN`` in the child environment (see :func:`token_env`).
    """
    if mode == "shushu":
        return [
            "shushu",
            "run",
            "--inject",
            f"{TOKEN_ENV_VAR}={value}",
            "--",
            "cloudflared",
            "tunnel",
            "run",
        ]
    return ["cloudflared", "tunnel", "run"]


def token_env(mode: str, value: str) -> dict[str, str]:
    """Extra child environment for the launcher: the plaintext token, or nothing.

    ``plain`` mode sets ``TUNNEL_TOKEN`` so cloudflared authenticates without the
    token ever touching argv. ``shushu`` mode injects it itself, so this is empty.
    """
    if mode == "plain":
        return {TOKEN_ENV_VAR: value}
    return {}


def redacted(command: list[str]) -> list[str]:
    """Copy of ``command`` with a plaintext ``--token <value>`` masked to ``***``.

    The token no longer rides on argv, so this is a defensive no-op for the commands
    this module builds; it stays as a guard against any future argv that carries a
    literal ``--token`` reaching stdout/logs.
    """
    out = list(command)
    for i, tok in enumerate(out):
        if tok == "--token" and i + 1 < len(out) and out[i + 1] != "$TOKEN":
            out[i + 1] = "***"
    return out


def _write_pidfile(deploy_dir: os.PathLike | str, pid: int, pgid: int, argv0: str) -> None:
    """Record the tracked process as JSON so stop can verify identity before signaling."""
    payload = json.dumps({"pid": pid, "pgid": pgid, "argv0": argv0})
    pid_path(deploy_dir).write_text(payload + "\n", encoding="utf-8")


def read_pidfile(deploy_dir: os.PathLike | str) -> dict | None:
    """Parse the recorded ``{pid, pgid, argv0}``, tolerating a legacy bare-integer file."""
    try:
        text = pid_path(deploy_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        rec = json.loads(text)
        pid = int(rec["pid"])
        return {"pid": pid, "pgid": int(rec.get("pgid", pid)), "argv0": str(rec.get("argv0", ""))}
    except (ValueError, TypeError, KeyError):
        try:
            pid = int(text)  # legacy / hand-written bare-integer pidfile
        except ValueError:
            return None
        return {"pid": pid, "pgid": pid, "argv0": ""}


def _process_matches(pid: int, argv0: str) -> bool:
    """Best-effort check that live ``pid`` is the program we recorded (PID-reuse guard).

    On Linux, require ``/proc/<pid>/cmdline`` to mention the recorded ``argv0``; if
    procfs is absent (e.g. macOS) fall back to the liveness probe alone. If procfs
    exists but the cmdline is unreadable (a reused pid owned by another user), treat
    it as a non-match so stop never signals an unrelated process.
    """
    if not argv0:
        return True  # nothing recorded to verify against (legacy pidfile)
    proc = Path("/proc")
    if not proc.is_dir():
        return True  # no procfs — rely on the liveness probe only
    try:
        raw = (proc / str(pid) / "cmdline").read_bytes()
    except OSError:
        return False  # can't read it (gone, or not ours) — do not signal it
    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    return argv0 in cmdline


def tunnel_pid(deploy_dir: os.PathLike | str) -> int | None:
    """Return the live, identity-verified cloudflared pid for this deployment, or ``None``."""
    rec = read_pidfile(deploy_dir)
    if rec is None:
        return None
    pid = rec["pid"]
    try:
        os.kill(pid, 0)  # signal 0 — liveness probe, sends nothing
    except OSError:
        return None
    if not _process_matches(pid, rec["argv0"]):
        return None
    return pid


def start_tunnel(
    deploy_dir: os.PathLike | str, command: list[str], env: dict[str, str] | None = None
) -> int:
    """Spawn cloudflared detached, append output to ``cloudflared.log``, record the pid.

    ``start_new_session=True`` makes the child a process-group leader so ``stop_tunnel``
    can tear down shushu + cloudflared together. ``env`` carries any extra child
    environment (the plaintext ``TUNNEL_TOKEN`` in fallback mode). Returns the pid.
    """
    deploy = Path(deploy_dir)
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    logf = log_path(deploy).open("ab")  # appended across restarts
    try:
        proc = subprocess.Popen(  # fixed argv, no shell
            command,
            cwd=str(deploy),
            env=child_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"could not start {command[0]}: {exc}",
            remediation=INSTALL_HINT,
        ) from exc
    finally:
        logf.close()  # the child holds its own dup of the fd
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = proc.pid  # start_new_session makes the child its own group leader
    _write_pidfile(deploy, proc.pid, pgid, os.path.basename(command[0]))
    return proc.pid


# How long stop waits for a graceful exit before escalating, and after SIGKILL.
TERM_TIMEOUT = 10.0
KILL_TIMEOUT = 5.0


def _still_running(pid: int) -> bool:
    """Liveness for the stop loop; reaps the process first if it is our own zombie child."""
    try:
        os.waitpid(pid, os.WNOHANG)  # reap a dead-but-unreaped child so kill(0) reflects reality
    except OSError:
        pass  # not our child (the common case — cloudflared was reparented to init)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _signal_group(pid: int, pgid: int, sig: int) -> None:
    """Signal the recorded process group, falling back to the bare pid."""
    try:
        os.killpg(pgid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _wait_gone(pid: int, timeout: float) -> bool:
    """Poll until ``pid`` exits or ``timeout`` elapses; True if it is gone."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _still_running(pid):
            return True
        time.sleep(0.1)
    return not _still_running(pid)


def stop_tunnel(deploy_dir: os.PathLike | str) -> tuple[str, int | None]:
    """Terminate the recorded tunnel, confirming exit before clearing the pidfile.

    Signals the recorded process *group* (SIGTERM, then SIGKILL on timeout) and only
    removes the pidfile once the process is confirmed gone. Returns ``(status, pid)``:
    ``("idle", None)`` nothing tracked (stale/foreign pidfile removed), ``("stopped",
    pid)`` confirmed terminated, ``("failed", pid)`` still alive after SIGKILL (the
    pidfile is kept so the user can retry).
    """
    rec = read_pidfile(deploy_dir)
    path = pid_path(deploy_dir)
    pid = tunnel_pid(deploy_dir)  # liveness + identity-verified
    if pid is None:
        path.unlink(missing_ok=True)  # nothing running, or stale / not ours
        return ("idle", None)
    pgid = rec["pgid"] if rec else pid
    _signal_group(pid, pgid, signal.SIGTERM)
    if not _wait_gone(pid, TERM_TIMEOUT):
        _signal_group(pid, pgid, signal.SIGKILL)
        if not _wait_gone(pid, KILL_TIMEOUT):
            return ("failed", pid)  # keep the pidfile so the user can retry
    path.unlink(missing_ok=True)
    return ("stopped", pid)
