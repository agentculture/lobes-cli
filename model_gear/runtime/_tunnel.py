"""Run the cloudflared tunnel that exposes the local vLLM API (stdlib only).

``model tunnel`` starts a standalone ``cloudflared tunnel run`` process (NOT a
compose service) that proxies the owner-chosen public hostname to the local
OpenAI-compatible server. The Cloudflare side (tunnel + ingress + DNS, run-token
sealed in ``shushu``) is provisioned once by ``cultureflare remote-login
--no-access``; this module owns the local lifecycle: resolve the hostname + token
source, build the command, and start/stop the background process via a pidfile.

All subprocess calls use fixed argv lists (no shell) — the bandit ``B404``/``B603``
skips in ``pyproject.toml`` cover them. The hostname and the run-token never live
in committed config: they come from ``$CULTURE_VLLM_PUBLIC_HOSTNAME`` / a gitignored
``.cf-tunnel.env`` in the deployment dir.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # fixed argv lists only, never shell=True
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


def resolve_hostname(explicit: str | None, deploy_dir: os.PathLike | str) -> str:
    """Public hostname: ``--hostname`` → ``$CULTURE_VLLM_PUBLIC_HOSTNAME`` → ``.cf-tunnel.env``.

    Never committed — raises a USER_ERROR (with a hint) when none of the three is set.
    """
    if explicit:
        return explicit
    env_val = os.environ.get(HOSTNAME_KEY)
    if env_val:
        return env_val
    file_val = _env.read_env(tunnel_env_path(deploy_dir), HOSTNAME_KEY)
    if file_val:
        return file_val
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
    ``("plain", token)``. Raises a USER_ERROR when neither key is set.
    """
    env_path = tunnel_env_path(deploy_dir)
    shushu_name = _env.read_env(env_path, TOKEN_SHUSHU_KEY)
    if shushu_name:
        return ("shushu", shushu_name)
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
    """The cloudflared command for the resolved token source (fixed argv, no shell).

    ``shushu``: inject the sealed token as ``$TOKEN`` and run cloudflared with it —
    the exact form ``cultureflare`` seals for. ``plain``: pass the literal run-token.
    """
    if mode == "shushu":
        return [
            "shushu",
            "run",
            "--inject",
            f"TOKEN={value}",
            "--",
            "cloudflared",
            "tunnel",
            "run",
            "--token",
            "$TOKEN",
        ]
    return ["cloudflared", "tunnel", "run", "--token", value]


def redacted(command: list[str]) -> list[str]:
    """Copy of ``command`` with a plaintext ``--token <value>`` masked to ``***``.

    The shushu form carries only a secret *name* plus the literal ``$TOKEN``, so it
    is safe to print as-is; the plain form carries the real run-token, which must
    never reach stdout/logs.
    """
    out = list(command)
    for i, tok in enumerate(out):
        if tok == "--token" and i + 1 < len(out) and out[i + 1] != "$TOKEN":
            out[i + 1] = "***"
    return out


def tunnel_pid(deploy_dir: os.PathLike | str) -> int | None:
    """Return the live cloudflared pid recorded for this deployment, or ``None``."""
    try:
        pid = int(pid_path(deploy_dir).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)  # signal 0 — liveness probe, sends nothing
    except OSError:
        return None
    return pid


def start_tunnel(deploy_dir: os.PathLike | str, command: list[str]) -> int:
    """Spawn cloudflared detached, append output to ``cloudflared.log``, record the pid.

    ``start_new_session=True`` makes the child a process-group leader so ``stop_tunnel``
    can tear down shushu + cloudflared together. Returns the pid.
    """
    deploy = Path(deploy_dir)
    logf = log_path(deploy).open("ab")  # appended across restarts
    try:
        proc = subprocess.Popen(  # fixed argv, no shell
            command,
            cwd=str(deploy),
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
    pid_path(deploy).write_text(f"{proc.pid}\n", encoding="utf-8")
    return proc.pid


def stop_tunnel(deploy_dir: os.PathLike | str) -> int | None:
    """Terminate the recorded cloudflared process (SIGTERM to its group).

    Returns the stopped pid, or ``None`` when nothing was running. Clears the
    pidfile either way (a stale pidfile is removed silently).
    """
    pid = tunnel_pid(deploy_dir)
    path = pid_path(deploy_dir)
    if pid is None:
        path.unlink(missing_ok=True)
        return None
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    path.unlink(missing_ok=True)
    return pid
