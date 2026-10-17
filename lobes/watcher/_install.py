"""Install, uninstall and report the watcher's systemd unit.

The first host unit lobes ships, so it is explicit and removable:

* **user unit** (default): ``~/.config/systemd/user/lobes-watcher.service``. No
  root needed, but a user unit only survives a reboot with linger on, so
  linger is REQUIRED (``loginctl show-user <user> -p Linger`` reads
  ``Linger=yes``); without it the apply refuses with the remediation.
* **system unit** (``--system``): ``/etc/systemd/system/lobes-watcher.service``
  with ``User=<invoking user>``, written through ``sudo -n`` (never prompts;
  refuses with a remediation when passwordless sudo is unavailable).

Every file written is recorded in ``<deploy>/.lobes-watcher.json``; uninstall
removes exactly those files (and only ones named ``lobes-watcher.service``),
then the manifest.

``root`` and ``runner`` are injectable so tests never touch real systemd.
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import subprocess  # fixed argv lists only, never shell=True
import sys
import tempfile
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from lobes.watcher import _run

UNIT_NAME = "lobes-watcher.service"
UNIT_TEMPLATE = "lobes-watcher.service.in"
MANIFEST = ".lobes-watcher.json"
_USER_UNIT_DIR = ".config/systemd/user"
_SYSTEM_UNIT_DIR = "/etc/systemd/system"
_TIMEOUT_S = 30

Runner = Callable[[list[str]], "tuple[int, str, str]"]


def _subprocess_runner(argv: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(  # fixed argv, no shell
            argv, capture_output=True, text=True, check=False, timeout=_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return p.returncode, p.stdout or "", p.stderr or ""


def _lobes_argv() -> list[str]:
    exe = shutil.which("lobes")
    if exe:
        return [str(Path(exe).resolve())]
    return [sys.executable, "-m", "lobes"]


@dataclass
class InstallContext:
    root: Path = Path("/")
    home: Path = field(default_factory=Path.home)
    user: str = field(default_factory=getpass.getuser)
    runner: Runner = _subprocess_runner
    exec_argv: list[str] = field(default_factory=_lobes_argv)
    system: bool = False
    path_env: str = field(default_factory=lambda: os.environ.get("PATH", ""))

    def under_root(self, path: Path | str) -> Path:
        return Path(self.root).joinpath(str(path).lstrip("/"))

    def unit_path(self, system: bool) -> Path:
        if system:
            return self.under_root(Path(_SYSTEM_UNIT_DIR) / UNIT_NAME)
        return self.under_root(Path(self.home) / _USER_UNIT_DIR / UNIT_NAME)

    def systemctl(self, system: bool, *args: str) -> list[str]:
        if system:
            return ["sudo", "-n", "systemctl", *args]
        return ["systemctl", "--user", *args]


def default_context(system: bool = False) -> InstallContext:
    return InstallContext(system=system)


@dataclass
class InstallPlan:
    deploy_dir: Path
    unit_type: str  # "user" | "system"
    unit_path: Path
    content: str
    linger: bool | None
    blockers: list[str]
    commands: list[list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "deploy_dir": str(self.deploy_dir),
            "unit_type": self.unit_type,
            "unit_path": str(self.unit_path),
            "linger": self.linger,
            "blockers": list(self.blockers),
            "commands": [list(c) for c in self.commands],
            "manifest": str(self.deploy_dir / MANIFEST),
            "content": self.content,
        }


def _quote(arg: str) -> str:
    """systemd ExecStart quoting: double-quote an arg with whitespace or quotes."""
    if arg and not any(ch.isspace() or ch in "\"'\\" for ch in arg):
        return arg
    return '"' + arg.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_unit(ctx: InstallContext, deploy_dir: Path, system: bool) -> str:
    template = files("lobes.templates").joinpath(UNIT_TEMPLATE).read_text(encoding="utf-8")
    # `watcher run` is dry-run without --apply (d8); the unit must commit.
    argv = [
        *ctx.exec_argv,
        "watcher",
        "run",
        "--compose-dir",
        str(Path(deploy_dir).resolve()),
        "--apply",
    ]
    env_line = f"Environment={_quote('PATH=' + ctx.path_env)}\n" if ctx.path_env else ""
    return (
        template.replace("@EXEC_START@", " ".join(_quote(a) for a in argv))
        .replace("@ENV_LINE@", env_line)
        .replace("@USER_LINE@", f"User={ctx.user}\n" if system else "")
        .replace("@WANTED_BY@", "multi-user.target" if system else "default.target")
    )


def linger_enabled(ctx: InstallContext) -> bool | None:
    """``True``/``False`` from ``loginctl show-user``; ``None`` when unknown."""
    rc, out, _ = ctx.runner(["loginctl", "show-user", ctx.user, "-p", "Linger"])
    if rc != 0:
        return None
    for line in out.splitlines():
        if line.strip().startswith("Linger="):
            return line.split("=", 1)[1].strip() == "yes"
    return None


def read_manifest(deploy_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(deploy_dir) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def plan_install(deploy_dir: Path, ctx: InstallContext) -> InstallPlan:
    """What ``install --apply`` would do. Read-only (one ``loginctl`` probe)."""
    deploy_dir = Path(deploy_dir)
    system = ctx.system
    linger = linger_enabled(ctx)
    blockers: list[str] = []
    if read_manifest(deploy_dir) is not None:
        blockers.append(
            f"already installed (manifest {deploy_dir / MANIFEST}); "
            "run 'lobes watcher uninstall --apply' first"
        )
    if not system and linger is not True:
        state = "unknown" if linger is None else "off"
        blockers.append(
            f"linger is {state} for {ctx.user}: a user unit would not start at boot; "
            f"run 'sudo loginctl enable-linger {ctx.user}' or pass --system"
        )
    unit = ctx.unit_path(system)
    commands = [
        ctx.systemctl(system, "daemon-reload"),
        ctx.systemctl(system, "enable", "--now", UNIT_NAME),
    ]
    if system:
        commands.insert(0, ["sudo", "-n", "install", "-m", "0644", "<rendered unit>", str(unit)])
    return InstallPlan(
        deploy_dir=deploy_dir,
        unit_type="system" if system else "user",
        unit_path=unit,
        content=render_unit(ctx, deploy_dir, system),
        linger=linger,
        blockers=blockers,
        commands=commands,
    )


def _check(ctx: InstallContext, argv: list[str], remediation: str) -> None:
    rc, _, err = ctx.runner(argv)
    if rc != 0:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"'{' '.join(argv)}' failed (exit {rc}): {err.strip() or 'no output'}",
            remediation=remediation,
        )


def _write_unit(plan: InstallPlan, ctx: InstallContext) -> None:
    if plan.unit_type == "user":
        plan.unit_path.parent.mkdir(parents=True, exist_ok=True)
        plan.unit_path.write_text(plan.content, encoding="utf-8")
        return
    fd, tmp = tempfile.mkstemp(prefix="lobes-watcher-", suffix=".service")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(plan.content)
        _check(
            ctx,
            ["sudo", "-n", "install", "-m", "0644", tmp, str(plan.unit_path)],
            "passwordless sudo is required for --system; install a user unit instead "
            f"(enable linger: 'sudo loginctl enable-linger {ctx.user}') or run with sudo access",
        )
    finally:
        Path(tmp).unlink(missing_ok=True)


def apply_install(plan: InstallPlan, ctx: InstallContext) -> dict[str, Any]:
    if plan.blockers:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="refusing to install the watcher: " + "; ".join(plan.blockers),
            remediation=plan.blockers[-1].split(": ", 1)[-1],
        )
    system = plan.unit_type == "system"
    _write_unit(plan, ctx)
    manifest = {
        "unit": UNIT_NAME,
        "unit_type": plan.unit_type,
        "files": [str(plan.unit_path)],
        "installed_at": int(time.time()),
    }
    # record before enabling, so a failed enable can still be uninstalled
    (plan.deploy_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    remediation = "inspect with 'systemctl status lobes-watcher' and the journal"
    _check(ctx, ctx.systemctl(system, "daemon-reload"), remediation)
    _check(ctx, ctx.systemctl(system, "enable", "--now", UNIT_NAME), remediation)
    return {**manifest, "manifest": str(plan.deploy_dir / MANIFEST)}


def _removable(path: str) -> bool:
    return Path(path).name == UNIT_NAME


def plan_uninstall(deploy_dir: Path, ctx: InstallContext) -> dict[str, Any]:
    manifest = read_manifest(deploy_dir)
    if manifest is None:
        return {"installed": False, "files": [], "commands": []}
    system = manifest.get("unit_type") == "system"
    listed = [str(f) for f in manifest.get("files") or []]
    return {
        "installed": True,
        "unit_type": "system" if system else "user",
        "files": [f for f in listed if _removable(f)],
        "skipped": [f for f in listed if not _removable(f)],
        "manifest": str(Path(deploy_dir) / MANIFEST),
        "commands": [
            ctx.systemctl(system, "disable", "--now", UNIT_NAME),
            ctx.systemctl(system, "daemon-reload"),
        ],
    }


def apply_uninstall(deploy_dir: Path, ctx: InstallContext) -> dict[str, Any]:
    plan = plan_uninstall(deploy_dir, ctx)
    if not plan["installed"]:
        return plan
    system = plan["unit_type"] == "system"
    # stopping a unit that is already gone is not an error worth refusing on
    ctx.runner(ctx.systemctl(system, "disable", "--now", UNIT_NAME))
    for f in plan["files"]:
        if system:
            _check(ctx, ["sudo", "-n", "rm", "-f", f], "remove it by hand with sudo")
        else:
            Path(f).unlink(missing_ok=True)
    ctx.runner(ctx.systemctl(system, "daemon-reload"))
    (Path(deploy_dir) / MANIFEST).unlink(missing_ok=True)
    return {**plan, "removed": plan["files"]}


def status(deploy_dir: Path, ctx: InstallContext) -> dict[str, Any]:
    deploy_dir = Path(deploy_dir)
    manifest = read_manifest(deploy_dir)
    system = bool(manifest and manifest.get("unit_type") == "system")
    out: dict[str, Any] = {
        "installed": manifest is not None,
        "unit_type": (manifest or {}).get("unit_type"),
        "files": (manifest or {}).get("files", []),
        "files_present": all(Path(f).exists() for f in (manifest or {}).get("files", [])),
        "linger": linger_enabled(ctx),
        "active": None,
    }
    if manifest is not None:
        _, stdout, _ = ctx.runner(ctx.systemctl(system, "is-active", UNIT_NAME))
        out["active"] = stdout.strip() or "unknown"
    state_file = deploy_dir / _run.STATE_FILE
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    out["last_clear"] = state.get("last_clear") if isinstance(state, dict) else None
    out["last_result"] = state.get("last_result") if isinstance(state, dict) else None
    return out
