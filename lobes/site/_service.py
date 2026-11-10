"""Keep ``lobes site`` up across reboots: a systemd USER unit.

Mirrors the box's other long-lived dashboards (heat-governor's ``mesh serve``):
a user unit under ``~/.config/systemd/user/lobes-site.service`` with
``Restart=always``, enabled for ``default.target``. A user unit only starts at
boot with linger on, so linger is REQUIRED; without it the apply refuses with
the remediation.

Binding a tailnet address at boot can race tailscaled; ``Restart=always`` with
a short ``RestartSec`` retries until the address exists rather than giving up.

Re-running ``install --apply`` with different flags rewrites the unit and
restarts it (the site holds no state, so there is nothing to lose). Every file
written is recorded in ``<deploy>/.lobes-site.json``; uninstall removes exactly
those files (only ones named ``lobes-site.service``), then the manifest.

``root``/``runner`` come from the watcher's :class:`InstallContext`, so tests
never touch real systemd.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from lobes.watcher._install import InstallContext, _quote, default_context, linger_enabled

UNIT_NAME = "lobes-site.service"
MANIFEST = ".lobes-site.json"
_USER_UNIT_DIR = ".config/systemd/user"

__all__ = [
    "MANIFEST",
    "UNIT_NAME",
    "SitePlan",
    "apply_install",
    "apply_uninstall",
    "default_context",
    "plan_install",
    "plan_uninstall",
    "status",
]


def unit_path(ctx: InstallContext) -> Path:
    return ctx.under_root(Path(ctx.home) / _USER_UNIT_DIR / UNIT_NAME)


def read_manifest(deploy_dir: Path) -> Optional[dict]:
    try:
        data = json.loads((Path(deploy_dir) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def render_unit(
    ctx: InstallContext,
    deploy_dir: Path,
    *,
    host: str,
    port: int,
    gateway: Optional[str],
) -> str:
    argv = [
        *ctx.exec_argv,
        "site",
        "--compose-dir",
        os.path.abspath(os.path.expanduser(str(deploy_dir))),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if gateway:
        argv += ["--gateway", gateway]
    argv.append("--apply")
    env_line = f"Environment={_quote('PATH=' + ctx.path_env)}\n" if ctx.path_env else ""
    return (
        "# lobes site (Mesh, Rollout, Residents) — read-only, kept up across reboots.\n"
        "# Installed by `lobes site install --apply`; removed by\n"
        "# `lobes site uninstall --apply`. Restart=always also covers a boot where\n"
        "# the bind address (e.g. the tailnet IP) is not up yet.\n"
        "[Unit]\n"
        f"Description=lobes site on {host}:{port}\n"
        "After=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"{env_line}"
        f"ExecStart={' '.join(_quote(a) for a in argv)}\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


@dataclass
class SitePlan:
    deploy_dir: Path
    unit_path: Path
    content: str
    linger: Optional[bool]
    blockers: list
    commands: list
    replaces: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "deploy_dir": str(self.deploy_dir),
            "unit_type": "user",
            "unit_path": str(self.unit_path),
            "linger": self.linger,
            "replaces": self.replaces,
            "blockers": list(self.blockers),
            "commands": [list(c) for c in self.commands],
            "manifest": str(self.deploy_dir / MANIFEST),
            "content": self.content,
        }


def plan_install(
    deploy_dir: Path,
    ctx: InstallContext,
    *,
    host: str,
    port: int,
    gateway: Optional[str] = None,
) -> SitePlan:
    """What ``install --apply`` would do. Read-only (one ``loginctl`` probe)."""
    deploy_dir = Path(deploy_dir)
    linger = linger_enabled(ctx)
    blockers = []
    if port == 0:
        blockers.append("--port 0 picks a new port on every restart: pass a fixed --port")
    if linger is not True:
        state = "unknown" if linger is None else "off"
        blockers.append(
            f"linger is {state} for {ctx.user}: a user unit would not start at boot; "
            f"run 'sudo loginctl enable-linger {ctx.user}'"
        )
    return SitePlan(
        deploy_dir=deploy_dir,
        unit_path=unit_path(ctx),
        content=render_unit(ctx, deploy_dir, host=host, port=port, gateway=gateway),
        linger=linger,
        blockers=blockers,
        commands=[
            ctx.systemctl(False, "daemon-reload"),
            ctx.systemctl(False, "enable", UNIT_NAME),
            ctx.systemctl(False, "restart", UNIT_NAME),
        ],
        replaces=read_manifest(deploy_dir) is not None or unit_path(ctx).exists(),
    )


def _check(ctx: InstallContext, argv: list, remediation: str) -> None:
    rc, _, err = ctx.runner(argv)
    if rc != 0:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"'{' '.join(argv)}' failed (exit {rc}): {err.strip() or 'no output'}",
            remediation=remediation,
        )


def apply_install(plan: SitePlan, ctx: InstallContext) -> dict[str, Any]:
    if plan.blockers:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="refusing to install the site service: " + "; ".join(plan.blockers),
            remediation=plan.blockers[-1].split(": ", 1)[-1],
        )
    plan.unit_path.parent.mkdir(parents=True, exist_ok=True)
    plan.unit_path.write_text(plan.content, encoding="utf-8")
    manifest = {
        "unit": UNIT_NAME,
        "unit_type": "user",
        "files": [str(plan.unit_path)],
        "installed_at": int(time.time()),
    }
    # record before enabling, so a failed enable can still be uninstalled
    (plan.deploy_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    remediation = f"inspect with 'systemctl --user status {UNIT_NAME}' and the journal"
    for argv in plan.commands:
        _check(ctx, argv, remediation)
    return {**manifest, "manifest": str(plan.deploy_dir / MANIFEST)}


def plan_uninstall(deploy_dir: Path, ctx: InstallContext) -> dict[str, Any]:
    manifest = read_manifest(deploy_dir)
    if manifest is None:
        return {"installed": False, "files": [], "commands": []}
    listed = [str(f) for f in manifest.get("files") or []]
    return {
        "installed": True,
        "files": [f for f in listed if Path(f).name == UNIT_NAME],
        "skipped": [f for f in listed if Path(f).name != UNIT_NAME],
        "manifest": str(Path(deploy_dir) / MANIFEST),
        "commands": [
            ctx.systemctl(False, "disable", "--now", UNIT_NAME),
            ctx.systemctl(False, "daemon-reload"),
        ],
    }


def apply_uninstall(deploy_dir: Path, ctx: InstallContext) -> dict[str, Any]:
    plan = plan_uninstall(deploy_dir, ctx)
    if not plan["installed"]:
        return plan
    ctx.runner(ctx.systemctl(False, "disable", "--now", UNIT_NAME))
    for f in plan["files"]:
        Path(f).unlink(missing_ok=True)
    ctx.runner(ctx.systemctl(False, "daemon-reload"))
    (Path(deploy_dir) / MANIFEST).unlink(missing_ok=True)
    return {**plan, "removed": plan["files"]}


def status(deploy_dir: Path, ctx: InstallContext) -> dict[str, Any]:
    manifest = read_manifest(deploy_dir)
    files = (manifest or {}).get("files", [])
    out: dict[str, Any] = {
        "installed": manifest is not None,
        "unit": UNIT_NAME,
        "files": files,
        "files_present": all(Path(f).exists() for f in files),
        "linger": linger_enabled(ctx),
        "enabled": None,
        "active": None,
    }
    if manifest is not None:
        _, act, _ = ctx.runner(ctx.systemctl(False, "is-active", UNIT_NAME))
        _, en, _ = ctx.runner(ctx.systemctl(False, "is-enabled", UNIT_NAME))
        out["active"] = act.strip() or "unknown"
        out["enabled"] = en.strip() or "unknown"
    return out
