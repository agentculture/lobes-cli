"""Rollback of one member: restore the pre-apply .env, recreate the gateway
from the retained ``lobes-gateway:<old-version>`` image (no build, no index).

Dry-run unless ``apply=True``. Never invoked automatically by apply; apply
only prints :func:`rollback_command`.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Any, Optional

from lobes.rollout import _transport
from lobes.rollout._transport import Runner, as_target

GATEWAY_IMAGE_REPO = "lobes-gateway"
DEFAULT_GATEWAY_IMAGE = "lobes-gateway:latest"
GATEWAY_CONTAINER = "model-gear-gateway"
_PLACEHOLDER = "<compose gateway image>"


@dataclass(frozen=True)
class Action:
    description: str
    argv: tuple  # remote argv

    @property
    def command(self) -> str:
        return " ".join(self.argv)


@dataclass
class RollbackResult:
    member: str
    actions: list = field(default_factory=list)
    applied: bool = False
    ok: bool = True
    error: str = ""


def rollback_command(member: str) -> str:
    return f"lobes rollout rollback --member {member} --apply"


def backup_from_journal(journal: Any, member: str) -> Optional[str]:
    """Latest ok backup_env path recorded for ``member``, if any."""
    found = None
    for r in journal.records():
        if r.get("member") == member and r.get("step") == "backup_env" and r.get("result") == "ok":
            d = r.get("detail")
            path = d.get("path") if isinstance(d, dict) else d
            if isinstance(path, str) and path:
                found = path
    return found


def build_actions(target: Any, old_version: str, backup_path: str, gateway_image: str) -> list:
    t = as_target(target)
    env_path = posixpath.join(posixpath.dirname(backup_path), ".env")
    return [
        Action("restore pre-apply .env", ("cp", "-p", backup_path, env_path)),
        Action(
            "retag retained gateway image",
            ("docker", "tag", f"{GATEWAY_IMAGE_REPO}:{old_version}", gateway_image),
        ),
        Action(
            "recreate gateway (no build)",
            (t.lobes_path or "lobes", "up", "gateway", "--apply"),
        ),
    ]


def rollback(
    member: str,
    target: Any,
    old_version: str,
    backup_path: Optional[str],
    *,
    gateway_image: Optional[str] = None,
    apply: bool = False,
    runner: Runner = _transport.default_runner,
    timeout: float = 300.0,
) -> RollbackResult:
    res = RollbackResult(member)
    if not backup_path or not backup_path.startswith("/"):
        res.ok, res.error = False, "no absolute pre-apply .env backup path known"
        return res
    if not old_version:
        res.ok, res.error = False, "old version unknown"
        return res
    image = gateway_image
    if apply and not image:
        probe = _transport.run(
            target,
            ["docker", "inspect", "--format", "{{.Config.Image}}", GATEWAY_CONTAINER],
            runner=runner,
        )
        image = probe.stdout.strip() if probe.ok and probe.stdout.strip() else DEFAULT_GATEWAY_IMAGE
    res.actions = build_actions(target, old_version, backup_path, image or _PLACEHOLDER)
    if not apply:
        return res
    res.applied = True
    for a in res.actions:
        r = _transport.run(target, a.argv, timeout=timeout, runner=runner)
        if not r.ok:
            res.ok, res.error = False, f"{a.description} failed (exit {r.returncode})"
            return res
    return res
