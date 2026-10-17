"""Machine-CLI adapter — card-aware wrapper over ``spark`` / ``thor`` / ``orin``.

The machine work (clocks, fans, sudo grants) lives in the per-machine CLIs, not
in lobes; the contract is ``docs/contracts/machine-cli-window.md``. This module
resolves the right CLI from the detected card + PATH, wraps the verbs lobes
calls, and, for the page-cache clear only, falls back to ``sudo -n`` when the
CLI is absent *and* an equivalent passwordless rule shows in ``sudo -n -l``.

Never prompts (every sudo call is ``-n``), never ``shell=True``, every
subprocess has a timeout, and every public call returns a typed result with a
reason instead of raising.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # fixed argv lists only, never shell=True
from dataclasses import dataclass, field
from typing import Any

from lobes.runtime._detect import detect_card

CARD_CLIS = {"spark": "spark", "thor": "thor", "orin": "orin"}
WINDOW_ACTIONS = ("show", "store", "lock", "restore")
SOURCE_CLI = "machine-cli"
SOURCE_SUDO = "sudo-fallback"

_TIMEOUT_S = 30
_SYSCTL_ARGV = ["sudo", "-n", "/usr/sbin/sysctl", "-w", "vm.drop_caches=3"]
_TEE_ARGV = ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"]
_SYSCTL_RE = re.compile(r"sysctl.*vm\.drop_caches")
_TEE_RE = re.compile(r"tee /proc/sys/vm/drop_caches")
_ALL_RE = re.compile(r"NOPASSWD:\s*ALL\b")


@dataclass(frozen=True)
class MachineResult:
    ok: bool
    reason: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    source: str | None = None  # "machine-cli" | "sudo-fallback" | None


@dataclass(frozen=True)
class GrantStatus:
    granted: bool
    rule: str | None = None  # "sysctl" | "tee" | "all" | None
    reason: str | None = None  # why not granted


@dataclass(frozen=True)
class _Proc:
    returncode: int
    stdout: str
    stderr: str


def _run(argv: list[str], *, stdin: str | None = None) -> _Proc | None:
    """Run a fixed argv with a timeout; ``None`` when it cannot run."""
    try:
        p = subprocess.run(  # fixed argv, no shell
            argv,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _Proc(p.returncode, p.stdout or "", p.stderr or "")


def machine_cli_for(card: str | None) -> str | None:
    """The machine CLI name for a detected card, only if it is on PATH."""
    name = CARD_CLIS.get(card or "")
    if name is None or shutil.which(name) is None:
        return None
    return name


def _current_cli() -> str | None:
    try:
        card = detect_card().resolved
    except Exception:  # detection must never break the adapter
        return None
    return machine_cli_for(card)


def _call_cli(cli: str, args: list[str]) -> MachineResult:
    proc = _run([cli, *args, "--json"])
    if proc is None:
        return MachineResult(False, "failed")
    try:
        data = json.loads(proc.stdout)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except ValueError:
        reason = "failed" if proc.returncode != 0 else "malformed_output"
        return MachineResult(False, reason, source=SOURCE_CLI)
    ok = bool(data.get("ok")) and proc.returncode == 0
    reason = None if ok else (data.get("reason") or "failed")
    return MachineResult(ok, reason, data, SOURCE_CLI)


def window(action: str) -> MachineResult:
    """``<cli> window show|store|lock|restore --json`` for the detected card."""
    if action not in WINDOW_ACTIONS:
        return MachineResult(False, "bad_action")
    cli = _current_cli()
    if cli is None:
        return MachineResult(False, "tool_missing")
    return _call_cli(cli, ["window", action])


def grant_status() -> GrantStatus:
    """Detect a passwordless drop-caches rule via ``sudo -n -l`` (never prompts)."""
    proc = _run(["sudo", "-n", "-l"])
    if proc is None or proc.returncode != 0:
        return GrantStatus(False, None, "not_granted")
    listing = proc.stdout
    if _SYSCTL_RE.search(listing):
        return GrantStatus(True, "sysctl")
    if _TEE_RE.search(listing):
        return GrantStatus(True, "tee")
    if _ALL_RE.search(listing):
        return GrantStatus(True, "all")
    return GrantStatus(False, None, "not_granted")


def drop_caches() -> MachineResult:
    """Clear the page cache: machine CLI first, else the sudo rule if granted."""
    cli = _current_cli()
    if cli is not None:
        return _call_cli(cli, ["drop-caches"])
    grant = grant_status()
    if not grant.granted:
        return MachineResult(False, "not_granted")
    try:
        os.sync()
    except OSError:
        pass
    if grant.rule == "tee":
        proc = _run(_TEE_ARGV, stdin="3")
    else:
        proc = _run(_SYSCTL_ARGV)
    if proc is None or proc.returncode != 0:
        return MachineResult(False, "failed", {"rule": grant.rule}, SOURCE_SUDO)
    return MachineResult(True, None, {"cleared": True, "rule": grant.rule}, SOURCE_SUDO)
