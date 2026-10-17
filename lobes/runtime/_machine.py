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
import time
from collections import OrderedDict
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


#: Seconds between two retries of a FAILED card detection (review2-b #6).
DETECT_RETRY_S = 60.0
#: Monotonic clock for the retry interval (a test seam).
_clock = time.monotonic
#: Most detectors the cache keeps (review3 #4): production uses one
#: (``detect_card``); tests swap many in, and an unbounded dict keyed by
#: function objects would grow for the life of the process.
CARD_CACHE_MAX = 8
#: detector -> (card, probed_at, failed). LRU-bounded at :data:`CARD_CACHE_MAX`.
_card_cache: OrderedDict[Any, tuple[str | None, float, bool]] = OrderedDict()


def _resolve_card(detect) -> str | None:
    """``detect().resolved``, memoised per detector object (review-b #10).

    Keyed on the detector itself, so the real ``detect_card`` (nvidia-smi +
    hostname probes) runs once per process, while a test that swaps the
    detector gets a fresh answer. Any detection that RETURNS is cached for
    good, ``UNKNOWN`` included (review3 #4): nvidia-smi answered, and asking
    again every minute only repeats the answer. Only a detector that RAISED
    (``None``) is retried, at most once per :data:`DETECT_RETRY_S`, so a
    long-running process (the watcher unit) recovers from an nvidia-smi that
    was not ready at boot (review2-b #6). The cache keeps the
    :data:`CARD_CACHE_MAX` most recently used detectors.
    """
    now = _clock()
    hit = _card_cache.get(detect)
    if hit is not None:
        card, at, failed = hit
        if not failed or now - at < DETECT_RETRY_S:
            _card_cache.move_to_end(detect)
            return card
    failed = False
    try:
        card = detect().resolved
    except Exception:  # detection must never break the adapter
        card, failed = None, True
    _card_cache[detect] = (card, now, failed)
    _card_cache.move_to_end(detect)
    while len(_card_cache) > CARD_CACHE_MAX:
        _card_cache.popitem(last=False)
    return card


def detected_card(detect=None) -> str | None:
    """The detected card name; a known card is probed once per process per detector."""
    return _resolve_card(detect if detect is not None else detect_card)


def _current_cli() -> str | None:
    return machine_cli_for(detected_card())


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
