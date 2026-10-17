"""The watcher loop (lane-safety contract, section 5).

Polls ``<log dir>/lane-state/*.gated`` (the markers ``mg-logwrap.sh`` keeps
while a lane waits on the memory gate, contract section 4). When at least one
marker is present it runs the machine adapter's ``drop_caches()`` — machine CLI
first, else a passwordless sudo rule — at most once per
``LOBES_WATCHER_MIN_INTERVAL_S`` (default 300). With no grant the adapter runs
nothing and the watcher only logs.

The watcher never issues a docker command: its only side effect besides its own
state file is the adapter call. Every action is one log line on stderr (the
unit's journal captures it).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lobes.runtime import _compose, _env
from lobes.runtime._machine import MachineResult, drop_caches

MIN_INTERVAL_ENV = "LOBES_WATCHER_MIN_INTERVAL_S"
GATE_TIMEOUT_ENV = "LOBES_MEM_GATE_TIMEOUT_S"
DEFAULT_MIN_INTERVAL_S = 300
DEFAULT_GATE_TIMEOUT_S = 900  # mirrors mg-logwrap.sh's default
DEFAULT_POLL_S = 10
STATE_FILE = ".lobes-watcher-state.json"
LANE_STATE_DIRNAME = "lane-state"
MARKER_SUFFIX = ".gated"


def _stderr_log(line: str) -> None:
    sys.stderr.write(line.rstrip("\n") + "\n")
    sys.stderr.flush()


def scan_markers(state_dir: Path) -> list[dict[str, Any]]:
    """Every ``*.gated`` marker in ``state_dir``, sorted by lane name.

    Symlinks are skipped (never follow one out of the state dir). A marker that
    is not a JSON object still counts — its lane is still waiting — and is
    flagged ``malformed``.
    """
    if not state_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(state_dir.glob(f"*{MARKER_SUFFIX}")):
        if p.is_symlink() or not p.is_file():
            continue
        lane = p.name[: -len(MARKER_SUFFIX)]
        entry: dict[str, Any] = {"lane": lane, "path": str(p), "malformed": False}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except (OSError, ValueError):
            entry["malformed"] = True
            out.append(entry)
            continue
        for key in ("since", "need_kib", "avail_kib"):
            val = data.get(key)
            entry[key] = val if isinstance(val, int) and not isinstance(val, bool) else None
        out.append(entry)
    return out


@dataclass
class Watcher:
    state_dir: Path
    state_file: Path
    min_interval_s: int = DEFAULT_MIN_INTERVAL_S
    stale_after_s: int = 2 * DEFAULT_GATE_TIMEOUT_S
    clear: Callable[[], MachineResult] | None = None
    clock: Callable[[], float] = time.time
    log: Callable[[str], None] = _stderr_log
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _quiet: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._state = self._load_state()

    # -- persisted state (so a restart cannot bypass the rate limit) --------
    def _load_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"last_attempt": None, "last_clear": None, "last_result": None}
        return data if isinstance(data, dict) else {}

    def _save_state(self) -> None:
        try:
            tmp = self.state_file.with_name(self.state_file.name + ".tmp")
            tmp.write_text(json.dumps(self._state), encoding="utf-8")
            os.replace(tmp, self.state_file)
        except OSError as exc:
            self.log(f"lobes-watcher: could not persist state to {self.state_file}: {exc}")

    @property
    def last_clear(self) -> float | None:
        return self._state.get("last_clear")

    # -- one poll ----------------------------------------------------------
    def _describe(self, m: dict[str, Any]) -> str:
        if m["malformed"]:
            return f"lane={m['lane']} marker=malformed"
        return f"lane={m['lane']} need_kib={m.get('need_kib')} avail_kib={m.get('avail_kib')}"

    def _is_stale(self, m: dict[str, Any], now: float) -> bool:
        since = m.get("since")
        return isinstance(since, int) and now - since > self.stale_after_s

    def tick(self) -> list[dict[str, Any]]:
        """Poll once; return one action record per marker (empty when none)."""
        markers = scan_markers(self.state_dir)
        if not markers:
            self._quiet.clear()
            return []
        now = self.clock()
        last = self._state.get("last_attempt")
        if isinstance(last, (int, float)) and now - last < self.min_interval_s:
            action, reason = "rate-limited", None
        else:
            clear = self.clear if self.clear is not None else drop_caches
            result = clear()
            self._state["last_attempt"] = now
            if result.ok:
                action = "cleared"
                self._state["last_clear"] = now
            else:
                action = f"skipped:{result.reason or 'failed'}"
            self._state["last_result"] = action
            self._save_state()
            reason = result.source
            self._quiet.clear()
        actions = []
        for m in markers:
            stale = self._is_stale(m, now)
            rec = {
                "lane": m["lane"],
                "need_kib": m.get("need_kib"),
                "avail_kib": m.get("avail_kib"),
                "stale": stale,
                "action": action,
            }
            actions.append(rec)
            # a rate-limited marker is logged once per window, not every poll
            if action == "rate-limited":
                if m["lane"] in self._quiet:
                    continue
                self._quiet.add(m["lane"])
            line = f"lobes-watcher: {self._describe(m)} action={action}"
            if reason:
                line += f" via={reason}"
            if stale:
                line += " (stale marker: older than the gate timeout; acting anyway)"
            self.log(line)
        return actions


def _int_knob(env: dict[str, str], key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        raw = env.get(key)
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return val if val >= 0 else default


def lane_state_dir(deploy_dir: Path) -> Path:
    env = _env.read_env_file(deploy_dir / _compose.ENV_FILE)
    log_dir = _compose.durable_log_dir(deploy_dir, env.get(_compose.LOG_DIR_ENV) or None)
    return log_dir / LANE_STATE_DIRNAME


def watcher_for(deploy_dir: Path, **kw: Any) -> Watcher:
    """A :class:`Watcher` wired to a deployment dir's log dir and ``.env`` knobs."""
    deploy_dir = Path(deploy_dir)
    env = _env.read_env_file(deploy_dir / _compose.ENV_FILE)
    gate_timeout = _int_knob(env, GATE_TIMEOUT_ENV, DEFAULT_GATE_TIMEOUT_S)
    return Watcher(
        state_dir=lane_state_dir(deploy_dir),
        state_file=deploy_dir / STATE_FILE,
        min_interval_s=_int_knob(env, MIN_INTERVAL_ENV, DEFAULT_MIN_INTERVAL_S),
        stale_after_s=2 * gate_timeout,
        **kw,
    )


def run_forever(watcher: Watcher, poll_s: float = DEFAULT_POLL_S) -> None:  # pragma: no cover
    watcher.log(
        f"lobes-watcher: watching {watcher.state_dir} "
        f"(min interval {watcher.min_interval_s}s, poll {poll_s}s)"
    )
    while True:
        try:
            watcher.tick()
        except Exception as exc:  # noqa: BLE001 - one bad poll must not kill the loop
            watcher.log(f"lobes-watcher: poll failed: {exc.__class__.__name__}: {exc}")
        time.sleep(poll_s)
