"""The watcher loop (lane-safety contract, section 5).

Polls ``<log dir>/lane-state/*.gated`` (the markers ``mg-logwrap.sh`` keeps
while a lane waits on the memory gate, contract section 4). When at least one
marker is present it runs the machine adapter's ``drop_caches()`` — machine CLI
first, else a passwordless sudo rule — at most once per
``LOBES_WATCHER_MIN_INTERVAL_S`` (default 300). With no grant the adapter runs
nothing and the watcher only logs.

A marker older than twice ``LOBES_MEM_GATE_TIMEOUT_S``, or whose lane has no
running container, is stale: logged once and not acted on. While any OTHER
lane is mid-boot (docker ``health: starting``) the clear is deferred, because a
drop during a sibling vLLM's memory profiling can trip it on unified memory.

The watcher runs no docker command that changes anything: its one docker call
is the fixed-shape, read-only ``docker ps`` of
:func:`lobes.runtime._compose.lane_health_argv` (filtered by the deployment's
compose project label)
(unreadable docker = unknown: age-only staleness, no deferral). Its only side
effect besides its own state file is the adapter call. Every action is one log
line on stderr (the unit's journal captures it).
"""

from __future__ import annotations

import json
import os
import signal
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
LANE_STATE_DIRNAME = _compose.LANE_STATE_DIRNAME
MARKER_SUFFIX = ".gated"


def _stderr_log(line: str) -> None:
    sys.stderr.write(line.rstrip("\n") + "\n")
    sys.stderr.flush()


def _marker_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_marker(path: Path) -> dict[str, Any] | None:
    """The marker's JSON object, or ``None`` when unreadable / not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _int_field(data: dict[str, Any], key: str) -> int | None:
    """``data[key]`` when it is a real int (a bool is not), else ``None``."""
    val = data.get(key)
    return val if isinstance(val, int) and not isinstance(val, bool) else None


def _marker_entry(p: Path) -> dict[str, Any]:
    """One marker's scan entry: lane, path, mtime and its int fields, or ``malformed``."""
    entry: dict[str, Any] = {
        "lane": p.name[: -len(MARKER_SUFFIX)],
        "path": str(p),
        "malformed": False,
        "mtime": _marker_mtime(p),
    }
    data = _read_marker(p)
    if data is None:
        entry["malformed"] = True
    else:
        entry.update({key: _int_field(data, key) for key in ("since", "need_kib", "avail_kib")})
    return entry


def scan_markers(state_dir: Path) -> list[dict[str, Any]]:
    """Every ``*.gated`` marker in ``state_dir``, sorted by lane name.

    Symlinks are skipped (never follow one out of the state dir). A marker that
    is not a JSON object still counts — its lane is still waiting — and is
    flagged ``malformed``.
    """
    if not state_dir.is_dir():
        return []
    return [
        _marker_entry(p)
        for p in sorted(state_dir.glob(f"*{MARKER_SUFFIX}"))
        if not p.is_symlink() and p.is_file()
    ]


#: Actions logged once per lane until the action changes (not every poll).
ACTION_STALE = "ignored:stale"
ACTION_DEFERRED = "deferred:lane-starting"
_QUIET_ACTIONS = ("rate-limited", ACTION_DEFERRED, ACTION_STALE, "would-clear")


@dataclass
class Watcher:
    state_dir: Path
    state_file: Path
    min_interval_s: int = DEFAULT_MIN_INTERVAL_S
    stale_after_s: int = 2 * DEFAULT_GATE_TIMEOUT_S
    clear: Callable[[], MachineResult] | None = None
    clock: Callable[[], float] = time.time
    log: Callable[[str], None] = _stderr_log
    #: ``MG_LOG_NAME`` -> ``"starting"`` | ``"running"`` for the running lanes, or
    #: ``None`` when unknown (see :func:`lobes.runtime._compose.lane_health`).
    health: Callable[[], dict[str, str] | None] | None = None
    #: ``False`` = dry run (d8): log ``would-clear``, never call the clear and
    #: never persist state. ``lobes watcher run`` sets it from ``--apply``.
    apply: bool = True
    #: The deployment's cached compose project candidates (set by
    #: :func:`watcher_for`; ``None`` for a bare test watcher).
    projects: "ProjectCandidates | None" = None
    _state: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _logged: dict[str, str] = field(default_factory=dict, init=False, repr=False)

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

    def _stale_reason(
        self, m: dict[str, Any], now: float, health: dict[str, str] | None
    ) -> str | None:
        """Why a marker is stale (not acted on), or ``None`` when it is live.

        A wrapper killed by SIGKILL (or a lane removed while gated) never
        removes its marker; acting on it would clear the page cache every
        ``min_interval_s`` forever (review-b #3b).
        """
        since = m.get("since")
        if not isinstance(since, int):
            since = m.get("mtime")  # malformed marker: age it by its mtime
        if isinstance(since, (int, float)) and now - since > self.stale_after_s:
            return f"older than {self.stale_after_s}s (2x the gate timeout)"
        if health is not None and m["lane"] not in health:
            return "its lane has no running container"
        return None

    def _read_health(self) -> dict[str, str] | None:
        if self.health is None:
            return None
        try:
            return self.health()
        except Exception:  # noqa: BLE001 - an unreadable health is "unknown"
            return None

    def _attempt(self, now: float) -> tuple[str, str | None]:
        last = self._state.get("last_attempt")
        if isinstance(last, (int, float)) and now - last < self.min_interval_s:
            return "rate-limited", None
        if not self.apply:
            return "would-clear", None  # dry run: no clear, no state written
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
        return action, result.source

    def _log_action(
        self,
        m: dict[str, Any],
        act: str,
        reason: str | None,
        via: str | None,
        booting: list[str],
    ) -> None:
        """Log one marker's action line, once per lane until the action changes."""
        if act in _QUIET_ACTIONS and self._logged.get(m["lane"]) == act:
            return
        self._logged[m["lane"]] = act
        line = f"lobes-watcher: {self._describe(m)} action={act}"
        if via and not reason:
            line += f" via={via}"
        if reason:
            line += f" (stale marker: {reason}; not acted on)"
        elif act == ACTION_DEFERRED:
            line += f" (lane(s) still starting: {', '.join(booting)})"
        self.log(line)

    def tick(self) -> list[dict[str, Any]]:
        """Poll once; return one action record per marker (empty when none)."""
        markers = scan_markers(self.state_dir)
        present = {m["lane"] for m in markers}
        self._logged = {k: v for k, v in self._logged.items() if k in present}
        if not markers:
            return []
        now = self.clock()
        health = self._read_health()
        stale = {m["lane"]: self._stale_reason(m, now, health) for m in markers}
        live = [m for m in markers if stale[m["lane"]] is None]
        # gated (markers above) and held (boot/counter hold) lanes are not booting
        held = _compose.marker_lanes(self.state_dir, ".held")
        booting = _compose.lanes_starting(health, present | held)
        via: str | None = None
        if not live:
            action = ACTION_STALE
        elif booting:
            # a drop mid-boot can trip a sibling vLLM's memory profiling (#3a)
            action = ACTION_DEFERRED
        else:
            action, via = self._attempt(now)
        actions = []
        for m in markers:
            reason = stale[m["lane"]]
            act = ACTION_STALE if reason else action
            actions.append(
                {
                    "lane": m["lane"],
                    "need_kib": m.get("need_kib"),
                    "avail_kib": m.get("avail_kib"),
                    "stale": reason is not None,
                    "action": act,
                }
            )
            self._log_action(m, act, reason, via, booting)
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
    """The deployment's ``lane-state/`` dir (:func:`lobes.runtime._compose.lane_state_dir`)."""
    return _compose.lane_state_dir(deploy_dir)


#: Seconds the watcher reuses a deployment's compose project candidates before
#: re-reading them (``.env``, the compose files, the dir's symlink target).
#: ``SIGHUP`` refreshes them at once (review3 #7).
PROJECT_REFRESH_S = 300.0


class ProjectCandidates:
    """:func:`lobes.runtime._compose.compose_project_candidates`, cached per deployment.

    The watcher polls every few seconds; the candidates only change when the
    operator edits ``.env`` / the compose files or re-points a symlink, so they
    are re-read every :data:`PROJECT_REFRESH_S` or after :meth:`invalidate`
    (wired to ``SIGHUP`` by :func:`install_sighup_refresh`).
    """

    def __init__(
        self,
        deploy_dir: Path,
        clock: Callable[[], float] = time.monotonic,
        ttl_s: float = PROJECT_REFRESH_S,
    ) -> None:
        self.deploy_dir = deploy_dir
        self.clock = clock
        self.ttl_s = ttl_s
        self._value: frozenset[str] | None = None
        self._at = 0.0

    def invalidate(self) -> None:
        self._value = None

    def get(self) -> frozenset[str]:
        now = self.clock()
        if self._value is None or now - self._at > self.ttl_s:
            self._value = _compose.compose_project_candidates(self.deploy_dir)
            self._at = now
        return self._value


def watcher_for(
    deploy_dir: Path, *, cache_clock: Callable[[], float] = time.monotonic, **kw: Any
) -> Watcher:
    """A :class:`Watcher` wired to a deployment dir's log dir and ``.env`` knobs."""
    deploy_dir = Path(deploy_dir)
    env = _env.read_env_file(deploy_dir / _compose.ENV_FILE)
    gate_timeout = _int_knob(env, GATE_TIMEOUT_ENV, DEFAULT_GATE_TIMEOUT_S)
    projects = ProjectCandidates(deploy_dir, clock=cache_clock)
    kw.setdefault("health", lambda: _compose.lane_health(deploy_dir, candidates=projects.get()))
    watcher = Watcher(
        state_dir=lane_state_dir(deploy_dir),
        state_file=deploy_dir / STATE_FILE,
        min_interval_s=_int_knob(env, MIN_INTERVAL_ENV, DEFAULT_MIN_INTERVAL_S),
        stale_after_s=2 * gate_timeout,
        **kw,
    )
    watcher.projects = projects
    return watcher


def install_sighup_refresh(watcher: Watcher) -> None:
    """``SIGHUP`` drops the watcher's cached project candidates (re-read next poll)."""
    projects = watcher.projects
    if projects is None:
        return

    def _on_hup(_signum, _frame) -> None:
        projects.invalidate()

    signal.signal(signal.SIGHUP, _on_hup)


def run_forever(watcher: Watcher, poll_s: float = DEFAULT_POLL_S) -> None:  # pragma: no cover
    install_sighup_refresh(watcher)
    mode = "" if watcher.apply else "; DRY RUN, pass --apply to clear"
    watcher.log(
        f"lobes-watcher: watching {watcher.state_dir} "
        f"(min interval {watcher.min_interval_s}s, poll {poll_s}s{mode})"
    )
    while True:
        try:
            watcher.tick()
        except Exception as exc:  # noqa: BLE001 - one bad poll must not kill the loop
            watcher.log(f"lobes-watcher: poll failed: {exc.__class__.__name__}: {exc}")
        time.sleep(poll_s)
