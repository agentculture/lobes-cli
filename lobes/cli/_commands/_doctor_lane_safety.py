"""``lobes doctor`` lane-safety findings and the targeted wrapper refresh (t16).

Contract: ``docs/contracts/lane-safety-runtime.md``. Seven read-only findings,
none of them ``error`` (lane safety never makes a deployment "unhealthy"), and
each returns ``None`` — no row at all — when it does not apply:

* ``logwrap_stale`` (warn) — the deployed ``mg-logwrap.sh`` differs from the
  packaged template, so the boot hold / start counter / memory gate may be
  missing. Absent wrapper: skipped (``scaffold_files`` already reports it).
* ``lane_state_writable`` (warn) — ``logs/lane-state/`` exists but the operator
  cannot write it (a root container created it first), so lifecycle verbs
  cannot reset start counters. Absent: skipped.
* ``compile_cache_unmounted`` (warn) — a vLLM service lacks the ``/cache/vllm``
  bind or ``VLLM_CACHE_ROOT``, or (d7) the ``/cache/flashinfer`` bind or
  ``FLASHINFER_WORKSPACE_BASE``. A hand-kept compose fires this on purpose; the
  remediation is override-file lines, never ``init --force``.
* ``compile_cache_size`` (info) — bytes under the compile-cache dir.
* ``jetson_clocks_locked`` (warn) — the machine CLI's ``window show`` reports
  locked clocks while no benchmark window holds the host lock. ``clocks_locked``
  (the devfreq min == max read) also catches a lock made outside any lobes
  window; without it (older CLI, unreadable) only ``active`` is consulted.
* ``drop_caches_grant_missing`` (warn) — Spark/Thor/Orin only: no passwordless
  page-cache rule, so the lifecycle clear and the watcher do nothing.
* ``watcher_inactive`` (warn) — the watcher manifest exists but its unit is not
  active. Not installed is an ``info`` pass ("optional"): the watcher is opt-in.

The refresh (``lobes doctor --refresh-logwrap``) is its own named flag, after
the ``--repin-version`` precedent: plain ``--fix`` stays missing-only (#174),
and the refresh writes exactly two paths — ``mg-logwrap.sh.bak`` (the current
file) then the packaged ``mg-logwrap.sh``, renamed into place from a temp file
so a running lane's bind-mounted inode is never rewritten.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
import tempfile
from importlib.resources import files as _resource_files
from pathlib import Path

from lobes.runtime import _compose, _detect, _machine, _window
from lobes.watcher import _install

_TEMPLATES_PACKAGE = "lobes.templates"
_BACKUP_SUFFIX = ".bak"
_REFRESH_FLAG = "lobes doctor --refresh-logwrap --apply"
_GRANT_CARDS = ("spark", "thor", "orin")
_CACHE_TARGET = "/cache/vllm"
_ENV_LINE = "VLLM_CACHE_ROOT=${LOBES_VLLM_CACHE_ROOT:-/cache/vllm}"
# Lane-safety d7: FlashInfer's JIT cache (vllm#62149), a second fixed target.
_FI_TARGET = "/cache/flashinfer"
_FI_ENV_KEY = "FLASHINFER_WORKSPACE_BASE"
_FI_ENV_LINE = f"{_FI_ENV_KEY}=${{LOBES_FLASHINFER_WORKSPACE_BASE:-{_FI_TARGET}}}"
_FI_SUFFIX = "-flashinfer"
_STATE_KEYS = ("mount", "env", "fi_mount", "fi_env")
_SIZE_WALK_LIMIT = 200_000  # entries; a bounded walk, never a runaway scan


def _check(id_: str, passed: bool, severity: str, message: str, remediation: str = "") -> dict:
    return {
        "id": id_,
        "passed": passed,
        "severity": severity,
        "message": message,
        "remediation": remediation,
    }


def _packaged_logwrap() -> str:
    return _compose._read_template(_resource_files(_TEMPLATES_PACKAGE), _compose.LOG_WRAPPER)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- logwrap_stale ----------------------------------------------------------


def logwrap_stale_check(deploy_dir: Path) -> dict | None:
    path = Path(deploy_dir) / _compose.LOG_WRAPPER
    try:
        deployed = path.read_bytes()
    except OSError:
        return None  # absent: scaffold_files owns that finding
    packaged = _packaged_logwrap().encode("utf-8")
    if _sha256(deployed) == _sha256(packaged):
        return _check("logwrap_stale", True, "info", f"{_compose.LOG_WRAPPER} matches the package")
    return _check(
        "logwrap_stale",
        False,
        "warn",
        f"{_compose.LOG_WRAPPER} differs from this CLI's packaged copy "
        f"(sha256 {_sha256(deployed)[:12]} vs {_sha256(packaged)[:12]}) — the lane "
        "boot hold, start counter and memory gate may be missing",
        f"run '{_REFRESH_FLAG}' (backs the current file up to "
        f"{_compose.LOG_WRAPPER}{_BACKUP_SUFFIX}, writes no other file), then recreate the lanes",
    )


# --- compile-cache findings -------------------------------------------------


def _compose_chain(deploy_dir: Path) -> list[str]:
    """The compose files ``docker compose`` reads for this deployment, in order."""
    names = [a for a in _compose._compose_files(deploy_dir) if a != "-f"]
    if not names:
        names = [_compose.COMPOSE_FILE]
        if (deploy_dir / _compose.LOCAL_OVERRIDE).is_file():
            names.append(_compose.LOCAL_OVERRIDE)
    return names


def _strip_comment(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return ""
    idx = stripped.find(" #")
    return stripped[:idx].rstrip() if idx >= 0 else stripped


def _service_blocks(text: str) -> dict[str, list[str]]:
    """``services.<name>`` -> its comment-stripped lines (stdlib indent scan)."""
    blocks: dict[str, list[str]] = {}
    top: str | None = None
    current: str | None = None
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent == 0:
            top = line.rstrip(":")
            current = None
            continue
        if top != "services":
            continue
        if indent == 2 and line.endswith(":"):
            current = line[:-1]
            blocks.setdefault(current, [])
            continue
        if current is not None and indent > 2:
            blocks[current].append(line)
    return blocks


def _tokens(lines: list[str]) -> str:
    out = []
    for line in lines:
        tok = line[2:] if line.startswith("- ") else line
        out.append(tok.strip().strip("'\""))
    return " ".join(out)


def _runs_vllm(lines: list[str]) -> bool:
    joined = f" {_tokens(lines)} "
    return " vllm serve " in joined or "Dockerfile.decider-vllm" in joined


def _binds(bare: str, target: str) -> bool:
    """True when a volume token binds onto ``target`` (``src:target`` or ``src:target:ro``)."""
    return f":{target}" in bare and (bare.endswith(target) or f"{target}:" in bare)


def _sets_key(bare: str, key: str) -> bool:
    """True when a env token assigns ``key`` (``KEY=value`` or ``KEY: value``)."""
    return bare.startswith((f"{key}=", f"{key}:"))


def _scan_service_lines(lines: list[str], entry: dict[str, bool]) -> None:
    """Fold one service block's volume/env tokens into its ``entry`` flags."""
    for line in lines:
        bare = line.strip("-'\" ")
        if _binds(bare, _CACHE_TARGET):
            entry["mount"] = True
        if _binds(bare, _FI_TARGET):
            entry["fi_mount"] = True
        if _sets_key(bare, "VLLM_CACHE_ROOT"):
            entry["env"] = True
        if _sets_key(bare, _FI_ENV_KEY):
            entry["fi_env"] = True


def _merge_compose_file(text: str, state: dict[str, dict[str, bool]], vllm: set[str]) -> None:
    """Merge one compose file's services into ``state``, noting vLLM runners."""
    for service, lines in _service_blocks(text).items():
        entry = state.setdefault(service, dict.fromkeys(_STATE_KEYS, False))
        if _runs_vllm(lines):
            vllm.add(service)
        _scan_service_lines(lines, entry)


def _parked_services(deploy_dir: Path) -> set[str]:
    """Services the shape overlay parks (not hosted here); empty without one."""
    shape = deploy_dir / _compose.SHAPE_OVERLAY
    if not shape.is_file():
        return set()
    return _compose.shape_parked_service_keys(shape.read_text(encoding="utf-8"))


def _vllm_mount_state(deploy_dir: Path) -> dict[str, dict[str, bool]]:
    """Per vLLM service: ``mount``/``env`` (t8) + ``fi_mount``/``fi_env`` (d7), merged."""
    state: dict[str, dict[str, bool]] = {}
    vllm: set[str] = set()
    for name in _compose_chain(deploy_dir):
        try:
            text = (deploy_dir / name).read_text(encoding="utf-8")
        except OSError:
            continue
        _merge_compose_file(text, state, vllm)
    parked = _parked_services(deploy_dir)
    return {s: state[s] for s in sorted(vllm) if s not in parked}


def _override_snippet(deploy_dir: Path, services: list[str]) -> str:
    log_names = _compose._service_log_names(deploy_dir, _compose_chain(deploy_dir))
    lines = ["services:"]
    for svc in services:
        lane = log_names.get(svc, svc)
        root = f"${{{_compose.COMPILE_CACHE_ENV}:-./{_compose.COMPILE_CACHE_DIRNAME}}}"
        lines += [
            f"  {svc}:",
            "    environment:",
            f"      - {_ENV_LINE}",
            f"      - {_FI_ENV_LINE}",
            "    volumes:",
            f"      - {root}/{lane}:{_CACHE_TARGET}",
            f"      - {root}/{lane}{_FI_SUFFIX}:{_FI_TARGET}",
        ]
    return "\n".join(lines)


def _first_gap(entry: dict[str, bool]) -> str:
    """The first missing piece, t8's original gaps before d7's."""
    gaps = {
        "mount": "no mount",
        "env": "no VLLM_CACHE_ROOT",
        "fi_mount": "no FlashInfer mount",
        "fi_env": f"no {_FI_ENV_KEY}",
    }
    return next(gaps[k] for k in _STATE_KEYS if not entry[k])


def compile_cache_unmounted_check(deploy_dir: Path) -> dict | None:
    deploy_dir = Path(deploy_dir)
    state = _vllm_mount_state(deploy_dir)
    if not state:
        return None
    lacking = [s for s, e in state.items() if not all(e[k] for k in _STATE_KEYS)]
    if not lacking:
        return _check(
            "compile_cache_unmounted",
            True,
            "info",
            f"all {len(state)} vLLM service(s) mount the compile cache at {_CACHE_TARGET} "
            f"and the FlashInfer JIT cache at {_FI_TARGET}",
        )
    detail = ", ".join(f"{s} ({_first_gap(state[s])})" for s in lacking)
    return _check(
        "compile_cache_unmounted",
        False,
        "warn",
        f"vLLM service(s) without a persistent compile cache: {detail} — every "
        "restart recompiles from scratch",
        f"add these lines to {_compose.LOCAL_OVERRIDE} (keeps a hand-kept compose "
        "intact), then run 'lobes doctor --fix --apply' to create the cache dirs "
        "and recreate the lanes:\n" + _override_snippet(deploy_dir, lacking),
    )


def _tree_bytes(root: Path) -> tuple[int, bool]:
    """Total ``lstat`` size of regular files under ``root``; ``(bytes, truncated)``."""
    total = 0
    seen = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            seen += 1
            if seen > _SIZE_WALK_LIMIT:
                return total, True
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if not os.path.islink(os.path.join(dirpath, name)):
                total += st.st_size
    return total, False


def compile_cache_size_check(deploy_dir: Path) -> dict | None:
    root = _compose.compile_cache_dir(deploy_dir)
    if not root.is_dir():
        return None
    total, truncated = _tree_bytes(root)
    prefix = "at least " if truncated else ""
    return _check(
        "compile_cache_size",
        True,
        "info",
        f"compile cache {root}: {prefix}{total} bytes ({total / 2**30:.2f} GiB); "
        "wipe a lane by deleting its directory with no lane running",
    )


# --- lane_state_writable (review-b #1) ----------------------------------------


def _lane_state_dir(deploy_dir: Path) -> Path:
    return _compose.lane_state_dir(deploy_dir)


def lane_state_writable_check(deploy_dir: Path) -> dict | None:
    """``lane-state/`` must be writable by the operator, or counter resets fail.

    A root container that wrote the first start counter before any lobes verb
    created the dir leaves it root 0755: ``lobes up <role> --apply`` then cannot
    delete ``<lane>.starts`` and the lane stays held. Absent: skipped (the next
    lifecycle verb creates it as the operator).
    """
    state = _lane_state_dir(Path(deploy_dir))
    if not state.is_dir():
        return None
    if os.access(state, os.W_OK | os.X_OK):
        return _check("lane_state_writable", True, "info", f"{state} is writable by this user")
    return _check(
        "lane_state_writable",
        False,
        "warn",
        f"{state} is not writable by this user (created by a root container): "
        "lifecycle verbs cannot reset a lane's start counter, so a held lane stays held",
        f"run 'sudo chown -R \"$(id -un)\" {state}' (the root containers can still write it)",
    )


# --- machine findings -------------------------------------------------------


def _detected_card() -> str | None:
    # per-process cached (review-b #10); never raises
    return _machine.detected_card(_detect.detect_card)


def _benchmark_window_active() -> bool:
    """True iff another process holds the benchmark window's host lock."""
    try:
        fd = os.open(_window.lock_path(), os.O_RDONLY)
    except OSError:
        return False  # no lock file: no window has ever run here
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)  # closing releases a lock we may have just taken
    return False


def _clocks_locked(payload: dict) -> bool | None:
    """The CLI's own read of the clock lock, or ``None`` to use ``active``.

    ``clocks_locked`` (contract addition 2026-10-17) is the devfreq
    ``min_freq == max_freq`` read, so it sees a lock made OUTSIDE any lobes
    window. ``null`` (unreadable, e.g. GB10) or absent (an older CLI) falls
    back to ``active``, which only tracks lobes' own window state.
    """
    value = payload.get("clocks_locked")
    return value if isinstance(value, bool) else None


def jetson_clocks_locked_check() -> dict | None:
    try:
        res = _machine.window("show")
    except Exception:  # the adapter never raises; belt and braces
        return None
    if not res.ok:
        return None  # tool_missing / unknown card / not_granted: not applicable
    payload = res.payload
    locked = _clocks_locked(payload)
    window_state = bool(payload.get("active") or payload.get("stored"))
    if locked is None:
        locked = bool(payload.get("active"))
        window_state = True  # the fallback only ever sees a lobes window
    if not locked:
        return _check("jetson_clocks_locked", True, "info", "clocks are not locked")
    if _benchmark_window_active():
        return _check(
            "jetson_clocks_locked",
            True,
            "info",
            "clocks locked by a benchmark window in progress",
        )
    cli = _machine._current_cli() or "<machine-cli>"
    if window_state:
        return _check(
            "jetson_clocks_locked",
            False,
            "warn",
            "clocks are locked by a lobes window but no benchmark holds it (a "
            "crashed or interrupted benchmark left them at max) — a warning, "
            "not an error",
            f"run '{cli} window restore' to return to the stored state",
        )
    return _check(
        "jetson_clocks_locked",
        False,
        "warn",
        "clocks are locked outside any lobes window (operator jetson_clocks or a "
        "crashed run) — a warning, not an error: the operator may have locked "
        "them on purpose",
        "if unintended, run 'sudo jetson_clocks --restore' (needs a state saved "
        "by 'jetson_clocks --store') or reboot the box to return to default clocks",
    )


_MANUAL_RULE = (
    "or add the passwordless sudoers rule by hand: "
    "'<user> ALL=(root) NOPASSWD: /usr/sbin/sysctl -w vm.drop_caches=3'"
)


def drop_caches_grant_missing_check() -> dict | None:
    card = _detected_card()
    if card not in _GRANT_CARDS:
        return None
    grant = _machine.grant_status()
    if grant.granted:
        return _check(
            "drop_caches_grant_missing",
            True,
            "info",
            f"page-cache clear is granted (rule: {grant.rule})",
        )
    cli = _machine.machine_cli_for(card)
    if cli:
        remediation = f"run '{cli} grant-permissions' ({_MANUAL_RULE})"
    else:
        remediation = (
            f"install the machine CLI ('uv tool install lobes-cli[{card}]' provides "
            f"'{card}') and run '{card} grant-permissions', {_MANUAL_RULE}"
        )
    return _check(
        "drop_caches_grant_missing",
        False,
        "warn",
        f"no passwordless page-cache clear on this {card}: lifecycle verbs and the "
        "watcher skip the clear, so a crashed lane's memory can stay unreclaimed",
        remediation,
    )


def _watcher_active_state(system: bool) -> str:
    """``systemctl [--user] is-active lobes-watcher.service`` (test seam)."""
    ctx = _install.default_context(system)
    _rc, stdout, _err = ctx.runner(ctx.systemctl(system, "is-active", _install.UNIT_NAME))
    return stdout.strip() or "unknown"


def _unit_runs_dry(manifest: dict) -> bool:
    """True when an installed unit's ExecStart runs ``watcher run`` without ``--apply``."""
    for f in manifest.get("files") or []:
        try:
            text = Path(str(f)).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("ExecStart=") and "watcher run" in line:
                return "--apply" not in line.split()
    return False


def watcher_inactive_check(deploy_dir: Path) -> dict | None:
    manifest = _install.read_manifest(Path(deploy_dir))
    if manifest is None:
        return _check(
            "watcher_inactive",
            True,
            "info",
            "host watcher not installed (optional; 'lobes watcher install' clears "
            "the page cache for memory-gated lanes)",
        )
    system = manifest.get("unit_type") == "system"
    state = _watcher_active_state(system)
    if state == "active" and _unit_runs_dry(manifest):
        return _check(
            "watcher_inactive",
            False,
            "warn",
            "host watcher is active but its unit runs 'lobes watcher run' without "
            "--apply (installed before d8), so it only logs what it would clear",
            "reinstall it: 'lobes watcher uninstall --apply' then 'lobes watcher install --apply'",
        )
    if state == "active":
        return _check("watcher_inactive", True, "info", "host watcher is active")
    systemctl = "sudo systemctl" if system else "systemctl --user"
    return _check(
        "watcher_inactive",
        False,
        "warn",
        f"host watcher is installed but {state}: memory-gated lanes get no page-cache clear",
        f"run '{systemctl} enable --now {_install.UNIT_NAME}'; inspect with "
        "'lobes watcher status'",
    )


# --- registration -----------------------------------------------------------


def lane_safety_checks(deploy_dir: Path) -> list[dict]:
    """Every applicable lane-safety finding for ``deploy_dir`` (the one doctor hook)."""
    deploy_dir = Path(deploy_dir)
    candidates = [
        logwrap_stale_check(deploy_dir),
        lane_state_writable_check(deploy_dir),
        compile_cache_unmounted_check(deploy_dir),
        compile_cache_size_check(deploy_dir),
        jetson_clocks_locked_check(),
        drop_caches_grant_missing_check(),
        watcher_inactive_check(deploy_dir),
    ]
    return [c for c in candidates if c is not None]


def missing_cache_dirs(deploy_dir: Path) -> list[str]:
    """The ``--fix`` plan entry: absent compile-cache dirs (missing-only)."""
    return [str(p) for p in _compose.ensure_compile_cache_dirs(deploy_dir, dry_run=True)]


def create_cache_dirs(deploy_dir: Path) -> list[str]:
    """The ``--fix --apply`` action: create absent compile-cache dirs."""
    return [f"created {p}" for p in _compose.ensure_compile_cache_dirs(deploy_dir)]


# --- the targeted refresh ---------------------------------------------------


def _refresh_state(deploy_dir: Path) -> str:
    """``absent`` | ``current`` | ``stale``."""
    path = Path(deploy_dir) / _compose.LOG_WRAPPER
    try:
        deployed = path.read_bytes()
    except OSError:
        return "absent"
    return "current" if deployed == _packaged_logwrap().encode("utf-8") else "stale"


def refresh_logwrap_plan(deploy_dir: Path) -> list[str]:
    state = _refresh_state(deploy_dir)
    wrapper = _compose.LOG_WRAPPER
    if state == "absent":
        return [f"{wrapper} is absent — run 'lobes doctor --fix --apply' to write it"]
    if state == "current":
        return []
    return [
        f"would copy {wrapper} -> {wrapper}{_BACKUP_SUFFIX}",
        f"would write the packaged {wrapper}",
    ]


def apply_refresh_logwrap(deploy_dir: Path) -> list[str]:
    """Back up then replace a STALE wrapper. Writes exactly two paths, or none.

    The new wrapper goes to a temp file in the same dir and is ``os.replace``d
    over the old name (review-b #2): running containers bind-mount the old
    INODE and a held lane's bash is still reading it, so truncating and
    rewriting in place would make bash resume at a stale offset in the new
    bytes. The rename leaves the old inode intact for them; the ``.bak`` is an
    independent copy. The old file's mode is kept, and so is its owner when
    this runs as root (review2-b #7: under ``sudo`` the temp file and the copy
    would otherwise be root-owned) — best-effort, a failure is reported as an
    extra action line and never stops the refresh.
    """
    if _refresh_state(deploy_dir) != "stale":
        return []
    wrapper = Path(deploy_dir) / _compose.LOG_WRAPPER
    backup = wrapper.with_name(wrapper.name + _BACKUP_SUFFIX)
    old = wrapper.stat()
    shutil.copy2(wrapper, backup)
    mode = stat.S_IMODE(old.st_mode)
    as_root = os.geteuid() == 0
    failed: list[str] = []
    if as_root:
        failed += _keep_owner(backup, old)
    fd, tmp = tempfile.mkstemp(prefix=f".{wrapper.name}.", dir=str(wrapper.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_packaged_logwrap())
        os.chmod(tmp, mode)
        if as_root:
            failed += _keep_owner(Path(tmp), old, shown=wrapper.name)
        os.replace(tmp, wrapper)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return [
        f"backed up {wrapper.name} -> {backup.name}",
        f"wrote packaged {wrapper.name}",
        *failed,
    ]


def _keep_owner(path: Path, old: os.stat_result, shown: str | None = None) -> list[str]:
    """chown *path* to *old*'s uid/gid; ``[]`` on success, else one report line."""
    try:
        os.chown(path, old.st_uid, old.st_gid)
    except OSError as exc:
        reason = exc.strerror or exc.__class__.__name__
        name = shown or path.name
        owner = f"{old.st_uid}:{old.st_gid}"
        return [
            f"could not keep the owner {owner} on {name} ({reason})"
            f" - run 'sudo chown {owner} {path.parent / name}'"
        ]
    return []
