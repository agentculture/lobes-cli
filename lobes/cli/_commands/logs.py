"""``model logs`` — read the durable, restart-surviving vLLM logs.

Read-only. ``mg-logwrap`` (the compose entrypoint) tees each vLLM service's
stdout+stderr to a per-boot file under the host log dir, so a crash trace
survives container restart/recreate — the investigation gap behind issue #50.
This verb lists those files and tails one, reading the **host** files directly so
it works even after the crashed container is gone (``docker logs`` would not).

    model logs                 # list per-boot log files (newest first) + the dir
    model logs vllm            # tail the latest log for a service (vllm/primary/embed/rerank)
    model logs primary -n 200  # tail more lines
    model logs --list --json   # structured listing
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lobes.cli import _runtime_ops
from lobes.cli._output import emit_result
from lobes.runtime import _compose, _env

# Per-boot files are "<service>-<ISO8601>.log"; the "<service>-latest.log" symlink
# is a convenience pointer we skip when listing real boots.
_LATEST_SUFFIX = "-latest.log"


def collect_logs(log_dir: Path, service: str | None = None) -> list[dict]:
    """Per-boot log files under ``log_dir``, newest first (pure; no docker).

    Each entry: ``{name, service, path, size, mtime}``. The ``<service>-latest.log``
    symlinks are skipped (they point at a file already listed). ``service`` filters
    by filename prefix (``vllm`` / ``primary`` / ``embed`` / ``rerank``).
    """
    if not log_dir.is_dir():
        return []
    out: list[dict] = []
    for p in log_dir.glob("*.log"):
        # Skip symlinks (the <service>-latest.log pointer AND any planted symlink):
        # never follow a symlink out of the log dir to a file like /etc/shadow —
        # mg-logwrap only ever writes regular per-boot files (security, Qodo review).
        if p.is_symlink() or p.name.endswith(_LATEST_SUFFIX) or not p.is_file():
            continue
        svc = p.name.split("-", 1)[0]
        if service and svc != service:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(
            {
                "name": p.name,
                "service": svc,
                "path": str(p),
                "size": st.st_size,
                "mtime": st.st_mtime,
            }
        )
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def tail_lines(path: Path, n: int, max_bytes: int = 262144) -> str:
    """Last ``n`` lines of ``path`` without reading the whole (possibly huge) file.

    vLLM logs every few seconds, so a boot file can be large; read only the final
    ``max_bytes`` and return the last ``n`` lines of that window.
    """
    # Defense in depth: refuse to read through a symlink even if one reaches here
    # (collect_logs already filters them out of the selection).
    if path.is_symlink():
        return f"(refusing to read a symlink: {path})"
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _resolve_log_dir(args: argparse.Namespace) -> Path:
    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    configured = _env.read_env(env_path, _compose.LOG_DIR_ENV) or None
    return _compose.durable_log_dir(deploy_dir, configured)


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{int(size)}{unit}"
        size /= 1024
    return f"{int(size)}G"


def _emit_tail(args, log_dir: Path, service: str, entries: list[dict], json_mode: bool) -> None:
    """Tail mode: show the latest boot for ``service`` (or the crashed one with --previous)."""
    if not entries:
        msg = f"no logs for '{service}' in {log_dir}"
        emit_result(
            {"log_dir": str(log_dir), "service": service, "files": []} if json_mode else msg,
            json_mode=json_mode,
        )
        return
    # --previous tails the boot *before* the latest — i.e. the boot that crashed,
    # the one to investigate after a restart created a fresh (healthy) boot file.
    want_prev = bool(getattr(args, "previous", False))
    idx = 1 if want_prev and len(entries) > 1 else 0
    chosen = entries[idx]
    # Flag the case where --previous was asked but there is no earlier boot, so the
    # reader isn't misled into thinking the only boot is the crashed one.
    only_one = want_prev and len(entries) == 1
    n = int(getattr(args, "lines", 40))
    body = tail_lines(Path(chosen["path"]), n)
    if json_mode:
        emit_result(
            {
                "log_dir": str(log_dir),
                "service": service,
                "file": chosen["path"],
                "lines": n,
                "only_boot": only_one,
                "tail": body,
            },
            json_mode=True,
        )
    else:
        note = " (only 1 boot — showing latest)" if only_one else ""
        emit_result(f">> {chosen['path']} (last {n} lines){note}\n{body}", json_mode=False)


def _emit_listing(log_dir: Path, entries: list[dict], json_mode: bool) -> None:
    """Listing mode (default / --list): per-boot files, newest first, latest flagged."""
    if json_mode:
        emit_result({"log_dir": str(log_dir), "files": entries}, json_mode=True)
        return
    if not entries:
        emit_result(
            f"no durable logs yet in {log_dir}\n"
            ">> they appear once a vLLM service starts (model serve / fleet up).",
            json_mode=False,
        )
        return
    lines = [f"log dir: {log_dir}", "boots (newest first):"]
    seen_services: set[str] = set()
    for e in entries:
        latest = "" if e["service"] in seen_services else "  <- latest"
        seen_services.add(e["service"])
        lines.append(f"  {e['name']:<34} {_human_size(e['size']):>6}{latest}")
    lines.append(">> tail one with: model logs <service>  (e.g. model logs vllm)")
    emit_result("\n".join(lines), json_mode=False)


def cmd_logs(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    log_dir = _resolve_log_dir(args)
    service = getattr(args, "service", None)
    entries = collect_logs(log_dir, service)
    if service and not getattr(args, "list", False):
        _emit_tail(args, log_dir, service, entries, json_mode)
    else:
        _emit_listing(log_dir, entries, json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "logs",
        help="Read-only: list/tail the durable vLLM logs that survive restart "
        "(model logs [service]; issue #50).",
    )
    p.add_argument(
        "service",
        nargs="?",
        help="Service to tail (vllm / primary / embed / rerank). Omit to list all boots.",
    )
    p.add_argument("-n", "--lines", type=int, default=40, help="Lines to tail (default 40).")
    p.add_argument(
        "-p",
        "--previous",
        action="store_true",
        help="Tail the boot before the latest — the crashed boot to investigate after a restart.",
    )
    p.add_argument(
        "--list", action="store_true", help="List boot files even when a service is given."
    )
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_logs)
