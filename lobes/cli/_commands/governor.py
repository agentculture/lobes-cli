"""``lobes governor`` — the graded thermal governor for one container.

    lobes governor run --container NAME [--once] [--apply] [--bands ...]

Hottest thermal zone -> ``docker update --cpus`` (16/12/8/4), then
``docker pause`` as the last resort, resuming below 86 C at 4 cpus. ``run`` is
dry-run unless ``--apply`` (mutation-safety rule): it reads the container state
and temperature and reports what it WOULD do, changing nothing. With
``--apply`` it logs one line per band change to ``<deployment>/logs/governor.log``
(``--log`` / ``$LOBES_GOVERNOR_LOG`` override) and, on SIGINT/SIGTERM, restores
the top-band cap and unpauses only a pause it made itself. Mechanism and
history: ``lobes explain lane-safety``; code: ``lobes/runtime/_governor.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.runtime import _governor as gv


def _stderr_log(line: str) -> None:
    sys.stderr.write(line.rstrip("\n") + "\n")
    sys.stderr.flush()


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        bands = gv.parse_bands(
            args.bands,
            pause_c=args.pause_c,
            resume_c=args.resume_c,
            resume_cpus=args.resume_cpus,
            top_cpus=args.top_cpus,
        )
    except ValueError as exc:
        raise ModelGearError(
            EXIT_USER_ERROR,
            str(exc),
            "Bands look like 84:12,88:8,91:4 (ascending, below --pause-c).",
        ) from exc
    deploy = _runtime_ops.deployment_dir(args)
    log_path = Path(args.log or os.environ.get(gv.LOG_ENV) or deploy / "logs" / gv.LOG_NAME)
    sink = gv.file_log(log_path) if args.apply else _stderr_log

    def log(line: str) -> None:
        sink(line)
        if args.apply and not args.once:
            _stderr_log(line)

    gov = gv.Governor(
        args.container,
        bands,
        docker=gv.run_docker,
        read_temp=gv.read_hottest_c,
        log=log,
        apply=bool(args.apply),
    )
    if args.once:
        alive = gov.tick()
        payload = {
            "dry_run": not args.apply,
            "container": args.container,
            "alive": alive,
            "temp_c": gov.last_temp,
            "action": gov.last_action,
            "log": str(log_path) if args.apply else None,
        }
        if args.json:
            emit_result(payload, json_mode=True)
        else:
            tail = "" if args.apply else "\n>> dry run: re-run with --apply to act"
            emit_result(
                f"{args.container}: temp={gov.last_temp} action={gov.last_action}{tail}",
                json_mode=False,
            )
        return 0
    gv.run_forever(gov, poll_s=args.poll)  # pragma: no cover - endless loop
    return 0  # pragma: no cover


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "governor",
        help="Graded thermal governor: cap a container's CPUs as the host heats, "
        "pause only as a last resort (dry-run unless --apply).",
    )
    gsub = p.add_subparsers(dest="governor_command", required=True)
    r = gsub.add_parser("run", help="Run the governor loop in the foreground.")
    r.add_argument("--container", required=True, help="Container to govern.")
    r.add_argument("--once", action="store_true", help="Poll once and exit.")
    r.add_argument(
        "--apply",
        action="store_true",
        help="Actually run docker update/pause/unpause (default: report only).",
    )
    r.add_argument("--poll", type=float, default=gv.DEFAULT_POLL_S, help="Seconds between polls.")
    r.add_argument(
        "--bands",
        default="84:12,88:8,91:4",
        help="'limit_c:cpus,...': from each limit upward use that cpus "
        "(default 84:12,88:8,91:4; below the first limit use --top-cpus).",
    )
    r.add_argument("--top-cpus", type=int, default=16, help="Cap below the first limit.")
    r.add_argument("--pause-c", type=float, default=93.5, help="Pause at or above this.")
    r.add_argument("--resume-c", type=float, default=86.0, help="Resume below this.")
    r.add_argument("--resume-cpus", type=int, default=4, help="Cap applied on resume.")
    r.add_argument("--log", help=f"Log file (default <deployment>/logs/{gv.LOG_NAME}).")
    r.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    r.add_argument("--json", action="store_true", help="Emit structured JSON (--once).")
    r.set_defaults(func=_cmd_run)
