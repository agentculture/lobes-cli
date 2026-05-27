"""``model init [TARGET]`` — scaffold a deployment directory.

Copies the packaged ``docker-compose.yml`` + ``env.example``→``.env`` into
``TARGET`` (default ``~/.model-gear``; ``model init .`` for the local folder).
Mutating: dry-run by default; ``--apply`` writes, ``--force`` overwrites.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_gear.cli._output import emit_result
from model_gear.runtime import _compose


def _emit_dry_run(target: Path, json_mode: bool) -> None:
    plan = _compose.scaffold_plan(target)
    if json_mode:
        emit_result(
            {
                "dry_run": True,
                "target": str(target),
                "files": [{"name": name, "exists": exists} for name, exists in plan],
            },
            json_mode=True,
        )
        return
    lines = [f"DRY RUN — would scaffold into {target}:"]
    for name, exists in plan:
        note = " (exists; needs --force to overwrite)" if exists else ""
        lines.append(f"  {name}{note}")
    lines.append("Re-run with --apply to write.")
    emit_result("\n".join(lines), json_mode=False)


def _emit_apply(target: Path, force: bool, json_mode: bool) -> None:
    written = _compose.write_scaffold(target, force=force)
    if json_mode:
        emit_result({"scaffolded": str(target), "files": [p.name for p in written]}, json_mode=True)
        return
    emit_result(
        f">> scaffolded {target}:\n"
        + "\n".join(f"  {p.name}" for p in written)
        + "\n>> next: docker login nvcr.io && model serve --apply",
        json_mode=False,
    )


def cmd_init(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    target = Path(args.target).expanduser() if args.target else _compose.default_deployment_dir()
    if args.apply:
        _emit_apply(target, args.force, json_mode)
    else:
        _emit_dry_run(target, json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "init",
        help="Scaffold a deployment dir (default ~/.model-gear; dry-run by default; --apply).",
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Where to scaffold (default ~/.model-gear; '.' for the current folder).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p.add_argument("--apply", action="store_true", help="Actually write the files.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_init)
