"""``model init [TARGET]`` — scaffold a deployment directory.

Copies the packaged ``docker-compose.yml`` + ``env.example``→``.env`` into
``TARGET`` (default ``~/.model-gear``; ``model init .`` for the local folder).
``--fleet`` scaffolds the 3-container gateway deployment instead (two always-warm
vLLM backends + a single OpenAI front). Mutating: dry-run by default; ``--apply``
writes, ``--force`` overwrites.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_gear import __version__
from model_gear.cli._output import emit_result
from model_gear.runtime import _compose, _env


def _templates(fleet: bool) -> dict[str, str]:
    return _compose.FLEET_TEMPLATES if fleet else _compose.SINGLE_TEMPLATES


def _emit_dry_run(target: Path, fleet: bool, json_mode: bool) -> None:
    plan = _compose.scaffold_plan(target, _templates(fleet))
    if json_mode:
        emit_result(
            {
                "dry_run": True,
                "fleet": fleet,
                "target": str(target),
                "files": [{"name": name, "exists": exists} for name, exists in plan],
            },
            json_mode=True,
        )
        return
    lines = [f"DRY RUN — would scaffold {'fleet ' if fleet else ''}into {target}:"]
    for name, exists in plan:
        note = " (exists; needs --force to overwrite)" if exists else ""
        lines.append(f"  {name}{note}")
    lines.append("Re-run with --apply to write.")
    emit_result("\n".join(lines), json_mode=False)


def _emit_apply(target: Path, fleet: bool, force: bool, json_mode: bool) -> None:
    written = _compose.write_scaffold(target, force=force, templates=_templates(fleet))
    if fleet:
        # Pin the gateway image to the model-gear release that scaffolded this.
        _env.set_env(target / _compose.ENV_FILE, "MODEL_GEAR_VERSION", __version__)
    if json_mode:
        emit_result(
            {"scaffolded": str(target), "fleet": fleet, "files": [p.name for p in written]},
            json_mode=True,
        )
        return
    next_step = (
        "docker login nvcr.io && model fleet up --apply"
        if fleet
        else "docker login nvcr.io && model serve --apply"
    )
    emit_result(
        f">> scaffolded {target}:\n"
        + "\n".join(f"  {p.name}" for p in written)
        + f"\n>> next: {next_step}",
        json_mode=False,
    )


def cmd_init(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    fleet = bool(getattr(args, "fleet", False))
    target = Path(args.target).expanduser() if args.target else _compose.default_deployment_dir()
    if args.apply:
        _emit_apply(target, fleet, args.force, json_mode)
    else:
        _emit_dry_run(target, fleet, json_mode)
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
    p.add_argument(
        "--fleet",
        action="store_true",
        help="Scaffold the 3-container gateway deployment (2 vLLM backends + 1 front) "
        "instead of a single model.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p.add_argument("--apply", action="store_true", help="Actually write the files.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_init)
