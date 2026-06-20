"""``model init [TARGET]`` — scaffold a deployment directory.

Copies the packaged ``docker-compose.yml`` + ``env.example``→``.env`` into
``TARGET`` (default ``~/.model-gear``; ``model init .`` for the local folder).
``--fleet`` scaffolds the gateway deployment instead (the always-warm Qwen
primary + co-resident embedding/reranker gears behind one OpenAI front, routed by
task family; one generate backend by default, opt-in generate fallback).
Mutating: dry-run by default; ``--apply`` writes, ``--force`` overwrites.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_gear import __version__
from model_gear.cli._errors import EXIT_USER_ERROR, ModelGearError
from model_gear.cli._output import emit_result
from model_gear.runtime import _compose, _env


def _templates(fleet: bool, audio: bool) -> dict[str, str]:
    if not fleet:
        return _compose.SINGLE_TEMPLATES
    templates = dict(_compose.FLEET_TEMPLATES)
    if audio:
        templates.update(_compose.AUDIO_TEMPLATES)
    return templates


def _emit_dry_run(target: Path, fleet: bool, audio: bool, json_mode: bool) -> None:
    plan = _compose.scaffold_plan(target, _templates(fleet, audio))
    if json_mode:
        emit_result(
            {
                "dry_run": True,
                "fleet": fleet,
                "audio": audio,
                "target": str(target),
                "files": [{"name": name, "exists": exists} for name, exists in plan],
            },
            json_mode=True,
        )
        return
    scope = "fleet+audio " if (fleet and audio) else "fleet " if fleet else ""
    lines = [f"DRY RUN — would scaffold {scope}into {target}:"]
    for name, exists in plan:
        note = " (exists; needs --force to overwrite)" if exists else ""
        lines.append(f"  {name}{note}")
    if audio:
        lines.append("  .env (+ audio keys appended)")
    lines.append("Re-run with --apply to write.")
    emit_result("\n".join(lines), json_mode=False)


def _emit_apply(target: Path, fleet: bool, audio: bool, force: bool, json_mode: bool) -> None:
    written = _compose.write_scaffold(target, force=force, templates=_templates(fleet, audio))
    # Create the durable-log dir now (as the invoking user) so the compose bind-mount
    # source exists before `model serve` / `fleet up` — otherwise Docker makes it
    # root-owned. The mg-logwrap entrypoint writes per-boot logs here (issue #50).
    _compose.ensure_log_dir(target)
    if fleet:
        # Pin the gateway image to the model-gear release that scaffolded this.
        _env.set_env(target / _compose.ENV_FILE, "MODEL_GEAR_VERSION", __version__)
    if audio:
        # Extend the fleet .env with the audio keys (NGC_API_KEY, ports, AUDIO_URL …).
        _compose.append_audio_env(target)
    if json_mode:
        emit_result(
            {
                "scaffolded": str(target),
                "fleet": fleet,
                "audio": audio,
                "files": [p.name for p in written],
            },
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
        + (f"\n  {_compose.ENV_FILE} (+ audio keys)" if audio else "")
        + f"\n>> next: {next_step}",
        json_mode=False,
    )


def cmd_init(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    fleet = bool(getattr(args, "fleet", False))
    audio = bool(getattr(args, "audio", False))
    if audio and not fleet:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="--audio requires --fleet",
            remediation="the audio overlay layers on the fleet: run 'model init --fleet --audio'",
        )
    target = Path(args.target).expanduser() if args.target else _compose.default_deployment_dir()
    if args.apply:
        _emit_apply(target, fleet, audio, args.force, json_mode)
    else:
        _emit_dry_run(target, fleet, audio, json_mode)
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
        help="Scaffold the gateway deployment (the Qwen primary + co-resident "
        "embedding/reranker gears behind 1 OpenAI front; one generate backend by "
        "default, opt-in generate fallback) instead of a single model.",
    )
    p.add_argument(
        "--audio",
        action="store_true",
        help="Also scaffold the audio overlay (STT + TTS + realtime bridge). Requires --fleet.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p.add_argument("--apply", action="store_true", help="Actually write the files.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_init)
