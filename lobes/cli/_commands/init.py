"""``lobes init [TARGET]`` — scaffold a deployment directory.

Copies the packaged compose + ``env.example``→``.env`` (+ gateway Dockerfile)
into ``TARGET`` (default ``~/.lobes``; ``lobes init .`` for the local folder).

The DEFAULT topology is the **fleet duo** (issue #69): the always-warm Qwen
generate primary + the multimodal Gemma gear, fronted by the stdlib gateway with
the co-resident embedding/reranker gears (the legacy 4B ``minor`` / 14B
``middle`` generate gears stay behind opt-in compose profiles). ``--single``
(alias ``--legacy``) restores the old single-model scaffold (one vLLM server, no
gateway). ``--fleet`` is now a default-implied no-op kept for back-compat.
``--audio`` layers the realtime audio overlay on the fleet (incompatible with
``--single``). Mutating: dry-run by default; ``--apply`` writes, ``--force``
overwrites.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lobes import __version__
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.runtime import _compose, _env


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
                "single": not fleet,
                "audio": audio,
                "target": str(target),
                "files": [{"name": name, "exists": exists} for name, exists in plan],
            },
            json_mode=True,
        )
        return
    if fleet and audio:
        scope = "the fleet duo + audio overlay "
    elif fleet:
        scope = "the fleet duo (main + multimodal) "
    else:
        scope = "the legacy single-model "
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
    # source exists before `lobes serve` / `fleet up` — otherwise Docker makes it
    # root-owned. The mg-logwrap entrypoint writes per-boot logs here (issue #50).
    _compose.ensure_log_dir(target)
    if fleet:
        # Pin the gateway image to the lobes-cli release that scaffolded this.
        _env.set_env(target / _compose.ENV_FILE, "MODEL_GEAR_VERSION", __version__)
    if audio:
        # Extend the fleet .env with the audio keys (NGC_API_KEY, ports, AUDIO_URL …).
        _compose.append_audio_env(target)
    if json_mode:
        emit_result(
            {
                "scaffolded": str(target),
                "fleet": fleet,
                "single": not fleet,
                "audio": audio,
                "files": [p.name for p in written],
            },
            json_mode=True,
        )
        return
    next_step = (
        "docker login nvcr.io && lobes fleet up --apply"
        if fleet
        else "docker login nvcr.io && lobes serve --apply"
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
    # The fleet duo is the DEFAULT (issue #69); --single (alias --legacy) opts out
    # to the legacy single-model scaffold. --fleet is a default-implied no-op alias.
    single = bool(getattr(args, "single", False))
    fleet = not single
    audio = bool(getattr(args, "audio", False))
    if audio and not fleet:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="--audio is incompatible with --single",
            remediation="the audio overlay layers on the fleet (the default): "
            "drop --single, e.g. 'lobes init --audio'",
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
        help="Scaffold a deployment dir (default ~/.lobes; dry-run by default; --apply).",
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Where to scaffold (default ~/.lobes; '.' for the current folder).",
    )
    # Topology selector. Default is the fleet duo (main primary + multimodal gear
    # + gateway + embed/rerank); --single (alias --legacy) restores the legacy
    # single-model scaffold. --fleet is the now-default-implied no-op kept for
    # back-compat. They are mutually exclusive.
    topology = p.add_mutually_exclusive_group()
    topology.add_argument(
        "--single",
        "--legacy",
        dest="single",
        action="store_true",
        help="Scaffold the legacy single-model deployment (one vLLM server, no "
        "gateway) instead of the default fleet duo.",
    )
    topology.add_argument(
        "--fleet",
        action="store_true",
        help="Default-implied no-op (kept for back-compat): the fleet duo — the "
        "Qwen primary + the multimodal gear behind 1 OpenAI gateway with the "
        "co-resident embedding/reranker gears — is now the default scaffold.",
    )
    p.add_argument(
        "--audio",
        action="store_true",
        help="Also scaffold the audio overlay (STT + TTS + realtime bridge). "
        "Layers on the fleet (the default); incompatible with --single.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p.add_argument("--apply", action="store_true", help="Actually write the files.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_init)
