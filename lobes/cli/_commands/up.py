"""``lobes up <role>`` — start (or ``--down``: stop) ONE Colleague role's gear.

**r3 (issue #81) — verb shape.** ``lobes up`` is a NEW top-level verb (matching
#81's ``lobes up cortex``), not a subcommand of ``fleet``. It reuses the
fleet/compose machinery (:mod:`lobes.runtime._compose`) rather than duplicating
any docker logic — it just targets ONE compose *service* (a role) instead of the
whole fleet, so a role is toggleable without disturbing the others.

Roles → compose services (issue #81, t7)::

    cortex    → vllm-primary       (the Qwen 27B generate primary)
    senses    → vllm-multimodal    (the Gemma 4 12B multimodal gear)
    embedder  → vllm-embed         (Qwen3-Embedding-0.6B pooling gear)
    reranker  → vllm-rerank        (Qwen3-Reranker-0.6B score gear)
    stt       → stt                (Parakeet — audio overlay, opt-in)
    tts       → chatterbox         (Chatterbox — audio overlay, opt-in)

These are the compose SERVICE names (top-level keys under ``services:``), NOT the
``container_name:`` values — ``docker compose up -d <service>`` addresses services.

**r4 (issue #81) — colleague-stack bundles audio.** ``colleague-stack`` is a
first-class target that brings up the FULL six-role Colleague set = the default
fleet roles (cortex/senses/embedder/reranker) PLUS the audio-overlay roles
(stt/tts). It therefore REQUIRES the audio overlay compose file
(``docker-compose.audio.yml``), scaffolded by ``lobes init --fleet --audio``; if
that file is absent the command explains how to add it rather than silently
yielding only four roles.

**Why a CLI-level target and NOT a compose ``profiles: [colleague-stack]`` block.**
A compose ``profiles:`` key is opt-IN: a service that declares one is NOT started
by a plain ``docker compose up``. Tagging the already-default-on services
(``vllm-primary`` …) with ``profiles: [colleague-stack]`` would DEMOTE them out of
the default fleet — a regression. Selecting the services across the two compose
files at the CLI layer keeps the default-on semantics intact, so colleague-stack
is a real, named CLI target instead.

**Mutation safety (repo rule).** ``up`` is a WRITE verb — dry-run by DEFAULT
(prints the exact ``docker compose …`` command it WOULD run); ``--apply`` is
required to execute it. ``--down`` toggles a role OFF via a scoped ``docker
compose stop`` (never a project-wide ``down``, which would remove every container).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lobes import roles
from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.runtime import _compose, _env

# role → the compose SERVICE name (the top-level key under ``services:`` — NOT the
# container_name). ``docker compose up -d <service>`` targets exactly these, so a
# role toggles without touching the rest of the fleet.
ROLE_SERVICE: dict[str, str] = {
    "cortex": "vllm-primary",
    "senses": "vllm-multimodal",
    "embedder": "vllm-embed",
    "reranker": "vllm-rerank",
    "stt": "stt",
    "tts": "chatterbox",
}

# The roles whose service lives in the audio overlay (docker-compose.audio.yml):
# any target that includes one needs the ``-f`` overlay AND the file scaffolded.
_AUDIO_ROLES: frozenset[str] = frozenset({"stt", "tts"})

# The colleague-stack bundle (r4): the FULL six-role Colleague set. Not a role in
# :data:`lobes.roles.ROLES` — it is ``up``'s own composite target, defined here.
COLLEAGUE_STACK = "colleague-stack"

# Every valid ``up`` target: the six roles (canonical order) + the bundle. Keyed
# off :data:`lobes.roles.ROLES` so this and the role registry never drift.
TARGETS: tuple[str, ...] = roles.ROLES + (COLLEAGUE_STACK,)


def _resolve(target: str) -> tuple[list[str], bool]:
    """``(services, needs_audio)`` for a target; raise USER_ERROR for an unknown one.

    ``needs_audio`` is True when any selected service lives in the audio overlay
    (stt/tts, or colleague-stack which always includes them, r4).
    """
    if target == COLLEAGUE_STACK:
        return [ROLE_SERVICE[r] for r in roles.ROLES], True
    if target in ROLE_SERVICE:
        return [ROLE_SERVICE[target]], target in _AUDIO_ROLES
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"unknown role '{target}'",
        remediation="valid: " + ", ".join(TARGETS),
    )


def _shape_blocked_services(deploy_dir: Path, services: list[str], target: str) -> bool:
    """Raise USER_ERROR if the deployment shape drops any of ``services`` for ``target``;
    otherwise return whether a shape overlay is present at all (for the ``-f`` chain).

    A role the deployment shape drops must not start here — name the shape
    instead of letting compose fail with "no such service" (t4b overlay).
    """
    shape_present = _compose.shape_overlay_present(deploy_dir)
    if not shape_present:
        return False
    overlay_text = (Path(deploy_dir) / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8")
    dropped = _compose._override_service_keys(overlay_text) - {"gateway"}
    blocked = sorted(set(services) & dropped)
    if blocked:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=(
                f"target '{target}' needs service(s) {', '.join(blocked)}, which this "
                f"deployment's shape drops ({_compose.SHAPE_OVERLAY})"
            ),
            remediation=(
                "pick a role this shape hosts, or re-scaffold with a shape that "
                "hosts it ('lobes init --shape machine-as-brain --apply')"
            ),
        )
    return True


def _compose_file_args(needs_audio: bool, shape_present: bool) -> list[str]:
    """The ``-f`` chain for the compose invocation.

    Shape overlay goes LAST (same rationale as _compose_files: its !reset on
    gateway.depends_on must be applied after every other file).
    """
    if not (needs_audio or shape_present):
        return []
    compose_files = ["-f", _compose.COMPOSE_FILE]
    if needs_audio:
        compose_files += ["-f", _compose.AUDIO_OVERLAY]
    if shape_present:
        compose_files += ["-f", _compose.SHAPE_OVERLAY]
    return compose_files


def cmd_up(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    target = args.role
    action = "stop" if getattr(args, "down", False) else "up"

    # Validate the target FIRST so `lobes up bogus` errors on the role name even
    # when nothing is scaffolded yet (acceptance criterion 4).
    services, needs_audio = _resolve(target)

    deploy_dir = _runtime_ops.deployment_dir(args)

    # colleague-stack / stt / tts reach into the audio overlay — it MUST be
    # scaffolded (r4). Explain how to add it rather than silently dropping audio.
    if needs_audio and not _compose.audio_overlay_present(deploy_dir):
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=(
                f"target '{target}' needs the audio overlay "
                f"({_compose.AUDIO_OVERLAY}), which is not scaffolded in {deploy_dir}"
            ),
            remediation=(
                "re-scaffold with 'lobes init --fleet --audio --apply' to add the "
                "stt/tts overlay, then retry"
            ),
        )

    # A role the deployment shape drops must not start here — name the shape
    # instead of letting compose fail with "no such service" (t4b overlay).
    shape_present = _shape_blocked_services(deploy_dir, services, target)

    compose_files = _compose_file_args(needs_audio, shape_present)
    argv = _compose.compose_service_argv(action, compose_files, services)
    command = " ".join(argv)

    if not args.apply:
        payload = {
            "dry_run": True,
            "target": target,
            "action": action,
            "services": services,
            "command": command,
            "deployment_dir": str(deploy_dir),
        }
        verb_word = "STOP" if action == "stop" else "START"
        text = (
            f"DRY RUN — would run: {command} in {deploy_dir} "
            f"({verb_word} target {target}: {', '.join(services)}).\n"
            "Re-run with --apply to execute."
        )
        emit_result(payload if json_mode else text, json_mode=json_mode)
        return 0

    verb_word = "stopping" if action == "stop" else "starting"
    emit_diagnostic(f">> {verb_word} {target} ({', '.join(services)}) in {deploy_dir}")
    if action == "up":
        # Create the durable-log dir (user-owned) before compose bind-mounts it —
        # the same guard serve / fleet up use.
        _compose.ensure_log_dir(
            deploy_dir,
            _env.read_env(deploy_dir / _compose.ENV_FILE, _compose.LOG_DIR_ENV) or None,
        )
    _runtime_ops.compose_check(_compose.run_compose(deploy_dir, argv), command)
    result = {
        ("started" if action == "up" else "stopped"): True,
        "target": target,
        "services": services,
        "command": command,
        "deployment_dir": str(deploy_dir),
    }
    done = "started" if action == "up" else "stopped"
    emit_result(result if json_mode else f">> {target} {done} in {deploy_dir}", json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "up",
        help="Start (or --down: stop) ONE Colleague role's gear, or the full "
        "'colleague-stack' (dry-run by default; --apply to commit).",
    )
    p.add_argument(
        "role",
        metavar="ROLE",
        help="cortex | senses | embedder | reranker | stt | tts | colleague-stack.",
    )
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--apply", action="store_true", help="Actually run docker compose.")
    p.add_argument(
        "--down",
        action="store_true",
        help="Stop the target service(s) instead of starting — a scoped "
        "'docker compose stop' that leaves the rest of the fleet untouched.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_up)
