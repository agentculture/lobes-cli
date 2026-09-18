"""``lobes up <role>`` — start (or ``--down``: stop) ONE Colleague role's gear.

**r3 (issue #81) — verb shape.** ``lobes up`` is a NEW top-level verb (matching
#81's ``lobes up cortex``), not a subcommand of ``fleet``. It reuses the
fleet/compose machinery (:mod:`lobes.runtime._compose`) rather than duplicating
any docker logic — it just targets ONE compose *service* (a role) instead of the
whole fleet, so a role is toggleable without disturbing the others.

Roles → compose services (issue #81, t7)::

    cortex    → vllm-primary       (the Qwen 27B generate primary)
    senses    → vllm-multimodal    (the Gemma 4 12B multimodal gear)
    muse      → vllm-muse          (the Gemma 4 31B creative lobe — opt-in,
                                    hosted only by a muse-hosting shape)
    worker    → vllm-worker        (the Qwen3.6-35B-A3B fast ground-work
                                    doer — opt-in, hosted only by a
                                    worker-hosting shape)
    embedder  → vllm-embed         (Qwen3-Embedding-0.6B pooling gear)
    reranker  → vllm-rerank        (Qwen3-Reranker-0.6B score gear)
    stt       → stt                (Parakeet — audio overlay, opt-in)
    tts       → chatterbox         (Chatterbox — audio overlay, opt-in)

These are the compose SERVICE names (top-level keys under ``services:``), NOT the
``container_name:`` values — ``docker compose up -d <service>`` addresses services.

**The gateway is a target too (issue #222).** ``gateway`` is not a Colleague role
— it is the stdlib reverse proxy fronting them — but it is the service an operator
most often needs to touch ALONE: it is pure stdlib, rebuilds in seconds, and bakes
``MODEL_GEAR_VERSION`` into its image, while the lobes behind it take minutes and
hold tens of GiB. Before #222 there was no verb for "reinstall the gateway, touch
nothing else", so the operator hand-wrote ``docker compose up -d --build gateway``
— which walked ``depends_on`` and recreated the whole fleet. ``lobes up gateway
--build`` is that need, named.

**Isolation is real, not just documented (issue #222).** Every ``up`` here runs
with ``--no-deps`` (:data:`lobes.runtime._compose.NO_DEPS_FLAG`); without it
``docker compose up -d <service>`` follows ``depends_on`` and (re)creates the
dependencies too, which is how a gateway-only restart became a fleet-wide one on
a live Thor. ``--down`` needs no equivalent: compose ``stop`` never walks
``depends_on``.

**r4 (issue #81) — colleague-stack bundles audio.** ``colleague-stack`` is a
first-class target that brings up the FULL seven-role default-hosted set = the default
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
from lobes.cli._commands.mesh import trigger_reannounce
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.profiles.shape_render import GATEWAY_SERVICE
from lobes.profiles.shapes import (
    DEFAULT_HOSTED_ROLES,
    OPT_IN_CORE_ROLES,
    builtin_shape_names,
    load_builtin_shape,
)
from lobes.runtime import _compose, _env

# role → the compose SERVICE name (the top-level key under ``services:`` — NOT the
# container_name). ``docker compose up -d <service>`` targets exactly these, so a
# role toggles without touching the rest of the fleet.
ROLE_SERVICE: dict[str, str] = {
    "cortex": "vllm-primary",
    "senses": "vllm-multimodal",
    "muse": "vllm-muse",
    "worker": "vllm-worker",
    "associate": "vllm-associate",
    "hand": "vllm-hand",
    "embedder": "vllm-embed",
    "reranker": "vllm-rerank",
    "stt": "stt",
    "tts": "chatterbox",
    # `innereye` (issue #82) — the ComfyUI render tenant, the eleventh role.
    # Its service is `comfyui` (NOT a `vllm-*` gear: it is not a vLLM lane at
    # all), and like muse/worker/associate it is an opt-in core role, so
    # `lobes up innereye` is gated by _opt_in_core_activated below. The
    # service itself is declared by a later task; this entry is the role's
    # lifecycle registration only.
    "innereye": "comfyui",
}

# The roles whose service lives in the audio overlay (docker-compose.audio.yml):
# any target that includes one needs the ``-f`` overlay AND the file scaffolded.
_AUDIO_ROLES: frozenset[str] = frozenset({"stt", "tts"})

# The colleague-stack bundle (r4): the DEFAULT-HOSTED Colleague set — the SEVEN
# roles machine-as-brain hosts (cortex, senses, hand, embedder, reranker, stt,
# tts). Deliberately NOT all of :data:`lobes.roles.ROLES`: the opt-in ``muse``
# and ``worker`` lobes are hosted only by their own hosting shapes and their
# services are compose-profile-gated, so bundling them here would break
# colleague-stack on every default deployment. `hand` IS bundled — it is
# default-hosted and its service carries no profile gate. Not a role itself —
# ``up``'s own composite target.
COLLEAGUE_STACK = "colleague-stack"

# The gateway service (issue #222). NOT a Colleague role — deliberately absent
# from :data:`lobes.roles.ROLES`, from ROLE_SERVICE, and from the colleague-stack
# bundle, so nothing that enumerates roles starts counting it as one. It is a
# first-class ``up`` TARGET only: the one service an operator legitimately
# restarts or re-images on its own.
# The TARGET name a caller types. Equal to the compose SERVICE name
# (:data:`lobes.profiles.shape_render.GATEWAY_SERVICE`, imported above rather
# than re-spelled) but a separate constant: those are two namespaces, and every
# other target here is a ROLE whose name differs from its service.
GATEWAY_TARGET = GATEWAY_SERVICE

# Every valid ``up`` target: the TEN roles (canonical order) + the bundle + the
# gateway. Keyed off :data:`lobes.roles.ROLES` so this and the role registry
# never drift.
TARGETS: tuple[str, ...] = roles.ROLES + (COLLEAGUE_STACK, GATEWAY_TARGET)


def _resolve(target: str) -> tuple[list[str], bool]:
    """``(services, needs_audio)`` for a target; raise USER_ERROR for an unknown one.

    ``needs_audio`` is True when any selected service lives in the audio overlay
    (stt/tts, or colleague-stack which always includes them, r4).
    """
    if target == COLLEAGUE_STACK:
        return [ROLE_SERVICE[r] for r in DEFAULT_HOSTED_ROLES], True
    if target == GATEWAY_TARGET:
        # The gateway lives in the BASE fleet file and fronts the audio lanes over
        # HTTP rather than declaring them, so it never pulls in the audio overlay.
        return [GATEWAY_SERVICE], False
    if target in ROLE_SERVICE:
        return [ROLE_SERVICE[target]], target in _AUDIO_ROLES
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"unknown role '{target}'",
        remediation="valid: " + ", ".join(TARGETS),
    )


def _hosting_shapes_for(role: str) -> tuple[str, ...]:
    """Every built-in shape that actually hosts ``role``, sorted by name.

    Looked up from the shapes themselves rather than assumed from a naming
    convention: the opt-in core roles do NOT all live behind a
    ``thor-<role>`` shape (``associate``'s is ``orin-associate``, and
    ``innereye``'s is ``spark-innereye``), so a hardcoded ``f"thor-{role}"``
    would point an operator at a shape that does not exist for those two.
    """
    return tuple(
        name
        for name in builtin_shape_names()
        if (shape := load_builtin_shape(name)) is not None and shape.hosts_role(role)
    )


def _opt_in_core_activated(deploy_dir: Path, target: str) -> None:
    """Raise USER_ERROR when an opt-in core role's compose profile isn't active.

    ``vllm-muse`` is parked behind the ``muse`` Docker Compose profile in the
    base fleet template; without ``COMPOSE_PROFILES`` naming it (rendered by a
    muse-hosting shape's activation env), ``docker compose up vllm-muse`` fails
    with an unexplained "no such service". Name the real fix instead.
    """
    if target not in OPT_IN_CORE_ROLES:
        return
    profiles = _env.read_env(Path(deploy_dir) / _compose.ENV_FILE, "COMPOSE_PROFILES") or ""
    if target in [p.strip() for p in profiles.split(",")]:
        return
    hosting_shapes = _hosting_shapes_for(target)
    if hosting_shapes:
        example_shape = hosting_shapes[0]
        remediation = (
            f"re-scaffold with a {target}-hosting shape "
            f"('lobes init --shape {example_shape} --apply'), then retry"
        )
    else:
        # No built-in shape hosts this role today (a future opt-in core role
        # added here before its shape lands) -- name the gap honestly rather
        # than a guessed shape name that would 404 in `resolve_shape`.
        remediation = (
            f"no built-in shape hosts '{target}' yet; write a custom shape that "
            f"lists it in 'hosts' and re-scaffold with 'lobes init --shape <name> "
            "--apply', then retry"
        )
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=(
            f"role '{target}' is opt-in and this deployment does not activate it "
            f"(COMPOSE_PROFILES in .env does not include '{target}')"
        ),
        remediation=remediation,
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
    # Only the blocks carrying the `shape-dropped` profile marker count as drops.
    # The override also holds blocks that do NOT park anything: a pre-#222
    # `gateway: depends_on: !reset null`, the associate-first start-order
    # reversals (#260), and a `comfyui: ports:` block when INNEREYE_UI_PORT
    # publishes the ComfyUI UI. Reading mere presence as "dropped" would refuse
    # `lobes up` for a lane this deployment very much hosts.
    dropped = _compose.shape_parked_service_keys(overlay_text) - {GATEWAY_SERVICE}
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


def _compose_file_args(
    needs_audio: bool,
    shape_present: bool,
    local_override: bool,
    gpu_present: bool = False,
    audio_he_present: bool = False,
) -> list[str]:
    """The ``-f`` chain for the compose invocation — delegates to the single
    composition authority (:func:`lobes.runtime._compose.compose_file_args`,
    issue #137), so ``lobes up <role>`` and ``lobes fleet up`` cannot disagree
    about which files a deployment is made of.

    ``needs_audio`` is this verb's own semantics — "only the overlays the
    TARGETED services need", not "the overlay exists" — which is why the
    booleans are passed in rather than probed from the deployment dir. This
    used to be a parallel hand-rolled builder; it drifted from
    ``_compose_files`` exactly once (#135/#136) before being folded in.

    ``gpu_present`` (the csv-mode GPU-access override) is a deployment fact,
    not a per-target one: EVERY targeted gear is a GPU gear on this board, so
    it is probed rather than derived from the target. Its audio half is paired
    with the audio overlay by the authority itself, so a non-audio target still
    never pulls in a file naming services the chain does not declare.

    ``audio_he_present`` (the Hebrew overlay, t15) is a deployment fact too —
    it rides with the audio overlay, so the authority drops it for any target
    that does not need audio.
    """
    return _compose.compose_file_args(
        audio=needs_audio,
        shape=shape_present,
        local=local_override,
        gpu=gpu_present,
        audio_he=audio_he_present,
    )


def _audio_overlay_required(deploy_dir: Path, target: str, needs_audio: bool) -> None:
    """Raise USER_ERROR when the target reaches into an unscaffolded audio overlay.

    ``colleague-stack`` / ``stt`` / ``tts`` need ``docker-compose.audio.yml`` (r4).
    Explain how to add it rather than silently yielding only the non-audio roles.
    """
    if not needs_audio or _compose.audio_overlay_present(deploy_dir):
        return
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


def _resolve_build(args: argparse.Namespace, action: str) -> bool:
    """``--build``, refusing the one combination that cannot mean anything.

    ``--build`` maps to ``docker compose up --build``; a ``--down`` is a
    ``stop``, which never builds an image. Silently ignoring the flag there
    would let an operator believe they had re-imaged something.
    """
    build = bool(getattr(args, "build", False))
    if build and action == "stop":
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="--build has no meaning with --down (a stop never builds an image)",
            remediation="drop --build, or drop --down to rebuild and restart the target",
        )
    return build


def cmd_up(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    target = args.role
    action = "stop" if getattr(args, "down", False) else "up"

    # Validate the target FIRST so `lobes up bogus` errors on the role name even
    # when nothing is scaffolded yet (acceptance criterion 4).
    services, needs_audio = _resolve(target)

    deploy_dir = _runtime_ops.deployment_dir(args)

    _audio_overlay_required(deploy_dir, target, needs_audio)

    # An opt-in core role (muse) needs its compose profile activated by a
    # hosting shape — name the real fix instead of compose's "no such service".
    _opt_in_core_activated(deploy_dir, target)

    # A role the deployment shape drops must not start here — name the shape
    # instead of letting compose fail with "no such service" (t4b overlay).
    shape_present = _shape_blocked_services(deploy_dir, services, target)

    compose_files = _compose_file_args(
        needs_audio,
        shape_present,
        _compose.local_override_present(deploy_dir),
        _compose.gpu_overlay_present(deploy_dir),
        _compose.audio_he_overlay_present(deploy_dir),
    )
    build = _resolve_build(args, action)
    argv = _compose.compose_service_argv(action, compose_files, services, build=build)
    command = " ".join(argv)

    if not args.apply:
        payload = {
            "dry_run": True,
            "target": target,
            "action": action,
            "services": services,
            "command": command,
            "build": build,
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
    # Mesh-brain-join (t8 follow-up): starting or stopping a role changes what
    # this box serves, so peers should learn it within one probe refresh
    # rather than waiting for the next scheduled heartbeat. No-op when
    # LOBES_MESH_KEY is unset; best-effort (never fails an already-successful
    # up/down).
    trigger_reannounce(
        _runtime_ops.resolve_port(args, deploy_dir / _compose.ENV_FILE),
        _env.read_env_file(deploy_dir / _compose.ENV_FILE),
    )
    result = {
        ("started" if action == "up" else "stopped"): True,
        "target": target,
        "services": services,
        "command": command,
        "build": build,
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
        help="cortex | senses | muse | worker | associate | hand | embedder | "
        "reranker | stt | tts | colleague-stack | gateway.",
    )
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--apply", action="store_true", help="Actually run docker compose.")
    p.add_argument(
        "--down",
        action="store_true",
        help="Stop the target service(s) instead of starting — a scoped "
        "'docker compose stop' that leaves the rest of the fleet untouched.",
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="Rebuild the target's image before starting it (up only). Only the "
        "gateway is built from a local Dockerfile — use this to re-image it at a "
        "new MODEL_GEAR_VERSION without touching the lobes behind it (#222).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_up)
