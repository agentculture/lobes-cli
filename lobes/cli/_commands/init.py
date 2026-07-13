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
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.cli._runtime_ops import resolve_init_profile
from lobes.profiles.render import profile_env
from lobes.runtime import _compose, _env


def _templates(fleet: bool, audio: bool) -> dict[str, str]:
    if not fleet:
        return _compose.SINGLE_TEMPLATES
    templates = dict(_compose.FLEET_TEMPLATES)
    if audio:
        templates.update(_compose.AUDIO_TEMPLATES)
    return templates


def _resolve_fleet_profile(target: Path, profile_name: str | None):
    """Resolve the per-machine profile for a fleet init; emits a stderr warning
    when ``--profile`` forces a name onto a card it wasn't validated for (or an
    undetected one), and likewise when detection itself comes back UNKNOWN with
    no ``--profile`` override — that case now resolves the conservative 'base'
    built-in (t14) rather than refusing; see ``resolve_init_profile``."""
    profile, card, warning = resolve_init_profile(profile_name, target)
    if warning:
        emit_diagnostic(f"warning: {warning}")
    return profile, card


def _profile_plan_lines(profile, card, profile_name: str | None) -> list[str]:
    facts = (
        f"device_name={card.device_name!r}, compute_capability={card.compute_capability!r}, "
        f"total_memory_gb={card.total_memory_gb!r}"
    )
    if profile_name:
        why = f"forced via --profile; detected card={card.resolved!r}, {facts}"
    else:
        why = f"auto-detected: {facts}"
    lines = [f"Profile: {profile.name} ({why})"]
    env = profile_env(profile)
    lines.append(f"  would set {len(env)} profile env var(s) in {_compose.ENV_FILE}")
    return lines


def _values_equal(current: str, new: str) -> bool:
    """True when two env-var strings represent the same value.

    A straight string match covers everything but numbers; a profile-resolved
    float (``str(0.3) == "0.3"``) and the template's own literal (``"0.30"``,
    written for human readability — see ``env.example``) are the SAME value
    with different spellings, so a numeric-aware fallback avoids rewriting a
    key whose resolved value merely restates the shipped default in fewer
    digits. A non-numeric mismatch (e.g. an actually different model id or
    flag token) still compares unequal and gets written.
    """
    if current == new:
        return True
    try:
        return float(current) == float(new)
    except ValueError:
        return False


def _apply_profile_env(env_path: Path, env: dict[str, str]) -> None:
    """Write a profile's rendered env vars into ``.env``, skipping no-op writes.

    ``write_scaffold`` has already copied the template's own ``env.example``
    defaults into ``.env`` by the time this runs. When the resolved profile's
    value for a key is the SAME as what's already there (see
    :func:`_values_equal`), the original line is left untouched instead of
    being rewritten in a different (but equal) format — this keeps a
    zero-divergence profile (e.g. ``spark`` on a freshly scaffolded fleet
    ``.env``, which already ships spark's own defaults) byte-identical to
    today's plain scaffold.
    """
    current = _env.read_env_file(env_path)
    for key, value in env.items():
        existing = current.get(key)
        if existing is not None and _values_equal(existing, value):
            continue
        _env.set_env(env_path, key, value)


def _profile_plan_dict(profile, card, profile_name: str | None) -> dict:
    return {
        "profile": profile.name,
        "profile_forced": bool(profile_name),
        "detected_card": card.resolved,
        "detected_facts": {
            "device_name": card.device_name,
            "compute_capability": card.compute_capability,
            "total_memory_gb": card.total_memory_gb,
        },
        "profile_env": profile_env(profile),
    }


def _emit_dry_run(
    target: Path, fleet: bool, audio: bool, json_mode: bool, profile_name: str | None
) -> None:
    plan = _compose.scaffold_plan(target, _templates(fleet, audio))
    profile = card = None
    if fleet:
        # Detection/warning happens on a dry run too — the plan must be honest
        # about what --apply would do, including the fallback profile it would
        # serve on an UNKNOWN card.
        profile, card = _resolve_fleet_profile(target, profile_name)
    if json_mode:
        payload = {
            "dry_run": True,
            "fleet": fleet,
            "single": not fleet,
            "audio": audio,
            "target": str(target),
            "files": [{"name": name, "exists": exists} for name, exists in plan],
        }
        if fleet:
            payload.update(_profile_plan_dict(profile, card, profile_name))
        emit_result(payload, json_mode=True)
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
    if fleet:
        lines.extend(_profile_plan_lines(profile, card, profile_name))
    lines.append("Re-run with --apply to write.")
    emit_result("\n".join(lines), json_mode=False)


def _emit_apply(
    target: Path, fleet: bool, audio: bool, force: bool, json_mode: bool, profile_name: str | None
) -> None:
    profile = card = None
    if fleet:
        # Resolve BEFORE writing anything — an explicit --profile mismatch or an
        # UNKNOWN card (falling back to the conservative 'base' profile, t14)
        # both warn here, before any file is written.
        profile, card = _resolve_fleet_profile(target, profile_name)
    written = _compose.write_scaffold(target, force=force, templates=_templates(fleet, audio))
    # Create the durable-log dir now (as the invoking user) so the compose bind-mount
    # source exists before `lobes serve` / `fleet up` — otherwise Docker makes it
    # root-owned. The mg-logwrap entrypoint writes per-boot logs here (issue #50).
    _compose.ensure_log_dir(target)
    if fleet:
        # Render the resolved profile's knobs into .env, the same way any other
        # env value gets written here (lobes.runtime._env.set_env) — skipping
        # keys the profile merely restates from the template default.
        _apply_profile_env(target / _compose.ENV_FILE, profile_env(profile))
        # Persist the profile choice itself for doctor/status to report
        _env.set_env(target / _compose.ENV_FILE, "LOBES_PROFILE", profile.name)
        # Pin the gateway image to the lobes-cli release that scaffolded this.
        _env.set_env(target / _compose.ENV_FILE, "MODEL_GEAR_VERSION", __version__)
    if audio:
        # Extend the fleet .env with the audio keys (NGC_API_KEY, ports, AUDIO_URL …).
        _compose.append_audio_env(target)
    if json_mode:
        payload = {
            "scaffolded": str(target),
            "fleet": fleet,
            "single": not fleet,
            "audio": audio,
            "files": [p.name for p in written],
        }
        if fleet:
            payload["profile"] = profile.name
            payload["profile_forced"] = bool(profile_name)
            payload["detected_card"] = card.resolved
        emit_result(payload, json_mode=True)
        return
    next_step = (
        "docker login nvcr.io && lobes fleet up --apply"
        if fleet
        else "docker login nvcr.io && lobes serve --apply"
    )
    profile_note = f"\n>> profile: {profile.name}" if fleet else ""
    emit_result(
        f">> scaffolded {target}:\n"
        + "\n".join(f"  {p.name}" for p in written)
        + (f"\n  {_compose.ENV_FILE} (+ audio keys)" if audio else "")
        + profile_note
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
    profile_name = getattr(args, "profile", None)
    if args.apply:
        _emit_apply(target, fleet, audio, args.force, json_mode, profile_name)
    else:
        _emit_dry_run(target, fleet, audio, json_mode, profile_name)
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
    p.add_argument(
        "--profile",
        help="Per-machine profile to render into .env (default: auto-detect the "
        "host card — spark, thor, ... — via lobes.runtime._detect). Overrides "
        "detection, including forcing a profile onto a card it was not "
        "validated for (warns, but proceeds). Fleet topology only. On an "
        "UNKNOWN card with no --profile, init warns and serves the conservative "
        "'base' profile (small generate model + pooling gears, no 27B) instead "
        "of guessing or refusing.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing files.")
    p.add_argument("--apply", action="store_true", help="Actually write the files.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_init)
