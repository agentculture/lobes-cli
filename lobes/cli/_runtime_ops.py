"""Glue shared by the model-ops command handlers.

Kept out of :mod:`lobes.cli._commands` so that package holds only verb
modules (each with a ``register``); these are plain helpers.
"""

from __future__ import annotations

import argparse
import os
import subprocess

from lobes import assess
from lobes.cli._errors import EXIT_ENV_ERROR, ModelGearError
from lobes.profiles.loader import resolve_profile
from lobes.profiles.schema import Profile
from lobes.runtime import _compose, _detect, _env

# The built-in profile `lobes init` resolves for an UNKNOWN card (t14): a
# conservative small generate model + the two 0.6B pooling gears, no 27B —
# see lobes/profiles/builtin/base.toml. Never returned/selected silently;
# resolve_init_profile always pairs it with a warning naming the detected
# facts and the assumption made.
UNKNOWN_CARD_PROFILE = "base"


def deployment_dir(args: argparse.Namespace):
    """Resolve the deployment dir, raising ModelGearError if it isn't scaffolded."""
    return _compose.resolve_deployment_dir(getattr(args, "compose_dir", None))


def resolve_port(args: argparse.Namespace, env_path) -> int:
    """Port from ``--port`` if given, else ``VLLM_PORT`` in ``.env``, else 8000."""
    explicit = getattr(args, "port", None)
    if explicit is not None:
        return _env.parse_port(explicit, "--port")
    return _env.parse_port(_env.read_env(env_path, "VLLM_PORT", "8000"), "VLLM_PORT")


def resolve_port_soft(args: argparse.Namespace) -> tuple[int, object]:
    """Best-effort ``(port, deploy_dir|None)`` for read-only probes.

    Used by ``assess``/``benchmark`` which target a running endpoint: ``--port``
    wins; otherwise read ``.env`` from the resolved deployment dir; if nothing is
    scaffolded, fall back to 8000 without erroring (the endpoint may be elsewhere).
    """
    explicit = getattr(args, "port", None)
    if explicit is not None:
        return _env.parse_port(explicit, "--port"), None
    try:
        deploy_dir = _compose.resolve_deployment_dir(getattr(args, "compose_dir", None))
    except ModelGearError:
        return 8000, None
    port = _env.parse_port(
        _env.read_env(deploy_dir / _compose.ENV_FILE, "VLLM_PORT", "8000"), "VLLM_PORT"
    )
    return port, deploy_dir


def deployment_env_soft(args: argparse.Namespace) -> dict[str, str]:
    """Best-effort ``.env`` contents as a plain dict (``{}`` when unscaffolded).

    Shared by every read-only role-registry consumer (``capabilities``,
    ``endpoint``, ``measure``, issue #81): a missing deployment must never turn
    a read-only introspection verb into a hard error — it just means every
    gateway-fronted role other than the always-present ``cortex`` resolves to
    its catalog default with ``loaded=False`` (see :mod:`lobes.roles`).
    """
    try:
        deploy_dir = _compose.resolve_deployment_dir(getattr(args, "compose_dir", None))
    except ModelGearError:
        return {}
    return _env.read_env_file(deploy_dir / _compose.ENV_FILE)


def probe_tool_calling(port: int, served: str | None) -> dict:
    """Verify tool calling on the just-(re)started server.

    Thin adapter over :func:`lobes.assess.probe_tool_calls` (which never
    raises — HTTP 400, malformed 200, connection failure, and undecodable bodies
    all fold into a structured result): builds the local URL and returns the
    probe result (``ok``/``tool_calls``/``finish``/``error``), so ``switch`` /
    ``serve`` always completes.
    """
    return assess.probe_tool_calls(f"http://localhost:{port}", served or "")


def format_tool_probe(tc: dict) -> str:
    """One-line PASS/FAIL summary of a :func:`probe_tool_calling` result."""
    if tc.get("ok"):
        return f"tool calling: PASS — called {', '.join(tc.get('tool_calls') or [])}"
    reason = tc.get("error") or f"no finish call (tool_calls={tc.get('tool_calls')})"
    return f"tool calling: FAIL — {reason}"


def resolve_init_profile(
    explicit_profile: str | None,
    deploy_dir: os.PathLike | str,
    *,
    detect_fn=None,
) -> tuple[Profile, _detect.DetectedCard, str | None]:
    """Resolve the per-machine :class:`Profile` ``lobes init`` should render.

    Detection ALWAYS runs (even when ``explicit_profile`` is given) — the
    caller needs the detected card facts either way: to name a profile when
    none is given, or to compare against a forced ``--profile`` and warn on a
    mismatch. ``detect_fn`` is injectable (tests pass a lambda); ``None`` (the
    default) resolves ``lobes.runtime._detect.detect_card`` at CALL time (not
    bound at import time), so a test can also just
    ``monkeypatch.setattr(_detect, "detect_card", ...)`` the module attribute,
    matching this repo's usual probe-neutralisation idiom (see
    ``tests/conftest.py``).

    * ``explicit_profile`` given: it always wins. A warning string (for the
      caller to print to stderr) is returned when the detected card is
      UNKNOWN, or known but different from the forced name — "forcing a
      profile onto a card it wasn't validated for" per the plan. Never
      raises for this branch: an explicit choice is always honored.
    * ``explicit_profile`` is ``None``: the detected card name is used. A
      known card resolves silently (no warning). An UNKNOWN card (t14) no
      longer raises — it resolves the built-in :data:`UNKNOWN_CARD_PROFILE`
      ("base": a small generate model + the two 0.6B pooling gears, no 27B —
      see ``lobes/profiles/builtin/base.toml``) and returns a warning string
      naming every detected fact and the assumption made, so ``init``
      proceeds instead of refusing. This never falls back SILENTLY to any
      built-in — the caller always gets (and is expected to print) the
      warning.

    Returns ``(profile, card, warning)``; ``warning`` is ``None`` on the
    ordinary "detected a known card, no override" path.
    """
    card = (detect_fn or _detect.detect_card)()
    facts = (
        f"device_name={card.device_name!r}, "
        f"compute_capability={card.compute_capability!r}, "
        f"total_memory_gb={card.total_memory_gb!r}"
    )
    warning: str | None = None
    if explicit_profile:
        name = explicit_profile
        if not card.is_known:
            warning = (
                f"--profile {name!r} used on an undetected card ({facts}) — "
                "proceeding, but this profile was not validated for this box."
            )
        elif card.resolved != name.strip().lower():
            # Compare NORMALISED forms: card.resolved is always the registry's
            # lowercase canonical name, but `name` is the raw --profile value
            # as typed — `--profile Spark` on a detected "spark" card must not
            # warn on casing alone (resolve_profile() normalises the same way).
            warning = (
                f"--profile {name!r} overrides the detected card "
                f"{card.resolved!r} ({facts}) — proceeding, but this profile "
                "was not validated for this box."
            )
    else:
        if card.is_known:
            name = card.resolved
        else:
            name = UNKNOWN_CARD_PROFILE
            warning = (
                f"unrecognised card ({facts}) — serving the conservative "
                f"{UNKNOWN_CARD_PROFILE!r} profile (a small generate model + the "
                "embed/rerank pooling gears, no 27B); pass --profile to override."
            )
    profile = resolve_profile(name, deploy_dir=deploy_dir)
    return profile, card, warning


def compose_check(completed: subprocess.CompletedProcess, label: str) -> None:
    """Raise ModelGearError when a ``docker compose`` call exits non-zero."""
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"{label} failed (exit {completed.returncode})",
            remediation=detail[-500:] if detail else "check docker and 'lobes status'",
        )
