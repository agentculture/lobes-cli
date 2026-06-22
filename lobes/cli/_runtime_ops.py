"""Glue shared by the model-ops command handlers.

Kept out of :mod:`lobes.cli._commands` so that package holds only verb
modules (each with a ``register``); these are plain helpers.
"""

from __future__ import annotations

import argparse
import subprocess

from lobes import assess
from lobes.cli._errors import EXIT_ENV_ERROR, ModelGearError
from lobes.runtime import _compose, _env


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


def compose_check(completed: subprocess.CompletedProcess, label: str) -> None:
    """Raise ModelGearError when a ``docker compose`` call exits non-zero."""
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"{label} failed (exit {completed.returncode})",
            remediation=detail[-500:] if detail else "check docker and 'model status'",
        )
