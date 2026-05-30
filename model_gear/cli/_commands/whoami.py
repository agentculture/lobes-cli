"""``model whoami`` — the smallest identity probe.

Reports model-gear's view of the world: the tool + version, the machine it runs
on, the model currently served (from the deployment ``.env``), the container's
health, and the agent that consumes the model (``model-gear``, read from
``culture.yaml``). Read-only.
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

from model_gear import __version__
from model_gear.cli._output import emit_result
from model_gear.runtime import _compose, _env, _health

_FALLBACK_AGENT = "model-gear"
_DEFAULT_MODEL = "mmangkad/Qwen3.6-27B-NVFP4"


def _find_culture_yaml() -> Path | None:
    """Locate the repo's ``culture.yaml`` by walking up from this module.

    Present in a source checkout (names the deployed agent); absent in a wheel
    install, where the caller falls back to the literal default.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "culture.yaml"
        if candidate.is_file():
            return candidate
    return None


def _agent_nick() -> str:
    """Return the deployed agent's ``suffix`` from ``culture.yaml`` (or the default)."""
    cfg = _find_culture_yaml()
    if cfg is None:
        return _FALLBACK_AGENT
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError:
        return _FALLBACK_AGENT
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- suffix:", "suffix:")):
            _, _, value = stripped.partition("suffix:")
            return value.strip().strip("'\"") or _FALLBACK_AGENT
    return _FALLBACK_AGENT


def _gpu_name() -> str:
    """First line of ``nvidia-smi -L`` (best-effort; ``unknown`` if unavailable)."""
    r = _compose._probe(["nvidia-smi", "-L"])
    if r is None or r.returncode != 0 or not r.stdout.strip():
        return "unknown"
    return r.stdout.splitlines()[0].strip() or "unknown"


def _served_and_port() -> tuple[str, int]:
    """The currently-served model + port, read from the deployment ``.env``."""
    try:
        deploy_dir = _compose.resolve_deployment_dir(None)
    except Exception:  # noqa: BLE001 - no deployment scaffolded yet
        return _DEFAULT_MODEL, 8000
    env_path = deploy_dir / _compose.ENV_FILE
    served = _env.read_env(env_path, "VLLM_SERVED_NAME", _DEFAULT_MODEL)
    port = _env.parse_port(_env.read_env(env_path, "VLLM_PORT", "8000"))
    return served, port


def _container_health(port: int) -> str:
    state = _compose.inspect_state()
    if state == "not created":
        return "not created"
    return "ok" if _health.is_healthy(port) else "down"


def report() -> dict[str, object]:
    served, port = _served_and_port()
    return {
        "tool": "model-gear",
        "version": __version__,
        "machine": {"host": socket.gethostname(), "gpu": _gpu_name()},
        "served_model": served,
        "port": port,
        "container_health": _container_health(port),
        "agent": _agent_nick(),
    }


def cmd_whoami(args: argparse.Namespace) -> None:
    identity = report()
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(identity, json_mode=True)
        return
    machine = identity["machine"]
    text = (
        f"tool: {identity['tool']}\n"
        f"version: {identity['version']}\n"
        f"machine: {machine['host']} ({machine['gpu']})\n"
        f"served_model: {identity['served_model']}  port: {identity['port']}\n"
        f"container_health: {identity['container_health']}\n"
        f"agent: {identity['agent']}"
    )
    emit_result(text, json_mode=False)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "whoami",
        help="Report the tool, machine, served model, container health, and the deployed agent.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_whoami)
