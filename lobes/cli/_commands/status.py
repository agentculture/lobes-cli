"""``lobes status`` — read-only snapshot of the current deployment.

Reports the configured model/served-name/port (from ``.env``), the container
lifecycle + health state, and whether ``/health`` is responding. This is the
*configured* served model (from ``.env``) + health — not a live ``/v1/models``
query; for the full supported catalog you can switch to, use ``lobes overview --list``.
"""

from __future__ import annotations

import argparse

from lobes.cli import _runtime_ops
from lobes.cli._output import emit_result
from lobes.runtime import _compose, _env, _health

# Shown when a key is absent/empty in .env (matches _env.read_env's default).
_UNSET = "(unset)"


def cmd_status(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)

    report = {
        "model": _env.read_env(env_path, "VLLM_MODEL", _UNSET),
        "served_name": _env.read_env(env_path, "VLLM_SERVED_NAME", _UNSET),
        "port": port,
        "tool_call_parser": _env.read_env(env_path, "VLLM_TOOL_CALL_PARSER", _UNSET),
        "deployment_dir": str(deploy_dir),
        "container": _compose.CONTAINER,
        "state": _compose.inspect_state(),
        "health": "ok" if _health.is_healthy(port) else "not responding",
    }

    if json_mode:
        emit_result(report, json_mode=True)
    else:
        emit_result(
            "\n".join(
                [
                    f"model:  {report['model']}",
                    f"served: {report['served_name']}  port: {report['port']}",
                    f"parser: {report['tool_call_parser']}",
                    f"dir:    {report['deployment_dir']}",
                    f"state:  {report['container']} — {report['state']}",
                    f"health: {report['health']} (:{port})",
                ]
            ),
            json_mode=False,
        )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "status",
        help="Read-only: the configured served model (from .env), container state, "
        "/health (catalog: lobes overview --list).",
    )
    p.add_argument("--port", type=int, help="Host port (default: VLLM_PORT in .env, else 8000).")
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_status)
