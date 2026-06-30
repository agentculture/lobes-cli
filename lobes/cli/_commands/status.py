"""``lobes status`` — read-only snapshot of the current deployment.

Reports the configured model/served-name/port (from ``.env``), the container
lifecycle + health state, and whether ``/health`` is responding. This is the
*configured* served model (from ``.env``) + health — not a live ``/v1/models``
query; for the full supported catalog you can switch to, use ``lobes overview --list``.

``--pressure`` sub-mode (issue #68/#69, t6/t7)
----------------------------------------------
Reads host memory pressure from ``/proc`` (via
:func:`lobes.runtime._pressure.sample_pressure`) and maps it to the highest
tier currently permitted under that pressure via
:func:`lobes.gateway._pressure_policy.decide` (called with
``requested_tier="main"`` so the result shows how far the top tier would be
downgraded right now).  The tier/model are reported in the new **main / minor /
multimodal** vocabulary (issue #69): under degraded pressure the ceiling drops
to ``minor`` — the only cheaper target — because ``multimodal`` is a *different
capability*, not a cheaper rung below ``main`` (the t6 seam).  This path is
**strictly read-only**: it touches neither the deployment dir nor the Docker
socket.
"""

from __future__ import annotations

import argparse

from lobes import catalog
from lobes.cli import _runtime_ops
from lobes.cli._output import emit_result
from lobes.gateway import _pressure_policy
from lobes.runtime import _compose, _env, _health
from lobes.runtime import _pressure as _pressure_mod

# Shown when a key is absent/empty in .env (matches _env.read_env's default).
_UNSET = "(unset)"


def _cmd_status_pressure(json_mode: bool) -> int:
    """Handle ``lobes status --pressure`` — a read-only /proc snapshot.

    Computes the highest tier currently permitted under live host pressure and
    resolves the model id that would serve that tier.  No deployment dir,
    no docker compose, no .env reads.
    """
    p = _pressure_mod.sample_pressure()
    d = _pressure_policy.decide(
        p["swap_used_percent"],
        p["iowait_percent"],
        requested_tier="main",
    )
    # New-vocabulary ceiling: "main" (full) when warm, "minor" under degraded
    # pressure (the only cheaper target — multimodal is not a rung; t6 seam).
    tier = d["max_allowed_tier"]
    model = catalog.resolve_tier(tier).id

    result = {
        "tier": tier,
        "model": model,
        "mode": d["mode"],
        "reason": d["reason"],
        "pressure": p,
    }

    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(
            "\n".join(
                [
                    f"tier:    {result['tier']}",
                    f"model:   {result['model']}",
                    f"mode:    {result['mode']}",
                    f"reason:  {result['reason']}",
                    f"swap:    {p['swap_used_percent']:.1f}%",
                    f"iowait:  {p['iowait_percent']:.1f}%",
                ]
            ),
            json_mode=False,
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))

    # --pressure path: /proc sampler only — no deploy dir, no docker, no .env.
    if getattr(args, "pressure", False):
        return _cmd_status_pressure(json_mode)

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
    p.add_argument(
        "--pressure",
        action="store_true",
        help=(
            "Read-only: sample host memory pressure (/proc) and report the highest "
            "tier currently permitted. No deployment dir or docker needed."
        ),
    )
    p.set_defaults(func=cmd_status)
