"""``lobes status`` — read-only snapshot of the current deployment.

Reports the configured model/served-name/port (from ``.env``), the container
lifecycle + health state, and whether ``/health`` is responding. This is the
*configured* served model (from ``.env``) + health — not a live ``/v1/models``
query; for the full supported catalog you can switch to, use ``lobes overview --list``.

``--pressure`` sub-mode (issue #68/#69/#85)
-------------------------------------------
Reads host memory pressure from ``/proc`` (via
:func:`lobes.runtime._pressure.sample_pressure`) and maps it to a busy/serve
decision via :func:`lobes.gateway._pressure_policy.decide` (called with
``requested_tier="main"`` so the result shows what a full-tier request would get
right now).  Under swap/iowait pressure the gateway no longer degrades a
``main``/``senses`` request onto a different model — it **sheds** it with HTTP
429 (busy, retry shortly), issue #85.  So this command reports the *box-level*
``mode`` (``warm`` / ``busy``), whether a full-tier request is being ``shed``,
the ``servable_tier`` that still answers under pressure (``minor`` — the floor —
is always served, never shed), and the ``retry_after`` a busy caller would get.
Because it calls the same :func:`decide` handle_post consults, the reported
decision matches what a live request would receive — without issuing one.  This
path is **strictly read-only**: it touches neither the deployment dir nor the
Docker socket.
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

    Samples live host pressure and reports the busy/serve decision a full-tier
    (``main``) request would receive right now: the box-level ``mode``, whether
    such a request is ``shed`` (HTTP 429), the ``servable_tier`` that still
    answers (``minor`` — the floor — is never shed), and the ``retry_after`` a
    busy caller would get.  Calls the same :func:`decide` handle_post consults,
    so the reported decision matches a live request without issuing one.  No
    deployment dir, no docker compose, no .env reads.
    """
    p = _pressure_mod.sample_pressure()
    d = _pressure_policy.decide(
        p["swap_used_percent"],
        p["iowait_percent"],
        requested_tier="main",
    )
    # Under pressure a full-tier (main / senses) request is shed with 429; the
    # only tier that still serves is ``minor`` (the floor). ``servable_tier`` is
    # "minor" when busy, else "main" — the model resolves that tier (#85).
    servable_tier = d["servable_tier"]
    model = catalog.resolve_tier(servable_tier).id
    shed = d["shed"]
    retry_after = _pressure_policy.BUSY_RETRY_AFTER_SECONDS if shed else None

    result = {
        "mode": d["mode"],
        "shed": shed,
        "servable_tier": servable_tier,
        "model": model,
        "reason": d["reason"],
        "retry_after": retry_after,
        "pressure": p,
    }

    if json_mode:
        emit_result(result, json_mode=True)
    else:
        shed_line = (
            f"shed:    main/cortex + senses requests return 429 busy "
            f"(retry after {retry_after}s)"
            if shed
            else "shed:    none — all tiers served"
        )
        emit_result(
            "\n".join(
                [
                    f"mode:    {result['mode']}",
                    shed_line,
                    f"servable: {result['servable_tier']}",
                    f"model:   {result['model']}",
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

    if _compose.is_fleet(deploy_dir):
        containers = [
            {"name": name, "state": _compose.inspect_state(name)}
            for name in _compose.fleet_containers(deploy_dir)
        ]
        report = {
            "model": _env.read_env(env_path, "VLLM_MODEL", _UNSET),
            "served_name": _env.read_env(env_path, "VLLM_SERVED_NAME", _UNSET),
            "port": port,
            "tool_call_parser": _env.read_env(env_path, "VLLM_TOOL_CALL_PARSER", _UNSET),
            "deployment_dir": str(deploy_dir),
            "deployment": "fleet",
            "containers": containers,
            "health": "ok" if _health.is_healthy(port) else "not responding",
        }

        if json_mode:
            emit_result(report, json_mode=True)
        else:
            lines = [
                f"model:  {report['model']}",
                f"served: {report['served_name']}  port: {report['port']}",
                f"parser: {report['tool_call_parser']}",
                f"dir:    {report['deployment_dir']}",
            ]
            for c in containers:
                lines.append(f"  {c['name']} — {c['state']}")
            lines.append(f"health: {report['health']} (:{port})")
            lines.append(
                "see 'lobes fleet status' / 'lobes capabilities' for the full fleet/role view"
            )
            emit_result("\n".join(lines), json_mode=False)
    else:
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
