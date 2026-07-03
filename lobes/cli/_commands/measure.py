"""``lobes measure`` — read-only per-role RUNTIME measurement (issue #81, t8).

Probes each of the six first-class roles (``cortex``/``senses``/``embedder``/
``reranker``/``stt``/``tts``) on its own live endpoint and reports **runtime**
metrics, organised BY ROLE: TTFT/decode-tps/prefill-tps/context(+mem, when
cheaply available) for the LLM roles, requests-or-docs-per-sec/latency/batch/
loaded for the pooling roles, RTF/latency/duration/failure-rate for the audio
overlay roles. The actual probing logic lives in :mod:`lobes.roles_measure`;
this module is the CLI wiring (deployment resolution + ``--json``/table
rendering), the same split ``lobes capabilities`` uses over :mod:`lobes.roles`.

RUNTIME-ONLY (boundary c7/h14): every metric emitted here is a serving/runtime
measurement. **No field asserts answer correctness, task quality, or
agent-task success** — that judgment is Colleague's job, not lobes'. See
:mod:`lobes.roles_measure` for the full contract and the closed metric
vocabulary (:data:`lobes.roles_measure.ALLOWED_METRIC_KEYS`).

Read-only: resolves the deployment the same soft way ``lobes capabilities``
does (:func:`lobes.cli._runtime_ops.deployment_env_soft` +
:func:`lobes.roles.role_registry_from_env`) and probes each loaded role's live
endpoint with a short timeout (GET/POST only) — never touches docker/compose,
no ``--apply``.
"""

from __future__ import annotations

import argparse

from lobes.cli import _runtime_ops
from lobes.cli._output import emit_result
from lobes.roles import ROLES, RoleInfo, role_registry_from_env
from lobes.roles_measure import DEFAULT_TIMEOUT, measure_registry

_JSON_HELP = "Emit structured JSON."
_COMPOSE_DIR_HELP = "Deployment dir (default: $LOBES_DIR or ~/.lobes)."
_PORT_HELP = "Gateway host port (default: VLLM_PORT in .env)."


def _registry(args: argparse.Namespace) -> dict[str, RoleInfo]:
    env = _runtime_ops.deployment_env_soft(args)
    port, _ = _runtime_ops.resolve_port_soft(args)
    gateway_url = f"http://localhost:{port}"
    return role_registry_from_env(env, gateway_url=gateway_url)


def _resolve_timeout(args: argparse.Namespace) -> float:
    raw = getattr(args, "timeout", None)
    return float(raw) if raw else DEFAULT_TIMEOUT


def _resolve_roles(args: argparse.Namespace) -> tuple[str, ...]:
    role = getattr(args, "role", None)
    return (role,) if role else ROLES


def _render_table(results: dict[str, dict], roles: tuple[str, ...]) -> str:
    header = f"{'role':<9} {'family':<12} ready  metrics"
    lines = [header, "-" * len(header)]
    for role in roles:
        r = results[role]
        metrics = (
            ", ".join(f"{k}={v}" for k, v in r["metrics"].items() if v is not None)
            or "(unavailable)"
        )
        lines.append(
            f"{r['role']:<9} {r['family']:<12} {'yes' if r['ready'] else 'no ':<6} {metrics}"
        )
    return "\n".join(lines)


def cmd_measure(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    roles = _resolve_roles(args)
    registry = _registry(args)
    timeout = _resolve_timeout(args)
    results = measure_registry(registry, roles=roles, timeout=timeout)

    if json_mode:
        emit_result(results, json_mode=True)
    else:
        emit_result(_render_table(results, roles), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "measure",
        help="Read-only: per-role RUNTIME metrics (latency/throughput/RTF/mem/"
        "readiness), organized by role — cortex/senses/embedder/reranker/stt/tts "
        "(issue #81).",
    )
    p.add_argument("--role", choices=ROLES, help="Measure a single role instead of all six.")
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Per-probe timeout in seconds (default {DEFAULT_TIMEOUT}).",
    )
    p.add_argument("--port", type=int, help=_PORT_HELP)
    p.add_argument("--compose-dir", help=_COMPOSE_DIR_HELP)
    p.add_argument("--json", action="store_true", help=_JSON_HELP)
    p.set_defaults(func=cmd_measure)
