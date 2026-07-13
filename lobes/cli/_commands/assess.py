"""``lobes assess`` — correctness probes against the served model.

Read-only. Runs the two fixed correctness probes and detects the reasoning-trace
field, then emits a markdown block (plus host-side facts) ready to paste into a
per-model doc under ``docs/``. ``--tools`` additionally probes OpenAI tool
calling. ``--preserve-thinking`` swaps in the two-turn ``preserve_thinking``
token-delta diagnostic (issue #93) instead of the correctness probes: it re-sends
the assistant reasoning trace in history and reports how many extra
``prompt_tokens`` that costs — a positive delta proves the trace survives across
turns. ``--probes`` swaps in the per-role CORRECTNESS probes (issue #81, t7):
cortex/embedder/reranker are each probed on their OWN endpoint (resolved via
the same role registry ``lobes capabilities``/``lobes measure`` use) for a
SEMANTIC answer, not just ``/health`` — a role that is healthy but wrong FAILS
its probe. Throughput lives in ``lobes benchmark``; RUNTIME-only metrics
(latency/throughput/RTF, never correctness) live in ``lobes measure``.
"""

from __future__ import annotations

import argparse

from lobes import assess as _assess
from lobes.cli import _runtime_ops
from lobes.cli._output import emit_result
from lobes.roles import role_registry_from_env
from lobes.runtime import _compose, _env


def cmd_assess(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    model = args.model
    if model is None and deploy_dir is not None:
        model = _env.read_env(deploy_dir / _compose.ENV_FILE, "VLLM_SERVED_NAME")

    url = f"http://localhost:{port}"

    # --probes is a standalone read-only diagnostic (issue #81, t7): per-role
    # CORRECTNESS probes for cortex/embedder/reranker, each resolved to its own
    # endpoint via the shared role registry (same builder `lobes capabilities`/
    # `lobes measure` use) rather than the single `url` above. No host facts /
    # single-model correctness probes are needed.
    if bool(getattr(args, "probes", False)):
        env = _runtime_ops.deployment_env_soft(args)
        registry = role_registry_from_env(env, gateway_url=url)
        roles = (args.role,) if getattr(args, "role", None) else _assess.PROBE_ROLES
        timeout = float(getattr(args, "timeout", None) or _assess.DEFAULT_PROBE_TIMEOUT)
        endpoints = {
            role: (info.endpoint, info.model) if info.loaded and info.endpoint else None
            for role, info in registry.items()
        }
        results = _assess.run_role_probes(endpoints, roles=roles, timeout=timeout)
        passed = all(r["ok"] for r in results.values())
        if json_mode:
            emit_result({"passed": passed, "probes": results}, json_mode=True)
        else:
            emit_result(_assess.render_role_probes(results), json_mode=False)
        return 0

    # --preserve-thinking is a standalone read-only diagnostic (issue #93): it
    # runs the two-turn token-delta probe and reports both prompt-token counts +
    # the delta. No host facts / correctness probes are needed.
    if bool(getattr(args, "preserve_thinking", False)):
        pt = _assess.run_preserve_thinking_probe(url, model)
        emit_result(pt if json_mode else _assess.render_preserve_thinking(pt), json_mode=json_mode)
    else:
        result = _assess.run_correctness(
            url, model, check_tools=bool(getattr(args, "tools", False))
        )
        host = {"image": _compose.container_image(), "gpu_memory": _compose.gpu_engine_mem()}
        if json_mode:
            emit_result({**result, "host": host}, json_mode=True)
        else:
            header = (
                "### Host-side\n"
                f"- Image: `{host['image']}`  ·  GPU memory (EngineCore): {host['gpu_memory']}\n"
            )
            emit_result(header + "\n" + _assess.render_correctness(result), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "assess",
        help="Correctness probes against the served model (markdown for a per-model doc).",
    )
    p.add_argument("--port", type=int, help="Host port (default: VLLM_PORT in .env, else 8000).")
    p.add_argument(
        "--model", help="Served model name (default: VLLM_SERVED_NAME, else first /v1/models)."
    )
    p.add_argument(
        "--tools",
        action="store_true",
        help="Also probe OpenAI tool calling (tool_choice:auto must return a tool_calls array).",
    )
    p.add_argument(
        "--preserve-thinking",
        action="store_true",
        help=(
            "Run the two-turn preserve_thinking token-delta diagnostic instead: "
            "re-send the assistant reasoning trace in history and report the "
            "prompt-token cost (a positive delta proves the trace survives)."
        ),
    )
    p.add_argument(
        "--probes",
        action="store_true",
        help=(
            "Run per-role CORRECTNESS probes (cortex/embedder/reranker) instead: "
            "each role is probed on its own endpoint for a semantic answer, not "
            "just /health — a role that is healthy but wrong FAILS its probe."
        ),
    )
    p.add_argument(
        "--role",
        choices=_assess.PROBE_ROLES,
        help="With --probes, run only one role's probe (default: all three).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "With --probes, per-probe hard timeout in seconds "
            f"(default {_assess.DEFAULT_PROBE_TIMEOUT})."
        ),
    )
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_assess)
