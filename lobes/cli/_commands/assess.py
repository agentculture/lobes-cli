"""``lobes assess`` — correctness probes against the served model.

Read-only. Runs the two fixed correctness probes and detects the reasoning-trace
field, then emits a markdown block (plus host-side facts) ready to paste into a
per-model doc under ``docs/``. ``--tools`` additionally probes OpenAI tool
calling. Throughput lives in ``lobes benchmark``.
"""

from __future__ import annotations

import argparse

from lobes import assess as _assess
from lobes.cli import _runtime_ops
from lobes.cli._output import emit_result
from lobes.runtime import _compose, _env


def cmd_assess(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    model = args.model
    if model is None and deploy_dir is not None:
        model = _env.read_env(deploy_dir / _compose.ENV_FILE, "VLLM_SERVED_NAME")

    url = f"http://localhost:{port}"
    result = _assess.run_correctness(url, model, check_tools=bool(getattr(args, "tools", False)))
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
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_assess)
