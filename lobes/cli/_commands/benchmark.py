"""``model benchmark`` — decode throughput + prefill latency for the served model.

Read-only. The workload shape is the active *purpose*: it defaults to the
configured ``VLLM_PURPOSE`` (so the numbers track the serve config) and can be
overridden with ``--purpose`` or explicit ``--input-len`` / ``--output-len``.
Forces a fixed decode length over a couple of runs and measures a prompt-sized
prefill, then emits a markdown block (plus host-side facts) for a per-model doc
under ``docs/``. Correctness lives in ``model assess``.
"""

from __future__ import annotations

import argparse

from lobes import assess as _assess
from lobes import profiles
from lobes.cli import _runtime_ops
from lobes.cli._output import emit_result
from lobes.runtime import _compose, _env


def _resolve_shape(args, deploy_dir) -> tuple[profiles.WorkloadProfile, int, int]:
    """Resolve the (purpose, input_len, output_len) shape — flag > .env > default."""
    purpose = args.purpose
    if purpose is None and deploy_dir is not None:
        purpose = _env.read_env(
            deploy_dir / _compose.ENV_FILE, "VLLM_PURPOSE", profiles.DEFAULT_PURPOSE
        )
    wl = profiles.workload_profile(purpose or profiles.DEFAULT_PURPOSE)
    input_len = args.input_len if args.input_len is not None else wl.bench_input_len
    output_len = args.output_len if args.output_len is not None else wl.bench_output_len
    return wl, input_len, output_len


def cmd_benchmark(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    model = args.model
    if model is None and deploy_dir is not None:
        model = _env.read_env(deploy_dir / _compose.ENV_FILE, "VLLM_SERVED_NAME")

    wl, input_len, output_len = _resolve_shape(args, deploy_dir)

    url = f"http://localhost:{port}"
    result = _assess.run_benchmark(
        url,
        model,
        purpose=wl.name,
        input_len=input_len,
        output_len=output_len,
        runs=args.runs,
    )
    host = {"image": _compose.container_image(), "gpu_memory": _compose.gpu_engine_mem()}

    if json_mode:
        emit_result({**result, "host": host}, json_mode=True)
    else:
        header = (
            "### Host-side\n"
            f"- Image: `{host['image']}`  ·  GPU memory (EngineCore): {host['gpu_memory']}\n"
        )
        emit_result(header + "\n" + _assess.render_benchmark(result), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "benchmark",
        help="Decode throughput + prefill latency for the served model (markdown for a doc).",
    )
    p.add_argument("--port", type=int, help="Host port (default: VLLM_PORT in .env, else 8000).")
    p.add_argument(
        "--model", help="Served model name (default: VLLM_SERVED_NAME, else first /v1/models)."
    )
    p.add_argument(
        "--purpose",
        choices=[wp.name for wp in profiles.WORKLOAD_PROFILES],
        default=None,
        help="Workload shape (default: the configured VLLM_PURPOSE, else balanced).",
    )
    p.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Override prompt length (default: the purpose's shape).",
    )
    p.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Override forced decode length (default: the purpose's shape).",
    )
    p.add_argument("--runs", type=int, default=2, help="Decode-throughput repetitions (default 2).")
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_benchmark)
