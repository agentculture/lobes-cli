"""``model switch <model>`` — change the served vLLM model.

Mutating: dry-run by default (prints the plan, changes nothing); ``--apply``
writes the five ``VLLM_*`` vars to ``.env`` (plus ``VLLM_TOOL_CALL_PARSER`` when
``--tool-call-parser`` is given) then recreates the container
(``docker compose down && up -d``) and waits for ``/health``.
"""

from __future__ import annotations

import argparse

from model_gear.cli import _runtime_ops
from model_gear.cli._output import emit_diagnostic, emit_result
from model_gear.runtime import _compose, _env, _health


def cmd_switch(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)
    served = args.served_name or args.model
    plan = {
        "VLLM_MODEL": args.model,
        "VLLM_SERVED_NAME": served,
        "VLLM_PORT": str(port),
        "VLLM_MAX_MODEL_LEN": str(args.max_model_len),
        "VLLM_GPU_MEM_UTIL": str(args.gpu_mem_util),
    }
    # Only write the tool-call parser when explicitly chosen, so a switch that
    # just retunes (port/len/mem) doesn't clobber a previously-set parser.
    if args.tool_call_parser:
        plan["VLLM_TOOL_CALL_PARSER"] = args.tool_call_parser

    if not args.apply:
        if json_mode:
            emit_result(
                {"dry_run": True, "deployment_dir": str(deploy_dir), "env": plan},
                json_mode=True,
            )
        else:
            lines = [f"DRY RUN — would update {env_path}:"]
            lines += [f"  {k}={v}" for k, v in plan.items()]
            lines.append(
                f"  then: docker compose down && up -d in {deploy_dir}, wait for health on :{port}"
            )
            lines.append("Re-run with --apply to execute.")
            emit_result("\n".join(lines), json_mode=False)
    else:
        emit_diagnostic(
            f">> switching to {args.model} (port={port} max_model_len={args.max_model_len} "
            f"served-name={served} gpu-mem-util={args.gpu_mem_util})"
        )
        for key, value in plan.items():
            _env.set_env(env_path, key, value)
        _runtime_ops.compose_check(_compose.compose_down(deploy_dir), "docker compose down")
        _runtime_ops.compose_check(_compose.compose_up_detached(deploy_dir), "docker compose up -d")
        _health.wait_health(port)
        result = {
            "switched": args.model,
            "served_name": served,
            "port": port,
            "deployment_dir": str(deploy_dir),
        }
        if json_mode:
            emit_result(result, json_mode=True)
        else:
            emit_result(f">> done. assess with: model assess --port {port}", json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "switch",
        help="Switch the served vLLM model (dry-run by default; --apply to commit).",
    )
    p.add_argument("model", help="Model to serve, e.g. nvidia/Qwen3-32B-NVFP4.")
    p.add_argument("--port", type=int, help="Host port (default: VLLM_PORT in .env, else 8000).")
    p.add_argument(
        "--max-model-len", type=int, default=32768, help="Context window (default 32768)."
    )
    p.add_argument("--served-name", help="Name clients address (default: the model name).")
    p.add_argument(
        "--gpu-mem-util", type=float, default=0.6, help="GPU memory fraction (default 0.6)."
    )
    p.add_argument(
        "--tool-call-parser",
        help="OpenAI tool-call parser (e.g. hermes for Qwen3 dense, qwen3_coder for "
        "Qwen3-Coder/3.6). Only written when given; leaves VLLM_TOOL_CALL_PARSER otherwise.",
    )
    p.add_argument(
        "--compose-dir", help="Deployment dir (default: $MODEL_GEAR_DIR or ~/.model-gear)."
    )
    p.add_argument(
        "--apply", action="store_true", help="Commit the switch (recreate the container)."
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_switch)
