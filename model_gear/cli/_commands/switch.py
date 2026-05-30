"""``model switch <model>`` — change the served vLLM model.

Mutating: dry-run by default (prints the plan, changes nothing); ``--apply``
writes the five ``VLLM_*`` vars to ``.env`` (plus an auto-selected, or
``--tool-call-parser``-overridden, ``VLLM_TOOL_CALL_PARSER``), recreates the
container (``docker compose down && up -d``), waits for ``/health``, and then
probes ``tool_choice:"auto"`` to confirm tool calling survived the switch
(``--no-probe`` to skip).
"""

from __future__ import annotations

import argparse

from model_gear.cli import _runtime_ops
from model_gear.cli._output import emit_diagnostic, emit_result
from model_gear.runtime import _compose, _env, _health, _parser


def _select_parser(args: argparse.Namespace) -> tuple[str | None, str]:
    """Return ``(parser, message)`` so tool calling keeps working across a switch.

    An explicit ``--tool-call-parser`` wins; otherwise infer one from the model
    name. ``None`` means "unknown model" — leave the existing
    ``VLLM_TOOL_CALL_PARSER`` untouched (override it explicitly when needed).
    """
    if args.tool_call_parser:
        return args.tool_call_parser, f"tool-call parser (explicit): {args.tool_call_parser}"
    inferred = _parser.infer_parser(args.model)
    if inferred:
        return inferred, f"tool-call parser (auto-selected): {inferred}"
    return None, "tool-call parser: left unchanged (unknown model; pass --tool-call-parser)"


def _emit_dry_run(args, deploy_dir, env_path, plan, parser, parser_msg, port, json_mode) -> None:
    if json_mode:
        emit_result(
            {
                "dry_run": True,
                "deployment_dir": str(deploy_dir),
                "env": plan,
                "tool_call_parser": parser,
                "probe": not args.no_probe,
            },
            json_mode=True,
        )
        return
    lines = [f"DRY RUN — would update {env_path}:"]
    lines += [f"  {k}={v}" for k, v in plan.items()]
    lines.append(f"  {parser_msg}")
    lines.append(
        f"  then: docker compose down && up -d in {deploy_dir}, wait for health on :{port}"
    )
    if not args.no_probe:
        lines.append("  then: probe tool calling (tool_choice:auto)")
    lines.append("Re-run with --apply to execute.")
    emit_result("\n".join(lines), json_mode=False)


def _apply_switch(args, deploy_dir, env_path, plan, parser, parser_msg, port, served, json_mode):
    emit_diagnostic(
        f">> switching to {args.model} (port={port} max_model_len={args.max_model_len} "
        f"served-name={served} gpu-mem-util={args.gpu_mem_util})"
    )
    emit_diagnostic(f">> {parser_msg}")
    for key, value in plan.items():
        _env.set_env(env_path, key, value)
    _runtime_ops.compose_check(_compose.compose_down(deploy_dir), "docker compose down")
    _runtime_ops.compose_check(_compose.compose_up_detached(deploy_dir), "docker compose up -d")
    _health.wait_health(port)
    tc = None if args.no_probe else _runtime_ops.probe_tool_calling(port, served)
    result = {
        "switched": args.model,
        "served_name": served,
        "port": port,
        "deployment_dir": str(deploy_dir),
        "tool_call_parser": parser,
        "tool_calling": tc,
    }
    if json_mode:
        emit_result(result, json_mode=True)
        return
    out = [f">> done. assess with: model assess --port {port}"]
    if tc is not None:
        out.append(">> " + _runtime_ops.format_tool_probe(tc))
    emit_result("\n".join(out), json_mode=False)


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
    parser, parser_msg = _select_parser(args)
    if parser:
        plan["VLLM_TOOL_CALL_PARSER"] = parser

    if args.apply:
        _apply_switch(args, deploy_dir, env_path, plan, parser, parser_msg, port, served, json_mode)
    else:
        _emit_dry_run(args, deploy_dir, env_path, plan, parser, parser_msg, port, json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "switch",
        help="Switch the served vLLM model (dry-run by default; --apply to commit).",
    )
    p.add_argument("model", help="Model to serve, e.g. mmangkad/Qwen3.6-27B-NVFP4.")
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
        help="OpenAI tool-call parser (hermes for Qwen3 dense, qwen3_coder for "
        "Qwen3-Coder/3.6). Overrides the per-model auto-selection.",
    )
    p.add_argument(
        "--compose-dir", help="Deployment dir (default: $MODEL_GEAR_DIR or ~/.model-gear)."
    )
    p.add_argument(
        "--apply", action="store_true", help="Commit the switch (recreate the container)."
    )
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the post-switch tool-calling probe (tool_choice:auto).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_switch)
