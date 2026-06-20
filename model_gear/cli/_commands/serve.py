"""``model serve`` (alias ``start``) — start the vLLM server.

Mutating: dry-run by default; ``--apply`` runs ``docker compose up -d`` in the
deployment dir, waits for ``/health``, then probes ``tool_choice:"auto"`` to
confirm tool calling is live (``--no-probe`` to skip).
"""

from __future__ import annotations

import argparse

from model_gear.cli import _runtime_ops
from model_gear.cli._output import emit_diagnostic, emit_result
from model_gear.runtime import _compose, _env, _health


def cmd_serve(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)

    if not args.apply:
        if json_mode:
            emit_result(
                {"dry_run": True, "deployment_dir": str(deploy_dir), "port": port},
                json_mode=True,
            )
        else:
            emit_result(
                f"DRY RUN — would run: docker compose up -d in {deploy_dir}, "
                f"then wait for health on :{port}.\nRe-run with --apply to execute.",
                json_mode=False,
            )
    else:
        emit_diagnostic(f">> starting the vLLM server in {deploy_dir}")
        # Ensure the durable-log dir exists (user-owned) before compose bind-mounts it.
        _compose.ensure_log_dir(deploy_dir, _env.read_env(env_path, _compose.LOG_DIR_ENV) or None)
        _runtime_ops.compose_check(_compose.compose_up_detached(deploy_dir), "docker compose up -d")
        _health.wait_health(port)
        result = {"serving": True, "port": port, "deployment_dir": str(deploy_dir)}
        tc = None
        if not args.no_probe:
            served = _env.read_env(env_path, "VLLM_SERVED_NAME") or _env.read_env(
                env_path, "VLLM_MODEL"
            )
            tc = _runtime_ops.probe_tool_calling(port, served)
        result["tool_calling"] = tc
        if json_mode:
            emit_result(result, json_mode=True)
        else:
            out = [f">> serving on :{port}. assess with: model assess --port {port}"]
            if tc is not None:
                out.append(">> " + _runtime_ops.format_tool_probe(tc))
            emit_result("\n".join(out), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "serve",
        aliases=["start"],
        help="Start the vLLM server (dry-run by default; --apply to commit).",
    )
    p.add_argument(
        "--port", type=int, help="Host port for the health wait (default: VLLM_PORT in .env)."
    )
    p.add_argument(
        "--compose-dir", help="Deployment dir (default: $MODEL_GEAR_DIR or ~/.model-gear)."
    )
    p.add_argument("--apply", action="store_true", help="Actually start the server.")
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the post-start tool-calling probe (tool_choice:auto).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_serve)
