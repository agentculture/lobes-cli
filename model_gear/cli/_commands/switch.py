"""``model switch <model>`` — change the served vLLM model (and its gear).

Mutating: dry-run by default (prints the plan, changes nothing); ``--apply``
writes the resolved ``VLLM_*`` vars to ``.env``, recreates the container
(``docker compose down && up -d``), waits for ``/health``, and then probes
``tool_choice:"auto"`` to confirm tool calling survived the switch (``--no-probe``
to skip).

The serve config is resolved from three layers (explicit CLI flags win):

* the **machine** profile (``--machine``, default auto-detected) → GPU memory
  fraction, max context, attention backend;
* the **workload** profile (``--purpose``, default ``balanced``) → batching knobs
  (and the shape ``model benchmark`` exercises); and
* the **model** catalog entry → quantization + tool-call parser, plus a printed
  reminder for the MoE-only compose flags (which can't be defaulted safely).
"""

from __future__ import annotations

import argparse
import socket

from model_gear import profiles
from model_gear.catalog import supported_models
from model_gear.cli import _runtime_ops
from model_gear.cli._commands.whoami import _gpu_name
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


def _select_quantization(args: argparse.Namespace) -> tuple[str | None, str]:
    """Return ``(quantization, message)`` so the right vLLM ``--quantization`` is set.

    Quantization is a per-checkpoint property (ModelOpt FP4 vs compressed-tensors),
    so it can't be inferred from the model *family* the way the tool-call parser
    can. An explicit ``--quantization`` wins; otherwise it is read from the catalog
    for a known model. ``None`` means "uncatalogued" — leave the existing
    ``VLLM_QUANTIZATION`` untouched (override it explicitly when needed).
    """
    if args.quantization:
        return args.quantization, f"quantization (explicit): {args.quantization}"
    for model in supported_models():
        if model.id == args.model:
            return model.quantization, f"quantization (from catalog): {model.quantization}"
    return None, "quantization: left unchanged (uncatalogued model; pass --quantization)"


def _serve_notices(model_id: str) -> list[str]:
    """Reminders for catalog serve-extras a model needs as a manual compose edit.

    Some flags can't be defaulted in the single-model template — compose can't
    conditionally omit a flag, and an empty ``--moe-backend=`` / ``--speculative-config=``
    token breaks vLLM — so ``model switch`` surfaces them for a hand edit instead:

    * ``--moe-backend`` for an MoE checkpoint (emitted bare so it pastes verbatim as
      a compose ``command:`` list item); and
    * ``--speculative-config`` (MTP draft) for a checkpoint that ships MTP weights,
      together with the ``--trust-remote-code`` / ``--language-model-only`` flags and
      the ``VLLM_MAX_NUM_SEQS=2`` cap that MTP-grafted text-only build also needs.

    Returns one line per applicable extra (empty list for a plain model).
    """
    notices: list[str] = []
    for model in supported_models():
        if model.id != model_id:
            continue
        if model.moe_backend:
            notices.append(
                "MoE model — add this to the compose `command` by hand (not written "
                "to .env; see docs/qwen3.6-35b-a3b-nvfp4.md): "
                f"--moe-backend={model.moe_backend}"
            )
        if model.speculative_config:
            notices.append(
                "MTP/text-only model — add these to the compose `command` by hand "
                "(not written to .env; see docs/qwen3.6-27b-text-nvfp4-mtp.md): "
                f"--speculative-config '{model.speculative_config}' --trust-remote-code "
                "--language-model-only; and set VLLM_MAX_NUM_SEQS=2 (4 OOMs at n=3/256K)"
            )
    return notices


def _resolve_machine_name(machine_arg: str) -> str:
    """Resolve ``--machine`` to a concrete profile name (``auto`` → detect).

    Only shells out to ``nvidia-smi`` when detection is actually needed (an
    explicit ``--machine`` skips it).
    """
    raw = (machine_arg or "auto").strip().lower()
    gpu = _gpu_name() if raw in ("", "auto") else None
    return profiles.resolve_machine(machine_arg, gpu_name=gpu, hostname=socket.gethostname())


def _build_plan(args: argparse.Namespace, port: int, served: str) -> tuple[dict, list[str]]:
    """Build the ``VLLM_*`` env plan + the human messages (parser/quant lines)."""
    machine = _resolve_machine_name(args.machine)
    serve_cfg = profiles.resolve_serve_config(
        args.purpose,
        machine,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
    )
    plan = {
        "VLLM_MODEL": args.model,
        "VLLM_SERVED_NAME": served,
        "VLLM_PORT": str(port),
        **serve_cfg,
    }
    messages: list[str] = []
    parser, parser_msg = _select_parser(args)
    messages.append(parser_msg)
    if parser:
        plan["VLLM_TOOL_CALL_PARSER"] = parser
    quant, quant_msg = _select_quantization(args)
    messages.append(quant_msg)
    if quant:
        plan["VLLM_QUANTIZATION"] = quant
    return plan, messages


def _emit_dry_run(deploy_dir, env_path, plan, messages, notices, port, probe, json_mode) -> None:
    if json_mode:
        emit_result(
            {
                "dry_run": True,
                "deployment_dir": str(deploy_dir),
                "env": plan,
                "purpose": plan.get("VLLM_PURPOSE"),
                "machine": plan.get("VLLM_MACHINE"),
                "tool_call_parser": plan.get("VLLM_TOOL_CALL_PARSER"),
                "quantization": plan.get("VLLM_QUANTIZATION"),
                "compose_edits": notices,
                "probe": probe,
            },
            json_mode=True,
        )
        return
    lines = [f"DRY RUN — would update {env_path}:"]
    lines += [f"  {k}={v}" for k, v in plan.items()]
    lines += [f"  {m}" for m in messages]
    lines += [f"  NOTE: {notice}" for notice in notices]
    lines.append(
        f"  then: docker compose down && up -d in {deploy_dir}, wait for health on :{port}"
    )
    if probe:
        lines.append("  then: probe tool calling (tool_choice:auto)")
    lines.append("Re-run with --apply to execute.")
    emit_result("\n".join(lines), json_mode=False)


def _apply_switch(
    model, deploy_dir, env_path, plan, messages, notices, port, served, probe, json_mode
):
    emit_diagnostic(
        f">> switching to {model} (port={port} purpose={plan['VLLM_PURPOSE']} "
        f"machine={plan['VLLM_MACHINE']} max_model_len={plan['VLLM_MAX_MODEL_LEN']} "
        f"gpu-mem-util={plan['VLLM_GPU_MEM_UTIL']} served-name={served})"
    )
    for msg in messages:
        emit_diagnostic(f">> {msg}")
    for notice in notices:
        emit_diagnostic(f">> NOTE: {notice}")
    for key, value in plan.items():
        _env.set_env(env_path, key, value)
    _runtime_ops.compose_check(_compose.compose_down(deploy_dir), "docker compose down")
    _runtime_ops.compose_check(_compose.compose_up_detached(deploy_dir), "docker compose up -d")
    _health.wait_health(port)
    tc = None if not probe else _runtime_ops.probe_tool_calling(port, served)
    result = {
        "switched": model,
        "served_name": served,
        "port": port,
        "deployment_dir": str(deploy_dir),
        "purpose": plan["VLLM_PURPOSE"],
        "machine": plan["VLLM_MACHINE"],
        "tool_call_parser": plan.get("VLLM_TOOL_CALL_PARSER"),
        "quantization": plan.get("VLLM_QUANTIZATION"),
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
    plan, messages = _build_plan(args, port, served)
    notices = _serve_notices(args.model)

    if args.apply:
        _apply_switch(
            args.model,
            deploy_dir,
            env_path,
            plan,
            messages,
            notices,
            port,
            served,
            not args.no_probe,
            json_mode,
        )
    else:
        _emit_dry_run(
            deploy_dir, env_path, plan, messages, notices, port, not args.no_probe, json_mode
        )
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "switch",
        help="Switch the served vLLM model (dry-run by default; --apply to commit).",
    )
    p.add_argument("model", help="Model to serve, e.g. mmangkad/Qwen3.6-27B-NVFP4.")
    p.add_argument("--port", type=int, help="Host port (default: VLLM_PORT in .env, else 8000).")
    p.add_argument(
        "--purpose",
        choices=[wp.name for wp in profiles.WORKLOAD_PROFILES],
        default=profiles.DEFAULT_PURPOSE,
        help="Workload profile — tunes batching + the benchmark shape (default balanced).",
    )
    p.add_argument(
        "--machine",
        choices=["auto"] + [mp.name for mp in profiles.MACHINE_PROFILES],
        default=profiles.DEFAULT_MACHINE,
        help="Machine profile — sets GPU mem / context / attention defaults "
        "(default auto: detect from nvidia-smi + hostname).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Context window (default: the machine profile, e.g. 32768 on spark).",
    )
    p.add_argument("--served-name", help="Name clients address (default: the model name).")
    p.add_argument(
        "--gpu-mem-util",
        type=float,
        default=None,
        help="GPU memory fraction (default: the machine profile, e.g. 0.6 on spark).",
    )
    p.add_argument(
        "--tool-call-parser",
        help="OpenAI tool-call parser (hermes for Qwen3 dense, qwen3_coder for "
        "Qwen3-Coder/3.6, mistral for Mistral). Overrides the per-model auto-selection.",
    )
    p.add_argument(
        "--quantization",
        help="vLLM --quantization (modelopt_fp4 for nvidia/mmangkad NVFP4, "
        "compressed-tensors for RedHatAI NVFP4). Overrides the catalog value.",
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
