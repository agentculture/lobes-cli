"""``lobes switch <model>`` — change the served vLLM model (and its gear).

Mutating: dry-run by default (prints the plan, changes nothing); ``--apply``
writes the resolved ``VLLM_*`` vars to ``.env``, recreates the container
(``docker compose down && up -d``), waits for ``/health``, and then probes
``tool_choice:"auto"`` to confirm tool calling survived the switch (``--no-probe``
to skip).

The serve config is resolved from three layers (explicit CLI flags win):

* the **machine** profile (``--machine``, default auto-detected) → GPU memory
  fraction, max context, attention backend;
* the **workload** profile (``--purpose``, default ``balanced``) → batching knobs
  (and the shape ``lobes benchmark`` exercises); and
* the **model** catalog entry → quantization + tool-call parser, plus a printed
  reminder for the MoE-only compose flags (which can't be defaulted safely).

For embed/score gears (``--task embed`` / ``--task score``, or auto-detected from
the catalog), the plan skips tool-call parser and caps ``VLLM_MAX_MODEL_LEN`` to
8192 and ``VLLM_GPU_MEM_UTIL`` to 0.06 by default (tiny footprint for a pooling
model; 0.025 OOMs the cache blocks — load-tested 2026-06-19). The post-switch
tool-calling probe is skipped. ``--task`` is the friendly surface; on this vLLM
build the actual serve flags are ``--runner pooling`` + ``--convert {embed,
classify}`` (the notice prints them — the single-model template can't default them).
"""

from __future__ import annotations

import argparse
import socket

from lobes import profiles
from lobes.catalog import mtp_compose_command_items, supported_models
from lobes.cli import _runtime_ops
from lobes.cli._commands.whoami import _gpu_name
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.runtime import _compose, _env, _health, _parser


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

    The special sentinel value ``"none"`` (catalog or explicit flag) signals
    bf16/unquantized: ``None`` is returned so ``VLLM_QUANTIZATION`` is not written.
    The single-model template defaults to ``--quantization=modelopt`` when
    ``VLLM_QUANTIZATION`` is absent — a compose-edit notice fires separately (via
    ``_bf16_none_notice``) telling the operator to remove that line by hand.
    """
    _UNQUANTIZED = "none"
    if args.quantization:
        if args.quantization == _UNQUANTIZED:
            return None, "quantization: none (bf16/unquantized; --quantization omitted)"
        return args.quantization, f"quantization (explicit): {args.quantization}"
    for model in supported_models():
        if model.id == args.model:
            if model.quantization == _UNQUANTIZED:
                return None, "quantization: none (bf16/unquantized; --quantization omitted)"
            return model.quantization, f"quantization (from catalog): {model.quantization}"
    return None, "quantization: left unchanged (uncatalogued model; pass --quantization)"


def _resolve_task(args: argparse.Namespace) -> str:
    """Return the effective vLLM task for this switch.

    An explicit ``--task`` always wins — the flag defaults to ``None`` (not
    ``"generate"``) so an explicit ``--task generate`` is distinguishable from
    "flag omitted" and can override a catalogued embed/score model. When the flag
    was omitted (``None``), the catalog is consulted: embed/score models declare
    their task there and it is inferred automatically — so ``lobes switch
    Qwen/Qwen3-Embedding-0.6B`` auto-detects ``task=embed`` without the flag.
    """
    if args.task is not None:
        # Operator provided an explicit --task; honour it unconditionally.
        return args.task
    for model in supported_models():
        if model.id == args.model and model.task != "generate":
            return model.task
    return "generate"


def _mtp_primary():
    """The MTP default primary (the model whose serve flags the template bakes in)."""
    return next(
        (m for m in supported_models() if m.role_hint == "primary" and m.speculative_config),
        None,
    )


def _mtp_removal_notice(model) -> str | None:
    """Non-primary target: the template's baked MTP flags must be removed by hand.

    The item list is the single source of truth in the catalog (so it can't drift
    from the templates); each renders as a YAML ``command:`` list item —
    ``--speculative-config`` is single-quoted because its JSON value has ``: ``/``{``.

    Gated on the target being the default MTP **primary** specifically (by id),
    not merely on ``model.speculative_config`` being non-empty. A different gear
    can carry its own, incompatible ``speculative_config`` (e.g. the Gemma 4 12B
    multimodal entry's native ``mtp`` + assistant-draft config) — the single-model
    template only bakes in the Qwen MTP primary's exact flags
    (``qwen3_5_mtp`` + ``--language-model-only`` + ``--tokenizer=...``), so
    switching to any other speculative-config-carrying model still needs the
    manual compose edit; only the actual primary's flags match the template as
    shipped (Qodo #80).
    """
    primary = _mtp_primary()
    if primary is not None and model.id == primary.id:
        return None
    rendered = "".join(
        (f"\n      - '{item}'" if item.startswith("--speculative-config=") else f"\n      - {item}")
        for item in mtp_compose_command_items()
    )
    return (
        "non-MTP model — the template ships the MTP default primary's flags; "
        "REMOVE these `command:` list items by hand to serve this model (see "
        "docs/qwen3.6-27b-text-nvfp4-mtp.md):" + rendered
    )


def _moe_notice(model) -> str | None:
    """MoE checkpoint: ``--moe-backend`` must be added by hand (not written to .env)."""
    if not model.moe_backend:
        return None
    return (
        "MoE model — add this to the compose `command` by hand (not written "
        "to .env; see docs/qwen3.6-35b-a3b-nvfp4.md): "
        f"--moe-backend={model.moe_backend}"
    )


def _pooling_notice(model) -> str | None:
    """Embed/score gear: the turnkey path is the fleet; solo serving on the
    single-model template needs the full pooling add + chat/MTP-flag removal.

    This vLLM build (0.19.0+nv26.04) serves pooling models with
    ``--runner pooling`` + ``--convert {embed,classify}`` (the old ``--task`` is
    rejected). ``--hf-overrides`` is single-quoted because its JSON value has
    ``: ``/``{``/``[`` (unquoted, ``docker compose`` fails to parse it).

    The embed lane holds MORE than one gear, so the service name is resolved per
    model, not per task. Naming ``vllm-embed`` for every embed model would point
    an operator switching to the deep gear at the HOT-PATH service — i.e. tell
    them to replace the 0.6B in place, silently invalidating every vector in an
    index built with it (the two models occupy different vector spaces; see
    docs/qwen3-embedding-4b.md). ``role_hint == "embedding"`` IS the definition of
    the role-default gear — pinned by
    tests/test_catalog.py::test_exactly_one_embed_model_carries_the_embedding_role_hint
    — so any other embed-task entry is by construction a non-default gear and
    belongs to the deep slot.
    """
    if model.task not in ("embed", "score"):
        return None
    convert = "embed" if model.task == "embed" else "classify"
    if model.task == "score":
        service = "vllm-rerank"
    elif model.role_hint == "embedding":
        service = "vllm-embed"
    else:
        service = "vllm-embed-deep"
    return (
        "embed/score gear — the TURNKEY path is the fleet: `lobes init --fleet` "
        f"+ `lobes fleet up` serves it via the dedicated {service} service "
        "(already task-aware). To solo-serve on the single-model template "
        "instead, ADD these `command:` list items —"
        "\n      - --runner=pooling"
        f"\n      - --convert={convert}"
        f"\n      - '--hf-overrides={model.hf_overrides}'"
        "\n    and REMOVE the chat/MTP flags the single-model template bakes in "
        "(they break or are ignored by a pooling model): --quantization, "
        "--reasoning-parser, --enable-auto-tool-choice, --tool-call-parser, and "
        "the 4 MTP lines (--speculative-config / --trust-remote-code / "
        "--language-model-only / --tokenizer). VLLM_TASK in .env is a record only; "
        "the single-model template does not consume it."
    )


def _bf16_none_notice(model) -> str | None:
    """bf16/unquantized gear: the ``--quantization`` compose line must be removed.

    The single-model template hardcodes ``--quantization=${VLLM_QUANTIZATION:-modelopt}``
    — even when ``VLLM_QUANTIZATION`` is absent from ``.env``, the default silently
    applies ModelOpt post-processing, which would corrupt bf16/unquantized weights.
    ``lobes switch`` does not write ``VLLM_QUANTIZATION`` for a ``quantization="none"``
    gear; the operator must REMOVE the ``--quantization`` line from the compose
    ``command:`` by hand. See docs/qwen3.5-4b-minor.md.
    """
    return _BF16_NONE_NOTICE if model.quantization == "none" else None


_BF16_NONE_NOTICE = (
    "bf16/unquantized model (quantization=none) — REMOVE the --quantization line "
    "from the compose `command:` by hand: the template defaults to "
    "--quantization=modelopt when VLLM_QUANTIZATION is absent, which would corrupt "
    "bf16 weights. See docs/qwen3.5-4b-minor.md."
)


def _serve_notices(model_id: str, args: argparse.Namespace | None = None) -> list[str]:
    """Reminders for compose ``command:`` edits a switch implies.

    The catalog-keyed edits (non-MTP flag removal, MoE backend add, embed/score
    pooling serve) fire only for a catalogued model. The bf16/none quantization
    removal fires on the **effective** quantization choice — a catalogued
    ``quantization="none"`` gear *or* an explicit ``--quantization none`` (even
    for an uncatalogued model), matching ``_select_quantization``'s contract.
    """
    model = next((m for m in supported_models() if m.id == model_id), None)
    candidates: list[str | None] = []
    if model is not None:
        candidates += [
            _mtp_removal_notice(model),
            _moe_notice(model),
            _pooling_notice(model),
        ]
    # bf16/none keys off the effective choice, not solely catalog metadata.
    effective_none = (getattr(args, "quantization", None) == "none") or (
        model is not None and model.quantization == "none"
    )
    if effective_none:
        candidates.append(_BF16_NONE_NOTICE)
    return [notice for notice in candidates if notice]


def _resolve_machine_name(machine_arg: str) -> str:
    """Resolve ``--machine`` to a concrete profile name (``auto`` → detect).

    Only shells out to ``nvidia-smi`` when detection is actually needed (an
    explicit ``--machine`` skips it).
    """
    raw = (machine_arg or "auto").strip().lower()
    gpu = _gpu_name() if raw in ("", "auto") else None
    return profiles.resolve_machine(machine_arg, gpu_name=gpu, hostname=socket.gethostname())


POOLING_DEFAULT_UTIL = 0.06
"""Shared embed/score budget — sized for the ~0.6B gears, NOT for every pooling model.

A larger pooling gear declares its own via ``SupportedModel.default_gpu_mem_util``;
this value is only the fallback for models that don't. Handing the 4B embedder this
default would under-provision it below its own weights (7.56 GiB of weights against a
0.06 x 121.69 = 7.30 GiB budget — it cannot load), which is exactly why the per-model
override exists.
"""


def _pooling_default_util(model_id: str) -> float:
    """The pooling budget for ``model_id`` — its own if catalogued, else the shared one."""
    catalogued = next((m for m in supported_models() if m.id == model_id), None)
    if catalogued is not None and catalogued.default_gpu_mem_util > 0:
        return catalogued.default_gpu_mem_util
    return POOLING_DEFAULT_UTIL


def _serve_cfg(args: argparse.Namespace, machine: str, is_pooling: bool) -> dict:
    """Resolve the ``VLLM_*`` serve config from the purpose/machine profiles.

    Embed/score gears default to a tiny KV cache (8192 context) and a per-model
    pooling budget (:func:`_pooling_default_util` — the catalogued
    ``default_gpu_mem_util`` when the model declares one, else
    :data:`POOLING_DEFAULT_UTIL`); an explicit ``--max-model-len`` /
    ``--gpu-mem-util`` still wins. A chat model passes ``None`` so the machine
    profile's own defaults apply.
    """
    default_mml = 8192 if is_pooling else None
    default_util = _pooling_default_util(args.model) if is_pooling else None
    max_model_len = args.max_model_len if args.max_model_len is not None else default_mml
    gpu_mem_util = args.gpu_mem_util if args.gpu_mem_util is not None else default_util
    return profiles.resolve_serve_config(
        args.purpose, machine, max_model_len=max_model_len, gpu_mem_util=gpu_mem_util
    )


def _forced_task_warning(args: argparse.Namespace, effective_task: str) -> str | None:
    """Warn when an explicit ``--task`` is forced on a model the catalog does not
    declare as that task — pooling serve defaults apply but no compose-edit notice
    fires (the notice keys off the catalog task). Catalogued auto-detected gears
    are unaffected (``args.task`` is ``None`` for them)."""
    if args.task is None:
        return None
    catalogued = next((m for m in supported_models() if m.id == args.model), None)
    if catalogued is not None and catalogued.task == effective_task:
        return None
    convert = "embed" if effective_task == "embed" else "classify"
    return (
        f"WARNING: --task={effective_task} was forced but the catalog does not "
        f"declare {args.model} as a {effective_task} model — serving with pooling "
        f"defaults, but you must add `--runner pooling --convert {convert}` + the "
        "model's `--hf-overrides` to the compose `command:` by hand (no notice is "
        "emitted for an uncatalogued task)."
    )


def _task_messages(
    plan: dict, args: argparse.Namespace, effective_task: str, is_pooling: bool
) -> list[str]:
    """Set ``VLLM_TASK`` / the tool-call parser; return the human messages."""
    if not is_pooling:
        parser, parser_msg = _select_parser(args)
        if parser:
            plan["VLLM_TOOL_CALL_PARSER"] = parser
        return [parser_msg]
    # Embed/score models don't use tool calling — skip the parser entirely.
    plan["VLLM_TASK"] = effective_task
    messages = [f"tool-call parser: skipped (task={effective_task})"]
    warning = _forced_task_warning(args, effective_task)
    if warning:
        messages.append(warning)
    return messages


def _quant_messages(
    plan: dict, args: argparse.Namespace, effective_task: str, is_pooling: bool
) -> list[str]:
    """Set ``VLLM_QUANTIZATION``; embed/score gears with an empty catalog
    quantization stay unquantized (no flag written)."""
    quant, quant_msg = _select_quantization(args)
    if is_pooling and quant == "":
        return [f"quantization: none (task={effective_task})"]
    if quant:
        plan["VLLM_QUANTIZATION"] = quant
    return [quant_msg]


def _context_messages(plan: dict, args: argparse.Namespace, is_pooling: bool) -> list[str]:
    """Pooling: note the embed/score context defaults. Chat: clamp ``--max-model-len``
    DOWN to the catalogued native ceiling — vLLM refuses a larger value (no YaRN) and
    the container fails to boot, so a high machine default (spark's 256K) can't
    boot-fail a 32K-native model. An uncatalogued model has no ceiling to clamp
    against, so it inherits the machine default with a warning."""
    if is_pooling:
        messages = []
        if args.max_model_len is None:
            messages.append("max-model-len (embed/score default): 8192")
        if args.gpu_mem_util is None:
            messages.append(
                f"gpu-mem-util (embed/score default): {_pooling_default_util(args.model)}"
            )
        return messages
    if args.max_model_len is not None:
        return []
    catalogued = next((m for m in supported_models() if m.id == args.model), None)
    if catalogued is None:
        return [
            f"max-model-len: machine default {plan['VLLM_MAX_MODEL_LEN']} applied "
            "unclamped (uncatalogued model — native ceiling unknown); if the "
            "checkpoint's native context is smaller, vLLM will refuse to boot — pass "
            "--max-model-len or add the model to the catalog with native_max_model_len"
        ]
    if int(plan["VLLM_MAX_MODEL_LEN"]) <= catalogued.native_max_model_len:
        return []
    plan["VLLM_MAX_MODEL_LEN"] = str(catalogued.native_max_model_len)
    return [f"max-model-len (clamped to model native ceiling): {catalogued.native_max_model_len}"]


def _mtp_cap_messages(plan: dict, args: argparse.Namespace) -> list[str]:
    """The MTP primary caps decode slots at 2 (the balanced profile's 4 OOMs at high
    context with n=3 spec-decode); force it over the profile."""
    primary = _mtp_primary()
    if primary and args.model == primary.id:
        plan["VLLM_MAX_NUM_SEQS"] = "2"
        return ["max-num-seqs (MTP primary cap): 2"]
    return []


def _build_plan(args: argparse.Namespace, port: int, served: str) -> tuple[dict, list[str]]:
    """Build the ``VLLM_*`` env plan + the human messages (one helper per concern:
    serve config, task/parser, quantization, context clamp, MTP cap)."""
    machine = _resolve_machine_name(args.machine)
    effective_task = _resolve_task(args)
    is_pooling = effective_task != "generate"
    plan = {
        "VLLM_MODEL": args.model,
        "VLLM_SERVED_NAME": served,
        "VLLM_PORT": str(port),
        **_serve_cfg(args, machine, is_pooling),
    }
    messages = [
        *_task_messages(plan, args, effective_task, is_pooling),
        *_quant_messages(plan, args, effective_task, is_pooling),
        *_context_messages(plan, args, is_pooling),
        *_mtp_cap_messages(plan, args),
    ]
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
    out = [f">> done. assess with: lobes assess --port {port}"]
    if tc is not None:
        out.append(">> " + _runtime_ops.format_tool_probe(tc))
    emit_result("\n".join(out), json_mode=False)


def _apply_env_only(model, env_path, plan, notices, served, port, json_mode) -> None:
    """``--apply`` blocked on a required compose edit: write ``.env`` but DON'T restart.

    Switching to a model the shared template can't serve unedited (a non-MTP model,
    while the template ships the MTP primary's flags; or the MoE backend flag) would
    take a healthy container down and fail to bring it back. So we persist the plan
    to ``.env`` and stop, printing the edits to make by hand. The user applies them
    and then runs ``lobes serve --apply`` — or re-runs ``switch --apply --force`` to
    recreate the container anyway.
    """
    for key, value in plan.items():
        _env.set_env(env_path, key, value)
    if json_mode:
        emit_result(
            {
                "switched": model,
                "served_name": served,
                "port": port,
                "restarted": False,
                "blocked_on_compose_edits": True,
                "compose_edits": notices,
                "next": "apply the compose edits, then: lobes serve --apply"
                " (or re-run switch --apply --force)",
            },
            json_mode=True,
        )
        return
    lines = [f">> wrote .env for {model} but did NOT restart — a manual compose edit is required:"]
    lines += [f">> NOTE: {notice}" for notice in notices]
    lines.append(">> then: lobes serve --apply  (or re-run: lobes switch ... --apply --force)")
    emit_result("\n".join(lines), json_mode=False)


def cmd_switch(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)
    served = args.served_name or args.model
    plan, messages = _build_plan(args, port, served)
    notices = _serve_notices(args.model, args)

    effective_task = _resolve_task(args)
    is_pooling = effective_task != "generate"
    if is_pooling:
        # Embed/score gears have no tool calling — skip the probe unconditionally and
        # surface a curl hint so the operator can verify via the right endpoint.
        probe = False
        messages.append(
            "post-switch probe: skipped (embed/score has no tool calling) — "
            "verify with curl /v1/embeddings or /v1/score"
        )
    else:
        probe = not args.no_probe

    if args.apply:
        if notices and not args.force:
            # Required compose edit pending — don't take a healthy deployment down.
            _apply_env_only(args.model, env_path, plan, notices, served, port, json_mode)
        else:
            _apply_switch(
                args.model,
                deploy_dir,
                env_path,
                plan,
                messages,
                notices,
                port,
                served,
                probe,
                json_mode,
            )
    else:
        _emit_dry_run(deploy_dir, env_path, plan, messages, notices, port, probe, json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "switch",
        help="Switch the served vLLM model (dry-run by default; --apply to commit).",
    )
    p.add_argument("model", help="Model to serve, e.g. sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP.")
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
        help="Context window (default: the machine profile, e.g. 262144 on spark, "
        "clamped down to the model's native ceiling for 32K-native catalog models).",
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
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument(
        "--apply", action="store_true", help="Commit the switch (recreate the container)."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="With --apply, recreate the container even when a manual compose edit "
        "is required (default: write .env but skip the restart so a healthy "
        "deployment isn't taken down by an incompatible compose file).",
    )
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the post-switch tool-calling probe (tool_choice:auto).",
    )
    p.add_argument(
        "--task",
        choices=["generate", "embed", "score"],
        default=None,
        help="vLLM task; embed/score serve a pooling model (no tool calling, "
        "small context + low GPU fraction). Default unset: auto-detected from the "
        "catalog for known embed/score models. An explicit flag (incl. "
        "--task generate) always wins, so it can override the catalogued task.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_switch)
