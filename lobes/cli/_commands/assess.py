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

Exit code: plain ``lobes assess`` / ``--preserve-thinking`` always exit
``EXIT_SUCCESS`` — they only emit a report. ``--probes`` is scriptable: it
exits ``EXIT_SUCCESS`` when every probed role passes and ``EXIT_ENV_ERROR``
when any role fails (mirrors ``lobes tunnel``'s status-driven exit code) —
the ``--json``/text payload is unchanged either way, only the exit code
differentiates pass from fail.
"""

from __future__ import annotations

import argparse

from lobes import assess as _assess
from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_SUCCESS
from lobes.cli._output import emit_result
from lobes.roles import role_registry_from_env
from lobes.runtime import _compose, _env

# Roles whose `model` field the gateway resolves through its tier-alias table
# (`lobes.catalog.TIER_ROLE`, consumed by `lobes.gateway._routing.tier_aliases`
# — "cortex"/"senses"/"main"/... , issue #81/#92) rather than requiring the
# concrete served id. Every probe endpoint here IS the gateway (`RoleInfo.endpoint`
# is always `gateway_url`/`server.public_url` for the four gateway-fronted roles
# — see `lobes.roles._gateway_role`, `endpoint = gateway`), so sending the role
# alias exercises the SAME alias-resolution lane a real caller uses, including
# the hardware-infeasibility 404 (issue #92 t6) a concrete served name would
# never trigger. `embedder`/`reranker` have NO such gateway alias — TIER_ROLE
# only covers the generate lane — so sending those role names literally as
# `model` would 404 as `model_not_found` (see
# `lobes.gateway._routing.is_unknown_model`); those two probes must keep using
# the concrete served name (`info.model`).
_GATEWAY_ALIASED_PROBE_ROLES = frozenset({"cortex"})


def _probe_model(role: str, info) -> str:
    """The `model` field to send for `role`'s probe request.

    The stable gateway alias (the role name itself) when the gateway resolves
    one for this role, else the concrete served name — see
    `_GATEWAY_ALIASED_PROBE_ROLES`.
    """
    return role if role in _GATEWAY_ALIASED_PROBE_ROLES else info.model


def _cmd_assess_probes(args: argparse.Namespace, url: str, json_mode: bool) -> int:
    """``--probes``: per-role CORRECTNESS probes (issue #81, t7).

    Each of cortex/embedder/reranker is resolved to its own endpoint via the
    shared role registry (same builder `lobes capabilities`/`lobes measure`
    use) rather than the single gateway `url`. Exits EXIT_SUCCESS when every
    probed role passes, EXIT_ENV_ERROR when any fails — the payload shape is
    identical either way.

    The `model` field sent with each request is `_probe_model`'s choice, not
    always `info.model`: the cortex probe sends the stable role alias
    ("cortex") so it exercises the gateway's alias-resolution lane the way a
    real caller does; embedder/reranker send the concrete served name because
    the gateway has no alias for them (see `_GATEWAY_ALIASED_PROBE_ROLES`).
    """
    env = _runtime_ops.deployment_env_soft(args)
    registry = role_registry_from_env(env, gateway_url=url)
    roles = (args.role,) if getattr(args, "role", None) else _assess.PROBE_ROLES
    timeout = float(getattr(args, "timeout", None) or _assess.DEFAULT_PROBE_TIMEOUT)
    endpoints = {
        role: (info.endpoint, _probe_model(role, info)) if info.loaded and info.endpoint else None
        for role, info in registry.items()
    }
    results = _assess.run_role_probes(endpoints, roles=roles, timeout=timeout)
    passed = all(r["ok"] for r in results.values())
    if json_mode:
        emit_result({"passed": passed, "probes": results}, json_mode=True)
    else:
        emit_result(_assess.render_role_probes(results), json_mode=False)
    return EXIT_SUCCESS if passed else EXIT_ENV_ERROR


def _cmd_assess_preserve_thinking(url: str, model: str | None, json_mode: bool) -> int:
    """``--preserve-thinking``: the two-turn token-delta diagnostic (issue #93).

    Reports both prompt-token counts and the delta; no host facts / correctness
    probes are needed.
    """
    pt = _assess.run_preserve_thinking_probe(url, model)
    emit_result(pt if json_mode else _assess.render_preserve_thinking(pt), json_mode=json_mode)
    return EXIT_SUCCESS


def _cmd_assess_correctness(
    args: argparse.Namespace, url: str, model: str | None, json_mode: bool
) -> int:
    """Default path: the two fixed correctness probes plus host-side facts."""
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
    return EXIT_SUCCESS


def cmd_assess(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    model = args.model
    if model is None and deploy_dir is not None:
        model = _env.read_env(deploy_dir / _compose.ENV_FILE, "VLLM_SERVED_NAME")

    url = f"http://localhost:{port}"

    if bool(getattr(args, "probes", False)):
        return _cmd_assess_probes(args, url, json_mode)
    if bool(getattr(args, "preserve_thinking", False)):
        return _cmd_assess_preserve_thinking(url, model, json_mode)
    return _cmd_assess_correctness(args, url, model, json_mode)


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
