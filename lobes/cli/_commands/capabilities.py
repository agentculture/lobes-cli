"""``lobes capabilities`` / ``lobes endpoint`` — the role registry (issue #81, t5).

Read-only, CLI-side view of the SIX first-class Colleague-facing roles
(``cortex`` / ``senses`` / ``embedder`` / ``reranker`` / ``stt`` / ``tts``),
built by the ONE canonical registry builder in :mod:`lobes.roles` —
:func:`lobes.roles.role_registry_from_env` — the same builder the gateway's
``GET /capabilities`` (t6) will call, so the role→endpoint contract has exactly
one source of truth.

Deployment resolution
----------------------
The CLI process never has the deployment's env vars in its own ``os.environ``
(those are injected into the *containers* by ``docker compose``, not the host
shell), so the deployment dir's ``.env`` is read straight off disk via
:func:`lobes.runtime._env.read_env_file` and handed to
:func:`~lobes.roles.role_registry_from_env` as the ``env`` mapping — this is
what drives BOTH the routing table (which roles are wired) and the served-
context overlay (``PRIMARY_MAX_MODEL_LEN`` and friends, t5). Resolution is
**soft**: an unscaffolded deployment (no ``lobes init`` yet) degrades to ``{}``
rather than erroring, mirroring ``lobes overview --live`` / ``assess`` /
``benchmark`` — a read-only introspection verb should always answer, showing
catalog defaults with every role ``loaded=False`` except the always-present
``cortex``.

The reachable gateway URL is derived exactly like every other read-only probe
in this CLI (``status``, ``overview --live``, ``fleet status``):
``http://localhost:<published port>``, where the port is ``--port`` if given,
else ``VLLM_PORT`` from the deployment's ``.env``, else 8000
(:func:`lobes.cli._runtime_ops.resolve_port_soft`).

Both verbs are strictly read-only: no compose/docker call, no ``--apply``.
"""

from __future__ import annotations

import argparse
import dataclasses

from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.roles import ROLES, RoleInfo, role_registry_from_env

_JSON_HELP = "Emit structured JSON."
_COMPOSE_DIR_HELP = "Deployment dir (default: $LOBES_DIR or ~/.lobes)."
_PORT_HELP = "Gateway host port (default: VLLM_PORT in .env)."


def _registry(args: argparse.Namespace) -> dict[str, RoleInfo]:
    env = _runtime_ops.deployment_env_soft(args)
    port, _ = _runtime_ops.resolve_port_soft(args)
    gateway_url = f"http://localhost:{port}"
    return role_registry_from_env(env, gateway_url=gateway_url)


def _role_payload(info: RoleInfo) -> dict[str, object]:
    """The full ``RoleInfo`` field set as a JSON-safe dict.

    The #81 shape wants at least ``{endpoint, model, context, ready,
    responsibilities}`` present; ``dataclasses.asdict`` includes every field
    (tuples become arrays), so the rest (role/runtime/path/quant/mtp/
    forbidden_responsibilities/loaded) rides along too.
    """
    return dataclasses.asdict(info)


def _render_table(registry: dict[str, RoleInfo]) -> str:
    header = f"{'role':<9} {'model':<48} {'context':>8}  loaded  endpoint"
    lines = [header, "-" * len(header)]
    for role in ROLES:
        info = registry[role]
        model = info.model if len(info.model) <= 48 else info.model[:45] + "..."
        lines.append(
            f"{info.role:<9} {model:<48} {info.context:>8}  "
            f"{'yes' if info.loaded else 'no ':<6} {info.endpoint or '(none)'}"
        )
        lines.append(f"          responsibilities: {', '.join(info.responsibilities)}")
    return "\n".join(lines)


def cmd_capabilities(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    registry = _registry(args)
    if json_mode:
        emit_result({role: _role_payload(registry[role]) for role in ROLES}, json_mode=True)
    else:
        emit_result(_render_table(registry), json_mode=False)
    return 0


def cmd_endpoint(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    role = args.role
    if role not in ROLES:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"unknown role {role!r}",
            remediation=f"valid roles: {', '.join(ROLES)}",
        )
    registry = _registry(args)
    endpoint = registry[role].endpoint
    if json_mode:
        emit_result({"role": role, "endpoint": endpoint}, json_mode=True)
    else:
        emit_result(endpoint, json_mode=False)
    return 0


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, help=_PORT_HELP)
    p.add_argument("--compose-dir", help=_COMPOSE_DIR_HELP)
    p.add_argument("--json", action="store_true", help=_JSON_HELP)


def register(sub: argparse._SubParsersAction) -> None:
    cap = sub.add_parser(
        "capabilities",
        help="Read-only: the six first-class roles (cortex/senses/embedder/"
        "reranker/stt/tts) resolved to live endpoint + metadata (issue #81).",
    )
    _add_common_args(cap)
    cap.set_defaults(func=cmd_capabilities)

    ep = sub.add_parser(
        "endpoint",
        help="Read-only: print the base URL for one role, e.g. 'lobes endpoint cortex'.",
    )
    ep.add_argument("role", help=f"one of: {', '.join(ROLES)}")
    _add_common_args(ep)
    ep.set_defaults(func=cmd_endpoint)
