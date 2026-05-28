"""``model fleet up | down | status`` — drive the 3-container gateway deployment.

The fleet is two always-warm vLLM backends behind one stdlib gateway (scaffolded
by ``model init --fleet``). These verbs are the fleet-lane counterparts of the
single-model ``serve`` / ``stop`` / ``status``:

- ``model fleet up`` — ``docker compose up -d --build`` (builds the gateway image),
  then waits for the gateway ``/health``. Dry-run by default; ``--apply`` commits.
- ``model fleet down`` — ``docker compose down``. Dry-run by default; ``--apply``.
- ``model fleet status`` — read-only: each container's state, the gateway's
  ``/health``, and the routed model list (``/v1/models``).

``model switch`` does NOT drive the fleet (it rewrites the single-model ``VLLM_*``
keys); change fleet models by editing the fleet ``.env`` and re-running ``up``.
"""

from __future__ import annotations

import argparse

from model_gear import assess
from model_gear.cli import _runtime_ops
from model_gear.cli._output import emit_diagnostic, emit_result
from model_gear.runtime import _compose, _health

_UNSET = "(unset)"


def _fleet_models(port: int) -> list[str] | None:
    """Best-effort ``/v1/models`` ids via the gateway; ``None`` if unreachable."""
    if not _health.is_healthy(port):
        return None
    try:
        _, payload = assess._get(f"http://localhost:{port}", "/v1/models")
        data = payload.get("data") if isinstance(payload, dict) else None
        return [m.get("id") for m in data] if isinstance(data, list) else None
    except OSError:
        return None


def cmd_fleet_up(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)

    if not args.apply:
        msg = (
            f"DRY RUN — would run: docker compose up -d --build in {deploy_dir}, "
            f"then wait for the gateway /health on :{port}.\nRe-run with --apply to execute."
        )
        emit_result(
            (
                {"dry_run": True, "deployment_dir": str(deploy_dir), "port": port}
                if json_mode
                else msg
            ),
            json_mode=json_mode,
        )
        return 0

    emit_diagnostic(f">> building + starting the fleet in {deploy_dir}")
    _runtime_ops.compose_check(
        _compose.compose_up_build(deploy_dir), "docker compose up -d --build"
    )
    # The gateway answers /health within seconds (it doesn't block on backends);
    # the vLLM backends load in the background — check them via 'model fleet status'.
    _health.wait_health(port, deadline_seconds=120, interval=5, container=_compose.FLEET_GATEWAY)
    result = {
        "serving": True,
        "port": port,
        "deployment_dir": str(deploy_dir),
        "containers": list(_compose.FLEET_CONTAINERS),
    }
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(
            f">> gateway up on :{port}. Backends load in the background — "
            f"check: model fleet status --compose-dir {deploy_dir}",
            json_mode=False,
        )
    return 0


def cmd_fleet_down(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)

    if not args.apply:
        emit_result(
            (
                {"dry_run": True, "deployment_dir": str(deploy_dir)}
                if json_mode
                else f"DRY RUN — would run: docker compose down in {deploy_dir}.\n"
                "Re-run with --apply to execute."
            ),
            json_mode=json_mode,
        )
        return 0

    emit_diagnostic(f">> stopping the fleet in {deploy_dir}")
    _runtime_ops.compose_check(_compose.compose_down(deploy_dir), "docker compose down")
    result = {"stopped": True, "deployment_dir": str(deploy_dir)}
    emit_result(result if json_mode else f">> fleet stopped in {deploy_dir}", json_mode=json_mode)
    return 0


def cmd_fleet_status(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)

    containers = [
        {"name": name, "state": _compose.inspect_state(name)} for name in _compose.FLEET_CONTAINERS
    ]
    report = {
        "deployment_dir": str(deploy_dir),
        "port": port,
        "gateway_health": "ok" if _health.is_healthy(port) else "not responding",
        "containers": containers,
        "models": _fleet_models(port),
    }

    if json_mode:
        emit_result(report, json_mode=True)
    else:
        lines = [
            f"dir:     {report['deployment_dir']}",
            f"gateway: {report['gateway_health']} (:{port})",
        ]
        for c in containers:
            lines.append(f"  {c['name']} — {c['state']}")
        models = report["models"]
        lines.append("models:  " + (", ".join(models) if models else _UNSET))
        emit_result("\n".join(lines), json_mode=False)
    return 0


def _no_verb(args: argparse.Namespace) -> int:
    # Bare `model fleet` → the read-only status (safe default).
    return cmd_fleet_status(args)


def _add_compose_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--compose-dir", help="Deployment dir (default: $MODEL_GEAR_DIR or ~/.model-gear)."
    )


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fleet",
        help="Drive the gateway fleet (up / down / status). See 'model fleet status'.",
    )
    _add_compose_dir(p)
    p.add_argument("--port", type=int, help="Gateway host port (default: VLLM_PORT in .env).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=_no_verb, json=False)
    # Propagate the structured-error parser class to the noun's subparsers.
    noun = p.add_subparsers(dest="fleet_command", parser_class=type(p))

    up = noun.add_parser("up", help="Build + start the fleet (dry-run; --apply).")
    _add_compose_dir(up)
    up.add_argument("--port", type=int, help="Gateway host port (default: VLLM_PORT in .env).")
    up.add_argument("--apply", action="store_true", help="Actually build + start the fleet.")
    up.add_argument("--json", action="store_true", help="Emit structured JSON.")
    up.set_defaults(func=cmd_fleet_up)

    down = noun.add_parser("down", help="Stop the fleet (dry-run; --apply).")
    _add_compose_dir(down)
    down.add_argument("--apply", action="store_true", help="Actually stop the fleet.")
    down.add_argument("--json", action="store_true", help="Emit structured JSON.")
    down.set_defaults(func=cmd_fleet_down)

    st = noun.add_parser("status", help="Read-only: container states, gateway /health, /v1/models.")
    _add_compose_dir(st)
    st.add_argument("--port", type=int, help="Gateway host port (default: VLLM_PORT in .env).")
    st.add_argument("--json", action="store_true", help="Emit structured JSON.")
    st.set_defaults(func=cmd_fleet_status)
