"""``lobes fleet up | down | status`` — drive the gateway deployment.

The fleet is the always-warm Qwen generate primary plus co-resident embedding and
reranker gears behind one stdlib gateway (scaffolded by ``lobes init --fleet``),
routed by task family; there is one generate backend by default, with an opt-in
warm generate fallback. These verbs are the fleet-lane counterparts of the single-model
``serve`` / ``stop`` / ``status``:

- ``lobes fleet up`` — ``docker compose up -d --build`` (builds the gateway image),
  then waits for the gateway ``/health``. Dry-run by default; ``--apply`` commits.
- ``lobes fleet down`` — ``docker compose down``. Dry-run by default; ``--apply``.
- ``lobes fleet status`` — read-only: each container's state, the gateway's
  ``/health``, and the *warm* routed model list (``/v1/models``). The full catalog
  you can switch to is ``lobes overview --list`` / ``/v1/models/supported``.

``lobes switch`` does NOT drive the fleet (it rewrites the single-model ``VLLM_*``
keys); change fleet models by editing the fleet ``.env`` and re-running ``up``.
"""

from __future__ import annotations

import argparse

from lobes import assess
from lobes.cli import _runtime_ops
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.runtime import _compose, _env, _health

_UNSET = "(unset)"
_JSON_HELP = "Emit structured JSON."
_PORT_HELP = "Gateway host port (default: VLLM_PORT in .env)."


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
        payload = {"dry_run": True, "deployment_dir": str(deploy_dir), "port": port}
        emit_result(payload if json_mode else msg, json_mode=json_mode)
    else:
        emit_diagnostic(f">> building + starting the fleet in {deploy_dir}")
        # Ensure the durable-log dir exists (user-owned) before compose bind-mounts it.
        _compose.ensure_log_dir(deploy_dir, _env.read_env(env_path, _compose.LOG_DIR_ENV) or None)
        _runtime_ops.compose_check(
            _compose.compose_up_build(deploy_dir), "docker compose up -d --build"
        )
        # The gateway answers /health within seconds (it doesn't block on backends);
        # the vLLM backends load in the background — check them via 'lobes fleet status'.
        _health.wait_health(
            port, deadline_seconds=120, interval=5, container=_compose.FLEET_GATEWAY
        )
        result = {
            "serving": True,
            "port": port,
            "deployment_dir": str(deploy_dir),
            "containers": list(_compose.fleet_containers(deploy_dir)),
        }
        text = (
            f">> gateway up on :{port}. Backends load in the background — "
            f"check: lobes fleet status --compose-dir {deploy_dir}"
        )
        emit_result(result if json_mode else text, json_mode=json_mode)
    return 0


def cmd_fleet_down(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)

    if not args.apply:
        dry = (
            f"DRY RUN — would run: docker compose down in {deploy_dir}.\n"
            "Re-run with --apply to execute."
        )
        payload = {"dry_run": True, "deployment_dir": str(deploy_dir)}
        emit_result(payload if json_mode else dry, json_mode=json_mode)
    else:
        emit_diagnostic(f">> stopping the fleet in {deploy_dir}")
        _runtime_ops.compose_check(_compose.compose_down(deploy_dir), "docker compose down")
        result = {"stopped": True, "deployment_dir": str(deploy_dir)}
        emit_result(
            result if json_mode else f">> fleet stopped in {deploy_dir}", json_mode=json_mode
        )
    return 0


def cmd_fleet_status(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)

    containers = [
        {"name": name, "state": _compose.inspect_state(name)}
        for name in _compose.fleet_containers(deploy_dir)
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


def cmd_fleet_files(args: argparse.Namespace) -> int:
    """``lobes fleet files`` — print the resolved ``docker compose -f`` chain.

    Read-only (#137): emits exactly the argv tokens every lobes verb passes to
    ``docker compose`` (one token per line, relative to the deployment dir), so
    shell scripts consume the chain from the CLI instead of re-implementing it —
    ``mapfile -t files < <(lobes fleet files --compose-dir "$dir")``. A plain
    deployment (no lobes overlay) prints NOTHING: compose resolves the project
    itself and its own convention layers base + operator override, so an empty
    mapfile array — a bare ``docker compose`` call — is the faithful chain.
    """
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)
    files = _compose._compose_files(deploy_dir)
    if json_mode:
        emit_result({"deployment_dir": str(deploy_dir), "files": files}, json_mode=True)
    elif files:
        emit_result("\n".join(files), json_mode=False)
    return 0


def _no_verb(args: argparse.Namespace) -> int:
    # Bare `lobes fleet` → the read-only status (safe default).
    return cmd_fleet_status(args)


def _add_compose_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fleet",
        help="Drive the gateway fleet (up / down / status). See 'lobes fleet status'.",
    )
    _add_compose_dir(p)
    p.add_argument("--port", type=int, help=_PORT_HELP)
    p.add_argument("--json", action="store_true", help=_JSON_HELP)
    p.set_defaults(func=_no_verb, json=False)
    # Propagate the structured-error parser class to the noun's subparsers.
    noun = p.add_subparsers(dest="fleet_command", parser_class=type(p))

    up = noun.add_parser("up", help="Build + start the fleet (dry-run; --apply).")
    _add_compose_dir(up)
    up.add_argument("--port", type=int, help=_PORT_HELP)
    up.add_argument("--apply", action="store_true", help="Actually build + start the fleet.")
    up.add_argument("--json", action="store_true", help=_JSON_HELP)
    up.set_defaults(func=cmd_fleet_up)

    down = noun.add_parser("down", help="Stop the fleet (dry-run; --apply).")
    _add_compose_dir(down)
    down.add_argument("--apply", action="store_true", help="Actually stop the fleet.")
    down.add_argument("--json", action="store_true", help=_JSON_HELP)
    down.set_defaults(func=cmd_fleet_down)

    st = noun.add_parser(
        "status",
        help="Read-only: container states, gateway /health, warm /v1/models "
        "(catalog: overview --list).",
    )
    _add_compose_dir(st)
    st.add_argument("--port", type=int, help=_PORT_HELP)
    st.add_argument("--json", action="store_true", help=_JSON_HELP)
    st.set_defaults(func=cmd_fleet_status)

    files_p = noun.add_parser(
        "files",
        help="Read-only: print the resolved docker compose -f chain, one argv "
        "token per line (empty for a plain deployment — compose's own "
        "base+override resolution applies). For scripts: mapfile -t files "
        "< <(lobes fleet files).",
    )
    _add_compose_dir(files_p)
    files_p.add_argument("--json", action="store_true", help=_JSON_HELP)
    files_p.set_defaults(func=cmd_fleet_files)
