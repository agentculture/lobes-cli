"""``lobes site`` — serve the local site (Mesh, Rollout, Residents) and its /api.

Dry-run by default: prints the URL, the gateway it reads and the routes it
would serve. ``--apply`` starts the server in the foreground until Ctrl-C.
The site is read-only either way: it issues GETs to gateways, reads the
residents file and the rollout journal, and writes nothing. It binds
127.0.0.1 unless ``--host`` says otherwise; it has no authentication.

``lobes site install [--host --port --gateway] --apply`` keeps it up across
reboots as a systemd user unit (``Restart=always``; linger required);
``uninstall --apply`` removes it; ``status`` reports it. Install and uninstall
are dry-run without ``--apply``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.runtime import _compose, _env
from lobes.site import _service
from lobes.site import server as _server

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _deploy_dir(args: argparse.Namespace) -> Path:
    try:
        return _compose.resolve_deployment_dir(getattr(args, "compose_dir", None))
    except ModelGearError:
        if getattr(args, "compose_dir", None) is not None:
            raise
        return _compose.default_deployment_dir()


def _gateway(args: argparse.Namespace, deploy_dir: Path) -> str:
    if args.gateway:
        return args.gateway.rstrip("/")
    port = _env.read_env(deploy_dir / _compose.ENV_FILE, "VLLM_PORT", "8000")
    return f"http://127.0.0.1:{_env.parse_port(port, 'VLLM_PORT')}"


def _cmd_install(args: argparse.Namespace, deploy_dir: Path, json_mode: bool) -> int:
    ctx = _service.default_context()
    plan = _service.plan_install(
        deploy_dir, ctx, host=args.host, port=args.port, gateway=args.gateway
    )
    if not args.apply:
        if json_mode:
            emit_result({"dry_run": True, **plan.to_dict()}, json_mode=True)
        else:
            lines = [
                f"DRY RUN — would {'rewrite' if plan.replaces else 'write'} {plan.unit_path}:",
                plan.content.rstrip(),
                "then: " + "; ".join(" ".join(c) for c in plan.commands),
                f"manifest: {deploy_dir / _service.MANIFEST}",
            ]
            lines += [f"BLOCKED: {b}" for b in plan.blockers]
            lines.append("Re-run with --apply to install it.")
            emit_result("\n".join(lines), json_mode=False)
        return 0
    result = _service.apply_install(plan, ctx)
    url = f"http://{args.host}:{args.port}/"
    if json_mode:
        emit_result({"dry_run": False, "url": url, **result}, json_mode=True)
    else:
        emit_result(
            f"installed {_service.UNIT_NAME} (user unit, enabled, starts at boot): {url}",
            json_mode=False,
        )
    return 0


def _cmd_uninstall(args: argparse.Namespace, deploy_dir: Path, json_mode: bool) -> int:
    ctx = _service.default_context()
    if not args.apply:
        plan = _service.plan_uninstall(deploy_dir, ctx)
        if json_mode:
            emit_result({"dry_run": True, **plan}, json_mode=True)
        elif not plan["installed"]:
            emit_result("lobes site service is not installed; nothing to do", json_mode=False)
        else:
            emit_result(
                "DRY RUN — would run: "
                + "; ".join(" ".join(c) for c in plan["commands"])
                + f"\n  remove: {', '.join(plan['files'])}\nRe-run with --apply.",
                json_mode=False,
            )
        return 0
    result = _service.apply_uninstall(deploy_dir, ctx)
    if json_mode:
        emit_result({"dry_run": False, **result}, json_mode=True)
    else:
        emit_result(f"removed {', '.join(result.get('removed', [])) or 'nothing'}", json_mode=False)
    return 0


def _cmd_status(deploy_dir: Path, json_mode: bool) -> int:
    st = _service.status(deploy_dir, _service.default_context())
    if json_mode:
        emit_result(st, json_mode=True)
    elif not st["installed"]:
        emit_result(
            "lobes site service: not installed ('lobes site install --apply')", json_mode=False
        )
    else:
        emit_result(
            f"lobes site service: {st['active']}, {st['enabled']}, linger={st['linger']}, "
            f"unit {'present' if st['files_present'] else 'MISSING'}",
            json_mode=False,
        )
    return 0


def cmd_site(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    if not 0 <= args.port <= 65535:
        raise ModelGearError(
            code=EXIT_USER_ERROR, message=f"--port out of range: {args.port}", remediation=""
        )
    deploy_dir = _deploy_dir(args)
    action = getattr(args, "action", None) or "serve"
    if action == "install":
        return _cmd_install(args, deploy_dir, json_mode)
    if action == "uninstall":
        return _cmd_uninstall(args, deploy_dir, json_mode)
    if action == "status":
        return _cmd_status(deploy_dir, json_mode)
    gateway = _gateway(args, deploy_dir)
    if args.host not in _LOOPBACK:
        emit_diagnostic(
            f"warning: binding {args.host} exposes the unauthenticated site to that network"
        )
    if not args.apply:
        url = f"http://{args.host}:{args.port}/"
        payload = {
            "dry_run": True,
            "url": url,
            "host": args.host,
            "port": args.port,
            "gateway": gateway,
            "deployment_dir": str(deploy_dir),
            "routes": list(_server.ROUTES),
            "writes": [],
        }
        if json_mode:
            emit_result(payload, json_mode=True)
        else:
            emit_result(
                f"DRY RUN — would serve the lobes site at {url}\n"
                f"  reads: {gateway} (/health, /capabilities, /mesh/roster), "
                f"residents.toml and the rollout journal in {deploy_dir}\n"
                f"  routes: {', '.join(_server.ROUTES)}\n"
                "  writes: nothing\n"
                "Re-run with --apply to start it.",
                json_mode=False,
            )
        return 0
    sources = _server.default_sources(deploy_dir, gateway=gateway)
    try:
        srv = _server.SiteServer(sources, host=args.host, port=args.port)
    except OSError as exc:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"cannot bind {args.host}:{args.port}: {exc.strerror or exc}",
            remediation="pick another --port",
        ) from exc
    try:
        if json_mode:
            emit_result({"dry_run": False, "url": srv.url, "gateway": gateway}, json_mode=True)
        else:
            emit_result(srv.url, json_mode=False)
        emit_diagnostic(f">> serving the lobes site at {srv.url} (Ctrl-C stops it)")
        _server.run_until_interrupted(srv)
    finally:
        srv.httpd.server_close()
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "site",
        help="Serve the local lobes site (Mesh, Rollout, Residents) and its read-only /api. "
        "Dry-run prints the URL; --apply starts it. 'install --apply' keeps it up across "
        "reboots (systemd user unit).",
    )
    p.add_argument(
        "action",
        nargs="?",
        default="serve",
        choices=("serve", "install", "uninstall", "status"),
        help="serve (default, foreground); install/uninstall the systemd user unit; status.",
    )
    p.add_argument("--host", default=_server.DEFAULT_HOST, help="Bind address (default 127.0.0.1).")
    p.add_argument(
        "--port",
        type=int,
        default=_server.DEFAULT_PORT,
        help=f"Port (default {_server.DEFAULT_PORT}; 0 picks a free one).",
    )
    p.add_argument(
        "--gateway",
        default=None,
        help="This box's gateway origin (default http://127.0.0.1:<VLLM_PORT>).",
    )
    p.add_argument(
        "--compose-dir", default=None, help="Deployment dir (default $LOBES_DIR or ~/.lobes)."
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Commit: start the server (serve) or write/remove the unit (install/uninstall).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_site)
