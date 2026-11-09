"""``lobes site`` — serve the local site (Mesh, Rollout, Residents) and its /api.

Dry-run by default: prints the URL, the gateway it reads and the routes it
would serve. ``--apply`` starts the server in the foreground until Ctrl-C.
The site is read-only either way: it issues GETs to gateways, reads the
residents file and the rollout journal, and writes nothing. It binds
127.0.0.1 unless ``--host`` says otherwise; it has no authentication.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.runtime import _compose, _env
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


def cmd_site(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    if not 0 <= args.port <= 65535:
        raise ModelGearError(
            code=EXIT_USER_ERROR, message=f"--port out of range: {args.port}", remediation=""
        )
    deploy_dir = _deploy_dir(args)
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
        "Dry-run prints the URL; --apply starts it.",
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
    p.add_argument("--apply", action="store_true", help="Start the server (foreground).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_site)
