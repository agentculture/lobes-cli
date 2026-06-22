"""``lobes tunnel`` — expose the local vLLM API via a Cloudflare Tunnel.

Mutating: dry-run by default; ``--apply`` starts a standalone ``cloudflared tunnel
run`` (background, logging to the deployment dir) that proxies the owner-chosen
public hostname to the local OpenAI-compatible server, and ``--stop`` terminates
it. Mirrors ``serve``/``stop``. The Cloudflare side (tunnel + ingress + DNS,
run-token sealed in shushu) is provisioned once by ``cultureflare remote-login
--no-access`` — see the README. The public API is only safe to expose when
``CULTURE_VLLM_API_KEY`` gates it with a bearer token.
"""

from __future__ import annotations

import argparse

from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_SUCCESS, EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.runtime import _compose, _health, _tunnel


def _public_url(hostname: str) -> str:
    return f"https://{hostname}/v1"


def _cmd_stop(args: argparse.Namespace, deploy_dir, json_mode: bool) -> int:
    if not args.apply:
        pid = _tunnel.tunnel_pid(deploy_dir)
        target = f"cloudflared (pid {pid})" if pid else "cloudflared (no running tunnel found)"
        if json_mode:
            emit_result(
                {"dry_run": True, "action": "stop", "pid": pid, "deployment_dir": str(deploy_dir)},
                json_mode=True,
            )
        else:
            emit_result(
                f"DRY RUN — would stop {target} in {deploy_dir}.\n"
                "Re-run with --apply to execute.",
                json_mode=False,
            )
        return EXIT_SUCCESS
    emit_diagnostic(f">> stopping the cloudflared tunnel in {deploy_dir}")
    status, pid = _tunnel.stop_tunnel(deploy_dir)
    if json_mode:
        emit_result(
            {
                "stopped": status == "stopped",
                "status": status,
                "pid": pid,
                "deployment_dir": str(deploy_dir),
            },
            json_mode=True,
        )
    elif status == "stopped":
        emit_result(f">> stopped pid {pid}.", json_mode=False)
    elif status == "idle":
        emit_result(">> no running tunnel to stop.", json_mode=False)
    else:  # "failed" — signalled but still alive; pidfile kept for a retry
        emit_diagnostic(
            f">> pid {pid} did not exit after SIGTERM/SIGKILL; the pidfile is kept.\n"
            ">> inspect it (ps) and retry: lobes tunnel --stop --apply"
        )
    return EXIT_ENV_ERROR if status == "failed" else EXIT_SUCCESS


def cmd_tunnel(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)

    if getattr(args, "stop", False):
        return _cmd_stop(args, deploy_dir, json_mode)

    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)
    hostname = _tunnel.resolve_hostname(getattr(args, "hostname", None), deploy_dir)
    mode, value = _tunnel.resolve_token(deploy_dir)
    command = _tunnel.build_command(mode, value)
    safe_command = _tunnel.redacted(command)
    url = _public_url(hostname)

    if not args.apply:
        if json_mode:
            emit_result(
                {
                    "dry_run": True,
                    "hostname": hostname,
                    "url": url,
                    "port": port,
                    "token_source": mode,
                    "command": safe_command,
                    "deployment_dir": str(deploy_dir),
                },
                json_mode=True,
            )
        else:
            emit_result(
                f"DRY RUN — would run: {' '.join(safe_command)}\n"
                f"(in {deploy_dir}), exposing the local :{port} API at {url}.\n"
                "Re-run with --apply to execute.",
                json_mode=False,
            )
        return EXIT_SUCCESS

    # --apply: refuse to orphan an already-running tunnel, then preflight
    # (cloudflared / shushu on PATH, local server up) and start.
    running = _tunnel.tunnel_pid(deploy_dir)
    if running:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"a tunnel is already running (pid {running})",
            remediation="stop it first: lobes tunnel --stop --apply",
        )
    if not _tunnel.cloudflared_present():
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message="cloudflared not found on PATH",
            remediation=_tunnel.INSTALL_HINT,
        )
    if mode == "shushu" and not _tunnel.shushu_present():
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message="shushu not found on PATH",
            remediation=(
                "install shushu, or set CULTURE_CF_TUNNEL_TOKEN (plaintext) in "
                + str(_tunnel.tunnel_env_path(deploy_dir))
            ),
        )
    if not _health.is_healthy(port):
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"local server not healthy on :{port}",
            remediation="start it first: lobes serve --apply "
            "(a tunnel to a down server just serves errors)",
        )
    emit_diagnostic(f">> starting cloudflared tunnel for :{port} -> {url}")
    pid = _tunnel.start_tunnel(deploy_dir, command, _tunnel.token_env(mode, value))
    log = _tunnel.log_path(deploy_dir)
    if json_mode:
        emit_result(
            {
                "tunneling": True,
                "url": url,
                "hostname": hostname,
                "port": port,
                "pid": pid,
                "token_source": mode,
                "log": str(log),
                "deployment_dir": str(deploy_dir),
            },
            json_mode=True,
        )
    else:
        emit_result(
            f">> tunnel up (pid {pid}); model reachable at {url}\n"
            f">> logs: {log}; stop with: lobes tunnel --stop --apply",
            json_mode=False,
        )
    return EXIT_SUCCESS


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tunnel",
        help="Expose the local vLLM API via a Cloudflare Tunnel (dry-run by default; --apply).",
    )
    p.add_argument(
        "--hostname",
        help="Public hostname (default: $CULTURE_VLLM_PUBLIC_HOSTNAME or .cf-tunnel.env).",
    )
    p.add_argument(
        "--port", type=int, help="Local API port to preflight (default: VLLM_PORT in .env)."
    )
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument(
        "--apply", action="store_true", help="Actually start (or with --stop, stop) the tunnel."
    )
    p.add_argument(
        "--stop", action="store_true", help="Stop the running tunnel for this deployment."
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_tunnel)
