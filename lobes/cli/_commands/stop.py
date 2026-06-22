"""``lobes stop`` — stop and remove the vLLM server.

Mutating: dry-run by default; ``--apply`` runs ``docker compose down`` in the
deployment dir.
"""

from __future__ import annotations

import argparse

from lobes.cli import _runtime_ops
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.runtime import _compose


def cmd_stop(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _runtime_ops.deployment_dir(args)

    if not args.apply:
        if json_mode:
            emit_result({"dry_run": True, "deployment_dir": str(deploy_dir)}, json_mode=True)
        else:
            emit_result(
                f"DRY RUN — would run: docker compose down in {deploy_dir}.\n"
                "Re-run with --apply to execute.",
                json_mode=False,
            )
    else:
        emit_diagnostic(f">> stopping the vLLM server in {deploy_dir}")
        _runtime_ops.compose_check(_compose.compose_down(deploy_dir), "docker compose down")
        result = {"stopped": True, "deployment_dir": str(deploy_dir)}
        if json_mode:
            emit_result(result, json_mode=True)
        else:
            emit_result(">> stopped.", json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "stop",
        help="Stop the vLLM server (dry-run by default; --apply to commit).",
    )
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--apply", action="store_true", help="Actually stop the server.")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_stop)
