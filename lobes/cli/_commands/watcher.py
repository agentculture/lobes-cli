"""``lobes watcher`` — the host watcher for memory-gated lanes.

    lobes watcher run [--once] [--apply]  # the loop (the unit's ExecStart passes --apply)
    lobes watcher install [--system]    # dry-run plan; --apply installs the unit
    lobes watcher uninstall             # dry-run plan; --apply removes it
    lobes watcher status                # installed / active / linger / last clear

The watcher polls the deployment's ``logs/lane-state/*.gated`` markers and runs
the machine CLI's drop-caches at most once per ``LOBES_WATCHER_MIN_INTERVAL_S``
(default 300); with no grant it only logs. Its one docker call is a read-only
``docker ps`` (to skip a clear while another lane boots); it never changes a
container. ``run`` is dry-run unless ``--apply`` (operator decision d8): it
logs what it would clear and changes nothing; the installed unit passes
``--apply``. ``install``
and ``uninstall`` are write verbs: dry-run by default, ``--apply`` commits.
"""

from __future__ import annotations

import argparse
import json

from lobes.cli import _runtime_ops
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.watcher import _install, _run


def _cmd_run(args: argparse.Namespace) -> int:
    deploy = _runtime_ops.deployment_dir(args)
    watcher = _run.watcher_for(deploy, apply=bool(args.apply))
    if args.once:
        actions = watcher.tick()
        payload = {
            "dry_run": not args.apply,
            "state_dir": str(watcher.state_dir),
            "actions": actions,
        }
        if args.json:
            emit_result(payload, json_mode=True)
        else:
            tail = "" if args.apply else "\n>> dry run: re-run with --apply to clear"
            emit_result(
                f"{len(actions)} gated lane(s) in {watcher.state_dir}{tail}",
                json_mode=False,
            )
        return 0
    _run.run_forever(watcher, poll_s=args.poll)  # pragma: no cover - endless loop
    return 0  # pragma: no cover


def _cmd_install(args: argparse.Namespace) -> int:
    deploy = _runtime_ops.deployment_dir(args)
    ctx = _install.default_context(system=args.system)
    plan = _install.plan_install(deploy, ctx)
    if not args.apply:
        data = {"dry_run": True, **plan.to_dict()}
        if args.json:
            emit_result(data, json_mode=True)
        else:
            lines = [
                f"would install a {plan.unit_type} unit: {plan.unit_path}",
                f"manifest: {deploy / _install.MANIFEST}",
                f"linger: {plan.linger}",
                *(f"then: {' '.join(c)}" for c in plan.commands),
                *(f"BLOCKED: {b}" for b in plan.blockers),
                "--- unit ---",
                plan.content.rstrip(),
                ">> re-run with --apply to install",
            ]
            emit_result("\n".join(lines), json_mode=False)
        return 0
    result = _install.apply_install(plan, ctx)
    if args.json:
        emit_result({"dry_run": False, **result}, json_mode=True)
    else:
        written = ", ".join(result["files"])
        emit_result(
            f"installed {result['unit_type']} unit {_install.UNIT_NAME}: {written}\n"
            f"manifest: {result['manifest']}",
            json_mode=False,
        )
    return 0


def _cmd_uninstall(args: argparse.Namespace) -> int:
    deploy = _runtime_ops.deployment_dir(args)
    ctx = _install.default_context()
    if not args.apply:
        plan = _install.plan_uninstall(deploy, ctx)
        if args.json:
            emit_result({"dry_run": True, **plan}, json_mode=True)
        elif not plan["installed"]:
            emit_result("watcher not installed (no manifest); nothing to do", json_mode=False)
        else:
            lines = [f"would remove: {f}" for f in plan["files"]]
            lines += [f"would run: {' '.join(c)}" for c in plan["commands"]]
            lines += [f"would remove manifest: {plan['manifest']}"]
            lines += [">> re-run with --apply to uninstall"]
            emit_result("\n".join(lines), json_mode=False)
        return 0
    result = _install.apply_uninstall(deploy, ctx)
    if args.json:
        emit_result({"dry_run": False, **result}, json_mode=True)
    elif not result["installed"]:
        emit_result("watcher not installed (no manifest); nothing to do", json_mode=False)
    else:
        emit_result("removed: " + ", ".join(result["removed"]), json_mode=False)
    for skipped in result.get("skipped") or []:
        emit_diagnostic(f"warning: manifest listed {skipped}; not a watcher unit, left alone")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    deploy = _runtime_ops.deployment_dir(args)
    st = _install.status(deploy, _install.default_context())
    if args.json:
        emit_result(st, json_mode=True)
    else:
        emit_result(
            "\n".join(f"{k}: {json.dumps(v) if isinstance(v, list) else v}" for k, v in st.items()),
            json_mode=False,
        )
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "watcher",
        help="Host watcher: clear the page cache for memory-gated lanes "
        "(run / install / uninstall / status). Never changes a container.",
    )
    wsub = p.add_subparsers(dest="watcher_command", required=True)

    r = wsub.add_parser(
        "run",
        help="Run the watcher loop in the foreground (the unit's command). Dry-run "
        "(logs what it would clear) unless --apply.",
    )
    r.add_argument("--once", action="store_true", help="Poll once and exit.")
    r.add_argument(
        "--apply",
        action="store_true",
        help="Actually clear the page cache (default: log what would be cleared).",
    )
    r.add_argument(
        "--poll",
        type=float,
        default=_run.DEFAULT_POLL_S,
        help=f"Seconds between polls (default {_run.DEFAULT_POLL_S}).",
    )
    _common(r)
    r.set_defaults(func=_cmd_run)

    i = wsub.add_parser(
        "install",
        help="WRITE: install the boot-persistent watcher unit (dry-run unless --apply).",
    )
    i.add_argument(
        "--system",
        action="store_true",
        help="Install a system unit via 'sudo -n' instead of a user unit (no linger needed).",
    )
    i.add_argument("--apply", action="store_true", help="Commit (default: print the plan).")
    _common(i)
    i.set_defaults(func=_cmd_install)

    u = wsub.add_parser(
        "uninstall",
        help="WRITE: remove exactly the files the install recorded (dry-run unless --apply).",
    )
    u.add_argument("--apply", action="store_true", help="Commit (default: print the plan).")
    _common(u)
    u.set_defaults(func=_cmd_uninstall)

    s = wsub.add_parser("status", help="Read-only: installed / active / linger / last clear.")
    _common(s)
    s.set_defaults(func=_cmd_status)
