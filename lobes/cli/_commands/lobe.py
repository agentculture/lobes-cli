"""``lobes lobe export <role>... --member <name>`` — export a lobe's build recipe.

Read-only toward the member (ssh ``cat`` / ``docker inspect`` only). Writes a
local bundle (Dockerfile, build inputs, compose service or ``docker run``)
under ``--out``. Dry-run by default: lists the files; ``--apply`` writes them.
No ``.env`` / ``.secrets.env`` content is ever read, and every text is redacted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.rollout import _export, _residents
from lobes.rollout._transport import Target

_JSON_HELP = "Emit structured JSON."


def _target_for(base_dir: str, member: str, local: bool) -> Target:
    if local:
        res = _residents.load(base_dir)
        entry = res.residents.get(member) or res.guests.get(member) or {}
        lp = entry.get("lobes_path")
        try:
            path = _export.local_lobes_path(str(lp.value) if lp and lp.value else "")
        except _export.ExportError as exc:
            raise ModelGearError(
                code=EXIT_USER_ERROR, message=str(exc), remediation="set lobes_path"
            ) from exc
        return Target(local=True, lobes_path=path)
    res = _residents.load(base_dir)
    entry = res.residents.get(member) or res.guests.get(member)
    if entry is None:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"member {member!r} is not in residents.toml under {base_dir}",
            remediation="pass --base-dir (where residents.toml lives) or --local",
        )

    def val(name: str) -> str:
        f = entry.get(name)
        return str(f.value) if f is not None and f.value not in (None, "") else ""

    if not val("ssh") or not val("lobes_path"):
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"member {member!r} has no ssh target or lobes_path (needs_you)",
            remediation="fill the ssh and lobes_path fields in residents.toml",
        )
    return Target(ssh_target=val("ssh"), user=val("user"), lobes_path=val("lobes_path"))


def cmd_lobe_export(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    target = _target_for(args.base_dir, args.member, args.local)
    try:
        bundle = _export.export_member(args.roles, target, member=args.member)
    except _export.ExportError as exc:
        raise ModelGearError(
            code=EXIT_ENV_ERROR, message=str(exc), remediation="check ssh access to the member"
        ) from exc
    files = bundle.paths()
    roles = {
        e.role: {"status": e.status, "service": e.service, "reason": e.reason, "notes": e.notes}
        for e in bundle.exports
    }
    out_dir = Path(args.out)
    if args.apply:
        bundle.write(out_dir)
    payload = {
        "dry_run": not args.apply,
        "member": args.member,
        "out": str(out_dir),
        "roles": roles,
        "files": files,
    }
    if json_mode:
        emit_result(payload, json_mode=True)
    else:
        lines = [
            f"{e.role}: {e.status}" + (f" ({e.reason})" if e.reason else "") for e in bundle.exports
        ]
        lines += [f"  {f}" for f in files]
        emit_result("\n".join(lines), json_mode=False)
        if args.apply:
            emit_diagnostic(f"wrote {len(files)} files under {out_dir}")
        else:
            emit_diagnostic("DRY RUN — re-run with --apply to write the bundle.")
    return 0


def _no_verb(args: argparse.Namespace) -> None:
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message="lobe needs a verb",
        remediation="try: lobes lobe export <role> --member <name>",
    )


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("lobe", help="Lobe tooling: export a lobe's build recipe from a member.")
    p.add_argument("--json", action="store_true", help=_JSON_HELP)
    p.set_defaults(func=_no_verb)
    noun = p.add_subparsers(dest="lobe_command", parser_class=type(p))
    ex = noun.add_parser(
        "export",
        help="Export role(s) from a member: Dockerfile + inputs + compose service, "
        "or image + run command (image-only). Dry-run; --apply writes.",
    )
    ex.add_argument("roles", nargs="+", metavar="role", help="Role or lane name, e.g. stt-en.")
    ex.add_argument("--member", required=True, help="Member name in residents.toml.")
    ex.add_argument("--base-dir", default=".", help="Dir holding residents.toml (default: .).")
    ex.add_argument("--out", default="lobe-export", help="Output dir (default: ./lobe-export).")
    ex.add_argument("--local", action="store_true", help="Run on this box instead of ssh.")
    ex.add_argument("--apply", action="store_true", help="Write the bundle.")
    ex.add_argument("--json", action="store_true", help=_JSON_HELP)
    ex.set_defaults(func=cmd_lobe_export)
