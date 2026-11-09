"""``lobes rollout plan|apply|status|rollback`` — the gradual fleet rollout verb.

Wires the ``lobes.rollout`` modules together (gradual-fleet-rollout, t13); it
re-implements none of them:

* ``plan <version>`` — read-only. Members come from ``residents.toml`` in the
  deployment dir; each member's own gateway (its mesh-roster origin, or
  ``127.0.0.1`` for this box) answers ``/health`` + ``/capabilities``; its
  deployed compose files are read over SSH for drift. Prints waves (canary
  first, this box last), skips (guests, current, offline, downgrades), drift,
  lock-affected members and each member's addressing. Writes nothing.
* ``apply <version>`` — dry-run (the plan plus what would run) unless
  ``--apply``. With ``--apply``: refuse on unpatched drift, preflight the
  release in the gateway base image, then one member at a time: member step
  (``_member_step``) then verification through that member's gateway
  (``_verify``). Halts at the first failure with the member, the failed check
  and the exact rollback command; never rolls back by itself. Every step goes
  to the journal, and a re-run resumes from it (a verified member is not
  redone).
* ``status`` — read-only journal projection, with the resume point.
* ``rollback --member <m>`` — dry-run (the actions) unless ``--apply``.

Drift (gateway env keys the target template passes that a member's deployed
compose does not): ``plan`` reports it; ``apply --apply`` refuses a drifted
member unless ``--patch-drift``, which backs up and patches only
``docker-compose.override.yml`` (spec decision, 2026-11-09).

No secret value is printed or journaled: every gateway key read over SSH, the
mesh key and secret-looking local ``.env`` values are scrubbed from all output.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from lobes import __version__
from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.rollout import _drift, _residents, _transport
from lobes.rollout._journal import SKIPPED_OFFLINE, Journal
from lobes.rollout._member_lock import LockHeld, member_lock
from lobes.rollout._member_step import run_member_step
from lobes.rollout._observe import MemberObservation, observe_member
from lobes.rollout._plan import PLANNED, SKIPPED_CURRENT
from lobes.rollout._plan import SKIPPED_OFFLINE as PLAN_OFFLINE
from lobes.rollout._plan import build_plan
from lobes.rollout._preflight import check_release
from lobes.rollout._redact import redact
from lobes.rollout._rollback import backup_from_journal, rollback, rollback_command
from lobes.rollout._verify import verify_member
from lobes.runtime import _compose, _env

_JSON_HELP = "Emit structured JSON."
_COMPOSE_DIR_HELP = "Deployment dir holding residents.toml and the journal (default: ~/.lobes)."
_PORT_HELP = "This box's gateway port (default: VLLM_PORT in .env)."
_TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "fleet" / "docker-compose.yml"
_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)$")
_FILE_MARK = re.compile(r"(?m)^@@LOBES-FILE (\S+)\n?")
_SSH_TIMEOUT = 30.0
_STEPS = (
    "lock",
    "window check",
    "back up .env",
    "install the lobes CLI",
    "pin MODEL_GEAR_VERSION",
    "tag the current gateway image",
    "rebuild the gateway",
    "wait for health",
    "verify through the member's gateway",
)

# $1 = deployment dir ("" -> $HOME/.lobes). Scripts take args, never interpolation.
_DIR = 'd="${1:-$HOME/.lobes}"; '
_READ_KEY = (
    _DIR + "v=$(sed -n 's/^GATEWAY_API_KEY=//p' \"$d/.env\" 2>/dev/null | tail -n 1); "
    '[ -n "$v" ] || v=$(sed -n \'s/^CULTURE_VLLM_API_KEY=//p\' "$d/.env" 2>/dev/null '
    "| tail -n 1); printf '%s' \"$v\""
)
_FETCH_FILES = (
    _DIR + 'for f in "$d"/docker-compose*.yml; do [ -f "$f" ] || continue; '
    'printf \'\\n@@LOBES-FILE %s\\n\' "$(basename "$f")"; cat "$f"; done; '
    "[ -f \"$d/deployment.lock.toml\" ] && printf '\\n@@LOBES-FILE deployment.lock.toml\\n'; "
    "exit 0"
)
# $2 = backup file name ("" = none), $3 = new override text.
_WRITE_OVERRIDE = (
    _DIR + 'f="$d/docker-compose.override.yml"; '
    'if [ -n "$2" ] && [ -f "$f" ]; then cp -p "$f" "$d/$2" || exit 1; fi; '
    'printf \'%s\' "$3" > "$f"'
)


@dataclass
class Seams:
    """Injected I/O. Tests replace :data:`SEAMS`; nothing else is patched."""

    opener: Callable[..., Any] = urllib.request.urlopen  # nosec B310
    runner: Callable[..., Any] = _transport.default_runner
    preflight_runner: Optional[Callable[..., Any]] = None
    sleep: Callable[[float], Any] = time.sleep


SEAMS = Seams()


class _TargetJournal(Journal):
    """The journal restricted to one rollout target, so an earlier rollout's
    verified records never mark a member done for this one."""

    def __init__(self, deploy_dir, target: str, redact_fn: Callable[[str], str]):
        super().__init__(deploy_dir, redact=redact_fn)
        self.target = target

    def records(self) -> list:
        return [r for r in super().records() if r.get("target") == self.target]


# ---------------------------------------------------------------------------
# context: residents + mesh + per-member reads (all read-only)
# ---------------------------------------------------------------------------


@dataclass
class _Member:
    name: str
    ssh: Optional[str] = None
    user: Optional[str] = None
    port: Optional[int] = None
    lobes_path: Optional[str] = None
    has_key: bool = False
    needs_you: list = field(default_factory=list)
    is_local: bool = False
    origin: Optional[str] = None
    key: Optional[str] = field(default=None, repr=False)
    obs: Optional[MemberObservation] = None
    files: dict = field(default_factory=dict, repr=False)
    drift: Optional[_drift.DriftReport] = None
    note: str = ""
    patch_diff: str = ""

    def target(self) -> dict:
        return {
            "ssh_target": self.ssh or "",
            "user": self.user or "",
            "lobes_path": self.lobes_path or "",
            "local": self.is_local,
        }

    def addressing(self) -> dict:
        return {
            "ssh": self.ssh,
            "user": self.user,
            "port": self.port,
            "lobes_path": self.lobes_path,
            "origin": self.origin,
            "gateway_key": self.has_key,
            "needs_you": list(self.needs_you),
            "note": self.note,
        }


@dataclass
class _Ctx:
    deploy_dir: Path
    target: str
    local: str
    port: int
    secrets: list = field(repr=False)
    members: dict
    residents: Any
    template_text: str = field(repr=False)
    template_source: str
    plan: Any = None

    def clean(self, text: str) -> str:
        return redact(text or "", self.secrets)


def _scrub(value: Any, clean: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, dict):
        return {k: _scrub(v, clean) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v, clean) for v in value]
    return value


def _emit(data: Any, json_mode: bool, clean: Callable[[str], str]) -> None:
    emit_result(_scrub(data, clean), json_mode=json_mode)


def _local_secrets(env: dict) -> list:
    # Very short values (flags like FOO_KEY=1) would blank unrelated text; the
    # name-pattern redaction in _redact still covers their KEY=value form.
    return [v for k, v in env.items() if v and len(v) >= 8 and _SECRET_NAME.search(k.upper())]


def _deploy_dir(args) -> Path:
    return Path(_runtime_ops.deployment_dir(args))


def _local_env(deploy_dir: Path) -> dict:
    return _env.read_env_file(deploy_dir / _compose.ENV_FILE)


def _load_residents(deploy_dir: Path):
    try:
        res = _residents.load(deploy_dir)
    except (ValueError, OSError) as exc:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"cannot read {deploy_dir / _residents.DEFAULT_NAME}: {exc}",
            remediation="fix the residents file (TOML) and re-run",
        ) from exc
    if not res.residents:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"no residents in {deploy_dir / _residents.DEFAULT_NAME}",
            remediation="add the members you manage: lobes mesh resident add <name>",
        )
    return res


def _member_from_entry(name: str, entry: dict, local: str) -> _Member:
    def val(f):
        fld = entry.get(f)
        return None if fld is None or fld.state != _residents.DETECTED else fld.value

    m = _Member(name, is_local=(name == local))
    m.ssh, m.user, m.lobes_path = val("ssh"), val("user"), val("lobes_path")
    port = val("port")
    m.port = (
        int(port) if isinstance(port, int) or (isinstance(port, str) and port.isdigit()) else None
    )
    m.has_key = bool(val("gateway_key"))
    # gateway_key = false is a detected answer ("no key"), not a gap.
    m.needs_you = [f for f in _residents.FIELDS if val(f) is None or val(f) == ""]
    return m


def _roster_origins(port: int, op_key: Optional[str], mesh_key: Optional[str]) -> dict:
    if not mesh_key:
        return {}
    obs = observe_member(
        f"http://127.0.0.1:{port}", gateway_key=op_key, mesh_key=mesh_key, opener=SEAMS.opener
    )
    return {
        str(m["name"]): str(m["origin"])
        for m in obs.roster
        if m.get("name") and isinstance(m.get("origin"), str) and m.get("origin")
    }


def _read_key(m: _Member) -> Optional[str]:
    try:
        raw = _transport.read_remote_value(
            m.target(), ["sh", "-c", _READ_KEY, "sh", ""], runner=SEAMS.runner
        )
    except RuntimeError:
        m.note = "gateway key unreadable over ssh"
        return None
    return raw.strip().strip("\"'").strip() or None


def _fetch_files(m: _Member) -> Optional[dict]:
    try:
        raw = _transport.read_remote_value(
            m.target(), ["sh", "-c", _FETCH_FILES, "sh", ""], runner=SEAMS.runner
        )
    except RuntimeError:
        return None
    parts = _FILE_MARK.split(raw)
    files = {}
    for i in range(1, len(parts) - 1, 2):
        files[parts[i]] = parts[i + 1]
    return files


def _reachable_by_ssh(m: _Member) -> bool:
    return bool(m.lobes_path and str(m.lobes_path).startswith("/") and (m.ssh or m.is_local))


def _gather(args, target: str) -> _Ctx:
    deploy_dir = _deploy_dir(args)
    env = _local_env(deploy_dir)
    local = (getattr(args, "local", None) or env.get("LOBES_MESH_NAME") or "").strip()
    port = _runtime_ops.resolve_port(args, deploy_dir / _compose.ENV_FILE)
    mesh_key = (env.get("LOBES_MESH_KEY") or "").strip() or None
    op_key = (env.get("GATEWAY_API_KEY") or env.get("CULTURE_VLLM_API_KEY") or "").strip() or None
    secrets = _local_secrets(env)
    residents = _load_residents(deploy_dir)

    template_arg = getattr(args, "template", None)
    tpath = Path(template_arg).expanduser() if template_arg else _TEMPLATE
    try:
        template_text = tpath.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"cannot read compose template {tpath}: {exc}",
            remediation="pass --template <path to the target release's fleet docker-compose.yml>",
        ) from exc
    if not template_arg and __version__ != target:
        emit_diagnostic(
            f"note: drift is checked against lobes {__version__}'s gateway template, not "
            f"{target}'s; pass --template <target docker-compose.yml> to check the target's."
        )

    origins = _roster_origins(port, op_key, mesh_key)
    members: dict = {}
    for name, entry in residents.residents.items():
        m = _member_from_entry(name, entry, local)
        members[name] = m
        if m.is_local:
            m.origin = f"http://127.0.0.1:{port}"
        else:
            m.origin = origins.get(name)
        if not _reachable_by_ssh(m):
            m.note = "residents.toml needs: " + ", ".join(m.needs_you or ["lobes_path"])
            continue
        if not m.origin:
            m.note = "not in the mesh roster (no gateway origin)"
            continue
        if m.has_key:
            m.key = _read_key(m)
            if m.key:
                secrets.append(m.key)
        m.obs = observe_member(m.origin, gateway_key=m.key, opener=SEAMS.opener)
        if not m.obs.reachable:
            m.note = "gateway unreachable"
            continue
        files = _fetch_files(m)
        if files is None:
            m.note = "compose files unreadable over ssh (drift unknown)"
            continue
        m.files = files
        m.drift = _drift.detect_drift(template_text, files)

    ctx = _Ctx(
        deploy_dir=deploy_dir,
        target=target,
        local=local,
        port=port,
        secrets=secrets,
        members=members,
        residents=residents,
        template_text=template_text,
        template_source=str(tpath),
    )
    observations = {n: m.obs for n, m in members.items() if m.obs is not None}
    drift = {n: m.drift for n, m in members.items() if m.drift is not None}
    try:
        ctx.plan = build_plan(
            residents,
            observations,
            drift,
            target,
            local,
            canary=getattr(args, "canary", None),
            allow_downgrade=bool(getattr(args, "allow_downgrade", False)),
            wave_size=int(getattr(args, "wave_size", 1) or 1),
        )
    except ValueError as exc:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=str(exc),
            remediation="check the target version and --canary (a planned resident)",
        ) from exc
    for mp in ctx.plan.members:
        m = members.get(mp.name)
        if m is not None and mp.status == PLAN_OFFLINE and m.note:
            mp.reason = m.note
    return ctx


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _plan_dict(ctx: _Ctx) -> dict:
    plan = ctx.plan
    members = []
    for mp in plan.members:
        d = asdict(mp)
        m = ctx.members.get(mp.name)
        if m is not None:
            d.update(m.addressing())
        members.append(d)
    lock = sorted(n for n, m in ctx.members.items() if m.drift and m.drift.lock_affected)
    return {
        "target": plan.target,
        "local": plan.local,
        "canary": plan.canary,
        "waves": [asdict(w) for w in plan.waves],
        "members": members,
        "drift": {
            n: m.drift.missing for n, m in ctx.members.items() if m.drift and m.drift.drifted
        },
        "lock_affected": lock,
        "template": ctx.template_source,
    }


def _wave_label(ctx: _Ctx, wave) -> str:
    tags = []
    if ctx.plan.canary in wave.members and ctx.plan.canary != ctx.local:
        tags.append("canary")
    if ctx.local in wave.members:
        tags.append("operator box, last")
    return f"wave {wave.index}" + (f" ({', '.join(tags)})" if tags else "")


def _plan_text(ctx: _Ctx) -> str:
    plan = ctx.plan
    lines = [
        f"rollout plan: lobes-cli {plan.target}"
        + (f"  (this box: {ctx.local})" if ctx.local else "")
    ]
    if not plan.waves:
        lines.append("  nothing to upgrade")
    for w in plan.waves:
        for name in w.members:
            mp = plan.member(name)
            roles = ", ".join(mp.unique_roles) if mp.unique_roles else "-"
            lines.append(
                f"  {_wave_label(ctx, w):<28} {name:<12} {mp.running} -> {plan.target}"
                f"  roles: {roles}"
            )
    skipped = [mp for mp in plan.members if mp.status != PLANNED]
    if skipped:
        lines.append("skipped:")
        for mp in skipped:
            extra = " (pending)" if mp.pending else ""
            lines.append(f"  {mp.name:<12} {mp.status:<18} {mp.reason}{extra}")
    drifted = [(n, m.drift) for n, m in ctx.members.items() if m.drift and m.drift.drifted]
    if drifted:
        lines.append(
            "drift (gateway env keys the target passes, missing from the deployed compose):"
        )
        for n, d in drifted:
            lines.append(f"  {n}: {', '.join(d.missing)}")
    lock = [n for n, m in ctx.members.items() if m.drift and m.drift.lock_affected]
    if lock:
        lines.append(
            "lock-affected (deployment.lock.toml will report lock_drift after the repin; "
            "re-capture is yours): " + ", ".join(lock)
        )
    lines.append("addressing:")
    for n, m in ctx.members.items():
        dest = f"{m.user}@{m.ssh}" if m.user and m.ssh else (m.ssh or "-")
        if m.is_local:
            dest = "local"
        lines.append(
            f"  {n:<12} {dest:<24} port {m.port or '-':<6} {m.lobes_path or '-'}"
            + (f"  [{m.note}]" if m.note else "")
        )
    return "\n".join(lines)


def _would_refuse(ctx: _Ctx) -> dict:
    out = {}
    for mp in ctx.plan.members:
        m = ctx.members.get(mp.name)
        if mp.status == PLANNED and m is not None and m.drift and m.drift.drifted:
            out[mp.name] = list(m.drift.missing)
    return out


def _drift_error(refuse: dict) -> ModelGearError:
    detail = "; ".join(f"{n}: {', '.join(keys)}" for n, keys in refuse.items())
    return ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"compose drift on planned members, no member touched ({detail})",
        remediation=(
            "re-run with --patch-drift to back up and patch docker-compose.override.yml, "
            "or add the keys to the member's override by hand"
        ),
    )


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    json_mode = bool(args.json)
    ctx = _gather(args, args.version)
    data = _plan_dict(ctx)
    data.update({"dry_run": True, "state": "plan", "preflight": {"ran": False}})
    if json_mode:
        _emit(data, True, ctx.clean)
    else:
        _emit(
            _plan_text(ctx) + "\npreflight: not run (apply --apply runs it first)", False, ctx.clean
        )
    return 0


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def _up_roles(obs: Optional[MemberObservation]) -> list:
    if obs is None:
        return []
    return sorted(r for r, c in obs.capabilities.items() if c.get("feasible") and c.get("ready"))


def _before_from_journal(journal: Journal, name: str, origin: str) -> Optional[MemberObservation]:
    for r in journal.records():
        if r.get("member") == name and r.get("step") == "observe" and r.get("result") == "ok":
            d = r.get("detail") or {}
            roles = d.get("up_roles") or []
            return MemberObservation(
                origin=origin,
                version=r.get("current"),
                capabilities={x: {"feasible": True, "ready": True} for x in roles},
                reachable=True,
            )
    return None


def _order(ctx: _Ctx, journal: Journal) -> list:
    """Wave order, plus members a previous run left in progress at the target
    (their /health already reports it, so the plan calls them current)."""
    order = [n for w in ctx.plan.waves for n in w.members]
    status = journal.status()
    resumed = [
        mp.name
        for mp in ctx.plan.members
        if mp.status == SKIPPED_CURRENT
        and mp.name in status
        and status[mp.name].state == "in_progress"
    ]
    head = [n for n in resumed if n != ctx.local]
    tail = [n for n in order if n != ctx.local]
    last = [ctx.local] if ctx.local in order or ctx.local in resumed else []
    return head + tail + last


def _patch_override(ctx: _Ctx, m: _Member, journal: Journal) -> Optional[str]:
    """Back up + patch the member's override under its lock. Returns an error or None."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    patch = _drift.patch_override(ctx.template_text, m.files, now=stamp)
    if not patch.new_text:
        return None
    try:
        with member_lock(m.target(), runner=SEAMS.runner):
            r = _transport.run(
                m.target(),
                ["sh", "-c", _WRITE_OVERRIDE, "sh", "", patch.backup_name or "", patch.new_text],
                timeout=_SSH_TIMEOUT,
                runner=SEAMS.runner,
                secrets=ctx.secrets,
            )
    except LockHeld as exc:
        return str(exc)
    except RuntimeError as exc:
        return str(exc)
    if not r.ok:
        journal.append(
            m.name, "patch_override", "failed", target=ctx.target, detail={"exit": r.returncode}
        )
        return f"override patch failed (exit {r.returncode})"
    journal.append(
        m.name,
        "patch_override",
        "ok",
        target=ctx.target,
        detail={"added": patch.added, "backup": patch.backup_name},
    )
    emit_diagnostic(ctx.clean(f"{m.name}: patched docker-compose.override.yml"))
    m.patch_diff = ctx.clean(patch.diff)
    return None


class _Halt(Exception):
    def __init__(self, member: str, check: str, role: Optional[str], detail: str, rollback: bool):
        super().__init__(detail)
        self.member, self.check, self.role, self.detail = member, check, role, detail
        self.rollback = rollback_command(member) if rollback else None
        self.diff = ""

    def to_dict(self) -> dict:
        return {
            "member": self.member,
            "check": self.check,
            "role": self.role,
            "detail": self.detail,
            "rollback": self.rollback,
            "diff": self.diff,
        }


def _upgrade_member(ctx: _Ctx, m: _Member, journal: Journal, args) -> dict:
    try:
        return _upgrade_member_inner(ctx, m, journal, args)
    except _Halt as exc:
        exc.diff = m.patch_diff  # an override patched before the halt stays visible
        raise


def _upgrade_member_inner(ctx: _Ctx, m: _Member, journal: Journal, args) -> dict:
    before = _before_from_journal(journal, m.name, m.origin or "")
    if before is None:
        before = m.obs
        journal.append(
            m.name,
            "observe",
            "ok",
            target=ctx.target,
            current=before.version if before else None,
            detail={"up_roles": _up_roles(before)},
        )
    out: dict = {"member": m.name}
    if args.patch_drift and m.drift and m.drift.drifted:
        err = _patch_override(ctx, m, journal)
        if err:
            raise _Halt(m.name, "patch_override", None, err, rollback=False)
        out["patched"] = list(m.drift.missing)
        out["diff"] = m.patch_diff
    step = run_member_step(
        m.name,
        m.target(),
        ctx.target,
        allow_window=bool(args.allow_window),
        journal=journal.append,
        runner=SEAMS.runner,
        sleep=SEAMS.sleep,
        secrets=ctx.secrets,
    )
    if step.skipped:
        out.update(result="skipped", reason=step.reason)
        return out
    if not step.ok:
        raise _Halt(m.name, step.failed_step, None, step.error, rollback=step.changed)
    vr = verify_member(
        m.origin or "",
        target_version=ctx.target,
        before=before,
        gateway_key=m.key,
        opener=SEAMS.opener,
    )
    failures = [{"check": f.check, "role": f.role, "detail": f.detail} for f in vr.failures]
    journal.append(
        m.name,
        "verify",
        "ok" if vr.ok else "failed",
        target=ctx.target,
        current=ctx.target if vr.ok else None,
        detail={"probed": vr.probed, "not_probed": vr.not_probed, "failures": failures},
    )
    if not vr.ok:
        f = vr.failures[0]
        raise _Halt(m.name, f.check, f.role, f.detail, rollback=True)
    out.update(result="verified", changed=step.changed, probed=vr.probed)
    if m.drift and m.drift.lock_affected:
        out["note"] = "deployment.lock.toml now reports lock_drift; re-capture the lock"
    return out


def _apply_text(ctx: _Ctx, data: dict) -> str:
    lines = [
        f"preflight: lobes-cli {ctx.target} installable ({data['preflight']['attempts']} tries)"
    ]
    for r in data["results"]:
        line = f"  {r['member']:<12} {r['result']}"
        if r.get("reason"):
            line += f": {r['reason']}"
        if r.get("note"):
            line += f" — {r['note']}"
        lines.append(line)
        if r.get("diff"):
            lines.append(r["diff"].rstrip("\n"))
    if data["pending"]:
        lines.append("pending (offline, upgraded by a later apply): " + ", ".join(data["pending"]))
    if data["state"] == "halted":
        h = data["halt"]
        role = f" (role {h['role']})" if h["role"] else ""
        if h.get("diff"):
            lines.append(h["diff"].rstrip("\n"))
        lines.append(f"{h['member']} halted: {h['check']}{role}: {h['detail']}")
        lines.append("others untouched after it; nothing was rolled back")
        if h["rollback"]:
            lines.append(f"roll back with: {h['rollback']}")
    else:
        lines.append(f"rollout {data['state']}")
    return "\n".join(lines)


def cmd_apply(args: argparse.Namespace) -> int:
    json_mode = bool(args.json)
    ctx = _gather(args, args.version)
    refuse = _would_refuse(ctx) if not args.patch_drift else {}
    if not args.apply:
        data = _plan_dict(ctx)
        data.update(
            {
                "dry_run": True,
                "state": "plan",
                "preflight": {"ran": False},
                "would_refuse": refuse,
                "steps": list(_STEPS),
            }
        )
        if json_mode:
            _emit(data, True, ctx.clean)
            return 0
        text = _plan_text(ctx)
        text += (
            f"\nDRY RUN — would preflight lobes-cli=={ctx.target} in the gateway base image, "
            "then per member: " + ", ".join(_STEPS) + "."
        )
        if refuse:
            text += "\n" + _drift_error(refuse).message + "; add --patch-drift"
        text += "\nRe-run with --apply to execute."
        _emit(text, False, ctx.clean)
        return 0

    if refuse:
        raise _drift_error(refuse)
    pre = check_release(ctx.target, runner=SEAMS.preflight_runner, sleep=SEAMS.sleep)
    if not pre.ok:
        raise ModelGearError(
            code=EXIT_ENV_ERROR, message=pre.hint, remediation="re-run apply once it resolves"
        )

    journal = _TargetJournal(ctx.deploy_dir, ctx.target, ctx.clean)
    status = journal.status()
    pending = []
    for mp in ctx.plan.members:
        if mp.status == PLAN_OFFLINE:
            pending.append(mp.name)
            if not (mp.name in status and status[mp.name].done):
                journal.append(
                    mp.name,
                    "plan",
                    SKIPPED_OFFLINE,
                    target=ctx.target,
                    detail={"reason": mp.reason},
                )
    data: dict = {
        "dry_run": False,
        "target": ctx.target,
        "preflight": {"ok": True, "attempts": pre.attempts, "base_image": pre.base_image},
        "results": [],
        "pending": pending,
        "halt": None,
    }
    halt: Optional[_Halt] = None
    for name in _order(ctx, journal):
        st = journal.status().get(name)
        if st is not None and st.done:
            data["results"].append(
                {"member": name, "result": "verified", "note": "verified earlier"}
            )
            continue
        try:
            data["results"].append(_upgrade_member(ctx, ctx.members[name], journal, args))
        except _Halt as exc:
            halt = exc
            break
    if halt is not None:
        data["state"] = "halted"
        data["halt"] = halt.to_dict()
    elif any(r["result"] == "skipped" for r in data["results"]):
        data["state"] = "incomplete"
    else:
        data["state"] = "complete"
    _emit(data if json_mode else _apply_text(ctx, data), json_mode, ctx.clean)
    if halt is not None:
        role = f" (role {halt.role})" if halt.role else ""
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=ctx.clean(
                f"rollout halted at {halt.member}: {halt.check}{role}: {halt.detail}"
            ),
            remediation=(
                f"inspect the member, then roll back: {halt.rollback}"
                if halt.rollback
                else "fix the cause and re-run apply; it resumes from the journal"
            ),
        )
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _soft_deploy_dir(args) -> Path:
    try:
        return _deploy_dir(args)
    except ModelGearError:
        explicit = getattr(args, "compose_dir", None)
        return Path(explicit).expanduser() if explicit else _compose.default_deployment_dir()


def _latest_target(journal: Journal, member: Optional[str] = None) -> Optional[str]:
    target = None
    for r in journal.records():
        if member is None or r.get("member") == member:
            target = r.get("target") or target
    return target


def cmd_status(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    deploy_dir = _soft_deploy_dir(args)
    env = _local_env(deploy_dir) if (deploy_dir / _compose.ENV_FILE).is_file() else {}
    secrets = _local_secrets(env)

    def clean(t):
        return redact(t or "", secrets)

    base = Journal(deploy_dir, redact=clean)
    target = getattr(args, "target", None) or _latest_target(base)
    if target is None:
        data = {"target": None, "journal": str(base.path), "members": {}, "resume": None}
        _emit(data if json_mode else f"no rollout journal at {base.path}", json_mode, clean)
        return 0
    journal = _TargetJournal(deploy_dir, target, clean)
    status = journal.status()
    rp = journal.resume_point()
    data = {
        "target": target,
        "journal": str(journal.path),
        "members": {n: asdict(s) for n, s in status.items()},
        "resume": asdict(rp) if rp else None,
    }
    if json_mode:
        _emit(data, True, clean)
        return 0
    lines = [f"rollout status: lobes-cli {target}  (journal: {journal.path})"]
    for n, s in status.items():
        lines.append(
            f"  {n:<12} {s.state:<12} {s.current or '-':<16} "
            f"{s.last_step or '-'} {s.last_result or ''}"
        )
    if rp:
        lines.append(
            f"resume: {rp.member} at {rp.step} (re-run lobes rollout apply {target} --apply)"
        )
    _emit("\n".join(lines), False, clean)
    return 0


def _no_verb(args: argparse.Namespace) -> int:
    return cmd_status(args)


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def _old_version(journal: Journal, member: str) -> Optional[str]:
    for r in journal.records():
        if r.get("member") != member:
            continue
        if r.get("step") == "observe" and r.get("result") == "ok" and r.get("current"):
            return r["current"]
    for r in journal.records():
        d = r.get("detail") if isinstance(r.get("detail"), dict) else {}
        if r.get("member") == member and r.get("step") == "tag_image" and d.get("tag"):
            return str(d["tag"]).rsplit(":", 1)[-1]
    return None


def cmd_rollback(args: argparse.Namespace) -> int:
    json_mode = bool(args.json)
    deploy_dir = _deploy_dir(args)
    env = _local_env(deploy_dir)
    secrets = _local_secrets(env)

    def clean(t):
        return redact(t or "", secrets)

    residents = _load_residents(deploy_dir)
    member = args.member
    entry = residents.residents.get(member)
    if entry is None:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"{member!r} is not a resident in {deploy_dir / _residents.DEFAULT_NAME}",
            remediation="pass a resident name (see lobes rollout status)",
        )
    local = (env.get("LOBES_MESH_NAME") or "").strip()
    m = _member_from_entry(member, entry, local)
    base = Journal(deploy_dir, redact=clean)
    target = _latest_target(base, member)
    journal = _TargetJournal(deploy_dir, target or "", clean)
    backup = args.backup or (backup_from_journal(journal, member) if target else None)
    old = args.old_version or (_old_version(journal, member) if target else None)
    if not backup or not old:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=(
                f"no pre-apply .env backup / previous version recorded for {member} "
                f"in {base.path}"
            ),
            remediation="pass --backup <remote .env backup path> and --old-version <version>",
        )
    if not m.lobes_path or not str(m.lobes_path).startswith("/") or not (m.ssh or m.is_local):
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"{member} has no usable ssh target / lobes_path in residents.toml",
            remediation="fix the needs-you fields in residents.toml",
        )
    res = rollback(member, m.target(), old, backup, apply=bool(args.apply), runner=SEAMS.runner)
    data = {
        "member": member,
        "dry_run": not args.apply,
        "old_version": old,
        "backup": backup,
        "actions": [{"description": a.description, "command": a.command} for a in res.actions],
        "applied": res.applied,
        "ok": res.ok,
        "error": res.error,
        "health_version": None,
        "command": rollback_command(member),
    }
    if not res.ok and not res.actions:
        raise ModelGearError(
            code=EXIT_USER_ERROR, message=res.error, remediation="check the journal"
        )
    if args.apply:
        journal.append(
            member,
            "rollback",
            "ok" if res.ok else "failed",
            target=target,
            current=old if res.ok else None,
            detail={"error": res.error},
        )
        port = _runtime_ops.resolve_port(args, deploy_dir / _compose.ENV_FILE)
        origin = (
            f"http://127.0.0.1:{port}"
            if m.is_local
            else _roster_origins(
                port,
                (env.get("GATEWAY_API_KEY") or "").strip() or None,
                (env.get("LOBES_MESH_KEY") or "").strip() or None,
            ).get(member)
        )
        if origin:
            data["health_version"] = observe_member(origin, opener=SEAMS.opener).version
    if json_mode:
        _emit(data, True, clean)
    else:
        head = (
            f"DRY RUN — rollback {member} to {old}:"
            if not args.apply
            else f"rolled back {member} to {old}: {'ok' if res.ok else 'FAILED ' + res.error}"
        )
        lines = [head]
        lines += [
            f"  {i}. {a['description']}: {a['command']}" for i, a in enumerate(data["actions"], 1)
        ]
        if args.apply:
            lines.append(f"/health now reports: {data['health_version'] or 'unknown'}")
        else:
            lines.append(f"Re-run with --apply to execute: {data['command']}")
        _emit("\n".join(lines), False, clean)
    if args.apply and not res.ok:
        raise ModelGearError(
            code=EXIT_ENV_ERROR, message=clean(res.error), remediation="inspect the member"
        )
    return 0


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--compose-dir", help=_COMPOSE_DIR_HELP)
    p.add_argument("--json", action="store_true", help=_JSON_HELP)


def _add_planning(p: argparse.ArgumentParser) -> None:
    _add_common(p)
    p.add_argument("version", help="Target lobes-cli version, e.g. 0.95.0.")
    p.add_argument("--port", type=int, help=_PORT_HELP)
    p.add_argument("--local", help="This box's mesh name (default: LOBES_MESH_NAME in .env).")
    p.add_argument(
        "--canary", help="Member for wave 1 (default: fewest roles hosted nowhere else)."
    )
    p.add_argument("--wave-size", type=int, default=1, help="Members per wave (default: 1).")
    p.add_argument(
        "--allow-downgrade", action="store_true", help="Allow a target older than a member."
    )
    p.add_argument(
        "--template",
        help="The target release's fleet docker-compose.yml for drift (default: this CLI's own).",
    )


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "rollout",
        help="Gradual mesh upgrade: plan / apply / status / rollback. Dry-run; --apply.",
    )
    _add_common(p)
    p.set_defaults(func=_no_verb, json=False, target=None)
    noun = p.add_subparsers(dest="rollout_command", parser_class=type(p))

    plan = noun.add_parser("plan", help="Read-only: waves, skips, drift and addressing per member.")
    _add_planning(plan)
    plan.set_defaults(func=cmd_plan)

    apply = noun.add_parser(
        "apply",
        help="Preflight, then upgrade and verify one wave at a time. Dry-run; --apply.",
    )
    _add_planning(apply)
    apply.add_argument("--apply", action="store_true", help="Actually upgrade members.")
    apply.add_argument(
        "--patch-drift",
        action="store_true",
        help="Back up and patch docker-compose.override.yml with missing gateway env keys.",
    )
    apply.add_argument(
        "--allow-window",
        action="store_true",
        help="Recreate a gateway even during an active benchmark window.",
    )
    apply.set_defaults(func=cmd_apply)

    status = noun.add_parser("status", help="Read-only: per-member state from the rollout journal.")
    _add_common(status)
    status.add_argument("--target", help="Rollout target to show (default: the latest).")
    status.set_defaults(func=cmd_status)

    rb = noun.add_parser(
        "rollback",
        help="Restore a member's .env and previous gateway image. Dry-run; --apply.",
    )
    _add_common(rb)
    rb.add_argument("--member", required=True, help="Resident to roll back.")
    rb.add_argument("--port", type=int, help=_PORT_HELP)
    rb.add_argument(
        "--backup", help="Remote pre-apply .env backup path (default: from the journal)."
    )
    rb.add_argument("--old-version", help="Previous version (default: from the journal).")
    rb.add_argument("--apply", action="store_true", help="Actually roll back.")
    rb.set_defaults(func=cmd_rollback)
