"""``lobes mesh status|request|approve <name> [--for <duration>]|revoke <name>``
— the CLI surface for mesh-brain-join (issue "mesh-brain-join", task t10).

This module is a thin HTTP client of the ``/mesh/*`` routes already wired by
:mod:`lobes.gateway._mesh_routes` (task t6) — it never re-implements roster,
ledger, or wire-format logic. Registered as a noun exactly like ``lobes
fleet`` (``lobes/cli/_commands/fleet.py``): a bare ``lobes mesh`` is the
read-only ``status`` default, and every mutating sub-verb (``request``,
``approve``, ``revoke``) is dry-run by default — it prints the plan and
touches nothing — and only actually calls the gateway with ``--apply``.

Endpoint summary (see ``_mesh_routes.py`` for the authoritative behaviour):

* ``GET /mesh/roster`` — authenticated (join-key Bearer) member listing.
  ``lobes mesh status`` renders it as a table: name, origin, last-heartbeat
  age, approval expiry, verified/unverified/flapping, and roles. The pinned
  route today returns only ``{name, origin, capacity}`` per member — every
  other column is an ADDITIVE field this CLI renders when a payload carries
  it (a richer roster response, or a test fixture) and otherwise shows as
  ``-``/``unknown``, mirroring the tolerant-of-missing-fields convention
  already used in ``lobes.cli._commands.capabilities``.
* ``POST /mesh/join`` — keyless. ``lobes mesh request`` is the CLI's "ask to
  join" verb: it POSTs ``{name, origin, capacity}`` with NO Authorization
  header, so it works from a box that has no ``LOBES_MESH_KEY`` configured
  at all — the join key is asymmetric (only the ledger side needs it).
* ``POST /mesh/approve`` / ``POST /mesh/revoke`` — authenticated. Both need
  the operator's own ``LOBES_MESH_KEY`` (the join key) as a Bearer token;
  ``--apply`` without a configured key is refused with a clear remediation
  rather than sending an unauthenticated request that will 401.

Every write verb (``request``/``approve``/``revoke``) is strictly HTTP-only:
no docker/compose call, ever.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result

_JSON_HELP = "Emit structured JSON."
_COMPOSE_DIR_HELP = "Deployment dir (default: $LOBES_DIR or ~/.lobes)."
_PORT_HELP = "Gateway host port (default: VLLM_PORT in .env)."

# Bounded so an unreachable/foreign process on the resolved port degrades
# fast — this mirrors lobes.cli._commands.capabilities' own timeout policy.
_GATEWAY_TIMEOUT_SECONDS = 2.0
_DEFAULT_APPROVAL_SECONDS = 3600.0

_DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


# ---------------------------------------------------------------------------
# tiny stdlib HTTP helpers (no shared state with lobes.assess — mesh routes
# are a distinct JSON-in/JSON-out surface, not /v1/chat/completions)
# ---------------------------------------------------------------------------


def _get_json(url: str, path: str, headers: dict[str, str] | None, timeout: float) -> dict | None:
    """``GET`` *path*; ``None`` on any failure to get an authoritative 2xx JSON body.

    A 401 is re-raised (not folded into ``None``) so
    :func:`lobes.cli._runtime_ops.friendly_unauthorized_errors` can turn it
    into an actionable message — same convention as
    ``capabilities._fetch_gateway_capabilities``.
    """
    req = urllib.request.Request(url + path, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # local endpoint only
            if not (200 <= resp.status < 300):
                return None
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise
        return None
    except OSError:  # URLError (incl. other HTTPError codes) subclasses OSError
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _post_json(
    url: str,
    path: str,
    payload: dict,
    headers: dict[str, str] | None,
    timeout: float,
) -> tuple[int, dict | str]:
    """``POST`` a JSON body; returns ``(status, parsed_body_or_text)``.

    Raises :class:`urllib.error.HTTPError` on a non-2xx response (the caller
    decides how to translate that — a 400 is a validation message worth
    showing verbatim, a 401 is handled by
    :func:`lobes.cli._runtime_ops.friendly_unauthorized_errors`).
    """
    data = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url + path, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # local endpoint only
        raw = resp.read()
        status = resp.status
    try:
        return status, json.loads(raw)
    except (ValueError, TypeError):
        return status, raw.decode("utf-8", errors="replace")


def _mesh_key(env: dict[str, str]) -> str:
    """The operator's configured mesh join key, or ``""`` when unset."""
    return (env.get("LOBES_MESH_KEY") or "").strip()


def _key_headers(env: dict[str, str]) -> dict[str, str]:
    key = _mesh_key(env)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _parse_duration_seconds(raw: str) -> float:
    """Parse ``--for`` as seconds: a bare number, or ``<number><s|m|h|d>``.

    Raises :class:`ModelGearError` (user error) on anything else — never
    silently substitutes a guessed default for a malformed value the
    operator actually typed.
    """
    text = raw.strip().lower()
    if not text:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="--for must not be blank",
            remediation="pass a duration like '3600', '30m', '24h', or '7d'",
        )
    unit = text[-1]
    if unit in _DURATION_UNITS and not text[:-1].rstrip("0123456789.") and text[:-1]:
        number_part = text[:-1]
        multiplier = _DURATION_UNITS[unit]
    else:
        number_part = text
        multiplier = 1.0
    try:
        value = float(number_part)
    except ValueError as exc:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"--for {raw!r} is not a valid duration",
            remediation="use a number of seconds, or suffix with s/m/h/d (e.g. '24h')",
        ) from exc
    if value <= 0:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"--for {raw!r} must be a positive duration",
            remediation="use a number of seconds, or suffix with s/m/h/d (e.g. '24h')",
        )
    return value * multiplier


# ---------------------------------------------------------------------------
# lobes mesh status
# ---------------------------------------------------------------------------


def _fetch_roster(port: int, headers: dict[str, str]) -> dict | None:
    return _get_json(f"http://localhost:{port}", "/mesh/roster", headers, _GATEWAY_TIMEOUT_SECONDS)


def _member_status(member: dict) -> str:
    """Three-state verified/unverified/flapping (issue mesh-brain-join, c47).

    Additive: today's ``/mesh/roster`` payload carries neither ``verified``
    nor ``flapping`` per member (see the module docstring) — a member
    missing both keys renders ``unknown`` rather than guessing.
    """
    if member.get("flapping"):
        return "flapping"
    verified = member.get("verified")
    if verified is True:
        return "verified"
    if verified is False:
        return "unverified"
    return "unknown"


def _fmt_seconds(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{value:.0f}s"


def _render_roster_table(members: list[dict]) -> str:
    header = f"{'name':<20} {'origin':<28} {'age':>8}  {'expiry':>8}  {'status':<10}  roles"
    lines = [header, "-" * len(header)]
    if not members:
        lines.append("(no members)")
        return "\n".join(lines)
    for member in members:
        name = str(member.get("name") or "")
        origin = str(member.get("origin") or "")
        age = _fmt_seconds(member.get("last_seen_age"))
        expiry = _fmt_seconds(member.get("expiry"))
        status = _member_status(member)
        roles = member.get("roles") or []
        roles_s = ", ".join(roles) if roles else "-"
        lines.append(f"{name:<20} {origin:<28} {age:>8}  {expiry:>8}  {status:<10}  {roles_s}")
    return "\n".join(lines)


def cmd_mesh_status(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    env = _runtime_ops.deployment_env_soft(args)
    headers = _key_headers(env)
    with _runtime_ops.friendly_unauthorized_errors(deploy_dir):
        payload = _fetch_roster(port, headers)
    if payload is None:
        if json_mode:
            emit_result({"available": False, "members": []}, json_mode=True)
        else:
            emit_diagnostic(
                "mesh roster unavailable — gateway unreachable, or mesh is not "
                "enabled on this box (LOBES_MESH_KEY unset)."
            )
            emit_result("(no roster — gateway unreachable or mesh disabled)", json_mode=False)
        return 0
    members = payload.get("members") or []
    if json_mode:
        emit_result({"available": True, "members": members}, json_mode=True)
    else:
        emit_result(_render_roster_table(members), json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# lobes mesh request — keyless "ask to join"
# ---------------------------------------------------------------------------


def cmd_mesh_request(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    env = _runtime_ops.deployment_env_soft(args)

    name = args.name or (env.get("LOBES_MESH_NAME") or "").strip()
    if not name:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="no member name to request join with",
            remediation="pass --name, or set LOBES_MESH_NAME in the deployment .env",
        )
    origin = args.origin or (env.get("GATEWAY_SELF_ORIGIN") or "").strip()
    plan = {"name": name, "origin": origin, "capacity": args.capacity}

    if not args.apply:
        msg = (
            f"DRY RUN — would POST /mesh/join to http://localhost:{port} "
            f"with {plan} (no join key required).\nRe-run with --apply to execute."
        )
        emit_result(
            {"dry_run": True, "port": port, "plan": plan} if json_mode else msg,
            json_mode=json_mode,
        )
        return 0

    with _runtime_ops.friendly_unauthorized_errors(deploy_dir):
        try:
            status, body = _post_json(
                f"http://localhost:{port}", "/mesh/join", plan, headers={}, timeout=5.0
            )
        except urllib.error.HTTPError as exc:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"/mesh/join refused ({exc.code}): {exc.reason}",
                remediation="check the gateway logs; the pending-join queue may be full",
            ) from exc
    result = {"status": status, "response": body, "plan": plan}
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(f">> requested join as {name!r} — HTTP {status}: {body}", json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# lobes mesh approve / revoke — authenticated, need the join key
# ---------------------------------------------------------------------------


def _require_key(env: dict[str, str]) -> str:
    key = _mesh_key(env)
    if not key:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="LOBES_MESH_KEY is not set — cannot authenticate to /mesh/approve|revoke",
            remediation="set LOBES_MESH_KEY in the deployment .env before --apply",
        )
    return key


def cmd_mesh_approve(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    env = _runtime_ops.deployment_env_soft(args)

    duration = _parse_duration_seconds(args.for_) if args.for_ else _DEFAULT_APPROVAL_SECONDS
    plan = {"name": args.name, "expiry": duration, "approved_by": "operator"}

    if not args.apply:
        msg = (
            f"DRY RUN — would POST /mesh/approve to http://localhost:{port} "
            f"with {plan}, authenticated with the join key.\n"
            "Re-run with --apply to execute."
        )
        emit_result(
            {"dry_run": True, "port": port, "plan": plan} if json_mode else msg,
            json_mode=json_mode,
        )
        return 0

    key = _require_key(env)
    with _runtime_ops.friendly_unauthorized_errors(deploy_dir):
        try:
            status, body = _post_json(
                f"http://localhost:{port}",
                "/mesh/approve",
                plan,
                headers={"Authorization": f"Bearer {key}"},
                timeout=5.0,
            )
        except urllib.error.HTTPError as exc:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"/mesh/approve refused ({exc.code}): {exc.reason}",
                remediation="check the name and the configured LOBES_MESH_KEY",
            ) from exc
    result = {"status": status, "response": body, "plan": plan}
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(
            f">> approved {args.name!r} for {duration:.0f}s — HTTP {status}", json_mode=False
        )
    return 0


def cmd_mesh_revoke(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    env = _runtime_ops.deployment_env_soft(args)

    plan = {"name": args.name, "approved_by": "operator"}

    if not args.apply:
        msg = (
            f"DRY RUN — would POST /mesh/revoke to http://localhost:{port} "
            f"with {plan}, authenticated with the join key.\n"
            "Re-run with --apply to execute."
        )
        emit_result(
            {"dry_run": True, "port": port, "plan": plan} if json_mode else msg,
            json_mode=json_mode,
        )
        return 0

    key = _require_key(env)
    with _runtime_ops.friendly_unauthorized_errors(deploy_dir):
        try:
            status, body = _post_json(
                f"http://localhost:{port}",
                "/mesh/revoke",
                plan,
                headers={"Authorization": f"Bearer {key}"},
                timeout=5.0,
            )
        except urllib.error.HTTPError as exc:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"/mesh/revoke refused ({exc.code}): {exc.reason}",
                remediation="check the name and the configured LOBES_MESH_KEY",
            ) from exc
    result = {"status": status, "response": body, "plan": plan}
    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(f">> revoked {args.name!r} — HTTP {status}", json_mode=False)
    return 0


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def _no_verb(args: argparse.Namespace) -> int:
    # Bare `lobes mesh` → the read-only status (safe default, mirrors fleet).
    return cmd_mesh_status(args)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--compose-dir", help=_COMPOSE_DIR_HELP)
    p.add_argument("--port", type=int, help=_PORT_HELP)
    p.add_argument("--json", action="store_true", help=_JSON_HELP)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "mesh",
        help="Mesh membership: status / request / approve / revoke. " "See 'lobes mesh status'.",
    )
    _add_common(p)
    p.set_defaults(func=_no_verb, json=False)
    noun = p.add_subparsers(dest="mesh_command", parser_class=type(p))

    status = noun.add_parser(
        "status",
        help="Read-only: roster members, heartbeat age, approval expiry, "
        "verified/unverified/flapping, roles.",
    )
    _add_common(status)
    status.set_defaults(func=cmd_mesh_status)

    request = noun.add_parser(
        "request",
        help="Ask to join the mesh (keyless POST /mesh/join). Dry-run; --apply.",
    )
    _add_common(request)
    request.add_argument("--name", help="Member name (default: LOBES_MESH_NAME in .env).")
    request.add_argument("--origin", help="This box's origin (default: GATEWAY_SELF_ORIGIN).")
    request.add_argument("--capacity", help="Optional declared capacity.")
    request.add_argument("--apply", action="store_true", help="Actually send the join request.")
    request.set_defaults(func=cmd_mesh_request)

    approve = noun.add_parser(
        "approve",
        help="Approve a pending member name (needs the join key). Dry-run; --apply.",
    )
    _add_common(approve)
    approve.add_argument("name", help="Member name to approve.")
    approve.add_argument(
        "--for",
        dest="for_",
        help="Approval duration, e.g. '3600', '30m', '24h', '7d' (default: 3600s).",
    )
    approve.add_argument("--apply", action="store_true", help="Actually POST /mesh/approve.")
    approve.set_defaults(func=cmd_mesh_approve)

    revoke = noun.add_parser(
        "revoke",
        help="Revoke a member's approval (needs the join key). Dry-run; --apply.",
    )
    _add_common(revoke)
    revoke.add_argument("name", help="Member name to revoke.")
    revoke.add_argument("--apply", action="store_true", help="Actually POST /mesh/revoke.")
    revoke.set_defaults(func=cmd_mesh_revoke)
