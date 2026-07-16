"""``lobes capabilities`` / ``lobes endpoint`` — the live role→endpoint contract
(issue #81; the gateway-client rewrite is issue #96, plan "advertised implies
reachable" task t7).

Both verbs used to REBUILD the six-role registry from the deployment's
``.env`` and call that the truth — this module's docstring used to claim the
CLI and the gateway share "exactly one source of truth" because both called
the same pure builder, :func:`lobes.roles.role_registry_from_env` /
:func:`lobes.roles.build_role_registry`. That claim was false in practice: the
CLI reads what an operator's ``.env`` *says*, while the gateway answers with
what its *own container process* was actually started with. Those are two
independent derivations of the same contract, sourced from two different
places, and they have now drifted in **both** directions:

* issue #92 — the gateway under-reported: a live backend its background
  readiness probe hadn't caught up with yet looked dead over HTTP, while the
  CLI, going by config, was (in that instance) right.
* issue #96 — the CLI over-reported: ``AUDIO_URL`` was present in the
  deployment's ``.env`` but was never wired into the gateway *container's*
  environment (a ``docker compose`` env-passthrough gap), so the CLI's
  offline registry advertised ``stt``/``tts`` as ``ready=true`` on a path
  that actually 404s/503s — while the gateway, which only knows its own
  actual environment, correctly said ``ready=false``.

Calling a single shared *function* the "one source of truth" was never
enough, because the CLI and the gateway don't share an address space — each
evaluates that function against its own, independently-sourced ``env``
mapping. Making the two derivations agree by testing them against each other
after the fact is exactly the failure mode that produced both bugs: a config
file on the host disk is not evidence of what an already-started container
process actually has wired, and no amount of keeping :mod:`lobes.roles` in
sync can make it so.

So this module is no longer a second, independent derivation. It is a CLIENT
of the gateway:

* ``lobes capabilities`` / ``lobes endpoint <role>`` first try ``GET
  http://localhost:<resolved port>/capabilities`` against the gateway that
  would actually serve a request. On a clean 200 whose body is a JSON object
  carrying all seven roles, THAT payload is rendered verbatim — the gateway is
  the one process that knows what it actually has wired and what its own
  readiness probes actually last observed, so its answer is the only one
  worth reporting.
* On any failure to get that authoritative answer — no gateway listening,
  connection refused, a timeout, a non-200, or a malformed/incomplete body —
  this degrades to the offline view built from the deployment's ``.env``
  (:func:`lobes.roles.role_registry_from_env`), same as before t7. But the
  fallback is now honest about what it is: every role's ``ready`` is forced
  to ``False`` (a config file was never probed, so it can never be evidence
  of health — this generalises issue #96's fix past stt/tts to all six
  roles), and the mode is always discoverable — but NOT by adding a key to
  the JSON payload. A prior revision of this module added a top-level
  ``source`` sibling next to the six role keys in ``--json`` output; a Qodo
  action-required finding on PR #102 correctly flagged that as a second,
  new divergence from the gateway's own contract (``GET /capabilities``
  returns *exactly* ``{cortex, senses, embedder, reranker, stt, tts}``, so a
  caller doing ``set(payload) == ROLES`` broke on the extra key — ironic,
  since t7's whole point was to make the CLI and gateway agree). The fix:
  ``--json`` output is now, in EVERY mode, the bare six-role dict and
  nothing else — byte-for-byte what the gateway would return in gateway
  mode. The offline/gateway distinction is instead surfaced out-of-band: the
  human-readable table keeps its ``# source: ...`` header line (stdout,
  human-facing, not part of any machine contract), and ``--json`` mode
  additionally writes a one-line "offline" diagnostic to **stderr** — never
  into the stdout JSON object — when (and only when) it is rendering the
  fallback. A caller — human or agent — can still always tell live from
  offline; it just never costs the JSON payload an extra key.

This makes the honesty condition (h3: the CLI's and the gateway's answers
agree) true **by construction** whenever a gateway is reachable — there is
then exactly one source of truth, and the CLI is only ever rendering it, so
there is nothing left to drift. When no gateway is reachable there is nothing
to agree *with*, so the fallback trades completeness for honesty (an
"unverified, treat as down" default) instead of silently guessing.

Deployment resolution
----------------------
The gateway URL to dial is derived exactly like every other read-only probe
in this CLI (``status``, ``overview --live``, ``fleet status``):
``http://localhost:<published port>``, where the port is ``--port`` if given,
else ``VLLM_PORT`` from the deployment's ``.env``, else 8000
(:func:`lobes.cli._runtime_ops.resolve_port_soft`). Resolving that port never
requires a scaffolded deployment (an unscaffolded one degrades to 8000), so
both verbs still answer — falling straight to the offline path — even before
``lobes init`` has ever run, mirroring ``lobes overview --live`` / ``assess``
/ ``benchmark``.

Both verbs stay strictly read-only: the only I/O is one bounded HTTP GET to
the local gateway (stdlib ``urllib``, a short timeout so an unreachable
gateway degrades fast rather than stalling an agent's loop) plus, on the
fallback path only, a read of the deployment's ``.env`` off disk. No
compose/docker call, no ``--apply``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import urllib.error
import urllib.request

from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic, emit_result
from lobes.gateway._config import build_config
from lobes.roles import ROLES, RoleInfo, annotate_peer_referrals, role_registry_from_env

_JSON_HELP = "Emit structured JSON."
_COMPOSE_DIR_HELP = "Deployment dir (default: $LOBES_DIR or ~/.lobes)."
_PORT_HELP = "Gateway host port (default: VLLM_PORT in .env)."

# The offline-mode notice. Shown as the table's first line (stdout, human
# text, not a machine contract) AND, in --json mode only, written to stderr
# instead of being mixed into the JSON object — see the module docstring for
# why: the JSON payload's keys must equal exactly `set(ROLES)` in every mode,
# matching what a live `GET /capabilities` returns byte-for-byte.
_OFFLINE_NOTICE = (
    "source: offline — gateway unreachable; showing .env-configured "
    "defaults, NOT a live probe (every role's ready=false)"
)

# Bounded so an unreachable/foreign process on the resolved port degrades to
# the offline fallback quickly — this is a read-only introspection verb, and
# an agent looping on it must never be made to wait on a dead socket.
_GATEWAY_TIMEOUT_SECONDS = 2.0

# The full RoleInfo field set — used to sanity-check a gateway response before
# trusting it. A 200 from *something* listening on the resolved port whose
# body happens to be a dict keyed by all six role names but missing fields a
# real /capabilities response always carries is treated as malformed, not
# authoritative (see _fetch_gateway_capabilities).
_ROLE_INFO_FIELDS = {f.name for f in dataclasses.fields(RoleInfo)}


def _fetch_gateway_capabilities(
    port: int,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = _GATEWAY_TIMEOUT_SECONDS,
) -> dict[str, dict] | None:
    """``GET /capabilities`` from the live gateway on ``port``.

    Returns the parsed payload — a dict keyed by all six :data:`ROLES`, each
    value the role's JSON metadata — on a clean 200 with a well-shaped body.
    Returns ``None`` on ANY failure to get an authoritative answer:
    connection refused (nothing listening), a DNS/socket error, a timeout, a
    non-2xx status, an undecodable body, or a 200 whose body doesn't have the
    expected shape (missing a role, or a role missing an expected field — a
    stale/foreign process happens to be answering on this port; see the
    ``lobes.roles._gateway_base_url`` docstring for why a foreign daemon on a
    guessed port is a real hazard on this rig, not a hypothetical one).

    ``None`` is not an error to the caller: it means "fall back to the
    offline registry", exactly like every other read-only probe in this CLI
    degrades when the thing it would rather ask isn't there. The ONE
    exception (issue #127 t3): a 401 is not folded into ``None`` — silently
    showing stale offline data in place of an auth rejection would hide the
    actual problem — it is re-raised so the caller's
    :func:`lobes.cli._runtime_ops.friendly_unauthorized_errors` wrapper can
    turn it into a clear, actionable error instead.

    ``headers`` carries the outbound ``Authorization`` header (see
    :func:`lobes.cli._runtime_ops.gateway_auth_headers`) — ``None``/``{}`` on
    a keyless deployment sends no header at all, byte-identical to today.
    """
    url = f"http://localhost:{port}/capabilities"
    req = urllib.request.Request(url, headers=headers or {})
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
    if not isinstance(data, dict):
        return None
    for role in ROLES:
        entry = data.get(role)
        if not isinstance(entry, dict) or not _ROLE_INFO_FIELDS <= set(entry):
            return None
    return data


def _offline_payload(args: argparse.Namespace) -> dict[str, dict]:
    """The ``.env``-derived fallback payload, shaped like a gateway response.

    Built from the SAME pure pieces the gateway itself uses
    (:func:`lobes.roles.role_registry_from_env` +
    :func:`lobes.roles.annotate_peer_referrals` over the deployment env), so
    the opt-in honest referral (mesh-brain t3 — ``hosted_by`` on an unhosted
    role with a declared ``*_PEER_ORIGIN``) shows up on the offline path too.
    With no peer config the annotation is a no-op and the payload carries
    exactly the ``RoleInfo`` field set per role, unchanged.
    """
    env = _runtime_ops.deployment_env_soft(args)
    port, _ = _runtime_ops.resolve_port_soft(args)
    gateway_url = f"http://localhost:{port}"
    registry = role_registry_from_env(env, gateway_url=gateway_url)
    payload = {role: _role_payload(registry[role]) for role in ROLES}
    table, _server = build_config(env)
    return annotate_peer_referrals(payload, table)


def _role_payload(info: RoleInfo) -> dict[str, object]:
    """The full ``RoleInfo`` field set as a JSON-safe dict.

    The #81 shape wants at least ``{endpoint, model, context, ready,
    responsibilities}`` present; ``dataclasses.asdict`` includes every field
    (tuples become arrays), so the rest (role/runtime/path/quant/mtp/
    forbidden_responsibilities/loaded) rides along too.
    """
    return dataclasses.asdict(info)


def _capabilities_view(args: argparse.Namespace) -> tuple[dict[str, dict], str]:
    """Resolve the six-role payload, preferring the live gateway.

    Returns ``(payload, source)``: ``payload`` is a dict keyed by all six
    :data:`ROLES` (each value the role's JSON metadata), and ``source`` is
    ``"gateway"`` when it came straight from a live ``GET /capabilities``, or
    ``"offline"`` when it is the ``.env``-derived fallback. In the offline
    case every role's ``ready`` is forced to ``False`` — see the module
    docstring and issue #96: a config file was never probed, so it can never
    honestly claim a role is reachable, no matter what the offline registry's
    own ``loaded``/``ready`` computation would otherwise say.
    """
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    headers = _runtime_ops.gateway_auth_headers(deploy_dir)
    live = _fetch_gateway_capabilities(port, headers=headers)
    if live is not None:
        return live, "gateway"
    offline = _offline_payload(args)
    for role in ROLES:
        offline[role]["ready"] = False
    return offline, "offline"


def _render_table(registry: dict[str, dict], source: str) -> str:
    header = f"{'role':<9} {'model':<48} {'context':>8}  loaded  endpoint"
    lines: list[str] = []
    if source == "offline":
        lines.append(f"# {_OFFLINE_NOTICE}")
    else:
        lines.append("# source: gateway — live GET /capabilities")
    lines.append(header)
    lines.append("-" * len(header))
    for role in ROLES:
        info = registry[role]
        model = info["model"] if len(info["model"]) <= 48 else info["model"][:45] + "..."
        lines.append(
            f"{info['role']:<9} {model:<48} {info['context']:>8}  "
            f"{'yes' if info['loaded'] else 'no ':<6} {info['endpoint'] or '(none)'}"
        )
        # Hardware feasibility (task t6): a role this machine's per-machine
        # profile declared it can never serve — surfaced even though the row
        # above still shows loaded=yes (the backend can be structurally wired,
        # e.g. the always-present primary, yet infeasible on this box).
        # `.get` defaults True so an older/foreign payload missing this key
        # (pre-t6 gateway, or a hand-built fixture) never raises.
        if info.get("feasible", True) is False:
            lines.append("          ** infeasible on this machine — never served here **")
            # Third lobe state (proxy-lobes t6, issues #115/#127): this box
            # FOLLOWS its own referral — requests to this gateway are forwarded
            # to the hosting peer, so callers stay single-endpoint. Distinct
            # wording from the referral-only case below, which remains an
            # address the CALLER must dial directly (this box 404s).
            if info.get("proxied") and info.get("hosted_by"):
                lines.append(f"          proxied via this gateway from peer: {info['hosted_by']}")
            # Opt-in honest referral (mesh-brain t3): the operator-declared
            # peer origin that hosts this unhosted role, when one is set. An
            # address to dial DIRECTLY — this box never proxies to it.
            elif info.get("hosted_by"):
                lines.append(f"          hosted by peer: {info['hosted_by']} (dial it directly)")
        lines.append(f"          responsibilities: {', '.join(info['responsibilities'])}")
    return "\n".join(lines)


def cmd_capabilities(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    _port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    with _runtime_ops.friendly_unauthorized_errors(deploy_dir):
        payload, source = _capabilities_view(args)
    if json_mode:
        # The JSON payload is the bare six-role dict in EVERY mode — no
        # "source"/mode key is ever mixed into it. In gateway mode this is
        # the live GET /capabilities body rendered verbatim; in offline mode
        # it is the .env-derived fallback (every role's ready forced False),
        # shaped identically. `set(json.loads(out)) == set(ROLES)` holds
        # unconditionally — matching what the gateway itself returns
        # byte-for-byte (Qodo action-required finding on PR #102: a prior
        # revision's top-level "source" sibling broke exactly that
        # contract). The offline/gateway distinction still needs to be
        # discoverable, so it goes out-of-band: stderr, never stdout JSON.
        if source == "offline":
            emit_diagnostic(_OFFLINE_NOTICE)
        emit_result(payload, json_mode=True)
    else:
        emit_result(_render_table(payload, source), json_mode=False)
    return 0


def cmd_endpoint(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    role = args.role
    if role not in ROLES:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"unknown role {role!r}",
            remediation=f"valid roles: {', '.join(ROLES)}",
        )
    _port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    with _runtime_ops.friendly_unauthorized_errors(deploy_dir):
        payload, _source = _capabilities_view(args)
    endpoint = payload[role]["endpoint"]
    if json_mode:
        emit_result({"role": role, "endpoint": endpoint}, json_mode=True)
    else:
        emit_result(endpoint, json_mode=False)
    return 0


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--port", type=int, help=_PORT_HELP)
    p.add_argument("--compose-dir", help=_COMPOSE_DIR_HELP)
    p.add_argument("--json", action="store_true", help=_JSON_HELP)


def register(sub: argparse._SubParsersAction) -> None:
    cap = sub.add_parser(
        "capabilities",
        help="Read-only: the seven first-class roles (cortex/senses/muse/embedder/"
        "reranker/stt/tts) resolved to live endpoint + metadata (issue #81).",
    )
    _add_common_args(cap)
    cap.set_defaults(func=cmd_capabilities)

    ep = sub.add_parser(
        "endpoint",
        help="Read-only: print the base URL for one role, e.g. 'lobes endpoint cortex'.",
    )
    ep.add_argument("role", help=f"one of: {', '.join(ROLES)}")
    _add_common_args(ep)
    ep.set_defaults(func=cmd_endpoint)
