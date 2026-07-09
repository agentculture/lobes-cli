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
  carrying all six roles, THAT payload is rendered verbatim — the gateway is
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
  roles), and both the JSON and the table carry an explicit ``source``
  marker (``"gateway"`` when live, ``"offline"`` when degraded) so a caller —
  human or agent — can never mistake a configured-defaults guess for a live
  observation.

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
import urllib.request

from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.roles import ROLES, RoleInfo, role_registry_from_env

_JSON_HELP = "Emit structured JSON."
_COMPOSE_DIR_HELP = "Deployment dir (default: $LOBES_DIR or ~/.lobes)."
_PORT_HELP = "Gateway host port (default: VLLM_PORT in .env)."

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
    port: int, timeout: float = _GATEWAY_TIMEOUT_SECONDS
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
    degrades when the thing it would rather ask isn't there.
    """
    url = f"http://localhost:{port}/capabilities"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # local endpoint only
            if not (200 <= resp.status < 300):
                return None
            raw = resp.read()
    except OSError:  # URLError (incl. HTTPError) subclasses OSError
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


def _offline_registry(args: argparse.Namespace) -> dict[str, RoleInfo]:
    env = _runtime_ops.deployment_env_soft(args)
    port, _ = _runtime_ops.resolve_port_soft(args)
    gateway_url = f"http://localhost:{port}"
    return role_registry_from_env(env, gateway_url=gateway_url)


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
    port, _ = _runtime_ops.resolve_port_soft(args)
    live = _fetch_gateway_capabilities(port)
    if live is not None:
        return live, "gateway"
    registry = _offline_registry(args)
    offline = {role: _role_payload(registry[role]) for role in ROLES}
    for role in ROLES:
        offline[role]["ready"] = False
    return offline, "offline"


def _render_table(registry: dict[str, dict], source: str) -> str:
    header = f"{'role':<9} {'model':<48} {'context':>8}  loaded  endpoint"
    lines: list[str] = []
    if source == "offline":
        lines.append(
            "# source: offline — gateway unreachable; showing .env-configured "
            "defaults, NOT a live probe (every role's ready=false)"
        )
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
        lines.append(f"          responsibilities: {', '.join(info['responsibilities'])}")
    return "\n".join(lines)


def cmd_capabilities(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    payload, source = _capabilities_view(args)
    if json_mode:
        # "source" is an added top-level sibling of the six role keys — it never
        # collides with a role name, so every existing consumer that reads
        # payload[<role>] is unaffected; only a strict `set(payload) == ROLES`
        # check needs to account for it.
        emit_result({**payload, "source": source}, json_mode=True)
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
        help="Read-only: the six first-class roles (cortex/senses/embedder/"
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
