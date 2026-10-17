"""``lobes status`` — read-only snapshot of the current deployment.

Reports the configured model/served-name/port (from ``.env``), the container
lifecycle + health state, and whether ``/health`` is responding. This is the
*configured* served model (from ``.env``) + health — not a live ``/v1/models``
query; for the full supported catalog you can switch to, use ``lobes overview --list``.

``--pressure`` sub-mode (issue #68/#69/#85)
-------------------------------------------
Reads host memory pressure from ``/proc`` (via
:func:`lobes.runtime._pressure.sample_pressure`) and maps it to a busy/serve
decision via :func:`lobes.gateway._pressure_policy.decide` (called with
``requested_tier="main"`` so the result shows what a full-tier request would get
right now).  Under swap/iowait pressure the gateway no longer degrades a
``main``/``senses`` request onto a different model — it **sheds** it with HTTP
429 (busy, retry shortly), issue #85.  So this command reports the *box-level*
``mode`` (``warm`` / ``busy``), whether a full-tier request is being ``shed``,
the ``servable_tier`` that still answers under pressure (``minor`` — the floor —
is always served, never shed), and the ``retry_after`` a busy caller would get.
Because it calls the same :func:`decide` handle_post consults, the reported
decision matches what a live request would receive — without issuing one.  This
path is **strictly read-only**: it writes nothing and never touches the Docker
socket (it only reads the deployment ``.env`` for the thermal keys, below).

Thermal (lane-safety t14): when the sample carries ``thermal_c`` the report
adds a ``thermal`` block (the reading against ``LOBES_THERMAL_SHED_C`` /
``LOBES_THERMAL_RESUME_C``) and ``shed_signal`` names the signal that sheds
(``"thermal"`` above the shed band). ``LOBES_THERMAL_ZONES`` / ``_SHED_C`` /
``_RESUME_C`` live in the deployment ``.env`` (review-b #9), so they are read
from the resolved deployment (``--compose-dir`` / ``$LOBES_DIR`` / ``~/.lobes``,
best-effort: no deployment is fine), with this process's environment winning
for any key it sets. ``LOBES_SWAP_DEGRADED_THRESHOLD`` /
``LOBES_IOWAIT_DEGRADED_THRESHOLD`` are read the same way (review2-b #4), so an
Orin's raised iowait threshold is honoured here too. That one ``.env`` read is
the only deployment access.
"""

from __future__ import annotations

import argparse
import os

from lobes import catalog
from lobes.cli import _runtime_ops
from lobes.cli._output import emit_result
from lobes.gateway import _pressure_policy
from lobes.runtime import _compose, _env, _health, _lanes
from lobes.runtime import _pressure as _pressure_mod

# Shown when a key is absent/empty in .env (matches _env.read_env's default).
_UNSET = "(unset)"

#: Thermal knobs ``status --pressure`` reads from the deployment ``.env``.
_THERMAL_KEYS = ("LOBES_THERMAL_ZONES", "LOBES_THERMAL_SHED_C", "LOBES_THERMAL_RESUME_C")
_BAND_KEYS = _THERMAL_KEYS[1:]
#: The busy thresholds ``decide()`` compares against (review2-b #4).
_THRESHOLD_KEYS = (_pressure_policy.SWAP_THRESHOLD_ENV, _pressure_policy.IOWAIT_THRESHOLD_ENV)


def _pressure_env(args: argparse.Namespace) -> dict[str, str]:
    """The deployment ``.env``'s pressure keys, overridden by this process's env.

    The thermal knobs and the swap/iowait busy thresholds — every knob
    ``decide()`` reads — so the reported decision matches the gateway, which
    gets them from that same ``.env``.
    """
    dotenv = _runtime_ops.deployment_env_soft(args)
    merged = {k: dotenv[k] for k in (*_THERMAL_KEYS, *_THRESHOLD_KEYS) if dotenv.get(k)}
    merged.update(os.environ)
    return merged


def _thresholds(env: dict[str, str]) -> tuple[float, float] | None:
    """``(swap, iowait)`` from the merged env, or ``None`` (the constants) when unset."""
    if not any(k in env for k in _THRESHOLD_KEYS):
        return None
    return _pressure_policy.degraded_thresholds_from_env(env)


def _thermal_bands(env: dict[str, str]) -> tuple[float | None, float | None] | None:
    """Bands from the merged env, or ``None`` (the module constants) when no key is set."""
    if not any(k in env for k in _BAND_KEYS):
        return None
    return _pressure_policy.thermal_bands_from_env(env)


def _thermal_report(
    thermal_c: float | None, bands: tuple[float | None, float | None] | None = None
) -> dict:
    """The thermal reading against the card's shed band (lane-safety t14).

    ``band`` is ``off`` (no shed band configured), ``unknown`` (no zone
    reading), ``above_shed``, ``hysteresis`` (between resume and shed) or
    ``below_resume``. ``shedding`` is ``None`` inside the hysteresis band: the
    gateway sheds there only if it was already shedding, and that latch lives
    in the gateway process (``GET /status`` → ``pressure.thermal_shedding``),
    not in this one-shot read.
    """
    if bands is None:
        bands = (_pressure_policy.THERMAL_SHED_C, _pressure_policy.THERMAL_RESUME_C)
    shed_c, resume_c = bands
    shedding: bool | None = False
    if shed_c is None:
        band = "off"
    elif thermal_c is None:
        band = "unknown"
    elif _pressure_policy.thermal_hot(thermal_c, bands=bands):
        band, shedding = "above_shed", True
    elif _pressure_policy.thermal_hot(thermal_c, latched=True, bands=bands):
        band, shedding = "hysteresis", None
    else:
        band = "below_resume"
    return {
        "c": thermal_c,
        "shed_c": shed_c,
        "resume_c": resume_c,
        "band": band,
        "shedding": shedding,
    }


def _thermal_line(thermal: dict) -> str:
    """One text line for ``status --pressure``: the reading and its band."""
    if thermal["c"] is None:
        return "thermal: n/a"
    reading = f"thermal: {thermal['c']:.1f} C"
    band = thermal["band"]
    if band == "off":
        return f"{reading} (no shed band)"
    if band == "above_shed":
        return f"{reading} (above shed {thermal['shed_c']:.1f} C — shedding)"
    if band == "hysteresis":
        return (
            f"{reading} (between resume {thermal['resume_c']:.1f} C and shed "
            f"{thermal['shed_c']:.1f} C — sheds only if already shedding)"
        )
    return f"{reading} (below resume {thermal['resume_c']:.1f} C)"


def _cmd_status_pressure(args: argparse.Namespace, json_mode: bool) -> int:
    """Handle ``lobes status --pressure`` — a read-only /proc snapshot.

    Samples live host pressure and reports the busy/serve decision a full-tier
    (``main``) request would receive right now: the box-level ``mode``, whether
    such a request is ``shed`` (HTTP 429), the ``servable_tier`` that still
    answers (``minor`` — the floor — is never shed), and the ``retry_after`` a
    busy caller would get.  Calls the same :func:`decide` handle_post consults,
    so the reported decision matches a live request without issuing one.  No
    docker; the only deployment access is a best-effort read of the thermal
    keys (review-b #9) and the swap/iowait thresholds (review2-b #4) from its
    ``.env``.
    """
    env = _pressure_env(args)
    bands = _thermal_bands(env)
    p = _pressure_mod.sample_pressure(env=env)
    thermal_c = p.get("thermal_c")
    # A one-shot CLI holds no latch history, so it asks decide() the
    # unlatched question; the hysteresis band is reported as such below.
    d = _pressure_policy.decide(
        p["swap_used_percent"],
        p["iowait_percent"],
        requested_tier="main",
        thermal_c=thermal_c,
        thermal_bands=bands,
        thresholds=_thresholds(env),
    )
    thermal = _thermal_report(thermal_c, bands)
    # Under pressure a full-tier (main / senses) request is shed with 429; the
    # only tier that still serves is ``minor`` (the floor). ``servable_tier`` is
    # "minor" when busy, else "main" — the model resolves that tier (#85).
    servable_tier = d["servable_tier"]
    model = catalog.resolve_tier(servable_tier).id
    shed = d["shed"]
    retry_after = _pressure_policy.BUSY_RETRY_AFTER_SECONDS if shed else None

    result = {
        "mode": d["mode"],
        "shed": shed,
        "servable_tier": servable_tier,
        "model": model,
        "reason": d["reason"],
        "retry_after": retry_after,
        "pressure": p,
        "shed_signal": d["shed_signal"],
        "thermal": thermal,
    }

    if json_mode:
        emit_result(result, json_mode=True)
    else:
        shed_line = (
            f"shed:    main/cortex + senses + muse requests return 429 busy "
            f"(retry after {retry_after}s)"
            if shed
            else "shed:    none — all tiers served"
        )
        emit_result(
            "\n".join(
                [
                    f"mode:    {result['mode']}",
                    shed_line,
                    f"servable: {result['servable_tier']}",
                    f"model:   {result['model']}",
                    f"reason:  {result['reason']}",
                    f"swap:    {p['swap_used_percent']:.1f}%",
                    f"iowait:  {p['iowait_percent']:.1f}%",
                    _thermal_line(thermal),
                    f"signal:  {d['shed_signal'] or 'none'}",
                ]
            ),
            json_mode=False,
        )
    return 0


def _cmd_status_fleet(deploy_dir, env_path, port: int, json_mode: bool) -> int:
    """Render ``lobes status`` for a fleet deployment (one line per container)."""
    containers = [
        {"name": name, "state": _compose.inspect_state(name)}
        for name in _compose.fleet_containers(deploy_dir)
    ]
    # Specialist embed/rerank lanes (t9): listed only for the lanes this
    # deployment's compose set actually defines, so a box without the lane
    # overlay reports exactly what it did before.
    lanes = [
        {"name": name, "service": svc, "state": _compose.inspect_state(svc)}
        for name, svc in _lanes.defined_lane_services(
            deploy_dir, _compose._compose_files(deploy_dir)
        ).items()
    ]
    report = {
        "model": _env.read_env(env_path, "VLLM_MODEL", _UNSET),
        "served_name": _env.read_env(env_path, "VLLM_SERVED_NAME", _UNSET),
        "port": port,
        "tool_call_parser": _env.read_env(env_path, "VLLM_TOOL_CALL_PARSER", _UNSET),
        "deployment_dir": str(deploy_dir),
        "deployment": "fleet",
        "containers": containers,
        "health": "ok" if _health.is_healthy(port) else "not responding",
        "profile": _env.read_env(env_path, "LOBES_PROFILE", _UNSET),
    }
    if lanes:
        report["lanes"] = lanes

    if json_mode:
        emit_result(report, json_mode=True)
    else:
        lines = [
            f"model:  {report['model']}",
            f"served: {report['served_name']}  port: {report['port']}",
            f"parser: {report['tool_call_parser']}",
            f"dir:    {report['deployment_dir']}",
        ]
        for c in containers:
            lines.append(f"  {c['name']} — {c['state']}")
        for lane in lanes:
            lines.append(f"  lane {lane['name']} ({lane['service']}) — {lane['state']}")
        lines.append(f"health: {report['health']} (:{port})")
        if report["profile"] != _UNSET:
            lines.append(f"profile: {report['profile']}")
        lines.append("see 'lobes fleet status' / 'lobes capabilities' for the full fleet/role view")
        emit_result("\n".join(lines), json_mode=False)
    return 0


def _cmd_status_single(deploy_dir, env_path, port: int, json_mode: bool) -> int:
    """Render ``lobes status`` for a single-model (legacy) deployment."""
    report = {
        "model": _env.read_env(env_path, "VLLM_MODEL", _UNSET),
        "served_name": _env.read_env(env_path, "VLLM_SERVED_NAME", _UNSET),
        "port": port,
        "tool_call_parser": _env.read_env(env_path, "VLLM_TOOL_CALL_PARSER", _UNSET),
        "deployment_dir": str(deploy_dir),
        "container": _compose.CONTAINER,
        "state": _compose.inspect_state(),
        "health": "ok" if _health.is_healthy(port) else "not responding",
        "profile": _env.read_env(env_path, "LOBES_PROFILE", _UNSET),
    }

    if json_mode:
        emit_result(report, json_mode=True)
    else:
        lines = [
            f"model:  {report['model']}",
            f"served: {report['served_name']}  port: {report['port']}",
            f"parser: {report['tool_call_parser']}",
            f"dir:    {report['deployment_dir']}",
            f"state:  {report['container']} — {report['state']}",
            f"health: {report['health']} (:{port})",
        ]
        if report["profile"] != _UNSET:
            lines.append(f"profile: {report['profile']}")
        emit_result("\n".join(lines), json_mode=False)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))

    # --pressure path: /proc + /sys sampler; the deployment .env is read
    # best-effort for the thermal keys only — no docker, never an error.
    if getattr(args, "pressure", False):
        return _cmd_status_pressure(args, json_mode)

    deploy_dir = _runtime_ops.deployment_dir(args)
    env_path = deploy_dir / _compose.ENV_FILE
    port = _runtime_ops.resolve_port(args, env_path)

    if _compose.is_fleet(deploy_dir):
        return _cmd_status_fleet(deploy_dir, env_path, port, json_mode)
    return _cmd_status_single(deploy_dir, env_path, port, json_mode)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "status",
        help="Read-only: the configured served model (from .env), container state, "
        "/health (catalog: lobes overview --list).",
    )
    p.add_argument("--port", type=int, help="Host port (default: VLLM_PORT in .env, else 8000).")
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.add_argument(
        "--pressure",
        action="store_true",
        help=(
            "Read-only: sample host memory pressure (/proc) and thermal zones, and "
            "report the highest tier currently permitted. Thermal knobs come from "
            "the deployment .env if there is one; no docker needed."
        ),
    )
    p.set_defaults(func=cmd_status)
