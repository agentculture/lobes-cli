"""``lobes calibrate <role>`` — measure a role's replica-pool serving capacity.

Ramps concurrency against a named Colleague role and locates the throughput
plateau + TTFT-bound knee via :func:`lobes.assess.calibration_knee` (t2) — the
number the cortex replica pool (issue #199) uses to rank replicas by
``active / capacity`` instead of a hardcoded weight of 1.0.

**Capacity here is a MEASURED THROUGHPUT KNEE — explicitly NOT vLLM's
``--max-num-seqs`` OOM-safety cap and NOT its KV-cache-derived concurrency
ceiling.** Both of those are numbers chosen to avoid running out of memory or
to describe what physically fits, not to describe useful serving throughput.
Conflating them was named as a defect to avoid
(``docs/specs/2026-08-27-capacity-relative-pool-routing.md``, s3/c4).

**Scope of a measured number.** A calibrated capacity is only valid for the
*(box, checkpoint, context window, speculative config)* it was measured on —
a ``lobes switch`` or a shape re-render invalidates it. This verb does not
track that fingerprint itself (the gateway's replica cache does, per t4); an
operator re-runs ``lobes calibrate`` after any of those change.

Read-only by default: plain ``lobes calibrate <role>`` only ramps and reports
— the samples, the chosen concurrency, and whether the ramp genuinely
plateaued. Writing the measured number to ``.env`` (as ``<PREFIX>_MAX_ACTIVE``,
the same knob :mod:`lobes.gateway._config` reads, t1) requires ``--apply``,
following the repo's dry-run-by-default write-verb convention. ``--apply``
additionally REFUSES to write when the ramp never demonstrated a genuine
plateau (``stopped_by == "top_of_ramp"``, or the degenerate ``"empty"``/a
zero-concurrency ``"ttft_bound"`` result) — recording the top level merely
*tried* as though it were a measured knee would poison routing with a
number that was never actually validated as a plateau.

No calibration logic runs inside the gateway's request path: this verb is the
entire calibration surface, living in ``lobes/cli/_commands/`` exactly like
``lobes assess``/``lobes benchmark`` — the gateway only ever *consumes* the
number this verb writes.
"""

from __future__ import annotations

import argparse
import math

from lobes import assess as _assess
from lobes.assess import CalibrationKnee, calibration_knee, run_concurrent
from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_SUCCESS, EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.gateway._config import MAX_ACTIVE_ENV
from lobes.roles import ROLE_BACKEND, ROLES, role_registry_from_env
from lobes.runtime import _compose, _env

_DEFAULT_SCHEDULE: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

# Plain-language rendering of CalibrationKnee.stopped_by (assess.py's closed
# vocabulary) — a caller/operator reading the report should never need to
# know the raw enum string.
_STOPPED_BY_PROSE: dict[str, str] = {
    "empty": "no samples were collected",
    "plateau": "throughput plateaued (each further step bought too little)",
    "ttft_bound": "time-to-first-token crossed the declared bound",
    "top_of_ramp": "the ramp reached the top of its schedule while still rising "
    "meaningfully — it never plateaued",
}


def _parse_schedule(raw: str | None) -> tuple[int, ...]:
    """Parse ``--schedule "1,2,4,8"`` into an ascending tuple of positive ints."""
    if not raw:
        return _DEFAULT_SCHEDULE
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"--schedule must be a comma-separated list of integers; got {raw!r}",
            remediation="e.g. --schedule 1,2,4,8,16",
        ) from exc
    if not values or any(v <= 0 for v in values):
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"--schedule must contain only positive integers; got {raw!r}",
            remediation="e.g. --schedule 1,2,4,8,16",
        )
    return tuple(sorted(values))


def _require_finite(
    value: float, name: str, *, positive: bool = True, allow_none: bool = False
) -> float | None:
    """Reject NaN/Infinity (and, when *positive*, non-positive values) for a
    numeric CLI argument.

    ``float("nan")``/``float("inf")`` both parse cleanly through argparse's
    own ``type=float`` — a NaN ``--ttft-bound-s`` makes every
    ``ttft_s > ttft_bound_s`` comparison ``False``, silently disabling the
    TTFT guard and letting :func:`~lobes.assess.calibration_knee` publish a
    capacity that was never actually bound-checked (Qodo review finding F7 on
    #221). Every new numeric flag this verb added is checked the same way,
    not just ``--ttft-bound-s``.

    ``value`` may be ``None`` only when *allow_none* is set (for the
    optional ``--min-relative-gain``, whose absence means "use the default"),
    in which case ``None`` passes through unchanged.
    """
    if value is None:
        if allow_none:
            return None
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"{name} is required",
            remediation=f"pass a finite value for {name}",
        )
    if not math.isfinite(value):
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"{name} must be a finite number; got {value!r}",
            remediation=f"pass a normal, finite value for {name} (not NaN or Infinity)",
        )
    if positive and value <= 0:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"{name} must be a positive number; got {value!r}",
            remediation=f"pass a positive value for {name}",
        )
    return value


def _aggregate_tok_s(row: dict) -> float:
    """Aggregate decode tok/s from a :func:`run_concurrent` row.

    ``total_completion_tokens / total_s`` — the total completion tokens
    across the whole batch divided by the batch's wall-clock duration, the
    same quantity an operator would compute by hand.

    Previously this multiplied ``concurrency`` by the reciprocal of the MEAN
    per-request ``latency_ms / completion_tokens`` (``ms_per_token``), which
    is not aggregate tokens over batch wall time — unequal completion
    lengths or latencies across the batch distort it badly (Qodo review
    finding F3 on #221; a live example is preserved in
    ``tests/test_assess_perf.py::test_run_concurrent_total_completion_tokens_over_wall_time_is_the_true_aggregate``,
    where the two formulas disagree by ~90%). ``0.0`` when ``total_s`` is
    degenerate (no wall time elapsed), matching :func:`calibration_knee`'s
    own zero-baseline handling.
    """
    total_s = row.get("total_s") or 0.0
    if total_s <= 0:
        return 0.0
    total_tokens = row.get("total_completion_tokens") or 0
    return round(total_tokens / total_s, 3)


def _measure_level(
    url: str,
    model: str,
    concurrency: int,
    *,
    max_tokens: int,
    timeout: int,
    _measure=None,
) -> tuple[int, float, float]:
    """One ramp level -> ``(concurrency, aggregate_tok_s, ttft_s)``.

    Two :func:`run_concurrent` calls at the same concurrency: one shaped for
    throughput (the real ``max_tokens``), one shaped for TTFT (``max_tokens=1``,
    the same trick :func:`lobes.assess.measure_prefill_ttft` uses for a single
    request, applied here under the level's own concurrency so TTFT reflects
    load, not just a solo request). ``_measure`` is injectable for hermetic
    tests — defaults to :func:`lobes.assess.run_concurrent`.
    """
    measure = _measure or run_concurrent
    throughput_row = measure(
        url, model, concurrency=concurrency, max_tokens=max_tokens, timeout=timeout
    )
    ttft_row = measure(url, model, concurrency=concurrency, max_tokens=1, timeout=timeout)
    aggregate_tok_s = _aggregate_tok_s(throughput_row)
    ttft_s = (ttft_row.get("p50_latency_ms") or 0.0) / 1000.0
    return concurrency, aggregate_tok_s, ttft_s


def drive_calibration(
    url: str,
    model: str,
    *,
    schedule: tuple[int, ...] = _DEFAULT_SCHEDULE,
    ttft_bound_s: float,
    min_relative_gain: float | None = None,
    max_tokens: int = 128,
    timeout: int = 300,
    _measure=None,
) -> CalibrationKnee:
    """Ramp *schedule* against ``url``/``model`` and return the calibration knee.

    Re-evaluates :func:`lobes.assess.calibration_knee` after EVERY new level
    (not only once at the end) and stops early the moment a level resolves to
    anything other than ``"top_of_ramp"`` — mirrors
    :func:`lobes.assess.auto_ramp_concurrency`'s early-stop discipline, so a
    plateau or a TTFT-bound violation does not force unnecessary load onto the
    engine once the answer is already known. All decision logic stays inside
    the pure :func:`calibration_knee` — this function only drives the network
    calls that produce its input samples.
    """
    kwargs = {} if min_relative_gain is None else {"min_relative_gain": min_relative_gain}
    samples: list[tuple[int, float, float]] = []
    knee = calibration_knee(samples, ttft_bound_s=ttft_bound_s, **kwargs)
    for concurrency in schedule:
        samples.append(
            _measure_level(
                url, model, concurrency, max_tokens=max_tokens, timeout=timeout, _measure=_measure
            )
        )
        knee = calibration_knee(samples, ttft_bound_s=ttft_bound_s, **kwargs)
        if knee.stopped_by != "top_of_ramp":
            break
    return knee


def _resolve_role_info(args: argparse.Namespace, role: str):
    env = _runtime_ops.deployment_env_soft(args)
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    gateway_url = f"http://localhost:{port}"
    registry = role_registry_from_env(env, gateway_url=gateway_url)
    return registry[role], deploy_dir


def _refuse_reason(knee: CalibrationKnee) -> str | None:
    """Why ``--apply`` must refuse to write *knee*, or ``None`` when it may write.

    Per the acceptance criteria, a ramp that never demonstrated a genuine
    plateau (``stopped_by == "top_of_ramp"``) must not be persisted as though
    the top level tried were a measured knee. The same refusal is extended,
    deliberately, to any non-positive ``concurrency`` (``"empty"``, or a
    ``"ttft_bound"`` violation so severe even the lowest level tried was
    unusable) — writing ``0`` as a capacity would not merely be an unproven
    number, it would make the role permanently unselectable in the replica
    pool (``active >= capacity`` with ``capacity == 0`` is always true).
    """
    if knee.stopped_by == "top_of_ramp":
        return (
            "the ramp never plateaued — it was still rising meaningfully at the "
            "top of --schedule; widen --schedule or accept the report without "
            "--apply"
        )
    if knee.concurrency <= 0:
        return f"no usable concurrency level was measured (stopped_by={knee.stopped_by!r})"
    return None


def _render_text(result: dict) -> str:
    lines = [
        f"lobes calibrate — role={result['role']} (backend={result['backend']})",
        f"  concurrency: {result['concurrency']}",
        f"  plateaued:   {result['plateaued']}",
        f"  stopped by:  {_STOPPED_BY_PROSE.get(result['stopped_by'], result['stopped_by'])}",
        "  samples (concurrency, aggregate_tok_s, ttft_s):",
    ]
    for concurrency, tok_s, ttft_s in result["samples"]:
        lines.append(f"    {concurrency:>4}  {tok_s:>10.2f} tok/s  {ttft_s:>8.3f} s")
    if result["applied"]:
        lines.append(f"  wrote {result['env_key']}={result['concurrency']} to .env")
    elif result.get("refused"):
        lines.append(f"  --apply refused: {result['refused']}")
    else:
        lines.append("  (dry run — pass --apply to write this to .env)")
    return "\n".join(lines)


def cmd_calibrate(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    role = args.role
    schedule = _parse_schedule(getattr(args, "schedule", None))
    apply = bool(getattr(args, "apply", False))

    # Validate every numeric argument this verb added BEFORE any network call
    # — a non-finite (NaN/Infinity) or non-positive value must fail loudly
    # here, not silently disable a guard deep inside calibration_knee (F7).
    ttft_bound_s = _require_finite(args.ttft_bound_s, "--ttft-bound-s")
    min_relative_gain = _require_finite(
        getattr(args, "min_relative_gain", None), "--min-relative-gain", allow_none=True
    )
    if args.max_tokens <= 0 or not math.isfinite(args.max_tokens):
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"--max-tokens must be a positive integer; got {args.max_tokens!r}",
            remediation="pass a positive integer, e.g. --max-tokens 128",
        )
    timeout = _require_finite(float(args.timeout), "--timeout")

    info, deploy_dir = _resolve_role_info(args, role)
    if not info.loaded or not info.endpoint:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"role {role!r} is not loaded/wired in this deployment — nothing to calibrate",
            remediation=f"wire it first ('lobes up {role} --apply'), or calibrate a loaded role",
        )

    headers = _runtime_ops.gateway_auth_headers(deploy_dir)
    with _assess.auth_headers(headers), _runtime_ops.friendly_unauthorized_errors(deploy_dir):
        knee = drive_calibration(
            info.endpoint,
            info.model,
            schedule=schedule,
            ttft_bound_s=ttft_bound_s,
            min_relative_gain=min_relative_gain,
            max_tokens=args.max_tokens,
            timeout=int(timeout),
        )

    backend = ROLE_BACKEND[role]
    env_key = MAX_ACTIVE_ENV[backend]
    result: dict = {
        "role": role,
        "backend": backend,
        "env_key": env_key,
        "concurrency": knee.concurrency,
        "plateaued": knee.plateaued,
        "stopped_by": knee.stopped_by,
        "samples": [list(s) for s in knee.samples],
        "applied": False,
        "refused": None,
    }

    if apply:
        if deploy_dir is None:
            raise ModelGearError(
                code=EXIT_ENV_ERROR,
                message="no scaffolded deployment found to write a capacity into",
                remediation="scaffold one first ('lobes init --apply'), or pass --compose-dir",
            )
        refusal = _refuse_reason(knee)
        if refusal is not None:
            # Nothing goes to stdout here: this command is about to exit
            # nonzero, and this repo's strict stdout/stderr contract
            # (lobes/cli/_output.py) reserves stdout for a SUCCESSFUL result.
            # Emitting a success-shaped payload to stdout before raising the
            # actual failure (which the dispatcher writes to stderr) would
            # let automation that only reads stdout consume a misleading
            # "result" despite the nonzero exit (Qodo review finding F8 on
            # #221) — sibling write verbs refusing a mutation (e.g.
            # `lobes init`'s unknown-shape/--single-conflict checks, `lobes
            # up`'s wiring checks) raise directly with no prior emit_result,
            # and this matches that convention exactly.
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"refusing to write a measured capacity for {role!r}: {refusal}",
                remediation="re-run without --apply to inspect the ramp, or widen --schedule",
            )
        env_path = deploy_dir / _compose.ENV_FILE
        _env.set_env(env_path, env_key, str(knee.concurrency))
        result["applied"] = True

    if json_mode:
        emit_result(result, json_mode=True)
    else:
        emit_result(_render_text(result), json_mode=False)
    return EXIT_SUCCESS


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "calibrate",
        help=(
            "Ramp concurrency against a role and measure its replica-pool "
            "serving capacity (a throughput knee — NOT --max-num-seqs). "
            "Read-only by default; --apply writes it to .env."
        ),
    )
    p.add_argument(
        "role",
        choices=ROLES,
        help="The Colleague role to calibrate (e.g. cortex).",
    )
    p.add_argument(
        "--ttft-bound-s",
        type=float,
        required=True,
        dest="ttft_bound_s",
        help=(
            "Declared TTFT bound in seconds — a level is only admissible while "
            "its TTFT stays at or under this bound. Deployment-specific; there "
            "is deliberately no default."
        ),
    )
    p.add_argument(
        "--schedule",
        default=None,
        help=(
            "Comma-separated concurrency levels to ramp through, ascending "
            f"(default: {','.join(str(v) for v in _DEFAULT_SCHEDULE)})."
        ),
    )
    p.add_argument(
        "--min-relative-gain",
        type=float,
        default=None,
        dest="min_relative_gain",
        help="Minimum relative throughput gain to keep ramping (default: 10%%).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        dest="max_tokens",
        help="Forced decode length per throughput request at each level (default 128).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Per-request timeout in seconds (default 300).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write the measured concurrency to .env as <PREFIX>_MAX_ACTIVE "
            "(default: dry run, report only). Refused when the ramp never "
            "plateaued. NOTE: the measured number is only valid for this "
            "(box, checkpoint, context window, speculative config) — a "
            "'lobes switch' or a shape re-render invalidates it; recalibrate "
            "after either."
        ),
    )
    p.add_argument("--port", type=int, help="Gateway host port (default: VLLM_PORT in .env).")
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_calibrate)
