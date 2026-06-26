"""``lobes benchmark`` — decode throughput + prefill latency for the served model.

Read-only. The workload shape is the active *purpose*: it defaults to the
configured ``VLLM_PURPOSE`` (so the numbers track the serve config) and can be
overridden with ``--purpose`` or explicit ``--input-len`` / ``--output-len``.
Forces a fixed decode length over a couple of runs and measures a prompt-sized
prefill, then emits a markdown block (plus host-side facts) for a per-model doc
under ``docs/``. Correctness lives in ``lobes assess``.

``--all-lobes`` benchmarks BOTH the primary and minor lobes through the gateway
in one combined report (perf metrics + cat soft-score per lobe, rendered via
:func:`lobes.bench.report.render_report`).  Read-only — no ``--apply``, no writes.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from lobes import assess as _assess
from lobes import profiles
from lobes.assess import (
    _decode_throughput,
    auto_ramp_concurrency,
    measure_prefill_ttft,
    run_concurrent,
)
from lobes.bench.cat_probe import generate_case
from lobes.bench.cat_score import score_case
from lobes.bench.report import render_report
from lobes.cli import _runtime_ops
from lobes.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.runtime import _compose, _env

# Fixed seed set for the cat soft-score probe (3 cases; fully reproducible).
_CAT_SEEDS: tuple[int, ...] = (0, 1, 2)


def _resolve_shape(args, deploy_dir) -> tuple[profiles.WorkloadProfile, int, int]:
    """Resolve the (purpose, input_len, output_len) shape — flag > .env > default."""
    purpose = args.purpose
    if purpose is None and deploy_dir is not None:
        purpose = _env.read_env(
            deploy_dir / _compose.ENV_FILE, "VLLM_PURPOSE", profiles.DEFAULT_PURPOSE
        )
    wl = profiles.workload_profile(purpose or profiles.DEFAULT_PURPOSE)
    input_len = args.input_len if args.input_len is not None else wl.bench_input_len
    output_len = args.output_len if args.output_len is not None else wl.bench_output_len
    return wl, input_len, output_len


def _bench_one_lobe(
    url: str,
    model: str,
    *,
    concurrency: "str | int" = "auto",
    output_len: int = 128,
    runs: int = 2,
    input_len: int = 2000,
) -> dict:
    """Measure per-lobe perf (t3) + cat soft-score (t5); return the render_report shape.

    Returns a dict with keys matching the ``render_report`` input contract::

        {
            "decode_tok_s": float,
            "prefill_ttft_ms": float,
            "peak_req_s": float,
            "p50_latency_ms": float,
            "p95_latency_ms": float,
            "ms_per_token": float,
            "cat_soft_score": float,
        }

    Network calls: ``_decode_throughput``, ``measure_prefill_ttft``,
    ``auto_ramp_concurrency`` (or ``run_concurrent``), and ``score_case`` — all
    patchable at their imported names in this module for hermetic tests.
    """
    # Decode throughput: mean of _decode_throughput samples
    rates = _decode_throughput(url, model, output_len, runs)
    decode_tok_s = statistics.mean(rates) if rates else 0.0

    # Prefill TTFT
    ttft_result = measure_prefill_ttft(url, model, input_len=input_len)
    prefill_ttft_ms = ttft_result["ttft_ms"]

    # Concurrent throughput — auto ramp (knee-find) or fixed concurrency
    if concurrency == "auto":
        ramp = auto_ramp_concurrency(url, model)
        # _find_knee returns rows[:i] where the last entry is the knee (peak-throughput) row
        row = ramp["rows"][-1] if ramp["rows"] else {}
    else:
        row = run_concurrent(url, model, concurrency=int(concurrency))

    peak_req_s = float(row.get("requests_per_s", 0.0))
    p50_latency_ms = float(row.get("p50_latency_ms", 0.0))
    p95_latency_ms = float(row.get("p95_latency_ms", 0.0))
    ms_per_token = float(row.get("ms_per_token", 0.0))

    # Cat soft-score: mean over the fixed seed set
    cat_scores: list[float] = []
    for seed in _CAT_SEEDS:
        case = generate_case(seed=seed, mode="closed")
        scored = score_case(case, base_url=url.rstrip("/") + "/v1", model=model)
        cat_scores.append(float(scored["soft_score"]))
    cat_soft_score = statistics.mean(cat_scores) if cat_scores else 0.0

    return {
        "decode_tok_s": decode_tok_s,
        "prefill_ttft_ms": prefill_ttft_ms,
        "peak_req_s": peak_req_s,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "ms_per_token": ms_per_token,
        "cat_soft_score": cat_soft_score,
    }


def _bench_all_lobes(
    url: str,
    primary_model: str | None,
    minor_model: str | None,
    *,
    concurrency: "str | int" = "auto",
    output_len: int = 128,
    runs: int = 2,
    input_len: int = 2000,
) -> dict:
    """Orchestrate per-lobe benchmarks; returns the results dict for :func:`render_report`.

    Skips any lobe whose served name is unset (with a labelled note to stderr).
    Both lobes share the same gateway URL; they are distinguished only by model name.
    """
    results: dict = {}
    for lobe_name, model in (("primary", primary_model), ("minor", minor_model)):
        if not model:
            print(f"[{lobe_name}] served name not set — skipping", file=sys.stderr)
            continue
        results[lobe_name] = _bench_one_lobe(
            url,
            model,
            concurrency=concurrency,
            output_len=output_len,
            runs=runs,
            input_len=input_len,
        )
    return results


def cmd_benchmark(args: argparse.Namespace) -> int:
    all_lobes = bool(getattr(args, "all_lobes", False))
    json_mode = bool(getattr(args, "json", False))
    port, deploy_dir = _runtime_ops.resolve_port_soft(args)
    url = f"http://localhost:{port}"

    if all_lobes:
        # --all-lobes: benchmark BOTH lobes through the gateway in one combined report.
        primary_model: str | None = getattr(args, "model", None)
        minor_model: str | None = getattr(args, "minor_model", None)
        if primary_model is None and deploy_dir is not None:
            primary_model = _env.read_env(deploy_dir / _compose.ENV_FILE, "VLLM_SERVED_NAME")
        if minor_model is None and deploy_dir is not None:
            minor_model = _env.read_env(deploy_dir / _compose.ENV_FILE, "MINOR_SERVED_NAME")

        _, input_len, output_len = _resolve_shape(args, deploy_dir)
        concurrency_raw: str = getattr(args, "concurrency", "auto")

        # Validate --concurrency: must be "auto" or a positive integer string.
        # Parse to int once here so downstream helpers receive the typed value.
        if concurrency_raw == "auto":
            concurrency_val: "str | int" = "auto"
        else:
            try:
                concurrency_int = int(concurrency_raw)
            except (ValueError, TypeError):
                raise ModelGearError(
                    code=EXIT_USER_ERROR,
                    message=(
                        f"--concurrency must be 'auto' or a positive integer; "
                        f"got {concurrency_raw!r}"
                    ),
                    remediation=(
                        "pass --concurrency auto (default) or a positive integer "
                        "such as --concurrency 8"
                    ),
                )
            if concurrency_int <= 0:
                raise ModelGearError(
                    code=EXIT_USER_ERROR,
                    message=(f"--concurrency must be a positive integer; got {concurrency_int}"),
                    remediation=(
                        "pass --concurrency auto (default) or a positive integer "
                        "such as --concurrency 8"
                    ),
                )
            concurrency_val = concurrency_int

        results = _bench_all_lobes(
            url,
            primary_model,
            minor_model,
            concurrency=concurrency_val,
            output_len=output_len,
            runs=args.runs,
            input_len=input_len,
        )

        if not results:
            raise ModelGearError(
                code=EXIT_ENV_ERROR,
                message=(
                    "no lobes to benchmark: neither VLLM_SERVED_NAME nor "
                    "MINOR_SERVED_NAME is set"
                ),
                remediation=(
                    "set a served model name in the deployment .env, " "or run without --all-lobes"
                ),
            )

        markdown = render_report(results)
        if json_mode:
            emit_result({"results": results, "markdown": markdown}, json_mode=True)
        else:
            emit_result(markdown, json_mode=False)
        return 0

    # Original single-model path (unchanged when --all-lobes is absent).
    model = args.model
    if model is None and deploy_dir is not None:
        model = _env.read_env(deploy_dir / _compose.ENV_FILE, "VLLM_SERVED_NAME")

    wl, input_len, output_len = _resolve_shape(args, deploy_dir)
    result = _assess.run_benchmark(
        url,
        model,
        purpose=wl.name,
        input_len=input_len,
        output_len=output_len,
        runs=args.runs,
    )
    host = {"image": _compose.container_image(), "gpu_memory": _compose.gpu_engine_mem()}

    if json_mode:
        emit_result({**result, "host": host}, json_mode=True)
    else:
        header = (
            "### Host-side\n"
            f"- Image: `{host['image']}`  ·  GPU memory (EngineCore): {host['gpu_memory']}\n"
        )
        emit_result(header + "\n" + _assess.render_benchmark(result), json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "benchmark",
        help="Decode throughput + prefill latency for the served model (markdown for a doc).",
    )
    p.add_argument("--port", type=int, help="Host port (default: VLLM_PORT in .env, else 8000).")
    p.add_argument(
        "--model", help="Served model name (default: VLLM_SERVED_NAME, else first /v1/models)."
    )
    p.add_argument(
        "--purpose",
        choices=[wp.name for wp in profiles.WORKLOAD_PROFILES],
        default=None,
        help="Workload shape (default: the configured VLLM_PURPOSE, else balanced).",
    )
    p.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Override prompt length (default: the purpose's shape).",
    )
    p.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Override forced decode length (default: the purpose's shape).",
    )
    p.add_argument("--runs", type=int, default=2, help="Decode-throughput repetitions (default 2).")
    p.add_argument("--compose-dir", help="Deployment dir (default: $LOBES_DIR or ~/.lobes).")
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    # --all-lobes branch: combined per-lobe report (t7)
    p.add_argument(
        "--all-lobes",
        action="store_true",
        default=False,
        dest="all_lobes",
        help=(
            "Benchmark ALL lobes (minor + primary) through the gateway "
            "in one combined report (perf + cat soft-score per lobe)."
        ),
    )
    p.add_argument(
        "--minor-model",
        default=None,
        dest="minor_model",
        help="Minor lobe model name (default: MINOR_SERVED_NAME in .env).",
    )
    p.add_argument(
        "--concurrency",
        default="auto",
        help=(
            "Concurrency for throughput test: 'auto' (knee-find ramp) "
            "or an integer (default: auto)."
        ),
    )
    p.set_defaults(func=cmd_benchmark)
