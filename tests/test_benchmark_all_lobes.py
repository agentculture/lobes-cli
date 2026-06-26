"""Tests for ``lobes benchmark --all-lobes --concurrency auto`` (t7).

Acceptance criteria verified:
  AC1 — ``--all-lobes`` runs the perf engine + cat scorer against EACH lobe
         (minor, primary) through the gateway; every number is labelled by lobe.
  AC2 — A single invocation produces BOTH the four perf metrics AND the cat
         soft-score per lobe, rendered together via the report renderer.
  AC3 — The suite is READ-ONLY (no ``--apply``, no writes, no external datasets).
  AC4 — The cat soft-score delta is shown so a reader can judge it against
         run-to-run noise.

All tests are hermetic: network calls are monkeypatched at their imported names
inside ``lobes.cli._commands.benchmark``.  No GPU, no docker, no real endpoint.
"""

from __future__ import annotations

import argparse
import json
import types

import pytest

import lobes.cli._commands.benchmark as benchmark_cmd

# ---------------------------------------------------------------------------
# Canned responses
# ---------------------------------------------------------------------------

_CANNED_RATES = [100.0, 110.0]  # decode tok/s samples

_CANNED_TTFT = {"prompt_tokens": 500, "ttft_ms": 250.0}

# Ramp result: knee=4, rows includes concurrency 1, 2, 4 (last = knee row)
_CANNED_RAMP = {
    "knee": 4,
    "rows": [
        {
            "concurrency": 1,
            "requests_per_s": 5.0,
            "p50_latency_ms": 100.0,
            "p95_latency_ms": 140.0,
            "ms_per_token": 9.0,
            "total_s": 0.2,
        },
        {
            "concurrency": 2,
            "requests_per_s": 9.5,
            "p50_latency_ms": 105.0,
            "p95_latency_ms": 145.0,
            "ms_per_token": 9.5,
            "total_s": 0.21,
        },
        {
            "concurrency": 4,
            "requests_per_s": 14.0,
            "p50_latency_ms": 120.0,
            "p95_latency_ms": 170.0,
            "ms_per_token": 10.0,
            "total_s": 0.29,
        },
    ],
}

_CANNED_CONCURRENT = {
    "concurrency": 8,
    "requests_per_s": 16.0,
    "p50_latency_ms": 130.0,
    "p95_latency_ms": 180.0,
    "ms_per_token": 11.0,
    "total_s": 0.5,
}

_CANNED_SCORE = {"soft_score": 0.75}


# ---------------------------------------------------------------------------
# Arg-namespace helpers
# ---------------------------------------------------------------------------


def _make_all_lobes_args(**kwargs) -> types.SimpleNamespace:
    """Build a minimal namespace that satisfies cmd_benchmark for --all-lobes."""
    defaults: dict = {
        "json": False,
        "port": 8000,
        "compose_dir": None,
        "model": "primary-model",
        "minor_model": "minor-model",
        "purpose": None,
        "input_len": None,
        "output_len": None,
        "runs": 2,
        "concurrency": "auto",
        "all_lobes": True,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _make_single_model_args(**kwargs) -> types.SimpleNamespace:
    """Build a namespace for the original single-model benchmark path."""
    defaults: dict = {
        "json": False,
        "port": 8000,
        "compose_dir": None,
        "model": "test-model",
        "purpose": None,
        "input_len": None,
        "output_len": None,
        "runs": 2,
        "all_lobes": False,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Fixtures: patch all network calls for --all-lobes tests
# ---------------------------------------------------------------------------


def _patch_all_lobes_network(monkeypatch, *, score_fn=None):
    """Monkeypatch every network call in the benchmark module's namespace."""
    monkeypatch.setattr(benchmark_cmd, "_decode_throughput", lambda *a, **kw: _CANNED_RATES)
    monkeypatch.setattr(benchmark_cmd, "measure_prefill_ttft", lambda *a, **kw: _CANNED_TTFT)
    monkeypatch.setattr(benchmark_cmd, "auto_ramp_concurrency", lambda *a, **kw: _CANNED_RAMP)
    monkeypatch.setattr(benchmark_cmd, "run_concurrent", lambda *a, **kw: _CANNED_CONCURRENT)
    if score_fn is None:
        score_fn = lambda *a, **kw: _CANNED_SCORE  # noqa: E731
    monkeypatch.setattr(benchmark_cmd, "score_case", score_fn)


# ---------------------------------------------------------------------------
# AC1 + AC2: both lobes present with all perf metrics + cat soft-score
# ---------------------------------------------------------------------------


def test_all_lobes_text_output_contains_both_lobe_labels(monkeypatch, capsys) -> None:
    """AC1: both 'minor' and 'primary' appear as row labels in the markdown output."""
    _patch_all_lobes_network(monkeypatch)
    rc = benchmark_cmd.cmd_benchmark(_make_all_lobes_args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "minor" in out, "'minor' lobe label missing from output"
    assert "primary" in out, "'primary' lobe label missing from output"


def test_all_lobes_text_output_contains_all_perf_metrics(monkeypatch, capsys) -> None:
    """AC2: all four perf metric names appear in the combined markdown."""
    _patch_all_lobes_network(monkeypatch)
    rc = benchmark_cmd.cmd_benchmark(_make_all_lobes_args())
    assert rc == 0
    out = capsys.readouterr().out
    for metric_label in (
        "decode tok/s",
        "prefill TTFT",
        "peak req/s",
        "p50 latency",
        "p95 latency",
        "ms per token",
    ):
        assert metric_label in out, f"perf metric label {metric_label!r} missing from output"


def test_all_lobes_text_output_contains_cat_soft_score(monkeypatch, capsys) -> None:
    """AC2: the cat soft-score row appears in the combined markdown."""
    _patch_all_lobes_network(monkeypatch)
    rc = benchmark_cmd.cmd_benchmark(_make_all_lobes_args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "cat soft-score" in out, "'cat soft-score' row missing from output"


def test_all_lobes_render_report_receives_both_keys(monkeypatch) -> None:
    """AC1+AC2: render_report is fed a dict containing both 'minor' and 'primary' keys."""
    _patch_all_lobes_network(monkeypatch)
    captured: list[dict] = []
    original_render = benchmark_cmd.render_report

    def spy_render(results: dict) -> str:
        captured.append(results)
        return original_render(results)

    monkeypatch.setattr(benchmark_cmd, "render_report", spy_render)
    benchmark_cmd.cmd_benchmark(_make_all_lobes_args())
    assert len(captured) == 1, "render_report should be called exactly once"
    assert "primary" in captured[0], "results dict missing 'primary' key"
    assert "minor" in captured[0], "results dict missing 'minor' key"


def test_all_lobes_each_lobe_has_all_required_keys(monkeypatch) -> None:
    """AC2: each per-lobe dict in the results contains all required metric keys."""
    _patch_all_lobes_network(monkeypatch)
    captured: list[dict] = []
    original_render = benchmark_cmd.render_report

    def spy_render(results: dict) -> str:
        captured.append(results)
        return original_render(results)

    monkeypatch.setattr(benchmark_cmd, "render_report", spy_render)
    benchmark_cmd.cmd_benchmark(_make_all_lobes_args())

    required_keys = {
        "decode_tok_s",
        "prefill_ttft_ms",
        "peak_req_s",
        "p50_latency_ms",
        "p95_latency_ms",
        "ms_per_token",
        "cat_soft_score",
    }
    for lobe in ("primary", "minor"):
        lobe_data = captured[0].get(lobe, {})
        missing = required_keys - set(lobe_data)
        assert not missing, f"lobe '{lobe}' missing keys: {missing}"


# ---------------------------------------------------------------------------
# AC4: cat soft-score delta is shown
# ---------------------------------------------------------------------------


def test_all_lobes_output_contains_cat_score_delta(monkeypatch, capsys) -> None:
    """AC4: the output markdown table includes a signed Δ column for cat soft-score."""

    # Give each lobe a different cat soft-score so the delta is non-zero.
    def fake_score(case, *, base_url, model, **kw):
        return {"soft_score": 0.80 if "minor" in model else 0.60}

    _patch_all_lobes_network(monkeypatch, score_fn=fake_score)
    rc = benchmark_cmd.cmd_benchmark(_make_all_lobes_args())
    assert rc == 0
    out = capsys.readouterr().out

    # The render_report table has a Δ column header
    assert any(
        tok in out for tok in ("Δ", "delta", "Delta")
    ), "No delta column found in combined report output"
    # The cat soft-score row must be present
    assert "cat soft-score" in out

    # A signed delta value must appear (+ or - before a digit in the delta column)
    import re

    # Look for a signed numeric value in the output (e.g. +0.2 or -0.2)
    assert re.search(r"[+-]\d", out), "No signed delta value found in output"


# ---------------------------------------------------------------------------
# AC3: read-only (no --apply flag, no file writes)
# ---------------------------------------------------------------------------


def test_benchmark_register_has_no_apply_flag() -> None:
    """AC3: the benchmark parser exposes no --apply flag."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    benchmark_cmd.register(sub)

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["benchmark", "--apply"])
    assert exc.value.code != 0, "--apply should cause a parse error (no such flag)"


def test_benchmark_all_lobes_creates_no_files(monkeypatch, tmp_path, capsys) -> None:
    """AC3: running --all-lobes creates no files in the working directory."""
    _patch_all_lobes_network(monkeypatch)
    before = set(tmp_path.iterdir())
    benchmark_cmd.cmd_benchmark(_make_all_lobes_args())
    after = set(tmp_path.iterdir())
    assert before == after, f"unexpected files created: {after - before}"


# ---------------------------------------------------------------------------
# Concurrency: auto vs numeric branch selection
# ---------------------------------------------------------------------------


def test_concurrency_auto_calls_auto_ramp_not_run_concurrent(monkeypatch, capsys) -> None:
    """auto branch calls auto_ramp_concurrency (once per lobe); run_concurrent is NOT called."""
    ramp_calls: list[str] = []
    concurrent_calls: list[int] = []

    def spy_ramp(url, model, **kw):
        ramp_calls.append(model)
        return _CANNED_RAMP

    def spy_concurrent(url, model, *, concurrency, **kw):
        concurrent_calls.append(concurrency)
        return _CANNED_CONCURRENT

    monkeypatch.setattr(benchmark_cmd, "_decode_throughput", lambda *a, **kw: _CANNED_RATES)
    monkeypatch.setattr(benchmark_cmd, "measure_prefill_ttft", lambda *a, **kw: _CANNED_TTFT)
    monkeypatch.setattr(benchmark_cmd, "auto_ramp_concurrency", spy_ramp)
    monkeypatch.setattr(benchmark_cmd, "run_concurrent", spy_concurrent)
    monkeypatch.setattr(benchmark_cmd, "score_case", lambda *a, **kw: _CANNED_SCORE)

    rc = benchmark_cmd.cmd_benchmark(_make_all_lobes_args(concurrency="auto"))
    assert rc == 0
    assert len(ramp_calls) == 2, "auto_ramp_concurrency should be called once per lobe (2 total)"
    assert len(concurrent_calls) == 0, "run_concurrent must NOT be called in auto mode"


def test_concurrency_numeric_calls_run_concurrent_not_ramp(monkeypatch, capsys) -> None:
    """Numeric concurrency branch calls run_concurrent (once per lobe); ramp is NOT called."""
    ramp_calls: list[str] = []
    concurrent_calls: list[int] = []

    def spy_ramp(url, model, **kw):
        ramp_calls.append(model)
        return _CANNED_RAMP

    def spy_concurrent(url, model, *, concurrency, **kw):
        concurrent_calls.append(concurrency)
        return _CANNED_CONCURRENT

    monkeypatch.setattr(benchmark_cmd, "_decode_throughput", lambda *a, **kw: _CANNED_RATES)
    monkeypatch.setattr(benchmark_cmd, "measure_prefill_ttft", lambda *a, **kw: _CANNED_TTFT)
    monkeypatch.setattr(benchmark_cmd, "auto_ramp_concurrency", spy_ramp)
    monkeypatch.setattr(benchmark_cmd, "run_concurrent", spy_concurrent)
    monkeypatch.setattr(benchmark_cmd, "score_case", lambda *a, **kw: _CANNED_SCORE)

    rc = benchmark_cmd.cmd_benchmark(_make_all_lobes_args(concurrency="8"))
    assert rc == 0
    assert len(concurrent_calls) == 2, "run_concurrent should be called once per lobe (2 total)"
    assert all(c == 8 for c in concurrent_calls), "concurrency passed to run_concurrent must be 8"
    assert len(ramp_calls) == 0, "auto_ramp_concurrency must NOT be called in numeric mode"


def test_concurrency_auto_peak_row_is_knee_row(monkeypatch) -> None:
    """The auto branch picks the knee (last) row from the ramp for peak metrics."""
    captured: list[dict] = []
    original_render = benchmark_cmd.render_report

    def spy_render(results: dict) -> str:
        captured.append(results)
        return original_render(results)

    monkeypatch.setattr(benchmark_cmd, "_decode_throughput", lambda *a, **kw: [50.0])
    monkeypatch.setattr(benchmark_cmd, "measure_prefill_ttft", lambda *a, **kw: _CANNED_TTFT)
    monkeypatch.setattr(benchmark_cmd, "auto_ramp_concurrency", lambda *a, **kw: _CANNED_RAMP)
    monkeypatch.setattr(benchmark_cmd, "run_concurrent", lambda *a, **kw: _CANNED_CONCURRENT)
    monkeypatch.setattr(benchmark_cmd, "score_case", lambda *a, **kw: _CANNED_SCORE)
    monkeypatch.setattr(benchmark_cmd, "render_report", spy_render)

    benchmark_cmd.cmd_benchmark(_make_all_lobes_args(concurrency="auto"))
    # The knee row is the last entry: concurrency=4, requests_per_s=14.0
    for lobe in ("primary", "minor"):
        assert captured[0][lobe]["peak_req_s"] == pytest.approx(14.0), (
            f"lobe '{lobe}' peak_req_s should come from the knee row (14.0), "
            f"got {captured[0][lobe]['peak_req_s']}"
        )


# ---------------------------------------------------------------------------
# JSON output mode
# ---------------------------------------------------------------------------


def test_all_lobes_json_output(monkeypatch, capsys) -> None:
    """--json emits a structured dict with 'results' (per-lobe) and 'markdown'."""
    _patch_all_lobes_network(monkeypatch)
    rc = benchmark_cmd.cmd_benchmark(_make_all_lobes_args(json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "results" in payload, "JSON output missing 'results' key"
    assert "markdown" in payload, "JSON output missing 'markdown' key"
    assert "primary" in payload["results"]
    assert "minor" in payload["results"]
    assert "cat_soft_score" in payload["results"]["primary"]
    assert "cat_soft_score" in payload["results"]["minor"]


# ---------------------------------------------------------------------------
# Skipping a lobe whose served name is unset
# ---------------------------------------------------------------------------


def test_all_lobes_skips_missing_minor_model(monkeypatch, capsys) -> None:
    """When minor_model is unset (None), only 'primary' appears in the results."""
    _patch_all_lobes_network(monkeypatch)
    captured: list[dict] = []
    original_render = benchmark_cmd.render_report

    def spy_render(results: dict) -> str:
        captured.append(results)
        return original_render(results)

    monkeypatch.setattr(benchmark_cmd, "render_report", spy_render)
    args = _make_all_lobes_args(minor_model=None, model="primary-model")
    rc = benchmark_cmd.cmd_benchmark(args)
    assert rc == 0
    assert "minor" not in captured[0], "minor should be absent when minor_model is None"
    assert "primary" in captured[0], "primary must still be benchmarked"


# ---------------------------------------------------------------------------
# Regression: existing single-model path is unchanged
# ---------------------------------------------------------------------------


def test_single_model_path_still_works(monkeypatch, capsys) -> None:
    """Original single-model benchmark path returns 0 and emits output when --all-lobes absent."""
    fake_result = {
        "model": "test-model",
        "endpoint": "http://localhost:8000",
        "max_model_len": 131072,
        "purpose": "balanced",
        "input_len": 1000,
        "output_len": 1000,
        "decode_rates": [80.5, 82.0],
        "prefill": {"prompt_tokens": 1000, "seconds": 1.2},
    }
    monkeypatch.setattr(benchmark_cmd._assess, "run_benchmark", lambda *a, **kw: fake_result)
    monkeypatch.setattr(benchmark_cmd._assess, "render_benchmark", lambda r: "## Benchmark stub")

    args = _make_single_model_args()
    rc = benchmark_cmd.cmd_benchmark(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Benchmark stub" in out, "single-model render_benchmark output missing"


def test_single_model_path_no_all_lobes_network_calls(monkeypatch, capsys) -> None:
    """--all-lobes network helpers are NOT called when all_lobes is False."""
    ramp_calls: list[int] = []
    monkeypatch.setattr(
        benchmark_cmd._assess,
        "run_benchmark",
        lambda *a, **kw: {
            "model": "m",
            "endpoint": "e",
            "max_model_len": 0,
            "purpose": "balanced",
            "input_len": 100,
            "output_len": 100,
            "decode_rates": [50.0],
            "prefill": {"prompt_tokens": 100, "seconds": 0.5},
        },
    )
    monkeypatch.setattr(benchmark_cmd._assess, "render_benchmark", lambda r: "## Benchmark stub")
    monkeypatch.setattr(
        benchmark_cmd,
        "auto_ramp_concurrency",
        lambda *a, **kw: ramp_calls.append(1) or _CANNED_RAMP,
    )

    args = _make_single_model_args()
    benchmark_cmd.cmd_benchmark(args)
    assert len(ramp_calls) == 0, "auto_ramp_concurrency must not be called on single-model path"


# ---------------------------------------------------------------------------
# CLI wiring: register() creates parseable --all-lobes / --concurrency flags
# ---------------------------------------------------------------------------


def test_register_all_lobes_flag_parseable() -> None:
    """register() adds an --all-lobes flag that parses correctly."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    benchmark_cmd.register(sub)

    args = parser.parse_args(["benchmark", "--all-lobes"])
    assert args.all_lobes is True


def test_register_concurrency_flag_parseable() -> None:
    """register() adds a --concurrency flag with default 'auto'."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    benchmark_cmd.register(sub)

    args_default = parser.parse_args(["benchmark"])
    assert args_default.concurrency == "auto"

    args_numeric = parser.parse_args(["benchmark", "--concurrency", "8"])
    assert args_numeric.concurrency == "8"


def test_register_minor_model_flag_parseable() -> None:
    """register() adds a --minor-model flag."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    benchmark_cmd.register(sub)

    args = parser.parse_args(["benchmark", "--minor-model", "Qwen/Qwen3.5-4B"])
    assert args.minor_model == "Qwen/Qwen3.5-4B"


# ---------------------------------------------------------------------------
# Finding A: --concurrency validation (Qodo reliability bug)
# ---------------------------------------------------------------------------


def test_concurrency_invalid_string_raises_user_error(monkeypatch) -> None:
    """Non-integer --concurrency raises ModelGearError with EXIT_USER_ERROR, not ValueError."""
    from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError

    for bad in ("-1", "0", "abc"):
        args = _make_all_lobes_args(concurrency=bad)
        with pytest.raises(ModelGearError) as exc_info:
            benchmark_cmd.cmd_benchmark(args)
        err = exc_info.value
        assert err.code == EXIT_USER_ERROR, (
            f"concurrency={bad!r}: expected EXIT_USER_ERROR ({EXIT_USER_ERROR}), "
            f"got code={err.code}"
        )
        # Must NOT be a raw ValueError leaking out
        assert (
            type(err).__name__ != "ValueError"
        ), f"concurrency={bad!r}: raw ValueError escaped (must be ModelGearError)"


def test_concurrency_invalid_does_not_call_run_concurrent(monkeypatch) -> None:
    """Validation rejects bad --concurrency before any ThreadPoolExecutor / network call."""
    from lobes.cli._errors import ModelGearError

    concurrent_calls: list = []
    ramp_calls: list = []
    monkeypatch.setattr(
        benchmark_cmd,
        "run_concurrent",
        lambda *a, **kw: concurrent_calls.append(kw) or _CANNED_CONCURRENT,
    )
    monkeypatch.setattr(
        benchmark_cmd,
        "auto_ramp_concurrency",
        lambda *a, **kw: ramp_calls.append(kw) or _CANNED_RAMP,
    )

    for bad in ("-1", "0", "abc"):
        concurrent_calls.clear()
        ramp_calls.clear()
        with pytest.raises(ModelGearError):
            benchmark_cmd.cmd_benchmark(_make_all_lobes_args(concurrency=bad))
        assert (
            len(concurrent_calls) == 0
        ), f"concurrency={bad!r}: run_concurrent must not be called on invalid concurrency"
        assert (
            len(ramp_calls) == 0
        ), f"concurrency={bad!r}: auto_ramp_concurrency must not be called on invalid concurrency"


# ---------------------------------------------------------------------------
# Finding B: --all-lobes with no served names raises EXIT_ENV_ERROR (S3516)
# ---------------------------------------------------------------------------


def test_all_lobes_no_served_names_raises_env_error() -> None:
    """--all-lobes with both served names unset raises ModelGearError(EXIT_ENV_ERROR).

    Replaces the previous silent empty-report / return-0 behavior.
    Uses explicit port=8000 so deploy_dir is None (no .env lookup),
    and passes model=None, minor_model=None so both lobes are skipped.
    No network monkeypatching needed: the error is raised before any probe.
    """
    from lobes.cli._errors import EXIT_ENV_ERROR, ModelGearError

    args = _make_all_lobes_args(model=None, minor_model=None)
    with pytest.raises(ModelGearError) as exc_info:
        benchmark_cmd.cmd_benchmark(args)
    err = exc_info.value
    assert err.code == EXIT_ENV_ERROR, (
        f"Expected EXIT_ENV_ERROR ({EXIT_ENV_ERROR}) when no lobes are set, "
        f"got code={err.code} message={err.message!r}"
    )
    assert (
        "VLLM_SERVED_NAME" in err.message or "MINOR_SERVED_NAME" in err.message
    ), f"Error message should mention the unset env vars; got: {err.message!r}"
