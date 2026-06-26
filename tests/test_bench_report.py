"""Tests for lobes.bench.report — combined per-lobe markdown report renderer."""

from __future__ import annotations

from lobes.bench.report import render_report


def test_render_report_complete_data():
    """Test render_report with complete minor and primary results.

    Verifies:
    - Output is a markdown table (contains | and --- separator)
    - Contains columns for minor and primary
    - Includes one row per metric (decode_tok_s, prefill_ttft_ms, etc.)
    - Displays correct signed deltas (minor - primary)
    - Includes cat soft-score row with correct delta
    """
    results = {
        "minor": {
            "decode_tok_s": 45.2,
            "prefill_ttft_ms": 123.4,
            "peak_req_s": 15.8,
            "p50_latency_ms": 85.5,
            "p95_latency_ms": 150.2,
            "ms_per_token": 22.1,
            "cat_soft_score": 0.78,
        },
        "primary": {
            "decode_tok_s": 50.1,
            "prefill_ttft_ms": 110.2,
            "peak_req_s": 18.3,
            "p50_latency_ms": 80.1,
            "p95_latency_ms": 145.8,
            "ms_per_token": 20.0,
            "cat_soft_score": 0.85,
        },
    }

    output = render_report(results)

    # Verify it's a markdown table
    assert "|" in output, "Output should contain | (table column separators)"
    assert "---" in output, "Output should contain --- (table header separator)"

    # Verify column headers
    assert "minor" in output.lower(), "Output should contain 'minor' column header"
    assert "primary" in output.lower(), "Output should contain 'primary' column header"
    assert "metric" in output.lower(), "Output should contain 'metric' column header"
    assert (
        "delta" in output.lower() or "δ" in output.lower() or "Δ" in output.lower()
    ), "Output should contain delta column header"

    # Verify metric rows
    assert "decode" in output.lower() and "tok" in output.lower(), "Should contain decode_tok_s row"
    assert "prefill" in output.lower(), "Should contain prefill_ttft_ms row"
    assert "peak" in output.lower() and "req" in output.lower(), "Should contain peak_req_s row"
    assert "latency" in output.lower(), "Should contain p50/p95 latency rows"
    assert (
        "ms_per_token" in output.lower() or "ms per token" in output.lower()
    ), "Should contain ms_per_token row"

    # Verify cat soft-score row
    assert (
        "cat" in output.lower() and "soft" in output.lower() and "score" in output.lower()
    ), "Should contain 'cat soft-score' row"

    # Verify expected delta values (minor - primary)
    # decode_tok_s: 45.2 - 50.1 = -4.9
    assert (
        "-4.9" in output or "-4.90" in output
    ), f"Should contain decode_tok_s delta of ~-4.9, got: {output}"

    # cat_soft_score: 0.78 - 0.85 = -0.07
    assert (
        "-0.07" in output or "-0.070" in output
    ), f"Should contain cat soft-score delta of ~-0.07, got: {output}"


def test_render_report_missing_metric():
    """Test render_report handles missing metrics gracefully.

    Verifies:
    - Function returns a string (does not raise)
    - Missing metric cells are marked as 'n/a'
    - Delta for missing metric is also 'n/a'
    """
    results = {
        "minor": {
            "decode_tok_s": 45.2,
            "prefill_ttft_ms": 123.4,
            "peak_req_s": 15.8,
            "p50_latency_ms": 85.5,
            "p95_latency_ms": 150.2,
            "ms_per_token": 22.1,
            "cat_soft_score": 0.78,
        },
        "primary": {
            # Deliberately omit ms_per_token
            "decode_tok_s": 50.1,
            "prefill_ttft_ms": 110.2,
            "peak_req_s": 18.3,
            "p50_latency_ms": 80.1,
            "p95_latency_ms": 145.8,
            "cat_soft_score": 0.85,
        },
    }

    # Should not raise
    output = render_report(results)

    # Verify output is a string
    assert isinstance(output, str), "render_report should return a string"

    # Verify it's still a valid markdown table
    assert "|" in output, "Output should still be a markdown table"
    assert "---" in output, "Output should still have table separator"

    # Verify n/a appears for missing metric
    assert (
        "n/a" in output.lower()
    ), f"Output should contain 'n/a' for missing ms_per_token in primary, got: {output}"
