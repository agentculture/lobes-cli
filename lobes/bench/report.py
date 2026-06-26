"""Combined per-lobe markdown report renderer with deltas.

Renders benchmark results as a GitHub-flavored markdown table comparing
minor vs primary lobes, with explicit per-metric deltas.
"""

from __future__ import annotations

_NAME_MAP = {
    "decode_tok_s": "decode tok/s",
    "prefill_ttft_ms": "prefill TTFT (ms)",
    "peak_req_s": "peak req/s",
    "p50_latency_ms": "p50 latency (ms)",
    "p95_latency_ms": "p95 latency (ms)",
    "ms_per_token": "ms per token",  # nosec B105 — metric label, not a secret
    "cat_soft_score": "cat soft-score",
}


def _format_metric_name(key: str) -> str:
    """Convert snake_case metric key to title case display name."""
    return _NAME_MAP.get(key, key.replace("_", " ").title())


def _format_value(value) -> str:
    """Format a metric value, rounding to sensible precision."""
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)):
        return str(value)
    # Round to 2 decimal places for most metrics
    return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(int(value))


def _format_delta(minor_val, primary_val) -> str:
    """Format the delta (minor - primary) with sign."""
    if minor_val is None or primary_val is None:
        return "n/a"
    if not isinstance(minor_val, (int, float)) or not isinstance(primary_val, (int, float)):
        return "n/a"
    delta = minor_val - primary_val
    # Format with sign
    if delta >= 0:
        return f"+{delta:.2f}".rstrip("0").rstrip(".")
    else:
        return f"{delta:.2f}".rstrip("0").rstrip(".")


def render_report(results: dict) -> str:
    """Render per-lobe benchmark results as a markdown table.

    Parameters
    ----------
    results : dict
        A dict with keys "minor" and "primary", each containing a dict of metrics:

        >>> {
        ...     "minor": {
        ...         "decode_tok_s": float,
        ...         "prefill_ttft_ms": float,
        ...         "peak_req_s": float,
        ...         "p50_latency_ms": float,
        ...         "p95_latency_ms": float,
        ...         "ms_per_token": float,
        ...         "cat_soft_score": float,
        ...     },
        ...     "primary": {<same keys>},
        ... }

        Missing metrics in either lobe are rendered as 'n/a'.

    Returns
    -------
    str
        A GitHub-flavored markdown table with one row per metric.
        Columns: Metric | minor | primary | Δ (minor−primary).
        Deltas are signed (e.g., +12.3, -0.04).
    """
    minor_data = results.get("minor", {})
    primary_data = results.get("primary", {})

    # Collect all metric keys from both lobes to ensure we don't miss any
    all_metrics = set(minor_data.keys()) | set(primary_data.keys())

    # Define the display order: performance metrics first, then cat soft-score
    metric_order = [
        "decode_tok_s",
        "prefill_ttft_ms",
        "peak_req_s",
        "p50_latency_ms",
        "p95_latency_ms",
        "ms_per_token",
        "cat_soft_score",
    ]

    # Build ordered list, adding any extra metrics not in the order
    ordered_metrics = [m for m in metric_order if m in all_metrics]
    extra_metrics = sorted(all_metrics - set(metric_order))
    ordered_metrics.extend(extra_metrics)

    # Build the markdown table
    lines = []

    # Header
    lines.append("| Metric | minor | primary | Δ (minor−primary) |")
    lines.append("|--------|-------|---------|------------------|")

    # Data rows
    for metric_key in ordered_metrics:
        minor_val = minor_data.get(metric_key)
        primary_val = primary_data.get(metric_key)

        metric_name = _format_metric_name(metric_key)
        minor_str = _format_value(minor_val)
        primary_str = _format_value(primary_val)
        delta_str = _format_delta(minor_val, primary_val)

        lines.append(f"| {metric_name} | {minor_str} | {primary_str} | {delta_str} |")

    return "\n".join(lines)
