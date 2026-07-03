"""Combined per-lobe markdown report renderer with deltas.

Renders benchmark results as a GitHub-flavored markdown table comparing
minor vs primary lobes, with explicit per-metric deltas.

:func:`render_report` stays byte-for-byte unchanged (its ``minor``/``primary``
keys and delta-column wording are characterization-tested) — the general N-ary
sibling is :func:`render_side_by_side`, added for issue #81 t9's profile-
comparison mode, which formats an ARBITRARY set of named columns (role names
like ``cortex``/``senses``, or catalog-variant labels like ``nvfp4``/``bf16``)
side by side instead of the fixed two-lobe shape.
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


def render_side_by_side(columns: dict, *, metric_order: list[str] | None = None) -> str:
    """Render an ARBITRARY number of named columns side by side — issue #81 t9.

    The general sibling of :func:`render_report`: where that function is fixed
    to exactly the ``minor``/``primary`` pair, this one takes any column count
    and any column labels (e.g. a single ``cortex`` column, a ``cortex``/
    ``senses`` pair, or an ``nvfp4``/``bf16`` pair) — the labels come straight
    from ``columns``' keys, in insertion order.

    Parameters
    ----------
    columns : dict
        Maps a column label to a flat metric dict, e.g.
        ``{"cortex": {"ttft_ms": 12.3, ...}, "senses": {"ttft_ms": 45.6, ...}}``.
        Missing metrics in any column render as ``n/a`` (same contract as
        :func:`render_report`).
    metric_order : list[str] | None
        Preferred row order for metric keys that appear in ``columns``. Keys
        outside this list are appended alphabetically (same fallback
        :func:`render_report` uses) so the table is deterministic regardless
        of dict/frozenset iteration order upstream.

    Returns
    -------
    str
        A GitHub-flavored markdown table: ``Metric`` + one column per label.
        When ``columns`` has EXACTLY two entries, a trailing signed
        ``Δ (label0−label1)`` column is appended (mirrors :func:`render_report`);
        for any other column count a pairwise delta is ambiguous, so it is
        omitted rather than guessed.
    """
    names = list(columns)
    all_metrics: set[str] = set()
    for data in columns.values():
        all_metrics |= set(data.keys())

    preferred = metric_order or []
    ordered_metrics = [m for m in preferred if m in all_metrics]
    ordered_metrics.extend(sorted(all_metrics - set(ordered_metrics)))

    show_delta = len(names) == 2
    header_cells = ["Metric", *names]
    if show_delta:
        header_cells.append(f"Δ ({names[0]}−{names[1]})")
    lines = ["| " + " | ".join(header_cells) + " |"]
    lines.append("|" + "|".join("---" for _ in header_cells) + "|")

    for metric_key in ordered_metrics:
        values = [columns[name].get(metric_key) for name in names]
        row_cells = [_format_metric_name(metric_key), *[_format_value(v) for v in values]]
        if show_delta:
            row_cells.append(_format_delta(values[0], values[1]))
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)
