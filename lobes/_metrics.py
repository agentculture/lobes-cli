"""Parse vLLM Prometheus ``/metrics`` + probe a backend's live state (stdlib only).

Shared by the gateway's ``/status`` fan-out and ``lobes overview --live``. The
parser is pure; the probes are best-effort and **never raise** — an unreachable
backend folds into a structured result so the live view degrades gracefully
instead of erroring. vLLM serves ``/metrics`` and ``/health`` unauthenticated, so
no API key is needed for either.
"""

from __future__ import annotations

import json
import math
import urllib.request

# Cap a single GET body so a misbehaving backend can't stress memory/latency. A
# vLLM /metrics scrape is well under this; /health is tiny.
_MAX_BODY_BYTES = 5 * 1024 * 1024

# The handful of vLLM series the live view reports. "busy" = running/waiting now;
# "usage" = cumulative tokens + finished requests by reason. Summed across the
# engine/model labels vLLM attaches (a single backend may expose >1 engine).
_KV = "vllm:gpu_cache_usage_perc"
_SUCCESS = "vllm:request_success_total"
# Series that are simply summed → the live-view field they accumulate into.
_SUM_FIELDS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
}


def _label(label_block: str, key: str) -> str | None:
    """Extract ``key="value"`` from a Prometheus ``{...}`` label block (best-effort)."""
    needle = f'{key}="'
    start = label_block.find(needle)
    if start < 0:
        return None
    start += len(needle)
    end = label_block.find('"', start)
    return label_block[start:end] if end > start else None


def _iter_samples(text: str):
    """Yield ``(name, labels, value)`` for each finite metric sample line.

    Skips comments, blanks, malformed lines, and non-finite values (NaN/inf would
    later make ``int()`` raise — the parser is best-effort).
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, value = line.rsplit(" ", 1)
            val = float(value)
        except ValueError:
            continue
        if not math.isfinite(val):
            continue
        brace = left.find("{")
        name = left[:brace] if brace >= 0 else left
        labels = left[brace:] if brace >= 0 else ""
        yield name, labels, val


def parse_metrics(text: str) -> dict:
    """Reduce a vLLM ``/metrics`` exposition to the live-view numbers.

    Returns ints for counts/tokens and a ``by_finish_reason`` map; ``kv_cache_usage``
    (0..1) is included only when the gauge is present. Unknown/malformed lines are
    skipped, so a partial scrape still yields what it can.
    """
    sums = dict.fromkeys(_SUM_FIELDS.values(), 0.0)
    kv: float | None = None
    by_reason: dict[str, float] = {}
    for name, labels, val in _iter_samples(text):
        field = _SUM_FIELDS.get(name)
        if field is not None:
            sums[field] += val
        elif name == _KV:
            kv = val if kv is None else max(kv, val)
        elif name == _SUCCESS:
            reason = _label(labels, "finished_reason") or "?"
            by_reason[reason] = by_reason.get(reason, 0.0) + val
    out = {
        "running": int(sums["running"]),
        "waiting": int(sums["waiting"]),
        "prompt_tokens": int(sums["prompt_tokens"]),
        "generation_tokens": int(sums["generation_tokens"]),
        "requests_succeeded": int(sum(by_reason.values())),
        "by_finish_reason": {k: int(v) for k, v in by_reason.items() if v},
    }
    if kv is not None:
        out["kv_cache_usage"] = round(kv, 3)
    return out


def http_get_text(
    url: str, *, timeout: float = 3.0, max_bytes: int = _MAX_BODY_BYTES
) -> str | None:
    """Best-effort GET → body text, or ``None`` if unreachable / non-2xx / oversized.

    Reads at most ``max_bytes`` (+1 to detect overflow): an over-cap body is treated
    as unavailable rather than buffered whole, so a misbehaving backend can't stress
    memory. Never raises.
    """
    try:
        with urllib.request.urlopen(
            url, timeout=timeout
        ) as r:  # nosec B310 - http(s) only, fixed scheme
            if not (200 <= r.status < 300):
                return None
            data = r.read(max_bytes + 1)
            if len(data) > max_bytes:
                return None  # oversized → best-effort fail rather than buffer it whole
            return data.decode("utf-8", errors="replace")
    except (OSError, ValueError):  # URLError is an OSError subclass — covered
        return None


def http_get_json(url: str, *, timeout: float = 3.0) -> dict | None:
    """Best-effort GET → parsed JSON dict, or ``None`` (unreachable / non-dict). Never raises."""
    text = http_get_text(url, timeout=timeout)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def health_ok(base_url: str, *, timeout: float = 3.0) -> bool:
    """True when ``<base_url>/health`` returns 2xx."""
    return http_get_text(base_url.rstrip("/") + "/health", timeout=timeout) is not None


def probe_backend(base_url: str, *, timeout: float = 3.0) -> dict:
    """Live ``{health, metrics}`` for one backend base URL (best-effort, never raises).

    ``health`` is ``"ok"`` / ``"unreachable"``; ``metrics`` is the parsed dict, or
    ``None`` when ``/metrics`` is unreachable (an engine can be loading or down).
    """
    base = base_url.rstrip("/")
    if not health_ok(base, timeout=timeout):
        # Short-circuit: a down backend has no useful /metrics, so skip the second
        # request (halves the timeout cost for a dead backend).
        return {"health": "unreachable", "metrics": None}
    raw = http_get_text(base + "/metrics", timeout=timeout)
    return {"health": "ok", "metrics": parse_metrics(raw) if raw is not None else None}
