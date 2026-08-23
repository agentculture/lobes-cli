"""Parse a backend's Prometheus ``/metrics`` + probe its live state (stdlib only).

Shared by the gateway's ``/status`` fan-out and ``lobes overview --live``. The
parser is pure; the probes are best-effort and **never raise** — an unreachable
backend folds into a structured result so the live view degrades gracefully
instead of erroring. vLLM serves ``/metrics`` and ``/health`` unauthenticated, so
no API key is needed for either.

**Engines other than vLLM (issue: the llama.cpp lane).** Prometheus series are
namespaced per engine: vLLM emits ``vllm:*``, llama.cpp's server emits
``llamacpp:*``. A parser keyed on ``vllm:*`` alone therefore reads a *busy*
llama.cpp backend as all-zeros — and the live view would present those zeros as
real numbers. So :func:`parse_metrics` sniffs the engine from the series names
present and parses llama.cpp's own series where they map cleanly, while naming
the fields that engine genuinely cannot report in an ``unsupported`` list and
**omitting** their keys entirely. The invariant callers can rely on: a numeric
field that is present is measured, and ``0`` always means idle — never
"unknown". Anything else is absent and named in ``unsupported``.
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

# Engine sniffing. ``vllm`` is the historical default and its output shape is
# frozen (no ``engine``/``unsupported`` keys) so every existing consumer and
# golden stays byte-identical. Only a *recognised non-vLLM* engine, or an
# unrecognised one, grows the honesty fields.
ENGINE_VLLM = "vllm"
ENGINE_LLAMACPP = "llamacpp"
ENGINE_UNKNOWN = "unknown"
_PREFIXES = ((ENGINE_VLLM, "vllm:"), (ENGINE_LLAMACPP, "llamacpp:"))

# llama.cpp's ``llama-server --metrics`` surface. Only series verified against
# that server's own exposition are mapped; nothing is guessed. Deliberately
# UNMAPPED (see _LLAMACPP_ALWAYS_UNSUPPORTED): llama.cpp has no per-finish-reason
# success counter — no ``request_success_total`` equivalent exists — so
# ``requests_succeeded``/``by_finish_reason`` are reported unknown, not zero.
_LLAMACPP_KV = "llamacpp:kv_cache_usage_ratio"
_LLAMACPP_SUM_FIELDS = {
    "llamacpp:requests_processing": "running",
    "llamacpp:requests_deferred": "waiting",
    "llamacpp:prompt_tokens_total": "prompt_tokens",
    "llamacpp:tokens_predicted_total": "generation_tokens",
}
_LLAMACPP_ALWAYS_UNSUPPORTED = ("requests_succeeded", "by_finish_reason")

# Every live-view field, in the order ``unsupported`` lists them (deterministic
# output, so /status payloads diff cleanly).
_ALL_FIELDS = (
    "running",
    "waiting",
    "prompt_tokens",
    "generation_tokens",
    "requests_succeeded",
    "by_finish_reason",
    "kv_cache_usage",
)


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


def _detect_engine(samples: list) -> str | None:
    """Sniff the engine from the series namespaces present, or ``None`` if nothing matches.

    vLLM wins a (pathological) tie: it is the incumbent, and its parse is the one
    every existing deployment depends on.
    """
    for engine, prefix in _PREFIXES:
        if any(name.startswith(prefix) for name, _labels, _val in samples):
            return engine
    return None


def _parse_vllm(samples: list) -> dict:
    """The vLLM reduction — frozen shape, no ``engine``/``unsupported`` keys."""
    sums = dict.fromkeys(_SUM_FIELDS.values(), 0.0)
    kv: float | None = None
    by_reason: dict[str, float] = {}
    for name, labels, val in samples:
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


def _parse_llamacpp(samples: list) -> dict:
    """Reduce a llama.cpp ``llamacpp:*`` exposition, naming what it cannot report.

    A field is emitted **only if its series actually appeared** in the scrape, so a
    build that does not export one reads "unknown" rather than a fabricated ``0``
    — the whole point of the exercise is telling "0 because idle" apart from
    "unknown because unsupported".
    """
    sums: dict[str, float] = {}
    kv: float | None = None
    for name, _labels, val in samples:
        field = _LLAMACPP_SUM_FIELDS.get(name)
        if field is not None:
            sums[field] = sums.get(field, 0.0) + val
        elif name == _LLAMACPP_KV:
            kv = val if kv is None else max(kv, val)
    out: dict = {"engine": ENGINE_LLAMACPP}
    for field in _ALL_FIELDS:
        if field in sums:
            out[field] = int(sums[field])
    if kv is not None:
        out["kv_cache_usage"] = round(kv, 3)
    out["unsupported"] = [f for f in _ALL_FIELDS if f not in out]
    return out


def parse_metrics(text: str) -> dict:
    """Reduce a backend ``/metrics`` exposition to the live-view numbers.

    Returns ints for counts/tokens and a ``by_finish_reason`` map; ``kv_cache_usage``
    (0..1) is included only when the gauge is present. Unknown/malformed lines are
    skipped, so a partial scrape still yields what it can.

    A **vLLM** scrape (and an empty one) yields exactly the historical dict. A
    **llama.cpp** scrape yields the same fields it can genuinely report plus
    ``engine``/``unsupported``. A scrape carrying series from neither engine
    yields ``engine: "unknown"`` and **no numbers at all** — refusing to answer
    beats answering zero.
    """
    samples = list(_iter_samples(text))
    engine = _detect_engine(samples)
    if engine == ENGINE_LLAMACPP:
        return _parse_llamacpp(samples)
    if engine is None and samples:
        # Some third engine we have never parsed: every field is unknown. (An
        # EMPTY scrape stays on the vLLM path — that is the historical
        # "nothing to report yet" shape, not a foreign engine.)
        return {"engine": ENGINE_UNKNOWN, "unsupported": list(_ALL_FIELDS)}
    return _parse_vllm(samples)


def unsupported_fields(metrics: dict | None) -> frozenset[str]:
    """The live-view fields *this* backend cannot report (empty for vLLM)."""
    return frozenset((metrics or {}).get("unsupported") or ())


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
