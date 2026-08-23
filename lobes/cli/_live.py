"""Build the ``lobes overview --live`` sections from a running deployment.

Read-only and HTTP-only (no docker), so it works against a local deployment or a
remote tunnel alike. It probes the gateway ``/status`` (fleet) or a single
backend's ``/metrics`` + ``/health`` via :mod:`lobes._metrics` (best-effort —
never raises). A backend whose engine cannot report a field renders as
``unknown``, and any total it makes incomplete is flagged partial — never a
fabricated ``0``. The section builders are pure (they take the already-probed
payloads) so they unit-test without sockets; :func:`live_sections` is the thin
probing wrapper.

Sections answer the five "what is the fleet doing right now" questions: ONLINE
(health), OFFERED (models + task families), BUSY (in-flight/queued), USAGE
(tokens + finished requests), and ENDPOINTS.
"""

from __future__ import annotations

from lobes import _metrics

_FLEET_STATUS_OBJECT = "lobes.fleet_status"


# A field a backend genuinely cannot report renders as this word — never as a
# number. See :mod:`lobes._metrics`: a non-vLLM engine (llama.cpp) names such
# fields in ``unsupported`` and omits their keys, so the live view can tell
# "0 because idle" from "unknown because this engine does not export it".
_UNKNOWN = "unknown"

# The usage fields the fleet totals sum; a backend that cannot report one makes
# the corresponding total incomplete, which the Usage section then says out loud.
_USAGE_FIELDS = ("prompt_tokens", "generation_tokens", "requests_succeeded", "by_finish_reason")


def _fmt(m: dict, field: str, *, thousands: bool = False) -> str:
    """One live-view number, or ``unknown`` when this backend cannot report it."""
    if field not in m or field in _metrics.unsupported_fields(m):
        return _UNKNOWN
    value = int(m[field] or 0)
    return f"{value:,}" if thousands else str(value)


def _agg_usage(backends: list[dict]) -> tuple[int, int, int, dict[str, int], set[str]]:
    """Sum tokens + finished requests across backends (the fleet's cumulative usage).

    The fifth element names the usage fields at least one backend could not
    report — the totals are then honest-but-partial rather than silently short.
    """
    prompt = gen = succeeded = 0
    reasons: dict[str, int] = {}
    incomplete: set[str] = set()
    for b in backends:
        m = b.get("metrics") or {}
        unknown = _metrics.unsupported_fields(m)
        incomplete |= {f for f in _USAGE_FIELDS if f in unknown}
        prompt += int(m.get("prompt_tokens", 0) or 0)
        gen += int(m.get("generation_tokens", 0) or 0)
        succeeded += int(m.get("requests_succeeded", 0) or 0)
        for reason, count in (m.get("by_finish_reason") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    return prompt, gen, succeeded, reasons, incomplete


def _usage_items(
    prompt: str,
    gen: str,
    succeeded: str,
    reasons: dict[str, int],
    incomplete: set[str] = frozenset(),
) -> list[str]:
    line = f"requests succeeded: {succeeded}"
    if reasons:
        line += "  (" + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) + ")"
    items = [f"prompt tokens: {prompt}    generation tokens: {gen}", line]
    if incomplete:
        items.append(
            "(!) totals are partial — not reported by every backend: "
            + ", ".join(sorted(incomplete))
        )
    return items


def _backend_line(b: dict) -> str:
    m = b.get("metrics") or {}
    parts = [f"{b.get('name', '?')} ({b.get('task', '?')}): {b.get('health', '?')}"]
    if m:
        parts.append(f"run {_fmt(m, 'running')} wait {_fmt(m, 'waiting')}")
    if b.get("served_name"):
        parts.append(str(b["served_name"]))
    return " · ".join(parts)


def fleet_sections(status: dict) -> list[dict]:
    """Live sections from a gateway ``/status`` payload (the fleet case)."""
    backends = status.get("backends") or []
    busy = status.get("busy") or {}
    tasks = sorted({b.get("task") for b in backends if b.get("task")})
    models = [b.get("served_name") for b in backends if b.get("served_name")]
    prompt, gen, succeeded, reasons, incomplete = _agg_usage(backends)
    return [
        {
            "title": "Online (live)",
            "items": [_backend_line(b) for b in backends] or ["(no backends)"],
        },
        {
            "title": "Offered",
            "items": [
                f"default model: {status.get('default_model', '?')}",
                f"task families: {', '.join(tasks) or '?'}",
                f"models: {', '.join(models) or '?'}",
                "full catalog: lobes overview --list",
            ],
        },
        {
            "title": "Busy",
            "items": [
                f"running: {int(busy.get('running', 0))}    "
                f"waiting: {int(busy.get('waiting', 0))}"
                + (
                    "    (partial — a backend does not report in-flight counts)"
                    if busy.get("partial")
                    else ""
                )
            ],
        },
        {
            "title": "Usage",
            "items": _usage_items(f"{prompt:,}", f"{gen:,}", str(succeeded), reasons, incomplete),
        },
        {"title": "Endpoints", "items": list(status.get("endpoints") or []) or ["(none)"]},
    ]


def single_sections(
    port: int, served_name: str | None, *, healthy: bool, metrics: dict | None
) -> list[dict]:
    """Live sections for a bare single-model vLLM server (``/metrics`` + ``/health``)."""
    served = served_name or "(model unknown — no .env)"
    online = f"{served} on :{port} — " + ("ok" if healthy else "not responding")
    if metrics:
        busy = [f"running: {_fmt(metrics, 'running')}    waiting: {_fmt(metrics, 'waiting')}"]
        usage = _usage_items(
            _fmt(metrics, "prompt_tokens", thousands=True),
            _fmt(metrics, "generation_tokens", thousands=True),
            _fmt(metrics, "requests_succeeded"),
            metrics.get("by_finish_reason") or {},
        )
    else:
        busy = ["(metrics unavailable)"]
        usage = ["(metrics unavailable)"]
    return [
        {"title": "Online (live)", "items": [online]},
        {
            "title": "Offered",
            "items": [f"served model: {served}", "full catalog: lobes overview --list"],
        },
        {"title": "Busy", "items": busy},
        {"title": "Usage", "items": usage},
        {
            "title": "Endpoints",
            "items": [
                "GET /health",
                "GET /metrics",
                "GET /v1/models",
                "POST /v1/chat/completions",
                "POST /v1/completions",
            ],
        },
    ]


def live_sections(port: int, served_name: str | None) -> list[dict]:
    """Probe :``port`` and return the live sections (fleet gateway or single model).

    The gateway exposes ``/status`` (a lobes fan-out); a bare vLLM does not, so
    its absence + a healthy ``/health`` means single-model. Everything is
    best-effort: an unreachable endpoint yields a single "nothing serving" section
    rather than an error.
    """
    base = f"http://localhost:{port}"
    status = _metrics.http_get_json(base + "/status")
    if isinstance(status, dict) and status.get("object") == _FLEET_STATUS_OBJECT:
        return fleet_sections(status)
    if _metrics.health_ok(base):
        raw = _metrics.http_get_text(base + "/metrics")
        return single_sections(
            port, served_name, healthy=True, metrics=_metrics.parse_metrics(raw) if raw else None
        )
    return [
        {
            "title": "Live",
            "items": [
                f"no lobes endpoint reachable on :{port}",
                ">> start one: lobes serve --apply  (or lobes fleet up --apply)",
            ],
        }
    ]
