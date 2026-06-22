"""Build the ``lobes overview --live`` sections from a running deployment.

Read-only and HTTP-only (no docker), so it works against a local deployment or a
remote tunnel alike. It probes the gateway ``/status`` (fleet) or a single vLLM
``/metrics`` + ``/health`` via :mod:`lobes._metrics` (best-effort — never
raises). The section builders are pure (they take the already-probed payloads) so
they unit-test without sockets; :func:`live_sections` is the thin probing wrapper.

Sections answer the five "what is the fleet doing right now" questions: ONLINE
(health), OFFERED (models + task families), BUSY (in-flight/queued), USAGE
(tokens + finished requests), and ENDPOINTS.
"""

from __future__ import annotations

from lobes import _metrics

_FLEET_STATUS_OBJECT = "lobes.fleet_status"


def _agg_usage(backends: list[dict]) -> tuple[int, int, int, dict[str, int]]:
    """Sum tokens + finished requests across backends (the fleet's cumulative usage)."""
    prompt = gen = succeeded = 0
    reasons: dict[str, int] = {}
    for b in backends:
        m = b.get("metrics") or {}
        prompt += int(m.get("prompt_tokens", 0) or 0)
        gen += int(m.get("generation_tokens", 0) or 0)
        succeeded += int(m.get("requests_succeeded", 0) or 0)
        for reason, count in (m.get("by_finish_reason") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    return prompt, gen, succeeded, reasons


def _usage_items(prompt: int, gen: int, succeeded: int, reasons: dict[str, int]) -> list[str]:
    line = f"requests succeeded: {succeeded}"
    if reasons:
        line += "  (" + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) + ")"
    return [f"prompt tokens: {prompt:,}    generation tokens: {gen:,}", line]


def _backend_line(b: dict) -> str:
    m = b.get("metrics") or {}
    parts = [f"{b.get('name', '?')} ({b.get('task', '?')}): {b.get('health', '?')}"]
    if m:
        parts.append(f"run {int(m.get('running', 0))} wait {int(m.get('waiting', 0))}")
    if b.get("served_name"):
        parts.append(str(b["served_name"]))
    return " · ".join(parts)


def fleet_sections(status: dict) -> list[dict]:
    """Live sections from a gateway ``/status`` payload (the fleet case)."""
    backends = status.get("backends") or []
    busy = status.get("busy") or {}
    tasks = sorted({b.get("task") for b in backends if b.get("task")})
    models = [b.get("served_name") for b in backends if b.get("served_name")]
    prompt, gen, succeeded, reasons = _agg_usage(backends)
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
            ],
        },
        {"title": "Usage", "items": _usage_items(prompt, gen, succeeded, reasons)},
        {"title": "Endpoints", "items": list(status.get("endpoints") or []) or ["(none)"]},
    ]


def single_sections(
    port: int, served_name: str | None, *, healthy: bool, metrics: dict | None
) -> list[dict]:
    """Live sections for a bare single-model vLLM server (``/metrics`` + ``/health``)."""
    served = served_name or "(model unknown — no .env)"
    online = f"{served} on :{port} — " + ("ok" if healthy else "not responding")
    if metrics:
        busy = [f"running: {metrics['running']}    waiting: {metrics['waiting']}"]
        usage = _usage_items(
            metrics["prompt_tokens"],
            metrics["generation_tokens"],
            metrics["requests_succeeded"],
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
