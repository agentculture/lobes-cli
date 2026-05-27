"""API-side assessment and benchmark of a vLLM-served model (stdlib only).

Talks only to the OpenAI-compatible endpoint (``urllib``, no third-party deps).
Ported from the original ``_assess.py`` and split into two concerns:

* :func:`run_correctness` — fixed correctness probes + reasoning-trace detection
  (drives ``model assess``);
* :func:`run_benchmark` — decode throughput + prefill latency (drives
  ``model benchmark``).

Host-side facts (image tag, GPU memory) are gathered by the command handlers via
:mod:`model_gear.runtime._compose` and printed alongside this output.
"""

from __future__ import annotations

import contextlib
import json
import time
import urllib.error
import urllib.request

from model_gear.cli._errors import EXIT_ENV_ERROR, ModelGearError

# urllib.error.URLError is a subclass of OSError, so `except OSError` covers
# connection failures, timeouts, and HTTPError without listing it redundantly.


@contextlib.contextmanager
def _api_errors(what: str):
    """Turn raw HTTP / JSON / response-shape failures into a structured error.

    Without this, an ``HTTPError``/``URLError`` or an unexpected payload
    (``KeyError``/``JSONDecodeError``) bubbles to the dispatcher's catch-all and
    appears as ``unexpected: ...`` with no remediation.
    """
    try:
        yield
    except ModelGearError:
        raise
    except OSError as exc:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"{what} failed: {exc}",
            remediation="check 'model status' / 'docker logs model-gear-vllm'",
        ) from exc
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"{what}: unexpected response shape ({exc.__class__.__name__}: {exc})",
            remediation="the served model returned an unexpected payload; check the vLLM logs",
        ) from exc


# (prompt, expected-substring, table-label) — the two fixed correctness probes.
_PROBES = [
    ("What is 17 * 23?", "391", "`17 * 23 = 391`"),
    (
        "If a train leaves at 14:45 and arrives at 17:10, how long is the journey in minutes?",
        "145",
        "train 14:45→17:10 = 145 min",
    ),
]

# Tool-calling probe (opt-in via ``model assess --tools``): mirrors issue #9's
# acceptance check — a ``tool_choice:"auto"`` request must return a ``tool_calls``
# array naming the ``finish`` function. Requires the server's
# ``--enable-auto-tool-choice`` + ``--tool-call-parser`` flags.
_TOOL_PROBE_PROMPT = "Call the finish tool with summary hello."
_TOOL_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the task with a short summary.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    }
]


def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # local endpoint only
        return json.load(r)


def _get(url: str, path: str, timeout: int = 10):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:  # local endpoint only
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status, json.load(r)
        return r.status, r.read().decode()


def _trace_field(msg: dict) -> tuple[str | None, int]:
    """Return ``(field_name, length)`` of the reasoning trace, whichever key holds it.

    vLLM builds vary: the ``<think>`` trace lands in ``reasoning`` on the nv26.04
    image, ``reasoning_content`` on older builds.
    """
    for key in ("reasoning", "reasoning_content"):
        val = msg.get(key)
        if isinstance(val, str) and val:
            return key, len(val)
    return None, 0


def health_status(url: str) -> int:
    """Return the ``/health`` status code, or raise if the endpoint is unreachable."""
    try:
        status, _ = _get(url, "/health")
    except OSError as exc:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"/health unreachable at {url} ({exc})",
            remediation="start the server with 'model serve --apply'",
        ) from exc
    return status


def served_model(url: str, override: str | None = None) -> tuple[str, object]:
    """Return ``(model_id, max_model_len)`` from ``/v1/models``. Raises if none served."""
    with _api_errors("/v1/models"):
        _, models = _get(url, "/v1/models")
        data = models.get("data") if isinstance(models, dict) else None
        if not data:
            raise ModelGearError(
                code=EXIT_ENV_ERROR,
                message=f"/v1/models returned no models at {url}",
                remediation="check 'model status' / 'docker logs model-gear-vllm'",
            )
        first = data[0]
        return (override or first["id"]), first.get("max_model_len")


def _probe(url: str, model: str, prompt: str, expect: str) -> dict:
    d = _post(
        url,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.3,
        },
    )
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    field, tlen = _trace_field(msg)
    return {
        "ok": expect in content,
        "expect": expect,
        "trace_field": field,
        "trace_len": tlen,
        "finish": d["choices"][0].get("finish_reason"),
        "completion_tokens": d.get("usage", {}).get("completion_tokens"),
    }


def _tool_probe(url: str, model: str) -> dict:
    """Probe OpenAI tool calling; degrade gracefully if the server lacks the flags.

    A server without ``--enable-auto-tool-choice`` rejects ``tool_choice:"auto"``
    with HTTP 400. We surface that as ``ok=False`` with the server's message
    rather than aborting the whole assess run.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _TOOL_PROBE_PROMPT}],
        "tools": _TOOL_PROBE_TOOLS,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0,
    }
    try:
        d = _post(url, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace").strip()
        return {
            "ok": False,
            "tool_calls": [],
            "finish": None,
            "error": f"HTTP {exc.code}: {body[:200]}",
        }
    msg = d["choices"][0]["message"]
    names = [c["function"]["name"] for c in (msg.get("tool_calls") or []) if c.get("function")]
    return {
        "ok": "finish" in names,
        "tool_calls": names,
        "finish": d["choices"][0].get("finish_reason"),
        "error": None,
    }


def _decode_throughput(url: str, model: str, n_tokens: int, runs: int = 2) -> list[float]:
    rates = []
    for _ in range(runs):
        t0 = time.monotonic()
        d = _post(
            url,
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Write a detailed essay about distributed systems."}
                ],
                "max_tokens": n_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        dt = time.monotonic() - t0
        ct = d["usage"]["completion_tokens"]
        rates.append(round(ct / dt, 1))
    return rates


def _prefill(url: str, model: str) -> dict:
    prompt = "Summarize this. " + "The system processes events. " * 400
    t0 = time.monotonic()
    d = _post(
        url,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    dt = time.monotonic() - t0
    return {"prompt_tokens": d["usage"]["prompt_tokens"], "seconds": round(dt, 2)}


def run_correctness(url: str, model: str | None = None, check_tools: bool = False) -> dict:
    """Run the fixed correctness probes; return a structured result.

    When ``check_tools`` is set, also probe OpenAI tool calling and report it
    under ``tool_calling`` (``None`` otherwise). ``passed`` reflects the content
    probes only — a tool-less server still passes correctness.
    """
    url = url.rstrip("/")
    hstatus = health_status(url)
    model, max_len = served_model(url, model)
    probes = []
    tool_calling = None
    with _api_errors("correctness probe"):
        for prompt, expect, label in _PROBES:
            result = _probe(url, model, prompt, expect)
            result["label"] = label
            probes.append(result)
        if check_tools:
            tool_calling = _tool_probe(url, model)
    trace_field = next((p["trace_field"] for p in probes if p["trace_field"]), None)
    trace_len = max((p["trace_len"] for p in probes), default=0)
    return {
        "model": model,
        "endpoint": url,
        "health": hstatus,
        "max_model_len": max_len,
        "probes": probes,
        "trace_field": trace_field or "(none)",
        "trace_len": trace_len,
        "passed": all(p["ok"] for p in probes),
        "tool_calling": tool_calling,
    }


def run_benchmark(
    url: str, model: str | None = None, decode_tokens: int = 512, runs: int = 2
) -> dict:
    """Measure decode throughput + prefill latency; return a structured result."""
    url = url.rstrip("/")
    health_status(url)
    model, max_len = served_model(url, model)
    with _api_errors("benchmark"):
        rates = _decode_throughput(url, model, decode_tokens, runs)
        pf = _prefill(url, model)
    return {
        "model": model,
        "endpoint": url,
        "max_model_len": max_len,
        "decode_tokens": decode_tokens,
        "decode_rates": rates,
        "prefill": pf,
    }


def render_correctness(result: dict) -> str:
    """Render :func:`run_correctness` output as a markdown block for a per-model doc."""
    lines = [
        f"## Assessment — `{result['model']}`",
        "",
        f"- Endpoint: `{result['endpoint']}` · `/health` {result['health']} · "
        f"`max_model_len` {result['max_model_len']}",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for p in result["probes"]:
        mark = "PASS" if p["ok"] else "FAIL"
        lines.append(
            f"| {p['label']} | {mark} (finish={p['finish']}, {p['completion_tokens']} tok) |"
        )
    lines.append(
        f"| reasoning trace field | `{result['trace_field']}` (len {result['trace_len']}) |"
    )
    tc = result.get("tool_calling")
    if tc is not None:
        if tc["ok"]:
            detail = f"PASS — called {', '.join(tc['tool_calls'])}"
        else:
            detail = "FAIL — " + (
                tc.get("error") or f"no finish call (tool_calls={tc['tool_calls']})"
            )
        lines.append(f"| tool calling (`tool_choice:auto`) | {detail} |")
    return "\n".join(lines)


def render_benchmark(result: dict) -> str:
    """Render :func:`run_benchmark` output as a markdown block for a per-model doc."""
    rates = "/".join(str(r) for r in result["decode_rates"])
    pf = result["prefill"]
    return "\n".join(
        [
            f"## Benchmark — `{result['model']}`",
            "",
            f"- Endpoint: `{result['endpoint']}` · `max_model_len` {result['max_model_len']}",
            "",
            "| Metric | Result |",
            "|---|---|",
            f"| **decode throughput** | **{rates} tok/s** (batch=1, greedy, "
            f"{result['decode_tokens']} tok forced) |",
            f"| prefill | {pf['prompt_tokens']} prompt tokens + 16 gen in {pf['seconds']} s |",
        ]
    )
