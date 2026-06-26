"""API-side assessment and benchmark of a vLLM-served model (stdlib only).

Talks only to the OpenAI-compatible endpoint (``urllib``, no third-party deps).
Ported from the original ``_assess.py`` and split into two concerns:

* :func:`run_correctness` — fixed correctness probes + reasoning-trace detection
  (drives ``lobes assess``);
* :func:`run_benchmark` — decode throughput + prefill latency (drives
  ``lobes benchmark``).

Host-side facts (image tag, GPU memory) are gathered by the command handlers via
:mod:`lobes.runtime._compose` and printed alongside this output.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import statistics
import time
import urllib.error
import urllib.request

from lobes.cli._errors import EXIT_ENV_ERROR, ModelGearError

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
            remediation="check 'lobes status' / 'docker logs model-gear-vllm'",
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

# Tool-calling probe (opt-in via ``lobes assess --tools``): mirrors issue #9's
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
            remediation="start the server with 'lobes serve --apply'",
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
                remediation="check 'lobes status' / 'docker logs model-gear-vllm'",
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
    """Probe OpenAI tool calling; degrade gracefully, never abort the assess run.

    A server without ``--enable-auto-tool-choice`` rejects ``tool_choice:"auto"``
    with HTTP 400. A server that *has* the flags but returns an unexpected payload
    (no ``choices``/``message``, or a wrong-shaped ``tool_calls``) would otherwise
    raise inside :func:`run_correctness`'s ``_api_errors`` block and abort. Both
    cases are surfaced here as a structured ``ok=False`` result with a FAIL row.
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
    # Defensive parsing: a malformed 200 must not abort the run (documented
    # "FAIL row, no abort"). Use .get()/isinstance throughout, with a catch-all
    # net for any remaining shape surprise.
    try:
        choices = d.get("choices") if isinstance(d, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        msg = choice.get("message") or {}
        raw_calls = msg.get("tool_calls")
        calls = raw_calls if isinstance(raw_calls, list) else []
        names = []
        for c in calls:
            fn = c.get("function") if isinstance(c, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.append(name)
        return {
            "ok": "finish" in names,
            "tool_calls": names,
            "finish": choice.get("finish_reason"),
            "error": None,
        }
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        return {
            "ok": False,
            "tool_calls": [],
            "finish": None,
            "error": f"unexpected response shape ({exc.__class__.__name__}: {exc})",
        }


def probe_tool_calls(url: str, model: str) -> dict:
    """One-shot tool-calling probe, without the arithmetic correctness probes.

    Used by ``lobes switch`` / ``lobes serve`` to verify, the moment the
    container is healthy, that ``tool_choice:"auto"`` returns a ``tool_calls``
    response (no HTTP 400, a ``finish`` call present). Returns the same
    structured dict as the in-``assess`` probe (``ok``/``tool_calls``/``finish``/
    ``error``).

    Never raises. ``_tool_probe`` already folds HTTP 400 and malformed-200
    payloads into ``ok=False``; the two failure modes it lets through —
    a connection failure (``OSError``) or an undecodable body
    (``JSONDecodeError``) from ``_post``/``json.load`` — are caught here and
    likewise returned as a structured ``ok=False``, so a post-switch/post-serve
    probe can never abort the command.
    """
    try:
        return _tool_probe(url.rstrip("/"), model)
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "tool_calls": [], "finish": None, "error": f"probe failed: {exc}"}


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


def _prefill(url: str, model: str, input_len: int = 2000) -> dict:
    # ~6 tokens per "The system processes events. " phrase — scale the repeat
    # count so the prompt approximates the requested input_len (the actual
    # prompt_tokens is measured and reported, so the estimate need only be close).
    reps = max(1, input_len // 6)
    prompt = "Summarize this. " + "The system processes events. " * reps
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
    url: str,
    model: str | None = None,
    *,
    purpose: str = "balanced",
    input_len: int = 1000,
    output_len: int = 1000,
    runs: int = 2,
) -> dict:
    """Measure decode throughput + prefill latency for a workload shape.

    The shape (``input_len`` prompt, ``output_len`` decode) is the workload
    *purpose* — ``lobes benchmark`` derives it from the configured ``VLLM_PURPOSE``
    so the numbers track the serve config (see :mod:`lobes.profiles`).
    """
    url = url.rstrip("/")
    health_status(url)
    model, max_len = served_model(url, model)
    with _api_errors("benchmark"):
        rates = _decode_throughput(url, model, output_len, runs)
        pf = _prefill(url, model, input_len)
    return {
        "model": model,
        "endpoint": url,
        "max_model_len": max_len,
        "purpose": purpose,
        "input_len": input_len,
        "output_len": output_len,
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


# ---------------------------------------------------------------------------
# Per-lobe perf engine — t3: ttft · concurrent driver · auto-ramp knee
# ---------------------------------------------------------------------------


def measure_prefill_ttft(
    url: str,
    model: str,
    *,
    input_len: int = 2000,
    timeout: int = 300,
) -> dict:
    """Measure time-to-first-token (TTFT) by timing a ``max_tokens=1`` request.

    Sends a prompt of approximately *input_len* tokens with ``max_tokens=1`` and
    ``temperature=0``, timing the full round trip.  With only one decode step the
    elapsed time is dominated by the prefill phase, so it approximates TTFT.

    Returns:
        ``{"prompt_tokens": int, "ttft_ms": float}``
    """
    reps = max(1, input_len // 6)
    prompt = "Summarize this. " + "The system processes events. " * reps
    t0 = time.monotonic()
    d = _post(
        url.rstrip("/"),
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        },
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    return {
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "ttft_ms": round(elapsed * 1000, 1),
    }


def _pct(sorted_vals: list[float], p: int) -> float:
    """Return the *p*-th percentile of a pre-sorted list (0 ≤ p ≤ 100).

    Uses the nearest-rank method on the sorted input.  Since the input is sorted
    and we only ever call this with p50 < p95, the invariant ``p95 >= p50`` holds
    by construction.
    """
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = int((n - 1) * p / 100)
    return sorted_vals[idx]


def run_concurrent(
    url: str,
    model: str,
    *,
    concurrency: int,
    max_tokens: int = 128,
    timeout: int = 300,
) -> dict:
    """Fire *concurrency* chat requests concurrently; return throughput + latency stats.

    Uses :class:`concurrent.futures.ThreadPoolExecutor` with *max_workers=concurrency*
    so all requests are in-flight simultaneously.  Wall time is measured around the
    whole batch (``time.monotonic()``).

    Returns:
        ``{"concurrency": int, "requests_per_s": float, "p50_latency_ms": float,
        "p95_latency_ms": float, "ms_per_token": float, "total_s": float}``

    * ``requests_per_s`` = concurrency / total_s (batch throughput).
    * ``p50``/``p95`` are per-request round-trip latencies.
    * ``ms_per_token`` = mean of (latency_ms / completion_tokens) across requests.
    """
    url = url.rstrip("/")

    def _one_request() -> dict:
        t0 = time.monotonic()
        d = _post(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Write a short paragraph."}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
        )
        dt = time.monotonic() - t0
        ct = d["usage"]["completion_tokens"]
        return {"latency_ms": dt * 1000.0, "completion_tokens": ct}

    t_batch = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_request) for _ in range(concurrency)]
        results = [f.result() for f in futures]
    total_s = time.monotonic() - t_batch

    latencies = sorted(r["latency_ms"] for r in results)
    p50 = _pct(latencies, 50)
    p95 = _pct(latencies, 95)

    ms_per_token_vals = [
        r["latency_ms"] / r["completion_tokens"] for r in results if r["completion_tokens"] > 0
    ]
    ms_per_token = statistics.mean(ms_per_token_vals) if ms_per_token_vals else 0.0

    return {
        "concurrency": concurrency,
        "requests_per_s": round(concurrency / total_s, 3),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "ms_per_token": round(ms_per_token, 3),
        "total_s": round(total_s, 3),
    }


def _find_knee(rows: list[dict], *, threshold: float = 0.1) -> dict:
    """Pure throughput-plateau detector; no network calls.

    Walk *rows* (each a dict with ``"concurrency"`` and ``"requests_per_s"``).
    Stop when the relative gain between consecutive steps falls below *threshold*::

        gain = (rps[i] - rps[i-1]) / rps[i-1]

    The *knee* is the concurrency of the last step **before** the gain dropped
    (the peak-throughput concurrency).  ``rows`` in the result contains all rows
    up to (but not including) the declining step.

    Args:
        rows: List of per-step measurement dicts, ordered by ascending concurrency.
        threshold: Minimum relative gain to keep ramping (default 0.1 = 10 %).

    Returns:
        ``{"knee": int, "rows": list[dict]}``
    """
    if not rows:
        return {"knee": 0, "rows": []}
    if len(rows) == 1:
        return {"knee": rows[0]["concurrency"], "rows": list(rows)}

    for i in range(1, len(rows)):
        prev_rps = rows[i - 1]["requests_per_s"]
        curr_rps = rows[i]["requests_per_s"]
        if prev_rps == 0:
            continue  # guard against degenerate data
        gain = (curr_rps - prev_rps) / prev_rps
        if gain < threshold:
            return {"knee": rows[i - 1]["concurrency"], "rows": rows[:i]}

    # All steps gained enough — last step is the peak.
    return {"knee": rows[-1]["concurrency"], "rows": list(rows)}


def auto_ramp_concurrency(
    url: str,
    model: str,
    *,
    schedule: tuple = (1, 2, 4, 8, 16, 32),
    threshold: float = 0.1,
    _measure=None,
    **kw,
) -> dict:
    """Ramp concurrency through *schedule*, stopping when throughput gain plateaus.

    Calls *_measure* (defaults to :func:`run_concurrent`) at each step.  After
    each step beyond the first, computes the relative throughput gain; if it falls
    below *threshold*, the ramp stops early (avoiding unnecessary load on the GPU).
    The final knee is located via :func:`_find_knee` on the accumulated rows.

    Args:
        url: Base URL of the vLLM endpoint.
        model: Model identifier to benchmark.
        schedule: Concurrency levels to try, in ascending order.
        threshold: Minimum relative gain to keep ramping (default 0.1 = 10 %).
        _measure: Override the per-step measurement function (useful for testing).
            Must have signature ``(url, model, *, concurrency, **kw) -> dict``.
            Defaults to :func:`run_concurrent`.
        **kw: Extra keyword arguments forwarded to *_measure* (e.g. ``max_tokens``).

    Returns:
        ``{"knee": int, "rows": list[dict]}`` — same shape as :func:`_find_knee`.
    """
    if _measure is None:
        _measure = run_concurrent

    rows: list[dict] = []
    for c in schedule:
        row = _measure(url, model, concurrency=c, **kw)
        rows.append(row)
        if len(rows) >= 2:
            prev_rps = rows[-2]["requests_per_s"]
            curr_rps = rows[-1]["requests_per_s"]
            if prev_rps != 0 and (curr_rps - prev_rps) / prev_rps < threshold:
                break  # plateau detected — no need to go higher

    return _find_knee(rows, threshold=threshold)


def render_benchmark(result: dict) -> str:
    """Render :func:`run_benchmark` output as a markdown block for a per-model doc."""
    rates = "/".join(str(r) for r in result["decode_rates"])
    pf = result["prefill"]
    return "\n".join(
        [
            f"## Benchmark — `{result['model']}` ({result['purpose']})",
            "",
            f"- Endpoint: `{result['endpoint']}` · `max_model_len` {result['max_model_len']} · "
            f"shape {result['input_len']} in / {result['output_len']} out",
            "",
            "| Metric | Result |",
            "|---|---|",
            f"| **decode throughput** | **{rates} tok/s** (batch=1, greedy, "
            f"{result['output_len']} tok forced) |",
            f"| prefill | {pf['prompt_tokens']} prompt tokens + 16 gen in {pf['seconds']} s |",
        ]
    )
