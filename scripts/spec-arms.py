#!/usr/bin/env python3
"""Measure one speculative-decoding ARM across content SHAPES (stdlib only).

WHY THIS EXISTS
----------------
The DSpark spike (docs/plans/2026-08-24-dspark-speculation-on-the-spark-cortex.md,
task t5) compares three speculation arms on the same cortex lane:

    mtp-n2   the incumbent self-hosted MTP draft head, num_speculative_tokens=2
    dspark   a 1.36B block drafter
    none     speculation fully off

Acceptance behaviour is famously content-dependent (structured/code text drafts
well, free-form prose drafts poorly), so a single number for an arm is as
misleading as a single prefill number (see scripts/prefill-depth-curve.py) — this
script always reports per-SHAPE numbers, never a single blended figure.

This script does NOT switch arms itself. The operator restarts the vLLM lane
between arms (see scripts/spike-preflight.sh), and this script is run once per
arm with ``--arm`` naming which one is currently loaded. Each run's ``--json``
output is one transcript file; ``--combine`` reads up to three such transcripts
back in and refuses to print a silently-incomplete three-arm table — a missing
arm is reported as MISSING, never backfilled from an earlier run, and any
MISSING or FAILED cell makes ``--combine`` exit NONZERO unless
``--allow-partial`` is passed (Qodo finding 4), so automation cannot accept a
partial comparison by accident.

USAGE
-----
    # one run per arm, after the operator has restarted the lane for that arm
    python3 scripts/spec-arms.py --url http://127.0.0.1:8001 --arm mtp-n2 \\
        --api-key "$GATEWAY_API_KEY" --model cortex \\
        --metrics-url http://127.0.0.1:8000 --json > arm-mtp-n2.json

    python3 scripts/spec-arms.py --url http://127.0.0.1:8001 --arm dspark \\
        --api-key "$GATEWAY_API_KEY" --model cortex \\
        --docker-container model-gear-vllm-primary --json > arm-dspark.json

    python3 scripts/spec-arms.py --url http://127.0.0.1:8001 --arm none \\
        --api-key "$GATEWAY_API_KEY" --model cortex --json > arm-none.json

    # then, once some/all transcripts exist:
    python3 scripts/spec-arms.py --combine arm-mtp-n2.json arm-dspark.json arm-none.json

ACCEPTANCE-RATE SURFACES
-------------------------
Two surfaces are supported, tried in the order given on the command line, and
every reported acceptance figure NAMES which one produced it:

  --metrics-url   scrapes vLLM's own Prometheus ``/metrics`` (the
                   ``vllm:spec_decode_num_accepted_tokens_total`` /
                   ``..._num_draft_tokens_total`` counters), sampled before and
                   after each shape and reported as the DELTA over that shape's
                   generation — the cumulative counter is never reported as-is.
  --docker-container  greps ``docker logs --since <run start>`` for vLLM's own
                   periodic ``SpecDecoding metrics: ... Avg Draft acceptance
                   rate: NN.N%`` log line and reports the mean of every line
                   emitted during the shape's generation window.

Neither surface is required. With neither flag, or when the arm is ``none``
(no speculation to accept), acceptance is reported as ``"not_applicable"`` /
``"unavailable"`` — never a fabricated number.

BOTH surfaces are ENGINE-WIDE, not per-request (Qodo finding 9). vLLM exposes
no per-request spec-decode counter, so what is measured is "everything this
engine accepted/drafted during this shape's window" — if another client is
served concurrently, its tokens land in this figure too. The script is honest
about that rather than hiding it:

  * every acceptance dict carries ``"scope": "engine_wide_over_shape_window"``
    and ``"request_scoped": false``, and the table prints ``engine-wide``;
  * with ``--metrics-url`` the engine's own ``vllm:request_success_total``
    counter is deltaed over the same window and compared against the ONE
    request this tool issued. More completions than that means someone else's
    traffic is in the number, so the entry is marked
    ``"contaminated": true`` and the table prints ``CONTAMINATED``. Fewer (or
    the counter absent) is inconclusive and reported as ``null``, never as
    clean.
  * the ``docker logs`` surface has no request identifier at all, so its
    ``"contaminated"`` is always ``null`` (unknown) — it can never be shown
    clean. Run it on a quiescent lane or treat the figure as an upper bound.

``--max-seconds`` is a real WALL-CLOCK deadline across the whole streaming
read, not merely urllib's per-operation socket timeout (Qodo finding 10): a
shape that exceeds it is reported as TIMED OUT (an ``error`` entry carrying
``"timed_out": true``), never as a completed measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 - `docker logs` only, args are not shell-interpolated
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ARMS: tuple[str, ...] = ("mtp-n2", "dspark", "none")

# vLLM Prometheus counter names, and the keys this script reports them under.
ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTED_METRIC = "vllm:spec_decode_num_draft_tokens_total"
REQUESTS_METRIC = "vllm:request_success_total"

_ACCEPTED_KEY = "accepted_tokens"
_DRAFTED_KEY = "drafted_tokens"
_RATE_KEY = "acceptance_rate"
_SURFACE_KEY = "surface"
_SCOPE_KEY = "scope"
_CONTAMINATED_KEY = "contaminated"

# Neither acceptance surface can attribute tokens to a single request — see the
# module docstring's ACCEPTANCE-RATE SURFACES section.
ENGINE_WIDE_SCOPE = "engine_wide_over_shape_window"

# Throughput basis markers (Qodo finding 3): a chunk count is NOT a token count.
BASIS_USAGE = "usage_completion_tokens"
BASIS_CHUNKS = "sse_chunk_count_estimate"

# Three content shapes with markedly different draft-acceptance behaviour —
# see the module docstring. Each has its own prompt and generation kwargs.
SHAPES: dict[str, dict] = {
    "code": {
        "label": "structured/code",
        "prompt": (
            "Write a Python function `parse_config(path: str) -> dict` that "
            "reads a simple KEY=VALUE .env-style file, skipping blank lines "
            "and lines starting with '#'. Include type hints and a short "
            "docstring. Return only the function, no surrounding prose."
        ),
        "enable_thinking": False,
        "temperature": 0,
    },
    "reasoning": {
        "label": "reasoning trace",
        "prompt": (
            "A train leaves station A at 14:45 and arrives at station B at "
            "17:10, stopping for 12 minutes at an intermediate station. "
            "If the train's average moving speed was 84 km/h, how far apart "
            "are the two stations? Show your reasoning step by step."
        ),
        "enable_thinking": True,
        "temperature": 0,
    },
    "prose": {
        "label": "free-form prose",
        "prompt": (
            "Write a short, reflective paragraph about the first time you "
            "noticed the seasons changing somewhere you lived. Make it "
            "specific and sensory, not generic."
        ),
        "enable_thinking": False,
        "temperature": 0.7,
    },
}


# ---------------------------------------------------------------------------
# Pure parsing/bookkeeping helpers (offline-testable, no network/subprocess).
# ---------------------------------------------------------------------------


def parse_metrics_text(text: str) -> dict[str, float]:
    """Parse vLLM Prometheus ``/metrics`` text into a flat counter dict.

    Only the counters this script needs are extracted, keyed by their bare
    metric name (labels are ignored — this fleet runs one engine per lane, so a
    single label set is expected). Missing metrics are simply absent from the
    returned dict; callers must not assume presence.

    ``vllm:request_success_total`` is collected alongside the two spec-decoding
    counters purely so the engine-wide acceptance window can be checked for
    contamination by other clients' traffic (Qodo finding 9); it is summed
    across its ``finished_reason`` label sets.
    """
    wanted = (ACCEPTED_METRIC, DRAFTED_METRIC, REQUESTS_METRIC)
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for name in wanted:
            if line.startswith(name):
                # "metric{labels} value" or "metric value"
                try:
                    value = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    continue
                # Sum across label sets (e.g. multiple engines) rather than
                # overwrite — this fleet runs one engine today, but summing is
                # the honest behaviour if that ever changes.
                out[name] = out.get(name, 0.0) + value
    return out


def requests_delta(before: dict[str, float], after: dict[str, float]) -> float | None:
    """Completed-request count over the window, or ``None`` if the counter is absent."""
    if REQUESTS_METRIC not in before or REQUESTS_METRIC not in after:
        return None
    return after[REQUESTS_METRIC] - before[REQUESTS_METRIC]


def acceptance_delta(
    before: dict[str, float], after: dict[str, float], expected_requests: int = 1
) -> dict | None:
    """Delta two ``/metrics`` snapshots into an ENGINE-WIDE acceptance-rate figure.

    The counters are engine-wide, so the returned rate covers every request the
    engine served during the window, not just the one this tool issued — the
    returned dict says so explicitly (``scope``/``request_scoped``) and never
    presents itself as a per-request measurement (Qodo finding 9).

    Contamination is detected where the engine makes it detectable: the
    ``vllm:request_success_total`` delta is compared against
    ``expected_requests`` (the number of requests THIS tool issued in the
    window). More completions than that means foreign traffic is folded in and
    ``contaminated`` is ``True``. When the counter is missing — or reports
    FEWER completions than expected, which happens when a request is still
    in flight at sampling time — the answer is unknown and ``contaminated`` is
    ``None``. It is never reported ``False`` on absent evidence.

    Returns ``None`` when either snapshot lacks both spec counters, or when no
    draft tokens were produced in the window (rate is undefined, not zero).
    """
    if ACCEPTED_METRIC not in before or ACCEPTED_METRIC not in after:
        return None
    if DRAFTED_METRIC not in before or DRAFTED_METRIC not in after:
        return None
    accepted = after[ACCEPTED_METRIC] - before[ACCEPTED_METRIC]
    drafted = after[DRAFTED_METRIC] - before[DRAFTED_METRIC]
    if drafted <= 0:
        return None

    observed = requests_delta(before, after)
    if observed is None:
        contaminated: bool | None = None
        note = (
            f"{REQUESTS_METRIC} absent — cannot tell whether other requests "
            "were served in this window"
        )
    elif observed > expected_requests:
        contaminated = True
        note = (
            f"{observed:g} requests completed in this window but this tool "
            f"issued {expected_requests} — the acceptance figure includes "
            "other clients' traffic and is NOT a valid per-shape measurement"
        )
    elif observed < expected_requests:
        contaminated = None
        note = (
            f"only {observed:g} of {expected_requests} issued request(s) had "
            "completed at sampling time — window boundaries are approximate, "
            "contamination undetermined"
        )
    else:
        contaminated = False
        note = (
            f"exactly {expected_requests} request completed in this window — "
            "no foreign traffic detected, but the counters remain engine-wide"
        )

    return {
        _ACCEPTED_KEY: accepted,
        _DRAFTED_KEY: drafted,
        _RATE_KEY: round(accepted / drafted, 4),
        _SURFACE_KEY: "vllm_metrics_http",
        _SCOPE_KEY: ENGINE_WIDE_SCOPE,
        "request_scoped": False,
        "requests_in_window": observed,
        "expected_requests": expected_requests,
        _CONTAMINATED_KEY: contaminated,
        "note": note,
    }


# vLLM's periodic INFO line, e.g.:
#   SpecDecoding metrics: Mean acceptance length: 2.77, Accepted throughput:
#   5.50 tokens/s, Drafted throughput: 6.20 tokens/s, Accepted: 55 tokens,
#   Drafted: 62 tokens, Per-position acceptance rate: 0.903, 0.871, Avg Draft
#   acceptance rate: 88.7%
_LOG_LINE_RE = re.compile(
    r"Mean acceptance length:\s*(?P<mean_len>[\d.]+).*?"
    r"Accepted:\s*(?P<accepted>\d+)\s*tokens.*?"
    r"Drafted:\s*(?P<drafted>\d+)\s*tokens.*?"
    r"Avg Draft acceptance rate:\s*(?P<rate_pct>[\d.]+)%"
)


def parse_acceptance_log_line(line: str) -> dict | None:
    """Parse one vLLM ``SpecDecoding metrics:`` log line, or ``None`` if it doesn't match."""
    m = _LOG_LINE_RE.search(line)
    if not m:
        return None
    return {
        "mean_acceptance_length": float(m.group("mean_len")),
        _ACCEPTED_KEY: int(m.group("accepted")),
        _DRAFTED_KEY: int(m.group("drafted")),
        _RATE_KEY: round(float(m.group("rate_pct")) / 100.0, 4),
    }


def summarize_log_lines(lines: list[str]) -> dict | None:
    """Mean acceptance rate across every ``SpecDecoding metrics:`` line in a window.

    vLLM's periodic log line carries NO request identifier, so this aggregate
    covers every request the engine served while the shape ran — and unlike the
    ``/metrics`` surface there is nothing to cross-check it against. Its
    ``contaminated`` is therefore always ``None`` (unknown), never ``False``
    (Qodo finding 9).
    """
    parsed = [p for p in (parse_acceptance_log_line(ln) for ln in lines) if p]
    if not parsed:
        return None
    total_accepted = sum(p[_ACCEPTED_KEY] for p in parsed)
    total_drafted = sum(p[_DRAFTED_KEY] for p in parsed)
    return {
        _ACCEPTED_KEY: total_accepted,
        _DRAFTED_KEY: total_drafted,
        _RATE_KEY: (round(total_accepted / total_drafted, 4) if total_drafted > 0 else None),
        "sample_count": len(parsed),
        _SURFACE_KEY: "docker_logs",
        _SCOPE_KEY: ENGINE_WIDE_SCOPE,
        "request_scoped": False,
        _CONTAMINATED_KEY: None,
        "note": (
            "docker log lines carry no request id — this aggregates every "
            "request the engine served during the shape's window and cannot "
            "be shown clean; treat it as an upper bound unless the lane was "
            "known-quiescent"
        ),
    }


def build_comparison(
    transcripts: dict[str, dict], required_arms: tuple[str, ...] = ARMS
) -> tuple[list[dict], list[str]]:
    """Build per-shape, per-arm comparison rows from loaded transcripts.

    ``transcripts`` maps arm name -> its loaded JSON dict (as emitted by this
    script's ``--json`` mode). Returns ``(rows, missing_arms)``: ``rows`` has
    one entry per (shape, arm) pair for EVERY required arm — a missing arm's
    entries carry ``"status": "MISSING"`` and no fabricated numbers, and a
    shape whose measurement errored carries ``"status": "FAILED"`` — and
    ``missing_arms`` lists which required arms had no transcript at all. This
    function never raises on missing data; it is the caller's job to decide
    whether incomplete data is acceptable to print (see ``--allow-partial``).
    """
    missing_arms = [arm for arm in required_arms if arm not in transcripts]
    rows: list[dict] = []
    for shape_name in SHAPES:
        for arm in required_arms:
            if arm not in transcripts:
                rows.append({"shape": shape_name, "arm": arm, "status": "MISSING"})
                continue
            shapes = transcripts[arm].get("shapes", {})
            entry = shapes.get(shape_name)
            if not entry:
                rows.append(
                    {
                        "shape": shape_name,
                        "arm": arm,
                        "status": "MISSING",
                        "note": "arm transcript present but shape absent",
                    }
                )
                continue
            # A shape the arm's own run reported as errored (including a
            # wall-clock timeout) is FAILED, not a usable cell — see
            # incomplete_cells() and --allow-partial (Qodo finding 4).
            status = "FAILED" if entry.get("error") else "ok"
            row = {"shape": shape_name, "arm": arm}
            row.update(entry)
            row["status"] = status
            rows.append(row)
    return rows, missing_arms


def consume_sse_stream(raw_lines, *, now, t0: float, deadline: float) -> dict:
    """Consume an OpenAI SSE stream, bounded by a WALL-CLOCK ``deadline``.

    Pure over its inputs — ``raw_lines`` is any iterable of ``bytes``/``str``
    lines and ``now`` is a monotonic clock callable — so the deadline and the
    chunk/usage bookkeeping are testable without a server (Qodo finding 10).

    Returns ``{"ttft", "chunks", "usage", "last", "timed_out"}``. ``chunks``
    counts nonempty SSE deltas, which is NOT a token count — see
    ``measure_shape`` (Qodo finding 3). ``timed_out`` is ``True`` when the
    clock passed ``deadline`` before ``[DONE]``; the caller must not treat such
    a stream as a completed measurement.
    """
    ttft: float | None = None
    chunks = 0
    last = t0
    usage: dict = {}
    timed_out = False
    for raw in raw_lines:
        if now() > deadline:
            timed_out = True
            break
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        # A reasoning-mode shape streams its trace on delta.reasoning /
        # delta.reasoning_content, NOT delta.content (the served build varies
        # which key it uses — see lobes/assess.py's _trace_field for the same
        # split) — content only carries text once the model exits <think>.
        # Counting only "content" here would report a reasoning-heavy
        # generation as having streamed nothing at all.
        text = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
        if text:
            if ttft is None:
                ttft = now() - t0
            chunks += 1
            last = now()
    return {"ttft": ttft, "chunks": chunks, "usage": usage, "last": last, "timed_out": timed_out}


def incomplete_cells(rows: list[dict]) -> list[dict]:
    """Every comparison cell that is not a usable measurement.

    A three-arm comparison is only complete when every (shape, arm) cell is
    ``ok``. MISSING (no transcript / no shape) and FAILED (the shape errored or
    timed out) both make the table an incomplete comparison, and ``--combine``
    exits nonzero on either unless ``--allow-partial`` is given.
    """
    return [
        {"shape": r["shape"], "arm": r["arm"], "status": r.get("status", "MISSING")}
        for r in rows
        if r.get("status") != "ok"
    ]


# ---------------------------------------------------------------------------
# Network I/O (urllib only, mirrors lobes/assess.py and prefill-depth-curve.py).
# ---------------------------------------------------------------------------


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def fetch_max_model_len(url: str, model: str, api_key: str | None, timeout: float) -> object:
    """Best-effort ``max_model_len`` for ``model`` from ``/v1/models``. ``None`` if unknown."""
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/models", headers=dict(_headers(api_key) if api_key else {})
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    for entry in payload.get("data") or []:
        if entry.get("id") == model or model in ("cortex", "main"):
            if "max_model_len" in entry:
                return entry["max_model_len"]
    data = payload.get("data") or []
    return data[0].get("max_model_len") if data else None


def fetch_metrics_snapshot(metrics_url: str, timeout: float) -> dict[str, float] | None:
    req = urllib.request.Request(metrics_url.rstrip("/") + "/metrics")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    return parse_metrics_text(text)


def fetch_docker_log_window(container: str, since_iso: str, timeout: float) -> list[str] | None:
    """Lines vLLM logged since ``since_iso`` (RFC3339), via ``docker logs --since``."""
    try:
        # nosec B603/B607 - fixed argv, no shell, operator-supplied container name
        proc = subprocess.run(
            ["docker", "logs", "--since", since_iso, container],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (proc.stdout + proc.stderr).splitlines()


def measure_shape(
    url: str,
    model: str,
    shape_name: str,
    shape_spec: dict,
    api_key: str | None,
    max_seconds: float,
    gen_tokens: int,
) -> dict:
    """Stream one shape's generation and time TTFT/decode (mirrors prefill-depth-curve.py)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": shape_spec["prompt"]}],
        "max_tokens": gen_tokens,
        "temperature": shape_spec["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": shape_spec["enable_thinking"]},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=_headers(api_key),
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=max_seconds) as resp:
            stream = consume_sse_stream(
                resp, now=time.perf_counter, t0=t0, deadline=t0 + max_seconds
            )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"shape": shape_name, "label": shape_spec["label"], "error": str(exc)}

    base = {"shape": shape_name, "label": shape_spec["label"]}

    # A shape that blew the wall-clock deadline is a TIMED-OUT shape, never a
    # completed measurement, even though partial content did stream (Qodo
    # finding 10). urllib's own timeout is per-socket-operation: a server that
    # keeps emitting chunks resets it forever, so the deadline below is the
    # only thing that actually bounds a shape.
    if stream["timed_out"]:
        return {
            **base,
            "error": (
                f"wall-clock deadline of {max_seconds:g}s exceeded while "
                f"streaming (--max-seconds); {stream['chunks']} content "
                "chunk(s) had arrived — reported as TIMED OUT, not measured"
            ),
            "timed_out": True,
            "elapsed_s": round(stream["last"] - t0, 2),
            "sse_content_chunks": stream["chunks"],
        }

    ttft = stream["ttft"]
    if ttft is None:
        return {
            **base,
            "error": "no content or reasoning tokens streamed "
            "(lane may be pressure-shedding — a 429/busy body arrives as a "
            "non-SSE 200 payload with no 'data: ' lines)",
        }

    gen_window = max(stream["last"] - t0 - ttft, 1e-9)
    n_chunks = stream["chunks"]
    # NEVER label a chunk count as a token count (Qodo finding 3). One SSE
    # delta may carry several tokens, so the chunk count is a property of
    # server/network chunking, not of generation. The real
    # usage.completion_tokens is preferred (this script asks for it via
    # stream_options.include_usage); when the server omits it, the chunk count
    # is reported under its OWN key and the derived tok/s is flagged an
    # estimate everywhere it appears.
    completion_tokens = (stream["usage"] or {}).get("completion_tokens")
    if isinstance(completion_tokens, (int, float)):
        basis, tokens, estimated = BASIS_USAGE, completion_tokens, False
    else:
        completion_tokens = None
        basis, tokens, estimated = BASIS_CHUNKS, n_chunks, True

    row = {
        **base,
        "ttft_ms": round(ttft * 1000, 1),
        "completion_tokens": completion_tokens,
        "sse_content_chunks": n_chunks,
        "decode_tok_s": round(tokens / gen_window, 2),
        "decode_tok_s_estimated": estimated,
        "throughput_basis": basis,
        "wall_s": round(stream["last"] - t0, 2),
    }
    if estimated:
        row["throughput_note"] = (
            "server returned no usage.completion_tokens; decode_tok_s is an "
            "ESTIMATE derived from the count of nonempty SSE deltas, which "
            "depends on server/network chunking rather than tokens generated"
        )
    return row


def _no_acceptance(surface: str) -> dict:
    """A no-number acceptance entry, shaped like the real ones (never fabricates a rate)."""
    return {
        _RATE_KEY: None,
        _SURFACE_KEY: surface,
        _SCOPE_KEY: None,
        "request_scoped": False,
        _CONTAMINATED_KEY: None,
    }


def run_arm(
    url: str,
    model: str,
    arm: str,
    api_key: str | None,
    max_seconds: float,
    gen_tokens: int,
    metrics_url: str | None,
    docker_container: str | None,
) -> dict:
    """Measure every shape for one arm, attaching whichever acceptance surface is configured."""
    max_model_len = fetch_max_model_len(url, model, api_key, max_seconds)
    shapes_out: dict[str, dict] = {}
    for shape_name, shape_spec in SHAPES.items():
        since = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        metrics_before = fetch_metrics_snapshot(metrics_url, max_seconds) if metrics_url else None

        row = measure_shape(url, model, shape_name, shape_spec, api_key, max_seconds, gen_tokens)
        row["arm"] = arm
        row["max_model_len"] = max_model_len

        acceptance = None
        if arm == "none":
            acceptance = _no_acceptance("not_applicable")
        elif metrics_url:
            metrics_after = fetch_metrics_snapshot(metrics_url, max_seconds)
            if metrics_before is not None and metrics_after is not None:
                # exactly one request was issued for this shape, above
                acceptance = acceptance_delta(metrics_before, metrics_after, expected_requests=1)
            if acceptance is None:
                acceptance = _no_acceptance("unavailable")
        elif docker_container:
            lines = fetch_docker_log_window(docker_container, since, max_seconds)
            summary = summarize_log_lines(lines) if lines else None
            acceptance = summary or _no_acceptance("unavailable")
        else:
            acceptance = _no_acceptance("unconfigured")

        row["acceptance"] = acceptance
        shapes_out[shape_name] = row

    return {
        "arm": arm,
        "url": url,
        "model": model,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "shapes": shapes_out,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_row(shape_name: str, entry: dict) -> str:
    label = SHAPES.get(shape_name, {}).get("label", shape_name)
    arm = entry.get("arm", "?")
    if entry.get("status") == "MISSING":
        return f"  [{label:>16} | {arm:>7}]  MISSING"
    if "error" in entry:
        verdict = "TIMED OUT" if entry.get("timed_out") else "FAILED"
        return f"  [{label:>16} | {arm:>7}]  {verdict} — {entry['error']}"
    acc = entry.get("acceptance") or {}
    rate = acc.get(_RATE_KEY)
    rate_str = f"{rate * 100:.1f}%" if isinstance(rate, (int, float)) else "n/a"
    surface = acc.get(_SURFACE_KEY, "unconfigured")
    # Say out loud what the acceptance figure actually covers (Qodo finding 9).
    if acc.get(_SCOPE_KEY) == ENGINE_WIDE_SCOPE:
        contaminated = acc.get(_CONTAMINATED_KEY)
        if contaminated is True:
            surface += ", engine-wide, CONTAMINATED"
        elif contaminated is None:
            surface += ", engine-wide, contamination unknown"
        else:
            surface += ", engine-wide"
    # ... and never let an estimated tok/s pass for a measured one (finding 3).
    tok_s = entry.get("decode_tok_s", "?")
    tok_s_str = f"~{tok_s} tok/s (est)" if entry.get("decode_tok_s_estimated") else f"{tok_s} tok/s"
    return (
        f"  [{label:>16} | {arm:>7}]  "
        f"ttft {entry.get('ttft_ms', '?'):>8} ms | "
        f"decode {tok_s_str:>18} | "
        f"accept {rate_str:>7} (via {surface}) | "
        f"max_model_len {entry.get('max_model_len', '?')}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure one speculation arm across content shapes, or "
        "combine prior per-arm transcripts into a comparison."
    )
    ap.add_argument("--url", help="gateway or lane origin, e.g. http://127.0.0.1:8001")
    ap.add_argument("--model", default="cortex", help="model or role alias (default: cortex)")
    ap.add_argument("--arm", choices=ARMS, help="the speculation arm currently loaded on --url")
    ap.add_argument("--api-key", default=None, help="bearer token, if the gateway gate is armed")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=180.0,
        help="per-shape WALL-CLOCK ceiling (default 180); a shape that "
        "exceeds it is reported TIMED OUT, not measured",
    )
    ap.add_argument(
        "--gen-tokens", type=int, default=256, help="tokens to generate per shape (default 256)"
    )
    ap.add_argument(
        "--metrics-url",
        default=None,
        help="origin serving vLLM's own /metrics (tried before --docker-container)",
    )
    ap.add_argument(
        "--docker-container",
        default=None,
        help="container name to `docker logs --since` for the SpecDecoding log line",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument(
        "--combine",
        nargs="+",
        metavar="TRANSCRIPT.json",
        default=None,
        help="combine 1-3 prior --json transcripts into a comparison instead of measuring",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="with --combine, exit 0 even when a (shape, arm) cell is MISSING "
        "or FAILED (default: exit 1; the rows are printed either way)",
    )
    args = ap.parse_args(argv)

    if args.combine:
        transcripts: dict[str, dict] = {}
        for path in args.combine:
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as exc:
                print(f"error: could not read {path}: {exc}", file=sys.stderr)
                return 2
            arm = data.get("arm")
            if arm not in ARMS:
                print(f"error: {path} has no recognised 'arm' field", file=sys.stderr)
                return 2
            if arm in transcripts:
                print(f"error: two transcripts both claim arm={arm}", file=sys.stderr)
                return 2
            transcripts[arm] = data

        rows, missing_arms = build_comparison(transcripts)
        incomplete = incomplete_cells(rows)
        if args.json:
            print(
                json.dumps(
                    {
                        "rows": rows,
                        "missing_arms": missing_arms,
                        "incomplete_cells": incomplete,
                        "complete": not incomplete,
                    },
                    indent=2,
                )
            )
        else:
            if missing_arms:
                print(
                    f"MISSING arm data: {', '.join(missing_arms)} — comparison "
                    "is INCOMPLETE, not a full three-arm result.",
                    file=sys.stderr,
                )
            for row in rows:
                print(_fmt_row(row["shape"], row))
        # Any MISSING or FAILED cell means this is not the three-arm comparison
        # the caller asked for; exit nonzero so automation cannot accept it by
        # accident (Qodo finding 4).
        if incomplete and not args.allow_partial:
            print(
                f"INCOMPLETE comparison: {len(incomplete)} of {len(rows)} "
                "(shape, arm) cells are not usable measurements — "
                + ", ".join(f"{c['shape']}/{c['arm']}={c['status']}" for c in incomplete)
                + ". Re-run the affected arms, or pass --allow-partial to "
                "accept an incomplete table.",
                file=sys.stderr,
            )
            return 1
        return 0

    if not args.url or not args.arm:
        ap.error("--url and --arm are required unless --combine is given")

    result = run_arm(
        args.url,
        args.model,
        args.arm,
        args.api_key,
        args.max_seconds,
        args.gen_tokens,
        args.metrics_url,
        args.docker_container,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"arm: {result['arm']}  model: {result['model']}  url: {result['url']}")
        for shape_name, entry in result["shapes"].items():
            print(_fmt_row(shape_name, entry))
            sys.stdout.flush()

    any_error = any("error" in e for e in result["shapes"].values())
    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
