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
arm is reported as MISSING, never backfilled from an earlier run.

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
``"unavailable"`` — never a fabricated number. ``--max-seconds`` bounds every
individual shape's request so one hung shape cannot stall the run.
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

    Only the two spec-decoding counters this script needs are extracted, keyed
    by their bare metric name (labels are ignored — this fleet runs one engine
    per lane, so a single label set is expected). Missing metrics are simply
    absent from the returned dict; callers must not assume presence.
    """
    wanted = (
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
    )
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


def acceptance_delta(before: dict[str, float], after: dict[str, float]) -> dict | None:
    """Delta two ``/metrics`` snapshots into an acceptance-rate figure.

    Returns ``None`` when either snapshot lacks both counters, or when no
    draft tokens were produced in the window (rate is undefined, not zero).
    """
    accepted_key = "vllm:spec_decode_num_accepted_tokens_total"
    drafted_key = "vllm:spec_decode_num_draft_tokens_total"
    if accepted_key not in before or accepted_key not in after:
        return None
    if drafted_key not in before or drafted_key not in after:
        return None
    accepted = after[accepted_key] - before[accepted_key]
    drafted = after[drafted_key] - before[drafted_key]
    if drafted <= 0:
        return None
    return {
        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "acceptance_rate": round(accepted / drafted, 4),
        "surface": "vllm_metrics_http",
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
        "accepted_tokens": int(m.group("accepted")),
        "drafted_tokens": int(m.group("drafted")),
        "acceptance_rate": round(float(m.group("rate_pct")) / 100.0, 4),
    }


def summarize_log_lines(lines: list[str]) -> dict | None:
    """Mean acceptance rate across every ``SpecDecoding metrics:`` line in a window."""
    parsed = [p for p in (parse_acceptance_log_line(ln) for ln in lines) if p]
    if not parsed:
        return None
    total_accepted = sum(p["accepted_tokens"] for p in parsed)
    total_drafted = sum(p["drafted_tokens"] for p in parsed)
    return {
        "accepted_tokens": total_accepted,
        "drafted_tokens": total_drafted,
        "acceptance_rate": (
            round(total_accepted / total_drafted, 4) if total_drafted > 0 else None
        ),
        "sample_count": len(parsed),
        "surface": "docker_logs",
    }


def build_comparison(
    transcripts: dict[str, dict], required_arms: tuple[str, ...] = ARMS
) -> tuple[list[dict], list[str]]:
    """Build per-shape, per-arm comparison rows from loaded transcripts.

    ``transcripts`` maps arm name -> its loaded JSON dict (as emitted by this
    script's ``--json`` mode). Returns ``(rows, missing_arms)``: ``rows`` has
    one entry per (shape, arm) pair for EVERY required arm — a missing arm's
    entries carry ``"status": "MISSING"`` and no fabricated numbers — and
    ``missing_arms`` lists which required arms had no transcript at all. This
    function never raises on missing data; it is the caller's job to decide
    whether MISSING data is acceptable to print (see ``--allow-partial``).
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
            row = {"shape": shape_name, "arm": arm, "status": "ok"}
            row.update(entry)
            rows.append(row)
    return rows, missing_arms


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
    ttft = None
    n_content = 0
    last = t0
    usage: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=max_seconds) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
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
                # delta.reasoning_content, NOT delta.content (the served build
                # varies which key it uses — see lobes/assess.py's
                # _trace_field for the same split) — content only carries
                # text once the model exits <think>. Counting only "content"
                # here would report a reasoning-heavy generation as having
                # streamed nothing at all.
                text = (
                    delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                )
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_content += 1
                    last = time.perf_counter()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"shape": shape_name, "label": shape_spec["label"], "error": str(exc)}

    if ttft is None:
        return {
            "shape": shape_name,
            "label": shape_spec["label"],
            "error": "no content or reasoning tokens streamed "
            "(lane may be pressure-shedding — a 429/busy body arrives as a "
            "non-SSE 200 payload with no 'data: ' lines)",
        }

    gen_window = max(last - t0 - ttft, 1e-9)
    completion_tokens = usage.get("completion_tokens", n_content)
    return {
        "shape": shape_name,
        "label": shape_spec["label"],
        "ttft_ms": round(ttft * 1000, 1),
        "completion_tokens": completion_tokens,
        "decode_tok_s": round(completion_tokens / gen_window, 2),
        "wall_s": round(last - t0, 2),
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
            acceptance = {"acceptance_rate": None, "surface": "not_applicable"}
        elif metrics_url:
            metrics_after = fetch_metrics_snapshot(metrics_url, max_seconds)
            if metrics_before is not None and metrics_after is not None:
                acceptance = acceptance_delta(metrics_before, metrics_after)
            if acceptance is None:
                acceptance = {"acceptance_rate": None, "surface": "unavailable"}
        elif docker_container:
            lines = fetch_docker_log_window(docker_container, since, max_seconds)
            summary = summarize_log_lines(lines) if lines else None
            acceptance = summary or {"acceptance_rate": None, "surface": "unavailable"}
        else:
            acceptance = {"acceptance_rate": None, "surface": "unconfigured"}

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
        return f"  [{label:>16} | {arm:>7}]  FAILED — {entry['error']}"
    acc = entry.get("acceptance") or {}
    rate = acc.get("acceptance_rate")
    rate_str = f"{rate * 100:.1f}%" if isinstance(rate, (int, float)) else "n/a"
    surface = acc.get("surface", "unconfigured")
    return (
        f"  [{label:>16} | {arm:>7}]  "
        f"ttft {entry.get('ttft_ms', '?'):>8} ms | "
        f"decode {entry.get('decode_tok_s', '?'):>6} tok/s | "
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
        "--max-seconds", type=float, default=180.0, help="per-shape request ceiling (default 180)"
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
        help="with --combine, exit 0 even when an arm's transcript is missing "
        "(default: exit 1, MISSING rows still printed either way)",
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
        if args.json:
            print(json.dumps({"rows": rows, "missing_arms": missing_arms}, indent=2))
        else:
            if missing_arms:
                print(
                    f"MISSING arm data: {', '.join(missing_arms)} — comparison "
                    "is INCOMPLETE, not a full three-arm result.",
                    file=sys.stderr,
                )
            for row in rows:
                print(_fmt_row(row["shape"], row))
        if missing_arms and not args.allow_partial:
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
