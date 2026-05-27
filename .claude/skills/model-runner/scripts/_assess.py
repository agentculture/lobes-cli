#!/usr/bin/env python3
"""API-side assessment of a vLLM-served model: correctness probes + throughput.

Talks only to the OpenAI-compatible endpoint (stdlib urllib, no third-party
deps). Host-side facts (GPU memory, weights-on-disk, image tag) are gathered by
the `model-runner.sh assess` wrapper, which prints them alongside this output.

Usage: _assess.py --url http://localhost:8001 [--model NAME] [--decode-tokens 512]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _post(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (local URL)
        return json.load(r)


def _get(url: str, path: str, timeout: int = 10):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:  # noqa: S310
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status, json.load(r)
        return r.status, r.read().decode()


def _trace_field(msg: dict) -> tuple[str | None, int]:
    """Return (field_name, length) of the reasoning trace, whichever key holds it."""
    for key in ("reasoning", "reasoning_content"):
        val = msg.get(key)
        if isinstance(val, str) and val:
            return key, len(val)
    return None, 0


def _probe(url: str, model: str, prompt: str, expect: str) -> dict:
    d = _post(url, {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.3,
    })
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


def _decode_throughput(url: str, model: str, n_tokens: int, runs: int = 2) -> list[float]:
    rates = []
    for _ in range(runs):
        t0 = time.monotonic()
        d = _post(url, {
            "model": model,
            "messages": [{"role": "user", "content": "Write a detailed essay about distributed systems."}],
            "max_tokens": n_tokens,
            "temperature": 0,
            "ignore_eos": True,
        })
        dt = time.monotonic() - t0
        ct = d["usage"]["completion_tokens"]
        rates.append(round(ct / dt, 1))
    return rates


def _prefill(url: str, model: str) -> dict:
    prompt = "Summarize this. " + "The system processes events. " * 400
    t0 = time.monotonic()
    d = _post(url, {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
    })
    dt = time.monotonic() - t0
    return {"prompt_tokens": d["usage"]["prompt_tokens"], "seconds": round(dt, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--model", default=None, help="served model name (defaults to first /v1/models id)")
    ap.add_argument("--decode-tokens", type=int, default=512)
    args = ap.parse_args()
    url = args.url.rstrip("/")

    try:
        hstatus, _ = _get(url, "/health")
    except urllib.error.URLError as e:
        print(f"FAIL: /health unreachable at {url} ({e})", file=sys.stderr)
        return 1

    _, models = _get(url, "/v1/models")
    served = models["data"][0]
    model = args.model or served["id"]
    max_len = served.get("max_model_len")

    print(f"## Assessment — `{model}`\n")
    print(f"- Endpoint: `{url}` · `/health` {hstatus} · `max_model_len` {max_len}\n")

    pa = _probe(url, model, "What is 17 * 23?", "391")
    pb = _probe(url, model, "If a train leaves at 14:45 and arrives at 17:10, "
                            "how long is the journey in minutes?", "145")
    rates = _decode_throughput(url, model, args.decode_tokens)
    pf = _prefill(url, model)

    field = pa["trace_field"] or pb["trace_field"] or "(none)"
    print("| Check | Result |")
    print("|---|---|")
    print(f"| `17 * 23 = 391` | {'PASS' if pa['ok'] else 'FAIL'} "
          f"(finish={pa['finish']}, {pa['completion_tokens']} tok) |")
    print(f"| train 14:45→17:10 = 145 min | {'PASS' if pb['ok'] else 'FAIL'} "
          f"(finish={pb['finish']}, {pb['completion_tokens']} tok) |")
    print(f"| reasoning trace field | `{field}` (len {max(pa['trace_len'], pb['trace_len'])}) |")
    print(f"| **decode throughput** | **{'/'.join(str(r) for r in rates)} tok/s** "
          f"(batch=1, greedy, {args.decode_tokens} tok forced) |")
    print(f"| prefill | {pf['prompt_tokens']} prompt tokens + 16 gen in {pf['seconds']} s |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
