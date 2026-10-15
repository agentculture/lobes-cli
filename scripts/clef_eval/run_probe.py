"""Run the Clef probe set against one /v1/systemone endpoint (t11 / t13 / t23).

Stdlib only. Reads the materialized probe set (``build_sets.py materialize``),
POSTs each record, and writes one JSONL line per record with the full
per-option probabilities, the latency and the input token count:

    python3 run_probe.py --url http://127.0.0.1:18080/v1/systemone \\
        --probe probe.jsonl --out bf16-spark.jsonl [--header 'X-Lobes-Model: decider:general'] \\
        [--bearer-env GATEWAY_API_KEY] [--label bf16-spark]

The bearer token, if any, is read from the named environment variable and is
never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def probabilities(answer: dict) -> dict[str, float]:
    """Per-option probabilities of one SystemOne answer, for any question type."""
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def post(url: str, body: dict, headers: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 — operator-given URL
        payload = json.loads(resp.read())
    return payload, (time.perf_counter() - t0) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--header", action="append", default=[])
    ap.add_argument("--bearer-env", default="")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()

    headers = dict(h.split(": ", 1) for h in a.header)
    if a.bearer_env:
        headers["Authorization"] = "Bearer " + os.environ[a.bearer_env]
    records = [json.loads(line) for line in open(a.probe, encoding="utf-8")]
    for r in records[: a.warmup]:
        post(a.url, r["record"], headers, a.timeout)
    with open(a.out, "w", encoding="utf-8") as out:
        for r in records:
            try:
                payload, ms = post(a.url, r["record"], headers, a.timeout)
                row = {
                    "id": r["id"],
                    "source": r["source"],
                    "label": a.label,
                    "latency_ms": round(ms, 1),
                    "input_tokens": payload.get("usage", {}).get("input_tokens"),
                    "probs": {q: probabilities(ans) for q, ans in payload["answers"].items()},
                    "gold": r["gold"],
                }
            except Exception as exc:  # noqa: BLE001 — a failed record is recorded, never dropped
                row = {
                    "id": r["id"],
                    "source": r["source"],
                    "label": a.label,
                    "error": repr(exc)[:300],
                }
            out.write(json.dumps(row) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
