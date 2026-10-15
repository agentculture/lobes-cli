"""Compare Clef probe runs (from run_probe.py) against the bf16 reference (t23 / t13).

    python3 compare.py --reference bf16-spark.jsonl w4a16-orin.jsonl nvfp4-spark.jsonl

Per run: top-1 accuracy against gold, top-1 agreement with the reference, mean/max
KL(reference || run) per question, and latency (median / p90). A run is PROMOTABLE
only if its top-1 agreement with the reference is >= 0.97 (user-set threshold,
2026-10-14) and it has no failed records. Unit: one question (a record may carry
several questions). Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

THRESHOLD = 0.97


def load(path: str) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in open(path, encoding="utf-8"))}


def top1(probs: dict[str, float]) -> str:
    return max(probs, key=probs.__getitem__)


def kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-9) -> float:
    return sum(pv * math.log((pv + eps) / (q.get(k, 0.0) + eps)) for k, pv in p.items() if pv > 0)


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def score(run: dict[str, dict], ref: dict[str, dict] | None) -> dict:
    n = correct = agree = 0
    kls: list[float] = []
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, correct, agree
    failed = [i for i, row in run.items() if "error" in row]
    for rid, row in run.items():
        if "error" in row:
            continue
        for q, probs in row["probs"].items():
            n += 1
            hit = top1(probs) == row["gold"].get(q)
            correct += hit
            src = by_source[row["source"]]
            src[0] += 1
            src[1] += hit
            if (
                ref is not None
                and rid in ref
                and "error" not in ref[rid]
                and q in ref[rid]["probs"]
            ):
                same = top1(probs) == top1(ref[rid]["probs"][q])
                agree += same
                src[2] += same
                kls.append(kl(ref[rid]["probs"][q], probs))
    lat = [row["latency_ms"] for row in run.values() if "latency_ms" in row]
    out = {
        "questions": n,
        "failed_records": len(failed),
        "top1_accuracy": round(correct / n, 4) if n else None,
        "latency_ms_median": round(pct(lat, 0.5), 1),
        "latency_ms_p90": round(pct(lat, 0.9), 1),
        "by_source": {
            s: {"questions": v[0], "accuracy": round(v[1] / v[0], 4)}
            for s, v in sorted(by_source.items())
        },
    }
    if ref is not None:
        out["top1_agreement_vs_reference"] = round(agree / n, 4) if n else None
        out["kl_mean"] = round(sum(kls) / len(kls), 5) if kls else None
        out["kl_max"] = round(max(kls), 5) if kls else None
        for s, v in by_source.items():
            out["by_source"][s]["agreement"] = round(v[2] / v[0], 4)
        out["promotable"] = bool(n) and not failed and agree / n >= THRESHOLD
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    ref = load(a.reference)
    report = {"threshold_top1_agreement": THRESHOLD, a.reference: score(ref, None)}
    for path in a.runs:
        report[path] = score(load(path), ref)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
