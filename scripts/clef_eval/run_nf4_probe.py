"""t13 A/B: run the Clef probe set in-process through the NF4 transformers path (t1 baseline).

Run inside odm-clef-spike:local with the Aikimi/clef-nf4 flat snapshot as cwd.
Writes the same JSONL row shape as run_probe.py (plus peak_allocated_gib), so
compare.py scores both paths alike. Latency is runner.decide() wall time
(in-process; no HTTP hop). Records the runtime refuses (e.g. > 8192 tokens,
its hard max_length ceiling) are written as failed rows, never dropped.
"""

import json
import pathlib
import sys
import time

sys.path.insert(0, ".")

from clef_runtime.core import validate_request  # noqa: E402
from clef_runtime.runtime import Runner  # noqa: E402

PROBE, OUT, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
WARMUP = 2


def probabilities(answer):
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return {"true": p, "false": 1.0 - p}
    return {str(k): float(v) for k, v in answer["probabilities"].items()}


def emit(**kw):
    print(json.dumps(kw), flush=True)


def request(rec):
    return validate_request(
        {
            "profile": "clef-24gb",
            "questions": rec["questions"],
            "state": rec["state"],
            "max_length": 8192,
        }
    )


def main():
    records = [json.loads(line) for line in open(PROBE, encoding="utf-8")]
    t0 = time.perf_counter()
    runner = Runner.from_directory(pathlib.Path(".").resolve(), "clef-24gb")
    emit(event="loaded", load_s=round(time.perf_counter() - t0, 2))
    try:
        for r in records[:WARMUP]:
            runner.decide(request(r["record"]))
        emit(event="warm")
        with open(OUT, "w", encoding="utf-8") as out:
            for i, r in enumerate(records):
                try:
                    t = time.perf_counter()
                    res = runner.decide(request(r["record"]))
                    ms = (time.perf_counter() - t) * 1000.0
                    row = {
                        "id": r["id"],
                        "source": r["source"],
                        "label": LABEL,
                        "latency_ms": round(ms, 1),
                        "input_tokens": res["usage"]["input_tokens"],
                        "peak_allocated_gib": round(res["peak_allocated_gib"], 2),
                        "probs": {q: probabilities(a) for q, a in res["answers"].items()},
                        "gold": r["gold"],
                    }
                except (
                    Exception
                ) as exc:  # noqa: BLE001 — a failed record is recorded, never dropped
                    row = {
                        "id": r["id"],
                        "source": r["source"],
                        "label": LABEL,
                        "error": repr(exc)[:300],
                    }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                if i % 20 == 0:
                    emit(event="progress", done=i + 1, total=len(records))
        emit(event="done")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
