#!/usr/bin/env python3
"""Reranker calibration probe (issue #227) — stdlib only, read-only.

Scores a fixed probe set through the gateway's ``/v1/rerank`` and prints, per
document, the ``relevance_score`` plus the response's ``usage.prompt_tokens``
(total and per pair) — the one externally visible tell of whether the served
lane renders the model card's judge prompt (~24 tokens/pair bare, ~85
templated). Also times the doc's 1x5 rerank shape and probes whether a
top-level ``instruction`` changes the score.

Run BEFORE and AFTER the compose change so the two transcripts under
``docs/evidence/`` are the same script's verbatim output::

    uv run python scripts/probe_reranker_calibration.py --url http://localhost:8001 \
        | tee docs/evidence/<date>-baseline-reranker-untemplated-spark.txt

``--key`` (or ``GATEWAY_API_KEY`` in the env, or in ``~/.lobes/.env`` /
``~/.lobes/.secrets.env``) is the gateway bearer; ``--model`` defaults to the
catalog's reranker id. Nothing here mutates anything.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess  # nosec B404 - read-only `git`/`docker inspect` for the header
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# The #220 report's cases, the assess.py France probe, and the challenge
# pass's graded-relevance case (two relevant docs, one clearly better).
PROBE_SET: list[tuple[str, str, list[str]]] = [
    (
        "sky",
        "What colour is the sky on a clear day?",
        [
            "On a clear day the sky is blue because air scatters short wavelengths.",
            "Cats purr when they are content.",
            "The invoice is due on the last business day of the month.",
        ],
    ),
    (
        "ports-ledger",
        "Which file lists the ports the gateway exposes?",
        [
            "The gateway port ledger is docs/gateway-fleet.md, which lists 8000 and 8001.",
            "Cats purr when they are content.",
            "Bananas are a good source of potassium.",
        ],
    ),
    (
        "france (assess probe)",
        "What is the capital of France?",
        [
            "Paris is the capital and most populous city of France.",
            "The Amazon rainforest spans several South American countries.",
            "Bananas are a good source of potassium.",
        ],
    ),
    (
        "toolbatch inversion",
        "How does the tool batcher group tool calls into one request?",
        [
            "NOTICE: this file is generated; do not edit by hand.",
            "toolbatch collects consecutive tool calls and issues them as a single batched request to the gateway.",
            "Cats purr when they are content.",
        ],
    ),
    (
        "graded relevance",
        "How do I stop the lobes fleet without deleting the deployment directory?",
        [
            "Run `lobes stop --apply`; it runs `docker compose down`, removing the containers but leaving ~/.lobes and its .env untouched.",  # noqa: E501
            "`lobes stop` stops the fleet.",
            "Cats purr when they are content.",
        ],
    ),
]

LATENCY_QUERY = "What is the capital of France?"
LATENCY_DOCS = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "Rome is the capital of Italy.",
    "Madrid is the capital of Spain.",
    "Lisbon is the capital of Portugal.",
]


def _read_env_key(name: str) -> str | None:
    for p in (Path.home() / ".lobes" / ".env", Path.home() / ".lobes" / ".secrets.env"):
        try:
            for line in p.read_text().splitlines():
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip()
        except OSError:
            continue
    return None


def _post(url: str, key: str | None, body: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t0 = time.monotonic()
    with urllib.request.urlopen(
        req, timeout=timeout
    ) as resp:  # nosec B310 - operator-supplied http(s) URL
        out = json.load(resp)
    return out, (time.monotonic() - t0) * 1000


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(  # nosec B603 - fixed argv, read-only commands
            cmd, capture_output=True, text=True, timeout=20, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "(unavailable)"


def header(url: str, model: str) -> None:
    print("# reranker calibration probe (#227)")
    print(f"date:       {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"host:       {platform.node()}")
    print(f"url:        {url}")
    print(f"model:      {model}")
    print(f"git sha:    {_run(['git', 'rev-parse', '--short', 'HEAD'])}")
    print(
        f"container:  {_run(['docker', 'inspect', 'model-gear-vllm-rerank', '--format', '{{.Config.Image}}'])}"
    )
    args = _run(["docker", "inspect", "model-gear-vllm-rerank", "--format", "{{json .Args}}"])
    print(f"args:       {args}")
    print(f"templated:  {'--chat-template' in args}")
    ver = _run(
        [
            "docker",
            "exec",
            "model-gear-vllm-rerank",
            "python3",
            "-c",
            "import vllm;print(vllm.__version__)",
        ]
    )
    print(f"vllm:       {ver}")
    print()


def probe(url: str, key: str | None, model: str, timeout: float) -> None:
    print("## probe set — relevance_score per document, usage.prompt_tokens")
    for name, query, docs in PROBE_SET:
        out, ms = _post(
            f"{url}/v1/rerank",
            key,
            {"model": model, "query": query, "documents": docs},
            timeout,
        )
        usage = out.get("usage") or {}
        total = usage.get("prompt_tokens")
        per_pair = round(total / len(docs), 1) if isinstance(total, int) else None
        print(f"\n[{name}]  Q: {query}")
        print(f"  prompt_tokens total={total} per_pair={per_pair}  latency={ms:.0f}ms")
        by_index = {r["index"]: r["relevance_score"] for r in out["results"]}
        order = [r["index"] for r in sorted(out["results"], key=lambda r: -r["relevance_score"])]
        for i, d in enumerate(docs):
            print(f"  [{i}] {by_index[i]:.3f}  {d[:78]}")
        print(f"  ranking: {order}")


def instruction_probe(url: str, key: str | None, model: str, timeout: float) -> None:
    print("\n## instruction probe — same pair with and without a top-level `instruction`")
    query = "Which file lists the ports the gateway exposes?"
    docs = [
        "The gateway port ledger is docs/gateway-fleet.md, which lists 8000 and 8001.",
        "Cats purr when they are content.",
    ]
    for path, base in (
        ("/v1/rerank", {"query": query, "documents": docs}),
        ("/v1/score", {"text_1": query, "text_2": docs}),
    ):
        plain, _ = _post(f"{url}{path}", key, {"model": model, **base}, timeout)
        instr, _ = _post(
            f"{url}{path}",
            key,
            {"model": model, "instruction": DEFAULT_INSTRUCTION + " about lobes", **base},
            timeout,
        )
        pk = "results" if path == "/v1/rerank" else "data"
        sk = "relevance_score" if path == "/v1/rerank" else "score"
        a = sorted((r["index"], round(r[sk], 4)) for r in plain[pk])
        b = sorted((r["index"], round(r[sk], 4)) for r in instr[pk])
        ta = (plain.get("usage") or {}).get("prompt_tokens")
        tb = (instr.get("usage") or {}).get("prompt_tokens")
        print(f"  {path}: without={a} tokens={ta}")
        print(f"  {path}: with   ={b} tokens={tb}")
        print(f"  {path}: instruction changes scores: {a != b}")


def latency_probe(url: str, key: str | None, model: str, timeout: float, n: int) -> None:
    print(f"\n## latency — 1 query x {len(LATENCY_DOCS)} docs, median of {n} (1 warm-up discarded)")
    body = {"model": model, "query": LATENCY_QUERY, "documents": LATENCY_DOCS}
    _post(f"{url}/v1/rerank", key, body, timeout)
    samples = [_post(f"{url}/v1/rerank", key, body, timeout)[1] for _ in range(n)]
    print(f"  samples_ms={[round(s, 1) for s in samples]}")
    print(f"  median_ms={statistics.median(samples):.1f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument(
        "--key", default=os.environ.get("GATEWAY_API_KEY") or _read_env_key("GATEWAY_API_KEY")
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--latency-samples", type=int, default=5)
    a = ap.parse_args(argv)
    if a.latency_samples < 1:
        ap.error("--latency-samples must be >= 1")
    url = a.url.rstrip("/")
    try:
        header(url, a.model)
        probe(url, a.key, a.model, a.timeout)
        instruction_probe(url, a.key, a.model, a.timeout)
        latency_probe(url, a.key, a.model, a.timeout, a.latency_samples)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} from {e.url}: {e.read()[:300]!r}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, KeyError, ValueError) as e:
        print(f"probe failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
