"""Code-retrieval head-to-head harness: nDCG@10 over a labelled corpus.

HTTP-only against an OpenAI-compatible ``/v1/embeddings`` endpoint (the lobes
gateway). Scores whatever model ids it is given, each with its own prompts from
``prompts.json``. stdlib only; no mesh or network dependency beyond ``--url``.

    python h2h.py --url http://localhost:8000 --models google/embeddinggemma-2 ...

Decision rule (committed BEFORE any results, h18): the 8B code specialist wins
the slot only if it beats EmbeddingGemma 2 by >= THRESHOLD_POINTS nDCG@10
points on BOTH families; otherwise EmbeddingGemma 2 takes the code slot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

HERE = Path(__file__).resolve().parent

THRESHOLD_POINTS = 5.0  # nDCG@10 points (nDCG * 100); fixed before results exist
K = 10
FAMILIES = ("nl_to_code", "issue_to_source")
BASELINE_MODEL = "google/embeddinggemma-2"
SPECIALIST_MODEL = "Qwen/Qwen3-VL-Embedding-8B"

Embedder = Callable[[str, list[str]], list[list[float]]]


# --- data ------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_prompts(path: Path = HERE / "prompts.json") -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --- prompts ---------------------------------------------------------------


def render_query(prompts: dict, model: str, family: str, query: str) -> str:
    template = prompts["models"][model]["query"][family]
    if template is None:
        return query
    return template.format(q=query, task=prompts["tasks"][family])


def render_document(prompts: dict, model: str, chunk: dict) -> str:
    template = prompts["models"][model]["document"]
    if template is None:
        return chunk["text"]
    return template.format(filename=chunk["path"], code=chunk["text"])


# --- math ------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def dcg(gains: Iterable[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int = K) -> float:
    """nDCG@k with linear gain = graded relevance (2 primary, 1 related)."""
    ideal = sorted(relevance.values(), reverse=True)[:k]
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(relevance.get(i, 0) for i in ranked_ids[:k]) / denom


def rank(query_vec: list[float], doc_vecs: list[list[float]], doc_ids: list[str]) -> list[str]:
    scored = sorted(
        ((cosine(query_vec, v), i) for i, v in zip(doc_ids, doc_vecs)),
        key=lambda t: (-t[0], t[1]),  # deterministic tie-break on id
    )
    return [i for _, i in scored]


def score_model(
    embed: Embedder,
    model: str,
    prompts: dict,
    corpus: list[dict],
    queries: list[dict],
    batch: int = 32,
) -> list[dict]:
    """Rows of {model, family, ndcg_at_10, n} (one per family) plus per-query detail."""
    doc_ids = [c["id"] for c in corpus]
    doc_texts = [render_document(prompts, model, c) for c in corpus]
    doc_vecs: list[list[float]] = []
    for i in range(0, len(doc_texts), batch):
        doc_vecs.extend(embed(model, doc_texts[i : i + batch]))
    per_family: dict[str, list[float]] = {}
    for q in queries:
        text = render_query(prompts, model, q["family"], q["query"])
        qv = embed(model, [text])[0]
        score = ndcg_at_k(rank(qv, doc_vecs, doc_ids), q["relevant"])
        per_family.setdefault(q["family"], []).append(score)
    return [
        {"model": model, "family": fam, "ndcg_at_10": sum(s) / len(s), "n": len(s)}
        for fam, s in sorted(per_family.items())
    ]


def decide(
    rows: list[dict], candidate: str = SPECIALIST_MODEL, baseline: str = BASELINE_MODEL
) -> dict:
    """Apply the committed rule. Margins are in nDCG@10 points (x100)."""
    by = {(r["model"], r["family"]): r["ndcg_at_10"] * 100 for r in rows}
    margins = {f: by[(candidate, f)] - by[(baseline, f)] for f in FAMILIES}
    wins = all(m >= THRESHOLD_POINTS for m in margins.values())
    return {
        "candidate": candidate,
        "baseline": baseline,
        "threshold_points": THRESHOLD_POINTS,
        "margins_points": margins,
        "candidate_takes_code_slot": wins,
        "outcome": (
            f"{candidate} takes the code slot"
            if wins
            else f"{baseline} takes the code slot; the 8B slot goes to the next-best candidate"
        ),
    }


def render_table(rows: list[dict]) -> str:
    lines = ["| model | family | nDCG@10 | n |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['family']} | {r['ndcg_at_10']:.4f} | {r['n']} |")
    return "\n".join(lines)


# --- HTTP ------------------------------------------------------------------


def http_embedder(base_url: str, api_key: str | None = None, timeout: float = 600.0) -> Embedder:
    url = base_url.rstrip("/") + "/v1/embeddings"

    def embed(model: str, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": model, "input": texts}).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            data = json.load(resp)["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    return embed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", required=True, help="gateway base URL, e.g. http://localhost:8000")
    p.add_argument("--models", nargs="+", help="model ids (default: every id in prompts.json)")
    p.add_argument("--api-key-env", default="GATEWAY_API_KEY")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--out-prefix", help="write <prefix>.md and <prefix>.json (default: stdout only)"
    )
    args = p.parse_args(argv)

    prompts = load_prompts()
    corpus = load_jsonl(HERE / "corpus.jsonl")
    queries = load_jsonl(HERE / "queries.jsonl")
    models = args.models or list(prompts["models"])
    unknown = [m for m in models if m not in prompts["models"]]
    if unknown:
        print(f"no prompts for: {unknown}", file=sys.stderr)
        return 2
    embed = http_embedder(args.url, os.environ.get(args.api_key_env))
    rows: list[dict] = []
    for m in models:
        rows.extend(score_model(embed, m, prompts, corpus, queries, args.batch))
    table = render_table(rows)
    print(table)
    if args.out_prefix:
        Path(args.out_prefix + ".md").write_text(table + "\n", encoding="utf-8")
        Path(args.out_prefix + ".json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
