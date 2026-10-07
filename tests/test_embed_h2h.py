"""Offline tests for scripts/embed_h2h: scorer math, prompts, corpus integrity."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

H = Path(__file__).resolve().parent.parent / "scripts" / "embed_h2h"
_spec = importlib.util.spec_from_file_location("embed_h2h_h2h", H / "h2h.py")
h2h = importlib.util.module_from_spec(_spec)
sys.modules["embed_h2h_h2h"] = h2h
_spec.loader.exec_module(h2h)


def test_ndcg_perfect_and_zero():
    assert h2h.ndcg_at_k(["a", "b"], {"a": 2}) == 1.0
    assert h2h.ndcg_at_k(["x", "y"], {"a": 2}) == 0.0


def test_ndcg_hand_computed_graded():
    # relevant: a=2, b=1. ranked: b, x, a. DCG = 1/log2(2) + 0 + 2/log2(4) = 1 + 1 = 2.
    # ideal: a, b -> 2/1 + 1/log2(3).
    expected = 2.0 / (2.0 + 1.0 / math.log2(3))
    assert math.isclose(h2h.ndcg_at_k(["b", "x", "a"], {"a": 2, "b": 1}), expected)


def test_ndcg_cutoff_at_k():
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    assert h2h.ndcg_at_k(ranked, {"a": 1}, k=10) == 0.0
    assert h2h.ndcg_at_k(ranked, {"a": 1}, k=11) > 0.0


def test_cosine():
    assert math.isclose(h2h.cosine([1, 0], [1, 0]), 1.0)
    assert math.isclose(h2h.cosine([1, 0], [0, 1]), 0.0, abs_tol=1e-12)
    assert h2h.cosine([0, 0], [1, 1]) == 0.0


def test_rank_orders_by_cosine():
    assert h2h.rank([1, 0], [[0, 1], [1, 0.1], [1, 1]], ["a", "b", "c"]) == ["b", "c", "a"]


PROMPTS = {
    "tasks": {"nl_to_code": "TASK-NL", "issue_to_source": "TASK-IS"},
    "models": {
        "m1": {
            "query": {"nl_to_code": "Q1: {q}", "issue_to_source": "Q1I: {q}"},
            "document": "T: {filename} | {code}",
        },
        "m2": {
            "query": {"nl_to_code": "Instruct: {task}\nQuery:{q}", "issue_to_source": None},
            "document": None,
        },
    },
}
CORPUS = [
    {"id": "d1", "path": "a.py", "text": "alpha"},
    {"id": "d2", "path": "b.py", "text": "beta"},
]
QUERIES = [
    {"family": "nl_to_code", "query": "alpha", "relevant": {"d1": 2}},
    {"family": "issue_to_source", "query": "beta", "relevant": {"d2": 2}},
]


def fake_embedder(calls):
    def embed(model, texts):
        calls.append((model, list(texts)))
        out = []
        for t in texts:
            out.append([1.0, 0.0] if "alpha" in t else [0.0, 1.0])
        return out

    return embed


def test_each_model_gets_its_own_prompts():
    calls: list = []
    emb = fake_embedder(calls)
    h2h.score_model(emb, "m1", PROMPTS, CORPUS, QUERIES)
    h2h.score_model(emb, "m2", PROMPTS, CORPUS, QUERIES)
    m1 = [t for m, ts in calls if m == "m1" for t in ts]
    m2 = [t for m, ts in calls if m == "m2" for t in ts]
    assert "T: a.py | alpha" in m1 and "Q1: alpha" in m1 and "Q1I: beta" in m1
    assert "Instruct: TASK-NL\nQuery:alpha" in m2
    assert "beta" in m2 and "alpha" in m2  # null templates -> raw text
    assert not any(t.startswith("Q1") or t.startswith("T:") for t in m2)


def test_score_model_rows():
    rows = h2h.score_model(fake_embedder([]), "m1", PROMPTS, CORPUS, QUERIES)
    assert {r["family"] for r in rows} == {"nl_to_code", "issue_to_source"}
    assert all(r["ndcg_at_10"] == 1.0 and r["n"] == 1 and r["model"] == "m1" for r in rows)
    assert "| m1 | nl_to_code | 1.0000 | 1 |" in h2h.render_table(rows)


def _rows(cand, base):
    out = []
    for model, vals in ((h2h.SPECIALIST_MODEL, cand), (h2h.BASELINE_MODEL, base)):
        for fam, v in zip(h2h.FAMILIES, vals):
            out.append({"model": model, "family": fam, "ndcg_at_10": v, "n": 1})
    return out


def test_threshold_is_five_points_and_decision_rule():
    assert h2h.THRESHOLD_POINTS == 5.0
    assert h2h.decide(_rows((0.70, 0.60), (0.64, 0.54)))["candidate_takes_code_slot"]
    # one family short of 5 points -> baseline keeps the slot
    assert not h2h.decide(_rows((0.70, 0.57), (0.64, 0.54)))["candidate_takes_code_slot"]


def test_real_prompts_cover_all_candidates_with_provenance():
    prompts = h2h.load_prompts()
    for model in (
        "google/embeddinggemma-2",
        "Qwen/Qwen3-VL-Embedding-8B",
        "nvidia/Nemotron-3-Embed-8B-BF16",
        "nomic-ai/nomic-embed-code",
        "Qwen/Qwen3-Embedding-0.6B",
    ):
        entry = prompts["models"][model]
        assert "source" in entry and "verified" in entry
        for fam in h2h.FAMILIES:
            assert fam in entry["query"]
    q = h2h.render_query(prompts, "google/embeddinggemma-2", "issue_to_source", "boom")
    assert q == "task: code retrieval | query: boom"


def test_corpus_and_queries_integrity():
    corpus = h2h.load_jsonl(H / "corpus.jsonl")
    queries = h2h.load_jsonl(H / "queries.jsonl")
    ids = {c["id"] for c in corpus}
    assert len(ids) == len(corpus)
    assert {c["repo"] for c in corpus} == {"lobes-cli", "culture"}
    assert all(len(c["sha"]) == 40 and c["start"] <= c["end"] for c in corpus)
    fam = [q["family"] for q in queries]
    assert fam.count("nl_to_code") >= 40 and fam.count("issue_to_source") >= 20
    for q in queries:
        assert q["relevant"] and set(q["relevant"].values()) <= {1, 2}
        assert set(q["relevant"]) <= ids
    assert (H / "corpus.jsonl").stat().st_size < 5_000_000
