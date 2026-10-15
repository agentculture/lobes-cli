"""t23: the Clef probe comparator — accuracy, agreement, KL and the 97% promotion gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "clef_eval" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cmp = _mod("compare")
rp = _mod("run_probe")


def _row(rid, p_true, gold="true", source="boolq", ms=100.0):
    return {
        "id": rid,
        "source": source,
        "latency_ms": ms,
        "gold": {"a": gold},
        "probs": {"a": {"true": p_true, "false": 1 - p_true}},
    }


def test_probabilities_cover_every_answer_type():
    assert rp.probabilities({"type": "noul", "noul": 0.8}) == {
        "true": 0.8,
        "false": 0.19999999999999996,
    }
    assert rp.probabilities(
        {"type": "choice", "choice": "x", "probabilities": {"x": 0.7, "y": 0.3}}
    ) == {"x": 0.7, "y": 0.3}
    assert (
        rp.probabilities(
            {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
        )["1"]
        == 0.6
    )


def test_identical_run_agrees_fully_and_is_promotable():
    ref = {f"r{i}": _row(f"r{i}", 0.9) for i in range(100)}
    out = cmp.score(ref, ref)
    assert out["top1_agreement_vs_reference"] == 1.0
    assert out["kl_max"] < 1e-6
    assert out["promotable"] is True


def test_below_threshold_is_not_promotable():
    ref = {f"r{i}": _row(f"r{i}", 0.9) for i in range(100)}
    run = {k: dict(v) for k, v in ref.items()}
    for i in range(4):  # 96% agreement < 97%
        run[f"r{i}"] = _row(f"r{i}", 0.2)
    out = cmp.score(run, ref)
    assert out["top1_agreement_vs_reference"] == 0.96
    assert out["promotable"] is False


def test_exactly_at_threshold_is_promotable():
    ref = {f"r{i}": _row(f"r{i}", 0.9) for i in range(100)}
    run = {k: dict(v) for k, v in ref.items()}
    for i in range(3):
        run[f"r{i}"] = _row(f"r{i}", 0.2)
    assert cmp.score(run, ref)["promotable"] is True


def test_a_failed_record_blocks_promotion():
    ref = {f"r{i}": _row(f"r{i}", 0.9) for i in range(100)}
    run = {k: dict(v) for k, v in ref.items()}
    run["r0"] = {"id": "r0", "source": "boolq", "error": "timeout"}
    out = cmp.score(run, ref)
    assert out["failed_records"] == 1
    assert out["promotable"] is False


def test_accuracy_is_against_gold_not_reference():
    ref = {"r0": _row("r0", 0.9, gold="false")}
    out = cmp.score(ref, ref)
    assert out["top1_accuracy"] == 0.0
    assert out["top1_agreement_vs_reference"] == 1.0


def test_main_reads_jsonl(tmp_path, capsys):
    ref = tmp_path / "ref.jsonl"
    ref.write_text("".join(json.dumps(_row(f"r{i}", 0.9)) + "\n" for i in range(5)))
    import sys

    argv = sys.argv
    sys.argv = ["compare.py", "--reference", str(ref), str(ref)]
    try:
        assert cmp.main() == 0
    finally:
        sys.argv = argv
    report = json.loads(capsys.readouterr().out)
    assert report[str(ref)]["promotable"] is True
