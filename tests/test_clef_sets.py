"""t18: the Clef calibration set and the labelled probe set are disjoint and cited.

Runs offline over the committed manifests (scripts/clef_sets/*.manifest.jsonl);
third-party rows are referenced by pinned revision, mesh-authored rows rebuild.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS_DIR = ROOT / "scripts" / "clef_sets"


def _load():
    sys.path.insert(0, str(SETS_DIR))
    try:
        spec = importlib.util.spec_from_file_location("clef_build_sets", SETS_DIR / "build_sets.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(SETS_DIR))


bs = _load()


def test_manifests_pass_every_invariant():
    assert bs.check() == []


def test_sets_are_disjoint_by_id_and_by_content():
    cal, probe = bs.read_manifest("calibration"), bs.read_manifest("probe")
    assert not {e["id"] for e in cal} & {e["id"] for e in probe}
    assert not {e["sha256"] for e in cal} & {e["sha256"] for e in probe}


def test_minimum_sizes():
    assert len(bs.read_manifest("calibration")) >= 256
    assert len(bs.read_manifest("probe")) >= 100


def test_every_source_is_pinned_and_licensed():
    for name, spec in bs.SOURCES.items():
        assert len(spec["revision"]) == 40, name
        assert spec["license"], name
    for name, spec in bs.MESH_SOURCES.items():
        assert spec["license"], name


def test_calibration_never_draws_from_an_eval_split():
    for e in bs.read_manifest("calibration"):
        if e["source"] in bs.SOURCES:
            assert bs.SOURCES[e["source"]]["files"]["calibration"] in e["id"]


def test_question_types_are_mixed_in_both_sets():
    # choice (snli/mcq/...), noul (boolq, incidents), score (incidents)
    for set_name in bs.SETS:
        sources = {e["source"] for e in bs.read_manifest(set_name)}
        assert {"boolq", "snli", "incident"} <= sources


def test_incident_gold_is_consistent_with_its_log():
    record, gold = bs.incident_record(1_000_007)
    state = record["state"]
    down = {
        svc
        for svc in bs.incident_record.__globals__["SERVICES"]
        if f"[{svc}] " in state
        and any(f"[{svc}] {t}" in state for t in bs.incident_record.__globals__["FAILURE_TEXT"])
    }
    assert gold["failing"] == str(min(len(down), 4))
    assert gold["database_down"] == ("true" if "postgres" in down else "false")
    if down:
        assert gold["first_failure"] in down


def test_incidents_are_deterministic():
    assert bs.incident_record(1_000_001) == bs.incident_record(1_000_001)
