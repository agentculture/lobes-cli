"""Rollout journal: append-only JSONL, status projection, resume."""

import json

from lobes.rollout._journal import STEPS, Journal


def test_append_only_jsonl_under_deployment_dir(tmp_path):
    j = Journal(tmp_path)
    j.append("spark", "lock", "ok", target="1.0.0", current="0.9.0")
    j.append("spark", "window_check", "ok")
    lines = j.path.read_text().splitlines()
    assert j.path.parent == tmp_path
    assert len(lines) == 2
    assert json.loads(lines[0])["member"] == "spark"
    first = lines[0]
    j.append("spark", "backup_env", "ok")
    assert j.path.read_text().splitlines()[0] == first


def test_status_projection(tmp_path):
    j = Journal(tmp_path)
    j.append("a", "lock", "ok", target="2", current="1")
    j.append("a", "install_cli", "failed")
    st = j.status()["a"]
    assert (st.target, st.current, st.last_step, st.last_result) == (
        "2",
        "1",
        "install_cli",
        "failed",
    )
    assert st.done is False


def test_done_only_after_verified_record(tmp_path):
    j = Journal(tmp_path)
    for s in STEPS[:-1]:
        j.append("a", s, "ok", target="2")
    assert j.status()["a"].done is False
    j.append("a", "verify", "failed")
    assert j.status()["a"].done is False
    j.append("a", "verify", "ok")
    assert j.status()["a"].done is True
    assert j.resume_point() is None


def test_resume_point_names_member_and_step(tmp_path):
    j = Journal(tmp_path)
    for s in STEPS:
        j.append("done", s, "ok")
    for s in ("lock", "window_check", "backup_env"):
        j.append("b", s, "ok")
    rp = j.resume_point()
    assert (rp.member, rp.step) == ("b", "install_cli")
    j.append("b", "install_cli", "failed")
    assert j.resume_point().step == "install_cli"


def test_offline_members_pending(tmp_path):
    j = Journal(tmp_path)
    j.append("off", "lock", "skipped_offline", target="2")
    assert j.status()["off"].state == "pending"
    assert j.pending_members() == ["off"]
    assert j.resume_point() is None


def test_redact_applied_and_missing_file(tmp_path):
    j = Journal(tmp_path, redact=lambda s: s.replace("SECRET", "***"))
    assert j.records() == [] and j.status() == {}
    j.append("a", "lock", "ok", detail={"msg": "key=SECRET", "n": ["SECRET"]})
    assert "SECRET" not in j.path.read_text()
    assert j.records()[0]["detail"]["n"] == ["***"]
