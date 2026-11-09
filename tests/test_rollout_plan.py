"""Tests for lobes.rollout._plan (pure: plain data in, plan out)."""

from __future__ import annotations

import json
import socket
import subprocess

import pytest

from lobes.rollout import _plan as pl
from lobes.rollout._drift import DriftReport
from lobes.rollout._observe import MemberObservation
from lobes.rollout._residents import Residents
from lobes.rollout._versions import DowngradeRefused


def obs(version="1.0.0", roles=(), reachable=True, proxied=()):
    caps = {r: {"feasible": True, "ready": True} for r in roles}
    caps.update({r: {"feasible": True, "ready": True, "proxied": True} for r in proxied})
    return MemberObservation(
        origin="http://x",
        version=version if reachable else None,
        capabilities=caps,
        reachable=reachable,
    )


def build(members, observations, **kw):
    res = Residents(residents={m: {} for m in members}, guests=kw.pop("guests", {}))
    return pl.build_plan(
        res,
        observations,
        kw.pop("drift", {}),
        kw.pop("target", "2.0.0"),
        kw.pop("local", "zed"),
        **kw,
    )


BASIC = {
    "spark": obs(roles=["cortex", "worker", "hand"]),
    "thor": obs(roles=["reranker", "hand"]),
    "orin": obs(roles=["reranker"]),
    "zed": obs(roles=[]),
}


def test_one_member_per_wave_canary_first_local_last():
    p = build(["spark", "thor", "orin", "zed"], BASIC)
    names = [w.members for w in p.waves]
    assert all(len(w) == 1 for w in names)
    # orin: reranker is shared with thor -> 0 unique roles -> canary
    assert names[0] == ["orin"]
    assert names[-1] == ["zed"]
    assert p.canary == "orin"


def test_roles_hosted_elsewhere_do_not_count():
    p = build(["thor", "orin", "zed"], {**BASIC, "spark": obs(roles=["hand"])}, canary=None)
    assert p.canary in {"thor", "orin"}
    assert p.member("thor").unique_roles == []


def test_canary_override_and_local_override_still_last():
    p = build(["spark", "thor", "zed"], BASIC, canary="spark")
    assert p.waves[0].members == ["spark"]
    p = build(["spark", "thor", "zed"], BASIC, canary="zed")
    assert p.waves[-1].members == ["zed"]


def test_bad_canary_rejected():
    with pytest.raises(ValueError):
        build(["spark", "zed"], BASIC, canary="nobody")


def test_wave_size():
    p = build(["spark", "thor", "orin", "zed"], BASIC, wave_size=2)
    assert [len(w.members) for w in p.waves] == [2, 2]
    assert p.waves[-1].members[-1] == "zed"


def test_guests_and_current_skipped():
    o = {**BASIC, "thor": obs(version="2.0.0", roles=["hand"])}
    p = build(["spark", "thor", "zed"], o, guests={"gus": {}})
    assert p.member("gus").status == pl.SKIPPED_GUEST
    assert p.member("thor").status == pl.SKIPPED_CURRENT
    planned = [m for w in p.waves for m in w.members]
    assert "gus" not in planned and "thor" not in planned


def test_downgrade_refused_unless_allowed():
    o = {**BASIC, "spark": obs(version="3.0.0", roles=["cortex"])}
    p = build(["spark", "zed"], o)
    assert p.member("spark").status == pl.REFUSED_DOWNGRADE
    assert "spark" not in [m for w in p.waves for m in w.members]
    p = build(["spark", "zed"], o, allow_downgrade=True)
    assert p.member("spark").status == pl.PLANNED


def test_unreachable_is_skipped_offline_pending():
    o = {**BASIC, "thor": obs(reachable=False)}
    p = build(["spark", "thor", "zed"], o)
    m = p.member("thor")
    assert m.status == pl.SKIPPED_OFFLINE and m.pending
    assert not any("thor" in w.members for w in p.waves)
    # missing observation counts as offline too
    p = build(["spark", "ghost", "zed"], BASIC)
    assert p.member("ghost").status == pl.SKIPPED_OFFLINE


def test_unreachable_not_counted_as_hosting():
    o = {**BASIC, "thor": obs(reachable=False)}
    p = build(["thor", "orin", "zed"], o)
    assert p.member("orin").unique_roles == ["reranker"]


def test_drift_attached_per_member_and_serializable():
    d = {"spark": DriftReport(missing=["A"], lock_affected=True)}
    p = build(["spark", "zed"], BASIC, drift=d)
    assert p.member("spark").drift.missing == ["A"]
    assert p.member("spark").drift.lock_affected is True
    assert p.member("zed").drift is None
    data = json.loads(json.dumps(p.to_dict()))
    assert data["target"] == "2.0.0" and data["members"][0]["name"]


def test_local_alone_still_planned_and_no_io(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("io attempted")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr("builtins.open", boom)
    monkeypatch.setattr("pathlib.Path.write_text", boom)
    monkeypatch.setattr("pathlib.Path.read_text", boom)
    p = build(["zed"], BASIC)
    assert [w.members for w in p.waves] == [["zed"]]
    p.to_dict()


def test_downgrade_error_type_not_raised():
    # refusal is data, not an exception
    assert issubclass(DowngradeRefused, ValueError)
    build(["spark", "zed"], {**BASIC, "spark": obs(version="9.0.0")})
