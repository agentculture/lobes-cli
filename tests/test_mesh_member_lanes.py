"""Tests for the member-lane resolver (mesh-pool-load-sharing, t1).

``member_lanes`` / ``find_member_lane`` expose EVERY verified member of a role
(agreeing and disagreeing alike) plus the self lane, under ``{role}-{member}``.
"""

from __future__ import annotations

from lobes.gateway._mesh_routing import (
    MemberInfo,
    MemberLane,
    RoutingSnapshot,
    find_member_lane,
    find_suffixed_lane,
    member_lanes,
)
from tests.test_mesh_naming import _fp, _two_member_snapshot


def _info(name, *, announced=(), verified=(), probed=True):
    return MemberInfo(
        name=name,
        origin=f"http://{name}",
        announced_roles=tuple(announced),
        verified_roles=tuple(verified),
        capacity=1.0,
        probed=probed,
    )


def _snap(*members):
    return RoutingSnapshot(members=tuple(members), announcements=())


def test_agreeing_and_disagreeing_members_both_get_lanes():
    snap = _two_member_snapshot(_fp(quantization="NVFP4"), _fp(quantization="FP8"))
    lanes = member_lanes(snap, "cortex")
    assert lanes == (
        MemberLane("cortex-nameA", "cortex", "nameA", "http://a"),
        MemberLane("cortex-nameB", "cortex", "nameB", "http://b"),
    )
    agreeing = _two_member_snapshot(_fp(), _fp())
    assert [lane.name for lane in member_lanes(agreeing, "cortex")] == [
        "cortex-nameA",
        "cortex-nameB",
    ]


def test_self_lane_first_then_peers_sorted():
    snap = _snap(
        _info("zed", announced=("cortex",), verified=("cortex",)),
        _info("alpha", announced=("cortex",), verified=("cortex",)),
    )
    lanes = member_lanes(snap, "cortex", self_name="mid", self_hosts=True)
    assert [lane.name for lane in lanes] == ["cortex-mid", "cortex-alpha", "cortex-zed"]
    assert lanes[0] == MemberLane("cortex-mid", "cortex", "mid", "", is_self=True)
    assert not lanes[1].is_self


def test_self_not_hosting_has_no_self_lane():
    snap = _snap(_info("alpha", announced=("cortex",), verified=("cortex",)))
    lanes = member_lanes(snap, "cortex", self_name="mid", self_hosts=False)
    assert [lane.name for lane in lanes] == ["cortex-alpha"]


def test_peer_named_self_is_skipped():
    snap = _snap(_info("mid", announced=("cortex",), verified=("cortex",)))
    lanes = member_lanes(snap, "cortex", self_name="mid", self_hosts=True)
    assert len(lanes) == 1
    assert lanes[0].is_self


def test_pending_member_is_flagged():
    snap = _snap(
        _info("alpha", announced=("cortex",), verified=("cortex",)),
        _info("boot", announced=("cortex",), probed=False),
        _info("probed-nothing", announced=("cortex",), probed=True),
    )
    lanes = {lane.member: lane for lane in member_lanes(snap, "cortex")}
    assert set(lanes) == {"alpha", "boot"}
    assert lanes["boot"].pending is True
    assert lanes["alpha"].pending is False


def test_snapshot_none_returns_only_self():
    assert member_lanes(None, "cortex") == ()
    lanes = member_lanes(None, "cortex", self_name="mid", self_hosts=True)
    assert [lane.name for lane in lanes] == ["cortex-mid"]


def test_unforwardable_role_gets_no_peer_lanes_but_self_lane():
    snap = _snap(_info("alpha", announced=("innereye",), verified=("innereye",)))
    assert member_lanes(snap, "innereye") == ()
    lanes = member_lanes(snap, "innereye", self_name="mid", self_hosts=True)
    assert [lane.name for lane in lanes] == ["innereye-mid"]
    assert find_member_lane(snap, "innereye-alpha", ("innereye",)) is None


def test_find_resolves_agreeing_peer():
    snap = _two_member_snapshot(_fp(), _fp())
    lane = find_member_lane(snap, "cortex-nameA", ("cortex", "senses"))
    assert lane == MemberLane("cortex-nameA", "cortex", "nameA", "http://a")


def test_find_disagreeing_peer_matches_find_suffixed_lane():
    snap = _two_member_snapshot(_fp(quantization="NVFP4"), _fp(quantization="FP8"))
    old = find_suffixed_lane(snap, "cortex-nameB", ("cortex",))
    new = find_member_lane(snap, "cortex-nameB", ("cortex",))
    assert old is not None
    assert new is not None
    assert (new.name, new.role, new.member, new.origin) == (
        old.name,
        old.role,
        old.member,
        old.origin,
    )


def test_find_self_lane_only_when_self_hosts():
    snap = _snap(_info("alpha", announced=("cortex",), verified=("cortex",)))
    hit = find_member_lane(
        snap, "cortex-mid", ("cortex",), self_name="mid", hosted_roles={"cortex"}
    )
    assert hit is not None
    assert hit.is_self
    assert hit.origin == ""
    assert (
        find_member_lane(snap, "cortex-mid", ("cortex",), self_name="mid", hosted_roles=set())
        is None
    )


def test_find_unknown_member_and_absent_role_return_none():
    snap = _snap(_info("alpha", announced=("senses",), verified=("senses",)))
    assert find_member_lane(snap, "cortex-nobody", ("cortex",)) is None
    # alpha does not carry cortex (e.g. announced private -> stripped upstream)
    assert find_member_lane(snap, "cortex-alpha", ("cortex",)) is None
    assert find_member_lane(None, "cortex-alpha", ("cortex",)) is None


def test_hand_member_resolves_like_cortex():
    snap = _snap(_info("alpha", announced=("hand",), verified=("hand",)))
    lane = find_member_lane(snap, "hand-alpha", ("hand", "cortex"))
    assert lane == MemberLane("hand-alpha", "hand", "alpha", "http://alpha")
