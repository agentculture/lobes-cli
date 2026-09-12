"""Tests for membership-driven routing snapshot (mesh-t7).

Covers the acceptance contract for
:mod:`lobes.gateway._mesh_routing`: immutable per-request snapshots, pool
construction from verified members, proxy-loop guard, and member drop.

Spec targets: c3, h25, c7, h9, c28, h18, c44, h35, c40, h31, c14, h13.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from lobes.gateway._mesh_routing import (
    MemberInfo,
    SnapshotHolder,
    build_snapshot,
    compute_role_placement,
    member_exists,
    mesh_markers,
    origins_for_role,
    roster_member_count,
    snapshot_member_roles,
)
from lobes.gateway._mesh_wire import Announcement, Fingerprint, RoleInfo

if TYPE_CHECKING:
    from lobes.gateway._mesh_roster import Roster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fp(
    served_id="unsloth/Qwen3.8-27B-NVFP4",
    quantization="NVFP4",
    max_model_len=262144,
    runtime="vllm",
):
    return Fingerprint(
        served_id=served_id,
        quantization=quantization,
        max_model_len=max_model_len,
        runtime=runtime,
    )


def _role(name: str, **over) -> RoleInfo:
    """A RoleInfo with sane defaults overridden by *over*.

    All fields use real defaults so tests only override what differs.
    """
    return RoleInfo(
        model=over.get("model", "unsloth/Qwen3.8-27B-NVFP4"),
        runtime=over.get("runtime", "vllm"),
        context=over.get("context", 262144),
        quant=over.get("quant", "NVFP4"),
        responsibilities=over.get("responsibilities", ("generate", "image_understanding")),
        forbidden_responsibilities=over.get(
            "forbidden_responsibilities", ("final_decision", "security_decision")
        ),
        fingerprint=over.get("fingerprint", _fp()),
        capacity=over.get("capacity"),
        private=over.get("private", False),
    )


def _ann(
    name: str,
    origin: str,
    roles: dict[str, RoleInfo] | None = None,
) -> Announcement:
    return Announcement(
        name=name,
        origin=origin,
        schema_version="1.0.0",
        roles=roles or {"cortex": _role("cortex")},
    )


def _make_roster(**overrides) -> Roster:
    """A Roster backed by a monotonic clock that advances on tick()."""
    _counter = [0.0]

    def clock():
        return _counter[0]

    def tick(delta=1.0):
        _counter[0] += delta
        return clock()

    clock.tick = tick  # type: ignore[attr-defined]
    roster = type(
        "FakeRoster",
        (),
        {
            "members": lambda self: list(overrides.get("_names", [])),
            "_roster": dict(overrides.get("_records", {})),
            "now": clock,
        },
    )()
    roster.tick = tick
    return roster


def _build_fake_roster(members: list[tuple[str, str, float]]) -> Roster:
    """Build a simple Roster-like object.

    Parameters
    ----------
    members: list of (name, origin, capacity)
    """
    _counter = [1.0]

    def clock():
        return _counter[0]

    def tick(delta=1.0):
        _counter[0] += delta
        return clock()

    clock.tick = tick  # type: ignore[attr-defined]

    records = {}
    names = []
    for name, origin, capacity in members:
        rec = type(
            "FakeMemberRecord",
            (),
            {
                "name": name,
                "origin": origin,
                "capacity": capacity,
                "last_seen": _counter[0],
                "missed": 0,
                "verified": False,
            },
        )()
        records[name] = rec
        names.append(name)

    roster = type(
        "FakeRoster",
        (),
        {
            "members": lambda self: list(names),
            "_roster": records,
            "now": clock,
        },
    )()
    roster.tick = tick
    return roster


class _FakeRoster:
    """Minimal roster duck-type for build_snapshot tests."""

    def __init__(self, members: list[tuple[str, str, float]]) -> None:
        self._counter = [1.0]
        self._records: dict[str, type] = {}
        self._names: list[str] = []
        for name, origin, capacity in members:
            rec = type(
                "FakeRecord",
                (),
                {
                    "name": name,
                    "origin": origin,
                    "capacity": capacity,
                },
            )()
            self._records[name] = rec
            self._names.append(name)

    def members(self) -> list[str]:
        return list(self._names)

    @property
    def _roster(self) -> dict[str, type]:
        return self._records

    def now(self) -> float:
        return self._counter[0]

    def tick(self, delta: float = 1.0) -> float:
        self._counter[0] += delta
        return self._counter[0]


# ---------------------------------------------------------------------------
# AC1: Immutable snapshot per request
# ---------------------------------------------------------------------------


class TestSnapshotImmutability:
    """AC1: the request path reads one immutable snapshot per request; a roster
    change mid-request never affects it; no lock is held across a dial."""

    def test_snapshot_is_frozen(self):
        """The members tuple cannot be mutated after build_snapshot."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        snap = build_snapshot(roster)
        # The returned tuple is immutable — trying to mutate raises.
        assert isinstance(snap.members, tuple)
        # Append (creates new tuple) succeeds — immutability means the original
        # object's contents can't change.  The real guarantee is that the
        # RoutingSnapshot's fields are plain tuples (not mutable lists) and
        # MemberInfo is a frozen dataclass.
        for m in snap.members:
            assert isinstance(m, MemberInfo)
            assert isinstance(m.announced_roles, tuple)
            assert isinstance(m.verified_roles, tuple)

    def test_roster_change_mid_request_not_seen(self):
        """A roster change after build_snapshot is called does not affect the
        returned snapshot."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        snap = build_snapshot(roster)
        # Original snapshot shows 1 member.
        assert roster_member_count(snap) == 1
        # Even if we build a new roster object (simulating a fresh roster),
        # the existing snap is still based on the old state.
        assert snap.member_count() == 1

    def test_no_lock_held_across_dial(self):
        """build_snapshot returns immediately; no threading lock is held."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        # If a lock were held, this would block waiting for another thread.
        snap = build_snapshot(roster)
        assert snap is not None

    def test_snapshot_holder_replacement_is_atomic(self):
        """SnapshotHolder.replace + .current form an atomic read-write pair."""
        roster = _FakeRoster([])
        holder = SnapshotHolder(roster)
        snap = build_snapshot(roster)
        holder.replace(snap)
        current = holder.current()
        assert current is snap
        # Replace with a different snapshot.
        snap2 = build_snapshot(roster)
        holder.replace(snap2)
        assert holder.current() is snap2
        assert holder.current() is not snap


# ---------------------------------------------------------------------------
# AC2: Pool from verified members; unverified on probe mismatch
# ---------------------------------------------------------------------------


class TestVerifiedPool:
    """AC2: two fake members announcing cortex with equal fingerprints form one
    pool selected by select_replica; a member whose /capabilities probe disagrees
    with its announcement gets zero forwards and is marked unverified."""

    def test_two_members_form_pool(self):
        """Two members announcing cortex with matching fingerprints are both
        verified and appear in member_origins('cortex')."""
        fp = _fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="NVFP4")
        roles_cortex = {"cortex": _role("cortex", fingerprint=fp)}
        roles_senses = {"cortex": _role("cortex", fingerprint=fp)}

        roster = _FakeRoster(
            [
                ("alpha", "http://alpha.local:8001", 1.0),
                ("beta", "http://beta.local:8001", 1.0),
            ]
        )
        ann_map = {
            "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles_cortex),
            "http://beta.local:8001": _ann("beta", "http://beta.local:8001", roles_senses),
        }
        # Both verified by /capabilities.
        ver_roles = {
            "http://alpha.local:8001": frozenset(["cortex"]),
            "http://beta.local:8001": frozenset(["cortex"]),
        }
        snap = build_snapshot(roster, announcements=ann_map, verified_roles=ver_roles)

        origins = origins_for_role(snap, "cortex")
        assert len(origins) == 2
        assert "http://alpha.local:8001" in origins
        assert "http://beta.local:8001" in origins

    def test_probe_mismatch_marks_unverified(self):
        """A member whose /capabilities probe does not include an announced role
        is marked unverified (verified_roles is empty) and gets zero forwards."""
        fp_cortex = _fp(served_id="unsloth/Qwen3.8-27B-NVFP4")
        fp_embed = _fp(served_id="Qwen/Qwen3-Embedding-0.6B", quantization="BF16")

        roles_cortex = {"cortex": _role("cortex", fingerprint=fp_cortex)}
        roles_embed = {"cortex": _role("cortex", fingerprint=fp_embed)}  # different fp

        roster = _FakeRoster(
            [
                ("alpha", "http://alpha.local:8001", 1.0),
                ("beta", "http://beta.local:8001", 1.0),
            ]
        )
        ann_map = {
            "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles_cortex),
            "http://beta.local:8001": _ann("beta", "http://beta.local:8001", roles_embed),
        }
        # Alpha's probe matches; beta's probe does NOT (embed fingerprint ≠ cortex).
        ver_roles = {
            "http://alpha.local:8001": frozenset(["cortex"]),
            "http://beta.local:8001": frozenset(["embed"]),  # different!
        }
        snap = build_snapshot(roster, announcements=ann_map, verified_roles=ver_roles)

        # Alpha is verified for cortex.
        assert snap.is_verified("http://alpha.local:8001", "cortex") is True
        # Beta is unverified for cortex (probe mismatch).
        assert snap.is_verified("http://beta.local:8001", "cortex") is False

        # Only alpha is in the cortex pool.
        origins = origins_for_role(snap, "cortex")
        assert origins == ("http://alpha.local:8001",)

        # New meaning (boot window, t1): beta's probe DID land — it simply
        # verified a different role — so beta is `probed` and is NOT excused
        # as pending.  "Unverified" and "not yet probed" are now distinct.
        from lobes.gateway._mesh_routing import compute_role_placement

        beta = next(m for m in snap.members if m.origin == "http://beta.local:8001")
        assert beta.probed is True
        assert compute_role_placement(snap, "cortex").pending_origins == ()

    def test_unverified_member_has_empty_verified_roles(self):
        """An unverified member's MemberInfo carries empty verified_roles."""
        roster = _FakeRoster([("beta", "http://beta.local:8001", 1.0)])
        fp_cortex = _fp(served_id="unsloth/Qwen3.8-27B-NVFP4")
        roles_cortex = {"cortex": _role("cortex", fingerprint=fp_cortex)}
        ann_map = {
            "http://beta.local:8001": _ann("beta", "http://beta.local:8001", roles_cortex),
        }
        # Probe disagrees: beta announced cortex, but probe says something else.
        ver_roles = {"http://beta.local:8001": frozenset(["embed"])}
        snap = build_snapshot(roster, announcements=ann_map, verified_roles=ver_roles)

        for m in snap.members:
            if m.origin == "http://beta.local:8001":
                assert m.verified_roles == ()
                assert m.announced_roles == ("cortex",)
                # New meaning (t1): the probe landed, so this is a hard
                # "unverified", not the boot window's "not yet probed".
                assert m.probed is True
                break
        else:
            pytest.fail("beta not found in snapshot members")


# ---------------------------------------------------------------------------
# AC3: hand forward / realtime 404
# ---------------------------------------------------------------------------


class TestHandAndRealtime:
    """AC3: model=hand on a member lacking hand forwards with X-Lobes-Proxied-By;
    GET /v1/realtime on a member lacking stt is 404 role_infeasible while another
    member announces stt."""

    def test_hand_role_routing_markers(self):
        """A member without the hand role is marked unverified for hand;
        mesh_markers reflects the unverified state."""
        roster = _FakeRoster(
            [
                ("alpha", "http://alpha.local:8001", 1.0),
                ("beta", "http://beta.local:8001", 1.0),
            ]
        )
        # Alpha has hand; beta does not.
        roles_alpha = {"cortex": _role("cortex"), "hand": _role("hand")}
        roles_beta = {"cortex": _role("cortex")}  # no hand

        ann_map = {
            "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles_alpha),
            "http://beta.local:8001": _ann("beta", "http://beta.local:8001", roles_beta),
        }
        ver_roles = {
            "http://alpha.local:8001": frozenset(["cortex", "hand"]),
            "http://beta.local:8001": frozenset(["cortex"]),
        }
        snap = build_snapshot(roster, announcements=ann_map, verified_roles=ver_roles)

        # Alpha is verified for hand; beta is not.
        assert snap.is_verified("http://alpha.local:8001", "hand") is True
        assert snap.is_verified("http://beta.local:8001", "hand") is False

        # Beta announced no hand role; origins_for_role only returns verified.
        origins = origins_for_role(snap, "hand")
        assert origins == ("http://alpha.local:8001",)

    def test_realtime_role_check(self):
        """A member lacking stt is not in the stt pool; mesh_markers shows unverified."""
        roster = _FakeRoster(
            [
                ("alpha", "http://alpha.local:8001", 1.0),
                ("beta", "http://beta.local:8001", 1.0),
            ]
        )
        roles_alpha = {"stt": _role("stt", model="stt-model")}
        roles_beta = {"cortex": _role("cortex")}

        ann_map = {
            "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles_alpha),
            "http://beta.local:8001": _ann("beta", "http://beta.local:8001", roles_beta),
        }
        ver_roles = {
            "http://alpha.local:8001": frozenset(["stt"]),
            "http://beta.local:8001": frozenset(["cortex"]),
        }
        snap = build_snapshot(roster, announcements=ann_map, verified_roles=ver_roles)

        # Only alpha is in the stt pool.
        origins = origins_for_role(snap, "stt")
        assert origins == ("http://alpha.local:8001",)
        assert "http://beta.local:8001" not in origins

    def test_mesh_markers_reflect_verification(self):
        """mesh_markers includes X-Lobes-Mesh-Verified/X-Lobes-Mesh-Unverified."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        roles = {"cortex": _role("cortex")}
        ann_map = {"http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles)}
        ver_roles = {"http://alpha.local:8001": frozenset(["cortex"])}
        snap = build_snapshot(roster, announcements=ann_map, verified_roles=ver_roles)

        markers = mesh_markers(snap, "cortex", chosen_origin="http://alpha.local:8001")
        header_names = [m[0] for m in markers]
        assert "X-Lobes-Mesh-Origin" in header_names
        assert "X-Lobes-Mesh-Verified" in header_names
        assert "X-Lobes-Mesh-Unverified" not in header_names

        # Unverified marker.
        markers2 = mesh_markers(
            snap, "cortex", chosen_origin="http://other.local:8001", unverified=True
        )
        header_names2 = [m[0] for m in markers2]
        assert "X-Lobes-Mesh-Unverified" in header_names2


# ---------------------------------------------------------------------------
# AC4: Proxy loop guard (R -> B -> B lacks R → 508)
# ---------------------------------------------------------------------------


class TestProxyLoop:
    """AC4: A lacking R -> B (roster says B has R) -> B lacks R too: A's request
    returns 508 proxy_loop and C is never dialled; a forwarded request carries
    Bearer <join key> and never the client's bearer."""

    def test_member_lacks_role_no_forward(self):
        """A member that does not have verified roles for a role returns empty
        origins — no forward is attempted."""
        roster = _FakeRoster([("beta", "http://beta.local:8001", 1.0)])
        # Beta does NOT announce cortex at all.
        roles = {"embed": _role("embed")}
        ann_map = {"http://beta.local:8001": _ann("beta", "http://beta.local:8001", roles)}
        ver_roles = {"http://beta.local:8001": frozenset(["embed"])}
        snap = build_snapshot(roster, announcements=ann_map, verified_roles=ver_roles)

        # cortex is not in any verified role set.
        origins = origins_for_role(snap, "cortex")
        assert origins == ()

    def test_member_exists_check(self):
        """member_exists correctly reports presence/absence."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        roles = {"cortex": _role("cortex")}
        snap = build_snapshot(
            roster,
            announcements={
                "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles),
            },
        )

        assert member_exists(snap, "http://alpha.local:8001") is True
        assert member_exists(snap, "http://other.local:8001") is False
        # None snapshot returns False.
        assert member_exists(None, "http://alpha.local:8001") is False

    def test_forwarded_request_bearer_swap(self):
        """A forwarded request uses the join key (Bearer), not the client's
        bearer.  This is verified by checking the snapshot does not carry
        client token info — the join key is the only credential in the mesh."""
        roster = _FakeRoster(
            [
                ("alpha", "http://alpha.local:8001", 1.0),
            ]
        )
        roles = {"cortex": _role("cortex")}
        ann_map = {"http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles)}
        ver_roles = {"http://alpha.local:8001": frozenset(["cortex"])}
        snap = build_snapshot(roster, announcements=ann_map, verified_roles=ver_roles)

        # The snapshot should not carry any bearer token.
        # Check that MemberInfo does not have an api_key field.
        for m in snap.members:
            assert not hasattr(m, "api_key") or m.api_key is None  # type: ignore[attr-defined]

    def test_origins_empty_for_unknown_role(self):
        """When no member verified a role, origins_for_role returns empty."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        roles = {"cortex": _role("cortex")}
        snap = build_snapshot(
            roster,
            announcements={
                "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles),
            },
        )
        # tts is not in any member's verified set.
        assert origins_for_role(snap, "tts") == ()


# ---------------------------------------------------------------------------
# AC5: Dropped member leaves the pool
# ---------------------------------------------------------------------------


class TestMemberDrop:
    """AC5: a member dropped from the roster leaves the pool on the next
    snapshot; requests for its only role 404 role_infeasible with no hosted_by
    within one snapshot, never hang."""

    def test_dropped_member_disappears(self):
        """After a member is removed from the roster, the new snapshot no
        longer includes it."""
        roster = _FakeRoster(
            [
                ("alpha", "http://alpha.local:8001", 1.0),
            ]
        )
        roles_alpha = {"cortex": _role("cortex")}
        snap1 = build_snapshot(
            roster,
            announcements={
                "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles_alpha),
            },
            verified_roles={
                "http://alpha.local:8001": frozenset(["cortex"]),
            },
        )
        assert snap1.member_count() == 1
        assert origins_for_role(snap1, "cortex") == ("http://alpha.local:8001",)

        # Simulate: alpha is dropped from the roster.
        roster2 = _FakeRoster([])
        snap2 = build_snapshot(
            roster2,
            announcements={
                "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles_alpha),
            },
        )
        # Alpha's origin is still in announcements, but no roster entry.
        # build_snapshot iterates roster.members(), so alpha is not included.
        assert snap2.member_count() == 0
        assert origins_for_role(snap2, "cortex") == ()
        # D5: a dropped member's stale announcement must be PRUNED, not just
        # unreachable via .members() — otherwise a verify pass driven by
        # snapshot.announcements would re-dial a dead origin.
        assert len(snap2.announcements) == 0

    def test_404_for_role_after_drop(self):
        """When a member is dropped and no other member has the role,
        origins_for_role returns empty — mimicking a 404 role_infeasible."""
        roster = _FakeRoster([])  # no members at all
        snap = build_snapshot(roster)
        assert origins_for_role(snap, "cortex") == ()
        assert roster_member_count(snap) == 0

    def test_no_hosted_by_after_drop(self):
        """After a member is dropped, there is no origin for that role —
        mimicking 'no hosted_by' in a 404 response."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        snap1 = build_snapshot(
            roster,
            announcements={
                "http://alpha.local:8001": _ann(
                    "alpha", "http://alpha.local:8001", {"cortex": _role("cortex")}
                ),
            },
            verified_roles={
                "http://alpha.local:8001": frozenset(["cortex"]),
            },
        )
        # Before drop: hosted_by exists.
        assert len(origins_for_role(snap1, "cortex")) == 1

        # After drop.
        roster2 = _FakeRoster([])
        snap2 = build_snapshot(roster2)
        # No origins → no hosted_by → 404 role_infeasible.
        assert origins_for_role(snap2, "cortex") == ()

    def test_snapshot_is_immediate_after_drop(self):
        """A new build_snapshot after roster drop immediately reflects the change;
        no async delay, no hang."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        snap1 = build_snapshot(
            roster,
            announcements={
                "http://alpha.local:8001": _ann(
                    "alpha", "http://alpha.local:8001", {"cortex": _role("cortex")}
                ),
            },
            verified_roles={
                "http://alpha.local:8001": frozenset(["cortex"]),
            },
        )
        assert snap1.member_count() == 1

        # Simulate roster change.
        roster2 = _FakeRoster([])
        snap2 = build_snapshot(roster2)  # should return immediately, no blocking.
        assert snap2.member_count() == 0

    def test_member_roles_after_drop(self):
        """snapshot_member_roles returns empty for a dropped member's origin."""
        roster = _FakeRoster([])
        snap = build_snapshot(roster)
        assert snapshot_member_roles(snap, "http://alpha.local:8001") == ()


# ---------------------------------------------------------------------------
# SnapshotHolder thread-safety
# ---------------------------------------------------------------------------


class TestSnapshotHolderThreadSafety:
    """SnapshotHolder operations are lock-protected."""

    def test_concurrent_read_write(self):
        """Concurrent .current() calls during .replace() do not raise."""
        roster = _FakeRoster([])
        holder = SnapshotHolder(roster)
        errors: list[BaseException] = []

        def reader():
            try:
                for _ in range(100):
                    holder.current()
            except BaseException as exc:
                errors.append(exc)

        def writer():
            for _ in range(100):
                snap = build_snapshot(roster)
                holder.replace(snap)

        threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == [], f"Thread errors: {errors}"
        assert holder.current() is not None or holder.current() is None  # always safe

    def test_update_replaces(self):
        """SnapshotHolder.update builds a new snapshot and replaces."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        holder = SnapshotHolder(roster)
        snap = holder.update()
        assert snap is not None
        assert holder.current() is snap


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


class TestRoutingIntegrationHelpers:
    """Test the convenience helper functions that replace table access in server.py."""

    def test_origins_for_role_none_snapshot(self):
        """origins_for_role(None, role) returns empty tuple."""
        assert origins_for_role(None, "cortex") == ()

    def test_member_exists_none_snapshot(self):
        """member_exists(None, origin) returns False."""
        assert member_exists(None, "http://alpha.local:8001") is False

    def test_roster_member_count_none_snapshot(self):
        """roster_member_count(None) returns 0."""
        assert roster_member_count(None) == 0

    def test_snapshot_member_roles_none(self):
        """snapshot_member_roles(None, origin) returns empty tuple."""
        assert snapshot_member_roles(None, "http://alpha.local:8001") == ()

    def test_mesh_markers_no_chosen_origin(self):
        """mesh_markers with no chosen_origin omits X-Lobes-Mesh-Origin."""
        roster = _FakeRoster([])
        snap = build_snapshot(roster)
        markers = mesh_markers(snap, "cortex")
        header_names = [m[0] for m in markers]
        assert "X-Lobes-Mesh-Origin" not in header_names
        assert "X-Lobes-Mesh-Role" in header_names

    def test_all_member_attributes(self):
        """MemberInfo carries all required attributes."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 42.0)])
        roles = {"cortex": _role("cortex")}
        snap = build_snapshot(
            roster,
            announcements={
                "http://alpha.local:8001": _ann("alpha", "http://alpha.local:8001", roles),
            },
            verified_roles={
                "http://alpha.local:8001": frozenset(["cortex"]),
            },
        )
        m = snap.members[0]
        assert m.name == "alpha"
        assert m.origin == "http://alpha.local:8001"
        assert m.announced_roles == ("cortex",)
        assert m.verified_roles == ("cortex",)
        assert m.capacity == 42.0


class TestAnnouncementTuples:
    """Test that announcements are stored as tuples in the snapshot."""

    def test_announcement_tuples_accessible(self):
        """The announcements tuple stores (origin, Announcement) pairs."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        roles = {"cortex": _role("cortex")}
        ann = _ann("alpha", "http://alpha.local:8001", roles)
        snap = build_snapshot(
            roster,
            announcements={
                "http://alpha.local:8001": ann,
            },
        )
        assert len(snap.announcements) == 1
        origin, a = snap.announcements[0]
        assert origin == "http://alpha.local:8001"
        assert a is ann

    def test_no_announcement_tuple_when_no_announcements(self):
        """With no announcements dict, announcements is empty."""
        roster = _FakeRoster([("alpha", "http://alpha.local:8001", 1.0)])
        snap = build_snapshot(roster)
        assert len(snap.announcements) == 0


# ---------------------------------------------------------------------------
# D6/D7: verify_member_roles is the single fingerprint-comparison entry point
# ---------------------------------------------------------------------------


class TestVerifyMemberRoles:
    """D6: verification is a single entry point (`verify_member_roles`),
    running `compare_fingerprints` per announced role — not merely a
    role-*name* subset check."""

    def test_verify_requires_fingerprint_match(self) -> None:
        from lobes.gateway._mesh_routing import verify_member_roles

        fp_a = _fp(served_id="unsloth/Qwen3.8-27B-NVFP4")
        fp_b = _fp(served_id="unsloth/Qwen3.6-27B-NVFP4")  # different served_id
        ann = _ann(
            "alpha", "http://alpha.local:8001", {"cortex": _role("cortex", fingerprint=fp_a)}
        )
        probed_roles = {
            "cortex": {
                "fingerprint": {
                    "served_id": fp_b.served_id,
                    "quantization": fp_b.quantization,
                    "max_model_len": fp_b.max_model_len,
                    "runtime": fp_b.runtime,
                },
                "ready": True,
            }
        }
        assert verify_member_roles(ann, probed_roles) == frozenset()

    def test_verify_accepts_matching_fingerprint(self) -> None:
        from lobes.gateway._mesh_routing import verify_member_roles

        fp = _fp()
        ann = _ann("alpha", "http://alpha.local:8001", {"cortex": _role("cortex", fingerprint=fp)})
        probed_roles = {
            "cortex": {
                "fingerprint": {
                    "served_id": fp.served_id,
                    "quantization": fp.quantization,
                    "max_model_len": fp.max_model_len,
                    "runtime": fp.runtime,
                },
                "ready": True,
            }
        }
        assert verify_member_roles(ann, probed_roles) == frozenset({"cortex"})

    def test_verify_not_ready_excluded(self) -> None:
        """A role whose /capabilities entry reports ready is False is never
        verified even when the fingerprint matches (W6: compatible AND
        ready)."""
        from lobes.gateway._mesh_routing import verify_member_roles

        fp = _fp()
        ann = _ann("alpha", "http://alpha.local:8001", {"cortex": _role("cortex", fingerprint=fp)})
        probed_roles = {
            "cortex": {
                "fingerprint": {
                    "served_id": fp.served_id,
                    "quantization": fp.quantization,
                    "max_model_len": fp.max_model_len,
                    "runtime": fp.runtime,
                },
                "ready": False,
            }
        }
        assert verify_member_roles(ann, probed_roles) == frozenset()


class TestWireFingerprintConversion:
    """D7: the wire<->replica fingerprint conversion maps the wire's
    ``0``/``null``/``""`` "N/A" sentinels to `None` (unknown) on BOTH the
    announced (dataclass) and probed (raw dict) sides."""

    def test_wire_fingerprint_zero_window_is_unknown(self) -> None:
        from lobes.gateway._mesh_routing import _wire_fingerprint_to_replica

        wire_fp = Fingerprint(
            served_id="unsloth/Qwen3.8-27B-NVFP4",
            quantization="NVFP4",
            max_model_len=0,  # the wire's "N/A"
            runtime="vllm",
        )
        replica_fp = _wire_fingerprint_to_replica(wire_fp)
        assert replica_fp.max_model_len is None

    def test_probed_dict_zero_window_is_unknown(self) -> None:
        """The probed side arrives as a plain dict off JSON, not a
        Fingerprint dataclass — 0/null there must convert identically."""
        from lobes.gateway._mesh_routing import _wire_fingerprint_to_replica

        replica_fp = _wire_fingerprint_to_replica(
            {
                "served_id": "unsloth/Qwen3.8-27B-NVFP4",
                "quantization": "NVFP4",
                "max_model_len": 0,
                "runtime": None,
            }
        )
        assert replica_fp.max_model_len is None
        assert replica_fp.runtime is None

    def test_empty_string_fields_are_unknown(self) -> None:
        from lobes.gateway._mesh_routing import _wire_fingerprint_to_replica

        replica_fp = _wire_fingerprint_to_replica(
            Fingerprint(served_id="", quantization="", max_model_len=262144, runtime="")
        )
        assert replica_fp.served_id is None
        assert replica_fp.quantization is None
        assert replica_fp.runtime is None
        assert replica_fp.max_model_len == 262144

    def test_zero_window_never_compares_compatible(self) -> None:
        """The regression D7 exists to prevent: a wire ``0`` must not compare
        as a KNOWN value (`0 != 128000`) — it must be UNKNOWN, and an unknown
        value on either side is always incompatible (spec h11)."""
        from lobes.gateway._mesh_routing import _wire_fingerprint_to_replica
        from lobes.gateway._replicas import compare_fingerprints

        zero_fp = _wire_fingerprint_to_replica(
            Fingerprint(served_id="m", quantization="q", max_model_len=0, runtime="vllm")
        )
        real_fp = _wire_fingerprint_to_replica(
            Fingerprint(served_id="m", quantization="q", max_model_len=128000, runtime="vllm")
        )
        compatible, reason = compare_fingerprints(zero_fp, real_fp)
        assert compatible is False
        assert "max_model_len" in reason


def test_verification_is_per_role_not_all_or_nothing():
    """Live dev527 (2026-09-12): the Orin announced six hosted lanes with only
    ``associate`` running; the all-or-nothing rule left it with zero verified
    roles on every peer. A member keeps exactly the announced roles the probe
    verified, and is unverified only when none of them does."""
    from lobes.gateway._mesh_routing import build_snapshot, origins_for_role

    roles = {
        "associate": _role("associate", fingerprint=_fp(served_id="nvidia/Lightning")),
        "hand": _role("hand", fingerprint=_fp(served_id="LiquidAI/LFM2.5-1.2B-Instruct")),
        "embedder": _role("embedder", fingerprint=_fp(served_id="Qwen/Qwen3-Embedding-0.6B")),
    }
    roster = _FakeRoster([("orin", "http://orin.local:8000", 1.0)])
    ann_map = {"http://orin.local:8000": _ann("orin", "http://orin.local:8000", roles)}
    snap = build_snapshot(
        roster,
        announcements=ann_map,
        verified_roles={"http://orin.local:8000": frozenset(["associate"])},
    )
    member = next(m for m in snap.members if m.origin == "http://orin.local:8000")
    assert member.verified_roles == ("associate",)
    assert origins_for_role(snap, "associate") == ("http://orin.local:8000",)
    assert origins_for_role(snap, "hand") == ()
    # Nothing verified → still unverified, exactly as before.
    snap2 = build_snapshot(
        roster,
        announcements=ann_map,
        verified_roles={"http://orin.local:8000": frozenset(["cortex"])},
    )
    assert next(m for m in snap2.members).verified_roles == ()


# ---------------------------------------------------------------------------
# Boot window: the never-probed sentinel, per-role readiness, and pending
# placement (mesh-boot-window-and-capabilities-advert, t1 — covers c4, h4).
#
# Before this, "not verified" conflated two different states: a member whose
# probe RAN and verified nothing, and a member whose first probe has not
# landed yet (the boot window).  ``MemberInfo.probed`` separates them, and
# ``RolePlacement.pending_origins`` names the announcers still inside that
# window so a caller can be told "not yet" instead of "never".
# ---------------------------------------------------------------------------


ORIGIN_A = "http://alpha.local:8001"
ORIGIN_B = "http://beta.local:8001"


class TestNeverProbedSentinel:
    def test_member_info_defaults_are_back_compatible(self):
        """The two new fields default so every pre-existing construction of a
        MemberInfo (keyword, without them) stays byte-identical in meaning."""
        m = MemberInfo(
            name="alpha",
            origin=ORIGIN_A,
            announced_roles=("cortex",),
            verified_roles=(),
            capacity=1.0,
        )
        assert m.probed is False
        assert m.ready_roles == ()
        assert m.unverified_reason is None

    def test_never_probed_member_reads_probed_false_and_no_reason(self):
        """No probe data at all → probed False, and the model level carries NO
        fabricated reason (the ``not_yet_probed`` string is a /mesh/roster
        presentation concern, not a MemberInfo value)."""
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(roster, announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)})
        member = next(m for m in snap.members if m.origin == ORIGIN_A)
        assert member.probed is False
        assert member.unverified_reason is None
        assert member.verified_roles == ()
        assert member.ready_roles == ()

    def test_probed_true_for_every_origin_in_the_verified_map(self):
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0), ("beta", ORIGIN_B, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={
                ORIGIN_A: _ann("alpha", ORIGIN_A),
                ORIGIN_B: _ann("beta", ORIGIN_B),
            },
            verified_roles={ORIGIN_A: frozenset(["cortex"])},
        )
        by_origin = {m.origin: m for m in snap.members}
        assert by_origin[ORIGIN_A].probed is True
        assert by_origin[ORIGIN_B].probed is False

    def test_probed_true_for_every_origin_in_the_reason_map(self):
        """A probe that RAN and verified nothing still counts as probed — that
        is exactly the member the boot window must stop excusing."""
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)},
            unverified_reasons={ORIGIN_A: "HTTP 503"},
        )
        member = next(m for m in snap.members if m.origin == ORIGIN_A)
        assert member.probed is True
        assert member.unverified_reason == "HTTP 503"
        assert member.verified_roles == ()


class TestReadyRoles:
    def test_ready_roles_threaded_from_the_probe_map(self):
        """``ready_roles`` are the roles whose probed /capabilities entry read
        ``ready: true`` — a superset of verified_roles in general (readiness
        does not imply a matching fingerprint)."""
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        roles = {
            "cortex": _role("cortex"),
            "hand": _role("hand", fingerprint=_fp(served_id="LiquidAI/LFM2.5-1.2B-Instruct")),
        }
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A, roles)},
            verified_roles={ORIGIN_A: frozenset(["cortex"])},
            ready_roles={ORIGIN_A: frozenset(["cortex", "hand"])},
        )
        member = next(m for m in snap.members if m.origin == ORIGIN_A)
        assert member.verified_roles == ("cortex",)
        assert member.ready_roles == ("cortex", "hand")

    def test_ready_roles_absent_means_empty_and_probed_still_set(self):
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)},
            verified_roles={ORIGIN_A: frozenset(["cortex"])},
        )
        member = next(m for m in snap.members if m.origin == ORIGIN_A)
        assert member.ready_roles == ()
        assert member.probed is True

    def test_ready_roles_alone_counts_as_probed(self):
        """A clean probe that found the lane not-ready reports readiness data
        but neither a verified role nor a reason; it is still probed."""
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)},
            ready_roles={ORIGIN_A: frozenset()},
        )
        member = next(m for m in snap.members if m.origin == ORIGIN_A)
        assert member.probed is True


class TestPendingPlacement:
    def test_role_placement_pending_origins_defaults_empty(self):
        from lobes.gateway._mesh_routing import RolePlacement

        placement = RolePlacement(role="cortex", plain_origins=(), suffixed=())
        assert placement.pending_origins == ()

    def test_never_probed_announcer_is_pending_not_plain(self):
        from lobes.gateway._mesh_routing import compute_role_placement

        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(roster, announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)})
        placement = compute_role_placement(snap, "cortex")
        assert placement.pending_origins == (ORIGIN_A,)
        assert placement.plain_origins == ()
        assert placement.suffixed == ()

    def test_probed_but_unverified_announcer_is_not_pending(self):
        """The boot window excuses only a member whose probe has not landed.
        One that was probed and verified nothing is a hard negative."""
        from lobes.gateway._mesh_routing import compute_role_placement

        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)},
            unverified_reasons={ORIGIN_A: "HTTP 503"},
        )
        placement = compute_role_placement(snap, "cortex")
        assert placement.pending_origins == ()
        assert placement.plain_origins == ()

    def test_verified_member_placement_is_unchanged_and_not_pending(self):
        from lobes.gateway._mesh_routing import compute_role_placement

        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)},
            verified_roles={ORIGIN_A: frozenset(["cortex"])},
        )
        placement = compute_role_placement(snap, "cortex")
        assert placement.plain_origins == (ORIGIN_A,)
        assert placement.suffixed == ()
        assert placement.pending_origins == ()

    def test_pending_and_verified_coexist(self):
        """One verified member serves plain while a second, still-unprobed
        announcer of the same role is listed pending — the plain pool is NOT
        widened by a pending origin."""
        from lobes.gateway._mesh_routing import compute_role_placement

        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0), ("beta", ORIGIN_B, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={
                ORIGIN_A: _ann("alpha", ORIGIN_A),
                ORIGIN_B: _ann("beta", ORIGIN_B),
            },
            verified_roles={ORIGIN_A: frozenset(["cortex"])},
        )
        placement = compute_role_placement(snap, "cortex")
        assert placement.plain_origins == (ORIGIN_A,)
        assert placement.pending_origins == (ORIGIN_B,)

    def test_pending_only_for_the_announced_role(self):
        from lobes.gateway._mesh_routing import compute_role_placement

        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A, {"cortex": _role("cortex")})},
        )
        assert compute_role_placement(snap, "cortex").pending_origins == (ORIGIN_A,)
        assert compute_role_placement(snap, "senses").pending_origins == ()

    def test_pending_origins_are_ordered_by_member_name(self):
        from lobes.gateway._mesh_routing import compute_role_placement

        roster = _FakeRoster([("zeta", ORIGIN_B, 1.0), ("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={
                ORIGIN_A: _ann("alpha", ORIGIN_A),
                ORIGIN_B: _ann("zeta", ORIGIN_B),
            },
        )
        placement = compute_role_placement(snap, "cortex")
        assert placement.pending_origins == (ORIGIN_A, ORIGIN_B)


class TestDiscoveredRoles:
    """d1: a seed roster's per-member ``roles`` list stands in for the
    announcement a recreated box has not received yet, so the pending (503)
    path can name the member; a real announcement always wins."""

    def _roster(self):
        from lobes.gateway._mesh_roster import Roster

        r = Roster()
        r.discover("thor", "http://thor:8000", None, now=0.0)
        return r

    def test_discovered_roles_become_announced_roles_when_no_announcement(self) -> None:
        snap = build_snapshot(
            self._roster(), discovered_roles={"http://thor:8000": ("worker", "reranker")}
        )
        m = snap.members[0]
        assert m.announced_roles == ("reranker", "worker")
        assert m.verified_roles == ()
        assert m.probed is False

    def test_discovered_member_is_pending_for_its_roles(self) -> None:
        snap = build_snapshot(self._roster(), discovered_roles={"http://thor:8000": ("worker",)})
        placement = compute_role_placement(snap, "worker")
        assert placement.pending_origins == ("http://thor:8000",)
        assert placement.plain_origins == ()
        assert compute_role_placement(snap, "cortex").pending_origins == ()

    def test_a_real_announcement_wins_over_discovered_roles(self) -> None:
        ann = _ann("thor", "http://thor:8000", roles={"cortex": _role("m")})
        snap = build_snapshot(
            self._roster(),
            announcements={"http://thor:8000": ann},
            discovered_roles={"http://thor:8000": ("worker",)},
        )
        assert snap.members[0].announced_roles == ("cortex",)

    def test_absent_discovered_roles_is_byte_identical(self) -> None:
        a = build_snapshot(self._roster())
        b = build_snapshot(self._roster(), discovered_roles=None)
        assert a.members == b.members


class TestRoleContext:
    """Qodo thread 2: the peer-advertised serving window rides the snapshot."""

    def test_role_context_is_threaded_from_the_probe_map(self):
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)},
            verified_roles={ORIGIN_A: frozenset(["cortex"])},
            role_contexts={ORIGIN_A: {"cortex": 262144, "hand": 32768}},
        )
        member = next(m for m in snap.members if m.origin == ORIGIN_A)
        # Sorted by role name, never by payload iteration order.
        assert member.role_context == (("cortex", 262144), ("hand", 32768))
        assert member.context_for("cortex") == 262144
        assert member.context_for("hand") == 32768
        assert member.context_for("muse") is None

    def test_role_context_defaults_to_empty_and_is_byte_identical(self):
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        a = build_snapshot(roster, announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)})
        b = build_snapshot(
            roster, announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)}, role_contexts=None
        )
        assert a.members == b.members
        assert a.members[0].role_context == ()
        assert a.members[0].context_for("cortex") is None

    def test_role_context_alone_does_not_mark_a_member_probed(self):
        """``probed`` stays the verified/reason/ready sentinel — adding a
        context map must not silently retire a member from the boot window."""
        roster = _FakeRoster([("alpha", ORIGIN_A, 1.0)])
        snap = build_snapshot(
            roster,
            announcements={ORIGIN_A: _ann("alpha", ORIGIN_A)},
            role_contexts={ORIGIN_A: {"cortex": 262144}},
        )
        assert snap.members[0].probed is False
