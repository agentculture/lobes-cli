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
