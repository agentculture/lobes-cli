"""Tests for suffixed-lane naming and exposure (mesh-t8, issue #237).

Covers the acceptance contract:

1. two members announcing the same role with disagreeing fingerprints are
   BOTH exposed only as suffixed lanes ("{role}-{member}"), never plain.
2. a raw served id hosted only via suffixed lanes resolves to every
   suffixed name (never a plain origin); when locally hosted it is local.
3. a role announced private never appears in a peer's roster/placement.
4. a fingerprint change moves a member from the plain pool to a suffixed
   lane; the naming layer recomputes per-snapshot (i.e. within one probe
   refresh), not per-heartbeat.

Spec targets: c27, h1, c48, h39, h20, c46, h37, c15, h14, c16, h15.
"""

from __future__ import annotations

import json

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_routing import (
    MESH_MEMBER_HEADER,
    RolePlacement,
    SuffixedLane,
    build_snapshot,
    compute_role_placement,
    find_suffixed_lane,
    suffixed_lane_name,
)
from lobes.gateway._mesh_wire import Announcement, Fingerprint, RoleInfo
from lobes.gateway._replicas import Fingerprint as ReplicaFingerprint
from lobes.gateway._replicas import compare_fingerprints
from tests.test_mesh_routes import _fake_handler, _make_routes
from tests.test_mesh_routing_wiring import (
    _FakeMemberGateway,
)
from tests.test_mesh_routing_wiring import _fp as _wiring_fp
from tests.test_mesh_routing_wiring import _role as _wiring_role
from tests.test_mesh_routing_wiring import (
    _setup_mesh,
)

# ---------------------------------------------------------------------------
# Fixtures (mirrors tests/test_mesh_routing.py's own helpers)
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
    return RoleInfo(
        model=over.get("model", "unsloth/Qwen3.8-27B-NVFP4"),
        runtime=over.get("runtime", "vllm"),
        context=over.get("context", 262144),
        quant=over.get("quant", "NVFP4"),
        responsibilities=over.get("responsibilities", ("generate",)),
        forbidden_responsibilities=over.get("forbidden_responsibilities", ()),
        fingerprint=over.get("fingerprint", _fp()),
        capacity=over.get("capacity"),
        private=over.get("private", False),
    )


def _ann(name: str, origin: str, roles: dict[str, RoleInfo] | None = None) -> Announcement:
    return Announcement(
        name=name,
        origin=origin,
        schema_version="1.0.0",
        roles=roles or {"cortex": _role("cortex")},
    )


def _replica_fp(**over) -> ReplicaFingerprint:
    return ReplicaFingerprint(
        served_id=over.get("served_id", "unsloth/Qwen3.8-27B-NVFP4"),
        max_model_len=over.get("max_model_len", 262144),
        runtime=over.get("runtime", "vllm"),
        quantization=over.get("quantization", "NVFP4"),
        kv_cache_dtype="",
        reasoning_parser="",
        tool_parser="",
        speculative_config="",
    )


class _FakeRoster:
    """Minimal Roster stand-in exposing exactly what build_snapshot reads."""

    def __init__(self, members: list[tuple[str, str, float]]):
        self._names = [m[0] for m in members]
        self._records = {}
        for name, origin, capacity in members:
            rec = type(
                "Rec",
                (),
                {"name": name, "origin": origin, "capacity": capacity},
            )()
            self._records[name] = rec
        self._roster = self._records

    def members(self):
        return list(self._names)


def _two_member_snapshot(fp_a: Fingerprint, fp_b: Fingerprint, *, role="cortex"):
    roster = _FakeRoster([("nameA", "http://a", 1.0), ("nameB", "http://b", 1.0)])
    ann_a = _ann("nameA", "http://a", {role: _role(role, fingerprint=fp_a)})
    ann_b = _ann("nameB", "http://b", {role: _role(role, fingerprint=fp_b)})
    snap = build_snapshot(
        roster,
        announcements={"http://a": ann_a, "http://b": ann_b},
        verified_roles={"http://a": frozenset({role}), "http://b": frozenset({role})},
    )
    return snap


# ---------------------------------------------------------------------------
# Criterion 1: disagreeing peers → both suffixed, neither plain
# ---------------------------------------------------------------------------


def test_two_members_disagree_both_suffixed_never_plain():
    fp_a = _fp(quantization="NVFP4")
    fp_b = _fp(quantization="FP8")
    snap = _two_member_snapshot(fp_a, fp_b)

    placement = compute_role_placement(snap, "cortex", local_fingerprint=None)

    assert placement.plain_origins == ()
    names = set(placement.suffixed_names())
    assert names == {"cortex-nameA", "cortex-nameB"}
    # never exactly one exposed plain, and never only one suffixed
    assert len(placement.suffixed) == 2


def test_two_members_agree_both_plain_no_suffix():
    fp = _fp(quantization="NVFP4")
    snap = _two_member_snapshot(fp, fp)

    placement = compute_role_placement(snap, "cortex", local_fingerprint=None)

    assert set(placement.plain_origins) == {"http://a", "http://b"}
    assert placement.suffixed == ()


def test_local_reference_splits_agreeing_and_disagreeing_members():
    local_fp = _replica_fp(quantization="NVFP4")
    fp_agrees = _fp(quantization="NVFP4")
    fp_disagrees = _fp(quantization="FP8")
    snap = _two_member_snapshot(fp_agrees, fp_disagrees)

    placement = compute_role_placement(snap, "cortex", local_fingerprint=local_fp)

    assert placement.plain_origins == ("http://a",)
    assert placement.suffixed_names() == ("cortex-nameB",)


def test_suffixed_lane_name_format():
    assert suffixed_lane_name("cortex", "thor") == "cortex-thor"


# ---------------------------------------------------------------------------
# Criterion 2: raw id resolution — suffixed-only vs local
# ---------------------------------------------------------------------------


def test_find_suffixed_lane_resolves_requested_raw_name():
    fp_a = _fp(quantization="NVFP4")
    fp_b = _fp(quantization="FP8")
    snap = _two_member_snapshot(fp_a, fp_b)

    lane = find_suffixed_lane(snap, "cortex-nameB", ("cortex", "senses"))

    assert lane is not None
    assert lane == SuffixedLane(
        name="cortex-nameB", role="cortex", member="nameB", origin="http://b"
    )


def test_find_suffixed_lane_returns_none_for_unknown_name():
    fp_a = _fp(quantization="NVFP4")
    fp_b = _fp(quantization="FP8")
    snap = _two_member_snapshot(fp_a, fp_b)

    assert find_suffixed_lane(snap, "cortex-nameC", ("cortex",)) is None
    # A name that agrees (plain-eligible) never resolves as suffixed.
    agree_snap = _two_member_snapshot(fp_a, fp_a)
    assert find_suffixed_lane(agree_snap, "cortex-nameA", ("cortex",)) is None


def test_find_suffixed_lane_none_snapshot():
    assert find_suffixed_lane(None, "cortex-nameA", ("cortex",)) is None


# ---------------------------------------------------------------------------
# Criterion 3: private roles never appear in placement (dropped before this
# layer sees them — verified at the wire boundary: Announcement.public()).
# ---------------------------------------------------------------------------


def test_private_role_dropped_by_announcement_public_before_placement():
    fp = _fp()
    private_role = _role("hand", fingerprint=fp, private=True)
    ann = Announcement(
        name="nameA", origin="http://a", schema_version="1.0.0", roles={"hand": private_role}
    )
    public = ann.public()
    assert "hand" not in public.roles

    roster = _FakeRoster([("nameA", "http://a", 1.0)])
    snap = build_snapshot(
        roster,
        announcements={"http://a": public},
        verified_roles={"http://a": frozenset({"hand"})},
    )
    placement = compute_role_placement(snap, "hand", local_fingerprint=None)
    # No announcement carries the role any more, so it is never usable as a
    # plain pool member — the private role is simply not there to place.
    assert placement.plain_origins == ()


# ---------------------------------------------------------------------------
# Criterion 4: recomputed per snapshot — moves out of the plain pool the
# moment the snapshot is rebuilt (probe refresh), no heartbeat wait baked in.
# ---------------------------------------------------------------------------


def test_placement_moves_member_out_of_plain_pool_on_fingerprint_change():
    fp = _fp(quantization="NVFP4")
    snap_before = _two_member_snapshot(fp, fp)
    placement_before = compute_role_placement(snap_before, "cortex")
    assert set(placement_before.plain_origins) == {"http://a", "http://b"}
    assert placement_before.suffixed == ()

    # Member B's quantization changes — a brand-new snapshot (as a probe
    # refresh would build) reflects it immediately; nothing here depends on
    # a heartbeat interval or any wall-clock/timer state.
    changed_fp = _fp(quantization="FP8")
    snap_after = _two_member_snapshot(fp, changed_fp)
    placement_after = compute_role_placement(snap_after, "cortex")

    assert placement_after.plain_origins == ()
    assert set(placement_after.suffixed_names()) == {"cortex-nameA", "cortex-nameB"}


def test_mesh_member_header_constant_is_dedicated_not_route_reason():
    # X-Lobes-Route-Reason stays a closed set (see lobes/gateway/_selection.py);
    # the mesh member marker is its own, separate header.
    assert MESH_MEMBER_HEADER == "X-Lobes-Mesh-Member"
    from lobes.gateway.server import ROUTE_REASON_HEADER

    assert MESH_MEMBER_HEADER != ROUTE_REASON_HEADER


def test_role_placement_dataclass_helpers():
    lane = SuffixedLane(name="cortex-x", role="cortex", member="x", origin="http://x")
    placement = RolePlacement(role="cortex", plain_origins=(), suffixed=(lane,))
    assert placement.suffixed_names() == ("cortex-x",)
    assert placement.origin_for_suffixed("cortex-x") == "http://x"
    assert placement.origin_for_suffixed("cortex-y") is None


def test_compute_role_placement_no_candidates_returns_empty():
    roster = _FakeRoster([])
    snap = build_snapshot(roster)
    placement = compute_role_placement(snap, "cortex")
    assert placement.plain_origins == ()
    assert placement.suffixed == ()


# ---------------------------------------------------------------------------
# POST /mesh/reannounce — immediate local re-announce hook
# ---------------------------------------------------------------------------


def _wire_ann(name: str) -> Announcement:
    return Announcement(
        name=name,
        origin="http://self",
        schema_version="1.0.0",
        roles={"cortex": _role("cortex")},
    )


def test_reannounce_requires_join_key():
    routes = _make_routes()
    status, _headers, body = routes.reannounce(_fake_handler("/mesh/reannounce", "POST"))
    assert status == 401
    assert json.loads(body)["error"]["type"] == "invalid_api_key"


def test_reannounce_no_announcement_yet_is_a_noop():
    routes = _make_routes()
    status, _headers, body = routes.reannounce(
        _fake_handler("/mesh/reannounce", "POST", headers={"Authorization": "Bearer sk-test"})
    )
    assert status == 200
    assert json.loads(body)["status"] == "no-op"


def test_reannounce_rebroadcasts_last_announcement_immediately():
    routes = _make_routes()
    ann = _wire_ann("test-box")
    routes._announcement = ann  # noqa: SLF001 — simulate a prior heartbeat send
    assert not routes._reannounce_event.is_set()

    status, _headers, body = routes.reannounce(
        _fake_handler("/mesh/reannounce", "POST", headers={"Authorization": "Bearer sk-test"})
    )

    assert status == 200
    assert json.loads(body) == {"status": "reannounced", "name": "test-box"}
    # reannounce_now sets the event so the heartbeat loop wakes within 1s.
    assert routes._reannounce_event.is_set()  # noqa: SLF001


def test_reannounce_uses_builder_for_fresh_data_when_wired():
    routes = _make_routes()
    calls = []

    def builder():
        calls.append(1)
        return _wire_ann("rebuilt-box")

    routes.set_announcement_builder(builder)
    status, _headers, body = routes.reannounce(
        _fake_handler("/mesh/reannounce", "POST", headers={"Authorization": "Bearer sk-test"})
    )

    assert status == 200
    assert calls == [1]
    assert json.loads(body)["name"] == "rebuilt-box"
    assert routes._announcement.name == "rebuilt-box"  # noqa: SLF001


def test_reannounce_registered_in_route_table():
    from lobes.gateway._mesh_routes import _MESH_ROUTES

    assert _MESH_ROUTES[("POST", "/mesh/reannounce")] == "reannounce"


# ---------------------------------------------------------------------------
# End-to-end wiring: handle_post through the real mesh dispatch path
# ---------------------------------------------------------------------------


class TestTwoDisagreeingMembersEndToEnd:
    """Criterion 1, wired through handle_post via real /capabilities probes."""

    def test_disagreeing_members_404_lists_both_never_picks_one(self, monkeypatch):
        fp_a = _wiring_fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="NVFP4")
        fp_b = _wiring_fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="FP8")
        join_key = "sk-mesh-join-key"

        members = [_FakeMemberGateway(), _FakeMemberGateway()]
        try:
            for m in members:
                m.start(0)
            origins = []
            for m, fp in zip(members, (fp_a, fp_b)):
                port = m._server.server_address[1]
                origin = f"http://127.0.0.1:{port}"
                origins.append(origin)
                m.set_capabilities(
                    {
                        "cortex": {
                            "fingerprint": {
                                "served_id": fp.served_id,
                                "quantization": fp.quantization,
                                "max_model_len": fp.max_model_len,
                                "runtime": fp.runtime,
                            },
                            "ready": True,
                        },
                    }
                )

            member_names = ["nameA", "nameB"]
            _routes, snap = _setup_mesh(
                origins,
                join_key,
                announced_roles={
                    origins[0]: {"cortex": _wiring_role("cortex", fingerprint=fp_a)},
                    origins[1]: {"cortex": _wiring_role("cortex", fingerprint=fp_b)},
                },
                fingerprints={origins[0]: fp_a, origins[1]: fp_b},
                member_names=member_names,
            )

            env = {
                "PRIMARY_FEASIBLE": "false",
                "PRIMARY_PEER_ORIGIN": origins[0],
                "PRIMARY_PEER_ORIGINS": origins[0],
            }
            table, cfg = build_config(env)
            specs = S.peer_specs_from_table(table)
            monkeypatch.setenv("LOBES_MESH_KEY", join_key)

            opener_calls = []

            def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
                opener_calls.append(True)
                raise AssertionError("must never dial when both members disagree")

            monkeypatch.setattr(S, "open_upstream", fake_open)

            def fake_replica_snapshot(backend_name):
                return ()

            resp = S.handle_post(
                table,
                cfg,
                "/v1/chat/completions",
                [("Authorization", "Bearer sk-caller")],
                json.dumps({"model": "cortex"}).encode(),
                fake_open,
                peer_specs=specs,
                replica_snapshot=fake_replica_snapshot,
                mesh_snapshot=snap,
            )

            assert resp.status == 404
            assert len(opener_calls) == 0  # never silently picked one — zero dials
            body = json.loads(resp.body)
            assert body["error"]["type"] == "role_infeasible"
            assert not body["error"].get("hosted_by")
            assert set(body["error"]["suffixed_lanes"]) == {"cortex-nameA", "cortex-nameB"}
        finally:
            for m in members:
                m.stop()

    def test_suffixed_name_addresses_one_member_directly(self, monkeypatch):
        fp_a = _wiring_fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="NVFP4")
        fp_b = _wiring_fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="FP8")
        join_key = "sk-mesh-join-key"

        members = [_FakeMemberGateway(), _FakeMemberGateway()]
        try:
            for m in members:
                m.start(0)
            origins = []
            for m, fp in zip(members, (fp_a, fp_b)):
                port = m._server.server_address[1]
                origin = f"http://127.0.0.1:{port}"
                origins.append(origin)
                m.set_capabilities(
                    {
                        "cortex": {
                            "fingerprint": {
                                "served_id": fp.served_id,
                                "quantization": fp.quantization,
                                "max_model_len": fp.max_model_len,
                                "runtime": fp.runtime,
                            },
                            "ready": True,
                        },
                    }
                )

            member_names = ["nameA", "nameB"]
            _routes, snap = _setup_mesh(
                origins,
                join_key,
                announced_roles={
                    origins[0]: {"cortex": _wiring_role("cortex", fingerprint=fp_a)},
                    origins[1]: {"cortex": _wiring_role("cortex", fingerprint=fp_b)},
                },
                fingerprints={origins[0]: fp_a, origins[1]: fp_b},
                member_names=member_names,
            )

            env = {"PRIMARY_FEASIBLE": "false"}
            table, cfg = build_config(env)
            monkeypatch.setenv("LOBES_MESH_KEY", join_key)

            opener_calls = []

            def fake_open(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
                opener_calls.append({"backend": backend})
                from tests.test_mesh_routing_wiring import _FakeUpstream

                return _FakeUpstream(200, b'{"choices": [{"text": "ok"}]}')

            monkeypatch.setattr(S, "open_upstream", fake_open)

            def fake_replica_snapshot(backend_name):
                return ()

            resp = S.handle_post(
                table,
                cfg,
                "/v1/chat/completions",
                [("Authorization", "Bearer sk-caller")],
                json.dumps({"model": "cortex-nameB"}).encode(),
                fake_open,
                replica_snapshot=fake_replica_snapshot,
                mesh_snapshot=snap,
            )

            assert resp.status == 200
            assert len(opener_calls) == 1
            header_names = dict(resp.headers)
            assert header_names.get(MESH_MEMBER_HEADER) == "nameB"
        finally:
            for m in members:
                m.stop()


def test_capabilities_payload_lists_suffixed_lanes_for_disagreeing_role():
    """Criterion 1/2 via the /capabilities builder (roles.annotate_mesh_naming)."""
    from lobes.roles import annotate_mesh_naming

    fp_a = _fp(quantization="NVFP4")
    fp_b = _fp(quantization="FP8")
    snap = _two_member_snapshot(fp_a, fp_b)

    payload = {"cortex": {"model": "unsloth/Qwen3.8-27B-NVFP4", "loaded": False}}
    annotated = annotate_mesh_naming(payload, snap)

    assert set(annotated["cortex"]["suffixed_lanes"]) == {"cortex-nameA", "cortex-nameB"}
    assert "member" not in annotated["cortex"]  # never names one side of a disagreement


def test_capabilities_payload_noop_when_mesh_disabled():
    from lobes.roles import annotate_mesh_naming

    payload = {"cortex": {"model": "x"}}
    annotated = annotate_mesh_naming(payload, None)
    assert annotated == {"cortex": {"model": "x"}}


# ---------------------------------------------------------------------------
# t4: mesh-sourced hosted_by / members / ready / proxied on /capabilities
# ---------------------------------------------------------------------------


def _one_member_snapshot(*, role="cortex", ready=True, name="nameA", origin="http://a"):
    roster = _FakeRoster([(name, origin, 1.0)])
    ann = _ann(name, origin, {role: _role(role, fingerprint=_fp())})
    return build_snapshot(
        roster,
        announcements={origin: ann},
        verified_roles={origin: frozenset({role})},
        ready_roles={origin: frozenset({role})} if ready else None,
    )


def _two_agreeing_ready(*, role="cortex", ready_b=True):
    fp = _fp(quantization="NVFP4")
    roster = _FakeRoster([("nameA", "http://a", 1.0), ("nameB", "http://b", 1.0)])
    anns = {
        "http://a": _ann("nameA", "http://a", {role: _role(role, fingerprint=fp)}),
        "http://b": _ann("nameB", "http://b", {role: _role(role, fingerprint=fp)}),
    }
    ready = {"http://a": frozenset({role})}
    if ready_b:
        ready["http://b"] = frozenset({role})
    return build_snapshot(
        roster,
        announcements=anns,
        verified_roles={"http://a": frozenset({role}), "http://b": frozenset({role})},
        ready_roles=ready,
    )


def test_one_plain_origin_emits_hosted_by_ready_and_proxied():
    from lobes.roles import annotate_mesh_naming

    snap = _one_member_snapshot(ready=True)
    payload = {"cortex": {"model": "m", "loaded": False, "feasible": False, "ready": False}}
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    # hosted_by is string-equal to the member's announced origin in the roster
    assert entry["hosted_by"] == "http://a"
    assert entry["hosted_by"] == next(m.origin for m in snap.members if m.name == "nameA")
    assert entry["proxied"] is True
    assert entry["ready"] is True
    assert entry["member"] == "nameA"
    assert "members" not in entry
    assert entry["feasible"] is False


def test_one_plain_origin_ready_false_when_the_member_lane_is_not_ready():
    from lobes.roles import annotate_mesh_naming

    snap = _one_member_snapshot(ready=False)
    payload = {"cortex": {"model": "m", "loaded": False, "feasible": False, "ready": False}}
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    assert entry["hosted_by"] == "http://a"
    assert entry["proxied"] is True
    assert entry["ready"] is False


def test_two_plain_origins_emit_members_and_no_hosted_by():
    from lobes.roles import annotate_mesh_naming

    snap = _two_agreeing_ready()
    payload = {"cortex": {"model": "m", "loaded": False, "feasible": False, "ready": False}}
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    assert entry["members"] == ["nameA", "nameB"]
    assert "hosted_by" not in entry
    assert entry["proxied"] is True
    assert entry["ready"] is True
    assert entry["member"] == "nameA"


def test_two_plain_origins_drop_a_pre_existing_env_hosted_by():
    # annotate_peer_referrals (cite-don't-delete) may already have written a
    # hosted_by; with a POOL answer there is no single host to name, so the
    # mesh annotation must remove it rather than leave a stale single origin.
    from lobes.roles import annotate_mesh_naming

    snap = _two_agreeing_ready()
    payload = {
        "cortex": {
            "model": "m",
            "loaded": False,
            "feasible": False,
            "ready": False,
            "hosted_by": "http://stale-env-origin",
            "proxied": True,
        }
    }
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    assert "hosted_by" not in entry
    assert entry["members"] == ["nameA", "nameB"]


def test_mesh_hosted_by_overwrites_the_env_sourced_one_for_a_single_origin():
    from lobes.roles import annotate_mesh_naming

    snap = _one_member_snapshot(ready=True)
    payload = {
        "cortex": {
            "model": "m",
            "loaded": False,
            "feasible": False,
            "ready": False,
            "hosted_by": "http://stale-env-origin",
        }
    }
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    assert entry["hosted_by"] == "http://a"


def test_locally_hosted_role_gets_no_mesh_hosted_by_or_ready_override():
    from lobes.roles import annotate_mesh_naming

    snap = _one_member_snapshot(ready=True)
    payload = {"cortex": {"model": "m", "loaded": True, "feasible": True, "ready": False}}
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    assert entry == {"model": "m", "loaded": True, "feasible": True, "ready": False}


def test_mesh_disabled_payload_is_byte_identical_for_a_dropped_role():
    from lobes.roles import annotate_mesh_naming

    before = {"cortex": {"model": "m", "loaded": False, "feasible": False, "ready": False}}
    expected = json.dumps(before)
    got = json.dumps(annotate_mesh_naming(json.loads(expected), None))
    assert got == expected


def test_no_plain_origins_leaves_the_dropped_role_untouched():
    # A disagreeing pair places nobody plain — no hosted_by, no members, no
    # proxied claim; only the suffixed_lanes listing the t8 contract already
    # pins.
    from lobes.roles import annotate_mesh_naming

    snap = _two_member_snapshot(_fp(quantization="NVFP4"), _fp(quantization="FP8"))
    payload = {"cortex": {"model": "m", "loaded": False, "feasible": False, "ready": False}}
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    assert "hosted_by" not in entry
    assert "members" not in entry
    assert "proxied" not in entry
    assert entry["ready"] is False


def test_compare_fingerprints_sanity_used_by_placement():
    # Documents the primitive compute_role_placement is built on, so a
    # future change to compare_fingerprints's semantics is visible here too.
    a = _replica_fp(quantization="NVFP4")
    b = _replica_fp(quantization="FP8")
    compatible, reason = compare_fingerprints(a, b)
    assert compatible is False
    assert "quantization" in reason


# ---------------------------------------------------------------------------
# t8 follow-up (c27/h1, c46/h37): GET /mesh/roster field population
# ---------------------------------------------------------------------------


def test_roster_list_populates_fields_after_announce_verify(monkeypatch):
    """A fake member announce -> verify -> roster read shows populated fields."""
    fp = _wiring_fp(served_id="unsloth/Qwen3.8-27B-NVFP4", quantization="NVFP4")
    join_key = "sk-mesh-join-key"

    member = _FakeMemberGateway()
    member.start(0)
    try:
        port = member._server.server_address[1]
        origin = f"http://127.0.0.1:{port}"
        member.set_capabilities(
            {
                "cortex": {
                    "fingerprint": {
                        "served_id": fp.served_id,
                        "quantization": fp.quantization,
                        "max_model_len": fp.max_model_len,
                        "runtime": fp.runtime,
                    },
                    "ready": True,
                },
            }
        )

        from lobes.gateway._mesh_config import build_mesh_config
        from lobes.gateway._mesh_roster import Roster
        from lobes.gateway._mesh_routes import MeshRoutes, verify_members
        from lobes.gateway._mesh_routing import SnapshotHolder

        mesh_cfg = build_mesh_config(
            {
                "LOBES_MESH_KEY": join_key,
                "LOBES_MESH_NAME": "me",
                "LOBES_MESH_SEEDS": "",
                "LOBES_MESH_HEARTBEAT_S": "60",
                "LOBES_MESH_MISSED_MAX": "3",
            }
        )
        roster = Roster()
        routes = MeshRoutes(mesh_cfg, roster)
        routes.roster.announce("nameA", origin, 4.0)
        from tests.test_mesh_routing_wiring import _ann

        routes._announcements[origin] = _ann(
            "nameA", origin, {"cortex": _wiring_role("cortex", fingerprint=fp)}
        )
        # Ledger: approve nameA so `expiry` is populated (not None).
        roster.approve("nameA", "operator", roster.now() + 3600.0)

        holder = SnapshotHolder(roster)
        routes._holder = holder
        verify_members(routes, holder, join_key=join_key, timeout=2.0)

        status, _headers, body = routes.roster_list(
            _fake_handler("/mesh/roster", "GET", headers={"Authorization": f"Bearer {join_key}"})
        )
        assert status == 200
        payload = json.loads(body)
        assert len(payload["members"]) == 1
        row = payload["members"][0]
        assert row["name"] == "nameA"
        assert row["origin"] == origin
        assert isinstance(row["last_seen_age"], (int, float))
        assert row["last_seen_age"] >= 0.0
        assert row["expiry"] == pytest.approx(roster.now() + 3600.0)
        assert row["verified"] is True
        assert row["flapping"] is False
        assert row["roles"] == ["cortex"]
    finally:
        member.stop()


def test_roster_list_expiry_none_and_unverified_without_ledger_entry():
    from lobes.gateway._mesh_config import build_mesh_config
    from lobes.gateway._mesh_roster import Roster
    from lobes.gateway._mesh_routes import MeshRoutes
    from lobes.gateway._mesh_routing import SnapshotHolder

    mesh_cfg = build_mesh_config(
        {"LOBES_MESH_KEY": "sk-test", "LOBES_MESH_NAME": "me", "LOBES_MESH_SEEDS": ""}
    )
    roster = Roster()
    routes = MeshRoutes(mesh_cfg, roster)
    routes.roster.announce("nameB", "http://b", 1.0)
    routes._holder = SnapshotHolder(roster)  # never populated -> no snapshot yet

    status, _headers, body = routes.roster_list(
        _fake_handler("/mesh/roster", "GET", headers={"Authorization": "Bearer sk-test"})
    )
    assert status == 200
    row = json.loads(body)["members"][0]
    assert row["expiry"] is None  # no ledger entry
    assert row["verified"] is False  # no snapshot yet
    assert row["flapping"] is False
    assert row["roles"] == []


def test_mesh_status_cli_renders_the_populated_roster_row(capsys):
    """lobes mesh status renders the real (non-golden-fixture) roster shape."""
    from lobes.cli._commands.mesh import _render_roster_table

    members = [
        {
            "name": "nameA",
            "origin": "http://a",
            "capacity": 4.0,
            "last_seen_age": 12.0,
            "expiry": 3599.0,
            "verified": True,
            "flapping": False,
            "roles": ["cortex"],
        },
        {
            "name": "nameB",
            "origin": "http://b",
            "capacity": 4.0,
            "last_seen_age": 5.0,
            "expiry": None,
            "verified": False,
            "flapping": False,
            "roles": [],
        },
    ]
    out = _render_roster_table(members)
    assert "nameA" in out
    assert "nameB" in out
    assert "12s" in out
    assert "3599s" in out
    assert "verified" in out
    assert "unverified" in out
    assert "cortex" in out


def test_sole_verified_member_is_plain_even_with_an_unknown_field():
    """Live dev528 (2026-09-12): the Thor was the only verified embedder in the
    mesh and the Spark hosts none, yet the Spark exposed it as ``embedder-thor``
    only — the strict pool rule made the sole candidate disagree with itself
    over ``quantization: unknown``. One candidate has nothing to pool with, so
    it is plain; two candidates keep the strict rule."""
    fp_unknown = _fp(quantization="unknown")
    snap = _two_member_snapshot(fp_unknown, fp_unknown)
    two = compute_role_placement(snap, "cortex", local_fingerprint=None)
    assert two.plain_origins == ()  # unknown never pools (spec h11), unchanged
    assert len(two.suffixed) == 2

    from lobes.gateway._mesh_routing import MemberInfo, RoutingSnapshot

    sole = RoutingSnapshot(
        members=tuple(m for m in snap.members if m.origin == "http://a"),
        announcements=tuple(a for a in snap.announcements if a[0] == "http://a"),
    )
    one = compute_role_placement(sole, "cortex", local_fingerprint=None)
    assert one.plain_origins == ("http://a",)
    assert one.suffixed == ()
    assert isinstance(sole.members[0], MemberInfo)


# ---------------------------------------------------------------------------
# Qodo thread 2: a proxied entry publishes the SERVING lane's context
# ---------------------------------------------------------------------------


def _one_member_with_context(context, *, role="cortex", origin="http://a"):
    roster = _FakeRoster([("nameA", origin, 1.0)])
    ann = _ann("nameA", origin, {role: _role(role, fingerprint=_fp())})
    return build_snapshot(
        roster,
        announcements={origin: ann},
        verified_roles={origin: frozenset({role})},
        ready_roles={origin: frozenset({role})},
        role_contexts={origin: {role: context}} if context is not None else None,
    )


def _two_with_contexts(ctx_a, ctx_b, *, role="cortex"):
    fp = _fp(quantization="NVFP4")
    roster = _FakeRoster([("nameA", "http://a", 1.0), ("nameB", "http://b", 1.0)])
    anns = {
        "http://a": _ann("nameA", "http://a", {role: _role(role, fingerprint=fp)}),
        "http://b": _ann("nameB", "http://b", {role: _role(role, fingerprint=fp)}),
    }
    contexts = {}
    if ctx_a is not None:
        contexts["http://a"] = {role: ctx_a}
    if ctx_b is not None:
        contexts["http://b"] = {role: ctx_b}
    return build_snapshot(
        roster,
        announcements=anns,
        verified_roles={"http://a": frozenset({role}), "http://b": frozenset({role})},
        ready_roles={"http://a": frozenset({role}), "http://b": frozenset({role})},
        role_contexts=contexts,
    )


def _proxied_payload(context=65536):
    return {
        "cortex": {
            "model": "m",
            "loaded": False,
            "feasible": False,
            "ready": False,
            "context": context,
        }
    }


def test_a_single_hosting_member_overwrites_the_local_context():
    """Qodo thread 2: the entry carried this box's own env-derived 65536 for a
    lane it does not host; the peer serves 262144 and that is what discovery
    clients must read."""
    from lobes.roles import annotate_mesh_naming

    snap = _one_member_with_context(262144)
    entry = annotate_mesh_naming(_proxied_payload(), snap)["cortex"]
    assert entry["context"] == 262144
    assert entry["hosted_by"] == "http://a"


def test_a_member_that_advertised_no_context_leaves_the_legacy_value():
    from lobes.roles import annotate_mesh_naming

    snap = _one_member_with_context(None)
    entry = annotate_mesh_naming(_proxied_payload(), snap)["cortex"]
    assert entry["context"] == 65536


def test_a_pool_publishes_the_context_its_members_agree_on():
    from lobes.roles import annotate_mesh_naming

    snap = _two_with_contexts(262144, 262144)
    entry = annotate_mesh_naming(_proxied_payload(), snap)["cortex"]
    assert entry["members"] == ["nameA", "nameB"]
    assert entry["context"] == 262144


def test_a_pool_whose_members_disagree_leaves_the_context_untouched():
    """No single honest window across the pool: publishing the first member's
    would be the same first-match guess this fix exists to remove."""
    from lobes.roles import annotate_mesh_naming

    snap = _two_with_contexts(262144, 131072)
    entry = annotate_mesh_naming(_proxied_payload(), snap)["cortex"]
    assert entry["members"] == ["nameA", "nameB"]
    assert entry["context"] == 65536


def test_a_pool_with_one_silent_member_leaves_the_context_untouched():
    from lobes.roles import annotate_mesh_naming

    snap = _two_with_contexts(262144, None)
    entry = annotate_mesh_naming(_proxied_payload(), snap)["cortex"]
    assert entry["context"] == 65536


def test_a_locally_hosted_role_never_takes_a_peers_context():
    from lobes.roles import annotate_mesh_naming

    snap = _one_member_with_context(262144)
    payload = {
        "cortex": {"model": "m", "loaded": True, "feasible": True, "ready": True, "context": 65536}
    }
    entry = annotate_mesh_naming(payload, snap)["cortex"]
    assert entry["context"] == 65536


def test_mesh_disabled_payload_with_a_context_is_byte_identical():
    from lobes.roles import annotate_mesh_naming

    expected = json.dumps(_proxied_payload())
    assert json.dumps(annotate_mesh_naming(json.loads(expected), None)) == expected


# ---------------------------------------------------------------------------
# The same rule for `model`: a proxied entry names the SERVING lane's model,
# not this box's own env-derived one (live 2026-09-19: the Thor and Orin
# advertised the Spark's senses as the retired 12B after the Spark moved to
# the 26B, because only `context` was overlaid).
# ---------------------------------------------------------------------------


def _members_with_models(*models, role="cortex"):
    fp = _fp(quantization="NVFP4")
    origins = [f"http://{c}" for c in "ab"[: len(models)]]
    roster = _FakeRoster([(f"name{o[-1].upper()}", o, 1.0) for o in origins])
    anns = {
        o: _ann(f"name{o[-1].upper()}", o, {role: _role(role, fingerprint=fp)}) for o in origins
    }
    return build_snapshot(
        roster,
        announcements=anns,
        verified_roles={o: frozenset({role}) for o in origins},
        ready_roles={o: frozenset({role}) for o in origins},
        role_models={o: {role: m} for o, m in zip(origins, models) if m is not None},
    )


def test_a_single_hosting_member_overwrites_the_local_model():
    from lobes.roles import annotate_mesh_naming

    snap = _members_with_models("vendor/new-26b")
    entry = annotate_mesh_naming(_proxied_payload(), snap)["cortex"]
    assert entry["model"] == "vendor/new-26b"
    assert entry["hosted_by"] == "http://a"


def test_a_member_that_advertised_no_model_leaves_the_local_value():
    from lobes.roles import annotate_mesh_naming

    snap = _members_with_models(None)
    assert annotate_mesh_naming(_proxied_payload(), snap)["cortex"]["model"] == "m"


def test_a_pool_publishes_the_model_its_members_agree_on():
    from lobes.roles import annotate_mesh_naming

    snap = _members_with_models("vendor/x", "vendor/x")
    entry = annotate_mesh_naming(_proxied_payload(), snap)["cortex"]
    assert entry["members"] == ["nameA", "nameB"]
    assert entry["model"] == "vendor/x"


def test_a_pool_whose_members_disagree_leaves_the_model_untouched():
    from lobes.roles import annotate_mesh_naming

    snap = _members_with_models("vendor/x", "vendor/y")
    assert annotate_mesh_naming(_proxied_payload(), snap)["cortex"]["model"] == "m"


def test_a_locally_hosted_role_never_takes_a_peers_model():
    from lobes.roles import annotate_mesh_naming

    snap = _members_with_models("vendor/new-26b")
    payload = {"cortex": {"model": "m", "loaded": True, "feasible": True, "ready": True}}
    assert annotate_mesh_naming(payload, snap)["cortex"]["model"] == "m"
