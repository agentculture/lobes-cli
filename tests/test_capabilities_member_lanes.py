"""Tests for the member_lanes annotation on /capabilities (mesh-pool-load-sharing, t3).

Adds a sorted ``member_lanes`` list to every non-excluded role entry whose
members appear in the mesh snapshot — pending (unprobed) members are excluded,
and excluded roles (stt, tts, innereye) never get the key.
"""

from __future__ import annotations

import copy
import json

from lobes.gateway._mesh_routing import (
    MemberInfo,
    RoutingSnapshot,
    build_snapshot,
)
from lobes.gateway._mesh_wire import Announcement, RoleInfo
from lobes.gateway._replicas import Fingerprint as ReplicaFingerprint
from lobes.roles import annotate_mesh_naming

# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _fp(
    served_id="unsloth/Qwen3.8-27B-NVFP4",
    quantization="NVFP4",
    max_model_len=262144,
    runtime="vllm",
):
    return ReplicaFingerprint(
        served_id=served_id,
        max_model_len=max_model_len,
        runtime=runtime,
        quantization=quantization,
        kv_cache_dtype="",
        reasoning_parser="",
        tool_parser="",
        speculative_config="",
    )


def _role_info(name: str, fingerprint=None) -> RoleInfo:
    return RoleInfo(
        model="unsloth/Qwen3.8-27B-NVFP4",
        runtime="vllm",
        context=262144,
        quant="NVFP4",
        responsibilities=("generate",),
        forbidden_responsibilities=(),
        fingerprint=fingerprint or _fp(),
        capacity=1.0,
        private=False,
    )


def _ann(name: str, origin: str, roles: dict[str, RoleInfo]) -> Announcement:
    return Announcement(
        name=name,
        origin=origin,
        schema_version="1.0.0",
        roles=roles,
    )


class _FakeRoster:
    """Minimal Roster stand-in exposing exactly what build_snapshot reads."""

    def __init__(self, members):
        self._names = [m[0] for m in members]
        self._records = {}
        for name, origin, capacity in members:
            rec = type("Rec", (), {"name": name, "origin": origin, "capacity": capacity})()
            self._records[name] = rec
        self._roster = self._records

    def members(self):
        return list(self._names)


def _info(
    name: str,
    *,
    announced: tuple[str, ...] = (),
    verified: tuple[str, ...] = (),
    probed: bool = True,
) -> MemberInfo:
    return MemberInfo(
        name=name,
        origin=f"http://{name}",
        announced_roles=announced,
        verified_roles=verified,
        capacity=1.0,
        probed=probed,
    )


def _snap(*members: MemberInfo) -> RoutingSnapshot:
    return RoutingSnapshot(members=tuple(members), announcements=())


def _agreeing_snapshot(member_names, role="cortex", fp=None):
    """build_snapshot with matching fingerprints so placement agrees."""
    if fp is None:
        fp = _fp()
    roster = _FakeRoster([(n, f"http://{n}", 1.0) for n in member_names])
    anns = {
        f"http://{n}": _ann(n, f"http://{n}", {role: _role_info(n, fingerprint=fp)})
        for n in member_names
    }
    return build_snapshot(
        roster,
        announcements=anns,
        verified_roles={f"http://{n}": frozenset({role}) for n in member_names},
        ready_roles={f"http://{n}": frozenset({role}) for n in member_names},
    )


# ---------------------------------------------------------------------------
# AC1: agreeing members + self-hosting → member_lanes populated
# ---------------------------------------------------------------------------


def test_AC1_agreeing_self_hosting_member_lanes():
    """spark and spark2 agree on cortex; self_name='spark' hosts cortex.

    member_lanes == ['cortex-spark', 'cortex-spark2'] (sorted).
    suffixed_lanes does NOT gain entries (both agree — no disagreement).
    """
    snap = _agreeing_snapshot(["spark", "spark2"])
    payload = {"cortex": {"model": "x", "loaded": True, "feasible": True, "ready": True}}
    entry = annotate_mesh_naming(payload, snap, self_name="spark", hosted_roles={"cortex"})[
        "cortex"
    ]

    assert entry["member_lanes"] == ["cortex-spark", "cortex-spark2"]
    assert "suffixed_lanes" not in entry  # both agree — no disagreement


# ---------------------------------------------------------------------------
# AC2: pending member excluded from member_lanes
# ---------------------------------------------------------------------------


def test_AC2_pending_excluded():
    """A member announced but not probed (probed=False) never appears."""
    snap = _snap(
        _info("alpha", announced=("cortex",), verified=("cortex",)),
        _info("boot", announced=("cortex",), probed=False),
    )
    payload = {"cortex": {"model": "x", "loaded": False, "feasible": False, "ready": False}}
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    assert entry["member_lanes"] == ["cortex-alpha"]


# ---------------------------------------------------------------------------
# AC3: excluded roles (stt, tts, innereye) get no member_lanes key
# ---------------------------------------------------------------------------


def test_AC3_excluded_roles_no_member_lanes():
    """stt, tts, innereye never get a member_lanes key."""
    snap = _snap(
        _info("alpha", announced=("stt",), verified=("stt",)),
        _info("alpha", announced=("tts",), verified=("tts",)),
        _info("alpha", announced=("innereye",), verified=("innereye",)),
    )
    payload = {
        "stt": {"model": "x"},
        "tts": {"model": "x"},
        "innereye": {"model": "x"},
    }
    result = annotate_mesh_naming(payload, snap)

    for role in ("stt", "tts", "innereye"):
        assert "member_lanes" not in result[role]


# ---------------------------------------------------------------------------
# AC4: mesh_snapshot=None leaves payload byte-identical
# ---------------------------------------------------------------------------


def test_AC4_snapshot_none_byte_identical():
    """mesh_snapshot=None must return a payload that json-roundtrips identically."""
    payload = {
        "cortex": {"model": "m", "loaded": False, "feasible": False, "ready": False},
        "senses": {"model": "m"},
    }
    before = json.dumps(payload, sort_keys=True)
    got = json.dumps(annotate_mesh_naming(copy.deepcopy(payload), None), sort_keys=True)
    assert got == before


# ---------------------------------------------------------------------------
# AC5: role with no members in snapshot → no member_lanes key
# ---------------------------------------------------------------------------


def test_AC5_no_members_no_member_lanes():
    """A role whose member doesn't carry it gets no member_lanes."""
    snap = _snap(_info("alpha", announced=("senses",), verified=("senses",)))
    payload = {"cortex": {"model": "m"}}
    entry = annotate_mesh_naming(payload, snap)["cortex"]

    assert "member_lanes" not in entry
