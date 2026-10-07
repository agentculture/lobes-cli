"""Specialist embed/rerank lanes over the mesh (orin-embedding-specialist t8).

A lane a box hosts is a top-level ``/capabilities`` key (t7), so the box's
own announcement carries it like any role. These tests cover the RECEIVING
side: a member that does NOT host a lane learns it from a verified peer with
no per-lane env typed, forwards ``model=<lane>`` (or the raw catalog id) in
ONE hop with ``X-Lobes-Mesh-Member``, never pools two different checkpoints
under one lane name, answers 503 ``role_unverified`` in the boot window, and
lists the mesh-reached lane on its own ``/capabilities`` as proxied. An older
peer that does not know lane keys still verifies the announcer's real roles.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from lobes.embed_lanes import EMBED_LANES
from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_config import MeshConfig
from lobes.gateway._mesh_routes import announcement_from_capabilities
from lobes.gateway._mesh_routing import (
    MESH_MEMBER_HEADER,
    MemberInfo,
    RoutingSnapshot,
    lane_placement,
    verify_member_roles,
)
from lobes.gateway._mesh_wire import Fingerprint, decode, encode
from lobes.roles import ROLES
from tests.test_mesh_routing_wiring import _FakeUpstream

JOIN_KEY = "sk-mesh-join-key"
ORIN = "http://orin:8000"
THOR = "http://thor:8000"
LANE = next(lane for lane in EMBED_LANES if lane.name == "gemma2-embed")
RERANK_LANE = next(lane for lane in EMBED_LANES if lane.task == "score")


@pytest.fixture
def mesh_env(monkeypatch):
    monkeypatch.setenv("LOBES_MESH_KEY", JOIN_KEY)
    monkeypatch.setenv("LOBES_MESH_NAME", "spark")


def _lane_env(lane=LANE) -> dict[str, str]:
    return {f"{lane.name.upper().replace('-', '_')}_BASE_URL": "http://lane:8000"}


def _mesh_cfg(name: str) -> MeshConfig:
    return MeshConfig(
        enabled=True,
        key=JOIN_KEY,
        name=name,
        seeds=(),
        heartbeat_s=60,
        missed_max=3,
        ledger_path=None,
    )


def _hosting_payload(env: dict[str, str], origin: str = ORIN) -> dict:
    """The hosting box's own /capabilities, every lane ready."""
    table, cfg = build_config(env)
    ready = {b.name: True for b in table.backends}
    return S.capabilities_payload(
        table,
        cfg,
        env,
        gateway_url=origin,
        backend_ready=ready,
        # Mesh on (an empty roster): roles publish the fingerprint a peer verifies.
        mesh_snapshot=RoutingSnapshot(members=(), announcements=()),
    )


def _hosting_announcement(env: dict[str, str], name: str = "orin", origin: str = ORIN):
    payload = _hosting_payload(env, origin)
    cfg = _mesh_cfg(name)
    ann = announcement_from_capabilities(cfg, payload, self_origin=origin)
    # Over the wire and back, exactly as a peer receives it.
    return decode(encode(ann)), payload


def _probed(payload: dict) -> dict[str, dict]:
    return {
        name: {"fingerprint": entry.get("fingerprint"), "ready": entry.get("ready")}
        for name, entry in payload.items()
        if isinstance(entry, dict) and not entry.get("proxied")
    }


def _member(name, origin, ann, verified, *, probed=True, ready=None):
    return MemberInfo(
        name=name,
        origin=origin,
        announced_roles=tuple(ann.roles),
        verified_roles=tuple(sorted(verified)),
        capacity=1.0,
        probed=probed,
        ready_roles=tuple(sorted(ready if ready is not None else verified)),
    )


def _orin_snapshot(env=None, *, probed=True):
    ann, payload = _hosting_announcement(env or _lane_env())
    verified = verify_member_roles(ann, _probed(payload)) if probed else frozenset()
    member = _member("orin", ORIN, ann, verified, probed=probed)
    return RoutingSnapshot(members=(member,), announcements=((ORIN, ann),))


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        self.calls.append(
            {
                "base_url": backend.base_url,
                "path": path,
                "body": json.loads(fwd_body),
                "headers": list(headers),
            }
        )
        return _FakeUpstream(200, b'{"data": [{"embedding": [0.1]}]}')


def _post(table, cfg, model, rec, *, snap, path="/v1/embeddings", headers=()):
    return S.handle_post(
        table,
        cfg,
        path,
        [("Authorization", "Bearer sk-caller"), *headers],
        json.dumps({"model": model, "input": "hello"}).encode(),
        rec,
        replica_snapshot=lambda _name: (),
        mesh_snapshot=snap,
    )


def _hdr(resp, name):
    return [v for k, v in resp.headers if k.lower() == name.lower()]


# --- announcing side -----------------------------------------------------------


def test_announcement_from_capabilities_includes_a_hosted_lane():
    ann, _payload = _hosting_announcement(_lane_env())
    assert LANE.name in ann.roles
    assert ann.roles[LANE.name].fingerprint.served_id == LANE.catalog_id
    assert ann.roles[LANE.name].model == LANE.catalog_id


def test_hosted_lane_verifies_against_the_hosts_own_capabilities():
    ann, payload = _hosting_announcement(_lane_env())
    verified = verify_member_roles(ann, _probed(payload))
    assert LANE.name in verified


# --- receiving side: forward in one hop, no per-lane env -------------------------


@pytest.mark.parametrize("requested", [LANE.name, LANE.catalog_id])
def test_non_hosting_member_forwards_lane_in_one_hop(mesh_env, requested):
    table, cfg = build_config({})  # no per-lane env typed at all
    rec = _Recorder()
    resp = _post(table, cfg, requested, rec, snap=_orin_snapshot())

    assert resp.status == 200
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["base_url"].rstrip("/") == ORIN
    assert call["path"] == "/v1/embeddings"
    # The destination's own served id — a model id its gateway accepts.
    assert call["body"]["model"] == LANE.catalog_id
    assert any(k == S.PROXIED_HEADER for k, _ in call["headers"])  # single-hop marker
    auth = [v for k, v in call["headers"] if k.lower() == "authorization"]
    assert auth == [f"Bearer {JOIN_KEY}"]
    assert _hdr(resp, MESH_MEMBER_HEADER)[-1] == "orin"
    assert ORIN in _hdr(resp, S.PROXIED_BY_HEADER)


def test_rerank_lane_forwards_on_its_own_path(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    snap = _orin_snapshot(_lane_env(RERANK_LANE))
    resp = _post(table, cfg, RERANK_LANE.name, rec, snap=snap, path="/v1/rerank")
    assert resp.status == 200
    assert [c["path"] for c in rec.calls] == ["/v1/rerank"]
    assert rec.calls[0]["body"]["model"] == RERANK_LANE.catalog_id


def test_hop_marked_arrival_for_a_lane_is_never_forwarded_again(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(
        table, cfg, LANE.name, rec, snap=_orin_snapshot(), headers=[(S.PROXIED_HEADER, "x")]
    )
    assert rec.calls == []
    assert resp.status == 508


def test_pending_lane_answers_503_role_unverified(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, LANE.name, rec, snap=_orin_snapshot(probed=False))
    assert rec.calls == []
    assert resp.status == 503
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_unverified"
    assert body["error"]["hosted_by"] == ORIN


def test_lane_nobody_hosts_is_role_infeasible_never_the_default_model(mesh_env):
    # Gateway-only-ish: every core role dropped. Without the lane step the
    # peer-only pool resolves an unknown id to the default role and forwards
    # it as cortex — a lane request must never be answered by another model.
    env = {f"{p}_FEASIBLE": "false" for p in ("PRIMARY", "MULTIMODAL", "EMBED", "RERANK")}
    table, cfg = build_config(env)
    rec = _Recorder()
    ann, payload = _hosting_announcement({})  # a peer hosting ordinary roles only
    verified = verify_member_roles(ann, _probed(payload))
    snap = RoutingSnapshot(
        members=(_member("orin", ORIN, ann, verified),), announcements=((ORIN, ann),)
    )
    resp = _post(table, cfg, LANE.name, rec, snap=snap)
    assert rec.calls == []
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"


def test_a_different_checkpoint_under_the_lane_name_is_never_used(mesh_env):
    ann, payload = _hosting_announcement(_lane_env())
    info = ann.roles[LANE.name]
    wrong_fp = Fingerprint(
        served_id="someone/other-embedder",
        quantization=info.fingerprint.quantization,
        max_model_len=info.fingerprint.max_model_len,
        runtime=info.fingerprint.runtime,
    )
    forged = dataclasses.replace(
        ann,
        roles={
            **ann.roles,
            LANE.name: dataclasses.replace(
                info, model="someone/other-embedder", fingerprint=wrong_fp
            ),
        },
    )
    member = _member("orin", ORIN, forged, {LANE.name})
    snap = RoutingSnapshot(members=(member,), announcements=((ORIN, forged),))
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, LANE.name, rec, snap=snap)
    assert rec.calls == []
    assert resp.status == 404


def _two_host_snapshot(ann_orin, thor_info):
    ann_thor = dataclasses.replace(ann_orin, name="thor", origin=THOR, roles={LANE.name: thor_info})
    return RoutingSnapshot(
        members=(
            _member("orin", ORIN, ann_orin, {LANE.name}),
            _member("thor", THOR, ann_thor, {LANE.name}),
        ),
        announcements=((ORIN, ann_orin), (THOR, ann_thor)),
    )


def test_same_checkpoint_on_two_hosts_still_forwards(mesh_env):
    """Same served id = same vector space: a differing non-identity field never strands the lane."""
    ann_orin, _payload = _hosting_announcement(_lane_env())
    info = ann_orin.roles[LANE.name]
    other_fp = dataclasses.replace(info.fingerprint, max_model_len=1234)
    snap = _two_host_snapshot(ann_orin, dataclasses.replace(info, fingerprint=other_fp))
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, LANE.name, rec, snap=snap)
    assert resp.status == 200
    assert len(rec.calls) == 1


def test_a_member_serving_another_checkpoint_never_strands_the_correct_one(mesh_env):
    """Review finding: the served-id filter runs BEFORE any agreement check."""
    ann_orin, _payload = _hosting_announcement(_lane_env())
    info = ann_orin.roles[LANE.name]
    wrong_fp = dataclasses.replace(info.fingerprint, served_id="someone/other-embedder")
    snap = _two_host_snapshot(ann_orin, dataclasses.replace(info, fingerprint=wrong_fp))
    placement = lane_placement(snap, LANE.name, LANE.catalog_id)
    assert placement.plain_origins == (ORIN,)
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, LANE.name, rec, snap=snap)
    assert resp.status == 200
    assert len(rec.calls) == 1


def test_hosting_member_serves_its_own_lane_locally(mesh_env):
    env = _lane_env()
    table, cfg = build_config(env)
    rec = _Recorder()
    resp = _post(table, cfg, LANE.name, rec, snap=_orin_snapshot())
    assert resp.status == 200
    assert [c["base_url"] for c in rec.calls] == ["http://lane:8000"]
    assert rec.calls[0]["body"]["model"] == LANE.catalog_id


def test_wired_but_infeasible_lane_is_reached_through_the_mesh(mesh_env):
    env = {**_lane_env(), "GEMMA2_EMBED_FEASIBLE": "false"}
    table, cfg = build_config(env)
    rec = _Recorder()
    resp = _post(table, cfg, LANE.name, rec, snap=_orin_snapshot())
    assert resp.status == 200
    assert [c["base_url"].rstrip("/") for c in rec.calls] == [ORIN]
    payload = S.capabilities_payload(
        table, cfg, env, gateway_url="http://spark:8000", mesh_snapshot=_orin_snapshot()
    )
    assert payload[LANE.name]["proxied"] is True
    assert payload[LANE.name]["hosted_by"] == ORIN


def test_no_mesh_leaves_an_unwired_lane_model_not_found():
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, LANE.name, rec, snap=None)
    assert rec.calls == []
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "model_not_found"


# --- /capabilities on the non-hosting member -------------------------------------


def test_non_hosting_capabilities_lists_the_mesh_reached_lane_as_proxied(mesh_env):
    table, cfg = build_config({})
    payload = S.capabilities_payload(
        table, cfg, {}, gateway_url="http://spark:8000", mesh_snapshot=_orin_snapshot()
    )
    entry = payload[LANE.name]
    assert entry["lane"] is True
    assert entry["proxied"] is True
    assert entry["hosted_by"] == ORIN
    assert entry["model"] == LANE.catalog_id
    assert entry["dimension"] == LANE.dim
    assert entry["modalities"] == list(LANE.modalities)
    assert entry["ready"] is True
    assert entry["feasible"] is False
    assert entry["endpoint"] == "http://spark:8000"
    # An unannounced lane gets no key.
    assert "qwen3vl-embed" not in payload
    # ... and is never announced again by this member.
    ann = announcement_from_capabilities(_mesh_cfg("spark"), payload, self_origin="http://spark")
    assert LANE.name not in ann.roles


def test_capabilities_without_mesh_has_no_unwired_lane_key():
    table, cfg = build_config({})
    payload = S.capabilities_payload(table, cfg, {}, gateway_url="http://spark:8000")
    assert LANE.name not in payload


# --- mixed versions --------------------------------------------------------------


def test_older_peer_ignores_lane_keys_and_still_verifies_real_roles():
    """A peer whose code knows only ROLES decodes and verifies the announcer.

    Simulates the old path: the decoder is the shared wire decoder (it never
    filtered by name), and the older verifier only reads the probed entries
    whose name it knows. The lane key is ignored — not fatal — and every
    real role the announcer hosts still verifies.
    """
    ann, payload = _hosting_announcement(_lane_env())
    assert LANE.name in ann.roles
    old_probe = {k: v for k, v in _probed(payload).items() if k in ROLES}
    verified = verify_member_roles(ann, old_probe)
    real_roles = {r for r in ann.roles if r in ROLES}
    assert real_roles, "the announcer must host at least one real role"
    assert real_roles <= verified
    assert LANE.name not in verified


def test_announcement_with_unknown_lane_key_decodes_whole():
    ann, _payload = _hosting_announcement(_lane_env())
    raw = json.loads(encode(ann))
    raw["roles"]["some-future-lane"] = dataclasses.asdict(ann.roles[LANE.name])
    decoded = decode(json.dumps(raw).encode())
    assert "some-future-lane" in decoded.roles
    assert set(ann.roles) <= set(decoded.roles)
