"""Speech-to-text lanes over the mesh (stt-lanes).

The hosting box's ``/capabilities`` carries ``stt-he``/``stt-en`` as top-level
lane keys, so its announcement carries them and a peer verifies them exactly
like an embed lane. A member that does NOT host a lane forwards the multipart
body VERBATIM in one hop (no language injection on the way: the serving box
injects), answers 503 in the boot window, and 404s when nobody offers the lane.
``stt-he`` is gated on its pinned checkpoint; ``stt-en`` (unpinned) is not.
And no ``stt-*`` lane name can ever be read as a ``{role}-{member}`` lane.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_routing import (
    MESH_MEMBER_HEADER,
    RoutingSnapshot,
    find_member_lane,
    lane_placement,
    member_lanes,
    verify_member_roles,
)
from lobes.gateway._mesh_wire import Fingerprint
from lobes.stt_lanes import lane_by_name
from tests.test_gateway_stt_lanes import CT, _body, _fields, _Rec
from tests.test_mesh_lanes import JOIN_KEY, THOR, _hosting_announcement, _member, _probed

PATH = "/v1/audio/transcriptions"
HE = lane_by_name("stt-he")


@pytest.fixture
def mesh_env(monkeypatch):
    monkeypatch.setenv("LOBES_MESH_KEY", JOIN_KEY)
    monkeypatch.setenv("LOBES_MESH_NAME", "spark")


def _thor_snapshot(env=None, *, probed=True, name="thor", origin=THOR):
    env = env if env is not None else {"STT_HE_BASE_URL": "http://stt-he:9002"}
    ann, payload = _hosting_announcement(env, name=name, origin=origin)
    verified = verify_member_roles(ann, _probed(payload)) if probed else frozenset()
    member = _member(name, origin, ann, verified, probed=probed)
    return RoutingSnapshot(members=(member,), announcements=((origin, ann),))


def _post(snap, body, *, env=None, headers=()):
    table, cfg = build_config(env or {})
    rec = _Rec()
    resp = S.handle_audio_request(
        table,
        cfg,
        {},
        PATH,
        [("Content-Type", CT), ("Authorization", "Bearer sk-caller"), *headers],
        body,
        rec,
        mesh_snapshot=snap,
    )
    return resp, rec


def _hdr(resp, name):
    return [v for k, v in resp.headers if k.lower() == name.lower()]


# --- announcing side -------------------------------------------------------------


def test_hosting_box_announces_and_verifies_both_lanes():
    env = {
        "STT_HE_BASE_URL": "http://stt-he:9002",
        "STT_EN_BASE_URL": "http://stt:9002",
        "STT_MODEL": "ivrit-ai/whisper-large-v3-turbo",
        "STT_RUNTIME": "transformers",
    }
    ann, payload = _hosting_announcement(env)
    assert ann.roles["stt-he"].fingerprint.served_id == HE.catalog_id
    assert ann.roles["stt-he"].fingerprint.runtime == "sherpa-onnx"
    assert ann.roles["stt-en"].fingerprint.served_id == "ivrit-ai/whisper-large-v3-turbo"
    assert {"stt-he", "stt-en"} <= verify_member_roles(ann, _probed(payload))


def test_an_infeasible_lane_is_not_announced():
    ann, _ = _hosting_announcement({"STT_HE_BASE_URL": "http://x:1", "STT_HE_FEASIBLE": "false"})
    assert "stt-he" not in ann.roles


# --- receiving side ---------------------------------------------------------------


def test_non_hosting_member_forwards_the_body_verbatim_in_one_hop(mesh_env):
    body = _body(model="stt-he")
    resp, rec = _post(_thor_snapshot(), body)
    assert resp.status == 200
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["base_url"].rstrip("/") == THOR
    assert call["path"] == PATH
    assert call["body"] == body  # no language injected on the forwarding box
    assert "language" not in _fields(call["body"])
    assert any(k == S.PROXIED_HEADER for k, _ in call["headers"])
    assert [v for k, v in call["headers"] if k.lower() == "authorization"] == [f"Bearer {JOIN_KEY}"]
    assert _hdr(resp, MESH_MEMBER_HEADER)[-1] == "thor"
    assert THOR in _hdr(resp, S.PROXIED_BY_HEADER)
    assert _hdr(resp, S.LANE_HEADER) == ["stt-he"]


def test_header_selector_travels_with_the_forward(mesh_env):
    resp, rec = _post(_thor_snapshot(), _body(), headers=[("X-Lobes-Model", "stt-he")])
    assert resp.status == 200
    assert ("X-Lobes-Model", "stt-he") in rec.calls[0]["headers"]


def test_a_local_lane_is_never_forwarded(mesh_env):
    resp, rec = _post(
        _thor_snapshot(), _body(model="stt-he"), env={"STT_HE_BASE_URL": "http://stt-he:9002"}
    )
    assert resp.status == 200
    assert rec.calls[0]["base_url"] == "http://stt-he:9002"


def test_infeasible_here_falls_through_to_the_mesh(mesh_env):
    env = {"STT_HE_BASE_URL": "http://stt-he:9002", "STT_HE_FEASIBLE": "false"}
    resp, rec = _post(_thor_snapshot(), _body(model="stt-he"), env=env)
    assert resp.status == 200
    assert rec.calls[0]["base_url"].rstrip("/") == THOR


def test_hop_marked_arrival_is_refused_508(mesh_env):
    resp, rec = _post(
        _thor_snapshot(), _body(model="stt-he"), headers=[(S.PROXIED_HEADER, "stt-he")]
    )
    assert resp.status == 508
    assert rec.calls == []


def test_pending_member_is_503_role_unverified(mesh_env):
    resp, rec = _post(_thor_snapshot(probed=False), _body(model="stt-he"))
    assert resp.status == 503
    err = json.loads(resp.body)["error"]
    assert err["type"] == "role_unverified"
    assert err["hosted_by"] == THOR
    assert rec.calls == []


def test_nobody_offers_it_is_404_role_infeasible(mesh_env):
    snap = _thor_snapshot({"STT_EN_BASE_URL": "http://stt:9002"})
    resp, rec = _post(snap, _body(model="stt-he"))
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"
    assert rec.calls == []


def _with_served_id(snap, lane, served_id):
    origin, ann = snap.announcements[0]
    info = ann.roles[lane]
    fp = dataclasses.replace(info.fingerprint, served_id=served_id)
    roles = {**ann.roles, lane: dataclasses.replace(info, fingerprint=fp)}
    return RoutingSnapshot(
        members=snap.members,
        announcements=((origin, dataclasses.replace(ann, roles=roles)),),
    )


def test_stt_he_is_gated_on_its_pinned_checkpoint():
    snap = _with_served_id(_thor_snapshot(), "stt-he", "someone/else-he")
    assert lane_placement(snap, "stt-he", HE.catalog_id).plain_origins == ()


def test_stt_en_is_not_gated_on_a_checkpoint(mesh_env):
    snap = _thor_snapshot({"STT_EN_BASE_URL": "http://stt:9002", "STT_MODEL": "nvidia/x"})
    assert lane_placement(snap, "stt-en", None).plain_origins == (THOR,)
    resp, rec = _post(snap, _body(model="stt-en"))
    assert resp.status == 200
    assert rec.calls[0]["base_url"].rstrip("/") == THOR


# --- capabilities + /v1/models on the receiving side ---------------------------------


def test_mesh_reached_lane_is_advertised_proxied(mesh_env):
    table, cfg = build_config({})
    payload = S.capabilities_payload(
        table, cfg, {}, gateway_url="http://gw:8000", mesh_snapshot=_thor_snapshot()
    )
    entry = payload["stt-he"]
    assert entry["lane"] is True
    assert entry["proxied"] is True
    assert entry["hosted_by"] == THOR
    assert entry["language"] == "he"
    assert entry["ready"] is True
    assert "fingerprint" not in entry  # never re-announced by this box
    assert "stt-en" not in payload


def test_mesh_reached_lane_listed_on_v1_models_when_ready(mesh_env):
    table, _ = build_config({})
    assert S.stt_lane_model_ids(table, {}, _thor_snapshot()) == ("stt-he",)
    assert S.stt_lane_model_ids(table, {}, _thor_snapshot(probed=False)) == ()


# --- never a member lane -------------------------------------------------------------


def test_stt_lanes_are_never_member_lanes_even_for_members_named_he_or_en():
    snaps = [
        _thor_snapshot({"STT_HE_BASE_URL": "http://x:1"}, name="he", origin="http://he:8000"),
        _thor_snapshot({"STT_EN_BASE_URL": "http://x:1"}, name="en", origin="http://en:8000"),
    ]
    roles = S._member_lane_roles()
    assert "stt" not in roles
    for snap in snaps:
        for name in ("stt-he", "stt-en"):
            assert find_member_lane(snap, name, roles, self_name="spark") is None
        # The lane key is never read as the `stt` role on some member.
        assert member_lanes(snap, "stt", self_name="spark") == ()


def test_member_lane_ids_never_list_an_stt_lane(mesh_env):
    table, _ = build_config({"STT_HE_BASE_URL": "http://x:1"})
    snap = _thor_snapshot(name="he", origin="http://he:8000")
    assert not [i for i in S.member_lane_ids(table, snap) if i.startswith("stt")]


def test_json_model_field_never_reaches_an_stt_lane(mesh_env):
    """`model=stt-he` on chat/completions is not a lane request (path-routed)."""
    table, cfg = build_config({"STT_HE_BASE_URL": "http://stt-he:9002"})
    rec = _Rec()
    S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        [],
        json.dumps({"model": "stt-he", "messages": []}).encode(),
        rec,
        mesh_snapshot=_thor_snapshot(),
    )
    assert all(c["name"] != "stt-he" for c in rec.calls)
    assert all(c["base_url"].rstrip("/") != "http://stt-he:9002" for c in rec.calls)


def test_wire_fingerprint_shape_is_unchanged():
    assert [f.name for f in dataclasses.fields(Fingerprint)] == [
        "served_id",
        "quantization",
        "max_model_len",
        "runtime",
    ]
