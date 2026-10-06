"""Member-lane dispatch through handle_post (mesh-pool-load-sharing, t4).

``{role}-{member}`` pins a request to ONE mesh member: a peer's name forwards
exactly once (and the destination serves a hop-marked arrival locally, so the
pin survives any pooling), this box's OWN name is served by its local lane
(with the plain role's pressure policy, never forwarded), a not-yet-probed
member answers 503 ``role_unverified``, and with the mesh disabled nothing
changes.
"""

from __future__ import annotations

import json

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_routing import MESH_MEMBER_HEADER, MemberInfo, RoutingSnapshot
from tests.test_mesh_naming import _fp, _two_member_snapshot
from tests.test_mesh_routing_wiring import _FakeUpstream

JOIN_KEY = "sk-mesh-join-key"
SPARK2 = "http://spark2:8000"
BUSY = {"swap_used_percent": 90.0, "iowait_percent": 0.0}


def _member(name, origin, *, verified=("cortex",), announced=None, probed=True):
    return MemberInfo(
        name=name,
        origin=origin,
        announced_roles=tuple(announced if announced is not None else verified),
        verified_roles=tuple(verified),
        capacity=1.0,
        probed=probed,
    )


def _snap(*members):
    return RoutingSnapshot(members=tuple(members), announcements=())


@pytest.fixture
def mesh_env(monkeypatch):
    monkeypatch.setenv("LOBES_MESH_KEY", JOIN_KEY)
    monkeypatch.setenv("LOBES_MESH_NAME", "spark")


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        self.calls.append(
            {"base_url": backend.base_url, "body": json.loads(fwd_body), "headers": list(headers)}
        )
        return _FakeUpstream(200, b'{"choices": [{"text": "ok"}]}')


def _post(table, cfg, model, opener, *, snap, headers=(), pressure=None):
    return S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        [("Authorization", "Bearer sk-caller"), *headers],
        json.dumps({"model": model, "messages": []}).encode(),
        opener,
        pressure=pressure,
        replica_snapshot=lambda _name: (),
        mesh_snapshot=snap,
    )


def _hdr(resp, name):
    return [v for k, v in resp.headers if k.lower() == name.lower()]


def _local_url(table):
    return next(b.base_url for b in table.backends if b.name == "primary")


# --- a peer's name: forwarded once, pinned ----------------------------------


def test_agreeing_peer_name_forwards_once_to_that_member(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, "cortex-spark2", rec, snap=_snap(_member("spark2", SPARK2)))

    assert resp.status == 200
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["base_url"].rstrip("/") == SPARK2
    # d1: a model id the DESTINATION accepts. With no announced served_id the
    # role name is used ("primary", the backend name, is unknown to every
    # gateway — measured live 2026-10-06: the destination answered 404).
    assert call["body"]["model"] == "cortex"
    auth = [v for k, v in call["headers"] if k.lower() == "authorization"]
    assert auth == [f"Bearer {JOIN_KEY}"]
    assert any(k == S.PROXIED_HEADER for k, _ in call["headers"])  # single-hop marker
    assert set(_hdr(resp, S.PROXIED_BY_HEADER)) == {SPARK2}  # stamped twice, as pre-t4
    assert _hdr(resp, MESH_MEMBER_HEADER)[-1] == "spark2"


def test_hand_peer_name_forwards_like_cortex(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    snap = _snap(_member("spark2", SPARK2, verified=("cortex", "hand")))
    resp = _post(table, cfg, "hand-spark2", rec, snap=snap)

    assert resp.status == 200
    assert [c["base_url"].rstrip("/") for c in rec.calls] == [SPARK2]
    assert rec.calls[0]["body"]["model"] == "hand"


def test_destination_serves_hop_marked_arrival_locally_even_when_pooled(mesh_env):
    """The pin survives pooling: a forwarded request is never re-placed."""
    table, cfg = build_config({})
    pooled = _two_member_snapshot(_fp(), _fp())  # agreeing → a real plain pool
    unmarked = S._pool_selection(
        table, "primary", [], replica_snapshot=None, mesh_snapshot=pooled, local_busy=False
    )
    assert unmarked is not None
    assert len(unmarked.candidates) >= 2  # the pool is armed
    placement = S._pool_selection(
        table,
        "primary",
        [(S.PROXIED_HEADER, "primary")],
        replica_snapshot=None,
        mesh_snapshot=pooled,
        local_busy=False,
    )
    assert placement is not None
    assert placement.selection.local is True
    assert placement.selection.reason == S.REASON_SOLE_READY


def test_hop_marked_arrival_naming_a_peer_lane_is_a_proxy_loop(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(
        table,
        cfg,
        "cortex-spark2",
        rec,
        snap=_snap(_member("spark2", SPARK2)),
        headers=[(S.PROXIED_HEADER, "primary")],
    )
    assert resp.status == S._PROXY_LOOP_STATUS
    assert rec.calls == []


# --- this box's own name: local, never forwarded -----------------------------


def test_self_name_is_served_locally_even_with_a_pool_peer(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, "cortex-spark", rec, snap=_snap(_member("spark2", SPARK2)))

    assert resp.status == 200
    assert [c["base_url"] for c in rec.calls] == [_local_url(table)]
    assert _hdr(resp, S.SERVED_BY_HEADER)
    assert _hdr(resp, MESH_MEMBER_HEADER)[-1] == "spark"
    assert not _hdr(resp, S.PROXIED_BY_HEADER)


def test_self_name_under_pressure_sheds_429_and_dials_nothing(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(
        table, cfg, "cortex-spark", rec, snap=_snap(_member("spark2", SPARK2)), pressure=BUSY
    )
    assert resp.status == 429
    assert json.loads(resp.body)["error"]["type"] == "server_busy"
    assert rec.calls == []


def test_hop_marked_self_name_is_served_locally_not_508(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(
        table,
        cfg,
        "cortex-spark",
        rec,
        snap=_snap(_member("spark2", SPARK2)),
        headers=[(S.PROXIED_HEADER, "primary")],
    )
    assert resp.status == 200
    assert [c["base_url"] for c in rec.calls] == [_local_url(table)]


def test_self_name_on_a_box_that_does_not_host_the_role_is_role_infeasible(mesh_env):
    table, cfg = build_config({"PRIMARY_FEASIBLE": "false"})
    rec = _Recorder()
    resp = _post(table, cfg, "cortex-spark", rec, snap=_snap(_member("spark2", SPARK2)))
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"
    assert rec.calls == []


# --- boot window, exclusions, mesh disabled -----------------------------------


def test_unprobed_member_name_is_503_role_unverified(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    snap = _snap(_member("spark2", SPARK2, verified=(), announced=("cortex",), probed=False))
    resp = _post(table, cfg, "cortex-spark2", rec, snap=snap)

    assert resp.status == 503
    assert _hdr(resp, "Retry-After") == ["5"]
    err = json.loads(resp.body)["error"]
    assert err["type"] == "role_unverified"
    assert err["hosted_by"] == SPARK2
    assert _hdr(resp, MESH_MEMBER_HEADER)[-1] == "spark2"
    assert any(k.lower().endswith("-unverified") for k, _ in resp.headers)
    assert rec.calls == []


@pytest.mark.parametrize("model", ["innereye-spark2", "stt-spark2", "tts-spark2"])
def test_unforwardable_and_audio_member_names_never_dial_a_peer(mesh_env, model):
    table, cfg = build_config({})
    rec = _Recorder()
    snap = _snap(_member("spark2", SPARK2, verified=("cortex", "innereye", "stt", "tts")))
    resp = _post(table, cfg, model, rec, snap=snap)
    assert resp.status == 404
    assert not any(c["base_url"].rstrip("/") == SPARK2 for c in rec.calls)


def test_unknown_member_name_is_model_not_found(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, "cortex-nobody", rec, snap=_snap(_member("spark2", SPARK2)))
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "model_not_found"
    assert rec.calls == []


def test_mesh_disabled_member_names_are_unknown_as_before(monkeypatch):
    monkeypatch.delenv("LOBES_MESH_KEY", raising=False)
    monkeypatch.delenv("LOBES_MESH_NAME", raising=False)
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _post(table, cfg, "cortex-spark2", rec, snap=None)
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "model_not_found"
    assert rec.calls == []


# --- GET /v1/models lists the routable member lanes (c30/h17) ----------------


def test_member_lane_ids_lists_self_and_verified_peers_only(mesh_env):
    table, _cfg = build_config({})
    snap = _snap(
        _member("spark2", SPARK2, verified=("cortex", "stt")),
        _member("thor", "http://thor:8000", verified=(), announced=("cortex",), probed=False),
    )
    ids = S.member_lane_ids(table, snap)
    assert "cortex-spark" in ids
    assert "cortex-spark2" in ids
    assert "cortex-thor" not in ids  # pending: 503 until probed, so never advertised
    assert not any(i.startswith(("stt-", "tts-", "innereye-")) for i in ids)


def test_member_lane_ids_empty_with_mesh_disabled():
    table, _cfg = build_config({})
    assert S.member_lane_ids(table, None) == ()


def test_v1_models_handler_lists_member_lanes(mesh_env):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from lobes.gateway._mesh_routing import MeshRoutingView

    table, cfg = build_config({})
    snap = _snap(_member("spark2", SPARK2))

    class _Holder:
        def current(self):
            return MeshRoutingView(snapshot=snap, peer_states={})

    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), S._make_handler(table, cfg, mesh_snapshot_holder=_Holder())
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/models"
        with urllib.request.urlopen(url, timeout=5) as resp:
            ids = [m["id"] for m in json.loads(resp.read())["data"]]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
    assert ids[-2:] == ["cortex-spark", "cortex-spark2"]


# --- GET /capabilities lists this box's own lane beside its peers (c22/h8) ---


def test_capabilities_lists_self_and_peer_member_lanes_on_a_hosting_box(mesh_env):
    table, cfg = build_config({})
    payload = S.capabilities_payload(table, cfg, {}, mesh_snapshot=_snap(_member("spark2", SPARK2)))
    cortex = payload["cortex"] if "cortex" in payload else payload["roles"]["cortex"]
    assert cortex["member_lanes"] == ["cortex-spark", "cortex-spark2"]


def test_capabilities_has_no_member_lanes_with_mesh_disabled(monkeypatch):
    monkeypatch.delenv("LOBES_MESH_KEY", raising=False)
    table, cfg = build_config({})
    payload = S.capabilities_payload(table, cfg, {})
    assert "member_lanes" not in json.dumps(payload)


# --- d1 (t6 live findings): outbound model + non-hosting box ------------------


def test_peer_lane_forwards_the_destinations_announced_served_id(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    snap = _two_member_snapshot(_fp(served_id="org/Model-A"), _fp(served_id="org/Model-B"))
    resp = _post(table, cfg, "cortex-nameB", rec, snap=snap)
    assert resp.status == 200
    assert [c["base_url"].rstrip("/") for c in rec.calls] == ["http://b"]
    assert rec.calls[0]["body"]["model"] == "org/Model-B"


def test_non_hosting_box_pins_a_peer_lane_instead_of_pooling_it(mesh_env):
    """Live Thor bug: _peer_only_forward resolved `cortex-nameB` to the default
    model and pool-forwarded it (to nameA), which then refused it with 508."""
    table, cfg = build_config({"PRIMARY_FEASIBLE": "false"})
    rec = _Recorder()
    snap = _two_member_snapshot(_fp(), _fp())  # agreeing → a real plain pool
    for target, origin in (("cortex-nameA", "http://a"), ("cortex-nameB", "http://b")):
        rec.calls.clear()
        resp = _post(table, cfg, target, rec, snap=snap)
        assert resp.status == 200, target
        assert [c["base_url"].rstrip("/") for c in rec.calls] == [origin], target
        assert _hdr(resp, S.ROUTE_REASON_HEADER)[-1] == "mesh-forwarded"


# --- #92 on /v1/models: a listed member lane must reach a live engine --------


def test_member_lane_ids_skip_a_self_lane_whose_local_backend_is_not_ready(mesh_env):
    """Live: spark listed `reranker-spark` with no reranker container running."""
    table, _cfg = build_config({})
    ready = {b.name: b.name != "rerank" for b in table.backends}
    ids = S.member_lane_ids(table, _snap(), ready=ready)
    assert "cortex-spark" in ids
    assert "reranker-spark" not in ids


def test_member_lane_ids_skip_a_peer_lane_whose_probe_was_not_ready(mesh_env):
    table, _cfg = build_config({})
    snap = _snap(
        MemberInfo(
            name="spark2",
            origin=SPARK2,
            announced_roles=("cortex", "hand"),
            verified_roles=("cortex", "hand"),
            ready_roles=("cortex",),
            capacity=1.0,
            probed=True,
        )
    )
    ids = S.member_lane_ids(table, snap, ready={b.name: True for b in table.backends})
    assert "cortex-spark2" in ids
    assert "hand-spark2" not in ids
