"""serve()'s mesh wiring, exercised without binding a socket.

Regression for the 2026-09-12 live boot crash: serve() read ``cfg.self_origin``
while the attribute lives on the RoutingTable. serve() is ``pragma: no cover``,
so the only way to keep that class of typo out of a live box is to build the
wiring through a helper a test can call with a mesh-enabled config.
"""

from __future__ import annotations

import pytest

from lobes.gateway._config import build_config
from lobes.gateway._mesh_config import MeshConfigError
from lobes.gateway.server import build_mesh_wiring


def _env(**over: str) -> dict[str, str]:
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": "unsloth/Qwen3.8-27B-NVFP4",
        "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        "LOBES_MESH_KEY": "sk-test",
        "LOBES_MESH_NAME": "me",
    }
    env.update(over)
    return env


def test_mesh_enabled_builds_routes_and_holder_without_starting_the_thread() -> None:
    env = _env()
    table, cfg = build_config(env)
    routes, holder = build_mesh_wiring(table, cfg, None, {}, start=False, env=env)
    assert routes is not None and holder is not None
    assert routes._thread is None  # start=False never spawns the heartbeat
    assert holder.current() is not None


def test_mesh_disabled_wires_nothing() -> None:
    env = _env(LOBES_MESH_KEY="")
    table, cfg = build_config(env)
    assert build_mesh_wiring(table, cfg, None, {}, start=False, env=env) == (None, None)


def test_mesh_key_without_self_origin_refuses_with_a_named_error() -> None:
    env = _env(GATEWAY_SELF_ORIGIN="")
    table, cfg = build_config(env)
    with pytest.raises(MeshConfigError, match="GATEWAY_SELF_ORIGIN"):
        build_mesh_wiring(table, cfg, None, {}, start=False, env=env)


def test_capabilities_payload_accepts_the_holder_view_not_only_the_snapshot() -> None:
    """Regression: the holder publishes a MeshRoutingView; /capabilities crashed on it live."""
    from lobes.gateway._mesh_roster import Roster
    from lobes.gateway._mesh_routing import MeshRoutingView, as_routing_snapshot, build_snapshot
    from lobes.gateway.server import capabilities_payload

    env = _env()
    table, cfg = build_config(env)
    view = MeshRoutingView(snapshot=build_snapshot(Roster()), peer_states={})
    assert as_routing_snapshot(view) is view.snapshot
    assert as_routing_snapshot(None) is None
    payload = capabilities_payload(table, cfg, env, mesh_snapshot=view)
    assert "cortex" in payload


def test_reannounce_builder_is_wired_and_rebuilds_a_fresh_announcement() -> None:
    env = _env()
    table, cfg = build_config(env)
    routes, _ = build_mesh_wiring(table, cfg, None, {}, start=False, env=env)
    assert routes._announcement_builder is not None
    fresh = routes._announcement_builder()
    assert fresh.name == "me" and fresh.origin == "http://me.local:8000"
    assert "cortex" in fresh.roles


def test_announcement_is_the_hosted_slice_of_this_box_s_own_capabilities() -> None:
    """The first live cutover announced all six roles with empty served ids:
    the announcement must be built from /capabilities, hosted roles only."""
    from lobes.gateway.server import capabilities_payload

    env = _env()
    table, cfg = build_config(env)
    routes, _ = build_mesh_wiring(table, cfg, None, {}, start=False, env=env)
    ann = routes._announcement_builder()
    payload = capabilities_payload(table, cfg, env, mesh_snapshot=routes._holder.current().snapshot)
    hosted = {
        r
        for r, e in payload.items()
        if isinstance(e, dict) and e.get("feasible") and e.get("fingerprint")
    }
    assert set(ann.roles) == hosted and "cortex" in hosted
    fp = ann.roles["cortex"].fingerprint
    assert fp.served_id == "unsloth/Qwen3.8-27B-NVFP4"
    assert fp.runtime == "vllm"
    assert fp.served_id == payload["cortex"]["fingerprint"]["served_id"]
    assert (fp.max_model_len or 0) == int(
        payload["cortex"]["fingerprint"].get("max_model_len") or 0
    )


def test_verification_is_identity_so_unknown_matches_unknown() -> None:
    from lobes.gateway._mesh_routing import fingerprints_identical
    from lobes.gateway._replicas import Fingerprint as RF

    def rf(q: str) -> RF:
        return RF(
            served_id="Qwen/Qwen3-Reranker-0.6B",
            quantization=q,
            max_model_len=8192,
            runtime="vllm",
            kv_cache_dtype="unknown",
            reasoning_parser="unknown",
            tool_parser="unknown",
            speculative_config="unknown",
        )

    a, b, c = rf("unknown"), rf("unknown"), rf("none")
    assert fingerprints_identical(a, b)
    assert not fingerprints_identical(a, c)
    assert not fingerprints_identical(a, None)


def test_a_mesh_only_host_keeps_its_own_lane_in_the_pool() -> None:
    """Regression: with no env-declared pool there is no replica cache, so the
    local lane was never a candidate and every reranker request on the live
    Spark was forwarded to the Thor with reason 'sole-ready'."""
    from lobes.gateway._mesh_roster import Roster
    from lobes.gateway._mesh_routing import build_snapshot
    from lobes.gateway._mesh_wire import Announcement, Fingerprint, RoleInfo
    from lobes.gateway.server import _pool_selection

    env = {
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": "Qwen/Qwen3-Reranker-0.6B",
        "RERANK_QUANTIZATION": "none",
        "GATEWAY_SELF_ORIGIN": "http://me.local:8000",
        "LOBES_MESH_KEY": "sk-test",
        "LOBES_MESH_NAME": "me",
    }
    table, cfg = build_config(env)
    fp = Fingerprint(
        served_id="Qwen/Qwen3-Reranker-0.6B",
        quantization="none",
        max_model_len=8192,
        runtime="vllm",
    )
    ann = Announcement(
        name="peer",
        origin="http://peer.local:8000",
        schema_version="1",
        roles={
            "reranker": RoleInfo(
                model="Qwen/Qwen3-Reranker-0.6B",
                runtime="vllm",
                context=8192,
                quant="none",
                responsibilities=(),
                forbidden_responsibilities=(),
                fingerprint=fp,
            )
        },
    )
    roster = Roster()
    roster.announce("peer", "http://peer.local:8000", None, now=0.0)
    snap = build_snapshot(
        roster,
        announcements={"http://peer.local:8000": ann},
        verified_roles={"http://peer.local:8000": frozenset({"reranker"})},
    )
    placement = _pool_selection(
        table, "rerank", req_headers=[], replica_snapshot=None, mesh_snapshot=snap, local_busy=False
    )
    assert placement is not None
    origins = {c.origin for c in placement.candidates}
    assert "http://peer.local:8000" in origins
    assert "http://me.local:8000" in origins, "the hosting box's own lane must be a candidate"
    assert any(c.local for c in placement.candidates)
