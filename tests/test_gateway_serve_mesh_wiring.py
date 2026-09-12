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
