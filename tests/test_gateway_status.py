"""Tests for the gateway ``/status`` fan-out (live fleet aggregate).

Pure — the per-backend probe is injected, so no sockets. The gateway is the only
place that can see the internal-only backends' /health + /metrics, so this is the
source for ``model overview --live`` in the fleet.
"""

from __future__ import annotations

from model_gear.gateway import server as S
from model_gear.gateway._config import ServerConfig
from model_gear.gateway._routing import Backend, RoutingTable


def _cfg(audio_url=None) -> ServerConfig:
    return ServerConfig(
        host="0.0.0.0", port=8000, connect_timeout=1.0, read_timeout=1.0, audio_url=audio_url
    )


def _table() -> RoutingTable:
    return RoutingTable(
        backends=(
            Backend("primary", "http://p:8000", "P", "generate"),
            Backend("embed", "http://e:8000", "E", "embed"),
            Backend("rerank", "http://r:8000", "R", "score"),
        ),
        default_model="P",
        aliases={},
    )


def test_endpoints_for_by_task_family() -> None:
    eps = S._endpoints_for(_table(), audio=False)
    assert "POST /v1/chat/completions" in eps
    assert "POST /v1/embeddings" in eps  # embed backend present
    assert "POST /v1/rerank" in eps and "POST /v1/score" in eps  # score backend present
    assert "GET /status" in eps
    assert all("audio" not in e for e in eps)


def test_endpoints_for_audio_toggle() -> None:
    eps = S._endpoints_for(_table(), audio=True)
    assert "POST /v1/audio/transcriptions" in eps and "POST /v1/audio/speech" in eps


def test_endpoints_for_generate_only() -> None:
    t = RoutingTable(
        backends=(Backend("primary", "http://p:8000", "P"),), default_model="P", aliases={}
    )
    eps = S._endpoints_for(t, audio=False)
    assert "POST /v1/embeddings" not in eps
    assert "POST /v1/rerank" not in eps


def test_fleet_status_payload_aggregates_busy_and_health() -> None:
    def fake_probe(base_url, *, timeout):
        if "p:8000" in base_url:
            return {
                "health": "ok",
                "metrics": {
                    "running": 2,
                    "waiting": 1,
                    "prompt_tokens": 100,
                    "generation_tokens": 40,
                    "requests_succeeded": 3,
                    "by_finish_reason": {"stop": 3},
                },
            }
        return {"health": "unreachable", "metrics": None}

    payload = S.fleet_status_payload(_table(), _cfg(), probe=fake_probe)
    assert payload["object"] == "model-gear.fleet_status"
    assert payload["default_model"] == "P"
    assert payload["busy"] == {"running": 2, "waiting": 1}  # only the reachable backend contributes
    assert [b["name"] for b in payload["backends"]] == ["primary", "embed", "rerank"]
    embed = next(b for b in payload["backends"] if b["name"] == "embed")
    assert embed["health"] == "unreachable" and embed["metrics"] is None
    primary = next(b for b in payload["backends"] if b["name"] == "primary")
    assert primary["task"] == "generate" and primary["served_name"] == "P"


def test_fleet_status_payload_all_unreachable() -> None:
    payload = S.fleet_status_payload(
        _table(), _cfg(), probe=lambda *a, **k: {"health": "unreachable", "metrics": None}
    )
    assert payload["busy"] == {"running": 0, "waiting": 0}
    assert all(b["health"] == "unreachable" for b in payload["backends"])
