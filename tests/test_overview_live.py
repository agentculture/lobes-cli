"""Tests for the live overview layer: metrics parsing, section builders, the CLI.

All offline — the HTTP probes in :mod:`model_gear._metrics` are monkeypatched, so
no sockets and no running deployment are needed.
"""

from __future__ import annotations

from model_gear import _metrics
from model_gear.cli import _live, main

SAMPLE = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:gpu_cache_usage_perc{engine="0",model_name="m"} 0.5
vllm:prompt_tokens_total{engine="0",model_name="m"} 100.0
vllm:generation_tokens_total{engine="0",model_name="m"} 40.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 5.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 2.0
"""


# --- parse_metrics ---------------------------------------------------------


def test_parse_metrics_full() -> None:
    m = _metrics.parse_metrics(SAMPLE)
    assert m["running"] == 2 and m["waiting"] == 1
    assert m["prompt_tokens"] == 100 and m["generation_tokens"] == 40
    assert m["requests_succeeded"] == 7
    assert m["by_finish_reason"] == {"stop": 5, "length": 2}
    assert m["kv_cache_usage"] == 0.5


def test_parse_metrics_sums_across_engines() -> None:
    text = 'vllm:num_requests_running{engine="0"} 1.0\nvllm:num_requests_running{engine="1"} 2.0\n'
    assert _metrics.parse_metrics(text)["running"] == 3


def test_parse_metrics_empty_and_no_kv() -> None:
    m = _metrics.parse_metrics("")
    assert m["running"] == 0 and m["by_finish_reason"] == {}
    assert "kv_cache_usage" not in m  # absent gauge → key omitted


def test_parse_metrics_skips_malformed_lines() -> None:
    m = _metrics.parse_metrics(
        'garbage no value\nvllm:num_requests_running{e="0"} 4.0\n# comment 9'
    )
    assert m["running"] == 4


# --- section builders (pure) ----------------------------------------------


def _fleet_status() -> dict:
    return {
        "object": "model-gear.fleet_status",
        "default_model": "P",
        "busy": {"running": 2, "waiting": 0},
        "backends": [
            {
                "name": "primary",
                "task": "generate",
                "served_name": "P",
                "health": "ok",
                "metrics": {
                    "running": 2,
                    "waiting": 0,
                    "prompt_tokens": 100,
                    "generation_tokens": 40,
                    "requests_succeeded": 7,
                    "by_finish_reason": {"stop": 7},
                },
            },
            {
                "name": "embed",
                "task": "embed",
                "served_name": "E",
                "health": "unreachable",
                "metrics": None,
            },
        ],
        "endpoints": ["GET /health", "POST /v1/chat/completions", "POST /v1/embeddings"],
    }


def test_fleet_sections_shape_and_content() -> None:
    secs = _live.fleet_sections(_fleet_status())
    assert [s["title"] for s in secs] == ["Online (live)", "Offered", "Busy", "Usage", "Endpoints"]
    online = "\n".join(secs[0]["items"])
    assert "primary (generate): ok" in online and "embed (embed): unreachable" in online
    offered = "\n".join(secs[1]["items"])
    assert "default model: P" in offered and "task families: embed, generate" in offered
    usage = "\n".join(secs[3]["items"])
    assert "prompt tokens: 100" in usage and "stop=7" in usage
    assert secs[4]["items"] == ["GET /health", "POST /v1/chat/completions", "POST /v1/embeddings"]


def test_single_sections_with_metrics() -> None:
    secs = _live.single_sections(8001, "M", healthy=True, metrics=_metrics.parse_metrics(SAMPLE))
    assert "M on :8001 — ok" in secs[0]["items"][0]
    assert "running: 2" in secs[2]["items"][0]
    assert "generation tokens: 40" in "\n".join(secs[3]["items"])


def test_single_sections_metrics_unavailable() -> None:
    secs = _live.single_sections(8000, None, healthy=True, metrics=None)
    assert "(model unknown" in secs[0]["items"][0]
    assert secs[2]["items"] == ["(metrics unavailable)"]


# --- live_sections probing wrapper ----------------------------------------


def test_live_sections_fleet(monkeypatch) -> None:
    monkeypatch.setattr(_metrics, "http_get_json", lambda url, **k: _fleet_status())
    secs = _live.live_sections(8000, None)
    assert secs[0]["title"] == "Online (live)" and "primary" in "\n".join(secs[0]["items"])


def test_live_sections_single(monkeypatch) -> None:
    monkeypatch.setattr(_metrics, "http_get_json", lambda url, **k: None)  # no gateway /status
    monkeypatch.setattr(_metrics, "health_ok", lambda base, **k: True)
    monkeypatch.setattr(
        _metrics, "http_get_text", lambda url, **k: 'vllm:num_requests_running{e="0"} 3.0'
    )
    secs = _live.live_sections(8001, "M")
    busy = "\n".join(s for sec in secs if sec["title"] == "Busy" for s in sec["items"])
    assert "running: 3" in busy


def test_live_sections_nothing_serving(monkeypatch) -> None:
    monkeypatch.setattr(_metrics, "http_get_json", lambda url, **k: None)
    monkeypatch.setattr(_metrics, "health_ok", lambda base, **k: False)
    secs = _live.live_sections(8000, None)
    assert secs[0]["title"] == "Live"
    assert "no model-gear endpoint reachable" in secs[0]["items"][0]


# --- the CLI verb ----------------------------------------------------------


def test_overview_live_cli_single(monkeypatch, capsys) -> None:
    monkeypatch.setattr(_metrics, "http_get_json", lambda url, **k: None)
    monkeypatch.setattr(_metrics, "health_ok", lambda base, **k: True)
    monkeypatch.setattr(
        _metrics, "http_get_text", lambda url, **k: 'vllm:generation_tokens_total{e="0"} 9.0'
    )
    assert main(["overview", "--live", "--port", "8000"]) == 0
    out = capsys.readouterr().out
    assert "model-gear (live)" in out and "Usage" in out and "generation tokens: 9" in out


def test_overview_live_cli_json(monkeypatch, capsys) -> None:
    import json

    monkeypatch.setattr(_metrics, "http_get_json", lambda url, **k: None)
    monkeypatch.setattr(_metrics, "health_ok", lambda base, **k: False)  # nothing serving
    assert main(["overview", "--live", "--port", "8000", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == "model-gear (live)"
    assert payload["sections"][0]["title"] == "Live"
