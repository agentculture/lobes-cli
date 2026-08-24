"""Tests for the live overview layer: metrics parsing, section builders, the CLI.

All offline — the HTTP probes in :mod:`lobes._metrics` are monkeypatched, so
no sockets and no running deployment are needed.
"""

from __future__ import annotations

from lobes import _metrics
from lobes.cli import _live, main

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
    assert m["running"] == 2
    assert m["waiting"] == 1
    assert m["prompt_tokens"] == 100
    assert m["generation_tokens"] == 40
    assert m["requests_succeeded"] == 7
    assert m["by_finish_reason"] == {"stop": 5, "length": 2}
    assert m["kv_cache_usage"] == 0.5


def test_parse_metrics_sums_across_engines() -> None:
    text = 'vllm:num_requests_running{engine="0"} 1.0\nvllm:num_requests_running{engine="1"} 2.0\n'
    assert _metrics.parse_metrics(text)["running"] == 3


def test_parse_metrics_empty_and_no_kv() -> None:
    m = _metrics.parse_metrics("")
    assert m["running"] == 0
    assert m["by_finish_reason"] == {}
    assert "kv_cache_usage" not in m  # absent gauge → key omitted


def test_parse_metrics_skips_malformed_lines() -> None:
    m = _metrics.parse_metrics(
        'garbage no value\nvllm:num_requests_running{e="0"} 4.0\n# comment 9'
    )
    assert m["running"] == 4


def test_parse_metrics_skips_non_finite() -> None:
    # NaN/inf must be dropped, not crash the later int() (Qodo: best-effort contract).
    text = (
        'vllm:num_requests_running{e="0"} nan\n'
        'vllm:num_requests_running{e="1"} inf\n'
        'vllm:num_requests_running{e="2"} 2.0\n'
    )
    assert _metrics.parse_metrics(text)["running"] == 2


# --- parse_metrics: the non-vLLM (llama.cpp) backend path ------------------

# llama.cpp's server exposes its own ``llamacpp:*`` series — NOT ``vllm:*`` — so
# the vLLM parser reads a busy llama.cpp lane as all-zeros. Silent zeros
# presented as real numbers are the one unacceptable outcome (plan t6).
LLAMACPP_SAMPLE = """
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 100
llamacpp:tokens_predicted_total 40
llamacpp:requests_processing 2
llamacpp:requests_deferred 1
llamacpp:kv_cache_usage_ratio 0.5
"""


def test_parse_metrics_llamacpp_parses_its_own_series() -> None:
    m = _metrics.parse_metrics(LLAMACPP_SAMPLE)
    assert m["engine"] == "llamacpp"
    assert m["running"] == 2
    assert m["waiting"] == 1
    assert m["prompt_tokens"] == 100
    assert m["generation_tokens"] == 40
    assert m["kv_cache_usage"] == 0.5


def test_parse_metrics_llamacpp_busy_lane_never_reads_idle() -> None:
    # The acceptance criterion in prose: a busy llama.cpp backend must not
    # report running == 0 the way the vLLM-only parser did.
    assert _metrics.parse_metrics(LLAMACPP_SAMPLE)["running"] != 0


def test_parse_metrics_llamacpp_marks_finish_reasons_unsupported() -> None:
    # llama.cpp has no per-finish-reason success counter at all: the fields are
    # ABSENT and named in ``unsupported`` — never emitted as a zero.
    m = _metrics.parse_metrics(LLAMACPP_SAMPLE)
    assert "requests_succeeded" not in m
    assert "by_finish_reason" not in m
    assert m["unsupported"] == ["requests_succeeded", "by_finish_reason"]


def test_parse_metrics_llamacpp_absent_series_is_unknown_not_zero() -> None:
    # A build that does not export a series must read "unknown", not "0" —
    # distinguishing "0 because idle" from "unknown because unsupported".
    m = _metrics.parse_metrics("llamacpp:prompt_tokens_total 7\n")
    assert m["prompt_tokens"] == 7
    assert "running" not in m
    assert "waiting" not in m
    assert "kv_cache_usage" not in m
    assert set(m["unsupported"]) >= {"running", "waiting", "kv_cache_usage"}


def test_parse_metrics_unrecognised_engine_reports_no_numbers() -> None:
    m = _metrics.parse_metrics('someengine:num_requests_running{e="0"} 4.0\n')
    assert m["engine"] == "unknown"
    assert not any(k in m for k in ("running", "waiting", "prompt_tokens"))
    assert "running" in m["unsupported"]


def test_parse_metrics_vllm_output_is_unchanged() -> None:
    # Byte-identity guard: the vLLM dict grows no engine/unsupported keys.
    m = _metrics.parse_metrics(SAMPLE)
    assert "engine" not in m
    assert "unsupported" not in m
    assert _metrics.parse_metrics("") == {
        "running": 0,
        "waiting": 0,
        "prompt_tokens": 0,
        "generation_tokens": 0,
        "requests_succeeded": 0,
        "by_finish_reason": {},
    }


def test_parse_metrics_prefers_vllm_when_both_prefixes_present() -> None:
    text = 'vllm:num_requests_running{e="0"} 5.0\nllamacpp:requests_processing 1\n'
    m = _metrics.parse_metrics(text)
    assert m["running"] == 5
    assert "engine" not in m


# --- http_get_text body cap + probe short-circuit -------------------------


class _FakeResp:
    def __init__(self, data: bytes, status: int = 200) -> None:
        self._data, self.status = data, status

    def read(self, n: int = -1) -> bytes:
        return self._data[:n] if n and n > 0 else self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_get_text_caps_oversized_body(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=0: _FakeResp(b"x" * 1000))
    assert _metrics.http_get_text("http://x/m", max_bytes=100) is None  # over cap → unavailable
    assert _metrics.http_get_text("http://x/m", max_bytes=5000) == "x" * 1000


def test_probe_backend_short_circuits_when_unhealthy(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(_metrics, "health_ok", lambda base, **k: False)
    monkeypatch.setattr(_metrics, "http_get_text", lambda url, **k: calls.append(url))
    st = _metrics.probe_backend("http://dead:8000")
    assert st == {"health": "unreachable", "metrics": None}
    assert calls == []  # /metrics never fetched for a down backend


# --- section builders (pure) ----------------------------------------------


def _fleet_status() -> dict:
    return {
        "object": "lobes.fleet_status",
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
    assert "primary (generate): ok" in online
    assert "embed (embed): unreachable" in online
    offered = "\n".join(secs[1]["items"])
    assert "default model: P" in offered
    assert "task families: embed, generate" in offered
    usage = "\n".join(secs[3]["items"])
    assert "prompt tokens: 100" in usage
    assert "stop=7" in usage
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


# --- the live view must not present a llama.cpp lane as idle --------------


def test_single_sections_llamacpp_reports_busy_and_unknown() -> None:
    secs = _live.single_sections(
        8000, "G", healthy=True, metrics=_metrics.parse_metrics(LLAMACPP_SAMPLE)
    )
    busy = "\n".join(secs[2]["items"])
    usage = "\n".join(secs[3]["items"])
    # busy, not idle
    assert "running: 2" in busy
    assert "waiting: 1" in busy
    assert "prompt tokens: 100" in usage
    # No success counter on this engine → "unknown", never a fabricated 0.
    assert "requests succeeded: unknown" in usage
    assert "requests succeeded: 0" not in usage


def test_single_sections_unknown_engine_reports_unknown_everywhere() -> None:
    m = _metrics.parse_metrics("someengine:running 4.0\n")
    secs = _live.single_sections(8000, "X", healthy=True, metrics=m)
    busy = "\n".join(secs[2]["items"])
    assert "unknown" in busy
    assert "running: 0" not in busy


def _fleet_status_with_llamacpp() -> dict:
    st = _fleet_status()
    st["backends"].append(
        {
            "name": "cortex-llamacpp",
            "task": "generate",
            "served_name": "Q",
            "health": "ok",
            "metrics": _metrics.parse_metrics(LLAMACPP_SAMPLE),
        }
    )
    st["busy"] = {"running": 4, "waiting": 1, "partial": True}
    return st


def test_fleet_sections_flags_partial_usage_totals() -> None:
    secs = _live.fleet_sections(_fleet_status_with_llamacpp())
    online = "\n".join(secs[0]["items"])
    assert "cortex-llamacpp (generate): ok" in online
    assert "run 2 wait 1" in online
    busy = "\n".join(secs[2]["items"])
    assert "partial" in busy  # the aggregate is known-incomplete, and says so
    usage = "\n".join(secs[3]["items"])
    # prompt/generation tokens ARE reported by llama.cpp and fold into the total;
    # the success counter is not, so the total is flagged rather than implied whole.
    assert "prompt tokens: 200" in usage
    assert "requests_succeeded" in usage


def test_fleet_sections_all_vllm_has_no_partial_note() -> None:
    secs = _live.fleet_sections(_fleet_status())
    assert "partial" not in "\n".join(secs[2]["items"])
    assert len(secs[3]["items"]) == 2  # unchanged: no extra note line


# --- live_sections probing wrapper ----------------------------------------


def test_live_sections_fleet(monkeypatch) -> None:
    monkeypatch.setattr(_metrics, "http_get_json", lambda url, **k: _fleet_status())
    secs = _live.live_sections(8000, None)
    assert secs[0]["title"] == "Online (live)"
    assert "primary" in "\n".join(secs[0]["items"])


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
    assert "no lobes endpoint reachable" in secs[0]["items"][0]


# --- the CLI verb ----------------------------------------------------------


def test_overview_live_cli_single(monkeypatch, capsys) -> None:
    monkeypatch.setattr(_metrics, "http_get_json", lambda url, **k: None)
    monkeypatch.setattr(_metrics, "health_ok", lambda base, **k: True)
    monkeypatch.setattr(
        _metrics, "http_get_text", lambda url, **k: 'vllm:generation_tokens_total{e="0"} 9.0'
    )
    assert main(["overview", "--live", "--port", "8000"]) == 0
    out = capsys.readouterr().out
    assert "lobes (live)" in out
    assert "Usage" in out
    assert "generation tokens: 9" in out


def test_overview_live_cli_json(monkeypatch, capsys) -> None:
    import json

    monkeypatch.setattr(_metrics, "http_get_json", lambda url, **k: None)
    monkeypatch.setattr(_metrics, "health_ok", lambda base, **k: False)  # nothing serving
    assert main(["overview", "--live", "--port", "8000", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == "lobes (live)"
    assert payload["sections"][0]["title"] == "Live"
