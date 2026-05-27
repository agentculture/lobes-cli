"""Tests for the assess/benchmark probes (monkeypatched HTTP, no live server)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

import model_gear.assess as A
from model_gear.cli import main
from model_gear.cli._errors import ModelGearError


def _fake_get(url, path, timeout=10):
    if path == "/health":
        return 200, {"status": "ok"}
    if path == "/v1/models":
        return 200, {"data": [{"id": "foo/bar", "max_model_len": 32768}]}
    return 200, {}


def _fake_chat(reasoning_key="reasoning"):
    def _post(url, payload, timeout=300):
        prompt = payload["messages"][0]["content"]
        if payload.get("tool_choice") == "auto":
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": '{"summary": "hello"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"completion_tokens": 8, "prompt_tokens": 20},
            }
        if "17 * 23" in prompt:
            return {
                "choices": [
                    {
                        "message": {"content": "= 391", reasoning_key: "thinking"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 10, "prompt_tokens": 5},
            }
        if "train" in prompt:
            return {
                "choices": [
                    {
                        "message": {"content": "145 minutes", reasoning_key: "think"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 12, "prompt_tokens": 7},
            }
        # decode-throughput / prefill calls
        return {
            "choices": [{"message": {"content": "essay"}, "finish_reason": "length"}],
            "usage": {
                "completion_tokens": int(payload.get("max_tokens", 16)),
                "prompt_tokens": 2015,
            },
        }

    return _post


def test_run_correctness_passes_and_detects_reasoning(monkeypatch) -> None:
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _fake_chat("reasoning"))
    r = A.run_correctness("http://localhost:8000", None)
    assert r["passed"] is True
    assert r["model"] == "foo/bar"
    assert r["trace_field"] == "reasoning"
    md = A.render_correctness(r)
    assert "## Assessment" in md
    assert "PASS" in md


def test_run_correctness_detects_reasoning_content(monkeypatch) -> None:
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _fake_chat("reasoning_content"))
    r = A.run_correctness("http://localhost:8000", None)
    assert r["trace_field"] == "reasoning_content"


def test_run_correctness_tool_calling(monkeypatch) -> None:
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _fake_chat())
    r = A.run_correctness("http://localhost:8000", None, check_tools=True)
    assert r["passed"] is True  # content probes unaffected
    assert r["tool_calling"]["ok"] is True
    assert "finish" in r["tool_calling"]["tool_calls"]
    md = A.render_correctness(r)
    assert "tool calling" in md
    assert "PASS — called finish" in md


def test_run_correctness_skips_tool_probe_by_default(monkeypatch) -> None:
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _fake_chat())
    r = A.run_correctness("http://localhost:8000", None)
    assert r["tool_calling"] is None
    assert "tool calling" not in A.render_correctness(r)


def test_tool_probe_handles_400(monkeypatch) -> None:
    # A server without --enable-auto-tool-choice rejects tool_choice:auto with 400.
    monkeypatch.setattr(A, "_get", _fake_get)

    def _post(url, payload, timeout=300):
        if payload.get("tool_choice") == "auto":
            raise urllib.error.HTTPError(
                url,
                400,
                "Bad Request",
                {},
                io.BytesIO(
                    b'{"error":{"message":"\\"auto\\" tool choice requires '
                    b'--enable-auto-tool-choice and --tool-call-parser to be set"}}'
                ),
            )
        return _fake_chat()(url, payload, timeout)

    monkeypatch.setattr(A, "_post", _post)
    r = A.run_correctness("http://localhost:8000", None, check_tools=True)
    # graceful degrade: the run completes, tool probe reports failure with the message
    assert r["passed"] is True
    assert r["tool_calling"]["ok"] is False
    assert "HTTP 400" in r["tool_calling"]["error"]
    assert "FAIL" in A.render_correctness(r)


def test_run_benchmark(monkeypatch) -> None:
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _fake_chat())
    r = A.run_benchmark("http://localhost:8000", None, decode_tokens=64, runs=2)
    assert len(r["decode_rates"]) == 2
    assert r["prefill"]["prompt_tokens"] == 2015
    md = A.render_benchmark(r)
    assert "decode throughput" in md


def test_health_unreachable_raises(monkeypatch) -> None:
    def _bad_get(url, path, timeout=10):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(A, "_get", _bad_get)
    with pytest.raises(ModelGearError):
        A.run_correctness("http://localhost:8000", None)


def test_models_missing_id_raises_structured(monkeypatch) -> None:
    def _get(url, path, timeout=10):
        if path == "/health":
            return 200, {"status": "ok"}
        return 200, {"data": [{"max_model_len": 100}]}  # no 'id'

    monkeypatch.setattr(A, "_get", _get)
    with pytest.raises(ModelGearError) as exc:
        A.run_correctness("http://localhost:8000", None)
    assert "unexpected response shape" in exc.value.message


def test_chat_error_has_context(monkeypatch) -> None:
    monkeypatch.setattr(A, "_get", _fake_get)  # health + models OK

    def _boom(url, payload, timeout=300):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(A, "_post", _boom)
    with pytest.raises(ModelGearError) as exc:
        A.run_correctness("http://localhost:8000", None)
    assert "correctness probe" in exc.value.message


def test_assess_command_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _fake_chat())
    rc = main(["assess", "--port", "8000", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["host"]["image"] == "?"  # offline probe


def test_benchmark_command_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _fake_chat())
    rc = main(["benchmark", "--port", "8000", "--decode-tokens", "32"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Benchmark" in out
    assert "Host-side" in out
