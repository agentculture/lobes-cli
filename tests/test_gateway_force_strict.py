"""Tests for GATEWAY_FORCE_STRICT_TOOLS (colleague#320, gateway force-strict
knob) — pure, no sockets. Follows the ``handle_post`` + injected-``opener``
pattern established in ``tests/test_gateway_server.py``.

Three layers:

1. Knob OFF: byte-identical passthrough (hard compatibility guarantee).
2. Knob ON: strict:true injection scoped to primary-lane chat-completions
   requests carrying a non-empty ``tools`` array; caller wins on an explicit
   ``strict``; every other request/lane/endpoint is untouched.
3. Retry-without-strict fallback: a 4xx/5xx matching the (heuristic, pending
   live discovery — plan risk r1) compile-failure signature retries once with
   the original un-injected body and relays that response verbatim; a
   non-matching failure, or a caller-set strict that was never injected, does
   not retry.
"""

from __future__ import annotations

import json

from lobes.gateway import server as S
from lobes.gateway._config import build_config

# --- shared fixtures / helpers ---------------------------------------------


def _cfg(**over):
    """A primary + multimodal + embed fleet table, mirroring test_gateway_server's
    _cfg() shape but with a multimodal/embed backend wired so the "other lane"
    tests have somewhere real to route to."""
    env = {
        "PRIMARY_SERVED_NAME": "P",
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": "M",
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": "E",
    }
    env.update(over)
    return build_config(env)


class _FakeUpstream:
    """Duck-typed stand-in for server._Upstream (no socket)."""

    def __init__(self, status, body=b'{"ok":1}'):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body
        self.closed = False

    def read_all(self):
        return self._body

    def read(self, _n):
        data, self._body = self._body, b""
        return data

    def close(self):
        self.closed = True


def _opener(behavior):
    """behavior: {backend_name: status_int | Exception | (status, body)}.
    Records (name, body) for every call, in order."""
    calls = []

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append((backend.name, body))
        outcome = behavior[backend.name]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, tuple):
            status, resp_body = outcome
            return _FakeUpstream(status, body=resp_body)
        return _FakeUpstream(outcome)

    return opener, calls


def _sequenced_opener(sequence):
    """A single-backend opener that yields ``sequence`` entries — (status,
    body) — one per call, in order. Every retry test only ever dials
    "primary" twice at most, so a shared per-call counter (not keyed by
    backend name) is sufficient."""
    calls = []
    state = {"i": 0}

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append((backend.name, body))
        status, resp_body = sequence[state["i"]]
        state["i"] += 1
        return _FakeUpstream(status, body=resp_body)

    return opener, calls


def _tools_body(model="P", extra_tool=None, stream=False):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    if extra_tool is not None:
        tools.append(extra_tool)
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "tools": tools}
    if stream:
        body["stream"] = True
    return json.dumps(body).encode()


_SIGNATURE_BODY = json.dumps(
    {"error": {"message": "xgrammar failed to compile json_schema for tool"}}
).encode()

_NO_SIGNATURE_BODY = json.dumps(
    {"error": {"message": "invalid request: missing required field 'foo'"}}
).encode()

_FSM_SIGNATURE_BODY = json.dumps(
    {"error": {"message": "Failed to advance FSM for grammar-constrained decoding"}}
).encode()


# --- config wiring: GATEWAY_FORCE_STRICT_TOOLS env parsing ------------------


def test_config_force_strict_tools_defaults_false() -> None:
    _, cfg = build_config({"PRIMARY_SERVED_NAME": "P"})
    assert cfg.force_strict_tools is False


def test_config_force_strict_tools_truthy_tokens_case_insensitive() -> None:
    for truthy in ("1", "true", "TRUE", "True", "yes", "YES"):
        _, cfg = build_config({"PRIMARY_SERVED_NAME": "P", "GATEWAY_FORCE_STRICT_TOOLS": truthy})
        assert cfg.force_strict_tools is True, truthy


def test_config_force_strict_tools_falsy_tokens() -> None:
    for falsy in ("0", "false", "no", "", "nah", "  "):
        _, cfg = build_config({"PRIMARY_SERVED_NAME": "P", "GATEWAY_FORCE_STRICT_TOOLS": falsy})
        assert cfg.force_strict_tools is False, falsy


# --- inject_strict_tools: pure body helper -----------------------------------


def test_inject_strict_tools_no_tools_field_returns_none() -> None:
    assert S.inject_strict_tools(b'{"model":"P"}') is None


def test_inject_strict_tools_empty_tools_array_returns_none() -> None:
    assert S.inject_strict_tools(b'{"model":"P","tools":[]}') is None


def test_inject_strict_tools_malformed_json_returns_none() -> None:
    assert S.inject_strict_tools(b"not json") is None


def test_inject_strict_tools_all_already_strict_returns_none() -> None:
    body = json.dumps(
        {"model": "P", "tools": [{"type": "function", "function": {"name": "x", "strict": False}}]}
    ).encode()
    assert S.inject_strict_tools(body) is None


def test_inject_strict_tools_injects_absent_and_reports_names() -> None:
    body = json.dumps(
        {
            "model": "P",
            "tools": [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "function", "function": {"name": "write_file", "strict": False}},
            ],
        }
    ).encode()
    result = S.inject_strict_tools(body)
    assert result is not None
    new_body, names = result
    assert names == ["read_file"]  # only the one actually modified
    data = json.loads(new_body)
    assert data["tools"][0]["function"]["strict"] is True
    assert data["tools"][1]["function"]["strict"] is False  # caller's explicit value untouched


# --- _matches_strict_failure_signature: the heuristic (plan risk r1) --------


def test_strict_failure_signature_matches_case_insensitively() -> None:
    assert S._matches_strict_failure_signature(b"XGrammar compile error") is True
    assert S._matches_strict_failure_signature(b"structural_tag violation") is True
    assert S._matches_strict_failure_signature(b"invalid json_schema") is True
    assert S._matches_strict_failure_signature(b"a generic grammar error") is True
    assert S._matches_strict_failure_signature(b"totally unrelated 400 bad request") is False


# --- 1. knob unset: byte-identical passthrough (hard guarantee) ------------


def test_knob_unset_tools_request_forwarded_unchanged() -> None:
    table, cfg = _cfg()
    assert cfg.force_strict_tools is False
    opener, calls = _opener({"primary": 200})
    body = _tools_body()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert resp.status == 200
    # Exactly what the pre-existing rewrite_model would forward — nothing else
    # touched the body.
    assert calls[0][1] == S.rewrite_model(body, "P")


def test_knob_unset_tool_less_request_forwarded_unchanged() -> None:
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 200})
    body = b'{"model":"P","messages":[{"role":"user","content":"hi"}]}'
    S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert calls[0][1] == S.rewrite_model(body, "P")


# --- 2. knob on: injection scoped to primary + chat-completions + tools ----


def test_knob_on_primary_lane_tools_all_get_strict_true() -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _opener({"primary": 200})
    body = _tools_body()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert resp.status == 200
    assert calls[0][0] == "primary"
    fwd = json.loads(calls[0][1])
    assert fwd["model"] == "P"
    assert fwd["tools"][0]["function"]["strict"] is True


def test_knob_on_caller_strict_false_is_preserved() -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    other_tool = {
        "type": "function",
        "function": {"name": "write_file", "parameters": {}, "strict": False},
    }
    body = _tools_body(extra_tool=other_tool)
    opener, calls = _opener({"primary": 200})
    S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    fwd = json.loads(calls[0][1])
    funcs = {t["function"]["name"]: t["function"] for t in fwd["tools"]}
    assert funcs["read_file"]["strict"] is True  # injected — was absent
    assert funcs["write_file"]["strict"] is False  # caller wins — never overwritten


def test_knob_on_tool_less_request_unaffected() -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _opener({"primary": 200})
    body = b'{"model":"P","messages":[{"role":"user","content":"hi"}]}'
    S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert calls[0][1] == S.rewrite_model(body, "P")


def test_knob_on_non_primary_lane_unaffected_by_served_name() -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _opener({"multimodal": 200})
    body = _tools_body(model="M")
    S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert calls[0][0] == "multimodal"
    assert calls[0][1] == S.rewrite_model(body, "M")


def test_knob_on_multimodal_alias_lane_unaffected() -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _opener({"multimodal": 200})
    body = _tools_body(model="multimodal")
    S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert calls[0][0] == "multimodal"
    fwd = json.loads(calls[0][1])
    assert fwd["model"] == "M"  # alias resolved
    assert "strict" not in fwd["tools"][0]["function"]  # not the primary lane


def test_knob_on_embeddings_endpoint_unaffected() -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _opener({"embed": 200})
    body = json.dumps(
        {
            "model": "E",
            "input": "hello",
            "tools": [{"type": "function", "function": {"name": "x"}}],
        }
    ).encode()
    S.handle_post(table, cfg, "/v1/embeddings", [], body, opener)
    assert calls[0][0] == "embed"
    assert calls[0][1] == S.rewrite_model(body, "E")  # path-gated: never chat-completions


def test_knob_on_streaming_success_still_relays_via_upstream() -> None:
    # A successful first attempt must not lose the streaming relay path.
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _opener({"primary": 200})
    body = _tools_body(stream=True)
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert resp.status == 200
    assert resp.streaming is True
    assert resp.upstream is not None
    assert json.loads(calls[0][1])["tools"][0]["function"]["strict"] is True


# --- 3. retry-without-strict fallback ---------------------------------------


def test_retry_fallback_signature_match_retries_and_returns_retry_response(capsys) -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _sequenced_opener([(400, _SIGNATURE_BODY), (200, b'{"ok":1}')])
    body = _tools_body()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert resp.status == 200
    assert len(calls) == 2
    assert [c[0] for c in calls] == ["primary", "primary"]
    # First call carried the injected body...
    assert json.loads(calls[0][1])["tools"][0]["function"]["strict"] is True
    # ...second call's body is the ORIGINAL un-injected bytes (post model-rewrite only).
    assert calls[1][1] == S.rewrite_model(body, "P")
    # A log line names the offending tool.
    err = capsys.readouterr().err
    assert "read_file" in err


def test_retry_fallback_non_matching_4xx_returned_as_is_single_call() -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _sequenced_opener([(400, _NO_SIGNATURE_BODY)])
    body = _tools_body()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert resp.status == 400
    assert len(calls) == 1
    assert (
        json.loads(resp.body)["error"]["message"] == "invalid request: missing required field 'foo'"
    )


def test_retry_fallback_500_fsm_grammar_signature_retried_once() -> None:
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _sequenced_opener([(500, _FSM_SIGNATURE_BODY), (200, b'{"ok":1}')])
    body = _tools_body()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert resp.status == 200
    assert len(calls) == 2
    assert calls[1][1] == S.rewrite_model(body, "P")


def test_caller_sent_strict_no_injection_never_retries() -> None:
    # The caller already declared "strict" on its only tool → inject_strict_tools
    # returns None (nothing modified) → the strict-retry path never engages, even
    # though the (single) upstream response matches the compile-failure signature.
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    caller_tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        },
    }
    body = json.dumps(
        {"model": "P", "messages": [{"role": "user", "content": "hi"}], "tools": [caller_tool]}
    ).encode()
    opener, calls = _sequenced_opener([(400, _SIGNATURE_BODY)])
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert len(calls) == 1  # no retry — this is the caller's own outcome
    assert resp.status == 400
    assert json.loads(calls[0][1])["tools"][0]["function"]["strict"] is True


def test_retry_fallback_non_matching_5xx_maps_to_owner_down_503() -> None:
    # A 5xx whose body does NOT match the strict-compile signature is not our
    # injection's fault — it must follow the gateway's documented owner-down
    # contract (retryable 503 backend_unavailable), not leak upstream 5xx
    # verbatim, and must NOT trigger the without-strict retry.
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _sequenced_opener([(502, _NO_SIGNATURE_BODY)])
    body = _tools_body()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert len(calls) == 1
    assert resp.status == 503
    payload = json.loads(resp.body)
    assert payload["error"]["type"] == "backend_unavailable"
    assert "primary: HTTP 502" in payload["error"]["attempts"]


def test_retry_fallback_retry_5xx_maps_to_owner_down_503() -> None:
    # Signature match → one retry without strict; the retry itself 5xx-ing IS
    # an owner-down condition — mapped to the retryable 503, never relayed.
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _sequenced_opener([(400, _SIGNATURE_BODY), (500, b"boom")])
    body = _tools_body()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert len(calls) == 2
    assert resp.status == 503
    payload = json.loads(resp.body)
    assert payload["error"]["type"] == "backend_unavailable"
    assert "primary: HTTP 500" in payload["error"]["attempts"]


def test_retry_fallback_connect_refused_on_first_attempt_yields_503() -> None:
    # A connection-level failure (no HTTP response at all) is not a "4xx/5xx
    # body matched the signature" case — it degrades exactly like the
    # existing owner-down path (retryable 503), never a retry-without-strict.
    table, cfg = _cfg(GATEWAY_FORCE_STRICT_TOOLS="1")
    opener, calls = _opener({"primary": S.UpstreamError("refused")})
    body = _tools_body()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener)
    assert len(calls) == 1
    assert resp.status == 503
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"
