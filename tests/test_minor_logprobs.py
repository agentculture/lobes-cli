"""Tests for logprobs extensions in lobes.minor._client.

Covers:
  - chat_completion() forwarding logprobs + top_logprobs and surfacing per-token data.
  - completions_echo() hitting /v1/completions with echo=true and returning token logprobs.
  - gateway_supports_echo() returning True on success and False on 404/error (never raises).

Uses a real http.server.HTTPServer on ephemeral ports — no network beyond
localhost, no mocking of urllib internals.
"""

from __future__ import annotations

import http.server
import json
import threading
from typing import Any

from lobes.minor import chat_completion
from lobes.minor._client import completions_echo, gateway_supports_echo

# ---------------------------------------------------------------------------
# Canned response fixtures
# ---------------------------------------------------------------------------

_CANNED_CHAT_LOGPROBS: dict[str, Any] = {
    "id": "chatcmpl-logprobs-001",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "42"},
            "finish_reason": "stop",
            "logprobs": {
                "content": [
                    {
                        "token": "42",
                        "logprob": -0.5,
                        "bytes": [52, 50],
                        "top_logprobs": [
                            {"token": "42", "logprob": -0.5, "bytes": [52, 50]},
                            {"token": " forty", "logprob": -1.2, "bytes": [32, 102]},
                            {"token": " 42", "logprob": -2.1, "bytes": [32, 52, 50]},
                            {"token": "two", "logprob": -3.3, "bytes": [116]},
                            {"token": "four", "logprob": -4.0, "bytes": [102]},
                        ],
                    }
                ]
            },
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
}

_CANNED_ECHO_RESPONSE: dict[str, Any] = {
    "id": "cmpl-echo-001",
    "object": "text_completion",
    "created": 1700000000,
    "model": "test-model",
    "choices": [
        {
            "text": "Ping pong",
            "index": 0,
            "logprobs": {
                "tokens": ["Ping", " pong"],
                "token_logprobs": [-0.3, -0.7],
                "top_logprobs": [
                    {"Ping": -0.3},
                    {" pong": -0.7},
                ],
            },
            "finish_reason": "length",
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
}

# ---------------------------------------------------------------------------
# Multi-path handler: routes /v1/chat/completions and /v1/completions
# ---------------------------------------------------------------------------


class _MultiPathHandler(http.server.BaseHTTPRequestHandler):
    """Serves both /v1/chat/completions (with logprobs) and /v1/completions (echo)."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        self.server.last_request_body = json.loads(raw)  # type: ignore[attr-defined]
        self.server.last_request_path = self.path  # type: ignore[attr-defined]

        if self.path == "/v1/chat/completions":
            body = json.dumps(_CANNED_CHAT_LOGPROBS).encode()
        elif self.path == "/v1/completions":
            body = json.dumps(_CANNED_ECHO_RESPONSE).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass  # silence server log noise during tests


# ---------------------------------------------------------------------------
# Handler that returns 404 for /v1/completions (gateway without echo support)
# ---------------------------------------------------------------------------


class _No404Handler(http.server.BaseHTTPRequestHandler):
    """Returns 404 for every request — simulates a gateway with no echo route."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # drain body
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass


# ---------------------------------------------------------------------------
# Helpers: spin up a server, serve N requests in a daemon thread
# ---------------------------------------------------------------------------


def _make_server(handler_cls) -> tuple[http.server.HTTPServer, str]:
    """Create an HTTPServer on an ephemeral port; return (server, base_url)."""
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    server.last_request_body = None  # type: ignore[attr-defined]
    server.last_request_path = None  # type: ignore[attr-defined]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    return server, base_url


def _serve_n(server: http.server.HTTPServer, n: int) -> threading.Thread:
    """Start a daemon thread that handles exactly *n* requests; return the thread."""

    def _run() -> None:
        for _ in range(n):
            server.handle_request()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Tests: chat_completion with logprobs + top_logprobs (AC 1)
# ---------------------------------------------------------------------------


def test_chat_completion_sends_logprobs_fields() -> None:
    """chat_completion sends logprobs=True and top_logprobs=N in the POST body."""
    server, base_url = _make_server(_MultiPathHandler)
    t = _serve_n(server, 1)
    try:
        chat_completion(
            "What is 6 times 7?",
            base_url=base_url,
            model="test-model",
            logprobs=True,
            top_logprobs=5,
        )
        t.join(timeout=5)
        body = server.last_request_body
        assert body is not None, "server did not capture a request body"
        assert body.get("logprobs") is True, f"logprobs not in body: {body}"
        assert body.get("top_logprobs") == 5, f"top_logprobs not in body: {body}"
    finally:
        server.server_close()


def test_chat_completion_returns_top_logprobs_in_response() -> None:
    """chat_completion returns per-token top_logprobs accessible via choices[0].logprobs.content."""
    server, base_url = _make_server(_MultiPathHandler)
    t = _serve_n(server, 1)
    try:
        result = chat_completion(
            "What is 6 times 7?",
            base_url=base_url,
            model="test-model",
            logprobs=True,
            top_logprobs=5,
        )
        t.join(timeout=5)
        content_logprobs = result["choices"][0]["logprobs"]["content"]
        assert isinstance(content_logprobs, list) and len(content_logprobs) > 0
        # The answer-position token "42" must appear in top_logprobs.
        first_token_top = content_logprobs[0]["top_logprobs"]
        assert isinstance(first_token_top, list) and len(first_token_top) > 0
        tokens_in_top = [t_entry["token"] for t_entry in first_token_top]
        assert "42" in tokens_in_top, f"'42' not found in top_logprobs tokens: {tokens_in_top}"
    finally:
        server.server_close()


def test_chat_completion_omits_logprobs_when_not_set() -> None:
    """When logprobs/top_logprobs are not passed, they must NOT appear in the POST body."""
    server, base_url = _make_server(_MultiPathHandler)
    t = _serve_n(server, 1)
    try:
        chat_completion("Hello", base_url=base_url, model="m")
        t.join(timeout=5)
        body = server.last_request_body
        assert "logprobs" not in body, f"logprobs unexpectedly in body: {body}"
        assert "top_logprobs" not in body, f"top_logprobs unexpectedly in body: {body}"
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Tests: completions_echo (AC 2)
# ---------------------------------------------------------------------------


def test_completions_echo_sends_correct_body() -> None:
    """completions_echo POSTs to /v1/completions with echo=true, max_tokens=0, combined prompt."""
    server, base_url = _make_server(_MultiPathHandler)
    t = _serve_n(server, 1)
    prompt = "Ping"
    continuation = " pong"
    try:
        completions_echo(prompt, continuation, base_url=base_url, model="test-model")
        t.join(timeout=5)
        body = server.last_request_body
        assert body is not None, "server did not capture a request body"
        assert (
            server.last_request_path == "/v1/completions"
        ), f"wrong path: {server.last_request_path}"
        assert body.get("echo") is True, f"echo not True in body: {body}"
        assert body.get("max_tokens") == 0, f"max_tokens not 0 in body: {body}"
        assert (
            body.get("prompt") == prompt + continuation
        ), f"prompt mismatch: {body.get('prompt')!r}"
    finally:
        server.server_close()


def test_completions_echo_returns_token_logprobs() -> None:
    """completions_echo returns token_logprobs and tokens for the continuation."""
    server, base_url = _make_server(_MultiPathHandler)
    t = _serve_n(server, 1)
    try:
        result = completions_echo("Ping", " pong", base_url=base_url, model="test-model")
        t.join(timeout=5)
        logprobs = result["choices"][0]["logprobs"]
        token_logprobs = logprobs["token_logprobs"]
        tokens = logprobs["tokens"]
        assert (
            isinstance(token_logprobs, list) and len(token_logprobs) > 0
        ), f"token_logprobs empty: {token_logprobs}"
        assert isinstance(tokens, list) and len(tokens) > 0, f"tokens empty: {tokens}"
        # Verify the canned values pass through faithfully.
        assert token_logprobs == [-0.3, -0.7]
        assert tokens == ["Ping", " pong"]
    finally:
        server.server_close()


def test_completions_echo_default_logprobs_param() -> None:
    """completions_echo sends logprobs=1 by default."""
    server, base_url = _make_server(_MultiPathHandler)
    t = _serve_n(server, 1)
    try:
        completions_echo("Hello", " world", base_url=base_url, model="m")
        t.join(timeout=5)
        body = server.last_request_body
        assert body.get("logprobs") == 1
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Tests: gateway_supports_echo (AC 3)
# ---------------------------------------------------------------------------


def test_gateway_supports_echo_returns_true_on_valid_response() -> None:
    """gateway_supports_echo returns True when /v1/completions returns valid logprobs."""
    server, base_url = _make_server(_MultiPathHandler)
    t = _serve_n(server, 1)
    try:
        result = gateway_supports_echo(base_url=base_url, model="test-model")
        t.join(timeout=5)
        assert result is True
    finally:
        server.server_close()


def test_gateway_supports_echo_returns_false_on_404() -> None:
    """gateway_supports_echo returns False (never raises) when the gateway returns 404."""
    server, base_url = _make_server(_No404Handler)
    t = _serve_n(server, 1)
    try:
        result = gateway_supports_echo(base_url=base_url, model="test-model")
        t.join(timeout=5)
        assert result is False
    finally:
        server.server_close()


def test_gateway_supports_echo_never_raises_on_connection_error() -> None:
    """gateway_supports_echo returns False on connection failure — never raises."""
    # Point at a port with nothing listening.
    result = gateway_supports_echo(
        base_url="http://127.0.0.1:1",  # port 1 will refuse connections
        model="test-model",
        timeout=2,
    )
    assert result is False
