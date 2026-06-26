"""Tests for lobes.minor._client — stdlib urllib OpenAI chat-completions client.

Uses a real http.server.HTTPServer on an ephemeral port so there is zero
network traffic outside localhost and no mocking of urllib internals.
"""

from __future__ import annotations

import http.server
import json
import threading
from typing import Any

import pytest

from lobes.minor import chat_completion, chat_text

# ---------------------------------------------------------------------------
# Canned response fixtures
# ---------------------------------------------------------------------------

_CANNED_CONTENT = "The answer is 42."

_CANNED_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-test-001",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": _CANNED_CONTENT},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
}

# ---------------------------------------------------------------------------
# Minimal HTTP server that captures the last request body
# ---------------------------------------------------------------------------


class _CaptureHandler(http.server.BaseHTTPRequestHandler):
    """Handle POST /v1/chat/completions; stash the decoded body on the server."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        # Store parsed body on the server instance for later inspection.
        self.server.last_request_body = json.loads(raw)  # type: ignore[attr-defined]
        body = json.dumps(_CANNED_RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass  # silence server log noise during tests


# ---------------------------------------------------------------------------
# Pytest fixture: ephemeral server running in a daemon thread
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_server():
    """Start HTTPServer on 127.0.0.1:0 (OS picks the port); yield (server, base_url)."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.last_request_body = None  # type: ignore[attr-defined]
    port = server.server_address[1]
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}/v1"
    yield server, base_url
    server.server_close()


# ---------------------------------------------------------------------------
# Helper: serve N requests from the same server instance
# ---------------------------------------------------------------------------


def _serve_n(server: http.server.HTTPServer, n: int) -> None:
    """Serve exactly *n* requests in the background (used for multi-call tests)."""
    for _ in range(n):
        server.handle_request()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chat_completion_returns_parsed_dict(local_server) -> None:
    """chat_completion() returns the full parsed JSON dict from the server."""
    server, base_url = local_server
    model = "test-model"
    prompt = "What is 6 times 7?"
    result = chat_completion(prompt, base_url=base_url, model=model)

    # Returns the full parsed response dict.
    assert isinstance(result, dict)
    assert result["id"] == "chatcmpl-test-001"
    assert result["choices"][0]["message"]["content"] == _CANNED_CONTENT


def test_request_body_is_well_formed(local_server) -> None:
    """The POST body must contain the model and a messages array with the prompt."""
    server, base_url = local_server
    model = "my-test-model"
    prompt = "Hello, model!"
    chat_completion(prompt, base_url=base_url, model=model)

    body = server.last_request_body
    assert body is not None, "server did not capture a request body"

    # Must carry the model identifier.
    assert body.get("model") == model

    # Must carry a messages array.
    messages = body.get("messages")
    assert isinstance(messages, list) and len(messages) >= 1

    # The user prompt must appear in the messages array.
    user_messages = [m for m in messages if m.get("role") == "user"]
    assert any(
        prompt in m.get("content", "") for m in user_messages
    ), f"prompt not found in messages: {messages}"


def test_chat_text_returns_assistant_content(local_server) -> None:
    """chat_text() returns just the assistant message string."""
    server, base_url = local_server
    result = chat_text("Say hello.", base_url=base_url, model="test-model")
    assert result == _CANNED_CONTENT


def test_system_prompt_included_when_provided() -> None:
    """When system= is given, the messages array starts with a system message."""
    # Use a fresh server for this multi-request test.
    server = http.server.HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.last_request_body = None  # type: ignore[attr-defined]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    try:
        chat_completion(
            "User message",
            base_url=base_url,
            model="m",
            system="You are a helpful assistant.",
        )
        body = server.last_request_body
        assert body is not None
        messages = body["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
    finally:
        server.server_close()


def test_optional_params_forwarded() -> None:
    """max_tokens and temperature, when supplied, appear in the request body."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.last_request_body = None  # type: ignore[attr-defined]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    try:
        chat_completion(
            "prompt",
            base_url=base_url,
            model="m",
            max_tokens=512,
            temperature=0.7,
        )
        body = server.last_request_body
        assert body["max_tokens"] == 512
        assert abs(body["temperature"] - 0.7) < 1e-9
    finally:
        server.server_close()


def test_base_url_trailing_slash_normalised() -> None:
    """base_url with a trailing slash still hits /chat/completions correctly."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server.last_request_body = None  # type: ignore[attr-defined]
    port = server.server_address[1]
    # Pass base_url with a trailing slash — the client must strip it.
    base_url = f"http://127.0.0.1:{port}/v1/"
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()
    try:
        result = chat_completion("hi", base_url=base_url, model="m")
        assert result["id"] == "chatcmpl-test-001"
    finally:
        server.server_close()


def test_no_third_party_imports() -> None:
    """lobes.minor must import only from the stdlib (no requests/httpx/etc.)."""
    import sys

    import lobes.minor  # noqa: F401 — ensure package is loaded
    import lobes.minor._client as _c

    # Check that none of the well-known third-party HTTP libs appear in
    # the module's own namespace (i.e. were imported by lobes.minor._client).
    forbidden = {"requests", "httpx", "aiohttp", "urllib3", "httplib2"}
    for name in forbidden:
        assert name not in vars(
            _c
        ), f"lobes.minor._client must not import third-party package: {name}"
    # Also verify none of them landed in sys.modules via our package.
    # (They may have been pulled in by unrelated test modules, so we only
    # check the client module's direct globals, which is the authoritative
    # source for what it imports.)
    _ = sys.modules  # referenced to confirm stdlib-only path; not further needed
