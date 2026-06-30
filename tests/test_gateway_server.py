"""Gateway server tests: handle_post failover decisions (no sockets) + a loopback
integration covering the handler relay (buffered + chunked streaming) and the
``open_upstream`` http.client path."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config


def _cfg(**over):
    env = {"PRIMARY_SERVED_NAME": "P", "FALLBACK_SERVED_NAME": "F", "GATEWAY_DEFAULT_MODEL": "P"}
    env.update(over)
    return build_config(env)


class _FakeUpstream:
    """Duck-typed stand-in for server._Upstream (no socket)."""

    def __init__(self, status, body=b'{"ok":1}', chunks=None):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body
        self._chunks = list(chunks) if chunks is not None else None
        self.closed = False

    def read_all(self):
        return self._body

    def read(self, _n):
        if self._chunks is None:
            data, self._body = self._body, b""
            return data
        return self._chunks.pop(0) if self._chunks else b""

    def close(self):
        self.closed = True


def _opener(behavior):
    """behavior: {backend_name: status_int | Exception}. Records (name, body)."""
    calls = []

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append((backend.name, body))
        outcome = behavior[backend.name]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeUpstream(outcome)

    return opener, calls


# --- handle_post: failover / default / rewrite (no sockets) ---------------


def test_failover_on_connection_refused() -> None:
    table, cfg = _cfg()
    opener, calls = _opener({"primary": S.UpstreamError("refused"), "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert [c[0] for c in calls] == ["primary", "fallback"]
    assert resp.status == 200 and resp.upstream is not None


def test_failover_on_5xx() -> None:
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 503, "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert [c[0] for c in calls] == ["primary", "fallback"]
    assert resp.status == 200


def test_no_failover_on_4xx() -> None:
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 400, "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert [c[0] for c in calls] == ["primary"]  # 4xx is a client error → returned verbatim
    assert resp.status == 400


def test_all_backends_down_returns_502() -> None:
    table, cfg = _cfg()
    opener, _ = _opener({"primary": S.UpstreamError("x"), "fallback": S.UpstreamError("y")})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert resp.status == 502 and resp.upstream is None
    assert json.loads(resp.body)["error"]["attempts"] == ["x", "y"]


def test_missing_model_routes_to_default() -> None:
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 200, "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b"{}", opener)
    assert calls[0][0] == "primary"  # default model's owner first
    assert resp.status == 200


def test_explicit_fallback_routes_to_fallback_first() -> None:
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 200, "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"F"}', opener)
    assert calls[0][0] == "fallback"
    assert resp.status == 200


def test_alias_model_is_rewritten_in_forwarded_body() -> None:
    table, cfg = _cfg(GATEWAY_ALIASES="fast=F")
    opener, calls = _opener({"fallback": 200, "primary": 200})
    S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"fast"}', opener)
    name, fwd_body = calls[0]
    assert name == "fallback"  # alias resolved → fallback owns it
    assert json.loads(fwd_body)["model"] == "F"  # body rewritten to the served name


def test_streaming_flag_propagates() -> None:
    table, cfg = _cfg()
    opener, _ = _opener({"primary": 200, "fallback": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"P","stream":true}', opener
    )
    assert resp.streaming is True


# --- loopback integration: the real handler relay + open_upstream ---------


@pytest.fixture
def gateway(monkeypatch):
    """A real ThreadingHTTPServer on an ephemeral port; open_upstream is stubbed
    so no real backend is needed. Yields the base URL."""
    table, cfg = _cfg()

    def fake_open(backend, path, body, headers, *, connect_timeout, read_timeout):
        if S.is_streaming(body):
            return _FakeUpstream(200, chunks=[b"data: a\n\n", b"data: b\n\n"])
        return _FakeUpstream(200, body=b'{"echo": "' + backend.name.encode() + b'"}')

    monkeypatch.setattr(S, "open_upstream", fake_open)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_integration_health_and_models(gateway) -> None:
    with urllib.request.urlopen(gateway + "/health", timeout=5) as r:
        assert r.status == 200
        assert json.load(r)["status"] == "ok"
    with urllib.request.urlopen(gateway + "/v1/models", timeout=5) as r:
        payload = json.load(r)
    assert [m["id"] for m in payload["data"]] == ["P", "F"]


def test_integration_supported_models(gateway) -> None:
    # The non-OpenAI discovery endpoint: the full supported catalog with flags.
    with urllib.request.urlopen(gateway + "/v1/models/supported", timeout=5) as r:
        assert r.status == 200
        payload = json.load(r)
    assert payload["object"] == "lobes.supported_models"
    assert payload["default_model"] == "P"  # the fixture's default served name
    assert len(payload["data"]) >= 1
    for entry in payload["data"]:
        assert {"id", "loaded", "default"} <= set(entry)
    # The OpenAI-standard /v1/models must stay standard — no catalog fields leak in.
    with urllib.request.urlopen(gateway + "/v1/models", timeout=5) as r:
        std = json.load(r)
    assert [m["id"] for m in std["data"]] == ["P", "F"]
    assert all(set(m) == {"id", "object", "owned_by"} for m in std["data"])


def test_integration_unknown_get_404(gateway) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(gateway + "/nope", timeout=5)
    assert exc.value.code == 404


def test_integration_buffered_post(gateway) -> None:
    req = urllib.request.Request(
        gateway + "/v1/chat/completions",
        data=b'{"model":"P"}',
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
        assert r.headers.get("Content-Length") is not None  # buffered → Content-Length
        assert json.load(r)["echo"] == "primary"


def test_integration_chunked_request_body_is_decoded(gateway) -> None:
    # A chunked request body (no Content-Length) must reach the backend intact,
    # not be forwarded as empty. The stub echoes the backend it routed to; with a
    # valid `model` the body must parse and route to the primary (default).
    host, port = gateway.removeprefix("http://").split(":")
    body = b'{"model":"P"}'
    chunked = b"%X\r\n%s\r\n0\r\n\r\n" % (len(body), body)
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
    ) + chunked
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(request)
        buf = b""
        while b"\r\n\r\n" not in buf or b'"echo"' not in buf:
            data = sock.recv(4096)
            if not data:
                break
            buf += data
    assert b"200" in buf.split(b"\r\n", 1)[0]
    assert b'"echo": "primary"' in buf  # body decoded → default route, not empty


def test_integration_streaming_post_is_chunked(gateway) -> None:
    # Raw socket so we can see the chunked framing on the wire (urllib would decode it).
    host, port = gateway.removeprefix("http://").split(":")
    body = b'{"model":"P","stream":true}'
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: %d\r\n\r\n" % len(body)
    ) + body
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(request)
        buf = b""
        while b"0\r\n\r\n" not in buf:  # read until the chunked terminator
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
    assert b"Transfer-Encoding: chunked" in buf
    assert b"data: a\n\n" in buf and b"data: b\n\n" in buf
    assert buf.rstrip().endswith(b"0")  # final zero-length chunk terminates the body


# --- open_upstream over a real loopback backend ---------------------------


class _Backend(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        code = 503 if self.path == "/boom" else 200
        body = b'{"served": true}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def backend():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Backend)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_open_upstream_success_and_5xx(backend) -> None:
    from lobes.gateway._routing import Backend

    b = Backend("primary", backend, "P")
    up = S.open_upstream(b, "/v1/chat/completions", b"{}", [], connect_timeout=2, read_timeout=5)
    assert up.status == 200
    assert json.loads(up.read_all())["served"] is True
    up.close()

    up = S.open_upstream(b, "/boom", b"{}", [], connect_timeout=2, read_timeout=5)
    assert up.status == 503  # returned (not raised) so handle_post can fail over
    up.close()


def test_open_upstream_refused_raises_upstream_error() -> None:
    from lobes.gateway._routing import Backend

    # Nothing is listening on this port → connect fails fast.
    b = Backend("primary", "http://127.0.0.1:1", "P")
    with pytest.raises(S.UpstreamError):
        S.open_upstream(b, "/x", b"{}", [], connect_timeout=1, read_timeout=2)


def test_open_upstream_malformed_url_raises_upstream_error() -> None:
    from lobes.gateway._routing import Backend

    # A non-numeric port makes urlsplit's .port raise ValueError — must surface as
    # UpstreamError (→ failover), not an uncaught 500.
    b = Backend("primary", "http://host:not-a-port", "P")
    with pytest.raises(S.UpstreamError):
        S.open_upstream(b, "/x", b"{}", [], connect_timeout=1, read_timeout=2)


# --- embed / rerank routing via handle_post (task-aware, no sockets) ---------

_EMBED_SERVED = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_SERVED = "Qwen/Qwen3-Reranker-0.6B"


def _task_cfg():
    """A table with primary (generate) + embed + rerank backends."""
    return build_config(
        {
            "PRIMARY_SERVED_NAME": "P",
            "EMBED_URL": "http://vllm-embed:8000",
            "EMBED_SERVED_NAME": _EMBED_SERVED,
            "RERANK_URL": "http://vllm-rerank:8000",
            "RERANK_SERVED_NAME": _RERANK_SERVED,
        }
    )


def test_handle_post_embeddings_path_routes_to_embed_backend() -> None:
    # POST /v1/embeddings with the embed served_name in the body must reach the
    # embed backend. /v1/embeddings is NOT special-cased — routing is by model name.
    table, cfg = _task_cfg()
    opener, calls = _opener({"embed": 200, "primary": 200})
    body = json.dumps({"model": _EMBED_SERVED, "input": "hello"}).encode()
    resp = S.handle_post(table, cfg, "/v1/embeddings", [], body, opener)
    assert resp.status == 200
    # embed backend was called, not the primary generate backend.
    assert calls[0][0] == "embed"
    # The forwarded body's model field is the embed backend's served_name.
    assert json.loads(calls[0][1])["model"] == _EMBED_SERVED


def test_handle_post_rerank_path_routes_to_rerank_backend() -> None:
    # POST /v1/rerank with the rerank served_name must reach the rerank backend.
    # /v1/rerank is NOT special-cased — routing is purely by model name.
    table, cfg = _task_cfg()
    opener, calls = _opener({"rerank": 200, "primary": 200})
    body = json.dumps({"model": _RERANK_SERVED, "query": "q", "documents": []}).encode()
    resp = S.handle_post(table, cfg, "/v1/rerank", [], body, opener)
    assert resp.status == 200
    # rerank backend was called, not the primary.
    assert calls[0][0] == "rerank"
    assert json.loads(calls[0][1])["model"] == _RERANK_SERVED


# --- pressure-aware tier downgrade + manual override (t6, #68) ----------------

from lobes.gateway._tier_request import PressureCache  # noqa: E402


def _fleet_cfg():
    """A full three-tier generate fleet with identifiable served names."""
    return build_config(
        {
            "PRIMARY_SERVED_NAME": "PRIMARY",
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MINOR_SERVED_NAME": "MINOR",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
            "MULTIMODAL_SERVED_NAME": "MULTIMODAL",
        }
    )


_HIGH_SWAP = {"swap_used_percent": 80.0, "iowait_percent": 0.0}  # > 75 → degraded/cheap
_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}


def test_handle_post_downgrades_hard_to_cheap_under_pressure() -> None:
    # model=hard under simulated high swap → forwarded to the cheap served name
    # (the minor gear) with X-Lobes-Tier-Reason: pressure.
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"hard"}', opener, pressure=_HIGH_SWAP
    )
    assert resp.status == 200
    assert calls[0][0] == "minor"  # cheap → minor backend
    assert json.loads(calls[0][1])["model"] == "MINOR"  # body rewritten to served name
    headers = dict(resp.headers)
    assert headers["X-Lobes-Tier"] == "cheap"
    assert headers["X-Lobes-Tier-Reason"] == "pressure"


def test_handle_post_override_forces_hard_under_pressure() -> None:
    # X-Lobes-Override forces the requested tier despite degraded pressure.
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        [],
        b'{"model":"hard"}',
        opener,
        pressure=_HIGH_SWAP,
        override=True,
    )
    assert resp.status == 200
    assert calls[0][0] == "primary"  # override → still the 27B
    assert json.loads(calls[0][1])["model"] == "PRIMARY"
    headers = dict(resp.headers)
    assert headers["X-Lobes-Tier"] == "hard"
    assert headers["X-Lobes-Tier-Reason"] == "manual_override"


def test_handle_post_no_pressure_keeps_hard_reason_default() -> None:
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"hard"}', opener, pressure=_NO_PRESSURE
    )
    assert calls[0][0] == "primary"
    headers = dict(resp.headers)
    assert headers["X-Lobes-Tier"] == "hard"
    assert headers["X-Lobes-Tier-Reason"] == "default"


def test_handle_post_plain_model_gets_no_tier_headers() -> None:
    # A concrete model id is never downgraded and carries no tier headers, even
    # under high pressure — the existing routing path is untouched.
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"PRIMARY"}', opener, pressure=_HIGH_SWAP
    )
    assert calls[0][0] == "primary"
    headers = dict(resp.headers)
    assert "X-Lobes-Tier" not in headers
    assert "X-Lobes-Tier-Reason" not in headers


def test_handle_post_without_pressure_skips_downgrade_layer() -> None:
    # pressure=None (no cache wired) → tier aliases resolve via the static table
    # (t5 behaviour), no tier headers, no downgrade.
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"hard"}', opener)
    assert calls[0][0] == "primary"  # hard → primary via static alias
    assert "X-Lobes-Tier" not in dict(resp.headers)


# --- loopback: tier headers are emitted BEFORE the streamed body --------------


@pytest.fixture
def tier_gateway(monkeypatch):
    """A real ThreadingHTTPServer wired with a PressureCache fixed at high swap.

    open_upstream is stubbed (no real backend). Streaming responses echo the
    backend the request routed to so the test can see the downgrade on the wire.
    """
    table, cfg = _fleet_cfg()
    cache = PressureCache(sampler=lambda: dict(_HIGH_SWAP), interval=1000, start=False)

    def fake_open(backend, path, body, headers, *, connect_timeout, read_timeout):
        tag = backend.name.encode()
        if S.is_streaming(body):
            return _FakeUpstream(200, chunks=[b"data: " + tag + b"\n\n", b"data: end\n\n"])
        return _FakeUpstream(200, body=b'{"echo": "' + tag + b'"}')

    monkeypatch.setattr(S, "open_upstream", fake_open)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg, cache))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        cache.stop()


def test_integration_streaming_downgrade_headers_precede_body(tier_gateway) -> None:
    # Raw socket so we can see the header block lands before the SSE data chunks.
    host, port = tier_gateway.removeprefix("http://").split(":")
    body = b'{"model":"hard","stream":true}'
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: %d\r\n\r\n" % len(body)
    ) + body
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(request)
        buf = b""
        while b"0\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
    head, _, _rest = buf.partition(b"\r\n\r\n")
    # Tier headers are in the HTTP header block (before the body separator).
    assert b"X-Lobes-Tier: cheap" in head
    assert b"X-Lobes-Tier-Reason: pressure" in head
    # And the downgrade actually happened on the wire: routed to the minor gear.
    assert b"data: minor\n\n" in buf
    # Header block precedes the first data chunk (no metadata smuggled into body).
    assert buf.index(b"X-Lobes-Tier-Reason") < buf.index(b"data: minor")


def test_integration_override_header_forces_hard(tier_gateway) -> None:
    host, port = tier_gateway.removeprefix("http://").split(":")
    body = b'{"model":"hard","stream":true}'
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: x\r\n"
        b"Content-Type: application/json\r\n"
        b"X-Lobes-Override: 1\r\n"
        b"Content-Length: %d\r\n\r\n" % len(body)
    ) + body
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        sock.sendall(request)
        buf = b""
        while b"0\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
    head, _, _rest = buf.partition(b"\r\n\r\n")
    assert b"X-Lobes-Tier: hard" in head
    assert b"X-Lobes-Tier-Reason: manual_override" in head
    assert b"data: primary\n\n" in buf  # override → the 27B primary on the wire
