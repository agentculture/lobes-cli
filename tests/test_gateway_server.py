"""Gateway server tests: handle_post routing/status decisions (no sockets, no
cross-backend failover since #91), the readiness-gated /v1/models advertisement
(#92), and a loopback integration covering the handler relay (buffered + chunked
streaming) and the ``open_upstream`` http.client path."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lobes import __version__
from lobes.gateway import server as S
from lobes.gateway._config import build_config


def _cfg(**over):
    env = {
        "PRIMARY_SERVED_NAME": "P",
        "FALLBACK_URL": "http://vllm-fallback:8000",
        "FALLBACK_SERVED_NAME": "F",
        "GATEWAY_DEFAULT_MODEL": "P",
    }
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


# --- handle_post: status matrix / default / rewrite (no sockets, no failover) -


def test_dead_owner_refused_yields_retryable_503() -> None:
    # issue #91 + #14 (t6): a dead primary must NOT retry against fallback (which
    # serves a different model — the forwarded body still names "P", and fallback
    # would either 404 on an unknown id or, worse, silently answer as the wrong
    # model). order_backends yields only the owner, so handle_post attempts primary
    # once and stops. The owner is the ONLY backend that can serve "P", so a
    # connection refusal is a TRANSIENT owner-down state → a retryable 503 +
    # Retry-After (type backend_unavailable), NEVER a terminal 404/502.
    table, cfg = _cfg()
    opener, calls = _opener({"primary": S.UpstreamError("refused"), "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert [c[0] for c in calls] == ["primary"]  # fallback is never dialed
    assert resp.status == 503 and resp.upstream is None
    headers = dict(resp.headers)
    assert headers["Retry-After"] == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"


def test_dead_owner_5xx_yields_retryable_503() -> None:
    # issue #91 + #14 (t6): a same-owner 5xx (e.g. EngineDeadError) is not failed
    # over to a backend serving a different model. The owner is the only backend
    # that can serve "P", so its 5xx is a transient owner-down state → the same
    # retryable 503 (backend_unavailable), not a 502.
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 503, "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert [c[0] for c in calls] == ["primary"]  # fallback is never dialed
    assert resp.status == 503
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"
    assert dict(resp.headers)["Retry-After"] == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)


def test_owner_4xx_relayed_verbatim_as_client_error() -> None:
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 400, "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert [c[0] for c in calls] == ["primary"]  # 4xx is a client error → returned verbatim
    assert resp.status == 400
    assert resp.upstream is not None  # relayed upstream, not a gateway-generated body


def test_owner_404_relayed_verbatim_not_converted_to_503() -> None:
    # A 404 from the owner is its authoritative verdict, relayed verbatim — NOT
    # converted to a 503. Since the owner is the ONLY backend that can serve this
    # model (#91), its "model does not exist" IS a client error, not a transient
    # backend outage. (Only a refusal / timeout / >=500 becomes the retryable 503.)
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 404})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert [c[0] for c in calls] == ["primary"]
    assert resp.status == 404
    assert resp.upstream is not None


def test_dead_owner_all_down_returns_503_with_single_attempt() -> None:
    # With no cross-backend failover (issue #91), only the owner is ever
    # attempted — fallback's outcome is irrelevant to a "P" request and is
    # never dialed. attempts therefore has a single entry, not one per backend.
    # t6: the dead owner yields a retryable 503 + Retry-After, never a terminal 502.
    table, cfg = _cfg()
    opener, calls = _opener({"primary": S.UpstreamError("x"), "fallback": S.UpstreamError("y")})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert [c[0] for c in calls] == ["primary"]  # fallback never dialed
    assert resp.status == 503 and resp.upstream is None
    assert json.loads(resp.body)["error"]["attempts"] == ["x"]
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"
    assert dict(resp.headers)["Retry-After"] == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)


def test_empty_order_backends_is_terminal_502() -> None:
    # The ONLY 502 left after issue #91/#14: order_backends returns an EMPTY list,
    # which can happen solely for a malformed routing table (no backend owns the
    # served model AND none owns default_model — here, a table with zero backends).
    # This is a config/deploy bug, not a transient outage, so it is a terminal 502
    # upstream_unavailable with NO Retry-After — distinct from the retryable 503 a
    # (present but) dead owner yields.
    from lobes.gateway._routing import RoutingTable

    table = RoutingTable(backends=(), default_model="P", aliases={})
    _, cfg = _cfg()
    opener, calls = _opener({})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"P"}', opener)
    assert calls == []  # nothing to dial — no owner exists
    assert resp.status == 502 and resp.upstream is None
    assert json.loads(resp.body)["error"]["type"] == "upstream_unavailable"
    assert "Retry-After" not in dict(resp.headers)  # terminal, not retryable


def test_missing_model_routes_to_default() -> None:
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 200, "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b"{}", opener)
    assert calls[0][0] == "primary"  # default model's owner first
    assert resp.status == 200


def test_explicit_fallback_routes_to_fallback_only() -> None:
    # RENAMED from test_explicit_fallback_routes_to_fallback_first (issue #91):
    # "first" implied there could be a second (a failover to primary) — there
    # no longer is. An explicit "F" request is attempted at fallback and
    # nowhere else, even though primary is healthy and configured in the table.
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 200, "fallback": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"F"}', opener)
    assert [c[0] for c in calls] == ["fallback"]  # primary is never dialed
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
        health = json.load(r)
    assert health["status"] == "ok"
    # issue #99: /health reports the deployed lobes-cli release so a remote
    # client (or `lobes doctor`) can detect artifact skew without docker.
    assert health["version"] == __version__
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


_HIGH_SWAP = {"swap_used_percent": 80.0, "iowait_percent": 0.0}  # > 75 → busy/shed
_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}


def test_handle_post_sheds_main_with_429_busy_under_pressure() -> None:
    # model=hard (back-compat alias for main) under simulated high swap → SHED
    # with a 429 busy response; the request is NOT forwarded to any backend
    # (degrade-to-minor is removed; #85). No upstream is dialed (h10).
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"hard"}', opener, pressure=_HIGH_SWAP
    )
    assert resp.status == 429
    assert resp.upstream is None
    assert calls == []  # h10: no upstream backend was dialed on the shed path
    headers = dict(resp.headers)
    assert headers["Retry-After"] == str(S.BUSY_RETRY_AFTER_SECONDS)
    assert headers["X-Lobes-Tier-Reason"] == "busy"
    body = json.loads(resp.body)
    assert body["error"]["type"] == "server_busy"
    assert body["error"]["code"] == "busy"
    assert "cortex" in body["error"]["message"]


def test_handle_post_sheds_senses_with_429_busy_under_pressure() -> None:
    # model=normal (multimodal/senses) is ALSO shed under pressure — not degraded
    # to minor. Busy applies to any cross-capability substitution (cortex + senses).
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"normal"}', opener, pressure=_HIGH_SWAP
    )
    assert resp.status == 429
    assert calls == []
    assert dict(resp.headers)["X-Lobes-Tier-Reason"] == "busy"
    body = json.loads(resp.body)
    assert "senses" in body["error"]["message"]


def test_handle_post_minor_still_served_under_pressure() -> None:
    # An explicit minor request is the floor — served as requested, never shed.
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"minor"}', opener, pressure=_HIGH_SWAP
    )
    assert resp.status == 200
    assert calls[0][0] == "minor"  # served, not shed
    headers = dict(resp.headers)
    assert headers["X-Lobes-Tier"] == "minor"
    assert headers["X-Lobes-Tier-Reason"] == "default"


def test_busy_429_is_distinguishable_from_503_owner_down() -> None:
    # The busy 429 (type server_busy, the pressure shed) and the owner-down 503
    # (type backend_unavailable) differ by status code AND error type — a client
    # can tell "the fleet is under pressure, back off the tier" from "this model's
    # own backend is down, retry it". Both are retryable but semantically distinct,
    # and both are distinct from the terminal 502 (malformed table) and from a
    # relayed upstream 404 ("model does not exist").
    table, cfg = _fleet_cfg()
    opener, _ = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    busy = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"hard"}', opener, pressure=_HIGH_SWAP
    )
    down_opener, _ = _opener(
        {
            "minor": S.UpstreamError("x"),
            "multimodal": S.UpstreamError("y"),
            "primary": S.UpstreamError("z"),
        }
    )
    down = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"PRIMARY"}', down_opener
    )
    assert busy.status == 429 and json.loads(busy.body)["error"]["type"] == "server_busy"
    assert down.status == 503 and json.loads(down.body)["error"]["type"] == "backend_unavailable"


def test_handle_post_override_forces_main_under_pressure() -> None:
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
    assert headers["X-Lobes-Tier"] == "main"  # hard normalizes to main
    assert headers["X-Lobes-Tier-Reason"] == "manual_override"


def test_handle_post_no_pressure_keeps_main_reason_default() -> None:
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"hard"}', opener, pressure=_NO_PRESSURE
    )
    assert calls[0][0] == "primary"
    headers = dict(resp.headers)
    assert headers["X-Lobes-Tier"] == "main"  # hard normalizes to main
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


def test_integration_streaming_request_under_pressure_gets_429_busy(tier_gateway) -> None:
    # Under pressure a streaming request is SHED with 429 busy — NOT downgraded
    # and streamed. The busy status + Retry-After + X-Lobes-Tier-Reason: busy
    # arrive as a complete (Content-Length) response, and no SSE body is emitted.
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
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        content_length = next(
            (
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            ),
            0,
        )
        while len(rest) < content_length:
            chunk = sock.recv(4096)
            if not chunk:
                break
            rest += chunk
    # 429 busy, not a streamed 200 downgrade.
    assert buf.startswith(b"HTTP/1.1 429")
    assert b"Retry-After:" in head
    assert b"X-Lobes-Tier-Reason: busy" in head
    # No SSE body for a served (downgraded) model was ever sent.
    assert b"data: minor" not in buf
    assert json.loads(rest)["error"]["code"] == "busy"


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
    assert b"X-Lobes-Tier: main" in head  # hard normalizes to main
    assert b"X-Lobes-Tier-Reason: manual_override" in head
    assert b"data: primary\n\n" in buf  # override → the 27B primary on the wire


# --- /status surfaces the busy-policy state (#85, c13) ------------------------


def _stub_probe(base_url, timeout):
    """Probe stub — no real backend socket."""
    return {"health": "unreachable", "metrics": None}


def test_fleet_status_payload_surfaces_busy_pressure_block() -> None:
    table, cfg = _fleet_cfg()
    busy = S.fleet_status_payload(table, cfg, _HIGH_SWAP, probe=_stub_probe)
    assert busy["pressure"]["mode"] == "busy"
    assert busy["pressure"]["shed"] is True
    assert busy["pressure"]["reason"] == "pressure"
    assert busy["pressure"]["swap_used_percent"] == 80.0


def test_fleet_status_payload_pressure_block_warm() -> None:
    table, cfg = _fleet_cfg()
    warm = S.fleet_status_payload(table, cfg, _NO_PRESSURE, probe=_stub_probe)
    assert warm["pressure"]["mode"] == "warm"
    assert warm["pressure"]["shed"] is False


def test_fleet_status_payload_omits_pressure_block_when_unwired() -> None:
    # Back-compat: no pressure arg → no pressure block in the payload.
    table, cfg = _fleet_cfg()
    payload = S.fleet_status_payload(table, cfg, probe=_stub_probe)
    assert "pressure" not in payload


# --- issue #91 h4: a dead primary NEVER dials the multimodal (Gemma) backend ---


def test_dead_primary_never_opens_multimodal_connection() -> None:
    # h4 (the "no-Gemma" invariant): with the cortex/primary engine dead, a
    # request naming the cortex served model must NEVER open a connection to the
    # multimodal (senses/Gemma) backend — a caller who asked for cortex must not
    # silently receive a Gemma answer (final_authority role-contract, issue #81).
    # Asserted on the injected opener's CALL SITES, not just the status code.
    table, cfg = _fleet_cfg()
    opener, calls = _opener(
        {"primary": S.UpstreamError("engine dead"), "multimodal": 200, "minor": 200}
    )
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"PRIMARY"}', opener)
    dialed = [c[0] for c in calls]
    assert dialed == ["primary"]  # ONLY the owner
    assert "multimodal" not in dialed and "minor" not in dialed  # never a foreign gear
    assert resp.status == 503
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"


# --- the "converse" honesty question: an unknown (never-advertised) model id ---


def test_unknown_model_id_never_silently_served_returns_404() -> None:
    # h23 CONVERSE: an id that was NEVER in /v1/models (neither an alias nor any
    # wired backend's served name) must NOT be silently served by the default
    # backend under a different model's weights. It 404s (model_not_found) and NO
    # backend is dialed — the request never touches primary. This REPLACES t6's
    # test_unknown_model_id_routes_to_default_owner_documented_choice, which locked
    # in exactly the behaviour h23 forbids (unknown id served under primary's
    # weights, body rewritten to "P"). This is CONSISTENT with "advertised implies
    # reachable": a model *listed* in /v1/models never 404s; one never listed
    # SHOULD 404 (contrast the never-404 race test below).
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 200, "fallback": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"never-in-v1-models"}', opener
    )
    assert calls == []  # NOT silently served — no backend dialed
    assert resp.status == 404
    assert resp.upstream is None
    body = json.loads(resp.body)
    assert body["error"]["type"] == "model_not_found"
    assert "never-in-v1-models" in body["error"]["message"]


def test_wired_but_dead_backend_yields_503_not_404() -> None:
    # CRITICAL distinction (issue #91): "F" IS a wired backend's served name, so it
    # is KNOWN even though a dead F is filtered out of /v1/models. A request naming
    # F when F is down must route to F's owner and get the retryable 503 (owner
    # down), NOT the 404 an UNKNOWN (never-wired) id gets. Getting this backwards
    # would 404 a merely-down backend and reintroduce #91. Unknown-ness is decided
    # against the routing table, never the readiness-filtered /v1/models list.
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 200, "fallback": S.UpstreamError("refused")})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"F"}', opener)
    assert [c[0] for c in calls] == ["fallback"]  # dialed the owner — F is KNOWN
    assert resp.status == 503  # owner down → retryable, NOT 404
    assert dict(resp.headers)["Retry-After"] == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"


# --- issue #92: ReadinessCache wired into /v1/models (advertised => reachable) --

from lobes.gateway._readiness import ReadinessCache  # noqa: E402
from lobes.gateway._routing import list_models_payload  # noqa: E402


def test_list_models_payload_filters_by_readiness() -> None:
    # Pure filter: with a readiness map (keyed by backend NAME), only backends
    # whose signal is True are listed. None (dead/missing) and False (unhealthy)
    # are both hidden — "advertise it anyway" is exactly the #92 defect.
    table, _cfg_ = _cfg()  # backends: primary "P", fallback "F"
    all_listed = list_models_payload(table)  # no map → every wired backend
    assert [m["id"] for m in all_listed["data"]] == ["P", "F"]
    only_primary = list_models_payload(table, {"primary": True, "fallback": None})
    assert [m["id"] for m in only_primary["data"]] == ["P"]  # None hidden
    still_only_primary = list_models_payload(table, {"primary": True, "fallback": False})
    assert [m["id"] for m in still_only_primary["data"]] == ["P"]  # False hidden
    none_ready = list_models_payload(table, {"primary": None, "fallback": None})
    assert none_ready["data"] == []  # nothing ready → nothing advertised


@pytest.fixture
def ready_gateway(monkeypatch):
    """A loopback gateway whose ReadinessCache verdicts + owner liveness are both
    caller-controllable, and whose readiness probe COUNTS its calls.

    The daemon is deliberately NOT started (``start=False``); the fixture seeds
    the snapshot with a single synchronous ``refresh()`` — exactly mirroring what
    ``serve()`` does before it binds — so the probe-call count is deterministic
    (one per backend) and the hot-path test can assert it stays put.
    """
    from types import SimpleNamespace

    table, cfg = _cfg()  # backends: primary "P" @ vllm-primary, fallback "F" @ vllm-fallback
    verdicts = {b.base_url: True for b in table.backends}
    probe_calls: list[str] = []

    def probe(base_url):
        probe_calls.append(base_url)
        return verdicts.get(base_url)

    cache = ReadinessCache.from_backends(table.backends, probe=probe, start=False)
    cache.refresh()  # one synchronous seed BEFORE binding — the serve() ordering

    alive = {b.name: True for b in table.backends}  # owner liveness for open_upstream
    open_calls: list[str] = []

    def fake_open(backend, path, body, headers, *, connect_timeout, read_timeout):
        open_calls.append(backend.name)
        if not alive.get(backend.name, False):
            raise S.UpstreamError(f"{backend.name}: refused")
        return _FakeUpstream(200, body=b'{"echo": "' + backend.name.encode() + b'"}')

    monkeypatch.setattr(S, "open_upstream", fake_open)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg, None, cache))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield SimpleNamespace(
            base=f"http://{host}:{port}",
            verdicts=verdicts,
            probe_calls=probe_calls,
            alive=alive,
            open_calls=open_calls,
            cache=cache,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        cache.stop()


def test_v1_models_correct_on_very_first_request(ready_gateway) -> None:
    # The startup window is closed by the one synchronous refresh() before bind:
    # /v1/models is correct on the VERY FIRST request, not empty for a probe
    # interval. Both backends probe True → both listed.
    gw = ready_gateway
    with urllib.request.urlopen(gw.base + "/v1/models", timeout=5) as r:
        ids = [m["id"] for m in json.load(r)["data"]]
    assert ids == ["P", "F"]


def test_v1_models_hides_dead_or_unhealthy_backend(ready_gateway) -> None:
    # A wired-but-dead backend must not be advertised (issue #92, honesty h14).
    # None (container gone / nothing listening) and False (reached but unhealthy)
    # are BOTH "not ready" — only True advertises.
    gw = ready_gateway
    gw.verdicts["http://vllm-fallback:8000"] = None  # missing container → probes None
    gw.cache.refresh()
    with urllib.request.urlopen(gw.base + "/v1/models", timeout=5) as r:
        assert [m["id"] for m in json.load(r)["data"]] == ["P"]  # dead F dropped
    gw.verdicts["http://vllm-fallback:8000"] = False  # reached but unhealthy
    gw.cache.refresh()
    with urllib.request.urlopen(gw.base + "/v1/models", timeout=5) as r:
        assert [m["id"] for m in json.load(r)["data"]] == ["P"]  # False hidden too


def test_hot_path_opens_no_readiness_probe(ready_gateway) -> None:
    # The POST hot path must open NO probe connection: readiness is read only via
    # the cache's socket-free .current() (in fact do_POST never touches the
    # readiness cache). Seed once (probe count == 2 backends), fire N completions,
    # assert the probe count is UNCHANGED.
    gw = ready_gateway
    seeded = len(gw.probe_calls)  # from the single refresh() in the fixture
    assert seeded == 2  # one probe per wired backend, nothing more
    for _ in range(5):
        req = urllib.request.Request(
            gw.base + "/v1/chat/completions",
            data=b'{"model":"P"}',
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200
    assert len(gw.probe_calls) == seeded  # zero probes added by the hot path


def test_race_listed_then_owner_killed_yields_503_not_404(ready_gateway) -> None:
    # h23 (the race invariant): model "P" is listed in /v1/models; the owner is
    # then killed; a completion naming "P" returns 503 + Retry-After — NEVER 404.
    # "Advertised implies reachable" held when it was listed; once the owner dies
    # the honest answer is "retry", not the terminal "no such model" a
    # cross-backend failover 404 (issue #91) used to produce.
    gw = ready_gateway
    with urllib.request.urlopen(gw.base + "/v1/models", timeout=5) as r:
        assert "P" in [m["id"] for m in json.load(r)["data"]]  # advertised now
    gw.alive["primary"] = False  # kill the owner AFTER it was advertised
    req = urllib.request.Request(
        gw.base + "/v1/chat/completions",
        data=b'{"model":"P"}',
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 503  # NEVER 404
    assert exc.value.headers.get("Retry-After") == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)
    assert json.loads(exc.value.read())["error"]["type"] == "backend_unavailable"


def test_wired_but_dead_unlisted_backend_yields_503_not_404(ready_gateway) -> None:
    # The exact #91 trap, end to end: a backend that is WIRED but dead is filtered
    # OUT of /v1/models (readiness), yet its served name is still in the routing
    # table. A completion naming it must return 503 (retry), NOT 404 — unknown-ness
    # is decided against the routing TABLE, never the readiness-filtered list.
    # Getting this backwards (deciding unknown-ness against /v1/models) would 404 a
    # merely-dead backend and reintroduce #91.
    gw = ready_gateway
    gw.verdicts["http://vllm-fallback:8000"] = None  # dead → dropped from /v1/models
    gw.cache.refresh()
    gw.alive["fallback"] = False  # and the owner is actually down
    with urllib.request.urlopen(gw.base + "/v1/models", timeout=5) as r:
        assert "F" not in [m["id"] for m in json.load(r)["data"]]  # NOT listed
    req = urllib.request.Request(
        gw.base + "/v1/chat/completions",
        data=b'{"model":"F"}',
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 503  # wired + dead + unlisted → retry, NOT 404
    assert exc.value.headers.get("Retry-After") == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)
    assert json.loads(exc.value.read())["error"]["type"] == "backend_unavailable"
    assert "fallback" in gw.open_calls  # the owner WAS dialed — F is known


def test_unknown_unlisted_model_yields_404_not_served(ready_gateway) -> None:
    # h23 converse at the route: an id that is neither wired nor aliased is never in
    # /v1/models AND is never silently served under the default backend's weights —
    # it 404s (model_not_found) and no backend is dialed. Contrast the wired-but-dead
    # case above (503): the two differ precisely on "is it in the routing table".
    gw = ready_gateway
    before = len(gw.open_calls)
    req = urllib.request.Request(
        gw.base + "/v1/chat/completions",
        data=b'{"model":"phantom-never-advertised"}',
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 404  # never advertised → 404, not served under primary
    assert json.loads(exc.value.read())["error"]["type"] == "model_not_found"
    assert len(gw.open_calls) == before  # no backend dialed — not silently served
