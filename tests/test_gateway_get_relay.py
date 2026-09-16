"""GET-side gateway plumbing (t8): a method-general ``open_upstream`` and a
``do_GET`` path that reaches ``_relay_streaming``.

This file covers the PLUMBING only — no route or path family is added here
(that is t9). Two things are proven:

* ``open_upstream`` takes a ``method`` and still defaults to ``POST``, so every
  existing POST caller is unchanged, and a ``GET`` call keeps the same
  connect-timeout-then-read-timeout socket treatment.
* ``_relay_streaming`` is reachable from ``do_GET`` through the
  ``_dispatch_get_upstream`` seam and relays arbitrary BINARY bytes
  byte-for-byte.

Scope honesty: the binary-fidelity assertion here is made against a local
stdlib stub upstream, not a real ComfyUI. The compose-network half of the
acceptance criterion ("identical to the same render fetched directly from
ComfyUI on the compose network") is proven in the live acceptance run (t13),
NOT by this file.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._routing import Backend


def _cfg(**over):
    env = {
        "PRIMARY_SERVED_NAME": "P",
        "FALLBACK_URL": "http://vllm-fallback:8000",
        "FALLBACK_SERVED_NAME": "F",
        "GATEWAY_DEFAULT_MODEL": "P",
    }
    env.update(over)
    return build_config(env)


# A body big enough to span several `_CHUNK` reads in the relay loop, and
# deliberately NOT valid UTF-8 — a relay that decodes anywhere would corrupt it.
_BLOB = hashlib.sha256(b"t8-innereye").digest() * 8192  # 256 KiB


class _StubUpstream(BaseHTTPRequestHandler):
    """Serves known binary bytes on GET and records the method it was asked."""

    # HTTP/1.1 so the connection is kept alive and `conn.sock` survives
    # `getresponse()` — that is what lets the read-timeout test below observe
    # the socket. It is also what a real upstream (vLLM, ComfyUI) speaks.
    protocol_version = "HTTP/1.1"

    methods: list[str] = []

    def _record(self) -> None:
        type(self).methods.append(self.command)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()
        if self.path == "/binary":
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_BLOB)))
            self.end_headers()
            self.wfile.write(_BLOB)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self._record()
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = b'{"served": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence the test log
        pass


class _QuietServer(ThreadingHTTPServer):
    """Swallows the broken-pipe traceback a test that closes early provokes."""

    def handle_error(self, request, client_address):  # pragma: no cover - noise only
        pass


@pytest.fixture
def upstream():
    _StubUpstream.methods = []
    httpd = _QuietServer(("127.0.0.1", 0), _StubUpstream)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- AC1: a method-general opener, POST callers unchanged -----------------


def test_open_upstream_still_defaults_to_post(upstream) -> None:
    """No `method=` → the wire method is POST, exactly as before t8."""
    up = S.open_upstream(
        Backend("primary", upstream, "P"),
        "/v1/chat/completions",
        b"{}",
        [],
        connect_timeout=2,
        read_timeout=5,
    )
    try:
        assert up.status == 200
    finally:
        up.close()
    assert _StubUpstream.methods == ["POST"]


def test_open_upstream_get_reaches_the_upstream(upstream) -> None:
    up = S.open_upstream(
        Backend("primary", upstream, "P"),
        "/binary",
        b"",
        [],
        connect_timeout=2,
        read_timeout=5,
        method="GET",
    )
    try:
        assert up.status == 200
        assert up.read_all() == _BLOB
    finally:
        up.close()
    assert _StubUpstream.methods == ["GET"]


def test_open_upstream_get_keeps_the_read_timeout(upstream) -> None:
    """The GET path must keep connect_timeout-then-read_timeout (a long
    GATEWAY_READ_TIMEOUT is what lets a slow render finish)."""
    up = S.open_upstream(
        Backend("primary", upstream, "P"),
        "/binary",
        b"",
        [],
        connect_timeout=2,
        read_timeout=37,
        method="GET",
    )
    try:
        assert up._conn.sock.gettimeout() == pytest.approx(37)
    finally:
        up.close()


def test_open_upstream_get_refused_raises_upstream_error() -> None:
    with pytest.raises(S.UpstreamError):
        S.open_upstream(
            Backend("primary", "http://127.0.0.1:1", "P"),
            "/binary",
            b"",
            [],
            connect_timeout=1,
            read_timeout=2,
            method="GET",
        )


# --- AC2: do_GET can reach _relay_streaming with arbitrary binary ---------


def _gateway_for(dispatch):
    """A real gateway on an ephemeral port whose GET seam is `dispatch`."""
    table, cfg = _cfg()
    base = S._make_handler(table, cfg)
    handler = type("_T8Handler", (base,), {"_dispatch_get_upstream": dispatch})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    return httpd, f"http://{host}:{port}"


def test_get_relay_is_byte_for_byte_identical(upstream) -> None:
    """A streamed GET relay preserves arbitrary binary bytes exactly.

    Proven here against a local stdlib stub upstream. The compose-network
    half of the criterion (the same render fetched directly from ComfyUI)
    is proven in the live acceptance run, t13 — not here.
    """
    backend = Backend("innereye", upstream, "comfyui")

    def dispatch(self, route, *, mesh_snapshot=None):
        if route != "/t8-probe":
            return None
        up = S.open_upstream(
            backend,
            "/binary",
            b"",
            list(self.headers.items()),
            connect_timeout=self.server_config.connect_timeout,
            read_timeout=self.server_config.read_timeout,
            method="GET",
        )
        return S.GatewayResponse(status=up.status, headers=up.headers, upstream=up, streaming=True)

    httpd, gw = _gateway_for(dispatch)
    try:
        with urllib.request.urlopen(gw + "/t8-probe", timeout=30) as r:
            assert r.status == 200
            through_gateway = r.read()
            assert r.headers.get("Content-Type") == "image/png"
        with urllib.request.urlopen(upstream + "/binary", timeout=30) as r:
            direct = r.read()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert through_gateway == direct
    assert through_gateway == _BLOB
    assert hashlib.sha256(through_gateway).hexdigest() == hashlib.sha256(_BLOB).hexdigest()


def test_get_seam_default_is_inert() -> None:
    """With nothing wired into the seam (the shipped default), an unknown GET
    route 404s exactly as it did before t8."""
    table, cfg = _cfg()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://{host}:{port}/t8-probe", timeout=5)
        assert exc.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
