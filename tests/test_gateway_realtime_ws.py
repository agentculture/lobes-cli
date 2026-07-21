"""The gateway's ``/v1/realtime`` WebSocket tunnel (plan task t3, issue #149).

Everything here is socket-free or loopback-only: the upgrade DECISION is a pure
function over the routing table + config, the handshake serialisation is bytes
in / bytes out, and the byte pump is driven with duck-typed fake sockets. The
one loopback test proves the inbound bearer gate fires on the handshake before
any tunnel is planned.
"""

from __future__ import annotations

import socket
import threading
from http.server import ThreadingHTTPServer

import pytest

from lobes.gateway import _realtime as R
from lobes.gateway import server as S
from lobes.gateway._config import ServerConfig
from lobes.gateway._routing import Backend, RoutingTable

_WS_HEADERS = [
    ("Host", "gateway:8000"),
    ("Upgrade", "websocket"),
    ("Connection", "Upgrade"),
    ("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ=="),
    ("Sec-WebSocket-Version", "13"),
]


def _cfg(audio_url: str | None = "http://realtime:8080") -> ServerConfig:
    return ServerConfig(
        host="127.0.0.1", port=0, connect_timeout=1.0, read_timeout=5.0, audio_url=audio_url
    )


def _table(**kw) -> RoutingTable:
    return RoutingTable(
        backends=(Backend(name="primary", base_url="http://p:8001", served_name="m"),),
        default_model="m",
        aliases={},
        **kw,
    )


# --- path + upgrade recognition -------------------------------------------


def test_realtime_path_recognised_with_and_without_query() -> None:
    assert R.is_realtime_path("/v1/realtime") is True
    assert R.is_realtime_path("/v1/realtime?model=stt") is True
    assert R.is_realtime_path("/v1/realtimex") is False
    assert R.is_realtime_path("/v1/audio/transcriptions") is False


def test_websocket_upgrade_detection_is_case_and_list_tolerant() -> None:
    assert R.is_websocket_upgrade(_WS_HEADERS) is True
    # Browsers send "keep-alive, Upgrade"; header names/values vary in case.
    assert R.is_websocket_upgrade([("upgrade", "WebSocket"), ("connection", "keep-alive, Upgrade")])
    assert R.is_websocket_upgrade([("Connection", "Upgrade")]) is False  # no Upgrade header
    assert R.is_websocket_upgrade([("Upgrade", "websocket")]) is False  # no Connection token
    assert R.is_websocket_upgrade([]) is False


# --- the upgrade decision (pure) ------------------------------------------


def test_plain_get_without_upgrade_headers_is_refused_not_tunnelled() -> None:
    d = R.plan_realtime_upgrade(_table(), _cfg(), "/v1/realtime", [("Host", "g:8000")])
    assert isinstance(d, R.RealtimeRefusal)
    assert d.kind == "not_an_upgrade"
    assert d.status == 426  # RFC 7231 §6.5.15 — Upgrade Required


def test_text_only_fleet_has_no_realtime_surface() -> None:
    d = R.plan_realtime_upgrade(_table(), _cfg(audio_url=None), "/v1/realtime", _WS_HEADERS)
    assert isinstance(d, R.RealtimeRefusal)
    assert d.kind == "audio_not_configured"
    assert d.status == 404


def test_declared_off_stt_lane_404s_role_infeasible_naming_the_peer() -> None:
    table = _table(
        infeasible=frozenset({"stt"}),
        peer_origins={"stt": "http://thor:8000"},
    )
    d = R.plan_realtime_upgrade(table, _cfg(), "/v1/realtime", _WS_HEADERS)
    assert isinstance(d, R.RealtimeRefusal)
    assert d.kind == "role_infeasible"
    assert d.status == 404
    assert d.role == "stt"
    assert d.peer_origin == "http://thor:8000"


def test_an_armed_stt_peer_proxy_still_refuses_no_websocket_is_ever_forwarded() -> None:
    # Boundary c13: the #129 proxy-lobes forwarder is POST-only. An operator who
    # armed STT_PEER_PROXY must still get the honest referral 404 here — never a
    # cross-box WebSocket forward.
    table = _table(
        infeasible=frozenset({"stt"}),
        peer_proxied=frozenset({"stt"}),
        peer_origins={"stt": "http://thor:8000"},
    )
    d = R.plan_realtime_upgrade(table, _cfg(), "/v1/realtime", _WS_HEADERS)
    assert isinstance(d, R.RealtimeRefusal)
    assert d.kind == "role_infeasible"
    assert d.peer_origin == "http://thor:8000"


def test_a_served_lane_plans_a_tunnel_to_the_local_bridge() -> None:
    d = R.plan_realtime_upgrade(_table(), _cfg(), "/v1/realtime?x=1", _WS_HEADERS)
    assert isinstance(d, R.TunnelTarget)
    assert (d.host, d.port) == ("realtime", 8080)
    assert d.path == "/v1/realtime?x=1"  # query preserved verbatim


# --- handshake serialisation ----------------------------------------------


def test_upgrade_request_preserves_the_websocket_headers_and_rewrites_host() -> None:
    raw = R.upgrade_request_bytes("/v1/realtime", _WS_HEADERS, host="realtime:8080")
    head, _, rest = raw.partition(b"\r\n\r\n")
    assert rest == b""  # a handshake carries no body
    lines = head.decode("latin-1").split("\r\n")
    assert lines[0] == "GET /v1/realtime HTTP/1.1"
    sent = {k.lower(): v for k, v in (line.split(": ", 1) for line in lines[1:])}
    assert sent["host"] == "realtime:8080"  # rewritten, never the client's Host
    assert sent["upgrade"] == "websocket"  # NOT stripped as hop-by-hop
    assert sent["connection"] == "Upgrade"
    assert sent["sec-websocket-key"] == "dGhlIHNhbXBsZSBub25jZQ=="
    assert sent["sec-websocket-version"] == "13"


def test_upgrade_request_drops_framing_and_proxy_headers() -> None:
    raw = R.upgrade_request_bytes(
        "/v1/realtime",
        _WS_HEADERS + [("Content-Length", "0"), ("Proxy-Authorization", "Basic x"), ("TE", "gzip")],
        host="realtime:8080",
    )
    lowered = raw.decode("latin-1").lower()
    assert "content-length" not in lowered
    assert "proxy-authorization" not in lowered
    assert "\r\nte:" not in lowered


def test_handshake_response_head_is_read_verbatim_and_its_status_parsed() -> None:
    head = (
        b"HTTP/1.1 101 Switching Protocols\r\n"
        b"Upgrade: websocket\r\n"
        b"Connection: Upgrade\r\n"
        b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n"
    )
    reader = _FakeReader(head + b"\x81\x03abc")  # first WS frame follows the head
    got, leftover = R.read_head(reader)
    assert got == head  # relayed byte-for-byte: Sec-WebSocket-Accept must survive
    assert R.status_of(got) == 101
    assert leftover == b"\x81\x03abc"  # bytes already buffered past the head


def test_a_non_101_handshake_response_is_reported_as_its_own_status() -> None:
    reader = _FakeReader(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
    head, _ = R.read_head(reader)
    assert R.status_of(head) == 503


def test_read_head_returns_on_a_real_socket_that_stays_open() -> None:
    """The regression guard for the hang that fakes could not catch.

    A real bridge answers 101 and then goes SILENT, waiting for the client to
    speak — it neither fills a 64 KiB buffer nor closes. ``read_head`` must
    return as soon as the head is complete; anything that waits for more parks
    the handler thread forever and no session ever starts.
    """
    client, bridge = socket.socketpair()
    try:
        bridge.sendall(
            b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
            b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n"
        )
        reader = client.makefile("rb")
        result: list = []

        def run() -> None:
            try:
                result.append(R.read_head(reader))
            except Exception as exc:  # noqa: BLE001 - surfaced via the assert below
                result.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive(), "read_head blocked on an open, idle socket"
        head, leftover = result[0]
        assert R.status_of(head) == 101
        assert b"Sec-WebSocket-Accept" in head
        assert leftover == b""
    finally:
        client.close()
        bridge.close()


def test_read_head_keeps_bytes_the_bridge_packed_after_the_head() -> None:
    """A real socket delivering the 101 and a first frame in one segment."""
    client, bridge = socket.socketpair()
    try:
        bridge.sendall(b"HTTP/1.1 101 Switching Protocols\r\n\r\n" + b"\x81\x03abc")
        head, leftover = R.read_head(client.makefile("rb"))
        assert R.status_of(head) == 101
        assert leftover == b"\x81\x03abc"  # the session's first event survives
    finally:
        client.close()
        bridge.close()


def test_read_head_reassembles_a_head_split_across_reads() -> None:
    # One syscall per call, 8 bytes at a time: the loop must keep reading until
    # the terminator rather than giving up on the first partial chunk.
    reader = _FakeReader(
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\nZ", chunk=8
    )
    head, leftover = R.read_head(reader)
    assert head.endswith(b"\r\n\r\n") and b"Upgrade: websocket" in head
    assert R.status_of(head) == 101
    # `leftover` is only what was ALREADY read past the terminator — here the
    # loop stopped on the chunk that completed it, so the trailing byte is
    # still in the reader (the pump gets it). read1(_CHUNK) always asks for
    # more than the buffer holds, so nothing can be stranded inside it.
    assert leftover == b""


def test_read_head_refuses_an_unbounded_header_block() -> None:
    with pytest.raises(R.HandshakeError):
        R.read_head(_FakeReader(b"HTTP/1.1 101 x\r\n" + b"X: y\r\n" * 100_000))


def test_read_head_on_a_closed_upstream_raises_rather_than_hanging() -> None:
    with pytest.raises(R.HandshakeError):
        R.read_head(_FakeReader(b""))


# --- the byte pump --------------------------------------------------------


class _FakeReader:
    """Duck-typed buffered reader: ``read1`` semantics, one chunk per call.

    Deliberately exposes ONLY ``read1``. An earlier version of this fake had a
    ``read`` that returned immediately with whatever it had — which is NOT how
    ``BufferedReader.read`` behaves on a blocking socket (it waits to fill the
    buffer), and that mismatch hid a bug where the real handshake hung forever.
    The real-socket test below is the regression guard; this fake stays honest
    about which method the code may use.
    """

    def __init__(self, data: bytes, *, chunk: int | None = None) -> None:
        self._data = data
        self._chunk = chunk

    def read1(self, n: int) -> bytes:
        n = min(n, self._chunk) if self._chunk else n
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk


class _FakeSock:
    """A duck-typed socket: scripted inbound chunks, recorded outbound bytes."""

    def __init__(self, inbound: list[bytes] | None = None) -> None:
        self._inbound = list(inbound or [])
        self.sent = b""
        self.shutdown_calls = 0
        self.closed = False

    def recv(self, _n: int) -> bytes:
        if not self._inbound:
            return b""  # EOF — peer closed
        return self._inbound.pop(0)

    def sendall(self, data: bytes) -> None:
        if self.closed:
            raise OSError("send on closed socket")
        self.sent += data

    def shutdown(self, _how: int) -> None:
        self.shutdown_calls += 1

    def close(self) -> None:
        self.closed = True


def test_pump_relays_until_eof_then_half_closes_the_far_side() -> None:
    src, dst = _FakeSock([b"ab", b"cd"]), _FakeSock()
    R.pump(src, dst)
    assert dst.sent == b"abcd"
    assert dst.shutdown_calls == 1  # EOF propagates as a half-close, not a hang


def test_tunnel_relays_both_directions_and_returns_when_both_sides_are_done() -> None:
    client = _FakeSock([b"client-audio-1", b"client-audio-2"])
    upstream = _FakeSock([b"event-1", b"event-2"])
    R.run_tunnel(client, upstream, leftover=b"pre-buffered")
    assert upstream.sent == b"pre-bufferedclient-audio-1client-audio-2"
    assert client.sent == b"event-1event-2"


def test_tunnel_unwinds_both_directions_when_the_client_drops_mid_stream() -> None:
    # c26: a dropped robot must never strand a gateway thread. The client side
    # EOFs immediately; the upstream pump must still be unwound and joined.
    client = _FakeSock([])
    upstream = _FakeSock([b"event-1"])
    done = threading.Event()

    def run() -> None:
        R.run_tunnel(client, upstream)
        done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert done.wait(timeout=5), "run_tunnel did not unwind after the client closed"
    t.join(timeout=5)
    assert not t.is_alive()
    assert upstream.shutdown_calls >= 1  # the far side was told, not left dangling


def test_tunnel_unwinds_when_the_upstream_dies_first() -> None:
    client = _FakeSock([b"still-talking"])
    upstream = _FakeSock([])
    done = threading.Event()
    threading.Thread(
        target=lambda: (R.run_tunnel(client, upstream), done.set()), daemon=True
    ).start()
    assert done.wait(timeout=5), "run_tunnel did not unwind after the upstream closed"
    assert client.shutdown_calls >= 1


def test_pump_swallows_a_broken_pipe_instead_of_raising_into_the_handler() -> None:
    src, dst = _FakeSock([b"x"]), _FakeSock()
    dst.close()  # sendall now raises OSError, as a reset peer would
    R.pump(src, dst)  # must not propagate


# --- the inbound bearer gate on the handshake (loopback) -------------------


@pytest.fixture
def gateway_with_key(monkeypatch):
    """A loopback gateway with GATEWAY_API_KEY set and audio wired.

    ``planned`` counts every upgrade DECISION: an unauthorized handshake must
    add nothing to it — the gate runs before any tunnel is planned or any
    upstream socket is opened.
    """
    planned: list[str] = []
    real_plan = R.plan_realtime_upgrade

    def counting_plan(table, cfg, path, headers):
        planned.append(path)
        return real_plan(table, cfg, path, headers)

    monkeypatch.setattr(S, "plan_realtime_upgrade", counting_plan, raising=False)
    cfg = ServerConfig(
        host="127.0.0.1",
        port=0,
        connect_timeout=1.0,
        read_timeout=5.0,
        audio_url="http://realtime:8080",
        api_key="k-secret",
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(_table(), cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    yield (host, port), planned
    httpd.shutdown()


def _handshake(addr: tuple[str, int], authorization: str | None = None) -> tuple[int, bytes]:
    """Send a REAL WebSocket handshake over a raw socket.

    urllib cannot be used here: it forces ``Connection: close``, which is not a
    WebSocket upgrade at all — the gateway would (correctly) answer 426 and the
    test would prove nothing about the auth gate.
    """
    lines = [f"GET {R.REALTIME_PATH} HTTP/1.1", f"Host: {addr[0]}:{addr[1]}"]
    lines += [f"{k}: {v}" for k, v in _WS_HEADERS[1:]]
    if authorization is not None:
        lines.append(f"Authorization: {authorization}")
    sock = socket.create_connection(addr, timeout=5)
    try:
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        raw = b""
        sock.settimeout(2)
        while True:  # drain head AND body: the refusal bodies are asserted on
            try:
                chunk = sock.recv(4096)
            except OSError:
                break
            if not chunk:
                break
            raw += chunk
    finally:
        sock.close()
    return R.status_of(raw), raw


def test_handshake_without_a_bearer_is_401_before_any_tunnel_is_planned(gateway_with_key) -> None:
    addr, planned = gateway_with_key
    status, raw = _handshake(addr)
    assert status == 401
    assert b"invalid_api_key" in raw
    assert b"WWW-Authenticate: Bearer" in raw
    assert planned == [], "the upgrade was planned despite a failed auth gate"
    assert b"k-secret" not in raw  # the gate never echoes key material


def test_handshake_with_a_wrong_bearer_is_401_before_any_tunnel_is_planned(
    gateway_with_key,
) -> None:
    addr, planned = gateway_with_key
    status, _ = _handshake(addr, "Bearer not-the-key")
    assert status == 401
    assert planned == []


def test_an_authorized_handshake_reaches_the_planner_and_dials_the_bridge(gateway_with_key) -> None:
    addr, planned = gateway_with_key
    # The bridge host does not resolve from the test env, so the tunnel attempt
    # fails AFTER planning — which is what this asserts: the gate let it through
    # and the handler went on to dial, rather than refusing it as unauthorized.
    status, raw = _handshake(addr, "Bearer k-secret")
    assert status == 502
    assert b"realtime bridge is unavailable" in raw
    assert planned == [R.REALTIME_PATH]


def test_a_plain_get_to_the_route_is_426_not_404(gateway_with_key) -> None:
    addr, _ = gateway_with_key
    sock = socket.create_connection(addr, timeout=5)
    try:
        sock.sendall(
            f"GET {R.REALTIME_PATH} HTTP/1.1\r\nHost: x\r\n"
            f"Authorization: Bearer k-secret\r\nConnection: close\r\n\r\n".encode()
        )
        raw = sock.recv(4096)
    finally:
        sock.close()
    assert R.status_of(raw) == 426
    assert b"Upgrade: websocket" in raw  # tells the caller what to send instead
