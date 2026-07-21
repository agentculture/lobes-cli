"""The ``/v1/realtime`` WebSocket tunnel — decision + framing + byte pump.

Everything the gateway needs to put a caller's WebSocket session in front of
the realtime bridge, split so the parts that can be tested without sockets
ARE: :func:`plan_realtime_upgrade` is a pure function of the routing table and
config, :func:`upgrade_request_bytes` / :func:`read_head` are bytes in, bytes
out, and :func:`pump` / :func:`run_tunnel` take duck-typed sockets. Only
:mod:`lobes.gateway.server` opens a real one.

**Why a byte tunnel and not a WebSocket implementation.** The gateway does not
speak the WebSocket protocol at all: it relays the client's handshake to the
bridge verbatim, relays the bridge's ``101`` back (``Sec-WebSocket-Accept``
included — computing it here would mean re-deriving a value the bridge already
derived correctly), and then moves opaque bytes in both directions. uvicorn
inside the bridge owns framing, ping/pong, and close. That keeps the stdlib
proxy free of a protocol it has no reason to parse.

**No cross-box WebSocket** (spec boundary c13, issue #149). The #129
proxy-lobes forwarder is POST-only. A dropped ``stt`` lane refuses the
handshake with the honest ``role_infeasible`` 404 naming its peer — *even when
the operator armed* ``STT_PEER_PROXY``, because a half-served WebSocket that
silently crossed a box boundary would break the loop-guard and attribution
guarantees the proxy lane provides for POSTs.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

REALTIME_PATH = "/v1/realtime"

# The Colleague role that owns the realtime session lane. Sessions transcribe,
# so a box that dropped `stt` cannot serve one.
REALTIME_ROLE = "stt"

# Relay chunk size. Audio frames are small (a 32 ms PCM16 chunk at 24 kHz is
# ~1.5 KB); 64 KiB is simply "never the bottleneck" without holding much.
_CHUNK = 65536

# Cap on the handshake header block, so a hostile or broken upstream cannot
# make the gateway buffer without bound before the tunnel even starts.
_MAX_HEAD = 64 * 1024

# Dropped from the forwarded handshake. NOT the usual hop-by-hop set: this is
# the one request in the gateway where `Connection` and `Upgrade` are the
# POINT and must survive. `Host` is rewritten to the bridge; the framing and
# proxy-auth headers are meaningless on a bodyless handshake.
_DROP_FROM_HANDSHAKE = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
    }
)


class HandshakeError(Exception):
    """The upstream's handshake response was absent, truncated, or oversized."""


@dataclass(frozen=True)
class TunnelTarget:
    """Where an accepted handshake goes: the local realtime bridge."""

    host: str
    port: int
    path: str  # the request target, query string included, verbatim


@dataclass(frozen=True)
class RealtimeRefusal:
    """Why a handshake was refused, and with what status.

    ``kind`` is the caller-visible reason — ``not_an_upgrade`` (a plain GET),
    ``audio_not_configured`` (a text-only fleet), or ``role_infeasible`` (the
    lane is declared off, with ``peer_origin`` naming its host when the
    operator declared one). :mod:`lobes.gateway.server` turns these into the
    response bodies, so the error shapes stay defined in one place.
    """

    kind: str
    status: int
    role: str | None = None
    peer_origin: str | None = None


def is_realtime_path(path: str) -> bool:
    """True for the realtime session route (query string tolerated)."""
    return path.split("?", 1)[0] == REALTIME_PATH


def is_websocket_upgrade(headers: Iterable[tuple[str, str]]) -> bool:
    """True when these request headers are a WebSocket upgrade.

    Both halves are required: ``Upgrade: websocket`` and an ``Upgrade`` token
    in ``Connection`` (RFC 6455 §4.1). ``Connection`` is a comma-separated
    list in the wild (``keep-alive, Upgrade``), and both header names and
    values are case-insensitive.
    """
    upgrade = ""
    connection = ""
    for key, value in headers:
        lowered = key.lower()
        if lowered == "upgrade":
            upgrade = value.strip().lower()
        elif lowered == "connection":
            connection = value.lower()
    tokens = {token.strip() for token in connection.split(",")}
    return upgrade == "websocket" and "upgrade" in tokens


def plan_realtime_upgrade(table, cfg, path: str, headers: Iterable[tuple[str, str]]):
    """Decide what to do with a ``/v1/realtime`` request. Pure.

    Returns a :class:`TunnelTarget` to tunnel, or a :class:`RealtimeRefusal`.
    Precedence deliberately mirrors :func:`lobes.gateway.server.
    handle_audio_request` so the realtime lane refuses for the same reasons,
    in the same order, as the batch lane it sits beside — with the one
    documented divergence that an armed peer proxy does NOT forward here.
    """
    headers = list(headers)
    if not is_websocket_upgrade(headers):
        # A plain GET to a WebSocket-only route. 426 says what to do about it;
        # a 404 would wrongly suggest the route does not exist.
        return RealtimeRefusal(kind="not_an_upgrade", status=426)
    if REALTIME_ROLE in getattr(table, "infeasible", frozenset()):
        return RealtimeRefusal(
            kind="role_infeasible",
            status=404,
            role=REALTIME_ROLE,
            peer_origin=getattr(table, "peer_origins", {}).get(REALTIME_ROLE),
        )
    if not cfg.audio_url:
        return RealtimeRefusal(kind="audio_not_configured", status=404)
    parts = urlsplit(cfg.audio_url)
    return TunnelTarget(
        host=parts.hostname or "",
        port=parts.port or (443 if parts.scheme == "https" else 80),
        path=path,
    )


def upgrade_request_bytes(path: str, headers: Iterable[tuple[str, str]], *, host: str) -> bytes:
    """Serialise the client's handshake for the upstream leg.

    The WebSocket headers pass through untouched — ``Sec-WebSocket-Key`` in
    particular, because the bridge's ``Sec-WebSocket-Accept`` must be derived
    from the client's own key for the client to accept the 101. ``Host`` is
    rewritten to the bridge (the client's names this gateway).
    """
    lines = [f"GET {path} HTTP/1.1", f"Host: {host}"]
    lines += [f"{k}: {v}" for k, v in headers if k.lower() not in _DROP_FROM_HANDSHAKE]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def read_head(reader) -> tuple[bytes, bytes]:
    """Read the upstream response head verbatim.

    Returns ``(head, leftover)`` — the raw bytes through the terminating blank
    line, and any bytes already read past it (the bridge may pack its first
    WebSocket frame into the same TCP segment as the 101, and dropping those
    would silently lose the session's first event).

    ``reader`` must expose ``read1`` (``socket.makefile('rb')`` does).
    ``read1`` and not ``read``: on a blocking socket ``BufferedReader.read(n)``
    keeps issuing syscalls until it has ``n`` bytes or hits EOF, so asking for
    a chunk-sized read would park here forever — a 101 head is ~130 bytes and a
    bridge that just accepted a session sends nothing more until the client
    speaks, and never closes. ``read1`` returns whatever one syscall yielded.
    """
    buf = b""
    while b"\r\n\r\n" not in buf:
        if len(buf) > _MAX_HEAD:
            raise HandshakeError("upstream handshake headers exceeded the size cap")
        chunk = reader.read1(_CHUNK)
        if not chunk:
            raise HandshakeError("upstream closed before completing the handshake")
        buf += chunk
    head, _, leftover = buf.partition(b"\r\n\r\n")
    return head + b"\r\n\r\n", leftover


def status_of(head: bytes) -> int:
    """The status code from a raw response head (``0`` if unparseable)."""
    try:
        return int(head.split(b"\r\n", 1)[0].split(b" ")[1])
    except (IndexError, ValueError):
        return 0


def pump(src, dst) -> None:
    """Relay ``src`` → ``dst`` until EOF, then half-close ``dst``.

    Never raises: a reset peer mid-session is an ordinary end of session, not
    a gateway error. The half-close is what makes the *other* direction's pump
    see EOF and unwind — without it a dropped client would strand a thread
    (spec claim c26).
    """
    try:
        while True:
            data = src.recv(_CHUNK)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def run_tunnel(client, upstream, *, leftover: bytes = b"") -> None:
    """Pump bytes both ways until either side closes, then unwind both.

    ``leftover`` is whatever :func:`read_head` read past the handshake head —
    bytes the UPSTREAM sent, so they belong to the CLIENT, and it is written
    there before the pump starts. Sending them upstream instead loses the
    session's first event and bounces a server-side (unmasked) frame back at
    the bridge, which RFC 6455 §5.1 requires it to answer by closing — the
    session dies the moment it opens. Returns only once both directions have
    finished, so the caller's handler thread ends with no orphaned pump behind
    it.
    """
    if leftover:
        try:
            client.sendall(leftover)
        except OSError:
            return
    up_to_client = threading.Thread(
        target=pump, args=(upstream, client), name="realtime-up", daemon=True
    )
    up_to_client.start()
    try:
        pump(client, upstream)
    finally:
        up_to_client.join()
