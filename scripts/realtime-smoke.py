#!/usr/bin/env python3
"""Live smoke test for the ``/v1/realtime`` WebSocket session (issue #149).

Drives ONE ``/v1/realtime`` connection through the fleet gateway end to end —
connect, stream PCM16 audio, observe ``server_vad`` speech boundaries, and
receive a Parakeet transcription — all on that single connection (issue
#149's acceptance criterion 4: a consumer needs no separate batch POST). It
is modelled on ``scripts/audio-smoke.py``: stdlib-only, prints ``PASS``/
``FAIL`` per step, and exits non-zero on any failure.

This is NOT an offline CI test — it requires a live GPU box with
``lobes fleet up --apply`` (the ``--audio`` overlay) already running, exactly
like ``scripts/audio-smoke.py``.

Why a hand-rolled WebSocket client and not a library
------------------------------------------------------
Issue #149 exists to move ``server_vad`` endpointing OFF the ``reachy-mini-cli``
robot client and ONTO the server — the whole point is to keep heavyweight
deps (torch, an OpenAI SDK) out of a robot's dependency tree and its CI. A
smoke test that itself needs ``websocket-client`` or ``websockets`` would
undercut that motivation, so this script speaks just enough of RFC 6455 by
hand: the opening handshake is a plain HTTP GET with a ``Sec-WebSocket-Key``,
verified against the server's ``Sec-WebSocket-Accept``; every frame this
script sends to the server is masked (RFC 6455 §5.1 requires client-to-server
masking); frames received back are parsed generically enough to handle the
small JSON text frames and PING/PONG keepalives this route actually sends.
The wire contract itself is documented in ``lobes/realtime/app.py``'s
``/v1/realtime`` route docstring and ``lobes/realtime/_session.py``'s event
schema — this script drives exactly that contract, nothing assumed.

The pure pieces (handshake framing, the accept-key computation, frame
build/parse, PCM chunking, phrase matching) are unit-tested offline in
``tests/test_realtime_smoke_helpers.py`` with no socket and no live
deployment. The socket-owning glue in this file is not unit-tested — it
mirrors the rest of this codebase's convention (e.g. ``app.py``'s
``# pragma: no cover`` WebSocket route) of thin, live-only glue over tested
pure modules.

Acceptance-evidence procedure (issue #108's rule)
---------------------------------------------------
This script's own output is not evidence of anything until it is actually
run against real hardware, and no doc in this repo may describe the
``/v1/realtime`` surface as "validated" or "measured" until that run has
happened AND its transcript has landed in the repo. Concretely:

1. Bring up a real audio-enabled fleet: ``lobes init --fleet --audio --apply``
   then ``lobes fleet up --apply`` on the target box, and confirm
   ``GET /v1/health/ready`` on the realtime bridge (or ``lobes status``)
   reports both STT and TTS ready.
2. Run this script against that deployment, e.g.::

       python3 scripts/realtime-smoke.py --base-url http://localhost:8000

   (add ``--api-key`` if the gateway has ``GATEWAY_API_KEY`` set).
3. Capture the COMPLETE stdout (every PASS/FAIL line plus the final
   summary) into ``docs/evidence/<date>-accept-realtime-<box>.txt`` —
   ``<box>`` is the short hostname the run happened on (mirrors every other
   filename already under ``docs/evidence/``, e.g.
   ``2026-07-17-accept-muse-tool-calling-thor.txt``). Commit that file
   BEFORE editing any doc, README, or CLAUDE.md to claim the ``/v1/realtime``
   surface is validated — the evidence file is the thing that makes the
   claim true, not the other way around.
4. Only once that file exists in the repo may a doc call the surface
   "validated" or "measured live" — and it should cite the evidence file by
   path, the way ``docs/gemma-4-31b-nvfp4.md`` and CLAUDE.md's own "Colleague
   roles" section cite their evidence files today.

See ``docs/evidence/README-realtime-acceptance.md`` for the fuller version of
this procedure. As of this task (#149 t8), this script has been written and
syntax/lint-checked only — it has NOT been run against any live hardware, and
nothing in this repository should say otherwise until step 3 above happens.

Exit code 0 if every step passes; non-zero on any failure.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import socket
import struct
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from urllib.error import URLError
from urllib.parse import urlencode, urlsplit

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"  # RFC 6455 SS4.2.2, fixed by spec.

OPCODE_CONTINUATION = 0x0
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2
OPCODE_CLOSE = 0x8
OPCODE_PING = 0x9
OPCODE_PONG = 0xA

BYTES_PER_SAMPLE = 2  # PCM16
CHUNK_MS = 32  # matches lobes.realtime._segmenter's 512-sample/32ms VAD framing

REALTIME_PATH = "/v1/realtime"

DEFAULT_PHRASE = "The quick brown fox jumps over the lazy dog."
# Survives Parakeet's documented normalisations (numerals -> digits, proper
# nouns rendered phonetically) — the same curated list scripts/audio-smoke.py's
# check_round_trip already validates for this exact sentence.
DEFAULT_PHRASE_KEYWORDS = ("quick", "brown", "fox", "lazy", "dog")

EventDict = dict


# ---------------------------------------------------------------------------
# Pure helpers — unit-tested offline in tests/test_realtime_smoke_helpers.py.
# No socket, no torch, no third-party import anywhere in this section.
# ---------------------------------------------------------------------------


def make_sec_websocket_key() -> str:
    """A fresh, random base64-encoded 16-byte nonce (RFC 6455 SS4.1)."""
    return base64.b64encode(os.urandom(16)).decode("ascii")


def compute_accept_key(sec_websocket_key: str) -> str:
    """RFC 6455 SS4.2.2: base64(sha1(key + the fixed WebSocket GUID)).

    SHA-1 here is the protocol-mandated handshake check, not a security
    boundary — RFC 6455 requires exactly this algorithm.
    """
    digest = hashlib.sha1(  # nosec B324 - RFC 6455-mandated, not a security use
        (sec_websocket_key + WS_GUID).encode("ascii")
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def build_handshake_request(
    host: str, path: str, key: str, extra_headers: dict[str, str] | None = None
) -> bytes:
    """Serialise the WebSocket opening handshake (RFC 6455 SS4.1) as raw bytes."""
    lines = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    for name, value in (extra_headers or {}).items():
        lines.append(f"{name}: {value}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def parse_response_head(head: bytes) -> tuple[int, dict[str, str]]:
    """Parse a raw HTTP response head into ``(status_code, lowercased_headers)``."""
    text = head.decode("latin-1", errors="replace")
    lines = [line for line in text.split("\r\n") if line]
    status = 0
    if lines:
        match = re.match(r"HTTP/\d\.\d\s+(\d+)", lines[0])
        if match:
            status = int(match.group(1))
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return status, headers


def mask_payload(payload: bytes, mask_key: bytes) -> bytes:
    """XOR-mask (or unmask — the operation is its own inverse) *payload*."""
    if len(mask_key) != 4:
        raise ValueError("mask_key must be exactly 4 bytes")
    return bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))


def build_frame(opcode: int, payload: bytes = b"", *, mask: bool = True) -> bytes:
    """One complete, unfragmented RFC 6455 SS5.2 frame (FIN always set).

    ``mask=True`` (the default, and the only mode this script ever sends):
    every client-to-server frame MUST be masked per RFC 6455 SS5.1.
    """
    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))  # FIN=1, RSV=0
    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length < 65536:
        header.append(mask_bit | 126)
        header += struct.pack("!H", length)
    else:
        header.append(mask_bit | 127)
        header += struct.pack("!Q", length)
    if mask:
        mask_key = os.urandom(4)
        header += mask_key
        payload = mask_payload(payload, mask_key)
    return bytes(header) + payload


class FrameReadError(Exception):
    """The frame stream ended (EOF) or was malformed before a full frame arrived."""


def read_frame(recv_exact) -> tuple[bool, int, bytes]:
    """Read one frame using ``recv_exact(n) -> bytes``. Raises :class:`FrameReadError` on EOF.

    ``recv_exact`` is any zero-argument-free callable of one int argument that
    either returns exactly ``n`` bytes or raises/returns short on EOF — this
    function never assumes a real socket, which is what makes it testable
    against a plain :class:`io.BytesIO` reader fed pre-built frames.
    """
    first_two = recv_exact(2)
    if len(first_two) < 2:
        raise FrameReadError("connection closed before a frame header arrived")
    b0, b1 = first_two[0], first_two[1]
    fin = bool(b0 & 0x80)
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    if length == 126:
        ext = recv_exact(2)
        if len(ext) < 2:
            raise FrameReadError("connection closed while reading the 16-bit extended length")
        length = struct.unpack("!H", ext)[0]
    elif length == 127:
        ext = recv_exact(8)
        if len(ext) < 8:
            raise FrameReadError("connection closed while reading the 64-bit extended length")
        length = struct.unpack("!Q", ext)[0]
    mask_key = None
    if masked:
        mask_key = recv_exact(4)
        if len(mask_key) < 4:
            raise FrameReadError("connection closed while reading the mask key")
    payload = recv_exact(length) if length else b""
    if length and len(payload) < length:
        raise FrameReadError("connection closed before the full payload arrived")
    if masked and mask_key is not None:
        payload = mask_payload(payload, mask_key)
    return fin, opcode, payload


def chunk_pcm(data: bytes, chunk_bytes: int):
    """Yield *data* split into ``chunk_bytes``-sized pieces.

    A short final remainder (fewer than ``chunk_bytes``) is yielded as-is —
    never silently dropped, mirroring how the server's own segmenter treats
    a short trailing read.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    for start in range(0, len(data), chunk_bytes):
        yield data[start : start + chunk_bytes]


def silence_bytes(duration_ms: int, sample_rate: int) -> bytes:
    """``duration_ms`` of digital silence — PCM16 mono LE zero bytes."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00" * (num_samples * BYTES_PER_SAMPLE)


def bytes_per_chunk_for_rate(sample_rate: int, chunk_ms: int = CHUNK_MS) -> int:
    """Byte length of one ``chunk_ms`` PCM16 mono chunk at ``sample_rate``."""
    return int(round(sample_rate * chunk_ms / 1000)) * BYTES_PER_SAMPLE


def normalize_text(text: str) -> str:
    """Lowercase and strip everything but alphanumerics/spaces, for loose matching."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def missing_keywords(transcription: str, keywords) -> list[str]:
    """Keywords NOT present (case-insensitive substring) in *transcription*."""
    lowered = transcription.lower()
    return [kw for kw in keywords if kw.lower() not in lowered]


def keywords_for_phrase(phrase: str) -> tuple[str, ...]:
    """The keyword list a transcription is checked against for *phrase*.

    The curated default list is used for the exact default phrase (already
    proven to survive Parakeet's normalisations); any other phrase falls back
    to its own normalised words, which is a strictly looser but honest check
    — a custom phrase may hit the numeral/proper-noun quirks
    ``scripts/audio-smoke.py`` documents, and this script does not pretend
    otherwise.
    """
    if phrase == DEFAULT_PHRASE:
        return DEFAULT_PHRASE_KEYWORDS
    words = [w for w in normalize_text(phrase).split(" ") if w]
    return tuple(words)


def build_realtime_ws_target(base_url: str, input_sample_rate: int) -> tuple[str, int, str, str]:
    """Derive ``(scheme, host, port, path_with_query)`` for the WS handshake.

    ``scheme`` is ``"wss"``/``"ws"`` (informational only — this script always
    opens a plain TCP socket; TLS is out of scope, mirroring the rest of the
    fleet's private/tailnet-transport assumption).
    """
    parsed = urlsplit(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    query = urlencode({"input_sample_rate": input_sample_rate, "turn_detection": "server_vad"})
    path = f"{REALTIME_PATH}?{query}"
    return scheme, host, port, path


def classify_event_or_timeout(event: EventDict | None, expected_type: str) -> tuple[bool, str]:
    """Decide one wait step's PASS/FAIL, honestly distinguishing THREE outcomes.

    This is criterion 5's honesty check made concrete: the caller must never
    read "no event arrived" as equivalent to either a pass or an explicit
    server-reported error.

    - ``event is None`` -> a genuine timeout: nothing arrived at all (could be
      silence, a hung connection, or a dropped session) — reported as a
      TIMEOUT, never as success.
    - ``event["type"] == "error"`` -> the server reported a NAMED failure
      (e.g. ``vad_unavailable``) — reported with its code and message,
      distinct wording from a timeout.
    - anything else that isn't *expected_type* -> an unexpected event type.
    - *expected_type* -> PASS.
    """
    if event is None:
        return False, f"TIMEOUT waiting for {expected_type!r} — no event arrived at all"
    if event.get("type") == "error":
        return False, (
            f"server reported error code={event.get('code')!r} "
            f"message={event.get('message')!r} (an explicit error, not a timeout)"
        )
    if event.get("type") != expected_type:
        return False, f"expected {expected_type!r}, got {event.get('type')!r}: {event!r}"
    return True, f"received {expected_type!r}"


# ---------------------------------------------------------------------------
# Live-only glue: sockets, threads, real time. Not unit-tested (see module
# docstring) — mirrors app.py's own pragma-no-cover WebSocket route.
# ---------------------------------------------------------------------------


@dataclass
class SmokeResult:
    name: str
    ok: bool
    detail: str = ""


class WebSocketClient:
    """A minimal RFC 6455 client: one handshake, masked writes, framed reads."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = bytearray()
        self._send_lock = threading.Lock()

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        path: str,
        *,
        extra_headers: dict[str, str] | None = None,
        connect_timeout: float = 10.0,
    ) -> tuple["WebSocketClient", int, dict[str, str]]:
        """Open the TCP socket and perform the WS handshake.

        Returns ``(client, status_code, response_headers)``. A non-101
        *status_code* means the handshake was refused (by the gateway's
        auth gate, its realtime-refusal path, or the bridge itself) — the
        caller decides what to do with that; this method never raises on a
        clean HTTP refusal, only on a transport-level failure.
        """
        key = make_sec_websocket_key()
        sock = socket.create_connection((host, port), timeout=connect_timeout)
        client = cls(sock)
        request = build_handshake_request(f"{host}:{port}", path, key, extra_headers=extra_headers)
        sock.sendall(request)
        head = client._read_until(b"\r\n\r\n", timeout=connect_timeout)
        status, headers = parse_response_head(head)
        if status == 101:
            expected_accept = compute_accept_key(key)
            got_accept = headers.get("sec-websocket-accept", "")
            if got_accept != expected_accept:
                raise ConnectionError(
                    f"Sec-WebSocket-Accept mismatch: expected {expected_accept!r}, "
                    f"got {got_accept!r} — refusing to trust this handshake"
                )
        return client, status, headers

    def _read_until(self, marker: bytes, timeout: float) -> bytes:
        self._sock.settimeout(timeout)
        while marker not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("connection closed during the handshake")
            self._buf.extend(chunk)
        idx = self._buf.index(marker) + len(marker)
        head = bytes(self._buf[:idx])
        del self._buf[:idx]
        return head

    def _recv_exact(self, n: int, timeout: float | None = None) -> bytes:
        if timeout is not None:
            self._sock.settimeout(timeout)
        while len(self._buf) < n:
            chunk = self._sock.recv(max(65536, n))
            if not chunk:
                break
            self._buf.extend(chunk)
        take = min(n, len(self._buf))
        data = bytes(self._buf[:take])
        del self._buf[:take]
        return data

    def read_frame(self, timeout: float | None = None) -> tuple[bool, int, bytes]:
        return read_frame(lambda n: self._recv_exact(n, timeout=timeout))

    def read_raw(self, n: int, timeout: float | None = None) -> bytes:
        """Read up to *n* already-available-or-arriving bytes (best-effort diagnostics)."""
        return self._recv_exact(n, timeout=timeout)

    def send_frame(self, opcode: int, payload: bytes = b"") -> None:
        frame = build_frame(opcode, payload, mask=True)
        with self._send_lock:
            self._sock.sendall(frame)

    def send_binary(self, payload: bytes) -> None:
        self.send_frame(OPCODE_BINARY, payload)

    def send_close(self, code: int = 1000) -> None:
        try:
            self.send_frame(OPCODE_CLOSE, struct.pack("!H", code))
        except OSError:
            pass

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class EventReader:
    """Background thread: decode JSON text frames off *client* into an ordered log.

    Answers PING with PONG so an idle session (between the last audio chunk
    and the transcription arriving) does not get dropped by a keepalive
    timeout. Stops on a CLOSE frame, EOF, or any socket error — never raises
    into the caller; failures are recorded as ``self.closed_reason``.
    """

    def __init__(self, client: WebSocketClient) -> None:
        self._client = client
        self._lock = threading.Lock()
        self.events: list[EventDict] = []
        self.closed = threading.Event()
        self.closed_reason = ""
        self._thread = threading.Thread(target=self._run, name="realtime-smoke-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def snapshot(self) -> list[EventDict]:
        with self._lock:
            return list(self.events)

    def _run(self) -> None:
        # A plain loop, not recursion: read_frame is called with a short
        # per-attempt timeout so this thread can periodically notice nothing
        # is arriving yet. An idle session commonly sits quiet for seconds
        # while STT runs — recursing on every idle tick would eventually hit
        # Python's recursion limit, so each timeout is just a `continue`.
        try:
            while True:
                try:
                    fin, opcode, payload = self._client.read_frame(timeout=1.0)
                except socket.timeout:
                    continue
                if not fin:
                    # This route never fragments a message (see app.py); a
                    # fragment is unexpected but not fatal — skip it rather
                    # than mis-assemble it.
                    continue
                if opcode == OPCODE_TEXT:
                    try:
                        event = json.loads(payload.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        self.closed_reason = f"malformed text frame: {exc}"
                        break
                    with self._lock:
                        self.events.append(event)
                elif opcode == OPCODE_PING:
                    self._client.send_frame(OPCODE_PONG, payload)
                elif opcode == OPCODE_CLOSE:
                    code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 0
                    self.closed_reason = f"server closed (code={code})"
                    break
                # BINARY/PONG/CONTINUATION: this route sends none; ignored.
        except FrameReadError as exc:
            self.closed_reason = f"connection closed: {exc}"
        except OSError as exc:
            self.closed_reason = f"socket error: {exc}"
        finally:
            self.closed.set()


def wait_for_event(
    reader: EventReader, start_index: int, expected_type: str, deadline: float
) -> tuple[EventDict | None, int]:
    """Poll *reader* for the next event at/after *start_index*, up to *deadline*.

    Returns ``(event_or_None, next_start_index)``. Returns as soon as ANY new
    event arrives (not just a matching one) — the caller runs it through
    :func:`classify_event_or_timeout` to decide PASS/FAIL, so an unexpected
    event type is reported precisely rather than silently waited-past.
    """
    while True:
        events = reader.snapshot()
        if len(events) > start_index:
            return events[start_index], start_index + 1
        if reader.closed.is_set():
            # One more look before giving up: the reader may have appended
            # its very last event and then set `closed` a moment after our
            # snapshot() above — this closes that race almost entirely.
            events = reader.snapshot()
            if len(events) > start_index:
                return events[start_index], start_index + 1
            return None, start_index
        if time.monotonic() >= deadline:
            return None, start_index
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# HTTP step: synthesize the known phrase via /v1/audio/speech.
# ---------------------------------------------------------------------------


def synthesize_phrase(base_url: str, phrase: str, api_key: str | None, timeout: float) -> bytes:
    """POST the known *phrase* to ``/v1/audio/speech`` and return raw PCM16 @ 24 kHz.

    Raises :class:`URLError` / :class:`OSError` on any transport failure —
    the caller decides how to report it. Uses ``response_format=pcm`` so the
    bytes returned are exactly what ``/v1/realtime`` expects at its default
    ``input_sample_rate=24000`` — no client-side resampling anywhere in this
    script.
    """
    url = f"{base_url.rstrip('/')}/v1/audio/speech"
    payload = json.dumps({"input": phrase, "response_format": "pcm"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(  # nosec B310 - operator-supplied base_url
        req, timeout=timeout
    ) as resp:
        if resp.status != 200:
            raise URLError(f"/v1/audio/speech returned HTTP {resp.status}")
        return resp.read()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_smoke(args: argparse.Namespace) -> list[SmokeResult]:
    results: list[SmokeResult] = []

    def record(name: str, ok: bool, detail: str = "") -> bool:
        results.append(SmokeResult(name, ok, detail))
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name}" + (f" — {detail}" if detail else ""))
        return ok

    print(f"Testing /v1/realtime at {args.base_url}")
    print(f"Known phrase: {args.phrase!r}")
    print()

    # Step 1: synthesize the known phrase — the audio this session will stream.
    pcm: bytes | None = None
    try:
        pcm = synthesize_phrase(args.base_url, args.phrase, args.api_key, timeout=60.0)
        ok = len(pcm) > 0
        record("tts-source", ok, f"{len(pcm)} bytes PCM16 @ 24000 Hz" if ok else "empty body")
        if not ok:
            pcm = None
    except (URLError, OSError, TimeoutError) as exc:
        record("tts-source", False, f"POST /v1/audio/speech failed: {exc}")

    # Step 2: open the ONE WebSocket this whole session rides on.
    scheme, host, port, path = build_realtime_ws_target(args.base_url, args.input_sample_rate)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    client: WebSocketClient | None = None
    try:
        client, status, _headers = WebSocketClient.connect(
            host, port, path, extra_headers=headers, connect_timeout=10.0
        )
    except (OSError, ConnectionError) as exc:
        record("ws-handshake", False, f"could not reach {scheme}://{host}:{port}{path}: {exc}")
        _print_summary(results)
        return results

    if status != 101:
        detail = f"HTTP {status} (expected 101 Switching Protocols)"
        try:
            # Best-effort: read a short body for a more useful message. Not
            # fatal if this fails — the status code alone is still reported.
            body = client.read_raw(4096, timeout=2.0)
            if body:
                detail += f" body={body[:500]!r}"
        except OSError:
            pass
        client.close()
        record("ws-handshake", False, detail)
        _print_summary(results)
        return results
    record("ws-handshake", True, "101 Switching Protocols, Sec-WebSocket-Accept verified")

    reader = EventReader(client)
    reader.start()
    deadline = time.monotonic() + args.timeout
    idx = 0

    event, idx = wait_for_event(reader, idx, "session.created", deadline)
    ok, detail = classify_event_or_timeout(event, "session.created")
    record("session-created", ok, detail)

    if pcm is None:
        record("speech-started", False, "skipped — no audio available (tts-source step failed)")
        record("speech-stopped", False, "skipped — no audio available (tts-source step failed)")
        record(
            "transcription",
            False,
            "skipped — no audio available (tts-source step failed)",
        )
        client.send_close()
        reader.join(timeout=2.0)
        client.close()
        _print_summary(results)
        return results

    # Step 3: stream the PCM in real-time-ish 32ms chunks, then trailing
    # silence so server_vad's vad_silence_ms confirms the turn ended.
    chunk_bytes = bytes_per_chunk_for_rate(args.input_sample_rate)
    silence = silence_bytes(args.trailing_silence_ms, args.input_sample_rate)
    stream = bytes(pcm) + silence
    try:
        for chunk in chunk_pcm(stream, chunk_bytes):
            client.send_binary(chunk)
            time.sleep(CHUNK_MS / 1000.0)
    except OSError as exc:
        record("audio-stream", False, f"send failed mid-stream: {exc}")
        client.close()
        reader.join(timeout=2.0)
        _print_summary(results)
        return results
    record("audio-stream", True, f"streamed {len(stream)} bytes + trailing silence")

    event, idx = wait_for_event(reader, idx, "input_audio_buffer.speech_started", deadline)
    ok, detail = classify_event_or_timeout(event, "input_audio_buffer.speech_started")
    record("speech-started", ok, detail)

    event, idx = wait_for_event(reader, idx, "input_audio_buffer.speech_stopped", deadline)
    ok, detail = classify_event_or_timeout(event, "input_audio_buffer.speech_stopped")
    record("speech-stopped", ok, detail)

    event, idx = wait_for_event(
        reader, idx, "conversation.item.input_audio_transcription.completed", deadline
    )
    ok, detail = classify_event_or_timeout(
        event, "conversation.item.input_audio_transcription.completed"
    )
    if ok:
        text = event.get("text", "")
        keywords = keywords_for_phrase(args.phrase)
        missing = missing_keywords(text, keywords)
        if missing:
            ok = False
            detail = f"transcript={text!r} missing keywords={missing}"
        else:
            detail = f"transcript={text!r} contains every expected keyword {keywords}"
    record("transcription", ok, detail)

    client.send_close()
    reader.join(timeout=3.0)
    client.close()

    _print_summary(results)
    return results


def _print_summary(results: list[SmokeResult]) -> None:
    print()
    print("=" * 60)
    passed = sum(1 for r in results if r.ok)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")
    if passed == total and total > 0:
        print("SUCCESS: all realtime checks passed")
    else:
        print("FAILURE: some checks failed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Live smoke test for the lobes /v1/realtime WebSocket session. "
            "Requires a live fleet with the --audio overlay running "
            "(lobes fleet up --apply); NOT an offline CI test."
        )
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the fleet gateway (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token if the gateway has GATEWAY_API_KEY set (default: unset)",
    )
    parser.add_argument(
        "--phrase",
        default=DEFAULT_PHRASE,
        help="Known phrase to synthesize and expect back in the transcript "
        f"(default: {DEFAULT_PHRASE!r})",
    )
    parser.add_argument(
        "--input-sample-rate",
        type=int,
        default=24000,
        choices=(16000, 24000),
        help="Session input_sample_rate query param (default: 24000, "
        "matching Chatterbox's native TTS output rate — no client resample)",
    )
    parser.add_argument(
        "--trailing-silence-ms",
        type=int,
        default=1200,
        help="Trailing silence appended after the phrase so server_vad's "
        "vad_silence_ms (default 600ms) confirms the turn ended (default: 1200)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Overall deadline (seconds) waiting for the full event sequence "
        "after the handshake (default: 45.0)",
    )
    args = parser.parse_args()

    try:
        results = run_smoke(args)
    except Exception as exc:  # noqa: BLE001 - a smoke test must never traceback
        print(f"FAIL: unexpected error: {type(exc).__name__}: {exc}")
        return 1

    return 0 if results and all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
