"""Offline unit tests for the PURE helpers in ``scripts/realtime-smoke.py``.

The script (issue #149 t8) is a LIVE-only tool — it opens real sockets
against a deployed audio overlay and has never been run against hardware as
part of this task (see the script's own module docstring). What CAN be
tested offline, with no socket and no live deployment, is the RFC 6455
framing/handshake arithmetic and the small pure decision helpers the script
builds on: masking, frame build/parse, the accept-key computation, PCM
chunking, phrase matching, and the three-way honesty classification behind
criterion 5 (an explicit ``error`` event must never read the same as a bare
timeout).

The script lives under ``scripts/`` with a hyphenated filename, so it is not
an importable module by its normal name — loaded here via
``importlib.util`` from its file path, exactly the way a hyphenated script
has to be imported in Python.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import socket
import struct
import sys
import threading
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "realtime-smoke.py"
_spec = importlib.util.spec_from_file_location("realtime_smoke", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
realtime_smoke = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec: the script's @dataclass on SmokeResult
# resolves its (deferred, `from __future__ import annotations`) type hints by
# looking the module up in sys.modules — that lookup fails on a module that
# hasn't been registered yet.
sys.modules[_spec.name] = realtime_smoke
_spec.loader.exec_module(realtime_smoke)


def _reader_from(data: bytes):
    """A ``recv_exact``-shaped callable over static bytes: short-reads at EOF,
    never raises — the same contract a real socket helper follows."""
    buf = io.BytesIO(data)

    def _recv(n: int) -> bytes:
        return buf.read(n)

    return _recv


# --- import isolation --------------------------------------------------------


def test_module_imports_without_torch_or_third_party_ws_client() -> None:
    src = _SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden = ("torch", "fastapi", "numpy", "scipy", "websocket", "websockets", "aiohttp")
    offenders = [
        name
        for name in forbidden
        for line in src.splitlines()
        if line.strip().startswith((f"import {name}", f"from {name}"))
    ]
    assert not offenders, f"realtime-smoke.py imports a forbidden dep: {offenders}"


def test_module_does_not_import_the_lobes_package_either() -> None:
    """Issue #151's wire migration deliberately does NOT import
    ``lobes.realtime._wire`` even though that module is itself stdlib-only.

    This script already reimplements RFC 6455 by hand instead of importing
    any WebSocket library, specifically so it stays a single droppable file
    with zero repo-context dependencies — copyable to another machine or
    even another project's tree and runnable with nothing but ``python3``.
    Importing ``lobes.realtime._wire`` would require the whole ``lobes``
    package tree (or a pip-installed ``lobes-cli``) to be importable from
    wherever this script lands, breaking exactly that property. The
    base64-event codec this file needs is a few lines of
    ``base64.b64encode``/``b64decode`` around a JSON dict — small enough
    that inlining it costs nothing, so the "no repo-context import" rule
    wins. See ``test_build_append_event_round_trips_through_the_real_server_side_decoder``
    and its delta-side counterpart below for the offline proof that the
    inlined encode/decode stays wire-compatible with the real
    ``lobes.realtime._wire`` codec despite not importing it.
    """
    src = _SCRIPT_PATH.read_text(encoding="utf-8")
    offenders = [
        line for line in src.splitlines() if line.strip().startswith(("import lobes", "from lobes"))
    ]
    assert not offenders, f"realtime-smoke.py imports the lobes package: {offenders}"


# --- Sec-WebSocket-Key / Sec-WebSocket-Accept -------------------------------


def test_compute_accept_key_matches_the_rfc6455_worked_example() -> None:
    # RFC 6455 SS1.3's own worked example — the canonical correctness check
    # for this computation, not just a round trip against our own code.
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    assert realtime_smoke.compute_accept_key(key) == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_make_sec_websocket_key_is_16_random_bytes_base64_encoded() -> None:
    key = realtime_smoke.make_sec_websocket_key()
    decoded = base64.b64decode(key)
    assert len(decoded) == 16
    # Two independent calls must not collide (this would indicate a broken
    # RNG or a hardcoded nonce, either of which breaks the RFC's replay
    # protection intent for the handshake).
    assert realtime_smoke.make_sec_websocket_key() != key


# --- handshake request / response ------------------------------------------


def test_build_handshake_request_carries_every_mandatory_header() -> None:
    req = realtime_smoke.build_handshake_request(
        "gateway:8000",
        "/v1/realtime?input_sample_rate=24000",
        "dGhlIHNhbXBsZSBub25jZQ==",
        extra_headers={"Authorization": "Bearer secret"},
    )
    text = req.decode("latin-1")
    assert text.startswith("GET /v1/realtime?input_sample_rate=24000 HTTP/1.1\r\n")
    assert "Host: gateway:8000\r\n" in text
    assert "Upgrade: websocket\r\n" in text
    assert "Connection: Upgrade\r\n" in text
    assert "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" in text
    assert "Sec-WebSocket-Version: 13\r\n" in text
    assert "Authorization: Bearer secret\r\n" in text
    assert text.endswith("\r\n\r\n")


def test_build_handshake_request_omits_authorization_when_no_api_key() -> None:
    req = realtime_smoke.build_handshake_request(
        "gateway:8000", "/v1/realtime", "key==", extra_headers=None
    )
    assert b"Authorization" not in req


def test_parse_response_head_extracts_status_and_lowercases_header_names() -> None:
    head = (
        b"HTTP/1.1 101 Switching Protocols\r\n"
        b"Upgrade: websocket\r\n"
        b"Connection: Upgrade\r\n"
        b"Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n"
        b"\r\n"
    )
    status, headers = realtime_smoke.parse_response_head(head)
    assert status == 101
    assert headers["sec-websocket-accept"] == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    assert headers["upgrade"] == "websocket"


def test_parse_response_head_handles_a_refusal_status() -> None:
    head = b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n"
    status, headers = realtime_smoke.parse_response_head(head)
    assert status == 404
    assert headers["content-type"] == "application/json"


def test_parse_response_head_is_zero_on_unparseable_status_line() -> None:
    status, _headers = realtime_smoke.parse_response_head(b"not even http\r\n\r\n")
    assert status == 0


# --- masking -----------------------------------------------------------------


def test_mask_payload_is_its_own_inverse() -> None:
    payload = b"the quick brown fox"
    mask_key = b"\x01\x02\x03\x04"
    masked = realtime_smoke.mask_payload(payload, mask_key)
    assert masked != payload  # a real mask actually changes the bytes
    assert realtime_smoke.mask_payload(masked, mask_key) == payload


def test_mask_payload_rejects_a_mask_key_of_the_wrong_length() -> None:
    with pytest.raises(ValueError):
        realtime_smoke.mask_payload(b"data", b"\x00\x00\x00")


# --- frame build / parse round trips ----------------------------------------


def test_build_frame_then_read_frame_round_trips_a_small_masked_binary_frame() -> None:
    payload = b"\x00\x01" * 100  # 200 bytes, well under the 126-byte boundary
    frame = realtime_smoke.build_frame(realtime_smoke.OPCODE_BINARY, payload, mask=True)
    fin, opcode, got = realtime_smoke.read_frame(_reader_from(frame))
    assert fin is True
    assert opcode == realtime_smoke.OPCODE_BINARY
    assert got == payload


def test_build_frame_then_read_frame_round_trips_an_unmasked_text_frame() -> None:
    # Server->client frames are never masked (RFC 6455 SS5.1) — the reader
    # must handle that just as correctly as a masked one.
    payload = b'{"type": "session.created"}'
    frame = realtime_smoke.build_frame(realtime_smoke.OPCODE_TEXT, payload, mask=False)
    fin, opcode, got = realtime_smoke.read_frame(_reader_from(frame))
    assert fin is True
    assert opcode == realtime_smoke.OPCODE_TEXT
    assert got == payload


def test_build_frame_uses_16bit_extended_length_above_125_bytes() -> None:
    payload = b"x" * 5000  # forces the 126 marker + 16-bit length path
    frame = realtime_smoke.build_frame(realtime_smoke.OPCODE_BINARY, payload, mask=True)
    # Byte 1's low 7 bits (mask bit already stripped) must read 126, and the
    # next two bytes must decode to the true length.
    assert (frame[1] & 0x7F) == 126
    assert struct.unpack("!H", frame[2:4])[0] == 5000
    _fin, _opcode, got = realtime_smoke.read_frame(_reader_from(frame))
    assert got == payload


def test_build_frame_empty_payload_round_trips() -> None:
    frame = realtime_smoke.build_frame(realtime_smoke.OPCODE_CLOSE, b"", mask=True)
    fin, opcode, got = realtime_smoke.read_frame(_reader_from(frame))
    assert fin is True
    assert opcode == realtime_smoke.OPCODE_CLOSE
    assert got == b""


def test_read_frame_raises_on_a_header_truncated_before_two_bytes() -> None:
    reader = _reader_from(b"\x81")
    with pytest.raises(realtime_smoke.FrameReadError):
        realtime_smoke.read_frame(reader)


def test_read_frame_raises_on_a_payload_truncated_mid_frame() -> None:
    full = realtime_smoke.build_frame(realtime_smoke.OPCODE_TEXT, b"hello world", mask=False)
    truncated = full[:-3]  # header intact, payload cut short
    reader = _reader_from(truncated)
    with pytest.raises(realtime_smoke.FrameReadError):
        realtime_smoke.read_frame(reader)


# --- PCM chunking / silence ---------------------------------------------------


def test_chunk_pcm_splits_evenly_and_keeps_a_short_final_remainder() -> None:
    data = bytes(range(10))
    chunks = list(realtime_smoke.chunk_pcm(data, 4))
    assert chunks == [bytes(range(0, 4)), bytes(range(4, 8)), bytes(range(8, 10))]


def test_chunk_pcm_rejects_a_non_positive_chunk_size() -> None:
    # chunk_pcm is a generator: building it cannot raise, iterating it does.
    chunks = realtime_smoke.chunk_pcm(b"abcd", 0)
    with pytest.raises(ValueError):
        list(chunks)


def test_silence_bytes_is_all_zero_and_the_right_length() -> None:
    silence = realtime_smoke.silence_bytes(duration_ms=100, sample_rate=16000)
    # 100ms @ 16kHz = 1600 samples * 2 bytes/sample = 3200 bytes.
    assert len(silence) == 3200
    assert silence == b"\x00" * 3200


def test_bytes_per_chunk_for_rate_matches_the_segmenters_own_framing_constant() -> None:
    # lobes.realtime._segmenter.CHUNK_BYTES is the server's own 512-sample/
    # 32ms framing at its native 16kHz — this script's chunking at 16000 Hz
    # must agree with it exactly, not just approximately.
    from lobes.realtime._segmenter import CHUNK_BYTES

    assert realtime_smoke.bytes_per_chunk_for_rate(16000) == CHUNK_BYTES == 1024


def test_bytes_per_chunk_for_rate_at_the_tts_native_24k_rate() -> None:
    # 24000 Hz * 32ms = 768 samples * 2 bytes/sample = 1536 bytes.
    assert realtime_smoke.bytes_per_chunk_for_rate(24000) == 1536


# --- base64 event wire (issue #151): input_audio_buffer.append / response.audio.delta ----
#
# #149 shipped audio-in as raw BINARY WebSocket frames; #151 supersedes that with the
# OpenAI-shaped base64 JSON wire in both directions. These are the pure encode/decode
# helpers this script and scripts/realtime-voice-loop.py (which imports this module)
# now send/receive instead of a raw binary frame.


def test_build_append_event_wraps_pcm_as_base64_input_audio_buffer_append() -> None:
    pcm = b"\x01\x02\x03\x04\x05\x06"
    event = realtime_smoke.build_append_event(pcm)
    assert event["type"] == "input_audio_buffer.append"
    assert base64.b64decode(event["audio"]) == pcm


def test_build_append_event_round_trips_empty_audio() -> None:
    # Zero bytes of audio is a valid (if odd) chunk — never an error.
    event = realtime_smoke.build_append_event(b"")
    assert event["audio"] == ""
    assert base64.b64decode(event["audio"]) == b""


def test_build_append_event_round_trips_through_the_real_server_side_decoder() -> None:
    """Cross-checks this script's own encode against the REAL server-side
    decoder (``lobes.realtime._wire.parse_append_event``) — proof that the
    inlined encode (see ``test_module_does_not_import_the_lobes_package_either``)
    stays byte-exact wire-compatible with the server without this script
    ever importing ``lobes`` itself. Only the TEST imports ``lobes``.
    """
    from lobes.realtime._wire import parse_append_event

    pcm = bytes(range(256)) * 4  # 1024 bytes, exercises every byte value
    event = realtime_smoke.build_append_event(pcm)
    assert parse_append_event(event) == pcm


def test_decode_audio_delta_event_extracts_the_base64_delta_field() -> None:
    pcm = b"\x10\x20" * 50
    event = {
        "type": "response.audio.delta",
        "event_id": "evt_1",
        "delta": base64.b64encode(pcm).decode("ascii"),
    }
    assert realtime_smoke.decode_audio_delta_event(event) == pcm


def test_decode_audio_delta_event_rejects_a_missing_delta_field() -> None:
    with pytest.raises(ValueError):
        realtime_smoke.decode_audio_delta_event({"type": "response.audio.delta"})


def test_decode_audio_delta_event_rejects_a_non_string_delta_field() -> None:
    with pytest.raises(ValueError):
        realtime_smoke.decode_audio_delta_event({"type": "response.audio.delta", "delta": 123})


def test_decode_audio_delta_event_round_trips_the_real_server_side_serializer() -> None:
    """Cross-checks this script's decode against the REAL server-side
    serializer (``lobes.realtime._wire.serialize_audio_delta``) — the
    outbound-direction counterpart of the append round-trip test above.
    """
    from lobes.realtime._wire import serialize_audio_delta

    pcm = bytes(range(256)) * 3
    event = serialize_audio_delta(pcm, response_id="resp_1", item_id="item_1")
    assert realtime_smoke.decode_audio_delta_event(event) == pcm


# --- phrase / keyword matching ------------------------------------------------


def test_normalize_text_lowercases_and_strips_punctuation() -> None:
    assert realtime_smoke.normalize_text("Reachy is Online!") == "reachy is online"


def test_missing_keywords_reports_only_the_absent_ones() -> None:
    missing = realtime_smoke.missing_keywords("the quick brown fox jumps", ["quick", "fox", "dog"])
    assert missing == ["dog"]


def test_missing_keywords_is_case_insensitive() -> None:
    assert realtime_smoke.missing_keywords("QUICK BROWN FOX", ["quick", "fox"]) == []


def test_keywords_for_phrase_uses_the_curated_list_for_the_default_phrase() -> None:
    assert (
        realtime_smoke.keywords_for_phrase(realtime_smoke.DEFAULT_PHRASE)
        == realtime_smoke.DEFAULT_PHRASE_KEYWORDS
    )


def test_keywords_for_phrase_derives_words_for_a_custom_phrase() -> None:
    assert realtime_smoke.keywords_for_phrase("Reachy is online.") == (
        "reachy",
        "is",
        "online",
    )


# --- WS target derivation -----------------------------------------------------


def test_build_realtime_ws_target_derives_host_port_and_query() -> None:
    scheme, host, port, path = realtime_smoke.build_realtime_ws_target(
        "http://localhost:8000", 24000
    )
    assert scheme == "ws"
    assert host == "localhost"
    assert port == 8000
    assert path.startswith("/v1/realtime?")
    assert "input_sample_rate=24000" in path
    assert "turn_detection=server_vad" in path


def test_build_realtime_ws_target_uses_wss_for_an_https_base_url() -> None:
    scheme, _host, port, _path = realtime_smoke.build_realtime_ws_target(
        "https://gateway.example.ts.net", 16000
    )
    assert scheme == "wss"
    assert port == 443


# --- the criterion-5 honesty classification -----------------------------------


def test_classify_event_or_timeout_on_the_expected_event_passes() -> None:
    ok, detail = realtime_smoke.classify_event_or_timeout(
        {"type": "session.created"}, "session.created"
    )
    assert ok is True
    assert "session.created" in detail


def test_classify_event_or_timeout_distinguishes_a_true_timeout() -> None:
    ok, detail = realtime_smoke.classify_event_or_timeout(None, "input_audio_buffer.speech_started")
    assert ok is False
    assert "TIMEOUT" in detail


def test_classify_event_or_timeout_distinguishes_a_named_error_from_a_timeout() -> None:
    ok, detail = realtime_smoke.classify_event_or_timeout(
        {"type": "error", "code": "vad_unavailable", "message": "silero failed to load"},
        "input_audio_buffer.speech_started",
    )
    assert ok is False
    assert "vad_unavailable" in detail
    assert "TIMEOUT" not in detail


def test_classify_event_or_timeout_never_conflates_error_and_timeout_wording() -> None:
    # The literal honesty requirement (criterion 5): the two failure messages
    # must read differently, so a human or a CI log grep can always tell them
    # apart — "no events" is never allowed to look like an explicit error.
    _ok1, timeout_detail = realtime_smoke.classify_event_or_timeout(None, "x")
    _ok2, error_detail = realtime_smoke.classify_event_or_timeout(
        {"type": "error", "code": "vad_unavailable", "message": "m"}, "x"
    )
    assert timeout_detail != error_detail
    # The timeout message never mentions an error code/message — nothing
    # arrived, so there is nothing error-shaped to report.
    assert "code=" not in timeout_detail
    assert "vad_unavailable" not in timeout_detail
    # The error message never CLAIMS to be a timeout (it may mention the word
    # only to explicitly rule it out, e.g. "not a timeout" — that contrast is
    # the point) — it must not start with, or equal, timeout-shaped wording.
    assert not error_detail.lower().startswith("timeout")
    assert "code=" in error_detail
    assert "vad_unavailable" in error_detail


def test_classify_event_or_timeout_flags_an_unexpected_event_type() -> None:
    ok, detail = realtime_smoke.classify_event_or_timeout(
        {"type": "session.closed"}, "session.created"
    )
    assert ok is False
    assert "session.closed" in detail
    assert "session.created" in detail


# --- mid-frame timeout must not desync the reader --------------------------


def test_a_timeout_between_header_and_payload_does_not_corrupt_the_next_read() -> None:
    """A frame split across TCP segments with a gap longer than the timeout.

    The reader must resume at the true frame boundary; if the consumed header
    were dropped, the retry would read the payload bytes as a new header and
    mis-parse everything after it. Uses a REAL socketpair — since the client
    waits with select(), a fake without a fileno() cannot model this.
    """
    client_sock, peer = socket.socketpair()
    try:
        client = realtime_smoke.WebSocketClient(client_sock)
        text = b'{"type":"session.created"}'
        frame = realtime_smoke.build_frame(realtime_smoke.OPCODE_TEXT, text, mask=False)
        peer.sendall(frame[:2])  # header only, then a gap
        with pytest.raises(socket.timeout):
            client.read_frame(timeout=0.05)
        peer.sendall(frame[2:])  # the rest arrives late
        fin, opcode, payload = client.read_frame(timeout=2.0)
        assert fin is True
        assert opcode == realtime_smoke.OPCODE_TEXT
        assert payload == text  # NOT garbage from a mis-read header
    finally:
        client_sock.close()
        peer.close()


# --- duplex safety: a reader must not mutate shared socket state -------------


def test_read_timeout_never_puts_a_deadline_on_the_shared_socket() -> None:
    """A reader's timeout must not leak onto a concurrent writer.

    ``settimeout`` is a property of the SOCKET, not of one call, so a reader
    that sets it hands its deadline to any other thread writing on the same
    socket. A large-enough ``sendall`` can then time out part-sent, and a
    partially written frame desynchronises the peer until it closes the
    connection. Found live: a voice loop that listens while it talks dies with
    BrokenPipeError after a few turns, while the one-shot smoke run (stream,
    then read) never trips it.
    """
    client_sock, peer = socket.socketpair()
    try:
        client = realtime_smoke.WebSocketClient(client_sock)
        assert client_sock.gettimeout() is None  # blocking to begin with
        with pytest.raises(socket.timeout):
            client.read_frame(timeout=0.05)  # nothing to read → times out
        # The socket must be exactly as we found it: no deadline inherited by
        # whichever thread writes next.
        assert client_sock.gettimeout() is None
    finally:
        client_sock.close()
        peer.close()


def test_a_writer_survives_a_concurrent_reader_waiting_on_a_timeout() -> None:
    """Write while another thread is blocked in read_frame — the real shape."""
    client_sock, peer = socket.socketpair()
    try:
        client = realtime_smoke.WebSocketClient(client_sock)
        errors: list[BaseException] = []

        def reader() -> None:
            for _ in range(3):
                try:
                    client.read_frame(timeout=0.2)
                except socket.timeout:
                    pass
                except BaseException as exc:  # noqa: BLE001 - recorded, asserted below
                    errors.append(exc)

        t = threading.Thread(target=reader)
        t.start()
        for _ in range(20):  # keep writing while the reader waits
            client.send_binary(b"\x00" * 1024)
        t.join(timeout=5)
        # Assert the join actually completed: if the reader ever deadlocks,
        # this is the line that should say so.
        assert not t.is_alive(), "reader thread did not finish — possible deadlock"

        assert not errors, f"reader raised while a writer was active: {errors}"
        # 1024 > 125, so each frame carries a 16-bit extended length:
        # 2 base + 2 length + 4 mask = 8 bytes of overhead.
        expected = 20 * (1024 + 8)
        # Drain until the full total arrives: a stream socket may return fewer
        # bytes than asked for even when more are queued, so a single recv()
        # would be flaky rather than wrong.
        got = bytearray()
        peer.settimeout(5)
        while len(got) < expected:
            chunk = peer.recv(expected - len(got))
            if not chunk:
                break
            got.extend(chunk)
        # Every frame arrived intact and in order — no partial/torn writes.
        assert len(got) == expected
    finally:
        client_sock.close()
        peer.close()


# --- acceptance grep gate (issue #151 t9): no in-repo client sends raw binary audio --


def test_no_in_repo_client_calls_send_binary_for_audio() -> None:
    """Issue #151's honesty condition: "after the PR, grep finds no in-repo
    client sending raw binary audio frames." #149 shipped audio-in as raw
    BINARY WebSocket frames; #151 supersedes that with the base64
    ``input_audio_buffer.append`` JSON event wire (:func:`build_append_event`
    / :meth:`WebSocketClient.send_json_event`).

    ``send_binary``/``OPCODE_BINARY`` stay defined in this script as a
    generic RFC 6455 primitive (frame-level tests above still exercise them
    directly), so this checks for an actual CALL SITE — ``.send_binary(`` —
    in either in-repo client's source, not the method's mere existence.
    """
    voice_loop_path = _SCRIPT_PATH.parent / "realtime-voice-loop.py"
    for path in (_SCRIPT_PATH, voice_loop_path):
        src = path.read_text(encoding="utf-8")
        call_sites = [
            line for line in src.splitlines() if ".send_binary(" in line and "def " not in line
        ]
        assert not call_sites, f"{path.name} still sends raw binary audio: {call_sites}"
