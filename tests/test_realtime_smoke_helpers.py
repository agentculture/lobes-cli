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

import importlib.util
import io
import socket
import struct
import sys
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


# --- Sec-WebSocket-Key / Sec-WebSocket-Accept -------------------------------


def test_compute_accept_key_matches_the_rfc6455_worked_example() -> None:
    # RFC 6455 SS1.3's own worked example — the canonical correctness check
    # for this computation, not just a round trip against our own code.
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    assert realtime_smoke.compute_accept_key(key) == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_make_sec_websocket_key_is_16_random_bytes_base64_encoded() -> None:
    import base64

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


class _StutteringSock:
    """Delivers a frame's header, then times out, then delivers the rest.

    Models the real hazard: a frame split across TCP segments with a gap
    longer than the per-read timeout, landing between the base header and the
    payload.
    """

    def __init__(self, payload: bytes) -> None:
        header = struct.pack("!BB", 0x81, len(payload))
        self._script = [header, socket.timeout("timed out"), payload]

    def settimeout(self, _t) -> None:
        pass

    def recv(self, _n: int) -> bytes:
        if not self._script:
            return b""
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_a_timeout_between_header_and_payload_does_not_corrupt_the_next_read() -> None:
    text = b'{"type":"session.created"}'
    client = realtime_smoke.WebSocketClient.__new__(realtime_smoke.WebSocketClient)
    client._sock = _StutteringSock(text)
    client._buf = bytearray()

    with pytest.raises(socket.timeout):
        client.read_frame(timeout=0.01)  # dies after consuming the header

    # The retry must see the SAME frame, not the payload mis-read as a header.
    fin, opcode, payload = client.read_frame(timeout=0.01)
    assert fin is True
    assert opcode == realtime_smoke.OPCODE_TEXT
    assert payload == text
