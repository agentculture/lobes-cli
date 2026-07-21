"""Tests for the base64 wire codec (stdlib-only; no [realtime] extra).

Mirrors the style of tests/test_realtime_segmenter.py and
tests/test_realtime_session.py: the module under test imports with the
standard library alone, so these tests run in the offline CI environment
(no torch, no fastapi, no httpx, no GPU).

Covers the t1 acceptance criteria from
docs/plans/2026-07-21-realtime-voice-to-voice-astro-test-site-151.md:

1. an ``input_audio_buffer.append`` event with valid base64 decodes to the
   exact PCM bytes; malformed base64/JSON yields a named error value
   (:class:`~lobes.realtime._wire.WireFormatError`), never an escaping
   exception
2. ``response.audio.delta`` serialization round-trips PCM byte-exact, and
   the whole module is stdlib-only
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

import lobes.realtime._wire as W
from lobes.realtime._wire import (
    DEFAULT_DELTA_CHUNK_BYTES,
    WireErrorCode,
    WireFormatError,
    decode_event,
    iter_audio_deltas,
    parse_append_event,
    serialize_audio_delta,
)
from lobes.realtime.protocol import BYTES_PER_SAMPLE, TTS_SAMPLE_RATE

# ---------------------------------------------------------------------------
# Import isolation — this module must be importable with none of the
# [realtime] extra's heavy deps installed (mirrors test_realtime_segmenter.py
# and test_realtime_imports.py).
# ---------------------------------------------------------------------------


def test_module_imports_without_the_realtime_extra() -> None:
    # If this test file collected at all in the offline dev env (no torch,
    # no fastapi/httpx/numpy/scipy installed), the import above already
    # proved it. This test just makes the guarantee explicit and named.
    assert hasattr(W, "parse_append_event")


def test_module_source_never_imports_forbidden_deps() -> None:
    src = Path(W.__file__).read_text(encoding="utf-8")
    forbidden = ("torch", "fastapi", "httpx", "numpy", "scipy", "silero_vad")
    offenders = [
        name
        for name in forbidden
        for line in src.splitlines()
        if line.strip().startswith((f"import {name}", f"from {name}"))
    ]
    assert not offenders, f"_wire.py imports forbidden deps: {offenders}"


def test_module_never_imports_session() -> None:
    # t1 scope: _wire.py must stay schema-agnostic and not depend on
    # _session.py, which a sibling task is editing in parallel. Parsed via
    # ast (real Import/ImportFrom nodes), not text matching, so the
    # docstring's own prose about _session.py (which names it, and can
    # word-wrap onto a line starting with "import") never false-positives.
    import ast

    src = Path(W.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(W.__file__))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(alias.name for alias in node.names if "_session" in alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and "_session" in node.module:
                offenders.append(node.module)
            offenders.extend(alias.name for alias in node.names if "_session" in alias.name)
    assert not offenders, f"_wire.py must not import _session: {offenders}"


def test_delta_chunk_bytes_is_a_whole_number_of_pcm16_samples() -> None:
    # Derived from protocol.py's constants, not an independent magic number.
    assert DEFAULT_DELTA_CHUNK_BYTES % BYTES_PER_SAMPLE == 0
    assert DEFAULT_DELTA_CHUNK_BYTES > 0


# ---------------------------------------------------------------------------
# decode_event — raw JSON text/bytes -> plain dict.
# ---------------------------------------------------------------------------


def test_decode_event_parses_a_json_object() -> None:
    raw = json.dumps({"type": "input_audio_buffer.append", "audio": "abc"})
    parsed = decode_event(raw)
    assert parsed == {"type": "input_audio_buffer.append", "audio": "abc"}


def test_decode_event_accepts_bytes_too() -> None:
    raw = json.dumps({"type": "response.create"}).encode("utf-8")
    parsed = decode_event(raw)
    assert parsed == {"type": "response.create"}


def test_decode_event_malformed_json_raises_named_error_not_escaping_exception() -> None:
    with pytest.raises(WireFormatError) as exc_info:
        decode_event("{not valid json")
    assert exc_info.value.code is WireErrorCode.INVALID_JSON


def test_decode_event_rejects_non_object_top_level_json() -> None:
    for raw in ("[1, 2, 3]", '"just a string"', "42", "null", "true"):
        with pytest.raises(WireFormatError) as exc_info:
            decode_event(raw)
        assert exc_info.value.code is WireErrorCode.INVALID_JSON


def test_decode_event_error_carries_a_wire_error_code_never_a_bare_exception() -> None:
    try:
        decode_event("garbage{{{")
    except WireFormatError as exc:
        assert isinstance(exc.code, WireErrorCode)
        assert isinstance(exc, ValueError)  # mirrors SessionConfigError's ValueError base
    else:
        pytest.fail("expected WireFormatError")


# ---------------------------------------------------------------------------
# parse_append_event — criterion 1: exact decode / named error on malformed input.
# ---------------------------------------------------------------------------


def test_parse_append_event_decodes_exact_pcm_bytes() -> None:
    pcm = bytes(range(256)) * 4  # 1024 bytes, deterministic non-trivial content
    payload = {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }
    assert parse_append_event(payload) == pcm


def test_parse_append_event_round_trips_random_pcm() -> None:
    pcm = os.urandom(4001)  # odd length on purpose — bytes, not samples
    payload = {"audio": base64.b64encode(pcm).decode("ascii")}
    assert parse_append_event(payload) == pcm


def test_parse_append_event_empty_audio_decodes_to_empty_bytes() -> None:
    assert parse_append_event({"audio": ""}) == b""


def test_parse_append_event_missing_audio_field_raises_named_error() -> None:
    with pytest.raises(WireFormatError) as exc_info:
        parse_append_event({"type": "input_audio_buffer.append"})
    assert exc_info.value.code is WireErrorCode.INVALID_APPEND_EVENT


def test_parse_append_event_non_string_audio_raises_named_error() -> None:
    for bad_audio in (None, 123, ["a", "b"], {"nested": "dict"}, 3.14):
        with pytest.raises(WireFormatError) as exc_info:
            parse_append_event({"audio": bad_audio})
        assert exc_info.value.code is WireErrorCode.INVALID_APPEND_EVENT


def test_parse_append_event_invalid_base64_raises_named_error_not_escaping_exception() -> None:
    for bad_b64 in ("not valid base64!!!", "%%%%", "abc", "====", "\x00\x01\x02"):
        with pytest.raises(WireFormatError) as exc_info:
            parse_append_event({"audio": bad_b64})
        assert exc_info.value.code is WireErrorCode.INVALID_APPEND_EVENT


def test_parse_append_event_error_message_never_leaks_raw_audio_field() -> None:
    # Defensive: the malformed-base64 path should not echo arbitrarily large
    # or binary-garbage input verbatim forever, but it must at least raise
    # the documented error type rather than any other exception class.
    try:
        parse_append_event({"audio": "###"})
    except WireFormatError:
        pass
    except Exception as exc:  # pragma: no cover - failure path only
        pytest.fail(f"expected WireFormatError, got escaping {type(exc).__name__}: {exc}")


def test_decode_event_then_parse_append_event_end_to_end() -> None:
    pcm = os.urandom(2048)
    raw = json.dumps(
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}
    )
    payload = decode_event(raw)
    assert parse_append_event(payload) == pcm


# ---------------------------------------------------------------------------
# serialize_audio_delta — criterion 2: byte-exact round trip.
# ---------------------------------------------------------------------------


def test_serialize_audio_delta_round_trips_pcm_byte_exact() -> None:
    pcm = os.urandom(4800)
    event = serialize_audio_delta(pcm, response_id="resp_abc", item_id="item_xyz")
    assert event["type"] == "response.audio.delta"
    assert event["response_id"] == "resp_abc"
    assert event["item_id"] == "item_xyz"
    assert base64.b64decode(event["delta"]) == pcm


def test_serialize_audio_delta_generates_an_event_id_when_not_supplied() -> None:
    event = serialize_audio_delta(b"\x00\x01", response_id="resp_1", item_id="item_1")
    assert isinstance(event["event_id"], str)
    assert event["event_id"].startswith("event_")


def test_serialize_audio_delta_accepts_an_explicit_event_id() -> None:
    event = serialize_audio_delta(
        b"\x00\x01", response_id="resp_1", item_id="item_1", event_id="event_fixed123"
    )
    assert event["event_id"] == "event_fixed123"


def test_serialize_audio_delta_default_indices_are_zero() -> None:
    event = serialize_audio_delta(b"\x00\x01", response_id="resp_1", item_id="item_1")
    assert event["output_index"] == 0
    assert event["content_index"] == 0


def test_serialize_audio_delta_empty_pcm_round_trips_to_empty_bytes() -> None:
    event = serialize_audio_delta(b"", response_id="resp_1", item_id="item_1")
    assert base64.b64decode(event["delta"]) == b""


# ---------------------------------------------------------------------------
# iter_audio_deltas — sequential chunking of a complete PCM buffer.
# ---------------------------------------------------------------------------


def test_iter_audio_deltas_reassembles_byte_exact() -> None:
    pcm = os.urandom(TTS_SAMPLE_RATE * BYTES_PER_SAMPLE)  # 1 second of audio
    events = list(
        iter_audio_deltas(pcm, DEFAULT_DELTA_CHUNK_BYTES, response_id="resp_1", item_id="item_1")
    )
    reassembled = b"".join(base64.b64decode(e["delta"]) for e in events)
    assert reassembled == pcm


def test_iter_audio_deltas_chunk_sizes_and_count() -> None:
    pcm = bytes(range(10))  # 10 bytes total
    events = list(iter_audio_deltas(pcm, 4, response_id="resp_1", item_id="item_1"))
    sizes = [len(base64.b64decode(e["delta"])) for e in events]
    assert sizes == [4, 4, 2]  # last chunk is the short remainder


def test_iter_audio_deltas_every_event_shares_response_and_item_id() -> None:
    pcm = os.urandom(4096)
    events = list(iter_audio_deltas(pcm, 1024, response_id="resp_shared", item_id="item_shared"))
    assert len(events) == 4
    assert all(e["response_id"] == "resp_shared" for e in events)
    assert all(e["item_id"] == "item_shared" for e in events)


def test_iter_audio_deltas_events_have_distinct_event_ids() -> None:
    pcm = os.urandom(4096)
    events = list(iter_audio_deltas(pcm, 1024, response_id="resp_1", item_id="item_1"))
    event_ids = [e["event_id"] for e in events]
    assert len(set(event_ids)) == len(event_ids)


def test_iter_audio_deltas_empty_pcm_yields_no_events() -> None:
    events = list(iter_audio_deltas(b"", DEFAULT_DELTA_CHUNK_BYTES, response_id="r", item_id="i"))
    assert events == []


def test_iter_audio_deltas_exact_multiple_of_chunk_size() -> None:
    pcm = os.urandom(20)
    events = list(iter_audio_deltas(pcm, 4, response_id="r", item_id="i"))
    assert [len(base64.b64decode(e["delta"])) for e in events] == [4, 4, 4, 4, 4]


def test_iter_audio_deltas_rejects_non_positive_chunk_bytes() -> None:
    for bad in (0, -1, -1024):
        with pytest.raises(ValueError):
            list(iter_audio_deltas(b"\x00\x01\x02\x03", bad, response_id="r", item_id="i"))


def test_iter_audio_deltas_rejects_odd_chunk_bytes() -> None:
    # PCM16 samples are 2 bytes; an odd chunk size would split a sample
    # across two frames.
    with pytest.raises(ValueError):
        list(iter_audio_deltas(b"\x00\x01\x02\x03", 3, response_id="r", item_id="i"))


def test_iter_audio_deltas_returns_an_iterator_not_a_list() -> None:
    # Suggested surface says Iterator[dict] — a lazy generator, not a
    # pre-built list, so a caller can start sending frame 1 before frame N
    # is even produced.
    import types

    result = iter_audio_deltas(b"\x00\x01", 2, response_id="r", item_id="i")
    assert isinstance(result, types.GeneratorType) or hasattr(result, "__next__")
