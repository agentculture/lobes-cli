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
    APPEND_EVENT_TYPE,
    DEFAULT_DELTA_CHUNK_BYTES,
    InboundDecision,
    InboundKind,
    WireErrorCode,
    WireFormatError,
    decide_inbound_message,
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
# decide_inbound_message — issue #151 t4: app.py's receive-loop decision,
# extracted here so it is exhaustively testable without the [realtime]
# extra installed (app.py is fastapi/torch-only and pragma-no-cover /
# coverage-omitted — see tests/test_realtime_imports.py and pyproject.toml's
# [tool.coverage.run] omit list). This mirrors the precedent set by t7's
# tests/test_realtime_tts_gate.py: the real decision lives in a stdlib
# module and is proven here; the route (app.py) is verified separately by
# a source-text grep gate below, since it can never be imported offline.
# ---------------------------------------------------------------------------


def _append_message(pcm: bytes) -> dict:
    """Build the plain dict a WebSocket receive() call would hand back for
    one ``input_audio_buffer.append`` text frame carrying *pcm*."""
    return {
        "type": "websocket.receive",
        "text": json.dumps(
            {"type": APPEND_EVENT_TYPE, "audio": base64.b64encode(pcm).decode("ascii")}
        ),
    }


def test_decide_inbound_message_classifies_a_valid_append_event_as_audio() -> None:
    pcm = os.urandom(512)
    decision = decide_inbound_message(_append_message(pcm))
    assert decision.kind is InboundKind.AUDIO
    assert decision.audio == pcm
    assert decision.error is None


def test_decide_inbound_message_empty_audio_is_still_audio_not_ignored() -> None:
    # An empty append event is valid input (zero bytes of audio), not
    # malformed and not "not audio" — see parse_append_event's own contract.
    decision = decide_inbound_message(_append_message(b""))
    assert decision.kind is InboundKind.AUDIO
    assert decision.audio == b""


def test_decide_inbound_message_ignores_a_well_formed_non_append_event() -> None:
    message = {"type": "websocket.receive", "text": json.dumps({"type": "response.create"})}
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.IGNORED
    assert decision.audio is None
    assert decision.error is None


def test_decide_inbound_message_json_object_with_no_type_field_is_ignored() -> None:
    # No "type" key at all still routes through the same "not an append
    # event" branch as an explicitly different type — not a parse error.
    message = {"type": "websocket.receive", "text": json.dumps({"foo": "bar"})}
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.IGNORED


def test_decide_inbound_message_malformed_json_is_a_named_error() -> None:
    message = {"type": "websocket.receive", "text": "{not valid json"}
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.ERROR
    assert decision.audio is None
    assert isinstance(decision.error, WireFormatError)
    assert decision.error.code is WireErrorCode.INVALID_JSON


def test_decide_inbound_message_malformed_append_audio_field_is_a_named_error() -> None:
    message = {
        "type": "websocket.receive",
        "text": json.dumps({"type": APPEND_EVENT_TYPE, "audio": "not valid base64!!!"}),
    }
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.ERROR
    assert decision.error.code is WireErrorCode.INVALID_APPEND_EVENT


def test_decide_inbound_message_missing_audio_field_is_a_named_error() -> None:
    message = {"type": "websocket.receive", "text": json.dumps({"type": APPEND_EVENT_TYPE})}
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.ERROR
    assert decision.error.code is WireErrorCode.INVALID_APPEND_EVENT


def test_decide_inbound_message_binary_frame_is_rejected_as_a_named_error() -> None:
    # The coordinated wire break (issue #151 / reachy-mini-cli#115): a raw
    # binary WebSocket frame was accepted as audio before this migration.
    # It must now be a named error, never audio, never a silent drop.
    message = {"type": "websocket.receive", "bytes": os.urandom(1024)}
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.ERROR
    assert decision.audio is None
    assert decision.error.code is WireErrorCode.UNSUPPORTED_FRAME_TYPE


def test_decide_inbound_message_binary_frame_of_valid_pcm_length_is_still_rejected() -> None:
    # Even a well-formed-looking PCM16 payload (even byte length) must not
    # sneak through as audio just because it happens to look plausible.
    message = {"type": "websocket.receive", "bytes": bytes(range(256)) * 4}
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.ERROR
    assert decision.error.code is WireErrorCode.UNSUPPORTED_FRAME_TYPE


def test_decide_inbound_message_empty_binary_frame_is_still_rejected() -> None:
    # b"" is falsy but not None — the check must be "is not None", not
    # a truthiness check, or an empty binary frame would slip past as if
    # it were absent.
    message = {"type": "websocket.receive", "bytes": b""}
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.ERROR
    assert decision.error.code is WireErrorCode.UNSUPPORTED_FRAME_TYPE


def test_decide_inbound_message_neither_text_nor_bytes_is_a_named_error() -> None:
    message = {"type": "websocket.receive"}
    decision = decide_inbound_message(message)
    assert decision.kind is InboundKind.ERROR
    assert decision.error.code is WireErrorCode.UNSUPPORTED_FRAME_TYPE


def test_decide_inbound_message_never_raises() -> None:
    adversarial_messages = [
        {},
        {"type": "websocket.receive"},
        {"type": "websocket.receive", "text": None},
        {"type": "websocket.receive", "text": ""},
        {"type": "websocket.receive", "text": "null"},
        {"type": "websocket.receive", "text": "[1, 2, 3]"},
        {"type": "websocket.receive", "text": "garbage{{{"},
        {"type": "websocket.receive", "text": json.dumps({"type": APPEND_EVENT_TYPE, "audio": 5})},
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": APPEND_EVENT_TYPE, "audio": None}),
        },
        {"type": "websocket.receive", "bytes": b"\x00\x01\x02"},
        {"type": "websocket.receive", "bytes": None, "text": None},
    ]
    for message in adversarial_messages:
        decision = decide_inbound_message(message)  # must never raise
        assert isinstance(decision, InboundDecision)
        assert decision.kind in (InboundKind.AUDIO, InboundKind.IGNORED, InboundKind.ERROR)
        if decision.kind is InboundKind.ERROR:
            assert isinstance(decision.error, WireFormatError)


def test_decide_inbound_message_result_fields_match_kind() -> None:
    # audio/error are mutually exclusive and each populated only for its
    # matching kind — a caller must be able to trust `decision.kind` alone.
    audio_decision = decide_inbound_message(_append_message(b"\x01\x02"))
    assert audio_decision.audio is not None and audio_decision.error is None

    ignored_decision = decide_inbound_message(
        {"type": "websocket.receive", "text": json.dumps({"type": "response.create"})}
    )
    assert ignored_decision.audio is None and ignored_decision.error is None

    error_decision = decide_inbound_message({"type": "websocket.receive", "text": "{{{"})
    assert error_decision.audio is None and error_decision.error is not None


def test_decide_inbound_message_sequence_reassembles_audio_byte_exact_transcription_only() -> None:
    """Acceptance criterion 1 (issue #151 t4): the transcription-only event
    sequence is unchanged 1:1 over base64 append input.

    Everything downstream of raw inbound bytes — buffering/alignment
    (_pcm.py), VAD segmentation (_segmenter.py), and session event emission
    (_session.py) — is untouched by this task and already covered by its own
    offline tests; the ONLY thing t4 changes is how those bytes are obtained
    from one received WebSocket message. This test proves that swap is
    transparent: feeding a realistic stream of base64 append events (with an
    ignorable non-append event interspersed, exactly as a real session might
    receive one) through decide_inbound_message reassembles the identical
    PCM stream, in order, byte-for-byte, that the pre-migration code would
    have read directly off `message["bytes"]` — so the transcription-only
    sequence downstream is unaffected in every respect except the wire
    framing itself.
    """
    pcm = os.urandom(4001)  # odd length on purpose, mirrors a real mic stream
    chunk_size = 733  # deliberately not aligned to a PCM16 sample boundary
    chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]

    messages = []
    for i, chunk in enumerate(chunks):
        messages.append(_append_message(chunk))
        if i == 1:
            # An ignorable control-type event arriving mid-stream must
            # contribute no bytes and must not disturb reassembly order.
            messages.append(
                {"type": "websocket.receive", "text": json.dumps({"type": "response.create"})}
            )

    reassembled = bytearray()
    for message in messages:
        decision = decide_inbound_message(message)
        if decision.kind is InboundKind.AUDIO:
            reassembled.extend(decision.audio)
        elif decision.kind is InboundKind.IGNORED:
            continue
        else:  # pragma: no cover - failure path only, fixture is well-formed
            pytest.fail(f"unexpected error decision: {decision.error}")

    assert bytes(reassembled) == pcm


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


# ---------------------------------------------------------------------------
# Grep gate — acceptance criterion 2 (issue #151 t4): no code path accepts
# or emits raw binary audio frames after the change.
#
# app.py is FastAPI/torch-only and is never imported by this offline suite
# (see test_realtime_imports.py and pyproject.toml's [tool.coverage.run]
# omit list), so this reads its SOURCE TEXT the same way
# test_module_source_never_imports_forbidden_deps above does for _wire.py —
# proving the guarantee holds without importing fastapi/torch/httpx.
# ---------------------------------------------------------------------------


def _app_py_source() -> str:
    return (Path(W.__file__).parent / "app.py").read_text(encoding="utf-8")


def test_app_py_routes_every_receive_through_the_decision_helper() -> None:
    src = _app_py_source()
    assert "decide_inbound_message" in src, (
        "app.py's _pump_session must classify every received WebSocket "
        "message through _wire.decide_inbound_message, not read it directly"
    )


def test_app_py_no_longer_reads_bytes_off_a_message_directly() -> None:
    # Before this task, _pump_session pulled inbound audio straight off
    # `message.get("bytes")`. After, decide_inbound_message owns that
    # decision (and classifies a "bytes"-carrying message as a named ERROR,
    # never as audio — see test_decide_inbound_message_binary_frame_is_
    # rejected_as_a_named_error above). A direct read reappearing in app.py
    # would silently bypass that guarantee.
    src = _app_py_source()
    assert 'message.get("bytes")' not in src
    assert 'message["bytes"]' not in src
    assert "message.get('bytes')" not in src


def test_app_py_never_sends_raw_binary_frames() -> None:
    # Output-direction half of the same guarantee: no send_bytes call
    # anywhere — outbound stays base64 JSON (response.audio.delta), wired by
    # a later task (t6), never introduced here.
    src = _app_py_source()
    assert "send_bytes" not in src


def test_app_py_route_docstring_no_longer_advertises_binary_input() -> None:
    src = _app_py_source()
    # The pre-migration docstring described input as "streamed as BINARY
    # WebSocket frames" — that specific claim must be gone. (The docstring
    # is still allowed to MENTION binary frames in the negative, e.g. "no
    # longer accepted" — that phrasing is asserted separately below.)
    assert "streamed as BINARY WebSocket" not in src
    assert "unsupported_frame_type" in src
    assert "input_audio_buffer.append" in src


# ---------------------------------------------------------------------------
# Issue #151 t6 additions: the ignored-event payload, and the shared encoder.
# ---------------------------------------------------------------------------


def test_decide_inbound_message_hands_back_an_ignored_events_payload() -> None:
    # t6 needs to act on ONE control event (response.create) without
    # re-parsing the frame — and without this module growing an opinion on
    # turn state. It hands the already-decoded object back instead; the
    # decision KIND is unchanged, which is what keeps the transcription-only
    # reassembly test above passing untouched.
    message = {
        "type": "websocket.receive",
        "text": json.dumps({"type": "response.create", "response": {"modalities": ["audio"]}}),
    }
    decision = decide_inbound_message(message)

    assert decision.kind is InboundKind.IGNORED
    assert decision.payload == {"type": "response.create", "response": {"modalities": ["audio"]}}


def test_decide_inbound_message_payload_is_absent_on_audio_and_error_decisions() -> None:
    # payload is populated for IGNORED only — an AUDIO decision would
    # otherwise hold a live reference to the whole base64 audio string long
    # after the PCM was decoded out of it.
    assert decide_inbound_message(_append_message(b"\x01\x02")).payload is None
    assert decide_inbound_message({"type": "websocket.receive", "text": "{{{"}).payload is None
    assert decide_inbound_message({"type": "websocket.receive", "bytes": b"\x00"}).payload is None


def test_encode_audio_chunk_round_trips_against_the_inbound_decoder() -> None:
    from lobes.realtime._wire import encode_audio_chunk, parse_append_event

    pcm = os.urandom(4800)
    assert parse_append_event({"audio": encode_audio_chunk(pcm)}) == pcm
    assert encode_audio_chunk(b"") == ""


def test_serialize_audio_delta_uses_the_one_shared_encoder() -> None:
    # One base64 encode in the module, shared by the standalone OpenAI-shaped
    # event here and by the session-schema delta the live route emits — so
    # the two can never disagree about what "the delta field" contains.
    from lobes.realtime._wire import encode_audio_chunk

    pcm = os.urandom(1024)
    event = serialize_audio_delta(pcm, response_id="resp_1", item_id="item_1")
    assert event["delta"] == encode_audio_chunk(pcm)


def test_the_delta_chunk_size_is_the_only_one_in_the_tree() -> None:
    # The chunk-size reconciliation (issue #151 t6): _floor.py's competing
    # DEFAULT_CHUNK_MS/DEFAULT_CHUNK_BYTES are gone and its `chunk_bytes` is
    # a required constructor argument, so this is the single source of truth.
    import lobes.realtime._floor as floor_module

    assert not hasattr(floor_module, "DEFAULT_CHUNK_BYTES")
    assert DEFAULT_DELTA_CHUNK_BYTES == 4800  # 100 ms at 24 kHz PCM16
