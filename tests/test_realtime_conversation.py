"""Offline tests for the conversation bridge — issue #151 t6.

t6 is the convergence task: five independently-built modules (``_wire``,
``_floor``, ``_session``, ``_turn``, ``tts_client``) meet and become one
working turn. Everything that could be *decided* wrongly in that meeting
lives in :mod:`lobes.realtime._conversation`, a stdlib-only module, so this
file can drive the entire voice-to-voice flow — commit, generate, synthesize,
deliver, interrupt, expire — with no fastapi, no httpx, no torch, and no
socket.

``app.py`` itself is fastapi/torch-only and is never imported by this suite
(see ``test_realtime_imports.py`` and pyproject's coverage omit list), so the
route's own two runtime obligations — PUMP delivery with an await between
chunks, and drive ``tick()`` from a watchdog — are asserted against its
SOURCE with :mod:`ast`, structurally, not by grepping for a phrase. Get
either wrong and every interruption/deadline guarantee below is inert live,
while still passing as a unit test; that is precisely the failure this file
exists to make impossible.
"""

from __future__ import annotations

import ast
import base64
import importlib
import json
from pathlib import Path

import pytest

import lobes.realtime._conversation as C
import lobes.realtime._floor as F
import lobes.realtime._session as S
import lobes.realtime._wire as W
from lobes.realtime.protocol import BYTES_PER_SAMPLE, CLIENT_SAMPLE_RATE, TTS_SAMPLE_RATE

# ---------------------------------------------------------------------------
# import isolation — the convergence must stay offline-testable
# ---------------------------------------------------------------------------


def test_module_imports_without_the_realtime_extra() -> None:
    importlib.import_module("lobes.realtime._conversation")


def test_module_source_never_imports_forbidden_deps() -> None:
    src = Path(C.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    forbidden = {"fastapi", "httpx", "numpy", "scipy", "torch", "uvicorn", "anyio", "silero_vad"}
    assert not (imported & forbidden), f"_conversation.py must stay stdlib-only: {imported}"


def test_module_never_imports_settings() -> None:
    # The route resolves env-derived values and passes them in — same rule
    # _segmenter.py and _floor.py follow, so this module has no env-derived
    # state at import time.
    src = Path(C.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "_settings", "_conversation must not import _settings"


# ---------------------------------------------------------------------------
# fakes / helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """A monotonic-ms clock that only moves when a test moves it."""

    def __init__(self, start_ms: int = 1_000) -> None:
        self.now_ms = start_ms

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, ms: int) -> int:
        self.now_ms += ms
        return self.now_ms


class Cancels:
    """Counts the floor's two abandonment hooks."""

    def __init__(self) -> None:
        self.generate = 0
        self.tts = 0

    def on_generate(self) -> None:
        self.generate += 1

    def on_tts(self) -> None:
        self.tts += 1


# 10 ms of 24 kHz PCM16 = 480 bytes — round numbers in every ms assertion.
CHUNK = 480


def pcm(n_bytes: int) -> bytes:
    """Deterministic, position-tagged PCM so a test can prove WHICH bytes went out."""
    return bytes((i * 7) % 251 for i in range(n_bytes))


def chat_body(text: str | None) -> bytes:
    return json.dumps({"choices": [{"message": {"content": text}}]}).encode()


def make_bridge(config_payload: dict | None = None, **overrides):
    """A bridge over a real Session, with a fake clock and counted cancels.

    ``base_url``/``api_key``/``model`` may still be passed individually; they
    are folded into the :class:`~lobes.realtime._conversation.GenerateConfig`
    the bridge now takes, so a test that only cares about the model does not
    have to restate the whole generate config.
    """
    config = S.parse_session_config(config_payload or {})
    session, _created = S.Session.create(config)
    cancels = Cancels()
    clock = FakeClock()
    generate = overrides.pop(
        "generate",
        C.GenerateConfig(
            base_url=overrides.pop("base_url", "http://gateway:8000"),
            api_key=overrides.pop("api_key", "test-key"),
            model=overrides.pop("model", "multimodal"),
        ),
    )
    kwargs: dict = {
        "cancel_generate": cancels.on_generate,
        "cancel_tts": cancels.on_tts,
        "generate": generate,
        "chunk_bytes": CHUNK,
        "clock": clock,
    }
    kwargs.update(overrides)
    return C.ConversationBridge(session, **kwargs), cancels, clock


def types_of(payloads: list[dict]) -> list:
    """The ``type`` of each payload, as the schema's own EventType member."""
    return [payload["type"] for payload in payloads]


def commit_turn(bridge, *, at_start: int = 128, at_stop: int = 2048, reason: str = "silence"):
    bridge.on_speech_started(at_ms=at_start)
    bridge.on_speech_stopped(at_ms=at_stop, reason=reason)


def run_to_speaking(bridge, clock, *, text="what time is it", reply="it is half past four"):
    """Commit a turn and walk it to SPEAKING; returns the turn id."""
    commit_turn(bridge)
    bridge.on_transcript(text)
    turn_id = bridge.take_pending_response()
    assert turn_id is not None
    # Leave the barge-in guard window so a later onset is honoured.
    clock.advance(F.DEFAULT_BARGE_IN_WINDOW_MS)
    assert bridge.build_generate_request(turn_id) is not None
    bridge.on_generate_response(200, chat_body(reply), turn_id=turn_id)
    assert bridge.take_pending_synthesis() == (turn_id, reply)
    return turn_id


def pump(bridge, turn_id, *, limit: int | None = None) -> int:
    """Deliver chunks the way the route does — one call per iteration."""
    sent = 0
    while bridge.deliver_next(turn_id=turn_id):
        sent += 1
        if limit is not None and sent >= limit:
            break
    return sent


def delivered_audio(payloads: list[dict]) -> bytes:
    return b"".join(
        base64.b64decode(str(payload["delta"]))
        for payload in payloads
        if payload["type"] == S.EventType.RESPONSE_AUDIO_DELTA
    )


# ---------------------------------------------------------------------------
# RECONCILIATION 1 — one chunk size, owned by the wire codec.
# ---------------------------------------------------------------------------


def test_floor_no_longer_ships_its_own_chunk_size_default() -> None:
    # t2's _floor.DEFAULT_CHUNK_MS/DEFAULT_CHUNK_BYTES (40ms/1920B) disagreed
    # with t1's _wire.DEFAULT_DELTA_CHUNK_BYTES (100ms/4800B). Exactly one
    # survives, and it is the wire codec's — chunk size is wire framing.
    assert not hasattr(F, "DEFAULT_CHUNK_BYTES")
    assert not hasattr(F, "DEFAULT_CHUNK_MS")
    assert "DEFAULT_CHUNK_BYTES" not in F.__all__
    assert "DEFAULT_CHUNK_MS" not in F.__all__


def test_floor_requires_an_explicit_chunk_size() -> None:
    # Required, not defaulted: a second default is a value free to drift.
    with pytest.raises(TypeError):
        F.Floor(
            emit_event=lambda event: None,
            send_audio_chunk=lambda chunk: None,
            cancel_generate=lambda: None,
            cancel_tts=lambda: None,
        )


def test_the_bridge_passes_the_wire_codecs_chunk_size_to_the_floor() -> None:
    bridge, _, _ = make_bridge(chunk_bytes=W.DEFAULT_DELTA_CHUNK_BYTES)
    assert bridge.floor.chunk_bytes == W.DEFAULT_DELTA_CHUNK_BYTES


def test_the_bridge_default_chunk_size_is_the_wire_codecs() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    bridge = C.ConversationBridge(
        session,
        cancel_generate=lambda: None,
        cancel_tts=lambda: None,
        generate=C.GenerateConfig(base_url="http://gateway:8000"),
    )
    assert bridge.floor.chunk_bytes == W.DEFAULT_DELTA_CHUNK_BYTES
    assert W.DEFAULT_DELTA_CHUNK_BYTES % BYTES_PER_SAMPLE == 0


# ---------------------------------------------------------------------------
# RECONCILIATION 2 — one error vocabulary for the floor's six reasons.
# ---------------------------------------------------------------------------


def test_every_failure_reason_maps_to_a_named_session_error_code() -> None:
    assert set(C.FAILURE_ERROR_CODES) == set(F.FailureReason)
    assert all(isinstance(code, S.ErrorCode) for code in C.FAILURE_ERROR_CODES.values())


def test_transcribe_failed_reuses_the_existing_stt_forward_failed_code() -> None:
    # NOT a new code: a committed turn's Parakeet forward failing is the same
    # event whether or not the session went on to answer the turn.
    assert (
        C.FAILURE_ERROR_CODES[F.FailureReason.TRANSCRIBE_FAILED] is S.ErrorCode.STT_FORWARD_FAILED
    )


@pytest.mark.parametrize(
    "reason,code",
    [
        (F.FailureReason.GENERATE_FAILED, S.ErrorCode.GENERATE_FAILED),
        (F.FailureReason.TTS_FAILED, S.ErrorCode.TTS_FAILED),
        (F.FailureReason.TRANSCRIBE_TIMEOUT, S.ErrorCode.RESPONSE_TIMEOUT),
        (F.FailureReason.GENERATE_TIMEOUT, S.ErrorCode.RESPONSE_TIMEOUT),
        (F.FailureReason.TTS_TIMEOUT, S.ErrorCode.RESPONSE_TIMEOUT),
    ],
)
def test_the_failure_mapping_table_is_the_documented_one(reason, code) -> None:
    assert C.FAILURE_ERROR_CODES[reason] is code


@pytest.mark.parametrize(
    "reason,stage",
    [
        (F.FailureReason.TRANSCRIBE_TIMEOUT, F.Stage.TRANSCRIBE),
        (F.FailureReason.GENERATE_TIMEOUT, F.Stage.GENERATE),
        (F.FailureReason.TTS_TIMEOUT, F.Stage.TTS),
    ],
)
def test_every_timeout_names_its_stage_in_the_message(reason, stage) -> None:
    # Three timeout reasons share ONE code, so without the stage in the text
    # an operator cannot tell which stage wedged.
    event = F.ResponseFailed(
        at_ms=1, turn_id=1, stage=stage, reason=reason, message="ReadTimeout: nothing came back"
    )
    message = C.describe_failure(event)
    assert stage.value in message
    assert "ReadTimeout" in message


def test_the_floors_own_timeout_message_is_not_double_prefixed() -> None:
    event = F.ResponseFailed(
        at_ms=1,
        turn_id=1,
        stage=F.Stage.GENERATE,
        reason=F.FailureReason.GENERATE_TIMEOUT,
        message="generate stage exceeded 60000ms",
    )
    assert C.describe_failure(event) == "generate stage exceeded 60000ms"


def test_describe_failure_is_idempotent_when_its_own_output_is_fed_back() -> None:
    """Idempotence means the OUTPUT survives a second pass unchanged.

    Calling the same pure function twice on the same input proves nothing, so
    this prefixes a message the route phrased itself (a backend read timeout,
    which does not open with "<stage> stage"), then describes THAT result: the
    stage must not be prefixed a second time.
    """
    raw = F.ResponseFailed(
        at_ms=1_000,
        turn_id=1,
        stage=F.Stage.GENERATE,
        reason=F.FailureReason.GENERATE_TIMEOUT,
        message="ReadTimeout: backend took too long",
    )
    once = C.describe_failure(raw)
    assert once == "generate stage: ReadTimeout: backend took too long"

    already_described = F.ResponseFailed(
        at_ms=1_000,
        turn_id=1,
        stage=F.Stage.GENERATE,
        reason=F.FailureReason.GENERATE_TIMEOUT,
        message=once,
    )
    assert C.describe_failure(already_described) == once


def test_a_named_failure_message_is_passed_through_unchanged() -> None:
    # generate_failed/tts_failed/stt_forward_failed each identify their own
    # stage via the code, so nothing is prepended.
    event = F.ResponseFailed(
        at_ms=1,
        turn_id=1,
        stage=F.Stage.GENERATE,
        reason=F.FailureReason.GENERATE_FAILED,
        message="the generate lane returned an empty reply",
    )
    assert C.describe_failure(event) == "the generate lane returned an empty reply"


# ---------------------------------------------------------------------------
# RECONCILIATION 3 — the split wire/session error vocabulary, collapsed.
# ---------------------------------------------------------------------------


def test_error_codes_are_one_enumerable_list() -> None:
    # The site renders each code distinctly and the docs task documents them,
    # so there must be exactly ONE enum to enumerate.
    assert set(C.WIRE_ERROR_CODES) == set(W.WireErrorCode)
    assert set(C.WIRE_ERROR_CODES.values()) == {S.ErrorCode.INVALID_WIRE_EVENT}
    assert S.ErrorCode.INVALID_WIRE_EVENT.value == "invalid_wire_event"


@pytest.mark.parametrize("wire_code", list(W.WireErrorCode))
def test_a_malformed_frame_emits_one_session_error_code_naming_the_wire_reason(wire_code) -> None:
    bridge, _, _ = make_bridge()
    bridge.on_wire_error(W.WireFormatError(wire_code, "something was wrong with the frame"))
    (payload,) = bridge.drain()

    assert payload["type"] == S.EventType.ERROR
    assert payload["code"] is S.ErrorCode.INVALID_WIRE_EVENT
    # One code for all three, so the specific reason lives in the text.
    assert wire_code.value in str(payload["message"])
    assert "something was wrong with the frame" in str(payload["message"])


def test_a_malformed_frame_never_moves_the_session_state() -> None:
    bridge, _, _ = make_bridge()
    bridge.on_speech_started(at_ms=64)
    bridge.drain()

    bridge.on_wire_error(W.WireFormatError(W.WireErrorCode.INVALID_JSON, "bad"))

    assert bridge.session.state is S.SessionState.SPEECH  # still mid-speech
    assert bridge.session.has_open_item is True


def test_the_error_code_the_json_carries_is_a_session_error_code_string() -> None:
    # The pre-t6 leak: app.py put a WireErrorCode value into ErrorEvent.code
    # verbatim. json.dumps still worked (both are str-Enums), which is exactly
    # why it went unnoticed — assert on the VALUE, not just serializability.
    bridge, _, _ = make_bridge()
    bridge.on_wire_error(
        W.WireFormatError(W.WireErrorCode.UNSUPPORTED_FRAME_TYPE, "binary frames are gone")
    )
    payload = json.loads(json.dumps(bridge.drain()[0]))
    assert payload["code"] in {code.value for code in S.ErrorCode}
    assert payload["code"] not in {
        code.value for code in W.WireErrorCode if code.value != payload["code"]
    }


# ---------------------------------------------------------------------------
# RECONCILIATION 4 — at_ms and reason reach the wire (honesty h19).
# ---------------------------------------------------------------------------


def test_boundary_events_carry_the_segmenters_audio_stream_timing() -> None:
    bridge, _, _ = make_bridge()
    bridge.on_speech_started(at_ms=128)
    bridge.on_speech_stopped(at_ms=2048, reason="silence")
    started, stopped = bridge.drain()

    assert started["at_ms"] == 128
    assert stopped["at_ms"] == 2048
    assert stopped["reason"] == "silence"


def test_a_max_turn_force_commit_is_distinguishable_from_a_silence_stop() -> None:
    # CLAUDE.md calls the max_turn force-commit "a normal boundary event,
    # never an error" — before t6 no client could tell the two apart at all.
    bridge, _, _ = make_bridge()
    bridge.on_speech_started(at_ms=0)
    bridge.on_speech_stopped(at_ms=30_000, reason="max_turn")
    _started, stopped = bridge.drain()

    assert stopped["reason"] == "max_turn"
    assert stopped["type"] == S.EventType.SPEECH_STOPPED  # a boundary, not an error


def test_boundary_timing_survives_json_serialization() -> None:
    bridge, _, _ = make_bridge()
    bridge.on_speech_started(at_ms=288)
    payload = json.loads(json.dumps(bridge.drain()[0]))
    assert payload["at_ms"] == 288


def test_at_ms_is_absent_rather_than_wrong_when_the_caller_has_none() -> None:
    bridge, _, _ = make_bridge()
    bridge.on_speech_started()
    bridge.on_speech_stopped()
    started, stopped = bridge.drain()
    assert started["at_ms"] is None
    assert stopped["at_ms"] is None and stopped["reason"] is None


def test_the_segmenters_at_ms_is_never_fed_into_the_floors_clock() -> None:
    # Two clock domains: at_ms is 32ms-quantised audio-stream time, the
    # floor's clock is monotonic wall-clock. Mixing them would silently skew
    # the barge-in guard window. Asserted structurally against the source.
    src = Path(C.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "on_speech_started"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "floor"
    ]
    assert calls, "the bridge must drive floor.on_speech_started"
    for call in calls:
        assert not call.args and not call.keywords, "floor.on_speech_started takes no timestamp"


# ---------------------------------------------------------------------------
# The opt-in boundary — a session that never triggers stays ears-only.
# ---------------------------------------------------------------------------

EARS_ONLY_SEQUENCE = [
    S.EventType.SPEECH_STARTED,
    S.EventType.SPEECH_STOPPED,
    S.EventType.TRANSCRIPTION_COMPLETED,
]


def test_a_session_that_never_opts_in_emits_the_transcription_only_sequence() -> None:
    bridge, cancels, _ = make_bridge()
    commit_turn(bridge)
    bridge.on_transcript("hello there")

    assert types_of(bridge.drain()) == EARS_ONLY_SEQUENCE
    assert bridge.take_pending_response() is None
    assert bridge.take_pending_synthesis() is None
    assert (cancels.generate, cancels.tts) == (0, 0)


def test_an_unarmed_session_never_touches_the_floor_at_all() -> None:
    bridge, _, _ = make_bridge()
    commit_turn(bridge)
    bridge.on_transcript("hello there")
    assert bridge.floor.state is F.FloorState.LISTENING
    assert bridge.floor.turn_id == 0  # no turn was ever opened


def test_an_unarmed_session_accumulates_no_history() -> None:
    bridge, _, _ = make_bridge()
    commit_turn(bridge)
    bridge.on_transcript("hello there")
    assert bridge.session.get_history() == []


def test_an_unarmed_stt_failure_is_the_same_named_error_as_ever() -> None:
    bridge, _, _ = make_bridge()
    commit_turn(bridge)
    bridge.on_transcription_failed("STT backend unreachable: connect error")

    payloads = bridge.drain()
    assert types_of(payloads) == [
        S.EventType.SPEECH_STARTED,
        S.EventType.SPEECH_STOPPED,
        S.EventType.ERROR,
    ]
    assert payloads[-1]["code"] is S.ErrorCode.STT_FORWARD_FAILED


def test_an_unrelated_control_event_does_not_arm_the_session() -> None:
    bridge, _, _ = make_bridge()
    assert bridge.on_control_event({"type": "session.update", "session": {}}) is False
    assert bridge.on_control_event({"type": "input_audio_buffer.commit"}) is False
    assert bridge.on_control_event(None) is False
    assert bridge.on_control_event({}) is False
    assert bridge.armed is False


def test_response_create_is_what_arms_the_session() -> None:
    bridge, _, _ = make_bridge()
    assert bridge.on_control_event({"type": C.RESPONSE_CREATE_EVENT_TYPE}) is True
    assert bridge.armed is True


def test_is_response_create_recognizes_only_the_trigger() -> None:
    assert C.is_response_create({"type": "response.create"}) is True
    assert C.is_response_create({"type": "response.created"}) is False
    assert C.is_response_create({"type": "input_audio_buffer.append"}) is False
    assert C.is_response_create({}) is False
    assert C.is_response_create(None) is False


# ---------------------------------------------------------------------------
# The opt-in turn: commit -> generate -> TTS -> audio out.
# ---------------------------------------------------------------------------


def test_the_full_turn_emits_the_response_lifecycle_in_order() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 3), turn_id=turn_id)
    assert pump(bridge, turn_id) == 3

    assert types_of(bridge.drain()) == [
        S.EventType.SPEECH_STARTED,
        S.EventType.SPEECH_STOPPED,
        S.EventType.TRANSCRIPTION_COMPLETED,
        S.EventType.RESPONSE_CREATED,
        S.EventType.RESPONSE_TEXT_DONE,
        S.EventType.RESPONSE_AUDIO_DELTA,
        S.EventType.RESPONSE_AUDIO_DELTA,
        S.EventType.RESPONSE_AUDIO_DELTA,
        S.EventType.RESPONSE_DONE,
    ]
    assert bridge.floor.state is F.FloorState.LISTENING
    assert bridge.session.state is S.SessionState.IDLE


def test_audio_out_is_byte_exact_with_no_resample_anywhere_in_the_path() -> None:
    # protocol.py pins TTS_SAMPLE_RATE == CLIENT_SAMPLE_RATE, so the ONLY
    # transform between synthesize()'s PCM and the wire is a base64 encode.
    assert TTS_SAMPLE_RATE == CLIENT_SAMPLE_RATE
    audio = pcm(CHUNK * 4 + 37 * BYTES_PER_SAMPLE)
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(audio, turn_id=turn_id)
    pump(bridge, turn_id)

    assert delivered_audio(bridge.drain()) == audio


def test_no_resample_call_exists_in_the_tts_out_path() -> None:
    # Asserted on the emitted CODE, with the docstring stripped — the
    # docstring is allowed to say "no resample", the code is not allowed to
    # do one.
    src = Path(C.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_on_audio_chunk"
    )
    body = [stmt for stmt in node.body if not isinstance(stmt, ast.Expr)] or node.body[-1:]
    code = "\n".join(ast.unparse(stmt) for stmt in body)
    assert "resample" not in code.lower()
    assert "encode_audio_chunk" in code


def test_every_delta_shares_the_response_id_of_its_response() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 3), turn_id=turn_id)
    pump(bridge, turn_id)
    payloads = bridge.drain()

    created = next(p for p in payloads if p["type"] == S.EventType.RESPONSE_CREATED)
    deltas = [p for p in payloads if p["type"] == S.EventType.RESPONSE_AUDIO_DELTA]
    done = next(p for p in payloads if p["type"] == S.EventType.RESPONSE_DONE)
    assert {p["response_id"] for p in deltas} == {created["response_id"]}
    assert done["response_id"] == created["response_id"]


def test_every_outbound_event_carries_the_session_id() -> None:
    # Including the audio deltas — the session schema stamps it onto every
    # event, which is why the route emits Session.emit_audio_delta rather
    # than _wire.serialize_audio_delta's bare, session-free dict.
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 2), turn_id=turn_id)
    pump(bridge, turn_id)

    assert {p["session_id"] for p in bridge.drain()} == {bridge.session.session_id}


def test_the_response_answers_the_transcribed_item() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("what time is it")
    payloads = bridge.drain()

    stopped = next(p for p in payloads if p["type"] == S.EventType.SPEECH_STOPPED)
    created = next(p for p in payloads if p["type"] == S.EventType.RESPONSE_CREATED)
    assert created["item_id"] == stopped["item_id"]


def test_a_blank_transcript_releases_the_floor_without_a_response() -> None:
    bridge, _, _ = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("   ")

    assert types_of(bridge.drain()) == EARS_ONLY_SEQUENCE
    assert bridge.floor.state is F.FloorState.LISTENING
    assert bridge.take_pending_response() is None
    assert bridge.session.get_history() == []


def test_an_empty_reply_is_a_named_failure_not_unexplained_silence() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    turn_id = bridge.take_pending_response()
    bridge.on_generate_response(200, chat_body(""), turn_id=turn_id)

    payloads = bridge.drain()
    assert payloads[-1]["type"] == S.EventType.ERROR
    assert payloads[-1]["code"] is S.ErrorCode.GENERATE_FAILED
    assert bridge.floor.state is F.FloorState.LISTENING


def test_empty_tts_audio_is_a_named_failure_not_a_silent_completion() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(b"", turn_id=turn_id)

    payloads = bridge.drain()
    assert payloads[-1]["code"] is S.ErrorCode.TTS_FAILED
    assert bridge.floor.state is F.FloorState.LISTENING


# ---------------------------------------------------------------------------
# The generate call: model policy, system prompt, named failures.
# ---------------------------------------------------------------------------


def test_the_voice_lane_defaults_to_multimodal_when_openai_model_is_unset() -> None:
    # Honesty h4: _turn.py is deliberately policy-free (a falsy model omits
    # the key entirely) and t8 deliberately did not pre-apply this, so the
    # wiring layer is the one place it can happen.
    assert C.resolve_voice_model("") == "multimodal"
    assert C.resolve_voice_model(None) == "multimodal"
    assert C.DEFAULT_VOICE_MODEL == "multimodal"


def test_an_operator_set_openai_model_wins_outright() -> None:
    assert C.resolve_voice_model("cortex") == "cortex"
    assert C.resolve_voice_model("muse") == "muse"


def test_the_generate_request_carries_the_resolved_model_and_no_thinking() -> None:
    bridge, _, clock = make_bridge(model=C.resolve_voice_model(""))
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("what time is it")
    turn_id = bridge.take_pending_response()

    request = bridge.build_generate_request(turn_id)
    assert request.body["model"] == "multimodal"
    assert request.body["chat_template_kwargs"] == {"enable_thinking": False}
    assert request.url == "http://gateway:8000/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"


def test_the_generate_request_carries_the_sessions_system_prompt() -> None:
    bridge, _, _ = make_bridge({"system_prompt": "be terse"})
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    request = bridge.build_generate_request(bridge.take_pending_response())

    assert request.body["messages"][0] == {"role": "system", "content": "be terse"}


def test_the_system_prompt_default_reaches_the_generate_call() -> None:
    # The operator half of spec c34: parse_session_config resolves the env
    # default, Session holds it, and the turn builder sends it.
    config = S.parse_session_config({}, default_system_prompt="operator prompt")
    session, _ = S.Session.create(config)
    bridge = C.ConversationBridge(
        session,
        cancel_generate=lambda: None,
        cancel_tts=lambda: None,
        generate=C.GenerateConfig(base_url="http://gateway:8000"),
    )
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    request = bridge.build_generate_request(bridge.take_pending_response())

    assert request.body["messages"][0]["content"] == "operator prompt"


def test_a_stale_turn_builds_no_generate_request_at_all() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    turn_id = bridge.take_pending_response()
    clock.advance(F.DEFAULT_BARGE_IN_WINDOW_MS)
    bridge.on_speech_started(at_ms=5000)  # barge-in before the call went out

    assert bridge.build_generate_request(turn_id) is None


def test_role_infeasible_is_a_named_error_carrying_the_hosted_by_hint() -> None:
    # Spec c4/h4: never a silent fallback to another lane.
    body = json.dumps(
        {
            "error": {
                "code": "role_infeasible",
                "message": "role 'senses' is not hosted on this box",
                "hosted_by": "http://thor:8000",
            }
        }
    ).encode()
    bridge, _, clock = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    turn_id = bridge.take_pending_response()
    bridge.on_generate_response(404, body, turn_id=turn_id)

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.GENERATE_FAILED
    assert "role_infeasible" in str(error["message"]) or "senses" in str(error["message"])
    assert "http://thor:8000" in str(error["message"])
    assert bridge.floor.state is F.FloorState.LISTENING


def test_role_infeasible_without_a_peer_hint_still_names_the_failure() -> None:
    exc = C.RoleInfeasibleError("role 'senses' is not hosted on this box")
    assert C.describe_role_infeasible(exc) == "role 'senses' is not hosted on this box"
    assert "hosted_by" not in C.describe_role_infeasible(exc)


def test_a_plain_backend_failure_is_a_named_generate_error() -> None:
    bridge, _, _ = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    turn_id = bridge.take_pending_response()
    bridge.on_generate_response(503, b"upstream is down", turn_id=turn_id)

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.GENERATE_FAILED


def test_a_generate_read_timeout_is_a_response_timeout_naming_its_stage() -> None:
    bridge, _, _ = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    turn_id = bridge.take_pending_response()
    bridge.fail_generate("ReadTimeout: no answer", turn_id=turn_id, timed_out=True)

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.RESPONSE_TIMEOUT
    assert "generate" in str(error["message"])


def test_a_tts_timeout_is_a_response_timeout_naming_its_stage() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.fail_tts("ReadTimeout: chatterbox never answered", turn_id=turn_id, timed_out=True)

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.RESPONSE_TIMEOUT
    assert "tts" in str(error["message"])


# ---------------------------------------------------------------------------
# Barge-in — the whole point of pumped delivery.
# ---------------------------------------------------------------------------


def test_a_barge_in_mid_delivery_never_sends_the_undelivered_remainder() -> None:
    audio = pcm(CHUNK * 6)
    bridge, cancels, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(audio, turn_id=turn_id)
    assert pump(bridge, turn_id, limit=2) == 2

    bridge.on_speech_started(at_ms=9000)  # the user speaks over the reply

    payloads = bridge.drain()
    assert delivered_audio(payloads) == audio[: CHUNK * 2]
    assert pump(bridge, turn_id) == 0  # the remainder is gone, not queued
    assert delivered_audio(bridge.drain()) == b""
    assert (cancels.generate, cancels.tts) == (1, 1)  # cancel BOTH, always


def test_the_interruption_event_is_ordered_after_the_last_delivered_chunk() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 6), turn_id=turn_id)
    pump(bridge, turn_id, limit=2)
    bridge.on_speech_started(at_ms=9000)

    types = types_of(bridge.drain())
    interrupted = types.index(S.EventType.RESPONSE_INTERRUPTED)
    last_delta = len(types) - 1 - types[::-1].index(S.EventType.RESPONSE_AUDIO_DELTA)
    assert last_delta < interrupted
    # The onset's own boundary event follows the interruption, so the
    # session's floor-holder state ends up SPEECH and not IDLE.
    assert types[interrupted + 1] == S.EventType.SPEECH_STARTED


def test_the_interruption_returns_the_floor_and_the_session_to_the_caller() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 4), turn_id=turn_id)
    pump(bridge, turn_id, limit=1)
    bridge.on_speech_started(at_ms=9000)

    assert bridge.floor.state is F.FloorState.LISTENING
    assert bridge.session.current_response_id is None


@pytest.mark.parametrize("stage", ["responding", "synthesizing", "delivering"])
def test_a_barge_in_lands_from_every_machine_held_response_stage(stage) -> None:
    bridge, cancels, clock = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    turn_id = bridge.take_pending_response()
    clock.advance(F.DEFAULT_BARGE_IN_WINDOW_MS)
    if stage != "responding":
        bridge.on_generate_response(200, chat_body("a reply"), turn_id=turn_id)
        bridge.take_pending_synthesis()
    if stage == "delivering":
        bridge.on_tts_audio(pcm(CHUNK * 4), turn_id=turn_id)
        pump(bridge, turn_id, limit=1)
    bridge.drain()

    bridge.on_speech_started(at_ms=9000)

    types = types_of(bridge.drain())
    assert S.EventType.RESPONSE_INTERRUPTED in types
    assert (cancels.generate, cancels.tts) == (1, 1)


def test_a_committed_turn_also_interrupts_and_immediately_takes_the_floor() -> None:
    # An onset swallowed by the guard window still leaves the turn itself as
    # evidence — a committed turn is never discarded on the floor.
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 4), turn_id=turn_id)
    pump(bridge, turn_id, limit=1)
    bridge.drain()

    bridge.on_speech_stopped(at_ms=9000, reason="silence")

    assert S.EventType.RESPONSE_INTERRUPTED in types_of(bridge.drain())
    assert bridge.floor.state is F.FloorState.TRANSCRIBING
    assert bridge.floor.turn_id == turn_id + 1


def test_an_onset_inside_the_barge_in_guard_window_is_not_an_interruption() -> None:
    bridge, cancels, _clock = make_bridge()
    bridge.arm()
    commit_turn(bridge)  # the floor is taken; the guard window starts now
    bridge.on_transcript("hello")
    bridge.drain()

    bridge.on_speech_started(at_ms=200)  # the clock has not moved

    assert types_of(bridge.drain()) == [S.EventType.SPEECH_STARTED]
    assert (cancels.generate, cancels.tts) == (0, 0)


def test_a_stale_response_task_cannot_deliver_into_the_next_turn() -> None:
    # The pump's turn_id guard: a response task still unwinding from an
    # interruption must not start pushing the NEXT turn's audio.
    bridge, _, clock = make_bridge()
    bridge.arm()
    first = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 6), turn_id=first)
    pump(bridge, first, limit=1)
    bridge.on_speech_stopped(at_ms=9000, reason="silence")  # interrupt + new turn
    bridge.on_transcript("and now something else")
    second = bridge.take_pending_response()
    clock.advance(F.DEFAULT_BARGE_IN_WINDOW_MS)
    bridge.on_generate_response(200, chat_body("the second reply"), turn_id=second)
    bridge.take_pending_synthesis()
    bridge.on_tts_audio(pcm(CHUNK * 3), turn_id=second)
    bridge.drain()

    assert bridge.deliver_next(turn_id=first) is False  # the stale pump
    assert bridge.drain() == []
    assert bridge.deliver_next(turn_id=second) is True


def test_a_stale_completion_never_lands_on_a_later_turn() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    first = run_to_speaking(bridge, clock)
    bridge.on_speech_stopped(at_ms=9000, reason="silence")
    bridge.on_transcript("second question")
    bridge.drain()

    # The abandoned first turn's TTS finally returns.
    assert bridge.on_tts_audio(pcm(CHUNK * 3), turn_id=first) is False
    assert bridge.drain() == []


# ---------------------------------------------------------------------------
# Per-stage deadlines — they expire ONLY inside tick().
# ---------------------------------------------------------------------------


def test_a_generate_deadline_expires_in_tick_and_returns_the_floor() -> None:
    bridge, cancels, clock = make_bridge(generate_timeout_ms=5_000)
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    bridge.take_pending_response()
    bridge.drain()

    assert bridge.tick() is False  # not due yet
    clock.advance(5_000)
    assert bridge.tick() is True

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.RESPONSE_TIMEOUT
    assert "generate" in str(error["message"])
    assert bridge.floor.state is F.FloorState.LISTENING
    assert (cancels.generate, cancels.tts) == (1, 1)


def test_a_tts_deadline_expires_in_tick_and_returns_the_floor() -> None:
    bridge, _, clock = make_bridge(tts_timeout_ms=5_000)
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.drain()

    clock.advance(5_000)
    assert bridge.tick() is True

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.RESPONSE_TIMEOUT
    assert "tts" in str(error["message"])
    assert bridge.floor.state is F.FloorState.LISTENING
    del turn_id


def test_a_transcribe_deadline_expires_in_tick() -> None:
    bridge, _, clock = make_bridge(transcribe_timeout_ms=5_000)
    bridge.arm()
    commit_turn(bridge)
    bridge.drain()

    clock.advance(5_000)
    assert bridge.tick() is True

    error = bridge.drain()[-1]
    assert error["code"] is S.ErrorCode.RESPONSE_TIMEOUT
    assert "transcribe" in str(error["message"])


def test_without_a_tick_no_deadline_ever_expires() -> None:
    # The whole reason app.py must run a watchdog: a wedged backend is by
    # definition not calling anything else.
    bridge, _, clock = make_bridge(generate_timeout_ms=5_000)
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    bridge.drain()

    clock.advance(60_000)
    assert bridge.floor.state is F.FloorState.RESPONDING  # still wedged, silently
    assert bridge.tick() is True  # only tick() can free it
    assert bridge.floor.state is F.FloorState.LISTENING


def test_tick_on_an_unarmed_session_is_inert() -> None:
    bridge, _, clock = make_bridge()
    commit_turn(bridge)
    bridge.on_transcript("hello")
    bridge.drain()
    clock.advance(600_000)
    assert bridge.tick() is False
    assert bridge.drain() == []


# ---------------------------------------------------------------------------
# STT failure while the floor holds the turn — emitted exactly once.
# ---------------------------------------------------------------------------


def test_an_armed_stt_failure_emits_exactly_one_error_identical_to_ears_only() -> None:
    bridge, _, _ = make_bridge()
    bridge.arm()
    commit_turn(bridge)
    bridge.on_transcription_failed("STT backend returned HTTP 502")

    payloads = bridge.drain()
    errors = [p for p in payloads if p["type"] == S.EventType.ERROR]
    assert len(errors) == 1
    assert errors[0]["code"] is S.ErrorCode.STT_FORWARD_FAILED
    assert errors[0]["message"] == "STT backend returned HTTP 502"
    assert errors[0]["item_id"] is not None  # the same shape the ears-only path emits
    assert bridge.floor.state is F.FloorState.LISTENING


# ---------------------------------------------------------------------------
# History — server-side, ephemeral, and honest about interruptions.
# ---------------------------------------------------------------------------


def test_a_two_turn_conversation_carries_the_first_exchange_forward() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    first = run_to_speaking(bridge, clock, text="what time is it", reply="half past four")
    bridge.on_tts_audio(pcm(CHUNK * 2), turn_id=first)
    pump(bridge, first)

    bridge.on_speech_started(at_ms=9000)
    bridge.on_speech_stopped(at_ms=11_000, reason="silence")
    bridge.on_transcript("and the date?")
    second = bridge.take_pending_response()
    request = bridge.build_generate_request(second)

    assert request.body["messages"] == [
        {"role": "system", "content": bridge.session.system_prompt},
        {"role": "user", "content": "what time is it"},
        {"role": "assistant", "content": "half past four"},
        {"role": "user", "content": "and the date?"},
    ]


def test_an_interrupted_reply_records_only_what_was_plausibly_heard() -> None:
    # Recording the whole reply as if it had been spoken would put words in
    # the machine's mouth the user cut off before hearing.
    reply = "one two three four five six seven eight"
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock, text="count", reply=reply)
    bridge.on_tts_audio(pcm(CHUNK * 8), turn_id=turn_id)
    pump(bridge, turn_id, limit=2)
    bridge.on_speech_started(at_ms=9000)

    assistant = [turn for turn in bridge.session.get_history() if turn["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["content"] != reply
    assert reply.startswith(assistant[0]["content"])


def test_an_interruption_before_any_audio_records_no_assistant_turn() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_speech_started(at_ms=9000)  # cut during synthesis, nothing heard

    assert [t for t in bridge.session.get_history() if t["role"] == "assistant"] == []
    assert [t["role"] for t in bridge.session.get_history()] == ["user"]
    del turn_id


def test_history_dies_with_the_session() -> None:
    bridge, _, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 2), turn_id=turn_id)
    pump(bridge, turn_id)
    assert bridge.session.get_history()

    bridge.close()
    bridge.session.teardown()
    assert bridge.session.get_history() == []


def test_close_is_idempotent_and_cancels_whatever_was_in_flight() -> None:
    bridge, cancels, clock = make_bridge()
    bridge.arm()
    run_to_speaking(bridge, clock)
    bridge.drain()

    bridge.close()
    bridge.close()

    assert (cancels.generate, cancels.tts) == (1, 1)
    assert bridge.drain() == []  # teardown emits nothing; that is Session's job


# ---------------------------------------------------------------------------
# The response.create-after-transcript flow (the OpenAI-shaped per-turn form).
# ---------------------------------------------------------------------------


def test_arming_after_a_transcript_answers_the_waiting_turn() -> None:
    bridge, _, _ = make_bridge()
    commit_turn(bridge)
    bridge.on_transcript("what time is it")
    assert bridge.take_pending_response() is None  # not armed yet
    bridge.drain()

    bridge.on_control_event({"type": "response.create"})

    assert types_of(bridge.drain()) == [S.EventType.RESPONSE_CREATED]
    assert bridge.take_pending_response() is not None


def test_a_duplicate_trigger_cannot_answer_the_same_turn_twice() -> None:
    bridge, _, _ = make_bridge()
    commit_turn(bridge)
    bridge.on_transcript("what time is it")
    bridge.on_control_event({"type": "response.create"})
    bridge.take_pending_response()
    bridge.drain()

    bridge.on_control_event({"type": "response.create"})

    assert bridge.drain() == []
    assert bridge.take_pending_response() is None


def test_arming_with_nothing_said_yet_just_arms() -> None:
    bridge, _, _ = make_bridge()
    bridge.on_control_event({"type": "response.create"})
    assert bridge.armed is True
    assert bridge.drain() == []
    assert bridge.take_pending_response() is None


def test_arming_once_answers_every_committed_turn_thereafter() -> None:
    bridge, _, clock = make_bridge()
    bridge.on_control_event({"type": "response.create"})
    for _ in range(2):
        commit_turn(bridge)
        bridge.on_transcript("hello")
        turn_id = bridge.take_pending_response()
        assert turn_id is not None
        clock.advance(F.DEFAULT_BARGE_IN_WINDOW_MS)
        bridge.on_generate_response(200, chat_body("hi"), turn_id=turn_id)
        bridge.take_pending_synthesis()
        bridge.on_tts_audio(pcm(CHUNK), turn_id=turn_id)
        pump(bridge, turn_id)
    assert bridge.floor.turn_id == 2


# ---------------------------------------------------------------------------
# The sample-rate trap: the floor's rate is the OUTPUT rate.
# ---------------------------------------------------------------------------


def test_the_floor_is_given_the_output_rate_not_the_sessions_input_rate() -> None:
    # A 16 kHz session is accepted on input; passing that into the floor
    # would misreport every audio_end_ms by 1.5x and mis-size every chunk.
    bridge, _, _ = make_bridge({"input_sample_rate": 16000})
    assert bridge.session.config.input_sample_rate == 16000
    assert bridge.floor.sample_rate == TTS_SAMPLE_RATE == 24000


def test_the_truncation_marker_measures_output_milliseconds() -> None:
    bridge, _, clock = make_bridge({"input_sample_rate": 16000})
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    # 10 ms per CHUNK at the OUTPUT rate; two chunks heard = 20 ms.
    bridge.on_tts_audio(pcm(CHUNK * 5), turn_id=turn_id)
    pump(bridge, turn_id, limit=2)
    captured: list = []
    bridge.floor._emit = captured.append  # observe the floor-local fact itself
    bridge.on_speech_started(at_ms=9000)

    (event,) = [e for e in captured if isinstance(e, F.ResponseInterrupted)]
    assert event.audio_end_ms == 20
    assert event.audio_total_ms == 50
    assert event.truncated is True


# ---------------------------------------------------------------------------
# app.py's two runtime obligations — asserted structurally against its source.
# ---------------------------------------------------------------------------


def _app_source() -> str:
    return (Path(W.__file__).parent / "app.py").read_text(encoding="utf-8")


def _function(name: str):
    src = _app_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node, src
    raise AssertionError(f"app.py has no function named {name!r}")


def test_app_py_pumps_delivery_with_an_await_between_chunks() -> None:
    # THE runtime contract. Wired as a tight synchronous loop, every
    # interruption guarantee in _floor.py is dead live while every unit test
    # above still passes — so this is asserted on the loop's STRUCTURE.
    node, _src = _function("_drive_response")
    loops = [
        loop
        for loop in ast.walk(node)
        if isinstance(loop, ast.While)
        and any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "deliver_next"
            for call in ast.walk(loop.test)
            if isinstance(call, ast.Call)
        )
    ]
    assert loops, "app.py must pump deliver_next() in a loop"
    for loop in loops:
        assert any(
            isinstance(inner, ast.Await)
            for inner in ast.walk(ast.Module(body=loop.body, type_ignores=[]))
        ), "the delivery pump must await between chunks or a barge-in can never land"


def test_app_py_drives_tick_from_a_watchdog_loop() -> None:
    # Deadlines expire ONLY inside tick(); no watchdog means no timeouts and
    # the "floor never wedges" property evaporates silently.
    node, _src = _function("_watchdog")
    calls = [
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    ]
    assert "tick" in calls
    assert any(isinstance(inner, ast.While) for inner in ast.walk(node))
    assert any(isinstance(inner, ast.Await) for inner in ast.walk(node))


def test_app_py_starts_the_watchdog_when_the_session_arms() -> None:
    node, _src = _function("_pump_session")
    calls = {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert {"on_control_event", "ensure_watchdog", "on_wire_error"} <= calls


def test_app_py_threads_the_default_system_prompt_into_the_session_config() -> None:
    node, _src = _function("_open_session")
    keywords = {
        kw.arg for call in ast.walk(node) if isinstance(call, ast.Call) for kw in call.keywords
    }
    assert "default_system_prompt" in keywords


def test_app_py_applies_the_voice_lane_model_policy_at_the_call_site() -> None:
    # honesty h4: _turn.py refuses to hardcode it and t8 did not pre-apply it,
    # so if the wiring layer skips it the voice lane silently default-routes.
    node, src = _function("_build_bridge")
    segment = ast.get_source_segment(src, node) or ""
    assert "resolve_voice_model(settings.openai_model)" in segment
    # …and the policy literal itself stays in the stdlib module, not here.
    assert '"multimodal"' not in src


def test_app_py_uses_the_wire_codecs_chunk_size_not_a_literal() -> None:
    node, src = _function("_build_bridge")
    segment = ast.get_source_segment(src, node) or ""
    assert "DEFAULT_DELTA_CHUNK_BYTES" in segment
    assert "4800" not in src and "1920" not in src


def test_app_py_no_longer_builds_error_events_itself() -> None:
    # The route used to mint an ErrorEvent with a WireErrorCode in the `code`
    # field. Event construction belongs to _session.py; the mapping belongs
    # to _conversation.py.
    src = _app_source()
    assert "ErrorEvent(" not in src
    assert "_wire_error_event" not in src


def test_app_py_synthesizes_on_the_voice_lane() -> None:
    node, src = _function("_drive_response")
    segment = ast.get_source_segment(src, node) or ""
    assert "lane=VOICE_LANE" in segment
    assert "cancel_event=active.tts_cancel" in segment


def test_batch_speech_route_never_opts_into_a_tts_lane() -> None:
    """Boundary guard (issue #151 t15): POST /v1/audio/speech stays lane-blind.

    t7 gave ``tts_client.synthesize()`` a ``lane`` parameter so a voice reply
    never queues behind batch work — but its default is ``BATCH_LANE``
    (``normalize_tts_lane(None) == BATCH_LANE``, see
    ``lobes/realtime/_settings.py``), meaning the ORIGINAL, pre-#151 batch
    route keeps its exact pre-t7 behavior only as long as it never passes
    ``lane=`` at all. Nothing exercises ``speech()`` at runtime (fastapi
    route shells are pragma-no-cover and no CI lane installs the
    ``[realtime]`` extra — see test_realtime_imports.py), so this is the one
    place that would catch a future edit routing the batch handler onto the
    voice lane, or onto any explicit lane at all.
    """
    node, src = _function("speech")
    segment = ast.get_source_segment(src, node) or ""
    assert "lane=" not in segment, "the batch /v1/audio/speech route must stay lane-blind"
    assert "synthesize(" in segment, "expected speech() to still call tts_client.synthesize()"


def test_app_py_never_resamples_on_the_way_out() -> None:
    node, src = _function("_drive_response")
    segment = (ast.get_source_segment(src, node) or "").lower()
    assert "scipy" not in segment
    assert "_to_pcm16k" not in segment
    assert "_resample_to_16k" not in segment


def test_the_gateway_knows_nothing_about_the_conversation_surface() -> None:
    # Acceptance criterion 2's other half: the tunnel already relays both
    # directions, so audio-out needs no gateway change at all.
    gateway_dir = Path(W.__file__).parent.parent / "gateway"
    for path in gateway_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in ("_conversation", "_floor", "response.audio.delta", "ConversationBridge"):
            assert token not in text, f"{path.name} must not know about {token}"


# ---------------------------------------------------------------------------
# VAD loss mid-reply, the outbox counter, and the trigger's own refusal path.
# ---------------------------------------------------------------------------


def test_a_vad_failure_mid_reply_stops_the_machine_speaking() -> None:
    # A session whose VAD is gone can no longer detect the barge-in that
    # makes speaking safe, so it must not keep speaking either.
    bridge, cancels, clock = make_bridge()
    bridge.arm()
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 6), turn_id=turn_id)
    pump(bridge, turn_id, limit=1)
    bridge.drain()

    bridge.fail_vad("RuntimeError: silero fell over")

    payloads = bridge.drain()
    assert types_of(payloads) == [S.EventType.ERROR]
    assert payloads[0]["code"] is S.ErrorCode.VAD_UNAVAILABLE
    assert bridge.floor.state is F.FloorState.CLOSED
    assert pump(bridge, turn_id) == 0  # the undelivered remainder is dropped
    assert (cancels.generate, cancels.tts) == (1, 1)


def test_the_outbox_counter_tracks_undrained_payloads() -> None:
    bridge, _, _ = make_bridge()
    assert bridge.pending_payloads == 0
    commit_turn(bridge)
    assert bridge.pending_payloads == 2
    bridge.drain()
    assert bridge.pending_payloads == 0


def test_a_trigger_arriving_inside_the_guard_window_answers_nothing() -> None:
    # The floor refuses to open a turn while it holds the floor and the
    # barge-in guard has not elapsed; the trigger must then be a no-op rather
    # than half-opening a turn nobody will ever finish.
    bridge, _, _clock = make_bridge()
    commit_turn(bridge)
    bridge.on_transcript("first question")
    bridge.on_control_event({"type": "response.create"})  # answers the first
    assert bridge.take_pending_response() is not None
    bridge.drain()

    # A second turn commits while the machine still holds the floor, inside
    # the guard window (the clock has not moved), so the floor drops it.
    commit_turn(bridge, at_start=3000, at_stop=4000)
    bridge.on_transcript("second question")
    bridge.drain()

    bridge.on_control_event({"type": "response.create"})

    assert bridge.drain() == []
    assert bridge.take_pending_response() is None
    assert bridge.floor.state is F.FloorState.RESPONDING  # still on turn one


def test_app_py_never_swallows_its_own_cancellation() -> None:
    """No ``except asyncio.CancelledError`` in app.py may return instead of re-raising.

    ``_drive_response`` runs inside a tracked ``_run_response`` task that session
    teardown cancels, while barge-in cancels the two child tasks it awaits.
    Catching both as one exception and sorting them out afterwards is how a
    torn-down session keeps running: the driver returns normally, flushes a
    socket that is going away, and reports itself COMPLETED to whoever cancelled
    it. So the route does not await those children directly at all — it uses
    ``asyncio.wait()``, which surfaces a cancelled child as ``task.cancelled()``
    while letting this task's own cancellation propagate untouched.

    That leaves every remaining handler free to re-raise unconditionally, which
    is what this asserts. Structural (AST) because app.py imports fastapi and is
    never executed by the offline suite — the same reason its routes carry
    ``pragma: no cover``.
    """
    tree = ast.parse(_app_source())

    def _is_cancelled_error(node: ast.expr | None) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "CancelledError"

    handlers = [
        handler
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler) and _is_cancelled_error(handler.type)
    ]
    assert handlers, "expected app.py to handle asyncio.CancelledError somewhere"

    for handler in handlers:
        returns = [node for node in ast.walk(handler) if isinstance(node, ast.Return)]
        raises = [node for node in ast.walk(handler) if isinstance(node, ast.Raise)]
        assert not returns, (
            f"CancelledError handler at line {handler.lineno} returns — it can swallow "
            "this task's own cancellation. Use asyncio.wait() and check "
            "task.cancelled() instead of catching the child's cancellation here."
        )
        assert raises, f"CancelledError handler at line {handler.lineno} never re-raises"


def test_app_py_waits_on_its_child_tasks_instead_of_awaiting_them() -> None:
    """The generate and TTS stages must go through ``asyncio.wait()``.

    This is the mechanism the test above depends on: awaiting a child directly
    collapses "my child was cancelled by barge-in" and "I was cancelled by
    teardown" into one indistinguishable CancelledError.
    """
    node, _src = _function("_drive_response")
    waits = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "wait"
    ]
    assert len(waits) >= 2, (
        "_drive_response should await its generate and TTS tasks via asyncio.wait(), "
        f"found {len(waits)} wait() call(s)"
    )
    cancelled_checks = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "cancelled"
    ]
    assert len(cancelled_checks) >= 2, (
        "each waited task must be checked with task.cancelled() before its result "
        f"is read, found {len(cancelled_checks)} check(s)"
    )
