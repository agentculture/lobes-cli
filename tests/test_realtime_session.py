"""Tests for the realtime session engine (stdlib-only; no [realtime] extra).

Mirrors the style of tests/test_realtime_protocol.py and
tests/test_realtime_settings.py: the module under test imports with the
standard library alone, so these tests run in the offline CI environment
(no torch, no fastapi, no GPU).

Covers the t2 acceptance criteria from
docs/plans/2026-07-21-realtime-ws-server-vad-149.md:

1. config parse accepts PCM16 mono at 24000 (default) and 16000, rejecting
   others with a named error
2. AEC defaults to none and stays off unless the session explicitly enables it
3. VAD-unavailable yields the documented named error event and no boundary
   events; a silent healthy session yields neither — distinguishable by event
   type alone
4. teardown from every state (idle, mid-speech, mid-transcription) releases
   all session bookkeeping
5. every event and log record carries the session id; no key material
   appears in any log line; no code path persists session state to disk
"""

from __future__ import annotations

import inspect
import logging

import pytest

from lobes.realtime import _session as S
from lobes.realtime import protocol as P

# ---------------------------------------------------------------------------
# Reuse, not redefinition — protocol.py owns ids/constants/enums (CLAUDE.md).
# ---------------------------------------------------------------------------


def test_id_generators_are_reused_from_protocol_not_redefined() -> None:
    assert S.gen_session_id is P.gen_session_id
    assert S.gen_event_id is P.gen_event_id
    assert S.gen_item_id is P.gen_item_id


def test_audio_enums_are_reused_from_protocol_not_redefined() -> None:
    assert S.AudioFormat is P.AudioFormat
    assert S.TurnDetectionType is P.TurnDetectionType
    assert S.AECMode is P.AECMode


# ---------------------------------------------------------------------------
# Criterion 1 — config parse: PCM16 mono, 24000 default / 16000 accepted.
# ---------------------------------------------------------------------------


def test_default_config_is_pcm16_mono_24000_server_vad_aec_none() -> None:
    config = S.parse_session_config({})
    assert config.input_audio_format is P.AudioFormat.PCM16
    assert config.input_sample_rate == P.CLIENT_SAMPLE_RATE == 24000
    assert config.channels == 1
    assert config.turn_detection is P.TurnDetectionType.SERVER_VAD
    assert config.aec_mode is P.AECMode.NONE


def test_16000_hz_is_accepted() -> None:
    config = S.parse_session_config({"input_sample_rate": 16000})
    assert config.input_sample_rate == 16000 == P.STT_SAMPLE_RATE


@pytest.mark.parametrize("bad_rate", [8000, 44100, 48000, 0, -1])
def test_unsupported_sample_rates_are_rejected_with_a_named_error(bad_rate: int) -> None:
    with pytest.raises(S.SessionConfigError) as exc_info:
        S.parse_session_config({"input_sample_rate": bad_rate})
    assert exc_info.value.code is S.ErrorCode.INVALID_SESSION_CONFIG


def test_non_integer_sample_rate_is_rejected_with_a_named_error() -> None:
    with pytest.raises(S.SessionConfigError) as exc_info:
        S.parse_session_config({"input_sample_rate": "fast"})
    assert exc_info.value.code is S.ErrorCode.INVALID_SESSION_CONFIG


@pytest.mark.parametrize("bad_format", ["pcm_f32le", "opus", "mulaw", 42])
def test_unsupported_audio_formats_are_rejected_with_a_named_error(bad_format) -> None:
    with pytest.raises(S.SessionConfigError) as exc_info:
        S.parse_session_config({"input_audio_format": bad_format})
    assert exc_info.value.code is S.ErrorCode.INVALID_SESSION_CONFIG


@pytest.mark.parametrize("bad_channels", [2, 0, "stereo"])
def test_unsupported_channel_counts_are_rejected_with_a_named_error(bad_channels) -> None:
    with pytest.raises(S.SessionConfigError) as exc_info:
        S.parse_session_config({"input_channels": bad_channels})
    assert exc_info.value.code is S.ErrorCode.INVALID_SESSION_CONFIG


def test_unsupported_turn_detection_is_rejected_with_a_named_error() -> None:
    with pytest.raises(S.SessionConfigError) as exc_info:
        S.parse_session_config({"turn_detection": "manual"})
    assert exc_info.value.code is S.ErrorCode.INVALID_SESSION_CONFIG


def test_config_error_is_named_not_a_bare_exception_string() -> None:
    # The rejection carries a documented ErrorCode enum member (not just a
    # message string) and can build a proper named error Event from it.
    with pytest.raises(S.SessionConfigError) as exc_info:
        S.parse_session_config({"input_sample_rate": 8000})
    exc = exc_info.value
    assert isinstance(exc.code, S.ErrorCode)
    event = exc.to_error_event(session_id="sess_test123")
    assert isinstance(event, S.ErrorEvent)
    assert event.type is S.EventType.ERROR
    assert event.code is S.ErrorCode.INVALID_SESSION_CONFIG
    assert event.session_id == "sess_test123"
    assert event.event_id  # a real event id, not a placeholder


# ---------------------------------------------------------------------------
# Criterion 2 — AEC defaults to none; explicit opt-in only.
# ---------------------------------------------------------------------------


def test_aec_defaults_to_none_when_omitted() -> None:
    config = S.parse_session_config({})
    assert config.aec_mode is P.AECMode.NONE


def test_aec_stays_none_alongside_other_explicit_fields() -> None:
    config = S.parse_session_config({"input_sample_rate": 16000, "turn_detection": "server_vad"})
    assert config.aec_mode is P.AECMode.NONE


def test_aec_enables_only_via_explicit_opt_in() -> None:
    config = S.parse_session_config({"aec_mode": "aec"})
    assert config.aec_mode is P.AECMode.AEC


def test_unsupported_aec_mode_is_rejected_with_a_named_error() -> None:
    with pytest.raises(S.SessionConfigError) as exc_info:
        S.parse_session_config({"aec_mode": "echo_cancel_v2"})
    assert exc_info.value.code is S.ErrorCode.INVALID_SESSION_CONFIG


def test_default_aec_mode_is_threaded_from_settings_style_default() -> None:
    # A caller can pass its own settings-sourced default (mirrors _settings.py's
    # default_aec_mode); an explicit per-session value still overrides it.
    config = S.parse_session_config({}, default_aec_mode="none")
    assert config.aec_mode is P.AECMode.NONE
    config2 = S.parse_session_config({"aec_mode": "aec"}, default_aec_mode="none")
    assert config2.aec_mode is P.AECMode.AEC


# ---------------------------------------------------------------------------
# Criterion 3 — VAD-unavailable vs. silent-but-healthy, by event type alone.
# ---------------------------------------------------------------------------


def test_vad_unavailable_yields_named_error_event_and_no_boundary_events() -> None:
    config = S.parse_session_config({})
    session, created = S.Session.create(config)
    error_event = session.mark_vad_unavailable()

    events = [created, error_event]
    types = {e.type for e in events}

    assert error_event.type is S.EventType.ERROR
    assert error_event.code is S.ErrorCode.VAD_UNAVAILABLE
    assert error_event.session_id == session.session_id
    assert S.EventType.SPEECH_STARTED not in types
    assert S.EventType.SPEECH_STOPPED not in types


def test_silent_healthy_session_yields_neither_error_nor_boundary_events() -> None:
    config = S.parse_session_config({})
    session, created = S.Session.create(config)
    # Nothing else happens: no VAD failure, no speech. The only event so far
    # is session.created.
    events = [created]
    types = {e.type for e in events}

    assert S.EventType.ERROR not in types
    assert S.EventType.SPEECH_STARTED not in types
    assert S.EventType.SPEECH_STOPPED not in types


def test_vad_unavailable_and_silent_healthy_are_distinguishable_by_type_alone() -> None:
    config = S.parse_session_config({})

    healthy_session, healthy_created = S.Session.create(config)
    healthy_types = {healthy_created.type}

    down_session, down_created = S.Session.create(config)
    down_error = down_session.mark_vad_unavailable()
    down_types = {down_created.type, down_error.type}

    # The presence/absence of EventType.ERROR is the ENTIRE distinguishing
    # signal — no need to inspect message text or any other field.
    assert S.EventType.ERROR in down_types
    assert S.EventType.ERROR not in healthy_types


# ---------------------------------------------------------------------------
# Criterion 4 — teardown from every state releases all bookkeeping.
# ---------------------------------------------------------------------------


def test_teardown_from_idle_releases_bookkeeping() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    assert session.state is S.SessionState.IDLE

    closed_event = session.teardown()

    assert session.state is S.SessionState.CLOSED
    assert session.current_item_id is None
    assert session.has_open_item is False
    assert closed_event.type is S.EventType.SESSION_CLOSED
    assert closed_event.session_id == session.session_id


def test_teardown_from_mid_speech_releases_bookkeeping() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech()
    assert session.state is S.SessionState.SPEECH
    assert session.has_open_item is True

    session.teardown()

    assert session.state is S.SessionState.CLOSED
    assert session.current_item_id is None
    assert session.has_open_item is False


def test_teardown_from_mid_transcription_releases_bookkeeping() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech()
    session.end_speech()
    assert session.state is S.SessionState.TRANSCRIBING
    assert session.has_open_item is True

    session.teardown()

    assert session.state is S.SessionState.CLOSED
    assert session.current_item_id is None
    assert session.has_open_item is False


def test_teardown_is_idempotent() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech()

    first = session.teardown()
    second = session.teardown()  # a second close (e.g. both sides of a tunnel) must not blow up

    assert session.state is S.SessionState.CLOSED
    assert first.session_id == second.session_id


def test_state_transitions_after_teardown_are_refused() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.teardown()

    with pytest.raises(S.SessionClosedError):
        session.begin_speech()


def test_two_sessions_do_not_share_bookkeeping() -> None:
    config = S.parse_session_config({})
    a, _ = S.Session.create(config)
    b, _ = S.Session.create(config)

    a.begin_speech()

    assert a.state is S.SessionState.SPEECH
    assert b.state is S.SessionState.IDLE
    assert a.session_id != b.session_id


# ---------------------------------------------------------------------------
# Criterion 5 — session id on every record; no key material; no persistence.
# ---------------------------------------------------------------------------


def test_full_lifecycle_every_log_record_carries_the_session_id(caplog) -> None:
    config = S.parse_session_config({})
    with caplog.at_level(logging.INFO, logger="lobes.realtime._session"):
        session, _ = S.Session.create(config)
        session.begin_speech()
        session.end_speech()
        session.complete_transcription("hello world")
        session.teardown()

    records = [r for r in caplog.records if r.name == "lobes.realtime._session"]
    assert records, "expected at least one log record from the session lifecycle"
    for record in records:
        assert session.session_id in record.getMessage()


def test_redact_for_log_masks_credential_shaped_keys() -> None:
    raw = {
        "api_key": "sk-super-secret-value",
        "Authorization": "Bearer abc123",
        "auth_token": "tok-xyz",
        "secret": "shh",
        "password": "hunter2",
        "input_sample_rate": 24000,
    }
    redacted = S.redact_for_log(raw)
    assert redacted["input_sample_rate"] == 24000
    for key in ("api_key", "Authorization", "auth_token", "secret", "password"):
        assert redacted[key] == S.REDACTED_MARKER
        assert redacted[key] != raw[key]


def test_create_never_leaks_api_key_shaped_payload_fields_into_logs(caplog) -> None:
    config = S.parse_session_config({})
    secret = "sk-do-not-leak-this-9999"
    with caplog.at_level(logging.DEBUG, logger="lobes.realtime._session"):
        session, _ = S.Session.create(
            config, raw_payload={"api_key": secret, "input_sample_rate": 24000}
        )

    assert secret not in caplog.text
    assert session.session_id in caplog.text


def test_transcription_text_is_not_logged_verbatim(caplog) -> None:
    config = S.parse_session_config({})
    secret_sounding_text = "my password is hunter2-actually-transcribed-speech"
    with caplog.at_level(logging.INFO, logger="lobes.realtime._session"):
        session, _ = S.Session.create(config)
        session.begin_speech()
        session.end_speech()
        session.complete_transcription(secret_sounding_text)

    assert secret_sounding_text not in caplog.text


def test_module_source_has_no_disk_persistence_code_paths() -> None:
    # Sessions are ephemeral by contract (c31/h23): no resume, no on-disk
    # state. A static scan is a cheap, deterministic guard against a future
    # regression sneaking in a persistence code path.
    source = inspect.getsource(S)
    forbidden = ("open(", "pathlib", ".write(", "pickle", "shelve", "sqlite3")
    for token in forbidden:
        assert token not in source, f"found {token!r} in _session.py — sessions must not persist"


def test_no_disk_writes_occur_across_a_full_lifecycle(monkeypatch) -> None:
    def _forbidden_open(*args, **kwargs):
        raise AssertionError("_session must never touch the filesystem")

    monkeypatch.setattr("builtins.open", _forbidden_open)

    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech()
    session.end_speech()
    session.complete_transcription("ok")
    session.mark_vad_unavailable()
    session.teardown()


# ---------------------------------------------------------------------------
# Event schema shape — the schema the next tasks (t1 wiring, t6 route) consume.
# ---------------------------------------------------------------------------


def test_event_types_are_openai_realtime_flavoured_strings() -> None:
    assert S.EventType.SESSION_CREATED.value == "session.created"
    assert S.EventType.SESSION_CLOSED.value == "session.closed"
    assert S.EventType.SPEECH_STARTED.value == "input_audio_buffer.speech_started"
    assert S.EventType.SPEECH_STOPPED.value == "input_audio_buffer.speech_stopped"
    assert (
        S.EventType.TRANSCRIPTION_COMPLETED.value
        == "conversation.item.input_audio_transcription.completed"
    )
    assert S.EventType.ERROR.value == "error"


def test_error_codes_are_documented_members() -> None:
    assert S.ErrorCode.INVALID_SESSION_CONFIG.value == "invalid_session_config"
    assert S.ErrorCode.VAD_UNAVAILABLE.value == "vad_unavailable"
    assert S.ErrorCode.STT_FORWARD_FAILED.value == "stt_forward_failed"


def test_speech_started_and_stopped_carry_the_same_item_id() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)

    started = session.begin_speech()
    stopped = session.end_speech()

    assert started.item_id == stopped.item_id
    assert started.session_id == session.session_id == stopped.session_id


def test_complete_transcription_returns_named_event_with_text_and_clears_item() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech()
    session.end_speech()

    event = session.complete_transcription("hello there")

    assert event.type is S.EventType.TRANSCRIPTION_COMPLETED
    assert event.text == "hello there"
    assert session.state is S.SessionState.IDLE
    assert session.has_open_item is False


def test_fail_transcription_returns_named_error_event() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech()
    session.end_speech()

    event = session.fail_transcription("stt backend unreachable")

    assert event.type is S.EventType.ERROR
    assert event.code is S.ErrorCode.STT_FORWARD_FAILED
    assert session.state is S.SessionState.IDLE
    assert session.has_open_item is False


def test_event_to_dict_round_trips_through_json() -> None:
    import json

    config = S.parse_session_config({})
    session, created = S.Session.create(config)

    payload = json.loads(json.dumps(S.event_to_dict(created)))
    assert payload["type"] == "session.created"
    assert payload["session_id"] == session.session_id


# ---------------------------------------------------------------------------
# #151 t3 — response lifecycle + interruption schema, floor states, history.
#
# Covers the t3 acceptance criteria from
# docs/plans/2026-07-21-realtime-voice-to-voice-astro-test-site-151.md:
#
# 1. every new event dataclass is frozen and event_to_dict-serializable; the
#    floor holder is explicit in the schema (SessionState.RESPONDING/SPEAKING)
# 2. a two-turn offline conversation asserts the second generate request
#    carries the first exchange; history lives only on the Session object
# 3. teardown from every new state releases history and bookkeeping
#    (idempotent, safe from any state)
# ---------------------------------------------------------------------------


# --- new schema members exist with the documented values -------------------


def test_new_event_types_are_openai_realtime_flavoured_strings() -> None:
    assert S.EventType.RESPONSE_CREATED.value == "response.created"
    assert S.EventType.RESPONSE_TEXT_DONE.value == "response.text.done"
    assert S.EventType.RESPONSE_AUDIO_DELTA.value == "response.audio.delta"
    assert S.EventType.RESPONSE_DONE.value == "response.done"
    assert S.EventType.RESPONSE_INTERRUPTED.value == "response.interrupted"


def test_new_error_codes_are_documented_members() -> None:
    assert S.ErrorCode.GENERATE_FAILED.value == "generate_failed"
    assert S.ErrorCode.TTS_FAILED.value == "tts_failed"
    assert S.ErrorCode.RESPONSE_TIMEOUT.value == "response_timeout"


def test_new_session_states_exist_for_responding_and_speaking() -> None:
    assert S.SessionState.RESPONDING.value == "responding"
    assert S.SessionState.SPEAKING.value == "speaking"


# --- criterion 1: every new event dataclass is frozen + serializable -------


def test_response_created_event_is_frozen_and_serializable() -> None:
    import dataclasses
    import json

    event = S.ResponseCreatedEvent(
        session_id="sess_x",
        event_id="event_x",
        timestamp_ms=1,
        response_id="resp_x",
        item_id="item_x",
    )
    assert event.type is S.EventType.RESPONSE_CREATED
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.response_id = "resp_y"  # type: ignore[misc]

    payload = json.loads(json.dumps(S.event_to_dict(event)))
    assert payload["type"] == "response.created"
    assert payload["response_id"] == "resp_x"
    assert payload["item_id"] == "item_x"


def test_response_text_done_event_is_frozen_and_serializable() -> None:
    import dataclasses
    import json

    event = S.ResponseTextDoneEvent(
        session_id="sess_x", event_id="event_x", timestamp_ms=1, response_id="resp_x", text="hi"
    )
    assert event.type is S.EventType.RESPONSE_TEXT_DONE
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "bye"  # type: ignore[misc]

    payload = json.loads(json.dumps(S.event_to_dict(event)))
    assert payload["type"] == "response.text.done"
    assert payload["text"] == "hi"


def test_response_audio_delta_event_is_frozen_and_serializable() -> None:
    import dataclasses
    import json

    event = S.ResponseAudioDeltaEvent(
        session_id="sess_x", event_id="event_x", timestamp_ms=1, response_id="resp_x", delta="QUJD"
    )
    assert event.type is S.EventType.RESPONSE_AUDIO_DELTA
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.delta = "ZFJH"  # type: ignore[misc]

    payload = json.loads(json.dumps(S.event_to_dict(event)))
    assert payload["type"] == "response.audio.delta"
    assert payload["delta"] == "QUJD"


def test_response_done_event_is_frozen_and_serializable() -> None:
    import dataclasses
    import json

    event = S.ResponseDoneEvent(
        session_id="sess_x", event_id="event_x", timestamp_ms=1, response_id="resp_x"
    )
    assert event.type is S.EventType.RESPONSE_DONE
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.response_id = "resp_y"  # type: ignore[misc]

    payload = json.loads(json.dumps(S.event_to_dict(event)))
    assert payload["type"] == "response.done"
    assert payload["response_id"] == "resp_x"


def test_response_interrupted_event_is_frozen_and_serializable() -> None:
    import dataclasses
    import json

    event = S.ResponseInterruptedEvent(
        session_id="sess_x", event_id="event_x", timestamp_ms=1, response_id="resp_x"
    )
    assert event.type is S.EventType.RESPONSE_INTERRUPTED
    assert event.truncated is True  # the "truncated marker" is on by default
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.truncated = False  # type: ignore[misc]

    payload = json.loads(json.dumps(S.event_to_dict(event)))
    assert payload["type"] == "response.interrupted"
    assert payload["truncated"] is True


# --- criterion 1 (continued): the floor holder through a full response ----


def test_begin_response_moves_floor_to_responding() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech()
    stopped = session.end_speech()
    session.complete_transcription("what time is it")

    event = session.begin_response(item_id=stopped.item_id)

    assert session.state is S.SessionState.RESPONDING
    assert event.type is S.EventType.RESPONSE_CREATED
    assert event.item_id == stopped.item_id
    assert event.response_id  # a real id, not a placeholder


def test_complete_response_text_moves_floor_to_speaking() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    created = session.begin_response()

    event = session.complete_response_text("it is three o'clock")

    assert session.state is S.SessionState.SPEAKING
    assert event.type is S.EventType.RESPONSE_TEXT_DONE
    assert event.response_id == created.response_id
    assert event.text == "it is three o'clock"


def test_emit_audio_delta_does_not_change_floor_state() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_response()
    session.complete_response_text("hi there")

    event = session.emit_audio_delta("QUJD")

    assert session.state is S.SessionState.SPEAKING
    assert event.type is S.EventType.RESPONSE_AUDIO_DELTA
    assert event.delta == "QUJD"


def test_complete_response_returns_floor_to_idle() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_response()
    session.complete_response_text("hi there")
    session.emit_audio_delta("QUJD")

    event = session.complete_response()

    assert session.state is S.SessionState.IDLE
    assert session.current_response_id is None
    assert event.type is S.EventType.RESPONSE_DONE


@pytest.mark.parametrize("stage", ["responding", "speaking"])
def test_interrupt_returns_floor_to_idle_from_every_responding_state(stage: str) -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_response()
    if stage == "speaking":
        session.complete_response_text("a reply in progress")

    assert session.state is S.SessionState(stage)

    event = session.interrupt_response()

    assert session.state is S.SessionState.IDLE
    assert session.current_response_id is None
    assert event.type is S.EventType.RESPONSE_INTERRUPTED
    assert event.truncated is True


@pytest.mark.parametrize("stage", ["responding", "speaking"])
def test_fail_response_returns_floor_to_idle_with_named_error(stage: str) -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_response()
    if stage == "speaking":
        session.complete_response_text("a reply in progress")

    event = session.fail_response(S.ErrorCode.GENERATE_FAILED, "gateway 404 role_infeasible")

    assert session.state is S.SessionState.IDLE
    assert session.current_response_id is None
    assert event.type is S.EventType.ERROR
    assert event.code is S.ErrorCode.GENERATE_FAILED


def test_fail_response_accepts_tts_failed_and_response_timeout_codes() -> None:
    config = S.parse_session_config({})
    for code in (S.ErrorCode.TTS_FAILED, S.ErrorCode.RESPONSE_TIMEOUT):
        session, _ = S.Session.create(config)
        session.begin_response()
        event = session.fail_response(code, "stage exceeded its deadline")
        assert event.code is code
        assert session.state is S.SessionState.IDLE


# --- criterion 2: history lives only on Session; second request carries ---
# --- the first exchange -----------------------------------------------------


def test_two_turn_conversation_history_carries_first_exchange_into_second_request() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)

    # Turn 1.
    session.append_history("user", "what's the weather like")
    first_request_messages = [
        {"role": "system", "content": session.system_prompt}
    ] + session.get_history()
    assert first_request_messages[-1] == {"role": "user", "content": "what's the weather like"}
    session.append_history("assistant", "I can't check live weather from here")
    first_exchange = session.get_history()  # the whole turn-1 user+assistant pair

    # Turn 2 — the SECOND generate request's payload must carry turn 1 whole.
    session.append_history("user", "ok, tell me a joke instead")
    second_request_messages = [
        {"role": "system", "content": session.system_prompt}
    ] + session.get_history()

    assert second_request_messages == [
        {"role": "system", "content": session.system_prompt},
        {"role": "user", "content": "what's the weather like"},
        {"role": "assistant", "content": "I can't check live weather from here"},
        {"role": "user", "content": "ok, tell me a joke instead"},
    ]
    # The first exchange (both turn-1 messages) is a strict subsequence of
    # the second request, appearing before the new turn-2 user message.
    assert second_request_messages[1:3] == first_exchange


def test_get_history_returns_a_defensive_copy() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.append_history("user", "hello")

    snapshot = session.get_history()
    snapshot.append({"role": "user", "content": "mutated after the fact"})

    assert session.get_history() == [{"role": "user", "content": "hello"}]


def test_history_lives_only_on_session_object_not_shared_across_sessions() -> None:
    config = S.parse_session_config({})
    a, _ = S.Session.create(config)
    b, _ = S.Session.create(config)

    a.append_history("user", "only in session a")

    assert a.get_history() == [{"role": "user", "content": "only in session a"}]
    assert b.get_history() == []


def test_history_does_not_persist_to_disk(monkeypatch) -> None:
    def _forbidden_open(*args, **kwargs):
        raise AssertionError("_session must never touch the filesystem")

    monkeypatch.setattr("builtins.open", _forbidden_open)

    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.append_history("user", "hello")
    session.append_history("assistant", "hi there")
    session.teardown()


# --- criterion 3: teardown from every new state releases bookkeeping ------


def test_teardown_from_responding_releases_bookkeeping() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.append_history("user", "hi")
    session.begin_response()
    assert session.state is S.SessionState.RESPONDING

    closed_event = session.teardown()

    assert session.state is S.SessionState.CLOSED
    assert session.current_response_id is None
    assert session.get_history() == []
    assert closed_event.type is S.EventType.SESSION_CLOSED


def test_teardown_from_speaking_releases_bookkeeping() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.append_history("user", "hi")
    session.begin_response()
    session.complete_response_text("a reply")
    assert session.state is S.SessionState.SPEAKING

    session.teardown()

    assert session.state is S.SessionState.CLOSED
    assert session.current_response_id is None
    assert session.get_history() == []


def test_teardown_is_idempotent_from_speaking() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_response()
    session.complete_response_text("a reply")

    first = session.teardown()
    second = session.teardown()

    assert session.state is S.SessionState.CLOSED
    assert first.session_id == second.session_id


def test_response_methods_after_teardown_are_refused() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.teardown()

    with pytest.raises(S.SessionClosedError):
        session.begin_response()


def test_append_history_after_teardown_is_refused() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.teardown()

    with pytest.raises(S.SessionClosedError):
        session.append_history("user", "too late")


# --- system prompt: default, per-session override, never logged verbatim --


def test_default_system_prompt_is_used_when_not_overridden() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)

    assert session.system_prompt == S.DEFAULT_SYSTEM_PROMPT
    assert isinstance(session.system_prompt, str)
    assert session.system_prompt  # non-empty


def test_system_prompt_override_via_connect_config() -> None:
    config = S.parse_session_config({"system_prompt": "You are a pirate."})
    session, _ = S.Session.create(config)

    assert session.system_prompt == "You are a pirate."


def test_system_prompt_threaded_from_settings_style_default() -> None:
    # Mirrors default_aec_mode/default_turn_detection: a caller-supplied
    # settings-style default is used when the client omits system_prompt,
    # and an explicit per-session value still overrides it.
    config = S.parse_session_config({}, default_system_prompt="Operator default prompt.")
    assert config.system_prompt == "Operator default prompt."

    config2 = S.parse_session_config(
        {"system_prompt": "Client override."}, default_system_prompt="Operator default prompt."
    )
    assert config2.system_prompt == "Client override."


def test_non_string_system_prompt_is_rejected_with_a_named_error() -> None:
    with pytest.raises(S.SessionConfigError) as exc_info:
        S.parse_session_config({"system_prompt": 12345})
    assert exc_info.value.code is S.ErrorCode.INVALID_SESSION_CONFIG


def test_response_text_is_not_logged_verbatim(caplog) -> None:
    config = S.parse_session_config({})
    secret_sounding_reply = "my password is hunter2-actually-a-spoken-reply"
    with caplog.at_level(logging.INFO, logger="lobes.realtime._session"):
        session, _ = S.Session.create(config)
        session.begin_response()
        session.complete_response_text(secret_sounding_reply)

    assert secret_sounding_reply not in caplog.text


def test_history_content_is_not_logged_verbatim(caplog) -> None:
    config = S.parse_session_config({})
    secret_sounding_text = "my password is hunter2-actually-history-content"
    with caplog.at_level(logging.DEBUG, logger="lobes.realtime._session"):
        session, _ = S.Session.create(config)
        session.append_history("user", secret_sounding_text)

    assert secret_sounding_text not in caplog.text


# ---------------------------------------------------------------------------
# #151 t6 — boundary timings on the wire, and the single error vocabulary.
#
# _segmenter.py computed SpeechStarted.at_ms / SpeechStopped.at_ms+reason from
# the very first #149 commit; the route dropped all three before they reached
# the wire, so honesty condition h19 ("a live event stream that shows VAD
# boundaries and timings") failed live and no client could tell a max_turn
# force-commit from a silence-confirmed stop. The schema carries them now.
# ---------------------------------------------------------------------------


def test_begin_speech_carries_the_segmenters_audio_stream_time() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)

    event = session.begin_speech(at_ms=288)

    assert event.at_ms == 288
    # A DIFFERENT clock from timestamp_ms (a monotonic process clock) — the
    # two are separate fields precisely because they are not comparable.
    assert event.timestamp_ms != 288 or event.at_ms == event.timestamp_ms


def test_end_speech_carries_the_commit_reason_and_time() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech(at_ms=128)

    event = session.end_speech(at_ms=30_000, reason="max_turn")

    assert (event.at_ms, event.reason) == (30_000, "max_turn")
    assert event.type is S.EventType.SPEECH_STOPPED  # a boundary, never an error


def test_boundary_timings_are_absent_rather_than_invented() -> None:
    # A caller with no audio-stream time to report says so; the field is
    # None, and the site renders an honestly-labelled wall-clock fallback
    # rather than passing elapsed time off as audio-stream time.
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)

    started = session.begin_speech()
    stopped = session.end_speech()

    assert started.at_ms is None
    assert (stopped.at_ms, stopped.reason) == (None, None)


def test_boundary_timings_serialize_onto_the_wire() -> None:
    import json

    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech(at_ms=128)

    payload = json.loads(
        json.dumps(S.event_to_dict(session.end_speech(at_ms=2048, reason="silence")))
    )

    assert payload["at_ms"] == 2048
    assert payload["reason"] == "silence"


def test_fail_wire_event_is_a_named_error_in_the_session_vocabulary() -> None:
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)

    event = session.fail_wire_event("invalid_json: malformed JSON: line 1 column 1")

    assert event.type is S.EventType.ERROR
    assert event.code is S.ErrorCode.INVALID_WIRE_EVENT
    assert event.code.value == "invalid_wire_event"
    # The specific wire reason survives in the text — one code covers all
    # three, exactly like invalid_session_config covers every bad config.
    assert "invalid_json" in event.message


def test_fail_wire_event_changes_no_session_state() -> None:
    # A malformed frame is not a turn boundary: it must not open or close an
    # item, and the session stays open to keep receiving (#149 contract).
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.begin_speech()

    session.fail_wire_event("invalid_append_event: 'audio' field is not valid base64")

    assert session.state is S.SessionState.SPEECH
    assert session.has_open_item is True


def test_fail_wire_event_survives_a_teardown_race() -> None:
    # Adversarial client input arriving as the session tears down must never
    # become an exception — same rule mark_vad_unavailable follows.
    config = S.parse_session_config({})
    session, _ = S.Session.create(config)
    session.teardown()

    event = session.fail_wire_event("unsupported_frame_type: binary frames are gone")

    assert event.code is S.ErrorCode.INVALID_WIRE_EVENT


def test_the_error_code_enum_is_the_whole_wire_vocabulary() -> None:
    # One enumerable list: the site renders each code distinctly and the docs
    # task documents them, so a second enum reaching ErrorEvent.code (the
    # pre-t6 WireErrorCode leak) is a schema break, not a detail.
    assert {code.value for code in S.ErrorCode} == {
        "invalid_session_config",
        "vad_unavailable",
        "invalid_wire_event",
        "stt_forward_failed",
        "generate_failed",
        "tts_failed",
        "response_timeout",
    }
