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
