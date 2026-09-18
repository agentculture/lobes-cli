"""Offline unit tests for the PURE helpers in ``scripts/realtime-he-accept.py``.

The script (hebrew-realtime t14) is a LIVE-only acceptance client — it opens
real sockets and real audio subprocesses against a deployed realtime overlay,
and has not been run against hardware as part of this task (see the script's
own module docstring and the task report). What CAN be tested offline, with
no socket, no audio hardware, and no live deployment, are the PURE decision
functions the script builds on: argument validation, the mic/speaker device
pairing check, the tool sandbox path resolution, the tool body itself, the
event-order honesty check, the latency table builder, the base64 audio
chunking, and the WAV reader.

The script lives under ``scripts/`` with a hyphenated filename, so it is
loaded here via ``importlib`` from its file path, exactly like
``tests/test_realtime_smoke_helpers.py`` loads its sibling script.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import wave
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "realtime-he-accept.py"
_spec = importlib.util.spec_from_file_location("realtime_he_accept", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
rha = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rha
_spec.loader.exec_module(rha)


# --- import isolation / no-tool-framework grep gate (criterion 3) ----------


def test_module_imports_without_a_third_party_ws_client_or_audio_lib() -> None:
    src = _SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden = ("torch", "fastapi", "numpy", "scipy", "websocket", "websockets", "aiohttp")
    offenders = [
        name
        for name in forbidden
        for line in src.splitlines()
        if line.strip().startswith((f"import {name}", f"from {name}"))
    ]
    assert not offenders, f"realtime-he-accept.py imports a forbidden dep: {offenders}"


def test_module_does_not_import_the_lobes_package() -> None:
    src = _SCRIPT_PATH.read_text(encoding="utf-8")
    offenders = [
        line for line in src.splitlines() if line.strip().startswith(("import lobes", "from lobes"))
    ]
    assert not offenders, f"realtime-he-accept.py imports the lobes package: {offenders}"


def test_no_tool_implementation_or_agent_framework_import_under_lobes() -> None:
    """Criterion 3: grep shows no tool implementation, tool schema, or
    agent-framework import under ``lobes/`` — the only tool lives in
    ``scripts/`` and test fixtures."""
    repo_root = _SCRIPT_PATH.resolve().parent.parent
    lobes_dir = repo_root / "lobes"
    agent_framework_tokens = ("langchain", "openai-agents", "openai_agents", "autogen", "crewai")
    offenders: list[str] = []
    for path in lobes_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if rha.TOOL_NAME in text:
            offenders.append(f"{path}: contains tool name {rha.TOOL_NAME!r}")
        for token in agent_framework_tokens:
            if token in text.lower():
                offenders.append(f"{path}: contains agent-framework token {token!r}")
    assert not offenders, "\n".join(offenders)


# --- device pairing ----------------------------------------------------------


def test_normalize_pipewire_device_name_strips_direction_prefix_and_profile_suffix() -> None:
    mic = "alsa_input.usb-Seeed_reSpeaker_XVF3800-00.multichannel-input"
    speaker = "alsa_output.usb-Seeed_reSpeaker_XVF3800-00.analog-stereo"
    assert rha.normalize_pipewire_device_name(mic) == rha.normalize_pipewire_device_name(speaker)


def test_normalize_pipewire_device_name_keeps_genuinely_different_devices_apart() -> None:
    reachy = "alsa_input.usb-Pollen_Robotics_Reachy_Mini-00.multichannel-input"
    respeaker = "alsa_output.usb-Seeed_reSpeaker_XVF3800-00.analog-stereo"
    assert rha.normalize_pipewire_device_name(reachy) != rha.normalize_pipewire_device_name(
        respeaker
    )


def test_normalize_pipewire_device_name_empty_string_is_empty() -> None:
    assert rha.normalize_pipewire_device_name("") == ""


def test_device_identity_alsa_is_literal() -> None:
    assert rha.device_identity("alsa", "1") == "1"
    assert rha.device_identity("alsa", "2") == "2"


def test_device_identity_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        rha.device_identity("bogus", "x")


def test_validate_device_pair_accepts_the_same_alsa_card() -> None:
    rha.validate_device_pair("alsa", "1", "1", allow_split_devices=False)  # must not raise


def test_validate_device_pair_refuses_a_mismatched_alsa_pair_by_default() -> None:
    with pytest.raises(rha.DeviceMismatchError):
        rha.validate_device_pair("alsa", "1", "2", allow_split_devices=False)


def test_validate_device_pair_allows_a_mismatch_with_the_override_flag() -> None:
    rha.validate_device_pair("alsa", "1", "2", allow_split_devices=True)  # must not raise


def test_validate_device_pair_accepts_matching_pipewire_nodes() -> None:
    mic = "alsa_input.usb-Seeed_reSpeaker-00.multichannel-input"
    speaker = "alsa_output.usb-Seeed_reSpeaker-00.analog-stereo"
    rha.validate_device_pair("pipewire", mic, speaker, allow_split_devices=False)  # no raise


def test_validate_device_pair_refuses_mismatched_pipewire_nodes() -> None:
    mic = "alsa_input.usb-Pollen_Robotics_Reachy_Mini-00.multichannel-input"
    speaker = "alsa_output.usb-Seeed_reSpeaker-00.analog-stereo"
    with pytest.raises(rha.DeviceMismatchError):
        rha.validate_device_pair("pipewire", mic, speaker, allow_split_devices=False)


def test_device_mismatch_error_message_names_both_devices() -> None:
    with pytest.raises(rha.DeviceMismatchError) as exc_info:
        rha.validate_device_pair("alsa", "1", "2", allow_split_devices=False)
    message = str(exc_info.value)
    assert "1" in message
    assert "2" in message
    assert "--allow-split-devices" in message


# --- capture/playback argv building (pure — never executed) ------------------


def test_build_capture_argv_alsa_uses_plughw_and_requested_rate() -> None:
    argv = rha.build_capture_argv("alsa", "1", rate=16000, channels=1)
    assert argv[0] == "arecord"
    assert "plughw:1,0" in argv
    assert "16000" in argv
    assert "raw" in argv


def test_build_playback_argv_alsa_uses_plughw_and_requested_rate() -> None:
    argv = rha.build_playback_argv("alsa", "1", rate=24000, channels=1)
    assert argv[0] == "aplay"
    assert "plughw:1,0" in argv
    assert "24000" in argv


def test_build_capture_argv_pipewire_targets_the_named_source() -> None:
    argv = rha.build_capture_argv("pipewire", "my-source", rate=16000, channels=1)
    assert argv[0] == "pw-record"
    assert "--target" in argv
    assert "my-source" in argv


def test_build_playback_argv_pipewire_targets_the_named_sink() -> None:
    argv = rha.build_playback_argv("pipewire", "my-sink", rate=24000, channels=1)
    assert argv[0] == "pw-play"
    assert "my-sink" in argv


def test_build_capture_argv_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        rha.build_capture_argv("bogus", "1")


def test_build_playback_argv_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        rha.build_playback_argv("bogus", "1")


# --- tool declaration ----------------------------------------------------------


def test_build_tool_schema_is_the_openai_realtime_flat_shape() -> None:
    schema = rha.build_tool_schema()
    assert schema["type"] == "function"
    assert schema["name"] == rha.TOOL_NAME == "list_directory"
    assert "description" in schema
    assert schema["parameters"]["type"] == "object"
    assert "path" in schema["parameters"]["properties"]
    assert schema["parameters"].get("required") == []


def test_build_session_update_event_declares_the_one_tool_and_auto_choice() -> None:
    event = rha.build_session_update_event()
    assert event["type"] == "session.update"
    tools = event["session"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == rha.TOOL_NAME
    assert event["session"]["tool_choice"] == "auto"


# --- tool sandbox path resolution ---------------------------------------------


def test_resolve_tool_path_accepts_a_path_inside_the_root(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    resolved = rha.resolve_tool_path(tmp_path, "sub")
    assert resolved == (tmp_path / "sub").resolve()


def test_resolve_tool_path_defaults_to_the_root_itself(tmp_path) -> None:
    assert rha.resolve_tool_path(tmp_path, "") == tmp_path.resolve()
    assert rha.resolve_tool_path(tmp_path, ".") == tmp_path.resolve()


def test_resolve_tool_path_refuses_a_dotdot_escape(tmp_path) -> None:
    with pytest.raises(rha.ToolPathError):
        rha.resolve_tool_path(tmp_path, "../../etc/passwd")


def test_resolve_tool_path_refuses_an_absolute_path_outside_the_root(tmp_path) -> None:
    with pytest.raises(rha.ToolPathError):
        rha.resolve_tool_path(tmp_path, "/etc/passwd")


def test_resolve_tool_path_refuses_a_symlink_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    root = tmp_path / "root"
    root.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported in this environment")
    with pytest.raises(rha.ToolPathError):
        rha.resolve_tool_path(root, "escape")


# --- the tool body itself -------------------------------------------------------


def test_run_list_directory_lists_sorted_entries(tmp_path) -> None:
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    output = json.loads(rha.run_list_directory(tmp_path, "."))
    assert output["entries"] == ["a.txt", "b.txt", "sub"]
    assert output["truncated"] is False


def test_run_list_directory_caps_at_50_entries(tmp_path) -> None:
    for i in range(60):
        (tmp_path / f"f{i:03d}.txt").touch()
    output = json.loads(rha.run_list_directory(tmp_path, "."))
    assert len(output["entries"]) == 50
    assert output["truncated"] is True


def test_run_list_directory_refuses_an_escape_as_an_error_string_not_an_exception(tmp_path) -> None:
    output = json.loads(rha.run_list_directory(tmp_path, "../../etc"))
    assert "error" in output


def test_run_list_directory_reports_a_missing_path_as_an_error_string(tmp_path) -> None:
    output = json.loads(rha.run_list_directory(tmp_path, "does-not-exist"))
    assert "error" in output


def test_run_list_directory_reports_a_file_path_as_not_a_directory(tmp_path) -> None:
    (tmp_path / "file.txt").write_text("x")
    output = json.loads(rha.run_list_directory(tmp_path, "file.txt"))
    assert "error" in output
    assert "directory" in output["error"]


def test_run_list_directory_never_raises_on_any_of_the_above(tmp_path) -> None:
    # A tool that raised instead of returning a JSON error string would crash
    # the whole acceptance run on one bad path from the model.
    for bad_path in ("../escape", "/etc", "nope", None):
        rha.run_list_directory(tmp_path, bad_path)  # must not raise


# --- tool-call argument parsing + wire event builders -----------------------


def test_parse_function_call_arguments_parses_a_json_object() -> None:
    assert rha.parse_function_call_arguments('{"path": "."}') == {"path": "."}


def test_parse_function_call_arguments_rejects_malformed_json() -> None:
    with pytest.raises(rha.ToolArgumentError):
        rha.parse_function_call_arguments("{not json")


def test_parse_function_call_arguments_rejects_a_non_object_json_value() -> None:
    with pytest.raises(rha.ToolArgumentError):
        rha.parse_function_call_arguments("[1, 2, 3]")


def test_build_function_call_output_event_shape() -> None:
    event = rha.build_function_call_output_event("call_123", '{"entries": []}')
    assert event["type"] == "conversation.item.create"
    assert event["item"]["type"] == "function_call_output"
    assert event["item"]["call_id"] == "call_123"
    assert event["item"]["output"] == '{"entries": []}'


def test_build_response_create_event_shape() -> None:
    assert rha.build_response_create_event() == {"type": "response.create"}


# --- event-order honesty check (criterion 2) --------------------------------


def test_check_transcript_before_tool_call_passes_when_transcript_precedes_call() -> None:
    log = [
        {"type": "session.created"},
        {"type": "conversation.item.input_audio_transcription.completed", "text": "hi"},
        {"type": "response.function_call_arguments.done", "call_id": "c1"},
    ]
    ok, detail = rha.check_transcript_before_tool_call(log)
    assert ok is True
    assert "ORDER OK" in detail


def test_check_transcript_before_tool_call_fails_when_call_precedes_transcript() -> None:
    log = [
        {"type": "response.function_call_arguments.done", "call_id": "c1"},
        {"type": "conversation.item.input_audio_transcription.completed", "text": "hi"},
    ]
    ok, detail = rha.check_transcript_before_tool_call(log)
    assert ok is False
    assert "ORDER VIOLATION" in detail


def test_check_transcript_before_tool_call_fails_when_no_call_ever_arrived() -> None:
    log = [{"type": "conversation.item.input_audio_transcription.completed", "text": "hi"}]
    ok, detail = rha.check_transcript_before_tool_call(log)
    assert ok is False
    assert "no function_call_arguments.done" in detail


def test_check_transcript_before_tool_call_fails_when_neither_arrived() -> None:
    ok, detail = rha.check_transcript_before_tool_call([{"type": "session.created"}])
    assert ok is False
    assert "neither" in detail


# --- latency table -------------------------------------------------------------


def test_build_latency_table_reports_only_present_stages() -> None:
    events = [
        {
            "type": "response.done",
            "response_id": "resp_1",
            "timings": {"stt": 120, "generate": 300, "first_delta": 410},
        }
    ]
    table = rha.build_latency_table(events)
    assert "stt" in table
    assert "120" in table
    assert "generate" in table
    assert "tool_wait" not in table  # absent stage never invented
    assert "phonikud" not in table


def test_build_latency_table_reports_multiple_responses_in_order() -> None:
    events = [
        {"type": "response.done", "response_id": "r1", "timings": {"tts": 50}},
        {"type": "response.done", "response_id": "r2", "timings": {"tts": 75}},
    ]
    table = rha.build_latency_table(events)
    assert table.index("r1") < table.index("r2")


def test_build_latency_table_is_honest_about_no_timings() -> None:
    table = rha.build_latency_table([{"type": "response.done", "response_id": "r1"}])
    assert "no response.done event carried" in table


def test_build_latency_table_empty_input() -> None:
    table = rha.build_latency_table([])
    assert "no response.done" in table


# --- connect-URL construction --------------------------------------------------


def test_build_realtime_target_carries_language_and_rate() -> None:
    scheme, host, port, path = rha.build_realtime_target("http://localhost:8000", 16000, "he")
    assert scheme == "ws"
    assert host == "localhost"
    assert port == 8000
    assert path.startswith("/v1/realtime?")
    assert "input_sample_rate=16000" in path
    assert "language=he" in path
    assert "turn_detection=server_vad" in path


def test_build_realtime_target_uses_wss_for_https() -> None:
    scheme, _host, port, _path = rha.build_realtime_target(
        "https://gateway.example.ts.net", 16000, "en"
    )
    assert scheme == "wss"
    assert port == 443


# --- base64 audio chunking (pure) -----------------------------------------------


def test_iter_append_events_wraps_every_chunk_as_input_audio_buffer_append() -> None:
    pcm = bytes(range(20))
    events = list(rha.iter_append_events(pcm, chunk_bytes=8))
    assert len(events) == 3  # 8 + 8 + 4
    assert all(e["type"] == "input_audio_buffer.append" for e in events)
    import base64

    reassembled = b"".join(base64.b64decode(e["audio"]) for e in events)
    assert reassembled == pcm


def test_iter_append_events_empty_pcm_yields_nothing() -> None:
    assert list(rha.iter_append_events(b"", chunk_bytes=8)) == []


# --- JSONL log line format -------------------------------------------------------


def test_format_log_line_is_valid_json_with_direction_ts_and_event() -> None:
    line = rha.format_log_line("send", {"type": "session.update"}, 123.456)
    parsed = json.loads(line)
    assert parsed["direction"] == "send"
    assert parsed["ts"] == 123.456
    assert parsed["event"] == {"type": "session.update"}


# --- WAV reading -----------------------------------------------------------------


def _write_wav(path: Path, *, rate: int, channels: int, sampwidth: int, frames: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(frames)


def test_read_wav_pcm16_mono_reads_matching_wav(tmp_path) -> None:
    frames = struct.pack("<4h", 1, -1, 2, -2)
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, rate=16000, channels=1, sampwidth=2, frames=frames)
    assert rha.read_wav_pcm16_mono(wav_path, expected_rate=16000) == frames


def test_read_wav_pcm16_mono_rejects_wrong_sample_rate(tmp_path) -> None:
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, rate=24000, channels=1, sampwidth=2, frames=b"\x00\x00")
    with pytest.raises(rha.WavFormatError):
        rha.read_wav_pcm16_mono(wav_path, expected_rate=16000)


def test_read_wav_pcm16_mono_rejects_stereo(tmp_path) -> None:
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, rate=16000, channels=2, sampwidth=2, frames=b"\x00\x00\x00\x00")
    with pytest.raises(rha.WavFormatError):
        rha.read_wav_pcm16_mono(wav_path, expected_rate=16000)


def test_read_wav_pcm16_mono_rejects_non_16bit(tmp_path) -> None:
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, rate=16000, channels=1, sampwidth=1, frames=b"\x00\x00")
    with pytest.raises(rha.WavFormatError):
        rha.read_wav_pcm16_mono(wav_path, expected_rate=16000)


def test_read_wav_pcm16_mono_rejects_an_unreadable_file(tmp_path) -> None:
    bogus = tmp_path / "not-a-wav.wav"
    bogus.write_bytes(b"not a wav file at all")
    with pytest.raises(rha.WavFormatError):
        rha.read_wav_pcm16_mono(bogus, expected_rate=16000)


# --- CLI argument validation ------------------------------------------------------


def _args(**overrides):
    parser = rha.build_arg_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_validate_args_accepts_the_default_alsa_card(tmp_path) -> None:
    args = _args(tool_root=str(tmp_path))
    rha.validate_args(args)  # must not raise


def test_validate_args_rejects_a_mismatched_alsa_pair(tmp_path) -> None:
    args = _args(tool_root=str(tmp_path), speaker_card=2)
    with pytest.raises(rha.DeviceMismatchError):
        rha.validate_args(args)


def test_validate_args_accepts_a_mismatched_pair_with_the_override(tmp_path) -> None:
    args = _args(tool_root=str(tmp_path), speaker_card=2, allow_split_devices=True)
    rha.validate_args(args)  # must not raise


def test_validate_args_rejects_a_nonexistent_tool_root() -> None:
    args = _args(tool_root="/definitely/does/not/exist/anywhere")
    with pytest.raises(rha.ArgsError):
        rha.validate_args(args)


def test_validate_args_rejects_pipewire_without_source_and_sink(tmp_path) -> None:
    args = _args(tool_root=str(tmp_path), backend="pipewire")
    with pytest.raises(rha.ArgsError):
        rha.validate_args(args)


def test_validate_args_accepts_pipewire_with_matching_source_and_sink(tmp_path) -> None:
    args = _args(
        tool_root=str(tmp_path),
        backend="pipewire",
        source="alsa_input.usb-Seeed-00.multichannel-input",
        sink="alsa_output.usb-Seeed-00.analog-stereo",
    )
    rha.validate_args(args)  # must not raise


def test_validate_args_requires_wav_with_script_mode(tmp_path) -> None:
    args = _args(tool_root=str(tmp_path), script_mode=True, wav=None)
    with pytest.raises(rha.ArgsError):
        rha.validate_args(args)


def test_validate_args_rejects_wav_without_script_mode(tmp_path) -> None:
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, rate=16000, channels=1, sampwidth=2, frames=b"\x00\x00")
    args = _args(tool_root=str(tmp_path), script_mode=False, wav=str(wav_path))
    with pytest.raises(rha.ArgsError):
        rha.validate_args(args)


def test_validate_args_accepts_script_mode_with_an_existing_wav(tmp_path) -> None:
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, rate=16000, channels=1, sampwidth=2, frames=b"\x00\x00")
    args = _args(tool_root=str(tmp_path), script_mode=True, wav=str(wav_path))
    rha.validate_args(args)  # must not raise


def test_validate_args_rejects_a_missing_wav_file(tmp_path) -> None:
    args = _args(tool_root=str(tmp_path), script_mode=True, wav=str(tmp_path / "nope.wav"))
    with pytest.raises(rha.ArgsError):
        rha.validate_args(args)


def test_resolved_mic_speaker_alsa_defaults_speaker_to_card() -> None:
    args = _args()
    assert rha.resolved_mic_speaker(args) == ("1", "1")


def test_resolved_mic_speaker_alsa_honours_speaker_card_override() -> None:
    args = _args(speaker_card=2)
    assert rha.resolved_mic_speaker(args) == ("1", "2")


def test_resolved_mic_speaker_pipewire_uses_source_and_sink() -> None:
    args = _args(backend="pipewire", source="src", sink="snk")
    assert rha.resolved_mic_speaker(args) == ("src", "snk")


# --- --help works and default tool root exists ------------------------------------


def test_build_arg_parser_help_does_not_raise() -> None:
    parser = rha.build_arg_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


def test_default_tool_root_is_the_repo_docs_directory() -> None:
    assert rha.DEFAULT_TOOL_ROOT.name == "docs"
    assert rha.DEFAULT_TOOL_ROOT.is_dir()


def test_exit_codes_are_all_distinct() -> None:
    codes = [
        rha.EXIT_OK,
        rha.EXIT_UNEXPECTED,
        rha.EXIT_ARGS,
        rha.EXIT_DEVICE_MISMATCH,
        rha.EXIT_TOOL_ROOT_INVALID,
        rha.EXIT_HANDSHAKE_FAILED,
        rha.EXIT_SESSION_ERROR,
        rha.EXIT_TIMEOUT,
        rha.EXIT_ORDER_VIOLATION,
        rha.EXIT_AUDIO_BACKEND_FAILED,
        rha.EXIT_MIC_EOF,
        rha.EXIT_WAV_FORMAT,
    ]
    assert len(codes) == len(set(codes))


def test_pipewire_identity_ignores_a_numeric_profile_suffix() -> None:
    """Found live 2026-09-18: this box names the reSpeaker source '...analog-stereo.3'."""
    src = "alsa_input.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_1149-00.analog-stereo.3"
    snk = "alsa_output.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_1149-00.analog-stereo"
    assert rha.normalize_pipewire_device_name(src) == rha.normalize_pipewire_device_name(snk)


def test_select_channel_takes_one_channel_and_never_averages() -> None:
    stereo = struct.pack("<8h", 1, 100, 2, 200, 3, 300, 4, 400)
    assert struct.unpack("<4h", rha.select_channel(stereo, 2, 0)) == (1, 2, 3, 4)
    assert struct.unpack("<4h", rha.select_channel(stereo, 2, 1)) == (100, 200, 300, 400)
    assert rha.select_channel(stereo, 1, 0) == stereo
    assert len(rha.select_channel(stereo + b"\x01", 2, 1)) == 8  # a ragged tail is dropped


# --- playback drain (2026-09-18: operator heard a streamed reply cut short) ---


class _FakePlayer:
    def __init__(self, exits_after_waits: int | None):
        self.stdin = self
        self.closed = False
        self.terminated = False
        self.waits: list[float] = []
        self._exits_after = exits_after_waits
        self._done = False

    def close(self):
        self.closed = True

    def poll(self):
        return 0 if self._done else None

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self._done:
            return 0
        if self._exits_after is not None and len(self.waits) >= self._exits_after:
            self._done = True
            return 0
        raise rha.subprocess.TimeoutExpired("player", timeout)

    def terminate(self):
        self.terminated = True
        self._done = True

    def kill(self):
        self._done = True


def test_drain_lets_the_player_finish_buffered_audio_before_any_terminate():
    player = _FakePlayer(exits_after_waits=1)
    rha._drain_playback(player, timeout=20.0)
    assert player.closed
    assert player.waits == [20.0]
    assert not player.terminated


def test_drain_terminates_a_player_that_never_exits():
    player = _FakePlayer(exits_after_waits=None)
    rha._drain_playback(player, timeout=0.01)
    assert player.closed and player.terminated
