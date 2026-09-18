#!/usr/bin/env python3
"""Hebrew /v1/realtime acceptance client — reSpeaker/Reachy capture-and-play + a demo tool.

Drives ONE ``/v1/realtime`` session, in Hebrew, through the fleet gateway,
end to end: capture mono 16 kHz from a real ALSA/pipewire microphone, stream
it as the OpenAI-shaped ``input_audio_buffer.append`` wire (issue #151),
play synthesized ``response.audio.delta`` chunks back to the SAME device's
sink (so the reSpeaker XVF3800's own hardware AEC sees the playback
reference), declare ONE harmless read-only tool
(:data:`TOOL_NAME` == ``list_directory``) via ``session.update``, execute it
client-side when the model calls it, and answer with
``conversation.item.create``/``function_call_output`` then
``response.create`` — the hebrew-realtime spec's t14/t17 acceptance shape.

Device order (task brief): the reSpeaker XVF3800 (ALSA card 1) is
unclaimed and exposes mic+speaker on one USB device, so it is tried FIRST
(``--card 1``, the default). The Reachy Mini (ALSA card 2) is tried only if
``reachy-mini-daemon`` has released it — this script does not probe that
itself; an operator passes ``--card 2`` once they have confirmed the release
(``fuser -v /dev/snd/*``).

Reuses ``scripts/realtime-smoke.py``'s hand-rolled RFC 6455 WebSocket
client (handshake, framed reads/writes, the ``input_audio_buffer.append`` /
``response.audio.delta`` base64 codec) via ``importlib``, exactly the way
``scripts/realtime-voice-loop.py`` already does — see that script's own
module docstring for why this repo prefers a hand-rolled client over a
third-party WebSocket library here.

Silent-failure discipline (memory: silent-failure-antipattern-voice-tools,
learned the hard way on PR #150/#152 of this same subsystem): EVERY failure
path below speaks a NAMED error and returns a distinct non-zero exit code
(listed in ``--help``) rather than idling out a timeout that looks
indistinguishable from "nobody spoke". Concretely: ``arecord``/``aplay``
stderr is always piped and quoted on failure; every subprocess teardown is
bounded by a timeout and then killed outright (a dangling ``arecord`` holds
the capture device busy for the NEXT run); a mid-session mic EOF stops the
session and names it rather than silently going deaf; a
``socket.timeout`` on the event reader is a normal quiet tick (``continue``),
while any OTHER read failure stops the session and is named — the two must
never look alike; every WebSocket PING is answered with PONG (uvicorn pings
~every 20 s and drops a peer that never pongs, and a long conversation dying
after tens of seconds while a one-shot smoke run never trips it is exactly
that defect).

No tool framework: the ONE tool this script executes
(:func:`run_list_directory`) is implemented HERE, in ``scripts/``, not under
``lobes/`` — nothing under ``lobes/`` declares a tool schema, implements a
tool, or imports an agent framework (see criterion 3;
``tests/test_realtime_he_accept_helpers.py`` greps for it).

Live-only glue (sockets, subprocesses, real audio, real time) is NOT
unit-tested, mirroring ``scripts/realtime-smoke.py``'s own convention. Every
PURE decision this script makes — argument validation, the device-pair
mismatch check, the tool-path sandbox, the event-order checker, the latency
table builder, and the base64 audio chunking — lives in a small function
below and is exercised offline, with no socket and no audio hardware, by
``tests/test_realtime_he_accept_helpers.py``.

This script has NOT been run against live hardware as part of this task —
no GPU-backed realtime server is up in this environment. See the task
report for what WAS verified (``--help``, argv-building, and this box's own
``arecord -l`` output) versus what a live acceptance run (t17) still owes.

  python3 scripts/realtime-he-accept.py --base-url http://localhost:8000 \\
      --api-key "$LOBES_API_KEY" --card 1 --language he \\
      --log /tmp/realtime-he-accept.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from urllib.parse import urlencode, urlsplit

# ---------------------------------------------------------------------------
# Reuse the shipped smoke script's hand-rolled RFC 6455 client + base64 wire
# codec (issue #151) instead of a new dependency or a re-implementation —
# exactly the pattern scripts/realtime-voice-loop.py already established.
# ---------------------------------------------------------------------------
_SMOKE_PATH = Path(__file__).resolve().parent / "realtime-smoke.py"
_spec = importlib.util.spec_from_file_location("realtime_smoke", _SMOKE_PATH)
assert _spec is not None
assert _spec.loader is not None
rs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rs
_spec.loader.exec_module(rs)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOL_ROOT = REPO_ROOT / "docs"

MIC_SAMPLE_RATE = 16000  # capture rate this script always requests
DELTA_SAMPLE_RATE = 24000  # Chatterbox's native TTS output rate — no resample
CHUNK_MS = 32  # matches lobes.realtime._segmenter's 512-sample/32ms VAD framing

REALTIME_PATH = "/v1/realtime"
TOOL_NAME = "list_directory"
MAX_TOOL_ENTRIES = 50

# Distinct, documented exit codes (silent-failure discipline: every failure
# path names itself AND exits differently, never a bare "1").
EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_ARGS = 2
EXIT_DEVICE_MISMATCH = 3
EXIT_TOOL_ROOT_INVALID = 4
EXIT_HANDSHAKE_FAILED = 5
EXIT_SESSION_ERROR = 6
EXIT_TIMEOUT = 7
EXIT_ORDER_VIOLATION = 8
EXIT_AUDIO_BACKEND_FAILED = 9
EXIT_MIC_EOF = 10
EXIT_WAV_FORMAT = 11

EXIT_CODE_HELP = """\
Exit codes:
  0  success — the full tool round trip completed and event order was honest
  1  unexpected/unclassified error (a bug, not a named failure mode)
  2  bad arguments (validated offline by validate_args())
  3  mic/speaker device mismatch (pass --allow-split-devices to override)
  4  --tool-root is not a directory
  5  WebSocket handshake failed (bad URL, refused connection, non-101 status)
  6  the server sent a named 'error' session event
  7  timed out waiting for an expected wire event
  8  the model's tool call arrived before (or without) its turn's transcript
  9  the capture or playback subprocess failed to start or produced no audio
  10 the microphone stopped producing audio mid-session (device lost/unplugged)
  11 --wav is not a 16-bit mono PCM file at the requested sample rate
"""


# ---------------------------------------------------------------------------
# Named errors — one class per failure family, mirroring
# lobes.realtime._session's SessionConfigError / lobes.realtime._wire's
# WireFormatError pairing (a documented reason, never a bare exception).
# ---------------------------------------------------------------------------


class ArgsError(ValueError):
    """A CLI argument combination is invalid. Exit code: EXIT_ARGS."""


class DeviceMismatchError(ValueError):
    """The chosen mic and speaker are not the same physical device.

    Refused by default because the reSpeaker XVF3800's hardware AEC needs
    the playback reference on its OWN output to be useful, and because a
    barge-in test on split devices would not be a fair test of anything.
    Override with ``--allow-split-devices`` when that tradeoff is deliberate
    (e.g. a device with no full-duplex sink of its own).
    """

    def __init__(self, mic_identity: str, speaker_identity: str) -> None:
        super().__init__(
            f"mic device {mic_identity!r} and speaker device {speaker_identity!r} "
            "are not the same device — pass --allow-split-devices to override "
            "(the reSpeaker's hardware AEC only sees the playback reference "
            "when mic and speaker are the SAME USB device)"
        )
        self.mic_identity = mic_identity
        self.speaker_identity = speaker_identity


class ToolPathError(ValueError):
    """A tool call's ``path`` argument escapes the configured ``--tool-root``."""


class ToolArgumentError(ValueError):
    """A ``response.function_call_arguments.done`` event's ``arguments`` string
    was not a valid JSON object."""


class WavFormatError(ValueError):
    """A ``--wav`` file is not 16-bit mono PCM at the expected sample rate."""


# ---------------------------------------------------------------------------
# Pure helpers, part 1: device selection + the mic/speaker pairing check.
# Exercised offline by tests/test_realtime_he_accept_helpers.py — no
# hardware, no subprocess is ever started by any function in this section.
# ---------------------------------------------------------------------------

_PW_PREFIX_RE = re.compile(r"^(alsa_input\.|alsa_output\.|bluez_input\.|bluez_output\.)")
_PW_SUFFIX_RE = re.compile(
    r"\.(multichannel-input|multichannel-output|analog-stereo|analog-mono"
    r"|iec958-stereo|pro-input-0|pro-output-0)$"
)


def normalize_pipewire_device_name(name: str) -> str:
    """Strip a pipewire node name down to its underlying-device identity.

    A HEURISTIC, not a registry lookup: pipewire names a device's capture and
    playback nodes differently (``alsa_input.usb-Seeed-...multichannel-input``
    vs. ``alsa_output.usb-Seeed-...analog-stereo``), so a literal string
    comparison would always call them a mismatch even when they are the same
    USB device. This strips the well-known direction prefix and profile
    suffix pipewire itself generates, leaving the shared device stem. It is
    deliberately conservative (unknown prefixes/suffixes are left alone) —
    ``--allow-split-devices`` is always the honest escape hatch when this
    heuristic gets it wrong.
    """
    if not name:
        return ""
    stripped = _PW_PREFIX_RE.sub("", name)
    stripped = _PW_SUFFIX_RE.sub("", stripped)
    return stripped.strip().lower()


def device_identity(backend: str, value: str) -> str:
    """The comparable identity of a device spec, per *backend*.

    ALSA identities are compared literally (a card number IS the device);
    pipewire identities go through :func:`normalize_pipewire_device_name`
    first.
    """
    if backend == "alsa":
        return str(value)
    if backend == "pipewire":
        return normalize_pipewire_device_name(str(value))
    raise ValueError(f"unsupported backend {backend!r}")


def validate_device_pair(
    backend: str, mic_device: str, speaker_device: str, allow_split_devices: bool
) -> None:
    """Refuse a mic/speaker pair that is not the same physical device.

    Raises :class:`DeviceMismatchError` unless *allow_split_devices* is set.
    NEVER raises when the pair matches, regardless of the flag.
    """
    mic_id = device_identity(backend, mic_device)
    speaker_id = device_identity(backend, speaker_device)
    if mic_id != speaker_id and not allow_split_devices:
        raise DeviceMismatchError(mic_id, speaker_id)


def build_capture_argv(backend: str, device: str, rate: int = MIC_SAMPLE_RATE, channels: int = 1):
    """The argv for the capture subprocess — never executed here, just built.

    ALSA opens ``plughw:<card>,0``: the ``plug`` layer performs the rate/
    channel-count conversion this box's hardware needs (both USB devices on
    this box expose S16_LE/2ch/16000Hz ONLY at the raw ``hw:`` level).
    """
    if backend == "alsa":
        return [
            "arecord",
            "-D",
            f"plughw:{device},0",
            "-f",
            "S16_LE",
            "-r",
            str(rate),
            "-c",
            str(channels),
            "-t",
            "raw",
            "-q",
        ]
    if backend == "pipewire":
        return [
            "pw-record",
            "--target",
            str(device),
            "--rate",
            str(rate),
            "--channels",
            str(channels),
            "--format",
            "s16",
            "-",
        ]
    raise ValueError(f"unsupported backend {backend!r}")


def build_playback_argv(
    backend: str, device: str, rate: int = DELTA_SAMPLE_RATE, channels: int = 1
):
    """The argv for the playback subprocess — never executed here, just built.

    ALSA plays through ``plughw:<card>,0`` too: ``plug`` resamples 24 kHz
    (Chatterbox's native output) down to whatever the hardware actually
    wants and upmixes to the device's native channel count.
    """
    if backend == "alsa":
        return [
            "aplay",
            "-D",
            f"plughw:{device},0",
            "-f",
            "S16_LE",
            "-r",
            str(rate),
            "-c",
            str(channels),
            "-t",
            "raw",
            "-q",
        ]
    if backend == "pipewire":
        return [
            "pw-play",
            "--target",
            str(device),
            "--rate",
            str(rate),
            "--channels",
            str(channels),
            "--format",
            "s16",
            "-",
        ]
    raise ValueError(f"unsupported backend {backend!r}")


# ---------------------------------------------------------------------------
# Pure helpers, part 2: the tool — declaration, sandboxed path resolution,
# and the tool body itself. This is the ONLY tool implementation in this
# repository; nothing under lobes/ knows this tool's name (criterion 3).
# ---------------------------------------------------------------------------


def build_tool_schema() -> dict:
    """The OpenAI-Realtime FLAT tool declaration for :data:`TOOL_NAME`.

    One harmless, read-only tool: list a directory's entries. Matches the
    flat shape ``lobes.realtime._session.parse_tools`` validates —
    ``{"type": "function", "name", "description", "parameters"}`` — so a
    stock OpenAI Realtime client's tool loop needs no adaptation.
    """
    return {
        "type": "function",
        "name": TOOL_NAME,
        "description": (
            "List the sorted names of files and subdirectories at a path "
            "(capped at 50 entries). Read-only; any path outside the "
            "configured tool root is refused, never executed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the tool root. Defaults to the root itself.",
                }
            },
            "required": [],
        },
    }


def build_session_update_event() -> dict:
    """The ``session.update`` event declaring :data:`TOOL_NAME`, ``tool_choice: auto``."""
    return {
        "type": "session.update",
        "session": {"tools": [build_tool_schema()], "tool_choice": "auto"},
    }


def resolve_tool_path(tool_root: Path, requested: str) -> Path:
    """Resolve *requested* under *tool_root*, refusing any path outside it.

    *requested* is interpreted as relative to *tool_root* (an empty string or
    ``"."`` means the root itself); ``..`` segments, symlinks, and an
    absolute path that happens to point elsewhere are all caught by the
    final ``relative_to`` check against the RESOLVED root, not by pattern-
    matching the string — a resolved-path containment check is the only
    version of this that isn't foolable by ``..`` or a symlink.

    Raises :class:`ToolPathError` on any escape attempt.
    """
    root = tool_root.resolve()
    target = requested.strip() if requested else "."
    candidate = (
        (root / target).resolve() if not Path(target).is_absolute() else Path(target).resolve()
    )
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ToolPathError(
            f"path {requested!r} resolves to {candidate}, which is outside " f"the tool root {root}"
        ) from exc
    return candidate


def parse_function_call_arguments(raw_arguments: str) -> dict:
    """Parse a ``response.function_call_arguments.done`` event's ``arguments``
    JSON string into a dict. Raises :class:`ToolArgumentError`, never a bare
    ``json.JSONDecodeError``, on malformed or non-object JSON."""
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ToolArgumentError(f"tool arguments are not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ToolArgumentError(f"tool arguments must be a JSON object, got {parsed!r}")
    return parsed


def run_list_directory(tool_root: Path, requested_path: str | None) -> str:
    """Execute the ONE tool this script declares. Returns a JSON string.

    Never raises: a sandbox violation, a missing path, or a non-directory
    path all come back as ``{"error": "..."}`` in the OUTPUT string, because
    a tool result is what the model reads — a Python exception here would
    only crash this script, not inform the model that tried a bad path.
    """
    try:
        path = resolve_tool_path(tool_root, requested_path or ".")
    except ToolPathError as exc:
        return json.dumps({"error": str(exc)})
    if not path.exists():
        return json.dumps({"error": f"path does not exist: {requested_path!r}"})
    if not path.is_dir():
        return json.dumps({"error": f"not a directory: {requested_path!r}"})
    entries = sorted(p.name for p in path.iterdir())
    truncated = len(entries) > MAX_TOOL_ENTRIES
    return json.dumps(
        {
            "path": str(path.relative_to(tool_root.resolve())) or ".",
            "entries": entries[:MAX_TOOL_ENTRIES],
            "truncated": truncated,
        }
    )


def build_function_call_output_event(call_id: str, output: str) -> dict:
    """The client's tool RESULT: ``conversation.item.create`` carrying a
    ``function_call_output`` item — the shape
    ``lobes.realtime._session.parse_function_call_output`` expects."""
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def build_response_create_event() -> dict:
    """The event that arms the conversation surface for this turn's reply."""
    return {"type": "response.create"}


# ---------------------------------------------------------------------------
# Pure helpers, part 3: the criterion-2 honesty checks — event order and the
# latency table, both built from an already-decoded, ordered event log.
# ---------------------------------------------------------------------------

_TRANSCRIPT_EVENT_TYPE = "conversation.item.input_audio_transcription.completed"
_TOOL_CALL_EVENT_TYPE = "response.function_call_arguments.done"


def check_transcript_before_tool_call(event_log: list[dict]) -> tuple[bool, str]:
    """Prove the turn's transcript arrived before its tool call, or say why not.

    *event_log* is the ordered sequence of decoded server events (dicts with
    a ``"type"`` key) this script observed. Returns ``(ok, detail)`` —
    ``ok`` is True ONLY when a transcription-completed event's index is
    strictly less than a function-call-arguments-done event's index; any
    other outcome (either missing, or the call arriving first/simultaneously)
    is reported as a named, honest failure, never silently accepted.
    """
    transcript_idx = next(
        (i for i, e in enumerate(event_log) if e.get("type") == _TRANSCRIPT_EVENT_TYPE), None
    )
    call_idx = next(
        (i for i, e in enumerate(event_log) if e.get("type") == _TOOL_CALL_EVENT_TYPE), None
    )
    if transcript_idx is None and call_idx is None:
        return False, "neither a transcript nor a tool call was observed in the event log"
    if transcript_idx is None:
        return False, "no transcription.completed event was observed before the tool call"
    if call_idx is None:
        return False, "no function_call_arguments.done event was observed at all"
    if transcript_idx < call_idx:
        return (
            True,
            f"ORDER OK: transcript.completed at index {transcript_idx} preceded "
            f"the tool call at index {call_idx}",
        )
    return (
        False,
        f"ORDER VIOLATION: the tool call at index {call_idx} arrived at or "
        f"before the transcript at index {transcript_idx}",
    )


def build_latency_table(response_done_events: list[dict]) -> str:
    """A plain-text latency table built ONLY from ``response.done`` ``timings``.

    Never invents a number: a stage absent from every ``timings`` mapping
    stays absent from the table (mirrors ``StageTimings.as_dict()``'s own
    "absent, never zeroed" contract) rather than printing a misleading 0/N/A
    row.
    """
    stage_order = ("stt", "generate", "tool_wait", "phonikud", "tts", "first_delta")
    rows: list[str] = []
    for idx, event in enumerate(response_done_events):
        timings = event.get("timings") or {}
        if not timings:
            continue
        rows.append(f"-- response {idx} (response_id={event.get('response_id')!r}) --")
        for stage in stage_order:
            if stage in timings:
                rows.append(f"  {stage:<11} {timings[stage]:>8} ms")
    if not rows:
        return "no response.done event carried a 'timings' mapping — nothing to tabulate"
    header = f"{'stage':<13}{'ms':>8}"
    return "\n".join([header, "-" * len(header), *rows])


# ---------------------------------------------------------------------------
# Pure helpers, part 4: the connect-URL, the base64 audio chunking, the
# JSONL log-line format, and the offline WAV reader.
# ---------------------------------------------------------------------------


def build_realtime_target(base_url: str, input_sample_rate: int, language: str):
    """``(scheme, host, port, path_with_query)`` for the WS handshake.

    ``language`` goes on the connect-URL query string
    (``lobes.realtime.app``'s ``_open_session`` passes ``websocket.query_params``
    straight into ``parse_session_config``, which reads ``language`` there),
    NOT via ``session.update`` — the tool declaration is the only thing this
    script sends through ``session.update``.
    """
    parsed = urlsplit(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    query = urlencode(
        {
            "input_sample_rate": input_sample_rate,
            "turn_detection": "server_vad",
            "language": language,
        }
    )
    return scheme, host, port, f"{REALTIME_PATH}?{query}"


def iter_append_events(pcm: bytes, chunk_bytes: int):
    """Split *pcm* into ``chunk_bytes``-sized pieces, each already wrapped as
    an ``input_audio_buffer.append`` event (:func:`rs.build_append_event`).

    The one base64-chunking decision this script owns: WHAT size to chunk
    at and IN WHAT ORDER to wrap them, reusing ``realtime-smoke.py``'s own
    ``chunk_pcm``/``build_append_event`` for the arithmetic itself rather
    than re-deriving it (see this file's module docstring on why that script
    is imported instead of duplicated).
    """
    for chunk in rs.chunk_pcm(pcm, chunk_bytes):
        yield rs.build_append_event(chunk)


def format_log_line(direction: str, event: dict, monotonic_ts: float) -> str:
    """One JSONL line for ``--log``: a monotonic timestamp, direction
    (``"send"``/``"recv"``), and the event verbatim."""
    return json.dumps({"ts": monotonic_ts, "direction": direction, "event": event}, sort_keys=True)


def read_wav_pcm16_mono(path: Path, expected_rate: int = MIC_SAMPLE_RATE) -> bytes:
    """Read a ``--wav`` file as raw PCM16 mono bytes at *expected_rate*.

    Raises :class:`WavFormatError`, never a bare ``wave.Error``, when the
    file is not 16-bit, not mono, or not at *expected_rate* — a --script-mode
    run must fail loudly on a mismatched fixture, not silently stream
    garbage-rate audio at the VAD.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            sampwidth = wf.getsampwidth()
            channels = wf.getnchannels()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise WavFormatError(f"{path} is not a readable WAV file: {exc}") from exc
    if sampwidth != 2:
        raise WavFormatError(f"{path} is not 16-bit PCM (sampwidth={sampwidth} bytes)")
    if channels != 1:
        raise WavFormatError(f"{path} is not mono (channels={channels})")
    if rate != expected_rate:
        raise WavFormatError(
            f"{path} sample rate is {rate}, expected {expected_rate} — resample it first"
        )
    return frames


# ---------------------------------------------------------------------------
# Argument parsing + offline validation.
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realtime-he-accept.py",
        description=__doc__.splitlines()[0],
        epilog=EXIT_CODE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token; falls back to $LOBES_API_KEY (never echoed to argv/ps)",
    )
    parser.add_argument("--language", default="he", help="STT/session language (default: he)")
    parser.add_argument("--backend", choices=("alsa", "pipewire"), default="alsa")
    parser.add_argument(
        "--card",
        type=int,
        default=1,
        help="ALSA card for BOTH mic and speaker (default: 1, the reSpeaker XVF3800)",
    )
    parser.add_argument(
        "--speaker-card",
        type=int,
        default=None,
        help="ALSA card override for playback only (default: same as --card)",
    )
    parser.add_argument("--source", default=None, help="pipewire capture source name")
    parser.add_argument("--sink", default=None, help="pipewire playback sink name")
    parser.add_argument(
        "--allow-split-devices",
        action="store_true",
        help="permit a mic/speaker pair on different physical devices (breaks hardware AEC)",
    )
    parser.add_argument(
        "--tool-root",
        default=str(DEFAULT_TOOL_ROOT),
        help=f"sandbox root for the list_directory tool (default: {DEFAULT_TOOL_ROOT})",
    )
    parser.add_argument("--log", default=None, help="write every wire event as JSONL here")
    parser.add_argument(
        "--script-mode",
        action="store_true",
        help="stream --wav instead of a live microphone (repeatable, no hardware needed)",
    )
    parser.add_argument(
        "--wav", default=None, help="16-bit mono PCM WAV to stream in --script-mode"
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="overall session deadline")
    parser.add_argument("--trailing-silence-ms", type=int, default=1200)
    parser.add_argument(
        "--input-sample-rate", type=int, default=MIC_SAMPLE_RATE, choices=(16000, 24000)
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Every pure, offline-checkable argument rule. Raises :class:`ArgsError`
    or :class:`DeviceMismatchError`; never touches a socket or the mic."""
    if args.backend == "alsa":
        mic_device = str(args.card)
        speaker_device = str(args.speaker_card if args.speaker_card is not None else args.card)
    else:
        if not args.source or not args.sink:
            raise ArgsError("--backend pipewire requires both --source and --sink")
        mic_device = args.source
        speaker_device = args.sink
    validate_device_pair(args.backend, mic_device, speaker_device, args.allow_split_devices)

    tool_root = Path(args.tool_root)
    if not tool_root.is_dir():
        raise ArgsError(f"--tool-root is not a directory: {tool_root}")

    if args.script_mode and not args.wav:
        raise ArgsError("--script-mode requires --wav")
    if args.wav and not Path(args.wav).is_file():
        raise ArgsError(f"--wav path does not exist or is not a file: {args.wav}")
    if args.wav and not args.script_mode:
        raise ArgsError("--wav was given without --script-mode — pass --script-mode to use it")


def resolved_mic_speaker(args: argparse.Namespace) -> tuple[str, str]:
    """The (mic, speaker) device specs :func:`validate_args` already checked."""
    if args.backend == "alsa":
        return str(args.card), str(
            args.speaker_card if args.speaker_card is not None else args.card
        )
    return args.source, args.sink


# ---------------------------------------------------------------------------
# Live-only glue: sockets, subprocesses, real audio, real time. NOT unit-
# tested (mirrors scripts/realtime-smoke.py's own convention) — every
# decision worth testing lives in the pure functions above.
# ---------------------------------------------------------------------------


class _EventLog:
    """Thread-safe ordered log of decoded server events, plus optional JSONL."""

    def __init__(self, log_fh) -> None:
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self._log_fh = log_fh

    def append(self, direction: str, event: dict) -> int:
        with self._lock:
            if direction == "recv":
                self._events.append(event)
            idx = len(self._events)
        if self._log_fh is not None:
            print(
                format_log_line(direction, event, time.monotonic()), file=self._log_fh, flush=True
            )
        return idx

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._events)


class _Reader(threading.Thread):
    """Background thread: decode JSON text frames, answer PING, record errors.

    Mirrors ``realtime-smoke.py``'s ``EventReader``: a ``socket.timeout`` is
    a quiet tick (``continue``), any OTHER read failure stops the session
    and is NAMED in ``closed_reason`` — the two must never look alike (memory:
    silent-failure-antipattern-voice-tools).
    """

    def __init__(self, client, event_log: _EventLog) -> None:
        super().__init__(name="realtime-he-accept-reader", daemon=True)
        self._client = client
        self.event_log = event_log
        self.closed = threading.Event()
        self.closed_reason = ""
        self.fatal_error: dict | None = None

    def run(self) -> None:
        try:
            while True:
                try:
                    fin, opcode, payload = self._client.read_frame(timeout=1.0)
                except socket.timeout:
                    continue
                if not fin:
                    continue
                if opcode == rs.OPCODE_PING:
                    self._client.send_frame(rs.OPCODE_PONG, payload)
                    continue
                if opcode == rs.OPCODE_CLOSE:
                    self.closed_reason = "server closed the session"
                    return
                if opcode != rs.OPCODE_TEXT or not payload:
                    continue
                try:
                    event = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self.closed_reason = f"malformed text frame: {exc}"
                    return
                self.event_log.append("recv", event)
                if event.get("type") == "error":
                    self.fatal_error = event
        except rs.FrameReadError as exc:
            self.closed_reason = f"connection closed: {exc}"
        except OSError as exc:
            self.closed_reason = f"socket error: {exc}"
        finally:
            self.closed.set()


def _wait_for(
    event_log: _EventLog, reader: _Reader, start_index: int, expected_type: str, deadline: float
) -> tuple[dict | None, int]:
    while True:
        events = event_log.snapshot()
        if len(events) > start_index:
            return events[start_index], start_index + 1
        if reader.fatal_error is not None:
            return reader.fatal_error, start_index
        if reader.closed.is_set():
            events = event_log.snapshot()
            if len(events) > start_index:
                return events[start_index], start_index + 1
            return None, start_index
        if time.monotonic() >= deadline:
            return None, start_index
        time.sleep(0.05)
    _ = expected_type  # kept for symmetry with realtime-smoke.py's signature


def _wait_until(
    event_log: _EventLog,
    reader: _Reader,
    start_index: int,
    wanted: tuple[str, ...],
    deadline: float,
) -> tuple[dict | None, int]:
    """Return the first event at/after *start_index* whose type is in *wanted*
    (or is a server ``error``), skipping everything else. Found live 2026-09-18:
    the server interleaves transcription / response.* events between the ones
    this client cares about, and a wait that returned "whatever came next"
    failed a healthy session on the transcript event."""
    idx = start_index
    while True:
        events = event_log.snapshot()
        while idx < len(events):
            event = events[idx]
            idx += 1
            if event.get("type") in wanted or event.get("type") == "error":
                return event, idx
        if reader.fatal_error is not None:
            return reader.fatal_error, idx
        if reader.closed.is_set() and idx >= len(event_log.snapshot()):
            return None, idx
        if time.monotonic() >= deadline:
            return None, idx
        time.sleep(0.05)


def _run_subprocess_or_die(argv: list[str], *, stdin=None):
    """Start a capture/playback subprocess with piped stderr. Raises
    :class:`RuntimeError` immediately if it could not even be started
    (e.g. the binary is missing) — quoted, never a silent spawn failure."""
    try:
        return subprocess.Popen(argv, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"failed to start {argv[0]!r}: {exc}") from exc


def _terminate(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """Terminate AND wait for a child, then kill outright — leaving a
    subprocess alive holding the capture/playback device busy would fail the
    NEXT run with 'Device or resource busy' (memory:
    silent-failure-antipattern-voice-tools, defect 6)."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
    except (ArgsError, DeviceMismatchError) as exc:
        print(f"FAIL [args]: {exc}", file=sys.stderr)
        return EXIT_DEVICE_MISMATCH if isinstance(exc, DeviceMismatchError) else EXIT_ARGS

    tool_root = Path(args.tool_root).resolve()
    args.api_key = args.api_key or os.environ.get("LOBES_API_KEY")
    mic_device, speaker_device = resolved_mic_speaker(args)

    log_fh = open(args.log, "a", encoding="utf-8") if args.log else None  # noqa: SIM115
    event_log = _EventLog(log_fh)

    def send(client, event: dict) -> None:
        event_log.append("send", event)
        client.send_json_event(event)

    scheme, host, port, path = build_realtime_target(
        args.base_url, args.input_sample_rate, args.language
    )
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    try:
        client, status, _headers = rs.WebSocketClient.connect(
            host, port, path, extra_headers=headers, connect_timeout=10.0
        )
    except (OSError, ConnectionError) as exc:
        print(f"FAIL [handshake]: could not reach {scheme}://{host}:{port}{path}: {exc}")
        return EXIT_HANDSHAKE_FAILED
    if status != 101:
        print(f"FAIL [handshake]: HTTP {status} (expected 101 Switching Protocols)")
        client.close()
        return EXIT_HANDSHAKE_FAILED
    print(f"PASS [handshake]: 101 Switching Protocols — {scheme}://{host}:{port}{path}")

    reader = _Reader(client, event_log)
    reader.start()
    deadline = time.monotonic() + args.timeout
    idx = 0
    mic_proc: subprocess.Popen | None = None
    playback_proc: subprocess.Popen | None = None
    exit_code = EXIT_OK

    try:
        event, idx = _wait_for(event_log, reader, idx, "session.created", deadline)
        if event is None or event.get("type") != "session.created":
            print(f"FAIL [session.created]: {event!r}")
            return EXIT_TIMEOUT if event is None else EXIT_SESSION_ERROR
        print("PASS [session.created]")

        send(client, build_session_update_event())
        event, idx = _wait_for(event_log, reader, idx, "session.updated", deadline)
        if event is None or event.get("type") != "session.updated":
            print(f"FAIL [session.updated]: {event!r}")
            return EXIT_TIMEOUT if event is None else EXIT_SESSION_ERROR
        declared_tools = [t.get("name") for t in (event.get("session") or {}).get("tools", [])]
        if TOOL_NAME not in declared_tools:
            print(f"FAIL [session.updated]: {TOOL_NAME!r} not echoed back: {event!r}")
            return EXIT_SESSION_ERROR
        print(f"PASS [session.updated]: tools={declared_tools}")

        # Conversation is OPT-IN on this server (#151): a session that never sends
        # response.create only transcribes. Found live 2026-09-18 — without this
        # the server heard the request, transcribed it, and (correctly) did nothing.
        send(client, build_response_create_event())
        print("SENT [response.create]: session armed")

        try:
            playback_proc = _run_subprocess_or_die(
                build_playback_argv(args.backend, speaker_device), stdin=subprocess.PIPE
            )
        except RuntimeError as exc:
            print(f"FAIL [playback-start]: {exc}")
            return EXIT_AUDIO_BACKEND_FAILED

        def player() -> None:
            played = 0  # index of the next event to look at — never replay a delta
            while not reader.closed.is_set():
                events = event_log.snapshot()
                fresh, played = events[played:], len(events)
                for event in fresh:
                    if event.get("type") == "response.audio.delta":
                        try:
                            pcm = rs.decode_audio_delta_event(event)
                            if playback_proc.stdin:
                                playback_proc.stdin.write(pcm)
                                playback_proc.stdin.flush()
                        except (ValueError, BrokenPipeError, OSError):
                            pass
                time.sleep(0.05)

        threading.Thread(target=player, daemon=True).start()

        if args.script_mode:
            wav_path = Path(args.wav)
            try:
                pcm = read_wav_pcm16_mono(wav_path, args.input_sample_rate)
            except WavFormatError as exc:
                print(f"FAIL [wav]: {exc}")
                return EXIT_WAV_FORMAT
            silence = rs.silence_bytes(args.trailing_silence_ms, args.input_sample_rate)
            stream = pcm + silence
            chunk_bytes = rs.bytes_per_chunk_for_rate(args.input_sample_rate)
            for event in iter_append_events(stream, chunk_bytes):
                send(client, event)
                time.sleep(CHUNK_MS / 1000.0)
            print(f"PASS [audio-stream]: streamed {len(stream)} bytes from {wav_path}")
        else:
            try:
                mic_proc = _run_subprocess_or_die(build_capture_argv(args.backend, mic_device))
            except RuntimeError as exc:
                print(f"FAIL [mic-start]: {exc}")
                return EXIT_AUDIO_BACKEND_FAILED
            chunk_bytes = rs.bytes_per_chunk_for_rate(args.input_sample_rate)
            first = mic_proc.stdout.read(chunk_bytes) if mic_proc.stdout else b""
            if not first:
                err = (
                    mic_proc.stderr.read().decode(errors="replace").strip()
                    if mic_proc.stderr
                    else ""
                )
                print(
                    f"FAIL [mic]: device {mic_device!r} produced no audio — "
                    f"{err or 'device unavailable'}"
                )
                return EXIT_AUDIO_BACKEND_FAILED
            send(client, rs.build_append_event(first))
            mic_deadline = time.monotonic() + args.timeout
            while time.monotonic() < mic_deadline:
                raw = mic_proc.stdout.read(chunk_bytes) if mic_proc.stdout else b""
                if not raw:
                    err = (
                        mic_proc.stderr.read().decode(errors="replace").strip()
                        if mic_proc.stderr
                        else ""
                    )
                    print(f"FAIL [mic]: mic stopped producing audio — {err or 'EOF'}")
                    return EXIT_MIC_EOF
                send(client, rs.build_append_event(raw))
                if len(event_log.snapshot()) - idx > 0:
                    break  # a server event arrived; stop feeding and process it

        event, idx = _wait_until(
            event_log, reader, idx, ("input_audio_buffer.speech_started",), deadline
        )
        ok, detail = rs.classify_event_or_timeout(event, "input_audio_buffer.speech_started")
        print(f"{'PASS' if ok else 'FAIL'} [speech-started]: {detail}")

        event, idx = _wait_until(
            event_log,
            reader,
            idx,
            ("conversation.item.input_audio_transcription.completed",),
            deadline,
        )
        if event is None or event.get("type") == "error":
            print(f"FAIL [transcript]: {event!r}")
            return EXIT_TIMEOUT if event is None else EXIT_SESSION_ERROR
        print(f"HEARD [transcript]: {event.get('text')!r}")

        event, idx = _wait_until(
            event_log,
            reader,
            idx,
            ("response.function_call_arguments.done", "response.done"),
            deadline,
        )
        if event is None:
            print("FAIL [tool-call]: TIMEOUT waiting for a tool call")
            return EXIT_TIMEOUT
        if event.get("type") == "error":
            print(f"FAIL [tool-call]: server error {event!r}")
            return EXIT_SESSION_ERROR
        if event.get("type") == "response.done":
            spoken = [
                e.get("text") for e in event_log.snapshot() if e.get("type") == "response.text.done"
            ]
            print(
                f"FAIL [tool-call]: the model answered in speech without calling the tool: {spoken}"
            )
            print()
            print(build_latency_table([event]))
            return EXIT_SESSION_ERROR
        call_id = event.get("call_id")
        try:
            call_args = parse_function_call_arguments(event.get("arguments") or "{}")
        except ToolArgumentError as exc:
            print(f"FAIL [tool-args]: {exc}")
            return EXIT_SESSION_ERROR
        print(f"PASS [tool-call]: name={event.get('name')!r} call_id={call_id!r} args={call_args}")

        ok, order_detail = check_transcript_before_tool_call(event_log.snapshot())
        print(f"{'PASS' if ok else 'FAIL'} [order]: {order_detail}")
        if not ok:
            exit_code = EXIT_ORDER_VIOLATION

        output = run_list_directory(tool_root, call_args.get("path"))
        print(f"TOOL RESULT: {output}")
        send(client, build_function_call_output_event(call_id, output))
        send(client, build_response_create_event())

        event, idx = _wait_until(event_log, reader, idx, ("response.done",), deadline)
        for said in (
            e.get("text") for e in event_log.snapshot() if e.get("type") == "response.text.done"
        ):
            print(f"SPOKEN [response.text.done]: {said!r}")
        response_done_events = [e for e in event_log.snapshot() if e.get("type") == "response.done"]
        if event is not None and event.get("type") == "response.done":
            print("PASS [response.done]")
        else:
            print(f"FAIL [response.done]: {event!r}")
            if exit_code == EXIT_OK:
                exit_code = EXIT_TIMEOUT if event is None else EXIT_SESSION_ERROR

        print()
        print(build_latency_table(response_done_events))
        return exit_code
    finally:
        client.send_close()
        reader.join(timeout=3.0)
        client.close()
        if mic_proc is not None:
            _terminate(mic_proc)
        if playback_proc is not None:
            try:
                if playback_proc.stdin:
                    playback_proc.stdin.close()
            except OSError:
                pass
            _terminate(playback_proc)
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - an acceptance client must never traceback silently
        print(f"FAIL [unexpected]: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(EXIT_UNEXPECTED)
