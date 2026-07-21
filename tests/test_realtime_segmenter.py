"""Tests for the server_vad segmenter (stdlib-only; no torch, no [realtime] extra).

The Silero VAD model is never imported here — every test drives
:class:`lobes.realtime._segmenter.Segmenter` with a small scripted fake that
returns a pre-programmed probability per 512-sample/32ms chunk it is fed,
exactly the injection point the module's docstring documents.
"""

from __future__ import annotations

from pathlib import Path

import lobes.realtime._segmenter as _segmenter
from lobes.realtime._segmenter import (
    CHUNK_BYTES,
    Segmenter,
    SpeechStarted,
    SpeechStopped,
)
from lobes.realtime.protocol import BYTES_PER_SAMPLE, VAD_CHUNK_MS, VAD_CHUNK_SAMPLES

# --- import isolation -------------------------------------------------------


def test_module_imports_without_torch_or_the_realtime_extra() -> None:
    # If this test file collects at all in the offline dev env (no torch,
    # no fastapi/httpx/numpy/scipy installed), the import above already
    # proved it. This test just makes the guarantee explicit and named.
    assert hasattr(_segmenter, "Segmenter")


def test_module_source_never_imports_forbidden_deps() -> None:
    # The docstring talks ABOUT torch/fastapi (documenting what this module
    # is NOT allowed to import) — so this scans actual import statements,
    # not bare substrings, to avoid flagging its own prose.
    src = Path(_segmenter.__file__).read_text(encoding="utf-8")
    forbidden = ("torch", "fastapi", "httpx", "numpy", "scipy", "silero_vad")
    offenders = [
        name
        for name in forbidden
        for line in src.splitlines()
        if line.strip().startswith((f"import {name}", f"from {name}"))
    ]
    assert not offenders, f"_segmenter.py imports forbidden deps: {offenders}"


def test_chunk_bytes_matches_the_protocol_framing() -> None:
    # 512 samples * 2 bytes/sample = 1024 bytes — derived from protocol.py's
    # constants, not redefined independently.
    assert CHUNK_BYTES == VAD_CHUNK_SAMPLES * BYTES_PER_SAMPLE == 1024


# --- test helpers ------------------------------------------------------------


def make_chunk(tag: int) -> bytes:
    """One full 1024-byte chunk whose every byte encodes ``tag`` (0-255)."""
    return bytes([tag % 256]) * CHUNK_BYTES


def chunk_tags(audio: bytes) -> list[int]:
    """Recover the tag sequence baked into ``audio`` by make_chunk()."""
    assert len(audio) % CHUNK_BYTES == 0, "turn audio must be whole chunks"
    return [audio[i] for i in range(0, len(audio), CHUNK_BYTES)]


class ScriptedVad:
    """A fake vad_probability callable: pops one probability per call.

    Records every chunk it was called with, so tests can assert the
    segmenter never hands it a short (partial) frame.
    """

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self.seen_chunks: list[bytes] = []

    def __call__(self, chunk: bytes) -> float:
        self.seen_chunks.append(chunk)
        return self._probs.pop(0)


# --- framing / buffering ------------------------------------------------------


def test_short_chunks_are_buffered_never_handed_to_vad() -> None:
    vad = ScriptedVad([0.0])
    seg = Segmenter(vad)

    events = seg.feed(b"\x01" * 500)
    assert events == []
    assert vad.seen_chunks == []  # not a full chunk yet

    events = seg.feed(b"\x01" * (CHUNK_BYTES - 500))
    assert vad.seen_chunks == [b"\x01" * CHUNK_BYTES]  # exactly one full chunk
    assert events == []  # probability 0.0 < default threshold


def test_every_vad_call_receives_exactly_one_full_chunk() -> None:
    vad = ScriptedVad([0.0, 0.0, 0.0])
    seg = Segmenter(vad)

    # Two and a half chunks' worth of bytes in one feed() call.
    seg.feed(make_chunk(1) + make_chunk(2) + make_chunk(3)[:512])
    assert len(vad.seen_chunks) == 2
    for chunk in vad.seen_chunks:
        assert len(chunk) == CHUNK_BYTES

    # The trailing half-chunk completes here; still only ever whole chunks.
    seg.feed(make_chunk(3)[512:])
    assert len(vad.seen_chunks) == 3
    assert all(len(c) == CHUNK_BYTES for c in vad.seen_chunks)


# --- silence -> speech -> silence (criterion: padded onset + silence-ms stop) --


def test_silence_speech_silence_yields_padded_onset_and_silence_confirmed_stop() -> None:
    # 2 chunks of padding (64ms), 2 chunks of confirming silence (64ms).
    probs = [0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0]
    vad = ScriptedVad(probs)
    seg = Segmenter(
        vad,
        vad_threshold=0.5,
        vad_prefix_padding_ms=64,
        vad_silence_ms=64,
        max_turn_ms=10_000,
    )

    all_events: list[object] = []
    for tag in range(9):
        all_events.extend(seg.feed(make_chunk(tag)))

    starts = [e for e in all_events if isinstance(e, SpeechStarted)]
    stops = [e for e in all_events if isinstance(e, SpeechStopped)]
    assert len(starts) == 1
    assert len(stops) == 1

    # Onset fires on tag 3 (the first chunk whose probability crosses
    # threshold); its audio is the padded onset: 2 preceding silence chunks
    # (tags 1, 2 — 64ms of pre-roll, tag 0 already evicted by the 2-chunk
    # ring) plus the onset chunk itself (tag 3).
    assert chunk_tags(starts[0].audio) == [1, 2, 3]

    # Stop fires once 64ms (2 chunks) of continuous non-speech is confirmed
    # (tags 6, 7); the full turn spans padded-onset through the confirming
    # silence, inclusive.
    assert chunk_tags(stops[0].audio) == [1, 2, 3, 4, 5, 6, 7]
    assert stops[0].reason == "silence"

    # Onset precedes stop on the shared stream-time axis.
    assert starts[0].at_ms < stops[0].at_ms

    # The trailing silence chunk (tag 8) after the stop starts a fresh,
    # empty pre-roll window — no event, no leakage from the closed turn.
    assert not seg.speaking


def test_speech_started_audio_is_empty_padding_when_speech_opens_the_stream() -> None:
    # No preceding silence at all: padding is whatever's available (nothing).
    vad = ScriptedVad([0.9])
    seg = Segmenter(vad, vad_prefix_padding_ms=300)
    events = seg.feed(make_chunk(7))
    assert len(events) == 1
    assert isinstance(events[0], SpeechStarted)
    assert chunk_tags(events[0].audio) == [7]


def test_a_healthy_silent_session_emits_no_events() -> None:
    vad = ScriptedVad([0.0] * 20)
    seg = Segmenter(vad)
    events: list[object] = []
    for tag in range(20):
        events.extend(seg.feed(make_chunk(tag)))
    assert events == []
    assert not seg.speaking


# --- max-turn cap (criterion: never-silent stream -> documented force-commit) -


def test_never_silent_stream_force_commits_at_the_max_turn_cap() -> None:
    # Continuous speech, silence_ms set high enough it can never fire here;
    # max_turn_ms = 5 chunks (160ms) forces periodic commits instead.
    n_chunks = 12
    vad = ScriptedVad([0.9] * n_chunks)
    seg = Segmenter(
        vad,
        vad_prefix_padding_ms=0,
        vad_silence_ms=10_000,
        max_turn_ms=5 * VAD_CHUNK_MS,
    )

    all_events: list[object] = []
    for tag in range(1, n_chunks + 1):
        all_events.extend(seg.feed(make_chunk(tag)))

    stops = [e for e in all_events if isinstance(e, SpeechStopped)]
    starts = [e for e in all_events if isinstance(e, SpeechStarted)]

    # Never raises; every stop is force-committed, never confirmed by silence.
    assert stops, "expected at least one force-committed turn"
    assert all(s.reason == "max_turn" for s in stops)

    # Each force-committed turn is exactly 5 chunks (the cap), and a fresh
    # turn starts immediately on the very next chunk (the stream never
    # actually falls silent).
    for stop in stops:
        assert len(stop.audio) == 5 * CHUNK_BYTES

    # 12 chunks at a 5-chunk cap: two full turns force-commit (chunks 1-5,
    # 6-10), and a third turn starts on chunk 11 but is still 2 chunks short
    # of the cap when the stream ends — no exception, no silent data loss,
    # every force-commit exactly the cap (never larger, proving it's
    # actually enforced and not merely hit once by chance).
    assert len(stops) == 2
    assert len(starts) == 3  # one more start than stop: the trailing
    # in-progress (uncommitted) third turn
    assert seg.speaking  # the never-silent stream is still mid-turn


def test_flush_force_commits_an_in_progress_turn_with_the_given_reason() -> None:
    vad = ScriptedVad([0.9, 0.9, 0.0])
    seg = Segmenter(vad, vad_prefix_padding_ms=0, vad_silence_ms=10_000, max_turn_ms=10_000)
    seg.feed(make_chunk(1))
    seg.feed(make_chunk(2))
    assert seg.speaking

    event = seg.flush("closed")
    assert isinstance(event, SpeechStopped)
    assert event.reason == "closed"
    assert chunk_tags(event.audio) == [1, 2]
    assert not seg.speaking

    # Flushing an idle segmenter is a no-op.
    assert seg.flush("closed") is None


def test_a_short_trailing_partial_chunk_is_discarded_on_flush_not_committed() -> None:
    vad = ScriptedVad([0.9])
    seg = Segmenter(vad)
    seg.feed(make_chunk(1))
    seg.feed(b"\x02" * 100)  # a short trailing remainder, never a full chunk
    event = seg.flush("closed")
    assert isinstance(event, SpeechStopped)
    # Only the one full chunk that was actually processed is in the turn.
    assert chunk_tags(event.audio) == [1]


# --- VAD failures propagate (documented: not this module's job to translate) --


def test_a_raising_vad_probability_propagates_unmodified() -> None:
    class Boom(RuntimeError):
        pass

    def failing_vad(_chunk: bytes) -> float:
        raise Boom("silero unavailable")

    seg = Segmenter(failing_vad)
    try:
        seg.feed(make_chunk(0))
    except Boom:
        pass
    else:
        raise AssertionError("expected the fake VAD's exception to propagate")


# --- per-session isolation (criterion: two interleaved instances) ------------


def _run_to_completion(probs: list[float], **kwargs: object) -> list[object]:
    vad = ScriptedVad(list(probs))
    seg = Segmenter(vad, **kwargs)  # type: ignore[arg-type]
    events: list[object] = []
    for tag in range(len(probs)):
        events.extend(seg.feed(make_chunk(tag)))
    return events


def test_two_interleaved_sessions_never_cross_contaminate() -> None:
    # Two different scripts, two different configs.
    probs_a = [0.0, 0.0, 0.9, 0.9, 0.0, 0.0]
    kwargs_a = dict(vad_prefix_padding_ms=32, vad_silence_ms=64, max_turn_ms=10_000)

    probs_b = [0.9, 0.9, 0.0, 0.0, 0.9, 0.9]
    kwargs_b = dict(vad_prefix_padding_ms=0, vad_silence_ms=32, max_turn_ms=10_000)

    expected_a = _run_to_completion(probs_a, **kwargs_a)
    expected_b = _run_to_completion(probs_b, **kwargs_b)

    vad_a = ScriptedVad(list(probs_a))
    vad_b = ScriptedVad(list(probs_b))
    seg_a = Segmenter(vad_a, **kwargs_a)  # type: ignore[arg-type]
    seg_b = Segmenter(vad_b, **kwargs_b)  # type: ignore[arg-type]

    actual_a: list[object] = []
    actual_b: list[object] = []
    # Interleave chunk-by-chunk, alternating instances.
    for tag in range(len(probs_a)):
        actual_a.extend(seg_a.feed(make_chunk(tag)))
        actual_b.extend(seg_b.feed(make_chunk(tag)))

    assert actual_a == expected_a
    assert actual_b == expected_b
    # And they're genuinely distinct sequences — proving neither leaked into
    # the other (different configs would corrupt each other's timing/padding
    # if any state were shared at module level).
    assert actual_a != actual_b


def test_constructor_defaults_match_the_documented_settings_defaults() -> None:
    # Mirrors lobes.realtime._settings.build_settings()'s VAD defaults.
    seg = Segmenter(lambda _c: 0.0)
    assert seg.vad_threshold == 0.5
    assert seg.vad_silence_ms == 600
    assert seg.vad_prefix_padding_ms == 300
    assert seg.max_turn_ms == 30_000
