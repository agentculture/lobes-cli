"""The segmenter's provisional-pause events (hebrew-realtime d9, layer A).

A pause is NOT a boundary: nothing is committed, and the default (no
``eager_silence_ms``) emits exactly the pre-d9 event sequence.
"""

from __future__ import annotations

from lobes.realtime._segmenter import (
    CHUNK_BYTES,
    Segmenter,
    SpeechPaused,
    SpeechResumed,
    SpeechStarted,
    SpeechStopped,
)
from lobes.realtime.protocol import VAD_CHUNK_MS


class ScriptedVad:
    def __init__(self, script: str) -> None:
        self._probs = [0.9 if ch == "S" else 0.0 for ch in script]

    def __call__(self, chunk: bytes) -> float:
        return self._probs.pop(0)


def run(script: str, **kwargs) -> list:
    seg = Segmenter(ScriptedVad(script), vad_prefix_padding_ms=0, **kwargs)
    events = []
    for index in range(len(script)):
        events += seg.feed(bytes([index % 256]) * CHUNK_BYTES)
    return events


def kinds(events: list) -> list[str]:
    return [type(e).__name__ for e in events]


SILENCE_10 = VAD_CHUNK_MS * 10
EAGER_3 = VAD_CHUNK_MS * 3


def test_default_never_emits_pause_events():
    events = run("SS" + "." * 10, vad_silence_ms=SILENCE_10)
    assert kinds(events) == ["SpeechStarted", "SpeechStopped"]


def test_a_pause_fires_once_at_the_eager_threshold_then_the_commit_follows():
    events = run("SS" + "." * 10, vad_silence_ms=SILENCE_10, eager_silence_ms=EAGER_3)
    assert kinds(events) == ["SpeechStarted", "SpeechPaused", "SpeechStopped"]
    paused, stopped = events[1], events[2]
    assert isinstance(paused, SpeechPaused) and isinstance(stopped, SpeechStopped)
    assert len(paused.audio) == 5 * CHUNK_BYTES  # 2 speech + 3 silent chunks
    # the commit is the SAME turn plus trailing silence — what makes adoption safe
    assert stopped.audio.startswith(paused.audio)
    assert paused.at_ms == 5 * VAD_CHUNK_MS


def test_resumed_speech_after_a_pause_is_reported_so_the_speculation_can_die():
    events = run(
        "SS" + "." * 4 + "SS" + "." * 10, vad_silence_ms=SILENCE_10, eager_silence_ms=EAGER_3
    )
    assert kinds(events) == [
        "SpeechStarted",
        "SpeechPaused",
        "SpeechResumed",
        "SpeechPaused",
        "SpeechStopped",
    ]
    assert isinstance(events[2], SpeechResumed)
    # the second pause snapshots the WHOLE turn so far, not just the tail
    assert len(events[3].audio) == (2 + 4 + 2 + 3) * CHUNK_BYTES


def test_a_silence_shorter_than_the_eager_threshold_emits_nothing():
    events = run("SS..SS" + "." * 10, vad_silence_ms=SILENCE_10, eager_silence_ms=EAGER_3)
    assert kinds(events) == ["SpeechStarted", "SpeechPaused", "SpeechStopped"]


def test_an_eager_threshold_at_or_above_the_commit_silence_is_inert():
    for eager in (SILENCE_10, SILENCE_10 * 2, 0, -5):
        events = run("SS" + "." * 10, vad_silence_ms=SILENCE_10, eager_silence_ms=eager)
        assert kinds(events) == ["SpeechStarted", "SpeechStopped"], eager


def test_a_max_turn_commit_needs_no_pause():
    events = run("S" * 8, vad_silence_ms=SILENCE_10, eager_silence_ms=EAGER_3, max_turn_ms=8 * 32)
    assert kinds(events) == ["SpeechStarted", "SpeechStopped"]
    assert events[1].reason == "max_turn"


def test_started_event_is_unchanged():
    events = run("S", vad_silence_ms=SILENCE_10, eager_silence_ms=EAGER_3)
    assert isinstance(events[0], SpeechStarted)
