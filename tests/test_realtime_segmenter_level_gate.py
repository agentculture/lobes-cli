"""The input-level gate: ignore quiet voices (background talk, a TV, the next room).

Silero answers "is this speech?", not "is this speech ADDRESSED TO ME?" — a
quiet far voice scores as high as a near one. ``min_level_pct`` adds the
missing question: a chunk counts as speech only while the PEAK level, held over
a short window like a level meter, is at least that percentage of full scale.
Measured on the reSpeaker 2026-09-18: direct speech peaks 15-55 %, the room's
floor 1-2 %.
"""

from __future__ import annotations

import struct

import pytest

from lobes.realtime._segmenter import CHUNK_BYTES, Segmenter, chunk_peak_pct
from lobes.realtime.protocol import VAD_CHUNK_MS

SAMPLES = CHUNK_BYTES // 2


def chunk(peak_pct: float) -> bytes:
    """A chunk that is silent except for one sample at ``peak_pct`` of full scale."""
    value = int(32767 * peak_pct / 100)
    return struct.pack(f"<{SAMPLES}h", *([0] * (SAMPLES - 1) + [-value]))


def run(levels: list[float], **kwargs) -> list[str]:
    seg = Segmenter(
        lambda _c: 0.9, vad_prefix_padding_ms=0, vad_silence_ms=VAD_CHUNK_MS * 12, **kwargs
    )
    events = []
    for level in levels:
        events += seg.feed(chunk(level))
    return [type(e).__name__ for e in events]


def test_peak_is_a_percentage_of_full_scale_and_sign_blind():
    assert chunk_peak_pct(chunk(50)) == pytest.approx(50, abs=0.01)
    assert chunk_peak_pct(struct.pack(f"<{SAMPLES}h", *([-32768] * SAMPLES))) == 100.0
    assert chunk_peak_pct(bytes(CHUNK_BYTES)) == 0.0


def test_off_by_default_a_quiet_voice_still_opens_a_turn():
    assert run([2.0] * 5) == ["SpeechStarted"]


def test_a_voice_below_the_threshold_never_opens_a_turn():
    assert run([2.0] * 40, min_level_pct=5.0) == []


def test_a_voice_just_over_the_threshold_opens_one():
    assert run([5.1] * 3, min_level_pct=5.0) == ["SpeechStarted"]  # 5.0 quantizes to 4.998


def test_the_gaps_between_syllables_do_not_chop_a_loud_speaker():
    # loud, then 6 quiet chunks (192 ms) of the same voice, then loud again:
    # the held peak keeps the turn's speech run alive — no pause, no commit.
    levels = [30.0] + [1.0] * 6 + [30.0] + [1.0] * 6 + [30.0]
    assert run(levels, min_level_pct=5.0, eager_silence_ms=VAD_CHUNK_MS * 3) == ["SpeechStarted"]


def test_a_loud_speaker_followed_by_quiet_background_talk_still_ends_the_turn():
    # Silero keeps saying "speech" (someone is talking across the room), but it
    # is below the gate, so it counts as silence and the turn commits.
    levels = [30.0] * 3 + [2.0] * 30
    assert run(levels, min_level_pct=5.0) == ["SpeechStarted", "SpeechStopped"]


def test_without_the_gate_that_background_talk_holds_the_turn_open():
    assert run([30.0] * 3 + [2.0] * 30) == ["SpeechStarted"]


@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_zero_or_negative_is_off(bad):
    assert run([2.0] * 5, min_level_pct=bad) == ["SpeechStarted"]


def test_the_threshold_is_capped_below_full_scale():
    # 100 % would make the microphone deaf; the cap keeps a typo survivable.
    assert run([100.0] * 3, min_level_pct=500.0) == ["SpeechStarted"]
