"""Pure-logic tests for the TTS pause / truncation helpers.

``lobes/realtime/tts_client.py`` imports ``httpx`` at module top (it only ever
loads inside the ``realtime`` container), so this module is skipped in the
offline CI env — the same pattern ``tests/test_chatterbox_pcm16.py`` uses for
numpy. It still runs anywhere the ``[realtime]`` extra is installed.

The functions under test are pure and stdlib-only; the httpx import is the only
reason they cannot be measured offline. Extracting them into a stdlib module so
the offline suite covers them is a worthwhile follow-up.
"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx", reason="tts_client imports httpx at module top")

from lobes.realtime.tts_client import (  # noqa: E402
    _is_truncated,
    _min_plausible_duration,
    trailing_pause_ms,
)


class TestTrailingPauseMs:
    """Punctuation → inter-sentence pause, incl. the S8786 regex rewrite."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("wow!!!", 400),
            ("wow!!!!!!", 400),
            ("really?!", 350),
            ("really!?", 350),
            ("wow!!", 350),
            ("hmm...", 400),
            ("hmm…", 400),
            ("done.", 350),
            ("what?", 350),
            ("hey!", 300),
            ("no punctuation", 200),
            ("", 200),
            ("   ", 200),
        ],
    )
    def test_punctuation_mapping(self, text: str, expected: int) -> None:
        assert trailing_pause_ms(text) == expected

    def test_trailing_run_must_be_at_the_end(self) -> None:
        """A "!!!" run NOT at the end must not win the 400 ms branch.

        The old `!{3,}$` regex anchored at end-of-string; the rstrip-based
        replacement must keep exactly that meaning.
        """
        assert trailing_pause_ms("wow!!!then more") == 200
        assert trailing_pause_ms("wow!!! then more.") == 350

    def test_ignores_trailing_whitespace(self) -> None:
        assert trailing_pause_ms("wow!!!   ") == 400
        assert trailing_pause_ms("done.  ") == 350

    def test_long_bang_run_is_not_quadratic(self) -> None:
        """Regression for Sonar S8786.

        The former `!{3,}$` pattern backtracked per start position on a long
        run that is not at the end — ~2.6 s for 40k characters. This asserts
        the pathological input is handled essentially instantly.
        """
        import time

        pathological = "!" * 40_000 + "x"
        start = time.perf_counter()
        result = trailing_pause_ms(pathological)
        elapsed = time.perf_counter() - start

        assert result == 200  # run is not at the end, so no 400 ms branch
        assert elapsed < 0.1, f"took {elapsed:.3f}s — quadratic backtracking is back"


class TestTruncationDetection:
    """Ratio-based short-audio detection extracted for Sonar S3776."""

    def test_min_plausible_duration_has_a_floor(self) -> None:
        assert _min_plausible_duration("") == 0.5
        assert _min_plausible_duration("short") == 0.5

    def test_min_plausible_duration_scales_with_length(self) -> None:
        assert _min_plausible_duration("x" * 100) == pytest.approx(1.5)

    def test_short_text_is_never_truncated(self) -> None:
        """Texts of 10 chars or fewer are exempt, however short the audio."""
        assert _is_truncated("tiny", 0.0) is False

    def test_implausibly_short_audio_is_truncated(self) -> None:
        text = "x" * 100  # needs >= 1.5s
        assert _is_truncated(text, 0.2) is True

    def test_plausible_audio_is_not_truncated(self) -> None:
        text = "x" * 100
        assert _is_truncated(text, 6.0) is False
