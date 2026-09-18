"""Tests for the pure TTS text helpers (stdlib-only; no [realtime] extra).

These used to live only in ``lobes/realtime/tts_client.py``, which imports
``httpx`` at module top and is therefore coverage-omitted and never
unit-imported offline (see ``pyproject.toml``'s ``[tool.coverage.run]`` omit
list and ``tests/test_tts_pause_and_truncation.py``'s own docstring, which
names this extraction as a "worthwhile follow-up"). Task #151-hebrew t9
moves the pure helpers into :mod:`lobes.realtime._tts_text` — a plain
``re``-only module with no third-party imports — so this file is the first
offline test coverage they have ever had.

``tts_client.py`` still re-exports every name under its old spelling, so
``tests/test_tts_pause_and_truncation.py`` (which imports from
``lobes.realtime.tts_client`` and is skipped unless ``httpx`` is installed)
is untouched and keeps passing wherever the ``[realtime]`` extra is present.
"""

from __future__ import annotations

import pytest

from lobes.realtime._tts_text import (
    _base_char_length,
    _clean_for_tts,
    _is_truncated,
    _min_plausible_duration,
    _split_for_tts,
    trailing_pause_ms,
)

# The same mixed-script probe sentence named in
# docs/specs/2026-09-18-hebrew-realtime.md — Hebrew prose wrapping a
# filesystem path, a colon, two filenames and a hyphenated "and" prefix.
MIXED_SENTENCE = (
    "בתיקייה /home/spark/git/lobes-cli/docs/specs נמצאים שלושה קבצים: "
    "2026-09-18-hebrew-realtime.md ו-README.md."
)

# Real gershayim (U+05F4), used inside a Hebrew acronym exactly like צה"ל.
GERSHAYIM_WORD = "צה״ל"


# ---------------------------------------------------------------------------
# _clean_for_tts — English behaviour PINNED to the pre-refactor implementation.
# Every expected value below was captured by running the ORIGINAL
# lobes/realtime/tts_client.py._clean_for_tts against the same input, before
# any code moved. language="en" callers must see byte-identical output.
# ---------------------------------------------------------------------------


class TestCleanForTtsEnglishPinned:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                "Hello **world**, this is a test — really!",
                "Hello world, this is a test , really!",
            ),
            ("- item one\n- item two\n1. numbered", "item one item two numbered"),
            ("He said “hi” and it’s great \U0001f600", "He said hi and it's great"),
            ("no special chars here", "no special chars here"),
            ("", ""),
            ("   spaced   out   text  ", "spaced out text"),
        ],
    )
    def test_pinned_english_outputs(self, text: str, expected: str) -> None:
        assert _clean_for_tts(text) == expected

    def test_em_and_en_dash_become_comma(self) -> None:
        assert _clean_for_tts("a—b") == "a, b"
        assert _clean_for_tts("a–b") == "a, b"

    def test_markdown_markers_stripped(self) -> None:
        assert _clean_for_tts("*bold* _em_ ~tilde~ `code` #head") == "bold em tilde code head"


class TestCleanForTtsHebrew:
    def test_gershayim_survives_cleaning(self) -> None:
        # Real gershayim (U+05F4) is not an ASCII quote — the cleaner's
        # quote-stripping regex must never touch it.
        cleaned = _clean_for_tts(GERSHAYIM_WORD)
        assert "״" in cleaned
        assert cleaned == GERSHAYIM_WORD

    def test_ascii_quotes_still_stripped_around_hebrew(self) -> None:
        # Pre-existing, unchanged behaviour: an ASCII double-quote (as
        # opposed to real gershayim) is still stripped, exactly like it is
        # for English text.
        assert _clean_for_tts('צה"ל') == "צהל"

    def test_mixed_script_sentence_preserves_latin_digit_spans(self) -> None:
        cleaned = _clean_for_tts(MIXED_SENTENCE)
        for span in (
            "/home/spark/git/lobes-cli/docs/specs",
            "2026-09-18-hebrew-realtime.md",
            "README.md",
        ):
            assert span in cleaned


# ---------------------------------------------------------------------------
# _split_for_tts — English behaviour PINNED (captured pre-refactor).
# ---------------------------------------------------------------------------


class TestSplitForTtsEnglishPinned:
    def test_short_text_returns_single_chunk(self) -> None:
        assert _split_for_tts("hello world", max_chars=600) == ["hello world"]

    def test_space_break_pinned(self) -> None:
        text = ("word " * 200).strip()
        assert len(text) == 999
        chunks = _split_for_tts(text, max_chars=50)
        assert len(chunks) == 20
        assert all(c == "word word word word word word word word word word" for c in chunks)
        assert all(len(c) == 49 for c in chunks)

    def test_comma_break_pinned(self) -> None:
        text = ", ".join(f"clause number {i}" for i in range(30))
        assert len(text) == 528
        chunks = _split_for_tts(text, max_chars=60)
        expected = [
            "clause number 0, clause number 1, clause number 2,",
            "clause number 3, clause number 4, clause number 5,",
            "clause number 6, clause number 7, clause number 8,",
            "clause number 9, clause number 10, clause number 11,",
            "clause number 12, clause number 13, clause number 14,",
            "clause number 15, clause number 16, clause number 17,",
            "clause number 18, clause number 19, clause number 20,",
            "clause number 21, clause number 22, clause number 23,",
            "clause number 24, clause number 25, clause number 26,",
            "clause number 27, clause number 28, clause number 29",
        ]
        assert chunks == expected

    def test_hard_cut_when_no_break_point(self) -> None:
        text = "x" * 100
        chunks = _split_for_tts(text, max_chars=10)
        assert "".join(chunks) == text
        assert all(len(c) <= 10 for c in chunks)


class TestSplitForTtsCountsBaseCharsNotNiqqud:
    """Chunk sizing must count base characters, not niqqud combining marks
    (acceptance criterion 2) — a heavily-vocalized Hebrew string should not
    be split any more aggressively than the same string without niqqud.
    """

    @staticmethod
    def _mark(letters: str) -> str:
        # Interleave a patach (U+05B7) after every character — a trivial
        # stand-in for what a real diacritizer's output looks like.
        return "".join(ch + "ַ" for ch in letters)

    def test_niqqud_marks_do_not_count_toward_the_chunk_ceiling(self) -> None:
        bare = "שלום עולם"  # 9 base chars
        vocalized = self._mark(bare)  # niqqud doubles the raw length
        assert len(vocalized) > len(bare)
        assert _base_char_length(vocalized) == _base_char_length(bare) == len(bare)

        # A ceiling that comfortably fits the BASE length but not the RAW
        # (niqqud-inflated) length must still keep it as one chunk.
        assert _base_char_length(bare) <= 12 < len(vocalized)
        assert _split_for_tts(vocalized, max_chars=12) == [vocalized]

    def test_splitting_a_long_vocalized_string_still_bounds_base_chars(self) -> None:
        words = [self._mark("שלום") for _ in range(10)]  # each: 4 base, 8 raw
        text = " ".join(words)
        max_chars = 20
        chunks = _split_for_tts(text, max_chars=max_chars)
        assert len(chunks) > 1
        for chunk in chunks:
            assert _base_char_length(chunk) <= max_chars

    def test_ascii_text_base_length_equals_raw_length(self) -> None:
        # No niqqud present -> base length is just len(); this is the
        # guarantee that keeps language="en" chunking byte-identical.
        text = "plain ascii sentence with no combining marks at all"
        assert _base_char_length(text) == len(text)


# ---------------------------------------------------------------------------
# trailing_pause_ms — pinned to existing punctuation -> pause-ms mapping.
# ---------------------------------------------------------------------------


class TestTrailingPauseMsPinned:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("sentence.", 350),
            ("question?", 350),
            ("exclaim!", 300),
            ("wow!!", 350),
            ("wow?!", 350),
            ("wow!?", 350),
            ("trailing...", 400),
            ("trailing…", 400),
            ("wow!!!", 400),
            ("no terminal punctuation", 200),
            ("", 200),
        ],
    )
    def test_pinned_pause_values(self, text: str, expected: int) -> None:
        assert trailing_pause_ms(text) == expected


# ---------------------------------------------------------------------------
# _min_plausible_duration / _is_truncated — pinned.
# ---------------------------------------------------------------------------


class TestMinPlausibleDurationPinned:
    def test_has_a_floor(self) -> None:
        assert _min_plausible_duration("") == 0.5
        assert _min_plausible_duration("short") == 0.5

    def test_scales_with_length(self) -> None:
        assert _min_plausible_duration("x" * 100) == pytest.approx(1.5)

    def test_is_truncated_short_text_never_truncated(self) -> None:
        assert _is_truncated("tiny", 0.0) is False

    def test_is_truncated_long_text_short_audio(self) -> None:
        text = "x" * 100
        assert _is_truncated(text, 0.2) is True
        assert _is_truncated(text, 6.0) is False
