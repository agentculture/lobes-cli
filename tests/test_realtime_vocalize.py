"""Tests for the Hebrew vocalization hook (stdlib-only; no [realtime] extra).

:mod:`lobes.realtime._vocalize` takes an INJECTED diacritizer callable — it
never imports ``phonikud_onnx`` itself (that import is lazy, guarded, and
only ever exercised inside the ``realtime`` container image) — so every case
here drives the module directly with a fake callable and plain Python
values, mirroring the house style of :mod:`tests.test_realtime_turn` /
:mod:`tests.test_realtime_segmenter`.
"""

from __future__ import annotations

import logging
import time

import pytest

from lobes.realtime._vocalize import (
    DEFAULT_TIMEOUT_S,
    build_phonikud_diacritizer,
    vocalize_hebrew,
)

# A mixed-script sentence used elsewhere in the hebrew-realtime spec's own
# probes (docs/specs/2026-09-18-hebrew-realtime.md) — Hebrew prose wrapping a
# filesystem path, a colon, two filenames and a hyphenated "and" prefix.
MIXED_SENTENCE = (
    "בתיקייה /home/spark/git/lobes-cli/docs/specs נמצאים שלושה קבצים: "
    "2026-09-18-hebrew-realtime.md ו-README.md."
)


def _add_niqqud(word: str) -> str:
    """A trivial fake diacritizer: appends a single niqqud mark (patach,
    U+05B7) after every Hebrew letter. Good enough to prove marks reach the
    output without needing the real phonikud model.
    """
    out = []
    for ch in word:
        out.append(ch)
        if "א" <= ch <= "ת":
            out.append("ַ")
    return "".join(out)


class TestVocalizeHebrewHappyPath:
    def test_vocalizes_plain_hebrew_text(self) -> None:
        result = vocalize_hebrew("שלום", _add_niqqud)
        assert result == _add_niqqud("שלום")
        assert result != "שלום"

    def test_empty_text_is_a_no_op(self) -> None:
        calls = []
        result = vocalize_hebrew("", lambda t: calls.append(t) or "should not happen")
        assert result == ""
        assert calls == []

    def test_text_with_no_hebrew_never_calls_the_diacritizer(self) -> None:
        calls = []

        def diacritizer(t: str) -> str:
            calls.append(t)
            return t

        result = vocalize_hebrew("hello world 123", diacritizer)
        assert result == "hello world 123"
        assert calls == []


class TestLatinAndDigitSpansUnchanged:
    def test_mixed_sentence_latin_and_digit_spans_survive_byte_identical(self) -> None:
        result = vocalize_hebrew(MIXED_SENTENCE, _add_niqqud)
        # Every Latin/digit run from the source sentence must still appear,
        # verbatim, in the vocalized output.
        for span in (
            "/home/spark/git/lobes-cli/docs/specs",
            "2026-09-18-hebrew-realtime.md",
            "README.md",
        ):
            assert span in result

    def test_diacritizer_never_receives_a_latin_or_digit_span(self) -> None:
        seen: list[str] = []

        def diacritizer(t: str) -> str:
            seen.append(t)
            return t

        vocalize_hebrew(MIXED_SENTENCE, diacritizer)
        for chunk in seen:
            assert "/home" not in chunk
            assert "2026" not in chunk
            assert "README" not in chunk
            assert ".md" not in chunk

    def test_pure_latin_text_round_trips_exactly(self) -> None:
        text = "/home/spark/git/lobes-cli/docs/specs/2026-09-18-hebrew-realtime.md"
        assert vocalize_hebrew(text, _add_niqqud) == text


class TestVocalizeHebrewFailureModes:
    def test_exception_returns_input_text_unchanged(self) -> None:
        def boom(t: str) -> str:
            raise ValueError("model not loaded")

        result = vocalize_hebrew("שלום עולם", boom)
        assert result == "שלום עולם"

    def test_exception_logs_a_warning_naming_the_cause(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(t: str) -> str:
            raise ValueError("model not loaded")

        with caplog.at_level(logging.WARNING, logger="lobes.realtime._vocalize"):
            vocalize_hebrew("שלום עולם", boom)

        assert any("model not loaded" in rec.message for rec in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_timeout_returns_input_text_unchanged(self) -> None:
        def slow(t: str) -> str:
            time.sleep(2.0)
            return _add_niqqud(t)

        result = vocalize_hebrew("שלום עולם", slow, timeout=0.05)
        assert result == "שלום עולם"

    def test_timeout_logs_a_warning_naming_the_cause(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def slow(t: str) -> str:
            time.sleep(2.0)
            return t

        with caplog.at_level(logging.WARNING, logger="lobes.realtime._vocalize"):
            vocalize_hebrew("שלום עולם", slow, timeout=0.05)

        assert any(
            "timed out" in rec.message.lower() or "timeout" in rec.message.lower()
            for rec in caplog.records
        )

    def test_default_timeout_is_a_positive_number(self) -> None:
        assert DEFAULT_TIMEOUT_S > 0


class TestBuildPhonikudDiacritizerIsLazy:
    def test_import_of_this_module_does_not_require_phonikud_onnx(self) -> None:
        # Getting this far without an ImportError already proves it, but be
        # explicit: the factory itself must exist and be callable without
        # having imported phonikud_onnx at module top.
        assert callable(build_phonikud_diacritizer)

    def test_calling_the_factory_without_phonikud_installed_raises_import_error(self) -> None:
        # This offline env never installs phonikud-onnx (it is only ever
        # installed inside the realtime container image) — so calling the
        # factory here must fail with an ImportError, not something that
        # looks like a real diacritizer.
        with pytest.raises(ImportError):
            build_phonikud_diacritizer("/nonexistent/phonikud-1.0.int8.onnx")
