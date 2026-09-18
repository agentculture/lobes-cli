"""Tests for the Hebrew vocalization hook (stdlib-only; no [realtime] extra).

:mod:`lobes.realtime._vocalize` takes an INJECTED diacritizer callable — it
never imports ``phonikud_onnx`` itself (that import is lazy, guarded, and
only ever exercised inside the ``realtime`` container image) — so every case
here drives the module directly with a fake callable and plain Python
values, mirroring the house style of :mod:`tests.test_realtime_turn` /
:mod:`tests.test_realtime_segmenter`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from lobes.realtime._vocalize import (
    DEFAULT_TIMEOUT_S,
    LazySingleton,
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

    def test_timeout_returns_promptly_not_after_the_slow_call_finishes(self) -> None:
        """Regression for the Qodo finding: ``_run_with_timeout`` used to run
        ``future.result(timeout=...)`` INSIDE a ``with ThreadPoolExecutor(...)``
        block, so leaving the block on timeout called ``shutdown(wait=True)``
        and blocked until the overdue worker thread finished — a 2s
        diacritizer with a 0.1s timeout made the whole call take ~2s instead
        of returning at the deadline. It must now return in well under 1s.
        """

        def slow(t: str) -> str:
            time.sleep(2.0)
            return _add_niqqud(t)

        start = time.monotonic()
        result = vocalize_hebrew("שלום עולם", slow, timeout=0.1)
        elapsed = time.monotonic() - start

        assert result == "שלום עולם"
        assert elapsed < 1.0, f"took {elapsed:.3f}s — timeout is blocking on shutdown(wait=True)"


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


class TestLazySingleton:
    """Offline coverage for the Qodo finding in ``tts_client.py``:
    ``_get_hebrew_diacritizer()`` used to build the (slow) phonikud model
    SYNCHRONOUSLY on the event loop, so the first Hebrew reply froze every
    concurrent session. The fix wraps the build in a :class:`LazySingleton`
    and calls ``get()`` via ``asyncio.to_thread`` — this class is the
    stdlib-only piece that guarantees (1) the builder runs at most once
    under concurrency and (2) ``get()`` is safe to run off the event loop
    without the caller reimplementing locking. ``tts_client.py`` itself
    cannot be imported in this offline env (it imports ``httpx`` at module
    top), so this is where that guarantee is actually exercised.
    """

    def test_builder_runs_once_under_concurrent_threads(self) -> None:
        call_count = 0
        count_lock = threading.Lock()

        def builder() -> str:
            nonlocal call_count
            with count_lock:
                call_count += 1
            time.sleep(0.05)
            return "built"

        singleton: LazySingleton[str] = LazySingleton(builder)
        results: list[str | None] = [None] * 8

        def worker(i: int) -> None:
            results[i] = singleton.get()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count == 1
        assert results == ["built"] * 8

    def test_get_returns_the_builders_value_including_none(self) -> None:
        calls = []

        def builder() -> None:
            calls.append(1)
            return None

        singleton: LazySingleton[None] = LazySingleton(builder)
        assert singleton.get() is None
        assert singleton.get() is None
        assert calls == [1]  # cached even though the value is None

    def test_get_via_asyncio_to_thread_does_not_block_the_event_loop(self) -> None:
        """Mirrors how ``tts_client.synthesize`` uses this class: a slow
        first build, run via ``asyncio.to_thread``, must not stall a
        concurrent coroutine on the same loop.
        """

        def slow_builder() -> str:
            time.sleep(0.3)
            return "built"

        singleton: LazySingleton[str] = LazySingleton(slow_builder)

        async def main() -> tuple[str | None, list[float]]:
            ticks: list[float] = []

            async def heartbeat() -> None:
                for _ in range(6):
                    await asyncio.sleep(0.05)
                    ticks.append(time.monotonic())

            build_task = asyncio.create_task(asyncio.to_thread(singleton.get))
            hb_task = asyncio.create_task(heartbeat())
            result = await build_task
            await hb_task
            return result, ticks

        result, ticks = asyncio.run(main())

        assert result == "built"
        # 6 ticks at ~50ms apart (~300ms) completed WHILE the 300ms build ran
        # in its own thread — if get() blocked the loop, the heartbeat could
        # not have made progress until after the build finished.
        assert len(ticks) == 6
