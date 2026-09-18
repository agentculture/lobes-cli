"""Hebrew vocalization hook — phonikud in the TTS text path (stdlib-only).

``tts_client.synthesize()`` calls :func:`vocalize_hebrew` after
``_clean_for_tts`` and before chunking, when the caller's ``language`` is
``"he"`` (issue hebrew-realtime t9). This module never imports
``phonikud_onnx`` itself: :func:`vocalize_hebrew` takes an INJECTED
diacritizer callable (``Callable[[str], str]``), so it is fully testable
offline with a fake callable — the real one is built by
:func:`build_phonikud_diacritizer`, whose ``import phonikud_onnx`` happens
lazily, inside the function body, guarded by nothing more than the function
never being called outside the ``realtime`` container image (the only place
``phonikud-onnx`` is installed).

Failure handling
-----------------
On the diacritizer raising, or on it not finishing within ``timeout``
seconds, :func:`vocalize_hebrew` returns the ORIGINAL input text unchanged
and logs a warning naming the cause — never a partially-vocalized string,
never a raised exception the caller has to catch. A synchronous callable's
timeout is enforced by running it in a worker thread with a deadline
(``concurrent.futures``), since a plain function call has no way to be
interrupted mid-execution.

Span extraction — only Hebrew-letter spans reach the diacritizer
-------------------------------------------------------------------
:func:`vocalize_hebrew` never hands the WHOLE input text to the diacritizer.
It first segments the text into spans (:func:`_segment_hebrew_spans`): a
span is "Hebrew" when it is a maximal run of Hebrew letters (U+05D0-U+05EA),
Hebrew niqqud/cantillation combining marks (U+0591-U+05C7),
geresh/gershayim (U+05F3/U+05F4) and whitespace, AND that run contains at
least one actual Hebrew letter. Every other character — Latin letters,
digits, path separators, punctuation, and whitespace-only runs with no
adjacent Hebrew letter — falls into a non-Hebrew span that the diacritizer
never sees and that is copied through byte-identical. This is what makes
"Latin and digit spans come back unchanged" a property this module
guarantees itself, rather than something merely hoped of the injected
callable.

Debug observability
--------------------
When the ``TTS_DEBUG_TEXT`` env var is truthy (``1``/``true``/``yes``,
case-insensitive — read at call time, not at import time, so a test can
toggle it via ``monkeypatch.setenv``), the fully vocalized text is logged at
INFO on this module's logger — the "niqqud really reached TTS" signal an
acceptance run can point to.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

log = logging.getLogger(__name__)

Diacritizer = Callable[[str], str]

_T = TypeVar("_T")

# Generous default for a CPU int8 ONNX diacritizer on one sentence; the
# hebrew-realtime spec notes phonikud's per-sentence latency on this box is
# unmeasured (docs/specs/2026-09-18-hebrew-realtime.md), so this errs
# conservative rather than clipping a slow-but-working call.
DEFAULT_TIMEOUT_S = 5.0

_DEBUG_ENV_VAR = "TTS_DEBUG_TEXT"
_TRUTHY = frozenset({"1", "true", "yes"})

# Hebrew letters (aleph .. tav).
_HEBREW_LETTER_LO = 0x05D0
_HEBREW_LETTER_HI = 0x05EA
# Niqqud + cantillation combining marks.
_HEBREW_MARK_LO = 0x0591
_HEBREW_MARK_HI = 0x05C7
# Geresh (׳) and gershayim (״) — Hebrew punctuation, not ASCII quotes.
_GERESH = "׳"
_GERSHAYIM = "״"


class LazySingleton(Generic[_T]):
    """Thread-safe, build-at-most-once-per-process lazy value.

    ``get()`` calls the *builder* passed to ``__init__`` at most once, even
    under concurrent callers — guarded by an internal ``threading.Lock`` —
    and caches whatever it returns (``None`` included) for every later call.

    Extracted stdlib-only so it is testable offline: ``tts_client.py``'s
    ``_get_hebrew_diacritizer()`` wraps its (slow, lazily-imports
    ``phonikud_onnx``) build in one of these, and calls ``get()`` via
    ``asyncio.to_thread`` so the build never runs on the event loop (Qodo
    finding — a synchronous lazy build on the loop froze every concurrent
    session on the first Hebrew reply). Because ``tts_client.py`` imports
    ``httpx`` at module top and the offline test env has no ``httpx``, this
    class — not ``tts_client`` itself — is what the offline suite exercises
    for that concurrency guarantee.
    """

    def __init__(self, builder: Callable[[], _T]) -> None:
        self._builder = builder
        self._lock = threading.Lock()
        self._built = False
        self._value: _T | None = None

    def get(self) -> _T | None:
        if self._built:
            return self._value
        with self._lock:
            if not self._built:
                self._value = self._builder()
                self._built = True
        return self._value


def _is_hebrew_letter(ch: str) -> bool:
    return _HEBREW_LETTER_LO <= ord(ch) <= _HEBREW_LETTER_HI


def _is_hebrew_mark(ch: str) -> bool:
    return _HEBREW_MARK_LO <= ord(ch) <= _HEBREW_MARK_HI


def _is_hebrew_related(ch: str) -> bool:
    """True for any character allowed inside a Hebrew span: a Hebrew letter,
    a niqqud/cantillation mark, geresh/gershayim, or whitespace (so a space
    between two Hebrew words stays attached to its span instead of splitting
    it in two).
    """
    return (
        _is_hebrew_letter(ch) or _is_hebrew_mark(ch) or ch in (_GERESH, _GERSHAYIM) or ch.isspace()
    )


def _scan_hebrew_span(text: str, start: int, n: int) -> tuple[int, bool]:
    """Return ``(end, has_letter)`` for the maximal Hebrew-related run
    starting at *start* — the end index just past the run, and whether it
    contains at least one actual Hebrew letter (vs. only marks/geresh/
    whitespace). Extracted from :func:`_segment_hebrew_spans` to keep that
    function's cognitive complexity inside the gate (Sonar S3776).
    """
    j = start
    has_letter = False
    while j < n and _is_hebrew_related(text[j]):
        if _is_hebrew_letter(text[j]):
            has_letter = True
        j += 1
    return j, has_letter


def _scan_non_hebrew_span(text: str, start: int, n: int) -> int:
    """Return the end index (exclusive) of the maximal non-Hebrew-related
    run starting at *start*. Sibling of :func:`_scan_hebrew_span`, same
    extraction rationale.
    """
    j = start
    while j < n and not _is_hebrew_related(text[j]):
        j += 1
    return j


def _segment_hebrew_spans(text: str) -> list[tuple[bool, str]]:
    """Split *text* into ``(is_hebrew, span)`` pieces that concatenate back
    to *text* exactly. See the module docstring's "Span extraction" section
    for what makes a span "Hebrew".
    """
    spans: list[tuple[bool, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if _is_hebrew_related(text[i]):
            j, has_letter = _scan_hebrew_span(text, i, n)
            spans.append((has_letter, text[i:j]))
        else:
            j = _scan_non_hebrew_span(text, i, n)
            spans.append((False, text[i:j]))
        i = j
    return spans


def _debug_text_enabled(env: dict | None = None) -> bool:
    """True iff ``TTS_DEBUG_TEXT`` holds a truthy token, read at call time.

    ``env`` defaults to ``os.environ``; tests pass an explicit mapping.
    """
    source = os.environ if env is None else env
    return (source.get(_DEBUG_ENV_VAR) or "").strip().lower() in _TRUTHY


def _run_with_timeout(diacritizer: Diacritizer, span: str, timeout: float) -> str:
    """Run *diacritizer* on *span* in a worker thread, enforcing *timeout*.

    Raises whatever the diacritizer itself raised, or
    :class:`concurrent.futures.TimeoutError` if it did not finish in time —
    :func:`vocalize_hebrew` is the single place both are caught and turned
    into "return the input text and log a warning naming the cause".

    Deliberately NOT a ``with ThreadPoolExecutor(...)`` block: leaving that
    block joins the pool (``shutdown(wait=True)``), which on a timeout would
    block until the overdue worker thread finishes anyway — defeating the
    timeout entirely (Qodo finding). On timeout (or any other exception) this
    shuts the pool down with ``wait=False, cancel_futures=True`` instead and
    returns/raises immediately; the overdue worker thread is abandoned to run
    to completion on its own and its result is discarded.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(diacritizer, span)
        return future.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def vocalize_hebrew(
    text: str,
    diacritizer: Diacritizer,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Vocalize the Hebrew-letter spans of *text* via *diacritizer*.

    Only spans containing at least one Hebrew letter are ever handed to
    *diacritizer* — Latin and digit spans (paths, filenames, numbers) are
    copied through unchanged and never reach it (see the module docstring's
    "Span extraction" section). On *diacritizer* raising an exception, or
    not returning within *timeout* seconds on ANY span, this function gives
    up on the WHOLE call and returns the original *text* unchanged, logging
    one warning that names the cause — never a partially-vocalized result.

    Returns *text* unchanged (no call to *diacritizer* at all) when *text*
    is empty or contains no Hebrew letters.
    """
    if not text:
        return text

    spans = _segment_hebrew_spans(text)
    if not any(is_hebrew for is_hebrew, _ in spans):
        return text

    out_parts: list[str] = []
    for is_hebrew, span in spans:
        if not is_hebrew:
            out_parts.append(span)
            continue
        try:
            out_parts.append(_run_with_timeout(diacritizer, span, timeout))
        except concurrent.futures.TimeoutError:
            log.warning(
                "[vocalize] diacritizer timed out after %.2fs — using un-vocalized text", timeout
            )
            return text
        # Any diacritizer failure degrades to un-vocalized text, never raises.
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[vocalize] diacritizer failed (%s: %s) — using un-vocalized text",
                type(exc).__name__,
                exc,
            )
            return text

    result = "".join(out_parts)
    if _debug_text_enabled():
        log.info("[vocalize] TTS_DEBUG_TEXT vocalized text: %s", result)
    return result


def build_phonikud_diacritizer(model_path: str) -> Diacritizer:
    """Build a real :data:`Diacritizer` backed by phonikud (lazy import).

    ``import phonikud_onnx`` happens INSIDE this function, not at module
    top — this module (and the whole ``lobes.realtime._vocalize`` import)
    must stay importable in the offline test env and the base wheel, neither
    of which installs ``phonikud-onnx``. Only the ``realtime`` container
    image (issue hebrew-realtime) installs it and only that image's own
    startup code should ever call this factory.

    Raises ``ImportError`` when ``phonikud_onnx`` is not installed — that is
    a deployment/packaging bug (a caller in an environment without the
    dependency), not something this module should mask.
    """
    # Intentionally lazy import — see the docstring above.
    from phonikud_onnx import Phonikud  # noqa: PLC0415

    model = Phonikud(model_path)

    def _diacritize(text: str) -> str:
        return model.add_diacritics(text)

    return _diacritize
