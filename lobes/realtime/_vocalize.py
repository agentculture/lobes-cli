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
from collections.abc import Callable

log = logging.getLogger(__name__)

Diacritizer = Callable[[str], str]

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


def _segment_hebrew_spans(text: str) -> list[tuple[bool, str]]:
    """Split *text* into ``(is_hebrew, span)`` pieces that concatenate back
    to *text* exactly. See the module docstring's "Span extraction" section
    for what makes a span "Hebrew".
    """
    spans: list[tuple[bool, str]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if _is_hebrew_related(ch):
            j = i
            has_letter = False
            while j < n and _is_hebrew_related(text[j]):
                if _is_hebrew_letter(text[j]):
                    has_letter = True
                j += 1
            spans.append((has_letter, text[i:j]))
            i = j
        else:
            j = i
            while j < n and not _is_hebrew_related(text[j]):
                j += 1
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
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(diacritizer, span)
        return future.result(timeout=timeout)


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
        except Exception as exc:  # noqa: BLE001 - any diacritizer failure degrades, never raises
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
    from phonikud_onnx import Phonikud  # noqa: PLC0415 - intentionally lazy, see docstring

    model = Phonikud(model_path)

    def _diacritize(text: str) -> str:
        return model.add_diacritics(text)

    return _diacritize
