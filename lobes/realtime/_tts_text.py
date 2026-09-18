"""Pure text helpers for the TTS request path (stdlib-only, ``re`` only).

Extracted out of ``lobes/realtime/tts_client.py`` (issue hebrew-realtime t9):
``tts_client.py`` imports ``httpx`` at module top, so it only ever loads
inside the ``realtime`` container and is coverage-omitted
(``pyproject.toml``'s ``[tool.coverage.run]``) — the pure text-shaping logic
living there was, as a result, never actually unit-tested offline (see
``tests/test_tts_pause_and_truncation.py``'s own docstring, which names this
extraction as a "worthwhile follow-up"). This module holds exactly that
logic and nothing that touches a socket, so it imports and is fully tested
without the ``[realtime]`` extra.

``tts_client.py`` re-exports every name here under its historical spelling
(``from ._tts_text import _clean_for_tts, ...``), so nothing that already
imported these names from ``tts_client`` needs to change.

Hebrew note — base characters vs. niqqud
-----------------------------------------
:func:`_split_for_tts` sizes chunks by :func:`_base_char_length`, not
``len()``: Hebrew niqqud/cantillation combining marks (U+0591-U+05C7) ride on
the preceding base character and must not inflate the chunk-sizing count —
one vocalized Hebrew word with full niqqud can be twice as many *code
points* as its bare form while remaining exactly as many *characters* a
listener hears. For text with no combining marks in that range (every
English string, and any Hebrew string before vocalization) base length
equals raw length, so :func:`_split_for_tts`'s behavior for ``language="en"``
callers is unchanged.
"""

from __future__ import annotations

import re

# Regex to strip emoji (Supplementary Multilingual Plane + common emoji ranges)
_EMOJI_RE = re.compile(
    "[\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # zero-width joiner
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)

# Markdown-style formatting
_MARKDOWN_RE = re.compile(r"[*_~`#]")

# Max chars of *cleaned* text per TTS request.
# Conservative chunking ceiling for Chatterbox (no hard SSML or Triton token limit;
# kept at 600 to avoid extremely long single requests and preserve latency).
_MAX_CLEAN_CHARS = 600

# Hebrew niqqud + cantillation combining marks (Unicode block U+0591-U+05C7).
# See the module docstring's "Hebrew note" for why chunk sizing excludes
# these from its character count.
_NIQQUD_LO = 0x0591
_NIQQUD_HI = 0x05C7


def _is_niqqud_mark(ch: str) -> bool:
    """True for a single Hebrew niqqud/cantillation combining mark."""
    return _NIQQUD_LO <= ord(ch) <= _NIQQUD_HI


def _base_char_length(text: str) -> int:
    """Character count excluding Hebrew niqqud/cantillation combining marks.

    Equals ``len(text)`` whenever *text* has no such marks — which is every
    English string, so this never changes ``language="en"`` behavior.
    """
    return sum(1 for ch in text if not _is_niqqud_mark(ch))


def _raw_index_for_base_chars(text: str, max_base_chars: int) -> int:
    """The raw index ``i`` such that ``text[:i]`` holds at most
    *max_base_chars* base (non-niqqud) characters — the base-char-aware
    counterpart of ``text[:max_base_chars]``.

    Returns ``len(text)`` when the whole string fits within the budget.
    """
    if max_base_chars <= 0:
        return 0
    count = 0
    for idx, ch in enumerate(text):
        if not _is_niqqud_mark(ch):
            count += 1
            if count > max_base_chars:
                return idx
    return len(text)


def _clean_for_tts(text: str) -> str:
    """Strip emoji, markdown, dashes, quotes and normalize for TTS input."""
    text = _EMOJI_RE.sub(" ", text)
    text = _MARKDOWN_RE.sub("", text)
    # Em-dash / en-dash → comma (natural pause; raw dashes confuse TTS)
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    # Curly single quotes / apostrophes → ASCII apostrophe (preserves contractions)
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    # Strip double-quotes (TTS doesn't need to voice them). Real Hebrew
    # gershayim (U+05F4) is a distinct code point and is never matched here.
    text = re.sub(r'["“”]', "", text)
    # Remove markdown list markers at line start:  - item  /  1. item
    text = re.sub(r"(?m)^\s*-\s+", " ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", " ", text)
    # Collapse whitespace / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_for_tts(text: str, max_chars: int = _MAX_CLEAN_CHARS) -> list[str]:
    """Split *text* into chunks of at most *max_chars* BASE characters.

    Tries to break at the last ``", "`` before the limit, then last ``" "``,
    and hard-cuts only as a last resort. Returns a single-element list when
    the text already fits. Sizing is by :func:`_base_char_length` — Hebrew
    niqqud/cantillation marks never count toward *max_chars* (see the module
    docstring). For text with no such marks (every English string) this is
    identical to counting ``len()``.
    """
    if _base_char_length(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while _base_char_length(remaining) > max_chars:
        window_end = _raw_index_for_base_chars(remaining, max_chars)
        window = remaining[:window_end]
        # Prefer splitting at last ", " (natural pause)
        idx = window.rfind(", ")
        if idx > 0:
            cut = idx + 2  # keep the comma+space with the left chunk
        else:
            # Fall back to last space
            idx = window.rfind(" ")
            if idx > 0:
                cut = idx + 1
            else:
                # Hard cut — no good break point
                cut = window_end
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


# ---------------------------------------------------------------------------
# Punctuation-aware pause helpers
# ---------------------------------------------------------------------------


def trailing_pause_ms(original_text: str) -> int:
    """Return inter-sentence silence duration (ms) based on ending punctuation.

    Examines the *original* sentence text (before TTS cleaning) so that
    trailing emoji and raw punctuation are still visible.
    """
    s = original_text.rstrip()
    if not s:
        return 200

    # Check multi-char patterns first (longest match wins).
    # Count the trailing run of "!" by string ops rather than a `!{3,}$` regex:
    # on a long run that is not at the end, that pattern backtracks per start
    # position (quadratic — Sonar S8786). rstrip is linear and says the same thing.
    if len(s) - len(s.rstrip("!")) >= 3:
        return 400
    if s.endswith(("?!", "!?")):
        return 350
    if s.endswith("!!"):
        return 350
    if s.endswith(("...", "…")):
        return 400
    if s.endswith("."):
        return 350
    if s.endswith("?"):
        return 350
    if s.endswith("!"):
        return 300

    # Trailing emoji
    if _EMOJI_RE.search(s[-2:]):
        return 250

    return 200


def _min_plausible_duration(clean: str) -> float:
    """Shortest audio duration that is plausible for *clean*.

    Ratio-based: expect at least 15 ms per character (normal speech at 125 %
    runs 60–80 ms/char, so 15 ms is very conservative).
    """
    return max(0.5, len(clean) * 0.015)


def _is_truncated(clean: str, duration: float) -> bool:
    """True when returned audio is implausibly short for the text it should speak."""
    return len(clean) > 10 and duration < _min_plausible_duration(clean)
