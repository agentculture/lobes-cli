"""Incremental sentence chunking for streamed generation — stdlib only.

Before this module a spoken reply was produced in two WHOLE-reply stages: one
blocking, non-streaming POST to ``/v1/chat/completions``, then one whole-text
Chatterbox synthesis, and only then the first audio delta. MEASURED live on
the DGX Spark (2026-09-18, the Hebrew voice loop): first audio 2.6 s after the
turn for a one-sentence reply and 7.3 s for a longer one, of which TTS alone
was 1.9-5.9 s. The engine synthesizes ~1.5x faster than real time, so every
one of those seconds was the machine waiting for text it could already have
started speaking.

This module is the decision that fixes it (approved deviation d7,
2026-09-18): it turns a stream of generate deltas into COMPLETE sentences, so
TTS can start on sentence 1 while sentence 2 is still being generated and
first audio becomes ``time-to-first-sentence + synth of ONE short
sentence`` — independent of how long the reply turns out to be.

Pure and offline-testable, exactly like :mod:`._segmenter` and :mod:`._floor`:
no I/O, no sibling imports beyond :mod:`._tts_text`'s base-character helpers
(the same ones the TTS chunker already sizes by), and every threshold is a
constructor argument rather than module policy.

The four rules, and what each one trades
----------------------------------------
1. **A boundary must be CONFIRMED.** A terminator (``. ! ? … :``) counts only
   when whitespace — or, at :meth:`SentenceChunker.flush`, end of stream —
   follows it; a newline is a boundary on its own. A bare terminator at the
   end of the buffer is never a boundary, because the next delta can turn
   ``3.`` into ``3.5``. This is also why an ellipsis run splits once, after
   its last dot, instead of three times.
2. **Crumbs merge forward** (:attr:`SentenceChunker.min_sentence_chars`, 12
   base chars). Handing TTS ``"כן."`` costs a whole request for a fraction of
   a second of audio and inserts an unnatural pause where the model meant
   none. A crumb therefore waits for the next boundary — unless it is the
   whole reply, in which case it is spoken as it is.
3. **The FIRST sentence may break early** (:attr:`eager_first_min_chars`, 24
   base chars): once the opening clause is that long, a comma or semicolon
   is accepted as a boundary too. Time-to-first-audio is the number this
   change exists to move, and a long opening sentence would otherwise hold
   the whole benefit hostage. Only the first — later commas wait for a real
   terminator, because by then audio is already flowing and a mid-sentence
   TTS seam is audible.
4. **A run-on is cut anyway** (:attr:`max_sentence_chars`, 220 base chars),
   at the last ``", "`` then the last space, via
   :func:`lobes.realtime._tts_text._split_for_tts` — the same splitter the
   TTS client already uses, so the two cannot disagree about where a break
   is acceptable.

Sizing is by BASE characters (:func:`lobes.realtime._tts_text._base_char_length`),
never ``len()``: vocalized Hebrew carries niqqud combining marks that double
the code points of a word without adding a syllable, so counting code points
would let a crumb through as if it were a sentence.

Never split inside an abbreviation
-----------------------------------
A terminator immediately preceded by a geresh/gershayim/quote (``׳ ״ ' "``)
is not a boundary: that mark belongs to an abbreviation (``וכו׳``, ``צה"ל``,
``ד"ר``), and releasing there would hand TTS a fragment. The bias throughout
is MERGE, never mis-split — a late boundary costs a little latency once, a
wrong one is audible in every reply.
"""

from __future__ import annotations

from ._tts_text import _base_char_length, _split_for_tts

# The smallest piece worth a TTS request of its own, in base characters. See
# rule 2 above: below this a piece merges into the next sentence.
DEFAULT_MIN_SENTENCE_CHARS = 12

# The largest piece handed to TTS in one request, in base characters. Kept
# well under _tts_text._MAX_CLEAN_CHARS (600, the TTS client's own ceiling)
# because a sentence-level stream wants SHORT units — a 600-char segment
# would reintroduce exactly the whole-reply latency this module removes.
DEFAULT_MAX_SENTENCE_CHARS = 220

# How long the FIRST clause must be before a comma/semicolon counts as a
# boundary for it (rule 3). Short enough to fire on a real opening clause,
# long enough that "כן," never becomes a synthesis of its own.
DEFAULT_EAGER_FIRST_MIN_CHARS = 24

# Terminators that end a sentence outright, when whitespace follows.
_TERMINATORS = frozenset(".!?…:")

# Terminators the FIRST sentence may additionally break at (rule 3).
_EAGER_TERMINATORS = frozenset(",;")

# A terminator directly after one of these belongs to an abbreviation, not to
# the end of a thought — ASCII quote/apostrophe plus Hebrew geresh/gershayim.
_ABBREVIATION_MARKS = frozenset("\"'׳״")


class SentenceChunker:
    """Turn generate deltas into complete sentences, one reply at a time.

    Construct one per REPLY (or call :meth:`reset` between replies — the
    eager-first allowance is per reply, not per session). Every method is
    synchronous and total: :meth:`feed` returns the sentences that became
    complete because of this delta, possibly none, and :meth:`flush` returns
    whatever is left when the stream ends.
    """

    def __init__(
        self,
        *,
        min_sentence_chars: int = DEFAULT_MIN_SENTENCE_CHARS,
        max_sentence_chars: int = DEFAULT_MAX_SENTENCE_CHARS,
        eager_first_min_chars: int = DEFAULT_EAGER_FIRST_MIN_CHARS,
    ) -> None:
        self.min_sentence_chars = max(0, min_sentence_chars)
        # Never below 1: a zero cap would cut forever without consuming.
        self.max_sentence_chars = max(1, max_sentence_chars)
        self.eager_first_min_chars = max(0, eager_first_min_chars)
        self._buf = ""
        self._released = 0

    # -- observation ------------------------------------------------------

    @property
    def released(self) -> int:
        """How many sentences this reply has released so far."""
        return self._released

    @property
    def pending(self) -> str:
        """The text buffered but not yet released (observation/tests)."""
        return self._buf

    # -- inputs -----------------------------------------------------------

    def reset(self) -> None:
        """Forget this reply — the next one gets its own eager-first break."""
        self._buf = ""
        self._released = 0

    def feed(self, text_delta: str) -> list[str]:
        """Consume one generate delta; return the sentences it COMPLETED."""
        if text_delta:
            self._buf += text_delta
        return self._drain()

    def flush(self) -> list[str]:
        """End of stream: return everything still buffered.

        The remainder is released even when it is shorter than
        :attr:`min_sentence_chars` — there is no next sentence to merge it
        into, and dropping it would silently truncate the reply. When this
        same call already released a sentence, the crumb is appended to it
        instead, so a trailing ``"כן."`` never becomes its own request.
        """
        out = self._drain()
        rest = self._buf.strip()
        self._buf = ""
        if not rest:
            return out
        pieces = self._split_long(rest)
        if out and _base_char_length(pieces[0]) < self.min_sentence_chars:
            out[-1] = f"{out[-1]} {pieces[0]}".strip()
            pieces = pieces[1:]
        out.extend(pieces)
        self._released += len(pieces)
        return out

    # -- internals --------------------------------------------------------

    def _drain(self) -> list[str]:
        """Release every sentence the buffer can currently prove complete."""
        out: list[str] = []
        while True:
            piece = self._take_boundary_piece()
            if piece is None:
                break
            pieces = self._split_long(piece)
            out.extend(pieces)
            self._released += len(pieces)
        out.extend(self._take_overlong())
        return out

    def _take_boundary_piece(self) -> str | None:
        """The next CONFIRMED sentence, or ``None`` while the buffer is open.

        Scans left to right and skips a boundary whose piece would be a crumb
        (rule 2) — the scan simply continues to the next one, which is what
        makes "merge forward" fall out of the same loop.
        """
        buf = self._buf
        eager = self._released == 0
        for index, char in enumerate(buf):
            end = self._boundary_end(buf, index, char, eager)
            if end is None:
                continue
            piece = buf[:end].strip()
            if _base_char_length(piece) < self.min_sentence_chars:
                continue  # a crumb: merge it into whatever comes next
            self._buf = buf[end:].lstrip()
            return piece
        return None

    def _boundary_end(self, buf: str, index: int, char: str, eager: bool) -> int | None:
        """The cut index just past a boundary at *index*, or ``None``.

        A newline is a boundary on its own (and is consumed rather than
        spoken). Every other terminator needs a following whitespace
        character to be CONFIRMED — the next delta could still extend it.
        """
        if char == "\n":
            return index  # the newline itself is dropped by the caller's lstrip
        if char not in _TERMINATORS and not (eager and char in _EAGER_TERMINATORS):
            return None
        if index + 1 >= len(buf) or not buf[index + 1].isspace():
            return None
        if index and buf[index - 1] in _ABBREVIATION_MARKS:
            return None  # an abbreviation's own mark, not the end of a thought
        if eager and char in _EAGER_TERMINATORS:
            if _base_char_length(buf[: index + 1]) < self.eager_first_min_chars:
                return None
        return index + 1

    def _take_overlong(self) -> list[str]:
        """Cut a run-on that has no boundary in sight (rule 4).

        The LAST piece stays buffered: more text may still arrive to finish
        it, and releasing it now would cut a sentence that was about to end
        on its own.
        """
        if _base_char_length(self._buf) <= self.max_sentence_chars:
            return []
        pieces = _split_for_tts(self._buf, self.max_sentence_chars)
        self._buf = pieces[-1]
        released = [piece for piece in pieces[:-1] if piece]
        self._released += len(released)
        return released

    def _split_long(self, text: str) -> list[str]:
        """Size one released piece to the cap, using the TTS splitter."""
        if _base_char_length(text) <= self.max_sentence_chars:
            return [text]
        return [piece for piece in _split_for_tts(text, self.max_sentence_chars) if piece]


__all__ = [
    "DEFAULT_MIN_SENTENCE_CHARS",
    "DEFAULT_MAX_SENTENCE_CHARS",
    "DEFAULT_EAGER_FIRST_MIN_CHARS",
    "SentenceChunker",
]
