"""Hidden speculation — start answering before the turn is confirmed.

hebrew-realtime plan, approved deviation **d9**, layer A (2026-09-18).

Why
---
With sentence streaming (d7) and a fast voice (d8) the pipeline itself is
~600-700 ms from a committed turn to first audio, measured live. The largest
single cost left is not compute at all: it is ``VAD_SILENCE_MS`` — the
silence the segmenter waits through before it BELIEVES the turn has ended
(1000 ms on the Hebrew deployment, because 600 ms split real sentences). The
machine sits idle for that whole second.

So it stops being idle. At a short provisional pause (``VAD_EAGER_MS``,
:class:`~lobes.realtime._segmenter.SpeechPaused`) the route snapshots the
turn and runs the WHOLE pipeline on it out of sight — STT, the streamed
generate call, synthesis of the first sentences — while the segmenter keeps
listening. Two outcomes:

* the speaker resumes → everything is cancelled and dropped. Nothing was
  emitted, nothing entered history, no tool ran (tools only ever run on the
  client, after an event that was never sent). The cost is wasted compute —
  the operator's word for the trade was "spendy", and it is opt-in.
* the silence is confirmed → the real turn ADOPTS the work: the transcript
  is reused, the buffered generate stream is replayed into the bridge, and
  already-synthesized sentences are delivered at once.

What makes adoption safe
------------------------
Adoption is decided by :func:`can_adopt`, and it is deliberately dumb: the
REAL generate request, built by the bridge after the real commit, must be
byte-for-byte the request the speculation sent (same url, same JSON body —
history, system prompt, tools, model, transcript). Anything that changed in
between — a tool declared, a history entry, a different transcript — makes
the bodies differ and the speculation is discarded for the ordinary path.
There is no second notion of "close enough".

The transcript itself is reused because the committed audio is the snapshot
plus trailing NON-speech chunks only (the segmenter's prefix property; a
resumed speaker cancels the speculation before any commit can happen).

This module holds only the pure, offline-testable pieces — stdlib + asyncio.
The orchestration (tasks, httpx, TTS) lives in the route, ``app.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ._turn import TurnRequest

__all__ = ["LineBuffer", "SegmentAudioCache", "can_adopt"]


def can_adopt(speculative: TurnRequest | None, real: TurnRequest | None) -> bool:
    """Whether the real turn may reuse the speculative generate stream."""
    if speculative is None or real is None:
        return False
    return speculative.url == real.url and speculative.body == real.body


class LineBuffer:
    """SSE lines from a producer that started EARLY, for a reader that starts late.

    The reader sees every line from the beginning — buffered ones at once,
    then live ones as they arrive — and the producer's failure, if any, is
    raised in the reader at the point it happened (so the adopting driver's
    existing ``httpx`` error handling still names it).
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._closed = False
        self._error: BaseException | None = None
        self._changed = asyncio.Event()

    def put(self, line: str) -> None:
        self._lines.append(line)
        self._changed.set()

    def close(self, error: BaseException | None = None) -> None:
        self._closed = True
        self._error = error
        self._changed.set()

    @property
    def closed(self) -> bool:
        return self._closed

    async def __aiter__(self) -> AsyncIterator[str]:
        index = 0
        while True:
            while index < len(self._lines):
                yield self._lines[index]
                index += 1
            if self._closed:
                if self._error is not None:
                    raise self._error
                return
            self._changed.clear()
            if index < len(self._lines) or self._closed:
                continue
            await self._changed.wait()


class SegmentAudioCache:
    """Speculatively synthesized sentences, keyed by their exact text.

    A slot is RESERVED before synthesis starts, so a real synth worker that
    arrives mid-synthesis waits for the in-flight result instead of queueing a
    duplicate request behind it on a concurrency-1 TTS lane. A failed,
    cancelled or empty speculative synthesis is a plain miss: the real worker
    synthesizes normally and any real failure is named by the real path.
    """

    def __init__(self) -> None:
        self._slots: dict[str, asyncio.Future] = {}

    def reserve(self, text: str) -> asyncio.Future | None:
        """A future to fulfil for *text*, or ``None`` if one is already reserved."""
        if text in self._slots:
            return None
        slot: asyncio.Future = asyncio.get_running_loop().create_future()
        self._slots[text] = slot
        return slot

    async def get(self, text: str) -> bytes | None:
        slot = self._slots.get(text)
        if slot is None:
            return None
        # `asyncio.wait` never re-raises the slot's OWN outcome — a cancelled
        # or failed speculative synthesis merely leaves it done — so the
        # result is inspected rather than caught. It does still propagate OUR
        # caller's cancellation (barge-in/teardown), which is the one
        # CancelledError that must never be swallowed, and it leaves the slot
        # itself pending for whoever else is waiting on it.
        await asyncio.wait({slot})
        if slot.cancelled() or slot.exception() is not None:
            return None  # a speculative failure is only a miss
        return slot.result() or None

    def cancel_pending(self) -> None:
        for slot in self._slots.values():
            if not slot.done():
                slot.cancel()
