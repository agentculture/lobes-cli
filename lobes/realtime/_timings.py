"""Per-stage stopwatch for one spoken turn — stdlib only.

``response.done`` carries :class:`lobes.realtime._session.StageTimings`
(hebrew-realtime claim c33): how many milliseconds the turn spent in STT,
generate, a tool wait, phonikud vocalization, TTS, and how long it took the
first audio delta to leave the server. The *boundaries* are the route's —
only ``app.py`` knows when a POST went out or when a chunk hit the socket —
but the ACCUMULATION is a decision, so it lives here, offline-tested, exactly
like every other piece of turn bookkeeping the route would otherwise own.

Three properties this module exists to guarantee:

- **A stage that never ran is ABSENT, never zero.** ``0`` reads as "instant",
  which is the dishonesty ``StageTimings`` was designed to avoid. Only a
  stage that was actually stopped (or explicitly recorded) appears.
- **A stage entered twice ACCUMULATES.** A tool turn calls the generate lane
  twice — once for the tool call, once with the result folded in — and the
  reported ``generate`` is the sum of both legs, not the last one.
- **The clock is injected.** Tests drive a list; the route passes
  :func:`lobes.realtime.protocol.timestamp_ms` (monotonic milliseconds), the
  same clock the floor uses, so the two never disagree about what a
  millisecond is.

``first_delta`` is a stage like any other, started when the turn commits and
stopped by :meth:`StageClock.mark_first_delta` at the first delta — but only
by the FIRST one, since "time to first audio" stops being interesting the
moment audio is flowing.
"""

from __future__ import annotations

from typing import Callable

from ._session import STAGE_TIMING_KEYS, StageTimings
from .protocol import timestamp_ms

FIRST_DELTA_STAGE = "first_delta"


class StageClock:
    """Accumulating stopwatch over the six :data:`STAGE_TIMING_KEYS` stages.

    One per SESSION, reset at each committed turn — the earliest stage
    (``stt``) starts on the commit, long before the response task that runs
    the later ones exists, so a per-response clock could never measure it.
    """

    def __init__(self, clock: Callable[[], int] = timestamp_ms) -> None:
        self._clock = clock
        self._totals: dict[str, int] = {}
        self._running: dict[str, int] = {}
        self._first_delta_marked = False

    def reset(self) -> None:
        """Forget every measurement — a new turn measures itself from zero."""
        self._totals.clear()
        self._running.clear()
        self._first_delta_marked = False

    def start(self, stage: str) -> None:
        """Open *stage*. A stage already running is left on its first start.

        Re-starting a running stage keeps the EARLIER boundary rather than
        moving it forward: a route that opened a stage twice by mistake should
        over-report the span it is unsure about, never under-report it.
        """
        self._check(stage)
        self._running.setdefault(stage, self._clock())

    def stop(self, stage: str) -> None:
        """Close *stage*, adding its elapsed span to the total.

        A stage that is not running is a no-op — the route stops stages on
        paths that may not have started them (a turn that failed before the
        stage opened), and a stopwatch is not the place to police that.
        """
        self._check(stage)
        started = self._running.pop(stage, None)
        if started is None:
            return
        self.record(stage, max(0, self._clock() - started))

    def record(self, stage: str, elapsed_ms: int) -> None:
        """Add an already-measured span to *stage*.

        For a stage the route cannot bracket itself: ``phonikud`` happens
        INSIDE ``tts_client.synthesize``, which hands its own elapsed
        milliseconds back rather than exposing the diacritizer call.
        """
        self._check(stage)
        self._totals[stage] = self._totals.get(stage, 0) + max(0, int(elapsed_ms))

    def mark_first_delta(self) -> None:
        """Stop ``first_delta`` at the FIRST audio chunk; later calls do nothing."""
        if self._first_delta_marked:
            return
        self._first_delta_marked = True
        self.stop(FIRST_DELTA_STAGE)

    def snapshot(self) -> StageTimings:
        """The measurement so far — absent stages stay absent.

        A stage still RUNNING is absent too: an unfinished span has no honest
        duration to report, and reporting the partial one would read as a
        completed stage.
        """
        return StageTimings(**{stage: self._totals.get(stage) for stage in STAGE_TIMING_KEYS})

    @staticmethod
    def _check(stage: str) -> None:
        if stage not in STAGE_TIMING_KEYS:
            raise ValueError(f"unknown timing stage {stage!r}; expected one of {STAGE_TIMING_KEYS}")


__all__ = ["FIRST_DELTA_STAGE", "StageClock"]
