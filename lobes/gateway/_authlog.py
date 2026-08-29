"""Collapsed, source-naming logging for inbound auth rejections (issue #228).

The opt-in inbound bearer gate (#127) is correct — a 401 on an unauthenticated
POST is it working — but what it *tells the operator* was not. Measured on the
DGX Spark, 2026-08-28/29: **1190 rejections in four hours**, every one of them
this line and nothing else::

    [gateway] "POST /tokenize HTTP/1.1" 401 -

Three problems in one line, and the third is the reason the first two matter:

* **No source.** ``BaseHTTPRequestHandler``'s own ``log_message`` prefixes the
  client address; the gateway's override drops it (it was written to keep
  ``docker logs`` tidy). So the log could not answer the first question an
  operator asks — *who* — and therefore could not distinguish ONE misconfigured
  client from a scan.
* **No reason.** A client that never sends a key and a client sending the wrong
  one are different incidents with different fixes, and the line conflates them.
* **No collapsing.** 1190 identical lines is not a signal, it is a denial of
  service against the log: whatever else the gateway had to say in those four
  hours is now unfindable.

This module is the collapsing half. It is pure and clock-injected — no I/O, no
socket, no global state — so the whole policy is testable offline; the handler
owns the actual ``stderr`` write.

**On saying more in the log than in the response.** ``_invalid_api_key_body``
is deliberately static: it never distinguishes missing from malformed from
wrong, because a 401 must not become a key-material oracle for the *caller*.
The log has a different audience — the operator reading their own server's
stderr — and that asymmetry is the point: the reason categories below are what
make triage possible, and none of them ever travels back to the client. No
reason string here contains, or is derived from, any part of either the
presented credential or the configured key.

**No env knob.** The window and cap are module constants deliberately. A
``GATEWAY_*`` knob would enter the compose passthrough, the profile render
tables and (through them) the deployment lock's env allowlist — a lot of
surface for a logging cadence nobody has yet needed to tune.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

# How long one source's rejections collapse into a single line. Sized against
# the observed failure: the #228 resident retried in bursts of ~29 every ~2
# minutes, so a 60s window prints one line per burst instead of 29 — enough to
# see the problem is ongoing, not enough to bury the rest of the log.
COLLAPSE_WINDOW_SECONDS: float = 60.0

# The most sources tracked at once. Bounded because the keys are attacker-chosen
# (one entry per source address): an unbounded dict here would turn a
# distributed scan into a memory leak in the process it is scanning. Expired
# entries are pruned first; a genuine overflow evicts the least-recently-seen,
# which at worst costs that source one extra line.
MAX_TRACKED_SOURCES: int = 256

# --- rejection reasons (operator-facing; never sent to the caller) ----------
REASON_NO_HEADER = "no Authorization header"
REASON_NOT_BEARER = "not a Bearer credential"
# nosec B105 below: bandit's hardcoded-password heuristic fires on the NAME
# (it contains "TOKEN"), not the value. This is a human-readable reason string
# written into the operator's log — it is not a credential, is never compared
# against one, and is never sent to a caller.
REASON_EMPTY_TOKEN = "empty Bearer token"  # nosec B105
REASON_MISMATCH = "Bearer token did not match this gateway's key"


def rejection_reason(authorization: str | None) -> str:
    """Classify WHY a credential was rejected, for the operator's log.

    Mirrors :func:`lobes.gateway.server.bearer_token_matches`'s own fail-closed
    parse order, so the category always names the step that actually failed.
    Returns a fixed string from the constants above — never anything derived
    from the presented credential, so the classification cannot leak the token
    it classified (nor, obviously, the configured key, which is not consulted).
    """
    if not authorization:
        return REASON_NO_HEADER
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return REASON_NOT_BEARER
    if not token.strip():
        return REASON_EMPTY_TOKEN
    return REASON_MISMATCH


class _SourceState:
    """One tracked source's collapse window."""

    __slots__ = ("window_started", "last_seen", "suppressed")

    def __init__(self, now: float) -> None:
        self.window_started = now
        self.last_seen = now
        self.suppressed = 0


class RejectionLog:
    """Decides whether an auth rejection is worth a log line, and what it says.

    One instance per server, shared across handler threads (the gateway is a
    ``ThreadingHTTPServer``), hence the lock. :meth:`record` returns the line to
    write, or ``None`` when this rejection is being collapsed into a window that
    has already been reported — the caller suppresses BOTH the diagnostic and
    the ordinary access-log line in that case, which is what actually takes 1190
    lines down to ~24.

    ``clock`` is injected (default :func:`time.monotonic`, which is immune to
    wall-clock jumps) so the whole policy is testable without sleeping.
    """

    def __init__(
        self,
        *,
        window: float = COLLAPSE_WINDOW_SECONDS,
        max_sources: int = MAX_TRACKED_SOURCES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window = window
        self._max_sources = max_sources
        self._clock = clock
        self._lock = threading.Lock()
        self._sources: dict[str, _SourceState] = {}

    def record(self, source: str, method: str, path: str, reason: str) -> str | None:
        """The line to log for this rejection, or ``None`` to stay silent.

        The first rejection from a source opens a window and is always
        reported. Further rejections inside that window are counted, not
        printed. The first rejection AFTER a window expires is reported again,
        and carries the count of everything suppressed in between — so a
        sustained flood stays visible as a rising number instead of either
        vanishing or drowning the log.
        """
        now = self._clock()
        with self._lock:
            state = self._sources.get(source)
            if state is not None and now - state.window_started < self._window:
                state.suppressed += 1
                state.last_seen = now
                return None
            suppressed = state.suppressed if state is not None else 0
            elapsed = now - state.window_started if state is not None else 0.0
            self._sources[source] = _SourceState(now)
            self._evict_if_needed(now)
        return _format(method, path, source, reason, suppressed, elapsed)

    def _evict_if_needed(self, now: float) -> None:
        """Keep the tracking table bounded. Caller holds the lock."""
        if len(self._sources) <= self._max_sources:
            return
        # Expired windows first — they carry no unreported count by
        # construction (their next rejection reopens a window anyway).
        for key in [k for k, s in self._sources.items() if now - s.window_started >= self._window]:
            del self._sources[key]
            if len(self._sources) <= self._max_sources:
                return
        # Still over: drop the least-recently-seen. Costs that source one extra
        # line if it returns, which is the right way to be wrong here.
        while len(self._sources) > self._max_sources:
            oldest = min(self._sources, key=lambda k: self._sources[k].last_seen)
            del self._sources[oldest]


def _format(
    method: str, path: str, source: str, reason: str, suppressed: int, elapsed: float
) -> str:
    """The operator-facing line. ``suppressed`` is appended only when non-zero,
    so the common case (one stray request) reads as one plain sentence."""
    line = f"auth: rejected {method} {path} from {source} ({reason})"
    if suppressed:
        line += f" [+{suppressed} more from this source in the previous {elapsed:.0f}s]"
    return line
