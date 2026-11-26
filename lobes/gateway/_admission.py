"""The admission signal on the gateway's ``GET /status`` (issue #346).

``/status`` reports how busy the box is (``busy``) and what the pressure policy
would do to a full-tier request right now (``pressure``). A caller that wants
to know whether to START work that will use this gateway had to rebuild a rule
from those two blocks, and had no timestamp to tell a fresh answer from a
cached one. This module owns that rule and renders four additive top-level
fields:

``admit``
    ``True`` or ``False``: whether the gateway wants new work now.
``at``
    When the status was computed, RFC 3339 in UTC to the second
    (``2026-11-26T12:00:00Z``).
``stale_after_s``
    After how many seconds a reader should treat the answer as unknown.
``reason``
    Short text naming why ``admit`` is ``False``. Absent when it is ``True``.

The rule, in reporting order:

1. the pressure policy sheds a full-tier request → hold, ``shed:<signal>``
   (``swap``, ``thermal``, ``engine`` or ``iowait``; plain ``shed`` when the
   policy names no signal);
2. any backend reports requests waiting in its queue → hold, ``waiting``;
3. otherwise → admit.

It is deliberately the same decision a live request meets: ``admit`` is never
``True`` while ``pressure.shed`` is ``True``. It does NOT fold in backend
health (read ``backends[].health``) and cannot see the queue of a backend that
exports no in-flight counts (``busy.partial``).

Stdlib only, pure: the caller supplies the clock reading.
"""

from __future__ import annotations

from datetime import datetime, timezone

# How long a reader may trust one answer. The pressure sample behind `admit`
# refreshes every 2 s (`PressureCache`), so 10 s is five sample periods: long
# enough for a caller that polls every few seconds, short enough that a cached
# answer cannot outlive the state it described by much.
STALE_AFTER_S = 10.0

REASON_SHED = "shed"
REASON_WAITING = "waiting"


def rfc3339_utc(now: float) -> str:
    """Epoch seconds *now* as RFC 3339 in UTC, to the second (``...Z``)."""
    return datetime.fromtimestamp(int(now), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def admission(*, shed: bool, shed_signal: str, waiting: int, now: float) -> dict:
    """The ``/status`` admission fields for one computed status.

    *shed* and *shed_signal* come from the pressure decision ``/status``
    already makes for a full tier, *waiting* is the fleet's aggregate waiting
    count, *now* is the epoch-seconds instant the status was computed.
    """
    reason = ""
    if shed:
        reason = f"{REASON_SHED}:{shed_signal}" if shed_signal else REASON_SHED
    elif waiting > 0:
        reason = REASON_WAITING
    fields: dict = {
        "admit": not reason,
        "at": rfc3339_utc(now),
        "stale_after_s": STALE_AFTER_S,
    }
    if reason:
        fields["reason"] = reason
    return fields
