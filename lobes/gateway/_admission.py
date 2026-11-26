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
    When the pressure sample behind the answer was taken (the time the status
    was computed when that is not known), RFC 3339 in UTC to the second
    (``2026-11-26T12:00:00Z``).
``stale_after_s``
    After how many seconds a reader should treat the answer as unknown.
``reason``
    Short text naming why ``admit`` is ``False``. Absent when it is ``True``.

The rule, in reporting order:

1. the pressure reading is not a measurement (the latest sample failed, or
   none has succeeded) → hold, ``no-sample``;
2. the sample is older than ``stale_after_s``, or stamped in the future → hold,
   ``stale``;
3. the pressure policy sheds a full-tier request → hold, ``shed:<signal>``
   (``swap``, ``thermal`` or ``iowait`` on ``/status``; the policy always names
   a signal today, and plain ``shed`` is kept for one that does not);
4. no backend is healthy, which includes a fleet with none → hold,
   ``unreachable``;
5. any backend reports requests waiting in its queue → hold, ``waiting``;
6. otherwise → admit.

``admit`` is never ``True`` while ``pressure.shed`` is ``True``. ``at`` is the
time of the pressure sample behind the answer when the caller knows it, so a
reader's own ``stale_after_s`` check also catches a stalled sampler; when the
caller does not know it, ``at`` is the time the status was computed. The rule
cannot see the queue of a backend that exports no in-flight counts
(``busy.partial``).

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
REASON_UNREACHABLE = "unreachable"
REASON_STALE = "stale"
REASON_NO_SAMPLE = "no-sample"

# "The caller did not say": distinct from None, which means "no sample succeeded".
SAMPLE_TIME_UNKNOWN = object()


def rfc3339_utc(now: float) -> str:
    """Epoch seconds *now* as RFC 3339 in UTC, to the second (``...Z``)."""
    return datetime.fromtimestamp(int(now), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _reason(*, shed: bool, shed_signal: str, waiting: int, healthy: int | None, sample: str) -> str:
    """Why the gateway holds, or ``""`` when it admits. See the module docstring."""
    if sample in (REASON_NO_SAMPLE, REASON_STALE):
        return sample
    if shed:
        return f"{REASON_SHED}:{shed_signal}" if shed_signal else REASON_SHED
    if healthy == 0:
        return REASON_UNREACHABLE
    if waiting > 0:
        return REASON_WAITING
    return ""


def _judge_sample(sampled_at: object, now: float) -> tuple[str, float]:
    """``(verdict, stamp)`` for the pressure sample behind the answer.

    The verdict is ``""`` (trusted, or the caller could not say), ``no-sample``
    or ``stale``. A sample stamped AFTER *now* is stale too: the clock stepped
    back, so the stamp says nothing about the sample's age.
    """
    if sampled_at is SAMPLE_TIME_UNKNOWN:
        return "", now
    if isinstance(sampled_at, bool) or not isinstance(sampled_at, (int, float)):
        return REASON_NO_SAMPLE, now  # None, or not a time at all
    age = now - float(sampled_at)
    if age < 0 or age > STALE_AFTER_S:
        return REASON_STALE, float(sampled_at)
    return "", float(sampled_at)


def admission(
    *,
    shed: bool,
    shed_signal: str,
    waiting: int,
    now: float,
    healthy: int | None = None,
    sampled_at: object = SAMPLE_TIME_UNKNOWN,
) -> dict:
    """The ``/status`` admission fields for one computed status.

    *shed* and *shed_signal* come from the pressure decision ``/status``
    already makes for a full tier, *waiting* is the fleet's aggregate waiting
    count, *now* is the epoch-seconds instant the status was computed.
    *healthy* is how many backends answered healthy (``None``: not known).
    *sampled_at* is the epoch seconds of the sample the pressure reading came
    from, ``None`` when that reading is not a measurement, and left out when
    the caller cannot say.
    """
    sample, stamp = _judge_sample(sampled_at, now)
    reason = _reason(
        shed=shed, shed_signal=shed_signal, waiting=waiting, healthy=healthy, sample=sample
    )
    fields: dict = {
        "admit": not reason,
        "at": rfc3339_utc(stamp),
        "stale_after_s": STALE_AFTER_S,
    }
    if reason:
        fields["reason"] = reason
    return fields
