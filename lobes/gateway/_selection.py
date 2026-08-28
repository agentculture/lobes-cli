"""Pure, deterministic replica-selection policy for the cortex replica pool (#199).

Why: the pool (issue #199) lets a request addressed to a pooled role (today,
`cortex` only) be served by whichever compatible replica — local or a
declared peer — is least loaded, instead of always dialing the local
backend or the single proxy target `_proxied_owner` picks today. This
module is the SELECTION function only: given a snapshot of replica state,
it decides which one (if any) should serve the request and WHY. It has no
knowledge of HTTP, sockets, the readiness cache, docker, or wall-clock
time — the caller (a future `_replicas.py` / `server.py` integration,
tasks t4/t7/t8) is responsible for building the candidate list from a
cached snapshot and for actually dispatching to the chosen origin.

Policy (see docs/specs/2026-08-25-cortex-replica-pool-199.md, c3/h3, c37,
c24/h16, c35; amended by docs/specs/2026-08-27-capacity-relative-pool-routing.md,
c2/h2, c5/h4, c16/h10, c22/h15, c24/h17):

1. A candidate is SELECTABLE when it is `compatible` and `ready` and NOT FULL.
   "Full" is capacity-relative: a candidate that declares a calibrated
   capacity (see `selection_capacity` below) is full once its active count
   (`running + waiting`) REACHES that capacity. A candidate that declares no
   calibrated capacity is never full — an unpublished capacity must never
   make a replica unselectable.

   A PEER's `busy` flag is deliberately NOT part of this gate any more. That
   flag is the peer's HOST-level pressure verdict (swap/iowait), and host
   iowait is not a proxy for serving capacity: a fully idle DGX Spark
   (`running=0, waiting=0`) was excluded from the pool for a ~60% iowait
   reading traced to a single sleeping desktop terminal with zero block I/O
   ever charged. Capacity utilisation is the honest signal; the flag stays on
   the protocol as informational state other modules still consume.

   The LOCAL SHED DECISION stays separate and is unchanged. It is a
   first-party verdict — this box's own pressure policy deciding this box
   should stop taking work — and it reaches selection two ways that are
   treated identically: the `local_busy` argument, and the local candidate's
   own `busy` flag. Either one excludes the LOCAL candidate and only the
   local candidate. Under genuine local pressure this box therefore forwards
   to a peer with spare capacity rather than shedding (q3); a 429 remains
   only when no replica anywhere is selectable. The asymmetry is the whole
   point: a pressure verdict is authoritative about the box that MADE it and
   says nothing about a box across the network.
2. No selectable candidate -> `Selection(None, False, "none")`.
3. Exactly one selectable candidate wins with reason `"sole-ready"`, with two
   carve-outs. If that candidate is a PEER and the local candidate was
   excluded for being FULL, the reason is `"peer-less-loaded"` — the local
   replica lost the utilisation comparison, which is a load decision and not
   an absence of alternatives. And if that candidate is local, idle (`running + waiting == 0`),
   and it is the ONLY candidate in the whole input (a no-pool / no-peers-
   declared box), the reason is `"local-idle"` instead — this keeps a
   single-box deployment's marker identical to what a genuinely idle local
   replica would report once a peer exists.
4. Otherwise every selectable candidate is ranked by its CAPACITY-RELATIVE
   wait — `(running + waiting) / capacity`, i.e. utilisation — ascending;
   ties are broken by locality (`local` first) and then by `origin` string
   ascending, so ranking is fully deterministic. The winner of that ranking
   is the "best" availability candidate. Its reason is the most specific of,
   in priority order:
     - `"local-busy-forwarded"` — best is a peer and `local_busy` is True.
     - `"local-idle"` — best is local and its wait is exactly 0.
     - `"peer-less-loaded"` — best is a peer, a local candidate is present
       in the input (selectable or not), and the peer won on load.
     - otherwise, the availability winner is returned as-is (this only
       happens when there is no local candidate in the input at all, i.e. a
       peer-only view) with reason `"peer-less-loaded"` still — the fallback
       vocabulary is deliberately just these plus the two below.
5. Affinity: when `affinity` is given as a non-empty string, a preferred
   replica is computed with rendezvous (highest-random-weight, here
   lowest-crc32) hashing over the SELECTABLE set: the candidate minimizing
   `zlib.crc32(f"{affinity}|{origin}")`. This makes the preference sticky to
   a (key, origin) pair — independent of how many other replicas are in the
   set, so a replica joining or leaving never perturbs an existing key's
   preference among the replicas that were already present, unless the
   newcomer itself wins that key's hash. The preferred replica is honoured
   (returned with reason `"affinity"`) only when it is selectable AND its
   capacity-relative wait is no worse than the availability winner's by more
   than `affinity_margin`. Otherwise availability wins outright and the
   reason is whatever step 4 produced. An absent or empty affinity key makes
   selection purely availability-driven (steps 2-4 only).

Capacity and the uncalibrated fallback
--------------------------------------

`weight` is the capacity carrier: a replica's measured max active requests
(the throughput knee, NOT the `--max-num-seqs` OOM cap). `1.0` is the value
`_replicas.py` / `roles.py` leave in place when nothing was published (an
older lobes, or a non-lobes replica) — but whether a capacity was published
is carried by the SEPARATE `calibrated` flag, never inferred from the weight.
That separation is load-bearing: `lobes calibrate` can validly measure and
persist a knee of 1 (a box whose engine admits one request at a time has
exactly that capacity), and reading every 1.0 as "nothing published" made
such a replica never full and ranked it as uncalibrated, so its measured
capacity did not affect routing at all.

Reading an UNPUBLISHED capacity as a real one-slot capacity is the opposite
mistake, and equally wrong: it would rank an uncalibrated peer 8x worse than
a calibrated weight-8 peer at the same single active request (1/1.0 vs 1/8),
silently draining a mixed-version fleet toward whichever boxes happen to be
calibrated. So an uncalibrated replica ranks at a NEUTRAL capacity instead:
the median of the calibrated capacities present in the same candidate set,
falling back to `_DEFAULT_NEUTRAL_CAPACITY` when nothing in the set is
calibrated (in which case every replica shares one capacity and the ranking
is byte-identical to the pre-capacity policy). A non-positive weight is a
misdeclaration, not a capacity, and is treated as uncalibrated too.

The neutral capacity is used for RANKING only. The fullness gate in step 1
uses a replica's OWN calibrated capacity and nothing else, so a replica is
never declared full on the strength of a capacity it borrowed from a peer.

Reason vocabulary (the closed `X-Lobes-Route-Reason` set)
---------------------------------------------------------

The set is UNCHANGED in membership — `local-idle | peer-less-loaded |
local-busy-forwarded | affinity | sole-ready | none` — but
`"peer-less-loaded"` is REDEFINED: it now means "the peer is less loaded
RELATIVE TO ITS OWN CAPACITY" (lower utilisation), where it previously meant
"the peer has fewer active requests". A peer with MORE active requests than
this box can now legitimately win with that reason, because it has more
headroom. `"peer-less-loaded"` is also the reason emitted when the local
replica is excluded for being FULL (a state that previously could only arise
from a pressure verdict, which reported `"local-busy-forwarded"`). No new
reason value is introduced, so a caller parsing the old closed set keeps
parsing successfully — but the MEANING of one member changed, and that is a
contract change (t8 documents it where the set is declared, in
`lobes/gateway/server.py`).

Deliberately NOT done here (parked, #128): no learned/EMA routing, no
latency history, no clock or wall-time input, no I/O, no retries, no
knowledge of HTTP or the readiness cache. `select_replica` is a pure
function of its arguments — same input always yields the same output,
which is exactly what the acceptance criteria in t5 require.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

REASON_LOCAL_IDLE = "local-idle"
REASON_PEER_LESS_LOADED = "peer-less-loaded"
REASON_LOCAL_BUSY_FORWARDED = "local-busy-forwarded"
REASON_AFFINITY = "affinity"
REASON_SOLE_READY = "sole-ready"
REASON_NONE = "none"

# Floor to avoid division by zero when a replica declares weight <= 0.
_WEIGHT_EPSILON = 1e-9

# The weight a replica carries when NO capacity has been published for it.
# It is a FALLBACK VALUE only: "was a capacity published?" is answered by the
# replica's own `calibrated` flag, never by comparing against this number —
# see the module docstring's "Capacity and the uncalibrated fallback".
UNCALIBRATED_WEIGHT = 1.0

# Neutral capacity for an uncalibrated replica when NOTHING in the candidate
# set is calibrated either. Any fixed value reproduces the pre-capacity
# ranking exactly in that case (every replica divides by the same number), so
# the sentinel's own value is kept for continuity.
_DEFAULT_NEUTRAL_CAPACITY = 1.0


@runtime_checkable
class ReplicaLike(Protocol):
    """Structural shape a candidate must satisfy. Duck-typed deliberately —
    this module never imports the concrete dataclass a sibling task defines
    (`lobes/gateway/_replicas.py`), so it has no ordering dependency on it.
    """

    origin: str
    local: bool
    ready: bool
    # The replica's HOST-level pressure verdict. Kept on the protocol because
    # other modules (status surfaces, the local shed decision) consume it —
    # but selection no longer gates on it; see the module docstring, step 1.
    busy: bool
    compatible: bool
    running: int
    waiting: int
    # Capacity carrier: measured max active requests. Meaningful only when
    # `calibrated` is True; otherwise it is the `UNCALIBRATED_WEIGHT`
    # fallback and this module substitutes a neutral capacity for ranking.
    weight: float
    # Whether `weight` is a capacity actually IN FORCE for this replica. Set
    # at ingest by the producer (`_replicas.ReplicaState`), which is the only
    # place that knows whether anything was published, whether the kill switch
    # is engaged, and whether a pinned capacity was discarded on a fingerprint
    # change. A measured capacity of exactly 1 is legal and sets this True.
    calibrated: bool


@dataclass(frozen=True)
class Selection:
    """The outcome of a selection: which origin (if any) should serve the
    request, whether it is this box's own local replica, and why."""

    origin: Optional[str]
    local: bool
    reason: str


def estimated_wait(replica: ReplicaLike, capacity: Optional[float] = None) -> float:
    """Estimated queue wait for a replica: (running + waiting) / capacity.

    The arithmetic is unchanged from #199: it is exactly `active / capacity`,
    i.e. utilisation. `capacity` defaults to the replica's own `weight` (its
    declared max active requests) so single-argument callers behave exactly as
    before; the ranking path passes the capacity resolved by
    :func:`selection_capacity`, which substitutes a neutral value for an
    UNCALIBRATED replica. Floored at `_WEIGHT_EPSILON` so a misdeclared
    zero/negative capacity never divides by zero or flips the sign.
    """

    resolved = replica.weight if capacity is None else capacity
    if not resolved > _WEIGHT_EPSILON:
        resolved = _WEIGHT_EPSILON
    return (replica.running + replica.waiting) / resolved


def _active(replica: ReplicaLike) -> int:
    return replica.running + replica.waiting


def is_calibrated(replica: ReplicaLike) -> bool:
    """True when *replica* carries a real measured capacity.

    Two independent conditions, and the weight's VALUE is deliberately not one
    of them: the producer must have declared a capacity in force
    (`calibrated`), and that capacity must be usable arithmetic (a
    non-positive weight is a misdeclaration, not a capacity). A published
    capacity of exactly one slot therefore reads as calibrated — see the
    module docstring.
    """

    return replica.calibrated and replica.weight > _WEIGHT_EPSILON


def neutral_capacity(candidates: Sequence[ReplicaLike]) -> float:
    """The capacity an UNCALIBRATED replica ranks at: the median of the
    calibrated capacities present in *candidates*, or
    `_DEFAULT_NEUTRAL_CAPACITY` when none of them is calibrated."""

    known = sorted(c.weight for c in candidates if is_calibrated(c))
    if not known:
        return _DEFAULT_NEUTRAL_CAPACITY
    mid = len(known) // 2
    if len(known) % 2:
        return known[mid]
    return (known[mid - 1] + known[mid]) / 2.0


def selection_capacity(replica: ReplicaLike, candidates: Sequence[ReplicaLike]) -> float:
    """The capacity used to RANK *replica* within *candidates* — its own when
    calibrated, the neutral fallback otherwise."""

    if is_calibrated(replica):
        return replica.weight
    return neutral_capacity(candidates)


def selection_wait(replica: ReplicaLike, candidates: Sequence[ReplicaLike]) -> float:
    """The capacity-relative wait (utilisation) selection actually ranks by.

    Exposed so a caller can report the utilisation behind a routing decision
    (c24/h17) without re-deriving the neutral fallback.
    """

    return estimated_wait(replica, selection_capacity(replica, candidates))


def is_full(replica: ReplicaLike) -> bool:
    """True when *replica*'s active count has REACHED its own calibrated
    capacity. An uncalibrated replica is never full (h3)."""

    return is_calibrated(replica) and _active(replica) >= replica.weight


def _is_selectable(replica: ReplicaLike, *, local_busy: bool) -> bool:
    if not (replica.compatible and replica.ready):
        return False
    if is_full(replica):
        return False
    if replica.local and (local_busy or replica.busy):
        # First-party shed verdict about THIS box. A peer's identical flag is
        # ignored — see the module docstring, step 1.
        return False
    return True


def _rank_key(replica: ReplicaLike, neutral: float) -> tuple:
    # Ascending capacity-relative wait; local (True) sorts before peer
    # (False) on a tie, so invert the boolean; then origin ascending for full
    # determinism when both wait and locality tie.
    capacity = replica.weight if is_calibrated(replica) else neutral
    return (estimated_wait(replica, capacity), not replica.local, replica.origin)


def _preferred_by_affinity(
    selectable: Sequence[ReplicaLike], affinity: str
) -> Optional[ReplicaLike]:
    if not selectable:
        return None
    best = None
    best_hash = None
    for replica in selectable:
        digest = zlib.crc32(f"{affinity}|{replica.origin}".encode("utf-8"))
        if (
            best_hash is None
            or digest < best_hash
            or (digest == best_hash and (best is None or replica.origin < best.origin))
        ):
            best_hash = digest
            best = replica
    return best


def _selectable(candidates: Sequence[ReplicaLike], *, local_busy: bool) -> list[ReplicaLike]:
    """The subset of *candidates* eligible to serve at all (step 1)."""
    return [c for c in candidates if _is_selectable(c, local_busy=local_busy)]


def _rank(selectable: Sequence[ReplicaLike], neutral: float) -> list[ReplicaLike]:
    """*selectable*, ordered by :func:`_rank_key` (step 4's ranking)."""
    return sorted(selectable, key=lambda r: _rank_key(r, neutral))


def _sole_reason(
    only: ReplicaLike,
    candidates: Sequence[ReplicaLike],
    *,
    local_busy: bool,
    has_local_input: bool,
) -> str:
    """The reason for the exactly-one-selectable case (step 3)."""
    if only.local:
        if len(candidates) == 1 and _active(only) == 0:
            return REASON_LOCAL_IDLE
        return REASON_SOLE_READY
    if local_busy and has_local_input:
        return REASON_LOCAL_BUSY_FORWARDED
    if any(c.local and is_full(c) for c in candidates):
        # The local replica was dropped on capacity, not on absence: this is
        # a load decision, so report it as one.
        return REASON_PEER_LESS_LOADED
    return REASON_SOLE_READY


def _reason_for(best: ReplicaLike, *, local_busy: bool) -> str:
    """The availability-ranking reason for *best* (step 4, pre-affinity)."""
    if not best.local and local_busy:
        return REASON_LOCAL_BUSY_FORWARDED
    if best.local and _active(best) == 0:
        return REASON_LOCAL_IDLE
    if not best.local:
        return REASON_PEER_LESS_LOADED
    return REASON_SOLE_READY


def _affinity_pick(
    selectable: Sequence[ReplicaLike],
    best: ReplicaLike,
    best_reason: str,
    *,
    affinity: Optional[str],
    affinity_margin: float,
    neutral: float,
) -> Selection:
    """Apply step 5's affinity override on top of the availability winner."""
    if affinity:
        preferred = _preferred_by_affinity(selectable, affinity)
        if preferred is not None:
            preferred_wait = estimated_wait(
                preferred, preferred.weight if is_calibrated(preferred) else neutral
            )
            best_wait = estimated_wait(best, best.weight if is_calibrated(best) else neutral)
            if preferred_wait - best_wait <= affinity_margin:
                return Selection(preferred.origin, preferred.local, REASON_AFFINITY)
    return Selection(best.origin, best.local, best_reason)


def select_replica(
    candidates: Sequence[ReplicaLike],
    *,
    affinity: Optional[str] = None,
    local_busy: bool = False,
    affinity_margin: float = 1.0,
) -> Selection:
    """Deterministically choose which replica (if any) should serve a
    pooled request. See the module docstring for the full policy."""

    selectable = _selectable(candidates, local_busy=local_busy)

    if not selectable:
        return Selection(None, False, REASON_NONE)

    has_local_input = any(c.local for c in candidates)
    neutral = neutral_capacity(candidates)
    ranked = _rank(selectable, neutral)
    best = ranked[0]

    if len(selectable) == 1:
        reason = _sole_reason(
            best, candidates, local_busy=local_busy, has_local_input=has_local_input
        )
        return Selection(best.origin, best.local, reason)

    best_reason = _reason_for(best, local_busy=local_busy)
    return _affinity_pick(
        selectable,
        best,
        best_reason,
        affinity=affinity,
        affinity_margin=affinity_margin,
        neutral=neutral,
    )
