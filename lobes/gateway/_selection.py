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
c24/h16, c35):

1. A candidate is SELECTABLE when it is `compatible` and `ready` and not
   `busy`. When `local_busy` is True (the caller's own pressure policy has
   already decided this box is shedding), the local candidate is additionally
   excluded from selectability — a busy box never picks itself even if its
   own `busy` flag happens to read False in a stale snapshot.
2. No selectable candidate -> `Selection(None, False, "none")`.
3. Exactly one selectable candidate wins with reason `"sole-ready"`, with one
   carve-out: if that candidate is local, idle (`running + waiting == 0`),
   and it is the ONLY candidate in the whole input (a no-pool / no-peers-
   declared box), the reason is `"local-idle"` instead — this keeps a
   single-box deployment's marker identical to what a genuinely idle local
   replica would report once a peer exists.
4. Otherwise every selectable candidate is ranked by `estimated_wait`
   ascending; ties are broken by locality (`local` first) and then by
   `origin` string ascending, so ranking is fully deterministic. The winner
   of that ranking is the "best" availability candidate. Its reason is the
   most specific of, in priority order:
     - `"local-busy-forwarded"` — best is a peer and `local_busy` is True.
     - `"local-idle"` — best is local and its `estimated_wait` is exactly 0.
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
   `estimated_wait` is no worse than the availability winner's by more than
   `affinity_margin`. Otherwise availability wins outright and the reason is
   whatever step 4 produced. An absent or empty affinity key makes selection
   purely availability-driven (steps 2-4 only).

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


@runtime_checkable
class ReplicaLike(Protocol):
    """Structural shape a candidate must satisfy. Duck-typed deliberately —
    this module never imports the concrete dataclass a sibling task defines
    (`lobes/gateway/_replicas.py`), so it has no ordering dependency on it.
    """

    origin: str
    local: bool
    ready: bool
    busy: bool
    compatible: bool
    running: int
    waiting: int
    weight: float


@dataclass(frozen=True)
class Selection:
    """The outcome of a selection: which origin (if any) should serve the
    request, whether it is this box's own local replica, and why."""

    origin: Optional[str]
    local: bool
    reason: str


def estimated_wait(replica: ReplicaLike) -> float:
    """Estimated queue wait for a replica: (running + waiting) / weight.

    `weight` is the replica's declared decode weight (default 1.0,
    higher is faster/more capacity) — a first-cut proxy for capacity per
    #199's open park on calibrating this later from measurement. Floored
    at `_WEIGHT_EPSILON` so a misdeclared zero/negative weight never
    divides by zero or flips the sign of the estimate.
    """

    weight = replica.weight if replica.weight > _WEIGHT_EPSILON else _WEIGHT_EPSILON
    return (replica.running + replica.waiting) / weight


def _is_selectable(replica: ReplicaLike, *, local_busy: bool) -> bool:
    if not (replica.compatible and replica.ready and not replica.busy):
        return False
    if local_busy and replica.local:
        return False
    return True


def _rank_key(replica: ReplicaLike) -> tuple:
    # Ascending estimated_wait; local (True) sorts before peer (False) on
    # a tie, so invert the boolean; then origin ascending for full
    # determinism when both wait and locality tie.
    return (estimated_wait(replica), not replica.local, replica.origin)


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


def _rank(selectable: Sequence[ReplicaLike]) -> list[ReplicaLike]:
    """*selectable*, ordered by :func:`_rank_key` (step 4's ranking)."""
    return sorted(selectable, key=_rank_key)


def _sole_reason(
    only: ReplicaLike,
    candidates: Sequence[ReplicaLike],
    *,
    local_busy: bool,
    has_local_input: bool,
) -> str:
    """The reason for the exactly-one-selectable case (step 3)."""
    if only.local:
        if len(candidates) == 1 and only.running + only.waiting == 0:
            return REASON_LOCAL_IDLE
        return REASON_SOLE_READY
    if local_busy and has_local_input:
        return REASON_LOCAL_BUSY_FORWARDED
    return REASON_SOLE_READY


def _reason_for(best: ReplicaLike, *, local_busy: bool) -> str:
    """The availability-ranking reason for *best* (step 4, pre-affinity)."""
    if not best.local and local_busy:
        return REASON_LOCAL_BUSY_FORWARDED
    if best.local and estimated_wait(best) == 0:
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
) -> Selection:
    """Apply step 5's affinity override on top of the availability winner."""
    if affinity:
        preferred = _preferred_by_affinity(selectable, affinity)
        if preferred is not None:
            if estimated_wait(preferred) - estimated_wait(best) <= affinity_margin:
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
    ranked = _rank(selectable)
    best = ranked[0]

    if len(selectable) == 1:
        reason = _sole_reason(
            best, candidates, local_busy=local_busy, has_local_input=has_local_input
        )
        return Selection(best.origin, best.local, reason)

    best_reason = _reason_for(best, local_busy=local_busy)
    return _affinity_pick(
        selectable, best, best_reason, affinity=affinity, affinity_margin=affinity_margin
    )
