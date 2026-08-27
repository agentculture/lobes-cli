"""Tests for the pure replica-selection policy (lobes/gateway/_selection.py).

Task t5 of docs/plans/2026-08-25-cortex-replica-pool-199.md. Uses a local
frozen dataclass satisfying `ReplicaLike` rather than importing the sibling
task's `_replicas.py` module (which may not exist on this branch).
"""

from dataclasses import dataclass

from lobes.gateway._selection import (
    REASON_AFFINITY,
    REASON_LOCAL_BUSY_FORWARDED,
    REASON_LOCAL_IDLE,
    REASON_NONE,
    REASON_PEER_LESS_LOADED,
    REASON_SOLE_READY,
    Selection,
    estimated_wait,
    select_replica,
)


@dataclass(frozen=True)
class FakeReplica:
    origin: str
    local: bool
    ready: bool
    busy: bool
    compatible: bool
    running: int
    waiting: int
    weight: float = 1.0


def local(origin="local", **kw):
    defaults = dict(
        origin=origin,
        local=True,
        ready=True,
        busy=False,
        compatible=True,
        running=0,
        waiting=0,
        weight=1.0,
    )
    defaults.update(kw)
    return FakeReplica(**defaults)


def peer(origin="peer", **kw):
    defaults = dict(
        origin=origin,
        local=False,
        ready=True,
        busy=False,
        compatible=True,
        running=0,
        waiting=0,
        weight=1.0,
    )
    defaults.update(kw)
    return FakeReplica(**defaults)


class TestBasicAvailability:
    def test_local_idle_vs_peer_idle_prefers_local(self):
        result = select_replica([local(), peer()])
        assert result == Selection("local", True, REASON_LOCAL_IDLE)

    def test_local_loaded_vs_peer_idle_picks_peer(self):
        result = select_replica([local(running=3, waiting=2), peer(running=0, waiting=0)])
        assert result == Selection("peer", False, REASON_PEER_LESS_LOADED)

    def test_local_busy_forwards_to_ready_peer(self):
        result = select_replica(
            [local(running=0, waiting=0), peer(running=0, waiting=0)],
            local_busy=True,
        )
        assert result == Selection("peer", False, REASON_LOCAL_BUSY_FORWARDED)

    def test_no_selectable_candidates_returns_none(self):
        result = select_replica(
            [local(ready=False), peer(busy=True)],
        )
        assert result == Selection(None, False, REASON_NONE)

    def test_incompatible_candidates_never_selected(self):
        result = select_replica(
            [local(compatible=False), peer()],
        )
        assert result == Selection("peer", False, REASON_SOLE_READY)

    def test_unready_candidates_never_selected(self):
        # Two candidates in the input but only one selectable -> sole-ready
        # (local-idle is reserved for a genuinely single-candidate input,
        # i.e. no pool declared at all).
        result = select_replica(
            [local(), peer(ready=False)],
        )
        assert result == Selection("local", True, REASON_SOLE_READY)

    def test_sole_ready_peer_when_local_absent(self):
        result = select_replica([peer(running=1, waiting=0)])
        assert result == Selection("peer", False, REASON_SOLE_READY)

    def test_sole_ready_local_but_not_no_pool_input_is_still_local_idle(self):
        # Only one candidate at all (no pool declared) and it is idle ->
        # local-idle, matching a genuinely single-box deployment.
        result = select_replica([local()])
        assert result == Selection("local", True, REASON_LOCAL_IDLE)

    def test_sole_ready_local_loaded_alone_is_sole_ready(self):
        result = select_replica([local(running=2, waiting=0)])
        assert result == Selection("local", True, REASON_SOLE_READY)


class TestWeighting:
    def test_estimated_wait_divides_by_weight(self):
        r = FakeReplica(
            origin="x",
            local=False,
            ready=True,
            busy=False,
            compatible=True,
            running=3,
            waiting=1,
            weight=2.0,
        )
        assert estimated_wait(r) == 2.0

    def test_zero_weight_does_not_explode(self):
        r = FakeReplica(
            origin="x",
            local=False,
            ready=True,
            busy=False,
            compatible=True,
            running=1,
            waiting=0,
            weight=0.0,
        )
        # Should not raise, and should be a very large (but finite) number.
        assert estimated_wait(r) > 1e6

    def test_higher_weight_peer_beats_lower_weight_local(self):
        # peer: (3+0)/2.0 = 1.5 ; local: (2+0)/1.0 = 2.0 -> peer wins
        result = select_replica(
            [
                local(running=2, waiting=0, weight=1.0),
                peer(running=3, waiting=0, weight=2.0),
            ]
        )
        assert result == Selection("peer", False, REASON_PEER_LESS_LOADED)


class TestAffinity:
    def test_affinity_within_margin_wins_over_availability(self):
        candidates = [
            local(running=0, waiting=0),
            peer(origin="peer-a", running=0, waiting=1),
            peer(origin="peer-b", running=0, waiting=1),
        ]
        # Find whichever key makes a peer preferred within margin, to prove
        # the affinity path can override the plain-availability winner
        # (local-idle) when the margin allows it.
        chosen_key = None
        for i in range(200):
            key = f"key-{i}"
            baseline = select_replica(candidates)
            result = select_replica(candidates, affinity=key, affinity_margin=5.0)
            if result.reason == REASON_AFFINITY and result.origin != baseline.origin:
                chosen_key = key
                break
        assert chosen_key is not None, "expected some key to prefer a non-baseline replica"

    def test_affinity_preferred_busy_falls_back_to_availability(self):
        # Force a specific preferred origin among two selectable peers
        # (deliberately no local candidate, to keep the hash comparison
        # confined to the two peers under test) by brute-forcing a key
        # that prefers "peer-busy", then re-run with that replica marked
        # busy/not selectable and confirm the reason is no longer affinity.
        selectable_pair = [peer(origin="peer-busy"), peer(origin="peer-idle")]
        preferred_key = None
        for i in range(200):
            key = f"aff-{i}"
            result = select_replica(selectable_pair, affinity=key, affinity_margin=1e9)
            if result.origin == "peer-busy" and result.reason == REASON_AFFINITY:
                preferred_key = key
                break
        assert preferred_key is not None

        with_busy = select_replica(
            [peer(origin="peer-busy", busy=True), peer(origin="peer-idle")],
            affinity=preferred_key,
            affinity_margin=1e9,
        )
        assert with_busy.reason != REASON_AFFINITY
        assert with_busy.origin != "peer-busy"

    def test_affinity_preferred_not_ready_falls_back_to_availability(self):
        selectable_pair = [peer(origin="peer-a"), peer(origin="peer-b")]
        preferred_key = None
        preferred_origin = None
        for i in range(200):
            key = f"nr-{i}"
            result = select_replica(selectable_pair, affinity=key, affinity_margin=1e9)
            if result.reason == REASON_AFFINITY:
                preferred_key = key
                preferred_origin = result.origin
                break
        assert preferred_key is not None

        other_origin = "peer-b" if preferred_origin == "peer-a" else "peer-a"
        candidates = [
            peer(origin=preferred_origin, ready=False),
            peer(origin=other_origin),
        ]
        result = select_replica(candidates, affinity=preferred_key, affinity_margin=1e9)
        assert result.reason != REASON_AFFINITY
        assert result.origin == other_origin

    def test_absent_affinity_key_is_purely_availability_driven(self):
        candidates = [local(running=1, waiting=0), peer(running=0, waiting=0)]
        without = select_replica(candidates)
        with_none = select_replica(candidates, affinity=None)
        with_empty = select_replica(candidates, affinity="")
        assert without == with_none == with_empty

    def test_affinity_outside_margin_availability_wins(self):
        # Two peers only (no local candidate), so the affinity preference is
        # unambiguous between the two, and one is much slower than the
        # other -- a tiny margin should never honour a preference for it.
        candidates = [
            peer(origin="peer-fast", running=0, waiting=0),
            peer(origin="peer-slow", running=5, waiting=5),
        ]
        preferred_slow_key = None
        for i in range(200):
            key = f"margin-{i}"
            result = select_replica(candidates, affinity=key, affinity_margin=1e9)
            if result.origin == "peer-slow" and result.reason == REASON_AFFINITY:
                preferred_slow_key = key
                break
        assert preferred_slow_key is not None

        tight = select_replica(candidates, affinity=preferred_slow_key, affinity_margin=0.01)
        assert tight.reason != REASON_AFFINITY
        assert tight.origin == "peer-fast"


class TestRendezvousStability:
    def test_adding_a_third_replica_does_not_move_existing_preference(self):
        two = [peer(origin="peer-a"), peer(origin="peer-b")]
        key = "sticky-key"
        before = select_replica(two, affinity=key, affinity_margin=1e9)

        three = two + [peer(origin="peer-c")]
        after = select_replica(three, affinity=key, affinity_margin=1e9)

        # Either the new replica wins the hash for this key (valid, since
        # HRW lets a newcomer win), or the original preference is unchanged.
        assert after.origin in (before.origin, "peer-c")

    def test_removing_the_non_preferred_replica_keeps_preference(self):
        two = [peer(origin="peer-a"), peer(origin="peer-b")]
        key = "sticky-key-2"
        before = select_replica(two, affinity=key, affinity_margin=1e9)

        remaining = [c for c in two if c.origin != before.origin]
        one_left = [next(c for c in two if c.origin == before.origin)]
        after = select_replica(one_left, affinity=key, affinity_margin=1e9)

        assert after.origin == before.origin
        assert remaining  # sanity: we actually removed the other one


class TestDeterminism:
    def test_same_input_yields_same_output_across_many_calls(self):
        candidates = [
            local(running=2, waiting=1, weight=1.0),
            peer(origin="peer-a", running=1, waiting=0, weight=1.5),
            peer(origin="peer-b", running=0, waiting=0, weight=1.0),
        ]
        results = {
            select_replica(candidates, affinity="stable-key", affinity_margin=0.5)
            for _ in range(100)
        }
        assert len(results) == 1
