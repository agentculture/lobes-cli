"""Tests for the pure replica-selection policy (lobes/gateway/_selection.py).

Task t5 of docs/plans/2026-08-25-cortex-replica-pool-199.md. Uses a local
frozen dataclass satisfying `ReplicaLike` rather than importing the sibling
task's `_replicas.py` module (which may not exist on this branch).

Task t3 of docs/plans/2026-08-27-capacity-relative-pool-routing.md rewrote the
selectability gate to be capacity-relative. Three cases below changed with it,
each because it encoded an assumption the capacity work deliberately retires:

* `test_no_selectable_candidates_returns_none` used a pressure-`busy` peer to
  produce an empty selectable set; a pressure verdict no longer excludes an
  idle replica, so the case now uses a genuinely FULL peer.
* `test_affinity_preferred_busy_falls_back_to_availability` became
  `..._full_...` for the same reason.
* `test_higher_weight_peer_beats_lower_weight_local` pitted the uncalibrated
  sentinel weight 1.0 against a calibrated 2.0, i.e. it read the sentinel as a
  measured capacity of one slot -- exactly what c22/h15 forbids. It now pits
  two CALIBRATED capacities against each other, preserving the intent.
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
    _is_selectable,
    estimated_wait,
    is_calibrated,
    is_full,
    select_replica,
    selection_wait,
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
    # Whether `weight` is a capacity actually IN FORCE for this replica. The
    # producer (`_replicas.ReplicaState`) sets it at ingest; it exists because
    # a measured capacity of exactly one slot is legal and must not be read
    # back as "nothing published" (F2).
    calibrated: bool = False


def _fake(origin, is_local, kw):
    """Build a `FakeReplica`, defaulting `calibrated` from `weight`.

    Every pre-F2 case in this file expressed "calibrated" by passing a weight
    other than 1.0 and "uncalibrated" by leaving the 1.0 default, so that is
    the default derivation here — the intent of those cases is preserved
    verbatim. A case that needs the two decoupled (a MEASURED one-slot
    capacity, or a declared-then-discarded one) passes `calibrated` itself.
    """
    defaults = dict(
        origin=origin,
        local=is_local,
        ready=True,
        busy=False,
        compatible=True,
        running=0,
        waiting=0,
        weight=1.0,
    )
    defaults.update(kw)
    defaults.setdefault("calibrated", defaults["weight"] != 1.0)
    return FakeReplica(**defaults)


def local(origin="local", **kw):
    return _fake(origin, True, kw)


def peer(origin="peer", **kw):
    return _fake(origin, False, kw)


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
        # A not-ready local and a peer whose engine is genuinely full
        # (calibrated capacity 2, two active) -> nothing to select.
        result = select_replica(
            [local(ready=False), peer(weight=2.0, running=2)],
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
        # Both CALIBRATED (neither is the 1.0 sentinel), and both have
        # headroom, so the ranking is a straight utilisation comparison:
        # peer: (4+0)/8.0 = 0.5 ; local: (3+0)/4.0 = 0.75 -> peer wins.
        result = select_replica(
            [
                local(running=3, waiting=0, weight=4.0),
                peer(running=4, waiting=0, weight=8.0),
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

    def test_affinity_preferred_full_falls_back_to_availability(self):
        # Force a specific preferred origin among two selectable peers
        # (deliberately no local candidate, to keep the hash comparison
        # confined to the two peers under test) by brute-forcing a key
        # that prefers "peer-busy", then re-run with that replica FULL
        # (not selectable) and confirm the reason is no longer affinity.
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
            [
                peer(origin="peer-busy", weight=2.0, running=2, busy=True),
                peer(origin="peer-idle"),
            ],
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


def _old_is_selectable(replica, *, local_busy: bool = False) -> bool:
    """The PRE-t3 selectability gate, kept verbatim as a regression witness.

    This is the gate that excluded a fully idle DGX Spark (running=0,
    waiting=0) from the pool because a sleeping desktop terminal pushed the
    host's iowait reading over the pressure threshold. It is reproduced here
    (not imported) so the before-state stays provable after the real gate
    changed.
    """

    if not (replica.compatible and replica.ready and not replica.busy):
        return False
    if local_busy and replica.local:
        return False
    return True


class TestPressureDecoupling:
    """c5/h4: pool candidacy no longer keys on a peer's host-level verdict."""

    def test_before_state_idle_peer_with_pressure_busy_flag_was_unselectable(self):
        # The regression witness: identical replica, both gates, opposite
        # answers. Idle engine, pressure-derived busy flag set.
        idle_but_flagged = peer(origin="spark", running=0, waiting=0, busy=True)

        assert _old_is_selectable(idle_but_flagged) is False
        assert _is_selectable(idle_but_flagged, local_busy=False) is True

    def test_idle_peer_with_pressure_busy_flag_is_selected(self):
        result = select_replica(
            [local(running=3, waiting=2), peer(running=0, waiting=0, busy=True)]
        )
        assert result == Selection("peer", False, REASON_PEER_LESS_LOADED)

    def test_pressure_busy_peer_with_a_full_engine_is_still_excluded(self):
        # h4: decoupling must not make a saturated box look attractive.
        saturated = peer(origin="peer-full", weight=4.0, running=4, busy=True)
        assert _is_selectable(saturated, local_busy=False) is False

        result = select_replica([local(weight=4.0, running=3), saturated])
        assert result == Selection("local", True, REASON_SOLE_READY)

    def test_a_saturated_peer_never_outranks_a_replica_with_headroom(self):
        # Not-quite-full peer (7/8) still ranks behind a lightly loaded local.
        result = select_replica(
            [
                local(weight=8.0, running=1),
                peer(origin="peer-hot", weight=8.0, running=7),
            ]
        )
        assert result.origin == "local"

    def test_every_replica_full_yields_none(self):
        result = select_replica(
            [local(weight=2.0, running=2), peer(weight=4.0, running=4, waiting=1)]
        )
        assert result == Selection(None, False, REASON_NONE)


class TestCapacityGate:
    """c2/h2: unselectable when active reaches capacity, not before."""

    def test_active_below_capacity_is_selectable(self):
        assert _is_selectable(peer(weight=4.0, running=3), local_busy=False) is True

    def test_active_equal_to_capacity_is_not_selectable(self):
        assert _is_selectable(peer(weight=4.0, running=4), local_busy=False) is False

    def test_waiting_counts_toward_capacity(self):
        assert _is_selectable(peer(weight=4.0, running=2, waiting=2), local_busy=False) is False

    def test_a_full_local_forwards_to_a_peer_with_headroom(self):
        result = select_replica([local(weight=2.0, running=2), peer(weight=2.0, running=1)])
        assert result == Selection("peer", False, REASON_PEER_LESS_LOADED)

    def test_an_uncalibrated_replica_is_never_full(self):
        # h3: an unpublished capacity never makes a replica unselectable.
        assert _is_selectable(peer(running=99, waiting=99), local_busy=False) is True

    def test_a_nonsensical_weight_is_treated_as_uncalibrated_not_as_full(self):
        assert _is_selectable(peer(weight=0.0, running=1), local_busy=False) is True
        assert _is_selectable(peer(weight=-3.0, running=1), local_busy=False) is True


class TestUncalibratedNeutralFallback:
    """c22/h15: the 1.0 sentinel is not read as a measured one-slot capacity."""

    def test_uncalibrated_peer_is_not_ranked_eight_times_worse(self):
        calibrated = peer(origin="peer-cal", weight=8.0, running=1)
        uncalibrated = peer(origin="peer-unc", running=1)
        candidates = [calibrated, uncalibrated]

        assert selection_wait(uncalibrated, candidates) == selection_wait(calibrated, candidates)
        # And concretely: not 1.0-vs-0.125.
        assert selection_wait(uncalibrated, candidates) == 0.125

    def test_a_mixed_fleet_does_not_drain_toward_the_calibrated_box(self):
        # One calibrated weight-8 peer, one uncalibrated peer, both at one
        # active request: locality/origin tie-breaks decide, not capacity.
        result = select_replica(
            [
                peer(origin="a-uncalibrated", running=1),
                peer(origin="b-calibrated", weight=8.0, running=1),
            ]
        )
        assert result.origin == "a-uncalibrated"

    def test_neutral_fallback_is_the_median_of_the_calibrated_capacities(self):
        candidates = [
            peer(origin="p1", weight=2.0),
            peer(origin="p2", weight=4.0),
            peer(origin="p3", weight=12.0),
            peer(origin="unc", running=4),
        ]
        # median of (2, 4, 12) is 4 -> 4 active / 4 capacity = 1.0
        assert selection_wait(candidates[-1], candidates) == 1.0

    def test_with_nothing_calibrated_the_fallback_reproduces_todays_ranking(self):
        # Criterion 4: for a uniform (fixed) weight the policy is unchanged.
        candidates = [local(running=2, waiting=1), peer(running=1)]
        assert selection_wait(candidates[0], candidates) == estimated_wait(candidates[0])
        assert selection_wait(candidates[1], candidates) == estimated_wait(candidates[1])
        assert select_replica(candidates) == Selection("peer", False, REASON_PEER_LESS_LOADED)


class TestLocalBusyIsUnchanged:
    """local_busy is this box's OWN shed verdict, not a peer's host reading."""

    def test_local_busy_still_forwards_rather_than_shedding(self):
        result = select_replica([local(), peer()], local_busy=True)
        assert result == Selection("peer", False, REASON_LOCAL_BUSY_FORWARDED)

    def test_local_busy_with_a_pressure_flagged_but_idle_peer_forwards(self):
        result = select_replica([local(), peer(busy=True)], local_busy=True)
        assert result == Selection("peer", False, REASON_LOCAL_BUSY_FORWARDED)

    def test_local_busy_with_no_other_replica_still_sheds(self):
        assert select_replica([local()], local_busy=True) == Selection(None, False, REASON_NONE)

    def test_local_busy_with_only_a_full_peer_still_sheds(self):
        result = select_replica([local(), peer(weight=2.0, running=2)], local_busy=True)
        assert result == Selection(None, False, REASON_NONE)


class TestLocalShedVerdictIsFirstParty:
    """The local box's own pressure flag still excludes it; a peer's does not."""

    def test_local_own_busy_flag_still_excludes_the_local_replica(self):
        assert _is_selectable(local(busy=True), local_busy=False) is False

    def test_local_own_busy_flag_forwards_to_an_idle_peer(self):
        result = select_replica([local(busy=True), peer()])
        assert result == Selection("peer", False, REASON_SOLE_READY)

    def test_a_peers_busy_flag_does_not_exclude_it(self):
        assert _is_selectable(peer(busy=True), local_busy=False) is True


class TestPurityWithCapacity:
    def test_capacity_aware_selection_is_still_deterministic(self):
        candidates = [
            local(running=2, waiting=1, weight=8.0),
            peer(origin="peer-a", running=1, weight=4.0),
            peer(origin="peer-b", running=0),
        ]
        results = {
            select_replica(candidates, affinity="cap-key", affinity_margin=0.5) for _ in range(100)
        }
        assert len(results) == 1

    def test_estimated_wait_keeps_its_single_argument_arithmetic(self):
        r = peer(running=3, waiting=1, weight=2.0)
        assert estimated_wait(r) == 2.0
        assert estimated_wait(r, 4.0) == 1.0


class TestAMeasuredOneSlotCapacity:
    """F2 (Qodo, PR #221): `lobes calibrate` can validly measure and persist a
    knee of 1 — a box whose engine admits exactly one request at a time has
    precisely that capacity. Reading every weight of 1.0 back as the
    "uncalibrated" sentinel made such a replica never full and ranked it as
    uncalibrated, so its measured capacity did not affect routing at all.

    The sentinel VALUE is unchanged; what changed is that "was a capacity
    published?" is now carried explicitly (`calibrated`) instead of being
    inferred from a magic number.
    """

    def test_a_measured_one_slot_capacity_is_calibrated(self):
        assert is_calibrated(peer(weight=1.0, calibrated=True)) is True

    def test_an_unpublished_capacity_is_still_uncalibrated_at_the_same_weight(self):
        assert is_calibrated(peer(weight=1.0)) is False

    def test_a_measured_one_slot_replica_is_FULL_at_one_active_request(self):
        full = peer(weight=1.0, calibrated=True, running=1)
        assert is_full(full) is True
        assert _is_selectable(full, local_busy=False) is False

    def test_an_uncalibrated_replica_at_weight_one_is_never_full(self):
        # h3 is untouched: an unpublished capacity must never make a replica
        # unselectable, however loaded it is.
        loaded = peer(weight=1.0, running=99, waiting=99)
        assert is_full(loaded) is False
        assert _is_selectable(loaded, local_busy=False) is True

    def test_a_full_one_slot_local_forwards_instead_of_winning_the_tie(self):
        # The routing consequence. Local declares a MEASURED capacity of 1 and
        # is holding its one request; the peer has 4 slots and one request.
        # Before F2 the local replica was read as uncalibrated, ranked at the
        # neutral capacity (4), tied with the peer at 0.25 and won on
        # locality — dispatching a second request to a one-slot box.
        result = select_replica(
            [
                local(weight=1.0, calibrated=True, running=1),
                peer(weight=4.0, running=1),
            ]
        )
        assert result == Selection("peer", False, REASON_PEER_LESS_LOADED)

    def test_a_measured_one_slot_capacity_counts_toward_the_neutral_median(self):
        candidates = [
            peer(origin="p1", weight=1.0, calibrated=True),
            peer(origin="p2", weight=9.0),
            peer(origin="unc", running=5),
        ]
        # median of the calibrated capacities (1, 9) is 5.0 -> 5 active / 5.
        assert selection_wait(candidates[-1], candidates) == 1.0

    def test_a_declared_capacity_discarded_on_a_fingerprint_change_is_uncalibrated(self):
        # `_replicas.py` reverts `weight` to the sentinel AND clears
        # `calibrated` when a pinned capacity no longer matches the live
        # fingerprint; the two must agree, or the replica would be full at one.
        stale = peer(weight=1.0, calibrated=False, running=3)
        assert is_calibrated(stale) is False
        assert is_full(stale) is False
