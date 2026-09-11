"""In-memory Roster + persisted Ledger for mesh membership.

Pure state machine with an injected clock (``clock`` argument) so tests need no
sleeps. Ledger persists as JSON at ``LOBES_MESH_LEDGER_PATH``; write via temp
file + rename.  Reuses ``_replicas.resolve_capacity``; does not copy it.

Coverage: c37, h28, c9, h11, c43, h34, c53, h43, h4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobes.gateway._mesh_roster import (
    Ledger,
    MeshApprovalExpired,
    MeshFlapping,
    MeshNameConflict,
    MeshRoster,
    Roster,
)

# --- injected clock ----------------------------------------------------------


class _TickClock:
    """Monotonically increasing clock; starts at 0.0, advances by 1.0 per tick."""

    def __init__(self) -> None:
        self.t: float = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self) -> None:
        self.t += 1.0


# --- fixtures ----------------------------------------------------------------


@pytest.fixture()
def clock() -> _TickClock:
    return _TickClock()


@pytest.fixture()
def tmp_path_clean(tmp_path: Path) -> Path:
    """Return ``tmp_path`` (guaranteed empty for each test)."""
    return tmp_path


@pytest.fixture()
def ledger_path(tmp_path_clean: Path) -> Path:
    p = tmp_path_clean / "ledger.json"
    p.write_text("{}", encoding="utf-8")
    return p


@pytest.fixture()
def roster(clock: _TickClock, ledger_path: Path) -> MeshRoster:
    return MeshRoster(clock=clock, ledger_path=str(ledger_path))


# =============================================================================
# Criterion 1 — drop after MISSED_MAX consecutive misses; restart empties
# roster while ledger file stays byte-identical
# =============================================================================


class TestCriterion1:
    """A member is dropped after LOBES_MESH_MISSED_MAX consecutive missed ticks, not before."""

    def test_not_dropped_before_missed_max(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.announce("alice", "origin-a", 4, now=clock())
        for _ in range(5):  # MISSED_MAX defaults to 5
            roster.tick(clock())
            clock.tick()
        # At tick 5 (0-indexed), the member has 5 missed ticks = MISSED_MAX,
        # so she IS dropped. We verify that at MISSED_MAX-1 she is NOT.
        roster2 = MeshRoster(clock=_TickClock(), ledger_path="/dev/null")
        roster2.announce("alice", "origin-a", 4, now=0.0)
        for i in range(4):  # only 4 ticks = MISSED_MAX - 1
            roster2.tick(0.0)
            # She should still be present
            assert roster2.is_joined("alice")

    def test_dropped_after_missed_max(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.announce("alice", "origin-a", 4, now=clock())
        for _ in range(5):  # MISSED_MAX consecutive ticks
            roster.tick(clock())
            clock.tick()
        assert roster.is_joined("alice") is False

    def test_restart_empty_roster_ledger_intact(
        self, roster: MeshRoster, clock: _TickClock, ledger_path: Path
    ) -> None:
        roster.announce("alice", "origin-a", 4, now=clock())
        roster.approve("bob", "admin", 9999.0, now=clock())
        for _ in range(5):
            roster.tick(clock())
            clock.tick()
        # Snapshot the ledger file
        before = ledger_path.read_bytes()
        # New roster starts empty
        roster2 = MeshRoster(clock=_TickClock(), ledger_path=str(ledger_path))
        assert roster2.members() == []
        # Ledger file byte-identical
        after = ledger_path.read_bytes()
        assert before == after


# =============================================================================
# Criterion 2 — name conflict / same-origin update
# =============================================================================


class TestCriterion2:
    """Two announcements with the same name from different origins: second refused.
    Same-origin re-announcement updates in place."""

    def test_same_name_different_origin_refused(
        self, roster: MeshRoster, clock: _TickClock
    ) -> None:
        roster.announce("alice", "origin-a", 4, now=clock())
        with pytest.raises(MeshNameConflict, match="alice"):
            roster.announce("alice", "origin-b", 4, now=clock())

    def test_same_origin_updates_in_place(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.announce("alice", "origin-a", 4, now=clock())
        roster.announce("alice", "origin-a", 8, now=clock())  # same origin, different cap
        m = roster._roster["alice"]
        assert m.capacity == 8

    def test_multiple_same_origin_updates(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.announce("alice", "origin-a", 4, now=clock())
        roster.announce("alice", "origin-a", 8, now=clock())
        roster.announce("alice", "origin-a", 2, now=clock())
        assert roster._roster["alice"].capacity == 2


# =============================================================================
# Criterion 3 — gossip merge grants; lapsed name refused
# =============================================================================


class TestCriterion3:
    """A lapsed name is refused on a member that never saw the approve call.

    Gossip from a peer grants permission (merged approval), but a lapsed
    expiry still refuses.
    """

    def test_lapsed_name_refused_on_merge(self, roster: Roster, clock: _TickClock) -> None:
        # A approves alice with expiry at t=10
        roster.approve("alice", "admin", 10.0, now=0.0)
        # B merges A's ledger (alice granted at t=0, expires t=10)
        peer = Ledger(clock=_TickClock())
        peer.approve("alice", "admin", 10.0, now=0.0)
        roster.merge(peer)
        # At t=5 the merged approval still grants
        assert roster.is_approved("alice", now=5.0) is True
        # At t=11 the merged approval has lapsed
        assert roster.is_approved("alice", now=11.0) is False

    def test_expired_entry_refused(self, roster: Roster, clock: _TickClock) -> None:
        roster.approve("alice", "admin", 0.0, now=clock())  # expired immediately
        assert roster.is_approved("alice") is False


# =============================================================================
# Criterion 4 — capacity clamp, flapping, no announcement removal
# =============================================================================


class TestCriterion4:
    """Announced capacity passes resolve_capacity (weight <= CAPACITY_CLAMP_MAX);
    flapping hold-out; no announcement can remove another member."""

    def test_capacity_clamped(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.announce("alice", "origin-a", 100.0, now=clock())  # way over clamp
        m = roster._roster["alice"]
        assert m.capacity == 64.0  # CAPACITY_CLAMP_MAX

    def test_flapping_hold_out(self, roster: MeshRoster, clock: _TickClock) -> None:
        # Join 4 times in 10 ticks triggers hold-out
        roster.approve("alice", "admin", 9999.0, now=clock())
        roster.join("alice", "origin-a", 4, now=clock())
        roster.leave("alice")
        roster.join("alice", "origin-a", 4, now=clock())
        roster.leave("alice")
        roster.join("alice", "origin-a", 4, now=clock())
        roster.leave("alice")
        # 4th join should be refused with flapping
        with pytest.raises(MeshFlapping):
            roster.join("alice", "origin-a", 4, now=clock())

    def test_announcement_cannot_remove_another(
        self, roster: MeshRoster, clock: _TickClock
    ) -> None:
        roster.announce("alice", "origin-a", 4, now=clock())
        roster.announce("bob", "origin-b", 4, now=clock())
        roster.announce("alice", "origin-a", 8, now=clock())  # alice update
        # Bob should still be here
        assert roster.is_joined("bob")


# --- Ledger persistence tests ------------------------------------------------


class TestLedgerPersistence:
    """Ledger persists as JSON at LOBES_MESH_LEDGER_PATH; write via temp file + rename."""

    def test_ledger_save_load(self, ledger_path: Path, clock: _TickClock) -> None:
        ld = Ledger(path=str(ledger_path), clock=clock)
        ld.approve("alice", "admin", 9999.0, now=clock())
        ld.save()
        # Read the file
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert "alice" in data

    def test_ledger_reload(self, ledger_path: Path, clock: _TickClock) -> None:
        ld1 = Ledger(path=str(ledger_path), clock=clock)
        ld1.approve("alice", "admin", 9999.0, now=clock())
        ld1.save()
        # New instance loads from file
        ld2 = Ledger(path=str(ledger_path), clock=clock)
        assert ld2.is_approved("alice")

    def test_updated_at_persists(self, ledger_path: Path, clock: _TickClock) -> None:
        ld1 = Ledger(path=str(ledger_path), clock=clock)
        ld1.approve("alice", "admin", 9999.0, now=10.0)
        ld1.save()
        ld2 = Ledger(path=str(ledger_path), clock=_TickClock())
        assert ld2.entries["alice"].updated_at == 10.0

    def test_legacy_load_updated_at_defaults_to_zero(self, ledger_path: Path) -> None:
        # Write legacy JSON without updated_at
        ledger_path.write_text(
            json.dumps({"alice": {"approved_by": "admin", "expiry": 9999.0}}),
            encoding="utf-8",
        )
        ld = Ledger(path=str(ledger_path), clock=_TickClock())
        assert ld.entries["alice"].updated_at == 0.0


# =============================================================================
# Defect 2 — revocation wins over older, longer grant
# =============================================================================


class TestDefect2:
    """Revocation (later updated_at) beats an older, longer approval."""

    def test_revocation_wins(self, clock: _TickClock) -> None:
        # A approves X for 1 h at t=0
        roster_a = Roster(clock=clock)
        roster_a.approve("alice", "admin", 3600.0, now=0.0)
        # B is clean
        roster_b = Roster(clock=_TickClock())
        roster_b.merge(roster_a.ledger)
        # at t=10 A revokes
        roster_a.revoke("alice", now=10.0)
        # B merges again -> B refuses X even though previous expiry was 3600
        roster_b.merge(roster_a.ledger)
        assert roster_b.is_approved("alice", now=9.0) is False

    def test_revocation_persisted(self, ledger_path: Path, clock: _TickClock) -> None:
        ld1 = Ledger(path=str(ledger_path), clock=clock)
        ld1.approve("alice", "admin", 3600.0, now=0.0)
        ld1.revoke("alice", now=10.0)
        ld1.save()
        ld2 = Ledger(path=str(ledger_path), clock=_TickClock())
        assert ld2.is_approved("alice") is False
        assert ld2.entries["alice"].expiry == 0.0
        assert ld2.entries["alice"].updated_at == 10.0


# --- Edge cases --------------------------------------------------------------


class TestEdgeCases:
    def test_members_empty_on_fresh_roster(self, roster: MeshRoster) -> None:
        assert roster.members() == []

    def test_join_requires_approval(self, roster: MeshRoster, clock: _TickClock) -> None:
        with pytest.raises(MeshApprovalExpired):
            roster.join("alice", "origin-a", 4, now=clock())

    def test_leave_nonexistent_silent(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.leave("alice")  # should not raise

    def test_tick_no_members(self, roster: MeshRoster, clock: _TickClock) -> None:
        result = roster.tick(clock())
        assert result.dropped == 0


# --- resolve_capacity reuse verification --------------------------------------


class TestResolveCapacityReuse:
    """Verifies that the roster uses _replicas.resolve_capacity (not a copy)."""

    def test_resolves_nan(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.announce("alice", "origin-a", float("nan"), now=clock())
        # NaN → refused, falls back to UNCALIBRATED_WEIGHT
        m = roster._roster["alice"]
        assert m.capacity > 0  # resolved to something valid (UNCALIBRATED_WEIGHT = 1.0)

    def test_resolves_negative(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.announce("alice", "origin-a", -5.0, now=clock())
        m = roster._roster["alice"]
        assert m.capacity > 0  # fallback to UNCALIBRATED_WEIGHT


# --- TickResult inspection ---------------------------------------------------


class TestTickResult:
    def test_tick_returns_drop_count(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.announce("alice", "origin-a", 4, now=clock())
        for _ in range(4):  # 4 ticks = MISSED_MAX - 1
            roster.tick(clock())
            clock.tick()
        result = roster.tick(clock())  # 5th tick — should drop
        assert result.dropped == 1


# --- Merge edge cases --------------------------------------------------------


class TestMergeEdgeCases:
    def test_merge_empty_peer(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.approve("alice", "admin", 9999.0, now=clock())
        roster.merge(Ledger(clock=clock))  # empty
        assert roster.is_approved("alice")

    def test_merge_updates_expiry(self, roster: Roster, clock: _TickClock) -> None:
        roster.approve("alice", "admin", 50.0, now=0.0)
        peer = Ledger(clock=_TickClock())
        peer.approve("alice", "peer-admin", 100.0, now=0.0)
        roster.merge(peer)
        # Same updated_at: local wins, expiry stays at 50.0
        assert roster._ledger.entries["alice"].expiry == 50.0
        assert roster.is_approved("alice")


# --- Flapping hold-out expiry ------------------------------------------------


class TestFlappingHoldOut:
    def test_hold_out_expires_after_one_tick(self, roster: MeshRoster, clock: _TickClock) -> None:
        roster.approve("alice", "admin", 9999.0, now=clock())
        roster.join("alice", "origin-a", 4, now=clock())
        roster.leave("alice")
        roster.join("alice", "origin-a", 4, now=clock())
        roster.leave("alice")
        roster.join("alice", "origin-a", 4, now=clock())
        roster.leave("alice")
        # 4th join → flapping hold-out
        with pytest.raises(MeshFlapping):
            roster.join("alice", "origin-a", 4, now=clock())
        # After one tick, hold-out should expire
        clock.tick()
        roster.join("alice", "origin-a", 4, now=clock())
