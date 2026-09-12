"""In-memory Roster + persisted Ledger for mesh membership.

Why this module exists (mesh-brain-join, issue #…)
---------------------------------------------------
A mesh of lobes boxes must track **who is alive** (Roster) and **who is
approved** (Ledger).  The Roster is a pure state machine with an injected clock
so tests need no sleeps.  The Ledger persists to JSON on disk, writing via temp
file + rename for atomicity, and supports gossip merge.

This module reuses ``_replicas.resolve_capacity`` rather than copy-pasting it,
because capacity validation is a single entry-point shared across the gateway.

API
---
- ``Roster`` — in-memory member tracking with join/leave/announce/tick and
  ledger passthrough (approve, revoke, is_approved).
- ``MeshRoster`` — backward-compatible alias for ``Roster``.
- ``Ledger`` — name approval registry with JSON persistence and gossip merge.
- Pure errors: ``MeshNameConflict``, ``MeshApprovalExpired``, ``MeshFlapping``.

Coverage targets: c37, h28, c9, h11, c43, h34, c53, h43, h4.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable

from lobes.gateway._replicas import CAPACITY_CLAMP_MAX, resolve_capacity

# How many consecutive ticks without an announce before a member is dropped.
_MISSED_MAX: int = int(os.environ.get("LOBES_MESH_MISSED_MAX", "5"))

# How many successful join operations within a 10-tick window triggers flapping.
# "more than 3 times" means the 4th join (flap_count=3) is rejected.
_FLAPPING_THRESHOLD: int = 3

# How many ticks to hold out after flapping is detected.
_FLAPPING_HOLD_TICKS: int = 1


# --- errors ------------------------------------------------------------------


class MeshError(Exception):
    """Base error for mesh-roster operations."""


class MeshNameConflict(MeshError):
    """A name is already held by a different origin."""


class MeshApprovalExpired(MeshNameConflict):
    """The approval expired before this join attempt."""


class MeshFlapping(MeshError):
    """Member joined/left too many times in 10 ticks."""


# --- data --------------------------------------------------------------------


@dataclass(frozen=True)
class _LedgerEntry:
    approved_by: str
    expiry: float
    updated_at: float


@dataclass
class TickResult:
    """Result returned by :meth:`Roster.tick`.

    ``dropped_origins`` (review #252 finding 2) names EXACTLY the members
    this tick removed, captured before deletion — the heartbeat's
    announcement-pruning pass must remove only these origins, never derive
    "dropped" by diffing against the survivors (that inverted the set and
    wiped every healthy member's announcement on any single expiry).
    """

    dropped: int = 0
    dropped_origins: tuple[str, ...] = ()


@dataclass
class MemberRecord:
    """One announced member."""

    name: str
    origin: str
    capacity: float  # resolved through resolve_capacity
    last_seen: float  # clock value of last announce
    missed: int = 0  # consecutive ticks since last announce
    verified: bool = False


# --- Ledger ------------------------------------------------------------------


class Ledger:
    """Name → approved_by / expiry / updated_at registry.

    Persists to JSON at *path* when :meth:`save` is called.  Writes via
    ``tempfile`` + ``os.replace`` for atomicity.
    """

    def __init__(
        self, *, path: str | None = None, clock: Callable[[], float] | None = None
    ) -> None:
        self.path = path  # ``None`` means no persistence
        self._clock = clock or (lambda: 0.0)
        self.entries: dict[str, _LedgerEntry] = {}
        # review #252 findings 3/4: one re-entrant lock serializes every
        # entries mutation (approve/revoke/merge) against every read used for
        # an admission decision (is_approved) and against save()'s own
        # iteration of `entries` — a concurrent approve/revoke used to be
        # able to interleave with another mutation or with save()'s dict
        # iteration (ThreadingHTTPServer runs every route on its own
        # thread), corrupting the persisted ledger or raising
        # "dictionary changed size during iteration".
        self._lock = threading.RLock()
        self.load()

    def approve(
        self, name: str, approved_by: str, expiry: float, *, now: float | None = None
    ) -> None:
        now = now if now is not None else self._clock()
        with self._lock:
            self.entries[name] = _LedgerEntry(
                approved_by=approved_by,
                expiry=expiry,
                updated_at=now,
            )

    def revoke(self, name: str, *, now: float | None = None, approved_by: str = "system") -> None:
        now = now if now is not None else self._clock()
        with self._lock:
            self.entries[name] = _LedgerEntry(
                approved_by=approved_by,
                expiry=0.0,
                updated_at=now,
            )

    def is_approved(self, name: str, *, now: float | None = None) -> bool:
        now = now if now is not None else self._clock()
        with self._lock:
            entry = self.entries.get(name)
            if entry is None:
                return False
            return entry.expiry > now

    def join_key(self, name: str) -> str | None:
        """Return ``approved_by`` if approved, else ``None``."""
        with self._lock:
            return self.entries[name].approved_by if self.is_approved(name) else None

    def merge(self, peer: "Ledger") -> None:
        """Gossip-merge *peer*'s entries.  The entry with the greater
        ``updated_at`` wins (ties: keep local)."""
        with self._lock:
            for name, entry in peer.entries.items():
                existing = self.entries.get(name)
                if existing is None or entry.updated_at > existing.updated_at:
                    self.entries[name] = entry

    def save(self) -> None:
        """Persist to *path* via temp-file + rename (atomic)."""
        if self.path is None:
            return
        with self._lock:
            data = {
                n: {
                    "approved_by": e.approved_by,
                    "expiry": e.expiry,
                    "updated_at": e.updated_at,
                }
                for n, e in self.entries.items()
            }
        dirn = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=dirn, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self) -> None:
        """Restore entries from *path* (no-op if path is ``None``)."""
        if self.path is None:
            return
        if not os.path.isfile(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            for name, info in data.items():
                self.entries[name] = _LedgerEntry(
                    approved_by=info["approved_by"],
                    expiry=info["expiry"],
                    updated_at=info.get("updated_at", 0.0),
                )


# --- Roster ------------------------------------------------------------------


class Roster:
    """In-memory member registry.

    Members are tracked by *name*.  A member survives only while they keep
    announcing; ``tick()`` increments the missed counter and drops members
    that hit ``_MISSED_MAX`` consecutive missed ticks.
    """

    _roster: dict[str, MemberRecord]  # back-stop for property setter

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        ledger_path: str | None = None,
        capacity_max: float = CAPACITY_CLAMP_MAX,
        missed_max: int | None = None,
    ) -> None:
        self._clock = clock or (lambda: 0.0)
        self._roster = {}
        # RLock: announce()/tick()/members() are also called from paths that
        # already hold this lock (seed-roster merge, gated announce) — a plain
        # Lock deadlocked the heartbeat live on 2026-09-12.
        self._lock = threading.RLock()  # Protects _roster, _approved_here, _flap_*
        self._ledger = Ledger(path=ledger_path, clock=self._clock)
        self._approved_here: set[str] = set()  # names explicitly approved on this node
        # PER-NAME flap tracking (item A, t9) — was a single roster-wide
        # counter/timestamp before this task, which meant one flapping
        # member held out EVERY name. Keyed by member name so churn on one
        # name never affects another's join/announce eligibility.
        self._flap_counts: dict[str, int] = {}
        self._flap_times: dict[str, float] = {}  # name -> clock value of last flap detection
        self._capacity_max: float = capacity_max
        # Finding 10: inject missed_max; None → fall back to module-level env default.
        self._missed_max_override: int | None = missed_max

    # -- public API (Roster) ------------------------------------------------

    def now(self) -> float:
        """Return the roster's current clock value.

        Used to convert duration-based expiry to the roster's clock domain.
        """
        return self._clock()

    def announce(
        self, name: str, origin: str, capacity: object, *, now: float | None = None
    ) -> None:
        """Register or update a member.  Refuses on name conflict or bad capacity.

        Item A (t9): a NEW registration (no existing record for *name*) is
        the announce-path's equivalent of :meth:`join`'s join-transition, and
        now carries the SAME per-name flap tracking ``join`` already had —
        `/mesh/announce` is the only endpoint any live deployment drives, so
        before this task the flap counter was dead code (nothing but the
        unwired ``join`` ever touched it). A name that re-registers more
        than :data:`_FLAPPING_THRESHOLD` times inside one hold-out window is
        refused with :class:`MeshFlapping` for :data:`_FLAPPING_HOLD_TICKS`,
        exactly mirroring ``join``'s contract. An UPDATE to an existing
        record (same name, same origin — the common heartbeat case) never
        touches flap state at all, matching ``join``'s never-flap-on-update
        precedent and keeping every steady-state heartbeat byte-identical.
        """
        now = now if now is not None else self._clock()

        with self._lock:
            existing = self._roster.get(name)
            if existing is not None:
                # Same origin → update in place
                if existing.origin == origin:
                    self._update_capacity(existing, capacity)
                    existing.last_seen = now
                    existing.missed = 0
                    return
                # Different origin → conflict
                raise MeshNameConflict(f"name {name!r} already held by {existing.origin!r}")

            # New registration — apply the per-name flap check (item A).
            flap_count = self._flap_counts.get(name, 0)
            flap_time = self._flap_times.get(name, 0.0)
            if flap_count > 0 and now >= flap_time + _FLAPPING_HOLD_TICKS:
                flap_count = 0
            if flap_count >= _FLAPPING_THRESHOLD:
                raise MeshFlapping(f"name {name!r} flapping — held out {_FLAPPING_HOLD_TICKS} tick")

            # New member — validate capacity first (refuses before adding)
            resolved, _note = resolve_capacity(capacity, capacity_max=self._capacity_max)
            self._roster[name] = MemberRecord(
                name=name, origin=origin, capacity=resolved, last_seen=now, missed=0
            )
            self._flap_counts[name] = flap_count + 1
            self._flap_times[name] = now

    def announce_gated(
        self, name: str, origin: str, capacity: object, *, now: float | None = None
    ) -> None:
        """Atomically check ledger admission, then :meth:`announce`.

        review #252 finding 4: ``/mesh/announce`` used to check
        ``ledger.is_approved`` and then call :meth:`announce` as two
        separate, unsynchronized steps — a concurrent :meth:`revoke` could
        commit in between, letting the in-flight request admit (or refresh)
        a name that was just revoked. Both the ledger read and the roster
        admission now happen under the SAME lock :meth:`revoke`/:meth:`approve`
        use, so a revoke either fully precedes or fully follows one
        announce — never interleaves with it.

        Raises :class:`MeshApprovalExpired` when an existing ledger entry for
        *name* is not currently in force (identical gating to :meth:`join`).
        An ABSENT entry is "no restriction" (finding 5, unchanged): the join
        key alone still admits.
        """
        now = now if now is not None else self._clock()
        with self._ledger._lock:  # noqa: SLF001 — the shared admission lock
            entry = self._ledger.entries.get(name)
            if entry is not None and not self._ledger.is_approved(name, now=now):
                raise MeshApprovalExpired(f"name {name!r} approval expired")
            self.announce(name, origin, capacity, now=now)

    def approve_and_save(
        self, name: str, approved_by: str, expiry: float, *, now: float | None = None
    ) -> None:
        """:meth:`approve` + :meth:`save`, one transaction (review #252 finding 3)."""
        with self._ledger._lock:  # noqa: SLF001
            self.approve(name, approved_by, expiry, now=now)
            self.save()

    def revoke_and_save(
        self, name: str, *, now: float | None = None, approved_by: str = "system"
    ) -> None:
        """:meth:`revoke` + :meth:`save`, one transaction (review #252 finding 3)."""
        with self._ledger._lock:  # noqa: SLF001
            self.revoke(name, now=now, approved_by=approved_by)
            self.save()

    def tick(self, now: float | None = None) -> TickResult:
        """Check staleness; drop expired members.  Returns drop count."""
        now = now if now is not None else self._clock()
        with self._lock:
            local_missed_max = (
                self._missed_max_override if self._missed_max_override is not None else _MISSED_MAX
            )
            dropped: int = 0
            to_remove: list[str] = []

            for name, member in self._roster.items():
                member.missed += 1
                if member.missed >= local_missed_max:
                    to_remove.append(name)

            dropped_origins: list[str] = []
            for name in to_remove:
                dropped_origins.append(self._roster[name].origin)
                del self._roster[name]
                dropped += 1

            # Clear each name's flapping count once its own hold-out window
            # has elapsed (item A: per-name, not the old single roster-wide
            # counter).
            for name in list(self._flap_counts):
                if (
                    self._flap_counts[name] > 0
                    and now >= self._flap_times.get(name, 0.0) + _FLAPPING_HOLD_TICKS
                ):
                    self._flap_counts[name] = 0

        return TickResult(dropped=dropped, dropped_origins=tuple(dropped_origins))

    def is_joined(self, name: str) -> bool:
        """Is the member currently in the roster?"""
        with self._lock:
            return name in self._roster

    def members(self) -> list[str]:
        """List all member names."""
        with self._lock:
            return list(self._roster.keys())

    def records(self) -> list[tuple[str, str, float]]:
        """Return ``(name, origin, capacity)`` for every member."""
        with self._lock:
            return [(m.name, m.origin, m.capacity) for m in self._roster.values()]

    # -- Ledger passthrough --------------------------------------------------

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    def is_approved(self, name: str, *, now: float | None = None) -> bool:
        """A name is approved if it exists in the ledger and has not expired."""
        return self._ledger.is_approved(name, now=now)

    def approve(
        self, name: str, approved_by: str, expiry: float, *, now: float | None = None
    ) -> None:
        self._approved_here.add(name)
        self._ledger.approve(name, approved_by, expiry, now=now)

    def revoke(self, name: str, *, now: float | None = None, approved_by: str = "system") -> None:
        """Revoke approval for *name*.  Delegates to :meth:`Ledger.revoke`."""
        self._ledger.revoke(name, now=now, approved_by=approved_by)

    def merge(self, peer: "Ledger") -> None:
        """Merge a peer's ledger into ours."""
        self._ledger.merge(peer)

    def save(self) -> None:
        self._ledger.save()

    def load(self) -> None:
        self._ledger.load()

    def join(self, name: str, origin: str, capacity: object, *, now: float | None = None) -> None:
        now = now if now is not None else self._clock()

        with self._lock:
            # Check approval first
            if not self.is_approved(name, now=now):
                entry = self._ledger.entries.get(name)
                if entry is not None:
                    raise MeshApprovalExpired(
                        f"name {name!r} approval expired (approved_by={entry.approved_by!r})"
                    )
                raise MeshApprovalExpired(f"name {name!r} not approved")

            # Check flapping (item A: per-name state, shared with announce())
            # — clear hold-out if the window already passed.
            flap_count = self._flap_counts.get(name, 0)
            flap_time = self._flap_times.get(name, 0.0)
            if flap_count > 0 and now >= flap_time + _FLAPPING_HOLD_TICKS:
                flap_count = 0
            if flap_count >= _FLAPPING_THRESHOLD:
                raise MeshFlapping(f"name {name!r} flapping — held out {_FLAPPING_HOLD_TICKS} tick")

            # Remove old entry if it exists (explicit leave counts as transition)
            if name in self._roster:
                del self._roster[name]

            resolved, _note = resolve_capacity(capacity, capacity_max=self._capacity_max)
            self._roster[name] = MemberRecord(
                name=name, origin=origin, capacity=resolved, last_seen=now, missed=0
            )
            self._flap_counts[name] = flap_count + 1
            self._flap_times[name] = now

    def leave(self, name: str, *, now: float | None = None) -> None:
        with self._lock:
            if name in self._roster:
                del self._roster[name]

    # -- helpers -------------------------------------------------------------

    def _update_capacity(self, member: MemberRecord, capacity: object) -> None:
        resolved, _note = resolve_capacity(capacity, capacity_max=self._capacity_max)
        member.capacity = resolved


# --- MeshRoster — backward-compatible alias for Roster ---------------------


MeshRoster = Roster
