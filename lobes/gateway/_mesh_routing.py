"""Membership-driven routing snapshot for the mesh.

Builds an immutable view of verified member roles from a :class:`~lobes.gateway._mesh_roster.Roster`
and probe-based capability data, then exposes the snapshot for gateway routing
decisions.  The snapshot is copy-on-write: each request reads one frozen copy;
a roster change mid-request never affects it and no lock is held across a dial.

Public API
----------
* :class:`MemberInfo` — one verified or unverified member
* :class:`RoutingSnapshot` — immutable per-request view of the mesh roster
* :func:`build_snapshot` — build a :class:`RoutingSnapshot` from Roster + optional
  probe data (capacities from /capabilities)
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lobes.gateway._mesh_roster import Roster
    from lobes.gateway._mesh_wire import Announcement, Fingerprint
    from lobes.gateway._replicas import Fingerprint as ReplicaFingerprint, ReplicaState


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemberInfo:
    """One mesh member as known to the routing layer.

    Parameters
    ----------
    name:
        Operator-declared name (unique within the roster).
    origin:
        Gateway origin URL that peers dial.
    announced_roles:
        Role names the member advertised in its :class:`Announcement` (may be
        empty when the member has no announcement yet).
    verified_roles:
        Role names verified by the ``/capabilities`` probe.  These are the ONLY
        roles that should be used for forwarding; a member that is not verified
        for a role must never receive forwarded traffic for it.
    capacity:
        Resolved capacity from the roster.
    """

    name: str
    origin: str
    announced_roles: tuple[str, ...]
    verified_roles: tuple[str, ...]
    capacity: float


@dataclass(frozen=True)
class RoutingSnapshot:
    """Immutable per-request view of the mesh membership.

    Built once per request from :func:`build_snapshot`; a roster change mid-request
    never affects it and no lock is held across a dial.
    """

    # origin -> MemberInfo for every member in the roster.
    members: tuple["MemberInfo", ...]
    # origin -> Announcement (decoded) for every member that announced.
    announcements: tuple[tuple[str, "Announcement"], ...]

    # --- lookup helpers ---

    def member_origins(self, role: str) -> tuple[str, ...]:
        """Origins of members verified for *role*, in roster order."""
        return tuple(m.origin for m in self.members if role in m.verified_roles)

    def has_member(self, origin: str) -> bool:
        """True when *origin* is known."""
        for m in self.members:
            if m.origin == origin:
                return True
        return False

    def member_count(self) -> int:
        return len(self.members)

    def announced_roles(self, origin: str) -> tuple[str, ...]:
        """Roles the member *origin* announced (may include non-verified)."""
        for m in self.members:
            if m.origin == origin:
                return m.announced_roles
        return ()

    def is_verified(self, origin: str, role: str) -> bool:
        """True when *origin* is verified for *role*."""
        for m in self.members:
            if m.origin == origin and role in m.verified_roles:
                return True
        return False


# ---------------------------------------------------------------------------
# MeshRoutingView + wire fingerprint converter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeshRoutingView:
    """Bundle of routing snapshot + per-origin replica states.

    Replaces ``RoutingSnapshot | None`` on :class:`SnapshotHolder` so that
    consumers can also inspect live replica state per peer.
    """

    snapshot: RoutingSnapshot
    peer_states: Mapping[str, Mapping[str, "ReplicaState"]]


# ---------------------------------------------------------------------------
# Fingerprint verification
# ---------------------------------------------------------------------------


def verify_member_roles(
    announced: "Announcement",
    probed_roles: dict[str, dict],  # role -> {fingerprint: {...}, ready: bool|None}
) -> frozenset[str]:
    """Compare announced fingerprints against probed fingerprints.

    Returns the frozenset of role names that are verified (compatible AND ready).
    Roles with no fingerprint key or with incompatible fingerprints are excluded.
    """
    verified: list[str] = []
    for role_name, probed_entry in probed_roles.items():
        if role_name not in announced.roles:
            continue
        # Skip if probed entry has no fingerprint key.
        if "fingerprint" not in probed_entry:
            continue
        # Skip if probed entry is not ready.
        if probed_entry.get("ready") is not True:
            continue

        # Get the announced fingerprint.
        announced_fp = announced.roles[role_name].fingerprint
        # Get the probed fingerprint.
        probed_fp_data = probed_entry.get("fingerprint")

        # Convert both to replica Fingerprint for comparison.
        announced_replica_fp: ReplicaFingerprint | None = (
            _wire_fingerprint_to_replica(announced_fp)
            if announced_fp is not None
            else None
        )
        probed_replica_fp: ReplicaFingerprint | None = (
            _wire_fingerprint_to_replica(probed_fp_data)
            if probed_fp_data is not None
            else None
        )

        # Run comparison.
        from lobes.gateway._replicas import compare_fingerprints

        compatible, _reason = compare_fingerprints(
            announced_replica_fp, probed_replica_fp
        )
        if compatible:
            verified.append(role_name)

    return frozenset(verified)


# ---------------------------------------------------------------------------
# Wire fingerprint converter
# ---------------------------------------------------------------------------


def _wire_fingerprint_to_replica(
    fp: "Fingerprint | None",
) -> "lobes.gateway._replicas.Fingerprint | None":  # noqa: F821 — resolved at runtime
    """Convert a wire :class:`~lobes.gateway._mesh_wire.Fingerprint` to a
    replica :class:`~lobes.gateway._replicas.Fingerprint`.

    Conversions:
    * ``max_model_len=0`` → ``None`` (unknown)
    * ``null`` / ``""`` fields → ``None`` (unknown)
    * Otherwise pass through.
    """
    if fp is None:
        return None

    from lobes.gateway._replicas import Fingerprint as ReplicaFingerprint

    return ReplicaFingerprint(
        served_id=fp.served_id if fp.served_id else None,  # type: ignore[arg-type]
        max_model_len=None if fp.max_model_len == 0 else fp.max_model_len,
        runtime=fp.runtime if fp.runtime else None,  # type: ignore[arg-type]
        quantization=fp.quantization if fp.quantization else None,  # type: ignore[arg-type]
        kv_cache_dtype="",
        reasoning_parser="",
        tool_parser="",
        speculative_config="",
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_snapshot(
    roster: "Roster",
    *,
    # Optional per-member announcement data: origin -> Announcement.
    # When the roster was populated by /mesh/announce, MeshRoutes decodes the
    # announcement and can pass it here so roles are known.  Without it, members
    # appear with empty announced_roles and empty verified_roles.
    announcements: Mapping[str, "Announcement"] | None = None,
    # Optional per-origin role data from /capabilities probes.  Keys are member
    # origins, values are frozensets of role names the probe verified.  A member
    # whose probe result does NOT match its announcement is marked unverified
    # (verified_roles is empty).  When omitted, every member is treated as
    # unverified.
    verified_roles: Mapping[str, "frozenset[str]"] | None = None,
) -> RoutingSnapshot:
    """Build a :class:`RoutingSnapshot` from *roster* + probe data.

    Parameters
    ----------
    roster:
        The current :class:`~lobes.gateway._mesh_roster.Roster` — read under no
        lock; the caller owns concurrency.  Only ``members()`` and per-member
        reads are used (never ``announce``, ``tick``, or ``roster._roster``
        mutation).
    announcements:
        Per-member announcements decoded from ``POST /mesh/announce``.  Keys are
        member *origins* (URLs); values are the :class:`Announcement` objects.
        Without these, members appear with empty role lists.
    verified_roles:
        Per-origin sets of roles verified by a ``/capabilities`` probe.  A member
        whose *verified_roles* does NOT match its *announced_roles* (as a
        superset check — all announced roles must be present in verified_roles)
        is marked **unverified**: its ``verified_roles`` tuple is empty and it
        will receive zero forwards.  When a member has no announcement but
        *does* have verified_roles, those roles are carried as both announced
        and verified (the member is fully verified).

    Returns
    -------
    A :class:`RoutingSnapshot` that is immutable: the returned tuple of
    :class:`MemberInfo` objects cannot be modified, and the dict of per-member
    data is frozen at construction time.
    """
    ann_map: dict[str, Announcement] = {} if announcements is None else dict(announcements)
    ver_map: dict[str, frozenset[str]] = {} if verified_roles is None else dict(verified_roles)

    # Collect the set of known origins from the roster so we can prune stale data.
    roster_origins: set[str] = set()

    members: list[MemberInfo] = []

    for name in roster.members():
        rec = roster._roster.get(name)  # noqa: SLF001 — read-only access
        if rec is None:
            continue

        origin = rec.origin
        roster_origins.add(origin)
        # Announcement roles.
        ann: Announcement | None = ann_map.get(origin)
        if ann is not None:
            announced = tuple(ann.roles.keys())
        else:
            announced = ()

        # Verification: all announced roles must be present in the probe result.
        verified_set = ver_map.get(origin)
        if verified_set is not None and announced:
            # Full verification: every announced role must be in the probe result.
            announced_set = frozenset(announced)
            if not announced_set.issubset(verified_set):
                # Probe disagrees with announcement → unverified.
                verified = ()
            else:
                verified = announced
        elif verified_set is not None and not announced:
            # No announcement but probe data exists → fully verified from probe.
            verified = tuple(sorted(verified_set))
        else:
            # No probe data → unverified.
            verified = ()

        members.append(
            MemberInfo(
                name=rec.name,
                origin=origin,
                announced_roles=announced,
                verified_roles=verified,
                capacity=rec.capacity,
            )
        )

    # Prune stale announcements and verified roles to origins still in the roster.
    ann_map = {o: a for o, a in ann_map.items() if o in roster_origins}
    ver_map = {o: v for o, v in ver_map.items() if o in roster_origins}

    # Build announcement tuples for lookups.
    ann_tuples: list[tuple[str, Announcement]] = []
    for origin, ann in ann_map.items():
        ann_tuples.append((origin, ann))

    return RoutingSnapshot(
        members=tuple(members),
        announcements=tuple(ann_tuples),
    )


# ---------------------------------------------------------------------------
# Thread-safe snapshot holder (copy-on-write, no lock across dial)
# ---------------------------------------------------------------------------


class SnapshotHolder:
    """Thread-safe snapshot holder with copy-on-write semantics.

    Usage:
        holder = SnapshotHolder(roster)
        # Background thread:
        new_view = holder.update(...)  # Returns MeshRoutingView
        holder.replace(new_view)

        # Request handler:
        view = holder.current()  # Returns MeshRoutingView | None
        if view is not None:
            snap = view.snapshot
            ...
        # Dial: no lock.

    The holder stores a :class:`MeshRoutingView` (snapshot + peer states), not
    just a bare :class:`RoutingSnapshot`.  The ``update`` method is a
    convenience builder – it is **not** a context manager.
    """

    def __init__(self, roster: "Roster") -> None:
        self._roster = roster
        self._lock = threading.Lock()
        self._snapshot: MeshRoutingView | None = None

    def replace(self, view: MeshRoutingView) -> None:
        """Replace the current view.  Atomic with respect to .current()."""
        with self._lock:
            self._snapshot = view

    def current(self) -> MeshRoutingView | None:
        """Return the current view (immutable, not held under lock)."""
        with self._lock:
            snap = self._snapshot
        return snap

    def update(self, **kwargs) -> MeshRoutingView:
        """Build a new view from the current roster and kwargs, then replace.

        Convenience method: read the roster under the lock, build, and replace
        atomically.  Returns the new :class:`MeshRoutingView`.
        """
        # Read roster members under lock.
        with self._lock:
            roster_members = list(self._roster.members())

        # Build the snapshot outside the lock.
        new_snap = build_snapshot(self._roster, **kwargs)
        view = MeshRoutingView(
            snapshot=new_snap,
            peer_states={},  # peer_states wired by the caller; empty by default.
        )
        with self._lock:
            self._snapshot = view
        return view


# ---------------------------------------------------------------------------
# Route reason helpers — these keep the existing X-Lobes-Route-Reason values
# and add mesh-specific markers in NEW X-Lobes-Mesh-* headers.
# ---------------------------------------------------------------------------


# Mesh-specific header names (NOT X-Lobes-Route-Reason — those are closed).
MESH_VERIFIED_HEADER = "X-Lobes-Mesh-Verified"
MESH_ORIGIN_HEADER = "X-Lobes-Mesh-Origin"
MESH_ROLE_HEADER = "X-Lobes-Mesh-Role"
MESH_UNVERIFIED_HEADER = "X-Lobes-Mesh-Unverified"


def mesh_markers(
    snapshot: RoutingSnapshot,
    role: str,
    *,
    chosen_origin: str | None = None,
    unverified: bool = False,
) -> list[tuple[str, str]]:
    """Mesh-specific response markers for one routing decision.

    These are NEW headers, not ``X-Lobes-Route-Reason`` — the latter is closed
    and unchanged.  This function adds:

    * ``X-Lobes-Mesh-Origin`` — the origin that will serve (or would serve).
    * ``X-Lobes-Mesh-Role`` — the role being routed.
    * ``X-Lobes-Mesh-Verified`` / ``X-Lobes-Mesh-Unverified`` — whether the
      chosen member was verified for this role.
    """
    markers: list[tuple[str, str]] = []
    if chosen_origin:
        markers.append((MESH_ORIGIN_HEADER, chosen_origin))
    markers.append((MESH_ROLE_HEADER, role))
    if unverified:
        markers.append((MESH_UNVERIFIED_HEADER, "true"))
    elif chosen_origin and snapshot.is_verified(chosen_origin, role):
        markers.append((MESH_VERIFIED_HEADER, "true"))
    return markers


def origins_for_role(
    snapshot: RoutingSnapshot | None,
    role: str,
) -> tuple[str, ...]:
    """Origins that verified support *role*, from *snapshot*.

    Returns an empty tuple when *snapshot* is ``None`` or no member verified
    for the role.  This is the single-point lookup that replaces
    ``table.replica_origins`` and ``table.peer_origins`` in mesh-driven routing.
    """
    if snapshot is None:
        return ()
    return snapshot.member_origins(role)


def member_exists(
    snapshot: RoutingSnapshot | None,
    origin: str,
) -> bool:
    """True when *origin* is in the snapshot."""
    if snapshot is None:
        return False
    return snapshot.has_member(origin)


def roster_member_count(
    snapshot: RoutingSnapshot | None,
) -> int:
    """Number of members in the snapshot."""
    if snapshot is None:
        return 0
    return snapshot.member_count()


def snapshot_member_roles(
    snapshot: RoutingSnapshot | None,
    origin: str,
) -> tuple[str, ...]:
    """Announced roles for *origin*, or empty tuple."""
    if snapshot is None:
        return ()
    return snapshot.announced_roles(origin)
