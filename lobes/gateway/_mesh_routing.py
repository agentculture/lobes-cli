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
    from lobes.gateway._replicas import Fingerprint as ReplicaFingerprint
    from lobes.gateway._replicas import ReplicaState


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
    unverified_reason:
        Short, operator-facing reason the last verification probe of this
        member failed or found nothing verified (item C, t9) — e.g. an HTTP
        status or an exception class name from the ``/capabilities`` dial.
        ``None`` when the member has never been probed, or its last probe
        succeeded. Never derived from the member's own credential or
        response body verbatim (mirrors the ``RejectionLog`` reason
        convention) — just a short, stable category for triage.
    """

    name: str
    origin: str
    announced_roles: tuple[str, ...]
    verified_roles: tuple[str, ...]
    capacity: float
    unverified_reason: str | None = None


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


def as_routing_snapshot(obj: "object | None") -> "RoutingSnapshot | None":
    """Unwrap a :class:`MeshRoutingView` to its :class:`RoutingSnapshot`.

    The snapshot holder publishes a view (snapshot + per-peer replica states);
    every routing/advert consumer wants the snapshot. Accepting both here is
    what keeps a holder-returned view from reaching ``compute_role_placement``
    — the live 2026-09-12 ``/capabilities`` crash on the Thor (Qodo #12).
    """
    if obj is None:
        return None
    if isinstance(obj, MeshRoutingView):
        return obj.snapshot
    return obj  # already a RoutingSnapshot (or a test double)


# ---------------------------------------------------------------------------
# Fingerprint verification
# ---------------------------------------------------------------------------


def fingerprints_identical(a: object, b: object) -> bool:
    """Field-by-field identity of two replica fingerprints (verification rule).

    ``None`` / ``""`` / ``"unknown"`` / ``0`` all normalise to unknown, so an
    unknown on both sides is a match; any other difference is not.
    """
    if a is None or b is None:
        return False

    def norm(v: object) -> str:
        if v is None or v == "" or v == 0 or str(v).lower() == "unknown":
            return "unknown"
        return str(v)

    return all(
        norm(getattr(a, f, None)) == norm(getattr(b, f, None))
        for f in ("served_id", "quantization", "max_model_len", "runtime")
    )


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
            _wire_fingerprint_to_replica(announced_fp) if announced_fp is not None else None
        )
        probed_replica_fp: ReplicaFingerprint | None = (
            _wire_fingerprint_to_replica(probed_fp_data) if probed_fp_data is not None else None
        )

        # Run comparison.
        # VERIFICATION asks "does the member serve what it announced?", so the
        # announced and probed fingerprints must be IDENTICAL, unknowns
        # included — a lane with no declared quantization announces
        # `unknown` and advertises `unknown`, and that is a match. Pooling
        # keeps the strict compare_fingerprints rule (unknown never pools).
        compatible = fingerprints_identical(announced_replica_fp, probed_replica_fp)
        if compatible:
            verified.append(role_name)

    return frozenset(verified)


# ---------------------------------------------------------------------------
# Wire fingerprint converter
# ---------------------------------------------------------------------------


def _wire_fingerprint_to_replica(
    fp: "Fingerprint | Mapping[str, object] | None",
) -> "lobes.gateway._replicas.Fingerprint | None":  # noqa: F821 — resolved at runtime
    """Convert a wire-shaped fingerprint to a replica :class:`~lobes.gateway._replicas.Fingerprint`.

    *fp* may be either a decoded :class:`~lobes.gateway._mesh_wire.Fingerprint`
    (the announced side, from a stored :class:`Announcement`) or a plain
    ``dict`` parsed straight off a peer's ``GET /capabilities`` JSON body (the
    probed side) — both shapes carry the same four field names, so this reads
    them uniformly via ``.get``/``getattr`` rather than assuming one type.

    Conversions (D7):
    * ``max_model_len=0`` → ``None`` (unknown; ``0`` is the wire's "N/A", not
      a real window)
    * ``null`` / ``""`` fields → ``None`` (unknown)
    * Otherwise pass through.
    """
    if fp is None:
        return None

    from lobes.gateway._replicas import Fingerprint as ReplicaFingerprint

    if isinstance(fp, Mapping):
        served_id = fp.get("served_id")
        max_model_len = fp.get("max_model_len")
        runtime = fp.get("runtime")
        quantization = fp.get("quantization")
    else:
        served_id = fp.served_id
        max_model_len = fp.max_model_len
        runtime = fp.runtime
        quantization = fp.quantization

    return ReplicaFingerprint(
        served_id=served_id if served_id else None,
        max_model_len=None if not max_model_len else max_model_len,
        runtime=runtime if runtime else None,
        quantization=quantization if quantization else None,
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
    # Optional per-origin reason the last verification probe failed (item C,
    # t9) — keys are member origins, values are short operator-facing
    # strings (see MemberInfo.unverified_reason). Carried straight onto the
    # matching MemberInfo; an origin absent here simply gets None.
    unverified_reasons: Mapping[str, str] | None = None,
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
    reason_map: dict[str, str] = {} if unverified_reasons is None else dict(unverified_reasons)

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
                unverified_reason=reason_map.get(origin),
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
        # Build the snapshot (build_snapshot reads the roster via its own
        # public, lock-protected accessors — see D4/D3).
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


# ---------------------------------------------------------------------------
# Suffixed-lane naming (t8, issue #237 naming/exposure)
# ---------------------------------------------------------------------------
#
# When two or more mesh members disagree on the FINGERPRINT they serve for
# one role, the role name alone ("cortex") is ambiguous: a plain request
# cannot honestly pick one without silently favouring an arbitrary member. A
# member whose fingerprint for a role does not match the REFERENCE for that
# role (this box's own local fingerprint when it hosts the role, or — when it
# does not — the fingerprint every OTHER verified member agrees on) is
# exposed only as a suffixed lane ``"{role}-{member.name}"``, never through
# the plain role name.  A plain role name is exposed only when every
# candidate member's fingerprint agrees with the reference; on any
# disagreement with no local reference to arbitrate, NO member is exposed
# plain — every one of them is suffixed, so a plain-role request 404s rather
# than silently picking a side.

MESH_MEMBER_HEADER = "X-Lobes-Mesh-Member"


def suffixed_lane_name(role: str, member: str) -> str:
    """The wire name for one member's disagreeing lane of *role*."""
    return f"{role}-{member}"


@dataclass(frozen=True)
class SuffixedLane:
    """One member's role, exposed only under its suffixed name."""

    name: str  # "{role}-{member}"
    role: str
    member: str
    origin: str


@dataclass(frozen=True)
class RolePlacement:
    """Where one role's mesh candidates land: the plain pool, or suffixed.

    ``plain_origins`` are usable for the bare role name (merged into the
    existing replica-pool / mesh-forward machinery exactly as before);
    ``suffixed`` lists every member whose fingerprint disagreed with the
    reference, each addressable only by its own suffixed name.
    """

    role: str
    plain_origins: tuple[str, ...]
    suffixed: tuple[SuffixedLane, ...]

    def suffixed_names(self) -> tuple[str, ...]:
        return tuple(lane.name for lane in self.suffixed)

    def origin_for_suffixed(self, name: str) -> str | None:
        for lane in self.suffixed:
            if lane.name == name:
                return lane.origin
        return None


def _role_fingerprint(
    ann: "Announcement",
    role: str,
) -> "ReplicaFingerprint | None":
    """The replica-shaped Fingerprint a member announced for *role*, if any."""
    info = ann.roles.get(role)
    if info is None:
        return None
    return _wire_fingerprint_to_replica(info.fingerprint)


def compute_role_placement(
    snapshot: RoutingSnapshot,
    role: str,
    *,
    local_fingerprint: "ReplicaFingerprint | None" = None,
) -> RolePlacement:
    """Split *role*'s verified mesh members into plain vs. suffixed lanes.

    Parameters
    ----------
    snapshot:
        The current routing snapshot (verified members + their announcements).
    role:
        The role name being placed (e.g. ``"cortex"``).
    local_fingerprint:
        This box's OWN fingerprint for *role* when it hosts it locally
        (``None`` when this box does not host the role at all — the
        peers-agree-with-each-other fallback applies).
    """
    from lobes.gateway._replicas import compare_fingerprints

    ann_by_origin = dict(snapshot.announcements)

    candidates: list[tuple[str, str, "ReplicaFingerprint | None"]] = []
    for member in snapshot.members:
        if role not in member.verified_roles:
            continue
        ann = ann_by_origin.get(member.origin)
        fp = _role_fingerprint(ann, role) if ann is not None else None
        candidates.append((member.name, member.origin, fp))

    if not candidates:
        return RolePlacement(role=role, plain_origins=(), suffixed=())

    # Deterministic ordering (by member name) so the reference member and the
    # emitted suffixed order never depend on roster/dict iteration order.
    candidates.sort(key=lambda c: c[0])

    if local_fingerprint is not None:
        reference = local_fingerprint
    else:
        # No local hosting: the reference is only trustworthy when every
        # candidate agrees with the FIRST one — otherwise there is no
        # authority to arbitrate and nobody is exposed plain (see module
        # docstring above).
        reference = candidates[0][2]
        all_agree = all(compare_fingerprints(reference, fp)[0] for _n, _o, fp in candidates)
        if not all_agree:
            suffixed = tuple(
                SuffixedLane(
                    name=suffixed_lane_name(role, name),
                    role=role,
                    member=name,
                    origin=origin,
                )
                for name, origin, _fp in candidates
            )
            return RolePlacement(role=role, plain_origins=(), suffixed=suffixed)

    plain: list[str] = []
    suffixed_list: list[SuffixedLane] = []
    for name, origin, fp in candidates:
        compatible, _reason = compare_fingerprints(reference, fp)
        if compatible:
            plain.append(origin)
        else:
            suffixed_list.append(
                SuffixedLane(
                    name=suffixed_lane_name(role, name),
                    role=role,
                    member=name,
                    origin=origin,
                )
            )

    return RolePlacement(role=role, plain_origins=tuple(plain), suffixed=tuple(suffixed_list))


def find_suffixed_lane(
    snapshot: RoutingSnapshot | None,
    requested: str,
    roles: "tuple[str, ...] | list[str]",
    *,
    local_fingerprints: "Mapping[str, ReplicaFingerprint | None] | None" = None,
) -> SuffixedLane | None:
    """Resolve a raw requested name (e.g. ``"cortex-thor"``) to a suffixed lane.

    Tries every role the requested name could plausibly be suffixing (a
    ``"{role}-"`` prefix match), computing that role's placement and checking
    whether *requested* is one of its suffixed names. Returns ``None`` when
    *snapshot* is ``None`` or no role's placement contains *requested*.
    """
    if snapshot is None:
        return None
    local_fingerprints = local_fingerprints or {}
    for role in roles:
        prefix = role + "-"
        if not requested.startswith(prefix):
            continue
        placement = compute_role_placement(
            snapshot, role, local_fingerprint=local_fingerprints.get(role)
        )
        for lane in placement.suffixed:
            if lane.name == requested:
                return lane
    return None


def exposed_role_names(
    snapshot: RoutingSnapshot | None,
    origin: str,
) -> tuple[str, ...]:
    """Role names *origin* currently exposes — plain or suffixed (t8 follow-up).

    For every role *origin* is verified for, this names it as the PLAIN role
    (e.g. ``"cortex"``) when it agrees with that role's placement reference,
    or as its own suffixed lane (e.g. ``"cortex-thor"``) when it disagrees.
    A role announced ``private`` never appears here in the first place: it
    was already stripped from the stored :class:`~lobes.gateway._mesh_wire.Announcement`
    at the wire boundary (``Announcement.public()``, applied on receipt), so
    it is never in ``verified_roles`` to begin with — this function excludes
    nothing extra.

    No ``local_fingerprint`` is passed to :func:`compute_role_placement` here
    deliberately: this is a roster-wide listing (``GET /mesh/roster``), not a
    per-request dispatch with one box's own hosting fingerprint in hand — the
    peers-agree-with-each-other reference is the only one this view has.
    """
    if snapshot is None:
        return ()
    member = next((m for m in snapshot.members if m.origin == origin), None)
    if member is None:
        return ()
    names: list[str] = []
    for role in member.verified_roles:
        placement = compute_role_placement(snapshot, role)
        if origin in placement.plain_origins:
            names.append(role)
        else:
            lane = placement_origin_lane(placement, origin)
            if lane is not None:
                names.append(lane.name)
    return tuple(names)


def placement_origin_lane(placement: RolePlacement, origin: str) -> SuffixedLane | None:
    """The :class:`SuffixedLane` in *placement* served by *origin*, if any."""
    for lane in placement.suffixed:
        if lane.origin == origin:
            return lane
    return None
