"""Mesh route handlers + heartbeat daemon thread.

End-points
----------
* ``GET  /mesh/detect``  – keyless health / presence check
* ``POST /mesh/join``    – keyless join request (flood-collapsed logging)
* ``POST /mesh/announce`` – Bearer-join-key membership heartbeat
* ``GET  /mesh/roster``   – Bearer-join-key member listing
* ``POST /mesh/approve``  – Bearer-join-key approval
* ``POST /mesh/revoke``   – Bearer-join-key revocation
* ``POST /mesh/reannounce`` – Bearer-join-key immediate local re-announce
  (t8): triggered by a readiness-cache health transition or by
  ``lobes switch``/``lobes up``, so a fingerprint/health change reaches
  peers within one probe refresh instead of waiting for the next
  heartbeat.

Heartbeat
---------
A background daemon thread announces the local member to every seed and
roster member at the interval declared in :class:`~lobes.gateway._mesh_config.MeshConfig`.
Each peer gets its own socket so one hung connection never delays another.

The :meth:`MeshRoutes.reannounce_now` method can be called from a request path
to update the announcement payload and trigger an immediate re-broadcast – it
sets the underlying :class:`threading.Event` so the loop wakes within one
second, never blocking the caller.

Usage
-----
``build_mesh_routes()`` returns ``(routes, announcement)`` where
``routes`` is the :class:`MeshRoutes` instance and ``announcement`` is a
:class:`Announcement` carrying every roster member's roles so the heartbeat
thread can broadcast immediately.
"""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import sys
import threading
import time
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lobes.gateway._authlog import RejectionLog
from lobes.gateway._mesh_config import MeshConfig, MeshConfigError, build_mesh_config
from lobes.gateway._mesh_roster import TickResult
from lobes.gateway._mesh_routing import build_snapshot
from lobes.gateway._mesh_wire import (
    SCHEMA_MAJOR,
    Announcement,
    Fingerprint,
    MeshSchemaIncompatible,
    RoleInfo,
    decode,
)

if TYPE_CHECKING:
    from lobes.gateway._mesh_roster import Roster
    from lobes.gateway._mesh_routing import SnapshotHolder


# --- internal data ----------------------------------------------------------


@dataclass(frozen=True)
class _PendingJoin:
    """One pending (unapproved) join request."""

    name: str
    origin: str
    capacity: object
    joined_at: float


# Per-dial timeout constant – a named value, not a derived formula.
_DIAL_TIMEOUT_S: float = 10.0

# t2: how long the heartbeat's wait blocks before re-checking the second wake
# event (`_verify_now_event`).  `Event.wait` takes one object, so the wait is
# sliced; this is the resulting worst-case wake latency for a verify-now
# request, and the upper bound on how long a newly announced member stays
# inside its boot window before the first probe reaches it.
_WAKE_SLICE_S: float = 0.05

# S1192: literals repeated across route handlers, named once.
_JSON: str = "application/json"
_NAME_REQUIRED_MSG: str = "name is required"
_INVALID_KEY_MSG: str = "Invalid API key."
_ANNOUNCE_PATH: str = "/mesh/announce"
_CAPABILITIES_PATH: str = "/capabilities"


# --- shared route-handler helpers (S3776: extracted from the handlers below) -


def _read_request_body(handler: object) -> bytes:
    """Read the raw request body via Content-Length, defensively."""
    rfile = getattr(handler, "rfile", None)
    hdrs = getattr(handler, "headers", {})
    cl = hdrs.get("Content-Length")
    if cl is None:
        return b""
    try:
        length = int(cl)
        if rfile is not None and length > 0:
            return rfile.read(length)
    except (ValueError, AttributeError):
        pass
    return b""


def _parse_json_object(body: bytes) -> dict:
    """Best-effort JSON parse; a malformed body silently yields ``{}``.

    Mirrors the original inline ``try: json.loads(...) except (ValueError,
    TypeError): pass`` pattern exactly — a well-formed-but-non-dict payload
    (e.g. a JSON list) still passes through unchanged, matching prior
    behaviour of every caller.
    """
    try:
        return json.loads(body)
    except (ValueError, TypeError):
        return {}


def _source_from_handler(handler: object) -> str:
    """Derive the request source from the socket peer (finding 11).

    Never trusts a client-supplied field.
    """
    address = getattr(handler, "client_address", None)
    if isinstance(address, tuple) and address:
        return str(address[0])
    return "<unknown>"


def _flat_error_response(
    status: int, message: str, *, close: bool = False
) -> tuple[int, list[tuple[str, str]], bytes]:
    """Build a ``{"error": "<message>"}`` response (the flat error shape)."""
    headers: list[tuple[str, str]] = [("Content-Type", _JSON)]
    if close:
        headers.append(("Connection", "close"))
    return status, headers, json.dumps({"error": message}).encode()


def _error_response(
    status: int,
    message: str,
    err_type: str,
    *,
    close: bool = True,
    www_authenticate: bool = False,
    **extra: object,
) -> tuple[int, list[tuple[str, str]], bytes]:
    """Build a ``{"error": {"message": ..., "type": ..., ...}}`` response.

    The nested error shape shared by every authenticated/roster-mutating
    mesh route.
    """
    headers: list[tuple[str, str]] = [("Content-Type", _JSON)]
    if www_authenticate:
        headers.append(("WWW-Authenticate", "Bearer"))
    if close:
        headers.append(("Connection", "close"))
    payload: dict[str, object] = {"message": message, "type": err_type}
    payload.update(extra)
    return status, headers, json.dumps({"error": payload}).encode()


def _invalid_key_response() -> tuple[int, list[tuple[str, str]], bytes]:
    """The 401 shared by every Bearer-guarded mesh route (finding 12)."""
    return _error_response(
        401,
        _INVALID_KEY_MSG,
        "invalid_api_key",
        www_authenticate=True,
        code="invalid_api_key",
    )


def _name_required_response() -> tuple[int, list[tuple[str, str]], bytes]:
    return _flat_error_response(400, _NAME_REQUIRED_MSG)


# --- main class -------------------------------------------------------------


class MeshRoutes:
    """HTTP route handlers for the ``/mesh/*`` surface."""

    def __init__(
        self,
        config: MeshConfig,
        roster: "Roster",
        # Private: for collapsed flood logging on /mesh/join.
        _join_log: RejectionLog | None = None,
        # Private: for collapsed flood logging of failed verification probes
        # (item C, t9) — mirrors _join_log exactly: None (the default) keeps
        # verification silent-on-failure, unchanged from before this task;
        # server.py's real wiring passes a live RejectionLog.
        _verify_log: RejectionLog | None = None,
    ) -> None:
        self.config = config
        self.roster = roster
        self._announcement: Announcement | None = None
        self._announcement_bytes: bytes | None = None
        # d1: roles a SEED ROSTER listed for a member we hold no announcement
        # for yet (origin -> roles). Provisional: the pending/503 path may name
        # the member from it; nothing is ever routed or verified from it, and a
        # real announcement always wins in build_snapshot.
        self._discovered_roles: dict[str, tuple[str, ...]] = {}
        self._join_log = _join_log
        self._verify_log = _verify_log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._reannounce_event = threading.Event()
        self._pending: list[_PendingJoin] = []
        self._announcements: dict[str, Announcement] = {}
        self._verify_event = threading.Event()
        # t2: the boot window closes on its own.  `_verify_event` is the
        # ordinary "the snapshot is dirty, rebuild it on the next tick" flag;
        # `_verify_now_event` is the stronger "there is a member nobody has
        # probed yet (or one whose fingerprint just changed) — wake the loop
        # NOW" signal.  It is set only by `announce()` (gated: unprobed or a
        # changed fingerprint, so a steady heartbeat storm never re-probes)
        # and by seed discovery.  A pass still only ever RUNS on the
        # heartbeat thread: a request handler sets the event, never probes.
        self._verify_now_event = threading.Event()
        # Single flight: one verification pass at a time, always on the loop
        # thread.  Held non-blocking, so a would-be second pass is skipped
        # rather than queued behind the first (a queued pass would just
        # re-probe the same members with staler data).
        self._verify_pass_lock = threading.Lock()
        self._holder: SnapshotHolder | None = None
        # Rebuild hook for POST /mesh/reannounce (t8): a callable that returns
        # a fresh Announcement reflecting this box's CURRENT state (fingerprint
        # / readiness), so an immediate re-announce after `lobes switch`/`up`
        # or a health transition carries the new data, not the stale one from
        # start-up.  `None` (the default) falls back to re-broadcasting the
        # last-known announcement unchanged — still an immediate wake of the
        # heartbeat loop, just with no new data to send.
        self._announcement_builder = None

    def set_announcement_builder(self, builder) -> None:
        """Wire the callable :meth:`reannounce` uses to rebuild fresh data."""
        with self._lock:
            self._announcement_builder = builder

    @classmethod
    def build(
        cls,
        *,
        env: object | None = None,
        roster: "Roster | None" = None,
    ) -> "MeshRoutes":
        """Factory that wires up a fresh :class:`MeshRoutes`.

        Parameters
        ----------
        env:
            ``os.environ``-like mapping (``None`` → real environment).
        roster:
            Injected roster (``None`` → a new ``Roster(clock=<now>)``).
        """
        if env is None:
            import os

            env = os.environ
        config = build_mesh_config(env)
        clock = time.monotonic
        if roster is None:
            # type: ignore[name-defined] — Roster imported at TYPE_CHECKING guard
            from lobes.gateway._mesh_roster import Roster

            roster = Roster(clock=clock, ledger_path=config.ledger_path)
        return cls(config, roster)

    # -- helpers shared by every Bearer-guarded endpoint ---------------------

    def _check_key(self, headers: dict[str, str]) -> bool:
        """Return True when *headers* carries a valid Bearer join key."""
        # Finding 18: handle None key cleanly.
        if self.config.key is None:
            return False
        auth = headers.get("Authorization", "")
        if not auth:
            return False
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer":
            return False
        import hmac

        return hmac.compare_digest(token.strip().encode("utf-8"), self.config.key.encode("utf-8"))

    # -- route handlers -----------------------------------------------------

    def detect(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """GET /mesh/detect  – keyless presence check.

        Returns only name, schema version and ``mesh: true`` — never a
        member list (security h7).
        """
        body = {
            "mesh": True,
            "name": self.config.name,
            "schema_version": SCHEMA_MAJOR,
        }
        return 200, [("Content-Type", _JSON)], json.dumps(body).encode()

    def _register_pending_join(
        self, name: str, origin: str, capacity: object, now: float, ttl: float
    ) -> tuple[int, list[tuple[str, str]], bytes] | None:
        """Enforce the pending-join queue cap + per-origin dedup, then enqueue.

        Returns an error response tuple, or ``None`` on success. Runs
        entirely under ``self._lock``, matching the original inline block.
        """
        with self._lock:
            pending = [p for p in self._pending if p.joined_at + ttl > now]
            self._pending = pending

            # Capacity cap: 8 pending entries total.
            if len(pending) >= 8:
                return _flat_error_response(400, "pending join queue is full (8)")

            # Finding 11: one entry per ORIGIN (not per name).
            for p in pending:
                if p.origin == origin:
                    return _flat_error_response(400, "same origin already pending")
                # Drop entries with different name but same origin won't happen
                # now that we key on origin above.

            pending.append(_PendingJoin(name=name, origin=origin, capacity=capacity, joined_at=now))
        return None

    def join(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/join – keyless pending join (flood-collapsed logging).

        Capped at 8 pending entries (one per origin) with a 300 s TTL.
        When a source floods past 100 requests in the collapse window,
        one collapsed line is emitted via :class:`~lobes.gateway._authlog.RejectionLog`.
        """
        data = _parse_json_object(_read_request_body(handler))

        name = data.get("name")
        origin = data.get("origin", "")
        capacity = data.get("capacity")

        if not isinstance(name, str) or not name:
            return _name_required_response()

        now = time.monotonic()
        ttl = 300.0  # 5-minute TTL for pending joins
        source = _source_from_handler(handler)

        error = self._register_pending_join(name, origin, capacity, now, ttl)
        if error is not None:
            return error

        # Collapsed flood logging via the existing RejectionLog pattern.
        if self._join_log is not None:
            line = self._join_log.record(
                source,
                "POST",
                "/mesh/join",
                "join_flooded",
            )
            if line is not None:
                sys.stderr.write(f"[gateway] {line}\n")

        # Finding 14: static 202 body, no log state leaked to caller.
        return (
            202,
            [("Content-Type", _JSON)],
            json.dumps({"status": "pending join registered"}).encode(),
        )

    def _decode_announcement_or_response(
        self, body: bytes
    ) -> tuple[Announcement | None, tuple[int, list[tuple[str, str]], bytes] | None]:
        """Decode the wire *body*; returns ``(public_announcement, None)`` or
        ``(None, error_response)`` (finding 15: MeshSchemaIncompatible caught
        distinctly from every other malformed-body failure).
        """
        from lobes.gateway._mesh_wire import decode

        try:
            return decode(body).public(), None
        except MeshSchemaIncompatible:
            # Wrong schema major – reject before touching roster.
            return None, _error_response(
                400,
                "Schema version mismatch",
                "schema_incompatible",
                expected_major=SCHEMA_MAJOR,
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            # S5713: json.JSONDecodeError is a ValueError subclass, already
            # covered — listing it too was a redundant Exception class.
            # Finding 15 (review #252): KeyError/AttributeError added as a
            # boundary safeguard — decode() now validates structure and
            # raises ValueError consistently, but a malformed body must
            # never escape this route as an unhandled 500 either way.
            return None, _flat_error_response(400, "invalid JSON body", close=True)

    def _drop_stale_announcements_for(self, name: str) -> None:
        """Drop any stored announcement from an origin that no longer
        matches the roster record for *name* (the announce() name-conflict
        path)."""
        rec = self.roster._roster.get(name)  # noqa: SLF001
        if rec is None:
            return
        current_origin = rec.origin
        stale_origins = [o for o, a in self._announcements.items() if a.origin != current_origin]
        for o in stale_origins:
            self._announcements.pop(o, None)

    def _admit_announcement(
        self, name: str, origin: str, roster_now: float
    ) -> tuple[int, list[tuple[str, str]], bytes] | None:
        """Admit *name*/*origin* into the roster.

        Returns an error response, or ``None`` on success.

        Finding 5: ledger only RESTRICTS — an absent entry is "no
        restriction", key-holders are admitted permanently.  Only when an
        entry exists AND is not in force (expired/revoked) do we refuse.
        Finding 4 (review #252): the ledger check and the roster admission
        used to be two separate, unsynchronized steps here — a concurrent
        /mesh/revoke could commit in the gap and this request would still
        admit the just-revoked name. announce_gated() does both under one
        lock shared with approve/revoke, so a revoke can never interleave.
        """
        from lobes.gateway._mesh_roster import MeshApprovalExpired, MeshFlapping

        try:
            if name == self.config.name:
                # A peer relaying OUR name (or a misconfigured twin) never
                # enters our own roster — the live Spark listed itself.
                return _error_response(
                    409,
                    "a member cannot announce this box's own name",
                    "mesh_name_conflict",
                    name=name,
                )
            self.roster.announce_gated(name, origin, None, now=roster_now)
        except MeshApprovalExpired:
            return _error_response(403, "Approval has expired", "approval_expired", name=name)
        except MeshFlapping as exc:
            return _error_response(429, str(exc), "mesh_flapping", name=name)
        except Exception as exc:
            if "conflict" in str(exc).lower() or "already held" in str(exc).lower():
                self._drop_stale_announcements_for(name)
                return _error_response(409, "Name conflict", "name_conflict", name=name)
            raise
        return None

    def announce(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/announce – authenticated membership heartbeat."""
        if not self._check_key(getattr(handler, "headers", {})):
            return _invalid_key_response()

        body = _read_request_body(handler)

        public, error = self._decode_announcement_or_response(body)
        if error is not None:
            return error
        name = public.name

        if not name:
            return _name_required_response()

        error = self._ingest_public_announcement(public)
        if error is not None:
            return error

        reply: dict = {"status": "announced", "name": name, "schema_version": SCHEMA_MAJOR}
        # d1: answer with OUR OWN public announcement. A box that has just been
        # recreated holds no peer announcements at all — the seed rosters it
        # fetches list members and roles but carry no announcement — so its
        # first pass had nothing to verify and its mesh-provided roles 404'd
        # until every peer's next heartbeat (57 s measured live, 2026-09-12).
        # Carrying the responder's announcement in the reply lets the
        # announcer learn and verify us on the very pass that reached us.
        # Additive: a pre-d1 announcer ignores the field.
        own = self._own_announcement_object()
        if own is not None:
            reply["announcement"] = own
        return (200, [("Content-Type", _JSON)], json.dumps(reply).encode())

    def _ingest_public_announcement(
        self, public: "Announcement"
    ) -> tuple[int, list[tuple[str, str]], bytes] | None:
        """Admit *public* into the roster and store it; ``None`` on success.

        Shared by the inbound handler and by :meth:`ingest_reply_announcement`
        (d1) so a peer's announcement is admitted through exactly one path
        (self-name refusal, ledger gate, hold-down) whichever way it arrived.
        """
        name = public.name
        origin = public.origin
        error = self._admit_announcement(name, origin, self.roster.now())
        if error is not None:
            return error

        # Store the decoded announcement keyed by origin for snapshot rebuilds.
        previous = self._announcements.get(origin)
        self._announcements[origin] = public

        # Mark verification as dirty so the heartbeat loop rebuilds the
        # snapshot with updated announcement data.
        self._verify_event.set()

        # t2: and ask for an IMMEDIATE pass when — and only when — this
        # announcement carries something a probe has not seen yet.  A member
        # already probed that re-announces the same fingerprints every
        # heartbeat changes nothing, so waking the loop for it would turn a
        # three-member mesh announcing once a second into a probe storm.
        if self._needs_immediate_verify(origin, previous, public):
            self._verify_now_event.set()
        # d2: visible to routing at once — pending until its probe lands.
        self.refresh_routing_view()
        return None

    def refresh_routing_view(self) -> None:
        """Rebuild the routing view from the roster NOW, keeping probe results (d2).

        Cheap and network-free: every member the roster knows appears in the
        view at once — a member learned from an announce reply, an inbound
        announce or a seed roster is PENDING (``probed`` False, its announced
        or discovered roles known) from this instant, so a request for one of
        its roles answers 503 ``role_unverified`` instead of 404 while the
        first probe is still in flight (live 2026-09-12: the verify pass only
        replaced the view when EVERY probe returned, and a paused peer held
        it for the whole probe timeout). Members already probed carry their
        verified / reason / ready data forward unchanged, so a refresh never
        demotes a routable lane. No holder (unit-test wiring) is a no-op.
        """
        holder = self._holder
        if holder is None:
            return
        from lobes.gateway._mesh_routing import MeshRoutingView

        try:
            view = holder.current()
        except Exception:  # nosec B110 — a duck-typed holder never breaks ingest
            return
        prev = getattr(view, "snapshot", None)
        verified: dict[str, frozenset[str]] = {}
        reasons: dict[str, str] = {}
        ready: dict[str, frozenset[str]] = {}
        if prev is not None:
            for m in prev.members:
                if not m.probed:
                    continue
                # ready_roles carries the "probed" trace even when empty (t1).
                ready[m.origin] = frozenset(m.ready_roles)
                if m.verified_roles:
                    verified[m.origin] = frozenset(m.verified_roles)
                if m.unverified_reason is not None:
                    reasons[m.origin] = m.unverified_reason
        snap = build_snapshot(
            self.roster,
            announcements=self._announcements,
            verified_roles=verified,
            unverified_reasons=reasons,
            ready_roles=ready,
            discovered_roles=self._discovered_roles,
        )
        peer_states = getattr(view, "peer_states", None) or {}
        holder.replace(MeshRoutingView(snapshot=snap, peer_states=peer_states))

    def _own_announcement_object(self) -> dict | None:
        """This box's stored public announcement as a JSON object, or ``None``."""
        raw = self._announcement_bytes
        if raw is None:
            return None
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None

    def ingest_reply_announcement(self, body: bytes | None) -> bool:
        """Learn a peer from the ``announcement`` its announce reply carried (d1).

        Returns ``True`` when a well-formed announcement was admitted and
        stored. Anything else — no body, no field, bad JSON, an incompatible
        schema, our own name, a lapsed approval — is ``False`` and leaves the
        roster untouched: the reply is a courtesy, never a requirement.
        """
        if not body:
            return False
        try:
            obj = json.loads(body)
        except (ValueError, TypeError):
            return False
        if not isinstance(obj, dict) or not isinstance(obj.get("announcement"), dict):
            return False
        try:
            ann = decode(json.dumps(obj["announcement"]).encode()).public()
        except Exception:  # nosec B110 — a malformed courtesy field is ignored
            return False
        if not ann.name or not ann.origin:
            return False
        with self._lock:
            return self._ingest_public_announcement(ann) is None

    def _member_probed(self, origin: str) -> bool:
        """Has ANY ``/capabilities`` probe result for *origin* landed yet?

        Read from the current routing snapshot (t1's ``MemberInfo.probed``).
        No snapshot, no member, no holder — all mean "not yet", which is the
        safe answer: it asks for one extra pass, never suppresses one.
        """
        holder = self._holder
        if holder is None:
            return False
        try:
            view = holder.current()
        except Exception:  # nosec B110 — a duck-typed holder never breaks announce
            return False
        if view is None or getattr(view, "snapshot", None) is None:
            return False
        member = next((m for m in view.snapshot.members if m.origin == origin), None)
        return bool(member is not None and member.probed)

    @staticmethod
    def _announcement_fingerprints(ann: "Announcement | None") -> dict:
        """The per-role fingerprints of *ann* — the part a probe verifies."""
        if ann is None:
            return {}
        return {role: info.fingerprint for role, info in ann.roles.items()}

    def _needs_immediate_verify(
        self, origin: str, previous: "Announcement | None", current: "Announcement"
    ) -> bool:
        """Should this announcement wake the loop for an immediate pass?"""
        if not self._member_probed(origin):
            return True
        return self._announcement_fingerprints(previous) != self._announcement_fingerprints(current)

    def _build_member_record(self, mname: str, now: float, snapshot: object | None) -> dict | None:
        """Build one ``/mesh/roster`` member record, or ``None`` when the
        name has no roster record (e.g. dropped between the members() read
        and this lookup — both run under the same lock, so this is
        defensive, not expected).

        t8 follow-up (c27/h1, c46/h37): the CLI's `lobes mesh status` reads
        last_seen_age / expiry / verified / flapping / roles per member —
        all additive, all sourced from state this box already holds (the
        Roster, its Ledger, and the mesh routing snapshot), never new
        tracking. See lobes.gateway._mesh_routing.exposed_role_names for
        the roles column (suffixed names included, private roles excluded
        because they were never in a stored announcement to begin with).
        """
        from lobes.gateway._mesh_roster import _FLAPPING_THRESHOLD
        from lobes.gateway._mesh_routing import exposed_role_names

        rec = self.roster._roster.get(mname)  # noqa: SLF001
        if rec is None:
            return None
        entry = self.roster.ledger.entries.get(mname)
        expiry = entry.expiry if entry is not None else None
        member_info = None
        if snapshot is not None:
            member_info = next((m for m in snapshot.members if m.origin == rec.origin), None)
        verified = bool(member_info.verified_roles) if member_info is not None else False
        roles = list(exposed_role_names(snapshot, rec.origin)) if snapshot is not None else []
        # Item A (t9): the flapping signal is now PER-MEMBER —
        # Roster.announce() (the only path `/mesh/announce` drives)
        # tracks each name's own flap count, so a churning member
        # reports `flapping` for ITSELF without holding out every
        # other name in the roster the way the old roster-wide
        # counter did.
        flap_count = self.roster._flap_counts.get(mname, 0)  # noqa: SLF001
        flapping = flap_count >= _FLAPPING_THRESHOLD
        unverified_reason = member_info.unverified_reason if member_info is not None else None
        # t2 (boot window): "probed and verified nothing" and "not probed
        # yet" both used to read verified:false / unverified_reason:null —
        # indistinguishable, so an operator watching a box come up could not
        # tell a broken peer from one still inside its first pass.  `probed`
        # is the sentinel; while it is False the reason reads the plain
        # string "not_yet_probed" instead of null.
        probed = bool(member_info.probed) if member_info is not None else False
        if not probed:
            unverified_reason = "not_yet_probed"
        return {
            "name": rec.name,
            "origin": rec.origin,
            "capacity": rec.capacity,
            "last_seen_age": max(0.0, now - rec.last_seen),
            "expiry": expiry,
            "verified": verified,
            "flapping": flapping,
            "roles": roles,
            # t2: True once the FIRST probe result for this member has
            # landed, whatever it found.
            "probed": probed,
            # Item C (t9): a short, operator-facing reason the last
            # verification probe found nothing verified, or None on
            # a clean/never-probed member.
            "unverified_reason": unverified_reason,
        }

    def roster_list(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """GET /mesh/roster – authenticated member listing."""
        if not self._check_key(getattr(handler, "headers", {})):
            return _invalid_key_response()

        # Finding 13: take a snapshot under the route lock.
        with self._lock:
            now = self.roster.now()
            snapshot = None
            if self._holder is not None:
                view = self._holder.current()
                if view is not None:
                    snapshot = view.snapshot
            member_records = {}
            for mname in self.roster.members():
                record = self._build_member_record(mname, now, snapshot)
                if record is not None:
                    member_records[mname] = record

        # Finding 9 (review #252): include the ledger's own entries in the
        # authenticated roster exchange. Without this, a revocation made on
        # one node was purely local — /mesh/roster only ever serialized
        # membership records, so a peer that already accepted and verified
        # the member never learned it was revoked, and kept forwarding to it
        # forever. `_fetch_seed_roster` merges these into its own Ledger via
        # `Ledger.merge()` (updated_at wins), matching the announce/approve/
        # revoke gossip contract.
        with self.roster.ledger._lock:  # noqa: SLF001 — snapshot for serialization
            ledger_entries = {
                n: {
                    "approved_by": e.approved_by,
                    "expiry": e.expiry,
                    "updated_at": e.updated_at,
                }
                for n, e in self.roster.ledger.entries.items()
            }

        # Finding 3: return per-member objects with name AND origin.
        return (
            200,
            [("Content-Type", _JSON)],
            json.dumps(
                {"members": list(member_records.values()), "ledger": ledger_entries}
            ).encode(),
        )

    def approve(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/approve – authenticated approval."""
        if not self._check_key(getattr(handler, "headers", {})):
            return _invalid_key_response()

        data = _parse_json_object(_read_request_body(handler))

        name: object = data.get("name")
        expiry: object = data.get("expiry", 3600.0)
        approved_by: str = data.get("approved_by", "requester")

        if not isinstance(name, str) or not name:
            return _name_required_response()

        import math

        try:
            expiry_f = float(expiry)
        except (TypeError, ValueError):
            expiry_f = 3600.0
        # Finding 16 (review #252): reject nan/inf the same way the CLI's
        # _parse_duration_seconds now does — a duration presented as finite
        # must not silently become an immediately-expired (nan) or
        # effectively-permanent (inf) approval.
        if not math.isfinite(expiry_f) or expiry_f <= 0:
            expiry_f = 3600.0

        roster_now = self.roster.now()
        # Finding 6: convert duration to absolute expiry.
        expiry_abs = roster_now + expiry_f
        # Finding 3 (review #252): approve + persist as one transaction under
        # the ledger's own lock, so a concurrent approve/revoke/save can
        # never interleave with this mutation or observe a half-written file.
        self.roster.approve_and_save(str(name), approved_by, expiry_abs, now=roster_now)
        # Mark verification as dirty.
        self._verify_event.set()
        return (
            200,
            [("Content-Type", _JSON)],
            json.dumps({"status": "approved", "name": name}).encode(),
        )

    def revoke(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/revoke – authenticated revocation."""
        if not self._check_key(getattr(handler, "headers", {})):
            return _invalid_key_response()

        data = _parse_json_object(_read_request_body(handler))

        name: object = data.get("name")
        approved_by: str = data.get("approved_by", "requester")

        if not isinstance(name, str) or not name:
            return _name_required_response()

        roster_now = self.roster.now()
        # Finding 3 (review #252): revoke + persist as one transaction under
        # the ledger's own lock — see the matching approve_and_save note above.
        self.roster.revoke_and_save(str(name), now=roster_now, approved_by=approved_by)
        # Remove the revoked member's announcement and rebuild the snapshot
        # so revoked members get zero forwards immediately.
        with self._lock:
            self._announcements.pop(name, None)
            snap = build_snapshot(
                self.roster,
                announcements=self._announcements,
                discovered_roles=self._discovered_roles,
            )
            if self._holder is not None:
                from lobes.gateway._mesh_routing import MeshRoutingView

                self._holder.replace(MeshRoutingView(snapshot=snap, peer_states={}))
        return (
            200,
            [("Content-Type", _JSON)],
            json.dumps({"status": "revoked", "name": name}).encode(),
        )

    def reannounce(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/reannounce – authenticated, LOCAL-only immediate re-broadcast.

        Triggered by this box's own readiness-cache health transitions and by
        ``lobes switch``/``lobes up`` (t8): whenever this box's own served
        fingerprint or health changes, callers hit this endpoint so peers
        reflect the change within one probe refresh instead of waiting for
        the next scheduled heartbeat. Gated on the same Bearer join key as
        every other authenticated mesh route — it is a local operator/CLI
        call, never something a remote peer needs to invoke.
        """
        if not self._check_key(getattr(handler, "headers", {})):
            return _invalid_key_response()

        with self._lock:
            builder = self._announcement_builder
            current = self._announcement

        fresh: Announcement | None = None
        if builder is not None:
            try:
                fresh = builder()
            except Exception:  # nosec B110 — a broken builder must not wedge reannounce
                fresh = None

        announcement = fresh if fresh is not None else current
        if announcement is None:
            # Nothing has ever been announced from this box yet — nothing to
            # re-broadcast, but this is not an error condition.
            return (
                200,
                [("Content-Type", _JSON)],
                json.dumps({"status": "no-op", "reason": "no announcement yet"}).encode(),
            )

        reannounce_now(self, announcement)
        return (
            200,
            [("Content-Type", _JSON)],
            json.dumps({"status": "reannounced", "name": announcement.name}).encode(),
        )


# --- route registry --------------------------------------------------------

_MESH_ROUTES: dict[tuple[str, str], str] = {
    ("GET", "/mesh/detect"): "detect",
    ("POST", "/mesh/join"): "join",
    ("POST", _ANNOUNCE_PATH): "announce",
    ("GET", "/mesh/roster"): "roster_list",
    ("POST", "/mesh/approve"): "approve",
    ("POST", "/mesh/revoke"): "revoke",
    ("POST", "/mesh/reannounce"): "reannounce",
}


# --- server integration helpers --------------------------------------------


def is_mesh_route(path: str) -> bool:
    """True when *path* is a known mesh endpoint path."""
    route_path = path.split("?", 1)[0]
    return route_path.startswith("/mesh/")


def dispatch_mesh(
    handler: object,
    routes: "MeshRoutes",
) -> tuple[int, list[tuple[str, str]], bytes] | None:
    """Dispatch a mesh request.  Returns ``(status, headers, body)`` or ``None``."""
    route_path = handler.path.split("?", 1)[0] if hasattr(handler, "path") else ""
    method = getattr(handler, "command", "GET")
    key = (method, route_path)
    method_name = _MESH_ROUTES.get(key)
    if method_name is None:
        return None
    handler_func = getattr(routes, method_name)
    status, headers, body = handler_func(handler)
    return status, headers, body


def _fingerprint_to_wire(
    fp: object,
) -> Fingerprint:
    """Convert a Fingerprint-like mapping to a wire Fingerprint."""
    if fp is None:
        return Fingerprint(
            served_id="",
            quantization="",
            max_model_len=0,
            runtime="",
        )
    if isinstance(fp, Fingerprint):
        return fp
    return Fingerprint(
        served_id=str(fp.get("served_id", "")),
        quantization=str(fp.get("quantization", "")),
        max_model_len=int(fp.get("max_model_len", 0)),
        runtime=str(fp.get("runtime", "")),
    )


def _role_info_from_capability_entry(
    role: str,
    entry: Mapping[str, object],
    local_capacities: Mapping[str, float] | None,
) -> RoleInfo | None:
    """Build one :class:`RoleInfo` from a ``/capabilities`` payload entry.

    Returns ``None`` when the entry should be skipped — not feasible, no
    fingerprint, or proxied (a member announces only what it hosts here).
    """
    from lobes.roles import ROLE_BACKEND

    fp = entry.get("fingerprint")
    if not entry.get("feasible") or not isinstance(fp, Mapping) or entry.get("proxied"):
        return None
    try:
        max_len = int(fp.get("max_model_len") or 0)
    except (TypeError, ValueError):
        max_len = 0
    backend = ROLE_BACKEND.get(role, role)
    capacity = None
    if local_capacities and backend in local_capacities:
        capacity = local_capacities[backend]
    return RoleInfo(
        model=str(entry.get("model") or fp.get("served_id") or ""),
        runtime=str(entry.get("runtime") or ""),
        context=int(entry.get("context") or max_len or 0),
        quant=str(entry.get("quant") or ""),
        responsibilities=tuple(entry.get("responsibilities") or ()),
        forbidden_responsibilities=tuple(entry.get("forbidden_responsibilities") or ()),
        fingerprint=Fingerprint(
            served_id=str(fp.get("served_id") or ""),
            quantization=str(fp.get("quantization") or ""),
            max_model_len=max_len,
            runtime=str(fp.get("runtime") or ""),
        ),
        capacity=capacity,
    )


def announcement_from_capabilities(
    config: MeshConfig,
    payload: Mapping[str, Mapping[str, object]],
    *,
    self_origin: str,
    local_capacities: Mapping[str, float] | None = None,
) -> Announcement:
    """Build this box's announcement from its OWN ``/capabilities`` payload.

    A member announces exactly what it advertises — the roles it hosts here
    (``feasible`` and carrying a ``fingerprint``, never a proxied one), with
    the very fingerprint a peer will read back when it verifies. Building the
    announcement from any other source is how the first live cutover ended up
    announcing all six roles with empty served ids (2026-09-12).
    """
    roles: dict[str, RoleInfo] = {}
    for role, entry in payload.items():
        if not isinstance(entry, Mapping):
            continue
        info = _role_info_from_capability_entry(role, entry, local_capacities)
        if info is not None:
            roles[str(role)] = info
    return Announcement(
        name=config.name or "", origin=self_origin, schema_version=str(SCHEMA_MAJOR), roles=roles
    )


def _resolve_ready_roles(readiness_cache: object | None) -> dict[str, bool | None]:
    """Return the readiness cache's flat ``backend -> ready`` map, or ``{}``.

    Finding 1/6: ``ReadinessCache.current()`` returns a flat
    ``dict[str, bool | None]`` keyed by BACKEND name ("primary",
    "multimodal", …), never nested and never keyed by role name.
    """
    if readiness_cache is None:
        return {}
    try:
        return readiness_cache.current()  # type: ignore[return-value]
    except (AttributeError, TypeError):
        return {}


def _live_fingerprint_from_cache(
    replica_caches: dict[str, object] | None, backend_name: str
) -> object | None:
    """Look up the LOCAL replica's live fingerprint for *backend_name*.

    ``replica_caches`` (`build_replica_caches`) is keyed by BACKEND name,
    never role name — look it up the same way the pool itself does.
    """
    if not replica_caches or backend_name not in replica_caches:
        return None
    cache = replica_caches[backend_name]
    try:
        # cache.current() returns tuple[ReplicaState]; find local=True.
        states = cache.current()
        if isinstance(states, (list, tuple)):
            for st in states:
                if getattr(st, "local", False):
                    return getattr(st, "fingerprint", None)
    except (AttributeError, TypeError):
        pass
    return None


def _role_info_from_lane_config(
    backend_name: str,
    lane_config: dict[str, str],
    *,
    replica_caches: dict[str, object] | None,
    local_capacities: dict[str, float] | None,
) -> RoleInfo:
    """Build one :class:`RoleInfo` from a declared lane config.

    Enriched with a live replica-cache fingerprint when available; falls
    back to the DECLARED lane fields otherwise (the common case: no
    ``*_PEER_ORIGINS`` pool declared anywhere means ``build_replica_caches``
    returns ``{}`` outright) — so a plain, unpooled box still announces a
    real, comparable fingerprint instead of an all-empty one that can never
    be verified by a peer's probe.
    """
    fp = _live_fingerprint_from_cache(replica_caches, backend_name)

    capacity = None
    if local_capacities and backend_name in local_capacities:
        capacity = local_capacities[backend_name]

    model = lane_config.get("model", "")
    runtime = lane_config.get("runtime", "")
    context = int(lane_config.get("context", 0))
    quant = lane_config.get("quant", "")
    resp_list = lane_config.get("responsibilities", [])
    if isinstance(resp_list, str):
        resp_list = [resp_list]
    forbidden_list = lane_config.get("forbidden_responsibilities", [])
    if isinstance(forbidden_list, str):
        forbidden_list = [forbidden_list]

    if fp is None:
        fp = {
            "served_id": model,
            "quantization": quant,
            "max_model_len": context,
            "runtime": runtime,
        }

    return RoleInfo(
        model=model,
        runtime=runtime,
        context=context,
        quant=quant,
        responsibilities=tuple(resp_list),
        forbidden_responsibilities=tuple(forbidden_list),
        fingerprint=_fingerprint_to_wire(fp),
        capacity=capacity,
        private=lane_config.get("private", False),
    )


def _build_announcement(
    config: MeshConfig,
    *,
    self_origin: str | None = None,
    readiness_cache: object | None = None,
    replica_caches: dict[str, object] | None = None,
    local_capacities: dict[str, float] | None = None,
    declared_lane_configs: dict[str, dict[str, str]] | None = None,
) -> Announcement:
    """Build the wire Announcement for THIS box's own heartbeat broadcast.

    Parameters
    ----------
    config:
        MeshConfig with name and key.
    self_origin:
        The client-reachable origin (from reachable_origin / public_url).
    readiness_cache:
        Optional readiness cache for determining which roles are ready.
    replica_caches:
        Optional dict of backend → ReplicaCache for live fingerprint data.
    local_capacities:
        Optional dict of backend name → capacity.
    declared_lane_configs:
        Optional dict of backend name → lane config dict.
    """
    origin = self_origin or ""  # never a name: an origin is a URL an operator typed (#92)

    # Finding 1: build real roles from the gateway's own data.
    from lobes.roles import BACKEND_ROLE

    ready_roles = _resolve_ready_roles(readiness_cache)

    roles: dict[str, RoleInfo] = {}
    for backend_name, lane_config in (declared_lane_configs or {}).items():
        # Check readiness: only include ready+hosted roles (finding 6, review
        # #252 — see _resolve_ready_roles for why this is keyed by BACKEND
        # name, never role name).
        if ready_roles and ready_roles.get(backend_name) is not True:
            continue

        # Convert backend name → role name (mesh speaks roles: cortex, senses, …)
        role_name = BACKEND_ROLE.get(backend_name, backend_name)
        roles[role_name] = _role_info_from_lane_config(
            backend_name,
            lane_config,
            replica_caches=replica_caches,
            local_capacities=local_capacities,
        )

    return Announcement(
        name=config.name or "",
        origin=origin,
        schema_version=str(SCHEMA_MAJOR),
        roles=roles,
    )


def build_mesh_routes(
    *,
    env: object | None = None,
    roster: "Roster | None" = None,
    # Optional gateway data for building a real announcement (finding 1).
    self_origin: str | None = None,
    readiness_cache: object | None = None,
    replica_caches: dict[str, object] | None = None,
    local_capacities: dict[str, float] | None = None,
    declared_lane_configs: dict[str, dict[str, str]] | None = None,
    join_log: RejectionLog | None = None,
    verify_log: RejectionLog | None = None,
    missed_max: int | None = None,
) -> tuple["MeshRoutes", Announcement]:
    """Build and return a :class:`MeshRoutes` + the initial heartbeat announcement.

    Returns
    -------
    ``(routes, announcement)`` – the routes instance and a
    :class:`Announcement` carrying every roster member's roles so the
    heartbeat thread can broadcast immediately.
    """
    from lobes.gateway._mesh_roster import Roster as _Roster  # avoid circular import

    if env is None:
        import os

        env = os.environ

    config = build_mesh_config(env)
    clock = time.monotonic
    if roster is None:
        roster = _Roster(clock=clock, ledger_path=config.ledger_path)

    # Finding 10: inject missed_max into Roster.
    if missed_max is not None:
        roster = _Roster(clock=clock, ledger_path=config.ledger_path, missed_max=missed_max)

    routes = MeshRoutes(config, roster, _join_log=join_log, _verify_log=verify_log)
    # Create the snapshot holder and attach it to routes.
    from lobes.gateway._mesh_routing import SnapshotHolder

    holder = SnapshotHolder(roster)
    routes._holder = holder  # noqa: SLF001

    # Build the initial announcement from gateway data (finding 1).
    announcement = _build_announcement(
        config,
        self_origin=self_origin,
        readiness_cache=readiness_cache,
        replica_caches=replica_caches,
        local_capacities=local_capacities,
        declared_lane_configs=declared_lane_configs,
    )
    return routes, announcement


# --- heartbeat daemon ------------------------------------------------------


def require_self_origin(self_origin: str | None) -> str:
    """Return the operator-typed self origin, or refuse to start the mesh.

    The announced origin is the URL peers dial back; it is GATEWAY_SELF_ORIGIN,
    typed once by the operator, never a name or a guessed URL (#92). A mesh
    with the join key set but no self origin is a misconfiguration named at
    startup rather than a member that announces an unreachable address.
    """
    origin = (self_origin or "").strip()
    if not origin:
        raise MeshConfigError(
            "LOBES_MESH_KEY is set but GATEWAY_SELF_ORIGIN is empty — the mesh "
            "announces this box by that URL; set it to the origin peers dial."
        )
    return origin


def _post_announcement(
    url: str,
    body: bytes,
    timeout: float,
    key: bytes | str | None = None,
) -> bytes | None:
    """POST *body* to *url* via http.client with a hard timeout.

    Returns the reply body on a 200 (d1: it may carry the responder's own
    announcement), ``None`` otherwise. Silently drops on any failure — a down
    peer is handled by tick-based staleness in the roster.

    Parameters
    ----------
    url:
        Full URL (including scheme).
    body:
        JSON body bytes.
    timeout:
        Socket timeout in seconds.
    key:
        Bearer key bytes (finding 2).
    """
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or _ANNOUNCE_PATH
    if parsed.query:
        path = path + "?" + parsed.query

    # Finding 16: scheme-aware connection.
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(  # type: ignore[call-overload]
            host, port, timeout=timeout
        )
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)

    headers: dict[str, str] = {
        "Content-Type": _JSON,
    }
    # Finding 2: attach Bearer key.
    if key is not None:
        # The join key reaches here as the str MeshConfig parsed; a bytes key is
        # accepted too. Decoding a str raised AttributeError inside the
        # best-effort catch on every announce — the live 2026-09-12 silent mesh.
        key_text = key.decode("utf-8") if isinstance(key, bytes) else key
        headers["Authorization"] = f"Bearer {key_text}"

    reply: bytes | None = None
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers=headers,
        )
        resp = conn.getresponse()
        data = resp.read(_MAX_REPLY_BYTES)
        if resp.status == 200:
            reply = data
    except Exception:  # nosec B110 — best-effort: silently drop failed peer connections
        pass
    finally:
        # S5727: `conn` is always assigned above (HTTPSConnection or
        # HTTPConnection) — the prior `if conn is not None:` guard could
        # never be False. Close directly.
        try:
            conn.close()
        except Exception:  # nosec B110 — silently ignore close errors
            pass
    return reply


# d1: an announce reply is small (status + one public announcement); cap the
# read so a misbehaving peer cannot make the heartbeat thread buffer freely.
_MAX_REPLY_BYTES = 256 * 1024


def _wait_for_tick(
    deadline: float | None,
    interval: float,
    reannounce_event: threading.Event,
    verify_now_event: threading.Event | None = None,
) -> tuple[float, bool]:
    """Sleep until *deadline* or a wake event, whichever comes first.

    Waits in ≤1s increments so the caller's ``stop_event`` check stays
    responsive.  Returns ``(deadline, woken_by_event)`` — a ``None``
    input deadline is resolved to ``now + interval`` without waiting (the
    loop's first iteration). Finding 4: paces on the full interval, not
    ``min(interval, 1.0)``.

    t2: there are now TWO wake events — ``reannounce_event`` (re-broadcast
    this box's own announcement) and ``verify_now_event`` (a member nobody
    has probed yet, or one whose fingerprint just changed).  ``Event.wait``
    takes one object, so the wait is sliced: each slice blocks on the
    reannounce event and re-checks the verify-now event, which bounds the
    wake latency at ``_WAKE_SLICE_S`` for the verify half while keeping the
    reannounce half instantaneous.  Both events are consumed by the wait.
    """
    woken_by_event = False
    if deadline is None:
        deadline = time.monotonic() + interval
    else:
        budget = min(deadline - time.monotonic(), 1.0)
        while budget > 0 and not woken_by_event:
            if verify_now_event is not None and verify_now_event.is_set():
                woken_by_event = True
                break
            # Finding 17 (review #252): capture wait()'s own return value
            # BEFORE clearing the event. The old code cleared first, so
            # `reannounce_event.is_set()` below was always False and an
            # immediate re-announce (POST /mesh/reannounce, `lobes
            # switch`/`up`) never ran a pass until the ordinary deadline —
            # exactly the bug the event exists to avoid.
            slice_dur = min(budget, _WAKE_SLICE_S)
            woken_by_event = reannounce_event.wait(timeout=slice_dur)
            if verify_now_event is not None and verify_now_event.is_set():
                woken_by_event = True
            budget -= slice_dur
    reannounce_event.clear()
    if verify_now_event is not None:
        verify_now_event.clear()
    return deadline, woken_by_event


def _prune_dropped_announcements(
    routes: "MeshRoutes", tick_result: TickResult, holder: "SnapshotHolder | None"
) -> None:
    """Drop refresh: prune announcements for the DROPPED members only.

    review #252 finding 2: iterating the SURVIVORS and popping their
    announcements — the old code — did the exact opposite of what a single
    expiry should do: it wiped every still-healthy member's announcement,
    leaving nothing for the next verification pass to verify. `tick()` now
    names the dropped origins directly. Called under ``routes._lock``.
    """
    if not tick_result.dropped_origins or holder is None:
        return
    try:
        from lobes.gateway._mesh_routing import MeshRoutingView

        for origin in tick_result.dropped_origins:
            routes._announcements.pop(origin, None)
        snap = build_snapshot(
            routes.roster,
            announcements=routes._announcements,
            discovered_roles=routes._discovered_roles,  # noqa: SLF001
        )
        holder.replace(MeshRoutingView(snapshot=snap, peer_states={}))
    except Exception:  # nosec B110 — best-effort: drop refresh never blocks
        pass


def _tick_and_collect(
    routes: "MeshRoutes",
    announcement_bytes: bytes,
    holder: "SnapshotHolder | None",
) -> tuple[bytes | None, bool]:
    """Tick the roster, prune dropped members, and pick the bytes to send.

    Finding 10: tick the roster once per pass under the lock. Returns
    ``(to_send, verify_dirty)`` — only DECIDES about verification under the
    lock; the verification pass itself dials every member over the network
    and re-takes this lock inside :func:`verify_members` — running it here
    starved every roster read and inbound announce for the probe timeouts
    (live Spark, dev518: /mesh/roster never answered).
    """
    try:
        with routes._lock:
            try:
                tick_result = routes.roster.tick()
            except Exception:  # nosec B110 — best-effort: tick never blocks
                tick_result = TickResult()

            _prune_dropped_announcements(routes, tick_result, holder)

            verify_dirty = routes._verify_event.is_set()
            if verify_dirty:
                routes._verify_event.clear()

            to_send = (
                routes._announcement_bytes
                if routes._announcement_bytes is not None
                else announcement_bytes
            )
        return to_send, verify_dirty
    except Exception:  # nosec B110 — best-effort: lock block never blocks
        return announcement_bytes, False


def _collect_member_origins(routes: "MeshRoutes") -> list[tuple[str, str]]:
    """Gather ``(name, origin)`` for every roster member under the lock
    (finding 9), so the dial loop below can run outside it.
    """
    member_origins: list[tuple[str, str]] = []
    with routes._lock:
        for member_name in routes.roster.members():
            rec = routes.roster._roster.get(member_name)  # noqa: SLF001
            if rec is not None:
                member_origins.append((member_name, rec.origin))
    return member_origins


def _maybe_fetch_seed_roster(
    routes: "MeshRoutes", seeds: tuple[str, ...], dial_timeout: float
) -> None:
    """Finding 3: fetch + merge every seed's roster, best-effort."""
    if not seeds:
        return
    try:
        # Finding 5 (review #252): named arguments — the old positional call
        # swapped `timeout` and `routes`, handing a MeshRoutes instance to
        # http.client as a socket timeout. The resulting TypeError was
        # swallowed by this very except-Exception, so seed rosters were
        # never fetched.
        _fetch_seed_roster(
            seeds, routes.config.key, routes.roster, timeout=dial_timeout, routes=routes
        )
    except Exception:  # nosec B110 — best-effort: seed fetch never blocks
        pass


def _broadcast_announcement(
    to_send: bytes,
    seeds: tuple[str, ...],
    member_origins: list[tuple[str, str]],
    stop_event: threading.Event,
    dial_timeout: float,
    key: str | None,
    routes: "MeshRoutes | None" = None,
) -> None:
    """Finding 9: parallelize announces to every seed + roster member.

    d1: every reply is offered to :meth:`MeshRoutes.ingest_reply_announcement`
    so a peer's own announcement, carried in its reply, lands the moment we
    reach it — the recreated-box boot window closes on the first pass.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(seeds) + len(member_origins) + 1)
    ) as pool:
        futures = []

        # Announce to seeds.
        for seed in seeds:
            if stop_event.is_set():
                break
            url = seed + _ANNOUNCE_PATH
            futures.append(pool.submit(_post_announcement, url, to_send, dial_timeout, key))

        # Announce to every roster member.
        for _member_name, origin in member_origins:
            if stop_event.is_set():
                break
            url = origin + _ANNOUNCE_PATH
            futures.append(pool.submit(_post_announcement, url, to_send, dial_timeout, key))

        # Wait for all dials to complete (each has its own timeout).
        for fut in concurrent.futures.as_completed(futures):
            try:
                reply = fut.result(timeout=dial_timeout)
            except Exception:  # nosec B112 — best-effort: drop failed peer connections
                continue
            if routes is not None and reply:
                try:
                    routes.ingest_reply_announcement(reply)
                except Exception:  # nosec B110 — a bad reply never breaks the pass
                    pass


def _run_heartbeat_pass(
    routes: "MeshRoutes",
    announcement_bytes: bytes,
    stop_event: threading.Event,
    seeds: "tuple[str, ...] | list[str]",
    holder: "SnapshotHolder | None",
    force_verify: bool,
) -> None:
    """Run ONE heartbeat pass: tick, verify if dirty, then broadcast.

    Extracted from :func:`_heartbeat_loop` so each half stays within the
    project's cognitive-complexity budget; the sequencing is unchanged.
    *force_verify* is the loop's first iteration (see the comment below).
    """
    to_send, verify_dirty = _tick_and_collect(routes, announcement_bytes, holder)
    # t2: the first iteration always verifies.  It used to `continue`
    # here, so on a box that started with members already in its roster
    # (a restored roster, a seeded peer) nothing was probed until a whole
    # heartbeat interval had passed — a boot window at least one tick
    # long, for no reason.
    if force_verify:
        verify_dirty = True

    # Verification pass OUTSIDE the lock: network I/O never holds it.
    if verify_dirty:
        try:
            _run_verify_pass(routes, holder)
        except Exception:  # nosec B110 — verification is best-effort
            pass

    if to_send is None:
        return

    member_origins = _collect_member_origins(routes)
    # Find 10: use named per-dial budget, not missed_max * 10.
    dial_timeout = _DIAL_TIMEOUT_S

    _maybe_fetch_seed_roster(routes, seeds, dial_timeout)
    _broadcast_announcement(
        to_send,
        seeds,
        member_origins,
        stop_event,
        dial_timeout,
        routes.config.key,
        routes=routes,
    )


def _heartbeat_loop(
    routes: "MeshRoutes",
    announcement_bytes: bytes,
    interval: float,
    stop_event: threading.Event,
    reannounce_event: threading.Event,
    holder: "SnapshotHolder | None" = None,
    verify_now_event: threading.Event | None = None,
) -> None:
    """Background thread that announces the local member to seeds + roster.

    Each peer gets its own socket with its own timeout so one hung peer never
    delays another.  The loop exits when *stop_event* is set.
    """
    seeds = routes.config.seeds
    if verify_now_event is None:
        verify_now_event = routes._verify_now_event  # noqa: SLF001
    deadline: float | None = None
    first_iteration = True
    while not stop_event.is_set():
        deadline, woken_by_event = _wait_for_tick(
            deadline, interval, reannounce_event, verify_now_event
        )

        if stop_event.is_set():
            break

        # Check if we should run a pass: the first iteration, the deadline
        # elapsing, or a wake event (reannounce / verify-now).
        now = time.monotonic()
        run_pass = first_iteration or (now >= deadline) or woken_by_event

        if run_pass:
            # Advance to next deadline.
            deadline = now + interval
        else:
            continue

        _run_heartbeat_pass(routes, announcement_bytes, stop_event, seeds, holder, first_iteration)
        first_iteration = False


def _seed_connection(
    seed: str, timeout: float
) -> tuple[http.client.HTTPConnection, str, dict[str, str]]:
    """Build the connection, path and headers for one seed's GET /mesh/roster."""
    parsed = urllib.parse.urlsplit(seed)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/mesh/roster"
    if parsed.query:
        path = path + "?" + parsed.query

    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(  # type: ignore[call-overload]
            host, port, timeout=timeout
        )
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)

    return conn, path, {"Content-Type": _JSON}


def _record_discovered_roles(routes: "MeshRoutes | None", morigin: str, roles: object) -> None:
    """Remember the roles a seed listed for *morigin* (d1).

    Provisional, so the pending (503) path can name a member we hold no
    announcement for yet. Never routed, never verified from; a real
    announcement wins (see build_snapshot's discovered_roles).
    """
    if routes is None or not isinstance(roles, list):
        return
    routes._discovered_roles[morigin] = tuple(  # noqa: SLF001
        sorted(r for r in roles if isinstance(r, str) and r)
    )


def _merge_one_seed_member(roster: "Roster", routes: "MeshRoutes | None", member: object) -> None:
    """Discovery-merge ONE entry of a seed's ``members`` list into *roster*."""
    if not isinstance(member, dict):
        return
    mname = member.get("name", "")
    morigin = member.get("origin", "")
    if routes is not None and mname == routes.config.name:
        # A peer's roster lists US; never merge ourselves in.
        return
    if not mname or not morigin:
        return

    _record_discovered_roles(routes, morigin, member.get("roles"))
    # DISCOVERY only: a peer's roster tells us a member exists; it
    # is not a heartbeat FROM that member. `discover` never
    # refreshes a known name (a stopped Thor stayed alive 4+ min,
    # live 2026-09-12) and is refused during the post-drop
    # hold-down (the pass that dropped it re-learned it from the
    # Orin and the survivors revived it forever, dev526). Roster
    # takes its own lock; wrapping it in that same lock deadlocked
    # the heartbeat live (Orin).
    newly_discovered = mname not in roster.members()
    roster.discover(mname, morigin, None, now=time.monotonic())
    # t2: a member we have never seen before is, by definition,
    # never-probed — ask the loop for an immediate pass rather than
    # leaving it inside the boot window until the next tick.  Only
    # NEW names set the event: a seed roster re-lists every known
    # member on every fetch, and waking on those would make the
    # verify-now event fire once per tick per seed forever.
    if newly_discovered and routes is not None:
        routes._verify_event.set()  # noqa: SLF001
        routes._verify_now_event.set()  # noqa: SLF001
        # d2: in the routing view now, pending on its discovered roles.
        routes.refresh_routing_view()


def _merge_seed_members(roster: "Roster", routes: "MeshRoutes | None", members: object) -> None:
    """Discovery-merge a seed's ``members`` list into *roster* (finding 3)."""
    if not isinstance(members, list):
        return
    for member in members:
        _merge_one_seed_member(roster, routes, member)


def _rebuild_routing_after_revocations(
    routes: "MeshRoutes", roster: "Roster", newly_unapproved: list[str]
) -> None:
    """Rebuild routing so a merged revocation removes the member from
    forwarding immediately, exactly as the local /mesh/revoke handler
    already does.
    """
    with routes._lock:  # noqa: SLF001
        for pname in newly_unapproved:
            rec = roster._roster.get(pname)  # noqa: SLF001
            if rec is None:
                continue
            routes._announcements.pop(rec.origin, None)  # noqa: SLF001
        if routes._holder is not None:  # noqa: SLF001
            from lobes.gateway._mesh_routing import MeshRoutingView

            snap = build_snapshot(
                roster,
                announcements=routes._announcements,  # noqa: SLF001
                discovered_roles=routes._discovered_roles,  # noqa: SLF001
            )
            routes._holder.replace(MeshRoutingView(snapshot=snap, peer_states={}))  # noqa: SLF001


def _merge_seed_ledger(roster: "Roster", routes: "MeshRoutes | None", ledger_data: object) -> None:
    """Merge a seed's ``ledger`` entries into *roster*.

    Finding 9 (review #252): merge the peer's LEDGER too, not just its
    membership records — the ledger is what a revocation lives in. Without
    this, a revoke made on one node never reached a peer that had already
    accepted and verified the member, so that peer kept forwarding to it
    forever. `Ledger.merge()` applies the standard updated_at-wins gossip
    rule.
    """
    if not isinstance(ledger_data, dict) or not ledger_data:
        return
    from lobes.gateway._mesh_roster import Ledger, _LedgerEntry

    peer_ledger = Ledger(path=None)
    for pname, pinfo in ledger_data.items():
        if not isinstance(pinfo, dict):
            continue
        try:
            peer_ledger.entries[pname] = _LedgerEntry(
                approved_by=str(pinfo["approved_by"]),
                expiry=float(pinfo["expiry"]),
                updated_at=float(pinfo["updated_at"]),
            )
        except (KeyError, TypeError, ValueError):
            continue

    merge_now = time.monotonic()
    roster.merge(peer_ledger)
    roster.save()
    newly_unapproved = [
        pname for pname in peer_ledger.entries if not roster.is_approved(pname, now=merge_now)
    ]
    if routes is not None and newly_unapproved:
        _rebuild_routing_after_revocations(routes, roster, newly_unapproved)


def _process_seed_roster_response(
    roster: "Roster", routes: "MeshRoutes | None", raw: bytes
) -> None:
    """Parse and merge one seed's /mesh/roster 200 response body."""
    try:
        data = json.loads(raw)
        _merge_seed_members(roster, routes, data.get("members", []))
        _merge_seed_ledger(roster, routes, data.get("ledger", {}))
    except (json.JSONDecodeError, TypeError, KeyError):
        pass


def _fetch_one_seed_roster(
    seed: str,
    bearer: str | None,
    roster: "Roster",
    timeout: float,
    routes: "MeshRoutes | None",
) -> None:
    """GET /mesh/roster from one seed and merge its response."""
    conn, path, headers = _seed_connection(seed, timeout)
    if bearer:
        headers["Authorization"] = bearer

    try:
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        if resp.status == 200:
            _process_seed_roster_response(roster, routes, resp.read())
    except Exception:  # nosec B110 — best-effort: seed roster fetch never blocks
        pass
    finally:
        try:
            conn.close()
        except Exception:  # nosec B110 — silently ignore close errors
            pass


def _fetch_seed_roster(
    seeds: tuple[str, ...],
    key: str | None,
    roster: "Roster",
    timeout: float,
    routes: "MeshRoutes | None" = None,
) -> None:
    """GET /mesh/roster from every seed and merge entries (finding 3).

    d3: the seeds are dialed IN PARALLEL, one bounded dial each, and every
    roster is merged the moment it arrives. Sequential fetching let one dead
    or paused seed hold the first pass — and every other seed's discovery —
    for its whole dial timeout (live 2026-09-12: a recreated Spark showed no
    mesh activity for 13 s while its first seed was paused, so the d2
    pending refresh had nothing to show and every request 404'd).
    """
    bearer = f"Bearer {key}" if key else None
    if not seeds:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(seeds))) as pool:
        futures = [
            pool.submit(_fetch_one_seed_roster, seed, bearer, roster, timeout, routes)
            for seed in seeds
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception:  # nosec B110 — best-effort: a seed fetch never blocks
                pass


def _run_verify_pass(
    routes: "MeshRoutes",
    holder: "SnapshotHolder | None",
) -> None:
    """Run a verification pass and rebuild the snapshot.

    Lightweight wrapper around :func:`verify_members` that runs on the
    heartbeat thread.  Does nothing when the holder is not yet available.
    """
    if holder is None:
        return
    # Single flight (t2): one pass at a time, and only ever on this thread.
    # Held non-blocking — a second pass would re-probe the same members with
    # staler data, so it is skipped, not queued. The verify-now event is
    # already set by whoever asked for it, so the skipped work is not lost:
    # the next loop iteration picks it up.
    if not routes._verify_pass_lock.acquire(blocking=False):  # noqa: SLF001
        # Re-raise the dirty flag so the skipped work is not lost.
        routes._verify_event.set()  # noqa: SLF001
        return
    try:
        verify_members(routes, holder)
    except Exception:  # nosec B110 — verification is best-effort
        pass
    finally:
        routes._verify_pass_lock.release()  # noqa: SLF001


def _collect_members_to_verify(routes: "MeshRoutes") -> list[tuple[str, str, Announcement]]:
    """Gather ``(name, origin, announcement)`` for every roster member with a
    stored announcement, under the lock.
    """
    members_to_verify: list[tuple[str, str, Announcement]] = []
    with routes._lock:
        for mname in routes.roster.members():
            rec = routes.roster._roster.get(mname)  # noqa: SLF001
            if rec is None:
                continue
            origin = rec.origin
            ann = routes._announcements.get(origin)
            if ann is not None:
                members_to_verify.append((mname, origin, ann))
    return members_to_verify


def _probe_member_capabilities(
    member_data: tuple[str, str, Announcement], key: str | None, probe_timeout: float
) -> tuple[str, frozenset[str], frozenset[str], str | None]:
    """Probe one member's /capabilities and verify roles.

    Returns ``(origin, verified_roles, ready_roles, reason)``.  ``ready_roles``
    (t2, feeding t1's ``MemberInfo.ready_roles``) is what the probe read back
    as ``ready: true``, independent of verification — a lane can be ready and
    serving a fingerprint that disagrees with what its box announced, and a
    verified lane can be down.  A probe that never reached the peer reports
    both sets empty plus a reason.
    """
    from lobes.gateway._mesh_routing import verify_member_roles
    from lobes.gateway._readiness import _default_peer_opener

    _mname, origin, ann = member_data
    try:
        get_caps = _default_peer_opener
        status, body = get_caps(
            origin.rstrip("/") + _CAPABILITIES_PATH,
            probe_timeout,
            key,
        )
        if status != 200:
            return origin, frozenset(), frozenset(), f"HTTP {status}"

        payload = json.loads(body)
        # Finding 8 (review #252): GET /capabilities returns the role
        # mapping AT THE TOP LEVEL (see server.capabilities_payload —
        # ``payload = {role: dataclasses.asdict(registry[role]) ...}``,
        # never nested under a "roles" key). colleague and
        # capabilities_payload's own consumers already treat the
        # response this way; reading ``payload.get("roles", {})`` here
        # found nothing on every real gateway response, so no member was
        # ever verified.
        roles_data = payload if isinstance(payload, dict) else {}

        # Build the probed_roles dict per the verify_member_roles signature.
        probed_roles: dict[str, dict] = {}
        for role_name, role_entry in roles_data.items():
            if not isinstance(role_entry, dict):
                continue
            # d2: a PROXIED entry is the peer relaying someone else's lane —
            # never a lane of its own. Its ready bit (true since t4) and its
            # fingerprint must not make the peer a candidate: live
            # 2026-09-12 the Spark's proxied worker entry drew a raw-id
            # request that the Spark then refused 508 (single hop).
            # The announcement builder already skips proxied entries
            # (_hosted_role_slice); the probe now agrees.
            if role_entry.get("proxied"):
                continue
            role_fp = role_entry.get("fingerprint")
            probed_roles[role_name] = {
                "fingerprint": role_fp,
                "ready": role_entry.get("ready"),
            }

        # Compare announced vs probed fingerprints.
        verified = verify_member_roles(ann, probed_roles)
        ready = frozenset(
            role for role, entry in probed_roles.items() if entry.get("ready") is True
        )
        reason = None if verified else "no announced role verified against /capabilities"
        return origin, verified, ready, reason

    except Exception as exc:  # nosec B110 — best-effort: probe never blocks
        return origin, frozenset(), frozenset(), type(exc).__name__


def _run_verification_probes(
    members_to_verify: list[tuple[str, str, Announcement]],
    key: str | None,
    probe_timeout: float,
    verify_log: RejectionLog | None,
) -> tuple[dict[str, frozenset[str]], dict[str, str | None], dict[str, frozenset[str]]]:
    """Probe every member's /capabilities in parallel and collect results.

    Item C (t9): every probe that fails to verify anything carries a short
    reason string instead of vanishing into a bare `except: pass`. `None`
    means "verified cleanly" — never logged, never stored on the member.

    t2: returns a THIRD mapping, per-origin ready roles, and — importantly —
    gives every origin an entry in it even when the probe found nothing ready.
    That entry is what tells `build_snapshot` the member was probed at all: a
    clean probe of a box whose lanes are all still loading verifies nothing
    and records no reason, so without it the member would read exactly like
    one nobody has dialled yet.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    verified_by_origin: dict[str, frozenset[str]] = {}
    reason_by_origin: dict[str, str | None] = {}
    ready_by_origin: dict[str, frozenset[str]] = {}

    max_workers = min(8, len(members_to_verify))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_probe_member_capabilities, md, key, probe_timeout): md
            for md in members_to_verify
        }
        for fut in as_completed(futures):
            mname, origin, _ann = futures[fut]
            try:
                origin, verified, ready, reason = fut.result(timeout=probe_timeout)
            except Exception as exc:  # nosec B110 — best-effort: drop failed probes
                verified, ready, reason = frozenset(), frozenset(), type(exc).__name__
            if verified:
                verified_by_origin[origin] = verified
            reason_by_origin[origin] = reason
            ready_by_origin[origin] = ready
            if reason is not None and verify_log is not None:
                line = verify_log.record(origin, "GET", _CAPABILITIES_PATH, reason)
                if line is not None:
                    sys.stderr.write(f"[gateway] mesh verify {mname}: {line}\n")

    return verified_by_origin, reason_by_origin, ready_by_origin


def _log_pending_members(
    routes: "MeshRoutes",
    holder: "SnapshotHolder",
    members_to_verify: list[tuple[str, str, Announcement]],
) -> None:
    """Say once, per member, that this box is about to leave a boot window.

    Item C (t9) gave failed probes a collapsed stderr line; t2 gives the
    not-yet-probed state the same treatment, through the SAME throttled
    RejectionLog so a mesh that is slow to verify cannot flood the log. The
    throttle key is deliberately distinct from the failure key (`#pending`),
    because a pending line and a failure line for one origin are different
    facts and must not collapse into each other.
    """
    verify_log = routes._verify_log
    if verify_log is None:
        return
    try:
        view = holder.current()
    except Exception:  # nosec B110 — a duck-typed holder never breaks a pass
        return
    if view is None or getattr(view, "snapshot", None) is None:
        return
    probed = {m.origin for m in view.snapshot.members if m.probed}
    for mname, origin, _ann in members_to_verify:
        if origin in probed:
            continue
        # The RejectionLog is used for its THROTTLE only; its own line reads
        # "auth: rejected ..." (it was built for 401s) and misled the first
        # live run, so the wording here is ours.
        line = verify_log.record(f"{origin}#pending", "GET", _CAPABILITIES_PATH, "not_yet_probed")
        if line is not None:
            sys.stderr.write(
                f"[gateway] mesh pending {mname}: {origin} not yet probed — requests for its "
                "roles answer 503 role_unverified until the first /capabilities probe lands\n"
            )


def verify_members(
    routes: "MeshRoutes",
    holder: "SnapshotHolder",
    join_key: str | None = None,
    timeout: float | None = None,
) -> None:
    """Verify announced members by probing ``/capabilities``.

    For each roster member with a stored announcement:

    1. GET ``<origin>/capabilities`` with ``Authorization: Bearer <join_key>``
    2. For each announced role, compare fingerprints
    3. Store verified roles
    4. Rebuild snapshot
    """
    from lobes.gateway._mesh_routing import MeshRoutingView
    from lobes.gateway._readiness import _PEER_PROBE_TIMEOUT

    key = join_key or (routes.config.key if hasattr(routes.config, "key") else None)
    probe_timeout = timeout or _PEER_PROBE_TIMEOUT

    members_to_verify = _collect_members_to_verify(routes)

    if not members_to_verify:
        # Rebuild snapshot even without verification (e.g. stale data).
        snap = build_snapshot(
            routes.roster,
            announcements=routes._announcements,
            discovered_roles=routes._discovered_roles,  # noqa: SLF001
        )
        holder.replace(MeshRoutingView(snapshot=snap, peer_states={}))
        return

    _log_pending_members(routes, holder, members_to_verify)

    verified_by_origin, reason_by_origin, ready_by_origin = _run_verification_probes(
        members_to_verify, key, probe_timeout, routes._verify_log
    )

    # Build the verified_roles mapping for build_snapshot.
    snap = build_snapshot(
        routes.roster,
        announcements=routes._announcements,
        verified_roles=verified_by_origin,
        unverified_reasons={o: r for o, r in reason_by_origin.items() if r is not None},
        ready_roles=ready_by_origin,
        discovered_roles=routes._discovered_roles,  # noqa: SLF001
    )
    holder.replace(MeshRoutingView(snapshot=snap, peer_states={}))


def start_mesh(
    routes: "MeshRoutes",
    announcement: Announcement,
) -> threading.Thread:
    """Start the heartbeat daemon thread.  Returns the thread handle."""
    from lobes.gateway._mesh_wire import encode as _encode

    announcement_bytes = _encode(announcement)
    # Store the initial announcement on the routes so POST /mesh/reannounce has
    # something to resend before any builder-driven rebuild (live 2026-09-12:
    # it answered "no announcement yet" forever).
    routes._announcement_bytes = announcement_bytes  # noqa: SLF001

    # Finding 17: keep the __init__ stop event (no reassignment).
    # Use routes._stop which was already created in __init__.

    thread = threading.Thread(
        target=_heartbeat_loop,
        args=(
            routes,
            announcement_bytes,
            routes.config.heartbeat_s,
            routes._stop,  # Use the __init__ stop event.
            routes._reannounce_event,
            routes._holder,
            routes._verify_now_event,
        ),
        name="lobes-mesh-heartbeat",
        daemon=True,
    )
    thread.start()
    # Finding 17: assign _thread so serve()'s shutdown can join.
    routes._thread = thread  # noqa: SLF001
    return thread


def reannounce_now(routes: "MeshRoutes", announcement: Announcement) -> None:
    """Update the announcement and trigger an immediate re-broadcast.

    Non-blocking: sets the re-announce event so the loop wakes within one
    second.  The actual broadcast still happens on the heartbeat thread.
    """
    from lobes.gateway._mesh_wire import encode as _encode

    announcement_bytes = _encode(announcement)
    with routes._lock:
        routes._announcement = announcement  # noqa: SLF001
        routes._announcement_bytes = announcement_bytes  # noqa: SLF001
    routes._reannounce_event.set()  # noqa: SLF001
