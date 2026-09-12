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
        self._join_log = _join_log
        self._verify_log = _verify_log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._reannounce_event = threading.Event()
        self._pending: list[_PendingJoin] = []
        self._announcements: dict[str, Announcement] = {}
        self._verify_event = threading.Event()
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
        return 200, [("Content-Type", "application/json")], json.dumps(body).encode()

    def join(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/join – keyless pending join (flood-collapsed logging).

        Capped at 8 pending entries (one per origin) with a 300 s TTL.
        When a source floods past 100 requests in the collapse window,
        one collapsed line is emitted via :class:`~lobes.gateway._authlog.RejectionLog`.
        """
        rfile = getattr(handler, "rfile", None)
        hdrs = getattr(handler, "headers", {})
        cl = hdrs.get("Content-Length")
        body: bytes = b""
        if cl is not None:
            try:
                length = int(cl)
                if rfile is not None and length > 0:
                    body = rfile.read(length)
            except (ValueError, AttributeError):
                body = b""

        data: dict = {}
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            pass

        name = data.get("name")
        origin = data.get("origin", "")
        capacity = data.get("capacity")

        if not isinstance(name, str) or not name:
            return (
                400,
                [("Content-Type", "application/json")],
                json.dumps({"error": "name is required"}).encode(),
            )

        now = time.monotonic()
        ttl = 300.0  # 5-minute TTL for pending joins

        # Finding 11: derive source from socket peer, not client-supplied field.
        address = getattr(handler, "client_address", None)
        if isinstance(address, tuple) and address:
            source = str(address[0])
        else:
            source = "<unknown>"

        with self._lock:
            pending = self._pending

            # Remove expired entries first.
            pending = [p for p in pending if p.joined_at + ttl > now]
            self._pending = pending

            # Capacity cap: 8 pending entries total.
            if len(pending) >= 8:
                return (
                    400,
                    [("Content-Type", "application/json")],
                    json.dumps({"error": "pending join queue is full (8)"}).encode(),
                )

            # Finding 11: one entry per ORIGIN (not per name).
            for p in pending:
                if p.origin == origin:
                    return (
                        400,
                        [("Content-Type", "application/json")],
                        json.dumps({"error": "same origin already pending"}).encode(),
                    )
                # Drop entries with different name but same origin won't happen
                # now that we key on origin above.

            pending.append(_PendingJoin(name=name, origin=origin, capacity=capacity, joined_at=now))

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
            [("Content-Type", "application/json")],
            json.dumps({"status": "pending join registered"}).encode(),
        )

    def announce(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/announce – authenticated membership heartbeat."""
        if not self._check_key(getattr(handler, "headers", {})):
            # Finding 12: add Connection: close to prevent keep-alive framing poison.
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
                    ("Connection", "close"),
                ],
                json.dumps(
                    {
                        "error": {
                            "message": "Invalid API key.",
                            "type": "invalid_api_key",
                            "code": "invalid_api_key",
                        }
                    }
                ).encode(),
            )

        rfile = getattr(handler, "rfile", None)
        hdrs = getattr(handler, "headers", {})
        cl = hdrs.get("Content-Length")
        body: bytes = b""
        if cl is not None:
            try:
                length = int(cl)
                if rfile is not None and length > 0:
                    body = rfile.read(length)
            except (ValueError, AttributeError):
                body = b""

        name: str
        origin: str

        # Finding 15: catch MeshSchemaIncompatible distinctly.
        try:
            from lobes.gateway._mesh_wire import decode

            announced = decode(body)
            public = announced.public()
            name = public.name
            origin = public.origin
        except MeshSchemaIncompatible:
            # Wrong schema major – reject before touching roster.
            return (
                400,
                [
                    ("Content-Type", "application/json"),
                    ("Connection", "close"),  # finding 12
                ],
                json.dumps(
                    {
                        "error": {
                            "message": "Schema version mismatch",
                            "type": "schema_incompatible",
                            "expected_major": SCHEMA_MAJOR,
                        }
                    }
                ).encode(),
            )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError):
            # Finding 15 (review #252): KeyError/AttributeError added as a
            # boundary safeguard — decode() now validates structure and
            # raises ValueError consistently, but a malformed body must
            # never escape this route as an unhandled 500 either way.
            return (
                400,
                [
                    ("Content-Type", "application/json"),
                    ("Connection", "close"),  # finding 12
                ],
                json.dumps({"error": "invalid JSON body"}).encode(),
            )

        if not name:
            return (
                400,
                [("Content-Type", "application/json")],
                json.dumps({"error": "name is required"}).encode(),
            )

        # Finding 5: ledger only RESTRICTS — an absent entry is "no
        # restriction", key-holders are admitted permanently.  Only when an
        # entry exists AND is not in force (expired/revoked) do we refuse.
        # Finding 4 (review #252): the ledger check and the roster admission
        # used to be two separate, unsynchronized steps here — a concurrent
        # /mesh/revoke could commit in the gap and this request would still
        # admit the just-revoked name. announce_gated() does both under one
        # lock shared with approve/revoke, so a revoke can never interleave.
        roster_now = self.roster.now()
        from lobes.gateway._mesh_roster import MeshApprovalExpired, MeshFlapping

        try:
            self.roster.announce_gated(name, origin, None, now=roster_now)
        except MeshApprovalExpired:
            return (
                403,
                [
                    ("Content-Type", "application/json"),
                    ("Connection", "close"),
                ],
                json.dumps(
                    {
                        "error": {
                            "message": "Approval has expired",
                            "type": "approval_expired",
                            "name": name,
                        }
                    }
                ).encode(),
            )
        except MeshFlapping as exc:
            return (
                429,
                [
                    ("Content-Type", "application/json"),
                    ("Connection", "close"),
                ],
                json.dumps(
                    {
                        "error": {
                            "message": str(exc),
                            "type": "mesh_flapping",
                            "name": name,
                        }
                    }
                ).encode(),
            )
        except Exception as exc:
            if "conflict" in str(exc).lower() or "already held" in str(exc).lower():
                # Drop any stored announcement from an origin that no longer
                # matches the roster record for *name*.
                rec = self.roster._roster.get(name)  # noqa: SLF001
                if rec is not None:
                    current_origin = rec.origin
                    stale_origins = [
                        o for o, a in self._announcements.items() if a.origin != current_origin
                    ]
                    for o in stale_origins:
                        self._announcements.pop(o, None)
                return (
                    409,
                    [
                        ("Content-Type", "application/json"),
                        ("Connection", "close"),
                    ],
                    json.dumps(
                        {
                            "error": {
                                "message": "Name conflict",
                                "type": "name_conflict",
                                "name": name,
                            }
                        }
                    ).encode(),
                )
            raise

        # Store the decoded announcement keyed by origin for snapshot rebuilds.
        self._announcements[origin] = public

        # Mark verification as dirty so the heartbeat loop rebuilds the
        # snapshot with updated announcement data.
        self._verify_event.set()

        return (
            200,
            [("Content-Type", "application/json")],
            json.dumps(
                {"status": "announced", "name": name, "schema_version": SCHEMA_MAJOR}
            ).encode(),
        )

    def roster_list(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """GET /mesh/roster – authenticated member listing."""
        if not self._check_key(getattr(handler, "headers", {})):
            # Finding 12: add Connection: close.
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
                    ("Connection", "close"),
                ],
                json.dumps(
                    {
                        "error": {
                            "message": "Invalid API key.",
                            "type": "invalid_api_key",
                            "code": "invalid_api_key",
                        }
                    }
                ).encode(),
            )

        # Finding 13: take a snapshot under the route lock.
        # t8 follow-up (c27/h1, c46/h37): the CLI's `lobes mesh status` reads
        # last_seen_age / expiry / verified / flapping / roles per member —
        # all additive, all sourced from state this box already holds (the
        # Roster, its Ledger, and the mesh routing snapshot), never new
        # tracking. See lobes.gateway._mesh_routing.exposed_role_names for
        # the roles column (suffixed names included, private roles excluded
        # because they were never in a stored announcement to begin with).
        from lobes.gateway._mesh_roster import _FLAPPING_THRESHOLD
        from lobes.gateway._mesh_routing import exposed_role_names

        with self._lock:
            now = self.roster.now()
            snapshot = None
            if self._holder is not None:
                view = self._holder.current()
                if view is not None:
                    snapshot = view.snapshot
            member_names = list(self.roster.members())
            member_records = {}
            for mname in member_names:
                rec = self.roster._roster.get(mname)  # noqa: SLF001
                if rec is None:
                    continue
                entry = self.roster.ledger.entries.get(mname)
                expiry = entry.expiry if entry is not None else None
                member_info = None
                if snapshot is not None:
                    member_info = next(
                        (m for m in snapshot.members if m.origin == rec.origin), None
                    )
                verified = bool(member_info.verified_roles) if member_info is not None else False
                roles = (
                    list(exposed_role_names(snapshot, rec.origin)) if snapshot is not None else []
                )
                # Item A (t9): the flapping signal is now PER-MEMBER —
                # Roster.announce() (the only path `/mesh/announce` drives)
                # tracks each name's own flap count, so a churning member
                # reports `flapping` for ITSELF without holding out every
                # other name in the roster the way the old roster-wide
                # counter did.
                flap_count = self.roster._flap_counts.get(mname, 0)  # noqa: SLF001
                flapping = flap_count >= _FLAPPING_THRESHOLD
                unverified_reason = (
                    member_info.unverified_reason if member_info is not None else None
                )
                member_records[mname] = {
                    "name": rec.name,
                    "origin": rec.origin,
                    "capacity": rec.capacity,
                    "last_seen_age": max(0.0, now - rec.last_seen),
                    "expiry": expiry,
                    "verified": verified,
                    "flapping": flapping,
                    "roles": roles,
                    # Item C (t9): a short, operator-facing reason the last
                    # verification probe found nothing verified, or None on
                    # a clean/never-probed member.
                    "unverified_reason": unverified_reason,
                }

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
            [("Content-Type", "application/json")],
            json.dumps(
                {"members": list(member_records.values()), "ledger": ledger_entries}
            ).encode(),
        )

    def approve(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/approve – authenticated approval."""
        if not self._check_key(getattr(handler, "headers", {})):
            # Finding 12: add Connection: close.
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
                    ("Connection", "close"),
                ],
                json.dumps(
                    {
                        "error": {
                            "message": "Invalid API key.",
                            "type": "invalid_api_key",
                            "code": "invalid_api_key",
                        }
                    }
                ).encode(),
            )

        rfile = getattr(handler, "rfile", None)
        hdrs = getattr(handler, "headers", {})
        cl = hdrs.get("Content-Length")
        body: bytes = b""
        if cl is not None:
            try:
                length = int(cl)
                if rfile is not None and length > 0:
                    body = rfile.read(length)
            except (ValueError, AttributeError):
                body = b""

        data: dict = {}
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            data = {}

        name: object = data.get("name")
        expiry: object = data.get("expiry", 3600.0)
        approved_by: str = data.get("approved_by", "requester")

        if not isinstance(name, str) or not name:
            return (
                400,
                [("Content-Type", "application/json")],
                json.dumps({"error": "name is required"}).encode(),
            )

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
            [("Content-Type", "application/json")],
            json.dumps({"status": "approved", "name": name}).encode(),
        )

    def revoke(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/revoke – authenticated revocation."""
        if not self._check_key(getattr(handler, "headers", {})):
            # Finding 12: add Connection: close.
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
                    ("Connection", "close"),
                ],
                json.dumps(
                    {
                        "error": {
                            "message": "Invalid API key.",
                            "type": "invalid_api_key",
                            "code": "invalid_api_key",
                        }
                    }
                ).encode(),
            )

        rfile = getattr(handler, "rfile", None)
        hdrs = getattr(handler, "headers", {})
        cl = hdrs.get("Content-Length")
        body: bytes = b""
        if cl is not None:
            try:
                length = int(cl)
                if rfile is not None and length > 0:
                    body = rfile.read(length)
            except (ValueError, AttributeError):
                body = b""

        data: dict = {}
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            data = {}

        name: object = data.get("name")
        approved_by: str = data.get("approved_by", "requester")

        if not isinstance(name, str) or not name:
            return (
                400,
                [("Content-Type", "application/json")],
                json.dumps({"error": "name is required"}).encode(),
            )

        roster_now = self.roster.now()
        # Finding 3 (review #252): revoke + persist as one transaction under
        # the ledger's own lock — see the matching approve_and_save note above.
        self.roster.revoke_and_save(str(name), now=roster_now, approved_by=approved_by)
        # Remove the revoked member's announcement and rebuild the snapshot
        # so revoked members get zero forwards immediately.
        with self._lock:
            self._announcements.pop(name, None)
            snap = build_snapshot(self.roster, announcements=self._announcements)
            if self._holder is not None:
                from lobes.gateway._mesh_routing import MeshRoutingView

                self._holder.replace(MeshRoutingView(snapshot=snap, peer_states={}))
        return (
            200,
            [("Content-Type", "application/json")],
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
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
                    ("Connection", "close"),
                ],
                json.dumps(
                    {
                        "error": {
                            "message": "Invalid API key.",
                            "type": "invalid_api_key",
                            "code": "invalid_api_key",
                        }
                    }
                ).encode(),
            )

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
                [("Content-Type", "application/json")],
                json.dumps({"status": "no-op", "reason": "no announcement yet"}).encode(),
            )

        reannounce_now(self, announcement)
        return (
            200,
            [("Content-Type", "application/json")],
            json.dumps({"status": "reannounced", "name": announcement.name}).encode(),
        )


# --- route registry --------------------------------------------------------

_MESH_ROUTES: dict[tuple[str, str], str] = {
    ("GET", "/mesh/detect"): "detect",
    ("POST", "/mesh/join"): "join",
    ("POST", "/mesh/announce"): "announce",
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


def _build_announcement(
    config: MeshConfig,
    roster: "Roster",
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
    roster:
        Current Roster instance.
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

    roles: dict[str, RoleInfo] = {}

    # Collect ready, hosted roles from the local gateway.
    # Finding 1: build real roles from the gateway's own data.
    from lobes.roles import BACKEND_ROLE

    if readiness_cache is not None:
        # Use readiness cache to determine which roles are ready.
        # current() returns a flat dict[str, bool|None], not a nested dict.
        try:
            ready_roles: dict[str, bool | None] = (
                readiness_cache.current()
            )  # type: ignore[assignment]
        except (AttributeError, TypeError):
            ready_roles = {}
    else:
        ready_roles = {}

    # Build per-role RoleInfo from lane configs or from replica caches.
    for backend_name, lane_config in (declared_lane_configs or {}).items():
        # Convert backend name → role name (mesh speaks roles: cortex, senses, …)
        role_name = BACKEND_ROLE.get(backend_name, backend_name)

        # Check readiness: only include ready+hosted roles. Finding 6 (review
        # #252): ReadinessCache.current() is keyed by BACKEND name ("primary",
        # "multimodal", …) — see ReadinessCache.from_backends — never by role
        # name. Looking this up as `role_name` ("cortex", "senses") missed
        # every entry once `ready_roles` was non-empty, so every ready lane
        # was silently dropped from the announcement.
        if ready_roles and ready_roles.get(backend_name) is not True:
            continue

        # Get live fingerprint from replica cache if available. `replica_caches`
        # (build_replica_caches) is keyed by BACKEND name, never role name —
        # look it up the same way the pool itself does, not by `role_name`
        # (a mismatch here would silently miss every live fingerprint).
        fp = None
        if replica_caches and backend_name in replica_caches:
            cache = replica_caches[backend_name]
            try:
                # cache.current() returns tuple[ReplicaState]; find local=True.
                states = cache.current()
                if isinstance(states, (list, tuple)):
                    for st in states:
                        if getattr(st, "local", False):
                            fp = getattr(st, "fingerprint", None)
                            break
            except (AttributeError, TypeError):
                fp = None

        # Get capacity from local capacities.
        capacity = None
        if local_capacities and backend_name in local_capacities:
            capacity = local_capacities[backend_name]

        # Build RoleInfo from lane config, enriched with live data.
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

        # No live replica-cache entry (the common case: no *_PEER_ORIGINS pool
        # declared anywhere means build_replica_caches returns {} outright) —
        # fall back to the DECLARED lane fields so a plain, unpooled box still
        # announces a real, comparable fingerprint instead of an all-empty one
        # that can never be verified by a peer's probe.
        if fp is None:
            fp = {
                "served_id": model,
                "quantization": quant,
                "max_model_len": context,
                "runtime": runtime,
            }

        roles[role_name] = RoleInfo(
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
        roster,
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
) -> None:
    """POST *body* to *url* via http.client with a hard timeout.

    Silently drops on any failure — a down peer is handled by tick-based
    staleness in the roster.

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
    path = parsed.path or "/mesh/announce"
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
        "Content-Type": "application/json",
    }
    # Finding 2: attach Bearer key.
    if key is not None:
        # The join key reaches here as the str MeshConfig parsed; a bytes key is
        # accepted too. Decoding a str raised AttributeError inside the
        # best-effort catch on every announce — the live 2026-09-12 silent mesh.
        key_text = key.decode("utf-8") if isinstance(key, bytes) else key
        headers["Authorization"] = f"Bearer {key_text}"

    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers=headers,
        )
        conn.getresponse()
    except Exception:  # nosec B110 — best-effort: silently drop failed peer connections
        pass
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # nosec B110 — silently ignore close errors
                pass


def _heartbeat_loop(
    routes: "MeshRoutes",
    announcement_bytes: bytes,
    interval: float,
    stop_event: threading.Event,
    reannounce_event: threading.Event,
    holder: "SnapshotHolder | None" = None,
) -> None:
    """Background thread that announces the local member to seeds + roster.

    Each peer gets its own socket with its own timeout so one hung peer never
    delays another.  The loop exits when *stop_event* is set.
    """
    seeds = routes.config.seeds
    # Finding 4: pace on the full interval, not min(interval, 1.0).
    deadline: float = 0.0

    while not stop_event.is_set():
        # Compute the next deadline.
        woken_by_reannounce = False
        if deadline == 0.0:
            deadline = time.monotonic() + interval
        else:
            # Wait in ≤1s increments against the deadline.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Deadline elapsed – no wait needed.
                pass
            else:
                wait_dur = min(remaining, 1.0)
                # Finding 17 (review #252): capture wait()'s own return value
                # BEFORE clearing the event. The old code cleared first, so
                # `reannounce_event.is_set()` below was always False and an
                # immediate re-announce (POST /mesh/reannounce, `lobes
                # switch`/`up`) never ran a pass until the ordinary deadline —
                # exactly the bug the event exists to avoid.
                woken_by_reannounce = reannounce_event.wait(timeout=wait_dur)

        reannounce_event.clear()

        if stop_event.is_set():
            break

        # Check if we should run a pass: deadline elapsed or reannounce fired.
        now = time.monotonic()
        run_pass = (now >= deadline) or woken_by_reannounce

        if run_pass:
            # Advance to next deadline.
            deadline = now + interval
        else:
            continue

        # Finding 10: tick the roster once per pass under the lock.
        tick_result: TickResult
        try:
            with routes._lock:
                try:
                    tick_result = routes.roster.tick()
                except Exception:  # nosec B110 — best-effort: tick never blocks
                    tick_result = TickResult()

                # Drop refresh: prune announcements for the DROPPED members
                # only (review #252 finding 2). Iterating the SURVIVORS and
                # popping their announcements — the old code — did the exact
                # opposite of what a single expiry should do: it wiped every
                # still-healthy member's announcement, leaving nothing for
                # the next verification pass to verify. `tick()` now names
                # the dropped origins directly.
                if tick_result.dropped_origins and holder is not None:
                    try:
                        from lobes.gateway._mesh_routing import MeshRoutingView

                        for origin in tick_result.dropped_origins:
                            routes._announcements.pop(origin, None)
                        snap = build_snapshot(routes.roster, announcements=routes._announcements)
                        holder.replace(MeshRoutingView(snapshot=snap, peer_states={}))
                    except Exception:  # nosec B110 — best-effort: drop refresh never blocks
                        pass

                # Only DECIDE about verification under the lock; the pass
                # itself dials every member over the network and re-takes this
                # lock inside verify_members — running it here starved every
                # roster read and inbound announce for the probe timeouts
                # (live Spark, dev518: /mesh/roster never answered).
                verify_dirty = routes._verify_event.is_set()
                if verify_dirty:
                    routes._verify_event.clear()

                # Collect the announcement to send.
                if routes._announcement_bytes is not None:
                    to_send = routes._announcement_bytes
                else:
                    to_send = announcement_bytes
        except Exception:  # nosec B110 — best-effort: lock block never blocks
            tick_result = TickResult()
            to_send = announcement_bytes
            verify_dirty = False

        # Verification pass OUTSIDE the lock: network I/O never holds it.
        if verify_dirty:
            try:
                _run_verify_pass(routes, holder)
            except Exception:  # nosec B110 — verification is best-effort
                pass

        if to_send is None:
            continue

        # Finding 9: gather members list under the lock, dial in parallel.
        member_origins: list[tuple[str, str]] = []
        with routes._lock:
            for member_name in list(routes.roster.members()):
                rec = routes.roster._roster.get(member_name)  # noqa: SLF001
                if rec is not None:
                    member_origins.append((member_name, rec.origin))

        # Find 10: use named per-dial budget, not missed_max * 10.
        dial_timeout = _DIAL_TIMEOUT_S

        # Finding 3: first fetch from seed roster.
        if seeds:
            try:
                # Finding 5 (review #252): named arguments — the old
                # positional call swapped `timeout` and `routes`, handing a
                # MeshRoutes instance to http.client as a socket timeout.
                # The resulting TypeError was swallowed by this very
                # except-Exception, so seed rosters were never fetched.
                _fetch_seed_roster(
                    seeds, routes.config.key, routes.roster, timeout=dial_timeout, routes=routes
                )
            except Exception:  # nosec B110 — best-effort: seed fetch never blocks
                pass

        if to_send is not None:
            # Finding 9: parallelize announces with ThreadPoolExecutor.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, len(seeds) + len(member_origins) + 1)
            ) as pool:
                futures = []

                # Announce to seeds.
                for seed in seeds:
                    if stop_event.is_set():
                        break
                    url = seed + "/mesh/announce"
                    futures.append(
                        pool.submit(
                            _post_announcement, url, to_send, dial_timeout, routes.config.key
                        )
                    )

                # Announce to every roster member.
                for _member_name, origin in member_origins:
                    if stop_event.is_set():
                        break
                    url = origin + "/mesh/announce"
                    futures.append(
                        pool.submit(
                            _post_announcement, url, to_send, dial_timeout, routes.config.key
                        )
                    )

                # Wait for all dials to complete (each has its own timeout).
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result(timeout=dial_timeout)
                    except Exception:  # nosec B110 — best-effort: drop failed peer connections
                        pass


def _fetch_seed_roster(
    seeds: tuple[str, ...],
    key: str | None,
    roster: "Roster",
    timeout: float,
    routes: "MeshRoutes | None" = None,
) -> None:
    """GET /mesh/roster from every seed and merge entries (finding 3)."""
    bearer = f"Bearer {key}" if key else None

    for seed in seeds:
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

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if bearer:
            headers["Authorization"] = bearer

        try:
            conn.request("GET", path, headers=headers)
            resp = conn.getresponse()
            if resp.status == 200:
                raw = resp.read()
                try:
                    data = json.loads(raw)
                    members = data.get("members", [])
                    if isinstance(members, list):
                        for member in members:
                            if isinstance(member, dict):
                                mname = member.get("name", "")
                                morigin = member.get("origin", "")
                                if mname and morigin:
                                    if routes is not None:
                                        with roster._lock:
                                            roster.announce(
                                                mname, morigin, None, now=time.monotonic()
                                            )
                                    else:
                                        roster.announce(mname, morigin, None, now=time.monotonic())

                    # Finding 9 (review #252): merge the peer's LEDGER too, not
                    # just its membership records — the ledger is what a
                    # revocation lives in. Without this, a revoke made on one
                    # node never reached a peer that had already accepted and
                    # verified the member, so that peer kept forwarding to it
                    # forever. `Ledger.merge()` applies the standard
                    # updated_at-wins gossip rule.
                    ledger_data = data.get("ledger", {})
                    if isinstance(ledger_data, dict) and ledger_data:
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
                        newly_unapproved: list[str] = []
                        merge_now = time.monotonic()
                        roster.merge(peer_ledger)
                        roster.save()
                        for pname in peer_ledger.entries:
                            if not roster.is_approved(pname, now=merge_now):
                                newly_unapproved.append(pname)
                        # Rebuild routing so a merged revocation removes the
                        # member from forwarding immediately, exactly as the
                        # local /mesh/revoke handler already does.
                        if routes is not None and newly_unapproved:
                            with routes._lock:  # noqa: SLF001
                                for pname in newly_unapproved:
                                    rec = roster._roster.get(pname)  # noqa: SLF001
                                    if rec is None:
                                        continue
                                    routes._announcements.pop(rec.origin, None)  # noqa: SLF001
                                if routes._holder is not None:  # noqa: SLF001
                                    from lobes.gateway._mesh_routing import MeshRoutingView

                                    snap = build_snapshot(
                                        roster, announcements=routes._announcements  # noqa: SLF001
                                    )
                                    routes._holder.replace(  # noqa: SLF001
                                        MeshRoutingView(snapshot=snap, peer_states={})
                                    )
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        except Exception:  # nosec B110 — best-effort: seed roster fetch never blocks
            pass
        finally:
            try:
                conn.close()
            except Exception:  # nosec B110 — silently ignore close errors
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
    try:
        verify_members(routes, holder)
    except Exception:  # nosec B110 — verification is best-effort
        pass


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
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from lobes.gateway._mesh_routing import MeshRoutingView, verify_member_roles
    from lobes.gateway._readiness import (
        _PEER_PROBE_TIMEOUT,
        _default_peer_opener,
    )

    key = join_key or (routes.config.key if hasattr(routes.config, "key") else None)
    probe_timeout = timeout or _PEER_PROBE_TIMEOUT

    # Collect members with stored announcements.
    members_to_verify: list[tuple[str, str, Announcement]] = []
    with routes._lock:
        for mname in list(routes.roster.members()):
            rec = routes.roster._roster.get(mname)  # noqa: SLF001
            if rec is None:
                continue
            origin = rec.origin
            ann = routes._announcements.get(origin)
            if ann is not None:
                members_to_verify.append((mname, origin, ann))

    if not members_to_verify:
        # Rebuild snapshot even without verification (e.g. stale data).
        snap = build_snapshot(routes.roster, announcements=routes._announcements)
        holder.replace(MeshRoutingView(snapshot=snap, peer_states={}))
        return

    # Probe all members in parallel.
    verified_by_origin: dict[str, frozenset[str]] = {}
    # Item C (t9): every probe that fails to verify anything carries a short
    # reason string instead of vanishing into a bare `except: pass`. `None`
    # means "verified cleanly" — never logged, never stored on the member.
    reason_by_origin: dict[str, str | None] = {}

    def _probe_peer(
        member_data: tuple[str, str, Announcement],
    ) -> tuple[str, frozenset[str], str | None]:
        """Probe one member's /capabilities and verify roles."""
        _mname, origin, ann = member_data
        try:
            get_caps = _default_peer_opener
            status, body = get_caps(
                origin.rstrip("/") + "/capabilities",
                probe_timeout,
                key,
            )
            if status != 200:
                return origin, frozenset(), f"HTTP {status}"

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
                role_fp = role_entry.get("fingerprint")
                probed_roles[role_name] = {
                    "fingerprint": role_fp,
                    "ready": role_entry.get("ready"),
                }

            # Compare announced vs probed fingerprints.
            verified = verify_member_roles(ann, probed_roles)
            reason = None if verified else "no announced role verified against /capabilities"
            return origin, verified, reason

        except Exception as exc:  # nosec B110 — best-effort: probe never blocks
            return origin, frozenset(), type(exc).__name__

    max_workers = min(8, len(members_to_verify))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_probe_peer, md): md for md in members_to_verify}
        for fut in as_completed(futures):
            mname, origin, _ann = futures[fut]
            try:
                origin, verified, reason = fut.result(timeout=probe_timeout)
            except Exception as exc:  # nosec B110 — best-effort: drop failed probes
                verified, reason = frozenset(), type(exc).__name__
            if verified:
                verified_by_origin[origin] = verified
            reason_by_origin[origin] = reason
            if reason is not None and routes._verify_log is not None:
                line = routes._verify_log.record(origin, "GET", "/capabilities", reason)
                if line is not None:
                    sys.stderr.write(f"[gateway] mesh verify {mname}: {line}\n")

    # Build the verified_roles mapping for build_snapshot.
    snap = build_snapshot(
        routes.roster,
        announcements=routes._announcements,
        verified_roles=verified_by_origin,
        unverified_reasons={o: r for o, r in reason_by_origin.items() if r is not None},
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
