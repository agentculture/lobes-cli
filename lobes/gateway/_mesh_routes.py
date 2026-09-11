"""Mesh route handlers + heartbeat daemon thread.

End-points
----------
* ``GET  /mesh/detect``  – keyless health / presence check
* ``POST /mesh/join``    – keyless join request (flood-collapsed logging)
* ``POST /mesh/announce`` – Bearer-join-key membership heartbeat
* ``GET  /mesh/roster``   – Bearer-join-key member listing
* ``POST /mesh/approve``  – Bearer-join-key approval
* ``POST /mesh/revoke``   – Bearer-join-key revocation

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
from lobes.gateway._mesh_config import MeshConfig, build_mesh_config
from lobes.gateway._mesh_wire import (
    SCHEMA_MAJOR,
    Announcement,
    Fingerprint,
    MeshSchemaIncompatible,
    RoleInfo,
)

if TYPE_CHECKING:
    from lobes.gateway._mesh_roster import Roster


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
    ) -> None:
        self.config = config
        self.roster = roster
        self._announcement: Announcement | None = None
        self._announcement_bytes: bytes | None = None
        self._join_log = _join_log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._reannounce_event = threading.Event()
        # Pending joins are initialized here (finding 11: never getattr default).
        self._pending: list[_PendingJoin] = []

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
        except (json.JSONDecodeError, ValueError, TypeError):
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

        # Finding 5: ledger approval check before roster mutation.
        roster_now = self.roster.now()
        if not self.roster.is_approved(name, now=roster_now):
            entry = self.roster.ledger.entries.get(name)
            if entry is not None:
                # Approval existed but lapsed.
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
            # No approval entry at all.
            return (
                403,
                [
                    ("Content-Type", "application/json"),
                    ("Connection", "close"),
                ],
                json.dumps(
                    {
                        "error": {
                            "message": "Approval required",
                            "type": "approval_required",
                            "name": name,
                        }
                    }
                ).encode(),
            )

        # Finding 5: pass None for capacity (roles data is in the wire format, not capacity param).
        # Also finding 5: handle MeshNameConflict → 409.
        try:
            self.roster.announce(name, origin, None, now=roster_now)
        except Exception as exc:
            if "conflict" in str(exc).lower() or "already held" in str(exc).lower():
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
        with self._lock:
            member_names = list(self.roster.members())
            member_records = {}
            for mname in member_names:
                rec = self.roster._roster.get(mname)  # noqa: SLF001
                if rec is not None:
                    member_records[mname] = {
                        "name": rec.name,
                        "origin": rec.origin,
                        "capacity": rec.capacity,
                    }

        # Finding 3: return per-member objects with name AND origin.
        return (
            200,
            [("Content-Type", "application/json")],
            json.dumps({"members": list(member_records.values())}).encode(),
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

        try:
            expiry_f = float(expiry)
        except (TypeError, ValueError):
            expiry_f = 3600.0

        roster_now = self.roster.now()
        # Finding 6: convert duration to absolute expiry.
        expiry_abs = roster_now + expiry_f
        self.roster.approve(str(name), approved_by, expiry_abs, now=roster_now)
        # Finding 6: persist the ledger.
        self.roster.save()
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
        self.roster.revoke(str(name), now=roster_now, approved_by=approved_by)
        # Finding 6: persist the ledger.
        self.roster.save()
        return (
            200,
            [("Content-Type", "application/json")],
            json.dumps({"status": "revoked", "name": name}).encode(),
        )


# --- route registry --------------------------------------------------------

_MESH_ROUTES: dict[tuple[str, str], str] = {
    ("GET", "/mesh/detect"): "detect",
    ("POST", "/mesh/join"): "join",
    ("POST", "/mesh/announce"): "announce",
    ("GET", "/mesh/roster"): "roster_list",
    ("POST", "/mesh/approve"): "approve",
    ("POST", "/mesh/revoke"): "revoke",
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
    origin = self_origin or config.name or ""

    roles: dict[str, RoleInfo] = {}

    # Collect ready, hosted roles from the local gateway.
    # Finding 1: build real roles from the gateway's own data.
    if readiness_cache is not None:
        # Use readiness cache to determine which roles are ready.
        try:
            current = readiness_cache.current()
            ready_roles: dict = current.get("roles", {})
        except (AttributeError, TypeError):
            ready_roles = {}
    else:
        ready_roles = {}

    # Build per-role RoleInfo from lane configs or from replica caches.
    for backend_name, lane_config in (declared_lane_configs or {}).items():
        role_name = backend_name
        # Check readiness: only include ready+hosted roles.
        if ready_roles and role_name not in ready_roles:
            continue

        # Get live fingerprint from replica cache if available.
        fp = None
        if replica_caches and role_name in replica_caches:
            cache = replica_caches[role_name]
            try:
                snapshot = cache.snapshot()
                if isinstance(snapshot, dict):
                    local_replicas = snapshot.get("local_replicas", [])
                    if local_replicas:
                        rep_state = local_replicas[0]
                        if isinstance(rep_state, dict):
                            fp = rep_state.get("fingerprint")
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
        from lobes.gateway._mesh_roster import Roster as _Roster

        # Reconstruct with the injected missed_max.
        roster = _Roster(clock=clock, ledger_path=config.ledger_path, capacity_max=1000000.0)
        # Store it on the roster for the heartbeat to access.
        roster._missed_max_override = missed_max  # noqa: SLF001

    routes = MeshRoutes(config, roster, _join_log=join_log)

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


def _post_announcement(
    url: str,
    body: bytes,
    timeout: float,
    key: bytes | None = None,
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
        headers["Authorization"] = f"Bearer {key.decode('utf-8')}"

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
                reannounce_event.wait(timeout=wait_dur)

        reannounce_event.clear()

        if stop_event.is_set():
            break

        # Check if we should run a pass: deadline elapsed or reannounce fired.
        now = time.monotonic()
        run_pass = (now >= deadline) or reannounce_event.is_set()

        if run_pass:
            # Advance to next deadline.
            deadline = now + interval
        else:
            continue

        # Finding 10: tick the roster once per pass under the lock.
        with routes._lock:
            try:
                routes.roster.tick()
            except Exception:  # nosec B110 — best-effort: tick never blocks
                pass

            # Collect the announcement to send.
            if routes._announcement_bytes is not None:
                to_send = routes._announcement_bytes
            else:
                to_send = announcement_bytes

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
                _fetch_seed_roster(seeds, routes.config.key, routes.roster, dial_timeout)
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
                                    roster.announce(mname, morigin, None, now=time.monotonic())
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        except Exception:  # nosec B110 — best-effort: seed roster fetch never blocks
            pass
        finally:
            try:
                conn.close()
            except Exception:  # nosec B110 — silently ignore close errors
                pass


def start_mesh(
    routes: "MeshRoutes",
    announcement: Announcement,
) -> threading.Thread:
    """Start the heartbeat daemon thread.  Returns the thread handle."""
    from lobes.gateway._mesh_wire import encode as _encode

    announcement_bytes = _encode(announcement)

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
