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

import json
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lobes.gateway._authlog import RejectionLog
from lobes.gateway._mesh_config import MeshConfig, build_mesh_config
from lobes.gateway._mesh_wire import (
    SCHEMA_MAJOR,
    Announcement,
    Fingerprint,
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

        with self._lock:
            pending: list[_PendingJoin] = getattr(self, "_pending", [])

            # Remove expired entries first.
            pending = [p for p in pending if p.joined_at + ttl > now]
            self._pending = pending  # type: ignore[attr-defined]

            # Capacity cap: 8 pending entries total.
            if len(pending) >= 8:
                return (
                    400,
                    [("Content-Type", "application/json")],
                    json.dumps({"error": "pending join queue is full (8)"}).encode(),
                )

            # One entry per origin.
            for p in pending:
                if p.name == name:
                    if p.origin == origin:
                        return (
                            400,
                            [("Content-Type", "application/json")],
                            json.dumps({"error": "same origin already pending"}).encode(),
                        )
                    else:
                        pending.remove(p)

            pending.append(_PendingJoin(name=name, origin=origin, capacity=capacity, joined_at=now))

        # Collapsed flood logging via the existing RejectionLog pattern.
        if self._join_log is not None:
            source = "<join>"
            line = self._join_log.record(
                source,
                "POST",
                "/mesh/join",
                "join_flooded",
            )
            if line is not None:
                sys.stderr.write(f"[gateway] {line}\n")
            else:
                # Collapsed — count suppressed for the message.
                suppressed = 0
                with self._join_log._lock:  # noqa: SLF001
                    st = self._join_log._sources.get(source)  # noqa: SLF001
                    if st:
                        suppressed = st.suppressed
                msg = f"pending join registered for {name}"
                if suppressed:
                    msg += f" (+{suppressed} more suppressed)"
                return (
                    202,
                    [("Content-Type", "application/json")],
                    json.dumps({"status": msg}).encode(),
                )

        return (
            202,
            [("Content-Type", "application/json")],
            json.dumps({"status": f"pending join registered for {name}"}).encode(),
        )

    def announce(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/announce – authenticated membership heartbeat."""
        if not self._check_key(getattr(handler, "headers", {})):
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
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
        roles: dict[str, RoleInfo]

        try:
            from lobes.gateway._mesh_wire import decode

            announced = decode(body)
            public = announced.public()
            name = public.name
            origin = public.origin
            roles = public.roles
        except Exception:
            # Parse name/origin/roles from raw JSON if decode fails.
            try:
                data = json.loads(body)
                name = str(data.get("name", ""))
                origin = str(data.get("origin", ""))
                raw_roles: dict = data.get("roles", {})
                roles = {}
                for rn, rd in raw_roles.items():
                    if isinstance(rd, dict):
                        fp = rd.get("fingerprint", {})
                        roles[rn] = RoleInfo(
                            model=rd.get("model", ""),
                            runtime=rd.get("runtime", ""),
                            context=rd.get("context", 0),
                            quant=rd.get("quant", ""),
                            responsibilities=tuple(rd.get("responsibilities", [])),
                            forbidden_responsibilities=tuple(
                                rd.get("forbidden_responsibilities", [])
                            ),
                            fingerprint=Fingerprint(
                                served_id=fp.get("served_id", ""),
                                quantization=fp.get("quantization", ""),
                                max_model_len=fp.get("max_model_len", 0),
                                runtime=fp.get("runtime", ""),
                            ),
                            capacity=rd.get("capacity"),
                            private=rd.get("private", False),
                        )
            except Exception:
                return (
                    400,
                    [("Content-Type", "application/json")],
                    json.dumps({"error": "invalid JSON body"}).encode(),
                )

        if not name:
            return (
                400,
                [("Content-Type", "application/json")],
                json.dumps({"error": "name is required"}).encode(),
            )

        self.roster.announce(name, origin, roles, now=time.monotonic())

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
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
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

        members = self.roster.members()
        return (
            200,
            [("Content-Type", "application/json")],
            json.dumps({"members": members}).encode(),
        )

    def approve(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/approve – authenticated approval."""
        if not self._check_key(getattr(handler, "headers", {})):
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
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

        self.roster.approve(str(name), approved_by, expiry_f, now=time.monotonic())
        return (
            200,
            [("Content-Type", "application/json")],
            json.dumps({"status": "approved", "name": name}).encode(),
        )

    def revoke(self, handler: object) -> tuple[int, list[tuple[str, str]], bytes]:
        """POST /mesh/revoke – authenticated revocation."""
        if not self._check_key(getattr(handler, "headers", {})):
            return (
                401,
                [
                    ("Content-Type", "application/json"),
                    ("WWW-Authenticate", "Bearer"),
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

        self.roster.revoke(str(name), now=time.monotonic(), approved_by=approved_by)
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


def build_mesh_routes(
    *,
    env: object | None = None,
    roster: "Roster | None" = None,
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

    routes = MeshRoutes(config, roster)

    # Build the initial announcement from current roster members.
    roles: dict[str, RoleInfo] = {}
    for member_name in roster.members():
        rec = roster._roster.get(member_name)  # noqa: SLF001
        if rec is None:
            continue
        member_roles = rec.origin if isinstance(rec.origin, dict) else {}
        if isinstance(member_roles, dict):
            for role_name, role_data in member_roles.items():
                if isinstance(role_data, dict):
                    fp_obj = role_data.get("fingerprint", {})
                    roles[role_name] = RoleInfo(
                        model=role_data.get("model", ""),
                        runtime=role_data.get("runtime", ""),
                        context=role_data.get("context", 0),
                        quant=role_data.get("quant", ""),
                        responsibilities=tuple(role_data.get("responsibilities", [])),
                        forbidden_responsibilities=tuple(
                            role_data.get("forbidden_responsibilities", [])
                        ),
                        fingerprint=Fingerprint(
                            served_id=fp_obj.get("served_id", ""),
                            quantization=fp_obj.get("quantization", ""),
                            max_model_len=fp_obj.get("max_model_len", 0),
                            runtime=fp_obj.get("runtime", ""),
                        ),
                        capacity=role_data.get("capacity"),
                        private=role_data.get("private", False),
                    )

    announcement = Announcement(
        name=config.name,
        origin="",
        schema_version=str(SCHEMA_MAJOR),
        roles=roles,
    )
    return routes, announcement


# --- heartbeat daemon ------------------------------------------------------


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

    while not stop_event.is_set():
        # Event.wait(interval) returns True only when stop() is set, so this
        # paces the loop and exits promptly on shutdown.
        reannounce_event.wait(timeout=min(interval, 1.0))
        reannounce_event.clear()

        if stop_event.is_set():
            break

        # Collect the announcement to send.
        with routes._lock:  # noqa: SLF001
            if routes._announcement_bytes is not None:  # noqa: SLF001
                to_send = routes._announcement_bytes  # noqa: SLF001
            else:
                to_send = announcement_bytes

        if to_send is None:
            continue

        peer_timeout = routes.config.missed_max * 10  # generous timeout per peer

        # Announce to seeds first.
        for seed in seeds:
            if stop_event.is_set():
                break
            _post_announcement(seed + "/mesh/announce", to_send, peer_timeout)

        # Announce to every roster member.
        for member_name in list(routes.roster.members()):
            if stop_event.is_set():
                break
            rec = routes.roster._roster.get(member_name)  # noqa: SLF001
            if rec is None:
                continue
            origin = rec.origin
            _post_announcement(origin + "/mesh/announce", to_send, peer_timeout)


def _post_announcement(url: str, body: bytes, timeout: float) -> None:
    """POST *body* to *url* via http.client with a hard timeout.

    Silently drops on any failure — a down peer is handled by tick-based
    staleness in the roster.
    """
    import http.client
    import urllib.parse

    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    path = parsed.path or "/mesh/announce"
    if parsed.query:
        path = path + "?" + parsed.query

    conn = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
            },
        )
        conn.getresponse()
    except Exception:  # nosec B110 — best-effort: silently drop failed peer connections
        # Best-effort: a down peer is handled by tick-based staleness.
        pass
    finally:
        if conn is not None:
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

    stop_event = threading.Event()
    routes._stop = stop_event  # noqa: SLF001

    thread = threading.Thread(
        target=_heartbeat_loop,
        args=(
            routes,
            announcement_bytes,
            routes.config.heartbeat_s,
            stop_event,
            routes._reannounce_event,
        ),
        name="lobes-mesh-heartbeat",
        daemon=True,
    )
    thread.start()
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
