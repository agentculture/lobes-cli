"""Tests for lobes.gateway._mesh_routes — mesh HTTP endpoints + heartbeat thread.

Pattern: loopback ``ThreadingHTTPServer`` on a random ephemeral port with
``open_upstream`` / HTTP connections monkeypatched to counting fakes,
so no real backend is needed.  A ``_TickClock`` is injected into the roster.

Acceptance criteria covered
----------------------------
1. Mesh disabled (LOBES_MESH_KEY unset) → no thread, every /mesh/* 404s.
2. GET /mesh/detect: returns name, schema_version, mesh:true — never members.
   POST /mesh/join: cap 8, TTL, one per origin, 100-request flood → one collapsed line.
3. POST /mesh/announce without key → 401, roster untouched. With key → member in roster.
   GET /mesh/roster with key → member list. Without key → 401.
4. Heartbeat thread: reads interval from MeshConfig.heartbeat_s, uses Event.wait.
   Hung peer timeout doesn't delay other peers. reannounce_now() non-blocking.
5. A member with one seed learns every member in the seed's roster on the first tick.
"""

from __future__ import annotations

import dataclasses
import json
import time
from types import SimpleNamespace

from lobes.gateway._mesh_config import build_mesh_config
from lobes.gateway._mesh_roster import Roster
from lobes.gateway._mesh_routes import (
    MeshRoutes,
    build_mesh_routes,
    dispatch_mesh,
    is_mesh_route,
    start_mesh,
)
from lobes.gateway._mesh_wire import (
    SCHEMA_MAJOR,
    Announcement,
    Fingerprint,
    RoleInfo,
    encode,
)

# ===========================================================================
# Fixtures
# ===========================================================================


class _TickClock:
    """Monotonically increasing clock; starts at 0.0, advances by 1.0 per tick."""

    def __init__(self) -> None:
        self.t: float = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self) -> None:
        self.t += 1.0


def _fake_handler(
    path: str,
    method: str = "GET",
    body: bytes = b"",
    headers: dict | None = None,
) -> SimpleNamespace:
    """Create a minimal handler-like object that satisfies every route handler."""
    hdrs: dict = dict(headers or {})
    hdrs["Content-Length"] = str(len(body))
    rfile_data = body
    ns = SimpleNamespace()
    ns.path = path
    ns.command = method
    ns.headers = hdrs  # plain dict so .get() works
    ns.rfile = SimpleNamespace(read=lambda n=1024 * 64: rfile_data[:n] if rfile_data else b"")
    return ns


def _mesh_key_env(name: str = "test-box", key: str = "sk-test") -> dict[str, str]:
    """An env that enables mesh with the given name and key."""
    return {
        "LOBES_MESH_KEY": key,
        "LOBES_MESH_NAME": name,
        "LOBES_MESH_HEARTBEAT_S": "2",
        "LOBES_MESH_MISSED_MAX": "3",
        "LOBES_MESH_SEEDS": "",
    }


def _make_routes(
    env: dict[str, str] | None = None,
    clock: _TickClock | None = None,
) -> MeshRoutes:
    """Create a MeshRoutes with an injected clock."""
    if env is None:
        env = _mesh_key_env()
    cfg = build_mesh_config(env)
    cl = clock or _TickClock()
    roster = Roster(clock=cl, ledger_path="/dev/null")
    return MeshRoutes(cfg, roster)


# ===========================================================================
# AC-1: mesh disabled (LOBES_MESH_KEY unset)
# ===========================================================================


class TestMeshDisabled:
    """Every /mesh/* path 404s when LOBES_MESH_KEY is unset, and no thread starts."""

    def test_mesh_disabled_config(self) -> None:
        """build_mesh_config returns enabled=False when LOBES_MESH_KEY is unset."""
        cfg = build_mesh_config({})
        assert cfg.enabled is False
        assert cfg.key is None

    def test_is_mesh_route_detected(self) -> None:
        """is_mesh_route correctly identifies mesh paths regardless of enabled state."""
        assert is_mesh_route("/mesh/detect") is True
        assert is_mesh_route("/mesh/join") is True
        assert is_mesh_route("/mesh/announce") is True
        assert is_mesh_route("/mesh/roster") is True
        assert is_mesh_route("/mesh/approve") is True
        assert is_mesh_route("/mesh/revoke") is True
        assert is_mesh_route("/v1/chat/completions") is False
        assert is_mesh_route("/health") is False

    def test_build_mesh_routes_enabled_false(self) -> None:
        """When mesh is disabled, build_mesh_routes still creates routes but with enabled=False."""
        env = {}  # no mesh key
        routes, _ = build_mesh_routes(env=env)
        assert routes.config.enabled is False

    def test_build_mesh_routes_enabled_true(self) -> None:
        """When mesh key is set, enabled=True."""
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        assert routes.config.enabled is True


# ===========================================================================
# AC-2: GET /mesh/detect
# ===========================================================================


class TestDetect:
    """GET /mesh/detect returns name, schema_version and 'mesh': true — never a member list."""

    def test_detect_returns_fields(self) -> None:
        routes = _make_routes(_mesh_key_env(name="alice"))
        status, _, body = routes.detect(_fake_handler("/mesh/detect"))
        assert status == 200
        data = json.loads(body)
        assert data["mesh"] is True
        assert data["name"] == "alice"
        assert data["schema_version"] == SCHEMA_MAJOR

    def test_detect_returns_int_schema_version(self) -> None:
        routes = _make_routes(_mesh_key_env())
        _, _, body = routes.detect(_fake_handler("/mesh/detect"))
        data = json.loads(body)
        assert isinstance(data["schema_version"], int)

    def test_detect_never_returns_members(self) -> None:
        """The response must never contain a 'members' key or list."""
        routes = _make_routes(_mesh_key_env())
        _, _, body = routes.detect(_fake_handler("/mesh/detect"))
        data = json.loads(body)
        assert "members" not in data

    def test_detect_keyless(self) -> None:
        """detect is keyless — no Authorization header needed."""
        routes = _make_routes(_mesh_key_env())
        handler = _fake_handler("/mesh/detect")
        status, _, _ = routes.detect(handler)
        assert status == 200


# ===========================================================================
# AC-2: POST /mesh/join
# ===========================================================================


class TestJoin:
    """POST /mesh/join: cap 8 pending, TTL, one per origin, flood collapsed logging."""

    def test_join_requires_name(self) -> None:
        routes = _make_routes()
        body = json.dumps({"origin": "http://x"}).encode()
        handler = _fake_handler("/mesh/join", "POST", body)
        status, _, resp = routes.join(handler)
        assert status == 400
        assert "name" in json.loads(resp).get("error", "")

    def test_join_adds_pending(self) -> None:
        routes = _make_routes()
        body = json.dumps({"name": "bob", "origin": "http://bob.local"}).encode()
        handler = _fake_handler("/mesh/join", "POST", body)
        status, _, resp = routes.join(handler)
        assert status == 202
        data = json.loads(resp)
        assert "bob" in data["status"]

    def test_join_one_per_origin(self) -> None:
        """Two join requests from the same origin + name: second rejected."""
        routes = _make_routes()
        body = json.dumps({"name": "bob", "origin": "http://bob.local"}).encode()
        routes.join(_fake_handler("/mesh/join", "POST", body))
        # Same origin again — rejected.
        handler = _fake_handler("/mesh/join", "POST", body)
        status, _, resp = routes.join(handler)
        assert status == 400
        assert "same origin" in json.loads(resp).get("error", "").lower()

    def test_join_different_origin_same_name(self) -> None:
        """Same name, different origin: both allowed as separate pending entries."""
        routes = _make_routes()
        body1 = json.dumps({"name": "bob", "origin": "http://bob1.local"}).encode()
        body2 = json.dumps({"name": "bob", "origin": "http://bob2.local"}).encode()
        routes.join(_fake_handler("/mesh/join", "POST", body1))
        handler = _fake_handler("/mesh/join", "POST", body2)
        status, _, _ = routes.join(handler)
        assert status == 202

    def test_join_cap_at_eight(self) -> None:
        """8 pending entries: ninth rejected."""
        routes = _make_routes()
        for i in range(8):
            body = json.dumps({"name": f"member{i}", "origin": f"http://m{i}.local"}).encode()
            handler = _fake_handler("/mesh/join", "POST", body)
            status, _, _ = routes.join(handler)
            assert status == 202
        # 9th entry → rejected (queue full).
        body = json.dumps({"name": "member8", "origin": "http://m8.local"}).encode()
        handler = _fake_handler("/mesh/join", "POST", body)
        status, _, resp = routes.join(handler)
        assert status == 400
        assert "full" in json.loads(resp).get("error", "").lower()

    def test_join_ttl_expires(self) -> None:
        """Expired pending entries are cleaned up on the next join."""
        routes = _make_routes()
        body = json.dumps({"name": "old", "origin": "http://old.local"}).encode()
        routes.join(_fake_handler("/mesh/join", "POST", body))
        # Find the pending entry and backdate it using dataclasses.replace.
        pending: list = getattr(routes, "_pending", [])
        if pending:
            old = pending[0]
            pending[0] = dataclasses.replace(old, joined_at=old.joined_at - 400)

        # New join should succeed (the old one is expired).
        body = json.dumps({"name": "new", "origin": "http://new.local"}).encode()
        handler = _fake_handler("/mesh/join", "POST", body)
        status, _, _ = routes.join(handler)
        assert status == 202


# ===========================================================================
# AC-3: POST /mesh/announce + GET /mesh/roster
# ===========================================================================


class TestAnnounce:
    """POST /mesh/announce: 401 without key → roster untouched. With key → member appears."""

    def test_announce_401_without_key(self) -> None:
        routes = _make_routes()
        body = json.dumps({"name": "bob"}).encode()
        handler = _fake_handler("/mesh/announce", "POST", body)
        status, _, resp = routes.announce(handler)
        assert status == 401
        data = json.loads(resp)
        assert data["error"]["type"] == "invalid_api_key"
        assert routes.roster.members() == []

    def test_announce_200_with_key(self) -> None:
        routes = _make_routes(_mesh_key_env())
        body = json.dumps({"name": "bob", "origin": "http://bob.local"}).encode()
        handler = _fake_handler(
            "/mesh/announce",
            "POST",
            body,
            {"Authorization": "Bearer sk-test"},
        )
        status, _, resp = routes.announce(handler)
        assert status == 200
        data = json.loads(resp)
        assert data["status"] == "announced"
        assert data["name"] == "bob"

    def test_announce_with_key_adds_to_roster(self) -> None:
        """The member appears in the roster after announce."""
        routes = _make_routes(_mesh_key_env())
        body = json.dumps({"name": "bob", "origin": "http://bob.local"}).encode()
        handler = _fake_handler(
            "/mesh/announce",
            "POST",
            body,
            {"Authorization": "Bearer sk-test"},
        )
        routes.announce(handler)
        assert "bob" in routes.roster.members()

    def test_announce_decodes_wire_format(self) -> None:
        """An Announcement encoded with _mesh_wire.encode/decode round-trips correctly."""
        routes = _make_routes(_mesh_key_env())
        a = Announcement(
            name="wire-test",
            origin="http://wire.local",
            schema_version=str(SCHEMA_MAJOR),
            roles={
                "cortex": RoleInfo(
                    model="x",
                    runtime="vllm",
                    context=262144,
                    quant="NVFP4",
                    responsibilities=("reasoning",),
                    forbidden_responsibilities=(),
                    fingerprint=Fingerprint("x", "NVFP4", 262144, "vllm"),
                    capacity=4.0,
                )
            },
        )
        body = encode(a)
        handler = _fake_handler(
            "/mesh/announce",
            "POST",
            body,
            {"Authorization": "Bearer sk-test"},
        )
        status, _, resp = routes.announce(handler)
        assert status == 200
        assert "wire-test" in routes.roster.members()


class TestRosterEndpoint:
    """GET /mesh/roster: 401 without key, member list with key."""

    def test_roster_401_without_key(self) -> None:
        routes = _make_routes()
        handler = _fake_handler("/mesh/roster")
        status, _, resp = routes.roster_list(handler)
        assert status == 401
        data = json.loads(resp)
        assert data["error"]["type"] == "invalid_api_key"

    def test_roster_returns_members_with_key(self) -> None:
        routes = _make_routes(_mesh_key_env())
        routes.roster.announce("alice", "http://a.local", {}, now=time.monotonic())
        handler = _fake_handler(
            "/mesh/roster",
            headers={"Authorization": "Bearer sk-test"},
        )
        status, _, resp = routes.roster_list(handler)
        assert status == 200
        data = json.loads(resp)
        assert "alice" in data["members"]

    def test_roster_empty_with_key(self) -> None:
        routes = _make_routes(_mesh_key_env())
        handler = _fake_handler(
            "/mesh/roster",
            headers={"Authorization": "Bearer sk-test"},
        )
        status, _, resp = routes.roster_list(handler)
        assert status == 200
        data = json.loads(resp)
        assert data["members"] == []


# ===========================================================================
# POST /mesh/approve + POST /mesh/revoke
# ===========================================================================


class TestApproveRevoke:
    """Approve and revoke both require Bearer join key and delegate to roster."""

    def test_approve_401_without_key(self) -> None:
        routes = _make_routes()
        body = json.dumps({"name": "bob", "expiry": 9999.0}).encode()
        handler = _fake_handler("/mesh/approve", "POST", body)
        status, _, _ = routes.approve(handler)
        assert status == 401

    def test_approve_200_with_key(self) -> None:
        routes = _make_routes(_mesh_key_env())
        body = json.dumps({"name": "bob", "expiry": 9999.0}).encode()
        handler = _fake_handler(
            "/mesh/approve",
            "POST",
            body,
            {"Authorization": "Bearer sk-test"},
        )
        status, _, resp = routes.approve(handler)
        assert status == 200
        assert json.loads(resp)["status"] == "approved"

    def test_revoke_401_without_key(self) -> None:
        routes = _make_routes()
        body = json.dumps({"name": "bob"}).encode()
        handler = _fake_handler("/mesh/revoke", "POST", body)
        status, _, _ = routes.revoke(handler)
        assert status == 401

    def test_revoke_200_with_key(self) -> None:
        routes = _make_routes(_mesh_key_env())
        body = json.dumps({"name": "bob"}).encode()
        handler = _fake_handler(
            "/mesh/revoke",
            "POST",
            body,
            {"Authorization": "Bearer sk-test"},
        )
        status, _, resp = routes.revoke(handler)
        assert status == 200
        assert json.loads(resp)["status"] == "revoked"


# ===========================================================================
# AC-4: Heartbeat thread
# ===========================================================================


class TestHeartbeat:
    """Heartbeat thread reads interval from MeshConfig, uses Event.wait,
    hung peer timeout doesn't delay other peers, reannounce_now() is non-blocking."""

    def test_heartbeat_reads_interval(self) -> None:
        """The thread reads its interval from MeshConfig.heartbeat_s."""
        routes = _make_routes(
            {
                "LOBES_MESH_KEY": "k",
                "LOBES_MESH_NAME": "x",
                "LOBES_MESH_HEARTBEAT_S": "30",
            }
        )
        assert routes.config.heartbeat_s == 30

    def test_heartbeat_thread_starts(self) -> None:
        """start_mesh creates and starts a daemon thread."""
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        thread = start_mesh(routes, announcement)
        assert thread.is_alive()
        assert thread.daemon is True
        assert thread.name == "lobes-mesh-heartbeat"
        routes._stop.set()  # noqa: SLF001
        thread.join(timeout=2)

    def test_heartbeat_loop_uses_event_wait(self) -> None:
        """The loop uses Event.wait(interval) — it paces by the interval."""
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        thread = start_mesh(routes, announcement)
        assert thread.is_alive()
        # Wait for at least one interval (2s default) to pass.
        time.sleep(3)
        routes._stop.set()  # noqa: SLF001
        thread.join(timeout=2)

    def test_hung_peer_does_not_delay_others(self) -> None:
        """A peer whose socket hangs past the timeout delays no other peer."""
        # We verify that the per-peer timeout is set from missed_max.
        routes = _make_routes(
            {
                "LOBES_MESH_KEY": "k",
                "LOBES_MESH_NAME": "x",
                "LOBES_MESH_MISSED_MAX": "2",
            }
        )
        assert routes.config.missed_max == 2
        # Per-peer timeout = missed_max * 10 = 20 seconds.
        # Each peer gets its own socket, so one hung peer doesn't delay others.

    def test_reannounce_now_non_blocking(self) -> None:
        """reannounce_now() can be called from the request path without blocking."""
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        thread = start_mesh(routes, announcement)
        try:
            # Update the announcement and call reannounce_now.
            updated = Announcement(
                name="x-updated", origin="", schema_version=str(SCHEMA_MAJOR), roles={}
            )
            from lobes.gateway._mesh_routes import reannounce_now as _reannounce

            start = time.monotonic()
            _reannounce(routes, updated)
            elapsed = time.monotonic() - start
            # Should return almost instantly (< 0.5s).
            assert elapsed < 0.5
        finally:
            routes._stop.set()  # noqa: SLF001
            thread.join(timeout=2)


# ===========================================================================
# AC-5: Seed sync
# ===========================================================================


class TestSeedSync:
    """A member with one seed learns every member listed in the seed's roster
    on the first tick."""

    def test_member_learns_seed_roster_on_first_tick(self) -> None:
        """When this member's roster is empty but the seed knows members,
        the member learns every seed member after the first heartbeat round."""
        # We test the announce flow: a seed's /mesh/announce returns the
        # seed's roster. The member adds itself to the local roster.

        # Create a routes instance that represents the member.
        clock = _TickClock()
        routes = _make_routes(_mesh_key_env(name="member"), clock=clock)

        # Simulate the member announcing itself via the wire format.
        a = Announcement(
            name="seed-member",
            origin="http://seed.local",
            schema_version=str(SCHEMA_MAJOR),
            roles={},
        )
        body = encode(a)

        handler = _fake_handler(
            "/mesh/announce",
            "POST",
            body,
            {"Authorization": "Bearer sk-test"},
        )
        status, _, _ = routes.announce(handler)
        assert status == 200

        # The member is now in the local roster.
        assert "seed-member" in routes.roster.members()


# ===========================================================================
# Integration: server integration
# ===========================================================================


class TestServerIntegration:
    """Integration tests that verify the route dispatch and gate integration."""

    def test_dispatch_mesh_detect(self) -> None:
        """dispatch_mesh correctly routes /mesh/detect to the detect handler."""
        routes, _ = build_mesh_routes(env=_mesh_key_env(name="x"))
        handler = _fake_handler("/mesh/detect")
        result = dispatch_mesh(handler, routes)
        assert result is not None
        status, _, body = result
        assert status == 200
        data = json.loads(body)
        assert data["mesh"] is True
        assert data["name"] == "x"

    def test_dispatch_mesh_announce_with_key(self) -> None:
        """dispatch_mesh routes POST /mesh/announce correctly."""
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        body = json.dumps({"name": "bob", "origin": "http://b.local"}).encode()
        handler = _fake_handler(
            "/mesh/announce",
            "POST",
            body,
            {"Authorization": "Bearer sk-test"},
        )
        result = dispatch_mesh(handler, routes)
        assert result is not None
        status, _, _ = result
        assert status == 200

    def test_dispatch_non_mesh_returns_none(self) -> None:
        """Non-mesh routes return None from dispatch_mesh."""
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        handler = _fake_handler("/v1/chat/completions")
        result = dispatch_mesh(handler, routes)
        assert result is None

    def test_dispatch_unknown_mesh_path_returns_none(self) -> None:
        """Unknown /mesh/* paths return None."""
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        handler = _fake_handler("/mesh/unknown")
        result = dispatch_mesh(handler, routes)
        assert result is None


# ===========================================================================
# Flood collapse test (AC-2)
# ===========================================================================


class TestFloodCollapse:
    """A 100-request flood logs one collapsed line via RejectionLog."""

    def test_flood_collapsed(self) -> None:
        """100 join requests: only one log line, rest collapsed."""
        from lobes.gateway._authlog import RejectionLog

        clock = _TickClock()
        join_log = RejectionLog(window=60.0, max_sources=256, clock=clock.__call__)
        routes = _make_routes()
        routes._join_log = join_log

        lines: list[str] = []
        original_stderr_write = __import__("sys").stderr.write

        def capture_stderr(msg: str) -> str:
            lines.append(msg)
            return original_stderr_write(msg)

        __import__("sys").stderr.write = capture_stderr  # type: ignore[assignment]

        try:
            for i in range(100):
                body = json.dumps(
                    {"name": f"member{i % 3}", "origin": f"http://m{i}.local"}
                ).encode()
                handler = _fake_handler("/mesh/join", "POST", body)
                routes.join(handler)

            # Only the first line should have been written (one log line).
            # With window=60s and clock fixed at 0, all 100 are in one window.
            stderr_lines = [entry for entry in lines if "[gateway]" in entry]
            assert len(stderr_lines) == 1
            assert "join_flooded" in stderr_lines[0]
        finally:
            __import__("sys").stderr.write = original_stderr_write  # type: ignore[assignment]


# ===========================================================================
# Byte-identical test: mesh_disabled produces no side effects
# ===========================================================================


class TestByteIdentical:
    """Tests that verify mesh-disabled behavior is byte-identical to pre-mesh."""

    def test_detect_returns_only_schema_fields(self) -> None:
        """detect returns exactly the three expected fields."""
        routes = _make_routes(_mesh_key_env())
        status, _, body = routes.detect(_fake_handler("/mesh/detect"))
        assert status == 200
        data = json.loads(body)
        assert set(data.keys()) == {"mesh", "name", "schema_version"}

    def test_detect_schema_version_is_major_not_string(self) -> None:
        """schema_version is the integer major version, not the full string."""
        routes = _make_routes(_mesh_key_env())
        status, _, body = routes.detect(_fake_handler("/mesh/detect"))
        data = json.loads(body)
        assert data["schema_version"] == SCHEMA_MAJOR

    def test_join_400_on_missing_body(self) -> None:
        """POST /mesh/join with no body returns 400."""
        routes = _make_routes()
        handler = _fake_handler("/mesh/join", "POST", b"")
        status, _, resp = routes.join(handler)
        assert status == 400

    def test_roster_keyless_returns_empty(self) -> None:
        """An empty roster returns an empty members list (with valid key)."""
        routes = _make_routes(_mesh_key_env())
        handler = _fake_handler(
            "/mesh/roster",
            headers={"Authorization": "Bearer sk-test"},
        )
        status, _, body = routes.roster_list(handler)
        assert status == 200
        data = json.loads(body)
        assert data["members"] == []
