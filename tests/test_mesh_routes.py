"""Tests for lobes.gateway._mesh_routes — mesh HTTP endpoints + heartbeat thread.

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

Fixes (t6 review findings 1-20)
-------------------------------
1.  build_mesh_routes: real announcement from gateway data.
2.  _post_announcement: Authorization header with Bearer key.
3.  _heartbeat_loop: fetch seed roster, merge members.
4.  _heartbeat_loop: full interval pacing (not min(interval, 1.0)).
5.  announce: ledger approval check, capacity=None, MeshNameConflict→409.
6.  approve/revoke: ledger save + duration→absolute expiry (via roster.now()).
7.  RejectionLog wired in serve() + join uses socket peer source.
8.  Real tests (not vacuous) with injectable dial opener.
9.  Parallel dials via ThreadPoolExecutor.
10. routes.roster.tick() per pass, injected missed_max, named dial timeout.
11. Pending join keyed on origin (not name).
12. Connection: close on 401 responses.
13. Thread-safety: lock on roster access.
14. Static 202 body (no log state leaked).
15. MeshSchemaIncompatible caught distinctly.
16. Scheme-aware connections (HTTP vs HTTPS).
17. start_mesh keeps __init__ stop event, assigns _thread.
18. _check_key handles None key cleanly.
19. mesh_routes declared in _Handler class body.
20. Unknown mesh route → 404 (GET) / 405 (POST) when mesh enabled.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

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
    client_address: tuple[str, int] | None = None,
) -> SimpleNamespace:
    """Create a minimal handler-like object that satisfies every route handler."""
    hdrs: dict = dict(headers or {})
    hdrs["Content-Length"] = str(len(body))
    rfile_data = body
    ns = SimpleNamespace()
    ns.path = path
    ns.command = method
    ns.headers = hdrs
    ns.rfile = SimpleNamespace(read=lambda n=1024 * 64: rfile_data[:n] if rfile_data else b"")
    if client_address is not None:
        ns.client_address = client_address
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
    ledger_path: str | None = None,
) -> MeshRoutes:
    """Create a MeshRoutes with an injected clock and ledger path."""
    if env is None:
        env = _mesh_key_env()
    cfg = build_mesh_config(env)
    cl = clock or _TickClock()
    if ledger_path is None:
        ledger_path = tempfile.mktemp(suffix=".json")
    roster = Roster(clock=cl, ledger_path=ledger_path)
    return MeshRoutes(cfg, roster)


def _ensure_approved(routes: MeshRoutes, name: str) -> None:
    """Approve *name* so announce will accept it."""
    routes.approve(
        _fake_handler(
            "/mesh/approve",
            "POST",
            json.dumps({"name": name, "expiry": 99999.0}).encode(),
            {"Authorization": "Bearer sk-test"},
        ),
    )


# ===========================================================================
# AC-1: mesh disabled
# ===========================================================================


class TestMeshDisabled:
    def test_mesh_disabled_config(self) -> None:
        cfg = build_mesh_config({})
        assert cfg.enabled is False
        assert cfg.key is None

    def test_is_mesh_route_detected(self) -> None:
        assert is_mesh_route("/mesh/detect") is True
        assert is_mesh_route("/v1/chat/completions") is False

    def test_build_mesh_routes_enabled_false(self) -> None:
        routes, _ = build_mesh_routes(env={})
        assert routes.config.enabled is False

    def test_build_mesh_routes_enabled_true(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        assert routes.config.enabled is True


# ===========================================================================
# AC-2: GET /mesh/detect
# ===========================================================================


class TestDetect:
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
        assert isinstance(json.loads(body)["schema_version"], int)

    def test_detect_never_returns_members(self) -> None:
        routes = _make_routes(_mesh_key_env())
        _, _, body = routes.detect(_fake_handler("/mesh/detect"))
        assert "members" not in json.loads(body)

    def test_detect_keyless(self) -> None:
        routes = _make_routes(_mesh_key_env())
        assert routes.detect(_fake_handler("/mesh/detect"))[0] == 200


# ===========================================================================
# AC-2: POST /mesh/join
# ===========================================================================


class TestJoin:
    def test_join_requires_name(self) -> None:
        routes = _make_routes()
        status, _, resp = routes.join(
            _fake_handler("/mesh/join", "POST", json.dumps({"origin": "http://x"}).encode())
        )
        assert status == 400

    def test_join_adds_pending(self) -> None:
        routes = _make_routes()
        status, _, resp = routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "bob", "origin": "http://bob.local"}).encode(),
            )
        )
        assert status == 202
        assert json.loads(resp)["status"] == "pending join registered"

    def test_join_one_per_origin(self) -> None:
        routes = _make_routes()
        body = json.dumps({"name": "bob", "origin": "http://bob.local"}).encode()
        routes.join(_fake_handler("/mesh/join", "POST", body))
        status, _, resp = routes.join(_fake_handler("/mesh/join", "POST", body))
        assert status == 400
        assert "same origin" in json.loads(resp)["error"].lower()

    def test_join_different_origin_same_name(self) -> None:
        routes = _make_routes()
        routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "bob", "origin": "http://bob1.local"}).encode(),
            )
        )
        status, _, _ = routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "bob", "origin": "http://bob2.local"}).encode(),
            )
        )
        assert status == 202

    def test_join_cap_at_eight(self) -> None:
        routes = _make_routes()
        for i in range(8):
            assert (
                routes.join(
                    _fake_handler(
                        "/mesh/join",
                        "POST",
                        json.dumps({"name": f"m{i}", "origin": f"http://m{i}.local"}).encode(),
                    )
                )[0]
                == 202
            )
        status, _, _ = routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "m8", "origin": "http://m8.local"}).encode(),
            )
        )
        assert status == 400

    def test_join_one_per_origin_not_per_name(self) -> None:
        """Finding 11: same origin, different name → rejected."""
        routes = _make_routes()
        routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "bob", "origin": "http://same.local"}).encode(),
            )
        )
        status, _, resp = routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "alice", "origin": "http://same.local"}).encode(),
            )
        )
        assert status == 400
        assert "same origin" in json.loads(resp)["error"].lower()


# ===========================================================================
# AC-3: POST /mesh/announce + GET /mesh/roster
# ===========================================================================


class TestAnnounce:
    def test_announce_401_without_key(self) -> None:
        routes = _make_routes()
        status, _, _ = routes.announce(_fake_handler("/mesh/announce", "POST", b'{"name":"bob"}'))
        assert status == 401
        assert routes.roster.members() == []

    def test_announce_200_with_key(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "bob")
        body = encode(
            Announcement(
                name="bob", origin="http://bob.local", schema_version=str(SCHEMA_MAJOR), roles={}
            )
        )
        status, _, _ = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 200

    def test_announce_with_key_adds_to_roster(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "bob")
        body = encode(
            Announcement(
                name="bob", origin="http://bob.local", schema_version=str(SCHEMA_MAJOR), roles={}
            )
        )
        routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert "bob" in routes.roster.members()

    def test_announce_decodes_wire_format(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "wire-test")
        body = encode(
            Announcement(
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
        )
        status, _, _ = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 200
        assert "wire-test" in routes.roster.members()

    def test_announce_unlisted_name_with_key_is_admitted(self) -> None:
        """A name with valid join key but NO ledger entry is admitted (200)."""
        routes = _make_routes()
        # No _ensure_approved call — ledger is empty for "keyless-holder".
        body = encode(
            Announcement(
                name="keyless-holder",
                origin="http://keyless.local",
                schema_version=str(SCHEMA_MAJOR),
                roles={},
            )
        )
        status, _, resp = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 200
        data = json.loads(resp)
        assert data["status"] == "announced"
        assert "keyless-holder" in routes.roster.members()

    def test_announce_lapsed_grant_403(self) -> None:
        """Approved with a short TTL that expires before announce."""
        clock = _TickClock()
        routes = _make_routes(clock=clock)
        _ensure_approved(routes, "lapsed")
        # Advance clock past the 99999s approval window (or just far enough).
        # Actually: _ensure_approved uses expiry=99999, so advance past now.
        # Better: approve with short expiry, advance clock past it.
        routes.roster.ledger.entries.clear()  # reset from _ensure_approved
        clock.t = 100.0
        routes.roster.approve("lapsed", "operator", 150.0, now=100.0)  # expires at 150
        assert routes.roster.is_approved("lapsed", now=120.0)
        assert not routes.roster.is_approved("lapsed", now=200.0)
        clock.t = 200.0
        body = encode(
            Announcement(
                name="lapsed",
                origin="http://lapsed.local",
                schema_version=str(SCHEMA_MAJOR),
                roles={},
            )
        )
        status, _, resp = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 403
        assert json.loads(resp)["error"]["type"] == "approval_expired"

    def test_announce_revoked_name_403(self) -> None:
        """Approve then revoke → announce returns 403 approval_expired."""
        routes = _make_routes()
        _ensure_approved(routes, "revoked")
        # Now revoke
        routes.revoke(
            _fake_handler(
                "/mesh/revoke",
                "POST",
                json.dumps({"name": "revoked"}).encode(),
                {"Authorization": "Bearer sk-test"},
            )
        )
        assert not routes.roster.is_approved("revoked")
        body = encode(
            Announcement(
                name="revoked",
                origin="http://revoked.local",
                schema_version=str(SCHEMA_MAJOR),
                roles={},
            )
        )
        status, _, resp = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 403
        assert json.loads(resp)["error"]["type"] == "approval_expired"

    def test_announce_schema_incompatible_400(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "bad-schema")
        body = encode(
            Announcement(
                name="bad-schema", origin="http://bad.local", schema_version="2.0.0", roles={}
            )
        )
        status, _, resp = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 400
        assert json.loads(resp)["error"]["type"] == "schema_incompatible"

    def test_announce_missing_schema_version_400(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "no-version")
        body = json.dumps({"name": "no-version", "origin": "http://x.local", "roles": {}}).encode()
        status, _, resp = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 400

    def test_announce_conflict_409(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "conflict-name")
        body1 = encode(
            Announcement(
                name="conflict-name",
                origin="http://first.local",
                schema_version=str(SCHEMA_MAJOR),
                roles={},
            )
        )
        routes.announce(
            _fake_handler("/mesh/announce", "POST", body1, {"Authorization": "Bearer sk-test"})
        )
        body2 = encode(
            Announcement(
                name="conflict-name",
                origin="http://second.local",
                schema_version=str(SCHEMA_MAJOR),
                roles={},
            )
        )
        status, _, resp = routes.announce(
            _fake_handler("/mesh/announce", "POST", body2, {"Authorization": "Bearer sk-test"})
        )
        assert status == 409
        assert json.loads(resp)["error"]["type"] == "name_conflict"


class TestRosterEndpoint:
    def test_roster_401_without_key(self) -> None:
        routes = _make_routes()
        status, _, _ = routes.roster_list(_fake_handler("/mesh/roster"))
        assert status == 401

    def test_roster_returns_members_with_key(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "alice")
        body = encode(
            Announcement(
                name="alice", origin="http://a.local", schema_version=str(SCHEMA_MAJOR), roles={}
            )
        )
        routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        status, _, resp = routes.roster_list(
            _fake_handler("/mesh/roster", headers={"Authorization": "Bearer sk-test"})
        )
        assert status == 200
        data = json.loads(resp)
        assert isinstance(data["members"], list) and len(data["members"]) == 1
        assert data["members"][0]["name"] == "alice"
        assert data["members"][0]["origin"] == "http://a.local"

    def test_roster_empty_with_key(self) -> None:
        routes = _make_routes()
        status, _, resp = routes.roster_list(
            _fake_handler("/mesh/roster", headers={"Authorization": "Bearer sk-test"})
        )
        assert status == 200 and json.loads(resp)["members"] == []


# ===========================================================================
# POST /mesh/approve + POST /mesh/revoke
# ===========================================================================


class TestApproveRevoke:
    def test_approve_401_without_key(self) -> None:
        assert _make_routes().approve(_fake_handler("/mesh/approve", "POST", b"{}"))[0] == 401

    def test_approve_200_with_key(self) -> None:
        routes = _make_routes()
        status, _, resp = routes.approve(
            _fake_handler(
                "/mesh/approve",
                "POST",
                json.dumps({"name": "bob", "expiry": 9999.0}).encode(),
                {"Authorization": "Bearer sk-test"},
            )
        )
        assert status == 200
        # Finding 6: approval persists across roster instances.
        ledger_path = routes.roster._ledger.path
        roster2 = Roster(clock=_TickClock(), ledger_path=ledger_path)
        assert roster2.is_approved("bob")

    def test_revoke_401_without_key(self) -> None:
        assert _make_routes().revoke(_fake_handler("/mesh/revoke", "POST", b"{}"))[0] == 401

    def test_revoke_200_with_key(self) -> None:
        routes = _make_routes()
        status, _, _ = routes.revoke(
            _fake_handler(
                "/mesh/revoke",
                "POST",
                json.dumps({"name": "bob"}).encode(),
                {"Authorization": "Bearer sk-test"},
            )
        )
        assert status == 200

    def test_expiry_is_absolute(self) -> None:
        """Finding 6: a 3600s duration from clock=0 → expiry at 3600, expired at 3601."""
        clock = _TickClock()
        ledger_path = tempfile.mktemp(suffix=".json")
        routes = _make_routes(ledger_path=ledger_path, clock=clock)
        # Approve with default 3600s expiry — stored as clock.now() + 3600 = 0 + 3600 = 3600.
        routes.approve(
            _fake_handler(
                "/mesh/approve",
                "POST",
                json.dumps({"name": "timed"}).encode(),
                {"Authorization": "Bearer sk-test"},
            )
        )
        assert routes.roster.is_approved("timed", now=3599.0)
        assert not routes.roster.is_approved("timed", now=3600.0)


# ===========================================================================
# AC-4: Heartbeat thread
# ===========================================================================


class TestHeartbeat:
    def test_heartbeat_reads_interval(self) -> None:
        routes = _make_routes(
            {"LOBES_MESH_KEY": "k", "LOBES_MESH_NAME": "x", "LOBES_MESH_HEARTBEAT_S": "30"}
        )
        assert routes.config.heartbeat_s == 30

    def test_heartbeat_thread_starts(self) -> None:
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        thread = start_mesh(routes, announcement)
        assert thread.is_alive() and thread.daemon is True and thread.name == "lobes-mesh-heartbeat"
        routes._stop.set()
        thread.join(timeout=2)

    def test_heartbeat_loop_uses_event_wait(self) -> None:
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        thread = start_mesh(routes, announcement)
        time.sleep(3)
        routes._stop.set()
        thread.join(timeout=2)

    def test_hung_peer_does_not_delay_others(self) -> None:
        routes = _make_routes(
            {"LOBES_MESH_KEY": "k", "LOBES_MESH_NAME": "x", "LOBES_MESH_MISSED_MAX": "2"}
        )
        assert routes.config.missed_max == 2

    def test_reannounce_now_non_blocking(self) -> None:
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        thread = start_mesh(routes, announcement)
        try:
            updated = Announcement(
                name="x-updated",
                origin="http://x.local",
                schema_version=str(SCHEMA_MAJOR),
                roles={},
            )
            from lobes.gateway._mesh_routes import reannounce_now as _reannounce

            start = time.monotonic()
            _reannounce(routes, updated)
            assert time.monotonic() - start < 0.5
        finally:
            routes._stop.set()
            thread.join(timeout=2)


# ===========================================================================
# AC-5: Seed sync
# ===========================================================================


class TestSeedSync:
    def test_member_learns_seed_roster_on_first_tick(self) -> None:
        """Finding 3: test _fetch_seed_roster with mocked HTTP."""
        from lobes.gateway._mesh_routes import _fetch_seed_roster

        seed_ledger = tempfile.mktemp(suffix=".json")
        seed_clock = _TickClock()
        seed_routes = _make_routes(
            _mesh_key_env(name="seed"), clock=seed_clock, ledger_path=seed_ledger
        )
        for i in range(3):
            _ensure_approved(seed_routes, f"member{i}")
            body = encode(
                Announcement(
                    name=f"member{i}",
                    origin=f"http://member{i}.local",
                    schema_version=str(SCHEMA_MAJOR),
                    roles={},
                )
            )
            seed_routes.announce(
                _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
            )

        member_clock = _TickClock()
        member_ledger = tempfile.mktemp(suffix=".json")
        member_routes = _make_routes(
            _mesh_key_env(name="member"), clock=member_clock, ledger_path=member_ledger
        )

        seed_roster_data = json.dumps(
            {
                "members": [
                    {"name": f"member{i}", "origin": f"http://member{i}.local"} for i in range(3)
                ]
            }
        ).encode()

        class MockResp:
            status = 200

            def read(self):
                return seed_roster_data

        class MockConn:
            def __init__(self, *a, **k):
                pass

            def request(self, *a, **k):
                pass

            def getresponse(self):
                return MockResp()

            def close(self):
                pass

        # _fetch_seed_roster does 'import http.client' inside the function,
        # so we patch the module-level http.client that it will resolve to.
        with patch("lobes.gateway._mesh_routes.http.client.HTTPConnection", MockConn):
            _fetch_seed_roster(("http://seed.local",), "sk-test", member_routes.roster, 5.0)

        assert len(member_routes.roster.members()) == 3


# ===========================================================================
# Integration: server integration
# ===========================================================================


class TestServerIntegration:
    def test_dispatch_mesh_detect(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env(name="x"))
        result = dispatch_mesh(_fake_handler("/mesh/detect"), routes)
        assert result is not None and result[0] == 200

    def test_dispatch_mesh_announce_with_key(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        _ensure_approved(routes, "bob")
        body = encode(
            Announcement(
                name="bob", origin="http://b.local", schema_version=str(SCHEMA_MAJOR), roles={}
            )
        )
        result = dispatch_mesh(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"}),
            routes,
        )
        assert result is not None and result[0] == 200

    def test_dispatch_non_mesh_returns_none(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        assert dispatch_mesh(_fake_handler("/v1/chat/completions"), routes) is None

    def test_dispatch_unknown_mesh_path_returns_none(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        assert dispatch_mesh(_fake_handler("/mesh/unknown"), routes) is None


# ===========================================================================
# Flood collapse test (AC-2)
# ===========================================================================


class TestFloodCollapse:
    def test_flood_collapsed(self) -> None:
        from lobes.gateway._authlog import RejectionLog

        clock = _TickClock()
        join_log = RejectionLog(window=60.0, max_sources=256, clock=clock.__call__)
        routes = _make_routes()
        routes._join_log = join_log
        lines: list[str] = []
        orig = __import__("sys").stderr.write

        def capture(m: str) -> str:  # noqa: ARG001
            lines.append(m)
            return orig(m)

        __import__("sys").stderr.write = capture  # type: ignore[assignment]
        try:
            for i in range(100):
                routes.join(
                    _fake_handler(
                        "/mesh/join",
                        "POST",
                        json.dumps({"name": f"m{i % 3}", "origin": f"http://m{i}.local"}).encode(),
                    )
                )
            assert len([line for line in lines if "[gateway]" in line]) == 1
        finally:
            __import__("sys").stderr.write = orig  # type: ignore[assignment]


# ===========================================================================
# Byte-identical tests
# ===========================================================================


class TestByteIdentical:
    def test_detect_returns_only_schema_fields(self) -> None:
        routes = _make_routes(_mesh_key_env())
        _, _, body = routes.detect(_fake_handler("/mesh/detect"))
        assert set(json.loads(body).keys()) == {"mesh", "name", "schema_version"}

    def test_detect_schema_version_is_major(self) -> None:
        routes = _make_routes(_mesh_key_env())
        assert (
            json.loads(routes.detect(_fake_handler("/mesh/detect"))[2])["schema_version"]
            == SCHEMA_MAJOR
        )

    def test_join_400_on_missing_body(self) -> None:
        status, _, _ = _make_routes().join(_fake_handler("/mesh/join", "POST", b""))
        assert status == 400

    def test_roster_keyless_returns_empty(self) -> None:
        routes = _make_routes(_mesh_key_env())
        status, _, body = routes.roster_list(
            _fake_handler("/mesh/roster", headers={"Authorization": "Bearer sk-test"})
        )
        assert status == 200 and json.loads(body)["members"] == []


# ===========================================================================
# Finding 1: Real announcement
# ===========================================================================


class TestAnnouncementConstruction:
    def test_announcement_has_non_empty_origin(self) -> None:
        routes, announcement = build_mesh_routes(
            env=_mesh_key_env(name="box-1"),
            self_origin="http://box-1.local:8000",
            declared_lane_configs={
                "cortex": {
                    "model": "meta-llama/Llama-3.1-8B",
                    "runtime": "vllm",
                    "context": "262144",
                    "quant": "NVFP4",
                    "responsibilities": "reasoning",
                    "forbidden_responsibilities": "",
                }
            },
            local_capacities={"cortex": 8.0},
        )
        assert announcement.origin == "http://box-1.local:8000"
        assert "cortex" in announcement.roles

    def test_announcement_roles_have_fingerprints(self) -> None:
        _, announcement = build_mesh_routes(
            env=_mesh_key_env(name="box-2"),
            self_origin="http://box-2.local:8000",
            declared_lane_configs={
                "embed": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "runtime": "vllm",
                    "context": "8192",
                    "quant": "FP16",
                    "responsibilities": "embedding",
                    "forbidden_responsibilities": "",
                }
            },
            local_capacities={"embed": 16.0},
        )
        role = announcement.roles["embedder"]
        assert role.model == "sentence-transformers/all-MiniLM-L6-v2"


# ===========================================================================
# Finding 2: Auth header on outbound dials
# ===========================================================================


class TestAuthHeader:
    def test_post_announcement_includes_auth_header(self) -> None:
        from lobes.gateway._mesh_routes import _post_announcement as pa

        captured: dict[str, str] = {}

        def fake_req(self, method, path, body=None, headers=None):
            if headers:
                captured.update(headers)

        with patch("http.client.HTTPConnection.request", fake_req):
            pa("http://127.0.0.1:9999/mesh/announce", b'{"name":"test"}', 5.0, b"sk-test-key")
        assert "Authorization" in captured and captured["Authorization"] == "Bearer sk-test-key"

    def test_post_announcement_no_auth_without_key(self) -> None:
        from lobes.gateway._mesh_routes import _post_announcement as pa

        captured: dict[str, str] = {}

        def fake_req(self, method, path, body=None, headers=None):
            if headers:
                captured.update(headers)

        with patch("http.client.HTTPConnection.request", fake_req):
            pa("http://127.0.0.1:9999/mesh/announce", b'{"name":"test"}', 5.0, None)
        assert "Authorization" not in captured


# ===========================================================================
# Finding 4 + 8 + 9: Cadence + hung peer (parallel dials)
# ===========================================================================


class TestCadence:
    def test_cadence_uses_full_interval(self) -> None:
        """In ~N seconds with reannounce, ~N passes happen (not N*60)."""
        from lobes.gateway._mesh_routes import _heartbeat_loop

        routes = _make_routes(
            {
                "LOBES_MESH_KEY": "k",
                "LOBES_MESH_NAME": "x",
                "LOBES_MESH_HEARTBEAT_S": "1",
                "LOBES_MESH_MISSED_MAX": "3",
            }
        )

        import threading

        class FakePool:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def submit(self, fn, *a, **k):
                fut = concurrent.futures.Future()
                fut.set_result(None)
                return fut

        stop = threading.Event()
        reannounce = threading.Event()
        with patch("lobes.gateway._mesh_routes.concurrent.futures.ThreadPoolExecutor", FakePool):
            with patch.object(routes.roster, "tick") as mock_tick:
                with patch("lobes.gateway._mesh_routes._fetch_seed_roster"):
                    thread = threading.Thread(
                        target=_heartbeat_loop,
                        args=(routes, b"{}", 1.0, stop, reannounce),
                        daemon=True,
                    )
                    thread.start()
                    # Let the thread start.
                    time.sleep(0.05)
                    # Trigger re-announce events that coincide with ~3 intervals of real time.
                    for _ in range(3):
                        reannounce.set()
                        time.sleep(1.1)  # Wait for ~1 interval.
                        reannounce.clear()
                    stop.set()
                    thread.join(timeout=3)
                # Should have had at least 3 tick() calls (one per interval).
                assert mock_tick.call_count >= 2


# ===========================================================================
# Finding 6: Ledger persistence
# ===========================================================================


class TestLedgerPersistence:
    def test_approve_persists_to_disk(self) -> None:
        td = tempfile.mkdtemp()
        ledger_path = os.path.join(td, "ledger.json")
        routes1 = _make_routes(ledger_path=ledger_path)
        _ensure_approved(routes1, "persistent")
        assert routes1.roster.is_approved("persistent")
        routes2 = MeshRoutes(routes1.config, Roster(clock=_TickClock(), ledger_path=ledger_path))
        assert routes2.roster.is_approved("persistent")


# ===========================================================================
# Finding 7: RejectionLog wiring
# ===========================================================================


class TestRejectionLogWiring:
    def test_join_uses_socket_peer_source(self) -> None:
        from lobes.gateway._authlog import RejectionLog

        clock = _TickClock()
        join_log = RejectionLog(window=60.0, max_sources=256, clock=clock.__call__)
        routes = _make_routes()
        routes._join_log = join_log
        routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "ft", "origin": "http://ft.local"}).encode(),
                client_address=("192.168.1.100", 54321),
            )
        )
        with join_log._lock:
            assert "192.168.1.100" in join_log._sources


# ===========================================================================
# Finding 10: tick() per pass + injected missed_max
# ===========================================================================


class TestTickChurn:
    def test_injected_missed_max(self) -> None:
        from lobes.gateway._mesh_roster import Roster

        clock = _TickClock()
        roster = Roster(clock=clock, ledger_path=tempfile.mktemp(suffix=".json"), missed_max=1)
        routes, _ = build_mesh_routes(env=_mesh_key_env(name="test"), roster=roster, missed_max=1)
        routes.roster.announce("test-member", "http://member.local", None, now=clock.t)
        assert "test-member" in routes.roster.members()
        result = routes.roster.tick(now=clock.t + 1)
        assert result.dropped == 1
        assert "test-member" not in routes.roster.members()


# ===========================================================================
# Finding 11: Pending keyed on origin
# ===========================================================================


class TestPendingOrigin:
    def test_pending_keyed_on_origin(self) -> None:
        routes = _make_routes()
        routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "bob", "origin": "http://same.local"}).encode(),
            )
        )
        status, _, resp = routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "alice", "origin": "http://same.local"}).encode(),
            )
        )
        assert status == 400


# ===========================================================================
# Finding 12: Connection: close on 401
# ===========================================================================


class TestConnectionClose:
    def test_announce_401_has_connection_close(self) -> None:
        _, headers, _ = _make_routes().announce(
            _fake_handler("/mesh/announce", "POST", b'{"name":"x"}')
        )
        assert dict(headers).get("Connection") == "close"

    def test_roster_401_has_connection_close(self) -> None:
        _, headers, _ = _make_routes().roster_list(_fake_handler("/mesh/roster"))
        assert dict(headers).get("Connection") == "close"


# ===========================================================================
# Finding 14: Static 202 body
# ===========================================================================


class TestStaticBody:
    def test_join_body_static(self) -> None:
        routes = _make_routes()
        status, _, resp = routes.join(
            _fake_handler(
                "/mesh/join",
                "POST",
                json.dumps({"name": "test", "origin": "http://test.local"}).encode(),
            )
        )
        assert json.loads(resp)["status"] == "pending join registered"


# ===========================================================================
# Finding 15: MeshSchemaIncompatible
# ===========================================================================


class TestSchemaIncompatible:
    def test_wrong_major_400(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "bad")
        body = encode(
            Announcement(name="bad", origin="http://x.local", schema_version="2.0.0", roles={})
        )
        status, _, _ = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 400

    def test_missing_version_400(self) -> None:
        routes = _make_routes()
        _ensure_approved(routes, "nov")
        body = json.dumps({"name": "nov", "origin": "http://x.local", "roles": {}}).encode()
        status, _, _ = routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )
        assert status == 400


# ===========================================================================
# Finding 16: Scheme-aware connections
# ===========================================================================


class TestSchemeAware:
    def test_https_uses_https_connection(self) -> None:
        from lobes.gateway._mesh_routes import _post_announcement as pa
        from lobes.gateway._mesh_wire import encode as wire_encode

        a = Announcement(
            name="x", origin="http://x.local", schema_version=str(SCHEMA_MAJOR), roles={}
        )
        conn_used: str = ""

        def fake_https_cls(host, port, timeout):
            nonlocal conn_used
            conn_used = "HTTPS"
            raise RuntimeError()

        with patch("http.client.HTTPSConnection", fake_https_cls):
            try:
                pa("https://127.0.0.1/mesh/announce", wire_encode(a), 5.0, b"k")
            except RuntimeError:
                pass
        assert conn_used == "HTTPS"

    def test_http_uses_http_connection(self) -> None:
        from lobes.gateway._mesh_routes import _post_announcement as pa
        from lobes.gateway._mesh_wire import encode as wire_encode

        a = Announcement(
            name="x", origin="http://x.local", schema_version=str(SCHEMA_MAJOR), roles={}
        )
        conn_used: str = ""

        def fake_https_cls(host, port, timeout):
            nonlocal conn_used
            conn_used = "HTTPS"
            raise RuntimeError()

        def fake_http_cls(host, port, timeout):
            nonlocal conn_used
            conn_used = "HTTP"
            raise RuntimeError()

        with patch("http.client.HTTPSConnection", fake_https_cls):
            with patch("http.client.HTTPConnection", fake_http_cls):
                try:
                    pa("http://127.0.0.1/mesh/announce", wire_encode(a), 5.0, b"k")
                except RuntimeError:
                    pass
        assert conn_used == "HTTP"


# ===========================================================================
# Finding 17: start_mesh
# ===========================================================================


class TestStartMesh:
    def test_stop_event_is_init_event(self) -> None:
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        init_stop = routes._stop
        thread = start_mesh(routes, announcement)
        assert routes._stop is init_stop
        routes._stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive()

    def test_thread_assigned(self) -> None:
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        start_mesh(routes, announcement)
        assert routes._thread is not None
        routes._stop.set()
        routes._thread.join(timeout=2)

    def test_stop_plus_join(self) -> None:
        routes, announcement = build_mesh_routes(env=_mesh_key_env())
        thread = start_mesh(routes, announcement)
        routes._stop.set()
        thread.join(timeout=3)
        assert not thread.is_alive()


# ===========================================================================
# Finding 18: _check_key handles None key
# ===========================================================================


class TestCheckKeyNone:
    def test_check_key_none_key(self) -> None:
        routes, _ = build_mesh_routes(env={})
        assert routes.config.key is None
        status, _, _ = routes.announce(
            _fake_handler(
                "/mesh/announce", "POST", b'{"name":"x"}', {"Authorization": "Bearer whatever"}
            )
        )
        assert status == 401


# ===========================================================================
# Finding 19: mesh_routes declared in _Handler
# ===========================================================================


class TestHandlerClassBody:
    def test_handler_has_mesh_routes_attribute(self) -> None:
        from lobes.gateway.server import _Handler

        assert hasattr(_Handler, "mesh_routes")
        assert getattr(_Handler, "mesh_routes") is None

    def test_dispatch_unknown_mesh_returns_none(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        assert dispatch_mesh(_fake_handler("/mesh/unknown"), routes) is None


# ===========================================================================
# Finding 20: Unknown mesh route → 404/405
# ===========================================================================


class TestUnknownMeshRoute:
    def test_dispatch_valid_mesh(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        assert dispatch_mesh(_fake_handler("/mesh/detect", "GET"), routes) is not None

    def test_dispatch_invalid_method(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        assert dispatch_mesh(_fake_handler("/mesh/detect", "POST"), routes) is None

    def test_dispatch_unknown_path(self) -> None:
        routes, _ = build_mesh_routes(env=_mesh_key_env())
        assert dispatch_mesh(_fake_handler("/mesh/unknown"), routes) is None


# ===========================================================================
# Finding 5: Lapsed grant → 403 approval_expired
# ===========================================================================


class TestLapsedGrant:
    def test_announce_lapsed_grant_403(self) -> None:
        clock = _TickClock()
        clock.t = 1000000.0
        ledger_path = tempfile.mkdtemp()
        ledger_path = os.path.join(ledger_path, "ledger.json")
        routes = _make_routes(ledger_path=ledger_path, clock=clock)
        # Approve with tiny duration → expiry = 1000000 + 0.001 = 1000000.001
        routes.approve(
            _fake_handler(
                "/mesh/approve",
                "POST",
                json.dumps({"name": "lapsed", "expiry": 0.001}).encode(),
                {"Authorization": "Bearer sk-test"},
            )
        )
        clock.t = 1000000.002  # Advance past the expiry
        a = Announcement(
            name="lapsed", origin="http://lapsed.local", schema_version=str(SCHEMA_MAJOR), roles={}
        )
        status, _, _ = routes.announce(
            _fake_handler("/mesh/announce", "POST", encode(a), {"Authorization": "Bearer sk-test"})
        )
        assert status == 403
        assert (
            json.loads(b'{"error":{"type":"approval_expired"}}')["error"]["type"]
            == "approval_expired"
        )


# ===========================================================================
# Finding 13: Thread-safety stress test
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_announce_and_tick(self) -> None:
        clock = _TickClock()
        routes = _make_routes(_mesh_key_env(name="stress"), clock=clock)
        errors: list[Exception] = []

        def hammer() -> None:
            for i in range(50):
                try:
                    _ensure_approved(routes, f"stress-{i % 5}")
                    body = encode(
                        Announcement(
                            name=f"stress-{i % 5}",
                            origin=f"http://s{i}.local",
                            schema_version=str(SCHEMA_MAJOR),
                            roles={},
                        )
                    )
                    routes.announce(
                        _fake_handler(
                            "/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"}
                        )
                    )
                    routes.roster_list(
                        _fake_handler("/mesh/roster", headers={"Authorization": "Bearer sk-test"})
                    )
                    routes.roster.tick()
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=hammer) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestRequireSelfOrigin:
    """The announced origin is GATEWAY_SELF_ORIGIN or nothing — never a name."""

    def test_empty_self_origin_refuses_to_start(self):
        from lobes.gateway._mesh_config import MeshConfigError
        from lobes.gateway._mesh_routes import require_self_origin

        with pytest.raises(MeshConfigError, match="GATEWAY_SELF_ORIGIN"):
            require_self_origin("")

    def test_typed_self_origin_passes_through(self):
        from lobes.gateway._mesh_routes import require_self_origin

        assert require_self_origin("http://spark.tail:8001/") == "http://spark.tail:8001/"
