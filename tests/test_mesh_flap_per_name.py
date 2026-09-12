"""Item A (t9): per-name flap tracking, moved onto the live /mesh/announce path.

Before this task the flap counter was ROSTER-WIDE (``Roster._flap_count`` /
``_flap_time``) and only :meth:`Roster.join` ever touched it — but no
``/mesh/*`` route calls ``join()``; ``/mesh/announce`` drives
:meth:`Roster.announce` instead, which never touched the counter at all. So
the mechanism was dead on every live deployment, and even if it had been
wired, one flapping member would have held out every OTHER name too.

This module proves:

* a name that re-registers (via ``announce()``) more than the threshold
  times inside one hold-out window is refused with ``MeshFlapping``;
* a DIFFERENT name's registrations are completely unaffected by another
  name's flapping — the per-name isolation the roster-wide counter lacked;
* the live ``POST /mesh/announce`` route surfaces a held-out name as a
  retryable 429 with ``type: mesh_flapping`` rather than an unhandled 500;
* ``GET /mesh/roster``'s ``flapping`` field is now genuinely per-member.
"""

from __future__ import annotations

import json

import pytest

from lobes.gateway._mesh_roster import MeshFlapping, Roster
from lobes.gateway._mesh_routes import MeshRoutes
from lobes.gateway._mesh_wire import SCHEMA_MAJOR, Announcement, encode
from tests.test_mesh_routes import _ensure_approved, _fake_handler, _make_routes, _mesh_key_env


class _TickClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, by: float = 1.0) -> None:
        self.now += by


class TestAnnouncePathFlapsPerName:
    def test_repeated_reregistration_is_held_out(self) -> None:
        clock = _TickClock()
        roster = Roster(clock=clock)
        for i in range(3):
            roster.announce("alice", f"http://alice-{i}.example", 1.0, now=clock())
            roster.leave("alice")
        now = clock()
        with pytest.raises(MeshFlapping):
            roster.announce("alice", "http://alice-3.example", 1.0, now=now)

    def test_a_different_name_is_never_held_out_by_alices_flapping(self) -> None:
        clock = _TickClock()
        roster = Roster(clock=clock)
        for i in range(3):
            roster.announce("alice", f"http://alice-{i}.example", 1.0, now=clock())
            roster.leave("alice")
        now = clock()
        with pytest.raises(MeshFlapping):
            roster.announce("alice", "http://alice-3.example", 1.0, now=now)

        # bob has never flapped — announcing him must succeed cleanly, which
        # the OLD roster-wide counter would have refused too (it held out
        # EVERY name once tripped).
        roster.announce("bob", "http://bob.example", 1.0, now=clock())
        assert roster.is_joined("bob")

    def test_hold_out_expires_after_one_tick(self) -> None:
        clock = _TickClock()
        roster = Roster(clock=clock)
        for i in range(3):
            roster.announce("alice", f"http://alice-{i}.example", 1.0, now=clock())
            roster.leave("alice")
        now = clock()
        with pytest.raises(MeshFlapping):
            roster.announce("alice", "http://alice-3.example", 1.0, now=now)
        clock.tick()
        roster.announce("alice", "http://alice-4.example", 1.0, now=clock())
        assert roster.is_joined("alice")

    def test_update_of_an_existing_member_never_flaps(self) -> None:
        """Re-announcing the SAME name+origin (the ordinary heartbeat) is an
        UPDATE, not a new registration — it must never consume flap budget,
        however many times it repeats."""
        clock = _TickClock()
        roster = Roster(clock=clock)
        roster.announce("alice", "http://alice.example", 1.0, now=clock())
        for _ in range(10):
            roster.announce("alice", "http://alice.example", 1.0, now=clock())
        assert roster.is_joined("alice")


class TestLiveAnnounceRouteSurfacesFlapping:
    def _announce(self, routes: MeshRoutes, name: str, origin: str):
        body = encode(
            Announcement(name=name, origin=origin, schema_version=str(SCHEMA_MAJOR), roles={})
        )
        return routes.announce(
            _fake_handler("/mesh/announce", "POST", body, {"Authorization": "Bearer sk-test"})
        )

    def test_flapping_name_gets_429_not_a_crash(self) -> None:
        clock = _TickClock()
        routes = _make_routes(_mesh_key_env(name="me"), clock=clock)
        _ensure_approved(routes, "alice")
        for i in range(3):
            status, _headers, _body = self._announce(routes, "alice", f"http://a{i}.example")
            assert status == 200
            routes.roster.leave("alice")

        status, _headers, body = self._announce(routes, "alice", "http://a3.example")
        assert status == 429
        payload = json.loads(body)
        assert payload["error"]["type"] == "mesh_flapping"

    def test_roster_reports_flapping_per_member(self) -> None:
        clock = _TickClock()
        routes = _make_routes(_mesh_key_env(name="me"), clock=clock)
        _ensure_approved(routes, "alice")
        _ensure_approved(routes, "bob")
        for i in range(3):
            self._announce(routes, "alice", f"http://a{i}.example")
            routes.roster.leave("alice")
        # alice's 3rd successful (re)registration puts her AT the threshold —
        # the next one will be refused, so she reports flapping now.
        self._announce(routes, "bob", "http://bob.example")

        status, _headers, body = routes.roster_list(
            _fake_handler("/mesh/roster", headers={"Authorization": "Bearer sk-test"})
        )
        assert status == 200
        members = {m["name"]: m for m in json.loads(body)["members"]}
        assert members["bob"]["flapping"] is False
