"""Item C (t9): verification failures are no longer silent.

Before this task, ``verify_members``'s per-member probe and the heartbeat
loop's verification-pass wrapper swallowed every failure with a bare
``except Exception: pass`` — an unreachable/misbehaving mesh peer left no
trace anywhere: no log line, no roster field, nothing a ``lobes mesh
status`` reader could see.  This module proves:

* a failing probe records a short ``reason`` string on the member
  (``MemberInfo.unverified_reason`` / :func:`build_snapshot`'s
  ``unverified_reasons`` param);
* ``GET /mesh/roster`` exposes that reason as an ADDITIVE
  ``unverified_reason`` field (``None`` when the member verified cleanly or
  was never probed);
* one collapsed line is written to stderr per failing origin, via the same
  :class:`~lobes.gateway._authlog.RejectionLog` pattern ``/mesh/join``
  already uses — repeated failures from the same origin collapse into one
  line per window rather than flooding the log.
"""

from __future__ import annotations

import json

import pytest

from lobes.gateway._authlog import RejectionLog
from lobes.gateway._mesh_config import build_mesh_config
from lobes.gateway._mesh_roster import Roster
from lobes.gateway._mesh_routes import MeshRoutes, verify_members
from lobes.gateway._mesh_routing import build_snapshot
from tests.test_mesh_routes import _fake_handler, _mesh_key_env
from tests.test_mesh_routing_wiring import _ann


def _routes_with_one_member(join_key: str, origin: str) -> MeshRoutes:
    cfg = build_mesh_config(_mesh_key_env(key=join_key))
    roster = Roster()
    routes = MeshRoutes(cfg, roster, _verify_log=RejectionLog())
    routes.roster.announce("peer-a", origin, 1.0)
    routes._announcements[origin] = _ann("peer-a", origin)
    return routes


class TestUnreachableOriginRecordsAReason:
    def test_reason_recorded_on_the_member(self) -> None:
        # A bogus origin (no listener) fails fast and deterministically —
        # exactly the "peer unreachable" case the bare except was hiding.
        origin = "http://127.0.0.1:1"
        routes = _routes_with_one_member("sk-test", origin)
        holder = type("Holder", (), {"replace": lambda s, v: setattr(s, "_v", v)})()
        verify_members(routes, holder, join_key="sk-test", timeout=0.2)
        view = holder._v
        member = next(m for m in view.snapshot.members if m.origin == origin)
        assert member.unverified_reason is not None
        assert member.verified_roles == ()
        # Boot window (t1): a probe that RAN and failed marks the member
        # probed — it is unverified for real, not merely not-yet-probed.
        assert member.probed is True

    def test_roster_route_exposes_unverified_reason(self) -> None:
        origin = "http://127.0.0.1:1"
        routes = _routes_with_one_member("sk-test", origin)
        holder = type(
            "Holder",
            (),
            {
                "replace": lambda s, v: setattr(s, "_v", v),
                "current": lambda s: getattr(s, "_v", None),
            },
        )()
        routes._holder = holder
        verify_members(routes, holder, join_key="sk-test", timeout=0.2)

        status, _headers, body = routes.roster_list(
            _fake_handler("/mesh/roster", headers={"Authorization": "Bearer sk-test"})
        )
        assert status == 200
        payload = json.loads(body)
        member = next(m for m in payload["members"] if m["origin"] == origin)
        assert member["unverified_reason"] is not None
        assert member["verified"] is False

    def test_collapsed_stderr_line_written_once_per_window(self, capsys) -> None:
        origin = "http://127.0.0.1:1"
        routes = _routes_with_one_member("sk-test", origin)
        holder = type("Holder", (), {"replace": lambda s, v: None})()
        verify_members(routes, holder, join_key="sk-test", timeout=0.2)
        verify_members(routes, holder, join_key="sk-test", timeout=0.2)

        err = capsys.readouterr().err
        lines = [ln for ln in err.splitlines() if "mesh verify" in ln]
        # The second call lands inside the same collapse window as the first
        # (RejectionLog's default 60s), so it must NOT print a second line.
        assert len(lines) == 1
        assert origin in lines[0]


class TestCleanVerificationRecordsNoReason:
    def test_no_reason_when_no_announcement_exists(self) -> None:
        # A member with no stored announcement is never probed at all —
        # verify_members's own early-return path — so it must carry no
        # fabricated reason.
        cfg = build_mesh_config(_mesh_key_env(key="sk-test"))
        roster = Roster()
        routes = MeshRoutes(cfg, roster, _verify_log=RejectionLog())
        routes.roster.announce("lonely", "http://127.0.0.1:2", 1.0)
        holder = type("Holder", (), {"replace": lambda s, v: setattr(s, "_v", v)})()
        verify_members(routes, holder, join_key="sk-test", timeout=0.2)
        member = next(m for m in holder._v.snapshot.members if m.name == "lonely")
        assert member.unverified_reason is None
        # New meaning (t1): never probed is its own state. The member reads
        # probed False here and carries NO reason at the model level — the
        # "not_yet_probed" string belongs to the /mesh/roster presentation.
        assert member.probed is False
        assert member.ready_roles == ()


class TestDefaultVerifyLogIsNone:
    def test_no_verify_log_means_no_crash_and_no_stderr(self, capsys) -> None:
        """A MeshRoutes built without an explicit _verify_log (the pre-t9
        default, and every existing test/factory call site) must behave
        exactly as before: no logging, no exception."""
        origin = "http://127.0.0.1:1"
        cfg = build_mesh_config(_mesh_key_env(key="sk-test"))
        roster = Roster()
        routes = MeshRoutes(cfg, roster)  # no _verify_log
        routes.roster.announce("peer-a", origin, 1.0)
        routes._announcements[origin] = _ann("peer-a", origin)
        holder = type("Holder", (), {"replace": lambda s, v: setattr(s, "_v", v)})()
        verify_members(routes, holder, join_key="sk-test", timeout=0.2)
        assert capsys.readouterr().err == ""


@pytest.mark.parametrize("unverified_reasons", [None, {}])
def test_build_snapshot_tolerates_absent_reasons(unverified_reasons) -> None:
    """build_snapshot's new keyword is optional and back-compat: every
    pre-t9 caller that never passes it gets unverified_reason=None on every
    member, unchanged."""
    roster = Roster()
    roster.announce("a", "http://a.example:8000", 1.0)
    snap = build_snapshot(roster, unverified_reasons=unverified_reasons)
    assert all(m.unverified_reason is None for m in snap.members)
    # And with no probe data of any kind, every member is never-probed.
    assert all(m.probed is False for m in snap.members)
    assert all(m.ready_roles == () for m in snap.members)
