"""Tests for lobes.rollout._observe (fixtures only; the opener is injected)."""

from __future__ import annotations

import json

from lobes.rollout import _observe as ob


class FakeOpener:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append((req.get_method(), req.full_url, dict(req.header_items())))
        path = req.full_url.split("://", 1)[1].split("/", 1)[1]
        body = self.routes["/" + path]
        if isinstance(body, Exception):
            raise body
        return _Resp(json.dumps(body).encode())


class _Resp:
    status = 200

    def __init__(self, raw):
        self.raw = raw

    def read(self):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


ROUTES = {
    "/health": {"status": "ok", "service": "model-gear-gateway", "version": "0.94.0"},
    "/capabilities": {
        "roles": {
            "cortex": {"feasible": True, "ready": True, "model": "m"},
            "hand": {"feasible": True, "ready": True},
        }
    },
    "/mesh/roster": {"members": [{"name": "spark", "origin": "http://x", "verified": True}]},
}


def test_observe_reads_all_and_only_gets():
    op = FakeOpener(ROUTES)
    obs = ob.observe_member("http://h:1/", gateway_key="gk", mesh_key="mk", opener=op)
    assert obs.version == "0.94.0"
    assert obs.reachable
    assert set(obs.capabilities) == {"cortex", "hand"}
    assert obs.roster[0]["name"] == "spark"
    assert {c[0] for c in op.calls} == {"GET"}
    assert len(op.calls) == 3
    auth = {c[1].rsplit("/", 1)[1]: c[2].get("Authorization") for c in op.calls}
    assert auth["health"] is None
    assert auth["capabilities"] == "Bearer gk"
    assert auth["roster"] == "Bearer mk"


def test_unreachable_member():
    op = FakeOpener({p: OSError("down") for p in ROUTES})
    obs = ob.observe_member("http://h:1", opener=op)
    assert not obs.reachable and obs.version is None and obs.capabilities == {}


def test_flat_capabilities_shape():
    routes = dict(ROUTES, **{"/capabilities": {"cortex": {"feasible": True, "ready": True}}})
    obs = ob.observe_member("http://h", opener=FakeOpener(routes))
    assert "cortex" in obs.capabilities


def _obs(caps):
    return ob.MemberObservation(origin="o", version="v", capabilities=caps)


def test_diff_regressions_and_new():
    before = _obs(
        {
            "a": {"feasible": True, "ready": True},
            "b": {"feasible": True, "ready": True},
            "c": {"feasible": True, "ready": True},
            "d": {"feasible": True, "ready": False},
            "e": {"feasible": True, "ready": True},
        }
    )
    after = _obs(
        {
            "a": {"feasible": True, "ready": True},
            "b": {"feasible": True, "ready": False},
            "c": {"feasible": False, "ready": True},
            "d": {"feasible": True, "ready": True},
            "n": {"feasible": True, "ready": False},
        }
    )
    d = ob.diff(before, after)
    assert d.regressions == ["b", "c", "e"]
    assert d.new_roles == ["n"]
    assert d.improved == ["d"]
    assert not d.ok
    assert ob.diff(before, before).ok


def test_observe_keeps_the_health_body_without_extra_requests():
    health = dict(
        ROUTES["/health"],
        memory={"total_bytes": 10, "available_bytes": 4},
    )
    op = FakeOpener(dict(ROUTES, **{"/health": health}))
    obs = ob.observe_member("http://h:1", opener=op)
    assert obs.health == health
    assert obs.version == "0.94.0"  # extra /health keys are tolerated
    assert len(op.calls) == 3


def test_unreachable_member_has_empty_health():
    op = FakeOpener({p: OSError("down") for p in ROUTES})
    assert ob.observe_member("http://h:1", opener=op).health == {}
