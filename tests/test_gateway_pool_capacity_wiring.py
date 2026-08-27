"""The gateway's END of capacity-relative pool routing (issue #199, task t5).

`_replicas.py` (t4) knows how to INGEST a capacity and how to count a
dispatch; nothing there ever publishes this box's own number or calls
``begin_dispatch``. That wiring is ``lobes/gateway/server.py``'s job, and it
is what this module tests:

(1) **publication** — ``fleet_status_payload`` writes this box's declared
    capacity onto each hosted backend's ``/status`` entry under
    :data:`lobes.gateway._replicas.PEER_CAPACITY_KEY`, so a peer learns it
    from the probe it ALREADY makes (no new probe, no new endpoint). The key
    is additive: a box that declares nothing publishes nothing and an older
    lobes still parses the body.

(2) **in-flight accounting** — every replica the pool dispatches to is
    counted BEFORE the dial and released on completion, on the error and
    retry paths included. A leaked counter is the one failure mode that
    cannot self-heal: it makes a healthy box look permanently full.

(3) **burst distribution** — the point of (2). N concurrent arrivals against
    two idle replicas distribute across both, without waiting for the 5 s
    probe refresh that would otherwise show both idle to every one of them.

(4) **observability** — the capacity and utilisation a placement used ride
    back on the answer, so a capacity-driven choice is explainable from a
    trace alone rather than from gateway logs. Deliberately a NEW header
    rather than a new ``X-Lobes-Route-Reason`` value: that vocabulary is
    closed (server.py's header contract) and t3 already redefined
    ``peer-less-loaded`` once. It appears only on POOLED answers, so h1's
    byte-identity guarantee for a no-``*_PEER_ORIGINS`` deployment holds.

Everything here runs offline: the probe opener, the upstream opener and the
clock are all injected, exactly as tests/test_gateway_pool.py and
tests/test_gateway_replicas.py do. No socket is opened and no live gateway
is touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from lobes.gateway import _replicas as R
from lobes.gateway import server as S
from lobes.gateway._config import build_config

_CORTEX_ID = "unsloth/Qwen3.8-27B-NVFP4"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_LOCAL_URL = "http://vllm-primary:8000"
_PEER_ORIGIN = "http://thor.local:8001"
_SELF_ORIGIN = "http://spark.local:8001"


# --- env / table builders ---------------------------------------------------


def _env(**over) -> dict[str, str]:
    env = {
        "PRIMARY_URL": _LOCAL_URL,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
    }
    env.update(over)
    return env


def _pool_env(**over) -> dict[str, str]:
    return _env(
        PRIMARY_PEER_ORIGINS=_PEER_ORIGIN,
        PRIMARY_PEER_API_KEYS="",
        GATEWAY_SELF_ORIGIN=_SELF_ORIGIN,
        **over,
    )


def _build(env):
    table, cfg = build_config(env)
    return table, cfg, S.peer_specs_from_table(table, env)


# --- fakes ------------------------------------------------------------------


class _FakeUpstream:
    def __init__(self, status=200, body=b'{"ok":1}'):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body
        self.closed = False

    def read_all(self):
        return self._body

    def read(self, _n):
        data, self._body = self._body, b""
        return data

    def close(self):
        self.closed = True


def _opener(outcome=200):
    calls: list = []

    def opener(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        calls.append(SimpleNamespace(backend=backend, path=path, headers=list(headers)))
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome(len(calls))
        return _FakeUpstream(outcome)

    return opener, calls


_DECLARED = {
    "runtime": "vllm",
    "quantization": "compressed-tensors",
    "kv_cache_dtype": "fp8",
    "reasoning_parser": "qwen3",
    "tool_parser": "qwen3_coder",
    "speculative_config": "none",
}


def _fingerprint() -> dict:
    fp = {"served_id": _CORTEX_ID, "max_model_len": 262144}
    fp.update(_DECLARED)
    return fp


def _caps() -> dict:
    return {
        "object": "lobes.capabilities",
        "roles": {
            "cortex": {
                "role": "cortex",
                "model": _CORTEX_ID,
                "context": 262144,
                "ready": True,
                "fingerprint": _fingerprint(),
            }
        },
    }


def _state(origin, *, local=False, running=0, waiting=0, weight=1.0, **kw) -> R.ReplicaState:
    return R.ReplicaState(
        origin=origin,
        local=local,
        ready=kw.pop("ready", True),
        busy=kw.pop("busy", False),
        health="ok",
        running=running,
        waiting=waiting,
        fingerprint=None,
        compatible=kw.pop("compatible", True),
        reason="",
        last_seen=1.0,
        weight=weight,
        **kw,
    )


def _snapshot(*states):
    """A fixed replica snapshot provider — the injected dispatch seam."""

    def provider(_backend_name):
        return tuple(states)

    return provider


def _body(model: str) -> bytes:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()


def _header(resp, name: str):
    lowered = name.lower()
    return next((v for k, v in resp.headers if k.lower() == lowered), None)


def _post(table, cfg, specs, *, snapshot=None, counter=None, opener=None, headers=()):
    calls: list = []
    if opener is None:
        opener, calls = _opener()
    resp = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        list(headers),
        _body(_CORTEX_ID),
        opener,
        peer_specs=specs,
        replica_snapshot=snapshot,
        dispatch_counter=counter,
    )
    return resp, calls


# --- a recording dispatch counter ------------------------------------------


class _RecordingCounter:
    """A stand-in for the real ``ReplicaCache``-backed counter.

    Records every ``(backend, origin)`` begun and every release, so a test can
    assert BOTH that a dispatch was counted and that nothing stayed counted.
    """

    def __init__(self) -> None:
        self.begun: list[tuple[str, str]] = []
        self.outstanding: list[tuple[str, str]] = []

    def __call__(self, backend: str, origin: str):
        entry = (backend, origin)
        self.begun.append(entry)
        self.outstanding.append(entry)

        def release() -> None:
            if entry in self.outstanding:
                self.outstanding.remove(entry)

        return release


# ===========================================================================
# (1) publication: this box's capacity on its own /status body
# ===========================================================================


def _probe_stub(**per_backend):
    def probe(base_url, timeout=0.0):
        return per_backend.get(base_url, {"health": "ok", "metrics": {"running": 0, "waiting": 0}})

    return probe


def _status(env, **kw):
    table, cfg = build_config(env)
    return S.fleet_status_payload(table, cfg, probe=_probe_stub(), **kw)


def _backend_row(payload, name):
    return next(b for b in payload["backends"] if b["name"] == name)


def test_status_publishes_this_boxs_declared_capacity_per_backend() -> None:
    payload = _status(_pool_env(PRIMARY_MAX_ACTIVE="6", MULTIMODAL_MAX_ACTIVE="3"))
    assert _backend_row(payload, "primary")[R.PEER_CAPACITY_KEY] == 6.0
    assert _backend_row(payload, "multimodal")[R.PEER_CAPACITY_KEY] == 3.0


def test_status_omits_capacity_for_an_undeclared_backend() -> None:
    """Additive, never fabricated: an older lobes still parses the body and
    `_replicas.py` reads an absent capacity as uncalibrated, not as zero."""
    payload = _status(_pool_env(PRIMARY_MAX_ACTIVE="6"))
    assert R.PEER_CAPACITY_KEY not in _backend_row(payload, "multimodal")


def test_status_with_no_capacity_declared_anywhere_is_unchanged() -> None:
    before = _status(_env())
    for row in before["backends"]:
        assert R.PEER_CAPACITY_KEY not in row


def test_status_never_publishes_a_capacity_for_a_role_this_box_does_not_host() -> None:
    """A dropped lane has no local replica; publishing a capacity for it would
    advertise room on a box that cannot serve the role at all."""
    payload = _status(
        _pool_env(MULTIMODAL_FEASIBLE="false", MULTIMODAL_MAX_ACTIVE="3", PRIMARY_MAX_ACTIVE="6")
    )
    assert R.PEER_CAPACITY_KEY not in _backend_row(payload, "multimodal")
    assert _backend_row(payload, "primary")[R.PEER_CAPACITY_KEY] == 6.0


def test_published_capacity_is_the_one_a_peers_replica_cache_ingests() -> None:
    """The producer/consumer round trip, offline: this box's own /status body
    is fed straight into a peer's ReplicaCache and comes back as weight."""
    payload = _status(_pool_env(PRIMARY_MAX_ACTIVE="8"))
    routes = {
        _LOCAL_URL + "/v1/models": (200, b'{"object":"list","data":[]}'),
        _LOCAL_URL + "/metrics": (200, b""),
        _PEER_ORIGIN + "/status": (200, json.dumps(payload).encode()),
        _PEER_ORIGIN + "/capabilities": (200, json.dumps(_caps()).encode()),
    }

    def opener(url, _timeout, _key):
        return routes[url]

    cache = R.ReplicaCache(
        "cortex",
        R.LocalLane(base_url=_LOCAL_URL, served_name=_CORTEX_ID, declared=dict(_DECLARED)),
        (R.PeerReplica(_PEER_ORIGIN, ""),),
        backend_name="primary",
        urlopen=opener,
        start=False,
    )
    cache.refresh()
    peer = next(s for s in cache.current() if not s.local)
    assert peer.capacity == 8.0
    assert peer.weight == 8.0


# ===========================================================================
# (2) in-flight accounting at dispatch
# ===========================================================================


def test_local_dispatch_is_counted_and_released() -> None:
    table, cfg, specs = _build(_pool_env())
    counter = _RecordingCounter()
    snapshot = _snapshot(_state(_LOCAL_URL, local=True), _state(_PEER_ORIGIN, running=5))
    resp, _ = _post(table, cfg, specs, snapshot=snapshot, counter=counter)
    assert counter.begun == [("primary", _LOCAL_URL)]
    # A relayed answer's completion escapes the dispatch block, so the release
    # rides on the response and fires when the relay finishes.
    assert counter.outstanding == [("primary", _LOCAL_URL)]
    resp.release()
    assert counter.outstanding == []
    resp.release()  # idempotent
    assert counter.outstanding == []


def test_forwarded_dispatch_is_counted_against_the_peer() -> None:
    table, cfg, specs = _build(_pool_env())
    counter = _RecordingCounter()
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=5),
        _state(_PEER_ORIGIN),
    )
    resp, _ = _post(table, cfg, specs, snapshot=snapshot, counter=counter)
    assert counter.begun == [("primary", _PEER_ORIGIN)]
    resp.release()
    assert counter.outstanding == []


def test_a_retry_releases_the_failed_replica_before_dispatching_the_next() -> None:
    """The classic leak site: A refuses pre-dispatch, B serves. A must not stay
    counted, or the box that merely failed once looks loaded forever."""
    table, cfg, specs = _build(_pool_env())
    counter = _RecordingCounter()
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=5),
        _state(_PEER_ORIGIN),
    )

    def outcome(n):
        if n == 1:  # the peer forward fails pre-dispatch
            raise S.UpstreamError("refused")
        return _FakeUpstream(200)

    opener, _calls = _opener(outcome)
    resp, _ = _post(table, cfg, specs, snapshot=snapshot, counter=counter, opener=opener)
    assert counter.begun == [("primary", _PEER_ORIGIN), ("primary", _LOCAL_URL)]
    # The failed peer was released the moment it failed — only the serving
    # replica is still counted.
    assert counter.outstanding == [("primary", _LOCAL_URL)]
    resp.release()
    assert counter.outstanding == []


def test_an_exhausted_pool_leaves_nothing_counted() -> None:
    """Every replica fails pre-dispatch → the 503. No release call site is
    reached by a response, so the failure path itself must release."""
    table, cfg, specs = _build(_pool_env())
    counter = _RecordingCounter()
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True),
        _state(_PEER_ORIGIN),
    )
    opener, _calls = _opener(S.UpstreamError("refused"))
    resp, _ = _post(table, cfg, specs, snapshot=snapshot, counter=counter, opener=opener)
    assert resp.status == 503
    assert len(counter.begun) == 2
    assert counter.outstanding == []


def test_an_error_response_from_a_replica_releases_immediately() -> None:
    """A peer's own 4xx is an ANSWER, relayed with no upstream body to stream
    (the relay is buffered) — either way the counter must not survive it."""
    table, cfg, specs = _build(_pool_env())
    counter = _RecordingCounter()
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=5),
        _state(_PEER_ORIGIN),
    )
    opener, _calls = _opener(429)
    resp, _ = _post(table, cfg, specs, snapshot=snapshot, counter=counter, opener=opener)
    assert resp.status == 429
    resp.release()
    assert counter.outstanding == []


def test_an_unpooled_request_never_touches_the_counter() -> None:
    table, cfg, specs = _build(_env())
    counter = _RecordingCounter()
    resp, _ = _post(table, cfg, specs, snapshot=None, counter=counter)
    assert counter.begun == []
    assert resp.release() is None


def test_a_marked_arrival_is_not_counted_and_leaves_nothing_outstanding() -> None:
    """A single-hop arrival is served locally WITHOUT a placement: selection
    returns ``sole-ready`` with no origin, so the pool dispatches nothing and
    the pre-pool local dial takes it. Counting it would mean inventing a
    dispatch the pool never made — and the boundary that matters for the leak
    is the one asserted here: nothing is left outstanding either way."""
    table, cfg, specs = _build(_pool_env())
    counter = _RecordingCounter()
    snapshot = _snapshot(_state(_LOCAL_URL, local=True), _state(_PEER_ORIGIN))
    resp, _ = _post(
        table,
        cfg,
        specs,
        snapshot=snapshot,
        counter=counter,
        headers=[(S.PROXIED_HEADER, "primary")],
    )
    assert resp.status == 200
    assert counter.begun == []
    resp.release()
    assert counter.outstanding == []


# ===========================================================================
# (3) a burst distributes without waiting for a probe refresh
# ===========================================================================


def _live_cache_routes(*, capacity=2):
    status = {
        "object": "lobes.fleet_status",
        "default_model": _CORTEX_ID,
        "busy": {"running": 0, "waiting": 0},
        "backends": [
            {
                "name": "primary",
                "task": "generate",
                "served_name": _CORTEX_ID,
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                R.PEER_CAPACITY_KEY: capacity,
            }
        ],
        "endpoints": [],
    }
    caps = _caps()
    models = {"object": "list", "data": [{"id": _CORTEX_ID, "max_model_len": 262144}]}
    return {
        _LOCAL_URL + "/v1/models": (200, json.dumps(models).encode()),
        _LOCAL_URL + "/metrics": (200, b""),
        _PEER_ORIGIN + "/status": (200, json.dumps(status).encode()),
        _PEER_ORIGIN + "/capabilities": (200, json.dumps(caps).encode()),
    }


def _live_cache(capacity=2):
    routes = _live_cache_routes(capacity=capacity)

    def opener(url, _timeout, _key):
        return routes[url]

    cache = R.ReplicaCache(
        "cortex",
        R.LocalLane(
            base_url=_LOCAL_URL,
            served_name=_CORTEX_ID,
            declared=dict(_DECLARED),
            weight=capacity,
        ),
        (R.PeerReplica(_PEER_ORIGIN, ""),),
        backend_name="primary",
        urlopen=opener,
        start=False,
    )
    cache.refresh()
    return cache


def test_a_burst_against_two_idle_replicas_distributes_across_both() -> None:
    """Four arrivals, ONE probe pass, no refresh in between. Without the
    in-flight fold every one of them reads the same idle snapshot and lands on
    the same replica."""
    table, cfg, specs = _build(_pool_env())
    cache = _live_cache(capacity=2)
    caches = {"primary": cache}
    snapshot = S.replica_snapshot_provider(caches)
    counter = S.dispatch_counter(caches)
    placed = []
    held = []
    for _ in range(4):
        resp, _calls = _post(table, cfg, specs, snapshot=snapshot, counter=counter)
        held.append(resp)
        placed.append(_header(resp, S.PROXIED_BY_HEADER) or "local")
    assert "local" in placed, placed
    assert _PEER_ORIGIN in placed, placed
    for resp in held:
        resp.release()
    assert cache.in_flight(_LOCAL_URL) == 0
    assert cache.in_flight(_PEER_ORIGIN) == 0


def test_dispatch_counter_is_none_without_a_pool() -> None:
    assert S.dispatch_counter({}) is None
    assert S.dispatch_counter(None) is None


def test_dispatch_counter_for_an_unknown_backend_is_a_harmless_no_op() -> None:
    counter = S.dispatch_counter({"primary": _live_cache()})
    assert counter is not None
    release = counter("nosuchbackend", _LOCAL_URL)
    assert release() is None


def test_build_replica_caches_seeds_the_local_lane_from_declared_capacity() -> None:
    table, _cfg = build_config(_pool_env(PRIMARY_MAX_ACTIVE="8"))
    caches = S.build_replica_caches(
        table,
        urlopen=lambda url, _t, _k: (200, b"{}"),
        start=False,
        capacities={"primary": 8.0},
    )
    local = next(s for s in caches["primary"].current() if s.local)
    assert local.weight == 8.0


def test_build_replica_caches_honours_the_capacity_kill_switch_for_peers() -> None:
    table, _cfg = build_config(_pool_env())
    caches = S.build_replica_caches(
        table,
        urlopen=lambda url, _t, _k: (200, b"{}"),
        start=False,
        capacities={"primary": 8.0},
        capacity_kill_switch=True,
    )
    local = next(s for s in caches["primary"].current() if s.local)
    assert local.weight == 1.0


# ===========================================================================
# (4) the placement's capacity + utilisation are observable after the fact
# ===========================================================================


def test_a_pooled_local_answer_reports_the_capacity_and_utilisation_used() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=1, weight=8.0),
        _state(_PEER_ORIGIN, running=6, weight=8.0),
    )
    resp, _ = _post(table, cfg, specs, snapshot=snapshot)
    load = _header(resp, S.ROUTE_LOAD_HEADER)
    assert load is not None
    fields = dict(part.strip().split("=", 1) for part in load.split(";"))
    assert fields == {
        "active": "1",
        "capacity": "8",
        "utilisation": "0.125",
        "calibrated": "true",
    }


def test_a_pooled_forwarded_answer_reports_the_peers_numbers() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=6, weight=8.0),
        _state(_PEER_ORIGIN, running=2, weight=4.0),
    )
    resp, _ = _post(table, cfg, specs, snapshot=snapshot)
    assert _header(resp, S.PROXIED_BY_HEADER) == _PEER_ORIGIN
    fields = dict(
        part.strip().split("=", 1) for part in _header(resp, S.ROUTE_LOAD_HEADER).split(";")
    )
    assert fields["capacity"] == "4"
    assert fields["active"] == "2"
    assert fields["utilisation"] == "0.5"


def test_an_uncalibrated_placement_says_so_rather_than_claiming_one_slot() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=1),
        _state(_PEER_ORIGIN, running=6),
    )
    resp, _ = _post(table, cfg, specs, snapshot=snapshot)
    fields = dict(
        part.strip().split("=", 1) for part in _header(resp, S.ROUTE_LOAD_HEADER).split(";")
    )
    assert fields["calibrated"] == "false"


def test_an_unpooled_answer_carries_no_route_load_header() -> None:
    """h1: with no *_PEER_ORIGINS declared the response is byte-identical to
    the pre-pool release, headers included."""
    table, cfg, specs = _build(_env())
    resp, _ = _post(table, cfg, specs, snapshot=None)
    assert _header(resp, S.ROUTE_LOAD_HEADER) is None


def test_a_peers_route_load_header_is_not_relayed_alongside_ours() -> None:
    """A pooled peer stamps its OWN load marker; relayed verbatim a caller
    would see two. The forwarder's verdict is the honest one."""
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=6, weight=8.0),
        _state(_PEER_ORIGIN, running=2, weight=4.0),
    )

    def outcome(_n):
        up = _FakeUpstream(200)
        up.headers = [
            ("Content-Type", "application/json"),
            (S.ROUTE_LOAD_HEADER, "active=99; capacity=99; utilisation=1; calibrated=true"),
        ]
        return up

    opener, _calls = _opener(outcome)
    resp, _ = _post(table, cfg, specs, snapshot=snapshot, opener=opener)
    loads = [v for k, v in resp.headers if k.lower() == S.ROUTE_LOAD_HEADER.lower()]
    assert loads == ["active=2; capacity=4; utilisation=0.5; calibrated=true"]
