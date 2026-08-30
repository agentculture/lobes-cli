"""Peer-only replica pools — placing a role this box does NOT host.

Where #199 let a box that HOSTS a role forward some of that role's requests to
an equally-good peer, this covers the mirror case: a box that hosts the role
NOWHERE, spreading it across the replicas that do. The motivating measurement
is on the record — on 2026-08-30 the Jetson AGX Orin answered every
``model=cortex`` request with ``X-Lobes-Proxied-By`` pinned to one of two
equally-good peers, because the singular proxy branch consumed the request
before any placement could happen.

Six contracts, each traceable to a confirmed frame claim in
``docs/specs/2026-08-30-peer-only-replica-pools.md``:

(1) **Peers agree with each other** (c3/h2, decision c16). With no local lane
    there is nothing to compare a peer against, so the first READY peer in
    DECLARATION order supplies the reference fingerprint and the rest are
    compared to it. An unknown still never pools silently.

(2) **The cache exists at all** (c2/h1). ``build_replica_caches`` used to skip
    every infeasible backend outright; it now skips only the ones with no
    declared replicas, and builds the rest with ``local=None``.

(3) **Never worse than today** (decision c24, c10/h5). Nothing selectable
    falls through to the singular-proxy forward; no singular origin declared
    leaves the pre-change 404 ``role_infeasible`` byte for byte; a marked
    arrival still answers 508 ``proxy_loop``.

(4) **The credential survives the upgrade** (c19/h16). The plural and singular
    key channels parse independently, so a deployment that has been forwarding
    on the singular pair must not start sending an empty Authorization to the
    same peer the moment plural origins are added.

(5) **Pressure does not re-gate a role this box cannot serve** (c21/h18).

(6) **Listing and placement share one predicate** (c18/h15, decision c17).

Everything drives the real seams — an injected ``open_upstream``, an injected
snapshot, an injected ``urlopen`` — so nothing here opens a socket.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import ReplicaConfigError, build_config
from lobes.gateway._replicas import (
    NO_REFERENCE_REASON,
    REFERENCE_NOTE,
    Fingerprint,
    PeerReplica,
    ReplicaCache,
)
from lobes.gateway._routing import list_models_payload
from lobes.gateway._selection import REASON_PEER_LESS_LOADED

_CORTEX_ID = "unsloth/Qwen3.8-27B-NVFP4"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"

_SPARK = "http://spark.local:8001"
_THOR = "http://thor.local:8000"
_ORIN_LOCAL = "http://vllm-primary:8000"

_SPARK_KEY = "sk-spark-inbound-copy-0001"
_THOR_KEY = "sk-thor-inbound-copy-0002"

_HIGH_PRESSURE = {"swap_used_percent": 90.0, "iowait_percent": 90.0}


# --- builders ---------------------------------------------------------------


def _orin_env(**over) -> dict[str, str]:
    """The deployed Orin's shape: cortex dropped, both peers declared."""
    env = {
        "PRIMARY_URL": _ORIN_LOCAL,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "PRIMARY_FEASIBLE": "false",
        "PRIMARY_PEER_ORIGIN": _SPARK,
        "PRIMARY_PEER_PROXY": "true",
        "PRIMARY_PEER_API_KEY": _SPARK_KEY,
        "PRIMARY_PEER_ORIGINS": f"{_SPARK},{_THOR}",
        "PRIMARY_PEER_API_KEYS": f"{_SPARK_KEY},{_THOR_KEY}",
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
    }
    env.update(over)
    return env


def _build(env):
    table, cfg = build_config(env)
    return table, cfg, S.peer_specs_from_table(table, env)


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


def _opener(script=None):
    script = script or {}
    calls = []

    def opener(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        calls.append(
            SimpleNamespace(
                url=backend.base_url,
                body=fwd_body,
                headers=list(headers),
                backend=backend,
            )
        )
        outcome = script.get(backend.base_url, 200)
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeUpstream(outcome)

    return opener, calls


def _state(origin, **over):
    kw = {
        "origin": origin,
        "local": False,
        "ready": True,
        "busy": False,
        "health": "ok",
        "running": 0,
        "waiting": 0,
        "fingerprint": None,
        "compatible": True,
        "reason": "",
        "last_seen": 1.0,
        "weight": 1.0,
        "calibrated": False,
    }
    kw.update(over)
    from lobes.gateway._replicas import ReplicaState

    return ReplicaState(**kw)


def _snapshot(*states):
    return lambda _name: tuple(states)


def _body(model: str) -> bytes:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}]}).encode()


def _post(table, cfg, specs, body, **kw):
    opener = kw.pop("opener", None)
    calls = kw.pop("calls", None)
    if opener is None:
        opener, calls = _opener()
    resp = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        list(kw.pop("headers", ())),
        body,
        opener,
        pressure=kw.pop("pressure", None),
        peer_specs=specs,
        replica_snapshot=kw.pop("replica_snapshot", None),
    )
    assert not kw, f"unused kwargs: {kw}"
    return resp, calls


def _header(resp, name):
    lowered = name.lower()
    return next((v for k, v in resp.headers if k.lower() == lowered), None)


def _fp(**over) -> Fingerprint:
    kw = {
        "served_id": _CORTEX_ID,
        "max_model_len": 262144,
        "runtime": "vllm",
        "quantization": "compressed-tensors",
        "kv_cache_dtype": "fp8",
        "reasoning_parser": "qwen3",
        "tool_parser": "qwen3_coder_thinking",
        "speculative_config": "unknown",
    }
    kw.update(over)
    return Fingerprint(**kw)


def _peer_only_cache(payloads, peers=(_SPARK, _THOR)):
    """A lane-less cache whose peers answer the scripted /status + /capabilities."""

    def urlopen(url, _timeout, _key):
        for origin, payload in payloads.items():
            if url.startswith(origin):
                doc = (
                    payload["capabilities"] if url.endswith("/capabilities") else payload["status"]
                )
                if doc is None:
                    raise OSError("refused")
                return 200, json.dumps(doc).encode()
        raise OSError("refused")

    return ReplicaCache(
        role="cortex",
        local=None,
        peers=tuple(PeerReplica(origin=o, api_key="") for o in peers),
        backend_name="primary",
        urlopen=urlopen,
        start=False,
    )


def _peer_payload(served=_CORTEX_ID, *, ready=True, context=262144, runtime="vllm"):
    return {
        "status": {
            "backends": [
                {
                    "name": "primary",
                    "served_name": served,
                    "health": "ok" if ready else "down",
                    "metrics": {"running": 0, "waiting": 0},
                }
            ]
        },
        "capabilities": {
            "cortex": {
                "model": served,
                "context": context,
                "fingerprint": {
                    "served_id": served,
                    "max_model_len": context,
                    "runtime": runtime,
                    "quantization": "compressed-tensors",
                    "kv_cache_dtype": "fp8",
                    "reasoning_parser": "qwen3",
                    "tool_parser": "qwen3_coder_thinking",
                    "speculative_config": "unknown",
                },
            }
        },
    }


# ============================================================================
# (1) peers agree with each other
# ============================================================================


def test_first_ready_peer_in_declaration_order_is_the_reference() -> None:
    cache = _peer_only_cache({_SPARK: _peer_payload(), _THOR: _peer_payload()})
    cache.refresh()
    states = {s.origin: s for s in cache.current()}
    assert states[_SPARK].compatible and states[_SPARK].reason == REFERENCE_NOTE
    # The second peer is compatible on its own merits, not by being the reference.
    assert states[_THOR].compatible and states[_THOR].reason == ""


def test_reference_is_declaration_order_not_probe_order() -> None:
    # The Thor answers identically; the Spark is still the reference because it
    # is declared first — a verdict that must not move with network timing.
    cache = _peer_only_cache({_SPARK: _peer_payload(), _THOR: _peer_payload()})
    cache.refresh()
    reference = [s.origin for s in cache.current() if s.reason == REFERENCE_NOTE]
    assert reference == [_SPARK]


def test_a_disagreeing_peer_is_excluded_with_the_field_named() -> None:
    cache = _peer_only_cache(
        {_SPARK: _peer_payload(), _THOR: _peer_payload(context=131072)},
    )
    cache.refresh()
    states = {s.origin: s for s in cache.current()}
    assert states[_SPARK].compatible
    assert not states[_THOR].compatible
    assert "max_model_len" in states[_THOR].reason
    assert "262144" in states[_THOR].reason and "131072" in states[_THOR].reason


def test_a_different_checkpoint_never_pools_silently() -> None:
    cache = _peer_only_cache({_SPARK: _peer_payload(), _THOR: _peer_payload(served="other/model")})
    cache.refresh()
    thor = next(s for s in cache.current() if s.origin == _THOR)
    assert not thor.compatible
    assert "served_id" in thor.reason


def test_no_ready_peer_leaves_nothing_compatible() -> None:
    cache = _peer_only_cache(
        {_SPARK: _peer_payload(ready=False), _THOR: _peer_payload(ready=False)}
    )
    cache.refresh()
    states = cache.current()
    assert states and not any(s.compatible for s in states)
    assert all(NO_REFERENCE_REASON in s.reason for s in states)


def test_a_peer_that_never_answered_keeps_its_own_reason() -> None:
    cache = _peer_only_cache({_SPARK: _peer_payload()}, peers=(_SPARK, _THOR))
    cache.refresh()
    thor = next(s for s in cache.current() if s.origin == _THOR)
    assert not thor.compatible
    assert "peer gateway" in thor.reason
    assert NO_REFERENCE_REASON not in thor.reason


def test_a_hosted_cache_still_compares_against_its_own_lane() -> None:
    # The local-lane path is untouched: with a lane present, nothing defers and
    # no peer is ever stamped as the reference.
    table, _cfg = build_config(
        {
            "PRIMARY_URL": _ORIN_LOCAL,
            "PRIMARY_SERVED_NAME": _CORTEX_ID,
            "PRIMARY_PEER_ORIGIN": _SPARK,
            "PRIMARY_PEER_ORIGINS": _SPARK,
            "PRIMARY_PEER_API_KEYS": _SPARK_KEY,
        }
    )
    caches = S.build_replica_caches(table, urlopen=lambda *_a, **_k: (200, b"{}"), start=False)
    assert caches["primary"].current()[0].local is True
    assert all(s.reason != REFERENCE_NOTE for s in caches["primary"].current())


# ============================================================================
# (2) the cache exists at all
# ============================================================================


def test_a_dropped_role_with_declared_replicas_gets_a_lane_less_cache() -> None:
    table, _cfg = build_config(_orin_env())
    caches = S.build_replica_caches(table, urlopen=lambda *_a, **_k: (200, b"{}"), start=False)
    assert "primary" in caches
    assert all(not s.local for s in caches["primary"].current())
    assert {s.origin for s in caches["primary"].current()} == {_SPARK, _THOR}


def test_a_dropped_role_with_no_declared_replicas_is_still_skipped() -> None:
    table, _cfg = build_config(
        _orin_env(
            PRIMARY_PEER_ORIGINS="",
            PRIMARY_PEER_API_KEYS="",
            MULTIMODAL_PEER_ORIGIN=_SPARK,
            MULTIMODAL_PEER_ORIGINS=_SPARK,
            MULTIMODAL_PEER_API_KEYS=_SPARK_KEY,
            MULTIMODAL_FEASIBLE="false",
        )
    )
    caches = S.build_replica_caches(table, urlopen=lambda *_a, **_k: (200, b"{}"), start=False)
    assert "primary" not in caches


def test_no_pool_declared_builds_nothing_at_all() -> None:
    table, _cfg = build_config(
        {"PRIMARY_URL": _ORIN_LOCAL, "PRIMARY_SERVED_NAME": _CORTEX_ID, "PRIMARY_FEASIBLE": "false"}
    )
    assert S.build_replica_caches(table, start=False) == {}


def test_an_unreachable_peer_never_stops_the_gateway_binding() -> None:
    # c22/h19: build_replica_caches runs on the bind path. Every peer refusing
    # must leave a cache that reports not-ready, never an exception.
    def refusing(*_a, **_k):
        raise OSError("connection refused")

    table, _cfg = build_config(_orin_env())
    caches = S.build_replica_caches(table, urlopen=refusing, start=False)
    assert not any(s.ready for s in caches["primary"].current())


def test_a_cache_whose_refresh_explodes_does_not_abort_the_boot(monkeypatch) -> None:
    table, _cfg = build_config(_orin_env())

    def exploding_refresh(self):
        raise RuntimeError("probe pass blew up")

    monkeypatch.setattr(ReplicaCache, "refresh", exploding_refresh)
    caches = S.build_replica_caches(table, urlopen=lambda *_a, **_k: (200, b"{}"), start=False)
    assert "primary" in caches


# ============================================================================
# (3) never worse than today
# ============================================================================


def test_a_dropped_pooled_role_is_placed_on_the_least_loaded_peer() -> None:
    table, cfg, specs = _build(_orin_env())
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        replica_snapshot=_snapshot(_state(_SPARK, running=6), _state(_THOR, running=0)),
    )
    assert resp.status == 200
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_PEER_LESS_LOADED
    assert [c.url for c in calls] == [_THOR]


def test_the_busier_peer_is_chosen_when_the_load_flips() -> None:
    table, cfg, specs = _build(_orin_env())
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        replica_snapshot=_snapshot(_state(_SPARK, running=0), _state(_THOR, running=9)),
    )
    assert _header(resp, S.PROXIED_BY_HEADER) == _SPARK
    assert [c.url for c in calls] == [_SPARK]


def test_the_raw_served_id_takes_the_same_path_as_the_alias() -> None:
    table, cfg, specs = _build(_orin_env())
    resp, _calls = _post(
        table,
        cfg,
        specs,
        _body(_CORTEX_ID),
        replica_snapshot=_snapshot(_state(_SPARK, running=6), _state(_THOR, running=0)),
    )
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR


def test_nothing_selectable_falls_through_to_the_singular_forward() -> None:
    table, cfg, specs = _build(_orin_env())
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        replica_snapshot=_snapshot(
            _state(_SPARK, ready=False, compatible=False),
            _state(_THOR, ready=False, compatible=False),
        ),
    )
    assert resp.status == 200
    # The SINGULAR origin, and no pool markers: this is the pre-change path.
    assert _header(resp, S.PROXIED_BY_HEADER) == _SPARK
    assert _header(resp, S.ROUTE_REASON_HEADER) is None
    assert [c.url for c in calls] == [_SPARK]


def test_no_snapshot_at_all_is_the_pre_change_path() -> None:
    table, cfg, specs = _build(_orin_env())
    resp, calls = _post(table, cfg, specs, _body("cortex"), replica_snapshot=None)
    assert _header(resp, S.PROXIED_BY_HEADER) == _SPARK
    assert _header(resp, S.ROUTE_REASON_HEADER) is None


def test_without_a_singular_origin_the_404_is_unchanged() -> None:
    table, cfg, specs = _build(
        _orin_env(PRIMARY_PEER_ORIGIN="", PRIMARY_PEER_PROXY="false", PRIMARY_PEER_API_KEY="")
    )
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        replica_snapshot=_snapshot(_state(_SPARK), _state(_THOR)),
    )
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["code"] == "role_infeasible"
    assert calls == []


def test_a_marked_arrival_still_answers_508_and_never_places() -> None:
    table, cfg, specs = _build(_orin_env())
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        headers=[(S.PROXIED_HEADER, "primary")],
        replica_snapshot=_snapshot(_state(_SPARK), _state(_THOR)),
    )
    assert resp.status == 508
    assert json.loads(resp.body)["error"]["code"] == "proxy_loop"
    assert calls == []


def test_a_role_this_box_hosts_is_untouched_by_the_peer_only_branch() -> None:
    table, cfg, specs = _build(_orin_env())
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("senses"),
        replica_snapshot=_snapshot(_state(_SPARK), _state(_THOR)),
    )
    assert resp.status == 200
    assert [c.url for c in calls] == ["http://vllm-multimodal:8000"]
    assert _header(resp, S.PROXIED_BY_HEADER) is None


def test_the_pool_is_generic_across_prefixes_not_special_cased_to_cortex() -> None:
    # c11/h12: the same behaviour on a NON-cortex prefix. Here `senses` is the
    # dropped, pooled role and cortex is served locally.
    env = {
        "PRIMARY_URL": _ORIN_LOCAL,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
        "MULTIMODAL_FEASIBLE": "false",
        "MULTIMODAL_PEER_ORIGIN": _SPARK,
        "MULTIMODAL_PEER_PROXY": "true",
        "MULTIMODAL_PEER_API_KEY": _SPARK_KEY,
        "MULTIMODAL_PEER_ORIGINS": f"{_SPARK},{_THOR}",
        "MULTIMODAL_PEER_API_KEYS": f"{_SPARK_KEY},{_THOR_KEY}",
    }
    table, cfg, specs = _build(env)
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("senses"),
        replica_snapshot=_snapshot(_state(_SPARK, running=4), _state(_THOR, running=0)),
    )
    assert resp.status == 200
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR
    assert [c.url for c in calls] == [_THOR]


# ============================================================================
# (4) the credential survives the upgrade
# ============================================================================


def _auth(call):
    return next((v for k, v in call.headers if k.lower() == "authorization"), None)


def test_each_replica_gets_its_own_declared_key() -> None:
    table, cfg, specs = _build(_orin_env())
    _resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        replica_snapshot=_snapshot(_state(_SPARK, running=6), _state(_THOR, running=0)),
    )
    assert _auth(calls[0]) == f"Bearer {_THOR_KEY}"


def test_the_singular_key_is_inherited_when_no_plural_slots_are_declared() -> None:
    # c19/h16: the exact upgrade path of a box already forwarding on the
    # singular pair. Without inheritance this forward carries no Authorization
    # and the peer — which runs an inbound gate — answers 401.
    table, cfg, specs = _build(_orin_env(PRIMARY_PEER_ORIGINS=_SPARK, PRIMARY_PEER_API_KEYS=""))
    _resp, calls = _post(
        table, cfg, specs, _body("cortex"), replica_snapshot=_snapshot(_state(_SPARK))
    )
    assert _auth(calls[0]) == f"Bearer {_SPARK_KEY}"


def test_a_replica_that_is_not_the_singular_peer_never_borrows_its_key() -> None:
    table, _cfg = build_config(_orin_env(PRIMARY_PEER_API_KEYS=","))
    assert S._replica_api_key(table, "primary", _SPARK) == _SPARK_KEY
    assert S._replica_api_key(table, "primary", _THOR) == ""


def test_the_callers_own_authorization_never_reaches_a_peer() -> None:
    table, cfg, specs = _build(_orin_env())
    _resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        headers=[("Authorization", "Bearer caller-secret")],
        replica_snapshot=_snapshot(_state(_SPARK, running=6), _state(_THOR, running=0)),
    )
    assert "caller-secret" not in json.dumps([list(c.headers) for c in calls])


# ============================================================================
# (5) pressure does not re-gate a role this box cannot serve
# ============================================================================


def test_local_pressure_never_sheds_a_peer_only_pooled_request() -> None:
    table, cfg, specs = _build(_orin_env())
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        pressure=_HIGH_PRESSURE,
        replica_snapshot=_snapshot(_state(_SPARK, running=6), _state(_THOR, running=0)),
    )
    assert resp.status == 200
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR
    assert [c.url for c in calls] == [_THOR]


# ============================================================================
# (6) listing and placement share one predicate
# ============================================================================


def test_a_pooled_dropped_role_is_listed_in_v1_models() -> None:
    table, _cfg = build_config(_orin_env())
    pooled = S.pooled_backends(table, _snapshot(_state(_SPARK), _state(_THOR)))
    assert pooled == frozenset({"primary"})
    payload = list_models_payload(
        table, {"multimodal": True}, {"primary": _CORTEX_ID}, pooled=pooled
    )
    assert _CORTEX_ID in {entry["id"] for entry in payload["data"]}


def test_the_listing_disappears_when_every_replica_goes_unready() -> None:
    table, _cfg = build_config(_orin_env())
    pooled = S.pooled_backends(
        table, _snapshot(_state(_SPARK, ready=False), _state(_THOR, ready=False))
    )
    assert pooled == frozenset()
    payload = list_models_payload(
        table, {"multimodal": True}, {"primary": _CORTEX_ID}, pooled=pooled
    )
    assert _CORTEX_ID not in {entry["id"] for entry in payload["data"]}


def test_an_incompatible_replica_set_is_not_pooled_and_not_listed() -> None:
    table, _cfg = build_config(_orin_env())
    assert (
        S.pooled_backends(
            table, _snapshot(_state(_SPARK, compatible=False), _state(_THOR, compatible=False))
        )
        == frozenset()
    )


def test_a_referral_only_deployment_lists_nothing_new() -> None:
    table, _cfg = build_config(
        _orin_env(PRIMARY_PEER_ORIGINS="", PRIMARY_PEER_API_KEYS="", PRIMARY_PEER_PROXY="false")
    )
    assert S.pooled_backends(table, _snapshot(_state(_SPARK))) == frozenset()
    before = list_models_payload(table, {"multimodal": True}, None)
    after = list_models_payload(table, {"multimodal": True}, None, pooled=frozenset())
    assert before == after


def test_placement_and_listing_agree_on_the_same_snapshot() -> None:
    # h15: one source of truth. Whatever the predicate says is pooled is
    # exactly what gets placed — asserted over both verdicts of the predicate.
    table, cfg, specs = _build(_orin_env())
    for snapshot, expect_pooled in (
        (_snapshot(_state(_SPARK), _state(_THOR)), True),
        (_snapshot(_state(_SPARK, ready=False), _state(_THOR, ready=False)), False),
    ):
        pooled = "primary" in S.pooled_backends(table, snapshot)
        resp, _calls = _post(table, cfg, specs, _body("cortex"), replica_snapshot=snapshot)
        placed = _header(resp, S.ROUTE_REASON_HEADER) is not None
        assert pooled == expect_pooled == placed


# ============================================================================
# the arming gate
# ============================================================================


def test_plural_origins_without_the_singular_one_refuse_to_arm() -> None:
    table, _cfg = build_config(
        _orin_env(PRIMARY_PEER_ORIGIN="", PRIMARY_PEER_PROXY="false", PRIMARY_PEER_API_KEY="")
    )
    with pytest.raises(ReplicaConfigError) as excinfo:
        S.build_replica_caches(table, urlopen=lambda *_a, **_k: (200, b"{}"), start=False)
    assert "PRIMARY_PEER_ORIGINS" in str(excinfo.value)
    assert "PRIMARY_PEER_ORIGIN" in str(excinfo.value)


def test_a_hosted_pool_needs_no_singular_origin() -> None:
    # The #199 case: the Spark pools cortex while HOSTING it and publishes no
    # referral at all. Demanding a singular origin there would refuse every
    # pool that shipped before this feature existed.
    table, _cfg = build_config(
        {
            "PRIMARY_URL": _ORIN_LOCAL,
            "PRIMARY_SERVED_NAME": _CORTEX_ID,
            "PRIMARY_PEER_ORIGINS": _THOR,
            "PRIMARY_PEER_API_KEYS": _THOR_KEY,
            "GATEWAY_SELF_ORIGIN": _SPARK,
        }
    )
    caches = S.build_replica_caches(table, urlopen=lambda *_a, **_k: (200, b"{}"), start=False)
    assert "primary" in caches


# ============================================================================
# the capabilities advert
# ============================================================================


def test_ready_is_true_when_any_compatible_replica_is_ready() -> None:
    table, cfg = build_config(_orin_env())
    payload = S.capabilities_payload(
        table,
        cfg,
        env=_orin_env(),
        gateway_url="http://localhost:8000",
        replica_snapshot={"cortex": (_state(_SPARK, ready=False), _state(_THOR, ready=True))},
    )
    assert payload["cortex"]["ready"] is True
    assert payload["cortex"]["feasible"] is False  # pooling never promotes a host


def test_ready_is_false_when_no_compatible_replica_is_ready() -> None:
    table, cfg = build_config(_orin_env())
    payload = S.capabilities_payload(
        table,
        cfg,
        env=_orin_env(),
        gateway_url="http://localhost:8000",
        replica_snapshot={
            "cortex": (_state(_SPARK, ready=False), _state(_THOR, ready=False)),
        },
    )
    assert payload["cortex"]["ready"] is False


def test_context_is_the_agreed_window_not_the_catalog_ceiling() -> None:
    table, cfg = build_config(_orin_env())
    payload = S.capabilities_payload(
        table,
        cfg,
        env=_orin_env(),
        gateway_url="http://localhost:8000",
        replica_snapshot={
            "cortex": (
                _state(_SPARK, fingerprint=_fp(max_model_len=262144)),
                _state(_THOR, fingerprint=_fp(max_model_len=262144)),
            )
        },
    )
    assert payload["cortex"]["context"] == 262144


# ============================================================================
# in-flight accounting — the herd the live run caught
# ============================================================================


def test_concurrent_placements_spread_across_replicas() -> None:
    """Consecutive placements must not all land on the same peer.

    Measured live on the Orin, 2026-08-30: four concurrent ``model=cortex``
    requests were all placed onto the Spark while the Thor sat idle, because
    the peer-only branch dialled without COUNTING the dispatch. Probed load is
    up to one refresh interval stale, so every arrival read the same idle
    snapshot, ranked the same replica first (ties break on origin string
    ascending) and stampeded it. Counting is what makes the snapshot
    self-correct between probes — and a peer-only pool needs it more than a
    hosted one, having no local replica to absorb the tie.
    """
    table, cfg, specs = _build(_orin_env())
    cache = _peer_only_cache({_SPARK: _peer_payload(), _THOR: _peer_payload()})
    cache.refresh()
    caches = {"primary": cache}
    counter = S.dispatch_counter(caches)
    snapshot = S.replica_snapshot_provider(caches)

    served = []
    holds = []
    for _ in range(4):
        opener, calls = _opener()
        # A response still holding an upstream keeps its dispatch counted
        # until the handler relays it — exactly the in-flight window a burst
        # of concurrent requests occupies.
        resp = S.handle_post(
            table,
            cfg,
            "/v1/chat/completions",
            [],
            _body("cortex"),
            opener,
            peer_specs=specs,
            replica_snapshot=snapshot,
            dispatch_counter=counter,
        )
        holds.append(resp)
        served.append(calls[0].url)

    assert set(served) == {_SPARK, _THOR}, f"all four placed onto {set(served)}"
    # Releasing every hold returns the pool to an even footing.
    for resp in holds:
        if resp.on_complete is not None:
            resp.on_complete()


def test_a_released_dispatch_stops_counting_against_its_replica() -> None:
    table, cfg, specs = _build(_orin_env())
    cache = _peer_only_cache({_SPARK: _peer_payload(), _THOR: _peer_payload()})
    cache.refresh()
    caches = {"primary": cache}

    def place():
        opener, calls = _opener()
        resp = S.handle_post(
            table,
            cfg,
            "/v1/chat/completions",
            [],
            _body("cortex"),
            opener,
            peer_specs=specs,
            replica_snapshot=S.replica_snapshot_provider(caches),
            dispatch_counter=S.dispatch_counter(caches),
        )
        return resp, calls[0].url

    first, first_url = place()
    if first.on_complete is not None:
        first.on_complete()
    # With the first dispatch released, the pool is level again and the same
    # deterministic tie-break applies — so the second request repeats the
    # first's choice rather than alternating blindly.
    _second, second_url = place()
    assert first_url == second_url == _SPARK
