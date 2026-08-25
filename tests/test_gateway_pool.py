"""The replica-pool data plane (cortex-replica-pool t7, issue #199).

The pool is the FOURTH thing that can happen to a model-routed POST, and it
sits BEFORE local dispatch rather than replacing a 404 the way the
referral/proxy branch (issues #115/#127) does: a role this box HOSTS may
still be answered by a declared peer replica when the snapshot says that
peer is better placed. Everything here drives
:func:`lobes.gateway.server.handle_post` through its pure seams — an
injected ``open_upstream`` (no sockets) and an injected
``replica_snapshot`` (no probing, no clock) — so every assertion is about
the DISPATCH decision, never about live probe behaviour.

Contract under test (the plan's t7 acceptance criteria, verbatim):

(a) ``model=cortex`` and ``model=<raw served id>`` produce IDENTICAL
    selection, forward target and markers for the same snapshot — every
    deployed consumer pins the raw id (the 2026-07-31 audit), so an
    alias-only pool would never see a real caller (spec c31/h23);
(b) with no ``*_PEER_ORIGINS`` declared, every response is byte-identical
    to the pre-pool release — no new headers on any path, success or error
    (spec c1/h1);
(c) an inbound ``X-Lobes-Proxied`` request is served by the LOCAL replica
    only and opens ZERO outbound sockets even when the snapshot prefers a
    peer; a box with no local replica still answers 508 ``proxy_loop``
    (spec c4/h4);
(d) local answers carry ``X-Lobes-Served-By`` (this box's declared
    ``GATEWAY_SELF_ORIGIN``, or ``local`` when undeclared) and forwarded
    answers keep ``X-Lobes-Proxied-By``; BOTH carry
    ``X-Lobes-Route-Reason`` naming the selection reason (spec c19/h14,
    c37/h30);
(e) a forwarded streaming answer relays through the existing one-shot byte
    tunnel and a mid-stream drop is NEVER replayed on another replica;
(f) ``X-Lobes-Affinity`` reaches :func:`select_replica` and travels to the
    peer on a forward; absent, selection is purely availability-driven
    (spec c24/h16).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._replicas import ReplicaState
from lobes.gateway._selection import (
    REASON_AFFINITY,
    REASON_LOCAL_IDLE,
    REASON_NONE,
    REASON_PEER_LESS_LOADED,
    REASON_SOLE_READY,
    Selection,
)

# The deployed cortex checkpoint (docs/qwen3.8-27b-nvfp4.md) — the id every
# consumer pins in culture.yaml, which is exactly why (a) above matters.
_CORTEX_ID = "unsloth/Qwen3.8-27B-NVFP4"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"

_SPARK_ORIGIN = "http://spark.local:8001"
_THOR_ORIGIN = "http://thor.local:8001"
_ORIN_ORIGIN = "http://orin.local:8001"

_THOR_KEY = "sk-thor-inbound-copy-0001"
_ORIN_KEY = "sk-orin-inbound-copy-0002"
_CALLER_TOKEN = "sk-caller-inbound-token-9999"

_HIGH_PRESSURE = {"swap_used_percent": 90.0, "iowait_percent": 90.0}
_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}


# --- env builders ------------------------------------------------------------


def _base_env(**over) -> dict[str, str]:
    """machine-as-brain-ish: cortex + senses hosted locally, NO pool declared.

    This is the (b) control: byte-identical to the pre-pool release.
    """
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
    }
    env.update(over)
    return env


def _pool_env(**over) -> dict[str, str]:
    """The Spark with a cortex replica pool: hosts cortex AND declares the
    Thor as a peer replica of the SAME role, with that peer's inbound key."""
    env = {
        "PRIMARY_PEER_ORIGINS": _THOR_ORIGIN,
        "PRIMARY_PEER_API_KEYS": _THOR_KEY,
        "GATEWAY_SELF_ORIGIN": _SPARK_ORIGIN,
    }
    env.update(over)
    return _base_env(**env)


def _dropped_pool_env(**over) -> dict[str, str]:
    """A box that does NOT host cortex (thor-lobe shape) but declares both the
    singular proxy peer and a replica pool. The singular proxy branch owns
    this request — there is no local replica to serve a marked arrival, so the
    508 ``proxy_loop`` refusal must survive the pool landing (spec c4/h4)."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "PRIMARY_FEASIBLE": "false",
        "PRIMARY_PEER_ORIGIN": _SPARK_ORIGIN,
        "PRIMARY_PEER_PROXY": "true",
        "PRIMARY_PEER_API_KEY": _THOR_KEY,
        "PRIMARY_PEER_ORIGINS": _SPARK_ORIGIN,
        "PRIMARY_PEER_API_KEYS": _THOR_KEY,
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
    }
    env.update(over)
    return env


def _build(env):
    table, cfg = build_config(env)
    return table, cfg, S.peer_specs_from_table(table, env)


# --- fakes -------------------------------------------------------------------


class _FakeUpstream:
    def __init__(self, status=200, body=b'{"ok":1}', chunks=None, headers=None):
        self.status = status
        self.headers = headers if headers is not None else [("Content-Type", "application/json")]
        self._body = body
        self._chunks = list(chunks) if chunks is not None else None
        self.closed = False

    def read_all(self):
        return self._body

    def read(self, _n):
        if self._chunks is None:
            data, self._body = self._body, b""
            return data
        return self._chunks.pop(0) if self._chunks else b""

    def close(self):
        self.closed = True


def _opener(outcome=200, body=b'{"ok":1}', chunks=None):
    calls = []

    def opener(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        calls.append(
            SimpleNamespace(backend=backend, path=path, body=fwd_body, headers=list(headers))
        )
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeUpstream(outcome, body=body, chunks=chunks)

    return opener, calls


def _state(
    origin: str,
    *,
    local: bool = False,
    ready: bool = True,
    busy: bool = False,
    running: int = 0,
    waiting: int = 0,
    compatible: bool = True,
    weight: float = 1.0,
) -> ReplicaState:
    """A snapshot row, injected directly.

    Deliberately NOT produced by a live :class:`ReplicaCache` probe: until
    t6/t8 publish a peer fingerprint every probed peer reads ``unknown`` and
    therefore ``compatible=False``, which would make every pool test here
    vacuously fall through to local dispatch. The dispatch decision is what
    is under test, so the snapshot is the injected input.
    """
    return ReplicaState(
        origin=origin,
        local=local,
        ready=ready,
        busy=busy,
        health="ok",
        running=running,
        waiting=waiting,
        fingerprint=None,
        compatible=compatible,
        reason="",
        last_seen=1.0,
        weight=weight,
    )


def _snapshot(*states):
    return lambda _name: tuple(states)


def _body(model: str, *, stream: bool = False) -> bytes:
    payload: dict = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    if stream:
        payload["stream"] = True
    return json.dumps(payload).encode()


def _post(
    table,
    cfg,
    specs,
    body,
    *,
    headers=(),
    pressure=None,
    opener=None,
    calls=None,
    replica_snapshot=None,
    path="/v1/chat/completions",
):
    if opener is None:
        opener, calls = _opener()
    resp = S.handle_post(
        table,
        cfg,
        path,
        list(headers),
        body,
        opener,
        pressure=pressure,
        peer_specs=specs,
        replica_snapshot=replica_snapshot,
    )
    return resp, calls


def _header(resp, name: str):
    lowered = name.lower()
    for key, value in resp.headers:
        if key.lower() == lowered:
            return value
    return None


# ============================================================================
# (b) no pool declared ⇒ byte-identical to the pre-pool release
# ============================================================================

# One representative request per path through handle_post: the served happy
# path, the tier-alias path, the unknown-id 404 (h23), the busy 429 shed
# (#85), and the owner-down 503 (#14/#91).
_NO_POOL_CASES = [
    pytest.param(_body(_CORTEX_ID), None, 200, id="raw-id-served"),
    pytest.param(_body("cortex"), _NO_PRESSURE, 200, id="alias-served"),
    pytest.param(_body("nobody/never-advertised"), None, 404, id="unknown-model"),
    pytest.param(_body("cortex"), _HIGH_PRESSURE, 429, id="pressure-shed"),
]


@pytest.mark.parametrize("payload,pressure,status", _NO_POOL_CASES)
def test_no_pool_declared_is_byte_identical(payload, pressure, status) -> None:
    # h1: with no *_PEER_ORIGINS the pool code path must be provably inert.
    # The strongest available offline proof is a same-input comparison: the
    # SAME request through the SAME table with the snapshot provider present
    # vs absent must produce an identical status, body and header list — and
    # neither may carry a pool marker.
    table, cfg, specs = _build(_base_env())
    assert not table.replica_origins  # the control's premise
    snapshot = _snapshot(_state("http://vllm-primary:8000", local=True))

    without, calls_a = _post(table, cfg, specs, payload, pressure=pressure)
    with_provider, calls_b = _post(
        table, cfg, specs, payload, pressure=pressure, replica_snapshot=snapshot
    )

    for resp in (without, with_provider):
        assert resp.status == status
        assert _header(resp, S.SERVED_BY_HEADER) is None
        assert _header(resp, S.ROUTE_REASON_HEADER) is None
    assert without.status == with_provider.status
    assert without.headers == with_provider.headers
    assert without.body == with_provider.body
    assert len(calls_a) == len(calls_b)


def test_no_pool_owner_down_503_unchanged() -> None:
    table, cfg, specs = _build(_base_env())
    opener, calls = _opener(outcome=S.UpstreamError("connection refused"))
    resp, _ = _post(table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls)
    assert resp.status == 503
    assert _header(resp, S.ROUTE_REASON_HEADER) is None
    assert _header(resp, S.SERVED_BY_HEADER) is None


# ============================================================================
# (a) alias and raw served id place identically
# ============================================================================


@pytest.mark.parametrize("model", ["cortex", _CORTEX_ID])
def test_alias_and_raw_id_forward_to_the_same_replica(model) -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=5),
        _state(_THOR_ORIGIN),
    )
    resp, calls = _post(
        table, cfg, specs, _body(model), pressure=_NO_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR_ORIGIN
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_PEER_LESS_LOADED
    assert len(calls) == 1
    assert calls[0].backend.base_url == _THOR_ORIGIN
    # The forwarded body always names the peer's served id, never the alias.
    assert json.loads(calls[0].body)["model"] == _CORTEX_ID


def test_alias_and_raw_id_produce_identical_markers() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=5),
        _state(_THOR_ORIGIN),
    )
    alias, alias_calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_NO_PRESSURE, replica_snapshot=snapshot
    )
    raw, raw_calls = _post(
        table, cfg, specs, _body(_CORTEX_ID), pressure=_NO_PRESSURE, replica_snapshot=snapshot
    )
    assert _header(alias, S.PROXIED_BY_HEADER) == _header(raw, S.PROXIED_BY_HEADER)
    assert _header(alias, S.ROUTE_REASON_HEADER) == _header(raw, S.ROUTE_REASON_HEADER)
    assert alias_calls[0].backend.base_url == raw_calls[0].backend.base_url
    assert json.loads(alias_calls[0].body)["model"] == json.loads(raw_calls[0].body)["model"]


# ============================================================================
# (d) markers on local and forwarded answers
# ============================================================================


def test_local_answer_carries_self_origin_and_reason() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True),
        _state(_THOR_ORIGIN, running=9),
    )
    resp, calls = _post(table, cfg, specs, _body(_CORTEX_ID), replica_snapshot=snapshot)
    assert resp.status == 200
    assert _header(resp, S.SERVED_BY_HEADER) == _SPARK_ORIGIN
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_LOCAL_IDLE
    assert _header(resp, S.PROXIED_BY_HEADER) is None
    assert len(calls) == 1
    assert calls[0].backend.base_url == "http://vllm-primary:8000"


def test_local_answer_without_declared_self_origin_says_local() -> None:
    # GATEWAY_SELF_ORIGIN is operator-typed and never derived (#92) — an
    # undeclared box must say something honest rather than invent a hostname.
    table, cfg, specs = _build(_pool_env(GATEWAY_SELF_ORIGIN=""))
    assert table.self_origin == ""
    snapshot = _snapshot(_state("http://vllm-primary:8000", local=True))
    resp, _ = _post(table, cfg, specs, _body(_CORTEX_ID), replica_snapshot=snapshot)
    assert _header(resp, S.SERVED_BY_HEADER) == "local"


def test_forwarded_answer_keeps_proxied_by_and_adds_reason() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=4),
        _state(_THOR_ORIGIN),
    )
    resp, calls = _post(table, cfg, specs, _body(_CORTEX_ID), replica_snapshot=snapshot)
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR_ORIGIN
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_PEER_LESS_LOADED
    assert _header(resp, S.SERVED_BY_HEADER) is None
    # The departing request is marked so the receiver never re-forwards it.
    marks = [v for k, v in calls[0].headers if k == S.PROXIED_HEADER]
    assert marks == ["primary"]


def test_forward_swaps_the_caller_credential_for_the_replica_key() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=4),
        _state(_THOR_ORIGIN),
    )
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body(_CORTEX_ID),
        headers=[("Authorization", f"Bearer {_CALLER_TOKEN}")],
        replica_snapshot=snapshot,
    )
    assert resp.status == 200
    auth = [v for k, v in calls[0].headers if k.lower() == "authorization"]
    assert auth == [f"Bearer {_THOR_KEY}"]
    assert _CALLER_TOKEN not in json.dumps(calls[0].headers)


def test_replica_key_is_positional_per_origin() -> None:
    table, cfg, specs = _build(
        _pool_env(
            PRIMARY_PEER_ORIGINS=f"{_THOR_ORIGIN},{_ORIN_ORIGIN}",
            PRIMARY_PEER_API_KEYS=f"{_THOR_KEY},{_ORIN_KEY}",
        )
    )
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=9),
        _state(_THOR_ORIGIN, running=5),
        _state(_ORIN_ORIGIN, running=1),
    )
    _resp, calls = _post(table, cfg, specs, _body(_CORTEX_ID), replica_snapshot=snapshot)
    assert calls[0].backend.base_url == _ORIN_ORIGIN
    auth = [v for k, v in calls[0].headers if k.lower() == "authorization"]
    assert auth == [f"Bearer {_ORIN_KEY}"]


def test_empty_replica_key_slot_sends_no_authorization() -> None:
    # h29: an EMPTY slot is legal and means "this peer has no inbound gate"
    # (the Thor sets no GATEWAY_API_KEY today) — never a blank Bearer.
    table, cfg, specs = _build(_pool_env(PRIMARY_PEER_API_KEYS=""))
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=4),
        _state(_THOR_ORIGIN),
    )
    _resp, calls = _post(
        table,
        cfg,
        specs,
        _body(_CORTEX_ID),
        headers=[("Authorization", f"Bearer {_CALLER_TOKEN}")],
        replica_snapshot=snapshot,
    )
    assert [v for k, v in calls[0].headers if k.lower() == "authorization"] == []


# ============================================================================
# (c) single hop — an arriving marked request is served locally, never re-sent
# ============================================================================


def test_marked_arrival_is_served_locally_with_zero_outbound_forwards() -> None:
    table, cfg, specs = _build(_pool_env())
    # The snapshot STRONGLY prefers the peer (local saturated) — and is ignored.
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=99),
        _state(_THOR_ORIGIN),
    )
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body(_CORTEX_ID),
        headers=[(S.PROXIED_HEADER, "primary")],
        replica_snapshot=snapshot,
    )
    assert resp.status == 200
    assert len(calls) == 1
    assert calls[0].backend.base_url == "http://vllm-primary:8000"
    assert _header(resp, S.PROXIED_BY_HEADER) is None
    assert _header(resp, S.SERVED_BY_HEADER) == _SPARK_ORIGIN
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_SOLE_READY


def test_marked_arrival_with_no_local_replica_still_508s() -> None:
    table, cfg, specs = _build(_dropped_pool_env())
    snapshot = _snapshot(_state(_SPARK_ORIGIN))
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body(_CORTEX_ID),
        headers=[(S.PROXIED_HEADER, "primary")],
        replica_snapshot=snapshot,
    )
    assert resp.status == S._PROXY_LOOP_STATUS
    assert calls == []
    assert json.loads(resp.body)["error"]["code"] == "proxy_loop"


# ============================================================================
# (f) affinity
# ============================================================================


def test_affinity_header_reaches_select_replica_and_the_peer(monkeypatch) -> None:
    seen: list = []

    def spy(candidates, *, affinity=None, local_busy=False, affinity_margin=1.0):
        seen.append(affinity)
        return Selection(_THOR_ORIGIN, False, REASON_AFFINITY)

    monkeypatch.setattr(S, "select_replica", spy)
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True),
        _state(_THOR_ORIGIN),
    )
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body(_CORTEX_ID),
        headers=[(S.AFFINITY_HEADER, "  session-42  ")],
        replica_snapshot=snapshot,
    )
    assert seen == ["session-42"]  # stripped
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_AFFINITY
    # The key travels so the receiving gateway can make the same sticky choice.
    assert [v for k, v in calls[0].headers if k.lower() == S.AFFINITY_HEADER.lower()] == [
        "  session-42  "
    ]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_absent_or_blank_affinity_reaches_select_replica_as_none(monkeypatch, value) -> None:
    seen: list = []

    def spy(candidates, *, affinity=None, local_busy=False, affinity_margin=1.0):
        seen.append(affinity)
        return Selection(None, False, REASON_NONE)

    monkeypatch.setattr(S, "select_replica", spy)
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state("http://vllm-primary:8000", local=True))
    headers = [] if value is None else [(S.AFFINITY_HEADER, value)]
    _post(table, cfg, specs, _body(_CORTEX_ID), headers=headers, replica_snapshot=snapshot)
    assert seen == [None]


# ============================================================================
# nothing selectable ⇒ today's local dispatch, honestly marked
# ============================================================================


def test_no_selectable_replica_falls_through_to_local_dispatch() -> None:
    # t7 leaves the 429/503 semantics of "nothing anywhere" to t8; the honest
    # t7 behaviour is the pre-pool one — dial the local owner — with the
    # reason marker saying `none` rather than claiming a selection happened.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, compatible=False),
        _state(_THOR_ORIGIN, ready=False),
    )
    resp, calls = _post(table, cfg, specs, _body(_CORTEX_ID), replica_snapshot=snapshot)
    assert resp.status == 200
    assert len(calls) == 1
    assert calls[0].backend.base_url == "http://vllm-primary:8000"
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_NONE
    assert _header(resp, S.SERVED_BY_HEADER) == _SPARK_ORIGIN


def test_empty_snapshot_falls_through_to_local_dispatch() -> None:
    table, cfg, specs = _build(_pool_env())
    resp, calls = _post(table, cfg, specs, _body(_CORTEX_ID), replica_snapshot=lambda _name: ())
    assert resp.status == 200
    assert calls[0].backend.base_url == "http://vllm-primary:8000"
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_NONE


def test_unpooled_role_in_a_pooled_deployment_is_unmarked() -> None:
    # Only the role with a declared replica set takes the pool path: a senses
    # request on the same box stays byte-identical to the pre-pool contract.
    table, cfg, specs = _build(_pool_env())
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body(_SENSES_ID),
        replica_snapshot=_snapshot(_state("http://vllm-multimodal:8000", local=True)),
    )
    assert resp.status == 200
    assert _header(resp, S.ROUTE_REASON_HEADER) is None
    assert _header(resp, S.SERVED_BY_HEADER) is None
    assert calls[0].backend.base_url == "http://vllm-multimodal:8000"


# ============================================================================
# t8 SUPERSEDED t7 here: the pressure shed became a forward
# ============================================================================


def test_pressure_shed_is_forwarded_once_a_pool_is_declared() -> None:
    # This test asserted the pre-t8 behaviour (429 + zero outbound calls) and
    # is kept, inverted, as the record of the change: with a selectable peer
    # replica declared, #85's shed becomes a forward (spec c7/h6). The 429 is
    # now reserved for "no replica anywhere is available" — proven in
    # tests/test_gateway_pool_pressure.py, which owns the full t8 contract.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, busy=True),
        _state(_THOR_ORIGIN),
    )
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        pressure=_HIGH_PRESSURE,
        replica_snapshot=snapshot,
    )
    assert resp.status == 200
    assert len(calls) == 1
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR_ORIGIN
    assert _header(resp, S.ROUTE_REASON_HEADER) == "local-busy-forwarded"


def test_pressure_shed_on_an_unpooled_role_still_429s() -> None:
    # The control: no *_PEER_ORIGINS ⇒ #85's shed is untouched, headers and all.
    table, cfg, specs = _build(_base_env())
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        pressure=_HIGH_PRESSURE,
        replica_snapshot=_snapshot(_state("http://vllm-primary:8000", local=True, busy=True)),
    )
    assert resp.status == 429
    assert calls == []
    assert _header(resp, "X-Lobes-Tier-Reason") == "busy"
    assert _header(resp, S.ROUTE_REASON_HEADER) is None


# ============================================================================
# (e) streaming relays; a mid-stream drop is never replayed
# ============================================================================


def test_forwarded_streaming_answer_relays_through_the_byte_tunnel() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=4),
        _state(_THOR_ORIGIN),
    )
    opener, calls = _opener(chunks=[b"data: a\n\n", b"data: b\n\n"])
    resp, _ = _post(
        table,
        cfg,
        specs,
        _body(_CORTEX_ID, stream=True),
        opener=opener,
        calls=calls,
        replica_snapshot=snapshot,
    )
    assert resp.streaming is True
    assert resp.upstream is not None
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR_ORIGIN
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_PEER_LESS_LOADED
    assert resp.upstream.read(4096) == b"data: a\n\n"


def test_mid_stream_drop_is_not_replayed_on_another_replica() -> None:
    # The relay is a one-shot byte tunnel with NO buffering (server.py
    # _relay_streaming): once the peer answered 2xx the request is committed
    # to that replica, so a truncated stream ends honestly rather than being
    # re-issued (which would double-charge the model and duplicate tokens).
    table, cfg, specs = _build(
        _pool_env(
            PRIMARY_PEER_ORIGINS=f"{_THOR_ORIGIN},{_ORIN_ORIGIN}",
            PRIMARY_PEER_API_KEYS=f"{_THOR_KEY},{_ORIN_KEY}",
        )
    )
    snapshot = _snapshot(
        _state("http://vllm-primary:8000", local=True, running=9),
        _state(_THOR_ORIGIN, running=1),
        _state(_ORIN_ORIGIN, running=5),
    )
    opener, calls = _opener(chunks=[b"data: a\n\n"])
    resp, _ = _post(
        table,
        cfg,
        specs,
        _body(_CORTEX_ID, stream=True),
        opener=opener,
        calls=calls,
        replica_snapshot=snapshot,
    )
    assert resp.upstream.read(4096) == b"data: a\n\n"
    assert resp.upstream.read(4096) == b""  # peer dropped
    assert len(calls) == 1  # nothing was re-dialed


# ============================================================================
# the pool never overrides the existing precedence rules
# ============================================================================


def test_unknown_model_still_404s_before_the_pool() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_THOR_ORIGIN))
    resp, calls = _post(
        table, cfg, specs, _body("nobody/never-advertised"), replica_snapshot=snapshot
    )
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["code"] == "model_not_found"
    assert calls == []
    assert _header(resp, S.ROUTE_REASON_HEADER) is None


def test_pool_does_not_apply_to_audio_paths() -> None:
    # /v1/audio/* is path-routed through handle_audio_request, which never
    # consults the pool — the plural family is a GENERATE-lane concern in v1.
    table, cfg, specs = _build(_pool_env())
    opener, calls = _opener()
    resp = S.handle_audio_request(table, cfg, specs, "/v1/audio/speech", [], b"{}", opener)
    assert _header(resp, S.ROUTE_REASON_HEADER) is None
