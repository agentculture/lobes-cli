"""Replica-pool PRESSURE and FAILURE semantics (cortex-replica-pool t8, #199).

t7 decided *where* a pooled request goes; t8 decides what happens when the
answer is "nowhere good". Four contracts, each with a recorded reason:

(1) **Pressure forwards, it does not shed** (spec c7/h6). #85 shed a
    ``main``/``multimodal`` request with 429 because this box was the only
    place the model lived. With replicas that premise is gone: the honest
    answer to "this box is swapping" is "the request goes to the box that is
    not". The 429 + ``Retry-After`` is reserved for "no replica anywhere can
    take it", and a non-pooled box keeps the pre-pool 429 byte-for-byte.

(2) **At most ONE forward per request** (spec c35/h27). A peer's own 429 is
    ITS verdict under ITS pressure policy — pressure describes the box that
    samples it (#85) — so it rides back through the existing 4xx relay and is
    never retried locally or re-forwarded. Two mutually-loaded boxes therefore
    produce exactly one outbound forward and one 429, never a ping-pong. An
    inbound ``X-Lobes-Proxied`` request never forwards at all: under local
    pressure it gets THIS box's 429, which is the receiver applying its own
    policy.

(3) **Pre-dispatch retry, never a mid-stream replay** (spec c15/h12). A
    replica that refused / timed out / 5xx'd before any bytes came back never
    served the request, so the next selectable replica is tried — at most once
    per replica, the LOCAL replica included. A replica that answered 2xx and
    then dropped DID serve it (the relay is a one-shot byte tunnel with no
    buffering), so it is never re-issued. All fail → 503
    ``backend_unavailable`` naming every attempt.

(4) **The live surfaces are wired** (spec c9, c33, c34/h26). ``serve()``
    builds one :class:`~lobes.gateway._replicas.ReplicaCache` per participating
    backend, refreshes it BEFORE binding so the first request has a snapshot,
    and publishes this box's own live fingerprint on ``GET /capabilities`` —
    the only thing a peer can read to decide whether our replica is compatible
    with its own. With no pool declared none of that happens and every byte is
    the pre-pool byte (h1).

Every assertion drives :func:`lobes.gateway.server.handle_post` (or the cache
builders) through their pure seams — an injected ``open_upstream`` and an
injected snapshot/``urlopen`` — so nothing here opens a socket.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._replicas import ReplicaState
from lobes.gateway._selection import (
    REASON_LOCAL_BUSY_FORWARDED,
    REASON_NONE,
    REASON_SOLE_READY,
)

_CORTEX_ID = "unsloth/Qwen3.8-27B-NVFP4"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_LOCAL_URL = "http://vllm-primary:8000"

_SPARK_ORIGIN = "http://spark.local:8001"
_THOR_ORIGIN = "http://thor.local:8001"
_ORIN_ORIGIN = "http://orin.local:8001"

_THOR_KEY = "sk-thor-inbound-copy-0001"
_ORIN_KEY = "sk-orin-inbound-copy-0002"

_HIGH_PRESSURE = {"swap_used_percent": 90.0, "iowait_percent": 90.0}
_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}


# --- env / fixture builders --------------------------------------------------


def _base_env(**over) -> dict[str, str]:
    env = {
        "PRIMARY_URL": _LOCAL_URL,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
    }
    env.update(over)
    return env


def _pool_env(**over) -> dict[str, str]:
    env = {
        "PRIMARY_PEER_ORIGINS": _THOR_ORIGIN,
        "PRIMARY_PEER_API_KEYS": _THOR_KEY,
        "GATEWAY_SELF_ORIGIN": _SPARK_ORIGIN,
    }
    env.update(over)
    return _base_env(**env)


def _two_peer_env(**over) -> dict[str, str]:
    return _pool_env(
        PRIMARY_PEER_ORIGINS=f"{_THOR_ORIGIN},{_ORIN_ORIGIN}",
        PRIMARY_PEER_API_KEYS=f"{_THOR_KEY},{_ORIN_KEY}",
        **over,
    )


def _build(env):
    table, cfg = build_config(env)
    return table, cfg, S.peer_specs_from_table(table, env)


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


def _scripted_opener(script):
    """``script`` maps a target URL to an outcome: an ``Exception`` to raise,
    an int status, or a ready ``_FakeUpstream``. A missing URL is a 200."""
    calls = []

    def opener(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        calls.append(
            SimpleNamespace(
                backend=backend,
                path=path,
                body=fwd_body,
                headers=list(headers),
                url=backend.base_url,
            )
        )
        outcome = script.get(backend.base_url, 200)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, int):
            return _FakeUpstream(outcome)
        return outcome

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


def _post(table, cfg, specs, body, **kw):
    opener = kw.pop("opener", None)
    calls = kw.pop("calls", None)
    if opener is None:
        opener, calls = _scripted_opener({})
    resp = S.handle_post(
        table,
        cfg,
        kw.pop("path", "/v1/chat/completions"),
        list(kw.pop("headers", ())),
        body,
        opener,
        pressure=kw.pop("pressure", None),
        override=kw.pop("override", False),
        peer_specs=specs,
        replica_snapshot=kw.pop("replica_snapshot", None),
    )
    assert not kw, f"unused kwargs: {kw}"
    return resp, calls


def _header(resp, name: str):
    lowered = name.lower()
    for key, value in resp.headers:
        if key.lower() == lowered:
            return value
    return None


# ============================================================================
# (1) pressure forwards instead of shedding
# ============================================================================


@pytest.mark.parametrize("requested", ["cortex", "main", "hard"])
def test_busy_with_a_selectable_peer_forwards(requested) -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True, busy=True), _state(_THOR_ORIGIN))
    resp, calls = _post(
        table, cfg, specs, _body(requested), pressure=_HIGH_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR_ORIGIN
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_LOCAL_BUSY_FORWARDED
    # No X-Lobes-Served-By: this box did NOT serve it.
    assert _header(resp, S.SERVED_BY_HEADER) is None
    # A single attempt ⇒ no attempts header (the t7 header set is preserved).
    assert _header(resp, S.ROUTE_ATTEMPTS_HEADER) is None
    assert len(calls) == 1
    assert calls[0].backend.base_url == _THOR_ORIGIN
    # The peer's own inbound key travels; the caller's Authorization never does.
    assert ("Authorization", f"Bearer {_THOR_KEY}") in calls[0].headers
    # And the body was rewritten to the id the peer actually serves.
    assert json.loads(calls[0].body)["model"] == _CORTEX_ID


def test_busy_forward_is_placed_by_the_raw_served_id_too() -> None:
    # The raw id is not a tier alias, so it never reaches the pressure branch —
    # this pins that it is nevertheless pooled and forwarded off a busy local
    # replica, which is the shape every deployed consumer actually sends (c31).
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True, busy=True), _state(_THOR_ORIGIN))
    resp, calls = _post(
        table, cfg, specs, _body(_CORTEX_ID), pressure=_HIGH_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR_ORIGIN
    assert len(calls) == 1


def test_busy_with_no_selectable_peer_keeps_the_pre_pool_429() -> None:
    # The byte-for-byte comparison the plan asks for: the pooled 429 and the
    # UNPOOLED 429 differ by exactly one header — the honest route reason.
    pooled_table, cfg, specs = _build(_pool_env())
    plain_table, plain_cfg, plain_specs = _build(_base_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, busy=True), _state(_THOR_ORIGIN, ready=False)
    )

    pooled, pooled_calls = _post(
        pooled_table,
        cfg,
        specs,
        _body("cortex"),
        pressure=_HIGH_PRESSURE,
        replica_snapshot=snapshot,
    )
    plain, plain_calls = _post(
        plain_table, plain_cfg, plain_specs, _body("cortex"), pressure=_HIGH_PRESSURE
    )

    assert pooled.status == plain.status == 429
    assert pooled.body == plain.body
    assert pooled_calls == []
    assert plain_calls == []
    assert pooled.headers == [(S.ROUTE_REASON_HEADER, REASON_NONE)] + plain.headers
    assert _header(pooled, "Retry-After") == str(S.BUSY_RETRY_AFTER_SECONDS)
    assert _header(pooled, "X-Lobes-Tier-Reason") == "busy"
    assert _header(plain, S.ROUTE_REASON_HEADER) is None


def test_busy_with_an_incompatible_peer_is_not_forwarded() -> None:
    # A declared replica whose live fingerprint does not match is listed, never
    # pooled (spec c13/h11) — so it cannot absorb a shed either.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, busy=True), _state(_THOR_ORIGIN, compatible=False)
    )
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_HIGH_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 429
    assert calls == []
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_NONE


def test_busy_peer_that_is_itself_busy_is_not_forwarded_to() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True, busy=True), _state(_THOR_ORIGIN, busy=True))
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_HIGH_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 429
    assert calls == []


def test_override_serves_locally_and_never_forwards() -> None:
    # X-Lobes-Override forces the requested tier despite pressure; it must not
    # become a licence to forward. With the local replica idle it wins outright.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True), _state(_THOR_ORIGIN))
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        pressure=_HIGH_PRESSURE,
        override=True,
        replica_snapshot=snapshot,
    )
    assert resp.status == 200
    assert len(calls) == 1
    assert calls[0].backend.base_url == _LOCAL_URL
    assert _header(resp, S.SERVED_BY_HEADER) == _SPARK_ORIGIN
    assert _header(resp, "X-Lobes-Tier-Reason") == "manual_override"


def test_hand_is_the_servable_floor_under_pressure_and_never_forwards() -> None:
    # An explicit `minor`/`cheap` request is never shed (#85's floor), so it
    # never reaches the busy dispatch at all — and the primary's pool must not
    # drag it off-box.
    table, cfg, specs = _build(
        _pool_env(
            HAND_BASE_URL="http://vllm-hand:8000", HAND_SERVED_NAME="LiquidAI/LFM2.5-1.2B-Instruct"
        )
    )
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("minor"),
        pressure=_HIGH_PRESSURE,
        replica_snapshot=_snapshot(_state(_LOCAL_URL, local=True, busy=True), _state(_THOR_ORIGIN)),
    )
    assert resp.status == 200
    assert calls[0].backend.base_url == "http://vllm-hand:8000"
    assert _header(resp, S.ROUTE_REASON_HEADER) is None


def test_infeasible_role_still_404s_before_the_busy_forward() -> None:
    # The hardware feasibility gate outranks pressure AND the pool: a dropped
    # role is an absolute fact, not a load condition.
    table, cfg, specs = _build(_pool_env(PRIMARY_FEASIBLE="false"))
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        pressure=_HIGH_PRESSURE,
        replica_snapshot=_snapshot(_state(_THOR_ORIGIN)),
    )
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["code"] == "role_infeasible"
    assert calls == []


# ============================================================================
# (2) at most ONE forward per request
# ============================================================================


def test_both_boxes_busy_produces_exactly_one_forward_and_one_429() -> None:
    # The receiver is simulated by a peer that sheds too. Its 429 is ITS
    # verdict under ITS policy (#85) and rides straight back — the forwarder
    # neither retries it locally (which pressure just forbade) nor forwards it
    # onward (which would be the ping-pong c35/h27 rules out).
    table, cfg, specs = _build(_two_peer_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, busy=True),
        _state(_THOR_ORIGIN, running=0),
        _state(_ORIN_ORIGIN, running=1),
    )
    opener, calls = _scripted_opener({_THOR_ORIGIN: 429, _ORIN_ORIGIN: 429})
    resp, _ = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        pressure=_HIGH_PRESSURE,
        opener=opener,
        calls=calls,
        replica_snapshot=snapshot,
    )
    assert resp.status == 429
    assert len(calls) == 1  # never two forwards
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR_ORIGIN
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_LOCAL_BUSY_FORWARDED


def test_a_peers_4xx_is_relayed_and_never_retried() -> None:
    table, cfg, specs = _build(_two_peer_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=9),
        _state(_THOR_ORIGIN, running=0),
        _state(_ORIN_ORIGIN, running=1),
    )
    opener, calls = _scripted_opener({_THOR_ORIGIN: 400, _ORIN_ORIGIN: 400})
    resp, _ = _post(
        table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls, replica_snapshot=snapshot
    )
    assert resp.status == 400
    assert len(calls) == 1


def test_a_marked_arrival_under_local_pressure_gets_the_local_429() -> None:
    # Single hop (c4/h4): a request a peer already forwarded is served HERE or
    # refused. Under local pressure that means THIS box's 429 — the receiver
    # applying its own policy — with zero outbound sockets.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True, busy=True), _state(_THOR_ORIGIN))
    resp, calls = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        headers=[(S.PROXIED_HEADER, "primary")],
        pressure=_HIGH_PRESSURE,
        replica_snapshot=snapshot,
    )
    assert resp.status == 429
    assert calls == []
    assert _header(resp, "X-Lobes-Tier-Reason") == "busy"
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_SOLE_READY


def test_a_marked_arrival_is_served_locally_when_not_busy() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True, running=9), _state(_THOR_ORIGIN))
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
    assert calls[0].backend.base_url == _LOCAL_URL
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_SOLE_READY


# ============================================================================
# (3) pre-dispatch retry; a committed 2xx is never replayed
# ============================================================================


def test_first_replica_refuses_then_the_next_is_tried_once() -> None:
    table, cfg, specs = _build(_two_peer_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=9),
        _state(_THOR_ORIGIN, running=0),
        _state(_ORIN_ORIGIN, running=1),
    )
    opener, calls = _scripted_opener({_THOR_ORIGIN: S.UpstreamError("connection refused")})
    resp, _ = _post(
        table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert [c.backend.base_url for c in calls] == [_THOR_ORIGIN, _ORIN_ORIGIN]
    assert _header(resp, S.PROXIED_BY_HEADER) == _ORIN_ORIGIN
    assert _header(resp, S.ROUTE_ATTEMPTS_HEADER) == "2"


@pytest.mark.parametrize(
    "failure", [S.UpstreamError("timed out"), 500, 503], ids=["timeout", "500", "503"]
)
def test_every_pre_dispatch_failure_class_retries(failure) -> None:
    table, cfg, specs = _build(_two_peer_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=9),
        _state(_THOR_ORIGIN, running=0),
        _state(_ORIN_ORIGIN, running=1),
    )
    opener, calls = _scripted_opener({_THOR_ORIGIN: failure})
    resp, _ = _post(
        table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert len(calls) == 2


def test_the_local_replica_is_an_ordinary_retry_candidate() -> None:
    # The local owner is dialed first (idle ⇒ locality wins) and, when it
    # refuses PRE-DISPATCH, the pool tries the peer rather than 503ing — a
    # single-owner deployment's one shot becomes a real second chance.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True), _state(_THOR_ORIGIN, running=1))
    opener, calls = _scripted_opener({_LOCAL_URL: S.UpstreamError("connection refused")})
    resp, _ = _post(
        table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert [c.backend.base_url for c in calls] == [_LOCAL_URL, _THOR_ORIGIN]
    assert _header(resp, S.PROXIED_BY_HEADER) == _THOR_ORIGIN
    assert _header(resp, S.ROUTE_ATTEMPTS_HEADER) == "2"


def test_no_replica_is_dispatched_to_twice() -> None:
    table, cfg, specs = _build(_two_peer_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True),
        _state(_THOR_ORIGIN),
        _state(_ORIN_ORIGIN),
    )
    refuse = S.UpstreamError("connection refused")
    opener, calls = _scripted_opener(
        {_LOCAL_URL: refuse, _THOR_ORIGIN: refuse, _ORIN_ORIGIN: refuse}
    )
    resp, _ = _post(
        table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls, replica_snapshot=snapshot
    )
    dialed = [c.backend.base_url for c in calls]
    assert sorted(dialed) == sorted([_LOCAL_URL, _THOR_ORIGIN, _ORIN_ORIGIN])
    assert len(dialed) == len(set(dialed))
    assert resp.status == 503


def test_all_replicas_down_503s_and_lists_every_attempt() -> None:
    table, cfg, specs = _build(_two_peer_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True),
        _state(_THOR_ORIGIN),
        _state(_ORIN_ORIGIN),
    )
    opener, calls = _scripted_opener(
        {
            _LOCAL_URL: S.UpstreamError("primary: connection refused"),
            _THOR_ORIGIN: S.UpstreamError("peer:primary: connection refused"),
            _ORIN_ORIGIN: 500,
        }
    )
    resp, _ = _post(
        table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls, replica_snapshot=snapshot
    )
    assert resp.status == 503
    error = json.loads(resp.body)["error"]
    assert error["type"] == "backend_unavailable"
    joined = " | ".join(error["attempts"])
    for origin in (_LOCAL_URL, _THOR_ORIGIN, _ORIN_ORIGIN):
        assert origin in joined  # every attempt names the replica it dialed
    assert "HTTP 500" in joined
    assert _header(resp, "Retry-After") == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_NONE
    assert _header(resp, S.ROUTE_ATTEMPTS_HEADER) == "3"


def test_all_replicas_down_under_pressure_503s_not_429s() -> None:
    # "429 when nothing is FREE, 503 when nothing is UP" — the two are
    # different facts and the caller is told which one it hit.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True, busy=True), _state(_THOR_ORIGIN))
    opener, calls = _scripted_opener({_THOR_ORIGIN: S.UpstreamError("connection refused")})
    resp, _ = _post(
        table,
        cfg,
        specs,
        _body("cortex"),
        pressure=_HIGH_PRESSURE,
        opener=opener,
        calls=calls,
        replica_snapshot=snapshot,
    )
    assert resp.status == 503
    assert len(calls) == 1  # the local replica was NOT dialed: pressure forbade it
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"


def test_a_2xx_that_drops_mid_stream_is_never_retried() -> None:
    table, cfg, specs = _build(_two_peer_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=9),
        _state(_THOR_ORIGIN, running=0),
        _state(_ORIN_ORIGIN, running=1),
    )
    opener, calls = _scripted_opener({_THOR_ORIGIN: _FakeUpstream(200, chunks=[b"data: a\n\n"])})
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
    assert resp.upstream.read(4096) == b""  # the peer dropped mid-stream
    assert len(calls) == 1  # exactly one open_upstream after the 2xx


def test_a_peer_404_is_terminal_and_never_retried() -> None:
    table, cfg, specs = _build(_two_peer_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, running=9),
        _state(_THOR_ORIGIN, running=0),
        _state(_ORIN_ORIGIN, running=1),
    )
    opener, calls = _scripted_opener({_THOR_ORIGIN: _FakeUpstream(404, body=b'{"error":{}}')})
    resp, _ = _post(
        table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls, replica_snapshot=snapshot
    )
    assert resp.status == 404
    assert len(calls) == 1


def test_unpooled_owner_down_keeps_the_pre_pool_503() -> None:
    table, cfg, specs = _build(_base_env())
    opener, calls = _scripted_opener({_LOCAL_URL: S.UpstreamError("connection refused")})
    resp, _ = _post(
        table, cfg, specs, _body(_CORTEX_ID), opener=opener, calls=calls, replica_snapshot=None
    )
    assert resp.status == 503
    assert json.loads(resp.body)["error"]["message"].startswith("the backend serving this model")
    assert _header(resp, S.ROUTE_REASON_HEADER) is None
    assert _header(resp, S.ROUTE_ATTEMPTS_HEADER) is None


def test_a_single_successful_attempt_carries_no_attempts_header() -> None:
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True), _state(_THOR_ORIGIN))
    resp, calls = _post(table, cfg, specs, _body(_CORTEX_ID), replica_snapshot=snapshot)
    assert resp.status == 200
    assert len(calls) == 1
    assert _header(resp, S.ROUTE_ATTEMPTS_HEADER) is None
    assert _header(resp, S.SERVED_BY_HEADER) == _SPARK_ORIGIN


# ============================================================================
# (4) the live surfaces: ReplicaCache construction, serve() and /capabilities
# ============================================================================


def _capabilities_fingerprint(**over) -> dict:
    fingerprint = {
        "served_id": _CORTEX_ID,
        "max_model_len": 262144,
        "runtime": "vllm",
        "quantization": "compressed-tensors",
        "kv_cache_dtype": "auto",
        "reasoning_parser": "qwen3",
        "tool_parser": "qwen3_coder",
        "speculative_config": "mtp",
    }
    fingerprint.update(over)
    return fingerprint


def _fake_urlopen(*, peer_busy: bool = False, peer_health: str = "ok", peer_running: int = 0):
    """A probe seam standing in for the whole mesh: this box's own /v1/models
    and /metrics, and the peer gateway's /status + /capabilities."""
    seen: list[str] = []

    def urlopen(url, timeout, api_key):
        seen.append(url)
        if url == f"{_LOCAL_URL}/v1/models":
            return (
                200,
                json.dumps(
                    {"data": [{"id": _CORTEX_ID, "max_model_len": 262144, "owned_by": "vllm"}]}
                ).encode(),
            )
        if url == f"{_LOCAL_URL}/metrics":
            return 200, b"vllm:num_requests_running 0.0\n"
        if url == f"{_THOR_ORIGIN}/status":
            return (
                200,
                json.dumps(
                    {
                        "busy": peer_busy,
                        "backends": [
                            {
                                "name": "primary",
                                "served_name": _CORTEX_ID,
                                "health": peer_health,
                                "metrics": {"running": peer_running, "waiting": 0},
                            }
                        ],
                    }
                ).encode(),
            )
        if url == f"{_THOR_ORIGIN}/capabilities":
            return (
                200,
                json.dumps({"cortex": {"fingerprint": _capabilities_fingerprint()}}).encode(),
            )
        return 404, b"{}"

    return urlopen, seen


def test_declared_lane_config_adapts_the_env_suffixes() -> None:
    # The one diverging name: the lanes and the compose passthrough spell the
    # tool parser <PREFIX>_TOOL_CALL_PARSER (the vLLM flag), the fingerprint
    # field is `tool_parser`. Everything else is a plain lowercasing.
    adapted = S.declared_lane_config(
        {
            "QUANTIZATION": "compressed-tensors",
            "KV_CACHE_DTYPE": "fp8",
            "REASONING_PARSER": "qwen3",
            "TOOL_CALL_PARSER": "qwen3_coder",
            "SPECULATIVE_CONFIG": '{"method":"mtp"}',
        }
    )
    assert adapted == {
        "quantization": "compressed-tensors",
        "kv_cache_dtype": "fp8",
        "reasoning_parser": "qwen3",
        "tool_parser": "qwen3_coder",
        "speculative_config": '{"method":"mtp"}',
    }


def test_no_pool_deployment_builds_no_caches() -> None:
    table, _cfg = build_config(_base_env())
    urlopen, seen = _fake_urlopen()
    caches = S.build_replica_caches(table, urlopen=urlopen, start=False)
    assert caches == {}
    assert seen == []  # not one probe fired
    assert S.replica_snapshot_provider(caches) is None
    assert S.replica_role_snapshot(caches) is None


def test_build_replica_caches_refreshes_before_returning() -> None:
    table, _cfg = build_config(
        _pool_env(
            PRIMARY_QUANTIZATION="compressed-tensors",
            PRIMARY_KV_CACHE_DTYPE="fp8",
            PRIMARY_TOOL_CALL_PARSER="qwen3_coder",
        )
    )
    urlopen, seen = _fake_urlopen()
    caches = S.build_replica_caches(table, urlopen=urlopen, start=False)

    assert "primary" in caches
    # The publication job (c33): the co-resident senses lane gets a local-only
    # cache so its fingerprint reaches /capabilities too.
    assert "multimodal" in caches
    assert f"{_LOCAL_URL}/v1/models" in seen
    assert f"{_THOR_ORIGIN}/status" in seen

    states = caches["primary"].current()
    assert [s.origin for s in states] == [_LOCAL_URL, _THOR_ORIGIN]
    local, peer = states
    assert local.local is True
    assert local.ready is True
    assert local.fingerprint.served_id == _CORTEX_ID
    assert local.fingerprint.max_model_len == 262144
    assert local.fingerprint.runtime == "vllm"  # live, from owned_by
    assert local.fingerprint.kv_cache_dtype == "fp8"  # declared
    assert local.fingerprint.tool_parser == "qwen3_coder"  # the adapted suffix
    assert peer.ready is True
    assert peer.compatible is True
    assert peer.busy is False

    # And the dispatch seam reads it by BACKEND name, socket-free.
    provider = S.replica_snapshot_provider(caches)
    assert provider("primary") == states
    assert provider("nonexistent") == ()


def test_a_dropped_lane_gets_no_cache() -> None:
    table, _cfg = build_config(_pool_env(MULTIMODAL_FEASIBLE="false"))
    urlopen, _seen = _fake_urlopen()
    caches = S.build_replica_caches(table, urlopen=urlopen, start=False)
    assert "multimodal" not in caches


def test_a_busy_peer_is_snapshotted_busy_and_sheds_the_forward() -> None:
    table, cfg = build_config(_pool_env())
    specs = S.peer_specs_from_table(table, _pool_env())
    urlopen, _seen = _fake_urlopen(peer_busy=True)
    caches = S.build_replica_caches(table, urlopen=urlopen, start=False)
    provider = S.replica_snapshot_provider(caches)
    assert provider("primary")[1].busy is True

    # A live snapshot, not a hand-built one: the busy peer cannot absorb a shed.
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_HIGH_PRESSURE, replica_snapshot=provider
    )
    assert resp.status == 429
    assert calls == []


def test_capabilities_payload_carries_live_replicas_on_a_pooled_box() -> None:
    env = _pool_env(PRIMARY_QUANTIZATION="compressed-tensors", PRIMARY_KV_CACHE_DTYPE="fp8")
    table, cfg = build_config(env)
    urlopen, _seen = _fake_urlopen(peer_running=3)
    caches = S.build_replica_caches(table, urlopen=urlopen, start=False)
    payload = S.capabilities_payload(
        table, cfg, env=env, replica_snapshot=S.replica_role_snapshot(caches)
    )

    cortex = payload["cortex"]
    assert cortex["fingerprint"]["served_id"] == _CORTEX_ID
    assert cortex["fingerprint"]["kv_cache_dtype"] == "fp8"
    origins = {row["origin"]: row for row in cortex["replicas"]}
    assert set(origins) == {_LOCAL_URL, _THOR_ORIGIN}
    assert origins[_LOCAL_URL]["local"] is True
    assert origins[_LOCAL_URL]["ready"] is True
    assert origins[_THOR_ORIGIN]["ready"] is True
    assert origins[_THOR_ORIGIN]["busy"] is False
    assert origins[_THOR_ORIGIN]["running"] == 3
    assert origins[_THOR_ORIGIN]["compatible"] is True
    # Every existing key keeps its single-owner type and meaning (h1 / c9).
    assert cortex["feasible"] is True
    assert isinstance(cortex["ready"], bool)
    assert isinstance(cortex["loaded"], bool)
    assert "hosted_by" not in cortex  # a hosted role gains no referral


def test_capabilities_payload_has_no_replicas_key_without_a_pool() -> None:
    env = _base_env()
    table, cfg = build_config(env)
    caches = S.build_replica_caches(table, urlopen=_fake_urlopen()[0], start=False)
    with_wiring = S.capabilities_payload(
        table, cfg, env=env, replica_snapshot=S.replica_role_snapshot(caches)
    )
    pre_pool = S.capabilities_payload(table, cfg, env=env)
    assert with_wiring == pre_pool
    for role in with_wiring:
        assert "replicas" not in with_wiring[role]
        assert "fingerprint" not in with_wiring[role]


def test_serve_refreshes_and_wires_the_replica_caches_before_binding(monkeypatch) -> None:
    env = _pool_env()
    table, cfg = build_config(env)
    order: list[str] = []
    stub_cache = SimpleNamespace(current=lambda: (), stop=lambda: order.append("stop"))

    def fake_build(tbl, **kwargs):
        order.append("build_replica_caches")
        assert tbl is table
        return {"primary": stub_cache}

    class _StubServer:
        def __init__(self, addr, handler):
            order.append("bind")
            self.handler = handler
            _StubServer.last = handler

        def serve_forever(self):
            order.append("serve")
            raise SystemExit

    monkeypatch.setattr(
        S.ReadinessCache,
        "from_backends",
        lambda backends, **kw: SimpleNamespace(
            refresh=lambda: None, start=lambda: None, current=lambda: {}
        ),
    )
    monkeypatch.setattr(S, "build_replica_caches", fake_build)
    monkeypatch.setattr(S, "ThreadingHTTPServer", _StubServer)
    with pytest.raises(SystemExit):
        S.serve(table, cfg)

    # Refreshed (inside build_replica_caches) BEFORE the socket is bound, and
    # stopped on the way out.
    assert order == ["build_replica_caches", "bind", "serve", "stop"]
    handler = _StubServer.last
    assert handler.replica_caches == {"primary": stub_cache}
    assert handler.replica_snapshot is not None
    assert handler.replica_snapshot("primary") == ()
