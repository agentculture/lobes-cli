"""h1 — a deployment with NO pool declared is byte-identical to the pre-pool
contract, headers included (capacity-relative pool routing, t7).

Two changes in this plan claim to be inert without peers, and both touch the
response wire:

* **t5** added a new response header, ``X-Lobes-Route-Load``, carrying the
  capacity and utilisation a placement used.
* **t11 / deviation d1** narrowed the shed band: host ``iowait`` alone no
  longer refuses a POOLED request, and :func:`~lobes.gateway._pressure_policy.decide`
  grew a ``pooled`` parameter and an additive ``shed_signal`` return key.

"Both are inert without peers" is a claim, and this module refuses to take it
on trust. Rather than re-asserting today's behaviour against itself, the
expected values below are a **golden captured from the pre-pool code itself** —
commit ``3661a27``, the merge-base this plan branched from, extracted with
``git archive`` into a throwaway tree and driven through the identical request
matrix. Every entry in :data:`_GOLDEN` is what the OLD gateway actually
returned; the tests re-drive the same matrix against the CURRENT gateway and
demand equality of status, the full ordered header list, and the body bytes.

Regenerating the golden is deliberately NOT automated: if a case here starts
failing, the pre-pool contract moved, and that is a finding to adjudicate, not
a fixture to refresh.

Coverage of the matrix: idle / no-pressure / high (both signals) / iowait-only
/ swap-only pressure, across the ``cortex`` alias, the ``senses`` alias, the
raw served id, the ``hand`` floor, a streaming request, a manual override, a
marked (already-proxied) arrival, an owner-down 503, an infeasible-role 404,
and ``GET /status`` with the pressure block warm, busy, and unwired.
"""

from __future__ import annotations

import inspect
import json

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._pressure_policy import decide

_CORTEX_ID = "unsloth/Qwen3.8-27B-NVFP4"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_LOCAL_URL = "http://vllm-primary:8000"

_HIGH = {"swap_used_percent": 90.0, "iowait_percent": 90.0}
_IOWAIT_ONLY = {"swap_used_percent": 0.0, "iowait_percent": 90.0}
_SWAP_ONLY = {"swap_used_percent": 90.0, "iowait_percent": 0.0}
_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}


# --- the no-peer deployment --------------------------------------------------


def _env(**over) -> dict[str, str]:
    """A machine-as-brain box: three local lanes, not one peer key anywhere."""
    env = {
        "PRIMARY_URL": _LOCAL_URL,
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
        "HAND_BASE_URL": "http://vllm-hand:8000",
        "HAND_SERVED_NAME": "LiquidAI/LFM2.5-1.2B-Instruct",
    }
    env.update(over)
    return env


class _Upstream:
    def __init__(self, status=200, body=b'{"ok":1}'):
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body

    def read_all(self):
        return self._body

    def read(self, _n):
        data, self._body = self._body, b""
        return data

    def close(self):
        pass


def _opener_ok(_backend, _path, _body, _headers, *, connect_timeout, read_timeout):
    return _Upstream()


def _opener_down(_backend, _path, _body, _headers, *, connect_timeout, read_timeout):
    raise S.UpstreamError("connection refused")


def _body(model: str, *, stream: bool = False) -> bytes:
    payload: dict = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    if stream:
        payload["stream"] = True
    return json.dumps(payload).encode()


# name -> (env overrides, body, pressure, request headers, opener, override flag)
_CASES = {
    "cortex-idle": ({}, _body("cortex"), None, [], _opener_ok, False),
    "cortex-no-pressure": ({}, _body("cortex"), _NO_PRESSURE, [], _opener_ok, False),
    "cortex-high": ({}, _body("cortex"), _HIGH, [], _opener_ok, False),
    "cortex-iowait-only": ({}, _body("cortex"), _IOWAIT_ONLY, [], _opener_ok, False),
    "cortex-swap-only": ({}, _body("cortex"), _SWAP_ONLY, [], _opener_ok, False),
    "senses-high": ({}, _body("senses"), _HIGH, [], _opener_ok, False),
    "hand-high": ({}, _body("minor"), _HIGH, [], _opener_ok, False),
    "hand-iowait": ({}, _body("minor"), _IOWAIT_ONLY, [], _opener_ok, False),
    "hand-swap": ({}, _body("minor"), _SWAP_ONLY, [], _opener_ok, False),
    "hand-idle": ({}, _body("minor"), _NO_PRESSURE, [], _opener_ok, False),
    "rawid-high": ({}, _body(_CORTEX_ID), _HIGH, [], _opener_ok, False),
    "rawid-idle": ({}, _body(_CORTEX_ID), _NO_PRESSURE, [], _opener_ok, False),
    "stream-idle": ({}, _body("cortex", stream=True), _NO_PRESSURE, [], _opener_ok, False),
    "stream-high": ({}, _body("cortex", stream=True), _HIGH, [], _opener_ok, False),
    "override-high": ({}, _body("cortex"), _HIGH, [], _opener_ok, True),
    "marked-high": (
        {},
        _body("cortex"),
        _HIGH,
        [(S.PROXIED_HEADER, "primary")],
        _opener_ok,
        False,
    ),
    "marked-idle": (
        {},
        _body("cortex"),
        _NO_PRESSURE,
        [(S.PROXIED_HEADER, "primary")],
        _opener_ok,
        False,
    ),
    "owner-down": ({}, _body("cortex"), _NO_PRESSURE, [], _opener_down, False),
    "owner-down-high": ({}, _body("cortex"), _HIGH, [], _opener_down, False),
    "infeasible": (
        {"PRIMARY_FEASIBLE": "false"},
        _body("cortex"),
        _NO_PRESSURE,
        [],
        _opener_ok,
        False,
    ),
    "infeasible-high": (
        {"PRIMARY_FEASIBLE": "false"},
        _body("cortex"),
        _HIGH,
        [],
        _opener_ok,
        False,
    ),
}


def _observe(name: str) -> dict:
    """Drive one matrix case through the CURRENT gateway, socket-free."""
    over, body, pressure, headers, opener, override = _CASES[name]
    env = _env(**over)
    table, cfg = build_config(env)
    resp = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        list(headers),
        body,
        opener,
        pressure=pressure,
        override=override,
        peer_specs=S.peer_specs_from_table(table, env),
    )
    raw = resp.body
    return {
        "status": resp.status,
        "headers": [tuple(h) for h in resp.headers],
        "body": raw.decode() if isinstance(raw, (bytes, bytearray)) else None,
        "upstream": resp.upstream is not None,
    }


def _fake_probe(_url, timeout):  # noqa: ARG001 - signature must match _metrics.probe_backend
    return {"health": "ok", "metrics": {"running": 0, "waiting": 0}}


_STATUS_CASES = {
    "status-warm": _NO_PRESSURE,
    "status-busy": _HIGH,
    "status-none": None,
}


_GOLDEN = {
    "cortex-high": {
        "body": '{"error": {"message": "cortex is under pressure; retry shortly", '
        '"type": "server_busy", "code": "busy"}}',
        "headers": [
            ("Retry-After", "5"),
            ("X-Lobes-Tier-Reason", "busy"),
            ("Content-Type", "application/json"),
        ],
        "status": 429,
        "upstream": False,
    },
    "cortex-idle": {
        "body": None,
        "headers": [("Content-Type", "application/json")],
        "status": 200,
        "upstream": True,
    },
    "cortex-iowait-only": {
        "body": '{"error": {"message": "cortex is under pressure; retry '
        'shortly", "type": "server_busy", "code": "busy"}}',
        "headers": [
            ("Retry-After", "5"),
            ("X-Lobes-Tier-Reason", "busy"),
            ("Content-Type", "application/json"),
        ],
        "status": 429,
        "upstream": False,
    },
    "cortex-no-pressure": {
        "body": None,
        "headers": [
            ("X-Lobes-Tier", "main"),
            ("X-Lobes-Tier-Reason", "default"),
            ("Content-Type", "application/json"),
        ],
        "status": 200,
        "upstream": True,
    },
    "cortex-swap-only": {
        "body": '{"error": {"message": "cortex is under pressure; retry '
        'shortly", "type": "server_busy", "code": "busy"}}',
        "headers": [
            ("Retry-After", "5"),
            ("X-Lobes-Tier-Reason", "busy"),
            ("Content-Type", "application/json"),
        ],
        "status": 429,
        "upstream": False,
    },
    "hand-high": {
        "body": None,
        "headers": [
            ("X-Lobes-Tier", "hand"),
            ("X-Lobes-Tier-Reason", "default"),
            ("Content-Type", "application/json"),
        ],
        "status": 200,
        "upstream": True,
    },
    "hand-idle": {
        "body": None,
        "headers": [
            ("X-Lobes-Tier", "hand"),
            ("X-Lobes-Tier-Reason", "default"),
            ("Content-Type", "application/json"),
        ],
        "status": 200,
        "upstream": True,
    },
    "hand-iowait": {
        "body": None,
        "headers": [
            ("X-Lobes-Tier", "hand"),
            ("X-Lobes-Tier-Reason", "default"),
            ("Content-Type", "application/json"),
        ],
        "status": 200,
        "upstream": True,
    },
    "hand-swap": {
        "body": None,
        "headers": [
            ("X-Lobes-Tier", "hand"),
            ("X-Lobes-Tier-Reason", "default"),
            ("Content-Type", "application/json"),
        ],
        "status": 200,
        "upstream": True,
    },
    "infeasible": {
        "body": '{"error": {"message": "The model `cortex` is not feasible on this '
        "machine \\u2014 its backend (`primary`) is declared "
        "hardware-infeasible by this deployment's per-machine profile and "
        'will never be served here.", "type": "role_infeasible", "code": '
        '"role_infeasible"}}',
        "headers": [("Content-Type", "application/json")],
        "status": 404,
        "upstream": False,
    },
    "infeasible-high": {
        "body": '{"error": {"message": "The model `cortex` is not feasible on '
        "this machine \\u2014 its backend (`primary`) is declared "
        "hardware-infeasible by this deployment's per-machine profile "
        'and will never be served here.", "type": "role_infeasible", '
        '"code": "role_infeasible"}}',
        "headers": [("Content-Type", "application/json")],
        "status": 404,
        "upstream": False,
    },
    "marked-high": {
        "body": '{"error": {"message": "cortex is under pressure; retry shortly", '
        '"type": "server_busy", "code": "busy"}}',
        "headers": [
            ("Retry-After", "5"),
            ("X-Lobes-Tier-Reason", "busy"),
            ("Content-Type", "application/json"),
        ],
        "status": 429,
        "upstream": False,
    },
    "marked-idle": {
        "body": None,
        "headers": [
            ("X-Lobes-Tier", "main"),
            ("X-Lobes-Tier-Reason", "default"),
            ("Content-Type", "application/json"),
        ],
        "status": 200,
        "upstream": True,
    },
    "override-high": {
        "body": None,
        "headers": [
            ("X-Lobes-Tier", "main"),
            ("X-Lobes-Tier-Reason", "manual_override"),
            ("Content-Type", "application/json"),
        ],
        "status": 200,
        "upstream": True,
    },
    "owner-down": {
        "body": '{"error": {"message": "the backend serving this model is unavailable '
        '\\u2014 retry shortly", "type": "backend_unavailable", "attempts": '
        '["connection refused"]}}',
        "headers": [
            ("X-Lobes-Tier", "main"),
            ("X-Lobes-Tier-Reason", "default"),
            ("Retry-After", "5"),
            ("Content-Type", "application/json"),
        ],
        "status": 503,
        "upstream": False,
    },
    "owner-down-high": {
        "body": '{"error": {"message": "cortex is under pressure; retry '
        'shortly", "type": "server_busy", "code": "busy"}}',
        "headers": [
            ("Retry-After", "5"),
            ("X-Lobes-Tier-Reason", "busy"),
            ("Content-Type", "application/json"),
        ],
        "status": 429,
        "upstream": False,
    },
    "rawid-high": {
        "body": None,
        "headers": [("Content-Type", "application/json")],
        "status": 200,
        "upstream": True,
    },
    "rawid-idle": {
        "body": None,
        "headers": [("Content-Type", "application/json")],
        "status": 200,
        "upstream": True,
    },
    "senses-high": {
        "body": '{"error": {"message": "senses is under pressure; retry shortly", '
        '"type": "server_busy", "code": "busy"}}',
        "headers": [
            ("Retry-After", "5"),
            ("X-Lobes-Tier-Reason", "busy"),
            ("Content-Type", "application/json"),
        ],
        "status": 429,
        "upstream": False,
    },
    "status-busy": {
        "backends": [
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "primary",
                "served_name": "unsloth/Qwen3.8-27B-NVFP4",
                "task": "generate",
            },
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "hand",
                "served_name": "LiquidAI/LFM2.5-1.2B-Instruct",
                "task": "generate",
            },
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "multimodal",
                "served_name": "coolthor/gemma-4-12B-it-NVFP4A16",
                "task": "generate",
            },
        ],
        "busy": {"running": 0, "waiting": 0},
        "default_model": "unsloth/Qwen3.8-27B-NVFP4",
        "endpoints": [
            "GET /health",
            "GET /status",
            "GET /v1/models",
            "GET /v1/models/supported",
            "GET /capabilities",
            "POST /v1/chat/completions",
            "POST /v1/completions",
        ],
        "object": "lobes.fleet_status",
        "pressure": {
            "iowait_percent": 90.0,
            "mode": "busy",
            "reason": "pressure",
            "shed": True,
            "swap_used_percent": 90.0,
        },
    },
    "status-none": {
        "backends": [
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "primary",
                "served_name": "unsloth/Qwen3.8-27B-NVFP4",
                "task": "generate",
            },
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "hand",
                "served_name": "LiquidAI/LFM2.5-1.2B-Instruct",
                "task": "generate",
            },
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "multimodal",
                "served_name": "coolthor/gemma-4-12B-it-NVFP4A16",
                "task": "generate",
            },
        ],
        "busy": {"running": 0, "waiting": 0},
        "default_model": "unsloth/Qwen3.8-27B-NVFP4",
        "endpoints": [
            "GET /health",
            "GET /status",
            "GET /v1/models",
            "GET /v1/models/supported",
            "GET /capabilities",
            "POST /v1/chat/completions",
            "POST /v1/completions",
        ],
        "object": "lobes.fleet_status",
    },
    "status-warm": {
        "backends": [
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "primary",
                "served_name": "unsloth/Qwen3.8-27B-NVFP4",
                "task": "generate",
            },
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "hand",
                "served_name": "LiquidAI/LFM2.5-1.2B-Instruct",
                "task": "generate",
            },
            {
                "health": "ok",
                "metrics": {"running": 0, "waiting": 0},
                "name": "multimodal",
                "served_name": "coolthor/gemma-4-12B-it-NVFP4A16",
                "task": "generate",
            },
        ],
        "busy": {"running": 0, "waiting": 0},
        "default_model": "unsloth/Qwen3.8-27B-NVFP4",
        "endpoints": [
            "GET /health",
            "GET /status",
            "GET /v1/models",
            "GET /v1/models/supported",
            "GET /capabilities",
            "POST /v1/chat/completions",
            "POST /v1/completions",
        ],
        "object": "lobes.fleet_status",
        "pressure": {
            "iowait_percent": 0.0,
            "mode": "warm",
            "reason": "default",
            "shed": False,
            "swap_used_percent": 0.0,
        },
    },
    "stream-high": {
        "body": '{"error": {"message": "cortex is under pressure; retry shortly", '
        '"type": "server_busy", "code": "busy"}}',
        "headers": [
            ("Retry-After", "5"),
            ("X-Lobes-Tier-Reason", "busy"),
            ("Content-Type", "application/json"),
        ],
        "status": 429,
        "upstream": False,
    },
    "stream-idle": {
        "body": None,
        "headers": [
            ("X-Lobes-Tier", "main"),
            ("X-Lobes-Tier-Reason", "default"),
            ("Content-Type", "application/json"),
        ],
        "status": 200,
        "upstream": True,
    },
}

# The complete set of ``X-Lobes-*`` response header names the PRE-POOL gateway
# could emit on a no-peer box, derived from the golden above. A new header name
# appearing here is exactly the h1 regression this module exists to catch.
_PRE_POOL_LOBES_HEADERS = frozenset(
    name
    for case in _GOLDEN.values()
    if isinstance(case, dict) and "headers" in case
    for name, _value in case["headers"]
    if name.lower().startswith("x-lobes-")
)


# ---------------------------------------------------------------------------
# (1) the golden: every no-peer response, byte for byte
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_no_peer_response_is_byte_identical_to_the_pre_pool_golden(case: str) -> None:
    # Status, the FULL ordered header list, and the body bytes — not a subset,
    # and not "no pool headers present". Header ORDER is part of the contract
    # here because the golden records what actually went on the wire.
    assert _observe(case) == _GOLDEN[case]


@pytest.mark.parametrize("case", sorted(_STATUS_CASES))
def test_status_is_byte_identical_to_the_pre_pool_golden(case: str) -> None:
    # t5 made `/status` the capacity discovery path a peer probes. On a box
    # that declares no capacity the key must be absent, not zero — an absent
    # key reads as "uncalibrated" at a peer, a published 0 would read as a
    # measured capacity of nothing.
    table, cfg = build_config(_env())
    payload = S.fleet_status_payload(table, cfg, pressure=_STATUS_CASES[case], probe=_fake_probe)
    assert json.loads(json.dumps(payload)) == _GOLDEN[case]
    for row in payload["backends"]:
        assert "max_active" not in row  # the t5 capacity key, by any spelling
        assert set(row) == {"name", "task", "served_name", "health", "metrics"}


def test_the_golden_actually_exercises_the_shapes_it_claims_to() -> None:
    # A golden that silently lost its interesting rows would pass vacuously.
    statuses = {_GOLDEN[c]["status"] for c in _CASES}
    assert statuses == {200, 404, 429, 503}
    assert len(_CASES) == 21
    assert set(_GOLDEN) == set(_CASES) | set(_STATUS_CASES)


def test_no_new_x_lobes_header_appears_without_a_pool() -> None:
    # The set-level statement of the same guarantee, so a NEW header added by
    # some future task fails loudly even if it lands on a shape the matrix
    # above does not enumerate.
    assert _PRE_POOL_LOBES_HEADERS == {"X-Lobes-Tier", "X-Lobes-Tier-Reason"}
    for case in _CASES:
        emitted = {
            name for name, _v in _observe(case)["headers"] if name.lower().startswith("x-lobes-")
        }
        assert emitted <= _PRE_POOL_LOBES_HEADERS, case
    # And in particular none of the pool markers, named explicitly.
    pool_markers = {
        S.SERVED_BY_HEADER,
        S.ROUTE_REASON_HEADER,
        S.ROUTE_LOAD_HEADER,
        S.ROUTE_ATTEMPTS_HEADER,
        S.PROXIED_BY_HEADER,
    }
    assert pool_markers.isdisjoint(_PRE_POOL_LOBES_HEADERS)


# ---------------------------------------------------------------------------
# (2) the two mechanisms behind the claim, proven rather than read
# ---------------------------------------------------------------------------


def test_decide_defaults_to_unpooled_in_signature_and_in_behaviour() -> None:
    # The claim is that `pooled` defaults False, so every pre-d1 call site —
    # and every single-box deployment — decides exactly as before. Asserted
    # twice: the default itself, and the decision it produces.
    param = inspect.signature(decide).parameters["pooled"]
    assert param.default is False
    assert param.kind is inspect.Parameter.KEYWORD_ONLY

    unpooled = decide(0.0, 90.0, "main")
    assert unpooled["shed"] is True
    assert unpooled["shed_signal"] == "iowait"
    assert unpooled == decide(0.0, 90.0, "main", pooled=False)
    # ...and the carve-out is real, so the assertion above is not vacuous.
    assert decide(0.0, 90.0, "main", pooled=True)["shed"] is False


def test_the_route_load_header_is_unreachable_without_a_pool(monkeypatch) -> None:
    # t5 stamps X-Lobes-Route-Load from `_pool_marker_headers` only. Proven by
    # BOOBY-TRAPPING both seams: if any no-peer request reached them the whole
    # matrix would blow up. (`_stamp_pool_headers` is trapped too, since it is
    # the other half of the same marker set.)
    def _explode(*_a, **_kw):
        raise AssertionError("a no-peer request reached the pool marker path")

    monkeypatch.setattr(S, "_route_load_header", _explode)
    monkeypatch.setattr(S, "_pool_marker_headers", _explode)
    monkeypatch.setattr(S, "_stamp_pool_headers", _explode)

    for case in _CASES:
        assert _observe(case) == _GOLDEN[case]


def test_the_booby_trap_is_not_vacuous(monkeypatch) -> None:
    # The counter-test for the one above: with a pool declared and a live
    # snapshot, the trapped seam IS reached — so a green run there means
    # "never called", not "never callable".
    from lobes.gateway._replicas import ReplicaState

    def _explode(*_a, **_kw):
        raise AssertionError("reached")

    monkeypatch.setattr(S, "_pool_marker_headers", _explode)
    env = _env(
        PRIMARY_PEER_ORIGINS="http://thor.local:8001",
        PRIMARY_PEER_API_KEYS="sk-thor",
        GATEWAY_SELF_ORIGIN="http://spark.local:8001",
    )
    table, cfg = build_config(env)
    state = ReplicaState(
        origin=_LOCAL_URL,
        local=True,
        ready=True,
        busy=False,
        health="ok",
        running=0,
        waiting=0,
        fingerprint=None,
        compatible=True,
        reason="",
        last_seen=1.0,
        weight=1.0,
    )
    peer_specs = S.peer_specs_from_table(table, env)
    body = _body("cortex")
    with pytest.raises(AssertionError, match="reached"):
        S.handle_post(
            table,
            cfg,
            "/v1/chat/completions",
            [],
            body,
            _opener_ok,
            pressure=_NO_PRESSURE,
            peer_specs=peer_specs,
            replica_snapshot=lambda _name: (state,),
        )


def test_a_no_peer_box_needs_no_new_env_key() -> None:
    # h5/c14: the change is invisible to a single-box operator. The bare env
    # parses, and every capacity field resolves to its "nothing declared"
    # value without raising.
    _table, cfg = build_config(_env())
    assert cfg.local_capacities == {}
    assert cfg.capacity_kill_switch is False


# ---------------------------------------------------------------------------
# (3) the floor: `hand` is never shed, under any band d1 left standing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["hand", "minor", "cheap"])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize(
    "pressure",
    [_HIGH, _IOWAIT_ONLY, _SWAP_ONLY, _NO_PRESSURE],
    ids=["both", "iowait-only", "swap-only", "none"],
)
@pytest.mark.parametrize(
    "engine",
    [(None, None), (0.0, 4.0), (4.0, 4.0), (99.0, 4.0)],
    ids=["unknown", "idle", "full", "over"],
)
def test_hand_is_never_shed_under_any_d1_shed_band(tier, pooled, pressure, engine) -> None:
    # d1 rewrote the shed band into three signals (swap / engine / iowait) and
    # added a pooled carve-out. The floor is orthogonal to all of it: `hand`
    # and both back-compat spellings are served in every cell of the product.
    active, capacity = engine
    result = decide(
        pressure["swap_used_percent"],
        pressure["iowait_percent"],
        tier,
        pooled=pooled,
        engine_active=active,
        engine_capacity=capacity,
    )
    assert result["shed"] is False
    assert result["servable_tier"] == "hand"
    assert result["reason"] == "default"


def test_the_floor_matrix_is_not_vacuous() -> None:
    # Same cells, a full tier: every band that leaves `hand` untouched above
    # does shed `main`, so the parametrisation is exercising real sheds.
    assert decide(90.0, 0.0, "main", pooled=True)["shed_signal"] == "swap"
    assert (
        decide(0.0, 0.0, "main", pooled=True, engine_active=4.0, engine_capacity=4.0)["shed_signal"]
        == "engine"
    )
    assert decide(0.0, 90.0, "main", pooled=False)["shed_signal"] == "iowait"
