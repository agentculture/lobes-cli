"""A SATURATED fleet queues; it does not refuse (deviation ``d5``).

Deviation ``d1`` made engine saturation (``active >= capacity``) a shed
signal inside :func:`lobes.gateway._pressure_policy.decide`, and
``lobes.gateway.server._resolve_tier`` fed this box's own engine state into
it. The live t10 acceptance run measured the consequence on the Spark+Thor
cortex pair: an 8-way concurrent flood of ``model=cortex`` at
``PRIMARY_MAX_ACTIVE=2`` per box served **4 of 8** through the pool (four
HTTP 429s) against **8 of 8** with the pool bypassed.

Capacity was specified as a ROUTING PREFERENCE (spec c5/h4: "gate on capacity
utilisation instead" — of pool *candidacy*), never as admission control.
Turning a burst that previously queued in vLLM's own waiting queue into
refusals is an availability regression no claim in the frame asks for.

``d5`` therefore takes the engine fact OFF the shed path and leaves it
entirely on the SELECTION path, where :func:`lobes.gateway._selection.is_full`
already uses it. The split this file pins down:

* saturation → *selection*: a full replica is not chosen while any replica
  has room, and when none has room the local owner is dialled anyway (vLLM
  queues it) rather than the caller getting a 429;
* genuine pressure → *shedding*: ``swap > 75 %`` sheds exactly as before,
  pooled or not, and ``hand`` remains the servable floor.

The pure policy function's own engine parameters are UNCHANGED and still
tested by ``tests/test_pressure_policy.py``; what changed is that the
gateway's request path no longer supplies them. Both facts are asserted here.
"""

from __future__ import annotations

import inspect

import pytest

from lobes.gateway import server as S
from lobes.gateway._selection import (
    REASON_LOCAL_BUSY_FORWARDED,
    REASON_NONE,
    REASON_PEER_LESS_LOADED,
)

from .test_gateway_pool_pressure import (  # noqa: F401  (fixture reuse, not re-export)
    _LOCAL_URL,
    _THOR_ORIGIN,
    _body,
    _build,
    _header,
    _pool_env,
    _post,
    _snapshot,
    _state,
)

_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}
_IOWAIT_ONLY = {"swap_used_percent": 0.0, "iowait_percent": 90.0}
_SWAP_ONLY = {"swap_used_percent": 90.0, "iowait_percent": 0.0}


# ---------------------------------------------------------------------------
# (1) a saturated fleet QUEUES rather than refuses  — the t10 regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pressure", [_NO_PRESSURE, _IOWAIT_ONLY], ids=["no-pressure", "iowait-only"]
)
def test_every_replica_full_dials_local_instead_of_429(pressure) -> None:
    # The t10 shape reduced to its essentials: capacity 2 on both boxes, both
    # already holding 2. Nothing has room, so nothing is SELECTABLE — and the
    # honest answer is the pre-pool one: dial the local owner and let vLLM's
    # own queue hold the request, exactly as the pool-bypassed baseline did
    # when it served 8/8.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, weight=2.0, running=2),
        _state(_THOR_ORIGIN, weight=2.0, running=1, waiting=1),
    )
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=pressure, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert [c.backend.base_url for c in calls] == [_LOCAL_URL]
    # Honestly marked: no selection happened, so `none` — not a claim that
    # this box won a comparison it never entered.
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_NONE


def test_a_full_local_engine_alone_is_never_a_429() -> None:
    # The narrowest statement of d5: local at capacity, no peer reachable in
    # the snapshot at all, zero host pressure. Before d5 this shed 429 on the
    # engine signal; now it queues locally.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(_state(_LOCAL_URL, local=True, weight=2.0, running=2))
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_NO_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert [c.backend.base_url for c in calls] == [_LOCAL_URL]


def test_a_burst_beyond_fleet_capacity_serves_every_request() -> None:
    # The measured regression itself, in miniature: eight arrivals against a
    # fleet whose two boxes hold two slots each. Four fit; the other four must
    # still be SERVED (queued), not refused. Any 429 here is the t10 defect.
    table, cfg, specs = _build(_pool_env())
    statuses = []
    for arrival in range(8):
        local_active = min(arrival, 2)
        peer_active = min(max(arrival - 2, 0), 2)
        snapshot = _snapshot(
            _state(_LOCAL_URL, local=True, weight=2.0, running=local_active),
            _state(_THOR_ORIGIN, weight=2.0, running=peer_active),
        )
        resp, _calls = _post(
            table, cfg, specs, _body("cortex"), pressure=_NO_PRESSURE, replica_snapshot=snapshot
        )
        statuses.append(resp.status)
    assert statuses == [200] * 8


# ---------------------------------------------------------------------------
# (2) selection still prefers headroom (the t10 routing markers, unregressed)
# ---------------------------------------------------------------------------


def test_a_full_local_engine_still_prefers_a_peer_with_headroom() -> None:
    # `is_full` stays load-bearing in _selection.py: the full local replica is
    # excluded from candidacy, so the peer with room wins. d5 changes the
    # REASON this reports (the local replica lost on capacity, not on a
    # pressure verdict) but never the placement.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, weight=4.0, running=4),
        _state(_THOR_ORIGIN, weight=4.0, running=1),
    )
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_NO_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert [c.backend.base_url for c in calls] == [_THOR_ORIGIN]
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_PEER_LESS_LOADED


def test_the_less_utilised_peer_still_wins_on_capacity_relative_load() -> None:
    # Both boxes have room, so both are selectable; the capacity-relative
    # ranking (t3's whole point) still picks the peer even though it holds
    # MORE active requests, because it has more headroom.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, weight=2.0, running=1),
        _state(_THOR_ORIGIN, weight=8.0, running=2),
    )
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_NO_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert [c.backend.base_url for c in calls] == [_THOR_ORIGIN]
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_PEER_LESS_LOADED


def test_swap_pressure_still_forwards_with_local_busy_forwarded() -> None:
    # The other marker t10 measured. It belongs to the PRESSURE path, which d5
    # leaves intact: under swap the box is genuinely degraded, so it hands the
    # request to a peer and says so.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, weight=2.0, running=0),
        _state(_THOR_ORIGIN, weight=2.0, running=0),
    )
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_SWAP_ONLY, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert [c.backend.base_url for c in calls] == [_THOR_ORIGIN]
    assert _header(resp, S.ROUTE_REASON_HEADER) == REASON_LOCAL_BUSY_FORWARDED


# ---------------------------------------------------------------------------
# (3) genuine pressure still sheds; `hand` is still the floor
# ---------------------------------------------------------------------------


def test_swap_thrash_still_sheds_when_no_replica_can_take_it() -> None:
    # Criterion 3: removing the engine signal must not weaken the swap band.
    # Paging locally, and the only peer is full ⇒ nowhere to go ⇒ 429.
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, weight=2.0, running=0),
        _state(_THOR_ORIGIN, weight=2.0, running=2),
    )
    resp, calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_SWAP_ONLY, replica_snapshot=snapshot
    )
    assert resp.status == 429
    assert calls == []
    assert _header(resp, "X-Lobes-Tier-Reason") == "busy"


def test_swap_thrash_still_sheds_an_unpooled_box() -> None:
    from .test_gateway_pool_pressure import _base_env

    table, cfg, specs = _build(_base_env())
    resp, calls = _post(table, cfg, specs, _body("cortex"), pressure=_SWAP_ONLY)
    assert resp.status == 429
    assert calls == []


@pytest.mark.parametrize("tier", ["hand", "minor", "cheap"])
def test_hand_is_served_with_every_replica_saturated(tier) -> None:
    # Criterion 4. The floor is orthogonal to saturation as it is to pressure.
    table, cfg, specs = _build(
        _pool_env(
            HAND_BASE_URL="http://vllm-hand:8000",
            HAND_SERVED_NAME="LiquidAI/LFM2.5-1.2B-Instruct",
        )
    )
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, weight=2.0, running=2),
        _state(_THOR_ORIGIN, weight=2.0, running=2),
    )
    for pressure in (_NO_PRESSURE, _IOWAIT_ONLY, _SWAP_ONLY):
        resp, calls = _post(
            table, cfg, specs, _body(tier), pressure=pressure, replica_snapshot=snapshot
        )
        assert resp.status == 200, (tier, pressure)
        assert [c.backend.base_url for c in calls] == ["http://vllm-hand:8000"]


# ---------------------------------------------------------------------------
# (4) the mechanism: the request path stops FEEDING engine state to `decide`
# ---------------------------------------------------------------------------


def test_the_request_path_never_passes_engine_state_to_decide(monkeypatch) -> None:
    # d5's implementation choice, pinned. `decide` keeps its engine parameters
    # (they are a pure, documented, side-effect-free part of its contract and
    # tests/test_no_config_byte_identity.py exercises them directly); what
    # changed is that the gateway no longer supplies them, so saturation can
    # never reach the shed verdict through this seam.
    seen: list[dict] = []
    real = S.decide

    def _spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(S, "decide", _spy)
    table, cfg, specs = _build(_pool_env())
    snapshot = _snapshot(
        _state(_LOCAL_URL, local=True, weight=2.0, running=2),
        _state(_THOR_ORIGIN, weight=2.0, running=2),
    )
    resp, _calls = _post(
        table, cfg, specs, _body("cortex"), pressure=_NO_PRESSURE, replica_snapshot=snapshot
    )
    assert resp.status == 200
    assert seen, "the tier path did not consult the pressure policy at all"
    for call in seen:
        assert "engine_active" not in call
        assert "engine_capacity" not in call
        # ...and the d1 pooled carve-out is still supplied.
        assert call.get("pooled") is True


def test_the_engine_plumbing_is_gone_from_the_server_module() -> None:
    # The guidance's anti-misleading rule: dead plumbing that no longer feeds
    # anything is removed rather than left looking load-bearing.
    assert not hasattr(S, "_pooled_engine_state")
    # The BODY only — the docstring names the parameters to record why they
    # are no longer supplied, which is the opposite of misleading.
    body = inspect.getsource(S._resolve_tier).replace(S._resolve_tier.__doc__ or "", "")
    assert "engine_active" not in body
    assert "engine_capacity" not in body
