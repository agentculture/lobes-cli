"""Tests for the member_lane_ids parameter added to list_models_payload.

Every test builds a small RoutingTable by hand (matching the style of
test_gateway_routing.py::test_list_models_payload_shape) so the golden
payload is fully deterministic.
"""

from __future__ import annotations

import json

from lobes.gateway._routing import Backend, RoutingTable, list_models_payload

# ── tiny fixture ────────────────────────────────────────────────────────────

_BACKENDS = (
    Backend("primary", "http://vllm-primary:8000", "cortex-spark"),
    Backend("fallback", "http://vllm-fallback:8000", "cortex-thor"),
)
_TABLE = RoutingTable(
    backends=_BACKENDS,
    default_model="cortex-spark",
    aliases={"fast": "cortex-thor"},
)
_READY = {"primary": True, "fallback": True}

# ── golden: what the UNMODIFIED function must return for the fixture ──────
#
# Frozen by hand from the fixture above — NOT computed by the function under
# test (an import-time call would be self-referential and could not catch a
# regression in the default path). Two backends, both ready, nothing
# infeasible/proxied/pooled/adapted → exactly their served names in table
# order and no more.
_GOLDEN = {
    "object": "list",
    "data": [
        {"id": "cortex-spark", "object": "model", "owned_by": "lobes"},
        {"id": "cortex-thor", "object": "model", "owned_by": "lobes"},
    ],
}


# ── tests ──────────────────────────────────────────────────────────────────


def test_golden_matches_current_payload() -> None:
    """No kwarg and empty kwarg are byte-identical to the frozen golden."""
    golden = json.dumps(_GOLDEN)
    got0 = json.dumps(list_models_payload(_TABLE, _READY))
    assert got0 == golden

    got1 = json.dumps(list_models_payload(_TABLE, _READY, member_lane_ids=()))
    assert got1 == golden

    # Omitted kwarg and explicit empty tuple are byte-identical.
    assert got0 == got1


def test_append_new_member_lane_ids() -> None:
    """member_lane_ids appends entries in order, one per new id."""
    payload = list_models_payload(_TABLE, _READY, member_lane_ids=("cortex-spark", "cortex-spark2"))
    ids = [entry["id"] for entry in payload["data"]]
    # Existing entries first, unchanged order.
    assert ids[:2] == ["cortex-spark", "cortex-thor"]
    # Then one entry per new id.
    assert len(ids) == 3
    assert ids[2] == "cortex-spark2"
    # Entry shape matches the existing convention.
    assert payload["data"][2]["object"] == "model"
    assert payload["data"][2]["owned_by"] == "lobes"


def test_dedup_member_lane_ids_against_existing_and_internal() -> None:
    """De-dup: one id already listed + one duplicate within member_lane_ids."""
    payload = list_models_payload(
        _TABLE, _READY, member_lane_ids=("cortex-spark", "cortex-spark", "cortex-spark2")
    )
    ids = [entry["id"] for entry in payload["data"]]
    # "cortex-spark" already at position 0, so appended "cortex-spark" (dup) is skipped.
    # "cortex-spark2" is new → appended once (not twice from the duplicate).
    assert ids == ["cortex-spark", "cortex-thor", "cortex-spark2"]
    assert len(ids) == 3
