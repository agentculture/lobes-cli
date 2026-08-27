"""Tests for the additive replica-pool capabilities keys (#199, task t6).

``annotate_replicas`` (lobes/roles.py) is the sibling of
``annotate_peer_referrals``: it adds the ADDITIVE per-role ``fingerprint``/
``replicas`` keys the cortex replica pool needs, without touching any
existing key's type or meaning. Two provenances are exercised here:

* **offline** (``snapshot=None``) — the CLI's own path: a declared-only view
  built from ``RoutingTable.replica_origins``/``lane_fingerprints`` alone,
  every live field honestly ``None``.
* **live** (``snapshot`` supplied) — what a future gateway wiring (t8) will
  pass: a per-role tuple of :class:`~lobes.gateway._replicas.ReplicaState`,
  rendered straight through.

The byte-identical guarantee (h1/c9: a no-pool deployment is unaffected) is
proven the same way ``test_roles_proxied.py`` proves it for referrals — build
a payload two ways and diff the JSON.
"""

from __future__ import annotations

import copy
import dataclasses
import json

from lobes.gateway._config import build_config
from lobes.gateway._replicas import UNCALIBRATED_WEIGHT, Fingerprint, ReplicaState
from lobes.gateway._routing import Backend, RoutingTable
from lobes.roles import ROLES, annotate_peer_referrals, annotate_replicas, build_role_registry

_CORTEX_ID = "unsloth/Qwen3.8-27B-NVFP4"
_GATEWAY_URL = "http://localhost:8000"
_SPARK_ORIGIN = "http://spark.local:8001"
_THOR_ORIGIN = "http://thor.local:8000"


def _table(**over) -> RoutingTable:
    backend = Backend(name="primary", base_url="http://vllm-primary:8000", served_name=_CORTEX_ID)
    base = dict(
        backends=(backend,),
        default_model=_CORTEX_ID,
        aliases={},
    )
    base.update(over)
    return RoutingTable(**base)


def _fingerprint(**over) -> Fingerprint:
    base = dict(
        served_id=_CORTEX_ID,
        max_model_len=262144,
        runtime="vllm",
        quantization="nvfp4",
        kv_cache_dtype="fp8",
        reasoning_parser="qwen3",
        tool_parser="qwen3_coder",
        speculative_config="dspark",
    )
    base.update(over)
    return Fingerprint(**base)


def _state(**over) -> ReplicaState:
    base = dict(
        origin="http://x:8000",
        local=False,
        ready=True,
        busy=False,
        health="ok",
        running=0,
        waiting=0,
        fingerprint=None,
        compatible=True,
        reason="",
        last_seen=0.0,
        weight=1.0,
    )
    base.update(over)
    return ReplicaState(**base)


def _base_payload(table: RoutingTable) -> dict[str, dict]:
    _table_ignored, cfg = build_config({})
    registry = build_role_registry(table, cfg, gateway_url=_GATEWAY_URL)
    return {role: dataclasses.asdict(registry[role]) for role in ROLES}


# ============================================================================
# Byte-identical guarantee: no replica pool declared, no snapshot
# ============================================================================


def test_no_pool_payload_has_no_replicas_key_and_is_byte_identical() -> None:
    table = _table()  # replica_origins defaults to {}
    payload = _base_payload(table)
    before = copy.deepcopy(payload)

    after = annotate_replicas(copy.deepcopy(payload), table)

    assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True)
    for role in ROLES:
        assert "replicas" not in after[role]
        assert "fingerprint" not in after[role]


def test_no_pool_payload_matches_pre_replica_oracle_through_referral_annotator() -> None:
    """annotate_peer_referrals + annotate_replicas together, with no replica
    pool declared, must equal annotate_peer_referrals alone."""
    table = _table(
        infeasible=("multimodal",),
        peer_origins={"multimodal": _THOR_ORIGIN},
    )
    payload = _base_payload(table)
    via_referral_only = annotate_peer_referrals(copy.deepcopy(payload), table)
    via_both = annotate_replicas(
        annotate_peer_referrals(copy.deepcopy(payload), table),
        table,
    )
    assert json.dumps(via_both, sort_keys=True) == json.dumps(via_referral_only, sort_keys=True)


# ============================================================================
# Offline (declared-only) replica view
# ============================================================================


def test_two_declared_replicas_offline_view() -> None:
    table = _table(
        replica_origins={"primary": (_THOR_ORIGIN,)},
        self_origin=_SPARK_ORIGIN,
        lane_fingerprints={
            "primary": {
                "QUANTIZATION": "nvfp4",
                "KV_CACHE_DTYPE": "fp8",
                "REASONING_PARSER": "qwen3",
                "TOOL_PARSER": "qwen3_coder",
                "SPECULATIVE_CONFIG": "dspark",
            }
        },
    )
    payload = _base_payload(table)
    annotate_replicas(payload, table)

    cortex = payload["cortex"]
    # existing keys keep type/meaning
    assert cortex["feasible"] is True
    assert "hosted_by" not in cortex
    assert "proxied" not in cortex

    rows = cortex["replicas"]
    assert [r["origin"] for r in rows] == [_SPARK_ORIGIN, _THOR_ORIGIN]
    assert rows[0]["local"] is True
    assert rows[1]["local"] is False
    for row in rows:
        assert row["ready"] is None
        assert row["busy"] is None
        assert row["compatible"] is None
        assert row["reason"] == "not probed (offline)"
    # the local row carries the declared fingerprint; the undeclared peer does not
    assert rows[0]["fingerprint"]["served_id"] == _CORTEX_ID
    assert rows[1]["fingerprint"] is None

    fp = cortex["fingerprint"]
    assert fp["served_id"] == _CORTEX_ID
    assert fp["quantization"] == "nvfp4"
    assert fp["kv_cache_dtype"] == "fp8"
    # RUNTIME has no <PREFIX>_RUNTIME env knob (t2) — never invented from the
    # catalog (c33/h25), so it stays honestly unknown.
    assert fp["runtime"] == "unknown"


def test_offline_view_reports_no_capacity_and_uncalibrated_weight() -> None:
    """t6: an offline (not-probed) row must not GUESS a capacity — it reports
    ``capacity: None`` (honest, like every other live field on this row) and
    ``weight`` as the UNCALIBRATED_WEIGHT sentinel, never a measured-looking
    number the row has no evidence for.
    """
    table = _table(
        replica_origins={"primary": (_THOR_ORIGIN,)},
        self_origin=_SPARK_ORIGIN,
    )
    payload = _base_payload(table)
    annotate_replicas(payload, table)

    rows = payload["cortex"]["replicas"]
    assert len(rows) == 2
    for row in rows:
        assert row["capacity"] is None
        assert row["weight"] == UNCALIBRATED_WEIGHT


def test_offline_view_self_origin_defaults_to_local() -> None:
    table = _table(replica_origins={"primary": (_THOR_ORIGIN,)})
    payload = _base_payload(table)
    annotate_replicas(payload, table)
    assert payload["cortex"]["replicas"][0]["origin"] == "local"


def test_undeclared_fingerprint_fields_are_unknown_never_catalog() -> None:
    """No lane_fingerprints declared at all: every declared field is
    'unknown' — never the catalog's quant/mtp/runtime for the served id."""
    table = _table(replica_origins={"primary": (_THOR_ORIGIN,)})
    payload = _base_payload(table)
    annotate_replicas(payload, table)
    fp = payload["cortex"]["fingerprint"]
    assert fp["served_id"] == _CORTEX_ID  # from the payload's own served model
    assert fp["quantization"] == "unknown"
    assert fp["kv_cache_dtype"] == "unknown"
    assert fp["tool_parser"] == "unknown"
    assert fp["speculative_config"] == "unknown"


# ============================================================================
# Live snapshot replica view
# ============================================================================


def test_live_snapshot_renders_verbatim() -> None:
    table = _table(replica_origins={"primary": (_THOR_ORIGIN,)})
    local_fp = _fingerprint()
    peer_fp = _fingerprint(kv_cache_dtype="auto", speculative_config="mtp")
    snapshot = {
        "cortex": (
            _state(
                origin=_SPARK_ORIGIN,
                local=True,
                ready=True,
                busy=False,
                running=2,
                waiting=1,
                fingerprint=local_fp,
                compatible=True,
                reason="",
                weight=1.0,
            ),
            _state(
                origin=_THOR_ORIGIN,
                local=False,
                ready=True,
                busy=False,
                running=0,
                waiting=0,
                fingerprint=peer_fp,
                compatible=True,
                reason="",
                weight=1.0,
            ),
        )
    }
    payload = _base_payload(table)
    annotate_replicas(payload, table, snapshot=snapshot)

    rows = payload["cortex"]["replicas"]
    assert len(rows) == 2
    assert rows[0]["origin"] == _SPARK_ORIGIN
    assert rows[0]["local"] is True
    assert rows[0]["running"] == 2
    assert rows[0]["waiting"] == 1
    assert rows[1]["origin"] == _THOR_ORIGIN
    assert rows[1]["compatible"] is True
    assert rows[1]["fingerprint"]["kv_cache_dtype"] == "auto"

    # fingerprint at the top level comes from the LOCAL replica's live probe
    assert payload["cortex"]["fingerprint"] == dataclasses.asdict(local_fp)


def test_live_snapshot_row_carries_capacity_alongside_weight() -> None:
    """t6: a clamped peer is exactly the diagnostic case — the row must
    expose BOTH the raw claimed capacity and the resolved (post-clamp)
    weight the ranking arithmetic actually used, distinctly.
    """
    table = _table(replica_origins={"primary": (_THOR_ORIGIN,)})
    local_fp = _fingerprint()
    snapshot = {
        "cortex": (
            _state(
                origin=_SPARK_ORIGIN, local=True, fingerprint=local_fp, weight=2.0, capacity=2.0
            ),
            _state(
                origin=_THOR_ORIGIN,
                local=False,
                fingerprint=_fingerprint(),
                weight=32.0,  # clamped
                capacity=999.0,  # raw claim, pre-clamp
                reason="capacity clamped: 999 -> 32",
            ),
        )
    }
    payload = _base_payload(table)
    annotate_replicas(payload, table, snapshot=snapshot)

    rows = payload["cortex"]["replicas"]
    assert rows[0]["weight"] == 2.0
    assert rows[0]["capacity"] == 2.0
    assert rows[1]["weight"] == 32.0
    assert rows[1]["capacity"] == 999.0
    assert "clamped" in rows[1]["reason"]


def test_live_snapshot_row_capacity_none_when_not_published() -> None:
    table = _table(replica_origins={"primary": (_THOR_ORIGIN,)})
    snapshot = {
        "cortex": (
            _state(origin=_SPARK_ORIGIN, local=True, fingerprint=_fingerprint()),
            _state(origin=_THOR_ORIGIN, local=False, fingerprint=_fingerprint()),
        )
    }
    payload = _base_payload(table)
    annotate_replicas(payload, table, snapshot=snapshot)

    rows = payload["cortex"]["replicas"]
    for row in rows:
        assert row["capacity"] is None
        assert row["weight"] == UNCALIBRATED_WEIGHT


def test_live_snapshot_incompatible_peer_reports_reason() -> None:
    table = _table(replica_origins={"primary": (_THOR_ORIGIN,)})
    local_fp = _fingerprint()
    snapshot = {
        "cortex": (
            _state(origin=_SPARK_ORIGIN, local=True, fingerprint=local_fp, compatible=True),
            _state(
                origin=_THOR_ORIGIN,
                local=False,
                ready=True,
                fingerprint=_fingerprint(max_model_len=131072),
                compatible=False,
                reason="max_model_len: 262144 != 131072",
            ),
        )
    }
    payload = _base_payload(table)
    annotate_replicas(payload, table, snapshot=snapshot)
    peer_row = payload["cortex"]["replicas"][1]
    assert peer_row["compatible"] is False
    assert "max_model_len" in peer_row["reason"]


def test_hosted_role_stays_feasible_true_with_replicas_declared() -> None:
    table = _table(replica_origins={"primary": (_THOR_ORIGIN,)})
    payload = _base_payload(table)
    annotate_replicas(payload, table)
    assert payload["cortex"]["feasible"] is True
    assert "hosted_by" not in payload["cortex"]


def test_offline_fingerprint_reads_tool_call_parser_suffix():
    """The lane knob is *_TOOL_CALL_PARSER, and _lane_fingerprints stores that suffix."""
    from lobes.roles import _offline_fingerprint

    fp = _offline_fingerprint(
        {"TOOL_CALL_PARSER": "qwen3_coder_thinking"}, {"model": "m", "context": 1}
    )
    assert fp["tool_parser"] == "qwen3_coder_thinking"
