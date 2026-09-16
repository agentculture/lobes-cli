"""cortex-replica-pool CONFIG channels: self-origin + lane fingerprints (t2,
#199) — the plural peer family retired (t14).

The plural replica-pool env family this module used to pin —
``<PREFIX>_PEER_ORIGINS`` / ``<PREFIX>_PEER_API_KEYS`` (comma-separated,
positional), parsed into :class:`~lobes.gateway._routing.RoutingTable`'s
``replica_origins``/``replica_api_keys`` fields — is GONE (t14): t13 made
the mesh ``RoutingSnapshot`` the pool candidate source, so
:func:`~lobes.gateway._config.build_config` no longer reads either env var.
``PEER_ORIGINS_ENV``/``PEER_API_KEYS_ENV`` and the ``ReplicaConfigError``
raise for a positional length mismatch are deleted along with that parsing.
What survives here, unaffected by the retirement:

* ``self_origin`` from ``GATEWAY_SELF_ORIGIN`` only — never derived.
* Lane fingerprints, read per backend name from
  ``<PREFIX>_{QUANTIZATION,KV_CACHE_DTYPE,REASONING_PARSER,TOOL_CALL_PARSER,
  SPECULATIVE_CONFIG}``; only SET knobs appear.
* The capacity + kill-switch env knobs (``MAX_ACTIVE_ENV`` /
  ``CAPACITY_KILL_SWITCH_ENV``) — a completely separate channel from the
  peer family, explicitly kept per the operator instruction ("keep
  ReplicaCache/select_replica/capacity clamp; remove only the env SOURCE").
* A no-new-knobs env yields a table equal (==) to today's.
* The retired plural knobs are now INERT — setting them does nothing.
"""

from __future__ import annotations

import pytest

from lobes.catalog import TIER_ROLE
from lobes.gateway._config import (
    CAPACITY_KILL_SWITCH_ENV,
    FEASIBLE_ENV,
    LANE_FINGERPRINT_SUFFIXES,
    MAX_ACTIVE_ENV,
    CapacityConfigError,
    ServerConfig,
    build_config,
)
from lobes.gateway._routing import Backend, RoutingTable, tier_aliases

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_ORIGIN_A = "http://a:8000"
_ORIGIN_B = "http://b:8000"


def _base_env(**over: str) -> dict[str, str]:
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
    }
    env.update(over)
    return env


# ============================================================================
# Retired: the plural replica-pool env family is now inert
# ============================================================================


def test_replica_origins_env_knob_is_now_inert() -> None:
    table, _cfg = build_config(_base_env(PRIMARY_PEER_ORIGINS=f"{_ORIGIN_A},{_ORIGIN_B}"))
    assert dict(table.replica_origins) == {}


def test_replica_api_keys_env_knob_is_now_inert() -> None:
    table, _cfg = build_config(
        _base_env(
            PRIMARY_PEER_ORIGINS=f"{_ORIGIN_A},{_ORIGIN_B}",
            PRIMARY_PEER_API_KEYS="k1,k2",
        )
    )
    assert dict(table.replica_api_keys) == {}


def test_mismatched_replica_api_keys_length_no_longer_raises() -> None:
    # Pre-t14 this was a hard ReplicaConfigError (positional length
    # mismatch); with the env parsing gone there is nothing left to
    # mismatch — build_config simply ignores both keys.
    env = _base_env(
        PRIMARY_PEER_ORIGINS=f"{_ORIGIN_A},{_ORIGIN_B}",
        PRIMARY_PEER_API_KEYS="k1",
    )
    table, _cfg = build_config(env)
    assert dict(table.replica_origins) == {}
    assert dict(table.replica_api_keys) == {}


# ============================================================================
# self_origin: GATEWAY_SELF_ORIGIN only, never derived
# ============================================================================


def test_self_origin_from_env() -> None:
    table, _cfg = build_config(_base_env(GATEWAY_SELF_ORIGIN="http://spark.local:8001/"))
    assert table.self_origin == "http://spark.local:8001"


def test_self_origin_absent_defaults_empty() -> None:
    table, _cfg = build_config(_base_env())
    assert table.self_origin == ""


def test_self_origin_blank_defaults_empty() -> None:
    table, _cfg = build_config(_base_env(GATEWAY_SELF_ORIGIN="   "))
    assert table.self_origin == ""


# ============================================================================
# lane_fingerprints: per-backend declared knobs, only-set-ones appear
# ============================================================================


def test_lane_fingerprints_reads_declared_knobs_only() -> None:
    table, _cfg = build_config(
        _base_env(
            PRIMARY_QUANTIZATION="modelopt",
            PRIMARY_TOOL_CALL_PARSER="qwen3_coder",
        )
    )
    assert dict(table.lane_fingerprints) == {
        "primary": {"QUANTIZATION": "modelopt", "TOOL_CALL_PARSER": "qwen3_coder"}
    }


def test_lane_fingerprints_absent_backend_yields_no_entry() -> None:
    table, _cfg = build_config(_base_env())
    assert dict(table.lane_fingerprints) == {}


def test_lane_fingerprints_covers_all_five_suffixes() -> None:
    assert LANE_FINGERPRINT_SUFFIXES == (
        "QUANTIZATION",
        "KV_CACHE_DTYPE",
        "REASONING_PARSER",
        "TOOL_CALL_PARSER",
        "SPECULATIVE_CONFIG",
    )
    env = _base_env(**{f"PRIMARY_{suffix}": "x" for suffix in LANE_FINGERPRINT_SUFFIXES})
    table, _cfg = build_config(env)
    assert dict(table.lane_fingerprints["primary"]) == {s: "x" for s in LANE_FINGERPRINT_SUFFIXES}


def test_lane_fingerprints_across_backend_names() -> None:
    table, _cfg = build_config(
        _base_env(MULTIMODAL_QUANTIZATION="fp8", HAND_TOOL_CALL_PARSER="lfm2")
    )
    assert dict(table.lane_fingerprints) == {
        "multimodal": {"QUANTIZATION": "fp8"},
        "hand": {"TOOL_CALL_PARSER": "lfm2"},
    }


# ============================================================================
# No-new-knobs env: table equal (==) to today's on every new field default
# ============================================================================


def test_no_new_knobs_env_yields_todays_config_objects() -> None:
    env = _base_env()
    table, cfg = build_config(env)
    primary = Backend(
        name="primary",
        base_url="http://vllm-primary:8000",
        served_name=_CORTEX_ID,
    )
    assert table == RoutingTable(
        backends=(primary,),
        default_model=_CORTEX_ID,
        aliases=tier_aliases([primary], TIER_ROLE),
        infeasible=frozenset({"muse", "worker", "associate", "innereye"}),
    )
    assert cfg == ServerConfig(
        host="0.0.0.0",  # nosec B104 — asserting the existing default, not binding
        port=8000,
        connect_timeout=5.0,
        read_timeout=600.0,
    )
    assert dict(table.replica_origins) == {}
    assert dict(table.replica_api_keys) == {}
    assert table.self_origin == ""
    assert dict(table.lane_fingerprints) == {}


def test_new_routing_fields_default_inert_on_direct_construction() -> None:
    table = RoutingTable(
        backends=(Backend(name="primary", base_url="http://x:1", served_name="m"),),
        default_model="m",
        aliases={},
    )
    assert dict(table.replica_origins) == {}
    assert dict(table.replica_api_keys) == {}
    assert table.self_origin == ""
    assert dict(table.lane_fingerprints) == {}


# ============================================================================
# Capacity + kill-switch env knobs (issue #199 capacity-relative-pool-routing,
# t1). This box's own declared "max active requests" capacity, per backend
# name, plus a single global kill switch that pins every resolved capacity
# back to the 1.0 sentinel. NO ranking/selection behaviour lands here — see
# lobes/gateway/_selection.py (t3) and lobes/gateway/_replicas.py (t4) for
# what consumes these values.
# ============================================================================


def test_max_active_env_mirrors_feasible_env_prefixes() -> None:
    assert set(MAX_ACTIVE_ENV) == set(FEASIBLE_ENV)
    assert MAX_ACTIVE_ENV == {
        "primary": "PRIMARY_MAX_ACTIVE",
        "multimodal": "MULTIMODAL_MAX_ACTIVE",
        "muse": "MUSE_MAX_ACTIVE",
        "worker": "WORKER_MAX_ACTIVE",
        "associate": "ASSOCIATE_MAX_ACTIVE",
        "hand": "HAND_MAX_ACTIVE",
        "embed": "EMBED_MAX_ACTIVE",
        "rerank": "RERANK_MAX_ACTIVE",
        "stt": "STT_MAX_ACTIVE",
        "tts": "TTS_MAX_ACTIVE",
        "innereye": "INNEREYE_MAX_ACTIVE",
    }


def test_capacity_kill_switch_env_is_a_single_global_knob() -> None:
    assert CAPACITY_KILL_SWITCH_ENV == "GATEWAY_CAPACITY_KILL_SWITCH"


def test_no_capacity_knobs_yields_empty_capacities_and_switch_off() -> None:
    _table, cfg = build_config(_base_env())
    assert dict(cfg.local_capacities) == {}
    assert cfg.capacity_kill_switch is False


def test_declared_capacity_parses_into_config() -> None:
    _table, cfg = build_config(_base_env(PRIMARY_MAX_ACTIVE="8"))
    assert cfg.local_capacities["primary"] == 8.0


def test_declared_capacity_across_backend_names() -> None:
    _table, cfg = build_config(_base_env(PRIMARY_MAX_ACTIVE="8", HAND_MAX_ACTIVE="4"))
    assert dict(cfg.local_capacities) == {"primary": 8.0, "hand": 4.0}


def test_blank_capacity_is_treated_as_unset() -> None:
    _table, cfg = build_config(_base_env(PRIMARY_MAX_ACTIVE="  "))
    assert dict(cfg.local_capacities) == {}


def test_malformed_capacity_raises_loudly() -> None:
    env = _base_env(PRIMARY_MAX_ACTIVE="not-a-number")
    with pytest.raises(CapacityConfigError):
        build_config(env)


def test_kill_switch_forces_sentinel_for_every_role_name_ignoring_declared() -> None:
    _table, cfg = build_config(
        _base_env(
            GATEWAY_CAPACITY_KILL_SWITCH="true",
            PRIMARY_MAX_ACTIVE="8",
            HAND_MAX_ACTIVE="4",
        )
    )
    assert cfg.capacity_kill_switch is True
    assert dict(cfg.local_capacities) == {name: 1.0 for name in MAX_ACTIVE_ENV}


def test_kill_switch_engaged_with_no_declared_capacities_still_sentinels_all() -> None:
    _table, cfg = build_config(_base_env(GATEWAY_CAPACITY_KILL_SWITCH="1"))
    assert cfg.capacity_kill_switch is True
    assert dict(cfg.local_capacities) == {name: 1.0 for name in MAX_ACTIVE_ENV}


def test_kill_switch_falsy_token_leaves_switch_off() -> None:
    _table, cfg = build_config(
        _base_env(GATEWAY_CAPACITY_KILL_SWITCH="false", PRIMARY_MAX_ACTIVE="8")
    )
    assert cfg.capacity_kill_switch is False
    assert dict(cfg.local_capacities) == {"primary": 8.0}


def test_single_box_no_peers_no_capacity_keys_parses_unchanged() -> None:
    # The single-box, no-*_PEER_ORIGINS, no-capacity-keys deployment must
    # keep parsing exactly as it did before this task — no new required key.
    env = _base_env()
    table, cfg = build_config(env)
    assert dict(table.replica_origins) == {}
    assert dict(cfg.local_capacities) == {}
    assert cfg.capacity_kill_switch is False
