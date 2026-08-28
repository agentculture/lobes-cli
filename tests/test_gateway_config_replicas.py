"""cortex-replica-pool CONFIG channels: plural peers + self-origin (t2, #199).

This task adds the *config layer only* for the plural replica-pool family —
parsing ``<PREFIX>_PEER_ORIGINS`` / ``<PREFIX>_PEER_API_KEYS`` (comma-
separated, positional) plus ``GATEWAY_SELF_ORIGIN`` and per-backend lane
fingerprint knobs into new :class:`~lobes.gateway._routing.RoutingTable`
fields. NO selection/dialing/consistency-checking behaviour lands here —
that is later cortex-replica-pool work. The existing SCALAR peer channels
(``PEER_ORIGIN_ENV`` / ``PEER_PROXY_ENV`` / ``PEER_API_KEY_ENV``) and
``order_backends`` are untouched by this task.

Contract pinned below:

* ``replica_origins`` is comma-separated, each item stripped and
  trailing-slash-trimmed; empty items are dropped; an absent/blank key
  yields no entry.
* ``replica_api_keys`` is comma-separated and POSITIONAL against
  ``replica_origins`` for the same name — an empty slot is legal (no key
  for that replica), but a list whose length disagrees with its origins
  list (shorter OR longer) raises :class:`~lobes.gateway._config.
  ReplicaConfigError` naming the backend.
* ``self_origin`` comes from ``GATEWAY_SELF_ORIGIN`` only — never derived.
* Lane fingerprints are read per backend name from
  ``<PREFIX>_{QUANTIZATION,KV_CACHE_DTYPE,REASONING_PARSER,TOOL_CALL_PARSER,
  SPECULATIVE_CONFIG}``; only SET knobs appear.
* A no-new-knobs env yields a table equal (==) to today's.
"""

from __future__ import annotations

import pytest

from lobes.catalog import TIER_ROLE
from lobes.gateway._config import (
    CAPACITY_KILL_SWITCH_ENV,
    FEASIBLE_ENV,
    LANE_FINGERPRINT_SUFFIXES,
    MAX_ACTIVE_ENV,
    PEER_API_KEYS_ENV,
    PEER_ORIGINS_ENV,
    CapacityConfigError,
    ReplicaConfigError,
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
# The env channels: one <PREFIX>_<KNOB> convention, all nine backend names
# ============================================================================


def test_peer_origins_env_mirrors_feasible_env_prefixes() -> None:
    assert set(PEER_ORIGINS_ENV) == set(FEASIBLE_ENV)
    assert PEER_ORIGINS_ENV == {
        "primary": "PRIMARY_PEER_ORIGINS",
        "multimodal": "MULTIMODAL_PEER_ORIGINS",
        "muse": "MUSE_PEER_ORIGINS",
        "worker": "WORKER_PEER_ORIGINS",
        "associate": "ASSOCIATE_PEER_ORIGINS",
        "hand": "HAND_PEER_ORIGINS",
        "embed": "EMBED_PEER_ORIGINS",
        "rerank": "RERANK_PEER_ORIGINS",
        "stt": "STT_PEER_ORIGINS",
        "tts": "TTS_PEER_ORIGINS",
    }


def test_peer_api_keys_env_mirrors_feasible_env_prefixes() -> None:
    assert set(PEER_API_KEYS_ENV) == set(FEASIBLE_ENV)
    assert PEER_API_KEYS_ENV == {
        "primary": "PRIMARY_PEER_API_KEYS",
        "multimodal": "MULTIMODAL_PEER_API_KEYS",
        "muse": "MUSE_PEER_API_KEYS",
        "worker": "WORKER_PEER_API_KEYS",
        "associate": "ASSOCIATE_PEER_API_KEYS",
        "hand": "HAND_PEER_API_KEYS",
        "embed": "EMBED_PEER_API_KEYS",
        "rerank": "RERANK_PEER_API_KEYS",
        "stt": "STT_PEER_API_KEYS",
        "tts": "TTS_PEER_API_KEYS",
    }


# ============================================================================
# replica_origins: comma-separated, stripped, trailing-slash-trimmed
# ============================================================================


def test_replica_origins_parsed_comma_separated() -> None:
    table, _cfg = build_config(_base_env(PRIMARY_PEER_ORIGINS=f"{_ORIGIN_A},{_ORIGIN_B}"))
    assert dict(table.replica_origins) == {"primary": (_ORIGIN_A, _ORIGIN_B)}


def test_replica_origins_stripped_and_trailing_slash_trimmed() -> None:
    table, _cfg = build_config(_base_env(PRIMARY_PEER_ORIGINS=" http://a:8000/ , http://b:8000// "))
    assert dict(table.replica_origins) == {"primary": ("http://a:8000", "http://b:8000")}


def test_replica_origins_empty_items_dropped() -> None:
    table, _cfg = build_config(_base_env(PRIMARY_PEER_ORIGINS=f"{_ORIGIN_A},,{_ORIGIN_B},"))
    assert dict(table.replica_origins) == {"primary": (_ORIGIN_A, _ORIGIN_B)}


def test_replica_origins_absent_key_yields_no_entry() -> None:
    table, _cfg = build_config(_base_env())
    assert dict(table.replica_origins) == {}


def test_replica_origins_blank_key_yields_no_entry() -> None:
    table, _cfg = build_config(_base_env(PRIMARY_PEER_ORIGINS="   "))
    assert dict(table.replica_origins) == {}


# ============================================================================
# replica_api_keys: comma-separated, POSITIONAL, empty slot legal
# ============================================================================


def test_replica_api_keys_positional_with_trailing_empty_slot() -> None:
    table, _cfg = build_config(
        _base_env(
            PRIMARY_PEER_ORIGINS=f"{_ORIGIN_A},{_ORIGIN_B}",
            PRIMARY_PEER_API_KEYS="k1,",
        )
    )
    assert dict(table.replica_api_keys) == {"primary": ("k1", "")}


def test_replica_api_keys_bare_blank_value_yields_one_empty_slot_per_origin() -> None:
    table, _cfg = build_config(
        _base_env(
            PRIMARY_PEER_ORIGINS=f"{_ORIGIN_A},{_ORIGIN_B}",
            PRIMARY_PEER_API_KEYS="",
        )
    )
    assert dict(table.replica_api_keys) == {"primary": ("", "")}


def test_replica_api_keys_without_origins_is_inert() -> None:
    table, _cfg = build_config(_base_env(PRIMARY_PEER_API_KEYS="k1,k2"))
    assert dict(table.replica_api_keys) == {}


def test_replica_api_keys_shorter_list_raises_naming_prefix() -> None:
    env = _base_env(
        PRIMARY_PEER_ORIGINS=f"{_ORIGIN_A},{_ORIGIN_B}",
        PRIMARY_PEER_API_KEYS="k1",
    )
    with pytest.raises(ReplicaConfigError, match="PRIMARY"):
        build_config(env)


def test_replica_api_keys_longer_list_raises_naming_prefix() -> None:
    env = _base_env(
        PRIMARY_PEER_ORIGINS=_ORIGIN_A,
        PRIMARY_PEER_API_KEYS="k1,k2",
    )
    with pytest.raises(ReplicaConfigError, match="PRIMARY"):
        build_config(env)


def test_replica_api_keys_stripped_not_transformed() -> None:
    table, _cfg = build_config(
        _base_env(
            PRIMARY_PEER_ORIGINS=_ORIGIN_A,
            PRIMARY_PEER_API_KEYS="  MiXeD-Case-Key==  ",
        )
    )
    assert dict(table.replica_api_keys) == {"primary": ("MiXeD-Case-Key==",)}


def test_replica_api_key_values_never_appear_in_repr_or_str() -> None:
    secret = "sk-replica-secret-do-not-print"  # nosec B105 — test fixture, not a credential
    table, _cfg = build_config(
        _base_env(PRIMARY_PEER_ORIGINS=_ORIGIN_A, PRIMARY_PEER_API_KEYS=secret)
    )
    assert dict(table.replica_api_keys) == {"primary": (secret,)}
    for text in (repr(table), str(table)):
        assert secret not in text


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
        infeasible=frozenset({"muse", "worker", "associate"}),
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
