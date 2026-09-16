"""The env peer family is RETIRED (t14): PEER_ORIGIN/PEER_PROXY/PEER_API_KEY
and their plural PEER_ORIGINS/PEER_API_KEYS siblings are no longer parsed by
:func:`lobes.gateway._config.build_config` — t13 made the mesh
``RoutingSnapshot`` the candidate source for both pool placement and
forwarding, so no behaviour depended on these env vars any more.

This module used to pin the CONFIG-LAYER shape of that parsing (proxy-lobes
t1, issues #115/#127): ``PEER_ORIGIN_ENV``/``PEER_PROXY_ENV``/
``PEER_API_KEY_ENV``/``PEER_ORIGINS_ENV``/``PEER_API_KEYS_ENV`` and the
``RoutingTable.peer_proxied``/``peer_api_keys`` fields they populated. All
five dicts are DELETED along with the functions that read them — there is no
mesh-shaped replacement to pin here, because ``build_config`` never consulted
the mesh in the first place (the mesh is applied later, in
``lobes.gateway.server``, as a separate ``mesh_snapshot`` parameter). So this
file keeps only what survives the retirement:

* the still-live ``ServerConfig.api_key`` resolution
  (``GATEWAY_API_KEY`` -> ``CULTURE_VLLM_API_KEY`` -> ``None``);
* the still-live ``RoutingTable`` fields (``peer_proxied``/``peer_api_keys``/
  ``peer_origins``/``replica_origins``/``replica_api_keys``) defaulting to
  empty on direct construction — nothing deletes the fields, only their env
  population;
* a new, explicit "the retired env vars are now INERT" contract: setting any
  of them in ``.env`` no longer moves any field on the built table — this is
  the config-layer half of what ``lobes doctor``'s ``peer_family_retired``
  finding polices operationally;
* the no-new-knobs byte-identity claim, updated for the current field set;
* the secrets-never-in-repr contract, now proven against a directly
  constructed table (since ``build_config`` no longer carries a peer secret
  through from env at all).
"""

from __future__ import annotations

from lobes.catalog import TIER_ROLE
from lobes.gateway._config import ServerConfig, build_config
from lobes.gateway._routing import Backend, RoutingTable, tier_aliases

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"

# Full, dialable origins an operator would DECLARE per box in .env — never
# derived (#92). Retained here only to prove the retired knobs are inert.
_THOR_ORIGIN = "http://thor.local:8001"


def _spark_lobe_env(**over: str) -> dict[str, str]:
    """A rendered spark-lobe env: cortex hosted, senses DROPPED (infeasible)."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_FEASIBLE": "false",
    }
    env.update(over)
    return env


# ============================================================================
# Retired: every *_PEER_* env knob is now inert — build_config ignores all of
# them, on every shape they used to arm (proxy knob + origin on a dropped
# role, worker's opt-in-core variant, the plural replica-pool family).
# ============================================================================


def test_retired_peer_proxy_and_origin_knobs_are_inert() -> None:
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_PROXY="true",
            MULTIMODAL_PEER_API_KEY="sk-lobes-thor-0001",  # nosec B105 — test fixture
        )
    )
    assert table.peer_proxied == frozenset()
    assert dict(table.peer_origins) == {}
    assert dict(table.peer_api_keys) == {}


def test_retired_plural_replica_pool_knobs_are_inert() -> None:
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGINS=_THOR_ORIGIN,
            MULTIMODAL_PEER_API_KEYS="sk-lobes-thor-0001",  # nosec B105 — test fixture
        )
    )
    assert dict(table.replica_origins) == {}
    assert dict(table.replica_api_keys) == {}


def test_retired_worker_peer_knobs_are_inert() -> None:
    # worker (the opt-in-core eighth role) used to ride the exact same
    # channel as every other backend name — it is just as inert now.
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "WORKER_PEER_ORIGIN": _THOR_ORIGIN,
        "WORKER_PEER_PROXY": "true",
        "WORKER_PEER_API_KEY": "sk-lobes-thor-worker-0001",  # nosec B105
    }
    table, _cfg = build_config(env)
    assert table.peer_proxied == frozenset()
    assert dict(table.peer_origins) == {}
    assert dict(table.peer_api_keys) == {}


# ============================================================================
# ServerConfig.api_key: GATEWAY_API_KEY → CULTURE_VLLM_API_KEY → None
# (unrelated to the retired peer family — still fully live)
# ============================================================================


def test_gateway_api_key_explicit_value_wins() -> None:
    _table, cfg = build_config(
        _spark_lobe_env(
            GATEWAY_API_KEY="sk-gateway-explicit",
            CULTURE_VLLM_API_KEY="sk-culture-existing",
        )
    )
    assert cfg.api_key == "sk-gateway-explicit"


def test_gateway_api_key_falls_back_to_culture_vllm_api_key() -> None:
    _table, cfg = build_config(_spark_lobe_env(CULTURE_VLLM_API_KEY="sk-culture-existing"))
    assert cfg.api_key == "sk-culture-existing"


def test_gateway_api_key_blank_falls_through_to_culture() -> None:
    _table, cfg = build_config(
        _spark_lobe_env(
            GATEWAY_API_KEY="   ",
            CULTURE_VLLM_API_KEY="sk-culture-existing",
        )
    )
    assert cfg.api_key == "sk-culture-existing"


def test_gateway_api_key_both_unset_disables_auth() -> None:
    _table, cfg = build_config(_spark_lobe_env())
    assert cfg.api_key is None


def test_gateway_api_key_both_blank_disables_auth() -> None:
    _table, cfg = build_config(_spark_lobe_env(GATEWAY_API_KEY="", CULTURE_VLLM_API_KEY="  "))
    assert cfg.api_key is None


def test_gateway_api_key_is_stripped() -> None:
    _table, cfg = build_config(_spark_lobe_env(GATEWAY_API_KEY="  sk-padded  "))
    assert cfg.api_key == "sk-padded"


# ============================================================================
# Secrets never appear in repr/str — proven on a directly constructed table,
# since build_config no longer carries any peer secret through from env.
# ============================================================================


def test_key_values_never_appear_in_repr_or_str() -> None:
    peer_secret = "sk-peer-secret-do-not-print"  # nosec B105 — test fixture, not a credential
    gateway_secret = "sk-gateway-secret-do-not-print"  # nosec B105 — test fixture
    primary = Backend(name="primary", base_url="http://vllm-primary:8000", served_name=_CORTEX_ID)
    table = RoutingTable(
        backends=(primary,),
        default_model=_CORTEX_ID,
        aliases={},
        peer_api_keys={"multimodal": peer_secret},
    )
    cfg = ServerConfig(
        host="0.0.0.0",  # nosec B104
        port=8000,
        connect_timeout=5.0,
        read_timeout=600.0,
        api_key=gateway_secret,
    )
    # The values ARE carried (the proxy data plane still reads them off a
    # directly/mesh-constructed table) ...
    assert dict(table.peer_api_keys) == {"multimodal": peer_secret}
    assert cfg.api_key == gateway_secret
    # ... but NEVER surface in repr/str of either config object.
    for text in (repr(table), str(table), repr(cfg), str(cfg)):
        assert peer_secret not in text
        assert gateway_secret not in text


# ============================================================================
# No-new-knobs env: byte-identical config objects on every existing field
# ============================================================================


def test_no_new_knobs_env_yields_todays_config_objects() -> None:
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
    }
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
    assert table.peer_proxied == frozenset()
    assert dict(table.peer_api_keys) == {}
    assert cfg.api_key is None


def test_routing_table_peer_fields_default_inert_on_direct_construction() -> None:
    # Every existing RoutingTable(...) construction in the codebase/tests
    # omits these fields — they must default to empty.
    table = RoutingTable(
        backends=(Backend(name="primary", base_url="http://x:1", served_name="m"),),
        default_model="m",
        aliases={},
    )
    assert table.peer_proxied == frozenset()
    assert dict(table.peer_api_keys) == {}
    assert dict(table.peer_origins) == {}
    assert dict(table.replica_origins) == {}
    assert dict(table.replica_api_keys) == {}
