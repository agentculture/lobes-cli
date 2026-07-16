"""Proxy-lobes CONFIG channels: proxy knob, peer keys, gateway key (t1, #127/#115).

This task adds the *config layer only* for proxy-lobes — the ability to
declare that a dropped role should be PROXIED to its declared peer
(``<PREFIX>_PEER_PROXY``), per-peer outbound credentials
(``<PREFIX>_PEER_API_KEY``), and an inbound gateway API key
(``GATEWAY_API_KEY``, falling back to ``CULTURE_VLLM_API_KEY``). NO server
behaviour changes here: nothing dials a peer and nothing enforces auth — the
data-plane branch that consults these fields lands in a LATER task.

The contract pinned below:

* ``peer_proxied`` holds a backend name ONLY when its proxy knob is truthy
  AND a peer origin is declared AND the role is infeasible on this box.
  Origin without the knob stays referral-only (the issue #112 contract is
  preserved byte-for-byte); knob without an origin has nothing to dial and
  is ignored; knob+origin on a locally-feasible role is ignored (the local
  engine serves it — hosted behaviour unchanged).
* ``peer_api_keys`` carries keys verbatim (stripped), and ONLY for names
  with a declared peer origin — a key without an origin is inert.
* ``ServerConfig.api_key`` resolves ``GATEWAY_API_KEY`` →
  ``CULTURE_VLLM_API_KEY`` → ``None`` (auth disabled); both unset is
  today's no-auth behaviour.
* SECRETS NEVER APPEAR in ``repr``/``str`` of the config objects.
* A no-new-knobs env yields config objects equal to today's on every
  pre-existing field.
"""

from __future__ import annotations

import pytest

from lobes.catalog import TIER_ROLE
from lobes.gateway._config import (
    FEASIBLE_ENV,
    PEER_API_KEY_ENV,
    PEER_ORIGIN_ENV,
    PEER_PROXY_ENV,
    ServerConfig,
    build_config,
)
from lobes.gateway._routing import Backend, RoutingTable, tier_aliases

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"

# Full, dialable origins an operator would DECLARE per box in .env — never
# derived (#92).
_THOR_ORIGIN = "http://thor.local:8001"
_SPARK_ORIGIN = "http://spark.local:8001"


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
# The env channels: one <PREFIX>_<KNOB> convention, five core roles
# ============================================================================


def test_peer_proxy_env_mirrors_feasible_env_prefixes() -> None:
    # Channels scoped to the five core roles: the proxy knob names exactly
    # the backends the feasibility / peer-origin channels name.
    assert set(PEER_PROXY_ENV) == set(FEASIBLE_ENV) == set(PEER_ORIGIN_ENV)
    assert PEER_PROXY_ENV == {
        "primary": "PRIMARY_PEER_PROXY",
        "multimodal": "MULTIMODAL_PEER_PROXY",
        "muse": "MUSE_PEER_PROXY",
        "embed": "EMBED_PEER_PROXY",
        "rerank": "RERANK_PEER_PROXY",
    }


def test_peer_api_key_env_mirrors_feasible_env_prefixes() -> None:
    assert set(PEER_API_KEY_ENV) == set(FEASIBLE_ENV)
    assert PEER_API_KEY_ENV == {
        "primary": "PRIMARY_PEER_API_KEY",
        "multimodal": "MULTIMODAL_PEER_API_KEY",
        "muse": "MUSE_PEER_API_KEY",
        "embed": "EMBED_PEER_API_KEY",
        "rerank": "RERANK_PEER_API_KEY",
    }


# ============================================================================
# peer_proxied: knob AND origin AND infeasible — all three, or nothing
# ============================================================================


def test_proxy_knob_with_origin_on_infeasible_role_is_proxied() -> None:
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_PROXY="true",
        )
    )
    assert table.peer_proxied == frozenset({"multimodal"})
    # The referral annotation channel is independent and still populated.
    assert dict(table.peer_origins) == {"multimodal": _THOR_ORIGIN}


def test_proxy_knob_without_origin_is_ignored() -> None:
    # A proxy knob with no declared origin has nothing to dial — ignored.
    table, _cfg = build_config(_spark_lobe_env(MULTIMODAL_PEER_PROXY="true"))
    assert table.peer_proxied == frozenset()


def test_origin_without_knob_stays_referral_only() -> None:
    # The issue #112 contract preserved: a declared origin alone is
    # annotation-only referral, never a proxy opt-in.
    table, _cfg = build_config(_spark_lobe_env(MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN))
    assert table.peer_proxied == frozenset()
    assert dict(table.peer_origins) == {"multimodal": _THOR_ORIGIN}


def test_proxy_knob_and_origin_on_feasible_role_is_ignored() -> None:
    # The role is hosted locally (no MULTIMODAL_FEASIBLE=false) — the local
    # engine serves it; hosted behaviour is unchanged by the knob.
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_PEER_ORIGIN": _THOR_ORIGIN,
        "MULTIMODAL_PEER_PROXY": "true",
    }
    table, _cfg = build_config(env)
    assert table.peer_proxied == frozenset()


def test_proxy_knob_works_for_a_wired_but_infeasible_role() -> None:
    # thor-lobe shape: the primary is unconditionally wired yet dropped —
    # the knob still applies (infeasibility is a config fact, not wiring).
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "PRIMARY_FEASIBLE": "false",
        "PRIMARY_PEER_ORIGIN": _SPARK_ORIGIN,
        "PRIMARY_PEER_PROXY": "yes",
    }
    table, _cfg = build_config(env)
    assert table.peer_proxied == frozenset({"primary"})


@pytest.mark.parametrize("token", ["1", "true", "yes", "TRUE", "Yes", " true "])
def test_truthy_proxy_tokens_accepted(token: str) -> None:
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_PROXY=token,
        )
    )
    assert table.peer_proxied == frozenset({"multimodal"})


@pytest.mark.parametrize("token", ["false", "", "0", "no", "off", "banana"])
def test_non_truthy_proxy_tokens_rejected(token: str) -> None:
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_PROXY=token,
        )
    )
    assert table.peer_proxied == frozenset()


def test_absent_proxy_knob_is_not_proxied() -> None:
    table, _cfg = build_config(_spark_lobe_env(MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN))
    assert table.peer_proxied == frozenset()


# ============================================================================
# peer_api_keys: verbatim (stripped), and only alongside a declared origin
# ============================================================================


def test_peer_api_key_populated_verbatim_when_origin_present() -> None:
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_API_KEY="sk-lobes-thor-0001",
        )
    )
    assert dict(table.peer_api_keys) == {"multimodal": "sk-lobes-thor-0001"}


def test_peer_api_key_is_stripped_not_transformed() -> None:
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_API_KEY="  MiXeD-Case-Key==  ",
        )
    )
    assert dict(table.peer_api_keys) == {"multimodal": "MiXeD-Case-Key=="}


def test_peer_api_key_without_origin_is_inert() -> None:
    # A key with no origin has no peer to authenticate to — omitted.
    table, _cfg = build_config(_spark_lobe_env(MULTIMODAL_PEER_API_KEY="sk-orphan"))
    assert dict(table.peer_api_keys) == {}


def test_peer_api_key_blank_is_omitted() -> None:
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_API_KEY="   ",
        )
    )
    assert dict(table.peer_api_keys) == {}


def test_peer_api_key_needs_no_proxy_knob() -> None:
    # Keys ride the origin declaration, not the proxy knob — a referral-only
    # peer may still carry a credential (harmless until a later task dials).
    table, _cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_API_KEY="sk-referral-only",
        )
    )
    assert table.peer_proxied == frozenset()
    assert dict(table.peer_api_keys) == {"multimodal": "sk-referral-only"}


# ============================================================================
# ServerConfig.api_key: GATEWAY_API_KEY → CULTURE_VLLM_API_KEY → None
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
# Secrets never appear in repr/str
# ============================================================================


def test_key_values_never_appear_in_repr_or_str() -> None:
    peer_secret = "sk-peer-secret-do-not-print"  # nosec B105 — test fixture, not a credential
    gateway_secret = "sk-gateway-secret-do-not-print"  # nosec B105 — test fixture
    table, cfg = build_config(
        _spark_lobe_env(
            MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN,
            MULTIMODAL_PEER_API_KEY=peer_secret,
            GATEWAY_API_KEY=gateway_secret,
        )
    )
    # The values ARE carried (the later data-plane task needs them) ...
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
    # Equality against a table constructed with ONLY pre-existing fields —
    # proving the new fields default inert AND no pre-existing field moved.
    assert table == RoutingTable(
        backends=(primary,),
        default_model=_CORTEX_ID,
        aliases=tier_aliases([primary], TIER_ROLE),
        # The ONE deliberate delta since muse landed: the opt-in muse lobe is
        # unwired here (no MUSE_BASE_URL) and unflagged, so it defaults to
        # INFEASIBLE (OPT_IN_BACKENDS) — `model=muse` 404s role_infeasible
        # instead of silently upward-falling-back to the primary. Every
        # pre-muse behaviour is otherwise unchanged.
        infeasible=frozenset({"muse"}),
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


def test_new_routing_fields_default_inert_on_direct_construction() -> None:
    # Every existing RoutingTable(...) construction in the codebase/tests
    # omits the new fields — they must default to empty.
    table = RoutingTable(
        backends=(Backend(name="primary", base_url="http://x:1", served_name="m"),),
        default_model="m",
        aliases={},
    )
    assert table.peer_proxied == frozenset()
    assert dict(table.peer_api_keys) == {}
