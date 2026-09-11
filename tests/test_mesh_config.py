"""Mesh config: parse LOBES_MESH_* env vars into a frozen MeshConfig (_mesh_config.py, t1).

This task PARSES the mesh keys only — no network, no I/O.
Every test runs against a plain ``dict`` so nothing touches the real env.

Contract pinned below:

* ``build_mesh_config(env)`` returns ``enabled=False`` with every field default
  when ``LOBES_MESH_KEY`` is unset; when set, ``enabled=True`` and every other
  knob is read.
* ``LOBES_MESH_KEY`` set without ``LOBES_MESH_NAME`` raises
  :class:`MeshConfigError` naming the missing key; name is NEVER derived from
  hostname.
* ``LOBES_MESH_SEEDS`` parses as a comma-separated list with trailing slashes
  trimmed; empty items (stray commas, leading/trailing commas) are dropped.
* ``LOBES_MESH_HEARTBEAT_S`` defaults to 60; non-positive values raise
  :class:`MeshHeartbeatError`.
* ``LOBES_MESH_MISSED_MAX`` defaults to 3; non-positive values raise
  :class:`MeshMissedMaxError`.
* ``LOBES_MESH_LEDGER_PATH`` is carried verbatim (stripped) when non-blank.
* No other module's config output changes — existing ``build_config`` tests
  pass unchanged (asserted by a no-op env).
"""

from __future__ import annotations

import pytest

from lobes.gateway._mesh_config import (
    MeshConfigError,
    MeshHeartbeatError,
    MeshMissedMaxError,
    build_mesh_config,
)

# A minimal env that touches NOTHING mesh-related — exercises the
# ``enabled=False`` default path and asserts no side-effect on existing config.


def _noop_env(**over: str) -> dict[str, str]:
    """An env that sets nothing mesh-related."""
    env: dict[str, str] = {}
    env.update(over)
    return env


# ============================================================================
# enabled=False: LOBES_MESH_KEY unset
# ============================================================================


def test_mesh_key_unset_yields_enabled_false_defaults() -> None:
    cfg = build_mesh_config(_noop_env())
    assert cfg.enabled is False
    assert cfg.key is None
    assert cfg.name is None
    assert cfg.seeds == ()
    assert cfg.heartbeat_s == 60
    assert cfg.missed_max == 3
    assert cfg.ledger_path is None


def test_mesh_key_blank_yields_enabled_false_defaults() -> None:
    cfg = build_mesh_config(_noop_env(LOBES_MESH_KEY=""))
    assert cfg.enabled is False
    assert cfg.key is None


def test_mesh_key_whitespace_yields_enabled_false_defaults() -> None:
    cfg = build_mesh_config(_noop_env(LOBES_MESH_KEY="   "))
    assert cfg.enabled is False
    assert cfg.key is None


def test_enabled_false_fields_are_none_or_empty() -> None:
    cfg = build_mesh_config(_noop_env(LOBES_MESH_KEY=""))
    assert cfg.name is None
    assert cfg.ledger_path is None
    assert cfg.seeds == ()


# ============================================================================
# MeshConfigError: LOBES_MESH_KEY set without LOBES_MESH_NAME
# ============================================================================


def test_key_set_without_name_raises_mesh_config_error() -> None:
    with pytest.raises(MeshConfigError, match="LOBES_MESH_NAME"):
        build_mesh_config(_noop_env(LOBES_MESH_KEY="sk-abc"))


@pytest.mark.parametrize("blank", ["", "   "])
def test_key_set_name_blank_raises_mesh_config_error(blank: str) -> None:
    with pytest.raises(MeshConfigError, match="LOBES_MESH_NAME"):
        build_mesh_config(_noop_env(LOBES_MESH_KEY="sk-abc", LOBES_MESH_NAME=blank))


def test_name_is_never_derived_from_hostname() -> None:
    """Even without LOBES_MESH_NAME, the error names the env var, not a
    hostname — proving name is never fabricating a fallback from the local
    box's view of itself.
    """
    err = pytest.raises(MeshConfigError)
    with err as exc_info:
        build_mesh_config(_noop_env(LOBES_MESH_KEY="sk-abc"))
    # The error message names the env var, not "hostname" or any FQDN.
    assert "hostname" not in str(exc_info.value).lower() or "LOBES_MESH_NAME" in str(exc_info.value)


# ============================================================================
# Full config: key + name set
# ============================================================================


def test_full_config_enabled_true_with_all_fields() -> None:
    cfg = build_mesh_config(
        _noop_env(
            LOBES_MESH_KEY="sk-mesh-key",
            LOBES_MESH_NAME="spark-box",
            LOBES_MESH_SEEDS="http://seed1.local:8001,http://seed2.local:8001/",
            LOBES_MESH_HEARTBEAT_S="30",
            LOBES_MESH_MISSED_MAX="5",
            LOBES_MESH_LEDGER_PATH="/tmp/mesh-ledger.json",
        )
    )
    assert cfg.enabled is True
    assert cfg.key == "sk-mesh-key"
    assert cfg.name == "spark-box"
    assert cfg.seeds == ("http://seed1.local:8001", "http://seed2.local:8001")
    assert cfg.heartbeat_s == 30
    assert cfg.missed_max == 5
    assert cfg.ledger_path == "/tmp/mesh-ledger.json"


def test_name_stripped_not_validated() -> None:
    """Name is taken verbatim (stripped), never validated against DNS or
    hostnames.
    """
    cfg = build_mesh_config(_noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="  my-box  "))
    assert cfg.name == "my-box"


# ============================================================================
# Seeds: comma-separated, trailing-slash trimmed, empty items dropped
# ============================================================================


def test_seeds_trailing_slash_trimmed() -> None:
    cfg = build_mesh_config(
        _noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b", LOBES_MESH_SEEDS="http://a:8001/")
    )
    assert cfg.seeds == ("http://a:8001",)


def test_seeds_comma_separated_multi() -> None:
    cfg = build_mesh_config(
        _noop_env(
            LOBES_MESH_KEY="sk-x",
            LOBES_MESH_NAME="b",
            LOBES_MESH_SEEDS="http://a:8001,http://b:8002/",
        )
    )
    assert cfg.seeds == ("http://a:8001", "http://b:8002")


def test_seeds_empty_items_dropped() -> None:
    """Stray commas, leading/trailing commas yield no blank origin."""
    cfg = build_mesh_config(
        _noop_env(
            LOBES_MESH_KEY="sk-x",
            LOBES_MESH_NAME="b",
            LOBES_MESH_SEEDS=",http://a:8001,,",
        )
    )
    assert cfg.seeds == ("http://a:8001",)


def test_seeds_whitespace_items_stripped() -> None:
    cfg = build_mesh_config(
        _noop_env(
            LOBES_MESH_KEY="sk-x",
            LOBES_MESH_NAME="b",
            LOBES_MESH_SEEDS="  http://a:8001  ,  http://b:8002  ",
        )
    )
    assert cfg.seeds == ("http://a:8001", "http://b:8002")


def test_seeds_blank_unset_yields_empty_tuple() -> None:
    cfg = build_mesh_config(
        _noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b", LOBES_MESH_SEEDS="")
    )
    assert cfg.seeds == ()


# ============================================================================
# Heartbeat: defaults 60, positive ints accepted, non-positive rejected
# ============================================================================


def test_heartbeat_default_60() -> None:
    cfg = build_mesh_config(_noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b"))
    assert cfg.heartbeat_s == 60


def test_heartbeat_custom_positive() -> None:
    cfg = build_mesh_config(
        _noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b", LOBES_MESH_HEARTBEAT_S="120")
    )
    assert cfg.heartbeat_s == 120


@pytest.mark.parametrize("value", ["0", "-1", "-10", "abc"])
def test_heartbeat_rejects_non_positive(value: str) -> None:
    with pytest.raises(MeshHeartbeatError):
        build_mesh_config(
            _noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b", LOBES_MESH_HEARTBEAT_S=value)
        )


# ============================================================================
# Missed max: defaults 3, positive ints accepted, non-positive rejected
# ============================================================================


def test_missed_max_default_3() -> None:
    cfg = build_mesh_config(_noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b"))
    assert cfg.missed_max == 3


def test_missed_max_custom_positive() -> None:
    cfg = build_mesh_config(
        _noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b", LOBES_MESH_MISSED_MAX="10")
    )
    assert cfg.missed_max == 10


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_missed_max_rejects_non_positive(value: str) -> None:
    with pytest.raises(MeshMissedMaxError):
        build_mesh_config(
            _noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b", LOBES_MESH_MISSED_MAX=value)
        )


# ============================================================================
# Ledger path: verbatim, stripped, None when blank
# ============================================================================


def test_ledger_path_stored_verbatim() -> None:
    cfg = build_mesh_config(
        _noop_env(
            LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b", LOBES_MESH_LEDGER_PATH="/var/run/mesh.json"
        )
    )
    assert cfg.ledger_path == "/var/run/mesh.json"


def test_ledger_path_none_when_blank() -> None:
    cfg = build_mesh_config(
        _noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b", LOBES_MESH_LEDGER_PATH="")
    )
    assert cfg.ledger_path is None


def test_ledger_path_none_when_unset() -> None:
    cfg = build_mesh_config(_noop_env(LOBES_MESH_KEY="sk-x", LOBES_MESH_NAME="b"))
    assert cfg.ledger_path is None


# ============================================================================
# Frozen dataclass: MeshConfig is immutable
# ============================================================================


def test_mesh_config_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    cfg = build_mesh_config(
        _noop_env(
            LOBES_MESH_KEY="sk-x",
            LOBES_MESH_NAME="b",
        )
    )
    with pytest.raises(FrozenInstanceError):
        cfg.key = "new-key"


# ============================================================================
# No side-effects on existing config
# ============================================================================


def test_build_mesh_config_no_side_effect_on_build_config() -> None:
    """Calling build_mesh_config does not mutate any global state that
    affects lobes.gateway._config.build_config. A no-op env (no mesh keys,
    no gateway keys) still produces the expected default config.
    """
    from lobes.gateway._config import build_config

    # build_mesh_config touches nothing in _config.py
    build_mesh_config({})

    # A minimal env still produces the default RoutingTable + ServerConfig
    table, cfg = build_config({})
    # The primary backend is always present with its default served name.
    assert len(table.backends) >= 1
    assert table.backends[0].name == "primary"
