"""Every env key ``lobes/gateway/_config.py`` reads must reach the gateway
container through the fleet compose template's ``environment:`` passthrough.

Why this guard exists (capacity-relative-pool-routing, t12). The gateway
service deliberately does NOT use ``env_file`` — it would inherit ``HF_TOKEN``
and every other ``.env`` secret — so it enumerates the scoped keys it consumes,
one ``- NAME=${NAME:-}`` line at a time. That design makes a whole class of
defect possible and invisible: a new knob is parsed by ``_config.py``,
documented in ``env.example``, set by an operator in ``.env`` … and never
reaches the container, so the feature is permanently inert in every real
deployment while every unit test passes.

That is exactly what happened to the capacity signal: all ten
``<PREFIX>_MAX_ACTIVE`` knobs and ``GATEWAY_CAPACITY_KILL_SWITCH`` were parsed,
ranked and published, and none of them were in the compose passthrough
(verified live: ``docker exec <gateway> env | grep -c MAX_ACTIVE`` → 0). Every
existing test constructs ``ServerConfig`` or an env mapping in-process, which
bypasses compose entirely, so no test could have caught it.

The expected key set here is DERIVED from ``_config.py`` itself, never
hardcoded, so a knob added there without a compose line fails this test
automatically. Three derivation sources:

  1. Every module-level ``*_ENV`` constant — a ``str`` contributes its own
     value, a ``dict`` contributes all of its values. This covers
     ``FEASIBLE_ENV``, ``PEER_ORIGIN_ENV``, ``PEER_PROXY_ENV``,
     ``PEER_API_KEY_ENV``, ``PEER_ORIGINS_ENV``, ``PEER_API_KEYS_ENV``,
     ``MAX_ACTIVE_ENV``, ``CAPACITY_KILL_SWITCH_ENV``, ``HAND_LORA_MODULES_ENV``
     and anything added beside them later.
  2. The lane-fingerprint cross-product ``<PREFIX>_<SUFFIX>`` the config builds
     at runtime from ``FEASIBLE_ENV`` × ``LANE_FINGERPRINT_SUFFIXES`` — those
     keys exist only as an f-string, so no constant names them.
  3. An AST scan of the module for string literals passed to its own env
     readers (``env.get(...)``, ``_as_bool``/``_as_float``/``_as_int``) and to
     ``*_key`` keyword arguments (``_optional_backend(url_key="MINOR_BASE_URL",
     …)``). This catches one-off keys like ``GATEWAY_FORCE_STRICT_TOOLS`` that
     never got an ``*_ENV`` constant.

Anything genuinely not meant to be operator-settable via ``.env`` is listed in
:data:`_DELIBERATELY_NOT_PASSED_THROUGH` with its reason — an explicit,
reviewable exemption rather than a silent gap.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

from lobes.gateway import _config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_SOURCE = _REPO_ROOT / "lobes" / "gateway" / "_config.py"
_FLEET_COMPOSE = _REPO_ROOT / "lobes" / "templates" / "fleet" / "docker-compose.yml"

_ENV_KEY = re.compile(r"[A-Z][A-Z0-9_]*$")

# The env readers this module hands a literal key to. Any helper added beside
# them that takes ``(env, key)`` belongs here too.
_ENV_READERS = frozenset({"get", "_as_bool", "_as_float", "_as_int"})

# Keys ``_config.py`` reads that must NOT be plumbed from ``.env``, each with
# the reason it is exempt. Keep this list tiny and argued.
_DELIBERATELY_NOT_PASSED_THROUGH: dict[str, str] = {
    # The in-container bind address, deliberately pinned to 0.0.0.0 (see the
    # `nosec B104` note in _config.py): inside the container that is the only
    # correct value, and letting an operator's .env override it would make the
    # gateway unreachable from the compose network while looking healthy.
    # Reachability from the host is owned by compose's `ports:` mapping, and
    # the matching GATEWAY_PORT is pinned to 8000 in the template for the same
    # reason.
    "GATEWAY_HOST": "in-container bind address, pinned to 0.0.0.0 by design",
}


def _gateway_env_keys() -> set[str]:
    """The KEY of every ``- KEY=value`` line in the gateway service's env list."""
    compose = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))
    entries: list[str] = compose["services"]["gateway"]["environment"]
    return {entry.split("=", 1)[0] for entry in entries}


def _literal_env_keys() -> set[str]:
    """Env keys named as string literals inside ``_config.py`` (source 3 above)."""
    tree = ast.parse(_CONFIG_SOURCE.read_text(encoding="utf-8"))
    found: set[str] = set()

    def _maybe(node: ast.expr) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _ENV_KEY.match(node.value):
                found.add(node.value)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in _ENV_READERS:
            for arg in node.args:
                _maybe(arg)
        for keyword in node.keywords:
            if keyword.arg and keyword.arg.endswith("_key"):
                _maybe(keyword.value)
    return found


def _config_env_keys() -> set[str]:
    """Every env key ``_config.py`` reads, from all three derivation sources."""
    keys: set[str] = set()
    for name in dir(_config):
        if not name.endswith("_ENV"):
            continue
        value = getattr(_config, name)
        if isinstance(value, str):
            keys.add(value)
        elif isinstance(value, dict):
            keys.update(str(item) for item in value.values())
    for backend in _config.FEASIBLE_ENV:
        for suffix in _config.LANE_FINGERPRINT_SUFFIXES:
            keys.add(f"{backend.upper()}_{suffix}")
    keys |= _literal_env_keys()
    return {key for key in keys if _ENV_KEY.match(key)}


class TestGatewayEnvPassthroughGuard:
    def test_derivation_is_not_vacuous(self) -> None:
        """A derivation that silently produced nothing would pass everything."""
        keys = _config_env_keys()
        assert len(keys) > 100
        # One representative from each derivation source.
        assert "PRIMARY_PEER_ORIGINS" in keys  # dict *_ENV constant
        assert "GATEWAY_CAPACITY_KILL_SWITCH" in keys  # scalar *_ENV constant
        assert "PRIMARY_SPECULATIVE_CONFIG" in keys  # fingerprint cross-product
        assert "GATEWAY_FORCE_STRICT_TOOLS" in keys  # literal-only key

    def test_every_config_env_key_reaches_the_gateway_container(self) -> None:
        expected = _config_env_keys() - set(_DELIBERATELY_NOT_PASSED_THROUGH)
        missing = sorted(expected - _gateway_env_keys())
        assert not missing, (
            "lobes/gateway/_config.py reads env keys the fleet compose gateway "
            "service never passes through, so they are inert in every real "
            "deployment: " + ", ".join(missing)
        )

    def test_exemptions_are_real_keys_and_really_absent(self) -> None:
        """An exemption must name a key the config reads and compose omits.

        Otherwise the list rots into a place stale names hide, or — worse —
        silences a key that IS plumbed and could have been un-exempted.
        """
        config_keys = _config_env_keys()
        gateway_keys = _gateway_env_keys()
        for key, reason in _DELIBERATELY_NOT_PASSED_THROUGH.items():
            assert key in config_keys, f"{key} is exempted but _config.py never reads it"
            assert key not in gateway_keys, f"{key} is exempted but compose passes it"
            assert reason.strip(), f"{key} is exempted without a reason"

    def test_capacity_knobs_use_the_neighbouring_empty_default_style(self) -> None:
        """``${NAME:-}`` — an unset knob must stay empty, never a literal."""
        compose = yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))
        entries: list[str] = compose["services"]["gateway"]["environment"]
        raw = {entry.split("=", 1)[0]: entry.split("=", 1)[1] for entry in entries}
        capacity_keys = [*_config.MAX_ACTIVE_ENV.values(), _config.CAPACITY_KILL_SWITCH_ENV]
        for key in capacity_keys:
            assert raw[key] == "${" + key + ":-}", f"{key} must default to empty"
