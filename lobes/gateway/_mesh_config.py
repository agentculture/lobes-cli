"""Build a frozen :class:`MeshConfig` from environment variables.

Reads a mapping (``os.environ`` by default) and constructs a single
immutable config object. No sockets — pass a plain ``dict`` to unit-test
it offline. The env keys mirror the mesh-join keys that land in the
gateway compose template at a later task.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# Default heartbeat interval (seconds).
_DEFAULT_HEARTBEAT_S: int = 60
# Default consecutive missed heartbeat count before a member is dropped.
_DEFAULT_MISSED_MAX: int = 3


class MeshConfigError(ValueError):
    """A required mesh configuration field is missing or invalid.

    Raised by :func:`build_mesh_config` when ``LOBES_MESH_KEY`` is set but
    ``LOBES_MESH_NAME`` is absent/blank, or when another required field
    disagrees with its companion (future extensions).
    """


class MeshHeartbeatError(ValueError):
    """``LOBES_MESH_HEARTBEAT_S`` could not be parsed as a positive integer.

    Raised by :func:`build_mesh_config` when the value is ``0`` or negative,
    or non-numeric — a heartbeat of zero or less has no operational meaning.
    """


class MeshMissedMaxError(ValueError):
    """``LOBES_MESH_MISSED_MAX`` could not be parsed as a positive integer.

    Raised by :func:`build_mesh_config` when the value is ``0`` or negative,
    or non-numeric — a missed-max of zero or less would drop a member on
    the very first heartbeat tick.
    """


@dataclass(frozen=True)
class MeshConfig:
    """Immutable mesh-join configuration parsed from env vars.

    ``enabled`` is False when ``LOBES_MESH_KEY`` is unset or blank — all
    other fields carry their default/None value in that case, so callers
    can check ``cfg.enabled`` before inspecting anything else.
    """

    enabled: bool
    key: str | None
    name: str | None
    seeds: tuple[str, ...]
    heartbeat_s: int
    missed_max: int
    ledger_path: str | None


def _parse_seeds(raw: str) -> tuple[str, ...]:
    """Parse ``LOBES_MESH_SEEDS`` as a comma-separated list.

    Each item is stripped of leading/trailing whitespace and has its
    trailing slash removed (mirroring the origin-trimming convention
    used everywhere else in :mod:`lobes.gateway._config`). Empty items
    (stray commas, leading/trailing commas) are dropped — there is no
    such thing as a seed at the empty-string origin.
    """
    if not raw:
        return ()
    return tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip())


def _parse_positive_int(env: Mapping[str, str], key: str, *, label: str) -> int:
    """Parse ``env[key]`` as a positive integer.

    Raises :class:`MeshHeartbeatError` or :class:`MeshMissedMaxError`
    (chosen by ``label``) when the value is ``0``, negative, or
    non-numeric.
    """
    raw = (env.get(key) or "").strip()
    if not raw:
        return _DEFAULT_HEARTBEAT_S if "HEARTBEAT" in label else _DEFAULT_MISSED_MAX
    try:
        value = int(raw)
    except ValueError as exc:
        if "HEARTBEAT" in label:
            raise MeshHeartbeatError(
                f"{key}={raw!r} is not a valid positive integer — "
                "the heartbeat interval must be a whole number of seconds."
            ) from exc
        else:
            raise MeshMissedMaxError(
                f"{key}={raw!r} is not a valid positive integer — "
                "missed-max must be a whole number of ticks."
            ) from exc
    if value <= 0:
        if "HEARTBEAT" in label:
            raise MeshHeartbeatError(f"{key}={value} must be a positive integer (greater than 0).")
        else:
            raise MeshMissedMaxError(f"{key}={value} must be a positive integer (greater than 0).")
    return value


def build_mesh_config(env: Mapping[str, str] | None = None) -> MeshConfig:
    """Construct the mesh config from environment variables.

    When ``LOBES_MESH_KEY`` is unset/blank, returns a config with
    ``enabled=False`` and every field at its default — this is the
    no-op posture that leaves the box operating without mesh
    coordination.

    When ``LOBES_MESH_KEY`` is set but ``LOBES_MESH_NAME`` is absent or
    blank, raises :class:`MeshConfigError` naming the missing key.
    The name is NEVER derived from the local box's hostname or any
    other ambient fact — it must be operator-declared.

    ``LOBES_MESH_SEEDS`` is parsed as a comma-separated list with
    trailing slashes trimmed (see :func:`_parse_seeds`).

    ``LOBES_MESH_HEARTBEAT_S`` defaults to :data:`_DEFAULT_HEARTBEAT_S`
    (60); non-positive values raise :class:`MeshHeartbeatError`.

    ``LOBES_MESH_MISSED_MAX`` defaults to :data:`_DEFAULT_MISSED_MAX` (3);
    non-positive values raise :class:`MeshMissedMaxError`.

    ``LOBES_MESH_LEDGER_PATH`` is carried verbatim (stripped) when
    non-blank; ``None`` when absent or blank.
    """
    env = os.environ if env is None else env

    key = (env.get("LOBES_MESH_KEY") or "").strip()
    if not key:
        return MeshConfig(
            enabled=False,
            key=None,
            name=None,
            seeds=(),
            heartbeat_s=_DEFAULT_HEARTBEAT_S,
            missed_max=_DEFAULT_MISSED_MAX,
            ledger_path=None,
        )

    name = (env.get("LOBES_MESH_NAME") or "").strip()
    if not name:
        raise MeshConfigError(
            "LOBES_MESH_KEY is set but LOBES_MESH_NAME is missing or blank — "
            "the member name must be operator-declared; it is never derived "
            "from the local box's hostname or network view."
        )

    seeds = _parse_seeds(env.get("LOBES_MESH_SEEDS") or "")
    heartbeat_s = _parse_positive_int(env, "LOBES_MESH_HEARTBEAT_S", label="HEARTBEAT_S")
    missed_max = _parse_positive_int(env, "LOBES_MESH_MISSED_MAX", label="MISSED_MAX")
    ledger_path = (env.get("LOBES_MESH_LEDGER_PATH") or "").strip() or None

    return MeshConfig(
        enabled=True,
        key=key,
        name=name,
        seeds=seeds,
        heartbeat_s=heartbeat_s,
        missed_max=missed_max,
        ledger_path=ledger_path,
    )
