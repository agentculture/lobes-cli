"""``lobes doctor`` residency findings (read-only).

* ``new_mesh_member`` — a verified mesh member is in neither the residents
  nor the guests table. Needs the local gateway's ``GET /mesh/roster``; when
  the gateway is unreachable or no mesh key is set, the finding is skipped.
* ``resident_field_missing`` — a resident has a field the operator must supply.
* ``gateway_open`` — a resident's ``gateway_key`` is absent/false.

No ``residents.toml`` means no findings at all (mirrors ``lock_drift``).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from pathlib import Path

from lobes.rollout import _residents
from lobes.runtime import _env

_MESH_KEY_ENV = "LOBES_MESH_KEY"
_TIMEOUT = 3.0


def _finding(fid: str, message: str, remediation: str) -> dict:
    return {
        "id": fid,
        "passed": False,
        "severity": "warn",
        "message": message,
        "remediation": remediation,
    }


def _http_roster(port: int, key: str) -> dict | None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/mesh/roster", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def _roster(deploy_dir: Path, fetch: Callable[[int, str], dict | None]) -> dict | None:
    env_path = Path(deploy_dir) / ".env"
    key = _env.read_env(env_path, _MESH_KEY_ENV)
    if not key:
        return None
    try:
        port = _env.parse_port(_env.read_env(env_path, "VLLM_PORT", "8000"))
        return fetch(port, key)
    except Exception:  # noqa: BLE001 - doctor must never fail on a probe
        return None


def residents_checks(
    deploy_dir: Path, fetch: Callable[[int, str], dict | None] = _http_roster
) -> list[dict]:
    try:
        res = _residents.load(deploy_dir)
    except Exception:  # noqa: BLE001 - unreadable file: stay silent, not fatal
        return []
    if not res.residents and not res.guests:
        return []
    out: list[dict] = []
    roster = _roster(deploy_dir, fetch)
    if roster:
        known = set(res.residents) | set(res.guests)
        for name in _residents._verified(roster):
            if name not in known:
                out.append(
                    _finding(
                        "new_mesh_member",
                        f"verified mesh member '{name}' is not in residents.toml",
                        f"run 'lobes mesh resident add {name}'",
                    )
                )
    for name, entry in res.residents.items():
        for fname in _residents.FIELDS:
            f = entry[fname]
            if fname == "gateway_key":
                if f.state == _residents.NEEDS_YOU or f.value is False:
                    out.append(
                        _finding(
                            "gateway_open",
                            f"resident '{name}' has no gateway key — its gateway is open",
                            f"set GATEWAY_API_KEY on '{name}'",
                        )
                    )
            elif f.state == _residents.NEEDS_YOU:
                out.append(
                    _finding(
                        "resident_field_missing",
                        f"resident '{name}' is missing '{fname}'",
                        f"edit residents.toml: set '{fname}' under [resident.{name}]",
                    )
                )
    return out
