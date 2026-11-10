"""Read-only member observation: /health version, /capabilities, /mesh/roster.

Only GET requests are issued. The HTTP opener is injected so tests never
touch the network; credentials are passed by the caller and never logged.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

Opener = Callable[..., Any]


@dataclass
class MemberObservation:
    origin: str
    version: str | None = None
    capabilities: dict[str, dict] = field(default_factory=dict)
    roster: list[dict] = field(default_factory=list)
    reachable: bool = False
    # the raw /health body (t21: carries optional memory/tunnel evidence)
    health: dict = field(default_factory=dict)


@dataclass
class CapabilityDiff:
    regressions: list[str]
    new_roles: list[str]
    improved: list[str]

    @property
    def ok(self) -> bool:
        return not self.regressions


def _get_json(opener: Opener, url: str, key: str | None, timeout: float) -> Any:
    req = urllib.request.Request(url, method="GET")  # nosec B310 - operator-typed origin
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with opener(req, timeout=timeout) as r:
            if not (200 <= getattr(r, "status", 200) < 300):
                return None
            return json.loads(r.read())
    except (OSError, ValueError, TypeError):
        return None


def _roles(payload: Any) -> dict[str, dict]:
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("roles")
    src = inner if isinstance(inner, dict) else payload
    return {k: v for k, v in src.items() if isinstance(v, dict)}


def observe_member(
    origin: str,
    *,
    gateway_key: str | None = None,
    mesh_key: str | None = None,
    opener: Opener = urllib.request.urlopen,  # nosec B310
    timeout: float = 5.0,
) -> MemberObservation:
    """Observe one member with three GETs; failures degrade to empty fields."""
    base = origin.rstrip("/")
    health = _get_json(opener, f"{base}/health", None, timeout)
    caps = _get_json(opener, f"{base}/capabilities", gateway_key, timeout)
    roster = _get_json(opener, f"{base}/mesh/roster", mesh_key, timeout)
    members = roster.get("members") if isinstance(roster, dict) else None
    version = health.get("version") if isinstance(health, dict) else None
    return MemberObservation(
        origin=origin,
        version=version if isinstance(version, str) else None,
        capabilities=_roles(caps),
        roster=[m for m in members if isinstance(m, dict)] if isinstance(members, list) else [],
        reachable=isinstance(health, dict),
        health=health if isinstance(health, dict) else {},
    )


def _up(role: dict | None) -> bool:
    return bool(role and role.get("feasible") and role.get("ready"))


def diff(before: MemberObservation, after: MemberObservation) -> CapabilityDiff:
    """Roles feasible+ready before and not after are regressions."""
    regressions = sorted(
        r for r, v in before.capabilities.items() if _up(v) and not _up(after.capabilities.get(r))
    )
    new_roles = sorted(set(after.capabilities) - set(before.capabilities))
    improved = sorted(
        r for r, v in before.capabilities.items() if not _up(v) and _up(after.capabilities.get(r))
    )
    return CapabilityDiff(regressions, new_roles, improved)
