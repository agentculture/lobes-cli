"""Rollout plan: members, waves, skips. Pure over already-gathered inputs (no I/O)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from lobes.rollout._drift import DriftReport
from lobes.rollout._observe import MemberObservation
from lobes.rollout._residents import Residents
from lobes.rollout._versions import is_downgrade, parse_version

PLANNED = "planned"
SKIPPED_GUEST = "skipped-guest"
SKIPPED_CURRENT = "skipped-current"
SKIPPED_OFFLINE = "skipped-offline"
REFUSED_DOWNGRADE = "refused-downgrade"


@dataclass
class MemberPlan:
    name: str
    status: str
    reason: str = ""
    running: str | None = None
    unique_roles: list[str] = field(default_factory=list)
    pending: bool = False
    is_local: bool = False
    drift: DriftReport | None = None


@dataclass
class Wave:
    index: int
    members: list[str]


@dataclass
class RolloutPlan:
    target: str
    local: str
    canary: str | None = None
    members: list[MemberPlan] = field(default_factory=list)
    waves: list[Wave] = field(default_factory=list)

    def member(self, name: str) -> MemberPlan:
        for m in self.members:
            if m.name == name:
                return m
        raise KeyError(name)

    def to_dict(self) -> dict:
        return asdict(self)


def _serves(obs: MemberObservation | None) -> set[str]:
    if obs is None or not obs.reachable:
        return set()
    return {
        r
        for r, c in obs.capabilities.items()
        if isinstance(c, dict) and c.get("feasible") and not c.get("proxied")
    }


def build_plan(
    residents: Residents,
    observations: dict[str, MemberObservation],
    drift: dict[str, DriftReport],
    target: str,
    local: str,
    *,
    canary: str | None = None,
    allow_downgrade: bool = False,
    wave_size: int = 1,
) -> RolloutPlan:
    """Build a plan from plain data. Performs no filesystem, network or process access."""
    parse_version(target)
    size = max(1, int(wave_size))
    served = {n: _serves(o) for n, o in observations.items()}
    plan = RolloutPlan(target=target, local=local)

    for name in sorted(residents.guests):
        plan.members.append(MemberPlan(name, SKIPPED_GUEST, "guest, not managed"))

    eligible: list[str] = []
    for name in sorted(residents.residents):
        o = observations.get(name)
        mp = MemberPlan(name, PLANNED, is_local=name == local, drift=drift.get(name))
        plan.members.append(mp)
        if o is None or not o.reachable or not o.version:
            mp.status, mp.reason, mp.pending = SKIPPED_OFFLINE, "unreachable", True
            continue
        mp.running = o.version
        others = set().union(*(s for n, s in served.items() if n != name and n in observations))
        mp.unique_roles = sorted(served.get(name, set()) - others)
        if is_downgrade(target, o.version) and not allow_downgrade:
            mp.status, mp.reason = REFUSED_DOWNGRADE, f"{o.version} is newer than {target}"
        elif parse_version(target) == parse_version(o.version):
            mp.status, mp.reason = SKIPPED_CURRENT, "already at target"
        else:
            eligible.append(name)

    if canary is not None and canary not in eligible:
        raise ValueError(f"canary {canary!r} is not a plannable resident")
    if canary is None:
        pool = [n for n in eligible if n != local] or eligible
        if pool:
            canary = min(pool, key=lambda n: (len(plan.member(n).unique_roles), n))
    plan.canary = canary

    rest = [n for n in eligible if n not in (canary, local)]
    order = ([canary] if canary and canary != local else []) + rest
    if local in eligible:
        order.append(local)
    plan.waves = [
        Wave(i + 1, order[j : j + size]) for i, j in enumerate(range(0, len(order), size))
    ]
    return plan
