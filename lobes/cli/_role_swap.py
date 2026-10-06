"""Exclusive-role swaps and the memory gate for ``lobes up`` (``--replace``).

A card profile can declare roles that must never run together on that board
(``[[exclusive_roles]]``; the Spark declares ``cortex`` + ``innereye``, because
both draw on the GB10's single unified memory pool). Until now only ``lobes
init --shape`` read that declaration, and a box with hand-kept compose files
can't be re-scaffolded. This module lets ``lobes up <role>`` enforce it on a
live box:

* **Refuse** to start a role while an exclusive rival's container is running.
* **``--replace``** does the whole swap: stop the rivals, then rewrite the few
  ``.env`` keys the gateway and compose read, so a rival is reported
  ``feasible:false`` and the mesh serves it. The keys are the role's
  ``*_FEASIBLE`` flag, its compose profile, and its base-URL wiring. Then start
  the role and recreate the gateway so it reads the new keys.
* **Memory gate.** Refuse when ``MemAvailable`` is below what the role needs:
  the card's ``declared_peak_gib``, else the role's ``*_GPU_MEM_UTIL`` share of
  ``MemTotal``. Under ``--replace`` the gate runs after the rivals stop, and a
  shortfall restarts them, so a refused swap leaves the box as it was.
  ``--override-memory`` skips the gate.

``.env`` is backed up before the first write, and only keys whose value changes
are written.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from lobes.gateway._config import FEASIBLE_ENV
from lobes.profiles.loader import resolve_profile
from lobes.profiles.shape_render import OPT_IN_CORE_ACTIVATION_ENV
from lobes.profiles.shapes import OPT_IN_CORE_ROLES
from lobes.runtime import _env


def _role_backend(role: str) -> str:
    # Imported late: lobes.roles imports the gateway package, which imports
    # lobes.roles back, so a module-level import here can start that cycle
    # from the wrong end.
    from lobes.roles import ROLE_BACKEND

    return ROLE_BACKEND.get(role, role)


PROFILE_KEY = "LOBES_PROFILE"
PROFILES_KEY = "COMPOSE_PROFILES"
CONTAINER_PREFIX = "model-gear-"
MEMINFO = Path("/proc/meminfo")
_GIB_KB = 1024 * 1024

# How long to wait for a stopped rival's memory to come back before the gate
# decides. Unified-memory boards return a vLLM pool within seconds of `stop`.
RELEASE_WAIT_S = 60.0
RELEASE_POLL_S = 3.0


@dataclass
class Exclusivity:
    """The card's exclusive-role facts for one target role."""

    profile: str = ""
    rivals: list[str] = field(default_factory=list)
    reason: str = ""


def exclusivity(env: dict[str, str], deploy_dir: Path, target: str) -> Exclusivity:
    """The roles the deployment's card declares exclusive with ``target``.

    Empty when ``.env`` names no profile, the profile can't be resolved, or the
    card declares no group containing ``target``.
    """
    name = (env.get(PROFILE_KEY) or "").strip()
    if not name:
        return Exclusivity()
    try:
        profile = resolve_profile(name, deploy_dir)
    except Exception:  # an unknown/broken profile only disables the guard
        return Exclusivity(profile=name)
    found = Exclusivity(profile=profile.name)
    for group in profile.exclusive_roles:
        if target in group.roles:
            found.rivals += [r for r in group.roles if r != target and r not in found.rivals]
            found.reason = found.reason or group.reason
    return found


def container_for(service: str) -> str:
    return CONTAINER_PREFIX + service


def _profiles(env: dict[str, str]) -> list[str]:
    return [p.strip() for p in (env.get(PROFILES_KEY) or "").split(",") if p.strip()]


def env_changes(env: dict[str, str], target: str, rivals: list[str]) -> dict[str, str]:
    """The ``.env`` keys a swap to ``target`` writes, only those that change.

    Rivals are marked infeasible (an explicit ``false`` is what lets the mesh
    serve them) and leave ``COMPOSE_PROFILES``. The target is marked feasible,
    joins ``COMPOSE_PROFILES`` when it is profile-gated, and gets its base-URL
    wiring when that is unset.
    """
    want: dict[str, str] = {}
    profiles = _profiles(env)
    for rival in rivals:
        want[FEASIBLE_ENV[_role_backend(rival)]] = "false"
        if rival in OPT_IN_CORE_ROLES and rival in profiles:
            profiles.remove(rival)
    want[FEASIBLE_ENV[_role_backend(target)]] = "true"
    if target in OPT_IN_CORE_ROLES:
        if target not in profiles:
            profiles.append(target)
        for key, value in OPT_IN_CORE_ACTIVATION_ENV.get(target, {}).items():
            if not (env.get(key) or "").strip():
                want[key] = value
    want[PROFILES_KEY] = ",".join(profiles)
    return {k: v for k, v in want.items() if (env.get(k) or "").strip() != v}


def backup_env(env_path: Path, why: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = env_path.with_name(f"{env_path.name}.bak-{stamp}-{why}")
    backup.write_bytes(env_path.read_bytes())
    return backup


def write_env(env_path: Path, changes: dict[str, str]) -> None:
    for key, value in changes.items():
        _env.set_env(env_path, key, value)


# --- the memory gate ---------------------------------------------------------


def meminfo_gib(field_name: str, meminfo: Path | None = None) -> float | None:
    """One ``/proc/meminfo`` field in GiB, or None where it can't be read."""
    try:
        for line in (meminfo or MEMINFO).read_text(encoding="utf-8").splitlines():
            if line.startswith(field_name + ":"):
                return int(line.split()[1]) / _GIB_KB
    except (OSError, ValueError, IndexError):
        return None
    return None


def required_gib(
    env: dict[str, str], deploy_dir: Path, target: str, *, meminfo: Path | None = None
) -> tuple[float | None, str]:
    """``(GiB the role needs, where that figure came from)``; None when unknown."""
    name = (env.get(PROFILE_KEY) or "").strip()
    if name:
        try:
            role = resolve_profile(name, deploy_dir).roles.get(target)
        except Exception:  # an unresolvable profile just falls through
            role = None
        peak = getattr(role, "declared_peak_gib", None)
        if peak:
            return float(peak), f"declared_peak_gib in the {name} card profile"
    util_key = _role_backend(target).upper() + "_GPU_MEM_UTIL"
    raw = (env.get(util_key) or "").split("#")[0].strip()
    total = meminfo_gib("MemTotal", meminfo)
    try:
        util = float(raw)
    except ValueError:
        return None, ""
    if total is None or not 0 < util <= 1:
        return None, ""
    return util * total, f"{util_key}={raw} x MemTotal {total:.1f} GiB"


@dataclass
class MemoryVerdict:
    ok: bool
    required: float | None
    available: float | None
    source: str

    def describe(self) -> str:
        if self.required is None:
            return "memory: no requirement declared for this role; not checked"
        if self.available is None:
            return "memory: /proc/meminfo unreadable; not checked"
        word = "ok" if self.ok else "SHORT"
        return (
            f"memory {word}: needs {self.required:.1f} GiB ({self.source}), "
            f"MemAvailable {self.available:.1f} GiB"
        )


def memory_verdict(
    env: dict[str, str], deploy_dir: Path, target: str, *, meminfo: Path | None = None
) -> MemoryVerdict:
    required, source = required_gib(env, deploy_dir, target, meminfo=meminfo)
    available = meminfo_gib("MemAvailable", meminfo)
    ok = required is None or available is None or available >= required
    return MemoryVerdict(ok, required, available, source)


def wait_for_memory(
    env: dict[str, str],
    deploy_dir: Path,
    target: str,
    *,
    meminfo: Path | None = None,
    wait_s: float | None = None,
    poll_s: float | None = None,
    sleep=None,
) -> MemoryVerdict:
    """Re-check memory until it fits or ``wait_s`` passes (a stopped lane's
    pool takes a few seconds to come back)."""
    wait_s = RELEASE_WAIT_S if wait_s is None else wait_s
    poll_s = RELEASE_POLL_S if poll_s is None else poll_s
    sleep = sleep or time.sleep
    verdict = memory_verdict(env, deploy_dir, target, meminfo=meminfo)
    waited = 0.0
    while not verdict.ok and waited < wait_s:
        sleep(poll_s)
        waited += poll_s
        verdict = memory_verdict(env, deploy_dir, target, meminfo=meminfo)
    return verdict
