"""Exclusive-role swaps and the memory gate for ``lobes up`` (``--replace``).

A card profile can declare roles that must never run together on that board
(``[[exclusive_roles]]``; the Spark declares ``cortex`` + ``innereye``, because
both draw on the GB10's single unified memory pool). Until now only ``lobes
init --shape`` read that declaration, and a box with hand-kept compose files
can't be re-scaffolded. This module lets ``lobes up <role>`` enforce it on a
live box:

* **Refuse** to start a role while an exclusive rival's container is running.
* **``--replace``** does the whole swap: stop the rivals, then rewrite the few
  ``.env`` keys the gateway and compose read (a rival's ``*_FEASIBLE=false`` so
  the mesh serves it, the role's compose profile and base-URL wiring), start the
  role and recreate the gateway so it reads them. A failure after the stop puts
  ``.env`` back and restarts the rivals.
* **Memory gate**, for a role that has exclusive rivals or a declared peak on
  its card: refuse when ``MemAvailable`` is below what the role needs (the
  card's ``declared_peak_gib``, else its ``*_GPU_MEM_UTIL`` share of
  ``MemTotal`` -- meaningful because exclusivity is only declared on
  unified-memory cards). ``--override-memory`` skips it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from lobes.gateway._config import FEASIBLE_ENV
from lobes.profiles.loader import resolve_profile
from lobes.profiles.render import LLAMA_CPP_COMPOSE_PROFILE, ROLE_ENV_PREFIX
from lobes.profiles.schema import Profile
from lobes.profiles.shape_render import (
    LLAMA_CPP_ROLE_SERVICE,
    OPT_IN_CORE_ACTIVATION_ENV,
    OPT_IN_CORE_COMPOSE_PROFILE,
    ROLE_SERVICE,
)
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


# --- the card ----------------------------------------------------------------


@dataclass
class Card:
    """The deployment's card profile, resolved once per ``lobes up``."""

    name: str = ""
    profile: Profile | None = None
    warning: str = ""


def load_card(env: dict[str, str], deploy_dir: Path) -> Card:
    """Resolve ``LOBES_PROFILE``. A name that won't resolve is reported, not
    swallowed: it turns the exclusive-role guard off, and the operator should
    know that."""
    name = (env.get(PROFILE_KEY) or "").strip()
    if not name:
        return Card()
    try:
        return Card(name=name, profile=resolve_profile(name, deploy_dir))
    except Exception as exc:  # any load/validation error has the same consequence
        return Card(
            name=name,
            warning=(
                f"card profile {name!r} did not load ({exc}); exclusive-role and "
                "memory checks are off for this command"
            ),
        )


@dataclass
class Exclusivity:
    """The card's exclusive-role facts for one target role."""

    rivals: list[str] = field(default_factory=list)
    reason: str = ""


def exclusivity(card: Card, target: str) -> Exclusivity:
    """The roles ``card`` declares exclusive with ``target``."""
    found = Exclusivity()
    if card.profile is None:
        return found
    for group in card.profile.exclusive_roles:
        if target in group.roles:
            found.rivals += [r for r in group.roles if r != target and r not in found.rivals]
            found.reason = found.reason or group.reason
    return found


def declared_peak(card: Card, target: str) -> float | None:
    if card.profile is None:
        return None
    peak = getattr(card.profile.roles.get(target), "declared_peak_gib", None)
    return float(peak) if peak else None


# --- services ----------------------------------------------------------------


def container_for(service: str) -> str:
    return CONTAINER_PREFIX + service


def role_services(role: str) -> list[str]:
    """Every compose service that can serve ``role``: its vLLM lane plus any
    alternative-engine lane (``llamacpp-primary`` for cortex)."""
    services = [ROLE_SERVICE[role]]
    alt = LLAMA_CPP_ROLE_SERVICE.get(role)
    if alt:
        services.append(alt)
    return services


def service_profile(role: str, service: str) -> str | None:
    """The compose profile gating ``service``, if any. Compose can't see a
    profile-gated service, even to stop it, unless that profile is active."""
    if service == LLAMA_CPP_ROLE_SERVICE.get(role):
        return LLAMA_CPP_COMPOSE_PROFILE
    return OPT_IN_CORE_COMPOSE_PROFILE.get(role)


# --- .env --------------------------------------------------------------------


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
        rival_profile = OPT_IN_CORE_COMPOSE_PROFILE.get(rival)
        if rival_profile in profiles:
            profiles.remove(rival_profile)
    want[FEASIBLE_ENV[_role_backend(target)]] = "true"
    target_profile = OPT_IN_CORE_COMPOSE_PROFILE.get(target)
    if target_profile:
        if target_profile not in profiles:
            profiles.append(target_profile)
        for key, value in OPT_IN_CORE_ACTIVATION_ENV.get(target, {}).items():
            if not (env.get(key) or "").strip():
                want[key] = value
    want[PROFILES_KEY] = ",".join(profiles)
    return {k: v for k, v in want.items() if (env.get(k) or "").strip() != v}


# Backup names come from this table, never from the command line, so no
# caller-supplied text reaches a path.
_BACKUP_TAG = {role: f"switch-to-{role}" for role in ROLE_SERVICE}


def backup_env(env_path: Path, target: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = env_path.with_name(f"{env_path.name}.bak-{stamp}-{_BACKUP_TAG[target]}")
    backup.write_bytes(env_path.read_bytes())
    return backup


def _replace_atomically(env_path: Path, text: str) -> None:
    tmp = env_path.with_name(env_path.name + ".tmp-switch")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, env_path.stat().st_mode & 0o777)
    os.replace(tmp, env_path)


def restore_env(env_path: Path, backup: Path) -> None:
    _replace_atomically(env_path, backup.read_text(encoding="utf-8"))


def write_env(env_path: Path, changes: dict[str, str]) -> None:
    """Apply ``changes`` in one atomic write: existing keys are rewritten in
    place, missing ones appended, and the file is swapped in with
    ``os.replace`` so an interruption never leaves it half-switched."""
    for key, value in changes.items():
        _env.check_value(value, key)
    pending = dict(changes)
    out: list[str] = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key = line.split("=", 1)[0] if "=" in line else None
        if key in pending:
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in pending.items()]
    _replace_atomically(env_path, "\n".join(out) + "\n")


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


def gate_applies(card: Card, target: str) -> bool:
    """The gate runs only where its arithmetic means something: a role with
    exclusive rivals (declared only on unified-memory cards) or a declared
    peak. A discrete-GPU box never has its GPU budget compared with host RAM."""
    return bool(exclusivity(card, target).rivals) or declared_peak(card, target) is not None


def required_gib(
    env: dict[str, str], card: Card, target: str, *, meminfo: Path | None = None
) -> tuple[float | None, str]:
    """``(GiB the role needs, where that figure came from)``; None when unknown."""
    peak = declared_peak(card, target)
    if peak:
        return peak, f"declared_peak_gib in the {card.name} card profile"
    prefix = ROLE_ENV_PREFIX.get(target)
    if not prefix:
        return None, ""
    util_key = prefix + "_GPU_MEM_UTIL"
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
    env: dict[str, str], card: Card, target: str, *, meminfo: Path | None = None
) -> MemoryVerdict:
    required, source = required_gib(env, card, target, meminfo=meminfo)
    available = meminfo_gib("MemAvailable", meminfo)
    ok = required is None or available is None or available >= required
    return MemoryVerdict(ok, required, available, source)


def wait_for_memory(
    env: dict[str, str],
    card: Card,
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
    verdict = memory_verdict(env, card, target, meminfo=meminfo)
    waited = 0.0
    while not verdict.ok and waited < wait_s:
        sleep(poll_s)
        waited += poll_s
        verdict = memory_verdict(env, card, target, meminfo=meminfo)
    return verdict
