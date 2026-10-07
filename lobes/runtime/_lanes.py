"""Specialist embed/rerank lane <-> compose service mapping (orin-embedding-specialist, t9).

The registry (:data:`lobes.embed_lanes.EMBED_LANES`) is the single source of
truth for lane names. A lane's compose SERVICE is ``embed-<lane name>``; the
services themselves are declared by an overlay that a deployment may or may not
carry, so every consumer asks :func:`defined_lane_services` rather than assuming.
Stdlib line scan only (the runtime carries no YAML parser).
"""

from __future__ import annotations

from pathlib import Path

from lobes.embed_lanes import EMBED_LANES, EmbedLane
from lobes.runtime import _compose

SERVICE_PREFIX = "embed-"


def lane_service(name: str) -> str:
    return SERVICE_PREFIX + name


LANE_SERVICE: dict[str, str] = {lane.name: lane_service(lane.name) for lane in EMBED_LANES}


def lane_by_name(name: str) -> EmbedLane | None:
    return next((lane for lane in EMBED_LANES if lane.name == name), None)


def chain_files(deploy_dir: Path, chain: list[str]) -> list[str]:
    """File names behind a ``-f`` chain; ``[]`` chain means compose's own default."""
    named = [chain[i + 1] for i in range(0, len(chain) - 1) if chain[i] == "-f"]
    if named:
        return named
    files = [_compose.COMPOSE_FILE]
    if _compose.local_override_present(deploy_dir):
        files.append(_compose.LOCAL_OVERRIDE)
    return files


def declared_services(deploy_dir: Path, chain: list[str]) -> set[str]:
    """Every top-level service key declared by the files of ``chain``."""
    keys: set[str] = set()
    for name in chain_files(deploy_dir, chain):
        path = Path(deploy_dir) / name
        if path.is_file():
            keys |= _compose._override_service_keys(path.read_text(encoding="utf-8"))
    return keys


def defined_lane_services(deploy_dir: Path, chain: list[str]) -> dict[str, str]:
    """``{lane name: service}`` for the lanes the deployment's compose set defines."""
    declared = declared_services(deploy_dir, chain)
    return {name: svc for name, svc in LANE_SERVICE.items() if svc in declared}
