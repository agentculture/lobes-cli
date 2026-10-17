"""``lobes doctor`` plugin-copy finding and the targeted ``--refresh-plugin`` lane.

Each box bind-mounts its OWN scaffold copy of the think-aware tool-parser
plugin (``./qwen3_thinking_tool_parser.py``), and ``doctor --fix`` writes it
missing-only, so a copy left behind by an upgrade is never replaced. This
mirrors the ``--refresh-logwrap`` lane in :mod:`._doctor_lane_safety`:

* ``plugin_stale`` (warn) — the deployed copy differs from the packaged
  source. Absent copy: skipped (``--fix`` owns that, and the legacy
  single-model scaffold never has one).
* ``lobes doctor --refresh-plugin`` — its own named flag, never part of
  ``--fix`` (the #174 append-only lesson). With ``--apply`` it writes exactly
  two paths: ``qwen3_thinking_tool_parser.py.bak`` (the current copy), then the
  packaged copy via :func:`lobes.runtime._compose.write_plugin_file`.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from lobes.runtime import _compose

_BACKUP_SUFFIX = ".bak"
REFRESH_FLAG = "lobes doctor --refresh-plugin --apply"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _packaged() -> bytes:
    return _compose.plugin_source().encode("utf-8")


def _deployed(deploy_dir: Path) -> bytes | None:
    try:
        return (Path(deploy_dir) / _compose.PLUGIN_DEST_NAME).read_bytes()
    except OSError:
        return None


def plugin_stale_check(deploy_dir: Path) -> dict | None:
    deployed = _deployed(deploy_dir)
    if deployed is None:
        return None
    name = _compose.PLUGIN_DEST_NAME
    packaged = _packaged()
    if deployed == packaged:
        return {
            "id": "plugin_stale",
            "passed": True,
            "severity": "info",
            "message": f"{name} matches the package",
            "remediation": "",
        }
    return {
        "id": "plugin_stale",
        "passed": False,
        "severity": "warn",
        "message": f"{name} differs from this CLI's packaged copy "
        f"(sha256 {_sha256(deployed)[:12]} vs {_sha256(packaged)[:12]}) — the lane "
        "may be running an outdated tool-parser plugin",
        "remediation": f"run '{REFRESH_FLAG}' (backs the current file up to "
        f"{name}{_BACKUP_SUFFIX}, writes no other file), then recreate the cortex lane",
    }


def refresh_plugin_plan(deploy_dir: Path) -> list[str]:
    name = _compose.PLUGIN_DEST_NAME
    deployed = _deployed(deploy_dir)
    if deployed is None:
        return [f"{name} is absent — run 'lobes doctor --fix --apply' to write it"]
    packaged = _packaged()
    if deployed == packaged:
        return []
    path = Path(deploy_dir) / name
    return [
        f"{path}: deployed sha256 {_sha256(deployed)}, packaged sha256 {_sha256(packaged)}",
        f"would copy {name} -> {name}{_BACKUP_SUFFIX}",
        f"would write the packaged {name}",
    ]


def apply_refresh_plugin(deploy_dir: Path) -> list[str]:
    """Back up then replace a STALE plugin copy. Writes exactly two paths, or none."""
    deployed = _deployed(deploy_dir)
    if deployed is None or deployed == _packaged():
        return []
    path = Path(deploy_dir) / _compose.PLUGIN_DEST_NAME
    backup = path.with_name(path.name + _BACKUP_SUFFIX)
    shutil.copy2(path, backup)
    _compose.write_plugin_file(deploy_dir, force=True)
    return [f"backed up {path.name} -> {backup.name}", f"wrote packaged {path.name}"]
