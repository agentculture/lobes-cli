"""Compose drift and deployment-lock detection for rollout (pure functions).

The gateway service takes an explicit ``environment:`` list, never env_file,
so a key the target template passes but a member's compose chain does not is
silently dropped (2026-11-09: spark2 lacked STT_EN_BASE_URL). Text in, text
out; the transport fetches/writes files. PyYAML is not a dependency, so the
environment list is parsed with conservative line-based logic.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

OVERRIDE_NAME = "docker-compose.override.yml"
LOCK_NAME = "deployment.lock.toml"

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_skip(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("#")


def _find_env_block(lines: list[str]):
    """Return (env_line_idx, env_indent, last_entry_idx) or None."""
    svc = None
    for i, ln in enumerate(lines):
        if re.match(r"^services:\s*(#.*)?$", ln):
            svc = i
            break
    if svc is None:
        return None
    gw = None
    for i in range(svc + 1, len(lines)):
        ln = lines[i]
        if _is_skip(ln):
            continue
        if _indent(ln) == 0:
            break
        if re.match(r"^\s+gateway:\s*(#.*)?$", ln):
            gw = i
            break
    if gw is None:
        return None
    gi = _indent(lines[gw])
    env = None
    for i in range(gw + 1, len(lines)):
        ln = lines[i]
        if _is_skip(ln):
            continue
        if _indent(ln) <= gi:
            return None
        if re.match(r"^\s+environment:\s*(#.*)?$", ln):
            env = i
            break
    if env is None:
        return None
    ei = _indent(lines[env])
    last = env
    for i in range(env + 1, len(lines)):
        ln = lines[i]
        if _is_skip(ln):
            continue
        if _indent(ln) <= ei and not ln.lstrip().startswith("- "):
            break
        if _indent(ln) < ei:
            break
        last = i
    return env, ei, last


def _entry_key(line: str) -> str | None:
    s = line.strip()
    if s.startswith("- "):
        s = s[2:].strip().strip("\"'")
        key = re.split(r"[=:\s]", s, maxsplit=1)[0]
    else:
        key = s.split(":", 1)[0].strip()
    return key if _KEY.match(key) else None


def gateway_env_keys(text: str) -> list[str]:
    """Keys under services.gateway.environment, in order, comments ignored."""
    lines = text.splitlines()
    blk = _find_env_block(lines)
    if blk is None:
        return []
    env, _ei, last = blk
    keys: list[str] = []
    for ln in lines[env + 1 : last + 1]:
        if _is_skip(ln):
            continue
        k = _entry_key(ln)
        if k and k not in keys:
            keys.append(k)
    return keys


@dataclass(frozen=True)
class DriftReport:
    missing: list[str] = field(default_factory=list)
    lock_affected: bool = False

    @property
    def drifted(self) -> bool:
        return bool(self.missing)


def detect_drift(template_text: str, member_files: dict[str, str]) -> DriftReport:
    """Compare the target template's gateway env keys to a member's compose chain.

    ``member_files`` maps file name -> text. A key is present if ANY
    ``docker-compose*.yml`` file passes it to the gateway.
    """
    have: set[str] = set()
    for name, text in member_files.items():
        if name.startswith("docker-compose") and name.endswith((".yml", ".yaml")):
            have.update(gateway_env_keys(text))
    missing = [k for k in gateway_env_keys(template_text) if k not in have]
    return DriftReport(missing=missing, lock_affected=LOCK_NAME in member_files)


@dataclass(frozen=True)
class OverridePatch:
    added: list[str]
    new_text: str | None  # None when nothing to add
    backup_name: str | None  # None when no override existed to back up
    diff: str


def _line(key: str, indent: int, mapping: bool = False) -> str:
    if mapping:
        return f"{' ' * indent}{key}: ${{{key}:-}}"
    return f"{' ' * indent}- {key}=${{{key}:-}}"


class OverridePatchError(ValueError):
    """The override cannot be extended safely by line edits."""


def patch_override(template_text: str, member_files: dict[str, str], *, now: str) -> OverridePatch:
    """Build the new docker-compose.override.yml adding only missing gateway keys.

    Pure: returns the text, backup name (``<override>.bak.<now>``) and unified
    diff for the caller to write. Never involves docker-compose.yml or shapes.
    New entries follow the block's own form (list ``- KEY=...`` or mapping
    ``KEY: ...``) and the file's indentation. Raises :class:`OverridePatchError`
    for a gateway ``environment:`` written inline (``[...]``/``{...}``), which
    line edits cannot extend without producing a duplicate key.
    """
    report = detect_drift(template_text, member_files)
    if not report.missing:
        return OverridePatch([], None, None, "")
    old = member_files.get(OVERRIDE_NAME)
    if old is None:
        new = "services:\n  gateway:\n    environment:\n" + "".join(
            _line(k, 6) + "\n" for k in report.missing
        )
        backup = None
        old_text = ""
    else:
        old_text = old
        lines = old.splitlines()
        blk = _find_env_block(lines)
        if blk is not None:
            env, ei, last = blk
            item_indent, mapping = ei + 2, False
            for i in range(env + 1, last + 1):
                if _is_skip(lines[i]):
                    continue
                item_indent = _indent(lines[i])
                mapping = not lines[i].lstrip().startswith("- ")
                break
            ins = [_line(k, item_indent, mapping) for k in report.missing]
            lines[last + 1 : last + 1] = ins
        else:
            gw = next(
                (i for i, ln in enumerate(lines) if re.match(r"^\s+gateway:\s*(#.*)?$", ln)),
                None,
            )
            if gw is not None:
                gi = _indent(lines[gw])
                step = gi  # the file's own indent unit (services -> gateway)
                for i in range(gw + 1, len(lines)):
                    if _is_skip(lines[i]):
                        continue
                    if _indent(lines[i]) <= gi:
                        break
                    if re.match(r"^\s+environment:", lines[i]):
                        raise OverridePatchError(
                            "gateway environment in docker-compose.override.yml is written "
                            "inline; add the missing keys by hand: " + ", ".join(report.missing)
                        )
                block = [" " * (gi + step) + "environment:"] + [
                    _line(k, gi + 2 * step) for k in report.missing
                ]
                lines[gw + 1 : gw + 1] = block
            elif any(re.match(r"^services:\s*(#.*)?$", ln) for ln in lines):
                s = next(i for i, ln in enumerate(lines) if re.match(r"^services:", ln))
                block = ["    environment:"] + [_line(k, 6) for k in report.missing]
                lines[s + 1 : s + 1] = ["  gateway:"] + block
            else:
                block = ["    environment:"] + [_line(k, 6) for k in report.missing]
                lines += ["services:", "  gateway:"] + block
        new = "\n".join(lines) + "\n"
        backup = f"{OVERRIDE_NAME}.bak.{now}"
    diff = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=OVERRIDE_NAME,
            tofile=OVERRIDE_NAME,
        )
    )
    return OverridePatch(list(report.missing), new, backup, diff)
