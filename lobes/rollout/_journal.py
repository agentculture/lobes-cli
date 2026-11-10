"""Append-only rollout journal, status projection and resume point.

One JSON object per line in ``<deployment dir>/rollout-journal.jsonl``.
A member is done only after a ``verify`` record with result ``ok``, and stops
being done after a later ``rollback`` record with result ``ok`` (so a re-run
of apply upgrades it again rather than reporting it verified).
Secrets are scrubbed by an injected ``redact`` callable (str -> str).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

JOURNAL_NAME = "rollout-journal.jsonl"
STEPS = (
    "lock",
    "window_check",
    "backup_env",
    "install_cli",
    "repin",
    "tag_image",
    "rebuild_gateway",
    "wait_health",
    "verify",
)
DONE_STEP = "verify"
ROLLBACK_STEP = "rollback"
OK = "ok"
FAILED = "failed"
SKIPPED_OFFLINE = "skipped_offline"


@dataclass(frozen=True)
class MemberStatus:
    member: str
    target: Optional[str]
    current: Optional[str]
    last_step: Optional[str]
    last_result: Optional[str]
    done: bool
    state: str  # done | pending | in_progress


@dataclass(frozen=True)
class ResumePoint:
    member: str
    step: str


def _scrub(value: Any, redact: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _scrub(v, redact) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v, redact) for v in value]
    return value


class Journal:
    def __init__(
        self,
        deployment_dir,
        redact: Optional[Callable[[str], str]] = None,
        steps: tuple = STEPS,
    ):
        self.path = Path(deployment_dir).expanduser() / JOURNAL_NAME
        self._redact = redact or (lambda s: s)
        self.steps = tuple(steps)

    def append(
        self,
        member: str,
        step: str,
        result: str,
        target: Optional[str] = None,
        current: Optional[str] = None,
        detail: Any = None,
    ) -> dict:
        rec = {
            "ts": time.time(),
            "member": member,
            "step": step,
            "result": result,
            "target": target,
            "current": current,
            "detail": detail,
        }
        rec = _scrub(rec, self._redact)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def records(self) -> list:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # tolerate a torn trailing line
        return out

    def status(self) -> dict:
        acc: dict = {}
        for r in self.records():
            m = acc.setdefault(
                r["member"],
                {"target": None, "current": None, "step": None, "result": None, "done": False},
            )
            if r.get("target") is not None:
                m["target"] = r["target"]
            if r.get("current") is not None:
                m["current"] = r["current"]
            m["step"], m["result"] = r["step"], r["result"]
            if r["step"] == DONE_STEP and r["result"] == OK:
                m["done"] = True
            elif r["step"] == ROLLBACK_STEP and r["result"] == OK:
                m["done"] = False
        out = {}
        for name, m in acc.items():
            if m["done"]:
                state = "done"
            elif m["result"] == SKIPPED_OFFLINE:
                state = "pending"
            else:
                state = "in_progress"
            out[name] = MemberStatus(
                name, m["target"], m["current"], m["step"], m["result"], m["done"], state
            )
        return out

    def pending_members(self) -> list:
        return [n for n, s in self.status().items() if s.state == "pending"]

    def resume_point(self) -> Optional[ResumePoint]:
        """First interrupted (not done, not offline) member and the step to redo."""
        for name, st in self.status().items():
            if st.state != "in_progress":
                continue
            if st.last_result == OK and st.last_step in self.steps:
                idx = self.steps.index(st.last_step) + 1
                step = self.steps[min(idx, len(self.steps) - 1)]
            else:
                step = st.last_step
            return ResumePoint(name, step)
        return None


class TargetJournal(Journal):
    """The journal restricted to one rollout target, so an earlier rollout's
    verified records never mark a member done for this one."""

    def __init__(self, deployment_dir, target: str, redact: Optional[Callable[[str], str]] = None):
        super().__init__(deployment_dir, redact=redact)
        self.target = target

    def records(self) -> list:
        return [r for r in super().records() if r.get("target") == self.target]


def latest_target(journal: Journal, member: Optional[str] = None) -> Optional[str]:
    """The target of the most recent record (for *member*, when given)."""
    target = None
    for r in journal.records():
        if member is None or r.get("member") == member:
            target = r.get("target") or target
    return target
