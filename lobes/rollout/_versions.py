"""PEP 440 version ordering and downgrade refusal for the fleet rollout.

A small stdlib-only subset of PEP 440 (``N(.N)*``, optional pre-release
``a|b|rc``, optional ``.postN``, optional ``.devN``). The repo declares no
runtime dependencies, so ``packaging`` is deliberately not used. Local
version labels (``+...``) and epochs are rejected rather than mis-ordered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering

__all__ = ["DowngradeRefused", "Version", "check_target", "is_downgrade", "parse_version"]

_VERSION_RE = re.compile(
    r"""
    ^
    (?P<release>\d+(?:\.\d+)*)
    (?:[-_.]?(?P<pre_l>a|alpha|b|beta|rc|c|pre|preview)[-_.]?(?P<pre_n>\d+)?)?
    (?:[-_.]?(?P<post_kw>post|rev|r)[-_.]?(?P<post_n>\d+)?)?
    (?:[-_.]?(?P<dev_kw>dev)[-_.]?(?P<dev_n>\d+)?)?
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Pre-release spellings normalised to one rank, in PEP 440 order.
_PRE_RANK = {
    "a": 0,
    "alpha": 0,
    "b": 1,
    "beta": 1,
    "rc": 2,
    "c": 2,
    "pre": 2,
    "preview": 2,
}


class DowngradeRefused(ValueError):
    """Raised when a rollout target is older than the running version."""


@total_ordering
@dataclass(frozen=True)
class Version:
    """A parsed PEP 440 version (subset) with a total order."""

    text: str
    release: tuple[int, ...]
    pre: tuple[int, int] | None
    post: int | None
    dev: int | None

    def _key(self) -> tuple:
        release = list(self.release)
        while len(release) > 1 and release[-1] == 0:
            release.pop()
        # dev-only (no pre, no post) sorts before any pre-release of the base.
        if self.pre is None and self.post is None and self.dev is not None:
            pre_key = (0, 0, 0)
        elif self.pre is None:
            pre_key = (2, 0, 0)
        else:
            pre_key = (1, self.pre[0], self.pre[1])
        post_key = (0, 0) if self.post is None else (1, self.post)
        # A dev release sorts before the final release of the same base.
        dev_key = (2, 0) if self.dev is None else (1, self.dev)
        return (tuple(release), pre_key, post_key, dev_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() == other._key()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._key() < other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __str__(self) -> str:
        return self.text


def parse_version(text: str) -> Version:
    """Parse ``text`` into a :class:`Version`; raise ``ValueError`` if malformed."""
    if not isinstance(text, str):
        raise ValueError(f"version must be a string, got {type(text).__name__}")
    stripped = text.strip()
    match = _VERSION_RE.match(stripped)
    if match is None:
        raise ValueError(f"not a supported PEP 440 version: {text!r}")
    release = tuple(int(part) for part in match.group("release").split("."))
    pre = None
    if match.group("pre_l"):
        pre = (_PRE_RANK[match.group("pre_l").lower()], int(match.group("pre_n") or 0))
    post = int(match.group("post_n") or 0) if match.group("post_kw") else None
    dev = int(match.group("dev_n") or 0) if match.group("dev_kw") else None
    return Version(text=stripped, release=release, pre=pre, post=post, dev=dev)


def is_downgrade(target: str, running: str) -> bool:
    """True when ``target`` is strictly older than ``running``."""
    return parse_version(target) < parse_version(running)


def check_target(target: str, running: str, allow_downgrade: bool = False) -> None:
    """Refuse a rollout to an older version unless ``allow_downgrade`` is set."""
    if not allow_downgrade and is_downgrade(target, running):
        raise DowngradeRefused(
            f"refusing to downgrade from {running} to {target}; "
            "pass allow_downgrade=True to proceed deliberately"
        )
