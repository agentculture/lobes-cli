"""Explain catalog — markdown keyed by topic-path tuples.

See :mod:`lobes.explain.catalog` for the string bodies and :func:`resolve`
for lookup.
"""

from __future__ import annotations

from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.explain.catalog import ENTRIES


def resolve(path: tuple[str, ...]) -> str:
    """Return the markdown body for ``path`` or raise :class:`ModelGearError`."""
    if path in ENTRIES:
        return ENTRIES[path]
    display = " ".join(path) if path else "<root>"
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"no explain entry for: {display}",
        remediation="list known topics with: lobes explain lobes",
    )


def known_paths() -> list[tuple[str, ...]]:
    """Return every catalog path (used by tests)."""
    return list(ENTRIES.keys())
