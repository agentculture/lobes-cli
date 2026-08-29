"""Deployment variation identity — machine type/setup, never hostname.

A "deployment variation" (`docs/plans/2026-08-29-deployment-lock-per-box.md`,
`docs/specs/2026-08-29-deployment-lock-per-box.md`) is identified by MACHINE
TYPE OR SETUP, never by hostname: two physically different boxes of the same
type (e.g. two DGX Sparks) are the SAME variation, and for an AI accelerator
the accelerator itself is the identity. This module is a pure mapping from
already-gathered host facts to a stable variation id — it does not gather
facts itself.

It deliberately does not re-probe hardware. :mod:`lobes.runtime._detect`
already gathers host facts (nvidia-smi device name + compute capability,
``/proc/meminfo`` total, the Jetson device-tree model, and hostname — the last
gathered but never fed into resolution) and resolves them to a registered
card name via :mod:`lobes.machines` — the accelerator-signature registry that
already treats hostname as inert. :data:`~lobes.runtime._detect.UNKNOWN` is
that resolution's honest "no match" result: never a guessed nearest card.

The variation id this module hands back is exactly that resolved card name
(or the explicit unknown sentinel) — a thin, named, documented seam so
downstream consumers (the lock writer, ``--from-lock`` restore, the
``deployments/<id>/`` catalog) depend on one small function instead of poking
at :class:`~lobes.runtime._detect.DetectedCard` internals directly.
"""

from __future__ import annotations

from lobes.runtime._detect import UNKNOWN, DetectedCard

# Re-exported so callers of this module never need to import _detect just for
# the sentinel. Kept as the SAME string, not a new one, so a variation id and
# a detected card's ``resolved`` field always compare equal.
UNKNOWN_VARIATION = UNKNOWN


def resolve_variation_id(detected: DetectedCard) -> str:
    """Map a :class:`DetectedCard` to a stable deployment variation id.

    The id is derived from the resolved card name only — the accelerator
    signature :mod:`lobes.machines` already matched on (GPU device name /
    compute capability / device-tree model, constrained by total memory).
    ``detected.hostname`` is never consulted here: two hosts of the same
    machine type always resolve to the same id, whatever their hostnames.

    An unrecognised card (``detected.resolved == UNKNOWN``) returns the
    explicit :data:`UNKNOWN_VARIATION` sentinel — never a guessed "closest"
    variation. Callers that need to render this for a human should treat
    :data:`UNKNOWN_VARIATION` as "no variation identity available", not as a
    variation of its own.
    """
    return detected.resolved
