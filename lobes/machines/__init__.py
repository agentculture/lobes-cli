"""Per-chip hardware strategies — one module per card, one explicit registry.

Each supported card is a :class:`CardStrategy` living in its own module
(``spark``, ``thor``, ``blackwell``, ``generic``); importing that module registers
it. This package imports the four built-ins **in detection-precedence order**
(``spark`` before ``blackwell`` so the GB10 is never taken for a discrete
Blackwell), then re-exports the registry API and the strategy vocabulary.

Adding a chip is one new module (with its own ``register(...)`` line) plus one
import line here — nothing else in this file, and nothing in
:mod:`lobes.profiles`, has to change: profile resolution, detection and knob
rendering all read the live registry. Third parties and tests can skip even the
import line and call :func:`register` directly.

No entry-point / plugin machinery — the registry is a plain in-process dict and
the fleet is assembled by explicit imports (stdlib only).
"""

from __future__ import annotations

from ._registry import detect, get, names, register, strategies, unregister
from ._strategy import CardStrategy, DetectionSignature, Knob, MachineDefaults, Trait
from ._traits import SM_110

# Registration order is detection precedence (spark before blackwell, so the GB10
# is never taken for the discrete Blackwell). Importing each chip module runs its
# module-level register() call. Pinned order — kept off isort so it is never
# alphabetised (which would swap spark and blackwell).
from . import spark, thor, blackwell, generic  # noqa: F401  # isort: skip

__all__ = [
    # registry API
    "register",
    "unregister",
    "get",
    "names",
    "strategies",
    "detect",
    # vocabulary
    "CardStrategy",
    "DetectionSignature",
    "MachineDefaults",
    "Knob",
    "Trait",
    "SM_110",
]
