"""The shape of per-chip hardware knowledge — a *strategy* per card, not a table row.

Why a strategy object instead of a wider :class:`~lobes.profiles.MachineProfile`
row: per-machine tuning is keyed by a *causal capability trait*, not by a board
name. A :class:`CardStrategy` therefore carries three things the old flat row
could not express honestly:

* a :class:`DetectionSignature` — the *evidence* that identifies the card (name
  markers, and — for future richer detection — its compute capability and total
  memory), kept next to the knobs it selects rather than in a separate table;
* the per-role knobs it needs (a generate-lane ``cortex`` knob is a different
  thing from an ``embedder`` pooling knob), each paired with a **provenance**
  string naming the *cause* — so ``TRITON_ATTN`` on an sm_110 board reads as
  "sm_110: FLASH_ATTN pooling path hangs" and not as an unexplained constant; and
* :class:`Trait` bundles — knowledge shared across boards (an sm_110 quirk holds
  on every sm_110 card) expressed once and *composed* into each strategy, so a
  second sm_110 board reuses the trait instead of copy-pasting its knobs.

This module is dependency-free (stdlib only) and imports nothing from the rest of
:mod:`lobes`, so the registry and the per-chip modules stay decoupled from the
CLI. :mod:`lobes.profiles` derives its legacy ``MachineProfile`` surface *from*
these strategies rather than duplicating them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

# A knob value is a scalar the serving layer substitutes verbatim (a backend
# name, a context length, a boolean flag). Kept as ``object`` so a strategy can
# carry ints, floats, strs and bools without a union per knob.
KnobValue = object


@dataclass(frozen=True)
class Knob:
    """One tuning value plus the causal reason it holds that value.

    ``provenance`` must name the *cause*, not restate the value — it is the
    difference between an auditable default and a magic number. Example:
    ``Knob("TRITON_ATTN", "sm_110: FLASH_ATTN pooling path hangs — force TRITON_ATTN")``.
    """

    value: KnobValue
    provenance: str


@dataclass(frozen=True)
class DetectionSignature:
    """The evidence that identifies a card.

    ``name_markers`` are lowercase substrings matched against the GPU name and
    hostname (the only signal the legacy :func:`~lobes.profiles.detect_machine`
    had, preserved exactly). ``compute_capability`` (e.g. ``"sm_110"``) and
    ``total_memory_gb`` are declared traits carried for a future, richer detector
    (``lobes/runtime/_detect.py``, a later task) — they are informational here and
    do **not** participate in :meth:`matches`, so present-day detection behaviour
    is unchanged.
    """

    name_markers: tuple[str, ...]
    compute_capability: str | None = None
    total_memory_gb: int | None = None

    def matches(self, gpu_name: str | None, hostname: str | None) -> bool:
        """True when any marker is a substring of the GPU name or the hostname.

        An empty ``name_markers`` never matches — that is how the ``generic``
        fallback stays out of auto-detection (it is only ever chosen explicitly).
        """
        for hay in ((gpu_name or "").lower(), (hostname or "").lower()):
            if hay and any(marker in hay for marker in self.name_markers):
                return True
        return False


@dataclass(frozen=True)
class MachineDefaults:
    """The legacy single-model serving knobs (the ``MachineProfile`` surface).

    These feed the ``VLLM_*`` env of the single-model scaffold (``lobes serve``,
    no fleet). Each is a :class:`Knob` so the single-model defaults are as
    auditable as the per-role ones; :mod:`lobes.profiles` reads their ``.value``.
    """

    gpu_mem_util: Knob
    max_model_len: Knob
    attention_backend: Knob


def _freeze_role_knobs(
    role_knobs: Mapping[str, Mapping[str, Knob]],
) -> Mapping[str, Mapping[str, Knob]]:
    """Deep-wrap a role→knob→Knob mapping in read-only proxies."""
    return MappingProxyType(
        {role: MappingProxyType(dict(knobs)) for role, knobs in role_knobs.items()}
    )


@dataclass(frozen=True)
class Trait:
    """A causal capability trait shared across boards, keyed by cause not board.

    A trait bundles the per-role knobs that follow from one hardware fact (an
    sm_110 pooling quirk, say). Boards *compose* traits so the knowledge lives in
    one place — see :mod:`lobes.machines._traits`.
    """

    name: str  # the cause, e.g. "sm_110"
    role_knobs: Mapping[str, Mapping[str, Knob]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_knobs", _freeze_role_knobs(self.role_knobs))


@dataclass(frozen=True)
class CardStrategy:
    """Everything lobes knows about one card: how to spot it, how to tune it, why.

    ``defaults`` back the legacy single-model surface; ``traits`` and
    ``role_overrides`` together give the per-role fleet knobs. A board-specific
    override in ``role_overrides`` wins over the same knob supplied by a trait
    (last-writer-wins in :meth:`role_knobs`), so a card can accept an sm_110 trait
    yet still pin one knob differently with its own provenance.
    """

    name: str  # canonical machine name (== VLLM_MACHINE)
    summary: str  # one-line description
    signature: DetectionSignature
    defaults: MachineDefaults
    status: str  # "load-tested" (measured here) | "configured" (declared estimate)
    traits: tuple[Trait, ...] = ()
    role_overrides: Mapping[str, Mapping[str, Knob]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_overrides", _freeze_role_knobs(self.role_overrides))

    def role_knobs(self) -> dict[str, dict[str, Knob]]:
        """The merged per-role knobs: trait knobs first, board overrides last."""
        merged: dict[str, dict[str, Knob]] = {}
        for trait in self.traits:
            for role, knobs in trait.role_knobs.items():
                merged.setdefault(role, {}).update(knobs)
        for role, knobs in self.role_overrides.items():
            merged.setdefault(role, {}).update(knobs)
        return merged

    def render(self) -> dict[str, object]:
        """A JSON-friendly view of the whole strategy — knobs *with* provenance.

        This is the "knob rendering" surface: it flattens every :class:`Knob` to
        ``{"value": ..., "provenance": ...}`` so a caller (or a test) can show why
        each number is what it is without reaching into the dataclasses.
        """

        def _knob(k: Knob) -> dict[str, object]:
            return {"value": k.value, "provenance": k.provenance}

        return {
            "name": self.name,
            "summary": self.summary,
            "status": self.status,
            "signature": {
                "name_markers": list(self.signature.name_markers),
                "compute_capability": self.signature.compute_capability,
                "total_memory_gb": self.signature.total_memory_gb,
            },
            "defaults": {
                "gpu_mem_util": _knob(self.defaults.gpu_mem_util),
                "max_model_len": _knob(self.defaults.max_model_len),
                "attention_backend": _knob(self.defaults.attention_backend),
            },
            "role_knobs": {
                role: {name: _knob(k) for name, k in knobs.items()}
                for role, knobs in self.role_knobs().items()
            },
        }
