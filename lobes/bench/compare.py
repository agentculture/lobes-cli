"""Cross-profile RUNTIME comparison — issue #81, task t9.

Compares runtime/serving performance across deployment *profiles* — different
ways the fleet's roles are combined or consumed — and emits COMPARABLE,
side-by-side runtime metrics per profile. All the actual probing is delegated
to :mod:`lobes.roles_measure` (never re-implemented here); rendering is
delegated to :func:`lobes.bench.report.render_side_by_side`.

Four profiles (:data:`PROFILE_NAMES`):

* ``cortex-only``       — the Qwen cortex generate lane alone.
* ``cortex+senses``     — Qwen cortex + Gemma senses (the duo) side by side.
* ``senses-direct``     — Gemma senses addressed directly (cheap/front tasks).
* ``qwen-nvfp4-vs-bf16`` — Qwen NVFP4 vs BF16, ONLY when both a quantized and
  an unquantized (bf16) Qwen ~27B-class checkpoint are catalog-present (see
  :func:`qwen_nvfp4_bf16_variants`). When the catalog doesn't carry both sides
  (today's reality — the catalog has two NVFP4 27B entries and no bf16 27B
  entry), this profile is reported ``available=False`` with a ``reason`` —
  NEVER fabricated, and the network is never touched (mirrors the
  "not loaded -> no probe" contract :mod:`lobes.roles_measure` already uses
  for an unwired role).

RUNTIME-ONLY -- boundary c7/h14
--------------------------------
Every metric this module ever emits comes straight from
:func:`lobes.roles_measure.measure_role` / :func:`measure_registry` — this
module adds no metric fields of its own, only the profile-level envelope
(``profile``/``available``/``reason``/``columns``). **No field anywhere in a
profile result may assert answer correctness, task quality, or agent-task
success** — that judgment is Colleague's job, not lobes' (c7/h14, c6/h13). See
:data:`lobes.roles_measure.ALLOWED_METRIC_KEYS` for the closed vocabulary this
module is bound by transitively.

Read-only
---------
Every probe is delegated to :mod:`lobes.roles_measure` (plain HTTP,
short-timeout, never raises). This module adds no docker/compose calls and no
writes. The ``qwen-nvfp4-vs-bf16`` profile probes the CURRENT ``cortex``
endpoint once per catalog variant, addressing it by that variant's model id —
if the endpoint isn't actually serving that variant, the request degrades the
same way an unreachable/mismatched probe degrades in
:mod:`lobes.roles_measure` (``ready=False``, metrics null): it never spins up
a second deployment and never fabricates a number for a model that isn't
live.
"""

from __future__ import annotations

from collections.abc import Sequence

from lobes.catalog import SUPPORTED_MODELS, SupportedModel
from lobes.roles import RoleInfo
from lobes.roles_measure import DEFAULT_TIMEOUT, measure_registry, measure_role

# The four profiles, in canonical/display order.
PROFILE_NAMES: tuple[str, ...] = (
    "cortex-only",
    "cortex+senses",
    "senses-direct",
    "qwen-nvfp4-vs-bf16",
)

# profile name -> the roles.py role(s) it measures via measure_registry.
# The catalog-gated profile (below) is handled separately — it compares two
# MODEL VARIANTS of the cortex role, not two different roles.
_ROLE_PROFILES: dict[str, tuple[str, ...]] = {
    "cortex-only": ("cortex",),
    "cortex+senses": ("cortex", "senses"),
    "senses-direct": ("senses",),
}

_QWEN_NVFP4_VS_BF16 = "qwen-nvfp4-vs-bf16"


def _is_qwen_27b_class(model: SupportedModel) -> bool:
    """True for a Qwen catalog entry in the ~27B size class (id carries "27b")."""
    lowered = model.id.lower()
    return "qwen" in lowered and "27b" in lowered


def _is_bf16(model: SupportedModel) -> bool:
    """True for the catalog's bf16/unquantized sentinel.

    Mirrors the convention documented on ``Qwen/Qwen3.5-4B`` in
    :mod:`lobes.catalog`: ``quantization="none"`` marks an unquantized bf16
    checkpoint; an empty string covers pooling/score entries that carry no
    quantization at all (never matched here since they also fail the Qwen
    27B-class check above).
    """
    return model.quantization in ("", "none")


def qwen_nvfp4_bf16_variants(
    catalog: Sequence[SupportedModel] = SUPPORTED_MODELS,
) -> tuple[SupportedModel | None, SupportedModel | None]:
    """The (NVFP4, BF16) Qwen ~27B catalog pair; ``None`` on a missing side.

    Never raises. A catalog carrying only one side (today's reality — see the
    module docstring) returns ``(entry, None)``; a catalog with neither
    returns ``(None, None)``. Pure/offline: reads the in-memory catalog list,
    touches no network.
    """
    nvfp4 = next((m for m in catalog if _is_qwen_27b_class(m) and not _is_bf16(m)), None)
    bf16 = next((m for m in catalog if _is_qwen_27b_class(m) and _is_bf16(m)), None)
    return nvfp4, bf16


def _run_role_profile(name: str, registry: dict[str, RoleInfo], *, timeout: float) -> dict:
    """cortex-only / cortex+senses / senses-direct: straight measure_registry fan-out."""
    roles = _ROLE_PROFILES[name]
    columns = measure_registry(registry, roles=roles, timeout=timeout)
    available = all(columns[r]["ready"] for r in roles)
    return {
        "profile": name,
        "available": available,
        "reason": None if available else "one or more roles unreachable",
        "columns": columns,
    }


def _measure_catalog_variant(entry: SupportedModel, endpoint: str, *, timeout: float) -> dict:
    """Probe a catalog model id at ``endpoint`` as a synthetic ``cortex`` role.

    Reuses :func:`lobes.roles_measure.measure_role` unchanged — this is the
    SAME probe the ``cortex`` role gets in :mod:`lobes.roles_measure`, just
    addressed at a specific catalog variant's model id instead of whatever the
    registry's ``cortex`` entry currently resolves to. ``loaded=bool(endpoint)``
    so an empty endpoint (cortex unwired in this deployment) degrades exactly
    like an unwired role does elsewhere — never touches the network.
    """
    info = RoleInfo(
        role="cortex",
        model=entry.id,
        runtime="vllm",
        endpoint=endpoint,
        path="/v1/chat/completions",
        context=entry.native_max_model_len,
        quant=entry.quantization,
        mtp=bool(entry.speculative_config),
        responsibilities=(),
        forbidden_responsibilities=(),
        ready=bool(endpoint),
        loaded=bool(endpoint),
    )
    return measure_role("cortex", info, timeout=timeout)


def _run_qwen_nvfp4_vs_bf16(
    registry: dict[str, RoleInfo],
    *,
    timeout: float,
    catalog: Sequence[SupportedModel] = SUPPORTED_MODELS,
) -> dict:
    """The catalog-gated profile: only probes when BOTH variants are catalog-present."""
    nvfp4, bf16 = qwen_nvfp4_bf16_variants(catalog)
    if nvfp4 is None or bf16 is None:
        missing = [label for label, entry in (("NVFP4", nvfp4), ("BF16", bf16)) if entry is None]
        return {
            "profile": _QWEN_NVFP4_VS_BF16,
            "available": False,
            "reason": (
                "catalog does not carry both a Qwen ~27B NVFP4 and BF16 variant "
                f"(missing: {', '.join(missing)}) — comparison needs both, nothing probed"
            ),
            "columns": {},
        }

    cortex = registry.get("cortex")
    endpoint = cortex.endpoint if cortex else ""
    columns = {
        "nvfp4": _measure_catalog_variant(nvfp4, endpoint, timeout=timeout),
        "bf16": _measure_catalog_variant(bf16, endpoint, timeout=timeout),
    }
    available = all(c["ready"] for c in columns.values())
    return {
        "profile": _QWEN_NVFP4_VS_BF16,
        "available": available,
        "reason": (
            None if available else "one or both variants unreachable at the cortex endpoint"
        ),
        "columns": columns,
    }


def run_profile(
    name: str,
    registry: dict[str, RoleInfo],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    catalog: Sequence[SupportedModel] = SUPPORTED_MODELS,
) -> dict:
    """Run ONE named profile; never raises for a known name (every branch degrades).

    Returns ``{"profile", "available", "reason", "columns"}`` — ``columns`` maps
    a column label (a role name for the three role-based profiles, or
    ``"nvfp4"``/``"bf16"`` for the catalog-gated one) to a
    :func:`lobes.roles_measure.measure_role`-shaped result (``role``/``family``/
    ``model``/``runtime``/``endpoint``/``loaded``/``ready``/``metrics``).

    Raises :class:`ValueError` for an unrecognised ``name`` — a programmer
    error (the CLI layer restricts ``--profile`` to :data:`PROFILE_NAMES` via
    argparse ``choices``, so this should never fire from the CLI).
    """
    if name == _QWEN_NVFP4_VS_BF16:
        return _run_qwen_nvfp4_vs_bf16(registry, timeout=timeout, catalog=catalog)
    if name in _ROLE_PROFILES:
        return _run_role_profile(name, registry, timeout=timeout)
    raise ValueError(f"unknown profile {name!r} — choose one of {PROFILE_NAMES}")


def run_profiles(
    names: Sequence[str] | None,
    registry: dict[str, RoleInfo],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    catalog: Sequence[SupportedModel] = SUPPORTED_MODELS,
) -> dict[str, dict]:
    """Run every profile in ``names`` (default: all four, :data:`PROFILE_NAMES`).

    Never raises — each profile degrades independently (an unreachable
    ``senses`` backend doesn't stop ``cortex-only`` from reporting real
    numbers). Returns an ordered ``dict`` keyed by profile name.
    """
    wanted = names if names is not None else PROFILE_NAMES
    return {n: run_profile(n, registry, timeout=timeout, catalog=catalog) for n in wanted}
