"""Workload (`purpose`) and machine tuning profiles — the profiling axes of a "gear".

A pure, near-dependency-free package (it only imports the CLI error type for
friendly "unknown profile" messages, plus stdlib). Where :mod:`lobes.catalog`
answers *which model* to serve, this answers *how* to serve it — across THREE
axes:

* a **workload profile** (the ``purpose``: ``balanced`` / ``prompt-heavy`` /
  ``decode-heavy``) sets the throughput/batching knobs and the shape
  (``input_len``/``output_len``) ``lobes benchmark`` exercises;
* a **legacy machine profile** (``spark`` / ``thor`` / ``blackwell`` /
  ``generic``) sets the single-model memory/arch defaults
  (``gpu_memory_utilization``, ``max_model_len``, ``attention_backend``); and
* a **fleet :class:`~lobes.profiles.schema.Profile`** (this package's newer
  surface — :mod:`lobes.profiles.schema` + :mod:`lobes.profiles.loader`) is
  the per-ROLE declaration a machine-aware ``lobes init`` resolves: whether
  ``cortex``/``senses``/``embedder``/``reranker`` is even feasible on the
  target box, which model serves it, and every compose-template knob
  (``gpu_mem_util``, ``max_model_len``, ``quantization``, ``kv_cache_dtype``,
  ``attention_backend``, ``enforce_eager``, ``max_num_seqs``) — see
  :func:`resolve_profile`; and
* a **:class:`~lobes.profiles.shapes.Shape`** (:mod:`lobes.profiles.shapes`,
  brain-shapes issue #113) is the DEPLOYMENT-SHAPE axis, orthogonal to the
  above: which role subset a BOX hosts at all — ``machine-as-brain`` (every
  role the card can serve, the default) or a mesh-brain lobe (``spark-lobe``
  / ``thor-lobe``, each keeping some Colleague roles and leaving the rest to
  a peer box) — composed with a :class:`Profile` at render time (a later
  task), never re-implementing it — see :func:`resolve_shape`.

The legacy machine layer is no longer a table in this module: it is *derived*
from the per-chip strategy registry in :mod:`lobes.machines` (one
:class:`~lobes.machines.CardStrategy` per card, owning its own detection
signature + per-role knobs + provenance). ``MachineProfile`` /
``MACHINE_PROFILES`` / :func:`detect_machine` / :func:`resolve_serve_config`
remain the stable surface legacy callers use, rebuilt from the registry on
each access — so registering a new chip (``lobes.machines``) lights it up
here with no edit to this file. The newer per-role :class:`Profile` schema
reads the SAME registry for its machine-validated divergences (e.g. Thor's
sm_110 knobs) rather than duplicating them — see
:mod:`lobes.profiles.loader`.

The workload knobs and machine defaults are *configured heuristics* (a sensible
starting point), not load-tested numbers — confirm them with ``lobes benchmark``.
The one set of values taken straight from a measured source is the per-purpose
``(input_len, output_len)`` benchmark shapes and the ``balanced`` batching knobs,
which mirror shahizat's cross-machine report (see ``docs/tuning-profiles.md``).

Resolution layers (highest precedence last), assembled by
:func:`resolve_serve_config`: machine profile → workload profile → explicit CLI
overrides. The *model* layer (quantization, tool parser, MoE/MTP extras) stays in
:mod:`lobes.catalog`, applied by ``lobes switch`` alongside this.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from lobes import machines
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.profiles.loader import (
    available_profiles,
    builtin_names,
    discover_operator_profiles,
    load_builtin,
    resolve_profile,
)
from lobes.profiles.schema import KNOB_NAMES, ROLES, Profile, RoleProfile
from lobes.profiles.shapes import (
    AUDIO_ROLES,
    SHAPE_ROLES,
    Shape,
    builtin_shape_names,
    load_builtin_shape,
    resolve_shape,
)

__all__ = [
    # workload profiles
    "DEFAULT_PURPOSE",
    "DEFAULT_MACHINE",
    "WorkloadProfile",
    "WORKLOAD_PROFILES",
    "workload_profiles",
    "workload_profile",
    "workloads_as_dicts",
    # legacy machine profiles (derived from lobes.machines)
    "MachineProfile",
    "MACHINE_PROFILES",
    "machine_profiles",
    "machine_profile",
    "detect_machine",
    "resolve_machine",
    "resolve_serve_config",
    "machines_as_dicts",
    # fleet per-role Profile schema + loader (lobes.profiles.schema/.loader)
    "ROLES",
    "KNOB_NAMES",
    "Profile",
    "RoleProfile",
    "builtin_names",
    "load_builtin",
    "discover_operator_profiles",
    "available_profiles",
    "resolve_profile",
    # deployment-shape schema + built-in loader (lobes.profiles.shapes)
    "AUDIO_ROLES",
    "SHAPE_ROLES",
    "Shape",
    "builtin_shape_names",
    "load_builtin_shape",
    "resolve_shape",
]

DEFAULT_PURPOSE = "balanced"
# CLI default; resolved to a concrete machine by detect_machine() when "auto".
DEFAULT_MACHINE = "auto"


@dataclass(frozen=True)
class WorkloadProfile:
    """One serving *purpose* — how the request mix is shaped, and how to tune for it."""

    name: str  # canonical name (== VLLM_PURPOSE)
    aliases: tuple[str, ...]  # accepted spellings (incl. the canonical name)
    summary: str  # one-line description
    max_num_seqs: int  # vLLM --max-num-seqs (concurrent decode slots)
    max_num_batched_tokens: int  # vLLM --max-num-batched-tokens (prefill chunk budget)
    bench_input_len: int  # lobes benchmark prompt size for this purpose
    bench_output_len: int  # lobes benchmark decode length for this purpose


# Ordered; the first entry is the default (balanced). The balanced batching knobs
# (4 / 8192) and all three (input,output) shapes mirror shahizat's report.
WORKLOAD_PROFILES: tuple[WorkloadProfile, ...] = (
    WorkloadProfile(
        name="balanced",
        aliases=("balanced", "balance", "bal"),
        summary="even prompt/decode mix (≈1K in / 1K out) — the default gear",
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        bench_input_len=1000,
        bench_output_len=1000,
    ),
    WorkloadProfile(
        name="prompt-heavy",
        aliases=("prompt-heavy", "prompt_heavy", "prompt", "prefill"),
        summary="long inputs, short outputs (≈8K in / 1K out) — favor prefill",
        max_num_seqs=4,
        max_num_batched_tokens=16384,  # bigger prefill chunks for long prompts
        bench_input_len=8000,
        bench_output_len=1000,
    ),
    WorkloadProfile(
        name="decode-heavy",
        aliases=("decode-heavy", "decode_heavy", "decode", "gen"),
        summary="short inputs, long outputs (≈1K in / 8K out) — favor decode",
        max_num_seqs=8,  # more concurrent decode slots
        max_num_batched_tokens=4096,  # smaller prefill chunks, more decode headroom
        bench_input_len=1000,
        bench_output_len=8000,
    ),
)


@dataclass(frozen=True)
class MachineProfile:
    """One hardware target — the memory/arch defaults lobes tunes to.

    The stable legacy view of a :class:`lobes.machines.CardStrategy`: the strategy
    owns the detection signature, per-role knobs and provenance; this flat row is
    what single-model callers (``switch``/``benchmark``) still read. Built from a
    strategy by :func:`_machine_profile_from_strategy` — never hand-authored.
    """

    name: str  # canonical name (== VLLM_MACHINE)
    summary: str  # one-line description
    gpu_markers: tuple[str, ...]  # lowercase substrings matched against nvidia-smi / hostname
    gpu_mem_util: float  # vLLM --gpu-memory-utilization default
    max_model_len: int  # vLLM --max-model-len default
    attention_backend: str  # vLLM --attention-backend default
    status: str  # "load-tested" (measured here) | "configured" (declared estimate)


_GENERIC = "generic"


def _machine_profile_from_strategy(strategy: machines.CardStrategy) -> MachineProfile:
    """Project a per-chip strategy onto the legacy flat ``MachineProfile`` row."""
    d = strategy.defaults
    return MachineProfile(
        name=strategy.name,
        summary=strategy.summary,
        gpu_markers=strategy.signature.name_markers,
        gpu_mem_util=d.gpu_mem_util.value,
        max_model_len=d.max_model_len.value,
        attention_backend=d.attention_backend.value,
        status=strategy.status,
    )


def __getattr__(name: str) -> object:
    """Serve ``MACHINE_PROFILES`` from the live registry (PEP 562).

    Kept as a module attribute (not a static tuple) so a chip registered after
    import — a plugin or a test — is reflected without editing this module.
    """
    if name == "MACHINE_PROFILES":
        return machine_profiles()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def workload_profiles() -> tuple[WorkloadProfile, ...]:
    """The full set of workload (purpose) profiles."""
    return WORKLOAD_PROFILES


def machine_profiles() -> tuple[MachineProfile, ...]:
    """The full set of machine profiles, derived live from the chip registry."""
    return tuple(_machine_profile_from_strategy(s) for s in machines.strategies())


def workload_profile(name: str) -> WorkloadProfile:
    """Resolve a purpose by canonical name or alias; raise on an unknown value."""
    key = (name or "").strip().lower()
    for profile in WORKLOAD_PROFILES:
        if key in profile.aliases:
            return profile
    known = ", ".join(p.name for p in WORKLOAD_PROFILES)
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"unknown purpose {name!r}",
        remediation=f"choose one of: {known}",
    )


def machine_profile(name: str) -> MachineProfile:
    """Resolve a machine by canonical name; raise on an unknown value."""
    strategy = machines.get(name)
    if strategy is not None:
        return _machine_profile_from_strategy(strategy)
    known = ", ".join(machines.names())
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"unknown machine {name!r}",
        remediation=f"choose one of: {known}, or 'auto' to detect",
    )


def detect_machine(gpu_name: str | None = None, hostname: str | None = None) -> str:
    """Best-effort machine name from the GPU string + hostname; ``generic`` if unsure.

    Delegates to the chip registry's honest resolver
    (:func:`lobes.machines.detect`, which returns ``None`` when nothing matches)
    and keeps the legacy silent ``generic`` fallback for its callers. Registry
    order is detection precedence, so the GB10 (``spark``) is still matched before
    the discrete Blackwell. Never raises.
    """
    strategy = machines.detect(gpu_name, hostname)
    return strategy.name if strategy is not None else _GENERIC


def resolve_machine(name: str, *, gpu_name: str | None = None, hostname: str | None = None) -> str:
    """Resolve ``--machine`` to a concrete profile name (``auto`` → detect)."""
    if (name or "").strip().lower() in ("", "auto"):
        return detect_machine(gpu_name, hostname)
    return machine_profile(name).name


def resolve_serve_config(
    purpose: str,
    machine: str,
    *,
    max_model_len: int | None = None,
    gpu_mem_util: float | None = None,
) -> dict[str, str]:
    """Assemble the ``VLLM_*`` env plan for a purpose + machine (overrides win).

    ``machine`` must already be a concrete profile name (call :func:`resolve_machine`
    on a possibly-``auto`` value first). ``max_model_len`` / ``gpu_mem_util`` are
    the explicit ``lobes switch`` overrides — ``None`` means "use the machine
    default". The model layer (quantization, tool parser, MoE extras) is applied
    separately from :mod:`lobes.catalog`.
    """
    wl = workload_profile(purpose)
    mp = machine_profile(machine)
    return {
        "VLLM_PURPOSE": wl.name,
        "VLLM_MACHINE": mp.name,
        "VLLM_MAX_MODEL_LEN": str(mp.max_model_len if max_model_len is None else max_model_len),
        "VLLM_GPU_MEM_UTIL": str(mp.gpu_mem_util if gpu_mem_util is None else gpu_mem_util),
        "VLLM_ATTENTION_BACKEND": mp.attention_backend,
        "VLLM_MAX_NUM_SEQS": str(wl.max_num_seqs),
        "VLLM_MAX_NUM_BATCHED_TOKENS": str(wl.max_num_batched_tokens),
    }


def workloads_as_dicts() -> list[dict[str, object]]:
    """The workload profiles as plain dicts (for JSON / overview without the dataclass)."""
    return [asdict(p) for p in WORKLOAD_PROFILES]


def machines_as_dicts() -> list[dict[str, object]]:
    """The machine profiles as plain dicts (for JSON / overview without the dataclass)."""
    return [asdict(p) for p in machine_profiles()]
