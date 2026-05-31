"""Workload (`purpose`) and machine tuning profiles — the second axis of a "gear".

A pure, near-dependency-free data module (it only imports the CLI error type for
friendly "unknown profile" messages). Where :mod:`model_gear.catalog` answers
*which model* to serve, this answers *how* to serve it:

* a **workload profile** (the ``purpose``: ``balanced`` / ``prompt-heavy`` /
  ``decode-heavy``) sets the throughput/batching knobs and the shape
  (``input_len``/``output_len``) ``model benchmark`` exercises, and
* a **machine profile** (``spark`` / ``thor`` / ``blackwell`` / ``generic``) sets
  the memory/arch defaults (``gpu_memory_utilization``, ``max_model_len``,
  ``attention_backend``).

The workload knobs and machine defaults are *configured heuristics* (a sensible
starting point), not load-tested numbers — confirm them with ``model benchmark``.
The one set of values taken straight from a measured source is the per-purpose
``(input_len, output_len)`` benchmark shapes and the ``balanced`` batching knobs,
which mirror shahizat's cross-machine report (see ``docs/tuning-profiles.md``).

Resolution layers (highest precedence last), assembled by
:func:`resolve_serve_config`: machine profile → workload profile → explicit CLI
overrides. The *model* layer (quantization, tool parser, MoE/MTP extras) stays in
:mod:`model_gear.catalog`, applied by ``model switch`` alongside this.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from model_gear.cli._errors import EXIT_USER_ERROR, ModelGearError

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
    bench_input_len: int  # model benchmark prompt size for this purpose
    bench_output_len: int  # model benchmark decode length for this purpose


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
    """One hardware target — the memory/arch defaults model-gear tunes to."""

    name: str  # canonical name (== VLLM_MACHINE)
    summary: str  # one-line description
    gpu_markers: tuple[str, ...]  # lowercase substrings matched against nvidia-smi / hostname
    gpu_mem_util: float  # vLLM --gpu-memory-utilization default
    max_model_len: int  # vLLM --max-model-len default
    attention_backend: str  # vLLM --attention-backend default
    status: str  # "load-tested" (measured here) | "configured" (declared estimate)


# Ordered for detect_machine(): the GB10 (spark) is itself a Grace *Blackwell*
# part, so spark must be matched before blackwell and via the specific "gb10"
# marker (never a bare "blackwell", which would also match the GB10's GPU string).
MACHINE_PROFILES: tuple[MachineProfile, ...] = (
    MachineProfile(
        name="spark",
        summary="DGX Spark (GB10 Grace Blackwell, 128 GB unified, usually shared)",
        gpu_markers=("gb10", "dgx spark", "spark"),
        gpu_mem_util=0.6,  # conservative: the GB10 is shared with other mesh agents
        max_model_len=32768,  # 256K-native primary capped for the first load
        attention_backend="flashinfer",
        status="load-tested",
    ),
    MachineProfile(
        name="thor",
        summary="Jetson Thor (Blackwell, unified memory) — not yet measured here",
        gpu_markers=("thor",),
        gpu_mem_util=0.6,
        max_model_len=32768,
        attention_backend="flashinfer",
        status="configured",
    ),
    MachineProfile(
        name="blackwell",
        summary="Blackwell 6000 Pro (RTX PRO 6000, dedicated VRAM)",
        # Specific to the RTX PRO 6000 Blackwell — NOT a bare "rtx 6000" (which
        # would false-match the older RTX 6000 Ada / Quadro RTX 6000, non-Blackwell).
        # "rtx pro 6000" = the nvidia-smi name; "6000 pro" = shahizat's marketing name.
        gpu_markers=("rtx pro 6000", "6000 pro"),
        gpu_mem_util=0.85,  # dedicated discrete GPU — can reserve aggressively
        max_model_len=65536,  # dedicated VRAM allows the report's longer context
        attention_backend="flashinfer",
        status="configured",
    ),
    MachineProfile(
        name="generic",
        summary="unknown Blackwell-class box — conservative fallback",
        gpu_markers=(),  # never auto-matched; the explicit/last-resort default
        gpu_mem_util=0.6,
        max_model_len=32768,
        attention_backend="flashinfer",
        status="configured",
    ),
)

_GENERIC = "generic"


def workload_profiles() -> tuple[WorkloadProfile, ...]:
    """The full set of workload (purpose) profiles."""
    return WORKLOAD_PROFILES


def machine_profiles() -> tuple[MachineProfile, ...]:
    """The full set of machine profiles."""
    return MACHINE_PROFILES


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
    key = (name or "").strip().lower()
    for profile in MACHINE_PROFILES:
        if key == profile.name:
            return profile
    known = ", ".join(p.name for p in MACHINE_PROFILES)
    raise ModelGearError(
        code=EXIT_USER_ERROR,
        message=f"unknown machine {name!r}",
        remediation=f"choose one of: {known}, or 'auto' to detect",
    )


def detect_machine(gpu_name: str | None = None, hostname: str | None = None) -> str:
    """Best-effort machine name from the GPU string + hostname; ``generic`` if unsure.

    Matches each profile's ``gpu_markers`` (lowercase substrings) against the GPU
    name first, then the hostname, in table order — so the GB10 (``spark``) is
    matched before the discrete Blackwell. Never raises.
    """
    haystacks = [(gpu_name or "").lower(), (hostname or "").lower()]
    for profile in MACHINE_PROFILES:
        if not profile.gpu_markers:
            continue
        for hay in haystacks:
            if hay and any(marker in hay for marker in profile.gpu_markers):
                return profile.name
    return _GENERIC


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
    the explicit ``model switch`` overrides — ``None`` means "use the machine
    default". The model layer (quantization, tool parser, MoE extras) is applied
    separately from :mod:`model_gear.catalog`.
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
    return [asdict(p) for p in MACHINE_PROFILES]
