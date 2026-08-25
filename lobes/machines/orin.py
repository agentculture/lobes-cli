"""Orin — Jetson AGX Orin 64GB Developer Kit (Ampere sm_87, unified memory).

Live-validated (docs/orin-profiles.md, 2026-07-16/17): compute capability 8.7
(Ampere, `sm_87`), 61.3 GiB unified memory measured, hostname `orin`, CUDA
13.2 / driver 595.78. Registered with a bare ``"orin"`` marker — the box's
real hostname is literally that, and its device-tree model string
(``/proc/device-tree/model``, the Jetson-only fallback signal) also carries
it (``"NVIDIA Jetson AGX Orin Developer Kit"``).

**Ampere cannot serve the NVFP4 `cortex` primary, nor the NVFP4 `muse`
export.** The 27B checkpoint (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` / the
current primary `unsloth/Qwen3.8-27B-NVFP4`, and its predecessor
`unsloth/Qwen3.6-27B-NVFP4`), and muse's `nvidia/Gemma-4-31B-IT-NVFP4`,
quantize *activations* to FP4 (W4A4 on the MLP path), which needs
Blackwell-class tensor cores — sm_87 is Ampere, one generation short. That is
a hard architecture line, not a memory tradeoff, and it is a per-role
*feasibility* fact (declared in a fleet ``Profile`` TOML — out of scope here;
see the `orin.toml` operator profile in docs/orin-profiles.md and the
built-in `lobes/profiles/builtin/orin.toml`). This module only
owns card *detection* plus the knob divergences the live boot measured.

That statement is PER-CHECKPOINT, and it stands unweakened for those exports.
What it is NOT is a property of "NVFP4" as a word — which is what the next
paragraph carves out.

**The Lightning carve-out (2026-08-25).** ``worker``'s
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` used to be swept into the
W4A4 sentence above. That was false for it: the checkpoint's OWN
``hf_quant_config.json`` (fetched 2026-08-25) declares ``quant_algo`` =
``"MIXED_PRECISION"``, with every routed *and* shared expert
``up_proj``/``down_proj`` at ``"W4A16_NVFP4"`` (``group_size`` 16) — WEIGHT-only
FP4 with 16-bit activations — the mixer ``in_proj``/``out_proj`` at ``"FP8"``,
and ``kv_cache_quant_algo`` ``"FP8"``. W4A16 is structurally the same exception
this board already runs for the int4 W4A16 ``senses`` gear: weights dequantize
through Marlin into 16-bit arithmetic, so there is no FP4 activation path to be
a generation short of. CONFIRMED LIVE on this physical board
(docs/evidence/2026-08-25-spike-lightning-vllm-orin.txt): vLLM v0.27.1 took
``quantization=modelopt_mixed`` on sm_87, selected a full Marlin fallback stack
(``MarlinFP8ScaledMMLinearKernel``, ``MarlinNvFp4LinearKernel``, the ``MARLIN``
NvFp4 MoE backend) plus native FlashAttention 2, and served correct output at
~78-81 tok/s. The FP8 halves ARE a real sm_87 barrier — the FP8 KV cache needed
``--kv-cache-dtype bfloat16`` to get past it. This is a correction to the
stated REASON only; no feasibility declaration moves with it, and this module
owns none of them anyway.

Note the line is about the CHECKPOINT FORMAT, not the ROLE: since 2026-08-23
``builtin/orin.toml`` declares ``cortex`` feasible on a **GGUF** gear served by
llama.cpp (``unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M``), which is weight-only and
decodes on Ampere — the same reason the int4 W4A16 ``senses`` gear runs here.
That declaration carries no knob this module owns, so nothing below changes.

The pooling-lane divergences below (`embedder`/`reranker` forced onto
`TRITON_ATTN`, `reranker` additionally `enforce_eager`) mirror Thor's sm_110
findings as a **conservative first-boot choice** — docs/orin-profiles.md is
explicit that Thor's FLASH_ATTN/FlashInfer pooling hang is an sm_110-specific
finding that was **not independently re-tested on sm_87**, so these are
"safe boot" values, not proven-necessary ones (relaxing and measuring them is
open follow-up work). Because the cause is different (an untested carry-over,
not a measured sm_87 fact), orin does **not** compose the shared ``SM_110``
trait — that would misattribute the provenance to the wrong compute
capability. They are declared as orin's own ``role_overrides`` instead, each
provenance string saying so explicitly and citing the doc.
"""

from __future__ import annotations

from ._registry import register
from ._strategy import CardStrategy, DetectionSignature, Knob, MachineDefaults

_CARRIED_OVER_PROVENANCE = (
    "orin sm_87: conservative first-boot choice mirroring Thor's sm_110 pooling "
    "divergence — NOT independently proven necessary on sm_87, only carried over "
    "as a safe default (docs/orin-profiles.md, live-validated 2026-07-16/17); "
    "relaxing and measuring it is open follow-up work"
)

STRATEGY = register(
    CardStrategy(
        name="orin",
        summary="Jetson AGX Orin 64GB Developer Kit (Ampere sm_87, unified memory)",
        signature=DetectionSignature(
            name_markers=("orin",),
            compute_capability="sm_87",
            total_memory_gb=64,
        ),
        defaults=MachineDefaults(
            # Legacy single-model surface (`lobes serve`, no fleet). 0.45 is the
            # MEASURED live boot value (docs/orin-profiles.md): 0.30 was refused
            # by vLLM on this 64 GB unified board (2.25 GiB KV left where 131072
            # needed 3.08 GiB); 0.45 booted clean with 18.86 GiB KV to spare.
            gpu_mem_util=Knob(
                0.45,
                "measured 2026-07-16/17 live boot (docs/orin-profiles.md): util 0.30 "
                "was refused by vLLM on this 64GB unified board, 0.45 booted clean",
            ),
            # A conservative single-model cap, matching the thor/generic
            # convention — the full 131072 native window was only proven for the
            # senses FLEET role (docs/orin-profiles.md), not this legacy surface.
            max_model_len=Knob(
                32768, "sensible single-model cap on the unified board (thor/generic convention)"
            ),
            attention_backend=Knob(
                "TRITON_ATTN",
                "sm_87: every role validated live on this box (senses/embedder/reranker) "
                "used TRITON_ATTN — FlashInfer is unvalidated here (docs/orin-profiles.md)",
            ),
        ),
        status="load-tested",
        # Deliberately NOT traits=(SM_110,) — see module docstring: the cause is
        # different (untested carry-over on sm_87, not a measured sm_110 fact).
        role_overrides={
            "embedder": {
                "attention_backend": Knob("TRITON_ATTN", _CARRIED_OVER_PROVENANCE),
            },
            "reranker": {
                "attention_backend": Knob("TRITON_ATTN", _CARRIED_OVER_PROVENANCE),
                "enforce_eager": Knob(True, _CARRIED_OVER_PROVENANCE),
            },
        },
    )
)
