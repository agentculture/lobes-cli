# Qwen3.8-Flash-Next GGUF — llama.cpp vs vLLM on the Thor

**Status: NOT SERVED. Deferred 2026-08-27, not abandoned.** No box in this
fleet has booted this checkpoint. Nothing here is measured on it — every number
below is either read off a published card or measured on a *different* model,
and says which. Per #108 nothing may claim otherwise until a transcript lands
under `docs/evidence/`.

This doc answers **what we tried to do, why each engine, and why neither
happened yet.** For the container images themselves — digests, versions, what
each one is validated on — see `docs/image-ledger.md`.

Sibling doc: `docs/qwen3.8-27b-gguf-llamacpp.md`, the 27B GGUF gear, which is
where every llama.cpp number cited here was actually measured.

## The checkpoint

| fact | value | source |
|---|---|---|
| repo | `unsloth/Qwen3.8-Flash-Next-GGUF` | Unsloth |
| architecture | `qwen4exp` — an early preview of the Qwen4 architecture | Qwen |
| parameters | **125B MoE + 51B n-gram embedding**, ~6B active per token | Qwen announcement |
| MoE | 512 experts, 10 routed + 1 shared | Qwen |
| attention | hybrid — 3 of every 4 layers Gated DeltaNet, the 4th Qwen Sparse Attention (QSA) | Qwen |
| PLE / n-gram | a 20,000,000 x 2560 learned lookup table at layer 2, bigram/trigram indexed | Qwen |
| context | 262144 native, 1M via YaRN | Qwen |
| modality | multimodal MoE | Qwen |

**The n-gram table is the whole story.** It is memory-heavy and compute-light —
a lookup, not a matmul — and Qwen designed it to be offloadable and
asynchronously prefetched. That is what makes a 176B-stored-parameter model
plausible on one box, and it is also what breaks every convenient assumption
below.

## Why this was interesting for a 128 GB unified board

~6B active per token means decode reads ~6B of weights, not 125B. On paper that
is a frontier-class model at a small model's decode cost, and the fleet has two
128 GB unified-memory boards.

## Why not native — nothing fits

| format | size | fits 122 GiB? |
|---|---|---|
| BF16 | 335.28 GiB | no |
| `Qwen/Qwen3.8-Flash-Next-FP8` (official) | 172.78 GiB | no |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | 135 GB | **no — misses by ~13 GB** |
| GGUF `UD-Q3_K_XL` | 90 GB | yes |
| GGUF `UD-IQ1_M` | 74.5 GB | yes |

Sizes read off `recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next` and the respective
model cards.

The NVFP4 near-miss is the instructive one: it quantizes **only** the 48 main
MoE layers' routed experts to W4A4 and is published for SGLang. The ~35 GB PLE
table is the floor no expert-only quantization can go below — which is also why
`UD-IQ1_S` at 72.5 GB is not nearly as small as a 1-bit quant of 176B
parameters should be.

**So GGUF is currently the only format that fits.** Any engine choice here is a
choice about *which engine reads a GGUF*, not about serving the native weights.

## Route 1 — llama.cpp

**Why.** Unsloth ships this checkpoint against their own llama.cpp branch:
`unslothai/llama.cpp`, branch `qwen4exp/qwen3.8-flash-next`, the head of
`ggml-org/llama.cpp#27742` (open, unmerged). The repo already has a working
llama.cpp lane — the opt-in `llamacpp` compose profile and the `engine` axis in
`lobes/catalog.py` — so this is a second gear on a proven axis, not new
machinery.

The PR description calls the QSA indexer and PLE n-gram embedding unfinished.
**That is stale relative to its own branch:** both landed
(`c88c9166 llama: QSA sparse attention for qwen4exp`,
`ad4fa3fa llama: qwen4exp PLE n-gram hash embedding`), along with PLE conv
state across ubatches, per-context PLE history, indexer KV save/restore, and a
quantized KV cache in the QSA path.

**Why not — prefill.** This is the decisive objection, and it is measured
in-repo on a *different* model, on the Orin
(`docs/evidence/2026-08-26-accept-orin-associate.txt`):

| | llama.cpp | vLLM | ratio |
|---|---|---|---|
| prefill | ~64 tok/s | ~1,612 tok/s | **~25x** |
| TTFT @ 32768 | 610 s | 16.9 s | **~36x** |

Even the 27B GGUF's MAXN best is 253.84 tok/s prefill
(`docs/qwen3.8-27b-gguf-llamacpp.md`). A ten-minute TTFT at 32K makes a
125B model unusable for long-context work no matter what decode does.

Secondary: llama.cpp is a **second engine** the gateway, roles, pressure policy
and parser pairs do not otherwise assume.

## Route 2 — vLLM (the chosen direction)

**Why.** vLLM is the fleet's engine. And the version move it requires is a
shared gain rather than a cost carried by this test alone — a newer nightly may
speed up the incumbent Qwen3.8-27B NVFP4 cortex, so the upgrade is worth
benchmarking on its own and the fleet keeps that result even if this checkpoint
never lands.

**Why not — four stacked unknowns**, none of which is a judgement call:

1. **The version does not exist.** `recipes.vllm.ai` requires vLLM "0.29.0+".
   The latest tagged release is **`v0.28.0`** (2026-08-26). "0.29.0+" names
   main. There is no version to pin — only a nightly digest.
2. **`cu129` `sm_110` kernel coverage is unverified.** arm64 nightlies exist
   (`nightly-aarch64`, `cu129-nightly-aarch64`). That is arch *availability*,
   not kernel coverage — `cu128` ships no `sm_110` SASS and no PTX, and an
   image that installs cleanly with no kernels for this board is exactly how
   chatterbox sank (#145).
3. **`vllm-gguf-plugin` does not list this model or these quants.** Its tested
   quant list is `Q6_K / Q8_0 / IQ4_XS / Q4_K_M / Q4_0` — neither `UD-IQ1_M`
   nor `UD-Q3_K_XL` is on it; its architecture list stops at Qwen 2.5/3 with no
   `qwen4exp`; multi-shard GGUF is undocumented; and it warns in its own words
   that appearing in vLLM's supported-model list "does not by itself guarantee
   GGUF compatibility."
4. **File size is not the footprint on vLLM.** llama.cpp mmaps the file; vLLM
   loads and dequantizes, plus runtime buffers, dequant workspace, CUDA graphs
   and KV. Against 122 GiB that margin decides the outcome — the **load-time
   peak**, not the steady state, is what has to be measured first, and
   `UD-IQ1_M` (74.5 GB) may be the only rung that loads at all.

### The offload that does not help here

`VLLM_PLE_CPU_OFFLOAD=1` is real, shipped, and named in the vLLM recipe: it
keeps the 51B n-gram lookup in host RAM. It is the single most attractive vLLM
feature for this model — and it is a **no-op on this hardware.** Thor and Spark
are unified-memory boards: host RAM and GPU RAM are one physical pool. Moving
the PLE table to "host" frees nothing. The feature's whole premise is a
discrete-GPU memory split these boxes do not have.

Do not SSD-offload it either. The PLE's hashed random-access pattern is hostile
to NVMe, and `ggml-org/llama.cpp#27766` (open 2026-08-26) exists because
someone is already hitting that on unified memory.

## The honest ordering

Independent checks agreed with the operator-supplied research on this point:
**llama.cpp is the more likely of the two to actually run the 90 GB Q3**, and
vLLM GGUF is the more likely to hit architecture, weight-mapping or kernel
issues. The vLLM direction was chosen anyway, on prefill and fleet coherence —
a deliberate trade, recorded as one, not an expectation that it works better.

## What resuming needs

Not a re-derivation — the specs and the ledger carry it:

- `docs/specs/2026-08-27-flash-next-on-vllm.md` — the vLLM route, converged and challenged
- `docs/specs/2026-08-27-qwen3-8-flash-next-gguf-candidate.md` — the llama.cpp route, same
- `docs/image-ledger.md` — the two unbuilt image recipes and their blockers

Open items either spec would still have to settle:

- **Abort criterion.** `docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt`
  records an indefinite `sm_110` hang on a hybrid state-space decode path.
  Flash-Next is 3-of-4 layers Gated DeltaNet — same family. A wall-clock
  timeout per boot attempt, written down before the first boot.
- **The bar.** >= 25 tok/s decode at MAXN — roughly 2x this Thor's incumbent
  vLLM NVFP4 cortex at 12.1 tok/s
  (`docs/evidence/2026-08-20-accept-cortex-local-thor.txt`) — **and** prefill /
  TTFT in vLLM's territory rather than llama.cpp's. Below either, the incumbent
  27B is simply the better fit and this checkpoint buys nothing for ~75 GB of RAM.
- **Blast radius.** The Spark proxies its `embedder` to this Thor, so a
  fleet-down window on this box costs the mesh embeddings, not just cortex.
- **Quant choice is deferred to the data** — `UD-Q3_K_XL` is the ladder's
  ceiling, not its destination. A low rung that benchmarks well may be the one
  adopted.
- **Vision** is out of scope for a first pass by choice, not by engine limit —
  the llama.cpp branch does carry image work.
