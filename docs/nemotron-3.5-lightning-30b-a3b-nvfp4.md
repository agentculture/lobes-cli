# NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 — the "worker" role

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded
> *now* — see
> [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **Status: SERVING on TWO boxes — the DGX Spark GB10 as `worker`
> (deviation d1, 2026-08-20) and the Jetson AGX Orin 64GB as `associate`
> (2026-08-26, the first Ampere host). NO-GO on the Jetson AGX Thor.** This
> checkpoint is now the deployed `worker` — but not on the box the original
> plan intended. The covering plan
> (`docs/plans/2026-08-20-nemotron-lightning-worker.md`) targeted the Jetson
> AGX Thor; task t2's live spike found the Thor's Mamba-2 SSD decode path
> wedges on this fleet's pinned nightly (see "The Thor no-go" below), so
> operator-approved deviation d1
> (`.devague/deliveries/nemotron-lightning-worker.json`) swapped the
> topology: the **Spark** now hosts `worker` (this checkpoint), and the
> **Thor** took `cortex` instead (see
> [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) and
> `CLAUDE.md`). Evidence:
> `docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt` (the no-go) and
> `docs/evidence/2026-08-20-accept-worker-hand-spark.txt` (the Spark
> acceptance, numbers below).

**Model id:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
**Tier alias:** `worker` — the role name *is* the alias (capability order:
`hand` < `multimodal` < `worker` < `muse` < `main`/`cortex`).
**Role:** `worker` — the fleet's fast ground-work DOER (the eighth
Colleague role). Replaces `unsloth/Qwen3.6-35B-A3B-NVFP4` in this seat
(nemotron-lightning-worker plan, #187) — that checkpoint is demoted to a
kept `candidate` (cite-don't-delete), not deleted. See
[`qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md) for its own history
and its 61.2 tok/s incumbent baseline, captured just before the flip.
**Hosted on:** the **DGX Spark GB10** (`spark-f8a9`) as `worker`, and the
**Jetson AGX Orin 64GB** as `associate` since 2026-08-26 — see "Live numbers —
Jetson AGX Orin" below. Not the Thor the plan originally targeted — see deviation d1, above. **Text-only** — the
worker role LOSES `image_understanding`/`video_understanding` on this swap
(the outgoing Qwen worker was multimodal; this checkpoint carries no
`vision_config` at all — see "What it is" below).
**Status:** `load-tested` on the Spark (2026-08-20); NO-GO on the Thor on
this fleet's current nightly digest (2026-08-20).

## What it is

An UNGATED (no HF license wall) NVIDIA checkpoint: `NemotronHForCausalLM`,
`model_type: "nemotron_h"` — a hybrid architecture combining Mamba-2
state-space layers, sparse mixture-of-experts blocks, and selective
attention layers across 52 hidden layers. 128 routed experts, 1 shared
expert, 6 experts selected per token — ~3B active of 30B total parameters,
matching the card's own "30B/3B active" framing. This is a **different
engine-support family** from the outgoing Qwen worker
(`Qwen3_5MoeForConditionalGeneration`).

Checkpoint facts (read from the published config files, fetched 2026-08-20):

- **`max_position_embeddings = 1048576`** (1M native ceiling) —
  `config.json`. This is a config-verified ceiling, not card prose. The
  deployed Spark boot serves a trimmed **65536** window as a progressive
  start (below) — nothing has yet exercised the full 1M.
- **NO `vision_config` anywhere in the file** — this checkpoint is
  **TEXT-ONLY**, unlike the outgoing worker's ViT (image+video) intake. The
  `worker` role therefore LOSES `image_understanding`/`video_understanding`
  on this swap; see [`colleague-stack.md`](colleague-stack.md) for the
  current responsibility-token set.
- **`hf_quant_config.json`**: `producer.name = "modelopt"` (version
  `0.44.0rc5`); `quant_algo: "MIXED_PRECISION"` — FP8 on
  attention/lm_head-style projections, `W4A16_NVFP4` (`group_size: 16`) on
  the routed-expert up/down projections; `kv_cache_quant_algo: "FP8"`. This
  is the same nvidia-modelopt family as the `muse` 31B gear and the outgoing
  27B MTP primary (`quantization="modelopt"`), **NOT** the
  `compressed-tensors` format the outgoing Qwen worker used. The deployed
  Spark lane serves `WORKER_QUANTIZATION=modelopt` — Lightning's own
  `quant_method` — measured, not guessed.
- **No `mtp_num_hidden_layers` / draft-head field and no speculative-decoding
  field anywhere in `config.json`** — so the catalog entry's
  `speculative_config` stays empty (the honest sentinel), even though the
  model card separately advertises MTP/DSpark speculative-decoding support.
  **The deployed Spark boot serves plain decode, MTP/DSpark line removed**
  (progressive-start decision, #187) — that support is **declared by the
  card, still unmeasured on this fleet**: Lightning's self-hosted-draft MTP
  and DSpark remain UNEVALUATED (`docs/evidence/2026-08-20-accept-worker-hand-spark.txt`,
  "Still open").
- **Tool calling — VALIDATED live on the Spark, 2026-08-20.** The card's own
  example vLLM serve command pairs `--reasoning-parser nemotron_v3` with
  `--tool-call-parser qwen3_coder` — i.e. the publisher asserts this
  non-Qwen checkpoint emits Qwen3-Coder-shaped tool calls. The Spark
  acceptance run proved this pair live: a structured `get_weather` tool call
  parsed correctly (`finish_reason=tool_calls`, nothing leaked into
  `content`) — see "Live numbers" below. **Strict tool calling
  (`strict: true` schemas / xgrammar structural tags) is UNPROBED** — no run
  has yet exercised it against this checkpoint. The card's suggested
  `--moe-backend marlin` was NOT carried forward on the Spark boot; the
  deployed lane leaves MoE backend auto-selected.
- License: `OpenMDW-1.1` (per the model card).

## The Thor no-go (2026-08-20)

The plan's original target was the Jetson AGX Thor (sm_110). Task t2's
conservative spike (`--max-model-len 32768 --gpu-memory-utilization 0.25`,
no MTP/DSpark, the fleet-wide `8bd082` nightly, vLLM 0.26.1rc1.dev942)
loaded weights and completed `torch.compile` cleanly, then wedged
indefinitely at:

```text
(EngineCore) INFO [mamba_mixer2.py:597] Warming up Mamba2 SSD Triton kernels...
```

— no further output for 25+ minutes, `/health` never came up, operator
killed the process. See
`docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt` for the full
transcript. This was the **third** same-day failure of a non-dense decode
path on sm_110 on this digest, each with a different signature (Qwen3.6-35B
GDN MTP decode: "no kernel image is available"; LFM2.5 conv-hybrid: CUDA
unspecified launch failure; Lightning Mamba-2 SSD: infinite Triton-warmup
wedge) — dense-transformer serving on the same digest/box is unaffected.
This is **sm_110-SPECIFIC**, not hardware-inherent: the wedge is an isolated
sm_110 Mamba-2 issue across multiple vLLM versions. The **fleet's pinned
`8bd082` 0.26.1 nightly** reproduced it first; a follow-up spike tested
the upstream `v0.27.1` release image on the Thor and reproduced the identical
wedge (`docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt`, catalog
entry comment). However, NVIDIA's Lightning model card pins vLLM `0.27.1` and
validates on DGX Spark/GB200/H100, and **a 2026-08-25 physical Jetson AGX
Orin (sm_87) spike cleared the identical vLLM v0.27.1 with this checkpoint
at ~78–81 tok/s single-stream WITH DSpark speculation** — the warmup line
`[mamba_mixer2.py:596] Warming up Mamba2 SSD Triton kernels...` appears at
18:58:24 and proceeds normally, allocating KV cache 94 seconds later
(see `docs/evidence/2026-08-25-spike-lightning-vllm-orin.txt`). So the wedge is
inherent to sm_110's Mamba-2 SSD kernel interaction, not to vLLM or the
checkpoint. Note: `--mamba-backend flashinfer` was used on the Orin, but the
warmup line itself is unrelated to that flag (which governs SSU, logged
separately as "Using flashinfer Mamba SSU backend"). The d1 topology
(Thor=cortex, Spark=worker+hand) stands regardless. The Orin result is a
spike measurement, not a deployed lane — no lobes shape hosts this
checkpoint on sm_87 yet (issue #107, broader tuned-small-model work,
future).

## Live numbers — Spark GB10 (deviation d1, 2026-08-20)

`docs/evidence/2026-08-20-accept-worker-hand-spark.txt`. Deployment:
`lobes init --shape thor-worker --apply --force` rendered on the **Spark**
card (the shape name is historical — see
[`deployment-shapes.md`](deployment-shapes.md#shapes-are-card-agnostic-data-proven-live-by-d1)),
`WORKER_MAX_MODEL_LEN=65536` (progressive start; 1M is a ceiling, not a
served value), `WORKER_GPU_MEM_UTIL=0.30`.

| Metric | Value |
|---|---|
| Model loading | 17.85 GiB (744 s incl. first-time weight download) |
| KV cache dtype | `fp8_e4m3` |
| GPU KV cache size | 3,560,789 tokens |
| Max concurrency at 65,536 tokens/request | **54.33×** |
| Decode (single-stream, no MTP) | **75.1 tok/s** (861 completion tokens / 11.47 s) |
| Known-answer probe (via the Thor→Spark proxy) | PASS, 0.45 s end-to-end |
| Tool-call probe (`nemotron_v3` + `qwen3_coder`) | structured PASS, `finish_reason=tool_calls`, nothing leaked |
| TTFT, short prompts (median of 3) | 75 ms |
| TTFT, long generations (median of 3) | 77 ms, sustained 74.3–74.6 tok/s |

The probe ran from the Thor box through its own gateway's `worker` proxy
(`model=worker` at `thor:8000` → `X-Lobes-Proxied-By` →
`spark.tail0be7e0.ts.net:8001`), so this transcript validates the d1
referral/proxy chain live, not just the engine.

**Comparison with the incumbent it replaced** (`unsloth/Qwen3.6-35B-A3B-NVFP4`
on the Thor, production 0.23.1 engine, MTP ON, captured just before the
swap — `docs/evidence/2026-08-20-baseline-worker-qwen35b-thor.txt`): 61.2
tok/s single-stream, known-answer 5.43 s, tool-call 4.92 s. The new worker
is **+23% decode and ~7× faster short-turn latency** — but this is an
honest DEPLOYED-topology comparison (different box, different engine,
proxy hop included, no speculative decoding on either side today), not a
same-silicon A/B.

## Live numbers — Jetson AGX Orin 64GB as `associate` (2026-08-26)

The **second** box to serve this checkpoint, and the first on **Ampere**. Role:
`associate` — the tenth Colleague role, "they do, but not act" (worker's
responsibilities minus `repo_action`). Shape: `orin-associate`
(associate + hand + embedder + reranker; no cortex, no senses, no audio).
Evidence: `docs/evidence/2026-08-26-accept-orin-associate.txt`; budget
derivation in `docs/evidence/2026-08-25-measure-associate-budget-orin.txt`;
lane spike in `docs/evidence/2026-08-25-spike-lightning-vllm-orin.txt`.

**Why sm_87 can serve this NVFP4 checkpoint at all.** Its `hf_quant_config.json`
is `W4A16_NVFP4` — **weight-only**, 16-bit activations — on the experts, plus
FP8 on `in_proj`/`out_proj`. That is *not* the W4A4 activation quantization that
rules out the Qwen3.8-27B and Gemma-4-31B NVFP4 exports on Ampere. vLLM accepts
`quantization=modelopt_mixed` and selects a full **Marlin** fallback stack (FP8,
NVFP4 GEMM, NVFP4 MoE) with native FlashAttention 2. The Orin also **clears**
the `Warming up Mamba2 SSD Triton kernels` step that wedged the Thor
indefinitely on two engine versions — so that no-go is sm_110-specific.

### Configuration

| | |
|---|---|
| Board | Jetson AGX Orin 64GB, Ampere sm_87, 61.34 GiB unified, **ZERO swap** |
| `ASSOCIATE_GPU_MEM_UTIL` | **0.56** — the vendor's 0.70 and an earlier 0.63 were both REFUSED at boot |
| `ASSOCIATE_MAX_MODEL_LEN` | 128,000 (native ceiling 1,048,576 — unexercised) |
| `ASSOCIATE_QUANTIZATION` | `modelopt` → resolved to `modelopt_mixed` |
| `ASSOCIATE_KV_CACHE_DTYPE` | `bfloat16` — sm_87 has no FP8 KV path, so the checkpoint's declared FP8 `kv_cache_quant_algo` is overridden |
| Speculation | **OFF**. `ASSOCIATE_SPECULATIVE_CONFIG` exists but is default-off and UNMEASURED on the shape |
| Engine | `vllm/vllm-openai@sha256:7c5a10e9…` (the pre-bump nightly this box still pinned — not the `v0.27.1` the lane spike used) |

| Boot fact | Value |
|---|---|
| Model loading | 17.81 GiB / 43.5 s (weights warm in cache) |
| Available KV | 9.35 GiB |
| GPU KV cache size | **1,524,000 tokens** |
| Max concurrency @ 128,000 | **11.91×** |
| init engine | 228.7 s (compilation 40.2 s) |

Full shape resident: associate 34.85 + hand 5.80 + rerank 5.34 + embed 4.83 GiB
= **~50.9 GiB used, ~1 GiB free** on a zero-swap board. See issue #216.

### Probes

| Probe | Result |
|---|---|
| `17 * 23 = 391` | PASS (finish=stop) |
| train 14:45 → 17:10 = 145 min | PASS (finish=stop) |
| reasoning trace | present (`reasoning`, len 1356) |
| tool calling (`tool_choice:auto`) | PASS |
| unauthenticated `GET /v1/models` | **401** — including over the tailnet address |
| valid bearer | 200 |

### Throughput — depth sweep at the incumbent's own shape

128 max output tokens, cache-defeating unique prompts with `prompt_tokens` read
back from the server. (A first sweep using repetitive filler reported ~1,175×
at depth 32,768; that was a `--enable-prefix-caching` artifact and was
discarded.)

| Depth | `associate` TTFT | `associate` decode | incumbent TTFT | incumbent decode | TTFT gain |
|---:|---:|---:|---:|---:|---:|
| 0 | 129 ms | 50.28 tok/s | 1,566 ms | 2.61 tok/s | 12× |
| 512 | 316 ms | 51.52 tok/s | 12,407 ms | 2.62 tok/s | 39× |
| 2,048 | 1,179 ms | 50.98 tok/s | 41,581 ms | 2.62 tok/s | 35× |
| 8,192 | 3,749 ms | 50.52 tok/s | 143,122 ms | 2.58 tok/s | 38× |
| 32,768 | **16,939 ms** | **52.53 tok/s** | **610,020 ms** | **2.43 tok/s** | **36×** |
| decay 0→32k | — | **none** | — | ~7% | — |

Incumbent = `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` on llama.cpp, same board
(`docs/evidence/2026-08-23-spike-qwen38-gguf-llamacpp-orin.txt`).

| Metric | `associate` | incumbent | Ratio |
|---|---:|---:|---:|
| Decode @ 32,768 | 52.53 tok/s | 2.43 tok/s | **21.6×** |
| End-to-end @ 32,768 | 19.4 s | 662.7 s | **34×** |
| Prefill | ~1,612 tok/s | ~64 tok/s | **25×** |
| `lobes benchmark` (×3) | 54.3 tok/s | — | — |

### Cross-box — NOT a same-harness comparison

| Host | Role | Engine | Speculation | Decode |
|---|---|---|---|---:|
| Orin sm_87 | `associate` | vLLM | off | 52.5–54.3 tok/s |
| Spark GB10 | `worker` | vLLM | off | 75.1 tok/s |
| Orin sm_87 | `cortex` | llama.cpp | n/a | 2.61 tok/s |
| Orin (lane spike) | — | vLLM v0.27.1 | DSpark ×5 | ~78–81 tok/s |

The Orin reaches **~72%** of the Spark's plain-decode rate on materially weaker
silicon. Different engine builds and resident sets, so the DSpark row is
indicative only.

**Deliberately absent: NVIDIA's "89 tokens/sec on Jetson AGX Orin."** It is a
multi-step agentic-workload aggregate WITH speculation, not a single-stream
decode figure, and jetson-ai-lab publishes no Orin command to reproduce it.
**Three separate defects** were found in that page's published recipes when run
verbatim: a llama.cpp `-hf` tag naming a quantization that does not exist in the
repo; a GGUF that will not load in NVIDIA's own Jetson-Orin image (missing Mamba
SSM tensor `blk.5.ssm_in.weight`, llama.cpp b10373 — a build-version gap, NOT an
sm_87 limit); and a vLLM `--speculative_config.model` whose repo id omits its
`-NVFP4` infix. Treat those recipes as unverified drafts.

### Still open on the Orin

1. DSpark is wired but **default-off and unmeasured** on the full shape; the
   drafter costs KV and the shape leaves ~1 GiB free.
2. Engine drift — `v0.27.1`, `7c5a10e9` and the template's `8bd082` are all in
   play; none compared against the others on this board.
3. `lobes fleet up` cannot start this shape (it builds the audio overlay a
   no-audio shape does not host).
4. No cross-box probe has addressed this lane from a peer.
5. Marlin NVFP4 correctness on sm_87 rests on a small probe set — vLLM
   `#34694`/`#49070` report garbled output on this fallback on non-Blackwell parts.

## Status and gating

The Spark numbers above are **measured**, not the arithmetic/hypothesis
this doc previously carried. Still open, per the acceptance transcript:

1. `WORKER_MAX_MODEL_LEN=65536` is a progressive start; growing toward the
   1M ceiling needs its own live boot (54.33× concurrency at 65K suggests
   ample headroom).
2. Lightning's self-hosted MTP draft and DSpark speculative decoding are
   UNEVALUATED.
3. Strict-tools (`strict: true` / xgrammar structural tags) arming is
   UNPROBED against this checkpoint.
4. The senses/stt/tts adverts on the re-scaffolded Spark were not
   re-validated in this window (the audio overlay containers stayed up
   throughout, untouched).

See `docs/plans/2026-08-20-nemotron-lightning-worker.md` for the full task
list, `docs/qwen3.6-35b-a3b-nvfp4.md` for the checkpoint this one replaces in
the `worker` seat, and `docs/qwen3.6-27b-text-nvfp4-mtp.md` +
`CLAUDE.md` for the `cortex` half of the d1 swap.
