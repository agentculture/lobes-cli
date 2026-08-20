# NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 — the "worker" role

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded
> *now* — see
> [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **Status: SERVING on the DGX Spark GB10, deviation d1 (2026-08-20).** This
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
**Hosted on:** the **DGX Spark GB10** (`spark-f8a9`), not the Thor the plan
originally targeted — see deviation d1, above. **Text-only** — the
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
This is scoped to the **fleet's pinned `8bd082` 0.26.1 nightly**, not to the
Thor hardware itself: NVIDIA's Lightning model card pins vLLM `0.27.1` and
validates on DGX Spark/GB200/H100, and Jetson AI Lab separately publishes an
official Thor recipe for Lightning on the **release** image
`vllm/vllm-openai:v0.27.1`
(<https://www.jetson-ai-lab.com/models/nemotron3-5-lightning/#run-on-jetson>).
A future move to a `>=0.27.1`-based fleet image may reopen this — two
follow-up spikes are recorded but not yet run: (1) Lightning on the Thor via
`v0.27.1`, and (2) whether `v0.27.1` also restores the sm_110 GDN MTP decode
kernel, which would let the Thor-local `cortex` (deviation d1) re-enable
MTP. The d1 topology (Thor=cortex, Spark=worker+hand) stands regardless of
either follow-up's outcome.

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
