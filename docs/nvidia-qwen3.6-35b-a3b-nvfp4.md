# `nvidia/Qwen3.6-35B-A3B-NVFP4` — the Thor `worker` lane

**Status: VALIDATED live on the physical Jetson AGX Thor, 2026-09-10** — see
issue [#244](https://github.com/agentculture/lobes-cli/issues/244). This is the
deployed `worker` checkpoint. Every number below was
measured on that box; nothing here is copied from a vendor page or a forum
post, and the external figures that ARE quoted are labelled as external.

> **Read this first if you are re-running it.** The single most useful fact in
> this document is not a throughput number — it is that **you must not force
> `--moe-backend`** on this lane, and the reason is not the one the fleet
> believed. See [MoE backend](#moe-backend-two-backends-run-at-once).

## What it is

NVIDIA's own ModelOpt export of Qwen3.6-35B-A3B. Read from the checkpoint's own
`config.json` + `hf_quant_config.json` on 2026-09-10 (HF repo last modified
2026-08-29) — not from the model card:

| field | value |
|---|---|
| architecture | `Qwen3_5MoeForConditionalGeneration` (`model_type qwen3_5_moe`) |
| native context | **262144** |
| experts | 256 total / 8 active per token, 40 layers |
| modality | **MULTIMODAL** — `vision_config` present (deepstack ViT, image + video token ids) |
| quantization | ModelOpt `MIXED_PRECISION`: experts + shared-expert `W4A16_NVFP4` (group_size 16, **weight-only**), FP8 on the `linear_attn`/`self_attn` projections |
| KV cache | `kv_cache_quant_algo: FP8` **declared** |
| MTP | `mtp_num_hidden_layers: 1` — carries its own draft head — but `exclude_modules: ["mtp.layers.0*", "mtp*"]`, so **the MTP module is UNQUANTIZED** |
| on-disk | 21.82 GiB across 3 safetensors shards; 20.4–21.3 GiB resident once loaded |

It is a **different checkpoint** from the two siblings in
[`qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md): `unsloth/…`
(`compressed-tensors`, the pre-d1 Thor worker, source of every historical Thor
worker measurement) and `mmangkad/…` (32K native). Do not mix their numbers.

### Prior art: this id failed to load here once

On **2026-05-31**, on the DGX Spark GB10, this exact id would not load on vLLM
0.19.0/0.21.0: `marlin`/`flashinfer_trtllm` → *"not supported for unquantized
MoE"*, `triton`/auto → `KeyError: layers.0.mlp.experts.w2_input_scale`. That is
recorded in the sibling doc and was the origin of this work's risk r1.

**It does not reproduce on the fleet's pinned nightly.** `modelopt_mixed` is
now a first-class quantization method and the model serves. Two things changed
and neither alone is proof: the engine moved four minor versions, and the HF
repo was re-uploaded 2026-08-29.

## Engine

```text
image:  vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695
vLLM:   0.26.1rc1.dev942+g5a4c8d992
```

This is the **fleet-wide pin** (`VLLM_NIGHTLY_IMAGE`), not a special build. No
custom image, no from-source compile, no patched wheel.

Pre-boot probes against that image (cheap, non-destructive, worth re-running
before trusting any other pin):

```bash
docker run --rm --entrypoint python3 $IMG -c "
from vllm.model_executor.models.registry import ModelRegistry
print('Qwen3_5MoeForConditionalGeneration' in ModelRegistry.get_supported_archs())
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
print('modelopt_mixed' in QUANTIZATION_METHODS)"
# -> True / True

docker run --rm --entrypoint bash $IMG -c \
  '/usr/local/cuda-13.0/bin/cuobjdump --list-elf \
   /usr/local/lib/python3.12/dist-packages/vllm/_moe_C_stable_libtorch.abi3.so \
   | grep -o "sm_[0-9]*" | sort -u'
# -> sm_80 sm_87 sm_89 sm_90 sm_100 sm_110 sm_120
```

`dflash` and `dspark` are already in this image's `SpeculativeMethod` literal —
DFlash needs **no** custom build here, contrary to the issue's assumption.

## Deployed recipe (Jetson AGX Thor, sm_110)

`.env` on the box:

```bash
WORKER_FEASIBLE=true
WORKER_MODEL=nvidia/Qwen3.6-35B-A3B-NVFP4
WORKER_SERVED_NAME=nvidia/Qwen3.6-35B-A3B-NVFP4
WORKER_QUANTIZATION=modelopt
WORKER_MAX_MODEL_LEN=262144
WORKER_GPU_MEM_UTIL=0.45
WORKER_KV_CACHE_DTYPE=fp8
WORKER_MAX_NUM_SEQS=1
WORKER_REASONING_PARSER=qwen3
WORKER_BASE_URL=http://vllm-worker:8000
WORKER_SPECULATIVE_CONFIG="'--speculative-config={\"method\": \"dflash\", \"num_speculative_tokens\": 12, \"model\": \"z-lab/Qwen3.6-35B-A3B-DFlash\"}'"
COMPOSE_PROFILES=worker

# cortex leaves this box; the SINGULAR origin is REQUIRED beside the plural
# pool or the gateway refuses to boot (_check_pool_arming).
PRIMARY_FEASIBLE=false
PRIMARY_PEER_ORIGIN=http://spark.tail0be7e0.ts.net:8001
PRIMARY_PEER_PROXY=true
PRIMARY_PEER_API_KEY=<the Spark's own inbound key>
```

> **The quoting of `WORKER_SPECULATIVE_CONFIG` is load-bearing.** Outer double
> quotes, inner double quotes escaped, the payload wrapped in literal single
> quotes. Get it wrong and compose shell-lexes the JSON into separate argv
> tokens — observed live as
> `--speculative-config={method:` `dflash,` `num_speculative_tokens:` … and a
> crash loop. Always verify with `docker inspect`, never from `.env`.

Resulting argv, from `docker inspect model-gear-vllm-worker`:

```bash
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name=nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --host=0.0.0.0 --port=8000 \
  --quantization=modelopt \
  --max-model-len=262144 \
  --gpu-memory-utilization=0.45 \
  --kv-cache-dtype=fp8 \
  --max-num-seqs=1 \
  --speculative-config={"method": "dflash", "num_speculative_tokens": 12, "model": "z-lab/Qwen3.6-35B-A3B-DFlash"} \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  --trust-remote-code
```

Environment: `VLLM_GDN_DECODE_KERNEL=triton` (set box-wide; it is what makes
speculative decoding reachable on sm_110 at all — see
[`qwen3.8-27b-nvfp4.md`](qwen3.8-27b-nvfp4.md)).

**No `--moe-backend`, no `--attention-backend`, no `--load-format`, no
`--async-scheduling`.** Auto-select is correct on this board; the issue's
recipe named several flags that are either wrong here or unnecessary.

### Draft model

`z-lab/Qwen3.6-35B-A3B-DFlash` (737 MiB, `DFlashDraftModel`, block_size 16,
target_layer_ids `[1,6,11,16,22,27,32,37]`). Pulled from the hub like any
other checkpoint; nothing bespoke.

## MoE backend: two backends run at once

The standing rule — **never force `--moe-backend` on this lane** — is unchanged
from the unsloth era, but its recorded reason was wrong. From a live boot log:

```text
Using MarlinNvFp4LinearKernel for NVFP4 GEMM
Using 'MARLIN' NvFp4 MoE backend out of potential backends:
  ['FLASHINFER_TRTLLM','FLASHINFER_CUTEDSL','FLASHINFER_CUTEDSL_BATCHED',
   'FLASHINFER_CUTLASS','VLLM_CUTLASS','MARLIN','HUMMING','EMULATION']
Using FLASHINFER attention backend out of potential backends: ['FLASHINFER','TRITON_ATTN']
Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)
GDN decode kernel: triton
...
Loading drafter model...
Using TRITON Unquantized MoE backend out of potential backends:
  ['FlashInfer TRTLLM','FlashInfer CUTLASS','TRITON','BATCHED_TRITON']
```

The **target's** experts are `W4A16_NVFP4` → MARLIN. The **drafter's** experts
are unquantized (`exclude_modules: ["mtp*"]`) → TRITON. Both, simultaneously.

So the 2026-07-31 refusals recorded in `thor-worker.toml`
(`marlin: "not supported for unquantized MoE"`) were **never an sm_110 fact** —
they were one forced value applied to two differently-quantized MoEs. Marlin
genuinely IS the right kernel for the target's experts on sm_110; it simply
cannot also be the drafter's. The issue's `--moe-backend marlin` recommendation
is therefore *correct about the kernel* and *wrong as a flag*.

## Measured results

All figures: physical Thor, MAXN, L4T R38.2.2, clocks **not** pinned, Culture
production stack co-resident (not a quiesced box), streaming, batch 1, no
client timeouts. TTFT = wall time to the first content delta; decode tok/s =
`(completion_tokens - 1) / (last delta - first delta)`, prefill excluded.
Harness: `scripts/stream-measure.py`.

### Speculation sweep — 65536 window, util 0.45, fp8 KV

| arm | decode tok/s (median of 3) | acceptance | mean accept len | KV pool | ceiling @65536 |
|---|---|---|---|---|---|
| none | 68.5 | — | — | 2,100,601 | 32.05× |
| MTP k=1 | 101.8 | 97.7% | 1.98 / 2 | 1,569,667 | 23.95× |
| MTP k=3 | 129.0 | 93.2% | 3.82 / 4 | 1,379,105 | 21.04× |
| MTP k=5 | 136.7 | 96.2% | 5.81 / 6 | 1,201,073 | 18.33× |
| **MTP k=6** | **156.3** | 87.2% | — | 1,148,093 | 17.52× |
| MTP k=7 | 155.3 | 73.0% | 6.11 / 8 | 1,079,619 | 16.47× |
| **DFlash k=12** | **182.6** | 74.7% | **9.97 / 13** | 548,653 | 8.37× |

**k=6 is the MTP peak** and strictly dominates k=7 — faster *and* better
acceptance. k=7 is not recommended despite being near-fastest among MTP arms.

Per-position acceptance is where the shape lives:

```text
MTP k=5     1.000, 1.000, 1.000, 0.946, 0.865
MTP k=7     1.000, 1.000, 1.000, 0.889, 0.667, 0.333, 0.222
DFlash k=12 1.000, 1.000, 0.970, 0.909, 0.909, 0.727, 0.636, 0.636,
            0.545, 0.545, 0.545, 0.545
```

DFlash's tail is far flatter than MTP's — that is what buys its depth.

### Deployed lane, 262144 window, through the gateway (`model=worker`)

| probe | TTFT | decode tok/s |
|---|---|---|
| code (109 tok), 3 runs | 148.5 / 144.0 / 130.5 ms | 163.19 / 198.12 / **196.56** |
| deep prompt (9,020 tok) | 3033.9 ms | 91.52 |

Budget at 262144: **KV pool 1,199,883 tokens = 4.58× ceiling**. Weights load in
~47 s.

Comparison table (same box, same day):

| | tok/s |
|---|---|
| incumbent `cortex` (Qwen3.8-27B, MTP n=2, 262144) | **18.7** |
| the historical unsloth worker figure the issue cited (different engine) | 61.2 |
| the plan's target | 100 |
| **deployed worker, DFlash k=12, via the gateway** | **196.6** |

External reference, **not** an acceptance threshold: the Thor field guide at
`patrickbdevaney/qwen-3.6-35b-a3b-dflash-jetson-agx-thor` reports 139.1 tok/s
conc=1 for this model with a custom from-source build. Our measurement exceeds
it on the stock pinned image.

### Concurrency and prompt depth — read this before quoting 196.6 tok/s

**The headline decode rate is a SHORT-PROMPT number.** Decode speed on this lane
depends strongly on prompt depth, and agentic work is deep-prompt work:

| prompt | decode tok/s | TTFT |
|---|---|---|
| 25 tokens (the headline probe) | **185–197** | 230–430 ms |
| 8,786 tokens, distinct content | **41.5–46.5** | 2.7–3.9 s |

That is a ~4x difference from depth alone, before any concurrency. Quote the
short-prompt figure only as a short-prompt figure.

#### Width sweep with DISTINCT long prompts — the agentic-realistic one

Each stream gets its own ~8.8k-token pseudo-source file, with a per-invocation
nonce so prefix caching cannot dedupe across streams or across runs
(`max_num_seqs=4`, DFlash k=12, 262144 window):

| width | per-stream decode tok/s | **aggregate decode** | aggregate prefill | TTFT |
|---|---|---|---|---|
| 1 | 46.5 | 12.1 | 2,375 | 2,715 ms |
| 2 | 45.7 / 13.2 (mean 29.5) | 10.4 | 3,036 | 3,639 / 5,070 ms |
| 4 | 31.9 / 14.1 / 8.3 / 7.0 (mean 15.3) | 13.3 | 3,249 | 4,169 → 9,836 ms |

**Aggregate decode is FLAT (12.1 → 10.4 → 13.3).** Concurrency buys no extra
total throughput here; it divides the same work into slower, markedly uneven
streams (7.0 to 31.9 tok/s at width 4 — the stream that finishes first inherits
the tail) and pushes worst-case TTFT from 2.7 s to 9.8 s.

**The lane is PREFILL-BOUND for deep prompts**, saturating around 2.4–3.2k
tok/s of prefill. That ceiling, not decode, is what multi-agent capacity
planning must budget.

#### A correction, recorded rather than quietly fixed

An earlier revision of this document reported "concurrency scales: 348.1 tok/s
aggregate at width 4". **That was wrong.** It came from sending four IDENTICAL
short prompts with `--enable-prefix-caching` on, so all four streams shared one
cached prefix and the box was doing roughly one prompt's work. Repeating the
sweep with distinct prompts produced the flat aggregate above. The lesson is
general: **any concurrency benchmark that reuses one prompt across streams
measures the prefix cache, not the engine.**

#### Why `max_num_seqs=1`

Given a flat aggregate, the cap costs no total throughput and buys the best
per-request latency, which is what an interactive agent experiences. Measured
at `max_num_seqs=1`: 185.2 tok/s short-prompt (vs 196.6 at cap 4 — within
run-to-run variance), 41.5 tok/s on an 8.8k-token prompt.

The setting is `WORKER_MAX_NUM_SEQS=1`. Raise it only with a distinct-prompt
measurement that shows an aggregate gain; the identical-prompt sweep will
always flatter it.

#### Caveat in the other direction — real agents DO share prefixes

The distinct-prompt test is deliberately the pessimistic bound: it gives every
stream unique content. A real agent re-sends a growing conversation, so
successive turns of the SAME session share a long prefix and hit the cache.
Truth for a given deployment sits between the two tests, nearer the distinct
one whenever several independent agents run at once. Neither number is the
whole answer, which is why both are recorded.

#### The agentic sample that started this

Sampled while Qwen Code drove an agentic PR review against this lane
(4 running + 6 queued), 180 s window:

```text
generation tokens 26,625 -> 38,608   = 11,983 in 180 s =   66.6 tok/s aggregate
prompt tokens  3,033,992 -> 3,273,132 = 239,140 in 180 s = 1,328 tok/s prefill
```

20x more prefill than generation. Consistent with the distinct-prompt sweep:
the workload is prefill-bound and the decode aggregate is low because the lane
is busy prefilling, not because speculation misbehaves under batching.

## Correctness

| probe | result |
|---|---|
| known-answer ("capital of France") | "Paris" ✅ |
| structured tool call | `finish_reason: tool_calls`, `get_weather({"city":"Paris"})`, `content` empty ✅ |
| parser pair | `--tool-call-parser=qwen3_coder` + `--reasoning-parser=qwen3`, verified live ✅ |
| image intake + negative controls | red→"Red", blue→"Blue", green→"Green" ✅ |
| video intake | **declared by the checkpoint, NOT measured** (#108) |
| Qwen Code agentic loop | read → `write_file` → correct FizzBuzz, verified by executing it ✅ |

## Operational traps (all hit live during bring-up)

1. **`restart: unless-stopped` starves a heavy lane out of its own memory.**
   vLLM's startup free-memory precheck fails; the container restarts; the
   dying attempt still holds its allocation, so each retry sees *less* memory
   than the last (53 → 30 GiB observed) while `free -g` reports ~102 GiB.
   Reached `RestartCount=30`. **Not a leak** — with the container stopped,
   `torch.cuda.mem_get_info()` reads 100.80 GiB free and `ps` RSS matches
   `free`. Working sequence:

   ```bash
   docker compose stop vllm-worker && sleep 10
   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches && sleep 5
   docker compose up -d --no-deps vllm-worker     # NOT --force-recreate
   ```

2. **CUDA-visible free memory ≠ the OS view.** Check with
   `torch.cuda.mem_get_info()` inside the image.
3. **A window shrink breaks consumers no model-id audit will catch.** Qwen Code
   requests 64000 output tokens by default → `HTTP 400: maximum context length
   is 65536 … you requested 64000 output tokens`. This is why the lane serves
   the full native 262144 rather than the 65536 the sweep measured.
4. **`env -u GATEWAY_API_KEY docker compose …`** — this box's `~/.bashrc`
   exports `GATEWAY_API_KEY` and compose reads shell env ahead of `.env`, so an
   ssh-driven restart silently arms the inbound gate and 401s everything.
5. **The singular peer credential is separate from the pool's.**
   `PRIMARY_PEER_API_KEYS` (plural) and `PRIMARY_PEER_API_KEY` (singular) parse
   independently; a proxied `model=cortex` 401s with `X-Lobes-Proxied-By` set
   until the singular one is declared.

## Status and gating

* **VALIDATED** on the Thor for: load, MoE/attention backend selection, budget
  at both 65536 and 262144, single-stream throughput, speculation sweep,
  tool calls, image intake, and end-to-end agentic use.
* **NOT validated:** any peer reaching this lane cross-box (no box declares
  `WORKER_PEER_ORIGIN` pointing at the Thor); long-context retrieval at
  262144; video intake; `--load-format fastsafetensors`; `--async-scheduling`;
  concurrency beyond width 4.
* Evidence: `docs/evidence/2026-09-10-*` (seven transcripts + the raw sweep
  log). Delivery record:
  `docs/deliveries/2026-09-10-thor-worker-arm-qwen3-6-35b-a3b-recipes.md`.
