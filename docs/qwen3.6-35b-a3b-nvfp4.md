# MoE candidate: `mmangkad/Qwen3.6-35B-A3B-NVFP4`

A **MoE candidate** — the *former* fleet fallback. It was **superseded as the
fallback choice** by the dense `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`
([`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md)), because
this checkpoint never loaded on the GB10 (see the status note below). (The fleet
now runs one *generate* backend by default — any warm fallback is opt-in via the
`FALLBACK_*` keys.) It remains
in the **supported catalog** as a candidate to re-test on a quiet/dedicated box
(`model overview --list`). See [`docs/gateway-fleet.md`](gateway-fleet.md) for the
fleet topology and the
[catalog-vs-warm distinction](gateway-fleet.md#supported-catalog-vs-warm-backends)
(what you *can* load vs. what's loaded *now*).

Source: <https://huggingface.co/mmangkad/Qwen3.6-35B-A3B-NVFP4>.

> **Status: load-tested 2026-05-30 — does NOT load reliably on this GB10.** First
> live `model fleet up` on `spark-f8a9`: co-resident with the 27B primary it hit
> `CUDA error: out of memory` on engine init and crash-looped (14+ restarts);
> *solo* (65 GiB free) it still crashed/restarted and then stalled at "Loading
> safetensors checkpoint shards: 0%" with the GPU idle, never reaching `/health`
> in 8+ min. **No benchmark obtained.** The architecture-derived expectations
> below are *unconfirmed*. Two root causes are entangled and need separating:
> (1) co-residence with another ~30B model overruns the 121.7 GiB unified pool
> (see [`docs/gateway-fleet.md`](gateway-fleet.md)); (2) the checkpoint's own
> load path (MoE + multimodal ViT + Mamba, single 24 GiB safetensors) stalls/OOMs
> even solo under swap pressure. Re-test on a quiet box before relying on it.

**Update — load-tested 2026-05-31 — DOES load solo with the right flags.** With
the 27B primary stopped (so the 35B had the GB10 to itself) and shahizat's tuning
(`--moe-backend marlin`, flashinfer, async scheduling, chunked prefill) at
`--gpu-memory-utilization 0.70 --max-model-len 32768`, it loaded healthy in ~6 min
(~84 GiB resident) and served. Two caveats found: (1) `0.85` util fails the
pre-flight reservation on this *shared* box (only ~90 of 121.7 GiB free — the
audio NIMs + reachy hold the rest), so `0.70` is the working value; (2) the MTP
`--speculative-config` from shahizat's recipe **fails to load** on this `mmangkad/`
copy (`qwen3_5_mtp.py` weight-shape mismatch on vLLM nv26.04) — it is tied to his
`nvidia/` checkpoint. Measured numbers under "Live replication" below.

## What it is

- An **NVFP4 (Mixture-of-Experts)** checkpoint: ~35B total parameters, **~3B
  active per token** (`A3B`). vLLM loads *all* experts into memory; the small
  active set only reduces per-token compute.
- Decode is memory-bandwidth bound on the GB10 (~273 GB/s shared). Reading only
  ~3B active params per token (≈1.5 GB at 4-bit) gives an **expected decode
  ceiling far above the dense 32B** (which reads ~18 GB/token) — the reason it is
  the fast fallback. *Confirm live.*

## How it runs in the fleet

Configured via the `FALLBACK_*` keys in the fleet `.env` (scaffolded by
`model init --fleet`); served by the `model-gear-vllm-fallback` container:

```dotenv
FALLBACK_MODEL=mmangkad/Qwen3.6-35B-A3B-NVFP4
FALLBACK_SERVED_NAME=mmangkad/Qwen3.6-35B-A3B-NVFP4
FALLBACK_MAX_MODEL_LEN=32768
FALLBACK_GPU_MEM_UTIL=0.35          # both models warm: keep primary+fallback well under 1.0 (dedicated box)
FALLBACK_TOOL_CALL_PARSER=qwen3_coder
FALLBACK_QUANTIZATION=modelopt_fp4
```

Address it through the gateway by name (or set `GATEWAY_ALIASES` for a short
alias):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -d '{"model":"mmangkad/Qwen3.6-35B-A3B-NVFP4","messages":[{"role":"user","content":"hi"}]}'
```

## Caveats to confirm on first load

1. **Tool-call format.** Qwen3.6 emits the Qwen3-Coder **XML** function format, so
   the backend is served with `--tool-call-parser=qwen3_coder` (not the `hermes`
   parser the dense Qwen3-32B uses). `model_gear.runtime._parser.infer_parser`
   already maps `qwen3.6` → `qwen3_coder`. Verify a `tool_choice:"auto"` probe
   returns a `finish` tool call.
2. **Quantization format.** The fleet defaults `FALLBACK_QUANTIZATION=modelopt_fp4`
   (as for the `nvidia/` checkpoints). This community (`mmangkad`) checkpoint may
   instead be a compressed-tensors NVFP4 — if vLLM rejects `modelopt_fp4`, drop or
   change `FALLBACK_QUANTIZATION`.
3. **`--trust-remote-code`.** The fleet compose omits it (as the single-model
   template does). If this checkpoint ships custom modeling code, vLLM will say so
   on load; add it back deliberately (it lets repo code run in-container alongside
   `HF_TOKEN` and the mounted cache).
4. **Architecture support.** Confirm the engine registers the checkpoint's
   architecture, as done for the 27B sibling:
   `docker exec model-gear-vllm-fallback python3 -c "from
   vllm.model_executor.models.registry import ModelRegistry;
   print(ModelRegistry.get_supported_archs())"`.

## Benchmark — blocked (model would not load), 2026-05-30

A live run was attempted (`model fleet up --apply` on `spark-f8a9`, then
`model benchmark --model mmangkad/Qwen3.6-35B-A3B-NVFP4`). The model never reached
`/health`, so no numbers exist yet:

| Property | Value |
|---|---|
| Health / `max_model_len` | **never healthy** — crash-looped co-resident; stalled at safetensors 0 % solo |
| Weights on disk | 24 GiB (single `model.safetensors`; `Qwen3_5MoeForConditionalGeneration`) |
| Decode throughput | *blocked* — `model benchmark` returned HTTP 502 (backend not up) |
| Prefill / correctness / tool calling | *blocked* |
| Co-resident with 27B (util 0.55/0.30, then 0.40/0.35) | **OOM** — `CUDA error: out of memory` on engine init |
| Solo (util 0.30, 65 GiB free) | crashed/restarted, then stalled loading the 24 GiB shard with GPU idle |

Next: re-test on a **dedicated/quiet** GB10 (stop other GPU services first), and
isolate whether the failure is co-residence pressure or the checkpoint's own
load path. Consider `--enforce-eager` (skip CUDA-graph capture) and disabling
`--enable-prefix-caching` to shrink the warmup footprint on the first load.

## Reference serve recipe + benchmark (shahizat, dedicated boxes)

shahizat benchmarked this model — the **`nvidia/Qwen3.6-35B-A3B-NVFP4`** checkpoint
(a different repo from the catalogued `mmangkad/` copy above) — on dedicated DGX
Spark, Jetson Thor, and Blackwell 6000 Pro boxes, where it **did** load and serve:
[NVIDIA Developer Forums, 2026-05-31](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810).
This is the serve recipe to try when re-testing on a quiet box. The two
**MoE-only** flags (`--moe-backend=marlin` and the MTP `--speculative-config`) are
what make the MoE perform — they are recorded as catalog data
([`model_gear/catalog.py`](../model_gear/catalog.py)) and printed by
`model switch mmangkad/Qwen3.6-35B-A3B-NVFP4`, but are **not** in the default
single-model template (they break the dense/hybrid models, and compose can't
conditionally omit a flag). Add them to the compose `command` by hand:

```bash
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --port 8000 --tensor-parallel-size 1 --trust-remote-code --dtype auto \
  --quantization modelopt --kv-cache-dtype fp8 \
  --attention-backend flashinfer --moe-backend marlin \
  --gpu-memory-utilization 0.85 --max-model-len 65536 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
```

Output-token throughput across the three workloads (16 concurrent requests):

| workload | Blackwell 6000 Pro | DGX Spark | Jetson Thor |
|---|---|---|---|
| prompt-heavy (8K/1K) | 343.8 tok/s | 171.6 tok/s | 124.2 tok/s |
| decode-heavy (1K/8K) | 1052.7 tok/s | 268.2 tok/s | 239.1 tok/s |
| balanced (1K/1K) | 817.5 tok/s | 249.5 tok/s | 190.7 tok/s |

MTP speculative-decode acceptance was highest on the decode-heavy workload
(~80–84 %), lowest on balanced (~57–59 %). These are shahizat's numbers on
dedicated boxes (the `nvidia/` checkpoint, concurrency 16, **with** MTP) — see
[`tuning-profiles.md`](tuning-profiles.md) for how the `--purpose` knob maps to
these shapes.

## Live replication on this GB10 (2026-05-31)

We did not trust the posted numbers — we measured. On the shared DGX Spark
`spark-f8a9` (single GB10, 121.7 GiB unified, shared with the audio NIMs + reachy;
vLLM 0.19.0+nv26.04), with the 27B stopped and the recipe above **minus MTP** at
util 0.70 / 32768:

| Metric (single-stream, batch=1) | 35B MoE (no MTP) | 27B hybrid (primary) |
|---|---|---|
| decode throughput | **35.0 / 36.1 tok/s** | 7.8 / 7.9 tok/s |
| prefill (845 tok + 16 gen) | **0.62 s** | 2.33 s |

So the 35B MoE is **~4.6× faster on single-stream decode and ~3.8× faster on
prefill** than the 27B on the same box — the MoE's ~3B-active-params advantage,
reproduced. (`vllm bench serve` at concurrency 1 agrees: 34.7 tok/s, TTFT 0.70 s,
TPOT 28 ms.) We could **not** reproduce shahizat's exact figures — he ran the
`nvidia/` checkpoint on *dedicated* boxes at concurrency 16 **with** MTP (which
roughly doubles per-stream decode); our run is the `mmangkad/` copy on a *shared*
box, single-stream, **without** MTP (it does not load here). The qualitative
result — MoE = much faster decode — replicates; the headline tok/s does not, and
the gap is explained by box, concurrency, and the missing MTP draft.

## Why we serve the `mmangkad/` copy, not `nvidia/` (vLLM version, 2026-05-31)

shahizat used `nvidia/Qwen3.6-35B-A3B-NVFP4`. We tried to switch to it (and to a
newer vLLM) to get MTP working — and hit a hard wall on the GB10:

- **The `nvidia/` checkpoint will not load on the NGC image's vLLM 0.19.0.** Its
  NVFP4-MoE experts fail every backend: `marlin` / `flashinfer_trtllm` → "not
  supported for unquantized MoE"; `triton` / auto → `KeyError:
  layers.0.mlp.experts.w2_input_scale`. Both `--quantization modelopt` and
  `modelopt_fp4` behave the same.
- **A newer vLLM *does* run on the GB10.** A derived image with
  `pip install vllm==0.21.0` pulls upstream torch 2.11.0 + CUDA-13 wheels
  (aarch64 wheels exist); torch 2.11.0 works on the GB10 (`device_capability
  (12,1)`; `sm_121` is forward-compatible with its `sm_120` kernels — a GPU
  matmul ran). On 0.21.0 the quant is now **recognized** (`modelopt_mixed`), but
  the MoE expert loader still fails the same way (`marlin` → "unquantized";
  `triton`/auto → missing `w2_input_scale`).
- **0.22.0 / nightly are not pip-installable here** (aarch64): a
  `nvidia-cutlass-dsl[cu13]` dependency conflict with no matching distribution.

Net: the `nvidia/` checkpoint's MoE export needs a vLLM build with NVFP4-MoE
expert support that isn't installable on this Grace/Blackwell (aarch64) box yet.
shahizat's dedicated boxes were almost certainly x86, where a suitable vLLM
installs cleanly. **The working NVFP4 MoE on the GB10 remains the `mmangkad/`
copy** (loads on the stock NGC `26.04-py3` image with `--moe-backend marlin`,
~35 tok/s single-stream — above). Revisit `nvidia/` + MTP when a vLLM with the
right loader ships for aarch64 (a newer NGC image, or upstream ≥0.22 gaining
aarch64 wheels). The image stays **NGC `26.04-py3`** (latest tag; vLLM
0.19.0 + torch 2.12.0a0.nv26.04 + CUDA 13.2, all Blackwell-patched).

> **The 27B took the other route.** Rather than wait for a newer engine, the 27B
> gets MTP from a checkpoint that *ships the MTP draft weights*
> (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`,
> [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md), issue #26). The
> same strategy could unblock MTP here — a 35B re-export with the draft head
> grafted back, loadable on the stock `0.19.0` image — without the `nvidia/`
> NVFP4-MoE loader. Re-testing `nvidia/Qwen3.6-35B-A3B-NVFP4` + MTP is tracked as a
> follow-up.
