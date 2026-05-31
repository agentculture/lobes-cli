# MoE candidate: `mmangkad/Qwen3.6-35B-A3B-NVFP4`

A **MoE candidate** — the *former* fleet fallback. It has been **replaced as the
default fallback** by the dense `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`
([`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md)), because
this checkpoint never loaded on the GB10 (see the status note below). It remains
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
dedicated boxes — see [`tuning-profiles.md`](tuning-profiles.md) for how the
`--purpose` knob maps to these shapes. model-gear's own GB10 numbers for the
`mmangkad/` copy remain blocked on a reliable load (above).
