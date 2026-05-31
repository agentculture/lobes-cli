# Tuning profiles — purpose × machine × model

model-gear resolves the vLLM serve config from **three layers**, so the same
catalog model can be served in different *gears* without editing the compose
file. `model switch <model> --purpose <p> --machine <m>` writes the resolved
`VLLM_*` values to the deployment `.env`; `docker compose` substitutes them at
`up`.

Resolution order (highest precedence last):

1. **machine profile** (`--machine`, default `auto`) — memory/arch defaults
2. **workload profile** (`--purpose`, default `balanced`) — batching + bench shape
3. **model** (the [catalog](../model_gear/catalog.py)) — quantization, tool
   parser, and MoE-only serve extras
4. **explicit CLI flags** (`--max-model-len`, `--gpu-mem-util`, …) — always win

The tables live in [`model_gear/profiles.py`](../model_gear/profiles.py) (a pure
data module, guarded by `tests/test_profiles.py`). See `model explain tuning`.

## Workload profiles (`--purpose`)

The `purpose` is the request mix the deployment is tuned for. The default is
`balanced`. `model benchmark` defaults its `(input, output)` shape to the
configured `VLLM_PURPOSE`, so the measured numbers track the serve config.

| purpose | `--max-num-seqs` | `--max-num-batched-tokens` | benchmark shape |
|---|---|---|---|
| **balanced** (default) | 4 | 8192 | ≈1K in / 1K out |
| prompt-heavy | 4 | 16384 | ≈8K in / 1K out |
| decode-heavy | 8 | 4096 | ≈1K in / 8K out |

The `(input, output)` shapes and the `balanced` batching values come straight
from shahizat's report (below); the prompt-heavy / decode-heavy batching knobs
are **configured heuristics** — a sensible starting point, not a measured
optimum. Confirm them with `model benchmark --purpose <p>`.

## Machine profiles (`--machine`)

Auto-detected by default from `nvidia-smi -L` + the hostname (`GB10` → `spark`,
`Thor` → `thor`, `RTX PRO 6000` → `blackwell`, else `generic`). Override with an
explicit `--machine`.

| machine | `--gpu-memory-utilization` | `--max-model-len` | `--attention-backend` | status |
|---|---|---|---|---|
| **spark** (GB10, 128 GB unified, usually shared) | 0.6 | 32768 | flashinfer | load-tested |
| thor (Jetson Thor, unified) | 0.6 | 32768 | flashinfer | configured |
| blackwell (RTX PRO 6000, dedicated VRAM) | 0.85 | 65536 | flashinfer | configured |
| generic (unknown Blackwell-class) | 0.6 | 32768 | flashinfer | configured |

`spark` deliberately keeps the conservative `0.6` GPU-memory fraction: the GB10
is shared with other mesh agents, and co-residence OOMs at higher fractions (see
[`gateway-fleet.md`](gateway-fleet.md)). A *dedicated* single-model box can raise
it toward `0.85` with `model switch --gpu-mem-util 0.85`.

## Model layer (per-model serve extras)

Quantization and the tool-call parser are per-model (from the catalog).
**MoE-only** flags — `--moe-backend` and the MTP `--speculative-config` — apply
to `Qwen3.6-35B-A3B` (MoE) alone and would break the dense/hybrid models, so they
are **not** in the default template. `model switch` to the MoE prints them for a
manual compose edit; see [`qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md).

## Reference benchmark (shahizat)

The throughput flags model-gear adopts (the flashinfer attention backend, chunked
prefill, async scheduling, `--max-num-seqs` / `--max-num-batched-tokens`, and —
for the MoE — the marlin backend + MTP speculative decode) follow **shahizat's**
cross-machine NVFP4 benchmark of `nvidia/Qwen3.6-35B-A3B-NVFP4`:
[NVIDIA Developer Forums, 2026-05-31](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810).

shahizat ran **one** serve config across all three machines and three workloads
(16 concurrent requests, `vllm bench serve`). Output-token throughput (tok/s):

| workload | Blackwell 6000 Pro | DGX Spark | Jetson Thor |
|---|---|---|---|
| prompt-heavy (8K/1K) | 343.8 | 171.6 | 124.2 |
| decode-heavy (1K/8K) | 1052.7 | 268.2 | 239.1 |
| balanced (1K/1K) | 817.5 | 249.5 | 190.7 |

These are a cross-machine **reference baseline** (dedicated boxes, the `nvidia/`
checkpoint, concurrency 16, with MTP) — **not** model-gear's own measurements.

## Our own replication (shared DGX Spark, 2026-05-31)

We did not trust the post — we measured on `spark-f8a9` (single GB10, 121.7 GiB
unified, shared with the audio NIMs + reachy; vLLM 0.19.0+nv26.04), with the new
config. Single-stream decode (batch=1, identical probe — 1000 in / 512 out):

| model (new config) | decode tok/s | prefill (845 tok + 16 gen) |
|---|---|---|
| 35B MoE candidate (marlin, no MTP, util 0.70) | **35.0 / 36.1** | **0.62 s** |
| 27B hybrid primary (util 0.6) | 7.8 / 7.9 | 2.33 s |

The 35B MoE is **~4.6× faster on single-stream decode** than the 27B on the same
box (the MoE's ~3B-active advantage). The 27B itself gains a little from the new
flags (~7.1 → ~7.8 tok/s, prefill 2.51 → 2.33 s — a shippable bonus; decode is
memory-bandwidth bound, so the batching/scheduling flags help it less).

Findings that drove config choices:

- **util 0.85 fails the pre-flight on this shared box** — only ~90 of 121.7 GiB is
  free (other services hold the rest), so the `spark` profile's conservative `0.6`
  is correct; the 35B needed `0.70` solo (with the 27B stopped).
- **shahizat's MTP `--speculative-config` does not transfer** to the cached
  `mmangkad/` checkpoint (`qwen3_5_mtp.py` weight-shape mismatch on vLLM nv26.04) —
  so the catalog carries only `--moe-backend=marlin` for the MoE, not MTP. See
  [`qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md).
- The GB10 reports **no native FP4 compute** → vLLM falls back to the Marlin
  weight-only FP4 kernel (a perf caveat on this image).

We could not reproduce shahizat's exact tok/s (his were dedicated boxes,
concurrency 16, `nvidia/` checkpoint, with MTP), but the qualitative result —
MoE = much faster decode — replicates. Re-run with `model benchmark --purpose <p>`
to refresh these on the live server.
