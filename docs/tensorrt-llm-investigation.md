# Investigation: TensorRT-LLM as an alternative engine — 2026-06-26

> **Status: desk investigation, no live run.** This records a documentation +
> web-evidence study of whether [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
> (`trtllm-serve`) could serve lobes' primary model
> (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`) on the DGX Spark (GB10 / SM121) as
> an alternative to vLLM. **Nothing was deployed or benchmarked on hardware** —
> the TRT-LLM columns below are *unmeasured*. The verdict is a go/no-go on
> *whether to spend the bring-up effort*, not a measured comparison.

**Verdict: not yet — stay on vLLM.** TRT-LLM cannot serve *this* model with its
defining feature (MTP speculative decoding) on a stable release today. Revisit
when TRT-LLM **1.3.0 ships stable** with Qwen3-family MTP + GDN kernels (both are
in 1.3.0 *release-candidate* builds as of this writing). See
[Decision](#decision--revisit-trigger).

## What we asked

Can we replace vLLM with TensorRT-LLM for the MTP 27B primary and adopt TRT-LLM
as a *switchable engine* in lobes — and is it worth it? The motivation is the
usual one: NVIDIA's first-party engine sometimes wins on decode throughput. The
question is whether that win is reachable for *this* checkpoint (NVFP4 + restored
MTP draft head + hybrid Mamba/GDN attention) on *this* box (GB10/SM121, aarch64,
beta-supported).

## TL;DR — feasibility by dimension

| Dimension | TRT-LLM status | Confidence | Notes |
|---|---|---|---|
| OpenAI-compatible serving (`trtllm-serve`) | ✅ supported | HIGH | `/v1/chat/completions`, `/v1/completions`, `/health`, `/v1/models`, `/metrics` — drops in behind the lobes gateway unchanged |
| NVFP4 inference, pre-quantized HF checkpoint | ✅ beta | MEDIUM | `--backend pytorch` loads ModelOpt/HF NVFP4 directly, no `trtllm-build`; validated on DGX Spark in 1.2 beta |
| DGX Spark / GB10 / SM121 / aarch64 | ⚠️ beta + bugs | MEDIUM | 1.2 "beta support for single-node DGX Spark"; 1.2.1 ships an arm64 NGC container; several SM121 bugs closed-fixed only in RCs |
| 256K context on 128 GB unified | ✅ feasible | MEDIUM | paged KV cache, no hard ceiling; set `--max_seq_len 262144`; memory-bound at high batch |
| **MTP speculative decoding (Qwen3)** | ❌ **not in stable** | LOW | **the blocker** — MTP is DeepSeek-only in stable; Qwen3.5/3.6 MTP only in 1.3.0 RC builds |
| **Hybrid GDN / DeltaNet (Qwen3.6-27B)** | ⚠️ **RC-only, unvalidated** | LOW–MED | GDN FlashInfer kernels for Qwen3.x land in 1.3.0 RCs; Qwen3.6-27B (48 GDN + 16 attn) not in the stable support matrix |

## 1. Where a TRT-LLM engine would plug into lobes

The good news from the codebase side: **two of the three integration surfaces
are already engine-agnostic.**

- **Gateway** (`lobes/gateway/`) — the `Backend` dataclass keys on a `base_url`
  and forwards POSTs verbatim (`/v1/chat/completions`, `/v1/embeddings`,
  `/v1/score`, `/v1/rerank`). It has no vLLM-specific code. Any OpenAI-speaking
  HTTP server on the backend URL works. **No gateway change needed.**
- **Assessment** (`lobes/assess.py`, `lobes assess` / `lobes benchmark`) — speaks
  pure OpenAI HTTP: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`.
  A TRT-LLM backend that serves those passes the harness as-is, so the *same*
  decode-throughput / prefill / correctness-probe numbers are directly
  comparable. (MTP draft acceptance is read from `GET /metrics` — vLLM exposes
  `vllm:spec_decode_*`; TRT-LLM's `/metrics` surface differs, so that one metric
  would need a new reader.)

The bad news: **everything else hardcodes vLLM.** A real engine swap touches:

| Surface | What's vLLM-specific |
|---|---|
| `lobes/templates/docker-compose.yml`, `fleet/docker-compose.yml` | `image: nvcr.io/nvidia/vllm:26.04-py3`, `command: vllm serve …`, all `--speculative-config` / `--quantization` / `--attention-backend` / `--kv-cache-dtype` / `--tokenizer` / `--reasoning-parser` flags, container name `model-gear-vllm` |
| `lobes/templates/env.example` (+ fleet) | every `VLLM_*` / `PRIMARY_*` env var name |
| `lobes/catalog.py` | `SupportedModel` has **no `engine` field**; `quantization` / `tool_parser` / `speculative_config` / `moe_backend` / `hf_overrides` map 1:1 to vLLM CLI flags |
| `lobes/cli/_commands/switch.py` | writes `VLLM_*` keys; parser/quantization/MTP-notice helpers all know vLLM flag names |
| `lobes/runtime/_compose.py` | `CONTAINER = "model-gear-vllm"`, fleet container names |
| `lobes/runtime/_health.py` | polls `/health` (fine for vLLM *and* `trtllm-serve` — both expose it) |
| `lobes/profiles.py` | `attention_backend` stores `flashinfer`, a vLLM backend name |

So an honest "switchable engine" is a real feature: it needs an `engine` field on
`SupportedModel`, a second compose template, an engine selector in `switch`, and
engine-scoped env keys. The seam exists, but it is not free. For a *spike* none of
that is required — you stand up a second one-off compose file by hand and point a
`base_url` at it.

## 2. Can TRT-LLM serve *this* model? — dimension detail

### NVFP4 (MEDIUM confidence — beta)

TRT-LLM's PyTorch backend loads pre-quantized NVFP4 checkpoints directly
(`trtllm-serve <model> --backend pytorch`), no engine compile step. NVFP4 support
is tiered by arch: full on SM100 (B200), and **1.2 explicitly added beta DGX
Spark support validating "FP16/FP8/NVFP4"** on GB10 (SM121).[^trt-releases] We
already have the NVFP4 checkpoint, which matters because **on-device ModelOpt
quantization is broken on GB10** — `quantize.py` mis-reads free memory via NVML
and spuriously offloads to CPU (issue #15021, closed *not planned*; workaround is
the Python API with `device_map="cuda:0"`).[^trt-15021] We don't need it; the
checkpoint is pre-quantized.

### MTP speculative decoding — **the blocker** (LOW confidence)

This is the feature the lobes primary exists *for*: the MTP draft head buys
**~2.4× single-stream decode** under vLLM (8 → ~19 tok/s).[^lobes-mtp] In TRT-LLM:

- MTP is a first-class spec-decode method (added 0.19.0) — **but the
  speculative-decoding docs state plainly: "MTP is currently only supported by
  DeepSeek."**[^trt-specdec] Qwen3-family MTP is not in a stable release.
- "Qwen3.5 MTP support" and "Enable MTP for Step-3.7 NVFP4" appear only in
  **1.3.0 release-candidate** notes.[^trt-releases]
- Even once it ships: **NVFP4 ModelOpt exports strip the `mtp.*` weights** during
  quantization. Working vLLM deployments restore the MTP head from the BF16
  checkpoint at load — the same restore would be needed under TRT-LLM.[^spark-mtp]

Net: on a stable TRT-LLM today you would serve this model **without MTP**, i.e.
forfeit the entire reason this checkpoint was built — landing back near the
archived baseline's ~8 tok/s, not ~19. (Eagle3 is TRT-LLM's documented Qwen3
spec-decode path, but that's a *different* draft mechanism and would not use the
checkpoint's restored MTP head.)[^trt-qwen]

### Hybrid GDN / DeltaNet — secondary blocker (LOW–MEDIUM)

Qwen3.6-27B is dense with 64 layers in 16 groups of `(3× Gated DeltaNet → 1×
Gated Attention)` = 48 linear-attention + 16 standard-attention layers. TRT-LLM
is actively merging this: Mamba-hybrid support (0.19.0), GDN sharding fixes
(#11567, closed-fixed), `Qwen3HybridConfig` derivation fixes (PRs #13832/#14410),
and **"Enable FlashInfer GDN decoding kernel for Qwen3.5"** — but the last is a
**1.3.0 RC** item.[^trt-releases][^trt-11567] The stable support matrix lists
`Qwen3ForCausalLM` / `Qwen3MoeForCausalLM` only; **Qwen3.6-27B / GDN is not
listed**, and the Qwen3-Next quick-start covers only the *80B MoE* variant, not
the dense hybrid.[^trt-matrix][^trt-next] This is the second thing that has to
land in 1.3.0 stable before a serious attempt.

### `trtllm-serve` OpenAI surface (HIGH)

Confirmed OpenAI-compatible. Minimal checkpoint-free invocation:[^trt-serve]

```bash
trtllm-serve sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
  --backend pytorch \
  --host 0.0.0.0 --port 8000 \
  --tp_size 1 --max_seq_len 262144 \
  --trust_remote_code
```

That endpoint set (`/v1/chat/completions`, `/v1/completions`, `/health`,
`/v1/models`) is exactly what the lobes gateway and `lobes assess` expect.

### GB10 support + 256K (MEDIUM)

1.2 ships beta DGX Spark support; 1.2.1 (2026-04-20) publishes a multi-arch
container including arm64 on NGC (`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`).
Known SM121/aarch64 bugs exist and several are closed-fixed *in RCs* (e.g.
Qwen3-Next FP8/NVFP4 Triton "illegal instruction" #12230; `NCCL_SYMMETRIC`
crash).[^trt-releases][^trt-12230] 256K context is architecturally fine — paged
KV cache, no hard ceiling, `--max_seq_len 262144`; with ~14 GB NVFP4 weights the
remaining unified memory leaves ample KV headroom for single/small-batch
decode.[^trt-mem]

## 3. Blockers, ranked

1. **MTP is DeepSeek-only in stable TRT-LLM.** Serving this model on TRT-LLM
   today means no MTP → losing the ~2.4× decode win that is the model's whole
   point. (1.3.0 RC only; plus `mtp.*` weight-restore from BF16.)
2. **Hybrid GDN/DeltaNet for Qwen3.6-27B isn't validated in stable.** Kernels are
   in 1.3.0 RCs; the dense 27B hybrid isn't in the support matrix or any
   public end-to-end report.
3. **GB10 is beta with a tail of SM121/aarch64 bugs**, several fixed only in RCs —
   so the *practical* path forces RC builds, compounding (1) and (2).

Note all three blockers converge on the same release: **TRT-LLM 1.3.0 stable.**

## 4. Comparison vs the recorded vLLM baseline

vLLM numbers are from [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md)
(DGX Spark GB10, vLLM `0.19.0+nv26.04`). The TRT-LLM column is **unmeasured** —
this investigation ran no hardware.

| Metric | vLLM (measured) | TRT-LLM (this study) |
|---|---|---|
| Engine / image | vLLM `0.19.0+nv26.04`, `nvcr.io/nvidia/vllm:26.04-py3` | `trtllm-serve` 1.2.x stable / 1.3.0 RC; `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` |
| OpenAI endpoints | ✅ behind gateway | ✅ behind gateway (would drop in) |
| NVFP4 on GB10 | ✅ serving today | ✅ beta (pre-quantized checkpoint) |
| Decode, batch=1 (256K) | **17.8 tok/s** | unmeasured; **expected ~baseline (~8) without MTP** |
| Decode, batch=1 (fresh/128K) | 19.1 / 18.3 tok/s | unmeasured |
| MTP draft acceptance | **72–79 %** (~2.2 of 3 tokens/step) | **N/A — MTP unsupported (stable)** |
| Prefill | ~2,800 tok/s | unmeasured |
| GPU memory reserved (util 0.6) | ~72 GB | unmeasured |
| Correctness probes (17×23=391; 145 min) | ✅ both pass w/ reasoning trace | unmeasured |

The decisive cell is **MTP draft acceptance**: vLLM has it, stable TRT-LLM does
not. Without it the comparison isn't TRT-LLM-vs-vLLM on equal footing — it's
TRT-LLM-without-the-feature vs vLLM-with-it, which vLLM wins by construction.

## 5. The minimal spike, if we run it anyway

If we want *a* number before 1.3.0 stable, the cheapest honest experiment — no
lobes engine abstraction, no production wiring:

1. Pull `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` on the Spark; one throwaway
   compose service on an alternate port (e.g. 8002), `base_url` only.
2. `trtllm-serve … --backend pytorch --max_seq_len 262144 --trust_remote_code`
   (see §2). Expect to **disable MTP** and possibly fall back to a non-GDN-aware
   path — i.e. measuring the *floor*, not the ceiling.
3. Run `lobes assess` + `lobes benchmark` against the TRT-LLM port (the harness is
   engine-agnostic) → fill the TRT-LLM column of §4. Read MTP acceptance from
   vLLM's `/metrics` only; TRT-LLM won't report it (no MTP).
4. **Expected outcome:** decode near the ~8 tok/s baseline (no draft head),
   confirming there's no win *until* MTP + GDN land in 1.3.0 stable. The value of
   running it is a recorded floor and a working bring-up recipe to re-run on 1.3.0.

This is a half-day spike with a near-certain negative result — worth it only as a
dated baseline / smoke-test of the GB10 container, not as a decision input. The
decision input is the 1.3.0 stable release.

## Decision / revisit trigger

**Do not adopt TRT-LLM now. Stay on vLLM.** The engine-agnostic gateway +
assessment harness mean re-evaluation is cheap later, so there's no architectural
debt in waiting.

**Revisit when all three hold:**

- TRT-LLM **1.3.0 ships stable** (not RC), and its notes confirm **Qwen3-family
  MTP** + **GDN decoding kernels**.
- A public report (NVIDIA or community) shows **Qwen3.6-class hybrid** running
  end-to-end on **GB10/SM121**.
- We're willing to handle **MTP `mtp.*` weight restore** from the BF16 checkpoint
  under TRT-LLM (same dance vLLM already does).

At that point: run the §5 spike *with* MTP enabled, fill the §4 table for real,
and only then weigh the real engine abstraction in `catalog.py` / `switch.py` /
templates (§1).

## Sources

Web evidence gathered 2026-06-26 (TRT-LLM stable 1.2.1 / RC 1.3.0rc19). Codebase
findings are from the lobes-cli tree at the same date.

[^trt-releases]: TRT-LLM releases & release notes (1.2 DGX Spark beta, 1.2.1 arm64 container, 1.3.0 RC MTP/GDN items) — <https://github.com/NVIDIA/TensorRT-LLM/releases>, <https://nvidia.github.io/TensorRT-LLM/0.19.0/release-notes.html>
[^trt-specdec]: TRT-LLM speculative-decoding docs — "MTP is currently only supported by DeepSeek" — <https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/features/speculative-decoding.html>
[^trt-matrix]: TRT-LLM support matrix (lists `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`) — <https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html>
[^trt-next]: TRT-LLM Qwen3-Next quick-start (80B MoE only) — <https://nvidia.github.io/TensorRT-LLM/deployment-guide/quick-start-recipe-for-qwen3-next-on-trtllm.html>
[^trt-serve]: `trtllm-serve` command docs (OpenAI-compatible server) — <https://nvidia.github.io/TensorRT-LLM/1.0.0rc2/commands/trtllm-serve.html>
[^trt-15021]: Issue #15021 — `quantize.py` device_map broken on GB10 (closed, not planned) — <https://github.com/NVIDIA/TensorRT-LLM/issues/15021>
[^trt-11567]: Issue #11567 — GDN manual sharding config (closed/fixed) — <https://github.com/NVIDIA/TensorRT-LLM/issues/11567>
[^trt-12230]: Issue #12230 — Qwen3-Next FP8/NVFP4 illegal instruction on DGX Spark/SM121 — <https://github.com/NVIDIA/TensorRT-LLM/issues/12230>
[^trt-mem]: TRT-LLM memory-usage docs (paged KV cache scaling) — <https://nvidia.github.io/TensorRT-LLM/reference/memory.html>
[^trt-qwen]: TRT-LLM Qwen examples README (Eagle3 spec-decode for Qwen3) — <https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md>
[^spark-mtp]: Community: Qwen3.5 NVFP4 + MTP on DGX Spark via vLLM (NVFP4 strips `mtp.*`; restore from BF16; ~24.5 tok/s) — <https://github.com/bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4>
[^lobes-mtp]: lobes baseline numbers — [`docs/qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) (vLLM `0.19.0+nv26.04`, 17.8–19.1 tok/s, 72–79 % MTP draft acceptance, DGX Spark GB10).
