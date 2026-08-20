# Repro: Nemotron 3.5 Lightning wedges at Mamba2 SSD kernel warmup on Jetson AGX Thor

Minimal, self-contained reproduction for the Jetson AI Lab documentation
maintainers: the published Thor serve recipe for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
(<https://www.jetson-ai-lab.com/models/nemotron3-5-lightning/#run-on-jetson>,
upstream `vllm/vllm-openai:v0.27.1`) does not come up on our Jetson AGX Thor:
the engine wedges **indefinitely** at

```text
(EngineCore) INFO [mamba_mixer2.py:596] Warming up Mamba2 SSD Triton kernels...
```

and `/health` never serves. We would love for this to work — the same
checkpoint serves beautifully on a DGX Spark GB10 next to this box
(75 tok/s single-stream, ~75 ms TTFT), and Lightning is exactly the worker
model we want on the Thor.

## Environment (the box that fails)

| item | value |
|---|---|
| Device | Jetson AGX Thor devkit, 128 GB unified (sm_110, aarch64) |
| L4T / JetPack | R38.2.2 (GCID 42205042, 2025-09-25) / JetPack 7.0-b128 |
| Kernel | 6.8.12-tegra |
| NVIDIA driver | 580.00 |
| Docker | 29.0.0, `--runtime=nvidia` (nvidia-container-toolkit) |
| Image | `vllm/vllm-openai:v0.27.1` (upstream, multi-arch pull on this box) |
| Model cache | pre-downloaded via `hf download` (no HF throttling in play) |

## Exact command (the published recipe, verbatim*)

See [`run.sh`](run.sh). It is the documented command with two
repro-hygiene changes only: detached instead of `-it --rm` (to keep logs),
and `--port 18002` (8000 is occupied on this host). All model/engine flags —
including the full Mamba cache set and DSpark speculative config — are
byte-identical to the published recipe.

## Observed behavior (2026-08-20, four independent attempts)

| # | image | flags | outcome |
|---|---|---|---|
| 1 | `vllm-openai` nightly digest `8bd082…` (0.26.1rc1.dev942) | conservative: 32K, util 0.25, no MTP, `nemotron_v3` + `qwen3_coder` | weights load, torch.compile completes, then **wedge** at Mamba2 SSD warmup; killed after 25 min idle |
| 2 | `v0.27.1` | same conservative set | identical wedge, killed after 25 min |
| 3 | `v0.27.1` | + `--mamba-backend flashinfer` | identical wedge, killed after 15 min |
| 4 | `v0.27.1` | **the full published recipe** (`run.sh`: fp8 KV, util 0.7, 128K, prefix caching, DSpark n=5, flashinfer backend, `--mamba-ssm-cache-dtype float16`, stochastic rounding, philox 5, cache mode align) | ~2 min of real compile activity (~80 % CPU — visibly different from 1–3), then CPU falls to ~0.4 % and the same warmup line sits **idle 15+ min**; killed |

In every attempt the container process stays alive and healthy-looking —
`docker exec` works, no error is ever logged — the warmup simply never
completes. Full log of attempt 4: [`logs/jetson-recipe-full.log`](logs/jetson-recipe-full.log)
(the last line is the wedge; nothing follows it).

## Context that may help localize it

Same-day results on the SAME box and images suggest the gap is specific to
non-dense decode/warmup paths on sm_110:

- `Qwen3.6-35B-A3B` (GDN-hybrid MoE): decode hard-fails with
  `fused_gdn_decode_post_conv_mtp … no kernel image is available for
  execution on the device` (csrc `libtorch_stable/gdn/fused_gdn_decode_kernel.cu:412`)
  — but the **non-MTP** GDN path works (we serve `Qwen3.8-27B` at a 1M YaRN
  window on this box with MTP disabled).
- `LiquidAI/LFM2.5-1.2B` (gated-conv hybrid): boots, then produces corrupted
  deterministic output and dies with `CUDA error: unspecified launch failure`
  within 1–3 requests — while the identical config passes on the Spark GB10.
- Dense-transformer serving on the same images is fine on this box.
- `torch.cuda.get_arch_list()` in both images includes `sm_110`, and a live
  matmul on the device succeeds — torch-level SASS is present; the issue is
  in vLLM/Triton/FlashInfer kernel-level coverage or the warmup's autotune.

## Ask

Does the recipe's validation cover Jetson AGX Thor on JetPack 7.0/R38.2.2
with upstream `v0.27.1`, and if so, what differs from the environment above
(JetPack/driver revision, env vars, image digest)? Happy to run any
diagnostic build or patched wheel on this box and report back — transcripts
of everything above are in this repo under `docs/evidence/` (see
`2026-08-20-spike-lightning-thor-no-go.txt`).

Contact: Ori Nachum (repo owner) — agentculture/lobes-cli.
