# Gemma 4 31B-NVFP4 — the "muse" role (creative/ideation lobe)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **Status: DECLARED — memory budget measured live, acceptance pending.** Most
> of this doc is read from the checkpoint's own published config and declared
> as data in the repo (catalog entry, `thor-muse` shape, compose template).
> The **memory budget is measured** — a physical Jetson AGX Thor booted the
> gear 2026-07-17 (util 0.55 at the full 262144 window; see
> ["How to host it"](#how-to-host-it-the-thor-muse-shape) below) — but
> throughput, MTP acceptance, and the correctness probes remain unmeasured,
> and per the #108 rule nothing here claims *validated* until the acceptance
> run (`scripts/accept-shape.sh`) passes and its transcript lands under
> `docs/evidence/` (the measured-truth lesson from `spark-lobe`/`thor-lobe`).

**Model id:** `nvidia/Gemma-4-31B-IT-NVFP4`
**Tier alias:** `muse` — the role name *is* the alias (capability order:
`minor` < `multimodal` < `muse` < `primary`/`main`).
**Role:** `muse` — the fleet's **creative/ideation lobe**, the SEVENTH
first-class Colleague role. **Opt-in for hosting**: `machine-as-brain` never
hosts it; only an explicit muse-hosting shape (`thor-muse`) does.
**Status:** `configured` (declared 2026-07-17; the first live boot 2026-07-17
measured the memory budget — acceptance run pending, not yet validated).

## What it is

Gemma 4 31B IT (Google DeepMind), in **NVIDIA's official ModelOpt NVFP4
export** — unlike the community `coolthor` 12B export behind `senses`, this
one comes from nvidia/ and quantizes with `modelopt`, not
`compressed-tensors`. It backs the `muse` role: creative generation, long-form
writing, ideation, style variation, and a divergent second opinion — a
*different mind* next to `cortex`, never a replacement for it (muse proposes,
cortex decides).

Checkpoint facts (read from the published config, 2026-07-17):

- **30.4 GiB of NVFP4 weights** across 4 safetensors shards.
- **256K native context** (`text_config.max_position_embeddings = 262144`).
- **`modelopt` quantization** — `config.json` declares
  `quant_method="modelopt"` (resolves to `modelopt_fp4`); passing
  `compressed-tensors` (the 12B's format) would be a quant-method mismatch.
  `hf_quant_config.json` declares NVFP4 weights **plus calibrated FP8
  KV-cache scales** — unlike the Qwen MTP re-export, which ships none (#109).
- **Plain-gemma4 line, not the Unified family** — `model_type: "gemma4"`,
  architecture `Gemma4ForConditionalGeneration`. This is *not* the
  `Gemma4UnifiedForConditionalGeneration` class the 12B `senses` gear needs
  from vLLM nightly. The checkpoint still declares `vision_config` **and**
  `audio_config` with image/audio token ids — multimodal intake like
  `senses` — but the **#101 assumption applies until measured**: on the 12B,
  this vLLM serving path silently drops `input_audio` content parts, and we
  assume the same gap here until a live probe says otherwise. Treat `muse` as
  text+vision intake at most; for speech use the purpose-built `stt` role.
- **Pythonic tool calls** (`--tool-call-parser pythonic`, like the 12B).
- **Native MTP DECLARED, not measured** — see next section.

## Speculative decoding — declared, unmeasured

The catalog entry carries a native-MTP `speculative_config`:

```json
{"method": "mtp", "model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 1}
```

The rationale is the same public-assistant-draft route the 12B base gear
already validated for its own family: Google publishes one assistant draft per
Gemma 4 size, and `google/gemma-4-31B-it-assistant` is the 31B's — in the
**`gemma4_assistant`** (plain-line) family, distinct from the 12B's
`gemma4_unified_assistant` (see the family table in
[`gemma4-mtp-draft.md`](gemma4-mtp-draft.md)). vLLM's `hf_config_override`
normalizes it to `gemma4_mtp` with forced `n_predict=1`.

**Nothing about this is measured on the 31B target.** The 12B family taught us
draft acceptance varies wildly by checkpoint (57.9% on the base it-model,
30.8% on the coder fine-tune — one worth wiring, one not), so the 31B's
acceptance rate and decode multiplier are unknown until the first acceptance
run measures them. The config is declared so that run has something to
measure; a poor result would demote it exactly as the coder's was.

## Serving

The `vllm-muse` compose service is parked behind the **`muse` Docker Compose
profile** in the base fleet template (like `vllm-minor`) — a plain
`docker compose up` never starts a 31B by accident. It builds the **same
custom image as `vllm-multimodal`** (`Dockerfile.vllm-gemma4`, the pinned vLLM
nightly + audio extras; `MUSE_IMAGE` overrides the tag).

Key env vars (from `env.example`; values mirror the `thor-muse` shape's
declaration):

| Variable | Value | Notes |
|---|---|---|
| `MUSE_MODEL` / `MUSE_SERVED_NAME` | `nvidia/Gemma-4-31B-IT-NVFP4` | HF checkpoint id / OpenAI `model` id |
| `MUSE_BASE_URL` | `http://vllm-muse:8000` | set by a muse-hosting shape render; wires the gateway backend |
| `MUSE_GPU_MEM_UTIL` | `0.55` | **measured** 2026-07-17 (Thor): 26.47 GiB KV pool, 611,415 tokens, 2.33x at 262144; the 0.40 hypothesis was refused live (0.6 GiB KV) |
| `MUSE_MAX_MODEL_LEN` | `262144` | the FULL 256K native window (operator decision — no box-budget trim) |
| `MUSE_QUANTIZATION` | `modelopt` | the NVIDIA export's own quant_method — NOT `compressed-tensors` |
| `MUSE_ATTENTION_BACKEND` | `TRITON_ATTN` | Gemma 4's heterogeneous per-layer head sizes — the same divergence the 12B `senses` gear carries on every card |

The gateway wires the `muse` backend only when `MUSE_BASE_URL` is set, and —
uniquely among the core roles — an **unwired muse defaults to infeasible**
(`OPT_IN_BACKENDS` in `lobes/gateway/_config.py`): on a pre-muse or stale
`.env`, `model=muse` gets an honest `404 role_infeasible` (referable and
proxyable via `MUSE_PEER_ORIGIN` / `MUSE_PEER_PROXY` / `MUSE_PEER_API_KEY`,
like every core role) instead of silently upward-falling-back to `cortex`.
An explicit `MUSE_FEASIBLE` always wins. See
[`gateway-fleet.md`](gateway-fleet.md#generate-lane-tier-aliases).

Under swap/iowait pressure a `muse` request is shed (`429` busy) exactly like
`cortex`/`senses` — `minor` remains the servable floor.

## How to host it: the thor-muse shape

`muse` is an **opt-in core role** (`lobes/profiles/shapes.py`'s
`OPT_IN_CORE_ROLES`): it carries the full per-machine Profile knob set, but no
card profile declares it — a 31B cannot co-reside with the default
`cortex`+`senses` duo on a 128 GB box, so the **shape** that hosts it carries
the full declaration in its own `[overrides.muse]`, and the card profiles stay
silent (`base.toml` vetoes it for unrecognised cards). The one built-in
muse-hosting shape:

```bash
lobes init --shape thor-muse --apply   # on a Jetson AGX Thor
lobes fleet up --apply
lobes up muse --apply                  # or start just this role
```

`thor-muse` hosts `muse` + `embedder` + `reranker` + `stt`/`tts` and drops
BOTH heavy default lobes (`cortex` and `senses`) to peer boxes — declare
`PRIMARY_PEER_ORIGIN` / `MULTIMODAL_PEER_ORIGIN` (and optionally the
`*_PEER_PROXY` knobs) so callers get honest referrals or transparent proxying
for the dropped roles. See [`deployment-shapes.md`](deployment-shapes.md).

**The budget values are measured, not yet validated.** `gpu_mem_util=0.55` is
~67.55 GiB of the Thor's 122.82 GiB unified pool; the 2026-07-17 live boot
measured a ~41 GiB non-KV footprint at the full window (32.06 GiB in-memory
weights per vLLM's load report + the MTP draft + the 262144-token profiling
pass), leaving a **26.47 GiB KV pool — 611,415 fp8 tokens, 2.33x concurrency**
at `max_model_len=262144`, the FULL 256K native window (operator decision —
no box-budget trim). The first hypothesis, util 0.40 (~49.1 GiB), was refused
live: the profiling pass left only 0.6 GiB for KV. Both `spark-lobe` and
`thor-lobe` shipped values that vLLM **refused** at their paper-derived
reclaim-sums, and the same lesson repeated here — shape budgets are **measured
truths, not arithmetic**. The acceptance run and its `docs/evidence/`
transcript are still pending, so these numbers are measured but the shape is
not validated (#108).

## Validation status

**DECLARED — memory budget measured live 2026-07-17, acceptance run pending.**
A physical Thor booted this gear (native class, `TRITON_ATTN`, `modelopt`
quant all held) and measured the memory ceiling: util 0.55 → 26.47 GiB KV pool
at the full 262144 window; the 0.40 hypothesis was refused with 0.6 GiB KV.
Per the #108 rule, no doc, support table, or `lobes capabilities` output may
claim it validated until an acceptance run (`scripts/accept-shape.sh`) passes
on a physical Thor and its transcript lands under `docs/evidence/`. That run
still gates: the correctness probes, the MTP draft's acceptance rate (vs. the
declared config), and whether the #101 audio gap applies to this plain-gemma4
line as assumed.

## Related docs

- [`colleague-stack.md`](colleague-stack.md) — the seven-role contract and
  `muse`'s responsibilities/forbidden lists.
- [`deployment-shapes.md`](deployment-shapes.md) — the `thor-muse` shape, the
  opt-in-core-role concept, referral/proxy for the dropped roles.
- [`gateway-fleet.md`](gateway-fleet.md) — tier aliases, the inverted
  feasibility default, peer channels, pressure policy.
- [`gemma4-mtp-draft.md`](gemma4-mtp-draft.md) — the assistant-draft family
  table (`gemma4_assistant` vs `gemma4_unified_assistant`).
- [`gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md) — the 12B `senses` gear this
  shares an image (and the #101 caveat) with.
- [`machine-profiles.md`](machine-profiles.md) — the per-card tuning axis;
  why the card profiles stay silent on `muse`.
