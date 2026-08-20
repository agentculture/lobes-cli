# NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 — the "worker" role

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded
> *now* — see
> [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **Status: DECLARED, UNVALIDATED.** This checkpoint has never booted on any
> box in this mesh. Everything below is read from the checkpoint's own
> published `config.json` / `hf_quant_config.json` (fetched 2026-08-20), not
> from card prose, per this repo's #108 discipline — validated / measured
> facts land only once the covering plan's live-boot tasks (t1, t2, t8) and
> their evidence transcripts under `docs/evidence/` exist. See
> `docs/plans/2026-08-20-nemotron-lightning-worker.md`.

**Model id:** `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
**Tier alias:** `worker` — the role name *is* the alias (capability order:
`hand` < `multimodal` < `worker` < `muse` < `main`/`cortex`).
**Role:** `worker` — the fleet's fast ground-work DOER (the eighth
Colleague role). Replaces `unsloth/Qwen3.6-35B-A3B-NVFP4` in this seat
(nemotron-lightning-worker plan, #187) — that checkpoint is demoted to a
kept `candidate` (cite-don't-delete), not deleted. See
[`qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md) for its own history.
**Status:** `configured` (declared 2026-08-20 from the published config
files; never booted on this fleet's hardware).

## What it is

An UNGATED (no HF license wall) NVIDIA checkpoint: `NemotronHForCausalLM`,
`model_type: "nemotron_h"` — a hybrid architecture combining Mamba-2
state-space layers, sparse mixture-of-experts blocks, and selective
attention layers across 52 hidden layers. 128 routed experts, 1 shared
expert, 6 experts selected per token — ~3B active of 30B total parameters,
matching the card's own "30B/3B active" framing. This is a **different
engine-support family** from the outgoing Qwen worker
(`Qwen3_5MoeForConditionalGeneration`) — `nemotron_h` serving support on
this repo's pinned vLLM nightly is UNPROVEN until the covering plan's t1
(sm_110 SASS/PTX bring-up) and t2 (standalone serve spike) tasks run.

Checkpoint facts (read from the published config files, fetched 2026-08-20):

- **`max_position_embeddings = 1048576`** (1M native ceiling) —
  `config.json`. This is a config-verified ceiling, not card prose. Nothing
  in this catalog or any shape allocates the full 1M window by default; a
  live boot decides the served `max_model_len` (plan t2/t8).
- **NO `vision_config` anywhere in the file** — this checkpoint is
  **TEXT-ONLY**, unlike the outgoing worker's ViT (image+video) intake. The
  `worker` role therefore LOSES `image_understanding`/`video_understanding`
  on this swap (`roles.py`'s `ROLE_RESPONSIBILITIES` is redefined in a
  sibling plan task, t4).
- **`hf_quant_config.json`**: `producer.name = "modelopt"` (version
  `0.44.0rc5`); `quant_algo: "MIXED_PRECISION"` — FP8 on
  attention/lm_head-style projections, `W4A16_NVFP4` (`group_size: 16`) on
  the routed-expert up/down projections; `kv_cache_quant_algo: "FP8"`. This
  is the same nvidia-modelopt family as the `muse` 31B gear and the outgoing
  27B MTP primary (`quantization="modelopt"`), **NOT** the
  `compressed-tensors` format the outgoing Qwen worker used.
- **No `mtp_num_hidden_layers` / draft-head field and no speculative-decoding
  field anywhere in `config.json`** — so the catalog entry's
  `speculative_config` stays empty (the honest sentinel), even though the
  model card separately advertises MTP/DSpark speculative-decoding support.
  That support is **declared by the card, unmeasured by us** — plan task t2
  evaluates MTP/DSpark separately from plain decode, after the plain-decode
  number is in hand.
- **Tool calling — UNVALIDATED on our engine.** The card's own example vLLM
  serve command passes `--reasoning-parser nemotron_v3` (a reasoning-parser
  flag; there is no catalog field for it, it is tracked here and in the
  covering plan) and `--tool-call-parser qwen3_coder` — i.e. the publisher
  itself asserts this non-Qwen checkpoint emits Qwen3-Coder-shaped tool
  calls. This repo has been burned by silently-wrong parser pairs before
  (see the Gemma 4 `pythonic` history in `CLAUDE.md`), so this claim is
  carried as the **best-cited default**, not a validated fact, until plan
  task t2's structured `tool_calls` probe confirms or disproves it live. The
  card also suggests `--moe-backend marlin`; the catalog entry does NOT
  carry that forward — the outgoing worker's own sm_110 history showed every
  forced NVFP4 MoE backend refused on this exact hardware family, so the
  entry leaves `moe_backend=""` (auto-select) instead of repeating an
  unverified guess in the opposite direction.
- License: `OpenMDW-1.1` (per the model card).

## Status and gating

No `gpu_mem_util` or `max_model_len` budget is declared anywhere for this
gear — per the thor-muse/thor-worker rule established elsewhere in this
repo, those are **measured truths** on a unified-memory card, not
arithmetic. This entry, and the `worker` role's catalog binding, promote to
`load-tested` only once:

1. plan task t1 proves sm_110 compatibility for the pinned vLLM nightly,
2. plan task t2's standalone serve spike returns a structured `tool_calls`
   array (not a call leaked into `content`) and a plain-decode tok/s number,
3. plan task t8's live Thor boot measures the co-resident budget and records
   an evidence transcript under `docs/evidence/`.

See `docs/plans/2026-08-20-nemotron-lightning-worker.md` for the full task
list, and `docs/qwen3.6-35b-a3b-nvfp4.md` for the checkpoint this one
replaces in the `worker` seat.
