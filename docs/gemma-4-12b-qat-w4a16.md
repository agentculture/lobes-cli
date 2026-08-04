# Gemma 4 12B QAT W4A16 (`unsloth/gemma-4-12B-it-qat-w4a16`)

> **Status: DECLARED, not yet booted on any hardware.** The catalog entry
> (`lobes/catalog.py`) is **shipped** — verified against the checkpoint's own
> `config.json`, fetched unauthenticated 2026-08-04 (the repo is not gated),
> NOT card prose. Every capability claim below is **pending-live-probe**: no
> image, video, audio, reasoning-trace, or tool-calling check has been run
> against this checkpoint on any machine. It is cataloged as a switchable
> gear only. A later task in the
> `unsloth-qat-senses-first-class-orin-variation` plan boots it on a Jetson
> AGX Orin and backfills this page with live evidence (or a live-recorded
> refusal) — until that evidence lands under `docs/evidence/`, treat every
> row in the [capability table](#capability-table-all-rows-pending-live-probe)
> below as unproven.

**Model id:** `unsloth/gemma-4-12B-it-qat-w4a16`
**Role:** `candidate` (see [why not `role_hint="multimodal"`](#why-role_hintcandidate-not-multimodal) below)
**Status:** `configured` in the catalog — declared 2026-08-04, not yet booted.

## What it is

A second Gemma 4 12B **unified** multimodal checkpoint, from a different
publisher and export recipe than the fleet's current `senses`/`multimodal`
default (`coolthor/gemma-4-12B-it-NVFP4A16`, see
[`gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md)):

- **QAT (quantization-aware trained)**, finetuned from `google/gemma-4-12B-it`
  — unlike the coolthor gear, which is a post-training NVFP4 export.
- **Same architecture class**: `Gemma4UnifiedForConditionalGeneration`
  (`model_type: gemma4_unified`) — the same class this repo's existing
  `Dockerfile.vllm-gemma4` image already serves for the coolthor/coder gears,
  so this is a checkpoint swap on already-proven serve machinery, not a new
  engine-support question.
- **Compressed-tensors, serialized explicitly for vLLM.**

All facts below are read off the checkpoint's own `config.json` (fetched
unauthenticated 2026-08-04 —
`https://huggingface.co/unsloth/gemma-4-12B-it-qat-w4a16/resolve/main/config.json`),
never from the model card's marketing prose.

### Quantization: INT4 W4A16, NOT FP4 — a different kernel path than the incumbent

The coolthor/coder gears are `compressed-tensors` **NVFP4**
(`format="nvfp4-pack-quantized"`). This checkpoint's `quantization_config`
declares:

| Field | Value |
|---|---|
| `quant_method` | `compressed-tensors` |
| `format` | `pack-quantized` |
| `num_bits` | `4` |
| `strategy` | `group` |
| `group_size` | `32` |
| `symmetric` | `true` |

That is **INT4 weight-only (W4A16)** — activations stay 16-bit — a
**different kernel path** than the incumbent's FP4. Both resolve to the same
vLLM `--quantization=compressed-tensors` flag (vLLM reads the exact scheme
off the checkpoint's own config at load time), but the int4 path is
**unproven on any of this fleet's hardware** until a live boot exercises it.
This distinction matters operationally: weight-only int4 is what makes an
Ampere-class (`sm_87`) target *plausible* at all (no Blackwell FP4 tensor
cores needed for 16-bit activations) — the same rationale the
`coolthor/gemma-4-12B-it-NVFP4A16` gear already validated live on a Jetson
AGX Orin (see [`orin-profiles.md`](orin-profiles.md)) for its own NVFP4
weight-only path — but plausible is not proven for *this* checkpoint's int4
path specifically; the boot is the test.

### Context: 262144 native — double the incumbent 12B

`text_config.max_position_embeddings = 262144` (256K), confirmed from the
checkpoint's own config — **double** the coolthor incumbent's measured
131072 (128K). The HF card's summary also claims 256K, but per this repo's
the #108 discipline ("a card earns its tuning data by booting it") means the card is
the *attempt target*, not the evidence; `config.json`'s
`max_position_embeddings` is what is cited here, and the number still needs
a live boot on real hardware to prove it actually serves at that window
(vLLM can refuse a `gpu_mem_util` at any window — see the coolthor Orin
boot's own refused-then-accepted util history for a precedent of exactly
that).

### Modalities: text + image + audio + video declared

`config.json` carries both a `vision_config` (`gemma4_unified_vision`) and
an `audio_config` (`gemma4_unified_audio`), plus:

| Token | id |
|---|---|
| `image_token_id` | 258880 |
| `audio_token_id` | 258881 |
| `video_token_id` | 258884 |

**Video is natively declared** on this checkpoint — this repo's existing
Gemma 4 capability shape (`_SHAPE_GEMMA4_UNIFIED` in `lobes/catalog.py`,
"text+image+audio") predates this export and does not mention video. Whether
vLLM's `gemma4_unified` serving path actually *feeds* video content (as
opposed to merely declaring the token id, the way it silently drops audio —
see below) is unknown until probed live.

### No MTP / draft head

Unlike the coolthor gear (wired to the external
`google/gemma-4-12B-it-assistant` MTP draft), this checkpoint's `config.json`
carries no `mtp_num_hidden_layers` or draft-head field of any kind — the
catalog entry's `speculative_config` is correctly empty. Whether the public
assistant draft can be grafted onto this QAT export is unexplored; nothing
here assumes it can.

## Why `role_hint="candidate"`, not `"multimodal"`

The covering plan's requirement text describes this entry as mirroring the
coolthor gear including `role_hint="multimodal"`. The catalog entry
deliberately uses `role_hint="candidate"` instead: `tests/test_catalog.py`'s
`test_exactly_one_gemma_multimodal_gear` pins `role_hint="multimodal"` to
**exactly one** gear (the coolthor entry) — a second one would make
`resolve_tier("multimodal")` / `("senses")` / `("normal")` ambiguous by
first-match. This mirrors the exact precedent already set by the
`sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4` entry (also
`role_hint="candidate"` for the identical reason). Keeping this entry a
candidate leaves the fleet-wide `multimodal`/`senses`/`normal` tier default —
and every thor/spark profile that pins the raw coolthor id — untouched. A
per-box **operator profile** (not this catalog field) is how a deployment
actually selects this checkpoint for its `senses` role; see the covering
plan for that wiring.

## Capability table (all rows PENDING-LIVE-PROBE)

Every row below states what `config.json` **declares**, never what has been
observed serving. None of these have a live probe result yet — no image,
video, audio, reasoning, or tool-calling request has been sent to this
checkpoint on any machine as of this writing.

| Capability | What `config.json` declares | Known risk | Verdict |
|---|---|---|---|
| Image | `vision_config` (`gemma4_unified_vision`), `image_token_id=258880` | — | **pending-live-probe** |
| Video | `video_token_id=258884` (new — not on the coolthor family's declared shape) | unknown whether `gemma4_unified` actually feeds video content vs. only declaring the token, mirroring the audio gap below | **pending-live-probe** |
| Audio | `audio_config` (`gemma4_unified_audio`), `audio_token_id=258881` | **issue #101**: vLLM's `gemma4_unified` path silently **drops** `input_audio` content on the coolthor checkpoint (200 OK, fluent reply that ignored the audio) — a vLLM-path gap, not a per-checkpoint one, so it is expected to reproduce here until re-probed on the pinned nightly | **pending-live-probe** |
| Reasoning (thinking trace) | Gemma 4 `<\|channel>thought` markers, consumed by the paired `gemma4` reasoning parser | the gemma4 tool/reasoning parser PAIR is validated live only on the 31B `muse` gear (2026-07-17); 12B lanes inherit it as an UNVALIDATED family rule (#108) | **pending-live-probe** |
| Tool calling | Native `<\|tool_call>call:name{...}<tool_call\|>` syntax, matched by the `gemma-4*` rule in `runtime._parser.infer_parser` → `tool_parser="gemma4"` | same family-rule caveat as reasoning, above | **pending-live-probe** |

No capability claim in this table may be promoted to a verdict without a
live probe on real hardware with a negative control where one exists (vision:
wrong-colour image; video: reversed-motion clip; audio: the #101
silent-drop-detecting probe — token-count delta plus content assertion, not
just HTTP 200; reasoning: `sorted(message.keys())` dumped and
`usage.completion_tokens` reconciled against the visible field lengths before
any verdict, per this repo's `reasoning` vs. `reasoning_content` field trap).

## Related docs

- [`gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md) — the incumbent `senses`/
  `multimodal` default (`coolthor/gemma-4-12B-it-NVFP4A16`) and its own
  `role_hint="multimodal"` tier ownership.
- [`orin-profiles.md`](orin-profiles.md) — the live-validated Jetson AGX Orin
  deployment this checkpoint is a candidate replacement for.
- [`machine-profiles.md`](machine-profiles.md) — machine-profile / card
  detection mechanics (the Orin card strategy this plan's later tasks add).
- [`model-switch-playbook.md`](model-switch-playbook.md) — the measurement
  discipline (benchmark the incumbent first, `completion_tokens` not SSE
  chunk counts, etc.) the live-test task in the covering plan follows.
