# Gemma 4 12B-NVFP4 — "multimodal" tier (normal)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).
>
> **"Support both" (2026-07-02).** The catalog carries **two** Gemma 4 12B
> gears — see [`docs/vllm-nightly-migration.md` §7](vllm-nightly-migration.md)
> for the full benchmark evidence behind this split:
>
> - **Base (default)** — `coolthor/gemma-4-12B-it-NVFP4A16`, the fleet's
>   default `multimodal`/`normal` tier gear, **native MTP wired ON by
>   default**: measured **28.6 tok/s decode at 57.9% draft acceptance** — the
>   fastest Gemma config measured on this hardware.
> - **Coder (opt-in)** — `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`,
>   KEPT (cite-don't-delete) but DEMOTED to a `candidate`: coding-strong, but
>   native MTP only reaches **30.8%** draft acceptance on this fine-tune (a
>   marginal ~6% decode win), so it is **not** wired by default. Reachable by
>   explicit id or the opt-in `multimodal-coder` gateway alias.
>
> Everything below that is shared architecture (the `Gemma4UnifiedForConditionalGeneration`
> class, the serve-enablement story, `TRITON_ATTN`, `compressed-tensors` NVFP4,
> the #71/#73 live-validation) applies to **both** checkpoints — they are the
> same unified multimodal family, one a base it-model, one a coder fine-tune of
> it. Sections that differ between the two (model id, speculative decoding,
> serving env vars) call out each gear explicitly.

**Model id (default, base):** `coolthor/gemma-4-12B-it-NVFP4A16`
**Model id (opt-in, coder):** `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`
**Tier alias:** `multimodal` (and `normal` back-compat; resolves to the **base** gear via `model=multimodal` or `model=normal` at the gateway). The coder is reachable via `model=multimodal-coder` once its opt-in backend is wired, or by its explicit id.
**Role:** `multimodal` (base) / `candidate` (coder) — unified text+image+audio gear(s); the base is the fleet's "normal" tier
**Status:** `load-tested` (both). Serve-enablement **resolved** (#71/#73) for the
shared architecture; the base gear's native-MTP speedup was measured in §7
(2026-07-02, 28.6 tok/s @ 57.9% acceptance). See
["Live-validation status"](#live-validation-status-71) below.

## What it is

The Gemma 4 12B family is the fleet's **unified multimodal** gear(s): a single
checkpoint serving text, image, and audio through
`Gemma4UnifiedForConditionalGeneration` — no separate vision or audio sidecars
needed. Both the base and coder checkpoints are quantized to **NVFP4** in
**`compressed-tensors`** format (not nvidia modelopt).

Architecture highlights (shared by both gears):

- **Unified multimodal** (`Gemma4UnifiedForConditionalGeneration`) — text, image,
  and audio in one checkpoint; the generate lane gains native vision and audio
  without the realtime overlay.
- **`compressed-tensors` NVFP4** (`--quantization compressed-tensors`) — the
  checkpoint's `config.json` declares `quant_method="compressed-tensors"`,
  `format="nvfp4-pack-quantized"`. Passing `modelopt_fp4` fails with a
  quant-method mismatch (verified live, #71).
- **Non-square attention** — `global_head_dim=512` is double `head_dim=256`, so
  the o_proj input is `num_heads × global_head_dim = 8192`. The default
  FlashAttention backend emits `num_heads × head_dim = 4096` and serving crashes;
  **`VLLM_ATTENTION_BACKEND=TRITON_ATTN`** is required (see #71 below; §6 later
  found this env is a no-op *warning* on the pinned nightly digest — the native
  class auto-forces TRITON regardless — but it is kept, belt-and-suspenders).
- **Native MTP differs by checkpoint** — see
  ["Speculative decoding"](#speculative-decoding-support-both-7) below: the
  **base** gear has it wired ON by default (57.9% acceptance); the **coder**
  gear does not (measured 30.8%, not worth it).
- **Pythonic tool calls** (`--tool-call-parser pythonic`).
- **128K native context** — confirmed from `text_config.max_position_embeddings =
  131072` in the checkpoint config (#71); the base checkpoint shares this family
  config (not independently re-measured for the exact NVFP4A16 export).
- **No `--language-model-only`** — vision and audio towers are active by default;
  adding that flag would disable them and defeat the headline capability.

## Serving (fleet)

The **base** gear is **default-on** in the fleet — it starts with the standard
`docker compose up` (no `--profile` needed). The `vllm-multimodal` service is
always-warm alongside the primary, embed, and rerank gears. The **coder** gear
is **opt-in** — its `vllm-multimodal-coder` service sits behind the
`multimodal-coder` compose profile (mirrors `vllm-minor`/`vllm-middle`).

Once the fleet is running, the gateway automatically wires `MULTIMODAL_BASE_URL`
to `http://vllm-multimodal:8000` — no `.env` edit required for the base gear.

### Base gear (default)

Key env vars (from `env.example`):

| Variable | Default | Notes |
|---|---|---|
| `MULTIMODAL_MODEL` | `coolthor/gemma-4-12B-it-NVFP4A16` | HF checkpoint id |
| `MULTIMODAL_SERVED_NAME` | `coolthor/gemma-4-12B-it-NVFP4A16` | OpenAI `model` id the gateway routes to |
| `MULTIMODAL_GPU_MEM_UTIL` | `0.22` | ~26 GiB on the 128 GB GB10 (live-validated 2026-07-02, [§8](vllm-nightly-migration.md#8-always-on-duo-budget-live-validated-2026-07-02)) |
| `MULTIMODAL_MAX_MODEL_LEN` | `131072` | the full 128K native context, co-resident with the 64K-trimmed 27B primary (see [Live-validation status](#live-validation-status-71) and [§8](vllm-nightly-migration.md#8-always-on-duo-budget-live-validated-2026-07-02)) |
| `MULTIMODAL_QUANTIZATION` | `compressed-tensors` | the checkpoint's own quant_method |
| `MULTIMODAL_ATTENTION_BACKEND` | `TRITON_ATTN` | required for the non-square attention (#71) |
| `MULTIMODAL_IMAGE` | *(unset → local build)* | or a `ghcr.io`/local-registry tag |

Compose flags used by the `vllm-multimodal` service:

```text
--model ${MULTIMODAL_MODEL}
--served-model-name ${MULTIMODAL_SERVED_NAME}
--quantization ${MULTIMODAL_QUANTIZATION:-compressed-tensors}
--max-model-len ${MULTIMODAL_MAX_MODEL_LEN:-131072}
--gpu-memory-utilization ${MULTIMODAL_GPU_MEM_UTIL:-0.22}
--tool-call-parser=pythonic
--speculative-config={"method": "mtp", "model": "google/gemma-4-12B-it-assistant", "num_speculative_tokens": 1}
--trust-remote-code
# env: VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

Note: **no `--language-model-only`** — the vision and audio towers are active.
This is the key difference from the `vllm-minor` service, which *does* use
`--language-model-only` to drop the ViT tower.

### Coder gear (opt-in)

Activate with `docker compose --profile multimodal-coder up -d` (or add
`multimodal-coder` to `COMPOSE_PROFILES` in `.env`), then set
`MULTIMODAL_CODER_BASE_URL` so the gateway wires it:

| Variable | Default | Notes |
|---|---|---|
| `MULTIMODAL_CODER_MODEL` / `MULTIMODAL_CODER_SERVED_NAME` | `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4` | HF checkpoint id / OpenAI `model` id |
| `MULTIMODAL_CODER_GPU_MEM_UTIL` | `0.12` | adds ~15 GiB on top of the default 0.64 fleet budget |
| `MULTIMODAL_CODER_BASE_URL` | *(unset by default)* | set to activate gateway routing |

The `vllm-multimodal-coder` compose service is otherwise identical to
`vllm-multimodal` (same custom image, `Dockerfile.vllm-gemma4`) but carries
**no `--speculative-config`** — native MTP measured only 30.8% draft
acceptance on this checkpoint (§6/§7), not worth wiring.

## GPU memory budget

When the default fleet is active (primary + multimodal + embed + rerank):

| Gear | `--gpu-memory-utilization` | Approx GiB |
|---|---|---|
| `primary` (27B MTP, **64K**, trimmed from 128K) | **0.30** | ~38 |
| `multimodal` (12B unified base, 128K native) | **0.22** | ~26 |
| `embed` (0.6B) | 0.06 | ~7 |
| `rerank` (0.6B) | 0.06 | ~7 |
| **Total** | **0.64** | ~78 / 128 GB |
| *(opt-in)* `multimodal-coder` (12B unified coder) | +0.12 | +~15 |

This "always-on duo" budget was **live-validated co-resident on the DGX Spark
GB10, 2026-07-02**: the multimodal gear held its full 128K context at 4.67×
concurrency and the primary held 64K at 6.36× concurrency (measured at util
0.35, shaved to the shipped 0.30 for extra headroom) at the same time —
~108 GiB used / ~13 GiB free alongside embed + rerank and other co-tenant
services. See
[`vllm-nightly-migration.md` §8](vllm-nightly-migration.md#8-always-on-duo-budget-live-validated-2026-07-02)
for the full numbers; this supersedes the earlier #71 co-resident-safe
fallback (8192 tokens @ util 0.12, ~15.7 GiB) described in
["Live-validation status"](#live-validation-status-71) below, which predates
this duo-budget measurement.

## Tier alias usage

Callers use capability-tier aliases instead of hardcoded model ids:

| Alias | Routes to | Fallback when absent |
|---|---|---|
| `multimodal` | **12B `multimodal` (base, native MTP)** | primary |
| `normal` | **12B `multimodal` (base, native MTP)** | primary (back-compat) |
| `multimodal-coder` | **12B coder** (opt-in) | primary (unresolved when not wired) |
| `main` | 27B `primary` | always present |
| `hard` | 27B `primary` | always present (back-compat) |
| `minor` | 4B `minor` | primary |
| `cheap` | 4B `minor` | primary (back-compat) |

Send `model=multimodal` or `model=normal` and the gateway resolves to the
**base** gear when it is wired and healthy. If the multimodal backend is not
started, the alias falls back upward to the primary — the caller's code is
unchanged. `multimodal-coder` is not part of the tier vocabulary — it is a
dedicated alias, added only once the opt-in coder backend is wired (mirrors the
tier-fallback contract: an alias never points at a served name nothing actually
serves).

```python
# Before: hardcoded model id
response = client.chat.completions.create(
    model="coolthor/gemma-4-12B-it-NVFP4A16",
    messages=[{"role": "user", "content": "..."}],
)

# After: tier alias — the gateway resolves to the right gear
response = client.chat.completions.create(
    model="multimodal",
    messages=[{"role": "user", "content": "..."}],
)

# Coding-heavy workload: opt into the coder gear explicitly (once its
# --profile multimodal-coder service is up and MULTIMODAL_CODER_BASE_URL is set)
response = client.chat.completions.create(
    model="multimodal-coder",
    messages=[{"role": "user", "content": "..."}],
)
```

## DSpark experiment — INVALID, do not wire (§6)

The DSpark route (`deepseek-ai/dspark_gemma4_12b_block7`, a DeepSeek
*speculative-decoding draft model* for Gemma 4 12B) was investigated in #75 and
found **INVALID on vLLM 0.23** (`docs/vllm-nightly-migration.md` §6, live,
2026-07-01): its custom `Gemma4DSparkModel` drafter architecture is not in
vLLM 0.23's supported speculative-draft set (`Model architectures
['Gemma4DSparkModel'] are not supported for now`). Do not wire it — the native
`mtp` route (the public `google/gemma-4-12B-it-assistant` draft, wired on the
base gear above) is the one that works.

## Speculative decoding — "support both" (§7) {#speculative-decoding-support-both-7}

> **Status: RESOLVED for the base gear (2026-07-02).** The base gear
> (`coolthor/gemma-4-12B-it-NVFP4A16`) now serves with native MTP **ON by
> default**: `{"method": "mtp", "model": "google/gemma-4-12B-it-assistant",
> "num_speculative_tokens": 1}`, measured **28.6 tok/s decode at 57.9% draft
> acceptance** (up from 19.8 tok/s no-spec, ~1.45×) — see
> [`docs/vllm-nightly-migration.md` §7](vllm-nightly-migration.md) for the full
> comparison table (coder no-spec/+MTP, bf16 base+MTP, and this NVFP4 base+MTP).
> The **coder** gear (`sakamakismile/…`) was also measured with native MTP —
> only **30.8%** draft acceptance, a marginal ~6% decode win — so it is KEPT
> opt-in but **not** wired by default. The sections below are the historical
> framing from before this resolution (issue #75, closed with the route
> resolved but wire→measure→verdict unbuilt); retained for context.

**Audience:** lobes operators/maintainers and the Culture mesh that consumes the
`multimodal` (Gemma 4 12B) generate lane — i.e. anyone calling `model=multimodal`
or the back-compat `model=normal` (see [Tier alias usage](#tier-alias-usage)).

**Historical before-state (as of #71/#73, pre-§7).** The gemma catalog entry
(then the coder, `id="sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4"`)
carried **no `speculative_config`**: the native self-speculation path was closed
(`{"method": "gemma4_mtp"}` rejected as "Unsupported speculative method" on
vLLM 0.21/0.22), and the DSpark draft-model route documented below was
disabled and unvalidated. §6/§7 (live, 2026-07-01/02) resolved this: the
correct method is `"mtp"` (not `"gemma4_mtp"`) with the draft under the
`"model"` key (not `"draft_model_id"` — vLLM 0.23's `SpeculativeConfig` rejects
that outdated key), and DSpark was proven **invalid** (unsupported drafter
architecture). See [DSpark experiment](#dspark-experiment--invalid-do-not-wire-6)
above.

**The gap this leaves (now closed for the base gear).** The 27B `primary`/`main`
gear gets a measured **~2.4× single-stream decode speedup** from MTP speculative
decoding (72–79% draft acceptance) — see
[`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md). The base
`multimodal`/`normal` gear now gets its own multiplier (~1.45×, 28.6 tok/s) from
native MTP — see §7. The coder gear (opt-in) still has the smaller ~1.04× gap
this section originally described (24 tok/s vs the base's 28.6).

**Scope split (historical).**

| Concern | Owner |
|---|---|
| Serve-enablement — force `TRITON_ATTN` on the transformers backend so the gear actually serves | issue #71 (see [Live-validation status](#live-validation-status-71) below) |
| Draft-model training/distilling (building a native `gemma4_assistant` head from scratch) | a separate follow-up, not #75 |
| Resolve a draft route, wire `--speculative-config`, measure, and decide | **issue #75 (CLOSED — route resolved; wire/measure/verdict not implemented)** |

As scoped, #75 did not train a draft model. It resolved to exactly one concrete
route (a validated `draft_model` such as DSpark, a sourced `gemma4_assistant`
draft, or a documented "no compatible draft available") to wire through the same
catalog-to-compose pattern the 27B MTP primary uses today (`speculative_config`
on the catalog entry drives the compose `--speculative-config` items), then
measure draft acceptance and decode speedup on a live co-resident serve and
commit a verdict. **#75 closed with the route resolved (see
[`gemma4-mtp-draft.md`](gemma4-mtp-draft.md)) but the wire → measure → verdict
legs unbuilt.** Its serve gate (#71) has since landed, so a future issue can
resume from the resolved route against a now-serving gear.

## Live-validation status (#71/#73) {#live-validation-status-71}

> **RESOLVED on the DGX Spark (GB10, sm_121, 2026-07-01, #71/#73): the gear
> SERVES and answers text + image requests.** `status` = `load-tested`.
>
> This #71/#73 validation ran against the **coder** checkpoint (the default
> gear at the time). The **base** checkpoint (`coolthor/gemma-4-12B-it-NVFP4A16`,
> promoted to default in §7, 2026-07-02) shares the identical
> `Gemma4UnifiedForConditionalGeneration` architecture and serve story — the
> §7 measurement independently re-confirmed it *serves* and *decodes* correctly
> (28.6 tok/s + MTP). The gap this admission used to flag — whether the
> text+image+audio content-correctness checks further down this section hold
> against the base checkpoint specifically — is now resolved, and the answer
> **splits**: image + text does, audio + text doesn't. See the next
> subsection for the live evidence, and the correction to the coder's
> original "audio + text ✓" line further down.

### Base checkpoint (coolthor): image + text verified, audio + text not served (#101)

Live evidence gathered against `coolthor/gemma-4-12B-it-NVFP4A16` on the DGX
Spark (vLLM `0.23.1rc1.dev672+g93d8f834d`), via `model=multimodal`:

| request | `prompt_tokens` | result |
|---|---|---|
| text only | 15 | — |
| text + image (96×96 solid PNG, stdlib-generated) | **273** (+258) | replies `"Red"` for a red image, `"Blue"` for a blue one |
| text + audio (0.68 s WAV, 24 kHz, from the rig's own Chatterbox TTS) | 34 (+19) | `""` (immediate EOS) |
| text + audio (same clip resampled to 16 kHz) | 48 | *"I cannot hear any audio because you haven't provided a file or a link…"* |

**Image + text: VERIFIED**, against known ground truth, with a negative
control (a blue image correctly fails a `"red"` assertion). This replaces
the earlier hedge — the base checkpoint's image intake is confirmed, not
merely inherited by architectural similarity to the coder checkpoint.

**Audio + text: NOT SUPPORTED on this vLLM path.** The image expands the
prompt by 258 tokens — real content injected into the sequence. The audio
adds only ~19 placeholder tokens and no content: the model receives no
signal that audio was attached, so it either emits an immediate EOS (24 kHz)
or, resampled to 16 kHz, a fluent claim that no audio was provided. The
checkpoint's own `config.json` declares `audio_config` and `audio_token_id`
— so this is a **vLLM `gemma4_unified` gap, not a checkpoint gap**. vLLM
**drops** the `input_audio` content part rather than rejecting it: a caller
gets `200 OK` and a fluent answer that silently ignored the audio, which is
the worst failure mode for a caller relying on the advertised capability.
Tracked as **issue #101**. Until #101 lands, treat `senses`/`multimodal` as
vision-only intake — for speech, use the purpose-built `stt` role (Parakeet,
`POST /v1/audio/transcriptions`; see [`docs/colleague-stack.md`](colleague-stack.md)).

**The fix: vLLM nightly's native class.** `gemma4_unified` is **early-fusion**
multimodal (no separate vision/audio towers — a `vision_embedder` and audio
projection feed tokens straight into the shared 48-layer LM) with **heterogeneous
per-layer head sizes**: 40 `sliding_attention` layers at `head_dim=256` and 8
`full_attention` layers at `global_head_dim=512`. Serving it needs vLLM's **native
`Gemma4UnifiedForConditionalGeneration`** class, which gives the two attention
types different KV block sizes and auto-forces `TRITON_ATTN`. That class exists
**only in vLLM nightly (≥ 0.23.1rc1)**. The shipped image
([`Dockerfile.vllm-gemma4`](../lobes/templates/fleet/Dockerfile.vllm-gemma4)) is
now `FROM vllm/vllm-openai:nightly` (pinned by digest) + the `vllm[audio]` extra
(`av`/`soundfile`/`librosa`/`soxr` — audio input resamples via PyAV).

**Why released vLLM ≤ 0.22.1 can't serve it.** With no native unified class, vLLM
falls back to the generic **Transformers modeling backend**, which builds *every*
layer's attention with a **single** `head_size` (256). The 8 full-attention layers
then emit `16×256=4096` but their o_proj wants `16×512=8192` →
`RuntimeError: Shape mismatch: a.size(1)=4096, size_k=8192` (marlin_gemm) at the
profiling forward. **This is not an attention-backend problem**: engaging
`TRITON_ATTN` via the `--attention-backend` CLI flag (0.22.1) was proven live to
crash *identically* — the single head_size is the wall, and only the native class
gets per-layer head sizes right.

Runtime matrix tested:

| Image | vLLM | native `gemma4_unified` class | serves |
|---|---|---|---|
| `nvcr.io/nvidia/vllm:26.04-py3` | 0.19.0 | ❌ | ❌ |
| `…:26.06-py3` + Transformers `181beb3` | 0.22.1 | ❌ (transformers-backend fallback) | ❌ (o_proj 4096≠8192) |
| **`vllm/vllm-openai:nightly` (shipped)** | **0.23.1rc1.dev** | **✅** | **✅ text+image** (audio accepted, silently dropped — #101) |

**Live validation results (util 0.25, `--max-model-len 4096`, GB10):**

- ✅ **Serves** — `/health` 200, native class resolved, TRITON auto-forced.
- ✅ **Text** — arithmetic + factual answered correctly.
- ✅ **Image + text** — described a test image (red circle + text) correctly.
- ⚠️ **Audio + text** — recorded at the time as "transcribed a 24 kHz TTS clip
  **verbatim** (needed `av`)," but the check behind that line asserted only
  `HTTP 200` + non-empty response content against a placeholder clip — it never
  diffed the transcription against the clip's actual words, so it was **never
  verified against ground truth**. When audio + text was later tested properly
  against the base checkpoint (the evidence table in
  ["Live-validation status"](#live-validation-status-71) above), the claim did
  **not** hold: vLLM drops the `input_audio` content part on this serving path
  and the model answers as if no audio was attached. Do not rely on this line;
  see **#101**.
- ✅ **GPU util** — ~**15.7 GiB** actual (weights 8.1 + cudagraph 0.46 + KV 7.2) ≈
  **0.12** of the then-**0.69** default-fleet budget. **Superseded 2026-07-02**
  by the always-on duo retune — see the note below and
  [`vllm-nightly-migration.md` §8](vllm-nightly-migration.md#8-always-on-duo-budget-live-validated-2026-07-02).

**Two config gotchas (vLLM 0.23), now handled in the compose/env:**

- **Cudagraph memory over-estimate.** vLLM 0.23 reserves an *estimated* cudagraph
  headroom inside the util budget — here **12.74 GiB estimated vs 0.46 GiB actual**
  — which starves the KV cache so `util 0.12` fails with *"No available memory for
  the cache blocks"*. The compose sets `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
  so the util knob maps to true usage.
- **Context vs util (historical, superseded 2026-07-02).** At the *original*
  `util 0.12` the KV cache held only ~24K tokens, so the 128K native context was
  **not** serveable at that co-resident lane budget — `MULTIMODAL_MAX_MODEL_LEN`
  defaulted to `8192`. This has since been resolved: the shipped default is now
  `util 0.22` / `MULTIMODAL_MAX_MODEL_LEN=131072`, live-validated co-resident
  with the 64K-trimmed 27B primary — see
  [`vllm-nightly-migration.md` §8](vllm-nightly-migration.md#8-always-on-duo-budget-live-validated-2026-07-02).

Resolved:

- ✅ **Serve-enablement** — native `gemma4_unified` class on vLLM nightly + `TRITON_ATTN`.
- ✅ **Image** — functional, verified against ground truth (see the base-checkpoint
  table in ["Live-validation status"](#live-validation-status-71) above).
- ⚠️ **Audio** — NOT served on this vLLM path (#101): the `vllm[audio]` extra
  installs cleanly, but `gemma4_unified` silently drops the `input_audio`
  content part instead of feeding it to the model. Use the purpose-built `stt`
  role (Parakeet, `POST /v1/audio/transcriptions`) for speech instead.
- ✅ **Quantization** — `compressed-tensors` (not `modelopt_fp4`).
- ✅ **GPU util** — ~15.7 GiB ≈ 0.12 budget.
- ✅ **Native MTP (base gear, default-on)** — the public `google/gemma-4-12B-it-assistant`
  draft, wired via `{"method": "mtp", "model": "…", "num_speculative_tokens": 1}`:
  28.6 tok/s @ 57.9% draft acceptance (§7). ⛔ **Not wired on the coder gear**
  (opt-in) — measured only 30.8% acceptance there, not worth it.

## Benchmark — 2026-07-01, DGX Spark (GB10), standalone (coder, no-spec)

> First throughput/prefill numbers for the gear — measured on the (then-default)
> **coder** checkpoint, **no speculative decoding**. Measured on the shared GB10
> **standalone** (own container on host port 8010, **not** co-resident behind the
> gateway) so the live 27B primary kept serving mesh traffic. Image
> `lobes/vllm-gemma4:nightly-audio` (vLLM **0.23.1rc1.dev672**, native
> `Gemma4UnifiedForConditionalGeneration`, `TRITON_ATTN`, `compressed-tensors`
> NVFP4, `--max-model-len 8192`). Driven by `lobes benchmark` + a manual
> forced-decode probe. See
> [`docs/vllm-nightly-migration.md` §7](vllm-nightly-migration.md) for the
> **base** gear's numbers (19.8 tok/s no-spec, 28.6 tok/s + native MTP) and the
> coder's own +MTP number (24 tok/s @ 30.8% accept) measured the same way.

| Property | Value |
|---|---|
| Health / `max_model_len` | `/health` 200; `8192` |
| Architecture resolved | ✅ native `Gemma4UnifiedForConditionalGeneration` (not the transformers fallback) |
| Correctness | `17 × 23 = 391` ✅ (finish=stop, 4 tok) |
| **Decode throughput** | **~23 tok/s** (batch=1, greedy — 21.7–23.3 across balanced/prompt-heavy; **23.0 tok/s sustained over 1,500 forced tokens** in 65.2 s) |
| Prefill (balanced) | 847 prompt tokens + 16 gen in **0.32 s** (~2,650 tok/s) |
| Prefill (prompt-heavy) | 6,682 prompt tokens in **3.42 s** (~1,954 tok/s) |
| Weights (EngineCore) | **8,113 MiB** (~7.9 GiB) |
| CUDA-graph memory | 0.96 GiB (trimmed capture set — see config note) |
| KV cache | **8.47 GiB → 57,636 tokens**; **7.04×** max concurrency at 8,192 tokens/request |
| Speculative decoding | **none** — this is the coder's raw single-stream, no-spec baseline (§6). With native MTP the coder reaches ~24 tok/s @ 30.8% accept (§7) — kept opt-in, not wired by default. |

**Decode context.** At ~**23 tok/s single-stream** (coder, no-spec), the 12B
`multimodal` lane is actually a touch *faster* per-stream than the 27B `primary`
(~18–19 tok/s, and that is *with* its ~2.4× MTP boost) — it is less than half the
parameters. The **default** base gear now does even better with native MTP:
**28.6 tok/s** (§7) — see
[Speculative decoding — "support both" (§7)](#speculative-decoding-support-both-7)
above for the full picture across both gears.

**Config note (historical, 2026-07-01) — why not the then-production `util
0.12`.** At the time of this benchmark the default fleet lane ran
`MULTIMODAL_GPU_MEM_UTIL=0.12` with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.
That combination was validated on #71's earlier nightly digest; the **then-current**
`:nightly-audio` build (dev672) behaved differently in two ways that forced a
bench-only config:

1. **Cudagraph accounting flipped.** With `…ESTIMATE_CUDAGRAPHS=0`, dev672 now
   *warns* that CUDA-graph memory is **not accounted** during KV allocation and
   recommends re-enabling it and raising util (it suggested `0.12 → 0.2427`). The
   full capture list (51 sizes, 1…512) estimates **14.93 GiB** of graph memory,
   which starves KV at `util 0.12` (KV came out at 1.11 GiB < the 1.2 GiB needed
   for one 8,192-token request).
2. **Co-resident memory ceiling.** With the primary (reduced to 64K / util 0.38)
   plus embed and rerank live, only **~19 GiB was CUDA-visible free** to a new
   process on the unified memory — not enough for the weights (8 GiB) plus the
   full 15 GiB of graphs.

So the benchmark trimmed the capture set to
`--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}'` (graphs →
0.96 GiB) at `--gpu-memory-utilization 0.15`. **Single-stream decode and prefill
are util- and capture-set-independent** (batch=1 stays graph-captured), so the
headline numbers above are representative; only *aggregate throughput above
concurrency 64* would read low versus a full-capture production serve. Total
standalone footprint here: ~**17.4 GiB** (weights 7.9 + graphs 0.96 + KV 8.47) at
util 0.15.

> **Follow-up (config drift) — RESOLVED 2026-07-02.** This note flagged that the
> then-default lane's `util 0.12` + `…ESTIMATE_CUDAGRAPHS=0` should be
> re-validated against the pinned dev672 image and the util raised if the
> accounting change stuck. The always-on duo retune did exactly that: the
> shipped default is now `util 0.22` (close to the `0.2427` this benchmark
> suggested) with the full `131072` (128K) context, live-validated co-resident
> with the 64K-trimmed 27B primary — see
> [`vllm-nightly-migration.md` §8](vllm-nightly-migration.md#8-always-on-duo-budget-live-validated-2026-07-02).

## Related docs

- [`vllm-nightly-migration.md`](vllm-nightly-migration.md) — §6/§7: the live
  head-to-head + checkpoint-choice measurements behind the "support both"
  decision (coder vs base, DSpark-invalid finding, the exact native-MTP config).
- [`gemma4-mtp-draft.md`](gemma4-mtp-draft.md) — the draft-route research (#75,
  t1) that resolved the native `google/gemma-4-12B-it-assistant` route.
- [`gateway-fleet.md`](gateway-fleet.md) — fleet topology (main/minor/multimodal),
  tier alias routing, pressure policy, memory budget.
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  hard/`primary` gear (27B MTP).
- [`qwen3-14b-nvfp4.md`](qwen3-14b-nvfp4.md) — the legacy 14B candidate (demoted
  from the `normal` tier; no tier alias resolves to it).
