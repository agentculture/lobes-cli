# Gemma 4 12B-NVFP4 — "multimodal" tier (normal)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

**Model id:** `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`
**Tier alias:** `multimodal` (and `normal` back-compat; resolves here via `model=multimodal` or `model=normal` at the gateway)
**Role:** `multimodal` — unified text+image+audio gear, the fleet's new "normal" tier
**Status:** `load-tested` — serve-enablement **resolved** (#71/#73). The gear
**serves** on the custom image (vLLM nightly, native `gemma4_unified` class) and
was live-validated on the DGX Spark GB10 for **text + image + audio** (2026-07-01;
see ["Live-validation status"](#live-validation-status-71) below).

## What it is

The Gemma 4 12B is the fleet's **unified multimodal** gear: a single checkpoint
serving text, image, and audio through `Gemma4UnifiedForConditionalGeneration` —
no separate vision or audio sidecars needed. It is quantized to **NVFP4** in
**`compressed-tensors`** format (not nvidia modelopt).

Architecture highlights:

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
  **`VLLM_ATTENTION_BACKEND=TRITON_ATTN`** is required (see #71 below).
- **No native MTP via this checkpoint** — despite the `-MTP` name, it exposes no
  `gemma4_assistant` draft, and vLLM 0.21/0.22 enable Gemma4 MTP only via a
  *separate* draft model. So **no `--speculative-config`** is carried (the
  `gemma4_mtp` method is rejected). The gear serves without spec-decode.
- **Pythonic tool calls** (`--tool-call-parser pythonic`).
- **128K native context** — confirmed from `text_config.max_position_embeddings =
  131072` in the checkpoint config (#71).
- **No `--language-model-only`** — vision and audio towers are active by default;
  adding that flag would disable them and defeat the headline capability.

## Serving (fleet)

The multimodal gear is **default-on** in the fleet — it starts with the standard
`docker compose up` (no `--profile` needed). The `vllm-multimodal` service is
always-warm alongside the primary, embed, and rerank gears.

Once the fleet is running, the gateway automatically wires `MULTIMODAL_BASE_URL`
to `http://vllm-multimodal:8000` — no `.env` edit required.

Key env vars (from `env.example`):

| Variable | Default | Notes |
|---|---|---|
| `MULTIMODAL_MODEL` | `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4` | HF checkpoint id |
| `MULTIMODAL_SERVED_NAME` | `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4` | OpenAI `model` id the gateway routes to |
| `MULTIMODAL_GPU_MEM_UTIL` | `0.12` | ~15 GiB on the 128 GB GB10 |
| `MULTIMODAL_MAX_MODEL_LEN` | `131072` | 128K (native, confirmed) |
| `MULTIMODAL_QUANTIZATION` | `compressed-tensors` | the checkpoint's own quant_method |
| `MULTIMODAL_ATTENTION_BACKEND` | `TRITON_ATTN` | required for the non-square attention (#71) |
| `MULTIMODAL_IMAGE` | *(unset → local build)* | or a `ghcr.io`/local-registry tag |

Compose flags used by the `vllm-multimodal` service:

```text
--model ${MULTIMODAL_MODEL}
--served-model-name ${MULTIMODAL_SERVED_NAME}
--quantization ${MULTIMODAL_QUANTIZATION:-compressed-tensors}
--max-model-len ${MULTIMODAL_MAX_MODEL_LEN:-131072}
--gpu-memory-utilization ${MULTIMODAL_GPU_MEM_UTIL:-0.12}
--tool-call-parser=pythonic
--trust-remote-code
# env: VLLM_ATTENTION_BACKEND=TRITON_ATTN
# NO --speculative-config (Gemma4 native MTP needs a separate gemma4_assistant draft)
```

Note: **no `--language-model-only`** — the vision and audio towers are active.
This is the key difference from the `vllm-minor` service, which *does* use
`--language-model-only` to drop the ViT tower.

## GPU memory budget

When the default fleet is active (primary + multimodal + embed + rerank):

| Gear | `--gpu-memory-utilization` | Approx GiB |
|---|---|---|
| `primary` (27B MTP, 128K) | 0.45 | ~56 |
| `multimodal` (12B unified, 128K) | **0.12** | ~15 |
| `embed` (0.6B) | 0.06 | ~7 |
| `rerank` (0.6B) | 0.06 | ~7 |
| **Total** | **0.69** | ~85 / 128 GB |

This leaves ~38 GiB of headroom on the 128 GB GB10 for KV caches and other
services. The multimodal gear's footprint is **measured** (#71, 2026-07-01):
~15.7 GiB (weights 8.1 + cudagraph 0.46 + KV 7.2) ≈ 0.12 — see
["Live-validation status"](#live-validation-status-71).

## Tier alias usage

Callers use capability-tier aliases instead of hardcoded model ids:

| Alias | Routes to | Fallback when absent |
|---|---|---|
| `multimodal` | **12B `multimodal`** | primary |
| `normal` | **12B `multimodal`** | primary (back-compat) |
| `main` | 27B `primary` | always present |
| `hard` | 27B `primary` | always present (back-compat) |
| `minor` | 4B `minor` | primary |
| `cheap` | 4B `minor` | primary (back-compat) |

Send `model=multimodal` or `model=normal` and the gateway resolves to this gear
when it is wired and healthy. If the multimodal backend is not started, the
alias falls back upward to the primary — the caller's code is unchanged.

```python
# Before: hardcoded model id
response = client.chat.completions.create(
    model="sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4",
    messages=[{"role": "user", "content": "..."}],
)

# After: tier alias — the gateway resolves to the right gear
response = client.chat.completions.create(
    model="multimodal",
    messages=[{"role": "user", "content": "..."}],
)
```

## DSpark experiment

The DSpark experiment is **disabled by default**. DSpark
(`deepseek-ai/dspark_gemma4_12b_block7`) is a DeepSeek *speculative-decoding draft
model* for Gemma 4 12B. Because the default serve carries **no** `--speculative-config`
(see below), trying it requires both adding a `--speculative-config` flag to the
`vllm-multimodal` command in `docker-compose.yml` **and** setting in `.env`:

```env
MULTIMODAL_SPECULATIVE_CONFIG={"method": "draft_model", "draft_model_id": "deepseek-ai/dspark_gemma4_12b_block7", "num_speculative_tokens": 3}
```

DSpark is unvalidated on this checkpoint — measure before enabling.

## Speculative decoding (#75): before-state and scope

**Audience:** lobes operators/maintainers and the Culture mesh that consumes the
`multimodal` (Gemma 4 12B) generate lane — i.e. anyone calling `model=multimodal`
or the back-compat `model=normal` (see [Tier alias usage](#tier-alias-usage)).

**Before-state (verified in-repo).** The gemma catalog entry
(`lobes/catalog.py`, `id="sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4"`)
carries **no `speculative_config`** today: `SupportedModel.speculative_config`
defaults to `speculative_config: str = ""`, and the gemma entry never overrides
it — the comment directly above its closing paren reads "No speculative_config:
despite the '-MTP' name, this unified checkpoint exposes no gemma4_assistant
draft...". Compose mirrors this: the only `--speculative-config` item in
`lobes/templates/docker-compose*.yml` belongs to the 27B `vllm` (primary)
service (`'--speculative-config={"method": "qwen3_5_mtp", ...}'`); the
`vllm-multimodal` service carries none — see the compose-flags block above and
the `# NO --speculative-config` comment in it. The native self-speculation path
is also closed: `{"method": "gemma4_mtp"}` is rejected by vLLM 0.21/0.22 as an
"Unsupported speculative method" (see the ["No native MTP via this
checkpoint"](#what-it-is) bullet above; the [DSpark
experiment](#dspark-experiment) above is the one currently-documented, but
disabled and unvalidated, alternative route).

**The gap this leaves.** The 27B `primary`/`main` gear gets a measured **~2.4×
single-stream decode speedup** from MTP speculative decoding (72–79 % draft
acceptance) — see
[`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md). The
`multimodal`/`normal` lane has no equivalent boost: it serves with no
speculative config at all, so per-stream decode is comparatively slow relative
to the primary, and that gap is real (not assumed) once #71 lands and the lane
takes live mesh traffic.

**Scope split.**

| Concern | Owner |
|---|---|
| Serve-enablement — force `TRITON_ATTN` on the transformers backend so the gear actually serves | issue #71 (see [Live-validation status](#live-validation-status-71) below) |
| Draft-model training/distilling (building a native `gemma4_assistant` head from scratch) | a separate follow-up, not #75 |
| Resolve a draft route, wire `--speculative-config`, measure, and decide | **issue #75 (this work)** |

Issue #75 does not train a draft model. It resolves to exactly one concrete
route (a validated `draft_model` such as DSpark, a sourced `gemma4_assistant`
draft, or a documented "no compatible draft available"), wires it through the
same catalog-to-compose pattern the 27B MTP primary uses today
(`speculative_config` on the catalog entry drives the compose
`--speculative-config` items), measures draft acceptance and decode speedup on
a live co-resident serve, and commits a verdict: restore `speculative_config`
by default if it beats the no-spec baseline, or document the negative with the
numbers that ruled it out. **Done = a measured verdict, not merely a wired
draft.** #75 is gated on #71 — no draft can be measured until the gear serves.

## Live-validation status (#71/#73) {#live-validation-status-71}

> **RESOLVED on the DGX Spark (GB10, sm_121, 2026-07-01, #71/#73): the gear
> SERVES and answers text + image + audio requests.** `status` = `load-tested`.

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
| **`vllm/vllm-openai:nightly` (shipped)** | **0.23.1rc1.dev** | **✅** | **✅ text+image+audio** |

**Live validation results (util 0.25, `--max-model-len 4096`, GB10):**

- ✅ **Serves** — `/health` 200, native class resolved, TRITON auto-forced.
- ✅ **Text** — arithmetic + factual answered correctly.
- ✅ **Image + text** — described a test image (red circle + text) correctly.
- ✅ **Audio + text** — transcribed a 24 kHz TTS clip **verbatim** (needed `av`).
- ✅ **GPU util** — ~**15.7 GiB** actual (weights 8.1 + cudagraph 0.46 + KV 7.2) ≈
  **0.12** of the 128 GB GB10 → fits the 0.69 default-fleet budget.

**Two config gotchas (vLLM 0.23), now handled in the compose/env:**

- **Cudagraph memory over-estimate.** vLLM 0.23 reserves an *estimated* cudagraph
  headroom inside the util budget — here **12.74 GiB estimated vs 0.46 GiB actual**
  — which starves the KV cache so `util 0.12` fails with *"No available memory for
  the cache blocks"*. The compose sets `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
  so the util knob maps to true usage.
- **Context vs util.** At `util 0.12` the KV cache holds only ~24K tokens, so the
  128K native context is **not** serveable at the co-resident lane budget —
  `MULTIMODAL_MAX_MODEL_LEN` defaults to `8192` (raise it and the util together for
  more). 128K would need a much larger util than the lane allows.

Resolved:

- ✅ **Serve-enablement** — native `gemma4_unified` class on vLLM nightly + `TRITON_ATTN`.
- ✅ **Image + audio** — functional (audio needs the `vllm[audio]` extra / PyAV).
- ✅ **Quantization** — `compressed-tensors` (not `modelopt_fp4`).
- ✅ **GPU util** — ~15.7 GiB ≈ 0.12 budget.
- ⛔ **Native MTP** — still needs a separate `gemma4_assistant` draft model (scoped in #75, closed); the gear serves without spec-decode.

## Related docs

- [`gateway-fleet.md`](gateway-fleet.md) — fleet topology (main/minor/multimodal),
  tier alias routing, pressure policy, memory budget.
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  hard/`primary` gear (27B MTP).
- [`qwen3-14b-nvfp4.md`](qwen3-14b-nvfp4.md) — the legacy 14B candidate (demoted
  from the `normal` tier; no tier alias resolves to it).
