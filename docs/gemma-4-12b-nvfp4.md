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

> **Status: #75 is CLOSED — scoped, not implemented.** The draft route was
> resolved (see [`gemma4-mtp-draft.md`](gemma4-mtp-draft.md)), but the
> wire → measure → verdict legs never landed: the `multimodal` lane still serves
> with **no `speculative_config`**, and the benchmark below (~23 tok/s
> single-stream, no draft) *is* that no-spec baseline. Reviving speculative
> decoding for this gear is **new work** (a future issue), not #75. The scope
> recorded here is retained as the historical framing.

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
`multimodal`/`normal` lane has no equivalent multiplier: it serves with no
speculative config at all. In *absolute* terms the 12B still out-decodes the
primary single-stream (~23 vs ~18–19 tok/s — see the benchmark below) because it
is under half the parameters; the gap is against its own *potential* — a working
draft would push it well past 23 tok/s — not against the primary. Now that #71
has landed and the lane is measured, that potential gap is concrete, not assumed.

**Scope split.**

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

## Benchmark — 2026-07-01, DGX Spark (GB10), standalone

> First throughput/prefill numbers for the gear. Measured on the shared GB10
> **standalone** (own container on host port 8010, **not** co-resident behind the
> gateway) so the live 27B primary kept serving mesh traffic. Image
> `lobes/vllm-gemma4:nightly-audio` (vLLM **0.23.1rc1.dev672**, native
> `Gemma4UnifiedForConditionalGeneration`, `TRITON_ATTN`, `compressed-tensors`
> NVFP4, `--max-model-len 8192`). Driven by `lobes benchmark` + a manual
> forced-decode probe.

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
| Speculative decoding | **none** — no MTP draft (see ["No native MTP"](#what-it-is)); this is raw single-stream decode |

**Decode context.** At ~**23 tok/s single-stream**, the 12B `multimodal` lane is
actually a touch *faster* per-stream than the 27B `primary` (~18–19 tok/s, and
that is *with* its ~2.4× MTP boost) — it is less than half the parameters. So the
lane with **no spec-decode** still out-decodes the primary single-stream, while
adding native vision+audio; the primary remains the more capable text model. The
speculative-decoding gap called out in
[Speculative decoding (#75)](#speculative-decoding-75-before-state-and-scope)
is about closing the *potential* gap (a draft would push the 12B well past 23
tok/s), not a current regression.

**Config note — why not the production `util 0.12`.** The default fleet lane runs
`MULTIMODAL_GPU_MEM_UTIL=0.12` with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.
That combination was validated on #71's earlier nightly digest; the **current**
`:nightly-audio` build (dev672) behaves differently in two ways that forced a
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

> **Follow-up (config drift):** the default lane's `util 0.12` +
> `…ESTIMATE_CUDAGRAPHS=0` should be re-validated against the pinned dev672 image
> and, if the accounting change sticks, either the util raised or the capture set
> trimmed in the compose so the gear boots 8,192 co-resident. Tracked with the
> serve config in [`gemma4-mtp-draft.md`](gemma4-mtp-draft.md) / the fleet compose.

## Related docs

- [`gateway-fleet.md`](gateway-fleet.md) — fleet topology (main/minor/multimodal),
  tier alias routing, pressure policy, memory budget.
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  hard/`primary` gear (27B MTP).
- [`qwen3-14b-nvfp4.md`](qwen3-14b-nvfp4.md) — the legacy 14B candidate (demoted
  from the `normal` tier; no tier alias resolves to it).
