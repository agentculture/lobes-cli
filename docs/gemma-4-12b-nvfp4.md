# Gemma 4 12B-NVFP4 — "multimodal" tier (normal)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

**Model id:** `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`
**Tier alias:** `multimodal` (and `normal` back-compat; resolves here via `model=multimodal` or `model=normal` at the gateway)
**Role:** `multimodal` — unified text+image+audio gear, the fleet's new "normal" tier
**Status:** `configured` — the custom image **loads** it, but it does not yet
**serve** (issue #71 serve-enablement; see ["Live-validation status"](#live-validation-status-71) below)

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
services. The measured util for the multimodal gear (vision+audio KV) is an
accepted plan risk — confirm on the Spark in t7.

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

## Live-validation status (#71) {#live-validation-status-71}

> **Live validation on the DGX Spark (`spark-f8a9`, GB10, 2026-06-30, #71)
> established that the custom image LOADS the gear but it does not yet SERVE.**
> `status` stays `configured`. Serve-enablement is tracked as follow-ups.

**What the custom image fixed.** Gemma 4 12B's `model_type: gemma4_unified` is
registered by no *stock* NGC image. The custom image
([`Dockerfile.vllm-gemma4`](../lobes/templates/fleet/Dockerfile.vllm-gemma4)) —
`FROM nvcr.io/nvidia/vllm:26.06-py3` (vLLM 0.22.1, NGC torch 2.13.0a0) + a
from-source Transformers pinned to `181beb3` (5.13.0.dev0) — **registers
`gemma4_unified` and loads the weights**. The Transformers overlay swaps only
Transformers + safetensors; NGC's Blackwell torch is preserved.

Runtime matrix tested:

| Base image | vLLM | torch | `gemma4_unified` registers | serves |
|---|---|---|---|---|
| `nvcr.io/nvidia/vllm:26.04-py3` | 0.19.0 | — | ❌ (stock) | — |
| `…:26.05.post1-py3` + Transformers `181beb3` | 0.21.0 | NGC 2.12.0a0 | ✅ | ❌ (see below) |
| `…:26.06-py3` + Transformers `181beb3` **(shipped)** | 0.22.1 | NGC 2.13.0a0 | ✅ | ❌ (see below) |
| host venv (out-of-docker) | 0.23.1rc1.dev | stock 2.11.0+cu130 | ✅ | not run |

**Why it does not serve yet (the serve-enablement follow-up).** The model loads
but crashes at warmup forward with `RuntimeError: Shape mismatch: a.size(1)=4096,
size_k=8192` in the o_proj GEMM. Root cause: Gemma 4's **non-square attention**
(`global_head_dim=512` ≠ `head_dim=256`) — the o_proj expects
`num_heads×global_head_dim = 8192`, but FlashAttention emits
`num_heads×head_dim = 4096`. The fix is `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
(see [ai-muninn.com's recipe](https://ai-muninn.com/en/blog/dgx-spark-gemma4-12b-omni-nvfp4-weight-only),
which serves `coolthor/gemma-4-12B-it-NVFP4A16` this way), but vLLM runs
`gemma4_unified` via its *transformers-modeling backend*, which did **not** honor
the env var in our runs (it still selected `FLASH_ATTN`). The compose sets the env
in anticipation; making it engage is the open follow-up.

Resolved vs open:

- ✅ **Image / arch registration** — custom image registers `gemma4_unified`, loads weights.
- ✅ **Quantization** — `compressed-tensors` (not `modelopt_fp4`).
- ✅ **Native context** — 128K (`text_config.max_position_embeddings=131072`).
- ⛔ **Serve-enablement** — force `TRITON_ATTN` on the transformers backend (and/or
  validate the blog's `coolthor` checkpoint, possibly switching the catalog default).
- ⛔ **Native MTP** — needs a separate `gemma4_assistant` draft model; not enabled.
- ⏳ **GPU util / functional image+audio** — measurable only once it serves.

## Related docs

- [`gateway-fleet.md`](gateway-fleet.md) — fleet topology (main/minor/multimodal),
  tier alias routing, pressure policy, memory budget.
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  hard/`primary` gear (27B MTP).
- [`qwen3-14b-nvfp4.md`](qwen3-14b-nvfp4.md) — the legacy 14B candidate (demoted
  from the `normal` tier; no tier alias resolves to it).
