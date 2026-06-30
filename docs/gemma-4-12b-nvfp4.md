# Gemma 4 12B-NVFP4 — "multimodal" tier (normal)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

**Model id:** `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`
**Tier alias:** `multimodal` (and `normal` back-compat; resolves here via `model=multimodal` or `model=normal` at the gateway)
**Role:** `multimodal` — unified text+image+audio gear, the fleet's new "normal" tier
**Status:** `configured` — not yet load-tested on the DGX Spark (issue #69, t7)

## What it is

The Gemma 4 12B is the fleet's **unified multimodal** gear: a single checkpoint
serving text, image, and audio through `Gemma4UnifiedForConditionalGeneration` —
no separate vision or audio sidecars needed. It ships a **native MTP draft head**
for vLLM speculative decoding and is quantized to **NVFP4** (`modelopt_fp4`).

Architecture highlights:

- **Unified multimodal** (`Gemma4UnifiedForConditionalGeneration`) — text + IMAGE
  + AUDIO in one checkpoint; the generate lane gains native vision and audio
  without the realtime overlay.
- **Native MTP draft head** (`--speculative-config` with `gemma4_mtp` method) —
  speculative decoding baked into the checkpoint, like the 27B primary's
  `qwen3_5_mtp` config.
- **`modelopt_fp4`** quantization (`--quantization modelopt_fp4`).
- **Pythonic tool calls** (`--tool-call-parser pythonic`).
- **128K native context** (`--max-model-len 131072`); the exact native context
  is an accepted plan risk pending t7 live validation on the Spark.
- **No `--language-model-only`** — vision and audio towers are active by default;
  adding that flag would disable them and defeat the headline capability.

The checkpoint id `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`
is an **accepted plan risk** — verify it loads on the `nv26.04` vLLM image before
promoting status to `load-tested`. See issue #69 (t7) for the live-validation task.

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
| `MULTIMODAL_MAX_MODEL_LEN` | `131072` | 128K default; verify before raising |
| `MULTIMODAL_SPECULATIVE_CONFIG` | `{"method": "gemma4_mtp", "num_speculative_tokens": 3}` | Native MTP draft head |

Compose flags used by the `vllm-multimodal` service:

```text
--model ${MULTIMODAL_MODEL}
--served-model-name ${MULTIMODAL_SERVED_NAME}
--quantization ${MULTIMODAL_QUANTIZATION:-modelopt_fp4}
--max-model-len ${MULTIMODAL_MAX_MODEL_LEN:-131072}
--gpu-memory-utilization ${MULTIMODAL_GPU_MEM_UTIL:-0.12}
--tool-call-parser=pythonic
--speculative-config=${MULTIMODAL_SPECULATIVE_CONFIG:-{"method": "gemma4_mtp", "num_speculative_tokens": 3}}
--trust-remote-code
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

The DSpark experiment is **disabled by default**. An environment variable
override can swap in the DeepSeek DSpark checkpoint:

```env
MULTIMODAL_MODEL=deepseek-ai/dspark_gemma4_12b_block7
```

This replaces the Gemma 4 12B checkpoint with the DSpark variant for
experimental evaluation. The default fleet does not use this override.

## Accepted plan risks

The following items are pending t7 live validation on the DGX Spark:

1. **Exact checkpoint pick** — confirm `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`
   loads correctly on the `nv26.04` vLLM image.
2. **`gemma4_mtp` method string** — verify the native-MTP `--speculative-config`
   method against the served checkpoint (the catalog uses `"gemma4_mtp"` as the
   method; confirm it matches the checkpoint's draft head).
3. **131072 native context** — confirm the 128K native context window; the catalog
   uses this as a safe default until measured.
4. **Measured GPU utilization** — confirm the `0.12` util budget accounts for
   vision+audio KV cache overhead (the 14B middle used `0.12` for text-only;
   multimodal KV may differ).

## Related docs

- [`gateway-fleet.md`](gateway-fleet.md) — three-tier topology, tier alias routing,
  pressure policy, memory budget.
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  hard/`primary` gear (27B MTP).
- [`qwen3-14b-nvfp4.md`](qwen3-14b-nvfp4.md) — the legacy 14B candidate (demoted
  from the `normal` tier; no tier alias resolves to it).
