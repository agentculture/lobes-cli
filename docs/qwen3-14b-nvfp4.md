# Qwen3-14B-NVFP4 — "middle" tier (normal)

> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

**Model id:** `nvidia/Qwen3-14B-NVFP4`
**Tier alias:** `normal` (resolves here via `model=normal` at the gateway)
**Role:** `middle` — balanced capability and cost between the 4B `minor` (cheap)
and the 27B MTP `primary` (hard)
**Status:** `configured` — not yet load-tested on the DGX Spark (issue #68, t9)

## What it is

The 14B dense NVFP4 checkpoint from NVIDIA is the fleet's **middle/normal tier**:
more capable than the 4B companion but cheaper to run than the full 27B primary.
It is a pure **inference** gear — not a LoRA base. LoRA training stays on
`Qwen/Qwen3.5-4B` (the bf16 minor lobe); there is no `lobes train` verb.

Architecture highlights:

- **Dense** transformer (no MoE, no MTP draft head) — the same class as
  `nvidia/Qwen3-32B-NVFP4`.
- **`modelopt_fp4`** quantization (`--quantization modelopt_fp4`).
- **Hermes-style JSON tool calls** (`--tool-call-parser hermes`).
- **32K native context** (`--max-model-len 32768` default); extendable to
  131 072 tokens via an explicit YaRN `--rope-scaling` override — unverified
  on this hardware.
- **No `--hf-overrides`, `--moe-backend`, or `--speculative-config`** —
  straightforward dense load; no extra compose edits needed beyond the standard
  fleet flags.

The checkpoint id `nvidia/Qwen3-14B-NVFP4` is an **accepted plan risk** — verify
it loads on the `nv26.04` vLLM image before promoting status to `load-tested`.
See issue #68 (t9) for the live-validation task.

## Serving (fleet)

The middle gear is an **opt-in compose profile** in the fleet, not an always-warm
backend. Activate it with:

```bash
# In .env (or export in shell):
COMPOSE_PROFILES=middle
# Or pass at run time:
docker compose --profile middle up -d
```

Once active, uncomment `MIDDLE_BASE_URL` in `.env` so the gateway routes `normal`
requests here:

```env
MIDDLE_BASE_URL=http://vllm-middle:8000   # uncomment to activate gateway routing
```

Key env vars (from `env.example`):

| Variable | Default | Notes |
|---|---|---|
| `MIDDLE_MODEL` | `nvidia/Qwen3-14B-NVFP4` | HF checkpoint id |
| `MIDDLE_SERVED_NAME` | `nvidia/Qwen3-14B-NVFP4` | OpenAI `model` id the gateway routes to |
| `MIDDLE_GPU_MEM_UTIL` | `0.12` | ~15 GiB on the 128 GB GB10 |
| `MIDDLE_MAX_MODEL_LEN` | `32768` | 32K default; verify before raising |
| `MIDDLE_BASE_URL` | *(commented out)* | Uncomment to wire the gateway |

Compose flags used by the `vllm-middle` service:

```text
--model ${MIDDLE_MODEL}
--served-model-name ${MIDDLE_SERVED_NAME}
--quantization ${MIDDLE_QUANTIZATION:-modelopt_fp4}
--max-model-len ${MIDDLE_MAX_MODEL_LEN:-32768}
--gpu-memory-utilization ${MIDDLE_GPU_MEM_UTIL:-0.12}
--tool-call-parser ${MIDDLE_TOOL_CALL_PARSER:-hermes}
```

## GPU memory budget

When the middle gear is active alongside the rest of the fleet:

| Gear | `--gpu-memory-utilization` | Approx GiB |
|---|---|---|
| `primary` (27B MTP, 128K) | 0.45 | ~56 |
| `middle` (14B NVFP4) | **0.12** | ~15 |
| `minor` (4B bf16) | 0.10 | ~13 |
| `embed` (0.6B) | 0.06 | ~7 |
| `rerank` (0.6B) | 0.06 | ~7 |
| **Total** | **0.79** | ~98 / 128 GB |

Adding the middle gear required reducing the primary's util from 0.6 → 0.45 and
trimming its context from 256K → 128K (`PRIMARY_MAX_MODEL_LEN=131072`). If you run
without the middle gear, restore `PRIMARY_GPU_MEM_UTIL=0.6` and optionally raise
`PRIMARY_MAX_MODEL_LEN=262144` for the full 256K context.

## Tier alias usage

Callers use capability-tier aliases instead of hardcoded model ids:

| Alias | Routes to | Fallback when absent |
|---|---|---|
| `cheap` | 4B `minor` | middle, else primary |
| `normal` | **14B `middle`** | primary |
| `hard` | 27B `primary` | always present |

Send `model=normal` and the gateway resolves to this gear when it is wired and
healthy. If the middle backend is not started, `normal` falls back upward to the
primary — the caller's code is unchanged.

```python
# Before: hardcoded model id
response = client.chat.completions.create(
    model="nvidia/Qwen3-14B-NVFP4",
    messages=[{"role": "user", "content": "..."}],
)

# After: tier alias — the gateway resolves to the right gear
response = client.chat.completions.create(
    model="normal",
    messages=[{"role": "user", "content": "..."}],
)
```

## LoRA scope

LoRA adapter training uses the **4B bf16 minor lobe** (`Qwen/Qwen3.5-4B`) as its
base. The 14B NVFP4 middle is:

- **Inference-only** in this fleet configuration.
- **Not a LoRA base** — NVFP4 quantization is not unsloth-LoRA-compatible.
- Not the target of any `lobes train` verb (no such verb exists).

## Related docs

- [`gateway-fleet.md`](gateway-fleet.md) — three-tier topology, tier alias routing,
  pressure policy, memory budget.
- [`qwen3.5-4b-minor.md`](qwen3.5-4b-minor.md) — the cheap/`minor` gear and the
  LoRA fine-tune target.
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  hard/`primary` gear.
- [`qwen3-32b-nvfp4.md`](qwen3-32b-nvfp4.md) — the dense 32B candidate (same
  architecture class as this 14B).
