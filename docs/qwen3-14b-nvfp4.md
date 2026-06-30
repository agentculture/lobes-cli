# Qwen3-14B-NVFP4 — legacy candidate (demoted)

> **DEMOTED (issue #69):** This gear is no longer the fleet's `normal`/middle
> tier. The `normal` tier alias now resolves to the **Gemma 4 12B multimodal
> gear** (`sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`,
> [`docs/gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md)), which is **default-on**.
> The 14B is kept as a **legacy `candidate`** — selectable explicitly by model id
> or via `COMPOSE_PROFILES=middle` — but no tier alias resolves to it.
>
> One entry in lobes's **supported catalog** (`lobes overview --list`). For
> the catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* —
> see [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

**Model id:** `nvidia/Qwen3-14B-NVFP4`
**Tier alias:** none — demoted to legacy candidate (issue #69). Use the full model
id or `COMPOSE_PROFILES=middle` to activate. The `normal` back-compat alias now
routes to `multimodal` (the Gemma 4 12B gear).
**Role:** `candidate` (formerly `middle`) — kept as a selectable legacy option;
no longer the `normal`/middle tier between `minor` and `main`
**Status:** `configured` — not yet load-tested on the DGX Spark (issue #68, t9)

## What it is

The 14B dense NVFP4 checkpoint from NVIDIA was the fleet's **middle/normal tier**,
but is now a **legacy candidate** (demoted in issue #69). The `normal`/`multimodal`
slot is now filled by the Gemma 4 12B unified-multimodal gear. The 14B is kept in
the catalog for operators who explicitly need a text-only Qwen 14B, but no tier
alias routes to it automatically. It is a pure **inference** gear — not a LoRA
base. LoRA training stays on `Qwen/Qwen3.5-4B` (the bf16 minor lobe); there is no
`lobes train` verb.

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

Once active, uncomment `MIDDLE_BASE_URL` in `.env` so the gateway routes explicit
`nvidia/Qwen3-14B-NVFP4` model-id requests here (note: the `normal` alias no
longer routes to this gear — it routes to the Gemma 4 12B `multimodal` gear):

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

## Tier alias usage (legacy — no alias resolves to this gear)

As of issue #69, **no tier alias resolves to this gear**. The `normal` back-compat
alias now routes to the Gemma 4 12B `multimodal` gear. To use this gear explicitly,
address it by model id:

```python
# Use the full model id directly — no tier alias routes here:
response = client.chat.completions.create(
    model="nvidia/Qwen3-14B-NVFP4",
    messages=[{"role": "user", "content": "..."}],
)
```

Current tier alias table (for reference):

| Alias | Back-compat alias | Routes to | Notes |
|---|---|---|---|
| `main` | `hard` | 27B `primary` | always present |
| `minor` | `cheap` | 4B `minor` | opt-in (`--profile minor`) |
| `multimodal` | `normal` | Gemma 4 12B `multimodal` | default-on; replaced `middle`/`normal` in #69 |

The 14B is activated via `COMPOSE_PROFILES=middle` and addressed by its full model id.

## LoRA scope

LoRA adapter training uses the **4B bf16 minor lobe** (`Qwen/Qwen3.5-4B`) as its
base. The 14B NVFP4 middle is:

- **Inference-only** in this fleet configuration.
- **Not a LoRA base** — NVFP4 quantization is not unsloth-LoRA-compatible.
- Not the target of any `lobes train` verb (no such verb exists).

## Related docs

- [`gateway-fleet.md`](gateway-fleet.md) — fleet topology, tier alias routing,
  pressure policy, memory budget.
- [`gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md) — the Gemma 4 12B multimodal
  gear that now fills the `normal`/`multimodal` tier slot.
- [`qwen3.5-4b-minor.md`](qwen3.5-4b-minor.md) — the cheap/`minor` gear and the
  LoRA fine-tune target.
- [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md) — the
  hard/`primary` gear.
- [`qwen3-32b-nvfp4.md`](qwen3-32b-nvfp4.md) — the dense 32B candidate (same
  architecture class as this 14B).
