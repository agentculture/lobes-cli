# Qwen3-14B-NVFP4 — middle tier candidate

**Model id:** `nvidia/Qwen3-14B-NVFP4`
**Role:** middle tier (`cheap` / **`normal`** / `hard`)
**Status:** configured — not yet load-tested on the DGX Spark (issue #68, t9)

## Overview

The 14B dense NVFP4 checkpoint is the fleet's **normal/middle tier**: balanced
capability and memory cost sitting between the 4B `minor` (cheap, fast) and the
27B MTP `primary` (hard, full capability).

Architecture mirrors `nvidia/Qwen3-32B-NVFP4` — dense (no MoE, no MTP draft
head), `modelopt_fp4` quantization, Hermes-style JSON tool calls (`hermes`
parser), 32K native context (extendable to 131K via YaRN rope-scaling).

## Serve parameters

| Parameter | Value |
|-----------|-------|
| `--quantization` | `modelopt_fp4` |
| `--tool-call-parser` | `hermes` |
| `--max-model-len` | `32768` (default; `131072` with YaRN) |

## Notes

- The exact HF checkpoint id (`nvidia/Qwen3-14B-NVFP4`) is an accepted plan
  risk — verify it loads on the nv26.04 vLLM image before promoting to
  `load-tested`. See issue #68 (t9) for the live-validation task.
- LoRA training stays on the 4B `minor` (`Qwen/Qwen3.5-4B`); the 14B is
  inference-only in this fleet configuration.
- See `docs/gateway-fleet.md` for the three-tier memory budget and alias
  routing (`model=normal` → this gear).
