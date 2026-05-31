# MTP candidate: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`

A **text-only MTP candidate** — the 27B primary re-exported so vLLM **speculative
decoding (Multi-Token Prediction)** actually works. The fleet's default primary
(`mmangkad/Qwen3.6-27B-NVFP4`, [`qwen3.6-27b-nvfp4.md`](qwen3.6-27b-nvfp4.md)) is
slow single-stream on the GB10 (~8 tok/s decode); MTP drafts several tokens per
forward pass and is the lever that speeds that up. Filed as
[issue #26](https://github.com/agentculture/model-gear/issues/26).

> **Status: configured, not yet load-tested on this GB10.** It is in the
> **supported catalog** (`model overview --list`) as a candidate. The numbers
> below are the model card's (on other hardware) plus our hypotheses — replace
> them with measured values once the live test runs (checklist below). For the
> catalog-vs-warm distinction — what you *can* load vs. what's loaded *now* — see
> [`gateway-fleet.md`](gateway-fleet.md#supported-catalog-vs-warm-backends).

Source: <https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP>.

## Why a different checkpoint at all (the 35B lesson)

We tried to enable MTP the obvious way first and learned the constraint is the
**checkpoint, not the engine** — the same wall the
[35B MoE](qwen3.6-35b-a3b-nvfp4.md#why-we-serve-the-mmangkad-copy-not-nvidia-vllm-version-2026-05-31)
hit:

- A **baseline NVFP4 export drops the MTP draft head.** Both the original
  `unsloth/Qwen3.6-27B-NVFP4` and `mmangkad/Qwen3.6-27B-NVFP4` are plain NVFP4
  quantizations; per the model cards the MTP layer is discarded during export, so
  vLLM speculative decoding gets **~0 % draft acceptance** (and the 35B
  `mmangkad/` copy fails to load the draft outright — `qwen3_5_mtp.py`
  weight-shape mismatch).
- The fix the 35B doc points at — *a newer vLLM with the right loader* — is **not
  installable on this aarch64 GB10** yet (cutlass-dsl conflicts ≥ 0.22; the image
  stays NGC `26.04-py3` / vLLM `0.19.0`).

So the working path is the **other half of the lesson: serve a checkpoint that
ships the MTP draft weights.** `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` is the
27B with its **MTP head grafted back in bf16** (~850 MB, 15 tensors) specifically
for vLLM — no newer engine required.

## What it is

- An **NVFP4 (NVIDIA ModelOpt) re-export of `Qwen/Qwen3.6-27B`** with the **MTP
  draft head restored in bf16** for vLLM speculative decoding (the baseline export
  dropped it). 27.78B params, 64 layers + 1 MTP layer, 256K native context.
- **Text-only:** the ViT vision tower is physically deleted from this build, so it
  serves with `--language-model-only` and `AutoModelForCausalLM` (no image path).
  The fleet runs the 27B text-only anyway, so nothing is lost for our use.
- **Quantization is `modelopt`** (not `modelopt_fp4`) per the card — NVFP4 on the
  native SM120 fast path. `lm_head`, the Mamba `conv1d` SSM convolutions, and all
  `mtp.*` modules are kept in bf16. The catalog sets `VLLM_QUANTIZATION=modelopt`;
  **verify on first load** (if vLLM rejects it, try `modelopt_fp4`).

## Is it supported here? — to be confirmed (the #1 risk)

The MTP method this checkpoint declares is **`qwen3_5_mtp`** (vLLM normalises it to
`mtp`). The card recommends **vLLM 0.19.1rc1+**; our image is `0.19.0+...nv26.04`,
right at that edge. In its favour, the primary 27B doc records that the same image
already *registers* `Qwen3_5ForConditionalGeneration` **plus the MoE and MTP
variants** — so the architecture is known to the engine. What is unproven is
whether the nv26.04 patch accepts the `qwen3_5_mtp` **speculative method**. Confirm
before relying on it:

```text
# does this engine know the speculative method / MTP arch?
docker exec model-gear-vllm python3 -c \
  "from vllm.model_executor.models.registry import ModelRegistry; \
   print([a for a in ModelRegistry.get_supported_archs() if 'MTP' in a or 'Mtp' in a])"
```

If the engine rejects `qwen3_5_mtp`, MTP waits for a newer aarch64 vLLM (see the
35B doc) and this stays a configured-but-non-MTP candidate.

## How to run (compose edits `model switch` prints)

`model switch` writes the `VLLM_*` keys to `.env`, but the MTP draft and the
text-only flags **can't be defaulted in the shared template** (compose can't omit
an empty flag, and they break the dense/hybrid models). So `model switch` resolves
the env and then **prints the exact compose `command:` lines to add by hand** —
the same mechanism the MoE `--moe-backend` uses:

```bash
model switch sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --port 8001 --apply
# auto-selects VLLM_TOOL_CALL_PARSER=qwen3_coder + VLLM_QUANTIZATION=modelopt,
# then prints: add to the compose `command` by hand —
#   --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}'
#   --trust-remote-code
#   --language-model-only
# and set VLLM_MAX_NUM_SEQS=2  (see the OOM caveat below)
```

Reference serve recipe (from the card) once those lines are in the compose file:

```bash
# Production (256K context, KV FP8, MTP n=3)
vllm serve sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
  --trust-remote-code --language-model-only \
  --quantization modelopt --kv-cache-dtype fp8 \
  --max-model-len 262144 --max-num-seqs 2 \
  --gpu-memory-utilization 0.9 --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'

# Smaller context (16K) for the fastest single-request decode — same flags,
#   --max-model-len 16384  (and drop --kv-cache-dtype fp8 if it OOMs on warmup)
```

On the **shared GB10** keep `--gpu-memory-utilization` at the machine profile's
`0.6` (not `0.9` — the audio NIMs + reachy hold the rest of the 121.7 GiB), and
the first load should cap `--max-model-len 32768`.

## Caveats — to validate on first load

1. **`--max-num-seqs 2` is load-bearing.** The card warns that
   `--max-num-seqs 4` + KV-FP8 + `n=3` + 256K context **silently OOMs during
   CUDA-graph capture**. No `--purpose` profile yields 2 (balanced/prompt-heavy=4,
   decode-heavy=8), so set `VLLM_MAX_NUM_SEQS=2` in `.env` by hand for this model.
2. **Quantization label.** The card uses `--quantization modelopt`; the catalog
   defaults the nvidia/mmangkad NVFP4 checkpoints to `modelopt_fp4`. If vLLM
   rejects `modelopt`, fall back to `modelopt_fp4` (override:
   `model switch … --quantization modelopt_fp4`).
3. **`--trust-remote-code` is required** (this repo ships custom modeling code) —
   the shared template omits it deliberately, hence the compose edit.
4. **MTP + reasoning + structured output.** This build serves with
   `--reasoning-parser qwen3`; MTP has a known vLLM edge case where speculation
   can break `</think>` detection under structured output
   ([vLLM #34650](https://github.com/vllm-project/vllm/issues/34650)). Watch the
   reasoning trace + any JSON-schema responses during the assess run.
5. **Tool calling untested on this build.** The card documents reasoning, not
   tool calls. The catalog sets `qwen3_coder` (as the primary 27B uses); confirm
   the `tool_choice:"auto"` probe still returns a `finish` call (`model switch`
   runs it unless `--no-probe`).

## Live-test checklist (run on the DGX Spark — out of band)

1. **Confirm the premise (unsloth, expected to fail MTP).**
   `model switch unsloth/Qwen3.6-27B-NVFP4 --apply` + the MTP compose edit →
   measure draft acceptance; expect **~0 %** (baseline export dropped the MTP
   head). This is issue #26's literal ask, and it justifies the re-export.
2. **The real test (sakamakismile).** `model switch
   sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --apply`, add the printed compose
   lines, set `VLLM_MAX_NUM_SEQS=2`, `model status` → `/health`, then
   `model assess` (correctness + tool probe) and `model benchmark` for decode
   tok/s **and the MTP draft-acceptance rate**, against the `mmangkad/` baseline.
   Fill the table below.
3. **Risk gate.** If the engine rejects `qwen3_5_mtp` (method not in nv26.04),
   record it here and keep MTP deferred (newer aarch64 vLLM needed).

## Benchmark — pending (not yet load-tested on this GB10)

| Property | Value |
|---|---|
| Health / `max_model_len` | *pending* |
| Engine accepts `qwen3_5_mtp` | *pending* (the #1 risk) |
| Correctness / reasoning trace | *pending* |
| Decode throughput (batch=1) | *pending* (card: ~1.74× the baseline NVFP4) |
| MTP draft acceptance | *pending* (card: ~57–84 %, workload-dependent) |
| Tool calling (`tool_choice:auto`) | *pending* |
| GPU memory reserved | *pending* |

Card reference (RTX PRO 6000 Blackwell, 256K + KV-FP8 + n=3): 2-parallel
aggregate ~189–207 tok/s, **~1.74× faster** than the baseline `Qwen3.6-27B-NVFP4`.
Those are off-box numbers at concurrency 2 — our single-stream GB10 figures will
differ; measure, don't assume.

## Recommendation

If the live test confirms `qwen3_5_mtp` loads on the nv26.04 image and MTP gives a
real single-stream decode speedup over the `mmangkad/` baseline, this is the strong
candidate to **promote to primary** (text-only is no loss for the fleet). Until
then `mmangkad/Qwen3.6-27B-NVFP4` stays the primary and this is a catalogued
candidate to benchmark. Re-run `model assess` / `model benchmark` after any vLLM
image bump — a newer engine could also unblock MTP on the `nvidia/` checkpoints
(see the 35B follow-up).
