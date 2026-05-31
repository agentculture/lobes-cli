# MTP candidate: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`

A **text-only MTP candidate** — the 27B primary re-exported so vLLM **speculative
decoding (Multi-Token Prediction)** actually works. The fleet's default primary
(`mmangkad/Qwen3.6-27B-NVFP4`, [`qwen3.6-27b-nvfp4.md`](qwen3.6-27b-nvfp4.md)) is
slow single-stream on the GB10 (~8 tok/s decode); MTP drafts several tokens per
forward pass and is the lever that speeds that up. Filed as
[issue #26](https://github.com/agentculture/model-gear/issues/26).

> **Status: load-tested 2026-05-31 on this GB10 — MTP works.** Loaded and served
> on `spark-f8a9` (vLLM `0.19.0+nv26.04`) at **19.1 tok/s decode (~2.4× the
> baseline 27B's ~8 tok/s)** with **72 % MTP draft acceptance** — see the
> benchmark table below. One packaging fix was needed (a tokenizer override —
> caveat 1). It is in the **supported catalog** (`model overview --list`) as a
> candidate. For the catalog-vs-warm distinction — what you *can* load vs. what's
> loaded *now* — see
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

## Is it supported here? — yes, confirmed (2026-05-31)

The MTP method this checkpoint declares is **`qwen3_5_mtp`**, and **vLLM
`0.19.0+nv26.04` accepts it** — this was the open risk, now resolved. On load the
engine logs:

```text
WARNING [speculative.py:368] method `qwen3_5_mtp` is deprecated and replaced with mtp.
INFO    [model.py:549] Resolved architecture: Qwen3_5ForConditionalGeneration
INFO    [model.py:549] Resolved architecture: Qwen3_5MTP
INFO    [core.py:105] ... speculative_config=SpeculativeConfig(method='mtp', num_spec_tokens=3)
```

Both the target (`Qwen3_5ForConditionalGeneration`) and the **draft head**
(`Qwen3_5MTP`) resolve on the stock image — no newer engine needed. Live draft
acceptance came out at **72 %** (see the benchmark table).

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

## Caveats — validated on the live load (2026-05-31)

1. **Tokenizer override required (load-blocking).** The checkpoint's
   `tokenizer_config.json` declares `"tokenizer_class": "TokenizersBackend"`, a
   newer-`transformers` class **absent from the nv26.04 image** — vLLM crash-loops
   with `ValueError: Tokenizer class TokenizersBackend does not exist or is not
   currently imported`. Fix: point vLLM at the sibling base tokenizer (same
   Qwen3.6-27B vocab, already cached in production) — add
   `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4` to the compose command. With that it
   loads cleanly. (`model switch` prints this line.)
2. **`--max-num-seqs 2` is load-bearing.** The card warns that `--max-num-seqs 4`
   with KV-FP8, `n=3`, and 256K context **silently OOMs during CUDA-graph capture**.
   No `--purpose` profile yields 2 (balanced/prompt-heavy=4, decode-heavy=8), so
   set `VLLM_MAX_NUM_SEQS=2` in `.env` by hand. Tested at 32768 context / util 0.6
   on the shared box (~71.5 GiB resident).
3. **Quantization.** `--quantization modelopt` works — vLLM resolves it to
   `modelopt_fp4` (NVFP4) on load. The catalog sets `modelopt`.
4. **`--trust-remote-code` + `--language-model-only` are required** (custom
   modeling code; vision tower removed) — the shared template omits both, hence
   the compose edits.
5. **MTP + reasoning + structured output.** This build serves with
   `--reasoning-parser qwen3` (the `reasoning` trace field populated correctly in
   testing); MTP has a known vLLM edge case where speculation can break `</think>`
   detection under structured output
   ([vLLM #34650](https://github.com/vllm-project/vllm/issues/34650)). Watch
   JSON-schema responses.
6. **Tool calling not exercised in this run.** The minimal load recipe omitted
   `--enable-auto-tool-choice`; add it + `--tool-call-parser=qwen3_coder` to test
   `tool_choice:"auto"` (additive; the primary 27B uses qwen3_coder).

## Live-test (run on `spark-f8a9`, 2026-05-31) — how to reproduce

1. **Premise (unsloth).** The baseline NVFP4 export drops the MTP head → ~0 %
   acceptance (established by the unsloth + sakamakismile model cards; not re-run
   live to avoid a redundant ~28 GB download). The grafted re-export is what makes
   MTP work.
2. **The real test (sakamakismile).** Stop the primary first (one ~30B model fits
   on the GB10 at a time), serve with the compose edits above + `VLLM_MAX_NUM_SEQS=2`,
   `docker compose up -d` → `/health`, then `model assess` (correctness) +
   `model benchmark` (decode tok/s), plus `GET /metrics`
   (`vllm:spec_decode_num_accepted_tokens_total` / `…_draft_tokens_total`) for the
   draft-acceptance rate. **Result: healthy, 19.1 tok/s, 72 % acceptance** (below).
3. **Risk gate — passed.** vLLM `0.19.0+nv26.04` accepts `qwen3_5_mtp`, so no newer
   engine is needed.

## Benchmark — 2026-05-31, DGX Spark (GB10), shared box

Image `nvcr.io/nvidia/vllm:26.04-py3` (vLLM `0.19.0+nv26.04`), served on `:8001` at
`--max-model-len 32768 --gpu-memory-utilization 0.6 --max-num-seqs 2 --kv-cache-dtype fp8`
with `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'` and
`--tokenizer=mmangkad/Qwen3.6-27B-NVFP4`.

| Property | Value |
|---|---|
| Health / `max_model_len` | `/health` 200; `32768` |
| Engine accepts `qwen3_5_mtp` | ✅ resolves `Qwen3_5MTP` draft head; method → `mtp` |
| Correctness | `17 × 23 = 391` ✅ (finish=stop, 384 tok); reasoning trace `reasoning` (4,817 chars) |
| **Decode throughput** | **19.1 / 19.1 tok/s** (batch=1, greedy, 512 tok forced) |
| Prefill | 845 prompt tokens + 16 gen in 1.23 s |
| **MTP draft acceptance** | **72.2 %** (732 / 1,014 draft tokens; ~2.17 of 3 accepted per step) |
| GPU memory reserved | ~71.5 GB (71,525 MiB) at util 0.6 |
| Tool calling | not exercised (minimal recipe omitted `--enable-auto-tool-choice`) |

### Comparison — baseline 27B primary (same box)

| | MTP (this checkpoint) | baseline `mmangkad/Qwen3.6-27B-NVFP4` |
|---|---|---|
| Decode (batch=1, 512 tok) | **19.1 tok/s** | 7.8–8.0 tok/s |
| Prefill (~845 tok) | 1.23 s | 2.33 s |
| Speedup | **~2.4× decode** | — |

So MTP gives a **~2.4× single-stream decode speedup** on the GB10 — *above* the
card's 1.74× (which was concurrency 2 on a Blackwell box; single-stream with 72 %
acceptance is MTP's best case). One assess probe (the 145-min word problem) hit
`finish=length` at the 2,048-token cap mid-reasoning — a reasoning-verbosity
artifact, not a wrong answer.

## Recommendation

The live test landed the win: **~2.4× single-stream decode (8 → 19 tok/s) at 72 %
MTP acceptance**, same ~71 GB footprint, on the stock nv26.04 image. This is a
strong candidate to **promote to primary** — text-only is no loss for the fleet —
once two gaps close: (1) tool calling exercised with `--enable-auto-tool-choice`
(the mesh agent uses tool calls), and (2) the tokenizer override handled cleanly
(works via `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4`; ideally fix the checkpoint's
`tokenizer_config.json` upstream so no override is needed). Until then
`mmangkad/Qwen3.6-27B-NVFP4` stays the primary and this is a load-tested candidate.
The same MTP-grafted-checkpoint strategy is the path to MTP on the 35B (see the 35B
follow-up).
