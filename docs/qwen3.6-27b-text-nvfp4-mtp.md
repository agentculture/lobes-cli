# Default primary: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`

The **fleet's default primary since 2026-05-31** — a **text-only MTP build**: the
27B re-exported so vLLM **speculative decoding (Multi-Token Prediction)** actually
works. The archived baseline
(`mmangkad/Qwen3.6-27B-NVFP4`, [`qwen3.6-27b-nvfp4.md`](qwen3.6-27b-nvfp4.md)) is
slow single-stream on the GB10 (~8 tok/s decode); MTP drafts several tokens per
forward pass and roughly doubles that. Promoted from candidate after the
tool-calling gate passed (see "Tool calling — verified" below). Filed as
[issue #26](https://github.com/agentculture/model-gear/issues/26).

> **Status: the fleet default primary (promoted 2026-05-31 on this GB10).**
> Loaded and served on `spark-f8a9` (vLLM `0.19.0+nv26.04`) at **18.7–19.1 tok/s
> decode (~2.4× the archived baseline 27B's ~8 tok/s)** with **72–79 % MTP draft
> acceptance**, and the **tool-calling gate passed** under the production compose
> (valid `qwen3_coder` tool call + full round-trip + reasoning trace, MTP active)
> — see the benchmark and "Tool calling — verified" sections below. One packaging
> fix is needed (a tokenizer override — caveat 1), baked into the compose
> template. For the catalog-vs-warm distinction — what you *can* load vs. what's
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
the env and then **prints the exact compose `command:` list items to add by hand** —
the same mechanism the MoE `--moe-backend` uses. The compose `command:` is a YAML
list (one argv token per item), so each flag is a separate item and the
`--speculative-config` JSON uses the `=` form, single-quoted (its value contains
`{` and `:` characters):

```yaml
# add to the vLLM service `command:` list in docker-compose.yml
      - '--speculative-config={"method": "qwen3_5_mtp", "num_speculative_tokens": 3}'
      - --trust-remote-code
      - --language-model-only
      - --tokenizer=mmangkad/Qwen3.6-27B-NVFP4
```

`model switch sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --apply` also auto-selects
`VLLM_TOOL_CALL_PARSER=qwen3_coder` + `VLLM_QUANTIZATION=modelopt`; separately set
`VLLM_MAX_NUM_SEQS=2` in `.env` (see the OOM caveat below).

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
`0.6` (not `0.9` — the audio NIMs + reachy hold the rest of the 121.7 GiB). The
served default is now the **full 256K** (`--max-model-len 262144`), load-tested
2026-06-03 (see the 256K benchmark below): same ~70 GiB footprint as 32K/128K at
util 0.6 — `--gpu-memory-utilization` fixes the KV-pool reservation, so context
length doesn't change resident memory; only the addressable window grows. vLLM
reports **5.3× max concurrency at a full 256K request** — well above the seqs=2
decode cap, so there is no practical concurrency loss. Lower the context only if
heavier co-resident agents need the headroom.

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
   set `VLLM_MAX_NUM_SEQS=2` in `.env` by hand (`model switch` to this primary
   forces it). Tested at 32768 context / util 0.6 on the shared box (~71.5 GiB
   resident); the **full 256K** (`262144`) served default was load-tested too
   (2026-06-03) — boots clean **at seqs=2** (the seqs=2 cap is exactly what keeps
   the capture OOM off at full context), same ~70 GiB footprint. The OOM warning
   above is specifically about seqs=4.
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
6. **Tool calling — verified (2026-05-31), gate closed.** The first run's minimal
   recipe omitted `--enable-auto-tool-choice`; the promotion run served through the
   compose template (which enables it + `--tool-call-parser=qwen3_coder`) and the
   tool path passed — see "Tool calling — verified" below.

## Live-test (run on `spark-f8a9`, 2026-05-31) — how to reproduce

1. **Premise (unsloth baseline).** The ticket ([#26](https://github.com/agentculture/model-gear/issues/26))
   names a baseline serve command —
   `vllm serve unsloth/Qwen3.6-27B-NVFP4 --trust-remote-code --dtype bfloat16 --max-model-len 4096`.
   We deliberately **deviate**: that baseline NVFP4 export *drops the MTP head* →
   ~0 % draft acceptance (established by the unsloth + sakamakismile model cards),
   so running it cannot exercise MTP — which is the whole point. It was not re-run
   live to avoid a redundant ~28 GB download. The decode comparison below is against
   the **already-load-tested `mmangkad/Qwen3.6-27B-NVFP4`** (same arch + NVFP4, the
   archived primary), not the unsloth artifact. The grafted re-export is what makes
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

## Benchmark — 2026-06-03, 128K context (the new served default)

Re-tested on the same shared GB10 at the **128K** default (`--max-model-len
131072`), otherwise the same image and flags as the 2026-05-31 run (util 0.6,
`--max-num-seqs 2`, KV-FP8, MTP n=3, `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4`).
Reproduce with `model switch sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP
--max-model-len 131072 --apply`, then `model assess` + `model benchmark`.

| Property | Value |
|---|---|
| Health / `max_model_len` | `/health` 200; `131072` |
| Boots clean (no capture OOM) | ✅ CUDA-graph pool 0.49 GiB; loaded in 6m46s |
| KV cache | **48.39 GiB** available; **372,800 tokens**; **9.61×** max concurrency at 131,072 tokens/request |
| Correctness | `17 × 23 = 391` ✅ (391 tok); 14:45→17:10 = 145 min ✅ (692 tok, completes — no `finish=length` this time); reasoning trace `reasoning` (1,639 chars) |
| **Decode throughput** | **18.3 tok/s** (batch=1, greedy, 1000 tok forced) |
| Prefill | 845 prompt tokens + 16 gen in 1.22 s |
| **MTP draft acceptance** | **73.3 %** (2,176 / 2,967 draft tokens; ~2.20 of 3 accepted per step) |
| GPU memory (EngineCore) | **71,963 MiB (~70 GiB)** at util 0.6 — same as 32K |
| Tool calling | ✅ post-switch probe passed (`tool_choice:"auto"`, finish=tool_calls) |

The 2026-05-31 32K table above remains the original promotion measurement; 128K
matches it on throughput, draft acceptance, and footprint. Because util fixes the
KV-pool reservation, the resident footprint is unchanged from 32K — only the
addressable context grows (and the pool still holds 9.6× a full 128K request, so
there is room to push higher with headroom). That headroom was then spent: 256K is
the served default since 2026-06-03 (next section).

## Benchmark — 2026-06-03, 256K context (the new served default)

Re-tested on the same shared GB10 at the **full 256K** native context
(`--max-model-len 262144`), otherwise the same image and flags as the 128K run
(util 0.6, `--max-num-seqs 2`, KV-FP8, MTP n=3,
`--tokenizer=mmangkad/Qwen3.6-27B-NVFP4`). Reproduce with `model switch
sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --max-model-len 262144 --apply`, then
`model assess --tools` + `model benchmark`.

| Property | Value |
|---|---|
| Health / `max_model_len` | `/health` 200; `262144` |
| Boots clean (no capture OOM) | ✅ CUDA-graph capture (PIECEWISE) 0.71 GiB in 2 s |
| KV cache | **49.14 GiB** available; **377,600 tokens**; **5.29×** max concurrency at 262,144 tokens/request |
| Correctness | `17 × 23 = 391` ✅ (329 tok); 14:45→17:10 = 145 min ✅ (701 tok, completes); reasoning trace `reasoning` (1,589 chars) |
| **Decode throughput** | **17.8 tok/s** (batch=1, greedy, 1000 tok forced) |
| Prefill | 845 prompt tokens + 16 gen in 1.25 s |
| **MTP draft acceptance** | **74.0 %** (2,282 / 3,084 draft tokens; ~2.22 of 3 accepted per step) |
| GPU memory (EngineCore) | **71,601 MiB (~70 GiB)** at util 0.6 — same as 32K/128K |
| Tool calling | ✅ post-switch probe passed (`tool_choice:"auto"`, finish=tool_calls) |

256K matches the 32K and 128K runs on footprint, throughput (the gentle
context-decline 19.1→18.3→17.8 tok/s continues), and draft acceptance (~74 %), and
adds the full native window at no extra resident memory — `--gpu-memory-utilization`
fixes the KV-pool reservation, so only the addressable context grows. The KV pool
gives **5.29×** concurrency at a full 256K request, well above the `--max-num-seqs 2`
decode cap, so there is no practical concurrency cost vs the 128K default. The
seqs=2 cap is what holds off the capture OOM the seqs=4 caveat warns about.

**Boot note:** one engine-core attempt hit a transient
`InductorError: CUDA driver error: operation not permitted` during torch-inductor
autotuning (a CUDA forward-compatibility-mode artifact on this box — the driver is
in compat mode), *after* weights loaded. It is **not** an OOM and **not**
context-related; the engine retried and booted clean (adding ~3 min to the boot).
Watch for it on a cold boot, but it self-recovers.

## Tool calling — verified (2026-05-31)

The promotion gate was tool calling: the mesh agent rides on this model over the
`acp` backend and uses tool calls, so MTP speculative decoding must not break the
tool path. Served through the **production compose** (so the flags the original
minimal recipe omitted were all present: `--enable-auto-tool-choice`,
`--tool-call-parser=qwen3_coder`, plus `--async-scheduling` / `--enable-chunked-prefill`
/ `--enable-prefix-caching`), at `--max-model-len 32768 --gpu-memory-utilization 0.6
--max-num-seqs 2`:

| Gate | Result |
|---|---|
| Tool call emission (`tool_choice:"auto"`) | ✅ valid `qwen3_coder` call `get_weather({"city":"Tokyo"})`, parseable JSON |
| Full tool round-trip (result → final answer) | ✅ `finish=stop`, correct natural-language answer |
| Reasoning + tool calling coexist (vLLM [#34650](https://github.com/vllm-project/vllm/issues/34650)) | ✅ reasoning trace present, no `</think>` breakage |
| MTP spec-decode active **with tools on** | ✅ **78.6 %** draft acceptance (276/351) — not silently disabled |
| Correctness (`model assess`) | ✅ both probes `finish=stop` (the 145-min word problem that hit `length` in the minimal run now completes) |
| Decode throughput (`model benchmark`, production flags) | ✅ **18.7 tok/s** — the ~2.4× win survives the full flag set |
| GPU footprint | ~71.2 GiB (72,915 MiB) at util 0.6 — fits the shared box |

The one packaging caveat (the `--tokenizer` override, caveat 1) is handled by
baking it into the compose template, so a fresh deploy works out of the box.
Reproduce with `model assess` + `model benchmark` and a `tool_choice:"auto"`
request (or `model switch sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP --apply`, which
runs the tool-call probe automatically).

## Recommendation

Promoted to **fleet default primary (2026-05-31)** — both gates the earlier draft
called out are closed: tool calling is verified (above) and the tokenizer override
is handled in the template. The win: **~2.4× single-stream decode (8 → ~19 tok/s)
at 72–79 % MTP acceptance**, same ~71 GB footprint, on the stock nv26.04 image;
text-only is no loss (the fleet runs the 27B text-only anyway). The archived
baseline `mmangkad/Qwen3.6-27B-NVFP4` stays a candidate (the tokenizer source here
and the only vision-capable 27B). Ideally fix the checkpoint's
`tokenizer_config.json` upstream so no `--tokenizer` override is needed. The same
MTP-grafted-checkpoint strategy is the path to MTP on the 35B (see the 35B
follow-up).
