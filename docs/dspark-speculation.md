# DSpark block speculation — the SGLang recipe, recorded

> **Nothing in this document was measured by this repo.** Every throughput,
> acceptance and latency figure below is a **third-party published number**,
> cited to its source. This repo has never booted SGLang, never pulled a
> RadixArk checkpoint, and has never run a DSpark drafter against anything.
> The doc exists so the recipe survives in-tree instead of only in a forum
> post — see `docs/plans/2026-08-24-dspark-speculation-on-the-spark-cortex.md`,
> whose spike is what would (or would not) produce measured numbers here.

The house rule this document is written under is
`docs/measuring-lane-performance.md` **rule 3** — *a number without its
conditions is not reproducible* — plus the habit set in
`docs/qwen3.8-27b-gguf-llamacpp.md#what-was-not-compared-nvfp4`: keep
**"measured here"** and **"published elsewhere"** in separate columns, and
never let a citation drift into a claim.

## Provenance labels used below

| label | meaning |
|---|---|
| **PUBLISHED-ELSEWHERE** | someone else's number, cited to its source; not reproduced here |
| **MEASURED-HERE (dated)** | a figure from a named `docs/evidence/` transcript, valid for the date and config that transcript records |
| **STRUCTURAL** | a property of the algorithm or the artifact's own config, not a measurement of anything |

## The published recipe — 34–38 tok/s on a DGX Spark GB10

**PUBLISHED-ELSEWHERE.** Source: the NVIDIA DGX Spark developer-forum thread
*"Qwen3.8-27B at 34-38 tok/s on DGX Spark"*
(<https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257>),
together with the two RadixArk Hugging Face model cards it points at
(<https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4> and
<https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark>).

### Target checkpoint

| fact | value |
|---|---|
| checkpoint | `RadixArk/Qwen3.8-27B-NVFP4` |
| revision pin | `52d1adc5f38aa5ebf099c29ed7025ba34cfbb854` |
| quantization | NVFP4, **W4A4** (weights *and* activations at 4 bits) |

The revision is quoted because the recipe quotes it. This repo pins every
image by digest and every checkpoint by revision, so a floating `main` would
not be carryable here even as a citation.

Note this is **not** the checkpoint the fleet serves. The Spark and Thor
cortex lanes serve `unsloth/Qwen3.8-27B-NVFP4` (see
`docs/qwen3.8-27b-nvfp4.md`) — same upstream base model, a **different
publisher's export**.

### Engine

| fact | value |
|---|---|
| engine | **SGLang** |
| image | `lmsysorg/sglang:qwen38-27b` |
| attention | flashinfer |
| verification | CUDA-graphed |

SGLang is recorded in `lobes/catalog.py` as the `ENGINE_SGLANG` value of the
existing engine axis — as **data**, alongside `ENGINE_VLLM` and
`ENGINE_LLAMA_CPP`. No lane in this repo serves it, and adding the constant
does not change what any existing lane does.

### Drafter

`RadixArk/Qwen3.8-27B-DSpark` — **STRUCTURAL** facts, read off the model
card, not measured:

| property | value |
|---|---|
| size / dtype | 1.36B, bf16 |
| layers | 5, full attention |
| attention | GQA, 40 query heads / 8 KV heads |
| confidence head | rank-256 Markov head, chooses draft length **dynamically** |
| block size | 7 → verification width 8 |
| positions | 262144 |
| training | SpecForge |
| stated target | `Qwen/Qwen3.8-27B-FP8` |
| stated engine | SGLang |

**PUBLISHED-ELSEWHERE, from the same card:** a claimed **mean 3.39 accepted
tokens per step**, ranging **2.71–4.57 by task**.

The last two rows of the table are the ones to carry forward, because both
differ from this fleet's lane: the drafter's aux-hidden-state statistics were
learned against an **FP8** target and served by **SGLang**, while the
deployed cortex here is **W4A4 NVFP4** on **vLLM**.

### Serve flags

```text
--mem-fraction-static 0.50
--enable-torch-compile
--torch-compile-max-bs 4
--num-continuous-decode-steps 2
```

### The conditions the headline number was taken under

Rule 3 applies to a citation as much as to a measurement, so the conditions
travel with the number:

- **single-stream, batch 1** — not a concurrency figure;
- **content-dependent**, and strongly so:

| content shape | PUBLISHED-ELSEWHERE tok/s |
|---|---|
| math / code | **43–47** |
| headline (mixed) | **34–38** |
| free-form prose | **12–18** |

That is a ~3× spread across content, driven by how often the drafter's block
is accepted. **A single-shape measurement of a speculative lane is not a
description of it** — which is why the covering plan's own measurement task
(t5) requires at least three content shapes and labels every number with the
shape that produced it.

## "Lossless by construction" — what it is, and what it is not

Block speculation is **lossless by construction**: the drafter proposes a
block of tokens, and the **target model verifies every one of them**. A
drafted token that the target would not have produced is rejected, so the
accepted output distribution is the target's own. Acceptance rate changes
**how fast** tokens arrive, never **which** tokens arrive.

That is a **STRUCTURAL** statement about the algorithm. It is:

- **not** a measurement this work performed;
- **not** a claim that any particular implementation is bug-free — a broken
  verification path, a mangled config, or a silently-disabled speculative
  flag can all violate it in practice;
- **not** transferable to a quantization change. Swapping the *target* model
  is a quality question; swapping the *drafter* is not.

The covering plan's honesty condition says it plainly: if a live run ever
observes an output difference attributable to the speculative arm, **the
claim is retracted, not defended**. Until such a run exists, treat losslessness
as the property block speculation is *designed* to have, cited here for what
it explains — namely why speed and acceptance are the only axes the spike
measures.

## What this repo has actually measured on the same silicon

**MEASURED-HERE (dated).** The DGX Spark's deployed `cortex` is
`unsloth/Qwen3.8-27B-NVFP4` on **vLLM** with the checkpoint's own
**self-hosted MTP** draft head at `num_speculative_tokens=2`. On **2026-08-19**,
per `docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt`:

| metric | value |
|---|---|
| decode | **19.9–24.0 tok/s** (24.0 short / 22.3 medium / 19.9 long) |
| MTP acceptance | **54.4–61.1 %** at n=2 |
| TTFT | 0.20–0.35 s |
| `gpu_mem_util` | 0.58 |
| `max_model_len` | 1048576 (1M, YaRN) |
| `max_num_seqs` | 2 |

**Cite this as a dated measurement from that transcript — never as "the
current rate".** The `.env` generation it was taken under has changed since,
and the model-switch-playbook's rule applies: the incumbent baseline for any
comparison must be **re-measured on the same day, same harness, same
prompts** as the arm it is compared against. A 2026-08-19 number is history,
not a baseline.

It is also not a like-for-like counterpart to the 34–38 figure above:
different publisher's export, different engine, different speculative method,
different window, and — unless someone measures it — an unknown content mix.
Putting the two numbers in the same sentence without those five caveats would
be exactly the mistake this doc exists to prevent.

## Why the vLLM route is even conceivable

**STRUCTURAL**, probed against the pinned image rather than read off a release
note:

- the deployed nightly, `0.26.1rc1.dev942+g5a4c8d992`, already declares
  **`dspark`** — and `dflash` — in
  `vllm.config.speculative.SpeculativeMethod`;
- the target architecture's own vLLM implementation, `qwen3_5.py`, declares
  `SupportsEagle3` with `set_aux_hidden_state_layers()` and
  `get_eagle3_aux_hidden_state_layers()` — the aux-hidden-state extraction
  path a DSpark speculator consumes.

Both facts say the method is **declarable** on the engine already running.
Neither says it **works**: vLLM's DSpark speculator consumes mean-pooled
target aux hidden states, and RadixArk's drafter learned those statistics
from an FP8 target. The plausible failure mode is therefore not a loader
error but **silently low acceptance**, which reads as "DSpark is not worth
it" when the honest reading is "the drafter is mismatched to this target's
quantization". Any weak result must be reported as a **TARGET-MISMATCH
hypothesis with the quant difference named**, never as a verdict on DSpark as
a technique.

## The GGUF-on-Spark question is CLOSED

**PUBLISHED-ELSEWHERE.** Source: ggml-org/llama.cpp discussion #27080
(<https://github.com/ggml-org/llama.cpp/discussions/27080>). On a GB10:

| configuration | PUBLISHED-ELSEWHERE |
|---|---|
| llama.cpp, plain | **~15 tok/s** decode |
| llama.cpp, `--spec-type draft-mtp` | **~18.08 tok/s** decode |
| llama.cpp prefill | **~488 tok/s** |

These are cited under exactly the same label as the SGLang figures: someone
else's numbers, on someone else's box, not reproduced here.

**This question is closed, and this document does not re-open it.** Both
figures sit below the Spark cortex lane's own dated 19.9–24.0 tok/s, against
a bandwidth ceiling around 16.6 tok/s for that format on that box. The
surviving survey and its scope entries live with the earlier
`qwen3-8-27b-gguf-on-the-spark` frame; the llama.cpp gear that this repo
*did* measure is a **Jetson AGX Orin** lane, documented with its own
conditions in `docs/qwen3.8-27b-gguf-llamacpp.md`. Nothing here changes
either.

## What this document does NOT establish

- **That 34–38 tok/s is reachable on this fleet.** It is a number from a
  different publisher's checkpoint on a different engine with a different
  drafter, measured by someone else.
- **That the DSpark drafter loads in vLLM at all**, let alone against a W4A4
  NVFP4 target. The published recipe is SGLang; vLLM's own reference artifact
  targets an unquantized 8B. This pairing may never have been exercised by
  anyone.
- **That the 1.36B bf16 drafter fits.** It is ~2.7 GiB of additional weight on
  a box already at `gpu_mem_util=0.58` with a 1M KV pool. If it can only be
  funded by trading the 1M window down, that is a **served-contract change** —
  `max_model_len` is advertised through `lobes capabilities` and
  `GET /capabilities` — not a private tuning dial.
- **Any quality claim, in either direction.** Losslessness is structural, not
  measured; the NVFP4-vs-GGUF quality question stays where
  `docs/qwen3.8-27b-gguf-llamacpp.md` left it, and issue #194 owns the
  within-lane quant-quality axis.
- **An acceptance rate observable from this repo's own surfaces.**
  `lobes/_metrics.py` maps no spec-decode, draft or acceptance families for
  either engine, so any accepted-tokens-per-step figure must come from vLLM's
  own `/metrics` endpoint or a named log line — and must say **which**.

## Sources

- NVIDIA DGX Spark developer forum, *"Qwen3.8-27B at 34-38 tok/s on DGX
  Spark"* —
  <https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257>
- `RadixArk/Qwen3.8-27B-NVFP4` model card —
  <https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4>
- `RadixArk/Qwen3.8-27B-DSpark` model card —
  <https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark>
- ggml-org/llama.cpp discussion #27080 —
  <https://github.com/ggml-org/llama.cpp/discussions/27080>
- `docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt` — the only
  measured-here transcript cited above
- `docs/measuring-lane-performance.md` — rule 3 and the conditions checklist
- `docs/qwen3.8-27b-nvfp4.md` — the deployed cortex checkpoint
- `docs/qwen3.8-27b-gguf-llamacpp.md` — the measured-here llama.cpp lane (Orin)
