# DSpark block speculation — the SGLang recipe, and the vLLM spike measured here

> **Update, 2026-08-24: the spike ran.** The recipe below is still the
> **published, SGLang, third-party** recipe — unchanged, still cited to its
> source, still not reproduced. But this repo has now separately booted the
> `RadixArk/Qwen3.8-27B-DSpark` drafter against the fleet's own **vLLM**
> `unsloth/Qwen3.8-27B-NVFP4` target, on the DGX Spark GB10, and measured it.
> See `docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt` and the new
> **"Measured here: DSpark on the fleet's own vLLM lane"** section below. The
> SGLang recipe section that follows is left exactly as t3 wrote it — a
> citation, not a claim — because the vLLM run used a different engine,
> publisher's export, and target quantization, and is not a like-for-like
> repro of it.

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

## Measured here: DSpark on the fleet's own vLLM lane

**MEASURED-HERE (dated).** Box: **DGX Spark GB10** (`spark-f8a9`, compute cap
12.1). Date: **2026-08-24**. Engine: `vllm/vllm-openai`, image digest
`sha256:49d2eb65dc2a8dea24e43c27b226f650481ac97d4ba9c567b6e1ca08bc472303`,
resolved vLLM `0.26.1rc1.dev942+g5a4c8d992` — the same nightly the incumbent
MTP baseline runs on, unchanged. Target: the fleet's deployed
`unsloth/Qwen3.8-27B-NVFP4` (W4A4 NVFP4, compressed-tensors, kv fp8) — **not**
the RadixArk NVFP4 target the published recipe above pairs the drafter with.
Drafter: `RadixArk/Qwen3.8-27B-DSpark`, revision
`85ef153be924f17ce4bf62726954eeaa4a73e854`, block size 7. Full transcript:
`docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt`.

**Config diff from the incumbent.** Only `--speculative-config` changed —
from the incumbent's `{"method":"mtp","num_speculative_tokens":2}` to
`{"method":"dspark","model":"RadixArk/Qwen3.8-27B-DSpark","revision":"85ef153b...","num_speculative_tokens":7}`
(proven from the live container's rendered argv via `docker inspect`, not
read off `.env` — the `.env` line and the rendered argv can disagree, see
the deployment defect below). Everything else — image, target checkpoint,
`hf-overrides` YaRN block, `preserve_thinking` — was held constant.

1. **DSpark loads and serves.** vLLM resolved `DSparkDraftModel` against the
   W4A4 target with no code change, no SGLang, no checkpoint swap. This is
   the first live evidence that the vLLM `dspark` method works against a
   quantized NVFP4 target at all — the published recipe pairs it only with
   SGLang and an unquantized/FP8 target.
2. **The 1.36B/2.53 GiB drafter forced a window trade-down.** At the
   incumbent's `max_model_len=1048576` / `gpu_mem_util=0.58`, vLLM refused
   the DSpark boot outright: `ValueError: max seq len 1048576 needs 51.47
   GiB KV; available 40.76 GiB; est. max model length 824000`. The drafter's
   own 5 full-attention KV layers add per-token KV cost on top of its
   weights. DSpark and the advertised 1M YaRN window cannot both be served
   at `gpu_mem_util=0.58` on this box — a **served-contract change** (a
   `max_model_len` reduction is visible through `lobes capabilities` /
   `GET /capabilities`), not a private tuning dial.
3. **Decode throughput, single-stream batch-1, `docker_logs` acceptance
   surface** (all figures tok/s; "none" = no speculative decoding at all):

   | window | shape | none | mtp-n2 (incumbent) | dspark (block 7) |
   |---|---|---:|---:|---:|
   | 1,048,576 | code / reasoning / prose | — | 23.67 / 21.76 / 16.40 (accept ~76%) | not bootable — see above |
   | 786,432 | code / reasoning / prose | 11.29 / 11.14 / 11.42 | 27.37 / 23.98 / 18.21 (accept 74.5–95.2%) | 44.85 / 33.26 / 13.33 (accept 61.9% / 47.0% / 28.6%) |
   | 262,144 | code / reasoning / prose | 9.93 / 9.95 / 10.01 | 24.69 / 21.90 / 16.65 (accept n/a / 89.2% / 76.1%) | 46.20 / 31.73 / 13.71 (accept n/a / 49.3% / 32.1%) |

   The no-speculation floor is flat across windows (~9.9–11.4 tok/s), as a
   bandwidth-bound dense decode should be — every number above is a
   speculation multiplier on that floor. DSpark **never loses to
   no-speculation on any shape**, but it is a mixed result against the
   incumbent MTP: **+64% on code** (44.85 vs 27.37 @786K), **+39% on
   reasoning** (33.26 vs 23.98), and **−27% on prose** (13.33 vs 18.21).
   DSpark beats the floor everywhere and beats MTP everywhere except prose.
4. **The spread tracks acceptance, which tracks content, not the window.**
   DSpark's draft acceptance was ~61.9% on code, ~47.0–49.3% on reasoning,
   and ~28.6–32.1% on prose — the same ordering as the published SGLang
   spread (43–47 code / 12–18 prose tok/s), which is evidence of the same
   underlying phenomenon, not a reproduction of those numbers.
5. **Two hypotheses for the prose collapse were tested and refuted.**
   - *YaRN/out-of-range extrapolation:* re-run at `max_model_len=262144` —
     within both the target's 262144 native ceiling and the drafter's own
     262144-position training range — produced acceptance and throughput
     unchanged within noise (46.20/31.73/13.71 vs 44.85/33.26/13.33).
     **Refuted.**
   - *Activation quantization (the FP8-trained-drafter-vs-W4A4-target
     mismatch hypothesis):* a second target, `huginnfork/Qwen3.8-27B-NVFP4A16`
     (NVFP4 weights, **16-bit activations**), was measured with the same
     drafter at 262144. Acceptance came back within ~1 point of the W4A4
     target on every shape (code 61.9% both; reasoning 47.0–49.3% W4A4 vs
     49.3% A16; prose 28.6–32.1% W4A4 vs 28.5% A16). Keeping activations at
     16 bits did **not** recover prose acceptance. **Refuted as the
     explanation for the collapse** — though not proof of the reverse; see
     "what this does NOT establish" below for the caveat on how close some
     of those figures are. A16 is also strictly **slower** on this box (a
     7.2–7.3 tok/s floor vs W4A4's ~10, from 28.84 GiB of weights on a
     bandwidth-bound decode), so it is not a substitute target either way.
   - What remains unrefuted: the collapse looks like a property of the
     drafter itself on unstructured text, not of this fleet's quantization
     choice. That is a narrowing, not a proof.
6. **Measurement variance means small deltas are not real.** Every figure
   above is a single run. Re-measuring the incumbent MTP arm at three
   different windows produced a **non-monotonic** result (23.67 @1M / 27.37
   @768K / 24.69 @262K tok/s on code) and the no-speculation floor differed
   by 13% between two windows with nothing else changed (11.42 @768K vs
   10.01 @262K prose). Run-to-run variance of roughly **±10–13%** is present
   in this transcript. An earlier reading of this same data (since withdrawn
   in the transcript's own section 13) had attributed part of that spread to
   the window trade-down itself; that reading does not survive the
   re-measurement and should not be repeated. **Do not read anything at or
   below ~15% in these tables as a real effect** — only the large deltas
   (dspark code +64% vs mtp, prose −27% vs mtp, ~4x vs the no-spec floor)
   clear that bar.
7. **A deployment defect was found en route, not by design.** The deployed
   scaffold generation on this box (0.57.2) **hardcodes**
   `--speculative-config` in its rendered `docker-compose.yml`; it predates
   the `${PRIMARY_SPECULATIVE_CONFIG-...}` substitution that ships in this
   repo's current template (0.59.0). Setting `PRIMARY_SPECULATIVE_CONFIG` in
   `.env` on that box has **no effect** on the rendered argv — the
   documented off-switch is a dead knob there, which is also why the "none"
   arm above had to be driven a different way than the `.env` alone would
   suggest. This was caught only because the harness proves the argv from
   `docker inspect`, never from `.env` — a lesson worth repeating: **an
   `.env` value is a declaration, not a proof of what the container is
   actually running.** Re-rendering the compose file on this box (e.g. via
   `lobes init --apply`) would pick up the fixed template.
8. **Deployment adoption (2026-08-25, deviation d4), and the re-render trap
   it opened — now closed.** Following this spike, the operator adopted
   DSpark on the deployed Spark `cortex` lane at `max_model_len=262144`,
   withdrawing the 1M YaRN window documented in
   `docs/qwen3.8-27b-nvfp4.md`, per point 2 above. That began as a
   **hand-edit to the live deployment's compose/`.env`**, made directly on
   the box, and for a day it was ONLY that. The risk was concrete: a plain
   `lobes init --apply` — a normally-safe, routine operation since 0.59.0 —
   regenerates `docker-compose.yml` from shape/profile knobs that still said
   "mtp-n2 at 1M", so a re-render would have silently reverted the lane, or
   booted DSpark argv at a window vLLM refuses outright. **The shape now
   declares the adopted config** (see "Adopted in-tree" below), so the
   declaration and the box agree. The lesson that produced this section
   stands regardless: **an `.env` value is a declaration, not a proof of
   what the container is actually running** — check `docker inspect`.

### What the spike does NOT establish

- **No quality claim.** Block speculation is lossless by construction (the
  target verifies every drafted token); this spike measured speed and
  acceptance, not output equivalence.
- **The FP8-vs-W4A4 acceptance-mismatch hypothesis is not proven.** Only the
  YaRN/range explanation and the activation-quantization explanation were
  tested and refuted. Something about the drafter's behavior on unstructured
  text is unexplained, not identified.
- **`num_speculative_tokens` was 7 throughout** (the drafter's declared block
  size). No sweep was run; a lower block may suit prose better. Unmeasured.
- **`dspark_draft_topk` was never set.** Unmeasured.
- **Single-stream batch-1 only.** No concurrency measurement.
- **One prompt per shape, one run each.** No repeat-variance figure beyond
  the incumbent-arm re-measurements noted in point 6 above.
- **The two acceptance figures that came back numerically identical across
  targets** (61.9% code both targets, 49.3% reasoning A16 vs 47.0–49.3% W4A4)
  are flagged in the transcript as suspicious for independent measurements —
  `docker_logs` reports a running average at limited precision and may be
  coarse or aggregating. Read "within ~1 point" as the claim; do not read it
  as bit-identical acceptance.
- **ARM 2b's code acceptance is missing** (the metrics window held no
  completed `SpecDecoding` line at the moment it was read) — reported as
  unavailable, not inferred from the 786K-window run.

## Why the vLLM route was even conceivable (answered — see above)

**STRUCTURAL**, probed against the pinned image rather than read off a release
note, and the reasoning that motivated running the spike at all:

- the deployed nightly, `0.26.1rc1.dev942+g5a4c8d992`, already declares
  **`dspark`** — and `dflash` — in
  `vllm.config.speculative.SpeculativeMethod`;
- the target architecture's own vLLM implementation, `qwen3_5.py`, declares
  `SupportsEagle3` with `set_aux_hidden_state_layers()` and
  `get_eagle3_aux_hidden_state_layers()` — the aux-hidden-state extraction
  path a DSpark speculator consumes.

Both facts said the method was **declarable** on the engine already running,
and the spike above has now confirmed it **works**: vLLM's DSpark speculator
consumes mean-pooled target aux hidden states, and RadixArk's drafter learned
those statistics from an FP8 target, not this fleet's W4A4 one — the
plausible failure mode was silently low acceptance reading as "DSpark isn't
worth it" rather than "the drafter is mismatched to this target's
quantization". The spike's own deviation d2 (point 5 above) tested that exact
mismatch hypothesis by holding the drafter fixed and swapping target
activation precision, and did not find support for it — the content-dependent
collapse persisted at 16-bit activations too. Report any future weak result
on this pairing as a **content-dependence finding**, citing this section,
rather than re-opening the quantization-mismatch question from scratch.

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

Two of the five items this section originally listed (2026-08-24, t3) are now
**settled by the spike above** — recorded here as history so the retraction
is visible, not silently edited away:

- ~~That the DSpark drafter loads in vLLM at all~~ — **settled, YES**: it
  loads and serves against the fleet's own W4A4 NVFP4 target. See "Measured
  here" above.
- ~~That the 1.36B bf16 drafter fits~~ — **settled, only with a window
  trade-down**: at the incumbent's `gpu_mem_util=0.58` / `max_model_len=1048576`
  it does not fit and vLLM refuses the boot; it fits at 786432 or below. The
  served-contract-change framing this bullet flagged in advance turned out to
  be exactly right — see point 2 in "Measured here" above.

What remains genuinely unestablished:

- **That 34–38 tok/s is reachable on this fleet.** It is still a number from
  a different publisher's checkpoint on a different engine with a different
  target quantization, measured by someone else. The spike measured this
  fleet's own numbers instead (see above) rather than closing this gap.
- **Any quality claim, in either direction.** Losslessness is structural, not
  measured; the spike measured speed and acceptance only. The NVFP4-vs-GGUF
  quality question stays where `docs/qwen3.8-27b-gguf-llamacpp.md` left it,
  and issue #194 owns the within-lane quant-quality axis.
- **An acceptance rate observable from this repo's own surfaces.**
  `lobes/_metrics.py` maps no spec-decode, draft or acceptance families for
  either engine; the spike's acceptance figures all came from vLLM's own
  `docker_logs` "SpecDecoding metrics" line, read manually, not from a
  first-class metric this repo exposes.
- **The FP8-vs-W4A4 acceptance-mismatch hypothesis, in the affirmative.** The
  spike's deviation d2 removed activation quantization as an explanation for
  the prose-content acceptance collapse; it did not identify what does
  explain it, and does not prove the mismatch hypothesis in the direction
  originally suspected.
- **A `num_speculative_tokens` sweep, `dspark_draft_topk` tuning, concurrency
  behavior, or repeat-run variance below the ~10–13% floor already observed.**
  All unmeasured — see "What the spike does NOT establish" above for the
  full list.
- **That the 2026-08-25 adoption (deviation d4, `max_model_len` traded down
  to 262144) rests on any NEW measurement.** It does not. The in-tree
  declaration described below is backed by a *render* proof — the shape's
  rendered argv matches the deployed container's, byte for byte — not by a
  fresh performance run. The 262144-window DSpark numbers it relies on are
  the ones already in this document (`ARM 2b` / the d3 262144 arms), with
  their stated ±10–13% single-run variance.

## Adopted in-tree (2026-08-25)

The `spark-lobe` shape declares the adopted config, so a box rendering that
shape gets DSpark at 262144 rather than the template's default MTP head at
1M. Three things landed together:

1. **A new profile knob, `speculative_config`** — the raw
   `--speculative-config=…` argv token a shape or card wants on a lane,
   rendered to `<PREFIX>_SPECULATIVE_CONFIG`. Before this, a shape could
   express *window* and *rope* opinions but not *draft* opinions, so a
   non-default drafter could only ever be a hand-edit. `None` still means
   "no opinion" (the template default applies) and `""` still means "no
   speculative decoding at all".
2. **`spark-lobe.toml` moved to the adopted pair** — `max_model_len` 1048576
   → 262144, `allow_long_max_model_len` dropped (inert at exactly the native
   ceiling), `hf_overrides` **kept** (every DSpark arm was measured with that
   YaRN block in force; removing it would ship a rope config nothing has been
   measured under), and the DSpark `speculative_config` added with its
   revision pinned.
3. **A quoting fix that had to be found empirically.** The compose slot
   `${PRIMARY_SPECULATIVE_CONFIG-'--speculative-config={…}'}` is *unquoted*,
   because the default supplies its own quotes. A value substituted there
   crosses compose's dotenv parser and then its shell-lexer, and must carry
   both layers itself. Tested against real `docker compose config`: the
   bare-single-quote and unquoted spellings **both silently degrade** the
   JSON to `{method:dspark,…}` — no error, an invalid config, and a boot
   failure far from the cause. Only the double-wrapped form survives, and
   `tests/test_profile_render.py` now models both parsers to keep it that
   way.

**Verification (`docs/evidence/2026-08-25-accept-spark-lobe-dspark-render.txt`):**
a fresh `lobes init --shape spark-lobe --profile spark --apply` into a scratch
directory, its compose resolved with `docker compose config`, and the
resulting argv diffed against the live container's own `docker inspect` output
— identical token sets. This is a **render proof, not a performance run**: it
establishes that a re-render can no longer drift the lane, and says nothing
new about throughput.

## Attempted on the Thor (sm_110), 2026-08-25 — BLOCKED, not disproven

**MEASURED-HERE (dated).** Box: **Jetson AGX Thor** (sm_110). Transcript:
`docs/evidence/2026-08-25-spike-thor-cortex-speculation.txt`.

DSpark was carried to the Thor's cortex lane on the same pinned image and the
same pinned drafter revision. It **loads** — vLLM resolved `DSparkDraftModel`
against the W4A4 target — and its KV **fits**: 630,029 tokens, 2.40×
concurrency at 262144. It then dies during warmup with a CUDA **illegal memory
access** in the *draft attention* path
(`dflash/speculator.py:456 propose → _build_draft_attn_metadata →
attn_utils.build_attn_metadata`).

This is **not** the sm_110 GDN-MTP kernel gap. That gap is separately fixed on
this board by `VLLM_GDN_DECODE_KERNEL=triton`, and MTP-n2 boots and decodes
happily behind it at 26.8 tok/s on code (a +120% gain over the unspeculated
floor). Two traps from that work are worth carrying here: swapping drafter does
**not** dodge the GDN gap (the gate keys off the target's verification batch,
not the drafter), and MTP is **not** the riskier arm — the MTP head and the
DSpark drafter are both `full_attention` and neither touches GDN.

The surviving hypothesis — that FlashInfer-on-sm_110 is the cause, which
`lobes/machines/thor.py:40` already flags as "unvalidated/contradicted" — is
**untestable today**: the generate lanes expose no attention-backend knob, and
`VLLM_ATTENTION_BACKEND` is absent from this nightly. Tracked in **issue #206**.

**So DSpark's viability on sm_110 is UNKNOWN.** This section records a missing
knob, not evidence against the drafter. Nothing here revises the Spark results
above.

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
- `docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt` — the incumbent
  MTP-n2 baseline, dated 2026-08-19
- `docs/evidence/2026-08-25-spike-thor-cortex-speculation.txt` — the Thor
  (sm_110) three-arm run: the `VLLM_GDN_DECODE_KERNEL=triton` unlock, MTP-n2 at
  +120%, and DSpark blocked in the draft attention path (issues #206, #207, #208)
- `docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt` — the DSpark
  three-arm, three-shape spike this document's "Measured here" section
  reports, dated 2026-08-24
- `docs/measuring-lane-performance.md` — rule 3 and the conditions checklist
- `docs/qwen3.8-27b-nvfp4.md` — the deployed cortex checkpoint
- `docs/qwen3.8-27b-gguf-llamacpp.md` — the measured-here llama.cpp lane (Orin)
