# unsloth/Qwen3.8-27B-NVFP4 — the multimodal cortex, VALIDATED at 1M

The fleet's default primary since 2026-08-19 (plan
`docs/plans/2026-08-19-qwen3-8-cortex-upgrade.md`), replacing
`unsloth/Qwen3.6-27B-NVFP4` (demoted to candidate, kept per cite-don't-delete).
**VALIDATED live on the DGX Spark GB10** — evidence:
`docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt` (boot + gates),
`docs/evidence/2026-08-19-spike-qwen3.8-official-nightly-spark.txt` (engine
spike), `docs/evidence/2026-08-19-baseline-qwen3.6-cortex-spark.txt` (the
incumbent baseline it was measured against).

## Checkpoint facts (config-verified 2026-08-19, snapshot 7d6f8d4d)

- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5` —
  the same architecture family as the 3.6 it replaced, so the swap was a
  checkpoint change, not a new-arch bring-up.
- 262,144 native context; 64 layers, hybrid linear-attention. A YaRN
  `hf-overrides` reach to **1M (1,048,576)** was declared and measured
  2026-08-19 and **withdrawn 2026-08-25** when DSpark was adopted (the two
  do not fit together at `gpu_mem_util=0.58`) — see "Serving shape" below.
- **Multimodal** — own ViT (image + video intake); no `--language-model-only`.
- **Self-hosted MTP draft**: a separate 849 MB `model_mtp.safetensors` module;
  the generic `{"method": "mtp"}` speculative config applies, no external
  draft repo, no `qwen3_5_mtp` method key needed on vLLM >= 0.26.
- Quantization: compressed-tensors, mixed-precision (fp8 attention + nvfp4
  MLP; ViT unquantized). ~23.4 GB weights (single main shard + MTP module —
  a different layout from the 3.6's five shards).
- Ships its own tokenizer (**re-verify `tokenizer.json` `"truncation": null`
  after every download** — an early revision hardcoded silent 2048-token
  truncation; the current revision is fixed, verified locally).
- `chat_template.jinja` carries `preserve_thinking` — the issue #93 flag
  works unchanged.
- Tool parser: `qwen3_coder` family; the `qwen3_coder_thinking` strict-tools
  plugin (colleague#320) loads and behaves on vLLM 0.26.1 — proven live with a
  `strict: true` + `enable_thinking` call returning a schema-valid tool_call.

## Engine

`vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`
— the **official** Docker Hub nightly (built 2026-08-19), resolved vLLM
`0.26.1rc1.dev942+g5a4c8d992`. The NVIDIA forum thread (380244) claimed stock
`vllm-openai` lacks NVFP4 kernels for sm_121a and required a custom build —
**disproven**: this stock nightly serves the checkpoint on the GB10 with
FlashInfer autotuning under an `121a` arch path. No third-party image was
needed; official-first order was followed (operator decision q3).

## Serving shape (spark-lobe, ADOPTED 2026-08-25)

This is what `spark-lobe.toml` **declares today** and what the deployed GB10
runs. The 1M YaRN window below it is the previous declaration, retired.

| knob | value | provenance |
|---|---|---|
| `gpu_mem_util` | **0.58** | unchanged from the 1M shape (2026-08-19 measurement) |
| `max_model_len` | **262,144** | the checkpoint's own native ceiling — forced back down from 1M by the DSpark drafter's KV cost |
| `hf-overrides` | `{"text_config":{"rope_parameters":{"rope_type":"yarn",…}}}` | **kept**: every DSpark arm was measured with this rope config in force, so removing it would serve a shape nothing has been measured under |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | *(unset — renders 0)* | inert at exactly 262,144; it exists only to serve PAST the declared ceiling |
| KV pool | 760,806 tokens | **2.90× concurrency at 262,144** — better headroom than the 1M window's 1.21× |
| speculative | `{"method":"dspark","model":"RadixArk/Qwen3.8-27B-DSpark","revision":"85ef153b…","num_speculative_tokens":7}` | see the DSpark section below; the revision is pinned deliberately |

**The prose cost is real and named:** against the incumbent MTP head at n=2,
DSpark wins on code (46.20 vs 24.69 tok/s) and reasoning, and **loses on
prose** (13.71 vs 16.65 tok/s).

A prose-heavy deployment wants the incumbent MTP head back, or spec-decode off
entirely — but **there is no ergonomic per-box override today.** A shape
override composes on top of the card profile, so an operator profile's own
`speculative_config` loses to the shape's, and `lobes init --apply`
force-writes the rendered key back over a hand-edited `.env` line. The
supported routes are to select a different shape or to fork the shape file.
That gap is real and tracked in issue #204 (raised by review on PR #202); it
is named here rather than papered over.

### Retired shape: the 1M YaRN window (MEASURED 2026-08-19, no longer declared)

Retained per cite-don't-delete, in this doc and in `spark-lobe.toml`'s own
comment block, as the rollback recipe.

| knob | value | provenance |
|---|---|---|
| `gpu_mem_util` | **0.58** | 0.60 refused twice live (73.01 GiB demanded vs 72.22/65.22 free); 0.58 booted after the embed-deep reclaim |
| `max_model_len` | **1,048,576** | YaRN ×4 over the 262,144 native ceiling |
| `hf-overrides` | `{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144,…}}}` | stock rope fields preserved byte-for-byte; only type/factor/original added |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | vLLM refuses past the declared ceiling otherwise |
| KV pool | 42.07 GiB = 1,271,476 tokens | **1.21× ceiling at full 1M** — effectively single-request at max depth |
| speculative | `{"method":"mtp","num_speculative_tokens":2}` | 54–61% draft acceptance at n=2 |
| co-residency cost | the opt-in `embed-deep` 4B gear is **stopped** on this box | operator reclaim decision (spec q4) |

Older rollback pair (measured 2026-07-31, also retained in `spark-lobe.toml`):
`gpu_mem_util=0.44` / `max_model_len=262144`, no YaRN knobs.

## Measured performance (single-stream, via gateway, `usage.completion_tokens`)

| shape | TTFT | decode tok/s | 3.6 baseline (same day) |
|---|---|---|---|
| short | 0.20–0.25 s | 23.6–24.7 | 26.5–31.0 |
| medium | 0.35 s | 22.0–22.4 | 26.1–29.1 |
| long (700 tok) | 0.25 s | 19.6–20.1 | 23.7–24.1 |

The 1M lane runs ~2–5 tok/s under the 3.6 baseline. At native 262,144 (t2
spike, same engine) it measured 20.6–30.4 — the YaRN window costs ~0–1.5 tok/s
of decode.

Long-context, measured: 228,415-token prompt → ~723 tok/s prefill (316 s);
**328,379-token prompt (beyond native) → ~563 tok/s prefill (583 s), exact
needle retrieval**. Prefill runs *minutes* at depth — consumers should
**stream** near-1M requests; a non-streamed request happened to survive 9.7
silent minutes on this stack, but that is measured luck, not a contract.

## YaRN quality cost (measured, not assumed)

Always-on static YaRN was the challenge pass's biggest unargued assumption
(spec c26). Measured on an 8-prompt deterministic QA set, same engine and
flags: **native 7/8 vs 1M-YaRN 7/8 — identical score, identical single
failure** (a prompt whose thinking exhausts the token budget in both configs).
Zero measurable cost on this set; it is a small set — a broader eval remains
open.

## Reasoning effort — the default is `xhigh`, and lobes never sets it

The checkpoint's own `chat_template.jinja` (lines 57-71) implements a
`reasoning_effort` chat-template kwarg. **lobes sets it nowhere** — no env var,
no profile TOML, no compose line references it — so every cortex request runs at
the template's own default, **`xhigh`**, the most expensive rung. `high` is an
**alias for `xhigh`**; the real ladder is `low` / `medium` / `xhigh` and an
unrecognised value is refused with a 400 carrying the template's own
`raise_exception` text.

The mechanism is **one injected system sentence, not a token budget** — it steers
verbosity and bounds nothing (only `max_tokens` bounds anything). `medium`
injects no sentence at all: it is the un-nudged baseline. Measured cost of the
sentence itself: `xhigh` 42 prompt tokens, `low` 30, `medium` 0.

**MEASURED 2026-08-21** on the Spark's local cortex —
`docs/evidence/2026-08-21-measure-reasoning-effort-cortex-spark.txt`. Headline
numbers, 4 prompts x 3 rungs, n=1, `temperature=0`:

| config | thinking tokens | vs `xhigh` |
|---|---:|---:|
| `xhigh` (default) | 7136 | — |
| `medium` | 5441 | −23.8% |
| `low` | 5058 | −29.1% |
| `enable_thinking: false` | **0** (1794 total) | **−75% total** |

Three findings that matter to callers, all in that transcript:

- **The aggregate saving is not monotonic.** Lowering effort made 2 of 4 prompts
  *more* expensive (`medium` used +75% thinking tokens on an arithmetic prompt;
  both lower rungs cost ~2.2x `xhigh` on a one-word classification). The −29%
  aggregate is carried almost entirely by one open-ended judgement prompt.
- **Instruction-following degraded at lower effort.** Asked to answer with just a
  letter and a number, `xhigh` returned 6 answer tokens; `medium`/`low` returned
  319/192 tokens of tables. Lower effort is worse for a caller that *parses*.
- **For shallow calls, turn thinking OFF rather than down** — `enable_thinking:
  false` (already the pattern at `lobes/cli/_commands/route.py:60` and
  `lobes/realtime/_turn.py:143`) cut 75% against `low`'s 24%, and scored 4/4 on
  the same prompts.

So the fleet default stays `xhigh` (`lobes/templates/fleet/docker-compose.yml`),
and `low`/`medium` are a **per-call-site** choice for open-ended judgement, not a
default to change. A per-request `chat_template_kwargs` merges **per key** over
`--default-chat-template-kwargs`, so sending only `reasoning_effort` leaves the
issue #93 `preserve_thinking: true` intact.

> Scope (#108): that transcript measures **token cost and single-answer
> agreement**, not quality. `enable_thinking:false` matching `xhigh` 4/4 most
> likely means the prompts were too easy to discriminate, not that thinking is
> free to drop. Effort x tool-calling, x strict/xgrammar, and x depth in the 1M
> window are all unmeasured.

## Speculative decoding — DSpark spike, MEASURED 2026-08-24, and the d4 adoption

The self-hosted MTP head (`{"method":"mtp","num_speculative_tokens":2}`,
above) is not the only speculative arm this checkpoint's vLLM lane can run.
On **2026-08-24**, box **DGX Spark GB10** (`spark-f8a9`), image digest
`sha256:49d2eb65dc2a8dea24e43c27b226f650481ac97d4ba9c567b6e1ca08bc472303`
(vLLM `0.26.1rc1.dev942+g5a4c8d992` — the same nightly as this doc's own
1M measurements), this lane was spiked against `RadixArk/Qwen3.8-27B-DSpark`
(revision `85ef153be924f17ce4bf62726954eeaa4a73e854`), a third-party
block-speculative drafter published for a different (SGLang, RadixArk NVFP4)
recipe. Full transcript:
`docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt`; the full reading
lives in `docs/dspark-speculation.md#measured-here-dspark-on-the-fleets-own-vllm-lane`.
Headline, all MEASURED-HERE (dated) 2026-08-24, single-stream batch-1:

- **DSpark loads and serves against this checkpoint on vLLM**, no code
  change, no engine swap.
- **Config diff:** only `--speculative-config` changed, to
  `{"method":"dspark","model":"RadixArk/Qwen3.8-27B-DSpark","revision":"85ef153b...","num_speculative_tokens":7}`.
- **It does not fit at the 1M window.** At this doc's own
  `gpu_mem_util=0.58` / `max_model_len=1048576`, vLLM refused the DSpark boot
  (`needs 51.47 GiB KV; available 40.76 GiB; est. max model length 824000`).
  DSpark was measured at `max_model_len=786432` and `262144` instead — a
  **served-contract change**, not a private dial.
- **@262144, decode tok/s (code / reasoning / prose), single-stream:** none
  9.93 / 9.95 / 10.01; incumbent mtp-n2 24.69 / 21.90 / 16.65; dspark
  46.20 / 31.73 / 13.71. DSpark beats the no-speculation floor on every
  shape, beats mtp-n2 on code and reasoning, and loses to mtp-n2 on prose
  (content-dependent acceptance: ~61.9% code, ~47–49% reasoning, ~28.6–32.1%
  prose).
- **This doc's own 19.9–24.0 tok/s figure, above, is a DATED 2026-08-19
  measurement** taken under the incumbent MTP config at the full 1M YaRN
  window — not a live baseline, and not directly comparable to the
  262144/786432-window DSpark numbers above without naming the window
  difference, per `docs/model-switch-playbook.md`'s re-measure-same-day rule.
- **A deployment defect was found en route:** the deployed scaffold
  generation on this box (0.57.2) hardcodes `--speculative-config` in the
  rendered compose file, predating the `${PRIMARY_SPECULATIVE_CONFIG-...}`
  substitution in this repo's current template (0.59.0); setting the env var
  alone had no effect on the running container's argv there. Proven only by
  reading the argv from `docker inspect`, never from `.env`.
- **What this does NOT establish:** no output-quality/equivalence claim (only
  speed and acceptance were measured); the FP8-trained-drafter-vs-W4A4-target
  acceptance-mismatch hypothesis is narrowed, not proven, by a same-drafter
  A16-target comparison that found acceptance within ~1 point of the W4A4
  target on every shape; no `num_speculative_tokens` sweep, no
  `dspark_draft_topk` tuning, no concurrency measurement, and single-run
  variance of roughly ±10–13% was directly observed in the incumbent arm's
  own re-measurements. See the spike transcript's own section 9 and section
  13, and `docs/dspark-speculation.md`'s "What the spike does NOT establish",
  for the full list.
- **Deployment adoption (2026-08-25, deviation d4):** following this spike,
  the deployed Spark `cortex` lane was switched to DSpark at
  `max_model_len=262144`, **withdrawing the 1M YaRN window**. It began as a
  hand-edit to the live deployment; the `spark-lobe` shape now **declares
  it**, so the two agree.
- **The re-render trap that came with it is CLOSED.** While the adoption
  lived only as a hand-edit, a plain `lobes init --apply` on this box would
  have regenerated `docker-compose.yml` from shape/profile knobs that still
  said "mtp-n2 at 1M" — silently reverting the lane, or worse, booting DSpark
  argv at a window vLLM refuses. The shape now carries the DSpark
  `speculative_config` and the 262,144 window, and a fresh
  `lobes init --shape spark-lobe --profile spark --apply` was rendered into a
  scratch dir and diffed against the live container: **identical argv token
  sets** (`docs/evidence/2026-08-25-accept-spark-lobe-dspark-render.txt`).
  The `docker inspect`-not-`.env` rule still stands for verifying any box.

## Known limits

- **The "Measured performance" section below is dated 2026-08-19 and was
  taken at the RETIRED 1M window under the MTP head** — not at the adopted
  262,144/DSpark shape. Read it as history, never as the current rate; the
  DSpark section further down carries the 2026-08-24 numbers for the shape
  that ships today. `lobes status` / `lobes capabilities` on the box, or
  `docker inspect`'s rendered argv, remain the live source of truth for any
  particular box.
- **The 1M window is withdrawn, not disproven.** It was measured and it
  worked; it simply cannot be served alongside the DSpark drafter at
  `gpu_mem_util=0.58` on this box. Restoring it means restoring the MTP
  draft too — the rollback recipe is in `spark-lobe.toml`'s d4 block.
- 1.21× KV ceiling at 1M: one full-depth request effectively owns the lane
  for its multi-minute prefill; no per-request KV fairness in v1 (plan risk r3).
- MTP acceptance at 1M depth beyond the probes above is lightly measured
  (plan risk r2); `num_speculative_tokens` tuning (forum suggests 5 at native)
  is open.
- The digest bump reaches hand/worker **templates** too, but those lanes are
  UNVALIDATED on their hosts (Thor sm_110/Orin) until a future boot there
  (plan risk r1; Spark-only rollout boundary c25).

## Speculative decoding on sm_110 (Jetson AGX Thor) — 2026-08-25

**MEASURED-HERE (dated).** Transcript:
`docs/evidence/2026-08-25-spike-thor-cortex-speculation.txt`.

`docs/evidence/2026-08-20-accept-cortex-local-thor.txt` recorded "MTP MUST BE
OFF" on the Thor, because this checkpoint's GDN-hybrid decode carries an MTP
variant with no sm_110 kernel image. That is now **measured to be narrower**:
it holds only on the **CUDA** GDN decode path.

Setting **`VLLM_GDN_DECODE_KERNEL=triton`** — a supported vLLM env var, not a
patch — forces the non-fused path and the missing kernel is never launched.
Measured on the Thor at `max_model_len=262144`, `gpu_mem_util=0.58`,
`max_num_seqs=2`, single-stream, thinking disabled:

| arm | code | reasoning | prose |
|---|---:|---:|---:|
| Triton, no speculation (control) | 12.19 | 12.19 | 12.21 |
| Triton + **MTP-n2** | **26.79** | **26.73** | **19.46** |
| Triton + DSpark block-7 | — blocked, see below | | |

The Triton path costs **essentially nothing** on the unspeculated floor
(12.19–12.21 vs 12.1–12.2 tok/s on CUDA), so those are clean multipliers:
**+120% on code**, +119% reasoning, +59% prose. MTP draft acceptance ranged
45.8–93.8%, strongly content-dependent.

Three caveats travel with this, and none is cosmetic:

- **DSpark is blocked, not disproven.** It loads and its KV fits, then warmup
  dies with a CUDA illegal memory access in the *draft attention* path. The
  FlashInfer-on-sm_110 hypothesis is untestable because the generate lanes
  expose no attention-backend knob (**issue #206**).
- **Greedy output is NOT preserved.** At temperature 0, one of three
  deterministic probes diverged reproducibly between the speculative and
  non-speculative arms, with each arm internally stable across three runs. The
  practical claim "speculation yields the same tokens" is **retracted for this
  lane** (**issue #207**); distributional losslessness was never measured.
- **The served window drops to 262144.** MTP's draft head costs ~238k tokens of
  KV pool; whether MTP-n2 would also have fitted at the 1M YaRN window was not
  tried.

See `docs/qwen38-rollout-notes.md` for the consumer repoint checklist and
`docs/model-switch-playbook.md` for the swap/rollback procedure.
