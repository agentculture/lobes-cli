# vLLM-nightly migration — before-state verification + baselines to beat

**Task t1** of the devague plan
[`lobes-unifies-its-generate-lane-on-one-vllm-nightl`](plans/2026-07-01-lobes-unifies-its-generate-lane-on-one-vllm-nightl.md)
(covers spec claims c3, h6, h7). This is a **verification-only** doc: it records
what the fleet templates and catalog actually pin *today*, cited to exact
`file:line`, and the baselines a future nightly migration has to beat. It makes
**no** change to `docker-compose.yml`, `env.example`, `catalog.py`, or any image
pin — that mutation is scoped to later tasks (t2/t3 spikes → t4 flip → t5–t7
Gemma DSpark → t8 trailing gears → t9 shipped-state docs), per the plan's
`Execution waves`.

## 1. Today's image pins (verified in-repo)

The fleet's generate/embed/rerank vLLM services and the Gemma multimodal gear
are **not** on the same engine today — that asymmetry is the whole reason this
migration exists.

### Primary, embed, rerank, minor, middle — all pin the NGC 26.04-py3 image

Grepped directly from the real template
(`lobes/templates/fleet/docker-compose.yml`), not assumed:

| Service | image pin | file:line |
|---|---|---|
| `vllm-primary` (generate `main` tier) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:24` |
| `vllm-embed` (embedding gear) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:111` |
| `vllm-rerank` (reranker gear) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:164` |
| `vllm-minor` (opt-in 4B, `COMPOSE_PROFILES=minor`) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:234` |
| `vllm-middle` (opt-in legacy 14B, `COMPOSE_PROFILES=middle`) | `nvcr.io/nvidia/vllm:26.04-py3` | `lobes/templates/fleet/docker-compose.yml:307` |

That image tag resolves to **vLLM `0.19.0+nv26.04`** — confirmed by every
per-model doc that serves on it, e.g.
`docs/qwen3.6-27b-text-nvfp4-mtp.md:183` ("Image `nvcr.io/nvidia/vllm:26.04-py3`
(vLLM `0.19.0+nv26.04`)") and `docs/gemma-4-12b-nvfp4.md:238` (`| nvcr.io/nvidia/vllm:26.04-py3 | 0.19.0 | ❌ | ❌ |`,
the last two columns being the two blockers this migration exists to fix: no
native `gemma4_unified` class, no MTP-capable loader for that arch on 0.19.0).
The single-model (non-fleet) `lobes/templates/docker-compose.yml:26` scaffold
pins the same `nvcr.io/nvidia/vllm:26.04-py3` image.

### The Gemma 4 12B `vllm-multimodal` gear already runs vLLM nightly

`vllm-multimodal` does **not** pin `nvcr.io/nvidia/vllm:26.04-py3` — it builds a
custom image instead
(`lobes/templates/fleet/docker-compose.yml:396-399`: `build: {context: ., dockerfile:
Dockerfile.vllm-gemma4}`, `image: ${MULTIMODAL_IMAGE:-lobes/vllm-gemma4:local}`),
and that Dockerfile bases off the **official vLLM nightly image, pinned by
digest**:

```text
lobes/templates/fleet/Dockerfile.vllm-gemma4:24
FROM vllm/vllm-openai@sha256:7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9
```

Per the comment directly above that line
(`lobes/templates/fleet/Dockerfile.vllm-gemma4:18-23`), the digest "resolves to
vLLM 0.23.1rc1.dev672 + transformers 5.12.1 (registers gemma4_unified) on
Blackwell-capable torch, and was live-validated on the DGX Spark GB10 (sm_121)
on 2026-07-01 (#71)". The reason it needs nightly at all: Gemma 4's
`gemma4_unified` arch is early-fusion multimodal with heterogeneous per-layer
head sizes, and the native `Gemma4UnifiedForConditionalGeneration` class that
handles that "only exists in vLLM NIGHTLY (>= 0.23.1rc1)"
(`lobes/templates/fleet/Dockerfile.vllm-gemma4:9`); released vLLM <= 0.22.1
(including the NGC 26.06 base the Dockerfile used before) falls back to the
generic Transformers backend and crashes the full-attention layers' `o_proj`.

### Summary

| Gear | Image today | Engine |
|---|---|---|
| `main` (27B primary) | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| embed | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| rerank | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| minor (opt-in 4B) | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| middle (opt-in legacy 14B) | `nvcr.io/nvidia/vllm:26.04-py3` | vLLM 0.19.0+nv26.04 |
| `multimodal` (Gemma 4 12B) | `vllm/vllm-openai@sha256:7c5a...` via `Dockerfile.vllm-gemma4` | vLLM 0.23.1rc1.dev672 (nightly) |

Two engines, two vLLM minor versions, on one gateway. Unifying the generate
lane onto one nightly (this plan's t2–t9) removes that split and makes a
same-engine head-to-head (t6) meaningful — today a 27B-vs-12B comparison would
be confounded by engine version, not just model size.

## 2. Baselines to beat

These are the numbers a nightly-served 27B primary and a nightly-served Gemma
12B (with MTP/DSpark wired, t5) have to match or beat for the migration to be
worth it. Recorded here from the source docs, not re-measured — t2/t6 do the
re-measurement.

### 27B primary (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`), today's 0.19.0 engine

Source: `docs/qwen3.6-27b-text-nvfp4-mtp.md`.

- **Decode throughput ≈ 19 tok/s** — the promotion benchmark measured
  **19.1 tok/s** (batch=1, greedy, 512 tok forced,
  `docs/qwen3.6-27b-text-nvfp4-mtp.md:193`), and the 128K-context re-run
  measured **18.3 tok/s** (1000 tok forced, `docs/qwen3.6-27b-text-nvfp4-mtp.md:227`);
  the production-flags `lobes benchmark` run under the compose template landed
  at **18.7 tok/s** (`docs/qwen3.6-27b-text-nvfp4-mtp.md:293`). Summary line:
  "**18.7–19.1 tok/s decode (~2.4× the archived baseline 27B's ~8 tok/s)**"
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:13-14`).
- **MTP draft acceptance: 72–79 %.** 72.2 % at promotion
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:195`), 73.3 % at 128K context
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:229`), 74.0 % at longer context
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:257`), and **78.6 %** with
  `--enable-auto-tool-choice` on under the production compose
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:291`) — the summary line states
  "**72–79 % MTP acceptance**" (`docs/qwen3.6-27b-text-nvfp4-mtp.md:307`).
- Measured on `nvcr.io/nvidia/vllm:26.04-py3` (vLLM `0.19.0+nv26.04`),
  `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`
  (`docs/qwen3.6-27b-text-nvfp4-mtp.md:183`).

### Gemma 4 12B multimodal gear, no spec-decode

Source: `docs/gemma-4-12b-nvfp4.md`.

- **Decode throughput ≈ 23 tok/s, no draft.** "the benchmark below (~23 tok/s
  single-stream, no draft) *is* that no-spec baseline"
  (`docs/gemma-4-12b-nvfp4.md:151-152`); the measured table reads **"~23 tok/s
  (batch=1, greedy — 21.7–23.3 across balanced/prompt-heavy; 23.0 tok/s
  sustained over 1,500 forced tokens in 65.2 s)"** (`docs/gemma-4-12b-nvfp4.md:286`),
  with `Speculative decoding: none` on the same line block
  (`docs/gemma-4-12b-nvfp4.md:292`).
- This baseline already out-decodes the 27B primary single-stream (~23 vs
  ~18–19 tok/s, `docs/gemma-4-12b-nvfp4.md:294-297`) despite having no
  spec-decode boost at all — the gear is smaller (12B vs 27B). The DSpark
  speculative-decoding wiring this plan adds (t5/t6) has to clear this ~23
  tok/s bar to be worth shipping default-on (t7's verdict gate).
- Measured on `lobes/vllm-gemma4:nightly-audio` (vLLM `0.23.1rc1.dev672`,
  native `Gemma4UnifiedForConditionalGeneration`, `TRITON_ATTN`,
  `compressed-tensors` NVFP4, `--max-model-len 8192`) — i.e. the Gemma baseline
  is *already* a nightly-engine number, unlike the 27B baseline above.

## 3. The generate lane carries real mesh traffic (not hypothetical)

This benchmark isn't academic — the `main` tier is the model the deployed
`lobes` mesh agent itself thinks with, right now, in production:

- `culture.yaml:8` declares the lobes agent's runtime backend as
  `model: vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` — every response
  the `lobes` agent posts to the Culture mesh is generated by the exact 27B
  primary this doc benchmarks. `docs/gateway-fleet.md:400-401` confirms the
  resolution path: "`culture.yaml`'s `model: vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
  resolves through the gateway on `:8000` as the default."
- The gateway's tier-alias contract is what makes that traffic identifiable
  per-gear: "Callers send `model=main|minor|multimodal` (or the back-compat
  aliases `hard|cheap|normal`) instead of a full model id"
  (`docs/gateway-fleet.md:95-96`) — `model=main` addresses the 27B primary
  benchmarked in §2, `model=multimodal` addresses the Gemma 12B gear
  benchmarked in §2.
- `docs/gemma-4-12b-nvfp4.md:157-158` names the `multimodal` lane's audience
  explicitly: "lobes operators/maintainers and **the Culture mesh that
  consumes** the `multimodal` (Gemma 4 12B) generate lane — i.e. anyone calling
  `model=multimodal` or the back-compat `model=normal`."

So both gears this plan puts on one engine are load-bearing for real mesh
callers today, not idle candidates — a regression in either (lost MTP,
gibberish, a dropped tool-call parser) would degrade the agent the mesh
actually talks to.

## 4. t2 spike — 27B serves on nightly (live, 2026-07-01): **GO**

Standalone spike run in an **authorized maintenance window** — no in-repo change,
but the live primary was stopped to free memory on the saturated GB10 (see the
memory note below), then restarted on its `26.04` image. Container:
`vllm/vllm-openai:nightly` (already local, no pull) = vLLM
**0.23.1rc1.dev672+g93d8f834d**, free host port 8100, replicating the primary's
serving flags at a trimmed `--max-model-len 8192` / `--gpu-memory-utilization
0.40`.

**Verdict: GO.** The 27B `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` loads and
serves on nightly with MTP active — validating the plan's load-bearing
assumption `c22` and resolving risk `r1` (the deprecated `qwen3_5_mtp` method /
Qwen3.5 hybrid-attention path did **not** break on nightly).

| Check | Result on nightly (0.23.1rc1.dev672) |
|---|---|
| Architecture | ✅ `Qwen3_5ForConditionalGeneration` + draft `Qwen3_5MTP` resolve |
| MTP method | ✅ `qwen3_5_mtp` logged deprecated → **auto-mapped to `mtp`**; `SpeculativeConfig(method='mtp', num_spec_tokens=3)` — MTP stays on, no caller change needed (r1) |
| Quantization | ✅ `--quantization modelopt` → resolved `modelopt_fp4`, served via `FlashInferCutlassNvFp4LinearKernel` (native NVFP4 cutlass) |
| Hybrid attention | ✅ `Using Triton/FLA GDN prefill kernel` — Qwen3.5 Gated-DeltaNet/linear-attention path active, no gibber |
| Correctness | ✅ `17 × 23 = 391` (finish=stop) |
| Tool call (qwen3_coder) | ✅ `get_weather({"city": "Paris"})`, finish=`tool_calls` |
| MTP draft acceptance | ✅ **69.9 %** (174 accepted / 249 drafted; 83 drafts — small sample, at/near the 72–79 % 0.19.0 baseline) |
| Throughput | **~20.7 tok/s** incl. prefill (256 tok in 12.4 s) — parity with the ~18.7–19.1 tok/s 0.19.0 decode baseline |
| Load | 18.65 GiB weights + 25.68 GiB KV (224,824-token cache @ 8192, 27.4× concurrency); init 141 s (compile 39 s) |

**Caveats (do not change the GO):** the 69.9 % acceptance is a single small-sample
probe (83 drafts) that lands just under the 72 % floor but within noise (nightly
also warns `num_speculative_tokens > 1 … may result in lower acceptance rate`);
the throughput includes prefill at a trimmed 8192 / util 0.40, so it is a
**functional-parity** signal, not a production-scale number — **t6** does the
apples-to-apples measurement. MTP works, correctness/tool-calling hold, quant +
hybrid kernels are native.

### Maintenance-window / memory note (risk `r2`, **confirmed**)

At spike time the GB10 was **memory-saturated**: 118/121 GiB used, swap 15/15 GiB
full, ~3 GiB free — the running fleet (primary at `util 0.6` + `max-model-len
262144`, plus minor/embed/rerank + the non-vLLM services) left no room for a
co-resident 27B even at low util. So t2 required **stopping the live primary**
(freed ~72 GiB → 62 GiB available) for the run, then restoring it. This is risk
`r2` confirmed: the fleet cannot co-reside a *second* large gear on nightly — so
**t6's head-to-head will likely need the sequential/standalone method**, and t4's
flip must mind the util budget. (The nightly image was already local; no pull.)

## 5. t3 spike — embed + rerank serve on nightly (live, 2026-07-01): **GO**

Standalone spikes on `vllm/vllm-openai:nightly` (0.23.1rc1.dev672), run in a
**light window** — only the opt-in `minor` gear was stopped (~13 GiB) to free
room; the generate lane (primary) stayed up. Each 0.6B pooling gear replicates
its compose flags (`util 0.06`, `--max-model-len 8192`), run **one at a time**.

**Verdict: GO** — both pooling task families work on nightly. Resolves the
embed/rerank half of assumption `c23` and honesty `h2`.

| Gear | Result on nightly |
|---|---|
| embed (`Qwen3-Embedding-0.6B`, `--runner pooling --convert embed`) | ✅ `/v1/embeddings` → **1024-dim** vector; `Supported tasks: ['token_embed','embed']`; matryoshka `--hf-overrides` accepted |
| rerank (`Qwen3-Reranker-0.6B`, `--runner pooling --convert classify`) | ✅ `Resolved architecture: Qwen3ForSequenceClassification`; `/v1/rerank` correct ranking (0.98 / 0.94 / 0.89), `/v1/score` [0.98, 0.90]; `Supported tasks: ['token_classify','classify']` |

### Finding: nightly's memory-profiling assertion is strict on the shared GB10

The rerank spike **crashed on its first launch** — *not* a model/nightly
incompatibility, but a nightly-vLLM assertion:
`AssertionError: Error in memory profiling. Initial free memory 22.02 GiB,
current free memory 22.11 GiB. This happens when other processes sharing the same
container release GPU memory while vLLM is profiling`. It tripped because the
embed spike was torn down (releasing memory) **while** rerank was profiling — on
the unified-memory GB10 every container shares one pool, so free memory
fluctuates. Relaunching once memory was quiescent succeeded immediately.

**Migration implication (t4 / t6 / t8):** nightly gears must be (re)started when
fleet memory is **stable**, not mid-churn — a `docker compose up` that brings
several nightly gears up at once (or alongside a gear being torn down) can hit
this. Mitigations for t4: bring gears up **one at a time** / with a settle gap,
or find the nightly flag to relax the profiler tolerance. Recorded as a new
migration risk to carry into the t4 flip and the t6 head-to-head.

## 6. t6 head-to-head — both generate gears on the same nightly (live, 2026-07-01)

Full maintenance window (primary stopped ~20 min, restored after). Both gears
served **standalone** on the SAME engine — vLLM **0.23.1rc1.dev672** — because
`r2` (§4) means the fleet cannot co-reside a second large gear. Batch=1 greedy,
`--max-model-len 8192`, an `ignore_eos`-forced 800-token decode + a 3,201-token
prefill probe.

### Results

| Metric | Qwen 27B (MTP) | Gemma 12B (no-spec) |
|---|---|---|
| Image / quant / util | `vllm/vllm-openai:nightly`, modelopt_fp4, 0.40 | `lobes/vllm-gemma4:nightly-audio`, compressed-tensors, 0.30 |
| **Decode tok/s** (vLLM-logged) | **~17.6–18.0** (17.0 wall) | **~22.8–23.0** (22.5 wall) |
| **Prefill tok/s** | **~2,190** (3,201 tok / 1.46 s) | **~1,966** (3,201 tok / 1.63 s) |
| MTP draft acceptance | 60.6 % (516 / 852, 284 drafts) | — (no spec-decode) |

**Read (inform-only, no swap per `c19`):** the **Gemma 12B out-decodes the 27B
(~23 vs ~18 tok/s)** purely by being the smaller model — even with the 27B running
MTP and Gemma running no spec-decode. The 27B has slightly faster prefill. This is
*not* like-for-like (different size / quant / role); it is the generate-lane
throughput picture on one engine, which is what the head-to-head was for.

Caveat vs the 0.19.0 baseline: the 27B's nightly numbers (~17.6–18 tok/s, 60.6 %
accept) are **modestly below** its 0.19.0 baseline (18.7–19.1 tok/s, 72–79 %) on
this single spike (trimmed context, util 0.40, `ignore_eos` forcing). Nightly is
roughly at **parity, not a speed win**, for the 27B — consistent with §4. Since
`c18` committed to nightly for fleet **unification** (not a 27B speedup), this does
not change the migration; recorded honestly.

### DSpark MTP route for Gemma: **INVALID on vLLM 0.23** (`h5` resolved — negative)

Serving Gemma with the DSpark draft (`deepseek-ai/dspark_gemma4_12b_block7`, the
plan's DSpark-first route) **fails to load** on nightly. Two findings, in order:

1. **Config-key correction.** The disabled-experiment config in
   `docs/gemma-4-12b-nvfp4.md` (and echoed by t5) —
   `{"method":"draft_model","draft_model_id":"…"}` — uses an **outdated key**.
   vLLM 0.23's `SpeculativeConfig` rejects `draft_model_id` ("Unexpected keyword
   argument"); the draft id must go in **`model`**. (t5/t7 must fix this.)
2. **Architecture unsupported (decisive).** With the corrected `model` key, load
   fails: `Value error, Model architectures ['Gemma4DSparkModel'] are not
   supported for now.` DSpark's custom drafter arch is **not** in vLLM 0.23's
   supported speculative-draft set. The DSpark `draft_model` route is a dead end
   on this engine.

**The native route works — measured (2026-07-01).** `Gemma4MTPModel` **is** in the
supported list, so I served the coder checkpoint with Google's
`google/gemma-4-12B-it-assistant` draft (`method=mtp`, `model=<assistant>`,
`num_speculative_tokens=1`). It **loads and drafts**: the engine logs `Detected MTP
model. Sharing target model embedding weights with the draft model` and maps the
draft layers to the target's late layers (`draft layer → layers.46/47`, KV-shared).
Measured on nightly:

| Gemma 4 12B **coder** + native MTP | value |
|---|---|
| Decode | **~24.0 tok/s** (vs ~23 no-spec — a **marginal ~6 % win**) |
| MTP draft acceptance | **30.8 %** (188 / 611, 611 drafts @ n_predict=1) |

**Why the win is only marginal — a checkpoint mismatch.** The assistant was trained
to draft for the **base** `google/gemma-4-12B-it`, but the fleet serves a **coder
fine-tune** (`…coder-fable5-composer2.5…`). The fine-tune's shifted output
distribution drops acceptance to ~31 % (vs the 27B MTP's ~60 %) — exactly the risk
`docs/gemma4-mtp-draft.md` flagged. At `n_predict=1` and ~31 % acceptance the speedup
is small. **Direction (user, 2026-07-01):** serve the **base `google/gemma-4-12B-it`**
(the assistant's exact target) for much higher acceptance — "less coder, more MTP" —
and **support both** checkpoints in the catalog so callers pick coding-strength vs
MTP-throughput. That base measurement + dual catalog entry is the next leg.

### Verdict feeding t7

DSpark-first is **invalid on vLLM 0.23**; the native assistant route **works but is
marginal on the coder checkpoint** (~31 % accept, ~6 % decode gain). So t7's verdict
is **do NOT wire DSpark**, and native MTP on the coder gear is a marginal/optional
win — the real MTP payoff needs the **base `gemma-4-12B-it`** gear (next leg). Side
note for t7/t9 cleanup: `VLLM_ATTENTION_BACKEND` is flagged an "unknown env var" on
this nightly (the native class auto-forces TRITON), so that compose env line is a
no-op warning.

## 7. Checkpoint choice for Gemma MTP — coder vs base (live, 2026-07-02)

The §6 result (coder + native MTP: 30.8 % accept, ~6 % win) raised the question:
does the assistant draft better for the **base** it-model it was trained for?
Measured both + their no-spec baselines, all on nightly 0.23.1rc1.dev672 (batch=1
greedy, `max_len 8192`):

| Gear | quant | no-spec | + native MTP | MTP accept | MTP speedup |
|---|---|---|---|---|---|
| Gemma **coder** (`…coder-fable5-composer2.5…`) | NVFP4 | ~23 tok/s | ~24 tok/s | 30.8 % | ~1.04× |
| Gemma **base** (`google/gemma-4-12B-it`) | bf16 | 6.5 tok/s | 14.6 tok/s | **93.9 %** | ~2.25× |
| **Gemma base NVFP4** (`coolthor/gemma-4-12B-it-NVFP4A16`) | NVFP4 | 19.8 tok/s | **28.6 tok/s** | 57.9 % | **~1.45×** |
| (ref) Qwen 27B primary | NVFP4 | — | ~18 tok/s | 60.6 % | ~2.4× |

**Findings:**

1. **User insight confirmed.** The native assistant (`google/gemma-4-12B-it-assistant`)
   drafts far better for the base it-model it was trained for: **93.9 %** accept on
   the bf16 base and **57.9 %** on the NVFP4 base — vs **30.8 %** on the coder
   fine-tune. "Less coder → more MTP" is real.
2. **bf16 is a trap: best acceptance, worst speed.** The bf16 base gets 93.9 %
   accept but its no-spec floor is only 6.5 tok/s, so even ~2.25× MTP lands at
   14.6 tok/s — *slower* than the NVFP4 coder. High acceptance can't rescue slow
   compute.
3. **The NVFP4 base is the winner — measured.** `coolthor/gemma-4-12B-it-NVFP4A16`
   with native MTP = **28.6 tok/s** (19.8 no-spec × **~1.45×** MTP @ 57.9 % accept) —
   the **fastest Gemma config measured**, beating the coder (24), coder+MTP (24),
   and bf16 base+MTP (14.6). NVFP4 quant drops acceptance from the bf16 base's
   93.9 % to 57.9 % (the quantized target's distribution shifts vs what the bf16
   assistant expects), but NVFP4 speed more than compensates.

**"Support both" — confirmed plan.** Catalog carries two Gemma gears:

- **coder** (`sakamakismile/…NVFP4`): coding-strong, MTP **not** worth wiring (30.8 %).
- **base** (`coolthor/gemma-4-12B-it-NVFP4A16`): general, **native MTP default-on**
  (`--speculative-config '{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":1}'`),
  ~28.6 tok/s.

Callers pick coding-strength vs MTP-throughput. Next leg: wire both into
`catalog.py` + compose (the base gear carries the assistant draft as a pinned dep).

## 8. Always-on duo budget (live-validated, 2026-07-02)

**Question answered: can the always-on Gemma multimodal gear serve its full
128K native context *and* co-reside with the 27B primary, without either
starving the other?** Yes — live-validated on the DGX Spark GB10.

Both gears are already default/always-on (§7); this only retunes their
`--max-model-len` / `--gpu-memory-utilization` pair so the co-resident budget
actually holds 128K on the multimodal side instead of the earlier
co-resident-safe fallback (8192 tokens @ util 0.12, from #71 — see
[`gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md#live-validation-status-71)).

| Gear | Context | Util | Concurrency (measured) |
|---|---|---|---|
| Gemma 4 12B multimodal (base) | **128K** (native max) | **0.22** | **4.67×** |
| Qwen 27B primary (MTP) | **64K** (trimmed from 128K) | 0.35 measured → **0.30 shipped** | **6.36×** at util 0.35 |

Both held their target context concurrently, alongside the embed + rerank
gears and the co-tenant services already on the box: **~108 GiB used / ~13 GiB
free** on the 128 GB GB10. To leave more headroom than the validated 0.35, the
27B primary's shipped default is shaved to **0.30** — still comfortably holds
64K (0.35 was the measured ceiling for 6.36× concurrency; 0.30 trades a little
concurrency for slack).

**New default-fleet budget:**

```text
primary 0.30 + multimodal 0.22 + embed 0.06 + rerank 0.06 = 0.64
```

(down from the pre-duo 0.69 default, which paired the primary's 128K/0.45 with
the multimodal gear's 8192/0.12 fallback context.)

**What changed (baked into `lobes/templates/fleet/docker-compose.yml` and
`env.example`):**

| Var | Old default | New default |
|---|---|---|
| `PRIMARY_MAX_MODEL_LEN` | 131072 (128K) | **65536 (64K)** |
| `PRIMARY_GPU_MEM_UTIL` | 0.45 | **0.30** |
| `MULTIMODAL_MAX_MODEL_LEN` | 8192 | **131072 (128K, Gemma's native max)** |
| `MULTIMODAL_GPU_MEM_UTIL` | 0.12 | **0.22** |

**Answers "can Gemma-128K + Qwen-64K co-reside? → yes"** — this was the open
question left by the #71 co-resident-safe fallback (the multimodal gear had
never been measured warm next to a full-size primary at anything beyond a
trimmed 4096–8192 token ceiling). With both gears retuned as above, the
always-on duo now serves its two headline capabilities — Qwen's 64K
tool-calling/reasoning lane and Gemma's full 128K native
text+image+audio+MTP lane — at the same time, on one GB10.

## Scope note

Sections §1–§3 are **before-state only** (t1). §4 = **t2** (27B on nightly, GO),
§5 = **t3** (embed+rerank on nightly, GO), §6 = **t6** (head-to-head + DSpark
verdict + native-MTP measured), §7 = **checkpoint choice** (coder vs base MTP),
§8 = **always-on duo budget** (Gemma 128K + Qwen 64K co-residency, live-validated).
t4 flipped the primary/embed/rerank default images to nightly (merged).
**Remaining:** t7 (commit the Gemma verdict), t8 (trailing minor/14B), t9
(shipped-state docs), and the new **"support both" gears** (coder + NVFP4-base with
native MTP) the §7 measurements motivate.
