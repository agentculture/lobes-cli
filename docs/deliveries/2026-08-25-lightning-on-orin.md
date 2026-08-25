# Delivery Summary — Lightning on Orin

plan: `lightning-on-orin` · run: `complete` · date: `2026-08-25`
baseline: `devague plan (hand-assembled)`

## Intent

Serve NVIDIA Nemotron 3.5 Lightning 30B-A3B locally on the Jetson AGX Orin
64GB (Ampere sm_87) as a new Colleague role, `associate` — a lobe that does
ground work and hands the result back rather than enacting it. The board's only
local generate lobe above `hand` was a llama.cpp GGUF cortex measured at
**2.61 tok/s**, below its own plan's ≥5 tok/s gate; everything faster was a
proxy hop to a peer. The plan executed eleven tasks in five waves, plus a
twelfth added mid-run at the operator's instruction for the live acceptance.

## Planned Work

Quoted verbatim from `devague plan show`:

- `t1` — Capture the board's pre-spike container inventory and restore it
- `t2` — Finalise the vLLM spike transcript as citable evidence
- `t3` — Finalise the llama.cpp spike transcript as citable evidence
- `t4` — Carve Lightning out of the W4A4 infeasibility claim on the Orin card
- `t5` — Correct the stale v0.27.1-on-Thor sentence in the Lightning doc
- `t6` — Add the 'associate' role to the role vocabulary and capability ordering
- `t7` — Give the eight unexpressible vLLM serve flags a real home
- `t8` — Declare associate on the Orin card profile with a measured budget
- `t9` — Ship the orin-associate deployment shape
- `t10` — Put the associate lane behind the gateway's authenticated front
- `t11` — Verify the untouched boundaries by test, not by assertion
- `t12` — Live acceptance: boot the orin-associate shape on the physical Orin
  and produce the #108 transcript *(added mid-run, operator-originated)*

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | Verification-only; the board-state file already satisfied both criteria. Criterion 2 met via its **abandonment** branch, not restoration — the operator chose not to restore the stopped lanes. No container touched. Zero new commits, the correct outcome for a verification task. |
| `t2` | delivered | vLLM spike transcript finalised (`0e3599c`). Agent flagged that the DSpark-404 evidence was **absent**; the orchestrator had destroyed it by writing two runs to one filename, and restored the captured lines (`32d362e`). |
| `t3` | delivered | llama.cpp transcript finalised as a citable **NO-GO** (`2b2b27c`), attributed to a llama.cpp build-version gap and explicitly *not* to sm_87. |
| `t4` | delivered | W4A4 claim carved per-checkpoint in `orin.py` + `orin.toml` (`776ea20`), test-first, with two guarding tests that also pin `worker` at `feasible = false` so the task could not silently flip a role. |
| `t5` | delivered | Stale Thor sentence corrected; the wedge is now stated as sm_110-specific. |
| `t7` | delivered | `vllm-associate` compose lane + all eight serve flags + `ASSOCIATE_IMAGE` (`0cc5658`), 31 tests. |
| `t6` | delivered | The tenth role across roles/catalog/gateway-config (`9fa9440`). Golden regeneration moved **only the `base` card, by one line** — strong evidence for the additive-only criterion. |
| `t8` | delivered | Budget measured live and declared — **but the value it shipped with was wrong and t12 corrected it.** See Drift. |
| `t9` | delivered | `orin-associate` shape (`604dd42`) — and it **found a real bug**: `compose_profile` always took `feasible` from the card, so any shape hosting an opt-in-core role whose card abstains rendered a bare `*_FEASIBLE=false` with no knobs. Latent for `thor-muse`/`thor-worker` on non-native cards. |
| `t10` | delivered | Exposure surface guaranteed + a `lobes doctor` `associate_auth_gate` check (`8955a1c`), 32 tests. |
| `t11` | delivered | 19 standing tripwires (`596cd77`); **no boundary violation found**. One tripwire later fired on t9's merge and was investigated, not adjusted away — see Drift. |
| `t12` | delivered | Live acceptance transcript (`c86cfb6`) — **GO, with three corrections the run forced**. |

## Mid-work Decisions

- `d1` (approved, `acceptable`) — t8's measured budget knobs moved from the Orin
  **card** profile to the **orin-associate shape** overrides; the card declares
  `feasible = false` and keeps the numbers as documentation. Reason: `associate`
  is an `OPT_IN_CORE_ROLE`, and t8's acceptance criterion contradicted that
  architecture — declaring it on the card made `orin-lobe`, which *drops*
  associate, render its full model and budget ("card passthrough broken"), and
  made machine-as-brain over-host the board.
- **The operator declined to restore the board's stopped lanes.** cortex,
  embedder, reranker and gateway were stopped to free memory for the spikes and
  deliberately left down, the board being repurposed for `associate`. This
  resolved plan risk `r1` (t1/t8 tension) by removing the second outage.
- **`associate` was created as a tenth role rather than a responsibilities token
  on `worker`.** `colleague-stack.md` routes behaviour differences to a token;
  the operator's load-bearing reason was the *separate public address*, keeping
  the `worker` seat free for a possible future worker/cortex switch.
- **Both vendor recipes were run verbatim first, and both failed.** Three
  independent defects were found (a nonexistent llama.cpp quantization tag; a
  GGUF that will not load in NVIDIA's own Jetson-Orin image; a vLLM DSpark repo
  id missing its `-NVFP4` infix). Recorded as a frame decision: the vendor's
  Jetson recipes are **unverified drafts**, not evidence of a validated
  deployment.
- **A first TTFT depth sweep was discarded.** It reported a ~1,175× win at depth
  32768; that was a `--enable-prefix-caching` artifact of repetitive filler. The
  cache-defeating rerun is what is recorded, and the inflated figures are named
  in the transcript rather than quietly replaced.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t8` (`d1`) | Budget knobs moved from the card to the shape — the criterion contradicted the `OPT_IN_CORE_ROLES` architecture. Shipped behaviour identical. | acceptable |
| `t8` | **The declared budget was wrong.** `util 0.63` was measured with `hand` NOT resident, reserving its declared `0.06` (3.68 GiB) as arithmetic headroom — a caveat t8 itself wrote into the profile. `hand` actually holds 5.84 GiB, and vLLM refused 0.63 once the real shape was up. Corrected to **0.56** in t12. No deviation record covers this; it was found by the live run. | needs-follow-up |
| `t7` | **DSpark was unreachable as shipped.** The lane had no speculative-config knob, so the lane's own ~78–81 tok/s headline could not be rendered by any shape. Cause was the orchestrator's framing of criterion c16, which asserted the speculative config was "already expressible" via `WORKER_SPECULATIVE_CONFIG` — the *worker* prefix, which does not generalize. `ASSOCIATE_SPECULATIVE_CONFIG` added in t12, **default-off and unmeasured on the full shape**. | needs-follow-up |
| `t6`/`t8` | Planned as one wave; they are **not independent** — `schema.py` validates profile role names against `roles.py`, so t8 alone produced 76 failures. Dependency added after the fact (risk `r6`); t8 rebased onto merged t6. | acceptable |
| `t11` | A tripwire fired on t9's merge (`orin-associate__spark.env` mentions associate). Investigated: the goldens matrix is every shape × every card, and the Spark's **actual** goldens are byte-unchanged. Test narrowed to shapes that do not host associate, derived from shape data with an `assert swept` guard. | acceptable |
| `t12` | Ran on the **pre-bump nightly** (`7c5a10e9…`) that the deployment's `.env` still pinned, not the `v0.27.1` the lane was measured on nor the template default `8bd082…`. The merge-only `.env` writer preserved the stale key. | needs-follow-up |

## Evidence

- tests: full suite `uv run pytest -n auto` — **3637 passed, 15 skipped** (baseline at branch point: 3398)
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r lobes` — all clean
- rubric: `uv run afi cli doctor . --strict` — healthy
- commits: `dbd3b9a..c86cfb6` (27 commits on `feat/lightning-on-orin-associate`)
- evidence: `docs/evidence/2026-08-25-spike-lightning-vllm-orin.txt`,
  `…-spike-lightning-llamacpp-orin.txt`, `…-measure-associate-budget-orin.txt`,
  `docs/evidence/2026-08-26-accept-orin-associate.txt`,
  `docs/evidence/2026-08-25-orin-board-state-lightning-spikes.txt`
- issues: `#216` (headless recovers unified memory — opened during this run)
- spec/plan: `docs/specs/2026-08-25-lightning-on-orin.md`, `docs/plans/2026-08-25-lightning-on-orin.md`

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| `associate` is a first-class role addressable by name, alias and `lobes up`, ranked `hand < multimodal < worker < muse < associate < main` | high | commit `9fa9440` · `lobes capabilities` shows `associate feasible=True ready=True loaded=True` |
| The Orin serves Lightning locally through the `orin-associate` shape | high | `docs/evidence/2026-08-26-accept-orin-associate.txt` · live `/health` 200 |
| Correctness: known-answer, multi-step reasoning and structured tool calls all PASS | high | `lobes assess --tools --model associate` — 17×23=391 PASS, 14:45→17:10 PASS, tool calling PASS |
| **52.5 tok/s decode at 32768 depth, no decay across 0→32768** — ~21.6× this board's 2.61 tok/s llama.cpp cortex; TTFT 16.9 s vs 610.0 s (~36×); prefill ~1,612 vs ~64 tok/s | high | acceptance transcript §5, cache-defeating sweep with `prompt_tokens` read back from the server |
| Unauthenticated requests are refused, **including over the tailnet address two peers actually used** | high | 401 on no-key / wrong-key / tailnet-no-key; 200 with valid bearer; `lobes doctor` discriminates both ways on the real deployment |
| sm_87 admits `modelopt_mixed` via Marlin (FP8, NVFP4 GEMM, NVFP4 MoE) with FlashAttention 2 | high | boot log kernel-selection lines, both spike and acceptance |
| The Orin clears the Mamba2 SSD Triton warmup that wedged the Thor on two engine versions | high | acceptance transcript §2 — line appears once and completes |
| Budget `util 0.56` / `max_model_len 128000`: weights 17.81 GiB, KV 9.35 GiB, pool 1,524,000 tokens, 11.91× | high | acceptance transcript §2 · `orin-associate.toml` |
| No existing role's gateway behaviour changed | high | 19 tripwires in `tests/test_associate_gateway_boundaries.py` · Spark goldens byte-unchanged |
| DSpark yields ~+45% decode on this board (54.3 → ~78–81 tok/s) | **low** | different engine build and resident set between the two measurements — indicative, not a controlled A/B |
| DSpark works on the shipped `orin-associate` shape | **unverified** | knob added but default-off; never booted on the full shape |
| The shape is reproducible from a clean `lobes fleet up` | **unverified** | `lobes fleet up` **fails** on this shape (pulls in the audio overlay); the run used an explicit file set |
| A from-source box can build its own gateway | **unverified** | it cannot — `Dockerfile.gateway` needs a published wheel; this run used a local-wheel variant |

## Remaining Work / Follow-up

- **`ASSOCIATE_SPECULATIVE_CONFIG` is default-off and unmeasured on the full
  shape.** The drafter costs KV (1,249,280 tokens with it at 0.63 vs 1,524,000
  without at 0.56) and the shape leaves ~1 GiB free. Arming it needs its own boot.
- **~1 GiB of headroom on a ZERO-swap board.** Issue `#216` (headless recovers
  unified memory) is now about *margin*, not just a bigger KV pool. Operator
  deferred to a separate session.
- **`lobes fleet up` cannot start this shape** — it builds the audio overlay a
  no-audio shape does not host. Needs an issue.
- **A from-source gateway cannot be built** without a published wheel. Needs an
  issue.
- **`stt`/`tts` advertise `feasible=True`** on a shape hosting no audio — stale
  `.env` keys preserved by the merge-only writer.
- **Engine drift**: three engines now in play (`v0.27.1`, `7c5a10e9`,
  `8bd082`); none compared against the others on this board.
- **embedder co-boot flakiness** — failed and restarted twice during bring-up,
  each time taking dependents down; single-shot `docker compose up` is unreliable.
- **No cross-box probe** — no peer has addressed this associate lane.
- **Marlin NVFP4 correctness on sm_87** rests on a small probe set; vLLM
  `#34694`/`#49070` report garbled output on this fallback on non-Blackwell parts.
- **`v7` follow-up**: `associate` and the Spark's `worker` serve the **same
  checkpoint**. Post-`#199` they may be poolable replicas of one lobe rather than
  two addresses — unevaluated, and it bears on whether the tenth role was needed.
- **Board state**: cortex, senses, embedder and reranker remain deliberately
  stopped; restore with `docker start llamacpp-cortex model-gear-vllm-embed
  model-gear-vllm-rerank model-gear-gateway`.
