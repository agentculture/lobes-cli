# flash-next on vllm

> lobes serves Qwen3.8-Flash-Next through vLLM on the Jetson AGX Thor, on the fleet's own engine rather than a second one
> instruction: Probe through the separate deployment's gateway origin, never the container port.

## Audience

- the lobes operator on the Jetson AGX Thor, and the fleet as a whole -- because the vLLM image is shared by four generate lanes, this test's upgrade lands on cortex, embedder, reranker and hand whether or not Flash-Next works

## Before → After

- After: two things exist that do not today: (1) a docs/evidence transcript for the fleet's vLLM image moved from its CURRENT shared pin -- sha256:8bd082c2, vLLM 0.26.1rc1.dev942+g5a4c8d992 -- to a main/nightly digest, with the incumbent Qwen3.8-27B NVFP4 cortex benchmarked before and after on this Thor; and (2) a verdict transcript for Qwen3.8-Flash-Next GGUF served through vllm-gguf-plugin on that image -- GO or NO-GO, with decode, prefill and TTFT measured at depth
  - instruction: Either upgrade Thor and Spark together, or unset `PRIMARY_PEER_ORIGINS` on the Thor for the window.

## Why it matters

- llama.cpp's prefill penalty is measured in-repo at ~25x (64 vs 1,612 tok/s) with ~36x TTFT (610 s vs 16.9 s at 32768), so a 125B model behind it is unusable for long-context work regardless of decode. vLLM is the fleet's engine, its gateway, roles, pressure policy and parsers all assume it, and the image move is a gain the fleet keeps even if this checkpoint fails.

## Requirements

- `VLLM_PLE_CPU_OFFLOAD`=1 -- the one vLLM feature that makes this model attractive, keeping the 51B N-gram lookup in host RAM (recipes.vllm.ai names it, and the model card calls the table 'more amenable to offloading than MoE') -- is a NO-OP ON THIS HARDWARE. Thor and Spark are unified-memory boards: host RAM and GPU RAM are the same physical pool, so moving the PLE table to 'host' frees nothing. The feature's whole premise is a discrete-GPU memory split this box does not have.
  - instruction: Measure once with the flag on and once off; report both.
  - honesty: if `VLLM_PLE_CPU_OFFLOAD` is set at all, the transcript states it changed nothing measurable on this unified-memory box, rather than leaving the reader to assume it helped
- PREFILL is a first-class success dimension, not a footnote. The in-repo measured llama.cpp prefill penalty (~64 vs ~1,612 tok/s, TTFT 610 s vs 16.9 s at 32768 -- docs/evidence/2026-08-26-accept-orin-associate.txt) is the operator's primary reason for choosing the vLLM path, so the test must measure prefill tok/s and TTFT at depth, not only single-stream decode.
  - honesty: prefill and TTFT are measured with cache-defeating unique text at more than one depth, the way docs/evidence/2026-08-26-accept-orin-associate.txt measured them -- a single short-prompt TTFT is not evidence
- the upgrade re-opens the Thor's FOUR validated `sm_110` divergences, which were validated against 0.23 and are not automatically true on 0.29: cortex `kv_cache_dtype`=auto (#109), embedder and reranker `attention_backend`=`TRITON_ATTN` (`FLASH_ATTN` pooling broken on `sm_110`, #105), and reranker `enforce_eager`=true (CUDA graphs unstable on `sm_110`). Stage 1 must re-run the three correctness probes -- cortex known-answer, embed ranking, rerank ordering -- and record whether each divergence is still needed.
  - instruction: Run the three correctness probes per docs/machine-profiles.md and record a verdict per divergence.
  - honesty: each of the four divergences is individually re-tested on the new image and recorded as still-needed, no-longer-needed, or newly-broken -- never carried forward untested
- vllm-gguf-plugin is not a pip install of a published wheel: its README installs from a clone with 'uv pip install -e . --no-build-isolation' and requires a CUDA or ROCm toolkit present. On this fleet that means building the plugin INSIDE an arm64 Jetson container against the image's own CUDA, and the resulting image must be pinned by digest like every other lane.
  - instruction: Record the plugin commit sha, not a branch name.
  - honesty: the plugin build is reproducible from the transcript: base image digest, plugin commit sha, and the exact build command, with the resulting image pinned by digest
- the GGUF addressing spelling is UNVERIFIED and must be settled before the lane is written: llama.cpp accepts '-hf <repo>:<quant>', and the operator's research proposed 'vllm serve unsloth/Qwen3.8-Flash-Next-GGUF:UD-`Q3_K_XL` --tokenizer Qwen/Qwen3.8-Flash-Next', but nothing read in this pass confirms vLLM accepts that repo:quant form or resolves a MULTI-SHARD GGUF. The lane may need a local shard path instead, which changes the compose lane's volume mounts.
  - instruction: Settle this with a scratch probe before writing any compose lane.
  - honesty: the addressing form that actually worked is recorded verbatim, and if repo:quant did not resolve, the local-shard-path alternative that did is written down with its volume mount
- part of the deliverable is a committed IMAGE LEDGER: every container recipe this work produces or pins is recorded in-tree as data -- image digest, engine and version (e.g. vLLM 0.29.x, plugin commit), the machine and arch it was built and validated for (thor / `sm_110` / arm64), the model(s) served on it, the build recipe or Dockerfile, and the date plus the evidence transcript that validated it. The ledger is the answer to 'which image is this, where did it come from, and what did it actually run'.
  - instruction: docs/image-ledger.md: one row per pinned image -- digest, engine+version, machine+arch, model(s), build recipe, date, evidence link.
  - honesty: every image digest this work pins appears in the ledger, and every ledger row names the machine and arch it was validated on plus the transcript that validated it -- a row with no evidence link is marked UNVALIDATED rather than left ambiguous
- the ledger covers BOTH stages and is written even on a NO-GO: the upgraded fleet vLLM image (stage 1) and the plugin-carrying Flash-Next image (stage 2) each get a row, including a stage-2 image that failed -- a recipe that did NOT work, recorded with why, is the most valuable row in the ledger for the next attempt.
  - instruction: Add the stage-2 image row even on a NO-GO, with its failure mode in the notes column.
  - honesty: a failed image still gets its row, marked with its failure mode, rather than being deleted from the record
- the blast radius is unchanged and still larger than 'this box': the DGX Spark proxies its EMBEDDER to this Thor (spark /capabilities: embedder proxied=true `hosted_by`=<http://thor...:8000>), so the stage-2 window costs the mesh embeddings. Stage 1's upgrade ALSO touches the embedder lane, so the Spark is affected by both stages, not just the down-window.
  - instruction: Re-check the Spark's /capabilities immediately before each stage and announce the window as embed-affecting.
  - honesty: the Spark's embedder dependency is re-checked live before each stage, and either announced or removed by the Spark declaring a local embedder first
- an ABORT CRITERION is written before each stage-2 boot attempt. The `sm_110` precedent stands: docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt records an indefinite hang on a hybrid state-space decode path, and Flash-Next is 3-of-4 layers Gated DeltaNet. A wall-clock timeout per attempt, then a written wedge verdict.
  - instruction: Write the timeout into the transcript before the first boot; on expiry kill the container and record the verdict citing the 2026-08-20 NO-GO.
  - honesty: the timeout is written down before the first boot attempt, not chosen after one hangs
- vLLM 0.29.0 DOES NOT EXIST. The latest tagged release is v0.28.0, published 2026-08-26 (gh api vllm-project/vllm releases, checked this pass); 0.27.1 was 2026-08-11. recipes.vllm.ai's 'vLLM 0.29.0+' therefore names main/nightly, not a release the fleet can pin to a version. Stage 1 is a NIGHTLY migration, not a release upgrade -- which is precisely why the image ledger (c18) is load-bearing rather than nice-to-have.
  - honesty: the digest the fleet actually runs is recorded, and the vLLM version it reports is read from inside the running container -- a nightly tag is never treated as a version
- stage 1 must be reversible by an .env OVERRIDE, never by editing the template's default digest: `VLLM_NIGHTLY_IMAGE` has the old digest as its template default, so setting the new one in the deployment .env makes rollback a one-line unset. Changing the default in lobes/templates/fleet/docker-compose.yml would turn rollback into a repo revert and would push the untested image at every other box that re-renders.
  - honesty: rollback is demonstrated once, not assumed: unset the .env override, bring the fleet up, confirm the old digest is running
- the replica pool CAN break silently on stage 1, but not for the reason you would guess. lobes/gateway/`_replicas.py`'s disqualifying fingerprint fields are `served_id`, quantization, `max_model_len` and runtime -- and runtime is engine-grained ('vllm' | 'llamacpp' | UNKNOWN), so a vLLM VERSION change does NOT disqualify a pair. What DOES is `max_model_len`: if the new nightly's memory behaviour forces the Thor's cortex below 262144 while the Spark stays at its window, the two silently stop pooling and every request serves local -- no error, just the pre-pool behaviour back.
  - instruction: After the upgrade, compare the Thor's and the Spark's /capabilities `max_model_len` and confirm the pool still selects the peer (X-Lobes-Route-Reason).
  - honesty: the post-upgrade pool check is an actual request that returns X-Lobes-Route-Reason naming a peer forward, not a comparison of two capabilities documents
- vLLM does not serve a GGUF the way llama.cpp does, so the ~90 GB file size is NOT the footprint. llama.cpp mmaps the file; vLLM loads and dequantizes, and the operator's own research flags 'runtime buffers, quant dequantization workspace, CUDA graphs, KV/state caches' on top. On a 122 GiB box with a 90 GB file that margin is thin enough to decide the outcome, so the load-time peak -- not the steady state -- is what stage 2 must measure first, and the lower rungs (UD-`IQ1_M` 74.5 GB) may be the only ones that load at all.
  - instruction: Watch free/tegrastats through the load, not just after it; record the PEAK.
  - honesty: the load-time PEAK is recorded, not only the post-load steady state -- a rung that loaded is not the same as a rung that fits
- the image ledger must be MACHINE-CHECKED, not prose: a test asserts that every sha256 digest appearing in lobes/templates/fleet/docker-compose.yml has a row in the ledger. There are 10 such pins today plus 18 more digest mentions across docs/\*.md, and a hand-maintained markdown table will drift from them within one PR -- the same drift the ledger exists to end.
  - instruction: Add a test alongside the existing goldens tests that greps the template for sha256 and fails on any digest missing from the ledger.
  - honesty: the ledger test fails on a digest that exists in the template but not the ledger -- proven by adding a digest and watching it go red, not by inspection
- CORRECTED: the fleet is on vLLM 0.26.1rc1.dev942+g5a4c8d992, NOT 0.23.1. docs/vllm-nightly-migration.md:424 records the before/after table -- sha256:7c5a10e9 was 0.23.1rc1.dev672 and is SUPERSEDED; the shipped shared pin sha256:8bd082c2 is 0.26.1rc1.dev942, flipped by the qwen3.8-cortex-upgrade plan t5. The earlier 0.23 figure came from docs/lfm2.5-1.2b-hand.md, which cites the OLD digest. The real gap is 0.26.1 -> main/nightly, materially smaller than the frame claimed.
  - instruction: Read the version from inside the running container before planning any upgrade; do not trust a per-model doc's version note, which may cite a superseded digest.
  - honesty: the fleet's actual running vLLM version is read from the container, and docs/lfm2.5-1.2b-hand.md's stale 0.23.1 reference is corrected or dated
- PARTIAL UPGRADES ARE THE POOL HAZARD, and the frame previously recorded the opposite as reassurance. Because `_replicas.py`'s runtime field is engine-grained (vllm|llamacpp|UNKNOWN), a Thor on a nightly digest and a Spark on 0.26.1rc1.dev942 remain POOL-COMPATIBLE and will serve each other's cortex requests despite running different engine builds. That is a latent behavioural-mismatch hazard, not a safety property. Until the fingerprint carries an engine VERSION, the mitigation is procedural: upgrade every pooled box together, or drop the peer from \*`_PEER_ORIGINS` for the duration of a staggered upgrade.
  - instruction: Upgrade Thor and Spark together, or unset `PRIMARY_PEER_ORIGINS` on the Thor for the duration; record each pooled box's engine build in the transcript.
  - honesty: the before-state version in the transcript is the one read from the running container, and matches docs/image-ledger.md's row for the shipped digest
- the shared digest drives SEVEN lanes across THREE boxes, not four and not six: vllm-primary, vllm-embed, vllm-embed-deep, vllm-rerank, vllm-hand, vllm-worker and vllm-associate all interpolate `VLLM_NIGHTLY_IMAGE` (grepped from the template; vllm-embed-deep was omitted from an earlier count of this same list). worker and associate are the Nemotron Lightning lanes -- served on the SPARK and the ORIN from this same template -- so the upgrade's reach extends past this Thor to any box that re-renders.
  - instruction: grep -c `VLLM_NIGHTLY_IMAGE` on the template and name every matching service in the transcript; do not summarise the count from memory.
  - honesty: the transcript names all seven lanes and the three boxes the shared digest reaches, with embed-deep listed explicitly rather than folded into embed

## Honesty conditions

- the gear answers a real completion through a lobes gateway on this Thor, served by vLLM -- so 'lobes serves it on the fleet's own engine' is literally true
- the three native-format sizes are re-checked against their model cards before stage 2 starts, so 'nothing native fits' is current rather than a snapshot -- a new sub-122 GiB quant would change the whole approach
- the final transcript states plainly whether the vLLM path beat, matched, or lost to the llama.cpp expectation -- including the case where the research was right and this path fails
- the transcript names every lane the shared image touched, so a reader can tell which results are fleet-wide and which are Flash-Next-only
- the llama.cpp prefill comparison is cited to its transcript and stated as a different-board measurement, not presented as a Thor figure
- stage 1 is committed and its evidence landed before stage 2 begins -- provable from commit order, so a stage-2 NO-GO cannot retroactively strand it
- both axes are reported with their comparators and their limits (different engine, different quantization, different board for the llama.cpp prefill figure) so each is a bar rather than a like-for-like A/B
- the incumbent cortex baseline is captured BEFORE the image swap -- per docs/model-switch-playbook.md that number cannot be recovered afterwards
- the transcript records which engine build EACH pooled box ran during the window, so a mixed-version pool is visible in the evidence rather than invisible

## Success signals

- stage 2 passes only on BOTH axes: single-stream decode >= 25 tok/s at MAXN (the operator's bar, ~2x this box's incumbent vLLM NVFP4 cortex at 12.1 tok/s -- docs/evidence/2026-08-20-accept-cortex-local-thor.txt), AND prefill/TTFT in vLLM's own territory rather than llama.cpp's, measured at depth against the in-repo llama.cpp figures (~64 tok/s prefill, 610 s TTFT at 32768). Missing either axis is a NO-GO: decode alone was never the reason to choose this engine.
  - instruction: Report decode, prefill and TTFT at 0 / 8K / 32K depth beside the 12.1 tok/s incumbent and the llama.cpp prefill figures.
- stage 1 passes independently of stage 2: the fleet runs on the upgraded vLLM image with cortex, embedder, reranker and hand all healthy, the three `sm_110` correctness probes green, and the incumbent Qwen3.8-27B NVFP4 cortex benchmarked before and after -- a regression against 12.1 tok/s is a stage-1 FAILURE that rolls the image back regardless of what stage 2 would have shown.
  - instruction: Benchmark the incumbent cortex on the old digest first; that baseline is unrecoverable after the swap (docs/model-switch-playbook.md).

## Scope / boundaries

- GGUF IS THE ONLY FORMAT THAT FITS. Every native checkpoint exceeds this box's 122 GiB: BF16 335.28 GiB, the official Qwen/Qwen3.8-Flash-Next-FP8 172.78 GiB (both read off recipes.vllm.ai), and RadixArk/Qwen3.8-Flash-Next-NVFP4 135 GB -- which quantizes ONLY the 48 MoE layers' routed experts to W4A4 and is published for SGLang, not vLLM. So 'pivot to vLLM' cannot mean 'serve the native checkpoint'; it can only mean serving the SAME Unsloth GGUF through vllm-gguf-plugin.
  - instruction: Re-check the Qwen FP8, RadixArk NVFP4 and any newer quant cards before committing to the GGUF route.
- the operator's own cited research reaches the OPPOSITE conclusion from the pivot and that disagreement is recorded, not smoothed: it states 'llama.cpp -> most likely to run the 90 GB Q3 successfully' and 'vLLM GGUF -> worth testing, but likely to hit architecture/weight-mapping/kernel issues'. Every check this pass ran agrees with that ordering. The pivot is therefore a deliberate fleet-coherence choice, not an expected-to-work-better choice, and must be justified on those terms.
  - instruction: Write the comparison into the conclusion even on a NO-GO.
- this is a TWO-STAGE test and stage 1 must stand alone: the vLLM image upgrade with a cortex before/after benchmark is committed and valuable on its own. Flash-Next is stage 2. A stage-2 NO-GO does not roll back stage 1, and stage 1 is not allowed to depend on Flash-Next working.

## Non-goals

- no role is re-pointed and no tier alias changes. Flash-Next lands as `role_hint`='candidate' if it lands at all; the cortex primary stays unsloth/Qwen3.8-27B-NVFP4 throughout both stages.

## Assumptions

- vllm-gguf-plugin is the ONLY vLLM route to a file that fits, and it is the weakest link: its README's tested quant list is `Q6_K` / `Q8_0` / `IQ4_XS` / `Q4_K_M` / `Q4_0` -- neither UD-`IQ1_M` nor UD-`Q3_K_XL` is on it -- its architecture list stops at Qwen 2.5/3 with no qwen4exp entry, it documents no multi-shard GGUF support, and it warns in its own words that appearing in vLLM's supported-model list 'does not by itself guarantee GGUF compatibility'. The plugin repo is live (pushed 2026-08-25) but nothing in it names this architecture.
- an arm64 / `sm_110` vLLM 0.29.0+ image may not exist. The fleet's four generate lanes share vllm/vllm-openai@sha256:8bd082c2..., and the Gemma lane uses nvcr.io/nvidia/vllm:26.04-py3 specifically because it is the NGC ARM64/Blackwell build. Whether upstream publishes a 0.29 arm64 image carrying `sm_110` kernels -- or whether one must be built -- is UNVERIFIED and is the first thing stage 1 must settle.
- c14's worry is HALF answered: arm64 images DO exist -- vllm/vllm-openai publishes nightly, nightly-aarch64, cu129-nightly and cu129-nightly-aarch64, multi-arch amd64+arm64 (Docker Hub tag probe, this pass). What is NOT answered is `sm_110` KERNEL coverage: the in-repo memory 'CUDA wheel arch is not a family' records that cu128 ships no `sm_110` SASS and no PTX while cu130 is Thor-safe, and cu129 is UNTESTED either way. An arm64 image that boots on this board can still carry no kernels for it.

## Scope exploration

- `s1` — `recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next + HF cards for Qwen FP8 and RadixArk NVFP4`: the vLLM recipe targets 4x GB300 or 8x H200 with FP8 at 172.78 GiB and tensor-parallel-size 4 -- a multi-GPU datacenter recipe. No single-128GB-box path exists in any native format; the 135 GB NVFP4 misses this box by ~13 GB and is an SGLang checkpoint
  - seeds: `c2`
- `s2` — `VLLM_PLE_CPU_OFFLOAD in the vLLM recipe vs Jetson AGX Thor unified memory`: the offload knob assumes VRAM and host RAM are separate budgets; on `sm_110` unified memory they are one 122 GiB pool, so the headline argument for the vLLM path does not transfer to this fleet
  - seeds: `c3`
- `s3` — `lobes/templates/fleet/docker-compose.yml VLLM_NIGHTLY_IMAGE digest + docs/lfm2.5-1.2b-hand.md version note vs recipes.vllm.ai`: one digest is shared by four generate lanes, so the version axis is fleet-wide, not per-lane; the 0.23 -> 0.29 jump is the real cost of the vLLM path and the frame must price it
  - seeds: `c4` (rejected)
- `s4` — `github.com/vllm-project/vllm-gguf-plugin README + repo metadata`: the plugin exists and is actively pushed, but supports neither this architecture nor these quant types by its own documentation, and the checkpoint ships as multi-shard -- three separate unknowns stacked on the version gap
  - seeds: `c5`
- `s5` — `the operator-supplied research vs this pass's independent verification of all six cited artifacts`: every cited artifact verifies (vllm-gguf-plugin live, vLLM issue 53908 open 2026-08-26, llama.cpp issue 27766 open 2026-08-26, QwenLM repo live) -- so the research is sound AND its own recommendation points at llama.cpp
  - seeds: `c6`
- `s6` — `docs/evidence/2026-08-26-accept-orin-associate.txt prefill comparison + docs/qwen3.8-27b-gguf-llamacpp.md MAXN figures`: the repo already measures the llama.cpp prefill penalty at ~25x on one board and ~36x on TTFT; my earlier scope pass weighed only decode and therefore understated the case FOR vLLM
  - seeds: `c7`
- `s7` — `one VLLM_NIGHTLY_IMAGE digest shared by four generate lanes`: because the digest is fleet-wide, upgrading it is unavoidably a fleet event -- which is exactly why it should carry a cortex before/after benchmark rather than being treated as incidental to this lane
  - seeds: `c8` (rejected)
- `s8` — `lobes/templates/fleet/docker-compose.yml image pins (vllm/vllm-openai digest x4 + nvcr.io/nvidia/vllm:26.04-py3)`: the fleet already needs a special ARM64/Blackwell image for one lane, which is evidence that arch coverage is the binding constraint on this platform, not vLLM's version number alone
  - seeds: `c14`
- `s9` — `docs/machine-profiles.md Thor divergences + lobes/machines/_traits.py SM_110`: the divergences live in a chip-strategy registry, not the profile TOML, so an upgrade that fixes one requires a code change to drop it -- and an upgrade that BREAKS one would surface as a silent pooling or CUDA-graph failure, not a boot error
  - seeds: `c15`
- `s10` — `vllm-gguf-plugin README install instructions vs the fleet's digest-pinning convention`: the plugin adds a from-source build step to what is otherwise a pull-a-digest lane; the same no-floating-tag rule the llama.cpp lane records applies here
  - seeds: `c16`
- `s11` — `vllm-gguf-plugin README (no multi-shard section) + the operator's proposed serve command`: two separate unknowns -- the repo:quant addressing form and multi-shard resolution -- neither documented; both are cheap to settle with a scratch probe before any lane is written
  - seeds: `c17`
- `s12` — `existing digest pins: 10 sha256 lines in fleet/docker-compose.yml + 18 across docs/*.md, and 5 Dockerfile.* templates -- no ledger file exists`: image provenance is currently scattered across compose comments and per-model docs with no single index, which is exactly the drift the operator's ledger request addresses; nothing in docs/ is named ledger/image/recipe/pin today
  - seeds: `c18`
- `s13` — `challenge pass / unstated-assumptions lens: gh api vllm-project/vllm releases vs recipes.vllm.ai's stated floor`: the frame said 'upgrade to 0.29.0+' as though it were a release; no such release exists, so the target is a moving nightly and every claim about 'the upgraded image' must name a digest, never a version
- `s14` — `challenge pass / hardware lens: Docker Hub vllm/vllm-openai tag arch list vs the cu128/sm_110 SASS precedent (#145)`: arch availability and kernel coverage are two different questions; the frame conflated them, and the repo has a recorded case (chatterbox) of an image that installed fine and had no `sm_110` kernels
- `s15` — `challenge pass / adjacent-systems lens: grep VLLM_NIGHTLY_IMAGE in lobes/templates/fleet/docker-compose.yml`: SEVEN services, three boxes -- vllm-primary, vllm-embed, vllm-embed-deep, vllm-rerank, vllm-hand, vllm-worker, vllm-associate. Counted twice and got it wrong both times (four, then six, each dropping embed-deep), which is itself the argument for the ledger listing services rather than stating a number.
- `s16` — `challenge pass / reversibility lens: the <digest> default form`: the template already encodes the safe rollback path; the frame never said which of the two ways to apply the upgrade, and only one of them is reversible in one line
- `s17` — `challenge pass / adjacent-systems lens: lobes/gateway/_replicas.py fingerprint fields (L102-111, L177-180, L285-290)`: runtime is engine-grained so the version upgrade is SAFE on that axis -- a clean result worth recording -- while `max_model_len` is the field that actually carries silent-unpooling risk, which the frame never considered
- `s18` — `challenge pass / failure-modes lens: vLLM GGUF load path vs llama.cpp mmap, against 122 GiB`: the frame carried the llama.cpp-era assumption that file size approximates footprint; on vLLM that assumption does not hold and it inverts the quant ladder's logic -- the biggest rung may be the one that cannot load
- `s19` — `challenge pass / operations lens: 10 sha256 pins in the template + 18 in docs, no index, vs c18's ledger`: the ledger's value depends entirely on staying current; this repo already proves scattered pins drift, and it already has a goldens-test convention that can enforce the ledger the same way
- `s20` — `challenge pass / migration lens: docs/vllm-nightly-migration.md`: the fleet has migrated its nightly once already and left a verification-only doc but no ledger -- direct evidence for why c18 is worth building now, and a ready template for stage 1's before-state
- `s21` — `challenge pass / lenses examined with no finding`: SECURITY: images come from vllm/vllm-openai and a vendor plugin repo, same trust posture as the existing pinned digests, and the lane publishes no host port -- no new exposure. DATA LOSS: no schema, no user data, GGUF read-only. CONCURRENCY: single-stream test, -np/-max-num-seqs at 1. These three were examined and are clean; that is a record of what was looked at, not a claim that nothing remains.
- `s22` — `qodo PR#218 review comment 1 (runtime version omitted from fingerprint) vs lobes/gateway/_replicas.py`: the reviewer is right about the hazard and the frame framed it backwards -- engine-grained runtime was recorded as 'safe on that axis' when it in fact permits pooling behaviourally different builds. Changing the fingerprint is a behaviour change to the #199 validated replica-pool contract and needs its own spec; the spec-side fix is to name the hazard and give the procedural mitigation.

## Decisions

- carried over unchanged from the llama.cpp frame: Thor-only; all production models come down for the stage-2 window; the lane runs from a SEPARATE deployment dir (~/.lobes-vllm-next, --compose-dir) so ~/.lobes/.env is never opened; MAXN is verified and pasted beside every measurement; disk is reclaimed rung by rung; and the quant choice is DEFERRED to measured speed and quality rather than fixed in advance.
- stage 1 follows an existing in-repo precedent rather than inventing a shape: docs/vllm-nightly-migration.md is a 'before-state verification + baselines to beat' doc from the previous nightly migration, citing exact file:line pins and making no mutation. Stage 1's before-state doc is its sibling, and the image ledger (c18) is the durable index that migration should have left behind.
- the deliverable is TWO documents, not one: docs/image-ledger.md stays fleet-wide and about the IMAGES (digests, versions, arch validation), while docs/qwen3.8-flash-next-gguf-llamacpp-vllm.md carries this checkpoint's what / why / why-not and is named after the model so it sits beside docs/qwen3.8-27b-gguf-llamacpp.md. Some overlap between them is accepted deliberately; each cross-links the other.
- experiment docs live in docs/experiments/, not flat in docs/. An experiment is a checkpoint, engine or runtime seriously considered and not (or not yet) put into service; the folder's README defines what belongs there and what goes to docs/<model>.md, docs/evidence/, docs/specs/ or docs/image-ledger.md instead. The Flash-Next evaluation is its first entry: docs/experiments/qwen3.8-flash-next-gguf-llamacpp-vllm.md. Scoped deliberately narrow -- a docs/models/ reorg would change lobes/catalog.py's doc= contract (filename-only, resolved as docs / model.doc and asserted by tests/`test_catalog.py`::`test_every_doc_file_exists`), regenerate the switch-plan goldens, and sweep 456 docs/\*.md references; that is its own PR, not this one.
- the vLLM image move (0.26.1rc1.dev942+g5a4c8d992 -> a main/nightly digest; there is no 0.29 release to name) is scoped as a SHARED fleet benefit, not a cost carried by this test alone: it re-opens a measurement of the existing Qwen3.8-27B NVFP4 cortex on newer code, which may itself gain speed. The upgrade therefore gets its own before/after measurement of the incumbent cortex, so the fleet keeps the gain even if Flash-Next fails.

## Open parks

- [unknown_nonblocking] whether a Flash-Next quant small enough for a 122 GiB box will exist -- RadixArk's 135 GB NVFP4 quantizes only the routed experts, so the ~35 GB PLE table is the floor; a PLE-aware quant is the open question, and vLLM issue 53908 (opened 2026-08-26) shows the ecosystem is still designing that
- [unknown_nonblocking] whether the cu129 nightly carries `sm_110` SASS -- decidable only by pulling the image and running cuobjdump --list-elf, which is a stage-1 first step rather than a spec-time answer
- [follow_up] EXECUTION DEFERRED 2026-08-27, not abandoned: the operator stopped before stage 1. What ships now is the knowledge -- the exported spec and the image ledger. Resuming needs no re-derivation: the blockers are recorded (no vLLM 0.29 release exists so the target is a nightly digest; cu129 `sm_110` SASS unverified; vLLM GGUF load-time peak unknown against 122 GiB).
