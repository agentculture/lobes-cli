# nemotron-lightning-worker

> Thor's worker seat moves to nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 — a fast, text-only, non-coding doer (action selection, tool loops, RAG, digestion, repo inspection — never code authoring or final judgment). The Thor machine image rides the same official vLLM nightly the Spark validated for Qwen3.8. Thor and Orin repoint their cortex proxy at the Spark's new served id unsloth/Qwen3.8-27B-NVFP4 (#186), and the worker validation matrix lands per #187 (with #183's hand budgets tracked as their own effort).
> instruction: Verify by the four success-signal transcripts (c17) plus a green CI run on the repo changes; every 'validated' word in docs must cite a transcript path (#108)

## Audience

- mesh consumers that address roles through the gateway contract (colleague — under renovation, so the worker contract can change; embodiment; reachy-mini-cli; eidetic) and the operators of the three deployed boxes (Thor, Spark, Orin)

## Before → After

- Before: today worker is the Qwen3.6-35B multimodal doer on an older nightly, its advert claims vision the replacement won't have; this Thor checkout is 7 commits stale; Thor and Orin still mirror the OLD cortex id so a repoint-less box would 404; hand is validated on Orin only (Thor blocked on #181, Spark never exercised)
- After: Thor serves worker on Lightning NVFP4 (text-only, nemotron parser pair, measured budget) on the same official nightly the Spark runs; the advert tells the deployed truth (no vision tokens, probe-verified tools); Thor and Orin answer model=cortex 200 by proxy under the Qwen3.8 served id; and hand has a reproducible budget on Thor AND Spark per #183

## Why it matters

- the reference cognitive split (#187) puts presence on Gemma, agency on Lightning, hard thinking/coding on Qwen3.8 — a worker seat tuned for fast moderate-accuracy agent loops instead of a repurposed coder, plus proxy correctness and reproducible budgets, keeps the mesh honest and fast

## Requirements

- lobes/catalog.py gains a nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 entry with role_hint=worker (hybrid Mamba-2 + sparse-MoE ~3B active, text-only, 1M ceiling, official NVFP4); the unsloth/Qwen3.6-35B-A3B-NVFP4 entry (origin/main catalog.py:751, VALIDATED 2026-07-31) stays in-tree as a demoted candidate — cite-don't-delete, same as muse
  - honesty: the catalog entry's config facts (arch class, quant_method, context, tokenizer, parser names) are read from the checkpoint's own config.json/hf_quant_config.json fetched from HF — never from model-card prose or the vLLM Jetson guide
- lobes/roles.py ROLE_RESPONSIBILITIES['worker'] is redefined per #187: drop image_understanding/video_understanding (Lightning is text-only), keep tool_use/execution/ground_work, and express 'repo inspection/navigation/running-authorized-commands yes, code authoring no' — extending the responsibility vocabulary explicitly if repo_action cannot carry that distinction (per #187's instruction not to hide policy in prose)
  - honesty: the inspection-vs-authoring split is expressed in machine-readable responsibility tokens (extended vocabulary if needed), never in prose or model-name checks — a consumer reading only responsibilities/forbidden_responsibilities gets the whole policy
- lobes/profiles/builtin_shapes/thor-worker.toml swaps its overrides.worker model to the Lightning checkpoint; its gpu_mem_util/max_model_len budgets must be MEASURED on the physical Thor live boot, never copied from the Qwen 35B values (util 0.45 / 262144 / 14.07x ceiling are checkpoint-specific measurements from 2026-07-31; thor-muse's 0.40-refused-0.55-measured precedent is the discipline)
  - honesty: every budget number committed to thor-worker.toml traces to a live Thor boot transcript under docs/evidence/ dated for THIS checkpoint; no number is inherited from the Qwen 35B measurements
- #186 lands as per-box deployment ops on Thor and Orin, not gateway code: each box mirrors PRIMARY_SERVED_NAME=unsloth/Qwen3.8-27B-NVFP4 into its .env beside PRIMARY_PEER_ORIGIN/_PROXY/_API_KEY, recreates the gateway with its full -f overlay set, and proves cortex proxied=true/ready=true + a model=cortex 200 with X-Lobes-Proxied-By; evidence transcripts land under docs/evidence/ per #108
  - honesty: acceptance is observed on both boxes, not inferred: lobes capabilities shows cortex proxied=true/ready=true/hosted_by, and a model=cortex chat request answers 200 carrying X-Lobes-Proxied-By, captured in a transcript per box
- the runtime advert must describe the DEPLOYED worker (#187): served model id/revision, quantization, configured context, modalities=text-only, tools from a measured tool-call probe, MTP mode actually enabled, loaded/ready/feasible separately — stale image_understanding/video_understanding claims removed everywhere they surface (roles.py, capabilities, colleague-stack.md, CLAUDE.md, qwen3.6-35b doc cross-refs)
  - honesty: the tools=yes advert derives from a measured structured tool_calls probe against the live Lightning lane (parser pairs fail silently — 200 OK with the call leaked into content counts as FAILURE); a repo-wide grep proves no surface still claims worker image/video understanding
- docs follow the shipped state: a new docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md per-model doc (provenance-fetched config), demotion notes in docs/qwen3.6-35b-a3b-nvfp4.md, role-table updates in docs/colleague-stack.md and CLAUDE.md, and evidence transcripts under docs/evidence/ for the Thor Lightning boot, the validation matrix, and both #186 repoints
  - honesty: docs ship the deployed state only: the Lightning doc carries measured numbers with transcript citations, the Qwen 35B doc is marked demoted-not-deleted, and no doc or capabilities output claims validated without evidence (#108)
- hand validation per #183: reproduce the Orin probe on Thor (re-attributing #181's LoRA embedding-slot failure under the new nightly — fixed, changed, or re-blocked, reported faithfully) and on the Spark (its 0.06 is a declared hypothesis); the adapter-serving surface stays blocked on unsloth-cli#16 and is reported as such, never claimed validated
  - honesty: hand results are reported faithfully per card: a Thor failure is recorded as blocked-with-attribution (not skipped), the Spark run follows the Orin probe template (budget, known-answer, structured tool_calls via lfm2 parser), and adapter serving is stated blocked on unsloth-cli#16
- the WORKER served-id swap is its own raw-id break, separate from #186: docs/model-switch-playbook.md §2 records that no mesh consumer addresses roles by name — all pin raw served ids — so the swap needs its own rollout-notes audit (who pins unsloth/Qwen3.6-35B-A3B-NVFP4), and any box that refers/proxies worker (WORKER_PEER_ORIGIN) must mirror the NEW WORKER_SERVED_NAME or its readiness probe sits proxied=true/ready=false and 404s — the exact 2026-08-05 cortex lesson applied to worker
  - honesty: the audit greps the sibling checkouts (colleague, embodiment, eidetic, reachy-mini-cli) for the old worker id and the rollout note names every hit; on-box WORKER_SERVED_NAME mirrors are verified by capabilities showing worker ready=true from the referring box
- rollback stays cheap until acceptance: the Qwen 35B worker must remain restorable during the whole rollout — thor-worker shape re-render is byte-for-byte restorable by design, and the old checkpoint + old image stay cached on the Thor disk until the #187 validation matrix passes (verify disk headroom before pulling ~20 GiB of Lightning weights)
  - honesty: rollback is exercised or explicitly waived in the transcript: either one dry-run restore to the Qwen shape is shown working, or the operator records accepting untested rollback

## Honesty conditions

- the announcement holds only when all four c17 transcripts exist under docs/evidence/; until then every doc states declared/UNVALIDATED per #108
- verified by reading origin/main's compose template: worker/hand/primary/embed/rerank/minor all default to the 8bd082 digest; the only Thor-side template question is the Gemma lanes, which are OUT of this effort
- the swap lands with zero diffs under lobes/gateway/ and lobes/roles.py's channel tables; if any gateway code change proves necessary, this boundary is reopened explicitly rather than quietly widened
- the served max_model_len on Thor is whatever the live boot sustains, validated progressively; no config allocates the 1M ceiling by default, and DSpark ships only with its own Thor evidence or not at all
- consumer impact is bounded by contract: worker/cortex stay addressed by role/alias through the gateway, and the one raw-id break (#186 cortex served id) is announced via the rollout-notes pattern before any box flips
- each clause of the after-state maps to one of the four success-signal transcripts; any clause without a landed transcript is stated declared/UNVALIDATED, never claimed done
- each before-state fact is cited to what was read: origin/main catalog/roles/shape files, the 7-commit git delta, issues #186/#183, and docs/lfm2.5-1.2b-hand.md — none is asserted from memory
- the cognitive-split rationale is #187's own operator-stated design, quoted not invented; lobes-cli implements only its Thor/repo side (colleague owns orchestration, c8)
- a success signal counts only as a landed transcript file under docs/evidence/ — a green local run without the committed transcript does not satisfy it

## Success signals

- acceptance transcripts under docs/evidence/ for: (1) Lightning live boot on Thor with measured budget + tool-call probe, (2) the #187 matched Qwen-vs-Lightning validation matrix incl. negative/escalation tasks, (3) model=cortex 200-by-proxy from Thor AND Orin with X-Lobes-Proxied-By, (4) hand budget probes on Thor and Spark per the Orin template

## Scope / boundaries

- no gateway code change is expected for either the repoint or the model swap: WORKER_*/PRIMARY_* env channels are role-generic (roles.py maps worker->worker uniformly; the #165 fix already added worker to server.py's resolution tables), and #187 requires preserving the #178 peer-proxy behaviour so consumers resolve worker by role, never by hardcoded endpoint/model id
- Lightning's 1M context is a capability CEILING, not a default allocation (#187 point 7): the shape's max_model_len starts from measured Thor budget and validates long-context progressively; DSpark speculative decoding is optional/experimental on Thor and must earn its own evidence

## Assumptions

- the 'new nightly as on Spark' image move is already TEMPLATE-side done at origin/main: the fleet-wide VLLM_NIGHTLY_IMAGE default is the official-nightly digest sha256:8bd082... (vLLM 0.26.1rc1.dev942) that #185 validated for Qwen3.8 on the Spark, and vllm-worker falls back to it (WORKER_IMAGE unset) — so the Thor side is deployment work (git pull, re-render, image pull), not a template change
- this Thor checkout is 7 commits behind origin/main (0.54.6 vs 0.57.0 — Qwen3.8 cortex, hand role, Orin card, worker peer-proxy fix all upstream); all work starts from a pulled origin/main, and stale local memory (thor-lobes-deployment.md still says Thor hosts muse) must not drive decisions
- the capability-tier ordering stays minor < multimodal < worker < muse < main with worker now text-only — the order ranks general capability for alias resolution, not modality; catalog.py's tier_aliases last-occurrence ordering comment is re-read against this, not silently assumed

## Scope exploration

- `s1` — `lobes/catalog.py (origin/main:751-830 worker entry; :262 Qwen3.8 cortex entry)`: worker gear is a first-class catalog entry with fetched-config provenance (config.json/hf_quant_config.json comments), role_hint=worker, self-hosted-MTP speculative_config, moe_backend auto-select rationale; a Lightning entry needs the same provenance bar and the Qwen 35B entry stays demoted, mirroring how mmangkad candidates were kept
  - seeds: `c2`
- `s2` — `lobes/roles.py (origin/main:159-250 ROLE_RESPONSIBILITIES)`: worker's tuple today carries image_understanding, video_understanding, repo_action; cortex's comment claims cortex is 'the ONLY role that both SEES and DECIDES' partly BECAUSE worker is forbidden final_decision — a text-only Lightning worker must drop the two vision tokens and the vocabulary may need a new token pair to split repo inspection from code authoring
  - seeds: `c3`
- `s3` — `lobes/profiles/builtin_shapes/thor-worker.toml (origin/main)`: shape carries worker's FULL declaration (OPT_IN_CORE_ROLES mechanism, card profiles silent, base.toml veto); every budget number in it is annotated MEASURED-on-Thor with the ceiling-vs-saturation warning; the sm_110 MoE-backend auto-select note is Qwen-MoE-specific and must be re-derived for Lightning's Mamba-2+MoE hybrid
  - seeds: `c4`
- `s4` — `docs/vllm-nightly-migration.md + docs/qwen38-rollout-notes.md (origin/main)`: the nightly-unification plan and the Qwen3.8 rollout notes are the prior art: rollout notes record the 2026-08-05 lesson that peer-readiness probes look for the ADVERTISED id in the peer's /v1/models, and name every raw-id pinner that 404s on a served-id swap
  - seeds: `c5`
- `s5` — `lobes/templates/fleet/docker-compose.yml (origin/main:53,209,294,352,519,1109 image pins)`: primary/embed/rerank/minor/hand/worker all default to VLLM_NIGHTLY_IMAGE=vllm/vllm-openai@sha256:8bd082... (0.26.1rc1.dev942, the #185 Spark-validated official nightly); only the Gemma lanes still build the older 0.23.1 digest via Dockerfile.vllm-gemma4 — so the Thor image bump is a deployment re-render, with sm_110 compatibility of that digest parked as blocking unknown v1
  - seeds: `c5`
- `s6` — `issue #186 + origin/main env.example:56 (PRIMARY_SERVED_NAME=unsloth/Qwen3.8-27B-NVFP4)`: env.example already ships the new id, so fresh scaffolds are correct; only the two deployed Jetson boxes' .env files lag — the peer-readiness probe matches the ADVERTISED id against the peer's /v1/models, so an un-mirrored box sits proxied=true/ready=false and 404s
  - seeds: `c6`
- `s7` — `lobes/roles.py role->backend tables (origin/main:97-130) + git log 2ede672 (#165)`: worker rides the same FEASIBLE/PEER_ORIGIN/PROXY/KEY channels as every role since the #165 silent-inertness fix; a checkpoint swap is pure env/shape data through these channels
  - seeds: `c7`
- `s8` — `issue #187 (validation matrix + runtime truthfulness sections)`: the acceptance bar is a matched Qwen-35B-vs-Lightning task matrix on the same Thor runtime (worker-shaped tasks + explicit negative/escalation tasks) plus an advert-truthfulness audit; this seeds both the truthfulness requirement and the measured-budget discipline
  - seeds: `c10`, `c4`
- `s9` — `git HEAD..origin/main (7 commits, ea04e1c..2b05c9e)`: upstream already landed: 2b05c9e Qwen3.8 cortex at 1M (#185), 86967f4 hand ninth role (#184), 8339439 Orin first-class card (#176), 2ede672 worker peer-proxy fix (#165) — several assumptions in this box's local state and memory are stale
  - seeds: `c11`
- `s10` — `issue #183 + docs/lfm2.5-1.2b-hand.md (origin/main)`: #183 is the hand role's budget/adapter validation (Orin VALIDATED 2026-08-10; Thor blocked on #181; Spark declared-only; adapters blocked on unsloth-cli#16) — it overlaps this effort only through the shared Thor nightly image, hence pending question q1
- `s11` — `docs/ tree (colleague-stack.md, qwen3.6-35b-a3b-nvfp4.md, evidence/ naming convention)`: every prior model move shipped a per-model doc + evidence transcript (2026-07-31-accept-worker-thor.txt is the template for this swap); the #108 rule forbids claiming validated without the transcript
  - seeds: `c12`
- `s12` — `challenge pass / adjacent-systems lens: docs/model-switch-playbook.md §2 + docs/qwen38-rollout-notes.md`: rigorous depth (hardware + distributed-state + migration signals, c19); the playbook's every-consumer-pins-raw-ids rule applies to the WORKER id swap too — the frame had covered only the cortex break; seeded the worker rollout-audit requirement
  - seeds: `c22`
- `s13` — `challenge pass / probe: HF API + config.json of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`: read-only probe settled gating (UNGATED), arch (NemotronHForCausalLM/nemotron_h), 1M max_pos, quant_method=modelopt, no vision_config; nemotron_h serving support in the 8bd082 nightly remains park v2's spike — the probe answers the checkpoint side only
  - seeds: `c24`
- `s14` — `challenge pass / reversibility lens: shapes.py restore mechanic + Thor disk`: shape re-render is byte-for-byte restorable by design, but checkpoint/image cache retention and disk headroom were nowhere in the frame; seeded the rollback requirement
  - seeds: `c23`
- `s15` — `challenge pass / co-residency+concurrency lens: thor-worker.toml hosts list (hand, embedder, reranker, stt, tts)`: clean pass with one condition: the Lightning budget measurement must be taken with all co-residents up (the Qwen 0.45 was measured co-resident), and the hand-on-Thor probe (c20/#181) shares the same boot — residual risk is the known unified-memory first-boot race, an ops-runbook concern for the plan
- `s16` — `challenge pass / security+tool-calling lens: GATEWAY_FORCE_STRICT_TOOLS + muse precedent in docs/gemma-4-31b-nvfp4.md`: strict-tools arming for the Lightning lane is parked (v3) pending a measured structural-tag probe; no other security-sensitive surface found — the swap adds no new auth/key material beyond existing peer-key copies
- `s17` — `challenge pass / observability lens: capabilities honesty + at_ms/reason precedent`: clean pass: c10/h7 already require probe-derived adverts and loaded/ready/feasible separation; no new observability gap found beyond what the frame carries

## Decisions

- q1 resolved by operator: #183 hand validation is IN this effort — unblock/attribute #181 on the new nightly (Thor), first-exercise the Spark, adapter serving as far as unsloth-cli#16 allows
- q2 resolved by operator: no consumer relies on worker vision (colleague is under renovation) — the text-only swap is approved; perception belongs to the senses lane
- operator sign-off 2026-08-20: parks v1 (sm_110 SASS in the 8bd082 nightly digest) and v2 (vLLM serving Lightning end-to-end) are downgraded to unknown_nonblocking for the SPEC; the plan must run both as gating spikes on the physical Thor BEFORE any worker flip, per the declared/UNVALIDATED discipline (#108)
- cheap-probe results (2026-08-20, HF API + config.json fetched): nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 is UNGATED (no license wall blocking Thor download); architectures=[NemotronHForCausalLM], model_type=nemotron_h, max_position_embeddings=1048576, quant_method=modelopt, NO vision_config — so the catalog entry says quantization=modelopt and the env.example worker comment ('compressed-tensors, NOT modelopt') flips for Lightning
