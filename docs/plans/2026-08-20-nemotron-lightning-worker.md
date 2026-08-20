# Build Plan — nemotron-lightning-worker

slug: `nemotron-lightning-worker` · status: `exported` · from frame: `nemotron-lightning-worker`

> Thor's worker seat moves to nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 — a fast, text-only, non-coding doer (action selection, tool loops, RAG, digestion, repo inspection — never code authoring or final judgment). The Thor machine image rides the same official vLLM nightly the Spark validated for Qwen3.8. Thor and Orin repoint their cortex proxy at the Spark's new served id unsloth/Qwen3.8-27B-NVFP4 (#186), and the worker validation matrix lands per #187 (with #183's hand budgets tracked as their own effort).

## Tasks

### t1 — Thor groundwork spike: pull the deployment to merged main, render, pull the 8bd082 nightly digest, and PROVE sm_110 compatibility (cuobjdump SASS/PTX listing or a minimal engine boot) before anything else consumes the image

- covers: c15, h14
- acceptance:
  - Thor's repo checkout and ~/.lobes render are at >=0.57.1; the digest is pulled and a recorded probe shows sm_110 SASS (or working PTX JIT) — a negative result STOPS the flip and is reported, not worked around
  - the before-state facts (stale checkout, old mirrors) are re-verified against pulled main and corrections recorded

### t2 — Spike: Lightning serves on the nightly on Thor — standalone vllm serve of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 (nemotron_h) with --reasoning-parser nemotron_v3 + candidate tool parser; probe structured tool_calls, measure plain decode, then evaluate MTP/DSpark separately; grow max_model_len progressively from a modest window

- depends on: t1
- covers: c9, h6
- acceptance:
  - the lane answers a known-answer completion AND returns a structured tool_calls array (a 200 with the call leaked into content is FAILURE); reasoning/tool parser pair recorded
  - plain-decode tok/s measured BEFORE any speculative mode; MTP/DSpark results (or their absence) recorded separately; no config allocates the 1M ceiling by default

### t3 — catalog.py: add the Lightning entry (role_hint=worker, quantization=modelopt, NemotronHForCausalLM facts from the fetched config) and demote unsloth/Qwen3.6-35B-A3B-NVFP4 to a kept candidate (cite-don't-delete); tests updated

- covers: c2, h1
- acceptance:
  - uv run pytest tests/test_catalog.py passes with the new entry; every config fact in the entry cites config.json/hf_quant_config.json fetched from HF (c24 probe), none from model-card prose

### t4 — roles.py: redefine ROLE_RESPONSIBILITIES['worker'] per #187 — drop image_understanding/video_understanding, keep tool_use/execution/ground_work, add explicit tokens splitting repo inspection/navigation/run-authorized-commands from code authoring (extend the vocabulary, never prose or model-name checks); adjust cortex's only-role-that-sees-and-decides comment; tests updated

- covers: c3, h2
- acceptance:
  - a consumer reading only responsibilities/forbidden_responsibilities gets the full non-coder policy; no vision token remains on worker; existing roles tests pass and a new test asserts the worker token set

### t5 — Worker served-id rollout audit + rollout notes: grep sibling checkouts (colleague, embodiment, eidetic, reachy-mini-cli) for unsloth/Qwen3.6-35B-A3B-NVFP4, write the worker rollout-notes doc naming every pinner, and list every box needing a WORKER_SERVED_NAME mirror

- covers: c22, h17, c13, h13
- acceptance:
  - the rollout note names every raw-id hit found by the audit grep (or records a clean grep per repo) and states the new id + mirror instructions BEFORE the flip lands

### t6 — Incumbent baseline (playbook §1): benchmark the CURRENT Qwen 35B worker on the NEW 8bd082 engine on Thor before it is gone — this number is unrecoverable after the flip

- depends on: t1
- covers: c17
- acceptance:
  - a matched-condition baseline transcript (decode tok/s, tool-call probe, worker-shaped task sample) for the incumbent on the new engine lands under docs/evidence/ before any flip

### t7 — Rollback readiness: verify Thor disk headroom for ~20 GiB of Lightning weights alongside the kept Qwen checkpoint + image; exercise one dry-run restore to the Qwen shape (or record an explicit operator waiver)

- depends on: t1
- covers: c23, h18
- acceptance:
  - old checkpoint and image remain on disk until the validation matrix passes; the restore dry-run output (or the recorded waiver) is in the evidence transcript

### t8 — Thor live boot + measurement: flip the worker lane to Lightning (drop_caches BEFORE recreate; watch for the orphaned-dependent compose state), measure gpu_mem_util/max_model_len co-resident with hand+embedder+reranker+audio, run the structured tool-call probe against the served lane, and verify the advert (capabilities) tells the deployed truth — text-only, probe-derived tools, loaded/ready/feasible separate

- depends on: t2, t6, t7
- covers: c4, h3, c10, h7
- acceptance:
  - a live-boot transcript under docs/evidence/ records the measured budget with all co-residents up, the tool_calls probe result, and a capabilities dump showing no vision tokens and probe-backed tools for worker

### t9 — Commit the measured shape: thor-worker.toml swaps overrides.worker to Lightning with the t8-measured budgets (transcript-cited), env.example worker block updated (quantization=modelopt, parser flags, new served id), compose worker lane flags aligned; shape goldens regenerated

- depends on: t8
- covers: c4, h3
- acceptance:
  - every number in the committed shape cites the t8 transcript; shape golden tests pass; the Qwen values survive only as cited history in comments

### t10 — #186 Thor: mirror PRIMARY_SERVED_NAME=unsloth/Qwen3.8-27B-NVFP4 beside the peer knobs in the Thor .env, recreate the gateway with the full -f overlay set, and prove cortex by proxy

- depends on: t1
- covers: c6, h4
- acceptance:
  - lobes capabilities on Thor shows cortex proxied=true/ready=true/hosted_by=spark; a model=cortex request answers 200 with X-Lobes-Proxied-By; transcript lands under docs/evidence/

### t11 — #186 Orin: same mirror + gateway recreate + proof on the Orin box

- covers: c6, h4
- acceptance:
  - same observable acceptance as the Thor repoint, captured in the Orin's own transcript under docs/evidence/

### t12 — hand on Thor (#183/#181): re-run the Orin-template probe on the new nightly — budget at the card window, known-answer completion, structured tool_calls via the lfm2 parser — and re-attribute #181's LoRA embedding-slot failure (fixed, changed, or re-blocked)

- depends on: t1
- covers: c20, h9
- acceptance:
  - either a passing budget transcript lands, or the failure is recorded blocked-with-attribution under the new engine — never skipped or smoothed over

### t13 — hand on Spark (#183): first exercise of the declared 0.06 hypothesis with the Orin probe template; adapter serving recorded as blocked on unsloth-cli#16

- covers: c20, h9
- acceptance:
  - a Spark transcript establishes (or refutes) the 0.06 budget with known-answer + tool_calls probes; the adapter-serving surface is explicitly stated blocked, not validated

### t14 — Validation matrix (#187): run the matched worker-shaped tasks + explicit negative/escalation tasks against Lightning on Thor and compare with the t6 incumbent baseline; record the go/no-go

- depends on: t6, t8
- covers: c17, h16
- acceptance:
  - the matrix transcript covers #187's worker-shaped list (tool choice, tool loop, RAG, summarize, extract, repo navigation, run tests, long loop, failure recovery) and at least two negative tasks proving escalation to cortex; a no-go triggers the t7 rollback path

### t15 — Docs follow the shipped state: new docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md (fetched-config provenance + measured numbers), demotion notes in docs/qwen3.6-35b-a3b-nvfp4.md, role tables in docs/colleague-stack.md + CLAUDE.md, cognitive-split rationale quoted from #187; repo-wide grep proves no surface still claims worker vision

- depends on: t3, t4, t9, t13
- covers: c12, h8, c16, h15
- acceptance:
  - every 'validated' claim cites a transcript path; the vision-claims grep is clean; markdownlint passes

### t16 — Final acceptance audit: all four success-signal transcripts landed; after-state clauses each mapped to one; zero diffs under lobes/gateway/ and roles.py channel tables (or the boundary reopened explicitly); remaining gaps stated declared/UNVALIDATED

- depends on: t8, t9, t10, t11, t12, t13, t14, t15
- covers: c1, h12, c14, h11, c7, h5, c17, h16
- acceptance:
  - an audit note maps c14's clauses to transcript paths, confirms the c7 zero-gateway-diff boundary via git diff, and lists anything still declared/UNVALIDATED honestly

## Risks

- [unknown_nonblocking] sm_110 SASS/PTX coverage of the 8bd082 nightly digest is unproven until t1 — a negative result stops the flip and reopens the image choice (older Thor-validated 0.23.1 digest as fallback) (task t1)
- [unknown_nonblocking] vLLM end-to-end support for nemotron_h on this nightly (reasoning/tool parser pair, FP8 KV, Mamba backend, MTP/DSpark) is unproven until t2; parser pairs fail silently, hence the structured-probe acceptance (task t2)
- [follow_up] GATEWAY_FORCE_STRICT_TOOLS arming for the Lightning lane stays undecided until a live structural-tag probe (muse precedent: never advertise a grammar-constrained lane that isn't one) (task t8)
- [unknown_nonblocking] #181 (LoRA embedding-slot failure) may persist under the new nightly — t12 reports blocked-with-attribution rather than forcing a pass (task t12)
- [unknown_nonblocking] Thor unified-memory first-boot race + compose orphaned-dependent state can fail any heavy-lobe recreate — runbook: drop_caches BEFORE recreate, verify no service stuck in created (task t8)
- [follow_up] adapter serving end-to-end stays untestable until unsloth-cli#16 ships a real adapter — reported blocked, out of this plan's power (task t13)
- [unknown_nonblocking] the incumbent baseline is unrecoverable after the flip — t6 MUST complete before t8; ordering is load-bearing (playbook §1) (task t6)
