# Build Plan — thor cortex speculation

slug: `thor-cortex-speculation` · status: `exported` · from frame: `thor-cortex-speculation`

> the Thor cortex gets speculative decoding back — DSpark or MTP, behind the Triton GDN decode path

## Tasks

### t1 — Build the rollback safety net and prove it works BEFORE touching anything

- instruction: Do this FIRST and do not skip the exercise step — an unexercised snapshot is a hope, not a rollback. Follow d1's naming (~/.lobes.pre-speculation-<timestamp>). Remember this box's memory quirk: MemAvailable LIES here, so read tegrastats for the real free-block picture, and run `drop_caches` BEFORE the recreate rather than after the boot fails and the restart policy loops it.
- covers: c26, h18
- acceptance:
  - ~/.lobes is copied to ~/.lobes.pre-speculation-<timestamp> following d1's naming, with .env captured separately for the key-by-key diff in t2
  - the rollback is EXERCISED once before any arm runs: restore the snapshot, recreate the cortex container, and re-prove BOTH the `known_answer` probe and ~12.1 tok/s on the cuda path
  - if the restored lane does not reproduce the 2026-08-20 numbers, the plan STOPS — the box has drifted and the snapshot is not a rollback
  - `drop_caches` (sync; echo 3 > /proc/sys/vm/`drop_caches`) runs BEFORE the recreate, not after it fails; tegrastats confirms the free-block recovery

### t2 — Re-render the deployment scaffold, preserving every operator-typed key

- instruction: The riskiest step in the plan, because it regenerates a hand-edited deployment. Copy .env into the snapshot before rendering. Prefer a plain --apply: merge-only since 0.59.0 protects existing lines, whereas d1's --apply --force force-writes keys and nothing regenerates an operator-typed peer credential (#92). If --force proves necessary for a specific key, reconcile the rest by hand and re-verify both proxy paths.
- depends on: t1
- covers: c8, h6, c9, h7, c27, h19
- acceptance:
  - a DRY-RUN 'lobes init --fleet --shape spark-lobe --profile thor' is diffed against the deployed compose and the live 'docker inspect' argv BEFORE any --apply, confirming the Thor does render spark-lobe (falsifies or confirms c8)
  - the render uses a plain --apply, NOT --apply --force; --force is used only if a specific key demands it and then reconciled by hand
  - post-render .env is diffed key by key against the pre-render copy; `MULTIMODAL_PEER_ORIGIN`/PROXY, `HAND_PEER_ORIGIN`/PROXY/`API_KEY`, `WORKER_FEASIBLE` and every \*`_FEASIBLE` flag are present and unchanged
  - the rendered docker-compose.yml contains the ${`PRIMARY_SPECULATIVE_CONFIG`-...} slot (absent on the deployed 0.57.x-era file), verified by grep
  - no local vllm-hand container exists; model=hand still answers 200 with X-Lobes-Proxied-By pointing at the Spark, and model=senses likewise at the Orin

### t3 — ARM A — the control: Triton GDN decode path, NO speculation, 262144

- instruction: Decide `max_num_seqs` cold, before you start: keeping 2 matches both the deployed box and the Spark spike's 262144 tables, which is what makes the numbers comparable. This arm's number is the denominator for everything that follows, so the warmup discipline matters more here than anywhere else — a JIT-inflated control makes both speculation arms look better than they are.
- depends on: t2
- covers: c14, h9, c22, h14, c24, h16
- acceptance:
  - `VLLM_GDN_DECODE_KERNEL`=triton is set on the vllm-primary service and PROVEN to have landed via 'docker inspect --format {{json .Config.Env}}', with the container log line 'GDN decode kernel: triton' captured as proof the value was read
  - `PRIMARY_SPECULATIVE_CONFIG`= (set but EMPTY, so the flag is omitted from argv, not blanked) and `max_model_len`=262144; argv proven from docker inspect, never from .env
  - `PRIMARY_MAX_NUM_SEQS` is decided BEFORE this arm, stated in the transcript, and identical across all three arms — keeping 2 matches both the deployed box and the Spark spike's 262144 tables
  - one warmup generation is discarded before the measured set, and a second measured run reproduces the first within the ~10-15% variance floor — proving Triton JIT compile cost is not leaking into the baseline
  - three content shapes (code / reasoning / prose) measured, each labelled with the shape that produced it; KV pool size and concurrency multiple recorded as vLLM reports them
  - the d1 probe suite (`known_answer`, `tool_calls`, gateway smoke) runs on this arm and its outputs are retained for the cross-arm comparison in t6
  - wall-clock timestamps recorded so a mid-run environment change is detectable afterwards

### t4 — ARM B — MTP-n2: the unlock test

- instruction: This is the arm that decides whether the whole idea works. Watch the FIRST decode, not /health — d1's failure surfaced there. If the missing-kernel RuntimeError returns, stop cleanly and hand t7 a negative result; that outcome satisfies the success signal and is not a failed plan.
- depends on: t3
- covers: c1, h1, c4, h3
- acceptance:
  - the lane boots with the compose template's default MTP-n2 --speculative-config AND `VLLM_GDN_DECODE_KERNEL`=triton, and COMPLETES A MULTI-TOKEN DECODE — a healthy /health is explicitly NOT evidence, since the d1 failure surfaced on the first decode
  - no 'no kernel image is available for execution on the device' RuntimeError appears; if it does, h1 is falsified, the frame collapses to a negative result, and t7 reports it as one rather than the plan continuing
  - no second knob turns out to be required — if vLLM raises a config error revealing one, h3 is falsified and the finding is recorded
  - same three content shapes, same discarded warmup, same pinned `max_num_seqs` as ARM A; MTP draft acceptance read from vLLM's `docker_logs` SpecDecoding line and recorded per shape
  - the d1 probe suite runs on this arm and its outputs are retained for t6

### t10 — Pre-fetch and verify the DSpark drafter BEFORE the maintenance window

- instruction: Run this alongside t1, before the maintenance window opens. A 2.53 GiB cold pull discovered mid-window is how a three-arm session becomes a two-arm session. NOTE: this task was briefly self-confirmed by the agent in error and flipped back to proposed — it needs the user's own confirm.
- acceptance:
  - RadixArk/Qwen3.8-27B-DSpark at revision 85ef153be924f17ce4bf62726954eeaa4a73e854 is present in this box's HF cache, with the revision confirmed on disk rather than assumed from a successful download
  - free disk headroom is checked against the ~2.53 GiB of bf16 weights plus HF's staging overhead, before the pull rather than during it
  - a partial or interrupted download is detected and cleared here, not discovered later as a confusing model-load error inside ARM C
  - this task touches NO container and NO deployment file, so it is safe to run alongside t1 without affecting the cortex lane

### t5 — ARM C — DSpark block-7 against the pinned revision

- instruction: Copy the `speculative_config` value verbatim from spark-lobe.toml — both quoting layers, no retyping. The bare-single-quote and unquoted spellings degrade the JSON silently and the boot failure points nowhere near the cause. If DSpark will not load on the pinned digest, STOP: bumping the nightly to make it load would invalidate arms A and B.
- depends on: t4, t10
- covers: c6, h4, c7, h5, c25, h17
- acceptance:
  - the `speculative_config` value is COPIED VERBATIM from lobes/profiles/`builtin_shapes`/spark-lobe.toml rather than retyped, preserving both quoting layers — the bare-single-quote and unquoted spellings degrade the JSON silently
  - revision 85ef153be924f17ce4bf62726954eeaa4a73e854 is proven present in the container's rendered argv via docker inspect; a floating or absent revision voids the arm and it is re-run
  - DSparkDraftModel resolves on the ALREADY-PINNED image digest 8bd082c274fa with no image bump; if it will not load, the plan STOPS and reports rather than bumping the nightly, which would confound every arm
  - KV pool and concurrency multiple recorded at 262144; if headroom looks generous, one probe boot above 262144 records where vLLM actually refuses, so the trade-down is measured on THIS box rather than inherited from the Spark
  - same three content shapes, same discarded warmup, same pinned `max_num_seqs`; DSpark draft acceptance recorded per shape
  - the d1 probe suite runs on this arm and its outputs are retained for t6

### t6 — Cross-arm output comparison and the divergence protocol

- instruction: Resist the pull to explain a divergence before recording it. Write the raw observation down first, then work the cheap causes — argv, sampling params, parser — before reaching for anything algorithmic. Most 'the model changed' findings are config findings.
- depends on: t3, t4, t5
- covers: c21, h13
- acceptance:
  - the retained probe outputs from all three arms are compared; `known_answer` and structured `tool_calls` behave identically across arms
  - IF any output difference is observed, the raw divergence is written into the transcript IMMEDIATELY, before diagnosis — sitting on the observation until the investigation concludes falsifies h13 just as much as defending losslessness does
  - diagnosis checks the CHEAP implementation causes first — argv via docker inspect, sampling parameters, the `qwen3_coder_thinking` parser — before reaching for an algorithmic explanation
  - the investigation is BOUNDED: it resolves to a named cause (implementation defect or algorithmic divergence) or the losslessness claim is retracted, never left suspended indefinitely
  - no speed or acceptance number in this plan is presented as a quality claim

### t7 — Write the evidence transcript — including if the answer is no

- instruction: Write this even if the answer is no, and write it with the same care. The negative transcript is the more valuable artifact, because it closes a question that would otherwise be reopened every few months. Keep 'measured here' and 'published elsewhere' in separate columns; never let a citation drift into a claim.
- depends on: t6
- covers: c16, h2, c19, h12
- acceptance:
  - docs/evidence/2026-<date>-spike-thor-cortex-speculation.txt exists (spike-, not accept-, unless an arm is actually adopted, in which case a separate accept- transcript is added)
  - it carries three arms x three content shapes with per-shape tok/s and acceptance, the same-day triton no-spec control, argv proven from docker inspect, and an explicit statement of which arm was adopted and which rejected
  - every number carries its conditions per docs/measuring-lane-performance.md rule 3; deltas at or under ~15% are stated as noise, not as effects
  - a NEGATIVE result — Triton does not unlock speculation, or unlocks it at a net loss — is written up with the SAME care as a positive one; a verbal 'it did not work' with no transcript falsifies h2
  - the transcript does not compare against the published 34-38 tok/s SGLang figure, which is a different publisher, engine, target quantization and silicon

### t8 — Adopt the outcome in-tree and prove the served contract

- instruction: Update thor.toml in the same PR as the transcript — a follow-up that never lands is how the flat 'MTP MUST BE OFF' comment became stale in the first place. Announce the window change before it lands: per q3, after this no box in the mesh serves cortex above 262144.
- depends on: t7
- covers: c17, h10, c18, h11
- acceptance:
  - lobes/profiles/builtin/thor.toml's cortex comment (lines ~35-39, currently a flat 'MTP MUST BE OFF') is replaced with the MEASURED condition naming `VLLM_GDN_DECODE_KERNEL` — in the SAME PR as the transcript, never a follow-up
  - 'lobes capabilities' and GET /capabilities on the Thor report cortex context 262144, not 1048576; if they still advertise 1M that is a capabilities bug and gets its own issue rather than being papered over
  - the mesh is told BEFORE the window change lands — a caller relying on >262144 tokens breaks silently otherwise, and per q3 no box in the mesh will serve above 262144 afterwards
  - the Orin's cortex referral still resolves after the change (it proxies to the Spark, not here, per the dual-cortex topology — verify rather than assume)
  - docs/dspark-speculation.md and docs/qwen3.8-27b-nvfp4.md cite the new transcript

### t9 — Ship it: branch, version bump, PR

- instruction: Standard repo flow: branch, bump, PR, and reply to every reviewer thread before merging. The version bump is not optional even for a docs-and-config PR.
- depends on: t8
- acceptance:
  - python3 .claude/skills/version-bump/scripts/bump.py minor runs — CI's version-check job fails the PR if the version equals main's, with no exception for docs-only changes
  - uv run pytest -n auto passes, plus black/isort/flake8/bandit and 'uv run afi cli doctor . --strict'
  - every reviewer thread (Qodo, Copilot, human) is replied to and resolved before merge

## Risks

- [unknown_nonblocking] the DSpark drafter is a COLD PULL on this box — ~2.53 GiB of bf16 weights from RadixArk on Hugging Face, never fetched here. Neither the frame nor the challenge pass examined the Thor's HF cache state, disk headroom, or network path. A slow or failed pull stalls ARM C after the other two arms have already consumed the maintenance window, and a partial download can surface as a confusing load error rather than a clear network failure. Pre-fetch and verify the revision BEFORE the arm window opens (task t5)
- [out_of_scope] this plan has NO parallelism available — every arm mutates the same single physical box and the same cortex container, so 'devague plan waves' will emit narrow, effectively serial waves. That is honest, not a plan defect: do not fan this out to a workforce, and do not let wave width be read as an opportunity to run arms concurrently. Concurrent arms would also violate c14's same-day/same-harness/same-box-state condition
- [unknown_nonblocking] the Thor's cortex is DOWN for each container recreate across t2 through t5. Per the dual-cortex topology the mesh routes cortex to the Spark and the Orin proxies there, so blast radius is limited — but any local consumer on this box (the deployed lobes agent, culture-node containers) loses its local cortex for the duration. Announce the window; do not discover a dependent mid-arm
- [follow_up] issue #204 — a shape override beats a card profile, so builtin/thor.toml cannot express a draft opinion that survives rendering spark-lobe. If t8 adopts a Thor-specific speculative config, the only supported homes are a new thor-cortex shape (orin-cortex.toml is the precedent) or forking spark-lobe. Neither is ergonomic; this plan does NOT close that gap (task t8)
- [follow_up] no lobes surface exposes draft acceptance — lobes/`_metrics.py` maps no spec-decode, draft or acceptance families for either engine, so every acceptance figure here comes from vLLM's `docker_logs` read by hand. Fine for a dated spike; it means an ADOPTED speculative lane degrades invisibly, since a drafter whose acceptance collapses on a workload shift looks identical to a slow box from every lobes surface
- [out_of_scope] retiring the YaRN `hf_overrides` block at 262144 is deliberately OUT of this plan. The operator's quality rationale for the native ceiling argues for stock rope too, but every DSpark arm ever measured ran with that YaRN block in force, and folding it in would add a second uncontrolled variable to a three-arm comparison. It needs its own transcript at 262144 + stock rope
