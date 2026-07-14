# lobes serves the mesh-brain end-state: one lobe per box — the Spark GB10 gives its whole machine to the Qwen cortex, Thor 128GB to Gemma senses, the Orin 64GB hosts the small-model lobes — and the brain stays whole across the mesh

> lobes serves the mesh-brain end-state: one lobe per box — the Spark GB10 gives its whole machine to the Qwen cortex, Thor 128GB to Gemma senses, the Orin 64GB hosts the small-model lobes — and the brain stays whole across the mesh
> instruction: Ship the one-lobe end-state as data-only shapes over the near-term Shape schema, gated on the near-term plan landing; validate on the physical Spark and Thor; Orin stays declared-but-unvalidated until booted

## Audience

- Multi-box fleet operators running the Culture mesh brain across machines — DGX Spark GB10 and Jetson AGX Thor 128GB validated today, Jetson AGX Orin 64GB the named next box — plus the mesh agents consuming the roles across boxes
  - instruction: Validate each audience leg like the near-term spec did: goldens for the default path, physical boots for the reference shapes, a consumer smoke check via the Culture mesh

## Before → After

- Before: After the near-term shapes land (#113 spec, plan confirmed), boxes still co-host: spark-lobe keeps cortex+embedder+reranker+audio on the Spark, thor-lobe keeps senses+the pooling gears+audio on Thor; cortex sits below its 256K native, senses below its full context, and the cross-box story is per-box honesty only — proxying and a brain-level view were explicitly deferred to #112
  - instruction: Cite the near-term spec's Decisions line deferring cross-box design to #112; no code change needed for this claim
- After: The deployment-shape axis reaches full specialization: a box can host as few as one heavy lobe. Reference instances: the Spark GB10 gives its whole machine to the Qwen cortex at 262144 native context, Thor 128GB to Gemma senses at its full 128K instead of the 32K trim, and the Orin 64GB hosts the small-model lobes (minor-class generate + pooling) — while the cheap gears co-reside on every box that wants them, and specialized, multi-role, and mixed boxes (local or cloud) compose into one brain
  - instruction: Implement the three reference shapes over the near-term Shape schema; shapes stay per-box and orthogonal to peers so any specialized/multi-role/mixed fleet composes; Orin stays declared-but-unvalidated until booted

## Why it matters

- Co-residency taxes every lobe even after the near-term reclaim; a box that gives its whole machine to the one lobe it is best at repays the tax completely — and the fleet still has to present a whole brain, which is exactly the cross-box design this issue owns
  - instruction: Quote the co-resident vs full-native context values in the spec; the repayment itself is verified under the c9 measurement requirement

## Requirements

- The cross-box question is answered as a first-class design decision: how a consumer reaches a lobe its local box no longer hosts — gateway proxying to the peer that hosts it vs direct per-box addressing — including whether a brain-level capabilities view exists across boxes
  - instruction: Record the user's cross-box decision as a decision claim; implement and test exactly that behaviour — no half-proxy
  - honesty: The cross-box reachability decision (proxy vs direct addressing, and whether a brain-level capabilities view exists) is recorded in the spec as a confirmed decision and the shipped behaviour matches it, provable by a test or live transcript
- The cheap gears (embedder, reranker, stt, tts) have a designed home in the end-state: they are cheap enough to co-reside almost anywhere, so their placement is an explicit, recorded decision — not an accident of which box they started on
  - instruction: Placement is decided (see Decisions: co-residence); implement the shapes accordingly and record the consumer-facing consequence that embed/rerank/audio endpoints stay localhost on every box
  - honesty: The recorded placement decision is co-residence: every shape may include the pooling and audio gears; the shipped reference shapes keep embedder/reranker/stt/tts on the Spark and Thor (consumers keep localhost endpoints) and the orin shape adds minor + pooling — no gear is forced to move boxes for the end-state to hold
- Per-box honesty (#92) holds in the end-state: a box that does not host a lobe never advertises it — capabilities flag or omit it and the gateway 4xxes requests for it, never silently rerouting to a different model
  - instruction: Reuse the #110/#92 machinery and the near-term t5 tests; extend the per-dropped-role contract test matrix to the reference shapes (spark-lobe drops senses, thor-lobe drops cortex, orin-small drops both heavy lobes — co-residence keeps the cheap gears hosted)
  - honesty: On each one-lobe box, live checks and contract tests show: capabilities flag/omit every unhosted role, GET /v1/models omits them, and generate/embed/rerank/audio requests for them return 4xx — mirroring the near-term t5 contract per dropped role
- Reclamation is real and measured on the physical boxes: Qwen-only Spark serves cortex at a strictly larger context/budget than spark-lobe (target: 256K native), senses-only Thor serves Gemma at full context vs the 32K trim — measured, not asserted
  - instruction: Declare the raised budgets as shape overrides (never runtime mutation), then measure on the GB10 and Thor and record the shipped env values in docs
  - honesty: Measured on the physical boxes: qwen-only Spark serves cortex above the spark-lobe budget (target 262144 context), gemma-only Thor serves senses at its full context (128K — the value senses had before the 32K co-resident trim) — numbers recorded in docs, not estimated
- One-lobe shapes are pure data over the #108 Profile schema plus the near-term Shape schema — no per-shape code fork — proven by per-(shape,card) goldens like the near-term shapes
  - instruction: If any one-lobe shape needs code the near-term Shape schema cannot express, stop and re-spec — that is a schema gap, not a shape
  - honesty: The one-lobe shapes add zero new Python branches to the render path: they are shape TOML/data entries composed over existing #108+near-term machinery, proven by per-(shape,card) goldens byte-diffed in CI

## Honesty conditions

- Each of the three named boxes can be moved to its one-lobe shape with one safe command and back; the heavy lobes serve at measured full budgets; every role of the brain remains reachable per the confirmed cross-box design
- Every named audience leg is exercised: the multi-box operator path is validated on the physical Spark and Thor, the one-box/mixed path stays goldens-proven byte-identical, and a consuming mesh agent reaches every role it used before the shape change
- The before-state is checkable in the shipped tree: the near-term spec+plan are merged (docs/specs/2026-07-14-lobes-serves-the-brain-shape-you-choose-machine-as.md, plan t1–t8 confirmed) and that spec's Decisions section records per-box honesty only with proxying and the brain-level view explicitly deferred to #112
- The tax and its repayment are quoted with shipped numbers, not estimates: cortex 131072 co-resident vs 262144 native, senses 32768 trimmed vs 131072 full — and the post-specialization measured values recorded in docs
- Bare lobes init on a recognised card renders byte-identical before and after this change (machine-as-brain goldens untouched); no new required decisions on the default path
- No doc, table, or capabilities output claims Orin support until a physical Orin boots the shape and its probes pass; the orin shape ships as declared-but-unvalidated data at most
- The success-signal command sequence exists, runs unattended on both physical boxes, and its transcript (probes, capabilities, measured budgets) is attached to the PR
- The three reference one-heavy-lobe shapes exist as data (qwen-only spark, gemma-only thor, small-model orin), render to per-(shape,card) goldens, and the Spark and Thor shapes boot on their physical boxes with the heavy lobe at its measured full budget
- The 262144 / 131072 / 5-of-5 / 404-referral numbers are read from the acceptance-script transcripts run on the physical Spark and Thor (deployed env values, /capabilities context fields, probe output) attached to the PR — not asserted in prose

## Success signals

- On each box one safe command (dry-run diff, --apply) moves it to its one-lobe shape: the Spark serves only the Qwen cortex with measured larger context, Thor serves only Gemma senses at full context, every hosted role passes its correctness probe, per-box capabilities are honest, and a consumer can still reach every role of the brain per the answered cross-box design
  - instruction: Extend the near-term acceptance script to the one-lobe shapes; run it on the physical Spark and Thor and attach transcripts to the PR
- Measured on the physical boxes: the qwen-only Spark serves cortex at max_model_len=262144 (2.0x the co-resident 131072), the gemma-only Thor serves senses at 131072 (4.0x the 32768 trim), 5/5 hosted roles pass their correctness probes on each reference shape, and a request for an unhosted role returns 404 carrying the honest referral per the confirmed cross-box decision

## Scope / boundaries

- machine-as-brain stays the default and one-box users are unaffected: bare lobes init is byte-identical on recognised cards; every one-lobe shape is opt-in per box
  - instruction: Golden-diff the default path in CI as the regression proof
- Orin 64GB ships nothing until someone boots one (the #108 rule): the orin shape may be declared as data but is not claimed supported until validated on a physical Orin
  - instruction: Mark the orin shape unvalidated in the support table exactly like the #108 'base' fallback discipline; file the physical validation as its own follow-up

## Assumptions

- The near-term brain-shapes plan (t1–t8: Shape schema, budget re-derivation, --shape flag, spark-lobe/thor-lobe, dropped-lobe honesty, GB10+Thor validation) lands before this work; this spec composes on its Shape schema rather than duplicating or re-doing it

## Decisions

- Backward compatible, just a design option: either you have many machines (some cloud if you want) and each is specialized, or some machines take multiple roles, or any mix — machine-as-brain stays the default and the one-lobe shapes are the far end of the same shape axis, not a mandate
- Cross-box reachability = direct + honest referral: no data-plane proxying — consumers dial each box directly as the Culture mesh does today; with opt-in peer config, a box's capabilities and role_infeasible 404s name the peer that hosts an absent role; the #92 invariant holds — a box never serves what it does not host
- Cheap gears co-reside everywhere: 'one lobe per box' specializes the heavy generate lobes only; embedder/reranker/stt/tts stay on every box that wants them (~0.06 util each), consumers keep localhost embed/rerank, and the Orin takes minor + optionally pooling when validated
- The Qwen-only Spark stays a fleet shape: gateway + one heavy vLLM behind it, capabilities/honesty surfaces intact; 'return to 256K-native solo' means the full GPU budget and 262144 context as shape overrides, not the legacy single-model scaffold

## Hard questions

- RESOLVED (user decision, see Decisions): does Qwen-only Spark stay a fleet shape or revert to the legacy single-model scaffold that already serves 256K solo? → Fleet shape: gateway + one heavy vLLM; 'solo' means the full GPU budget and 262144 context as shape overrides.
- RESOLVED (user decision, see Decisions): when a consumer asks a one-lobe box for a role it no longer hosts — proxy to the peer, or direct addressing? Brain-level view? → Direct + honest referral: no data-plane proxying; with opt-in peer config, capabilities and role_infeasible 404s name the hosting peer.
- RESOLVED (user decision, see Decisions): where does each cheap gear land, given the #112 table puts the pooling gears on the unvalidated Orin? → Co-residence: embedder/reranker/stt/tts stay on every box that wants them; Orin takes minor + optionally pooling when validated.

## Open / follow-up

- Physical Jetson AGX Orin 64GB boot + probe validation — the orin small-model shape ships as declared-but-unvalidated data until someone boots one (the #108 rule); validation is its own follow-up with its own evidence
- Proxy-lobes — a box serves a *sleeping lobe* by following its own referral (gateway forwards to the hosting peer, advertised as proxied, never impersonating local serving); opt-in per box per role, direct+referral stays the default — tracked as agentculture/lobes-cli#115, sequenced after #112
