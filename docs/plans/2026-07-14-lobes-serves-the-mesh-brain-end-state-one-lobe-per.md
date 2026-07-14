# Build Plan — lobes serves the mesh-brain end-state: one lobe per box — the Spark GB10 gives its whole machine to the Qwen cortex, Thor 128GB to Gemma senses, the Orin 64GB hosts the small-model lobes — and the brain stays whole across the mesh

slug: `lobes-serves-the-mesh-brain-end-state-one-lobe-per` · status: `exported` · from frame: `lobes-serves-the-mesh-brain-end-state-one-lobe-per`

> lobes serves the mesh-brain end-state: one lobe per box — the Spark GB10 gives its whole machine to the Qwen cortex, Thor 128GB to Gemma senses, the Orin 64GB hosts the small-model lobes — and the brain stays whole across the mesh

## Tasks

### t1 — Full-native heavy-lobe budgets as shape data: spark-lobe cortex rises to 262144 context + reclaimed util (strictly above the co-resident 0.30), thor-lobe senses rises to 131072 + reclaimed util (strictly above 0.14) — declared shape overrides over the near-term Shape schema, cheap gears kept co-resident, zero new render-path code branches

- covers: c9, h5, c19, h11, c7, h12, c10, c5
- acceptance:
  - Rendered goldens show PRIMARY_MAX_MODEL_LEN=262144 for (spark-lobe, spark card) and MULTIMODAL_MAX_MODEL_LEN=131072 for (thor-lobe, thor card), each with util strictly above the co-resident value
  - Both shapes still include embedder+reranker+audio (co-residence) and drop exactly one heavy lobe; the diff is shape data + goldens only — no new Python branches in the render path

### t2 — orin-small reference shape: minor-class generate + the 0.6B pooling gears (audio opt-in like everywhere), declared-but-unvalidated per the #108 rule — renders against the base/unknown-card discipline until a physical Orin boots

- covers: c12, h8, c19, h11, c7, h12
- acceptance:
  - The orin-small shape renders to per-(shape,card) goldens; no doc, support table, or capabilities output claims Orin validated — marked exactly like the #108 base-fallback discipline
  - The shape hosts no heavy lobe (cortex and senses both absent) and includes minor + embedder + reranker

### t3 — Honest referral: opt-in peer config on the gateway — with peers configured, lobes capabilities / GET /capabilities annotate an unhosted role with the peer that hosts it and role_infeasible 404 bodies carry the referral; with no peer config, responses are byte-identical to the near-term contract; NO data-plane proxying exists

- covers: c6, h2
- acceptance:
  - With peer config set, capabilities and the 404 role_infeasible body name the hosting peer's origin for each unhosted role; with it unset, output is byte-identical to the pre-change contract (regression test)
  - No code path forwards a generate/embed/rerank/audio request to a peer — proven by a test asserting the gateway never opens an outbound connection on a role request for an unhosted role

### t4 — Contract-test matrix for the reference shapes: per-dropped-role honesty tests (spark-lobe drops senses; thor-lobe drops cortex; orin-small drops both heavies), referral-aware 404 assertions, and the machine-as-brain default-path regression goldens byte-identical before/after

- depends on: t1, t2, t3
- covers: c8, h4, c11, h7, h6, c10
- acceptance:
  - One contract test per (reference shape, dropped role) asserts: capabilities flag/omit it, /v1/models omits it, requests for it 4xx with the referral when peers are configured
  - Bare-init machine-as-brain goldens are byte-identical before and after the change; all goldens byte-diffed in CI

### t5 — Acceptance script + physical validation on the GB10 and Thor: extend the near-term acceptance script to the full-native shapes; run unattended on both boxes — dry-run diff, --apply, 5/5 hosted-role probes, deployed env + /capabilities show 262144/131072, live referral 404, re-init with the previous shape restores byte-for-byte, and a Culture mesh consumer reaches every role; transcripts attached to the PR

- depends on: t4
- covers: c1, h10, c2, h13, c14, h9, c20, h16, c9, h5, h11
- acceptance:
  - The script runs unattended on both physical boxes and its transcript records: 5/5 probes per shape, PRIMARY_MAX_MODEL_LEN=262144 (Spark) and MULTIMODAL_MAX_MODEL_LEN=131072 (Thor) in the deployed env and /capabilities, and a 404-with-referral for an unhosted role
  - Moving back: re-running init with the previous shape restores the prior rendering byte-for-byte, shown in the transcript
  - A mesh consumer (Culture agent) reaches every role of the brain across the two boxes per the direct+referral decision

### t6 — Docs + evidence: end-state section in the deployment-shapes doc — the shape axis (specialized / multi-role / any mix, local or cloud; backward compatible), the four recorded decisions (direct+referral, co-residence, fleet shape, mixable axis), the tax table quoting 131072→262144 and 32768→131072, the before-state citation of the near-term spec's Decisions deferral to #112, Orin marked unvalidated; lobes explain shapes + CLAUDE.md updated

- depends on: t5
- covers: c3, h14, c5, h15, c12
- acceptance:
  - The doc quotes the shipped co-resident vs full-native context values and the measured post-specialization numbers from the t5 transcripts — no estimated numbers
  - The doc cites the near-term spec's Decisions line deferring cross-box design to #112 and records all four #112 decisions; Orin appears as declared-but-unvalidated

## Risks

- [unknown_nonblocking] The near-term brain-shapes plan (#113, t1–t8) is spec+plan merged but NOT yet implemented — this entire plan sequences after it lands; if its t2/t6 measurement already lands full-native budgets, t1 here reduces to asserting/goldening those values
- [unknown_nonblocking] Exact reclaimed util values per box are measured during t5 physical validation, not declared up front (frame parked item v1)
- [follow_up] Physical Jetson AGX Orin 64GB boot + probe validation is its own follow-up with its own evidence (frame parked item v2); orin-small stays declared-but-unvalidated in this plan
- [unknown_nonblocking] Peer-config surface (env var vs file, naming, origin format) is decided at t3 implementation within the #92 lesson: never fabricate an absolute URL; referral origins must be operator-declared per peer, not inferred (task t3)
