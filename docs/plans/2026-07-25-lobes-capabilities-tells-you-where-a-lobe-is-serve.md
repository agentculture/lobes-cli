# Build Plan — lobes capabilities tells you WHERE a lobe is served — a proxied role reads by-proxy, never a bare loaded:no

slug: `lobes-capabilities-tells-you-where-a-lobe-is-serve` · status: `exported` · from frame: `lobes-capabilities-tells-you-where-a-lobe-is-serve`

> lobes capabilities tells you WHERE a lobe is served — a proxied role reads by-proxy, never a bare loaded:no

## Tasks

### t1 — Render a third 'by-proxy' state in the capabilities table renderer

- covers: c1, c5, c13, h2, h5, h9, h1
- acceptance:
  - In lobes/cli/_commands/capabilities.py _render_table, a role with feasible=false AND proxied=true prints 'by-proxy' in the loaded column and names its hosted_by peer; a locally-served role still prints 'yes'; a role that is neither prints 'no'.
  - The state is derived ONLY from the payload's existing proxied/hosted_by/feasible keys — no new key is read and the gateway JSON is not touched.
  - A payload missing the 'proxied' key (offline path, older gateway, hand-built fixture) renders without raising, matching the existing .get('feasible', True) convention.
  - The stale comment at capabilities.py:289-292 ('the row above still shows loaded=yes') is updated to describe the shipped behaviour.
  - Unit tests cover all four cases above against fixture payloads and FAIL against current main.

### t2 — Correct spark's stale MULTIMODAL wiring, shape-override-safe

- covers: c14, h3, h4
- acceptance:
  - MULTIMODAL_BASE_URL is commented out in spark ~/.lobes/.env while MULTIMODAL_SERVED_NAME is retained, so the peer served-name still resolves to coolthor/gemma-4-12B-it-NVFP4A16.
  - Only the gateway container is recreated, via docker compose -f docker-compose.yml -f docker-compose.shape.yml up -d --no-deps gateway.
  - Post-change: vllm-multimodal is NOT running and vllm-primary (cortex) has the same container id / start time as before — no 27B reload.
  - A live POST model=senses through the spark gateway returns 200 with X-Lobes-Proxied-By: orin, proving the peer probe still passes and the proxy stays armed.

### t3 — Confirm a named programmatic consumer actually reads GET /capabilities

- covers: c2, h6
- acceptance:
  - Establish by source inspection whether Colleague, webcam-cli or reachy-mini-cli reads GET /capabilities to decide role usability; record the finding with a file/line citation.
  - If NO consumer reads it programmatically, say so plainly and narrow the spec's audience claim rather than asserting an unverified consumer — the CLI-display-only decision is then trivially sufficient.

### t4 — Update the docs that describe the capabilities loaded column

- depends on: t1
- acceptance:
  - Every doc that shows or describes the capabilities table's loaded column (docs/colleague-stack.md, docs/deployment-shapes.md, docs/gateway-fleet.md as applicable) reflects the three-state rendering.
  - lobes explain (lobes/explain/catalog.py) matches the shipped table if it documents the column.
  - Docs state explicitly that GET /capabilities JSON is UNCHANGED — loaded stays a bool — so no reader infers a wire change.

### t5 — Acceptance probe: one transcript proving before, after and boundary

- depends on: t1, t2
- covers: c3, c4, c9, c10, h7, h8, h10, h11
- acceptance:
  - A single re-runnable script emits into one transcript under docs/evidence/: the capabilities table, plus live model=muse and model=senses POSTs showing 200 + X-Lobes-Proxied-By.
  - Both proxied roles read 'by-proxy' with their peer named, while cortex/embedder/reranker still read 'yes'.
  - The toggle-proof for h7/h9: flipping <PREFIX>_BASE_URL alone no longer moves the displayed state (before the change it did).
  - Boundary proof for h10: git diff --name-only shows no file on a routing or auth path, and the live POST responses are byte-identical to the pre-change capture, headers included.

### t6 — Version bump and CHANGELOG entry

- depends on: t5
- acceptance:
  - python3 .claude/skills/version-bump/scripts/bump.py patch run; pyproject version differs from main so the version-check CI job passes.
  - CHANGELOG entry describes the display-only change and states that the /capabilities wire contract is unchanged.

## Risks

- [follow_up] Issue #155's converged-but-unmerged spec (branch spec/stt-readiness-truth-155) edits the SAME two files this plan touches: lobes/roles.py and the lobes capabilities renderer. It targets 'ready' for audio roles, not 'loaded' for proxied generate roles, so the concerns are disjoint — but whichever merges second will need a rebase, and #155 also changes what the capabilities table reports.
- [unknown_nonblocking] Thor runs gateway 0.46.0 vs repo 0.54.1. The by-proxy display reads the proxied/hosted_by keys from the LOCAL gateway's payload, so Thor's version does not affect spark's rendering — but a by-proxy row rendered on Thor itself would depend on 0.46.0 emitting those keys, which is unverified.
