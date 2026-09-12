# Delivery Summary — mesh-brain-join

plan: `mesh-brain-join` · run: `complete` · date: `2026-09-11`
baseline: `devague summary skeleton`

## Intent

Deliver the mesh-brain join for lobes: every member is the brain, dynamic
membership by one shared join key (plus a persisted approval ledger) replaces
the operator-typed `*_PEER_*` env family, heartbeat-driven rosters with
trust-but-verify, same-role pools and auto-wired proxying derived from
verified members, `{role}-{machine-name}` naming on fingerprint disagreement,
a gateway-only shape, CLI and doctor support — then cut the live
Spark/Thor/Orin fleet over and measure the five success signals. The run
executed plan `mesh-brain-join` (14 tasks after deviation d6 added t13/t14),
spec `docs/specs/2026-09-11-mesh-brain-join.md`, plan
`docs/plans/2026-09-11-mesh-brain-join.md`, split artifact
`docs/plans/2026-09-11-mesh-brain-join-split.md`, on PR #252 (branch
`spec/mesh-brain-join`, version 0.76.0).

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Mesh config: parse `LOBES_MESH_KEY` / `LOBES_MESH_NAME` / `LOBES_MESH_SEEDS` / `LOBES_MESH_HEARTBEAT_S` (60) / `LOBES_MESH_MISSED_MAX` (3) / `LOBES_MESH_LEDGER_PATH` into a frozen MeshConfig in a new module
- `t2` — Announcement wire: schema-versioned Announcement dataclass + encode/decode in a new module, carrying name, origin, `schema_version`, and per-role RoleInfo fields plus fingerprint, capacity and a private flag
- `t3` — Roster + ledger: in-memory Roster (members by name, `last_seen`, missed count, verified/unverified, flapping hold-out, name-conflict refusal, capacity clamp) and a persisted approval Ledger (name -> `approved_by`, expiry) with gossip merge, in a new module
- `t4` — Templates and scan plumbing: gateway compose passthrough for every `LOBES_MESH_`\* key, the gateway's first bind mount for the ledger file, env.example docs for the keys, `LOBES_MESH_KEY` in `scan_deployment_secrets.py`, .gitignore for the runtime ledger
- `t5` — Gateway-only shape: a built-in 'gateway-only' shape (hosts=`[]`) with goldens, and relax `_routing.py`'s 'a built table always has a primary backend' assumption so a table with no local backend builds and routes only by mesh
- `t6` — Mesh endpoints + heartbeat thread: GET /mesh/detect (keyless, minimal), POST /mesh/join (keyless, bounded, collapsed log), POST /mesh/announce and GET /mesh/roster and POST /mesh/approve|revoke (join-key bearer); inbound gate accepts the join key for members; heartbeat daemon thread in serve() announcing to seeds + roster members with an immediate re-announce hook
- `t7` — Membership-driven routing (core): copy-on-write routing snapshot swapped atomically from the roster; pools and auto-proxy derived from verified members (probe /capabilities must match the announcement before any forward); hand proxied like any role; realtime stays local-only; single-hop 508 preserved
- `t8` — Naming and exposure: '{role}-{machine-name}' suffixed lanes on fingerprint disagreement, listed on /v1/models and /capabilities of every member; raw served-id on divergent lanes only 404s listing the suffixed names; private roles never exposed; immediate re-announce on lobes switch/up/unhealthy; X-Lobes-Mesh-Member on every mesh answer
- `t9` — Retire the \*`_PEER_`\* family in code + doctor: delete `PEER_ORIGIN`/ORIGINS/PROXY/`API_KEY`(S) parsing and ReplicaConfigError paths from `_config.py`/`_routing.py`; doctor gains three findings: leftover \*`_PEER_`\* keys, shell `LOBES_MESH_KEY` differing from .env, and mesh passthrough coverage
- `t10` — CLI: `lobes mesh status|request|approve <name> [--for <duration>]|revoke <name>` (write verbs dry-run, --apply commits) plus lobes capabilities rendering members, suffixed and private lanes
- `t11` — Docs and catalog follow the replacement: gateway-fleet.md, deployment-shapes.md, colleague-stack.md, openai-api.md, secret-rotation.md (join-key rotation = fleet-wide restart), env.example, lobes explain, CLAUDE.md; old peer sections moved under 'Retired'; deployments/jetson-agx-`thor__thor`-worker re-captured; version bump
- `t12` — Live cutover and validation on Spark + Thor + Orin: baseline capture, per-box backups, migrate all three onto the join key with `LOBES_MESH_NAME`/SEEDS, boot the gateway-only shape on the Orin as a fourth-member test, measure the five success signals, file the evidence transcript
- `t13` — Swap the peer SOURCE to the mesh: pools and forward targets are built from the mesh RoutingSnapshot with no env-declared peer origins; the pool selection no longer exits early when table.`replica_origins` is empty; end-to-end tests with NO `<PREFIX>_PEER_*` keys in any fixture; byte-identical tests intact
- `t14` — Delete the env peer-family parsing (`PEER_ORIGIN_ENV`, `PEER_ORIGINS_ENV`, `PEER_PROXY_ENV`, `PEER_API_KEY_ENV`, `PEER_API_KEYS_ENV`, their ReplicaConfigError paths, `peer_specs_from_table` env inputs, `_PEER_SERVED_NAME_ENV`/`_PEER_ROLE_HINT` if env-only) and rewrite the six test files that declared peers through env keys to declare them through fake mesh announcements; doctor's `_pool_arming_check` retired

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `lobes/gateway/_mesh_config.py`, frozen MeshConfig with the six keys and exact defaults, key hidden from repr; commit `7ed6512` (worker lane, 20 min) |
| `t2` | delivered | `lobes/gateway/_mesh_wire.py`, schema-versioned Announcement with per-role RoleInfo (fingerprint, capacity, private) after a merger follow-up moved fingerprint/capacity from the announcement to the role; commit `95f5201` |
| `t3` | delivered | `lobes/gateway/_mesh_roster.py`, Roster + persisted Ledger with `updated_at` gossip merge and revoke-wins after two ledger defects found at review; commit `c5a5846` |
| `t4` | delivered | fleet compose passthrough for `LOBES_MESH_*`, the gateway's first bind mount (`${LOBES_MESH_DIR:-./mesh}`), env.example, secrets-scan pattern, .gitignore; the file bind mount that auto-created a root-owned dir was corrected to a directory mount; commit `6189d26` |
| `t5` | delivered | `lobes/profiles/builtin_shapes/gateway-only.toml` with spark/thor/orin/base goldens and the relaxed order_backends invariant; commit `fcf1ff2` |
| `t6` | delivered | `lobes/gateway/_mesh_routes.py`: /mesh/detect, join, announce, roster, approve, revoke, reannounce and the heartbeat daemon; built by the d2/d3 worker→cortex→worker loop, then three merger fixes (boot crash on a non-existent config field, self-origin guard); commit `8696935` |
| `t7` | delivered | `lobes/gateway/_mesh_routing.py`: RoutingSnapshot, SnapshotHolder, verification by probed fingerprint; the worker's fix leg died at the tool-call cap and a Sonnet subagent completed it (d4), finding the real defect (probed fingerprint was a dict, every verification silently failed); commit `5773b22` |
| `t8` | delivered | suffixed-lane placement, raw-id 404 listing divergent names, private roles stripped, `/mesh/reannounce` + CLI hook, `X-Lobes-Mesh-Member`; commit `ef513e6` |
| `t9` | partial | doctor's three findings, per-name flapping on the live path, verification reasons landed (commit `9fcb000`); the env-family code deletion was declined by the agent after measuring >100 call sites and moved to t13/t14 by deviation d6 |
| `t10` | delivered | `lobes/cli/_commands/mesh.py` (`lobes mesh status\|request\|approve\|revoke`, dry-run by default) and capabilities rendering of members and suffixed lanes; commit `e196868` |
| `t11` | delivered | gateway-fleet.md mesh contract + Retired section, deployment-shapes.md, colleague-stack.md, openai-api.md, secret-rotation.md, env.example, `lobes explain mesh`, CLAUDE.md, catalog re-capture, 0.76.0 bump; commit `5b26b79` and the t11 series |
| `t12` | delivered (post-merge) | Live cutover of all three boxes onto the join key; five signals measured on the final build (`docs/evidence/2026-09-12-accept-mesh-brain-join-fleet.txt`): 1, 2, 4, 5 PASS, 3 measured but its literal expectation unmet; baseline and backups filed (`27a91a8`). The two criteria left unchecked at merge time (gateway-only fourth member, Qwen Code through the mesh; lapse l2) were measured on the released 0.76.0 the same day and both PASS |
| `t13` | delivered | pool candidates, peer-only forward and busy dispatch sourced from the mesh RoutingSnapshot; wiring tests with zero peer keys; commit `d078cdc` |
| `t14` | delivered | env peer parsing deleted from `_config.py`/`_routing.py`/`server.py`/`doctor.py`, 20+ test files rewritten off env keys, template/env.example/catalog/docs closed; commit `5db3c33` and the t14 series to `4cbcc63`. Residue: docstrings and comments still name the retired keys as history |

## Mid-work Decisions

- `d1` — Workforce backend switch after wave 1: tasks t6-t11 run on Claude Sonnet subagents (Agent tool, model sonnet) instead of the gate-2 pairings (pi+associate exploration, qwen-code on worker/cortex). t1-t5 stay on the original pairings (t4/t5 still running). t12 stays in-house. — Operator decision 2026-09-11 after observing wave-1 pace and output quality: qwen-code on cortex took >70 min per task before its first edit, and pi+associate headless runs returned summaries instead of maps twice.
- `d2` — Experiment on wave 2 (t6): worker (qwen-code, worker lane) builds first; cortex reviews the committed diff via ask-colleague review and returns feedback/instructions ONLY (no fixing); worker applies the fixes. Timings recorded per leg to compare against d1's Sonnet path; the winner runs waves 3-6. — Operator hypothesis 2026-09-11: worker is the fast implementer, cortex the better judge; the loop puts each lane on its strength. Supersedes d1 for t6 only; d1 stands for t7-t11 pending the result.
- `d3` — Review leg harness: Qwen Code on the cortex lane in analyze-only (plan) approval mode replaces ask-colleague review for the d2 experiment's cortex review leg; same reviewer prompt, feedback and instructions only. — Operator decision 2026-09-11 mid-experiment: keep one harness (Qwen Code) across build, review and fix legs.
- `d4` — Waves 4-6 (t8, t9, t11) run on Claude-only agents: Sonnet subagents build in isolated worktrees, the main agent (Claude) reviews and merges. No local-lane legs (no worker, cortex or associate) for the remaining tasks. t7's in-flight worker fix leg is allowed to finish; if it has not reported within 30 minutes of this record it is cut and t7 is completed by a Sonnet subagent from the committed state. — Operator decision 2026-09-11 after the measured comparison: Sonnet control (t10) 10 min merged as-is vs local loop 146 min (t6) and 3 h+ (t7); the local-lane architecture that would fix this is filed as colleague#497 and is out of scope for this plan.
- `d5` — Amendment to d4: pi+associate stays in as a read-only FACTS pass on every Sonnet diff (t7 completion, t8, t9, t11), run in parallel with the main agent's review so it never gates a merge. Rules: facts only, quoted lines not line numbers, output to a scratch file, verdict line ignored. Retained only if the merger acts on at least one fact associate alone found; otherwise dropped at the end of the plan. — Operator decision 2026-09-11: keep associate as an independent fact-check between lanes while the build/merge path is Claude-only.
- `d6` — Scope gap found at t9: the replica-pool and auto-proxy dispatch still SOURCE their peers from the env-declared `<PREFIX>_PEER_ORIGIN(S)` routing-table fields; t7's wiring layered the mesh snapshot beside them and its end-to-end tests set `PRIMARY_PEER_ORIGIN`/ORIGINS in their fixtures. A mesh-only deployment (zero peer keys) does not pool or forward, so the confirmed 'replace' decision (c22) is not delivered and the cutover t12 cannot run yet. PROPOSED: add t13 'swap the peer source' and t14 'delete the env peer parsing and rewrite the six test files', both Sonnet, sequential, before t12; t11 proceeds now. — t9's agent (Sonnet) declined AC1 after measuring >100 call sites reading table.`replica_origins`/`peer_`\* and reported it honestly; the merger verified the wiring tests' fixtures depend on the env keys. Outcome: t13 and t14 both delivered; the gap is closed (evidence e10, e2).
- Cutover deployment settings, no record covers them: the fleet runs `LOBES_MESH_MISSED_MAX=2` (delta b5) so a lost member drops within two intervals, and both reranker hosts declare `RERANK_QUANTIZATION=none` so the strict pool rule can pass. The Spark's hand-maintained compose lacked the fingerprint-knob passthrough, so that key went into its override file rather than a re-scaffold.
- d5 retention: associate's facts pass was acted on twice (t7: builder not passing the live replica snapshot; t8 pre-merge) and produced nothing actionable on the t12 fix commits (2 hits in 3 passes, one wrong fact per pass); retained through the plan, its lane assessment posted on colleague#495.
- Four live-only defects found by the cutover after the plan's last task merged were fixed in-house rather than as new plan tasks: seed-roster merge made discovery-only (`163a4ec`), drop hold-down (`ffae309`), per-role verification (`3cef3ad`), sole-candidate plain placement (`39a2412`). Each carries a regression test in the live pass order and a delta record (b2–b4).

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|------------------------|-----------------|
| `t6` (`d1`) | Operator decision 2026-09-11 after observing wave-1 pace and output quality: qwen-code on cortex took >70 min per task before its first edit, and pi+associate headless runs returned summaries instead of maps twice. | `acceptable` |
| `t6` (`d2`) | Operator hypothesis 2026-09-11: worker is the fast implementer, cortex the better judge; the loop puts each lane on its strength. Supersedes d1 for t6 only; d1 stands for t7-t11 pending the result. | `acceptable` |
| `t6` (`d3`) | Operator decision 2026-09-11 mid-experiment: keep one harness (Qwen Code) across build, review and fix legs. | `acceptable` |
| `t8` (`d4`) | Operator decision 2026-09-11 after the measured comparison: Sonnet control (t10) 10 min merged as-is vs local loop 146 min (t6) and 3 h+ (t7); the local-lane architecture that would fix this is filed as colleague#497 and is out of scope for this plan. | `acceptable` |
| `t8` (`d5`) | Operator decision 2026-09-11: keep associate as an independent fact-check between lanes while the build/merge path is Claude-only. | `acceptable` |
| `t9` (`d6`) | t9's agent (Sonnet) declined AC1 after measuring >100 call sites reading table.`replica_origins`/`peer_`\* and reported it honestly; the merger verified the wiring tests' fixtures depend on the env keys. | `risky` |
| `t12` | The gateway-only shape was never booted on the Orin as a fourth member, and Qwen Code was not run through the mesh; both acceptance criteria were reported unchecked (lapse l2), not passed. | `needs-follow-up` |
| `t12` | Signal 3's literal expectation (4 concurrent cortex requests served by two members) could not be met: no two members host cortex in this fleet, and the reranker pool keeps an idle local lane local (delta b1, evidence e3 filed as fail). | `needs-follow-up` |
| `t12` | The plan assumed the code default `missed_max=3`; the fleet was cut over at `LOBES_MESH_MISSED_MAX=2` so signal 4 could meet its two-interval bound (delta b5). | `acceptable` |
| `t14` | Docstrings and comments across the gateway and roles modules still mentioned the retired `*_PEER_*` names as live behaviour after the parsing was gone; scrubbed on the branch in `f25b53e` before the PR gate. | `acceptable` |

## Evidence

- tests: `uv run pytest -n auto -q` at `537c849` — 4860 passed, 15 skipped (live gates)
- tests: the nine mesh test files (`tests/test_mesh_*.py`, `tests/test_gateway_serve_mesh_wiring.py`, `tests/test_cli_mesh.py`) — 293 passed at `9ce1a2a` (evidence e10)
- tests, live pass order: `tests/test_mesh_heartbeat_live.py::test_a_dropped_member_is_not_revived_by_a_peer_roster_that_still_lists_it`, `tests/test_mesh_routing.py::test_verification_is_per_role_not_all_or_nothing`, `tests/test_mesh_naming.py::test_sole_verified_member_is_plain_even_with_an_unknown_field` — pass
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r lobes`, `afi cli doctor . --strict` — clean; CI lint, secrets-scan, version-check, site-build green on `9ce1a2a`
- SonarCloud quality gate on PR #252: passed (reliability A, security A, maintainability A, new coverage 95.6 %); the 85 open code smells were fixed on the branch by three Sonnet agents (`537c849`, `52d0787`, `67fcaef`) plus one by hand, and three accepted in SonarCloud with rationale (the pre-existing 117-complexity dispatcher, a load-bearing `list()` copy, a shared handler signature)
- live re-validation on 0.76.0.dev532 (the refactored code): signals 1, 2, 3b, 4 PASS again, 3 measured as before; the Thor → Spark cortex forward answered 200 through the proxy this time (`docs/evidence/2026-09-12-accept-mesh-brain-join-fleet.txt`, re-validation section)
- live: `docs/evidence/2026-09-12-baseline-mesh-cutover.txt` (pre-cutover state, backups `~/.lobes.pre-mesh-20260911T222244Z` on every box) and `docs/evidence/2026-09-12-accept-mesh-brain-join-fleet.txt` (five signals on 0.76.0.dev529, findings table of nine live-only defects)
- validate-delivery ledger: obligations o1–o11, evidence e1–e10 (e3 = fail), deltas b1–b5, lapses l1–l3 — all approved by the operator 2026-09-12
- commits: `7216340..537c849` (107 commits on `spec/mesh-brain-join`)
- PRs / issues: PR #252; agentculture/associate#3, #4; agentculture/colleague#495, #496, #497; follow-ups #232, #215

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| A box joins the mesh with the key alone and is listed by every member within one interval | high | e1 (signal 1, 10 s) · `docs/evidence/2026-09-12-accept-mesh-brain-join-fleet.txt` |
| Dynamic membership replaced the env peer family: zero `*_PEER_*` keys on any box, doctor clean, no parsing left in the gateway | high | e2 · e10 · commits `d078cdc`, `5db3c33` |
| Every member verifies every other by fingerprint identity against its own /capabilities, per role | high | roster reads in the transcript (all `v=True`), test `test_verification_is_per_role_not_all_or_nothing`, commit `3cef3ad` |
| Roles a box lacks are reachable by their plain name on its own gateway by proxy (embedder, worker → Thor; associate → Orin; cortex Thor → Spark) | high | e6 · e8 (curl half) · signals 3b/3c |
| A stopped member leaves every roster within two intervals, its exclusive role 404s in 0.01 s, and it rejoins verified after restart | high | e4 · commits `163a4ec`, `ffae309` · test `test_a_dropped_member_is_not_revived_by_a_peer_roster_that_still_lists_it` |
| With no join key nothing mesh-related runs and responses are byte-identical to 0.75.x apart from the retired family's own entries | high | e5 · byte-identical tests in the suite |
| Two members serving the same role form one pool and share load | medium | pool formed (both reranker lanes plain, e3 basis); spill-over never observed — the idle local lane stays local; the forward path was observed on dev524 only with the local lane absent |
| Differing lanes are exposed as `{role}-{machine-name}` and the plain name stays honest | medium | observed live on dev528 (`embedder-thor`, `reranker-spark`) for a real quantization disagreement; the sole-candidate case fixed in `39a2412`; tests in `tests/test_mesh_naming.py` |
| A gateway-only member boots, joins and serves every role the mesh hosts by proxy | high | release run on 0.76.0: spark-gw joined, verified all three in 90 s, served cortex/worker/associate/embedder/reranker by proxy (`docs/evidence/2026-09-12-accept-mesh-brain-join-fleet.txt`, h8 section); `hand` 404s role_infeasible on it exactly as on every member, because no box in this fleet runs the hand lane — there is nobody to forward to |
| A robot client (Qwen Code) reaches mesh roles its local gateway does not host | high | release run on 0.76.0: `qwen -m worker` and `qwen -m associate` against the Spark gateway (which hosts neither) fixed the test file through the Thor and the Orin (h6 section of the transcript); the other proxied roles (cortex from the Thor, embedder, reranker) were measured with curl, not with Qwen Code |
| Cutover is rollback-safe | medium | e7: backups named per box; a rollback was never exercised |
| Approval ledger: a lapsed or revoked name is refused and revocation gossips | medium | unit tests in `tests/test_mesh_roster.py`, `tests/test_mesh_routes.py`; not exercised live (the fleet joined on the key alone) |

Lapse ledger evidence:

| Lapse | Code | What |
|-------|------|------|
| `l1` | `provenance-missing` | During /think I resolved park v4 (#215 raw-id pressure gate) with the decision text meant for v3 (fleet hygiene) — a wrong-id resolution, not a reasoned one. |
| `l2` | `n-below-claim` | t12 cutover skipped two of its own acceptance checks: the gateway-only shape was never booted on a fourth member (h8/o9) and Qwen Code was not run through the mesh (h6 curl half only, o8); both reported unchecked, not passed |
| `l3` | `assumption-for-measurement` | signal 4 took three builds (dev525-dev527): the first two fixes were reasoned from the code path instead of reproducing the live pass ORDER (tick, then seed merge) in a test first |

## Remaining Work / Follow-up

- (closed) gateway-only member and Qwen Code through the mesh — both measured on the 0.76.0 release run; see the transcript's release section.
- (closed by PR #254, 0.77.0) A mesh-provided role answered 404 role_infeasible for the first ~60 s after a gateway recreate — measured fixed 2026-09-12 (`docs/evidence/2026-09-12-accept-mesh-boot-window-fleet.txt`): the first reachable request after a recreate answers 200, and with a peer paused it answers 503 `role_unverified` + `Retry-After: 5` until the peer is probed.
- (closed by PR #254, 0.77.0) `/capabilities` on a member that reaches a role only via the mesh reported ready:false / hosted_by:null — measured fixed 2026-09-12 on the gateway-only member (same transcript): hosted_by, ready and proxied are mesh-sourced; a pooled role carries `members`.
- `t12` signal 3 spill-over — generate real local load on a pooled role (or add a second cortex host) and observe `X-Lobes-Proxied-By` from a pooled member; the policy keeps an idle local lane local by design.
- `t14` residue — done on the branch (`f25b53e`): docstrings and comments now describe routing via the mesh; deliberately historical mentions (Retired (t14), dated findings, issue numbers) kept; `_check_pool_arming` kept because a hand-built RoutingTable can still reach it.
- Announce only loaded lanes — the Orin announces five hosted-but-not-running lanes; harmless with per-role verification, still noise on every peer.
- Cortex proxy 200 — closed: the dev532 re-validation obtained a chat completion through the Thor → Spark forward (earlier runs were shed 429 by the Spark's own pressure policy).
- `/capabilities` JSON `hosted_by` — closed by PR #254 (mesh-sourced since 0.77.0); service-rate weighting for heterogeneous pools (#232); the raw-id pressure gate (#215); the `./mesh` mount is root-owned on first recreate (chmod applied by hand on all three boxes).
- Colleague lane architecture (colleague#495/#496/#497) and the associate facts-pass fixes (associate#3/#4) — filed, outside this plan.
