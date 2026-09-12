# Delivery Summary — mesh-boot-window-and-capabilities-advert

plan: `mesh-boot-window-and-capabilities-advert` · run: `complete` · date: `2026-09-12`
baseline: `devague summary skeleton`

## Intent

Fix the two live-test follow-ups PR #253 recorded (`docs/deliveries/2026-09-11-mesh-brain-join.md`, lines 127-128): a
mesh-provided role answered 404 `role_infeasible` for ~60 s after a gateway recreate, and `/capabilities` on a member
that reaches a role only via the mesh reported `ready:false` / `hosted_by:null` while routing worked. The plan executed
eight file-disjoint tasks fanned out to worktrees by `/assign-to-workforce` (waves t1/t5/t6/t7 → t2/t3 → t4 → t8, the
last in-house), merged as PR #254 (squash `1350fe2`, 0.77.0), after three approved mid-run deviations and a review
round (Qodo, SonarCloud, and a `pi --model associate` review).

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Routing model: never-probed sentinel, per-role ready, and pending placement
- `t2` — Heartbeat: first pass at start, event-woken single-flight verify on announce and seed discovery, roster sentinel and throttled log
- `t3` — Gateway: 503 `role_unverified` for a never-probed candidate; mismatch, empty roster, realtime and raw ids pinned
- `t4` — /capabilities: mesh-sourced `hosted_by`, members, ready and proxied
- `t5` — CLI: render the never-probed sentinel and the pooled members list
- `t6` — Docs and explain: caller-facing 503 contract, roster sentinel, mesh-sourced capabilities fields
- `t7` — Live capabilities gate: accept the pooled members rule and the 503 boot window
- `t8` — Release and live acceptance: version bump, gateway re-image on all three boxes, Thor lock re-capture, evidence transcript, delivery and CLAUDE.md closure

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `MemberInfo.probed` / `ready_roles` (later `role_context`), `RolePlacement.pending_origins`, `build_snapshot(ready_roles=, discovered_roles=, role_contexts=)`; agent branch `agent/t1`, merged `8b82e8a` |
| `t2` | delivered, then extended by `d1`/`d2`/`d3` | first pass at start, `_verify_now_event`, gated announce wake, single-flight pass, roster `probed` + `not_yet_probed`, pending log; plus (d1) announce replies carry the responder's announcement and seed-listed roles feed pending, (d2) `refresh_routing_view` on ingest/discovery, (d3) parallel seed fetch; review round: lossless wake consumption, changed-announcement guard in both paths |
| `t3` | delivered | `_role_unverified_response` / `_role_unverified_body` at the fall-through sites, raw ids follow the alias, realtime pinned; merged `85bb560` |
| `t4` | delivered | `_annotate_plain_member` sources `hosted_by` (sole origin) / `members` (pooled) / `ready` / `proxied`; review round added the peer-advertised `context`; merged `a77ec3a`, `546125c` |
| `t5` | delivered | `lobes mesh status` prints `(not probed)`; `lobes capabilities` prints `proxied via mesh members: …`; merged `cb90ad5` |
| `t6` | delivered | five docs + `lobes/explain/catalog.py`, first DECLARED then flipped to MEASURED by t8; merged `bee2e36`, `d3990f3` |
| `t7` | delivered, criterion amended by `d2` (3) | pooled `members` accepted, `role_unverified` retried, a still-unverified role is a fault (review round), zero feasible generate lobes is a fault only when nothing answered by proxy; merged `dbd588b` |
| `t8` | delivered, criterion amended by `d2` (3) | five re-images (dev540 → dev544) on Spark/Thor/Orin + a gateway-only member, transcript `docs/evidence/2026-09-12-accept-mesh-boot-window-fleet.txt`, Thor catalog re-captured from the live box, docs MEASURED, CHANGELOG 0.77.0, delivery follow-ups closed; the live suite passes except its two raw-id checks (#236, recorded); `lobes doctor` lock_drift on the Thor NOT run (the lock lives in the repo catalog) |

## Mid-work Decisions

- `d1` — announce replies carry the responder's own public announcement; seed-listed roles are provisional announced roles for the pending path; the pending log line got its own wording — a recreated Spark answered 404 for 57 s and never 503 on dev540 because seed rosters carry no announcement, so nothing was pending or verifiable until the peers' 60 s heartbeats.
- `d2` — the probe ignores proxied capabilities entries; the routing view refreshes on ingest/discovery carrying probe results forward; the live suite accepts a proxy-only brain and records the two #236 raw-id failures instead of suppressing them — on dev541 the staged 503 demo still 404'd (the view was replaced only after the whole first pass) and the 35B raw id resolved to the Spark (508).
- `d3` — seed rosters are fetched in parallel — on dev543 the first pass blocked 13 s on the paused Thor's seed-roster GET, so the Orin's roster (listing the Thor's roles) merged too late for d2 to show anything.
- Review round (no deviation record; post-merge-gate fixes on the PR branch): peer-advertised `context` on proxied roles (Qodo 2), changed-announcement guard in the verify and refresh paths (Qodo 3 + the review agent's flagged residual), lossless wake consumption (Qodo 4), live-gate verdict after retries (Qodo 5), 16 SonarCloud findings + 1 more; Qodo 1 and 6 pushed back (dropped-role 404 is the pre-mesh contract; a not-ready member is never placed). The `pi --model associate` review contributed the 503 message wording.
- Operator decisions during t8: the Orin `associate` lane moved to `ASSOCIATE_GPU_MEM_UTIL=0.70` + `ASSOCIATE_MAX_NUM_BATCHED_TOKENS=8192` for concurrency resilience after it crash-looped on a plain restart and died under two concurrent long prefills at the shape's 0.80 (deployment-side; recorded on #256).

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t2` (`d1`) | measured live 2026-09-12T11:00:47Z (bootwindow-worker): a recreated Spark gateway answered 404 `role_infeasible` for 57 s and never 503 — seed rosters carry members and roles but no announcement, so the first pass had nothing to verify and `pending_origins` stayed empty until the peers' own 60 s heartbeats delivered their announcements | needs-follow-up |
| `t8` (`d2`) | measured on 0.77.0.dev541 2026-09-12T11:24Z: the staged 503 demo (Thor gateway paused, Spark recreated) still answered 404 because the snapshot is replaced only when the whole first pass completes and a paused peer holds it for the probe timeout; the 35B raw id resolved to the Spark (508 `proxy_loop`) via spark-gw and the Thor because t4 marks proxied entries ready and the probe collects ready from every entry (0.76.0 control: 404 `model_not_found`); the suite's discovery check requires a feasible generate lobe, which a gateway-only box never has | needs-follow-up |
| `t2` (`d3`) | measured on 0.77.0.dev543 2026-09-12T11:42:26Z (demo503): with the Thor gateway paused, the recreated Spark's first pass blocked on the Thor's seed-roster GET from ~11:42:37 until the unpause at 11:42:50; the Orin's roster (which lists the Thor's roles) was merged only after that, so d2's immediate-pending refresh had nothing to show and every request 404'd for the whole window | acceptable |
| `t8` | the live suite criterion "passes against spark-gw" became "passes except the two #236 raw-id checks, recorded" (`d2` (3)); the 0.76.0 control cited to the user as a regression signal was run inside the old boot window and is invalid (lapse `l7`) — a fair control 100 s after start showed 0.76.0 identical, so the raw-id mis-route is pre-existing #236 | acceptable |
| `t8` | `lobes doctor` lock_drift on the Thor after re-image was not run: the re-captured lock lives in `deployments/`, not beside the Thor's deployment, so doctor there has no lock to compare | acceptable |
| `t3` | the served-backend fall-through site is unreachable from any `build_config` table; its guard shares the tested helper but the site itself has no test (lapse `l4`, pre-existing dead path) | risky |
| `d2` (1) | the probe-skips-proxied change did not fix the raw-id 508 it was proposed for (lapse `l8`); it stays as a correct invariant, the symptom is #236 | acceptable |

## Evidence

- tests (merged main `1350fe2`, 2026-09-12T14:16:36Z): 401 passed across the mesh/gateway modules; full suite 4947 passed, 15 skipped on the PR head; per-obligation node ids in `devague evidence --list` (`e1`–`e11`)
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r lobes`, `afi cli doctor . --strict`, `scripts/scan_deployment_secrets.py` — all clean at the PR head; SonarCloud 0 open issues, quality gate OK
- live: `docs/evidence/2026-09-12-accept-mesh-boot-window-fleet.txt` (sections A–H, raw-id matrix + fair 0.76.0 control, Orin recovery and resilience)
- commits: `d41f5ac..1350fe2` (squash of PR #254; branch history `39b1d4b..7872825`)
- PRs / issues: PR #254; follow-ups #256 (umbrella), #236, #255; comments on #236 and #256
- ledgers: deviations `d1`–`d3` (approved), evidence `e1`–`e11` and deltas `b1`–`b5` (proposed), lapses `l1`–`l11` (proposed)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| a recreated gateway answers its first reachable request 200 — the ~60 s 404 window is gone | high | transcript section A on dev541/dev543/dev544 vs A0 on dev540 · `e11` (`test_a_cold_box_learns_a_seed_peers_announcement_from_the_reply_and_verifies_it`, `test_a_hung_seed_does_not_delay_the_other_seeds_discovery`) |
| a role whose only candidate is announced but not yet probed answers 503 `role_unverified` with `Retry-After: 5`, `hosted_by` and the `X-Lobes-Mesh-*` headers; mismatch and empty roster keep 404; realtime untouched | high | `e1` `TestNeverProbedMemberIsPending` · `e2` transcript section B (13 × 503 then 200) · `TestMismatchVariant`/`TestDropMember` unchanged · `e8` realtime test |
| `/mesh/roster` carries `probed` + `not_yet_probed`; both CLIs render the new fields | high | `e5` `TestRosterRecordBootWindow` · `e6` live rosters · `tests/test_cli_mesh.py`, `tests/test_cli_capabilities.py` |
| `/capabilities` on a mesh-only member sources `hosted_by` (sole origin) / `members` (pooled) / `ready` / `proxied` from the roster; byte-identical with the mesh disabled | high | `e3` `tests/test_mesh_naming.py` · `e4` transcript section C/D (spark-gw) · section F (ready flips within 31 s) |
| a proxied role's `context` is the serving peer's advertised window | medium | `tests/test_mesh_naming.py` context cases (review round) — test-proven only, not measured live (lapse `l11`, proposed) |
| verification is single-flight on the heartbeat thread; an unchanged re-announce triggers no extra probe; wakes are never lost | high | `e7` announce-storm + never-overlap tests · `TestVerifyNowEvent` |
| a probe result is discarded when the member's announcement changed mid-probe, in both the verify and refresh paths | high | `test_an_announcement_replaced_mid_probe_stays_pending_until_the_next_pass` · `TestRefreshDropsResultsForAChangedAnnouncement` |
| no new rendered or allowlisted key; `hosted_by` is an announced `GATEWAY_SELF_ORIGIN`; `SCHEMA_MAJOR` unchanged; mixed-version members interoperated during the staggered rollout | high | `e9` grep gate + empty `lobes/profiles`/`lobes/templates` diff · `e10` transcript re-image sections |
| the Thor catalog entry is re-captured from the live box and validates | high | `deployments/jetson-agx-thor__thor-worker/` (`d3990f3`), `tests/test_variation_catalog.py` |
| `lobes doctor` reports no lock_drift on the Thor after re-image | unverified | not run — the lock is a catalog entry, not a deployment-side lock |
| the live capabilities suite passes on a gateway-only member | medium | 3 of 5 pass; the two raw-id checks fail for pre-existing #236 (fair 0.76.0 control identical) — recorded, not claimed |
| seed-discovered members are probed "in the same pass" | medium | `e11` at strength `fidelity`: the probe lands in the event-woken pass immediately after discovery (lapse `l5`); no tick wait is what the tests assert |

Lapse ledger: `l1`–`l11` are all still proposed, so none has capped a confidence above; when adjudicated, `l4` (untested fall-through site), `l5` (same-pass wording), `l7` (invalid first control), `l8` (d2 (1) misattribution) and `l11` (context test-only) are the ones that bear on rows here.

## Remaining Work / Follow-up

- #256 umbrella: #236 raw checkpoint-id addressing through a non-hosting front (pre-existing; the suite's two raw-id checks fail on every mesh member) and #255 the verify-failure log line's `auth: rejected` wording.
- Release run: the three fronts still run `0.77.0.dev544` from TestPyPI; re-pin to `0.77.0` from PyPI and re-measure section A once, as #253 did for 0.76.0.
- Adjudicate `e1`–`e11`, `b1`–`b5`, `l1`–`l11` (`devague evidence/delta/lapse --confirm|--reject`).
- `lobes doctor` on the Thor against the re-captured lock is unverified (see Delivery Claims).
- Repo inconsistency found on the way: CLAUDE.md's associate paragraph says `ASSOCIATE_GPU_MEM_UTIL=0.56`, the `orin-associate` shape pins 0.80 (measured gears-first); the deployed Orin now runs 0.70 + 8192 batched tokens at the operator's request — reconcile the docs and the shape.
- The served-backend mesh branch in `server.py` is dead code with no test (`l4`); decide whether to delete or reach it.
