# mesh-boot-window-and-capabilities-advert

> lobes gateway: a mesh-provided role is verified the moment it is announced, answers a retryable not-yet-verified status during the boot window instead of 404 `role_infeasible`, and /capabilities reports mesh-provided roles with a mesh-sourced `hosted_by` and ready
> instruction: acceptance transcript under docs/evidence/ dated after the merge; CLAUDE.md/docs say DECLARED until it lands

## Audience

- Colleague-side callers that address roles by name through any mesh member (Qwen Code, culture agents, reachy-mini-cli), and the operator running lobes capabilities / lobes mesh status after a gateway recreate.
  - instruction: grep both docs for `role_unverified`

## Before → After

- Before: After a gateway recreate a mesh-provided role 404s `role_infeasible` for ~60 s until the first periodic verification pass (docs/evidence/2026-09-12-accept-mesh-brain-join-fleet.txt:411), and /capabilities on a member that reaches a role only via the mesh shows ready:false / `hosted_by`:null beside a correct member name (same transcript, line 439).
  - instruction: read the new transcript header
- After: A recreated gateway verifies announced members within seconds (at boot, on announce, on seed discovery); a request that lands before that answers 503 `role_unverified` with Retry-After 5 s and mesh headers naming the pending member; /capabilities on any member reports a mesh-provided role with `hosted_by` = the verified peer origin when the role has one plain origin (a members list when pooled), ready from the peer's probe, proxied:true.
  - instruction: same acceptance transcript
  - ⚠ contested by `d1` (needs-follow-up): measured live 2026-09-12T11:00:47Z (bootwindow-worker): a recreated Spark gateway answered 404 `role_infeasible` for 57 s and never 503 — seed rosters carry members and roles but no announcement, so the first pass had nothing to verify and `pending_origins` stayed empty until the peers' own 60 s heartbeats delivered their announcements

## Why it matters

- Callers cannot tell a transient boot window from a role nobody hosts, so a retry loop gives up on a 404 that would have succeeded a minute later; and an advert that says ready:false while routing works breaks the 'advertised implies reachable' contract the live capabilities gate enforces.
  - instruction: script it in the acceptance run

## Requirements

- The heartbeat loop runs its first verification pass at thread start, not one full `LOBES_MESH_HEARTBEAT_S` later: `_wait_for_tick` (`_mesh_routes.py`:1108-1136) resolves a None deadline to now+interval without waiting and the loop then 'continue's (lines 1284-1297), so the first real pass lands ~60 s after a gateway recreate — the measured window.
  - instruction: extend tests/`test_mesh_heartbeat_live.py` with a timestamped first-probe assertion
  - honesty: With `LOBES_MESH_HEARTBEAT_S`=1 the first verification probe reaches a fake peer in < 1 s of `start_mesh`, not ~1 interval later.
- An inbound announce and a seed-roster discovery each trigger an immediate verification of that member: MeshRoutes.announce() (`_mesh_routes.py`:459-463) only sets `_verify_event`, which is read on the next periodic tick; `_merge_seed_members` (1342-1363) runs after `_tick_and_collect` in the same pass, so a seeded member verifies one interval later. `verify_members` already runs its probes outside routes.`_lock` (dev518 note 1170-1175; `test_mesh_heartbeat_live.py`:80-163), so calling it out-of-band is lock-safe.
  - instruction: new tests beside `test_a_slow_peer_probe_never_blocks_the_roster_or_inbound_announces`
  - ⚠ contested by `d1` (needs-follow-up): measured live 2026-09-12T11:00:47Z (bootwindow-worker): a recreated Spark gateway answered 404 `role_infeasible` for 57 s and never 503 — seed rosters carry members and roles but no announcement, so the first pass had nothing to verify and `pending_origins` stayed empty until the peers' own 60 s heartbeats delivered their announcements
  - honesty: A POST /mesh/announce from a new member results in a /capabilities probe of that member before the next periodic tick, and a seed-discovered member is probed in the same pass that discovered it; roster reads and inbound announces still return in < 1 s while a slow probe is in flight.
- MemberInfo/the /mesh/roster payload distinguish 'never probed yet' from 'probed and clean': today `unverified_reason`=None means both (`_mesh_routing.py`:54-61 docstring; `_build_member_record` `_mesh_routes.py`:498-521), pinned by tests/`test_mesh_verify_reason.py`:96-108 and 127-135, which must be updated deliberately, not broken.
  - instruction: run the two named tests after the change; assert the new sentinel on a never-probed member
  - honesty: GET /mesh/roster distinguishes a never-probed member (e.g. `unverified_reason` '`not_yet_probed`' or an explicit probed:false) from a cleanly verified one, and tests/`test_mesh_verify_reason.py`:96-135 are updated to the new meaning rather than deleted.
- A request for a role that an announced-but-not-yet-probed member hosts answers 503, error.type/code `role_unverified`, Retry-After = `BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS` (5 s), error.`hosted_by` naming the pending member's origin, X-Lobes-Mesh-Member/-Origin/-Role plus X-Lobes-Mesh-Unverified via `mesh_markers`(unverified=True) — instead of 404 `role_infeasible`. Today `_collect_role_candidates` (`_mesh_routing.py`:608-625) drops every unverified member silently and server.py:3161/3268/3337 fall through to the plain 404; the 503+Retry-After builder at server.py:1414-1424 and the `_error_body` type list (686-703) are the shape to extend.
  - instruction: tests in `test_mesh_routing_wiring.py`: one new never-probed case, TestMismatchVariant and TestDropMember unchanged and green
  - honesty: A request for a role announced by a never-probed member answers 503, body error.type=`role_unverified` with `hosted_by` = that member's origin, headers Retry-After=`heartbeat_s` and X-Lobes-Mesh-Member/-Origin/-Role/-Unverified; a probed-and-disagreed member and an empty roster still answer 404 `role_infeasible` with no `hosted_by`.
- GET /capabilities on a member that reaches a role only via the mesh sources `hosted_by`, ready and proxied from the RoutingSnapshot: `hosted_by` = the verified peer's announced origin ONLY when the role has exactly one plain origin; when the role is pooled across several plain origins the entry carries a members list instead and no `hosted_by` (q4, resolved a), so X-Lobes-Proxied-By always equals `hosted_by` whenever `hosted_by` is present; ready = the peer's per-role ready bit fetched in `_probe_member_capabilities` (`_mesh_routes.py`:1540-1554, currently discarded before MemberInfo); proxied:true. Today `annotate_peer_referrals` (roles.py:1169-1266) reads only the retired table.`peer_origins`/`peer_proxied` and is a no-op on every real deployment.
  - instruction: flip tests/`test_peer_referral.py`:255-263 and add a mesh-snapshot fixture case; byte-identical payload with mesh disabled
  - honesty: `capabilities_payload` on a member that hosts a role nowhere but has a verified plain-placed peer for it emits `hosted_by` = that peer's announced origin, ready = the peer's probed ready bit, proxied:true; with no verified peer the three keys are absent/false exactly as today.
- The tests that pin the current gap flip to the new contract: tests/`test_peer_referral.py`:255-263 `test_capabilities_hosted_by_stays_env_sourced_and_the_env_knob_is_inert` and its neighbouring no-op tests (266-289); tests/`test_live_capabilities.py`'s 'advertised ready implies reachable' gate (line 348) and proxied-vs-X-Lobes-Proxied-By cross-check (~511-586) must still hold for mesh-sourced values.
  - instruction: run the live suite against spark-gw during the acceptance re-run
  - honesty: tests/`test_live_capabilities.py`'s advertised-ready-implies-reachable gate and the proxied-vs-X-Lobes-Proxied-By cross-check pass on the gateway-only member live.
- Docs and explain follow in lockstep: docs/gateway-fleet.md 643-653 (trust-but-verify, 'receives nothing'), docs/colleague-stack.md 421-425 and 589-680 (`hosted_by`/proxied/ready semantics), docs/deployment-shapes.md 652-654 and 682 (404 body and state table), docs/openai-api.md `role_infeasible` mentions (204, 541, 621-626), lobes/explain/catalog.py ~1467-1471, and the two follow-up lines in docs/deliveries/2026-09-11-mesh-brain-join.md (127-128, 134) are closed with a citation to a new evidence transcript.
  - instruction: grep the docs for 'receives nothing' and '`hosted_by` is not mesh-sourced' after the edit; markdownlint-cli2 on touched docs
  - honesty: Every doc line cited in s8-s10 is updated in the same PR and the delivery follow-up lines 127-128/134 cite the new evidence transcript; markdownlint passes.
- Announce- and discovery-triggered verification wakes the heartbeat loop through an event (the `_reannounce_event` pattern, `_mesh_routes.py`:1108-1136) and only when the announcing member is unverified or its announcement changed; the pass itself stays single-flight on the loop thread and is never run inside the announce handler. Why: server.py:69/5338 is a ThreadingHTTPServer, so a handler-thread `verify_members` would overlap the loop's pass and SnapshotHolder.replace (`_mesh_routing.py`:419-427) is last-writer-wins — an older pass could overwrite a newer snapshot; and `_collect_members_to_verify` (1491-1506) probes EVERY member, so one pass per announce with N members re-announcing each heartbeat is N² probes per heartbeat.
  - instruction: new test beside `test_a_slow_peer_probe_never_blocks_the_roster_or_inbound_announces` counting probe hits per fake member
  - honesty: With three fake members each announcing every second for 10 s (`heartbeat_s`=1), each fake /capabilities is probed at most once for its first announce plus once per periodic tick — never once per announce; and POST /mesh/announce returns in < 50 ms while a 3 s slow probe is in flight.
- Observability: each 503 `role_unverified` is logged through the existing throttled RejectionLog (`verify_log`, `_mesh_routes.py`:1585-1590) once per member, not per request, and lobes mesh status renders the never-probed sentinel — mesh.py:215-258 already reads verified/`unverified_reason` via .get, so an older CLI tolerates the new field.
  - instruction: unit test on the RejectionLog path; CLI render test in tests/`test_cli_mesh.py`
  - honesty: Ten consecutive 503s for one pending member produce one gateway stderr line; lobes mesh status shows the sentinel text for a never-probed member.
- Rollout: the fix ships only in the gateway image, so every member re-images its front (`MODEL_GEAR_VERSION` bump, lobes up gateway --build --apply); the announcement wire schema (`SCHEMA_MAJOR`) is unchanged so a mixed-version mesh interoperates during the rollout; and deployments/jetson-agx-`thor__thor`-worker/ is re-captured because its lock tracks the .env digest (CLAUDE.md, deployment lock section).
  - instruction: git diff on `_mesh_wire.py`; transcript roster reads mid-rollout; doctor output on the Thor
  - honesty: `SCHEMA_MAJOR` is unchanged in the diff; a 0.76.x member and the new member verify each other live during the rollout; the Thor catalog lock is re-captured in the same PR and lobes doctor reports no `lock_drift` on the Thor after re-image.

## Honesty conditions

- All three behaviours (verify-on-announce/boot, 503 `role_unverified`, mesh-sourced `hosted_by`/ready/proxied) are measured live on the fleet before any doc calls them validated (#108).
- TestMismatchVariant.`test_mismatch_fingerprint` and TestDropMember stay green unchanged.
- `hosted_by` values in the payload are string-equal to an announced `GATEWAY_SELF_ORIGIN` in the roster; no new key appears in env.example, render.py tables or the lock allowlist; the profile grep gate in tests/`test_shape_gateway_only.py` passes.
- `_mesh_config.py` defaults are byte-identical after the change and no forwarded request gains a second hop.
- docs/colleague-stack.md and docs/gateway-fleet.md name the caller-facing status and the operator-facing roster sentinel respectively.
- The two transcript lines (411, 439) are quoted as the baseline in the new evidence transcript's header.
- The live re-run shows the 503-then-200 sequence and the populated capabilities dump.
- A one-line retry loop (curl with Retry-After honoured) succeeds through the boot window on the live fleet.
- Every number in the success signal appears in the acceptance transcript with a timestamp or a count.
- A realtime handshake against a declared-off stt lane still answers 404 `role_infeasible` with a never-probed member announcing stt; a raw-id chat request for that member's checkpoint answers 503 `role_unverified`.
- In the live transcript the gateway-only member's roster shows every member re-verified on each heartbeat (a verify log line or a changing last-verified marker per tick), and stopping one member's vLLM lane flips its ready to false within one heartbeat plus the probe timeout.

## Success signals

- On the 3-box fleet, a request for a mesh-provided role issued < 10 s after a gateway recreate answers 503 `role_unverified` (never 404), and a retry after Retry-After answers 200 within 1 heartbeat (60 s); the gateway-only member's /capabilities shows `hosted_by` non-null, ready:true and proxied:true for >= 5 mesh-provided roles (cortex, worker, associate, embedder, reranker); the offline suite passes with the two pinning tests flipped.
  - instruction: read the transcript
  - ⚠ contested by `d1` (needs-follow-up): measured live 2026-09-12T11:00:47Z (bootwindow-worker): a recreated Spark gateway answered 404 `role_infeasible` for 57 s and never 503 — seed rosters carry members and roles but no announcement, so the first pass had nothing to verify and `pending_origins` stayed empty until the peers' own 60 s heartbeats delivered their announcements

## Scope / boundaries

- A member whose probe RAN and disagreed on fingerprint keeps today's 404 `role_infeasible` with no `hosted_by`: TestMismatchVariant.`test_mismatch_fingerprint` (tests/`test_mesh_routing_wiring.py`:556-635) and the spec's honesty clause (docs/specs/2026-09-11-mesh-brain-join.md:133, 'requests for its roles either pool elsewhere or 404 `role_infeasible` naming no `hosted_by` — never hang') stay true; only the never-probed boot window gets the retryable status. A roster with zero members (TestDropMember, 883-946; tests/`test_mesh_routing.py`:500-586) also stays 404.
  - instruction: uv run pytest tests/`test_mesh_routing_wiring.py` -k 'Mismatch or Drop'
- \#92 holds: `hosted_by` is only ever the peer's own announced `GATEWAY_SELF_ORIGIN` as the roster verified it, never derived from a socket, hostname or DNS — docs/colleague-stack.md:677-680 already sanctions 'on a mesh-brain member, the origin the mesh roster verified'. The profile renderer stays mesh-blind (tests/`test_shape_gateway_only.py`:22-26, 318-341 grep gate) and no new .env or render-time key is added — membership is runtime state (spec: 'Membership state is RUNTIME state, never a rendered .env key').
  - instruction: assert origin equality in the new test; git diff --stat shows no lobes/profiles/ or templates/env.example change
- Forwarding stays single-hop and the heartbeat period / `LOBES_MESH_MISSED_MAX` defaults (60 s / 3, `_mesh_config.py`:79-107,158) are unchanged: the fix is first-pass timing plus announce-triggered verification, not a faster clock.
  - instruction: git diff lobes/gateway/`_mesh_config.py` is empty; 508 `proxy_loop` test still green
- GET /v1/realtime keeps today's 404 `role_infeasible` for a declared-off stt lane (never proxied, `_realtime.py`:20,151) and is untouched by the 503 path; a raw model-id request whose checkpoint is announced only by a never-probed member follows the SAME 503 `role_unverified` rule as the role alias, because the served-backend branch (server.py:3268-3283) shares the fall-through.
  - instruction: two tests: one in the realtime suite, one beside the never-probed case in `test_mesh_routing_wiring.py`

## Non-goals

- Not in this fix: announcing only loaded lanes (the Orin's five hosted-but-not-running lanes, delivery line 131), #232 service-rate weighting, #215 raw-id pressure gate, the root-owned ./mesh mount, and the two gateway-only scaffold frictions (`VLLM_PORT` publish, fixed `container_name`) recorded in the h8 transcript section.
- The Orin/Thor/Spark live re-run is the acceptance, not a unit-test claim: per the #108 rule the new status and the mesh-sourced /capabilities fields are DECLARED until a transcript under docs/evidence/ shows a fresh gateway recreate answering 503 `role_unverified` then 200 within one heartbeat, and a gateway-only member's /capabilities showing `hosted_by`/ready for a mesh-provided role.

## Assumptions

- The periodic verification pass runs only when `_verify_event` was set since the last tick (`_tick_and_collect`, `_mesh_routes.py`:1186-1188); because every member re-announces each heartbeat this is every tick in practice, so the accepted ≤1-heartbeat staleness of a mesh-sourced ready bit holds, and a member that stops announcing is pruned after `missed_max` rather than left advertising ready:true.
  - instruction: acceptance run: stop the Orin's associate lane, poll /capabilities on spark-gw every 5 s

## Scope exploration

- `s1` — `lobes/gateway/_mesh_routes.py (_wait_for_tick 1108-1136, _heartbeat_loop 1280-1318)`: first iteration resolves deadline=now+interval without waiting and then 'continue's, so the first tick+verify pass runs a full `heartbeat_s` (60 s) after thread start — the measured ~60 s window; `build_mesh_routes` (963-1017) never verifies at boot
  - seeds: `c2`
- `s2` — `lobes/gateway/_mesh_routing.py (MemberInfo 35-69, _collect_role_candidates 608-625, mesh_markers 466-490) + tests/test_mesh_verify_reason.py:96-135`: `unverified_reason`=None means both never-probed and clean (pinned by two tests); the candidate filter drops unverified members silently so 'nobody hosts it' and 'not probed yet' are one 404; `mesh_markers` already has an unverified=True path unused by the 404 branch
  - seeds: `c4`, `c5`
- `s3` — `lobes/gateway/server.py (_role_infeasible_body 729-778, _error_body 686-703, 503 builder 1414-1424, mesh fall-through 3161/3268/3337, forward headers 3143-3153)`: the 404 body has no field for a known-but-unverified candidate; a 503+Retry-After builder and the `BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS` constant exist to clone; the docstring enumerates four deliberately distinct error types a fifth must join
  - seeds: `c5`, `q1` (question, resolved)
- `s4` — `tests/test_mesh_routing_wiring.py (TestMismatchVariant 556-635, TestDropMember 883-946) + tests/test_mesh_routing.py:290-330,500-586 + docs/specs/2026-09-11-mesh-brain-join.md:133`: probed-and-disagreed and zero-member rosters are pinned 404 `role_infeasible` without `hosted_by`, and the spec's honesty clause requires it — the retryable status must key off never-probed only
  - seeds: `c9`
- `s5` — `lobes/gateway/server.py capabilities_payload 3953-4077 + lobes/roles.py (build_role_registry 960-1115, annotate_peer_referrals 1169-1266, _annotate_plain_member 1506-1528, annotate_mesh_naming 1531-1576)`: feasible/loaded/ready come from the local RoutingTable only; `hosted_by`/proxied come from the retired table.`peer_origins`/`peer_proxied` and are a documented no-op post-t14; only 'member' is mesh-sourced via `compute_role_placement` — hence ready:false / `hosted_by`:null beside a correct member
  - seeds: `c6`
- `s6` — `lobes/gateway/_mesh_routes.py _probe_member_capabilities 1508-1554 + lobes/gateway/_mesh_routing.py verify_member_roles 173-216`: the peer's per-role ready bit and fingerprint are fetched during verification, used only as a pass/fail gate, and discarded before MemberInfo — the raw material for a mesh-sourced ready already exists at probe time
  - seeds: `c6`, `q2` (question, resolved)
- `s7` — `tests/test_peer_referral.py:22-26,255-289 + tests/test_live_capabilities.py:348,~511-586 + tests/test_shape_gateway_only.py:22-26,291-341`: one test explicitly pins `hosted_by` absent and ready:false for an unhosted role as a recorded gap; the live gate requires advertised-ready roles to be reachable and proxied to match X-Lobes-Proxied-By; the profile-render grep gate forbids mesh knowledge in profiles
  - seeds: `c7`, `c10`
- `s8` — `docs/colleague-stack.md 421-425, 589-595, 661-680`: the /capabilities contract already says `hosted_by` may be 'on a mesh-brain member, the origin the mesh roster verified' and ready is the live peer probe — the docs sanction the fix; the shipped code does not do it
  - seeds: `c10`, `c8`
- `s9` — `docs/gateway-fleet.md 615-653 + lobes/explain/catalog.py ~1467-1471 + docs/deployment-shapes.md 652-654,682 + docs/openai-api.md 204,541,621-626`: trust-but-verify prose promises 'an unverified member's roles receive nothing' with only a roster-level `unverified_reason`; no caller-facing status for the window is documented anywhere; the 404 body/state table and explain mirror must gain the new case in lockstep
  - seeds: `c8`
- `s10` — `docs/deliveries/2026-09-11-mesh-brain-join.md 124-134 + docs/evidence/2026-09-12-accept-mesh-brain-join-fleet.txt 411,439,443-448 + docs/plans/2026-09-11-mesh-brain-join.md`: both follow-ups are recorded verbatim in the delivery (127-128, 134) with no issue number; the plan is silent on them; the adjacent follow-ups (loaded-lanes-only announce, #232, #215, ./mesh mount, `VLLM_PORT`/`container_name` frictions) are separate
  - seeds: `c12`, `c13`
- `s11` — `lobes/gateway/_mesh_config.py 79-107,158 + GitHub open issues (gh api, 2026-09-12)`: heartbeat default 60 s / `missed_max` 3 are the knobs the window depends on; no open issue tracks either follow-up (#118 records the same 5xx-without-Retry-After pattern for the audio facade)
  - seeds: `c11`
- `s12` — `lobes/gateway/_mesh_routes.py (announce 437-471, approve 605, _merge_seed_members 1342-1363, dev518 lock note 1170-1175, verify_members 1597-1637)`: announce/approve set only `_verify_event` (read on the next periodic tick, never wakes the loop — only `_reannounce_event` does); seed discovery runs after the verify decision in the same pass; `verify_members` holds routes.`_lock` only for the snapshot copy and probes in a ThreadPool outside it, so an out-of-band call is safe (regression test `test_mesh_heartbeat_live.py`:80-163). Full-pass vs single-member probe cost is parked as v1
  - seeds: `c3`
- `s13` — `challenge pass / concurrency lens: server.py:69,5338 (ThreadingHTTPServer), _mesh_routing.py SnapshotHolder 405-440, _mesh_routes.py _collect_members_to_verify 1491-1506, _run_verification_probes 1557-1594 (pool of ≤8)`: handler-thread verification would overlap the loop's pass with last-writer-wins snapshot replace, and every pass probes every member — seeded the single-flight, event-wake requirement
  - seeds: `c21`
- `s14` — `challenge pass / adjacent-systems lens: lobes/cli/_commands/capabilities.py 306-370, lobes/cli/_commands/mesh.py 215-258, ../culture/culture (grep hosted_by, role_infeasible: no hits)`: the CLI already prints 'proxied via this gateway from peer: `hosted_by`' for feasible:false+proxied and 'served by mesh member'; the mesh status CLI reads roster fields with .get; no sibling consumer in the culture checkout reads `hosted_by` or the 404 type
  - seeds: `c23`
- `s15` — `challenge pass / unstated-assumption lens: decision c14 vs after_state c18; server.py:678-683 BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS=5; _mesh_routes.py _DIAL_TIMEOUT_S`: a 60 s Retry-After is honest only for the periodic-pass design the fix removes — routed to a blocking hard question on c14
  - seeds: `q3` (question, resolved)
- `s16` — `challenge pass / data-flow lens: docs/evidence/2026-09-12-accept-mesh-brain-join-fleet.txt h8 (reranker peer-less-loaded across spark+thor) + tests/test_live_capabilities.py ~511-586`: a pooled role has no single `hosted_by` while the live gate demands X-Lobes-Proxied-By equal it — routed to a blocking hard question on c6
  - seeds: `q4` (question, resolved)
- `s17` — `challenge pass / failure-mode + lifecycle lens: _realtime.py 20,151; server.py served-backend branch 3268-3283; _mesh_wire.py SCHEMA_MAJOR; deployments/jetson-agx-thor__thor-worker/`: realtime must stay out of the 503 path and raw ids in; no wire schema change so mixed versions interoperate mid-rollout; the Thor catalog lock re-capture is part of shipping
  - seeds: `c22`, `c24`
- `s18` — `challenge pass / security lens: announce-gated probes (_mesh_routes.py announce 437-471 behind _check_key), roster capacity clamp (delivery t3, not re-read this pass)`: an authenticated peer can already make this box GET any origin it announces on the periodic pass; the fix changes cadence only and c21's unverified-only gate bounds it to one extra probe per new identity; amplification under a flood of distinct names rests on the roster capacity clamp, which this pass did not re-read
- `s19` — `challenge pass / reversibility + observability lens: doctor --repin-version (CLAUDE.md), RejectionLog verify_log 1585-1590, _tick_and_collect verify_dirty 1186-1188`: no persisted state or env key changes, so rollback is a gateway re-pin; the ready-refresh cadence depends on announces keeping the verify flag dirty every tick — captured as assumption c25 with a live honesty check
  - seeds: `c25`

## Decisions

- Boot-window status is 503 with error.type/code `role_unverified` and Retry-After = `BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS` (5 s, server.py:678-683) — the same retryable discipline as `backend_unavailable` / `server_busy`, not a 404 variant and not the 60 s heartbeat (q3, resolved b).
- A mesh-provided role on /capabilities reports proxied:true and ready from the peer's probe-time ready bit, refreshed on every verification pass; up to one heartbeat of staleness is accepted.

## Hard questions

- Boot-window status: 503 `role_unverified` + Retry-After (retryable, matches `backend_unavailable`/`server_busy` discipline) or keep 404 `role_infeasible` and add a 'pending' marker so existing 404 consumers see no new status code? Recommendation: 503. (resolved: 503 `role_unverified` with Retry-After = `heartbeat_s` (user confirmed 2026-09-12))
- For a mesh-provided role should /capabilities set proxied:true (this box does forward, via auto-wiring) and should ready be the peer's probe-time ready bit (up to one heartbeat stale) or a live per-request probe? Recommendation: proxied:true, probe-time bit, refreshed on each verify pass. (resolved: proxied:true; ready = the peer's probe-time ready bit, refreshed on each verify pass (user confirmed 2026-09-12))
- A pooled role with more than one plain origin (live: reranker placed across spark+thor, X-Lobes-Route-Reason peer-less-loaded, transcript h8) has no single `hosted_by`: the forward may go to a member other than the one named, and the live gate (tests/`test_live_capabilities.py` ~511-586) requires X-Lobes-Proxied-By to name `hosted_by`. Options: (a) `hosted_by` only when exactly one plain origin, plus a members list when pooled; (b) `hosted_by` = first origin and relax the gate to 'X-Lobes-Proxied-By is in the pool'. Recommendation: (a). (resolved: (a) `hosted_by` only when exactly one plain origin; a members list when the role is pooled — user decided 2026-09-12)
- Retry-After = `heartbeat_s` (60 s) contradicts `after_state` c18 ('verifies within seconds') once c2/c3/c21 land: a caller honouring it waits ~12× longer than the probe round-trip. Options: (a) Retry-After = `_DIAL_TIMEOUT_S` (the probe budget); (b) `BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS` = 5, the existing 503 constant; (c) keep 60. Recommendation: (b). (resolved: (b) Retry-After = `BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS` (5 s), the existing 503 constant — user decided 2026-09-12)

## Open parks

- [unknown_nonblocking] Probe amplification under a flood of distinct member names is bounded by the roster capacity clamp recorded in the mesh-brain-join delivery (t3); this pass did not re-read that clamp or measure the bound

## Resolved vagueness

- [unknown_nonblocking] Whether an announce-triggered verify should be a full `verify_members` pass (all members, ThreadPool probes) or a single-member probe; the cost is one /capabilities GET per member per announce and is bounded by `_DIAL_TIMEOUT_S`, but the fan-in on a 3-box mesh recreate has not been measured — resolved: wake the loop via an event; single-flight full pass (c21)
