# Delivery Summary — cortex replica pool (#199)

plan: `cortex-replica-pool-199` · run: `complete` · date: `2026-08-25`
baseline: `devague summary skeleton`

## Intent

Deliver issue #199 — one logical `cortex` served by N compatible replicas,
every gateway a front, each request placed on the most available replica,
every answer naming the replica that served it — by executing the converged
plan `cortex-replica-pool-199` (spec
`docs/specs/2026-08-25-cortex-replica-pool-199.md`, plan
`docs/plans/2026-08-25-cortex-replica-pool-199.md`, both in PR #212) through
`/assign-to-workforce`: 11 tasks in 5 waves, one subagent worktree per code
task, the two live-box tasks (t1, t11) run in-house, every merge TDD-gated on
the full suite. Implementation PR: #213 (0.63.0).

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Capture the PRE-POOL baseline transcript on the live Spark+Thor pair (docs/evidence/2026-08-XX-baseline-cortex-single-owner.txt): three concurrent model=cortex requests to one gateway, then the same under local pressure while the peer idles; record gateway pins and headers
- `t2` — Config: parse the plural peer family and self-origin — <PREFIX>`_PEER_ORIGINS` / <PREFIX>`_PEER_API_KEYS` (comma-separated, positional, empty key slot legal, shorter key list = startup error) and `GATEWAY_SELF_ORIGIN` — into new RoutingTable fields (`replica_origins`: Mapping\[str, tuple\[str,...\]\], `replica_api_keys`, `self_origin`) plus per-lane declared fingerprint keys (quantization, `kv_cache_dtype`, reasoning/tool parser, speculative config); existing scalar fields and `order_backends` untouched. Files: lobes/gateway/`_config.py`, lobes/gateway/`_routing.py` (fields only), tests/`test_gateway_config_replicas.py`
- `t3` — Templates + doctor: add gateway-service passthrough lines for <PREFIX>`_PEER_ORIGINS`, <PREFIX>`_PEER_API_KEYS`, `GATEWAY_SELF_ORIGIN` and the per-lane declared fingerprint keys (`PRIMARY_QUANTIZATION`, `PRIMARY_KV_CACHE_DTYPE`, `PRIMARY_REASONING_PARSER`, `PRIMARY_TOOL_PARSER`, `PRIMARY_SPECULATIVE_CONFIG` and the other prefixes' equivalents) in lobes/templates/fleet/docker-compose.yml and document them in env.example; teach lobes doctor to flag a deployed docker-compose.yml lacking any passthrough for a key set in .env. Files: lobes/templates/fleet/docker-compose.yml, lobes/templates/fleet/env.example, lobes/cli/`_commands`/doctor.py, tests
- `t4` — Replica snapshot: new lobes/gateway/`_replicas.py` with ReplicaState (origin, local: bool, ready, busy, health, running, waiting, fingerprint, compatible: bool, reason, `last_seen`) and a ReplicaCache that, on the ReadinessCache daemon-thread pattern (separate peer thread, bounded timeouts, O(1) current()), probes each peer gateway's GET /status (busy, backends\[\].health, metrics.running/waiting) and GET /capabilities fingerprint with the peer key, probes the local lane's own /v1/models (id, `max_model_len`) and merges the declared lane config; computes compatible from served id + quantization + max context + runtime only, with `kv_cache_dtype`/draft/rope recorded as informational. Files: lobes/gateway/`_replicas.py`, tests/`test_gateway_replicas.py`
- `t5` — Selection policy: pure function `select_replica`(candidates: Sequence\[ReplicaState\], \*, affinity: str | None, `local_busy`: bool) -> Selection(origin, local, reason) in new lobes/gateway/`_selection.py` — deterministic weighted least-load: filter compatible+ready+not-busy, estimated wait = (running+waiting)/`declared_weight`, local wins ties (locality), affinity (a stable hash of the key -> preferred replica) honoured only when the preferred replica is selectable and not worse than the best by more than a declared margin; reason vocabulary local-idle | peer-less-loaded | local-busy-forwarded | affinity | sole-ready | none. Files: lobes/gateway/`_selection.py`, tests/`test_gateway_selection.py`
- `t6` — Capabilities surfaces: additive per-role 'replicas' list (origin, local, ready, busy, running, waiting, compatible, reason, fingerprint) and per-role 'fingerprint' object on the payload built by lobes/roles.py, rendered by lobes capabilities / lobes capabilities --replicas and lobes endpoint \<role\> --replicas including a 'would choose: \<origin\> (\<reason\>)' line; existing keys keep type and meaning; lobes route untouched. Files: lobes/roles.py, lobes/cli/`_commands`/capabilities.py, lobes/cli/`_commands`/endpoint.py, tests/`test_roles_replicas.py`, tests/`test_cli_capabilities_replicas.py`
- `t7` — Gateway dispatch (part 1): pool path in lobes/gateway/server.py — for a request whose model resolves (alias OR raw served id) to a role with a declared replica set, consult ReplicaCache.current() + `select_replica` before local dispatch; forward to a chosen peer origin via a generalized `_proxy_to_peer`(origin, `api_key`) that stamps X-Lobes-Proxied and returns X-Lobes-Proxied-By; an inbound X-Lobes-Proxied request is served by the local replica only (508 only when no local replica); read X-Lobes-Affinity; stamp X-Lobes-Served-By: <`GATEWAY_SELF_ORIGIN` or 'local'> on local answers and X-Lobes-Route-Reason on every pooled answer; forward X-Lobes-Affinity to the peer. Files: lobes/gateway/server.py, tests/`test_gateway_pool.py`
- `t8` — Gateway dispatch (part 2): pressure and failure semantics in lobes/gateway/server.py — under local busy pressure a pooled request is forwarded to a selectable peer instead of shed; 429 busy + Retry-After only when no replica is selectable (503 `backend_unavailable` when all are down); at most ONE forward per request (a peer's 429/4xx rides back via the existing relay and the forwarder never retries locally); pre-dispatch failure (refused/timeout/5xx before any 2xx) retries the next selectable replica at most once per replica; wire the replicas/fingerprint payload from t5 into GET /capabilities and start ReplicaCache in serve(). Files: lobes/gateway/server.py, tests/`test_gateway_pool_pressure.py`
- `t9` — N-gateway loopback integration suite: extend tests/`test_proxy_integration.py` with `_n_gateways`(n, `pool_env`) and drive the acceptance scenarios offline — spread across two replicas, local-busy forward, peer down, single hop under a marked request, mutual busy, raw-id equivalence, affinity stickiness within margin, and a no-pool byte-identical golden. Files: tests/`test_proxy_integration.py`, tests/goldens/ (no-pool golden)
- `t10` — Docs, explain and changelog: describe the replica pool (config family, empty-slot rule, `GATEWAY_SELF_ORIGIN`, selection policy, affinity header, markers, route reasons, failure table, compose passthrough + doctor check, rollback = delete the \*`_PEER_ORIGINS` line and recreate the gateway) in docs/gateway-fleet.md, docs/deployment-shapes.md, docs/colleague-stack.md (capabilities schema), docs/openai-api.md, lobes/explain/catalog.py (`_GATEWAY`/`_SHAPES`/`_API`/`_ROLES`), CLAUDE.md, CHANGELOG.md — all labelled DECLARED/UNVALIDATED until t10's transcript lands, cortex-only validation stated explicitly. Files: docs/\*.md, lobes/explain/catalog.py, CLAUDE.md, CHANGELOG.md, pyproject.toml (version bump)
- `t11` — Live acceptance on Spark+Thor: re-scaffold both deployed docker-compose.yml from the packaged template, set `GATEWAY_SELF_ORIGIN`, `PRIMARY_PEER_ORIGINS` and `PRIMARY_PEER_API_KEYS` per box (Spark lists the Thor at :8000 with an empty key slot; Thor lists the Spark at :8001 with the Spark's key), align `MODEL_GEAR_VERSION` on both, recreate only the gateway containers, and run the three scenarios — spread, Spark busy -> Thor, Thor down -> Spark — plus the affinity check; record docs/evidence/2026-XX-XX-accept-cortex-replica-pool-spark-thor.txt and flip the docs from DECLARED to VALIDATED for cortex only. Files: docs/evidence/, docs/\*.md (status flip), CLAUDE.md

Note: the plan's prose refers to sibling tasks by a label one lower than the
id the CLI assigned (e.g. t8's "payload from t5" is t6, t10's "until t10's
transcript" is t11, t11's baseline is t1). The dependency graph is correct;
only the prose labels drift.

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `docs/evidence/2026-08-25-baseline-cortex-single-owner.txt` (fda9186), run in-house before any pool code. Scenario C (429 while the peer idles) was observed *organically* — the Spark sat under iowait pressure — rather than induced; an 8-way flood queued at 11.0 tok/s aggregate (equal to one request) with the Thor at `running=0`. Absolute tok/s contaminated by background mesh load, stated in a Caveats section. |
| `t2` | delivered | `_config.py`/`_routing.py`: `PEER_ORIGINS_ENV`/`PEER_API_KEYS_ENV`, `ReplicaConfigError`, `RoutingTable.replica_origins/replica_api_keys/self_origin/lane_fingerprints`, `LANE_FINGERPRINT_SUFFIXES` (1c11326). Suffix `TOOL_PARSER` reconciled to the lanes' real `TOOL_CALL_PARSER` in-house (8b4702d). |
| `t3` | delivered | Gateway passthrough for the plural family, `GATEWAY_SELF_ORIGIN` and lane fingerprint keys; `env.example` block; `doctor` `gateway_passthrough` finding (3bb90ee). Extended in-house to scan every compose overlay (9e616a7) and then scoped to `services.gateway.environment` after Qodo's finding (5619d9b, e779a54). |
| `t4` | delivered | `lobes/gateway/_replicas.py` — `Fingerprint`, `ReplicaState`, `ReplicaCache`, `compare_fingerprints` (4ab241f). Runtime fallback from `/v1/models` `owned_by` added in-house (6b74b9f) — without it every replica read `runtime: unknown` and could never pool. |
| `t5` | delivered | `lobes/gateway/_selection.py` — `select_replica`, rendezvous-hash affinity, `REASON_*` (d60fadf); refactored under Sonar S3776 without policy change (009babb). |
| `t6` | delivered | `annotate_replicas` in `roles.py`; `--replicas` on `lobes capabilities` / `lobes endpoint` with the would-choose line; no-pool payload byte-identical; `route.py` untouched (3ad091a). Offline fingerprint's `TOOL_PARSER` lookup fixed after Qodo (5619d9b). |
| `t7` | delivered | Pool dispatch before local dispatch keyed on the owning backend (alias and raw id share it), `_ForwardTarget`, `X-Lobes-Served-By`/`X-Lobes-Route-Reason`, `X-Lobes-Affinity` read and forwarded, marked arrivals local-only (b5ac496). |
| `t8` | delivered | Busy→forward, one-forward rule, pre-dispatch retry with `X-Lobes-Route-Attempts`, 503 exhausted body with `error.attempts`, `build_replica_caches` in `serve()`, live `/capabilities` wiring (a46a191). Deviation from the brief, recorded in-code: no local-only cache is built on a no-pool box, so the no-pool payload stays byte-identical (h1) at the cost of not publishing a fingerprint there. |
| `t9` | delivered | `_n_gateways` harness + 14 real-socket tests + `tests/goldens/no-pool-gateway.json` (bc136c2). **Caught a production-only defect**: the snapshot provider was bound as a method by the descriptor protocol, 500-ing every pooled POST in a real process (de6dd3c). Two acceptance criteria re-shaped honestly: "both busy → relayed 429" cannot occur at the tier gate (see Drift); "change the key → may differ" is not a rendezvous guarantee. |
| `t10` | delivered | `docs/gateway-fleet.md`, `deployment-shapes.md`, `colleague-stack.md`, `openai-api.md`, `explain/catalog.py`, `CLAUDE.md`, CHANGELOG; 0.62.1 → 0.63.0 (6f4f696); flipped to VALIDATED (cortex-only) by t11 (e2e6558). |
| `t11` | delivered (with drift) | `docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt` (e2e6558), run off the PR's TestPyPI wheel `0.63.0.dev428`. Spark: passthrough via `docker-compose.override.yml`, base compose untouched (`d1`); Thor: compose re-scaffolded from the dev wheel's template (verified additive-only). Two runs: the first was spoiled by a 401 trap on the Thor (below); the second is the transcript. |

## Mid-work Decisions

- `d1` — t11: do NOT re-scaffold the Spark's docker-compose.yml from the packaged template — the deployed file is hand-edited with the DSpark speculative config baked into the vllm-primary command (d4, 2026-08-25; issue #204: no per-box override exists), and the template would silently revert it to the MTP default. Instead the Spark gets the new gateway passthrough lines in its existing docker-compose.override.yml (precedent: `HAND_FEASIBLE` already lives there) and only the gateway container is rebuilt; the Thor, whose deployed compose is byte-identical to the template, IS re-scaffolded as planned. lobes doctor was taught to scan every overlay so the override placement is not flagged. — Live inspection during t11 prep: diff of ~/.lobes/docker-compose.yml vs the packaged template shows 20+ Spark-only lines including the DSpark --speculative-config; the Thor diff is empty. (approved by the operator; issue #214 filed to make committed rendered compose a practice)
- Wave 2 started before wave 1 fully merged — t6/t7 depend only on t2/t4/t5 and touch disjoint files, so they were fanned out while t3 was still running; t3 merged in parallel. No conflict resulted.
- Wave-1 reconciles done in-house rather than re-briefing agents: `TOOL_PARSER` → `TOOL_CALL_PARSER` (t2 vs t3), and the `owned_by` runtime fallback (t4 vs t6's finding that no runtime knob exists).
- The Thor's compose was replaced with the dev wheel's packaged file rather than via `lobes init --apply --force`, because a force re-render would also rewrite profile keys into the Thor's hand-tuned `.env` (#204).
- Both boxes were **left running the pooled dev gateway** `0.63.0.dev428` after acceptance (operator-visible in the PR body); switching to the released 0.63.0 is a post-merge step.
- t8's brief said to build a local-only `ReplicaCache` for every hosted lane so `/capabilities` always publishes a fingerprint (c33); the agent gated it on a declared pool instead, to keep the no-pool payload byte-identical (h1). Accepted — every pooled box still publishes, and a no-pool box has no peer to publish to.
- Sonar's 21 new code smells were swept in a dedicated worktree (009babb, e779a54) before merge, per repo practice, with no behaviour change.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t11` (`d1`) | Live inspection during t11 prep: diff of ~/.lobes/docker-compose.yml vs the packaged template shows 20+ Spark-only lines including the DSpark --speculative-config; the Thor diff is empty. | risky |
| `t9` | Acceptance criterion "both gateways busy → exactly one forward and a 429 relayed" cannot occur at the tier gate: a forwarded request arrives at the peer with `model` rewritten to the raw served id, which is not a tier alias, so the receiver's pressure gate never sheds it. Split into two tests (peer *engine* 429 relayed after one forward; both busy → local 429 with reason `none`, zero forwards). Recorded as a plan risk. | needs-follow-up |
| `t11` | The plan's h23 ("raw id and alias placed identically") holds under load but **not under pressure**: the busy→forward hook sits on the tier-alias-only pressure branch (#85), so a raw-id request to a busy box is served/queued locally while the alias forwards. Observed live (scenario 2), recorded in the transcript, filed as issue #215. Deployed consumers pin the raw id, so this is the material gap. | needs-follow-up |
| `t11` | "Affinity keeps follow-up turns on the prior replica" was demonstrable only when the Spark's pressure stayed warm (F2, F4 held); on the live box the iowait oscillated around the 50 % default and availability overrode affinity on F1/F3/F5 — by design, but steady-state stickiness was not measured. | acceptable |
| `t11` | The first live run returned 401 for every request reaching the Thor: `thor@thor`'s `~/.bashrc` exports `GATEWAY_API_KEY` (the Spark's key) and compose interpolates the shell environment ahead of `.env`, so the ssh-driven recreate armed the Thor's inbound gate. Fixed by recreating with `env -u GATEWAY_API_KEY`; recorded in the transcript and in memory. Not a code defect. | acceptable |
| `t7`/`t8` | Live run showed a forwarded answer relaying the peer's own `X-Lobes-Served-By`/`X-Lobes-Route-Reason` next to the forwarder's (two route reasons on one response). Fixed post-run (3eff677); the transcript shows the pre-fix headers. | acceptable |
| `t6` | Offline fingerprint's `reasoning_parser` is always `unknown` (no `PRIMARY_REASONING_PARSER` env key exists — the lane hardcodes `--reasoning-parser=qwen3`) and `speculative_config` is `unknown` on both boxes because neither declares it in `.env` (the Spark's DSpark is baked into compose, the Thor's MTP into the template default). The drafter difference is therefore NOT visible in the fingerprint. Declared-only fields are only as honest as `.env`; #214 would close it. | needs-follow-up |
| `t8` | `readiness_cache.stop()` is not called on the new `finally` teardown (a pre-existing test stubs the cache with an object lacking `stop`); only the replica caches are stopped. | acceptable |

## Evidence

- tests: full suite `uv run pytest -n auto -q` — **3398 passed, 15 skipped** (from 3182 on main) at e779a54
- tests (pool-related, read-only re-run for this summary): `tests/test_gateway_config_replicas.py`, `tests/test_doctor_passthrough.py`, `tests/test_gateway_replicas.py`, `tests/test_gateway_selection.py`, `tests/test_roles_replicas.py`, `tests/test_cli_capabilities_replicas.py`, `tests/test_gateway_pool.py`, `tests/test_gateway_pool_pressure.py`, `tests/test_proxy_integration.py`, `tests/test_shape_goldens.py` — 280 passed
- tests (named): `tests/test_gateway_pool.py::test_forwarded_answer_carries_only_the_forwarders_pool_markers` — pass; `tests/test_gateway_pool.py::test_pressure_shed_is_forwarded_once_a_pool_is_declared` — pass; `tests/test_doctor_passthrough.py::test_passthrough_under_another_service_does_not_count` — pass; `tests/test_roles_replicas.py::test_offline_fingerprint_reads_tool_call_parser_suffix` — pass; `tests/test_gateway_replicas.py::test_local_runtime_falls_back_to_owned_by_when_undeclared` — pass; `tests/test_proxy_integration.py::test_pool_forwards_to_the_idle_peer_when_this_box_is_loaded` — pass (the test that caught de6dd3c)
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r lobes` — clean; `afi cli doctor . --strict` — PASS; `markdownlint-cli2` — clean
- CI on PR #213 at e779a54: lint, test ×2, test-publish, site-build, version-check, GitGuardian — all pass; SonarCloud quality gate OK, **0 open issues** on the PR; review threads 0 unresolved
- commits: `8da53f7..e779a54` on `feat/199-cortex-replica-pool` (33 commits, 41 files, +8339/−80)
- live: `docs/evidence/2026-08-25-baseline-cortex-single-owner.txt` (t1), `docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt` (t11); TestPyPI wheel `lobes-cli==0.63.0.dev428` (publish run 428)
- PRs / issues: #212 (spec+plan), #213 (implementation), #214 (compose-as-lock practice), #215 (raw-id pressure gap), #199 (closed by #213)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| A no-pool deployment is byte-identical to the pre-pool release | high | golden `tests/goldens/no-pool-gateway.json` · `tests/test_gateway_pool.py::test_no_pool_declared_is_byte_identical` · all builtin shape goldens unchanged |
| Two boxes declare each other and form a cortex pool with live-probed, compatible fingerprints | high | transcript scenario 0 (both fronts list the peer `ready=true, compatible=true`, `max_model_len=262144`, `runtime=vllm`) · `lobes/gateway/_replicas.py` |
| Requests to a busy front are forwarded to the free peer instead of shed | high | transcript scenarios 1 & 4 (`X-Lobes-Route-Reason: local-busy-forwarded`, `X-Lobes-Proxied-By: <thor>`) · `test_pressure_shed_is_forwarded_once_a_pool_is_declared` |
| Merged capacity is measured, not asserted | high | transcript scenario 1: 900 tokens / 47.0 s = **19.1 tok/s aggregate** vs baseline 11.0 (+74 %), same contention |
| Peer down → the front keeps serving alias and raw id with no caller change | high | transcript scenario 6 (`sole-ready`, 200 for both) · `tests/test_proxy_integration.py` peer-down test |
| A marked arrival never re-forwards (single hop) | high | transcript scenario 7 · `tests/test_gateway_pool.py` marked-arrival tests |
| Every pooled answer identifies the serving replica and why | high | `X-Lobes-Served-By` / `X-Lobes-Proxied-By` / `X-Lobes-Route-Reason` on every transcript response · 3eff677 for the de-duplication |
| Raw id and alias are placed identically **under load** | high | `tests/test_proxy_integration.py` raw-id/alias parametrised test · scenario 0 shared snapshot |
| Raw id and alias are placed identically **under pressure** | low | **not delivered** — live scenario 2 shows divergence; issue #215 |
| Affinity keeps a session on its replica under steady conditions | medium | `tests/test_gateway_selection.py` + loopback affinity tests pass; live F2/F4 held, F1/F3/F5 yielded to a busy flip — steady-state not measured live |
| Pre-dispatch failure retries the next replica once each; mid-stream drops never replay | medium | `tests/test_gateway_pool_pressure.py` (refused/timeout/5xx parametrised; 2xx-then-drop not retried) — offline only, not exercised live |
| `lobes doctor` flags a missing gateway passthrough, scoped to the gateway service, across overlays | high | `tests/test_doctor_passthrough.py` (12 tests) · e779a54 |
| The drafter difference (DSpark vs MTP) is visible in the fingerprint | unverified | not true today — both boxes report `speculative_config: unknown` (no `.env` declaration); not claimed |
| Pooling of any non-cortex role | unverified | declared-only per docs; no evidence, not claimed |

## Remaining Work / Follow-up

- **#215** — forward raw-id requests under local pressure (compute the pressure decision for pooled raw-id requests and feed `local_busy`); the deployed consumers pin the raw id, so today's live behaviour under pressure differs from the alias's. Next step: small `server.py` change on the `_pooled_busy_dispatch` seam + loopback test; then amend h23 or keep it.
- **#214** — commit each box's rendered compose/Dockerfiles as a per-box lock with secrets out; would also make `PRIMARY_SPECULATIVE_CONFIG` declared and the drafter visible in the fingerprint.
- **Post-merge deployment step** — both boxes run `lobes-cli==0.63.0.dev428` from TestPyPI. After 0.63.0 publishes: set `MODEL_GEAR_VERSION=0.63.0`, clear `GATEWAY_PIP_EXTRA_INDEX_URL`, rebuild the gateway on each box — on the Thor with `env -u GATEWAY_API_KEY` (its `~/.bashrc` exports the Spark's key). Backups: `~/.lobes.pre-199-20260825T173929Z` on both.
- **Affinity client** — which culture-side component sets `X-Lobes-Affinity` remains a follow-up (plan risk); affinity is exercised via curl only.
- **Receiver-side pressure for marked raw-id arrivals** — the t9 spec gap (plan risk): decide whether a receiving box should apply pressure to forwarded requests; today the forwarder's snapshot (peer `busy`) is the only guard.
- **Steady-state affinity and replica weights** — measure on an uncontended pair; `weight` stays declared (1.0) until then (plan risk).
- **`readiness_cache.stop()` on teardown** — symmetric shutdown for the pre-existing readiness cache (t8 note).
- **Plan prose labels** — the plan's task text refers to siblings by an off-by-one label; cosmetic, left as exported.
