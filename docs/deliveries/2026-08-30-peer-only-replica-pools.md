# Delivery Summary — peer-only replica pools

plan: `peer-only-replica-pools` · run: `complete` · date: `2026-08-30`
baseline: `devague summary skeleton`

## Intent

> A box that hosts none of a role still spreads that role's traffic across every declared replica of it.

The Jetson AGX Orin hosts no `cortex`. Its gateway answered every
`model=cortex` request by forwarding to the **singular**
`PRIMARY_PEER_ORIGIN`, pinning all traffic to one of two equally-compatible
peers while the Spark and Thor each published two ready cortex replicas of
each other. This run made a dropped role's request *placeable* across every
declared replica, and then measured whether that actually helped.

**Execution note:** this plan was **not** fanned out via
`/assign-to-workforce`. The nine tasks were executed serially by the main
agent in one session, in wave order, with the same TDD discipline applied at
each step (full suite + lint + rubric gate green before each commit). The
plan's waves were used as an ordering contract, not as a parallelism
schedule. Recording this because the drift and decisions below were made by
one agent in-session rather than surfaced by a merge gate.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Capture the pre-change baseline transcript on the physical Orin
- `t2` — Reference fingerprint for a lane-less pool in `_replicas.py`
- `t3` — Fold the capabilities ready/context relay across a replica set in roles.py
- `t4` — Peer-only cache construction, arming gate and credentials in server.py
- `t5` — Peer-only placement precedence, fallback and guards in `handle_post`
- `t6` — List a pooled non-hosted role in GET /v1/models
- `t7` — Byte-identity and generic-prefix regression suite
- `t8` — Document the peer-only pool and bump the version
- `t9` — Live acceptance run on the physical Orin against the Spark and Thor

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `docs/evidence/2026-08-30-baseline-orin-cortex-pinned.txt` — 4 concurrent requests, 4/4 pinned to the Spark at 88.1 tok/s, filed before any code change |
| `t2` | delivered | `_apply_reference` + `REFERENCE_NOTE` / `NO_REFERENCE_REASON` in `lobes/gateway/_replicas.py`; `_probe_peer` gained `defer_compatibility` |
| `t3` | delivered | `_pooled_peer_advert` in `server.py` + the pooled-role gate in `lobes/roles.py::_role_signals`; `ready` folds across replicas, `context` is the agreed window |
| `t4` | delivered | `build_replica_caches` builds a lane-less cache; `_check_pool_arming`; singular-key inheritance in `_replica_api_key`; guarded pre-bind refresh |
| `t5` | delivered | `_peer_only_forward` placed ahead of `_proxied_owner`, with fall-through, hop guard and pressure bypass |
| `t6` | delivered | `pooled_backends` predicate shared by placement and the models list; `pooled` param on `list_models_payload` |
| `t7` | delivered | `tests/test_gateway_peer_only_pool.py` — 40 tests; `_config.py` and `doctor.py` untouched (verifiable by diff) |
| `t8` | delivered | `docs/gateway-fleet.md`, `docs/deployment-shapes.md`, `CLAUDE.md`; version 0.69.2 → 0.70.0 + CHANGELOG |
| `t9` | delivered | `docs/evidence/2026-08-30-accept-peer-only-pool-orin.txt` — 3 of 4 measurable targets met, 1 explicitly NOT met |

All nine accounted for. No task dropped, blocked, or partial.

## Mid-work Decisions

No `devague deviate` records were created for this run — the deviations below
were made and reviewed in-session with the user rather than through
`/deviate`. Recording them here directly, which is the method's fallback for
decisions no record covers.

- **The arming gate was scoped to dropped roles only.** As first written,
  `_check_pool_arming` required a singular `<PREFIX>_PEER_ORIGIN` for *every*
  pooled backend, which refused every pre-existing #199 pool — a box that
  *hosts* the role it pools publishes no referral, so it has no `hosted_by`
  to protect. Caught by `test_build_replica_caches_seeds_the_local_lane_from_declared_capacity`.
- **Both new gates were placed in `server.py`, not `_config.py`.** Claim `c8`
  promised `_config.py` stays untouched, but the routing table and
  `_replica_api_keys` are built there (`_config.py:1326`). Putting the arming
  gate and key inheritance in `build_replica_caches` / `_replica_api_key`
  keeps the boundary without weakening either gate.
- **`_peer_only_forward` also requires the singular origin at request time**,
  not only at startup. The startup gate makes the state unreachable in
  production, but a hand-built table can still reach it; degrading to the
  pre-change path is safer than placing a request whose fall-through and
  `hosted_by` are both undefined. This preserved an existing test unchanged.
- **The `pass_start` clock read is scoped to the peer-only path.** Reading it
  unconditionally consumed a tick from an injected clock and shifted every
  timestamp a hosted cache records — caught by
  `test_last_seen_is_carried_forward_across_a_failing_cycle`.
- **The live `t9` run was conducted with both replica key slots empty**
  (`PRIMARY_PEER_API_KEYS=,`) rather than with the Thor's key filled in, so the
  run would exercise the inheritance path against a Spark that genuinely 401s
  unauthenticated callers. Confirmed the Thor runs no inbound gate at all.
- **The peer-down target was first measured by simulation**, then re-measured
  for real once the user set up ssh to the Thor. Both runs are kept in the
  transcript; the simulated one proves something the real one does not (the
  reference falling through to the second-declared peer).

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t4` | The arming gate as specified would have refused every existing #199 hosted pool; scoped to `table.infeasible` backends only. The spec's intent (protect `hosted_by`) is preserved — `hosted_by` only exists for dropped roles. | acceptable |
| `t4` | Claim `c8`'s "`_config.py` untouched" boundary held, but only because both gates moved into `server.py`. The plan's instruction had already anticipated this; recording it because a reader comparing spec text to diff will notice the gate is not where `c19`'s wording implies. | acceptable |
| `t5` | Two defects were found **after** the task's tests passed, both by the live run, both requiring code changes to `t5`'s surface: the missing dispatch counter and the stale-fingerprint compatibility resurrection. The task's acceptance criteria did not cover concurrency behaviour or probe-failure state. | needs-follow-up |
| `t9` | Success signal `c23`'s throughput target (≥ +40% over baseline) is **NOT MET** — measured 42.8 tok/s at N=4 vs an 88.1 baseline, and 85.9 vs 89.0 at N=8. Cause is a gap in #199's capacity model, not in this work. Filed as #232. | needs-follow-up |
| `t9` | The continuity target says "with one peer's gateway stopped". It was first satisfied by simulation (an unreachable declared origin) because no ssh access existed, then re-run as written. The transcript keeps both and labels which is which. | acceptable |

## Evidence

- tests: `tests/test_gateway_peer_only_pool.py` — 40 passed (whole-file node id)
- tests: `tests/test_no_config_byte_identity.py`, `tests/test_gateway_pool_pressure.py`, `tests/test_gateway_replicas.py` — 290 passed together with the above
- tests: full suite `uv run pytest -n auto tests/` — 4359 passed, 15 skipped
- lint: `uv run black --check`, `isort --check-only`, `flake8`, `bandit` — clean
- rubric: `uv run afi cli doctor . --strict` — exit 0
- commits: `4c027b1..4bc0f1b` (3 commits, 19 files, +3275/−30 vs `main`)
- evidence transcripts: `docs/evidence/2026-08-30-baseline-orin-cortex-pinned.txt`, `docs/evidence/2026-08-30-accept-peer-only-pool-orin.txt`
- spec / plan: `docs/specs/2026-08-30-peer-only-replica-pools.md`, `docs/plans/2026-08-30-peer-only-replica-pools.md`
- issues: #232 (capacity expresses concurrency, not service rate)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| A dropped role with declared plural origins is placed across its replicas, not pinned to one | high | live: 2/4 and 2/4 split at N=4, 6/2 at N=8, vs a 4/0 baseline — `docs/evidence/2026-08-30-accept-peer-only-pool-orin.txt` §2 |
| Compatibility for a lane-less pool is decided by peers agreeing with each other, first READY peer in declaration order | high | live: the Thor became `"fingerprint reference"` while an origin declared *ahead* of it was unreachable — transcript §5; tests `::test_first_ready_peer_in_declaration_order_is_the_reference`, `::test_reference_is_declaration_order_not_probe_order` |
| The singular API key is inherited by the replica that IS the singular peer | high | live: every Spark placement authenticated against a gateway that 401s unauthenticated callers — transcript §0; test `::test_the_singular_key_is_inherited_when_no_plural_slots_are_declared` |
| `ready`/`context` for a pooled dropped role reflect the replicas, not the catalog | high | live: `ready` false→true, `context` 1048576→262144 — transcript §1; commit `4c027b1` |
| Nothing selectable falls through to the singular forward; no singular origin leaves the 404 unchanged | high | live: 4 of 8 requests carried `route_reason=none` and were served by the singular path — transcript §2; test `::test_nothing_selectable_falls_through_to_the_singular_forward` |
| A pooled dropped role appears in `/v1/models` and disappears when replicas go unready | high | live: `['nvidia/...Lightning...', 'cortex']` — transcript §1; test `::test_the_listing_disappears_when_every_replica_goes_unready` |
| One peer's gateway stopped still serves 100% of requests | high | live: 4/4 answered 200 at 86.6 tok/s with `model-gear-gateway` stopped on the Thor — transcript §5b |
| Deployments with no `*_PEER_ORIGINS` are byte-identical | high | `tests/test_no_config_byte_identity.py` passes unchanged; full suite 4359 passed |
| The behaviour is generic across role prefixes, not cortex-specific | medium | test `::test_the_pool_is_generic_across_prefixes_not_special_cased_to_cortex` exercises the `multimodal` prefix — **no live run on a non-cortex role** |
| Pooling improves aggregate throughput | **unverified — DISPROVEN on this pair** | measured 42.8 vs 88.1 tok/s (N=4) and 85.9 vs 89.0 (N=8) — transcript §4. Not claimed; see #232 |
| Pooling helps a *homogeneous* replica pair | unverified | never tested — the only pair available differs 4.4x in speed |

## Remaining Work / Follow-up

- **#232 — capacity expresses concurrency, not service rate.** The blocking
  reason the throughput target failed. Two sub-gaps: no service-rate weight
  exists anywhere in the model, and `build_replica_caches` builds every
  `PeerReplica` with no weight, so a pooling box has no channel to declare a
  peer's worth. Until this is resolved, **no throughput benefit may be
  claimed for a heterogeneous pool** — the docs and `CLAUDE.md` say so
  explicitly.
- **Homogeneous-pair validation is untested.** The claim that pooling helps
  when replicas are matched in speed is plausible and unmeasured. #199's own
  +74% result is not evidence for it — that run was bounded by the local box
  being busy, not by replica speed.
- **The generic-prefix claim is unit-only.** No non-cortex role has been
  pooled live. Owner decision whether that needs a live run before the
  behaviour is described as validated for other roles.
- **`t5`'s acceptance criteria under-specified concurrency.** Both live
  defects (dispatch counting, stale-fingerprint compatibility) were in
  behaviour its criteria did not cover. Worth a habit rather than a ticket:
  a task that changes dispatch should carry a concurrency criterion.
- **Deployment state on the Orin.** The pool is left **armed**
  (`PRIMARY_PEER_ORIGINS` naming both peers). Given the measured throughput
  cost, disabling it is one commented line — an open operator decision, not a
  defect. The gateway runs a **local-wheel image** (0.70.0, unpublished);
  `Dockerfile.gateway.orig` and `Dockerfile.gateway.bak-0.63.1` are both kept
  beside it for restore.
- **`deployment.lock.toml` does not capture any of this.** Every `*_PEER_*`
  key is deliberately excluded from the lock, so a `--from-lock` restore
  reproduces this box with no pool. Pre-existing (recorded as scope finding
  s16), unchanged by this work, and now more load-bearing.
