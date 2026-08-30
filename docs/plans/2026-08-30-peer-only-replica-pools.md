# Build Plan — peer-only replica pools

slug: `peer-only-replica-pools` · status: `exported` · from frame: `peer-only-replica-pools`

> A box that hosts none of a role still spreads that role's traffic across every declared replica of it.

## Tasks

### t1 — Capture the pre-change baseline transcript on the physical Orin

- instruction: Read-only probing of the live Orin/Spark/Thor gateways; no container is stopped or recreated.
- covers: c13, h13
- acceptance:
  - docs/evidence/2026-08-30-baseline-orin-cortex-pinned.txt records: every cortex request answered 200 with X-Lobes-Proxied-By pinned to a single peer, both peers' /capabilities showing two compatible ready cortex replicas, and an aggregate tok/s figure for N concurrent requests
  - the transcript is filed BEFORE any code change lands, per the #108 measured-not-declared rule

### t2 — Reference fingerprint for a lane-less pool in `_replicas.py`

- instruction: Add the reference path inside ReplicaCache.`_refresh_peers`; keep `_probe_peer`'s signature by passing the resolved reference as `local_fp`.
- covers: c3, h2
- acceptance:
  - ReplicaCache with local=None takes the first READY peer in declaration order as the reference and compares every other peer to it with `compare_fingerprints`
  - a peer that disagrees is marked incompatible with a reason naming the differing field; peers that disagree with each other leave nothing compatible rather than guessing
  - the published replicas array names which peer supplied the reference
  - no local-lane behaviour changes: existing ReplicaCache tests pass untouched

### t3 — Fold the capabilities ready/context relay across a replica set in roles.py

- instruction: Extend the `peer_ready`/`peer_context` channel in lobes/roles.py only; do not touch `annotate_peer_referrals`' `hosted_by`/proxied logic.
- covers: c20, h17
- acceptance:
  - a non-hosted pooled role advertises ready = any(compatible and ready) across its replicas
  - context is the fingerprint-agreed `max_model_len`, and is absent (never catalog-derived) when no replica is compatible
  - feasible stays false and `hosted_by`/proxied are unchanged for the same role

### t4 — Peer-only cache construction, arming gate and credentials in server.py

- instruction: All four behaviours land in `build_replica_caches` and `_replica_api_key` in server.py — lobes/gateway/`_config.py` stays untouched (boundary c8), reusing its ReplicaConfigError.
- covers: c2, h1, c19, h16, c22, h19
- acceptance:
  - `build_replica_caches` skips an infeasible backend only when it has no declared replica origins; otherwise it builds a cache with local=None and the declared peers
  - declaring <PREFIX>`_PEER_ORIGINS` without <PREFIX>`_PEER_ORIGIN` raises a named ReplicaConfigError at startup, so `hosted_by` never silently vanishes
  - a replica origin equal to <PREFIX>`_PEER_ORIGIN` inherits <PREFIX>`_PEER_API_KEY` when no positional key slot is declared; a forward therefore carries the same Authorization the singular proxy path would have sent
  - every peer refusing the connection still lets serve() bind within the peer probe timeout, with the role reporting ready:false — `build_replica_caches` never raises for an unreachable peer
  - with no \*`_PEER_ORIGINS` declared anywhere the returned cache map is byte-identical to today for referral-only and singular-proxy deployments

### t5 — Peer-only placement precedence, fallback and guards in `handle_post`

- instruction: Insert the peer-only branch ahead of `_proxied_owner` in `handle_post`, on the same side of the pressure gate as the existing proxy branch.
- depends on: t2, t4
- covers: c4, h3, c7, h4, c12, h6, c21, h18
- acceptance:
  - peer-only placement runs before the `_proxied_owner` branch, gated on 'this backend is pooled' (plural origins declared AND at least one compatible ready replica)
  - nothing selectable falls through to the existing singular-proxy forward; with no singular origin the 404 `role_infeasible` body is byte-identical to today
  - a placed answer carries X-Lobes-Proxied-By naming the chosen peer and X-Lobes-Route-Reason from the existing vocabulary — no new reason string
  - an inbound X-Lobes-Proxied request never enters placement and still answers 508 `proxy_loop`
  - at most one forward per request: a test counting upstream opens proves a pre-dispatch failure retries at most once per remaining replica and a 2xx-then-dropped response is never replayed
  - under simulated swap/iowait pressure a peer-only pooled request is still forwarded, never shed 429

### t6 — List a pooled non-hosted role in GET /v1/models

- instruction: Expose the pooled predicate as one helper and call it from both the placement path and the models-list filter.
- depends on: t5
- covers: c18, h15
- acceptance:
  - a role with declared plural origins and at least one compatible ready replica appears in /v1/models
  - the entry disappears again when every replica goes unready
  - referral-only and singular-proxy deployments' /v1/models payloads are asserted byte-identical before and after
  - the listing is driven by the SAME 'is pooled' predicate the placement path uses — one source of truth, asserted by a test that would fail if the two diverged

### t7 — Byte-identity and generic-prefix regression suite

- instruction: Tests only; the diff for this task must add no non-test lines.
- depends on: t5, t6
- covers: c8, h11, c11, h12
- acceptance:
  - tests/`test_no_config_byte_identity.py` passes unchanged
  - lobes/gateway/`_config.py` and lobes/cli/`_commands`/doctor.py are untouched, verified by diff in the PR
  - the peer-only path is exercised on a non-cortex prefix (e.g. multimodal/senses) proving it is generic across all ten role prefixes
  - a referral-only 404 `role_infeasible` body and a singular-proxy forward are byte-identical before and after

### t8 — Document the peer-only pool and bump the version

- instruction: Mark the capability DECLARED in CLAUDE.md until t9's transcript lands; bump with the version-bump skill.
- depends on: t7
- covers: c1, h10, c14, h14
- acceptance:
  - docs/gateway-fleet.md and docs/deployment-shapes.md describe the peer-only state, the reference-fingerprint rule, the singular-origin requirement and the empty-pool fallback
  - CLAUDE.md's replica-pool paragraph records the new capability and marks it DECLARED until the acceptance transcript lands (#108)
  - the version is bumped via .claude/skills/version-bump/scripts/bump.py and CHANGELOG.md carries an entry

### t9 — Live acceptance run on the physical Orin against the Spark and Thor

- instruction: Requires rebuilding the Orin gateway image and recreating model-gear-gateway — ask the operator before running.
- depends on: t1, t8
- covers: c15, h7, h20, c23, h22
- acceptance:
  - at least 4 concurrent cortex requests produce BOTH declared peer origins in X-Lobes-Proxied-By, neither peer serving 0 nor 100 percent
  - aggregate tok/s is at least 40 percent above the t1 baseline, reported as raw numbers
  - with one peer's gateway stopped, 100 percent of requests still answer 200 from the other
  - the transcript lands under docs/evidence/ and any target not met is reported as NOT met rather than softened

## Risks

- [unknown_nonblocking] The Orin declares only the singular `PRIMARY_PEER_API_KEY` today, so the new arming gate must land WITH the key-inheritance behaviour or the live acceptance run 401s against the Spark on its first request. (task t4)
- [unknown_nonblocking] t4 and t5 both edit lobes/gateway/server.py, so they are formally sequential at merge — t5 depends on t4 for that reason, not only for content. Building them in parallel would collide. (task t5)
- [follow_up] N independent forwarding boxes cannot see each other's in-flight dispatches (frame park v2), so herding between 5s refreshes is unaddressed by this plan and may only appear at a mesh scale larger than the current three boxes.
