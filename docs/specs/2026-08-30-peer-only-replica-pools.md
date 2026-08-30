# peer-only replica pools

> A box that hosts none of a role still spreads that role's traffic across every declared replica of it.
> instruction: Ship as a gateway-only change: lobes/gateway/{server,`_replicas`}.py plus tests and docs; no new env knobs, no template or profile changes.

## Audience

- Operators of a mesh box that hosts none of a heavy role but declares peers for it — the Jetson AGX Orin today; any future gear-only or edge box tomorrow — plus every caller that addresses that box's gateway by role name.
  - instruction: Exercise the peer-only path in tests on a non-cortex prefix (e.g. senses/multimodal) to prove it is generic.

## Before → After

- Before: Today the Orin's gateway forwards every cortex request to one declared origin (measured 2026-08-30: 200 with X-Lobes-Proxied-By = the Spark, always), even though the Spark and Thor each publish two compatible ready cortex replicas — so the Orin's traffic cannot use the pool that already exists.
  - instruction: Before merging, file the 2026-08-30 Orin baseline probe (200 + X-Lobes-Proxied-By pinned to the Spark on every request; both peers publishing two compatible ready cortex replicas) as docs/evidence/2026-08-30-baseline-orin-cortex-pinned.txt.
- After: A box with no local replica of a role places each request across all declared <PREFIX>`_PEER_ORIGINS` by live-probed load, stamping X-Lobes-Proxied-By and X-Lobes-Route-Reason, instead of pinning every request to the singular <PREFIX>`_PEER_ORIGIN`.
  - instruction: Stamp X-Lobes-Proxied-By with the chosen peer origin and X-Lobes-Route-Reason from the existing vocabulary on every peer-only placed answer; add no new reason string.

## Why it matters

- Capacity the mesh already has is unreachable from the boxes most likely to need it: a gear-only box is exactly the one that hosts no heavy lobe, and pinning its traffic to one peer wastes half the fleet's cortex capacity and makes that one peer a single point of failure for it.
  - instruction: Report aggregate tok/s for N concurrent Orin-originated cortex requests before and after, in the same shape as docs/evidence/2026-08-25-baseline-cortex-single-owner.txt.

## Requirements

- `build_replica_caches` skips every backend in table.infeasible (lobes/gateway/server.py:3752, 'a dropped lane has no local replica to probe or publish'), so a non-hosted role gets no ReplicaCache at all today; that skip is the change point — a role with declared <PREFIX>`_PEER_ORIGINS` must get a peer-only cache.
  - instruction: In `build_replica_caches` (lobes/gateway/server.py:3752) replace the unconditional infeasible skip with: skip only when the backend has no declared replica origins; otherwise build a cache with local=None and the declared peers.
  - honesty: With no \*`_PEER_ORIGINS` declared for an infeasible backend, `build_replica_caches` still returns no cache for it — proven by a test asserting the pre-change cache map for a referral-only and a proxy deployment.
- A peer-only pool needs a compatibility REFERENCE that is not the local lane: `compare_fingerprints`(local=None, peer) returns (False, 'fingerprint: unknown (local replica not probed)') at lobes/gateway/`_replicas.py`:598, so with no local lane every peer is marked incompatible and selection would pick nothing.
  - instruction: Add a reference-fingerprint path to ReplicaCache.`_refresh_peers`: when local is None, probe peers first, take the first READY peer in declaration order as the reference, then run `compare_fingerprints`(reference, peer) for the rest. Publish which peer supplied the reference.
  - honesty: A peer whose fingerprint disagrees with the chosen reference is excluded with a reason naming the differing field, and a pool whose peers do not agree with each other selects nothing rather than guessing — no unknown ever pools silently (#199 h11 restated for the lane-less case).
- Precedence must change: `handle_post` calls `_proxied_owner` and forwards to the SINGULAR <PREFIX>`_PEER_ORIGIN` (lobes/gateway/server.py:2456) before the feasibility 404 and before `_pool_selection` (server.py:1550), so a pooled non-hosted role has to be placed ahead of the singular proxy branch or the one declared origin always wins.
  - instruction: In `handle_post`, run peer-only placement BEFORE the `_proxied_owner` branch (server.py:2456) and gate it on 'this backend is pooled': plural origins declared and at least one compatible ready replica. On no selectable replica, fall through to the existing singular-proxy forward unchanged.
  - honesty: A pooled non-hosted role never produces more than one forward per request: placement happens once, the chosen peer is stamped X-Lobes-Proxied, and a pre-dispatch failure retries at most once per remaining replica (the #199 t8 semantics) — verified by a test that counts upstream opens.
- Credential blind spot: the plural channel is positional and INDEPENDENT of the singular one — `_replica_api_keys` (lobes/gateway/`_config.py`:473) returns no entry when <PREFIX>`_PEER_API_KEYS` is unset, so `_replica_api_key` yields "" and the forward carries no Authorization. On the exact target deployment (the Orin declares only the singular `PRIMARY_PEER_API_KEY`, and the Spark runs an inbound gate) every pooled forward would 401. The change must either inherit the singular key for an origin equal to <PREFIX>`_PEER_ORIGIN`, or refuse to arm the pool with a named config error — never silently 401.
  - instruction: Decide the inherit-vs-refuse behaviour explicitly and test both: origin == `PEER_ORIGIN` inherits `PEER_API_KEY`; an origin with neither a key slot nor a singular match is a named ReplicaConfigError at startup.
  - honesty: A pooled deployment whose plural origins have no matching key slot is caught at startup or by an inherited key — proven by a test that arms a pool with only the singular `PEER_API_KEY` set and asserts the forward carries the same Authorization the singular proxy path would have sent.
- A pooled non-hosted role's advertised ready must be the OR across its compatible replicas, and its context the fingerprint-agreed `max_model_len`: `probe_peer_ready` (lobes/gateway/`_readiness.py`:212) and the #220 relay are built for ONE peer origin, so with N replicas the capabilities surface has no defined answer today.
  - instruction: Extend the peer-ready/peer-context relay to fold across the replica set: ready = any(compatible and ready), context = the fingerprint-agreed `max_model_len`, absent when the set is empty.
  - honesty: With two ready replicas the role advertises ready:true and the agreed context; with every replica unready it advertises ready:false and never fabricates a context from the catalog (the #220 lesson: a role this box does not host has no <PREFIX>`_MAX_MODEL_LEN` to read).
- A down or slow peer must never delay or fail the gateway's bind: `build_replica_caches` refreshes every cache SYNCHRONOUSLY before serve() binds (lobes/gateway/server.py:3775), so adding peer-only caches puts cross-box probes of roles this box does not host on the boot path — the peer probe timeout must bound it and an unreachable peer must leave the gateway serving every other role normally.
  - instruction: Keep the synchronous pre-bind refresh but bound it by the existing peer probe timeout; never let a peer-only cache raise out of `build_replica_caches`.
  - honesty: A test constructs a peer-only cache whose every peer refuses the connection and asserts serve() still binds within the probe timeout and that the role reports ready:false rather than the gateway failing to start.

## Honesty conditions

- The claim is literal: after this ships, a box hosting NO replica of a role still distributes that role's requests across every declared replica — demonstrated by a live run where one box's requests are answered by two different peers, not by a config that merely permits it.
- A request arriving at this box already marked X-Lobes-Proxied is never placed across peers: with no local replica it still answers 508 `proxy_loop`, exactly as the pre-change `_proxied_owner` path does.
- No change lands in lobes/gateway/`_config.py`'s origin/key parsing or in doctor's pass-through suffix tables — proven by those files being untouched in the diff, with the existing config and doctor tests passing unchanged.
- The existing no-config byte-identity suite (tests/`test_no_config_byte_identity.py`) passes unchanged, and a referral-only deployment's 404 `role_infeasible` body plus a singular-proxy deployment's forward are byte-identical before and after.
- The behaviour is generic across all ten role prefixes, not special-cased to cortex or to the Orin: the peer-only path keys off `PEER_ORIGINS_ENV`'s existing mapping, and a test exercises it on a second, non-cortex prefix.
- The placement is observable: every answer carries X-Lobes-Proxied-By naming the serving peer and X-Lobes-Route-Reason from the existing closed vocabulary — no new reason string is invented for the peer-only case (peer-less-loaded already covers it, `_selection.py` l.63-66).
- The before-state is measured, not asserted: the 2026-08-30 probe on the physical Orin (200, X-Lobes-Proxied-By = the Spark on every request, both peers publishing two compatible ready cortex replicas) is filed as the baseline transcript under docs/evidence/ before the change lands.
- The capacity claim is quantified against the baseline rather than assumed: the acceptance run reports aggregate throughput for concurrent Orin-originated cortex requests before and after, in the same shape as the #199 single-owner baseline.
- The acceptance transcript is a LIVE cross-box run on the physical Orin against the Spark and Thor, filed under docs/evidence/ per the #108 rule — declared-only data may not be described as validated anywhere in docs, CLAUDE.md, or lobes capabilities.
- The measurable targets in the new `success_signal` claim are reported as raw numbers in the transcript, not as a summary judgement.
- The reference is honest about its own provenance: the published replicas array names which peer supplied the reference, and a peer excluded for disagreeing carries the same field-naming reason string a local-lane mismatch produces today (`compare_fingerprints`, `_replicas.py`:587).
- A pooled role's /v1/models entry is proven to appear ONLY with plural origins and a compatible ready replica: tests assert the payload is unchanged for referral-only and singular-proxy deployments, and that the entry disappears again when every peer goes unready.
- A referral-only and a singular-proxy deployment's GET /v1/models payloads are asserted byte-identical before and after, and the pooled listing is driven by the same predicate the placement path uses — a single source of truth, not two independent conditions that can drift.
- A test drives a peer-only pooled box under simulated swap/iowait pressure and asserts the pooled role is still forwarded, never shed 429.
- Every number in the target is reported as a measured value in the acceptance transcript — per-peer split, aggregate tok/s against the recorded baseline, and the one-peer-down result — or the claim is reported as NOT met; no target is quietly softened after the run.
- Both halves are tested: a pool whose peers are all unready forwards to the singular origin with the same bytes the pre-change path produced, and `PEER_ORIGINS` without `PEER_ORIGIN` raises a named startup error rather than arming a pool with no referral.

## Success signals

- A live cross-box transcript from the Orin under load: concurrent cortex requests answered by BOTH the Spark and the Thor (distinct X-Lobes-Proxied-By values with X-Lobes-Route-Reason), a peer taken down still served by the other, and a deployment with no \*`_PEER_ORIGINS` byte-identical to today.
  - instruction: Acceptance run on the physical Orin: at least 4 concurrent cortex requests must yield BOTH peer origins in X-Lobes-Proxied-By, aggregate throughput must exceed the pinned-to-one-peer baseline, and stopping one peer's gateway must still serve 200 from the other.
- Measurable target for the acceptance run on the physical Orin: at least 4 concurrent cortex requests produce BOTH declared peer origins in X-Lobes-Proxied-By (neither peer serving 0 and neither serving 100%), aggregate tok/s is at least 40% above the pinned-to-one-peer baseline, and with one peer's gateway stopped 100% of requests still answer 200 from the other.
  - instruction: Report the per-peer request split, aggregate tok/s vs baseline, and the one-peer-down result as raw numbers in the acceptance transcript.

## Scope / boundaries

- The single-hop guard is untouched: `_proxy_to_peer` stamps X-Lobes-Proxied (lobes/gateway/server.py:1253) and a marked arrival skips selection and is served locally (`_pool_selection`, server.py:1578). This work adds fan-out at the ORIGINATING box only — never a second hop, never a peer re-forwarding.
  - instruction: Assert in tests that an inbound X-Lobes-Proxied request to a peer-only pooled box still answers 508 `proxy_loop` and never enters placement.
- Config parsing and doctor are already generic and stay unchanged: `_replica_origins`/`_replica_api_keys` (lobes/gateway/`_config.py`:439/473) are keyed by backend name with no feasibility check, and doctor passes `PEER_ORIGINS`/`PEER_API_KEYS` through for every role prefix (lobes/cli/`_commands`/doctor.py:269).
  - instruction: Leave lobes/gateway/`_config.py` and lobes/cli/`_commands`/doctor.py untouched; verify by diff.
- The /v1/models change is scoped to POOLED roles only: a referral-only role (`hosted_by`, no proxy) and a singular-proxy role (`PEER_ORIGIN` + `PEER_PROXY`, no plural origins) keep today's payload byte for byte — listing is gated on the same 'is this role pooled' predicate the placement path uses, never on infeasible-plus-a-peer-origin.
  - instruction: Drive the /v1/models listing from the same 'is pooled' predicate the placement path uses; assert unchanged payloads for referral-only and singular-proxy deployments.
- Local pressure must not shed a peer-only pooled request: `handle_post`'s proxy branch deliberately runs BEFORE pressure shedding (lobes/gateway/server.py:2417 — 'pressure describes THIS box's load; the model runs on the peer'), and placing the pool ahead of that branch must preserve the bypass, not re-gate a role this box cannot serve on its own swap/iowait.
  - instruction: Place the peer-only branch ahead of `_proxied_owner` but on the same side of the pressure gate as the existing proxy branch.

## Non-goals

- Pooling does not make a box a host: feasible stays false for a non-hosted role (lobes/roles.py:1104) and `hosted_by`/proxied/ready remain the honest usability fields — no role is promoted to feasible by having replicas.
- No behaviour change for any deployment that declares no \*`_PEER_ORIGINS` (h1 of #199): with no plural origins the peer-only path must not construct a cache, place a request, or alter one byte of the referral 404 or the singular proxy forward.

## Assumptions

- The substrate already supports a lane-less pool: ReplicaCache.`__init__` takes local: LocalLane | None (lobes/gateway/`_replicas.py`:686), seeds no local state when it is None, and `_refresh_local` returns early on a None lane (line 1247) — so no new local-optional plumbing is needed inside the cache.
- `select_replica` already contemplates a peer-only candidate set: lobes/gateway/`_selection.py` lines 63-66 document 'no local candidate in the input at all, i.e. a peer-only view' returning reason peer-less-loaded — so ranking needs no new reason vocabulary.

## Scope exploration

- `s1` — `lobes/gateway/server.py — build_replica_caches (l.3711-3776)`: skips 'if backend.name in table.infeasible: continue' so a dropped lane gets no ReplicaCache; the LocalLane is built unconditionally for every cache it does make
  - seeds: `c2`
- `s2` — `lobes/gateway/_replicas.py — compare_fingerprints (l.587) and _probe_peer (l.1183)`: peer compatibility is computed against `local_fp`; a None local fingerprint yields (False, 'fingerprint: unknown (local replica not probed)') and `_peer_backend_entry` also keys off the local served id first, falling back to backend/role name
  - seeds: `c3`
- `s3` — `lobes/gateway/server.py — handle_post precedence (l.2455-2470, 1300-1345, 1550-1590)`: order is proxy branch -> `_resolve_served_or_early` (feasibility 404) -> `_pool_selection`; the pool sits after the owning backend resolves and is unreachable for an infeasible role
  - seeds: `c4`
- `s4` — `lobes/gateway/_replicas.py — ReplicaCache.__init__ (l.685) and _refresh_local (l.1247)`: local is already Optional: no local state is seeded when None and the local refresh pass returns early, so a peer-only cache is representable without touching the constructor
  - seeds: `c5`
- `s5` — `lobes/gateway/_selection.py — policy docstring (l.36-70) and _reason_for (l.348)`: a peer-only candidate set is already a documented input, returning peer-less-loaded; `local_busy` and the local carve-outs simply never fire
  - seeds: `c6`
- `s6` — `lobes/gateway/server.py — _proxy_to_peer (l.1215-1260) and _pool_selection hop guard (l.1578)`: every forward stamps X-Lobes-Proxied and a marked arrival is pinned to the receiver (sole-ready) — measured live on this box: an Orin forward to the Spark is served by the Spark and never re-routed to the Thor
  - seeds: `c7`
- `s7` — `lobes/gateway/_config.py (l.330-500) and lobes/cli/_commands/doctor.py (l.262-300)`: plural origins/keys parse per backend name with no feasibility gate and raise ReplicaConfigError on a length mismatch; doctor already asserts `PEER_ORIGINS`/`PEER_API_KEYS` reach the gateway container for every role prefix
  - seeds: `c8`
- `s8` — `lobes/roles.py — annotate_peer_referrals (l.1049-1130) and annotate_replicas (l.1252)`: feasible stays false for proxied roles by design, proxied/`hosted_by`/ready carry usability, and the replicas/fingerprint keys are additive per role — a peer-only pool has an existing publication surface
  - seeds: `c9`
- `s9` — `live probe: Orin/Spark/Thor gateways, 2026-08-30`: Spark and Thor each publish two compatible ready cortex replicas (each other plus local, `served_id` unsloth/Qwen3.8-27B-NVFP4, `max_model_len` 262144, runtime vllm); the Orin publishes cortex feasible:false and, with `PRIMARY_PEER_PROXY`=true, returns 200 with X-Lobes-Proxied-By pinned to the Spark
  - seeds: `c1`
- `s10` — `lobes/gateway/_routing.py — RoutingTable.infeasible/replica_origins (l.58-141), order_backends (l.504)`: infeasible backends are filtered out of ordering and the models list while still present in table.backends, which is why `build_replica_caches` can see and skip them; open question q2 asks whether a pooled non-hosted role should re-enter /v1/models
- `s11` — `challenge pass / security lens: lobes/gateway/_config.py:473 (_replica_api_keys), server.py:1529 (_replica_api_key)`: the plural key channel is positional and independent of the singular one; an unset `PEER_API_KEYS` yields no entry and an empty outbound credential, which 401s against a peer that runs an inbound gate — exactly the Orin/Spark pairing this feature targets
  - seeds: `c19`
- `s12` — `challenge pass / adjacent-systems lens: lobes/gateway/_readiness.py:212 (probe_peer_ready), lobes/roles.py:794-836 (peer_ready/peer_context relay)`: both the readiness probe and the #220 relay are single-origin by construction; with N replicas neither ready nor context has a defined answer
  - seeds: `c20`
- `s13` — `challenge pass / operations lens: lobes/gateway/server.py:2417 (proxy-before-pressure ordering)`: the proxy branch's pressure bypass is deliberate and documented; moving the pool ahead of it risks re-gating a non-hosted role on this box's own swap/iowait
  - seeds: `c21`
- `s14` — `challenge pass / lifecycle lens: lobes/gateway/server.py:3711-3776 (build_replica_caches synchronous pre-bind refresh)`: boot currently probes only roles this box hosts; peer-only caches put cross-box probes on the bind path
  - seeds: `c22`
- `s15` — `challenge pass / concurrency lens: lobes/gateway/_replicas.py in-flight accounting (begin_dispatch/_reconcile/INFLIGHT_MAX_AGE)`: per-gateway accounting reconciles against the peer's probed /status in both directions, so a single forwarder self-corrects; the residual hazard is N independent forwarders herding between refreshes — recorded as park v2, not fixed here
- `s16` — `challenge pass / reversibility lens: lobes/runtime/_lock.py:15-35 and MERGE_ONLY_FILES`: every \*`_PEER_`\* key is deliberately excluded from the deployment lock (operator-typed, 'peer origins are internal information by operator decision'), so a --from-lock restore reproduces a box with NO pool config — pre-existing, unchanged by this work, but this feature makes that omission more load-bearing
- `s17` — `challenge pass / observability lens: lobes/gateway/_config.py:587 (_self_origin), server.py:1653`: clean pass — `GATEWAY_SELF_ORIGIN` feeds only X-Lobes-Served-By on LOCALLY served answers, so a peer-only box needs no self origin declared; every pooled answer it emits is a forward carrying X-Lobes-Proxied-By instead
- `s18` — `challenge pass / failure-mode lens: server.py _pool_dispatch retry semantics (t8) vs _feasibility_response (l.1321)`: exhausting a pool yields 503 `backend_unavailable` with X-Lobes-Route-Attempts while today a non-hosted role yields a terminal 404 `role_infeasible` — a caller-visible contract change with no decided answer; raised as question q3

## Decisions

- Compatibility reference for a lane-less pool (q1): the first READY peer in declaration order supplies the reference fingerprint and every other declared peer is compared against it — peers agree with each other, no operator-declared expectation and no ungated trust. Disagreement excludes the peer with a reason naming the field; total disagreement (or no ready peer) means nothing is selectable and the request falls back to the existing singular-proxy forward.
- GET /v1/models lists a pooled non-hosted role (q2): a role with declared plural origins and at least one compatible ready replica re-enters the models payload that `order_backends` currently filters (lobes/gateway/`_routing.py`:504), so plain OpenAI clients that never read /capabilities can discover it.
- Empty-pool fallback (q3) and the singular-origin requirement (v3): a peer-only pool with no selectable replica takes the existing singular-proxy forward, and a deployment may not declare <PREFIX>`_PEER_ORIGINS` without <PREFIX>`_PEER_ORIGIN` — enforced as a named ReplicaConfigError at startup, so `hosted_by` (lobes/roles.py:1126) never silently vanishes.
  - instruction: Enforce the singular-origin requirement in `_replica_origins`' consumer at table-build time, reusing ReplicaConfigError; implement the fallback as the existing `_proxied_owner` branch running after an empty placement.

## Open parks

- [unknown_nonblocking] If the Orin's llama.cpp cortex lane ever comes back up as a local replica, it is fingerprint-incompatible with the NVFP4 vLLM peers by design — does a pool then hold one local-incompatible lane plus N compatible peers, and which one serves?
- [unknown_nonblocking] Two peer-only boxes pooling the SAME peers cannot see each other's in-flight dispatches — ReplicaCache's local accounting (lobes/gateway/`_replicas.py`, `begin_dispatch`/`_reconcile`) is per-gateway and reconciles only against the peer's own probed /status, so N forwarding boxes can herd onto the same replica between 5s refreshes.

## Resolved vagueness

- [unknown_blocking] A box declaring ONLY plural origins (no singular <PREFIX>`_PEER_ORIGIN`) has no `hosted_by` to publish — `annotate_peer_referrals` reads table.`peer_origins`, the singular channel (lobes/roles.py:1126) — so the honest-referral annotation would silently vanish for a plural-only deployment. — resolved: Plural origins are an ADDITION to the singular channel, never a replacement: arming a pool without <PREFIX>`_PEER_ORIGIN` is a named config error at startup, so `hosted_by` keeps reading the singular origin unchanged and composes with the q3 fallback.
