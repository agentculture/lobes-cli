# Build Plan — capacity-relative pool routing

slug: `capacity-relative-pool-routing` · status: `exported` · from frame: `capacity-relative-pool-routing`

> The cortex replica pool routes by capacity-relative load: every box publishes its own max active requests, and a request goes to whichever replica is least full — not to whichever box a host-level iowait reading happens to like.

## Tasks

### t1 — Capacity + kill-switch env knobs in lobes/gateway/`_config.py`

- instruction: touch only lobes/gateway/`_config.py` and tests/`test_gateway_config_replicas.py`; follow the existing `PEER_ORIGINS_ENV` nine-prefix pattern rather than inventing a new shape
- covers: c25, h18, c14, h8
- acceptance:
  - a box's own declared capacity parses from env into config; absent, it resolves to the neutral default and no error is raised
  - the kill-switch knob, when truthy, forces the resolved capacity to the 1.0 sentinel for every replica local and peer alike
  - a single-box deployment with no peers and no capacity keys parses exactly as it does today - no new required key
  - tests/`test_gateway_config_replicas.py` covers set / unset / kill-switch-engaged

### t2 — Calibration knee measurement in lobes/assess.py: throughput plateau + TTFT guard

- instruction: keep the knee function pure over a list of samples - no HTTP, no clock - so it unit-tests without a live engine, mirroring how lobes/gateway/`_selection.py` is pure over a replica snapshot
- acceptance:
  - given a ramp of (concurrency, `aggregate_tok_s`, ttft) samples the knee is the highest level whose throughput still rises meaningfully AND whose TTFT is under the declared bound
  - a ramp that never plateaus returns the top level tried, and is reported as un-plateaued rather than silently treated as the knee
  - a ramp where TTFT crosses the bound before any plateau returns the last level under the bound
  - the knee function is pure over a sample list - no HTTP, no clock - and is unit-tested without a live engine

### t3 — Capacity-relative selectability + reason vocabulary in lobes/gateway/`_selection.py`

- instruction: do not change `estimated_wait`()'s arithmetic - it already computes capacity-relative utilisation; change only selectability, the fallback constant, and the reason vocabulary
- covers: c2, h2, c5, h4, c16, h10, c22, h15, c24, h17
- acceptance:
  - `_is_selectable` no longer excludes a candidate for a pressure-derived busy flag alone; a candidate is unselectable when its active count reaches its capacity
  - a peer under pressure whose engine is genuinely full still ranks last - decoupling does not make a saturated box attractive
  - an uncalibrated peer uses a neutral fallback, not the 1.0 sentinel: with one calibrated weight-8 peer and one uncalibrated peer each at one active request, the uncalibrated peer is not ranked 8x worse
  - `select_replica` stays a pure function - same inputs, same output - and existing tests/`test_gateway_selection.py` cases pass unchanged for any fixed weight
  - a regression test captures the `before_state`: an idle peer carrying a pressure-busy flag is unselectable under the OLD gate and selectable under the new one
  - the reason vocabulary change is explicit - either a new constant or a documented redefinition of peer-less-loaded - and the closed set in the module docstring is updated to match

### t4 — Capacity ingest, clamp, fingerprint-keying and in-flight counter in lobes/gateway/`_replicas.py`

- instruction: the fattest task and alone in its wave: land ingest+clamp first, then fingerprint-keying, then the in-flight counter, so a partial merge is still coherent
- depends on: t1, t3
- covers: c3, h3, c19, h13, c23, h16
- acceptance:
  - a peer's published capacity is read on the existing /status probe and populates ReplicaState.weight; the local seed populates from this box's own declared capacity
  - a received capacity is clamped to a configured maximum on ingest, and the clamp is recorded in the replica row's reason field rather than applied silently
  - a peer publishing no capacity falls back to the neutral default and stays routable - an unpublished capacity never makes a replica unselectable
  - a stored capacity is discarded when the live fingerprint (`served_id` + quantization + `max_model_len` + runtime) no longer matches the one it was measured under
  - the module exposes an in-flight counter that callers increment at dispatch and decrement at completion, and `estimated_wait` consumes probed load PLUS local in-flight
  - tests/`test_gateway_replicas.py` covers ingest, clamp, missing-capacity fallback, and fingerprint mismatch discard

### t5 — Publish own capacity in /status and wire dispatch in-flight accounting in lobes/gateway/server.py

- instruction: increment before dispatch and decrement in a finally-equivalent path so an error or retry cannot leak a counter; a leaked in-flight count makes a healthy box look permanently full
- depends on: t4
- covers: c20, h14, c15, h9
- acceptance:
  - `fleet_status_payload` carries this box's declared capacity for each pooled role, so a peer learns it from the probe it already makes
  - the in-flight counter is incremented before a replica is dispatched to and decremented on completion, including on the error and retry paths
  - N concurrent arrivals against two idle replicas distribute across both rather than all landing on one, without waiting for a probe refresh
  - the capacity and utilisation used by a routing decision are observable after the fact, not only inferable from gateway logs

### t6 — Surface resolved capacity in the /capabilities replica row in lobes/roles.py

- instruction: keep the offline replica view honest - a not-probed row reports no capacity rather than guessing one, matching how it already reports None for every live field
- depends on: t4
- acceptance:
  - the hardcoded 1.0 at roles.py:1152 and :1167 is replaced by the resolved capacity, keeping the neutral default as the fallback
  - the replica row carries the capacity used alongside the existing weight field so a capacity-driven choice is explainable from /capabilities alone
  - the offline (not-probed) replica view still reports honest values rather than guessing a capacity
  - tests/`test_roles_replicas.py` and tests/`test_cli_capabilities_replicas.py` cover the populated and fallback cases

### t7 — Regression guards: untouched sampler, pool-pressure suite, and no-config byte-identity

- instruction: run the pool-pressure suite before and after; treat any required test change as a contract renegotiation to be named in the PR, not a test to quietly update
- depends on: t5, t6
- covers: c6, h5, c7, h6, h1
- acceptance:
  - git shows lobes/runtime/`_pressure.py` unchanged by this work and tests/`test_pressure.py` passes untouched
  - every test in tests/`test_gateway_pool_pressure.py` passes unchanged, or the PR carries an explicit note naming which behaviour was renegotiated and why
  - with no capacity published and no \*`_PEER_ORIGINS` declared, responses are byte-identical to the pre-pool contract, headers included
  - hand remains the servable floor under pressure and is never forwarded

### t8 — Document the capacity knobs and the calibrate verb in templates and docs

- instruction: state plainly in env.example that capacity is NOT the --max-num-seqs OOM cap - conflating the two is the specific confusion this spec exists to prevent
- depends on: t1, t5, t6
- acceptance:
  - lobes/templates/fleet/env.example documents the capacity and kill-switch knobs, and states plainly that capacity is NOT the --max-num-seqs OOM cap
  - docs/gateway-fleet.md's replica-pool section describes capacity-relative routing, the self-published discovery path, and the clamp
  - the X-Lobes-Route-Reason vocabulary change is stated where the closed set is currently documented, so a caller parsing the old set is told what changed
  - uv run afi cli doctor . --strict passes

### t9 — lobes calibrate verb wrapping the knee measurement

- instruction: follow the repo's dry-run-by-default convention: reporting is read-only, writing a measured capacity to .env requires --apply
- depends on: t2, t1
- acceptance:
  - the verb ramps concurrency against a named role and reports the measured knee, the samples behind it, and whether the ramp plateaued
  - it is read-only by default and follows the repo's dry-run/--apply convention if it writes the measured capacity to .env
  - it refuses to write a capacity when the ramp never plateaued, rather than recording the top level tried as if it were a knee
  - no calibration logic runs inside the gateway request path - the verb lives in lobes/cli/`_commands`/ and the gateway only consumes a number

### t10 — Live acceptance on the Spark+Thor cortex pair with the iowait artifact still present

- instruction: run against the live Spark+Thor pair with the ghostty iowait artifact still elevated - it is the fixture, not a problem to clear first; state validated scope as cortex-on-this-pair only
- depends on: t7, t8, t9
- covers: c1, c10, h7, c17, h11, c18, h12
- acceptance:
  - the Spark rejoins the pool as a selectable candidate while its host iowait still reads above the 50 percent threshold - the artifact is the fixture and is NOT cleared first
  - an 8-way concurrent flood of model=cortex against one gateway with both boxes idle reaches at least 18.7 tok/s aggregate, versus the recorded 11.0 tok/s single-owner baseline on the same flood shape
  - concurrent requests visibly distribute across both replicas rather than queueing on one, evidenced by the per-answer Served-By and Route-Reason markers
  - an acceptance transcript lands under docs/evidence/ recording both numbers, the capacity each box published, and the calibration runs behind them
  - scope is stated honestly: validated for cortex on the Spark+Thor NVFP4 pair only, every other pooled role declared-but-unvalidated
  - the single-owner baseline is re-measured on the same 8-way flood shape before the comparison is drawn - the recorded 11.0 tok/s is not trusted as authoritative, since it was captured under what looks like the same iowait artifact

## Risks

- [unknown_nonblocking] the 2026-08-25 acceptance evidence may be contaminated - its 'organic iowait pressure' premise is probably this same artifact, so the 11.0 tok/s baseline that t10's 1.7x target leans on should be re-verified before it is treated as authoritative (task t10)
- [unknown_nonblocking] whether capacity is a single scalar at all: a box may have different effective capacity for short agentic turns than for long-context requests, which one calibrated number cannot express (frame park v2)
- [unknown_nonblocking] calibration cost and cadence are unspecified - a ramp per box per checkpoint is not free, and nothing yet says who re-runs it after a lobes switch (task t9)
