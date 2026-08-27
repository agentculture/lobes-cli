# capacity-relative pool routing

> The cortex replica pool routes by capacity-relative load: every box publishes its own max active requests, and a request goes to whichever replica is least full — not to whichever box a host-level iowait reading happens to like.
> instruction: validate live on the Spark+Thor cortex pair, the only pool scope #199 validated, and land an acceptance transcript under docs/evidence/

## Audience

- fleet operators running lobes on more than one box, and the agents addressing model=cortex through any box's gateway
  - instruction: no new required env keys for a single-box deployment; capacity publication defaults on, capacity consumption defaults to the 1.0 fallback

## Before → After

- Before: every replica ranks as if it had a capacity of exactly one slot (weight hardcoded 1.0), and a peer's pool candidacy is gated on its host iowait/swap verdict - so an idle Spark with running=0 was excluded from the pool for a 60% iowait reading produced by a sleeping desktop terminal
  - instruction: capture the `before_state` as a regression test in tests/`test_gateway_selection.py`: idle peer + pressure-busy flag => currently unselectable
- After: a request addressed to a pooled role lands on whichever replica is least full relative to its own measured capacity; an idle box is never evicted from the pool by a host-level reading unrelated to serving
  - instruction: extend the /capabilities replica row (roles.py:1130) to carry the capacity used, alongside the existing weight field

## Requirements

- the selection arithmetic already computes capacity-relative utilization — `_selection.py`'s `estimated_wait` is (running + waiting) / weight — so this work populates weight with a real per-device capacity rather than changing the ranking policy
  - instruction: populate ReplicaState.weight from the probed peer capacity in `_replicas.py` (and the local capacity for the local seed); leave `estimated_wait`() and the ranking in `_selection.py` untouched
  - honesty: populating weight changes only the ranking INPUT: `select_replica` stays a pure function of its arguments, and tests/`test_gateway_selection.py` passes unchanged for any fixed weight
- weight is a designed-in hook that no env knob populates: it defaults to 1.0 in `_replicas.py` (fields at 214/241/258) and is hardcoded 1.0 in roles.py:1152,1167, so every replica currently ranks as if it had identical capacity of one slot
  - instruction: replace the hardcoded 1.0 at roles.py:1152 and :1167 with the resolved capacity, keeping 1.0 as the fallback when none is published
  - honesty: a peer that publishes no capacity (older lobes, or a non-lobes replica) falls back to the current weight=1.0 and is still routable - an unpublished capacity never makes a replica unselectable
- a peer's pool candidacy must stop depending on its host-level pressure verdict: `_replicas.py`:342 `_busy_from_status` derives busy from the peer's pressure.shed / mode=='busy', which is what excluded a fully idle Spark (GPU running=0) from the pool for a 60% iowait reading
  - instruction: stop `_is_selectable`() keying pool candidacy on a peer's pressure-derived busy flag; gate on capacity utilisation (active >= max) instead, and keep the local shed decision separate
  - honesty: a peer under genuine pressure with a FULL engine is still deprioritised - decoupling candidacy from the pressure verdict must not make an actually-saturated box look attractive
- capacity must be clamped or sanity-bounded on receipt: q1 makes capacity peer-CONTROLLED input, and `_selection.py` ranks by active/weight, so a peer publishing an inflated capacity ranks as near-zero wait at every load level and silently vacuums the whole pool - a black hole with no error anywhere
  - instruction: clamp received capacity to a configured maximum on ingest in `_replicas.py`, and record the clamp in the replica row's reason field
  - honesty: a peer publishing an absurd capacity cannot capture more than a bounded share of traffic, and the clamp is observable rather than silent
- the pool needs local in-flight accounting at dispatch, not just the probed snapshot: `_replicas.py` sources load solely from the 5s /status refresh (`_DEFAULT_REFRESH_INTERVAL`=5.0) with no counter incremented when a request is actually dispatched, so a burst of concurrent arrivals all read one stale snapshot and stampede the same replica
  - instruction: increment a local in-flight counter at dispatch and decrement on completion; feed probed load + local in-flight into `estimated_wait` so the snapshot self-corrects between refreshes
  - honesty: N concurrent arrivals against two idle replicas distribute across both, not all to one, without waiting for a probe refresh
- an uncalibrated peer must not be silently starved: h3's weight=1.0 fallback keeps it routable but `estimated_wait`=active/weight ranks it 8x worse than a calibrated weight-8 peer at the same single active request (1.0 vs 0.125), so mixed-version fleets systematically drain toward whichever boxes happen to be calibrated
  - instruction: pick the uncalibrated fallback so it is neutral rather than pessimistic - do not reuse the 1.0 sentinel as if it were a measured capacity of one slot
  - honesty: a fleet where exactly one box is calibrated does not drain toward it: an uncalibrated peer still receives work proportional to an honest default, not to the 1.0 sentinel
- a calibrated capacity is only valid for the (box, checkpoint, context window, speculative config) it was measured on; lobes switch or a shape re-render invalidates it, and the existing live-probed fingerprint (`served_id` + quantization + `max_model_len` + runtime, `_replicas.py`:177) is the natural key to invalidate against
  - instruction: store the fingerprint alongside the calibrated capacity and discard the capacity when the live fingerprint no longer matches
  - honesty: a capacity measured under a different fingerprint is not used: after lobes switch, routing falls back to the safe default until recalibrated
- capacity-relative routing carries its own kill switch: a knob that pins weight to 1.0 fleet-wide, leaving the pool armed but the capacity signal inert (q4)
  - instruction: implement as a single env knob read at gateway start that forces the resolved capacity to 1.0 for every replica, local and peer alike
  - honesty: with the kill switch engaged the pool still routes and still forwards - only the capacity input reverts to 1.0, reproducing today's ranking exactly

## Honesty conditions

- with no capacity published and no \*`_PEER_ORIGINS` declared, every response stays byte-identical to the pre-pool contract - the #199 no-config-no-change guarantee is not weakened
- lobes/runtime/`_pressure.py` is unchanged by this work, and tests/`test_pressure.py` passes untouched
- every test in tests/`test_gateway_pool_pressure.py` either passes unchanged or is accompanied by an explicit, recorded renegotiation of the behaviour it asserted
- the fix is verified by the Spark rejoining the pool while its host iowait is still reading high - not by the iowait reading itself going away
- the change is invisible to a single-box deployment: an operator running machine-as-brain with no peers sees no new required configuration
- 'least full relative to its own capacity' is observable after the fact - X-Lobes-Route-Reason and the /capabilities replica rows expose the capacity and utilisation the decision used
- the `before_state` is reproducible on demand: with weight at 1.0 and a peer reporting pressure-busy, an idle peer is provably excluded from selection
- the two success signals are measured on the Spark+Thor cortex pair with the ghostty iowait artifact still present, and both numbers land in an acceptance transcript
- the 1.7x target is a floor derived from an already-measured result (19.1 tok/s on the same pair), not an aspirational number invented for this spec
- the Route-Reason vocabulary change is stated explicitly in the PR and docs/gateway-fleet.md, and a caller parsing the old closed set is told what changed

## Success signals

- with both boxes idle, concurrent requests to one gateway distribute across replicas in proportion to measured capacity instead of queueing on one box; and a box whose host iowait is high but whose engine is idle continues to receive pooled work
  - instruction: run the 8-way flood against the Spark+Thor pair and record both numbers under docs/evidence/
- an 8-way concurrent flood of model=cortex against one gateway, both boxes idle, reaches at least 1.7x the 11.0 tok/s single-owner aggregate baseline recorded in docs/evidence/2026-08-25-baseline-cortex-single-owner.txt (i.e. >=18.7 tok/s), and the Spark stays a selectable pool candidate while its host iowait still reads above the 50% threshold
  - instruction: compare against the recorded 11.0 tok/s baseline on the same 8-way raw-id flood shape, not against a fresh single-request measurement

## Scope / boundaries

- the pressure sampler is NOT the defect and must not be 'fixed': lobes/runtime/`_pressure.py` computes iowait as a correct 150ms delta of /proc/stat, verified live on the Spark at ~60% across five samples — the number is accurate, it is the routing INFERENCE from it that is wrong
  - instruction: make no edit to lobes/runtime/`_pressure.py`; if a change there seems necessary, that is a signal the diagnosis was wrong
- shedding under genuine pressure stays: tests/`test_gateway_pool_pressure.py` encodes 20+ behaviours (busy-with-selectable-peer forwards, both-boxes-busy yields exactly one forward and one 429, hand is the servable floor, marked arrivals never re-forward) that must survive unchanged
  - instruction: run tests/`test_gateway_pool_pressure.py` before and after; any test that must change gets an explicit note in the PR saying which behaviour was renegotiated and why
- host-level signals stay poisonable by design and that is the argument for c5, not a bug to chase upstream: the Spark's 60% iowait traces to a single idle ghostty terminal cgroup (app-com.mitchellh.ghostty.service, 96.66% PSI io-full) whose io.stat is EMPTY - zero block I/O ever charged - and which sits in user.slice, entirely outside the docker/vLLM cgroups
  - instruction: verification runs against the live Spark while its ghostty-driven iowait is still elevated - do not clear the artifact first, it is the test fixture
- X-Lobes-Route-Reason is an explicitly CLOSED vocabulary (server.py:728-731: local-idle | peer-less-loaded | local-busy-forwarded | affinity | sole-ready | none); capacity-relative routing either reuses peer-less-loaded with changed meaning or extends the set - both are contract changes that must be stated, not slipped in
  - instruction: expose the capacity and utilisation used in the /capabilities replica row so a capacity-driven choice is explainable from a trace

## Non-goals

- no calibration logic lands inside the gateway's request path: the gateway consumes a capacity number, it never measures one mid-flight — calibration is a separate read-only CLI verb, mirroring how assess/benchmark already sit outside the serving lane

## Assumptions

- per-device capacity already exists as vLLM --max-num-seqs, declared per-card in the profile system (profiles/`__init__.py`:130 field; base=4, spark=2; render.py:116 maps it to <PREFIX>`_MAX_NUM_SEQS`) and rendered to env as `PRIMARY_MAX_NUM_SEQS` (fleet/env.example:59)
- the 2026-08-25 replica-pool acceptance evidence is contaminated by this defect: docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt attributes its forwarding to the Spark being under 'organic iowait pressure', which live re-measurement suggests was this same signal with the disk essentially idle (0 sectors read, static pswpin/pswpout, zero D-state tasks)
- accurate capacity makes the stale-snapshot herd WORSE, not better: today every replica ranks at weight=1.0 so ties resolve to local and bursts stay put, whereas a genuinely least-full peer attracts an entire burst before the next probe corrects it

## Scope exploration

- `s1` — `lobes/gateway/_selection.py`: `estimated_wait`() = (running + waiting) / weight, floored at `_WEIGHT_EPSILON`; setting weight to a device's max active requests makes this exactly active/max utilization. Its own docstring names the gap: weight is 'a first-cut proxy for capacity per #199's open park on calibrating this later from measurement'
  - seeds: `c2`
- `s2` — `lobes/gateway/_replicas.py + lobes/roles.py`: weight is plumbed end-to-end (dataclass fields at `_replicas.py`:214/241/258, seeded via `_seed`() at 422, surfaced in the /capabilities replica row at roles.py:1130) but is populated only by the 1.0 default and the hardcoded 1.0 at roles.py:1152/1167 — grep finds no env parsing for it anywhere in lobes/
  - seeds: `c3`
- `s3` — `lobes/profiles/ + lobes/templates/fleet/env.example`: `PRIMARY_MAX_NUM_SEQS`=2 is already a per-card declared value (profiles/`__init__.py`:130 WorkloadProfile field, render.py:116 mapping), but env.example:59 documents it as an OOM safety cap ('MTP cap - 4 OOMs at n=3/high context'), NOT a measured throughput capacity — the two meanings must not be silently conflated
  - seeds: `c4`
- `s4` — `lobes/gateway/_replicas.py:342 _busy_from_status`: a peer's busy flag is read from its /status pressure.shed or pressure.mode=='busy'; `_is_selectable`() in `_selection.py` then drops any busy candidate outright, so one host-level reading removes an idle box from the pool entirely — measured live: Spark reported running=0/waiting=0 yet mode=busy from iowait 61.9%
  - seeds: `c5`
- `s5` — `lobes/runtime/_pressure.py`: `parse_iowait_percent` takes two /proc/stat snapshots 150ms apart and divides `iowait_delta` by `total_delta` - correct methodology, not a since-boot artifact. Confirmed live on the Spark (5 samples, 58.5-62.9%). The sampler is sound; the defect is treating host iowait as a proxy for serving capacity
  - seeds: `c6`
- `s6` — `tests/test_gateway_pool_pressure.py`: the busy-to-forward contract is densely tested (`test_busy_with_a_selectable_peer_forwards`, `test_both_boxes_busy_produces_exactly_one_forward_and_one_429`, `test_hand_is_the_servable_floor_under_pressure_and_never_forwards`, `test_a_marked_arrival_under_local_pressure_gets_the_local_429`); any change to busy semantics must keep these green or explicitly renegotiate them
  - seeds: `c7`
- `s7` — `lobes/gateway/server.py:2350 fleet_status_payload + lobes/gateway/_config.py:330 PEER_ORIGINS_ENV`: both candidate wiring points already exist: `fleet_status_payload` builds the /status body a peer probes (so a box could publish its own capacity there), and `PEER_ORIGINS_ENV` maps all nine role prefixes to their peer knobs (the pattern a positional <PREFIX>`_PEER_WEIGHTS` would follow) — q1 chooses between them
  - seeds: `c2`
- `s8` — `docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt`: the pool's only acceptance run cites 'organic iowait pressure' on the Spark as the trigger for its measured +74% forwarding result; if that pressure was this artifact, the transcript still proves the forwarding mechanism but not that it triggers on real load
  - seeds: `c9`
- `s9` — `/sys/fs/cgroup PSI walk on the Spark`: io.pressure localizes to user.slice/user-1000.slice/user@1000.service/app.slice/app-com.mitchellh.ghostty.service at 96.66% full, holding one sleeping (Ssl) ghostty process up 6d21h with an empty io.stat - an unrelated desktop process can therefore evict a serving box from the pool
  - seeds: `c10`
- `s10` — `issue #215 (pressure gate is tier-alias-only)`: the existing divergence - a raw-id request under local PRESSURE is not forwarded, only tier-alias requests are - is entangled with c5: if pool candidacy stops keying on pressure, the alias/raw-id split may resolve or may need separate treatment
- `s11` — `challenge pass / security lens: q1 self-published capacity + server.py:2713 _authorized`: the inbound auth gate protects who may ASK, but nothing validates what a declared peer ANSWERS; capacity arrives as trusted peer input. Note Thor currently runs `GATEWAY_API_KEY` empty, so its /status is unauthenticated on the tailnet
  - seeds: `c19`
- `s12` — `challenge pass / concurrency lens: lobes/gateway/_replicas.py refresh loop`: load is probe-sourced only (`_DEFAULT_REFRESH_INTERVAL`=5.0, refresh threads at 732/737); grep finds no in-flight counter incremented at dispatch, so selection decides on data up to 5s stale
  - seeds: `c20`
- `s13` — `challenge pass / migration lens: h3 weight fallback arithmetic`: worked the fallback through `estimated_wait`(): an uncalibrated peer at weight 1.0 with one active request scores 1.0 against a calibrated weight-8 peer's 0.125 - routable, as h3 promises, but deprioritised by the capacity ratio. Idle-vs-idle still ties at 0.0
  - seeds: `c22`
- `s14` — `challenge pass / observability lens: server.py:726-745 header contract`: Route-Reason is documented as a closed set and Served-By/Route-Reason appear only on pooled answers (h1); the capacity used is not currently exposed on any header, so a capacity-driven decision would be unexplainable from a trace alone
  - seeds: `c24`
- `s15` — `challenge pass / lifecycle lens: lobes switch + shape re-render vs a calibrated number`: capacity is measured against one checkpoint at one window; lobes switch is a down+up with a model swap and a shape re-render force-writes keys, so a stored capacity outlives the conditions it was measured under unless keyed to the fingerprint
  - seeds: `c23`
- `s16` — `challenge pass / reversibility lens: env knob surface`: no `POOL_ENABLED`/`DISABLE_POOL`/`REPLICA_POOL` knob exists anywhere in lobes/; the only rollback today is unsetting \*`_PEER_ORIGINS`, which removes pooling wholesale - there is no way to keep the pool but disable the capacity signal. Raised as q4
- `s17` — `challenge pass / adjacent-systems lens: lobes/realtime voice lane`: examined `_turn.py`: the voice lane addresses model=multimodal, and only cortex is a validated pooled role today, so voice latency is untouched by this change - CLEAN for now, with residual risk if multimodal is ever pooled, since a voice turn budgets ~1s and a cross-box forward adds a proxy hop

## Decisions

- a box learns a peer's capacity by probing it: each box declares its own max active requests locally and publishes it in its /status body, which ReplicaCache already fetches on its 5s refresh (q1)
- capacity is a MEASURED throughput knee produced by a calibration routine - explicitly not the --max-num-seqs OOM cap and not vLLM's KV-derived concurrency ceiling, both of which are numbers chosen for other reasons (q2)
- under genuine local pressure a box forwards to a selectable peer with spare capacity rather than shedding; 429 remains only when no replica anywhere is selectable (q3)

## Open parks

- [unknown_nonblocking] whether capacity should be a single scalar at all - a box may have different effective capacity for short agentic turns than for long-context requests, which one calibrated number cannot express

## Resolved vagueness

- [unknown_blocking] the calibration routine's stopping rule is unspecified - ramp concurrency until aggregate throughput plateaus, until TTFT crosses a bound, or until the engine refuses admission — resolved: the knee is the highest concurrency level at which aggregate throughput still rises meaningfully per added slot AND TTFT stays under a declared bound; the TTFT guard prevents a throughput-optimal but unusable answer
