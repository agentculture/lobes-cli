# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.69.1] - 2026-08-29

### Added

### Changed

### Fixed

## [0.69.0] - 2026-08-29

### Added

- `lobes up gateway` (and `--build`) — a first-class path to restart or re-image the gateway alone at a new `MODEL_GEAR_VERSION` (issue #222).

### Fixed

- `lobes up <role>` now runs with `--no-deps` and the fleet gateway declares no `depends_on`, so a targeted start can no longer recreate the whole fleet or boot a lane the box declares infeasible (issue #222).
- A streamed `/v1/chat/completions` relay always terminates the client stream: an upstream that dies mid-stream now yields an SSE error event, `data: [DONE]` and the chunked terminator, and a client that hung up is logged instead of leaving a wedged socket (issue #220).
- A proxied role advertises the hosting peer's own `ready` and `context`, read from the peer's `GET /capabilities`, instead of a locally-guessed catalog ceiling and an unsatisfiable `/v1/models` check (issue #220).

## [0.68.4] - 2026-08-29

### Fixed

- The last three membership assertions in `tests/test_variation_catalog.py` are subject-first, completing SonarCloud S3415 for the file — the previous pass missed module constants on the left because it scanned only for bare literals.

## [0.68.3] - 2026-08-29

### Fixed

- `tests/test_variation_catalog.py` uses one assertion argument order throughout (SonarCloud S3415); the membership checks became subject-first `.count()` assertions, which are also more specific than the bare `in` they replaced.

## [0.68.2] - 2026-08-29

### Fixed

- The `VARIATION.md` heading regex uses an unquantified `[ \t]` separator, removing the ambiguity between it and the trailing capture group (SonarCloud S8786). Measured behaviour was already linear; the change removes the ambiguity rather than relying on it.

## [0.68.1] - 2026-08-29

### Fixed

- Secret scanner: `${VAR:-literal-secret}` and Dockerfile whitespace `ENV KEY secret` no longer bypass the required `secrets-scan` CI gate. An existing test had asserted the first bypass was safe behaviour.
- `init --from-lock` refuses to write through a symlink in the target directory, and stages every file before committing so a failed restore leaves the deployment untouched rather than mixed.
- A malformed `deployment.lock.toml` is now a controlled user error instead of a traceback out of `init`/`doctor`.
- `lock_drift` compares the lock and the deployment symmetrically, so a knob added after capture is reported (tagged added/changed/removed).
- The buildability guard routes `.devN` pins to TestPyPI rather than PyPI, where this repo actually publishes them.

## [0.68.0] - 2026-08-29

### Added

- `lobes init --from-lock` restores a committed deployment variation verbatim, bypassing profile/shape resolution, with a machine-type guard and an offline buildability preflight.
- `deployment.lock.toml` — a secret-free-by-construction lock whose key set is an allowlist derived from `profile_env`.
- `deployments/` — the published variation catalog, its info-file honesty contract, and a validator. Ships with zero real variations; capture needs physical hardware.
- `lobes doctor` reports `lock_drift`, naming the specific differing files; `lobes switch` warns that a committed lock has gone stale.
- A required `secrets-scan` CI job over every committed deployment artifact.
- `docs/secret-rotation.md` — the leaked-key drill and the full key-copy inventory.
- `lobes explain lock` and `docs/deployment-lock.md`.

### Changed

- `.gitignore` now uses a positional rule: a `.env` suffix is ignored, a `.env.` prefix is allowed. The committed goldens keep an explicit negation.
- Every fleet service reads an optional `.secrets.env` alongside `.env`, so no secret is ever retyped by hand.

## [0.67.1] - 2026-08-28

### Fixed

- **Qodo review round on #221 — seven findings, all real.** Two were latent bugs
  in `lobes/assess.py`'s `run_concurrent` fan-out, so they affected every caller,
  not just the new verb: aggregate throughput was computed as
  `concurrency x reciprocal of the mean per-request rate` rather than **total
  completion tokens over batch wall time** (the two diverge ~90% when completion
  lengths differ, which would have made `calibration_knee` persist the wrong
  capacity); and gateway auth was installed in a `ContextVar` that
  `ThreadPoolExecutor` **does not propagate to worker threads** — the module's
  own comment claimed the opposite — so a ramp against an authenticated gateway
  would have 401'd on every request. Each submitted task now runs under its own
  `copy_context()` snapshot, since re-running one `Context` concurrently raises
  `RuntimeError: cyclic call`.
- **The `1.0` capacity sentinel collided with a genuine measured capacity of 1.**
  A box with `max_num_seqs=1` publishes a real knee of 1, which `is_calibrated`
  read as "nothing published" — so it was never considered full and its measured
  capacity never affected routing. Replaced the magic number with an explicit
  `ReplicaState.calibrated` flag set at the single ingest point, and made the
  operator-declared `weight` fields `float | None` so `PRIMARY_MAX_ACTIVE=1` is
  no longer read as undeclared.
- **Work executing locally is now counted even when no routing decision was
  made.** A proxied arrival, and `d5`'s nothing-selectable queue-locally path,
  both dialled the local backend without touching the dispatch counter — so a
  receiver undercounted its own load and kept selecting an already-loaded local
  replica. The counter now records what the engine is *executing*, not what the
  router *decided*.
- **Probe/in-flight reconciliation now corrects in both directions.** A dispatch
  registered just before a probe sampled the engine was counted by neither
  (stale-low, the burst undercount this accounting exists to prevent); and load
  from a burst that had already completed was still read as current (stale-high
  — measured live, it suppressed forwarding entirely in one run of five, 50.82
  vs 98.6 tok/s). Each dispatch is now bucketed against the probe's own stamp
  and corrections are bounded by the probed count, so a box never invents load a
  peer never reported nor cancels more than it dispatched.
- **`lobes calibrate` argument and output hygiene.** Non-finite and non-positive
  values are rejected before any network call (`--ttft-bound-s nan` silently
  disabled the TTFT guard entirely, and worse, fell through to a real ramp
  against a nonexistent server), and a refused `--apply` no longer writes a
  success-shaped payload to stdout before failing — matching how `up` and `init`
  already handle a refused write.

- **SonarCloud on #221's new code: seven smells cleared, three refused.** Cleared
  a `dict.fromkeys` simplification, three multi-throw `pytest.raises` blocks
  (each narrowed to the single call under test, then sanity-checked by
  temporarily breaking the pinned behaviour), and three composite assertions.
  **Refused** three `S1940` "use the opposite operator" findings and marked them
  ACCEPTED with rationale: `not (value > 0.0)` and `value <= 0.0` are NOT
  equivalent under IEEE-754, since every comparison against NaN is False. The
  negated form is a deliberate single check refusing NaN, zero and negatives
  together — the rewrite would let a NaN capacity reach division and ranking,
  which is the same defect class flagged on this PR's CLI arguments.
- **`sonarclaude` gained `--pull-request/--pr`.** The vendored script could only
  query project-wide, so PR-scoped new-code findings were invisible through it
  and a project-wide query returned "0 issues in this PR's files" while the PR
  view showed ten. Needs upstreaming to steward.

### Notes

- One finding was **pushed back**, not fixed: Qodo flagged `_resolve_tier` for
  omitting engine state from the pressure decision, citing "return 429 when no
  replica is selectable". That requirement is exactly what deviation `d5`
  reversed — live acceptance measured it shedding 4 of 8 requests the fleet
  previously queued. `_is_selectable` still excludes a full replica, so routing
  still prefers headroom; only the *shed* verdict changed. See
  `docs/evidence/2026-08-28-accept-capacity-relative-pool-thor-spark.txt`.

## [0.67.0] - 2026-08-27

### Added

- **Spec: capacity-relative pool routing** — `docs/specs/2026-08-27-capacity-relative-pool-routing.md`,
  a converged devague frame (`/scope` → `/think` → `/challenge`) for routing the
  cortex replica pool by each box's own measured capacity instead of a
  host-level pressure verdict. Redeems the `#199` parked `weight` calibration
  hook: `_selection.py` already computes `estimated_wait = (running + waiting) /
  weight`, which IS capacity-relative utilisation, but `weight` is populated by
  nothing beyond the hardcoded `1.0` at `roles.py:1152,1167`. Decisions
  recorded: capacity is **peer self-published** via `/status` (O(machines)
  config, no per-pair drift); capacity is a **measured throughput knee** —
  neither the `--max-num-seqs` OOM cap nor vLLM's KV-derived ceiling; a box
  under genuine pressure **forwards to a spare-capacity peer** rather than
  shedding 429; and the feature carries its own kill switch pinning `weight` to
  1.0 fleet-wide.
- **Six findings from the rigorous `/challenge` pass**, all spec-side: capacity
  clamping (self-published capacity is peer-CONTROLLED input, and an inflated
  value ranks as near-zero wait at every load level — a silent traffic black
  hole); local in-flight accounting at dispatch (load is probe-sourced only, on
  a 5 s refresh, so concurrent arrivals stampede one stale snapshot — a herd
  that accurate capacity makes *worse*, since today's uniform `weight=1.0` ties
  resolve to local); a neutral rather than pessimistic uncalibrated-peer
  fallback (the `1.0` sentinel ranks an uncalibrated peer 8× worse than a
  calibrated weight-8 peer at one active request, draining mixed-version fleets
  toward whichever boxes happen to be calibrated); fingerprint-keyed capacity
  invalidation across `lobes switch`; and the `X-Lobes-Route-Reason` closed
  vocabulary as a stated contract change.
- **Plan: capacity-relative pool routing** —
  `docs/plans/2026-08-27-capacity-relative-pool-routing.md`, seeded from the
  converged frame and covering **36/36** spec targets across ten tasks in five
  dependency waves, each wave file-disjoint so parallel execution matches
  serial: config knobs + kill switch, the pure knee-measurement function,
  selection policy + reason vocabulary, capacity ingest/clamp/fingerprint-keying
  /in-flight counter, `/status` publication + dispatch accounting, the
  `/capabilities` row, regression guards, docs, the `calibrate` verb, and live
  acceptance on the Spark+Thor pair. Three non-blocking risks recorded, incl.
  that the `11.0 tok/s` single-owner baseline is re-measured before t10's
  throughput target leans on it.

### Changed

- **The pool no longer treats a host-level pressure verdict as a serving
  verdict — on either side of a forward.** Sending side (`_selection.py`): a
  peer's `busy` flag no longer gates pool candidacy at all; a candidate is
  unselectable when its active count reaches its own capacity. A local shed
  verdict still excludes the local replica only. `weight == 1.0` is now the
  *uncalibrated sentinel* rather than a measured one-slot capacity, and an
  uncalibrated replica ranks at the median of the calibrated capacities in its
  candidate set — without this, an uncalibrated peer ranked 8x worse than a
  calibrated weight-8 peer at one active request and mixed-version fleets
  drained toward whichever boxes happened to be calibrated. Receiving side
  (`_pressure_policy.py`, `server.py`, deviation `d1`): a **pooled** arrival is
  no longer shed on iowait alone. A box still sheds on `swap > 75%` (it is
  paging the very memory holding weights and KV) and on engine saturation
  (`active >= published capacity` — deliberately the same fact selection gates
  candidacy on, so a box can never select itself and refuse itself on
  contradictory evidence). The carve-out is pooled-only: for a single-owner
  role a 429 is honest backpressure, but for a pooled role it is a lie whenever
  a replica has room. `decide()` defaults `pooled=False`, so every pre-`d1`
  call site and every single-box deployment decides byte-identically (`h1`).
  `mode` still reports the host honestly to `/status` and peer probes; only the
  shed band narrowed.
- **`weight` now carries a real, self-published capacity.** A peer publishes its
  own on the `/status` body already probed (`backends[].capacity`); local and
  peer values pass through one ingest gate. An inflated value is **clamped**
  (`CAPACITY_CLAMP_MAX = 64.0`, above every concurrency figure this fleet has
  measured) and the clamp is recorded in the replica row's `reason` rather than
  applied silently — unclamped, a peer publishing 10,000 would score near-zero
  wait at every load level and vacuum the pool. A capacity is **pinned to the
  fingerprint it was measured under** and discarded when that changes, with two
  deliberate asymmetries: an unknown/absent fingerprint is never read as
  evidence of change, and the pin travels with the number rather than the probe,
  so a republished stale value cannot resurrect itself one refresh later.
- **Local in-flight accounting closes the stale-snapshot herd.** Load was
  probe-sourced only, on a 5 s refresh, so concurrent arrivals stampeded one
  snapshot — a hazard accurate capacity makes *worse*, since today's uniform
  weights make ties resolve to local. Dispatches are now folded into `waiting`
  so the snapshot self-corrects between probes, guarded three ways because a
  leaked counter cannot self-heal: a context manager that releases on any exit,
  an idempotent release that tolerates double-free, and a 900 s expiry so even a
  leak decays instead of pinning a healthy box at "full" forever.
- **`X-Lobes-Route-Reason` vocabulary: membership unchanged, `peer-less-loaded`
  redefined.** It now means "less loaded *relative to its own capacity*", so a
  peer with **more** active requests can legitimately win it, and it is now also
  emitted when the local replica is excluded for being full — a state that
  previously could only arise from a pressure verdict and reported
  `local-busy-forwarded`.
- **Narrows issue #215 without widening it.** In the iowait-only band — exactly
  where the live Spark sits — `model=cortex` and the raw served id now both
  serve locally instead of diverging. In the swap / full-engine band the
  alias-only gate is unchanged, still #215's to fix.

- **`X-Lobes-Route-Load`** — a new response header on pooled answers only,
  reporting the numbers a routing decision was actually made on:
  `active=1; capacity=8; utilisation=0.125; calibrated=true`. `calibrated=false`
  says out loud that the `1.0` sentinel is a neutral substitute rather than a
  measured one-slot capacity. The `X-Lobes-Route-Reason` vocabulary was
  deliberately NOT widened again to carry this. A peer's copy is stripped from a
  relayed answer, and the header is absent entirely from a no-peer deployment
  (`h1`).
- **Capacity is published on `/status`** at `backends[].capacity`, so a peer
  learns it from the probe it already makes. The key is withheld — rather than
  fabricated as `1.0` — when nothing is declared, when the role is infeasible
  here, or when the backend is never-proxied: an absent key reads as
  *uncalibrated* at the peer, whereas a fabricated `1.0` would read as a
  calibrated one-slot capacity and starve this box.
- **`lobes calibrate <role>`** — measures a box's serving capacity by ramping
  concurrency and locating the throughput knee, then reports the level, the
  samples behind it, and *whether the ramp plateaued*. Read-only by default;
  `--apply` writes `<PREFIX>_MAX_ACTIVE` to `.env`. It **refuses to write** when
  the ramp never plateaued (recording the top level tried would be a guess, not
  a knee) or when the level resolves to zero (which would make a role
  permanently unselectable). Stops early once the knee is settled. The measured
  number is scoped to one `(box, checkpoint, context window, speculative
  config)` and goes stale after a `lobes switch` — the help text says so.

### Fixed

- **VALIDATED live on the Spark+Thor cortex pair, 2026-08-28**
  (`docs/evidence/2026-08-28-accept-capacity-relative-pool-thor-spark.txt`).
  With the Spark still declaring `mode: busy, shed: true` at ~62-72% iowait and
  its engine idle, it rejoins the pool as a `ready` candidate and is genuinely
  dispatched to — two of eight requests per run carry
  `X-Lobes-Proxied-By: <spark>` with `peer-less-loaded`, reproducible across
  three runs. An 8-way flood improves from a re-measured single-owner median of
  **51.66 tok/s to 66.92 tok/s (+29.5%)** with identical work (888 completion
  tokens) and zero errors on either side. Recorded honestly: the peer takes only
  25% of the burst (the overflow falls through to local queueing, so the gain is
  a floor); `c18`'s **ratio** target of 1.7x is NOT met on its own terms (1.295x
  against the honest baseline — the 1.7x was calibrated against a baseline
  captured under this same artifact, per risk `r1`); `lobes calibrate` was not
  exercised on hardware; and capacity is an even 2:2 split ceilinged by
  `max_num_seqs`, so the capacity-*proportional* arm of the design remains
  untested. **Amended the same day** after operator review: the 8-way figure was
  the wrong shape — beyond fleet capacity the overflow queues locally and
  dilutes the result, and the aggregate cannot separate "work was distributed"
  from "work landed on the faster box" (the Spark is ~2.7x faster per stream:
  49.24 vs 18.18 tok/s). Measured at fleet capacity (4 concurrent = 2 slots + 2
  slots) the pool delivers **50.71 -> ~98.6 tok/s, +94%**. That is **97% of the
  even-split ceiling** of 101.4 tok/s — 138.5 is unreachable with an even split
  because the faster box finishes its half early and idles, which is the
  intrinsic cost of the operator's decision (risk `r6`) that capacity encodes
  each machine's PARALLELISM capacity rather than its throughput. One further
  defect is recorded and NOT fixed: one run in five forwarded nothing because
  the 5 s probe still reported the peer at `run=2` from a completed burst, which
  at `capacity=2` reads as full — local in-flight accounting corrects herding
  within a burst but not probe data that is stale-high between bursts.

- **Engine saturation drives selection, not shedding (`d5`, found by live
  acceptance).** `d1`'s receiving-side work made `active >= capacity` a shed
  signal, which routed a saturated request down the busy path (429) instead of
  `_PoolFallthrough`'s local dial — so a burst the fleet previously queued was
  refused. Measured live on the Spark+Thor pair at `PRIMARY_MAX_ACTIVE=2` each:
  an 8-way flood through the pool served **4/8 with four HTTP 429s**, where the
  same flood bypassing the pool served **8/8 with zero errors**. Capacity was
  specified as a routing preference, not admission control. The gateway no
  longer passes engine state to `decide()`; `_is_selectable` still excludes a
  full replica, so routing still prefers headroom while a saturated fleet falls
  through to local queueing. `decide()`'s own contract is unchanged (its
  `engine_*` parameters remain, now unsupplied) because the `h1` byte-identity
  golden pins that pure-function surface.
- **The capacity knobs never reached the gateway container — the feature was
  inert in every real deployment.** `lobes/templates/fleet/docker-compose.yml`
  enumerates each env var the gateway receives (there is no `env_file`), and
  none of the ten `<PREFIX>_MAX_ACTIVE` knobs nor
  `GATEWAY_CAPACITY_KILL_SWITCH` were listed, so an operator's `.env` value
  could never arrive. Every test passed because tests construct config
  in-process, bypassing compose — this class of gap was structurally invisible
  to the suite. A new guard test now derives the required env set from
  `lobes/gateway/_config.py` itself (its `*_ENV` constants, the runtime
  feasible×fingerprint cross-product, and an AST scan for literals passed to the
  module's own env readers), so a future knob added without a compose line fails
  automatically.
- **Three unrelated knobs were already inert and are now plumbed**, surfaced by
  that same guard: **`GATEWAY_FORCE_STRICT_TOOLS`** — documented to operators in
  `env.example` and consumed by `server.py`, but never passed through, so
  setting it did nothing — and **`FALLBACK_URL` / `FALLBACK_SERVED_NAME`**,
  which `env.example` presented as `.env` keys while the compose comment told
  operators to edit the template instead. Defaults are unchanged and empty, so
  the default fleet renders identically. `GATEWAY_HOST` is deliberately NOT
  passed through (inside the container `0.0.0.0` is the only correct bind; an
  override could make the gateway unreachable while looking healthy) and that
  exemption is recorded in a structure the guard itself validates.

- **Diagnosed why the cortex replica pool has been running single-box.** The
  DGX Spark sits permanently in `mode: busy, shed: true` from a ~60 % iowait
  reading, which removes it from the pool entirely (`_is_selectable()` drops any
  busy candidate), so the Thor never forwards. Live measurement shows the box is
  not under I/O pressure at all: 0 sectors read and static `pswpin`/`pswpout`
  over 5 s, zero `D`-state tasks, load average 0.72 on 20 cores. PSI localises
  the pressure to `user.slice/.../app-com.mitchellh.ghostty.service` at 96.66 %
  io-full — one sleeping desktop terminal, up 6d21h, whose `io.stat` is **empty**
  (zero block I/O ever charged), entirely outside the docker/vLLM cgroups. The
  sampler itself is correct (`lobes/runtime/_pressure.py` takes a proper 150 ms
  `/proc/stat` delta) and is explicitly out of scope: the defect is treating
  host iowait as a proxy for serving capacity. Recorded as boundary claims
  `c6`/`c10` on the frame; no code changes in this release.

## [0.66.2] - 2026-08-27

### Fixed

- **The #199 spec and plan records no longer read as if the Jetson AGX Orin
  were a pool member.** Both exported devague documents
  (`docs/specs/2026-08-25-cortex-replica-pool-199.md`,
  `docs/plans/2026-08-25-cortex-replica-pool-199.md`) open with the verbatim
  `/scope` announcement (frame claim `c1`, `origin: user`), whose parenthetical
  names "Spark, Thor, Orin at 256K" as compatible replicas. The converged
  frame's own live-probed-fingerprint claim, the 0.63.0 implementation, and the
  acceptance transcript all admit **Spark + Thor only** — the Orin's llama.cpp
  `Qwen3.8-27B-UD-Q4_K_M` cortex serves under the id `cortex` at a different
  quantization and runtime, so it is marked `compatible: false` and never enters
  the candidate set. The announcement is kept unedited (it is the record of what
  was announced); a **correction note** now sits directly beneath it in both
  documents, naming the exemption and citing `docs/gateway-fleet.md` and
  `docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt`. No
  behaviour changes; `CLAUDE.md`, `docs/gateway-fleet.md` and the 0.62.1
  changelog entry already stated the exemption correctly.

### Changed

- **PR #212 (the #199 spec + plan) reconciled with the implementation that
  shipped ahead of it.** Both exported documents, the frame JSON and the
  `docs/evidence/` transcripts landed in `main` via PR #213 (0.63.0), which was
  stacked on this branch; `main` has since advanced to 0.66.1. Merging `main`
  back in resolves every conflict to `main`'s side — the plan JSON keeps its
  delivery-time task `t12` and follow-up risks `r5`/`r6`, `.devague/current*`
  keep `main`'s pointers, and the eidetic store keeps the post-validation
  record — so this branch adds only the correction note above.

## [0.66.1] - 2026-08-27

### Changed

- **Cleared every open SonarCloud issue on the project — all 96, all in
  `tests/`.** Two rules, both test-only: `python:S9073` ("Split this composite
  assertion into separate assertions", 92 MAJOR) and `python:S9083` ("Remove
  empty parentheses from this decorator", 4 MINOR). No file under `lobes/` is
  touched and no runtime behaviour changes — a failing conjunct now names
  itself instead of hiding behind a compound `and`.
  - **92 composite assertions split** across 30 test modules. 83 were
    single-line with no message and split mechanically; **9 carried assertion
    messages**, and each conjunct now gets **its own message naming which
    condition failed** — a strict improvement, since `assert A and B, "X"`
    reported `X` regardless of which half broke.
  - **4 empty-paren `@pytest.fixture()` decorators** reduced to
    `@pytest.fixture` (`test_assess_perf.py`, `test_bench_compare.py`,
    `test_cli_benchmark_profiles.py`, `test_minor_client.py`).

### Fixed

- **Indentation integrity of a mechanical rewrite, proven structurally rather
  than by a green suite.** An earlier substring-based pass matched *mid-line*
  and dedented the second `assert` of a split pair out of its enclosing block —
  out of a `try` in `test_minor_logprobs.py` and out of a `for` in
  `test_realtime_conversation.py` — and **the full suite still passed**, because
  the asserts ran, just at the wrong nesting. The diff is now verified by an
  AST-level equivalence check: main's tree with the canonical split applied
  in place must `ast.dump`-match this branch's tree for all 142 test files
  (assert messages normalised out). That check passes, so no statement escaped
  its block. An independent AST audit reproduces SonarCloud's counts exactly —
  92 + 4 on `main`, **0 + 0** here.

## [0.66.0] - 2026-08-27

### Added

- **`docs/image-ledger.md` — one index for every container image the fleet
  pins.** Digest, engine + version, the machine and arch it was validated on,
  the model(s) it serves, its build recipe, and the evidence transcript that
  validated it. Provenance was previously scattered across 10 image pins in
  `lobes/templates/fleet/docker-compose.yml`, 18 further digest mentions in
  `docs/*.md`, and 5 `Dockerfile.*` templates, with no index tying them
  together. Failed and superseded recipes keep their rows; a row with no
  evidence link is marked UNVALIDATED (#108).
- **`docs/experiments/` — a home for checkpoints and engines we evaluated and
  did not put into service.** Its `README.md` defines the contract: an
  experiment doc answers what it is, why we wanted it, and why it is not
  running, and it cites `docs/<model>.md` / `docs/evidence/` / `docs/specs/` /
  `docs/image-ledger.md` rather than replacing them. #108 applies here more
  sharply than elsewhere, because these files describe things that were never
  served. Deliberately narrow: a `docs/models/` reorg would change
  `lobes/catalog.py`'s `doc=` contract and sweep 456 references, and is its own
  change.
- **`docs/experiments/qwen3.8-flash-next-gguf-llamacpp-vllm.md` — what Qwen3.8-Flash-Next
  is, why each engine was considered, and why neither happened yet.** Names the
  checkpoint the way the sibling per-model docs do, so it sits next to
  `docs/qwen3.8-27b-gguf-llamacpp.md` and is findable by model name. Records
  that **nothing native fits** a 122 GiB board (BF16 335.28 GiB, official FP8
  172.78 GiB, RadixArk NVFP4 135 GB — the ~35 GB PLE table is the floor), the
  ~25x prefill / ~36x TTFT penalty that argued against llama.cpp, and the four
  stacked unknowns that stalled vLLM. **Status: NOT SERVED**, deferred
  2026-08-27 (#108 — no box has booted it, no transcript exists).
- **`docs/specs/2026-08-27-qwen3-8-flash-next-gguf-candidate.md`** — the
  converged, challenged spec for serving Qwen3.8-Flash-Next (125B MoE, 6B
  active) from an Unsloth Dynamic GGUF through the llama.cpp lane on the Thor.
- **`docs/specs/2026-08-27-flash-next-on-vllm.md`** — the converged, challenged
  spec for the vLLM route: a fleet nightly move with a cortex before/after
  benchmark, then the same GGUF through `vllm-gguf-plugin`. **Execution is
  deferred, not abandoned** — the ledger's "Deferred" section records both
  recipes and every blocker found, so resuming needs no re-derivation.

### Changed

### Fixed

- **Recorded the fleet's actual vLLM version.** `docs/lfm2.5-1.2b-hand.md` and
  `docs/qwen3-reranker-0.6b.md` describe the shared pin as `0.23.1rc1.dev672`,
  which is the **superseded** digest `sha256:7c5a10e9…`. The shipped pin
  `sha256:8bd082c2…` has been `0.26.1rc1.dev942+g5a4c8d992` since the
  qwen3.8-cortex-upgrade (`docs/vllm-nightly-migration.md:424`). The ledger
  records the live value and flags both stale references; the per-model docs
  are left untouched in this docs-only change.
- **The shared nightly digest drives SEVEN services, not four or six**
  (Qodo review, PR #218). `vllm-embed-deep` was dropped from two successive
  counts of the same list. The ledger now enumerates the services rather than
  stating a number: `vllm-primary`, `vllm-embed`, `vllm-embed-deep`,
  `vllm-rerank`, `vllm-hand`, `vllm-worker`, `vllm-associate`.

## [0.65.1] - 2026-08-26

### Fixed

- **`base.toml`'s vetoes were being bypassed for every opt-in core role**
  (Qodo review, PR #217). The overlay introduced for hosted opt-in roles set
  `feasible=True` unconditionally, so `--shape thor-muse` on an unrecognised
  card would render a full 31B budget the conservative fallback exists to
  prevent — and the same for `worker` and `associate`. Root cause was deviation
  `d1`, which had the Orin card declare `[roles.associate] feasible = false`;
  the card is now **silent** on associate, exactly as every card already is for
  `muse`/`worker`, so the plain overlay composes correctly and the special case
  is gone. `thor-worker` on `base`/`orin` renders `WORKER_FEASIBLE=false` again
  rather than a stale Qwen-era budget.
- `ASSOCIATE_TOOL_CALL_PARSER` reached only the gateway passthrough, not the
  lane. Since the gateway consumes it for the capabilities `tool_parser` field
  and the replica-pool fingerprint, setting it made `lobes capabilities`
  advertise a parser the lane was not running. It now reaches
  `--tool-call-parser` too (default `qwen3_coder` unchanged).
- markdownlint: three asterisk bullets added to `docs/colleague-stack.md`
  became the file's first list and flipped MD004's expected style, breaking 18
  pre-existing dash bullets. Converted to match the file's own convention
  rather than rewriting untouched lines.

### Changed

- `tests/test_profile_schema.py` splits a composite `in_proj and out_proj`
  assertion into two, so a failure names which projection is missing.

## [0.65.0] - 2026-08-26

### Changed

- **`orin-associate` is now a SOLO shape** (operator decision, deviation `d2`):
  hosts `associate` alone at `gpu_mem_util = 0.80`, dropping `hand`, `embedder`
  and `reranker`. It is the **first built-in shape to drop `hand`** — the
  pressure-policy servable floor every other shape hosts — so a box on this
  shape has **no floor**: under swap/iowait pressure it sheds 429 with nothing
  to degrade to, leaning entirely on this card's
  `LOBES_IOWAIT_DEGRADED_THRESHOLD=100` override. Accepted deliberately.

### Added

- **DSpark speculative decoding, MEASURED** on the Orin
  (`docs/evidence/2026-08-26-accept-orin-associate.txt`). Controlled A/B —
  identical engine, image, util and resident set, speculation the only
  variable: **96.8 / 121.1 / 90.6 / 93.9 / 59.1 tok/s** with DSpark versus
  54.8 / 54.7 / 54.6 / 54.0 / 52.1 plain across depths 0 / 512 / 2048 / 8192 /
  32768 — a **1.77–2.21× gain** at shallow-to-moderate depth. Draft acceptance
  35–64%; the drafter costs ~37% of the KV pool (3,806,000 → 2,395,428 tokens).
  Requires vLLM **v0.27.1**: the fleet nightly rejects `dspark` and refuses to
  boot, so `ASSOCIATE_IMAGE` and `ASSOCIATE_SPECULATIVE_CONFIG` are a matched
  pair.
- `docs/orin-associate-deployment.md` — a copy-pasteable deployment reference:
  the plain `docker run`, the compose service verbatim, the budget table for
  every co-residency measured, and the five things that will bite you.

### Fixed

- `tests/test_shape_contract_matrix.py` built chat-completions cells for
  **pooling and audio** roles, expecting `role_infeasible` where
  `model_not_found` is correct — such a role is not a chat model whether hosted
  or not. Never exercised until `orin-associate` became the first shape to drop
  `embedder`/`reranker`. The matrix now derives generate-ness from `TIER_ROLE`,
  and had also never been taught about `associate`.

## [0.64.0] - 2026-08-26

### Added

- **`associate` — the tenth Colleague role, serving NVIDIA Nemotron 3.5
  Lightning 30B-A3B on the Jetson AGX Orin 64GB (sm_87).** "They do, but not
  act": worker's responsibilities MINUS `repo_action`, ranked
  `hand < multimodal < worker < muse < associate < main`, opt-in-hosted, and
  shedding 429 under pressure like every other full tier. Ships with the
  `vllm-associate` compose lane (the eight vLLM serve flags NVIDIA's Jetson
  recipe needs, `ASSOCIATE_IMAGE`, and a default-off
  `ASSOCIATE_SPECULATIVE_CONFIG` for DSpark), the `orin-associate` deployment
  shape, and a `lobes doctor` check that fails a hosted-but-uncredentialed lane.
- **Live acceptance on the physical Orin**
  (`docs/evidence/2026-08-26-accept-orin-associate.txt`): known-answer and
  multi-step reasoning PASS, structured tool calls PASS, and a depth sweep at
  the incumbent's own shape — **52.5 tok/s decode at 32768-token depth with no
  decay across 0->32768, versus the board's llama.cpp GGUF cortex at 2.43 tok/s
  (~21.6x), TTFT 16.9 s versus 610.0 s (~36x), prefill ~1,612 versus ~64 tok/s.**
  Measured WITHOUT speculation; the vendor's "89 tok/s" agentic aggregate is
  deliberately not used as a comparator.

### Fixed

- **Opt-in core roles were silently un-hostable.** `compose_profile`/`_overlay`
  always took `feasible` from the card, so a shape hosting an
  `OPT_IN_CORE_ROLE` whose card abstains rendered a bare `*_FEASIBLE=false` with
  no model or knobs. Latent for `thor-muse`/`thor-worker` when force-applied to
  a non-native card; surfaced by `associate`.
- **The Orin card's W4A4 infeasibility claim is now per-checkpoint.** Lightning's
  own `hf_quant_config.json` is `W4A16_NVFP4` (weight-only) on the experts plus
  FP8 on `in_proj`/`out_proj` — not the W4A4 activation quantization that rules
  out the cortex and muse NVFP4 exports. sm_87 admits `modelopt_mixed` through a
  full Marlin stack; the W4A4 sentence still stands for the two checkpoints it
  really describes.
- Stale `v0.27.1`-on-Thor account in `docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md`;
  the Thor wedge is sm_110-specific — the Orin clears the identical Mamba2 SSD
  Triton warmup.

## [0.63.1] - 2026-08-25

### Added

- New opt-in `vllm-associate` fleet lane (lightning-on-orin plan, t7) giving
  NVIDIA's published Jetson serve recipe for
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` its eight previously
  unexpressible `vllm serve` flags a real home: `--mamba-backend`,
  `--mamba-ssm-cache-dtype`, `--enable-mamba-cache-stochastic-rounding`,
  `--mamba-cache-philox-rounds`, `--mamba-cache-mode`,
  `--enable-prefix-caching`, `--max-num-batched-tokens` (all env-parameterized
  knobs), and `--trust-remote-code` (hardcoded, matching every other lane).
  `ASSOCIATE_IMAGE` overrides the lane's image, proven by `docker compose
  config`. Gated behind the `associate` Compose profile; deliberately not
  wired into the Colleague role system yet (no `roles.py` entry, no gateway
  passthrough) — that lands in a later task.

### Fixed

- The Orin card's W4A4 infeasibility claim is now PER-CHECKPOINT (plan `lightning-on-orin`, t4). `lobes/machines/orin.py` and `lobes/profiles/builtin/orin.toml` used to blame one reason -- "NVFP4 quantizes ACTIVATIONS to FP4, sm_87 is Ampere" -- for all three NVFP4 generate lobes, naming `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` among them. That is false for Lightning: its own `hf_quant_config.json` (fetched 2026-08-25) declares `quant_algo` `MIXED_PRECISION`, with every routed and shared expert `up_proj`/`down_proj` at `W4A16_NVFP4` (`group_size` 16, WEIGHT-only, 16-bit activations), the mixer `in_proj`/`out_proj` at `FP8`, and `kv_cache_quant_algo` `FP8` -- structurally the same weight-only exception the board already runs for the int4 W4A16 `senses` gear. Both files now carry a named Lightning carve-out citing that config and the live sm_87 run (`docs/evidence/2026-08-25-spike-lightning-vllm-orin.txt`: vLLM v0.27.1 accepted `quantization=modelopt_mixed`, selected a full Marlin fallback stack plus FlashAttention 2, and served correct output at ~78-81 tok/s with `--kv-cache-dtype bfloat16` working around the real FP8 KV barrier). The W4A4 sentence itself is untouched and still names `unsloth/Qwen3.8-27B-NVFP4` and `nvidia/Gemma-4-31B-IT-NVFP4`, which it still describes; two new tests in `tests/test_profile_schema.py` pin both halves. No role feasibility changed -- `worker` stays `feasible = false` on this card

## [0.63.0] - 2026-08-25

### Added

- Docs, `lobes explain`, and CLAUDE.md coverage for the cortex replica pool (issue #199): the plural `<PREFIX>_PEER_ORIGINS`/`_PEER_API_KEYS` config family and `GATEWAY_SELF_ORIGIN`, live-probed compatibility (served id + quantization + max context + runtime, with `kv_cache_dtype`/parsers/speculative config informational), the deterministic least-load selection policy with `X-Lobes-Affinity` stickiness, the `X-Lobes-Served-By`/`X-Lobes-Route-Reason`/`X-Lobes-Route-Attempts` markers alongside the existing `X-Lobes-Proxied-By`, the compose passthrough + `lobes doctor` `gateway_passthrough` finding, the additive `replicas`/`fingerprint` capabilities fields and `--replicas` CLI view, and the failure/rollback table -- in `docs/gateway-fleet.md`, `docs/deployment-shapes.md`, `docs/colleague-stack.md`, `docs/openai-api.md`, `lobes/explain/catalog.py`, and `CLAUDE.md`

### Changed

- Every doc naming `hosted_by` or `X-Lobes-Proxied-By` now also names the replicas list, `X-Lobes-Served-By`, and `X-Lobes-Route-Reason`, and states explicitly that the pool is DECLARED/UNVALIDATED (#108) except `cortex` on the Spark+Thor NVFP4 pair, citing the pre-pool baseline transcript

## [0.62.1] - 2026-08-25

### Added

- Spec + plan for issue #199 — the cortex replica pool: one logical cortex
  served by N compatible replicas (Spark + Thor NVFP4 vLLM; the Orin's
  llama.cpp cortex exempt), every gateway a front, availability-aware
  selection from a cached peer `/status` snapshot, `X-Lobes-Affinity` as a
  preference, honest `X-Lobes-Served-By` / `X-Lobes-Proxied-By` /
  `X-Lobes-Route-Reason` markers, per-role comma-separated
  `<PREFIX>_PEER_ORIGINS` / `_PEER_API_KEYS`, and an operator-typed
  `GATEWAY_SELF_ORIGIN` (`docs/specs/2026-08-25-cortex-replica-pool-199.md`,
  `docs/plans/2026-08-25-cortex-replica-pool-199.md`). Scoped, thought,
  challenged and planned with devague; no code change in this PR.

## [0.62.0] - 2026-08-25

### Added

### Changed

### Fixed

## [0.61.2] - 2026-08-25

### Changed

- `tests/test_profile_render.py`'s quoting-layer test splits its composite `assert raw.startswith('"') and raw.endswith('"')` into two statements. The two ends fail for different reasons — a truncated write versus a value quoted on one side only — and a composite assertion reports neither, so the split is about diagnosability, not style.

## [0.61.1] - 2026-08-25

### Changed

- `speculative_config` is REFUSED at load for a role whose compose lane cannot expand `<PREFIX>_SPECULATIVE_CONFIG` (`schema.SPECULATIVE_CONFIG_ROLES` = cortex / senses / worker). `muse` hardcodes its `--speculative-config` token as a YAML list element, and `hand`/`embedder`/`reranker` carry no speculative flag at all, so declaring the knob there used to render an `.env` key nothing reads — a silent no-op. It now fails loudly with a message naming why, matching the existing rule that an unknown knob is a load error rather than a silently dropped one. Raised by review on PR #202 (Qodo finding 4).
- docs, CLAUDE.md and `spark-lobe.toml` no longer claim a per-box override for the DSpark default that does not exist. A shape override beats the card profile and a re-render force-writes the rendered key, so the honest routes are "select a different shape" or "fork the shape file" — the missing mechanism is tracked in #204 rather than papered over. Raised by review on PR #202 (Qodo finding 5).

### Fixed

- docs/evidence/2026-08-25-accept-spark-lobe-dspark-render.txt RETRACTS a false claim: it said an operator re-rendering the live box would drop the now-undeclared `PRIMARY_ALLOW_LONG_MAX_MODEL_LEN`. It will not — `.env` is merge-only, so a key the profile no longer renders is left in place, not removed. The error came from reading a fresh render, where nothing stale can survive. The transcript now also bounds its own argv-equivalence result to fresh renders. Raised by review on PR #202 (Qodo finding 3).

## [0.61.0] - 2026-08-25

### Added

- `speculative_config` — a new per-role machine-profile knob (`lobes/profiles/schema.py`, rendered to `<PREFIX>_SPECULATIVE_CONFIG`) carrying the lane's raw `--speculative-config=…` argv token. Until now a shape could declare a *window* and a *rope* opinion but not a *draft* opinion, so serving anything other than a checkpoint's own baked-in MTP head could only ever be a hand-edit to a rendered compose file. `None` still means "no opinion" (the template default applies); `""` still means "no speculative decoding at all".
- docs/evidence/2026-08-25-accept-spark-lobe-dspark-render.txt — a RENDER acceptance transcript (explicitly not a performance run): the three .env quoting spellings resolved through real `docker compose config`, a fresh `lobes init --shape spark-lobe --profile spark --apply` into a scratch dir, and its resolved cortex argv diffed against the live container's own `docker inspect` output — identical token sets.

### Changed

- **`spark-lobe` now declares the adopted DSpark @ 262144 cortex lane** (deviation d4, 2026-08-25), replacing the 1M-YaRN declaration: `max_model_len` 1048576 → 262144 (the checkpoint's native ceiling), `speculative_config` set to the RadixArk DSpark block-7 drafter with its revision pinned, and `allow_long_max_model_len` dropped as inert at exactly the native ceiling. `hf_overrides` (the YaRN rope block) is deliberately KEPT: every DSpark arm in the 2026-08-24 spike was measured with it in force, so removing it would ship a rope configuration nothing has been measured under. The prose regression this default carries is real and named — DSpark wins on code (46.20 vs 24.69 tok/s) and reasoning, loses on prose (13.71 vs 16.65) — and a prose-heavy box overrides the knob back.
- docs/qwen3.8-27b-nvfp4.md, docs/dspark-speculation.md, docs/deployment-shapes.md, CLAUDE.md: the 1M YaRN window is now described as the RETIRED validated shape (withdrawn, not disproven — it cannot be served alongside the DSpark drafter at `gpu_mem_util=0.58`), with the adopted 262144/DSpark pair as the current declaration. The 1M recipe and the older 0.44/262144 pair stay in-tree as documented rollbacks (cite-don't-delete).
- lobes/templates/fleet/env.example: the `PRIMARY_SPECULATIVE_CONFIG` block now spells out the VERIFIED retarget form instead of deferring to the `MULTIMODAL_*` caveat, and names the two spellings that fail silently.

### Fixed

- A `lobes init --apply` re-render of the DGX Spark can no longer silently revert its cortex lane. The 2026-08-25 DSpark adoption lived only as a hand-edit to the deployed compose file while the shape/profile knobs still said "mtp-n2 at `max_model_len=1048576`" — so a routine re-render would have either restored the incumbent MTP config or emitted DSpark argv at a window vLLM refuses outright (`needs 51.47 GiB KV; available 40.76 GiB`). The shape now renders an argv whose token set is identical to the deployed container's, proven by diff rather than by inspection.
- The `${PRIMARY_SPECULATIVE_CONFIG-…}` compose slot is unquoted (its default carries its own quotes), so a value substituted there must survive compose's dotenv parser AND its shell-lexer. Checked against real `docker compose config`: the bare-single-quote and unquoted spellings both degrade `{"method":"dspark",…}` to `{method:dspark,…}` — silently, with no error, and a boot failure far from the cause. The shape declares the one surviving form, and `tests/test_profile_render.py` models both parsers so a future edit cannot regress it unnoticed.

## [0.60.1] - 2026-08-25

### Added

- docs/dspark-speculation.md: a measured-here section recording the 2026-08-24 DSpark block-speculation spike on the deployed Spark cortex — vLLM DSpark loads and serves against the fleet's W4A4 NVFP4 target with no code/engine change, beats the incumbent MTP-n2 arm on code (+64%) and reasoning (+39%) but loses on prose (-27%), and beats the no-speculation floor on every shape and content type.

### Changed

- docs/qwen3.8-27b-nvfp4.md and docs/dspark-speculation.md: the 19.9-24.0 tok/s incumbent figure is now consistently cited as a DATED 2026-08-19 measurement (docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt), never as a live baseline, alongside the new DSpark numbers.

### Fixed

- `scripts/spec-arms.py` no longer presents an engine-wide acceptance aggregate as a per-request measurement (Qodo finding 9). Both surfaces now carry `scope: "engine_wide_over_shape_window"` / `request_scoped: false`, and the `/metrics` surface deltas `vllm:request_success_total` over the same window against the one request the tool issued — more completions than that marks the entry `contaminated: true` and prints `CONTAMINATED`. The `docker logs` surface has no request id at all, so its `contaminated` is always `null` (unknown) and can never be shown clean.
- `scripts/spec-arms.py` never labels a count of SSE deltas as `completion_tokens` (Qodo finding 3). One delta may carry several tokens, so the chunk count is now reported under its own `sse_content_chunks` key, `completion_tokens` stays `null` when the server omits `usage`, and the derived throughput is flagged `decode_tok_s_estimated: true` / `throughput_basis: "sse_chunk_count_estimate"` in JSON and `~N tok/s (est)` in the table.
- `scripts/spec-arms.py --max-seconds` is a real wall-clock deadline across the whole streaming read, not just urllib's per-operation socket timeout (Qodo finding 10). A server that keeps emitting chunks could previously reset that timeout forever and run a shape indefinitely; a shape that blows the deadline is now reported TIMED OUT rather than as a completed measurement.
- `scripts/spec-arms.py --combine` exits nonzero on any MISSING **or FAILED** cell, not only on a wholly absent arm file (Qodo finding 4). A transcript whose shape errored or timed out no longer yields a successful exit alongside a printed FAILED row, so automation cannot accept an incomplete three-arm comparison; `--allow-partial` is the explicit opt-in.

## [0.60.0] - 2026-08-24

### Added

- **`ask-colleague resume <task-id|last> [--detach]`** — pick a cut /
  timed-out / SIGTERM'd run back up from its persisted artifact, continuing on
  the original `colleague/<id>` work branch. `--detach` runs it under
  `setsid`/`nohup` and returns at once.
- **Per-seat thinking effort** for `ask-colleague` (colleague#416) —
  `--effort RUNG` (acting seat), `--seat-effort S=R` (any seat), `--role NAME`.
  `off` for small well-specified briefs, `xhigh` for open-ended judgement,
  `default` as the kill-switch.

### Changed

- **`ask-colleague` re-vendored byte-verbatim from `agentculture/colleague`
  @ 1.63.0** — all five files (`SKILL.md`, `scripts/ask-colleague.sh`,
  `prompts/{explore,review,write}.md`) match
  `diff -r ../colleague/.claude/skills/ask-colleague`.
- **Default colleague model is now `unsloth/Qwen3.8-27B-NVFP4`** (was the
  Qwen3.6 pin) — the lobes gateway on `:8001` no longer serves 3.6, so the old
  default only worked via colleague's auto-refresh warning path.
- `ask-colleague review` front-loads a filtered, capped diff into the review
  instruction so the model does not spend turns running `git diff` itself.

## [0.59.1] - 2026-08-23

### Added

### Changed

### Fixed

## [0.59.0] - 2026-08-23

### Added

- `_compose.MERGE_ONLY_FILES`, `scaffold_action()`, `env_keys()` and `merge_env_template()` — the merge-only machinery, with `MERGE_ONLY_FILES` naming the invariant so a symmetry-minded refactor has to delete a constant to break it.

### Changed

- `lobes init --apply` no longer aborts when a deployment is already scaffolded: existing template files are SKIPPED instead. Refusing outright was what pushed operators toward `--force`, and `--force` is the flag that used to destroy `.env` — the guard against a small accident was manufacturing a much larger one. A shape/profile re-render (`lobes init --shape spark-lobe --apply`) is now a safe, routine operation on a live box.
- `lobes init`'s dry run now names the exact per-file action — `create` / `merge` / `overwrite` / `skip` — in both the human plan and `--json` (new `action` key per file), instead of the blanket, now-false "needs --force to overwrite" note.
- `--force` help text states what it does and does not touch: it overwrites compose/Dockerfiles/plugin, never `.env`.

### Fixed

- `lobes init --apply --force` no longer TRUNCATES an existing `.env`. It is now merge-only always: missing template keys are appended, every existing line is left untouched. Previously `write_scaffold` rewrote every template including `.env`, destroying operator-typed state no template can regenerate — `GATEWAY_API_KEY`, every `*_PEER_ORIGIN`/`*_PEER_PROXY`/`*_PEER_API_KEY`, the per-role `*_FEASIBLE` declarations, `COMPOSE_PROFILES` and `HF_TOKEN`. Reproduced on a live DGX Spark deployment: 89 env keys in, 72 out, with the senses->Orin proxy, the embed->Thor proxy and the inbound auth gate all silently un-wired. This is what reverted that box's cortex from its 1M YaRN window to the card-profile 128K default on 2026-08-20, and the same trap was recorded a month earlier (eidetic `lobes-init-apply-is-not-a-repair-path`, 2026-07-17). Reset `.env` deliberately by deleting it first.
- `write_plugin_file` mirrors the same contract: an existing tool-parser plugin file is left alone without `--force` rather than aborting the command, and returns `None` when it skips so callers cannot report a file they did not write.
- `.env` permissions are re-hardened to 0600 on the MERGE path too. Overwriting used to chmod as a side effect; merge-only would otherwise have left a drifted, world-readable `.env` — with an API key in it — untightened.
- `lobes init --apply` reports what actually changed: `.env` is always listed on the fleet path (it is always rewritten with the shape/profile knobs, `LOBES_PROFILE` and `MODEL_GEAR_VERSION`), and a skipped tool-parser plugin is no longer listed as written.

## [0.58.4] - 2026-08-23

### Added

- docs/qwen3.8-27b-gguf-llamacpp.md — three sections measured *after* #193 merged:
  - **Agentic clients and prompt caching.** The `qwen` CLI sends ~35 000 tokens of
    preamble per turn, which cold costs ~150 s of prefill. llama.cpp's prompt cache
    (on by default) cuts warm turns to ~0.3 s — a ~500x reduction — so the preamble
    is paid **once per session, not per turn**. This retracts an earlier note that
    called agentic use impractical; that note generalised from two consecutive
    *cold* turns and was wrong.
  - **What was NOT compared: NVFP4.** It was never benchmarked on this box and
    cannot be (W4A4 needs Blackwell FP4 tensor cores — the reason this lane
    exists). Cross-engine quality parity rests on published benchmarks, not on
    anything measured here, and the doc now says so.
  - Throughput under real agentic load (8.2 tok/s) matches the synthetic benchmark
    (8.46 tok/s), so the lane behaves as characterised when a genuine client drives it.

### Changed

### Fixed

## [0.58.3] - 2026-08-23

### Added

- docs/plans/2026-08-22-qwen3-8-gguf-llamacpp.md — converged build plan for the
  Orin-local llama.cpp cortex: 11 tasks in 8 dependency waves, every task
  carrying TDD acceptance criteria and operator instructions, covering all 32
  spec targets. Wave 0 is the spike (t1) plus the Tegra iowait fix (t7); the
  remaining tasks are conditional on a GO, and a NO-GO ends the plan with a
  recorded verdict (risk r1). Also recorded: unmeasured concurrency (r2),
  expected feature losses versus the vLLM lane — MTP, ViT, preserve_thinking,
  strict tools (r3), out-of-scope mesh senses re-homing (r4), and the
  second-cortex-host routing question (r5). Plan state in
  `.devague/plans/qwen3-8-gguf-llamacpp.json`.

## [0.58.2] - 2026-08-23

### Added

- docs/specs/2026-08-22-qwen3-8-gguf-llamacpp.md — converged devague spec: the
  Jetson AGX Orin (sm_87) gains a LOCAL cortex serving
  `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` via a llama.cpp lane, replacing the
  `senses` Gemma 4 12B on that box (operator decision; mesh senses re-homing is
  out of scope). Spike-gated per the TRT-LLM-investigation pattern: live
  load/correctness/tok/s/context on the box before any integration lands.
  Measurable gate: decode >= 5 tok/s single-stream, context >= 32768 served and
  needle-probed, known-answer + tool-calling probes PASS through the gateway,
  evidence under docs/evidence/. Frame state in
  `.devague/frames/qwen3-8-gguf-llamacpp.json` (10 scope entries, 19 confirmed
  claims, 12 honesty conditions).

## [0.58.1] - 2026-08-21

### Added

- docs/evidence/2026-08-21-measure-reasoning-effort-cortex-spark.txt — a read-only COST measurement of the Qwen3.8 cortex reasoning_effort knob on the Spark (4 prompts x low/medium/xhigh/thinking-off, temperature=0, n=1 per cell).
- docs/qwen3.8-27b-nvfp4.md: a Reasoning effort section recording that the checkpoint template defaults to xhigh, that high is an alias for xhigh, and that the mechanism is an injected sentence rather than a token budget.
- docs/openai-api.md: a caller-facing Controlling the thinking trace subsection covering enable_thinking vs reasoning_effort, the 400 on an unrecognised value, the per-key merge over the server default, and the message.reasoning / reasoning_tokens response fields.

### Changed

- Documented that lobes sets reasoning_effort nowhere, so every cortex request runs at the template default xhigh. No default was changed: the measurement found lower effort is not monotonically cheaper (2 of 4 prompts cost more) and degrades instruction-following, while enable_thinking:false cut 75% of completion tokens against low s 24%.

## [0.58.0] - 2026-08-20

### Added

- Lightning worker + d1 topology swap (plan `nemotron-lightning-worker`, #187/#186/#183, deviation d1): the `worker` role serves `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` — VALIDATED on the DGX Spark GB10 (75.1 tok/s, ~75 ms TTFT, `nemotron_v3` + `qwen3_coder` pair) after a Thor sm_110 NO-GO (Mamba-2 SSD warmup wedge on both the fleet nightly and upstream v0.27.1); the Thor serves `cortex` locally (`unsloth/Qwen3.8-27B-NVFP4` @ the full 1M YaRN window, MTP off — sm_110 ships no GDN-MTP kernel). Nine evidence transcripts under `docs/evidence/2026-08-20-*`.
- `PRIMARY_SPECULATIVE_CONFIG` / `WORKER_SPECULATIVE_CONFIG` (set-but-empty = MTP off, the senses-lane mechanism) and `WORKER_REASONING_PARSER` (default `nemotron_v3`); the primary and worker compose lanes move to shell-lexed string commands, fixing a Compose brace-interpolation hazard with `${VAR:-{}}` defaults.
- `hand` becomes referable/proxyable: the `NEVER_PROXIED_BACKENDS` decision is REVERSED by d1 evidence (LFM2.5 inference is corrupt-then-fatal on Thor sm_110 while identical config passes on the Spark — #181 re-attributed); `HAND_PEER_ORIGIN`/`_PROXY`/`_API_KEY` land in both the config channels and server resolution tables together.
- `docs/worker-lightning-rollout-notes.md` (served-id audit: no external raw-id pinners), Lightning per-model doc, and the sm_110 non-dense-decode lesson in `docs/machine-profiles.md`.

### Changed

- `worker` responsibilities per #187: text-only, non-coding — `image_understanding`/`video_understanding` dropped; `repo_inspection`, `run_authorized_commands`, `action_selection`, `retrieval_synthesis`, `summarization`, `log_digestion`, `structured_extraction` added; `code_authoring` forbidden. `unsloth/Qwen3.6-35B-A3B-NVFP4` demoted to candidate (final production baseline 61.2 tok/s captured pre-swap).
- `thor.toml` cortex tracks the fleet default again (`unsloth/Qwen3.8-27B-NVFP4`, validated at 1M on the physical Thor); the spark↔thor divergence set returns to hardware-only.

## [0.57.2] - 2026-08-20

### Added

- `docs/plans/2026-08-20-nemotron-lightning-worker.md` — converged 16-task build plan (+ plan state) from the #188 spec: Lightning worker swap gated on sm_110/nemotron_h spikes with baseline-before-flip ordering, cortex repoints on Thor+Orin (#186), and hand budget probes on Thor+Spark (#183).

## [0.57.1] - 2026-08-20

### Added

- `docs/specs/2026-08-20-nemotron-lightning-worker.md` — converged devague spec (+ frame state) for the Lightning worker swap (#187), the cortex proxy repoint (#186), and hand budget validation (#183); the challenge pass ran rigorous (hardware + distributed-state + migration signals).

## [0.57.0] - 2026-08-20

### Added

### Changed

### Fixed

## [0.56.6] - 2026-08-19

### Changed

- Bumped the shared vLLM nightly digest for all six Qwen-lane fleet services (vllm-primary/embed/embed-deep/rerank/hand/worker) in one commit, flipped the fleet default primary to `unsloth/Qwen3.8-27B-NVFP4`, and wired the spark-lobe shape's 1M-token YaRN hypothesis (`hf-overrides` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN`) — see docs/plans/2026-08-19-qwen3-8-cortex-upgrade.md task t5.

## [0.56.5] - 2026-08-10

### Fixed

- Sonar S9073: a composite assertion in the collision-message test is now a single substring check — the two halves were only separate because the expected text wraps across two source lines.

## [0.56.4] - 2026-08-10

### Fixed

- CI lint: two markdownlint errors in my own CHANGELOG entries — `hand:<domain>` parsed as inline HTML (MD033) and bare snake_case identifiers parsed as emphasis markers (MD037). Both came from writing changelog prose with unbackticked code. My local markdownlint run had passed because I ran it BEFORE the version bump appended the entry.
- Sonar S8997 x2: the adapter-probe test swapped a module global by hand; it now uses the monkeypatch fixture, and a sibling test drops its inline contextlib/io redirect for the capsys fixture the rest of the module already uses.

## [0.56.3] - 2026-08-10

### Added

- Collision detection now covers LoRA adapter names, not just served names (Qodo #184-1): `_backend_for` matches `requested in backend.adapters`, so an adapter name is an ownership claim exactly like a served name — a collision between the two resolved silently by backend order. The warning also names the right remedy (`HAND_LORA_MODULES` vs `*_SERVED_NAME`) for whichever kind of duplicate it found.
- A new warning for an adapter name shadowed by an alias: `resolve_model` checks aliases FIRST, so an adapter named after a tier, role or operator alias is unreachable by its own name — a total shadow rather than an order-dependent race.

### Changed

- `build_config` cognitive complexity 18 -> well under the 15 limit (Sonar S3776) by extracting three alias derivations into named helpers: `_hand_adapter_aliases`, `_add_self_named_opt_in_aliases` and `_add_pooling_role_aliases`. Behaviour-preserving — the existing alias tests pass unchanged.

### Fixed

- The hand backend comment claimed `HAND_SERVED_NAME` alone wires the lane (Qodo #184-2). It does not — `_optional_backend` requires `*_BASE_URL`, by its own documented contract, since a served name with no URL describes a model rather than a reachable backend.

## [0.56.2] - 2026-08-10

### Added

- 29 tests over the hand LoRA-adapter honesty surface — the declaration parser (HAND_LORA_MODULES: partition-not-split so a path keeps an =, malformed segments dropped, dedupe, whitespace), the engine probe (intersection, undeclared ids ignored, empty declaration opens no socket, correct path, no API key, and every fail-closed mode: non-200, unreachable, malformed body), the ReadinessCache background refresh (empty seed, copy isolation, a raising probe degrading to empty without aborting the pass), the `hand:<domain>` alias derivation, and the `/v1/models` filter (declared-but-unconfirmed is invisible; an adapter cannot outlive its lane).

### Changed

- SonarCloud new-code coverage 51.2% -> 100%. The gate exposed a real gap rather than a metric one: the adapter surface — the honesty machinery this release is largely about — had NO direct tests. The delivery summary claim that it was covered offline was overstated and is corrected.

## [0.56.1] - 2026-08-10

### Added

- docs/evidence/2026-08-10-partial-hand-spark.txt — hand on the DGX Spark GB10: functional PASS on a third card (Lfm2ForCausalLM, bf16 sentinel, no reasoning parser, LoRA armed, known-answer, structured tool_calls, empty-inventory /v1/models, undeclared adapter 404), plus the first live confirmation that the d8 --attention-config fix resolves.

### Changed

- hand budget guidance now names the MECHANISM behind every non-reproducing budget: on a unified-memory card with co-resident tenants vLLM profiles against memory free AT THAT INSTANT, so the same gpu_mem_util yields a different KV pool run to run. Three Spark boots at the identical 0.06 gave 6.21 / 3.34 / 3.54 GiB. This retro-explains the Orin retraction as a property, not a fluke — and it is not a tight-margin artifact, since the Spark reproduced the 2x spread with several GiB of headroom. All four cards stay DECLARED (#183).

## [0.56.0] - 2026-08-10

### Added

- **`hand` — the NINTH Colleague role** (`LiquidAI/LFM2.5-1.2B-Instruct`), the fleet's designated **fine-tuning base**. "Muscle memory": one cheap base, many LoRA adapters, each mastering a domain. Where `worker` is an untrained generalist doer, `hand` is a trained specialist. At ~1.2B (~2.4 GiB bf16) it is cheap enough to co-reside on **every card**, so unlike `muse`/`worker` it is **default-hosted by every built-in shape** and **never proxied to a peer** (`NEVER_PROXIED_BACKENDS` names that absence so a symmetry-minded refactor must delete a constant to break it). Responsibilities `domain_mastery`/`learned_skill`/`specialized_task`/`tool_use`; forbidden `final_decision`/`repo_action`/`security_decision`. See `docs/lfm2.5-1.2b-hand.md`
- **LoRA adapter serving** — the `vllm-hand` lane ships **armed** (`--enable-lora`, `--max-loras=4`, `--max-lora-rank=32`) with the inventory **empty**: v1 has zero adapters. Adapters are declared once in `HAND_LORA_MODULES` (`name=path`, comma-separated), read by BOTH the engine's `--lora-modules` and the gateway's alias derivation so the two cannot disagree, and fixed at boot — there is no runtime hot-load. `model=hand` serves the base and never 404s on an empty inventory; `model=hand:<domain>` serves that adapter; an UNdeclared `hand:<domain>` is refused with `model_not_found`, never silently downgraded to the base
- **Adapter honesty (#92 for adapters)** — a declared adapter is advertised on `GET /v1/models` and `/capabilities` only once the lane's OWN `/v1/models` confirms the engine loaded it (`_readiness.probe_backend_adapters`). Deliberately asks the ENGINE, not the filesystem: adapter paths are mounted into `vllm-hand`, not the gateway, so a path check there would false-negative every correct config while still missing the failures that matter (unreadable file, rank above `--max-lora-rank`, a checkpoint vLLM refused)
- **`lfm2` tool-call parser** — LFM2 emits `<|tool_call_start|>…<|tool_call_end|>`, whose delimiters are **special tokens**: the same trap that made `pythonic` silently wrong for Gemma 4. vLLM's purpose-built `lfm2` parser resolves both delimiters in `__init__` and **raises** when either is missing, so a bad tokenizer revision fails loudly at startup rather than relaying a well-formed call as prose. No `--reasoning-parser`: this checkpoint has no thinking mode (LiquidAI ships `LFM2.5-1.2B-Thinking` separately), so unlike the cortex and Gemma 4 lanes there is no second half to pair with
- `docs/lfm2.5-1.2b-hand.md` — the per-model reference, and an **"adding a role is effectively irreversible"** callout in `docs/colleague-stack.md` enumerating the surfaces a role name lands on
- **VALIDATION STATUS: `hand` is DECLARED on every card and VALIDATED on NONE (#108).** Stated here rather than left unsaid, because two live findings in this release were *only* visible on hardware. The Orin **served** and passed every functional check — a structured `tool_calls` array through the `lfm2` parser, the tokenizer's `<|tool_call_start|>`/`<|tool_call_end|>` confirmed as special tokens (ids 10/11), the bf16 sentinel, a correct known-answer completion, an unknown id refused 404 — but its **budget did not reproduce**: three boots at the identical `gpu_mem_util=0.06` / `max_model_len=32768` profiled 2.7 GiB, 0.14 GiB and 0.09 GiB of available KV (one success, two refusals), because vLLM clamps against actual free memory on a box that also hosts `senses` and unrelated workloads. The Thor boot failed outright in LoRA embedding-slot allocation, cause unattributed. `docs/evidence/2026-08-10-partial-hand-orin.txt` (a PARTIAL record carrying its own retraction), `docs/evidence/2026-08-10-hand-lobe-budget-derivation.txt`, and issues #181/#182/#183

### Changed

- **`hand` replaced `Qwen/Qwen3.5-4B` as the `minor`/`cheap` tier.** Both spellings still work and now resolve to `hand`; the 4B stays in the catalog as a plain `candidate` (cite-don't-delete), still selectable via `lobes switch` and still runnable as the opt-in `vllm-minor` compose service, but no tier alias resolves to it any more — it is addressable only by explicit model id, exactly like the legacy 14B `middle` gear. Capability order is now `hand` < `multimodal` < `worker` < `muse` < `main`
- **`hand` is the pressure-policy servable floor.** Under swap > 75 % / iowait > 50 %, `cortex`/`senses`/`worker`/`muse` still shed with 429 + `Retry-After` and `hand` is always served — the floor's PROMISE is unchanged, only its name (a `minor`/`cheap` request normalizes to `hand`). `servable_tier` therefore reports `"hand"` where it previously reported `"minor"`
- `colleague-stack` is the **seven** default-hosted roles (`hand` joins; `muse`/`worker` stay excluded, both being opt-in-hosted and compose-profile-gated)
- `mg-logwrap` now drops an argument that is exactly `--flag=` with an empty value. A compose `command:` list cannot omit an argument conditionally, so an unset templated flag renders as a bare `--flag=` — and vLLM would parse `--lora-modules=` as a malformed `name=path` pair. The rule is narrow by construction: a flag with a value, a bare `--flag`, a lone `--`, a short `-x`, and every non-flag argument all pass through untouched
- Role-count prose swept to nine across `docs/`, `CLAUDE.md` and `README.md`. The sweep also corrected counts that were **already stale before this change** and had never caught up with `worker` — `lobes/explain/catalog.py` said SEVEN throughout, and several `capabilities`/`measure`/`learn` strings said six or seven
- `docs/qwen3.5-4b-minor.md` re-headed as the DEMOTED gear, with a note to read its (never-realised) LoRA promises in the past tense — that unfulfilled plan is precisely why the role moved

### Fixed

- **The `vllm-hand` lane shipped unbootable AND silently misconfigured — two defects no offline test could see.** (1) It omitted `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, which every other lane on the same nightly image sets; live, the lane profiled to **`Available KV cache memory: -9.25 GiB`** — negative, so no boot at any util. (2) It never substituted `HAND_ATTENTION_BACKEND`, a **dead knob**: `builtin/orin.toml` declares `attention_backend = "TRITON_ATTN"` and `lobes init` renders it into `.env`, where nothing read it — the operator sees a configured backend the engine never receives. Both were caught by booting the real committed compose lane; both rendered valid compose and passed the full suite. The second now has a class-level guard (`test_every_rendered_profile_knob_is_substituted_by_the_fleet_template`, verified to fail with the fix reverted); the first's equivalent is #182 and does not exist yet
- `lobes measure` would have raised `KeyError` on any role missing from `roles_measure._FAMILY_BY_ROLE` — a crash, not a degraded reading. Both that map and `_MEASURE_FN` now cover every role, and a parametrised test iterates `ROLES` so the class cannot recur
- `build_role_registry` iterated a **hand-typed copy** of the gateway-fronted roles instead of deriving them from `ROLES`, so a new role could be registered in all six per-role tables yet be silently missing from the registry the CLI and `GET /capabilities` both read. Now derived (`GATEWAY_FRONTED_ROLES`), with a parametrised completeness test per table — the same half-landed-role failure mode that made `WORKER_PEER_PROXY=true` inert in 0.54.6

## [0.55.1] - 2026-08-10

**Three surfaces still named the checkpoint 0.54.9 demoted.** The multimodal
`cortex` promotion swapped the served id to `unsloth/Qwen3.6-27B-NVFP4` and left
`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` behind as a text-only *candidate* —
but `whoami`'s fallback, the agent's own system prompt, and two README passages
kept describing the demoted one as what this box serves. Docs-and-fallback only;
no runtime routing changed.

### Fixed

- **`lobes whoami` reported a model the box 404s on.** `_DEFAULT_MODEL` — what `whoami` prints as "served" when there is no scaffold, or when `VLLM_SERVED_NAME` is unset — still named the demoted `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`, while the gateway's own fallback (`lobes.gateway._config._DEFAULT_PRIMARY`) had moved to `unsloth/Qwen3.6-27B-NVFP4` in 0.54.9. On an unscaffolded box the two answers disagreed, and `whoami`'s was the wrong one: a caller pinning the id it printed gets `model_not_found`. The two constants must track each other; the comment now says so.
- **`AGENTS.md` told the deployed agent it was text-only.** The runtime paragraph — the `acp` system prompt the lobes agent actually runs on — described the demoted checkpoint, its grafted MTP head, and a text-only capability set. It now describes the promoted one (self-hosted MTP baked into the checkpoint, compressed-tensors NVFP4) and states the capability that changed: the agent takes **image and video** input through the checkpoint's own ViT, so "look at this screenshot" is work it can do rather than hand off. An agent that does not know it can see will keep referring vision away.
- **`README.md` taught callers the addressing pattern that just broke.** The fleet-routing example pinned a raw checkpoint id — the exact id 0.54.9 retired — so copying it yielded a 404; it now addresses the stable role name (`model: "cortex"`) and says why. The example's comment also claimed an unknown model "defaults to the primary"; only a **missing** `model` field does that, an unknown id 404s `model_not_found`. The `culture.yaml` snippet in the identity section was updated to the served id as well.

## [0.55.0] - 2026-08-04

**The Jetson AGX Orin becomes a first-class card.** `orin` joins `spark`/`thor`
as a detected card with its own built-in profile and a new `orin-lobe`
deployment shape serving `senses` on the QAT checkpoint
`unsloth/gemma-4-12B-it-qat-w4a16`. `spark.toml`, `thor.toml`, and every
pre-existing golden are **byte-identical** — this release adds a card, it does
not move the Blackwell fleet. **The orin variation is DECLARED/UNVALIDATED
(#108): no physical board has booted it**, so its budget knobs are marked
MEASURED-PENDING and no doc, support table, or `lobes capabilities` output may
claim otherwise. Delivery record:
`docs/deliveries/2026-08-04-unsloth-qat-senses-first-class-orin-variation.md`.

### Added

- **`orin` card detection (t2).** `lobes/machines/orin.py` registers the Jetson AGX Orin 64GB (Ampere **sm_87**, 61.3 GiB unified) — one module plus one `register()` line, per the `_registry.py` convention. Verified live on a physical board: `detect_card()` resolves `orin` from the real nvidia-smi name, compute capability, device-tree model, and hostname; an unrecognised card still falls through to `base`. The pooling-gear divergences `docs/orin-profiles.md` records (`TRITON_ATTN`, reranker `enforce_eager`) are declared as orin's **own** `role_overrides` and deliberately **not** by composing the `SM_110` trait — those values are an untested carry-over from Thor, and attributing them to an sm_110 trait on an sm_87 board would misstate why they hold.
- **Built-in `orin` card profile (t6)** — `senses` = `unsloth/gemma-4-12B-it-qat-w4a16` (compressed-tensors **int4** W4A16 — weight-only, so Ampere qualifies without Blackwell FP4 tensor cores), `cortex`/`muse`/`worker` vetoed on the FP4-activations-need-Blackwell line. `gpu_mem_util = 0.45` / `max_model_len = 262144` are **hypotheses pending a live boot**, annotated with why 0.45 is not simply inherited (see #171 below).
- **`orin-lobe` deployment shape (t7)** — hosts `senses` + `embedder` + `reranker` and **no local `stt`/`tts`** (the Parakeet base image ships no sm_87 kernels; audio forwards to a peer via `AUDIO_URL`). Its `[overrides.senses]` carries the orin budget, fixing a real clobber: rendering `thor-lobe` over the orin card yielded Thor's `0.30`/`131072`, which the deployed box had been hand-patching in `.env`. A lockstep test pins the shape's overrides to `builtin/orin.toml`, so a live-boot backfill of one file without the other fails CI.
- **`Profile.host_env` (t7)** — an optional card-level table for non-role env keys. Used to persist `LOBES_IOWAIT_DEGRADED_THRESHOLD=100` on Tegra, where the `sugov:*` cpufreq kthreads inflate `/proc/stat` iowait to ~59% with **zero** disk I/O and the gateway then 429-sheds all `senses` traffic indefinitely. Declared on the **card**, not the shape, so a bare `lobes init` (`machine-as-brain`) is fixed too — that is the default path on this hardware. Values are strings and render *before* role keys, so a `host_env` entry can never shadow a measured budget.
- **`unsloth/gemma-4-12B-it-qat-w4a16` catalog entry + `docs/gemma-4-12b-qat-w4a16.md` (t1).** Every field cited from the checkpoint's own `config.json`: `Gemma4UnifiedForConditionalGeneration`, int4 pack-quantized (**not** the incumbent's FP4 path), `max_position_embeddings=262144` (double the incumbent), `vision_config` + `audio_config` present, and a **`video_token_id`** the repo's existing "text+image+audio" Gemma line predates. No MTP draft ships with it. Every capability row in the doc is marked **pending-live-probe**; nothing is claimed from the model card.
- **`gpu_access` — a card-declared knob so csv-mode GPU access survives a re-render (Orin variation t8).** A `Profile` may now declare `gpu_access = "devices"` (the default: the compose templates' own `deploy.resources.reservations.devices` request) or `"runtime"`. On a board whose NVIDIA container toolkit resolves to the **legacy csv mode** — measured live on the Jetson AGX Orin (toolkit 1.19.1, `mode = "auto"`; `docs/orin-profiles.md` divergence 1) — the `deploy.resources` request fails at container **create** with *"invoking the NVIDIA Container Runtime Hook directly … is not supported"*. Declaring `"runtime"` makes `lobes init --apply` GENERATE two compose overrides, `docker-compose.gpu.yml` + `docker-compose.gpu-audio.yml`, which `!reset` each GPU service's `deploy:` stanza and set `runtime: nvidia` instead — the same override mechanism `docker-compose.shape.yml` already proved, chosen because docker-compose has **no conditional-block syntax**, so no `${VAR}` substitution in one template can express both forms. The `orin` built-in opts in; every other card is untouched. Two files rather than one because a compose override may only name services some file in the same `-f` chain declares, and the audio sidecars live in the opt-in audio overlay that `lobes up <non-audio-role>` deliberately leaves out. The pair is rewritten on **every** `--apply` and **removed** when the resolved card no longer declares it — which is the entire point: the deployed Orin had been running the equivalent as a hand edit of its scaffolded compose files since 2026-07-16, and a re-init silently reverted it. `gpu_access` renders **no `.env` key at all**, so every profile and shape golden — `orin`'s included — is byte-unchanged. **UNVALIDATED (#108):** `docker compose config` on the real csv-mode board renders and merges the overrides correctly, which proves the compose merge, not a container create; only a live boot can do that. See `docs/machine-profiles.md#gpu_access---how-this-boards-container-runtime-is-asked-for-the-gpu`.

### Changed

- **The senses lane's `--speculative-config` is now parameterized (t5)** — `MULTIMODAL_SPECULATIVE_CONFIG`, defaulting to the existing `google/gemma-4-12B-it-assistant` MTP draft so an unset value renders **byte-identically** to before. An explicit empty value omits the flag **entirely**, which is what makes a draft-less checkpoint servable at all: the replacement senses checkpoint ships no MTP head, and the recorded decision is to drop speculative decoding rather than block the swap. The lane's `command:` became a shell-lexed string because compose cannot conditionally omit a list item, and the dash-only default operator (`${VAR-default}`, never `:-`) is load-bearing — the colon form treats empty as unset and would substitute the draft straight back, making the off-switch unreachable. The `test_catalog.py` byte-guard was **strengthened, not deleted**: it now pins the catalog item as the knob's full substitution, closing a hole where prose in a comment would have satisfied the old substring check.
- `lobes/profiles/loader.py` now derives **orin's** role overrides from the `lobes.machines` registry as well as thor's, so `builtin/orin.toml` stays silent on those knobs instead of re-typing the literals — the same single-source-of-truth arrangement `thor.toml` already used for its sm_110 divergences.
- `lobes.runtime._compose.compose_file_args` — the single `-f` chain authority (#137) — gained a `gpu` dimension, pairing each GPU-access override with the compose file whose services it patches (base half after `docker-compose.yml`, audio half after `docker-compose.audio.yml`). A csv-mode deployment with no other overlay is no longer a "plain" deployment: it now resolves an explicit chain rather than the bare argv, because the bare argv would silently drop the override and every gear would ask for the GPU the way that board refuses. With no `gpu_access` declaration anywhere — every pre-existing deployment — the chain is byte-identical to before.

### Fixed

- **The test suite was host-dependent (#172).** `tests/test_init.py` and `tests/test_cli_logs.py` called `main(["init", ...])` without mocking hardware detection, so they inherited the runner's **real** GPU card. This passed only because CI is not a recognised card — landing `orin` detection immediately turned 20 of them red on a physical Orin. Both files now pin detection with an autouse fixture, and a parametrized test proves the **injected** card drives the outcome rather than the host. No assertion was weakened and no test skipped. The same latent break would have fired for anyone running the suite on a Spark or a Thor.
- `lobes init --shape` help no longer hand-lists shape names — it derives them from `builtin_shape_names()`. The hand-written list had already gone stale, omitting `thor-worker`.

### Known issues found while building this release

Filed, not fixed here — each is a pre-existing defect this work surfaced:

- **#171** — `docs/orin-profiles.md` overstates the Orin senses KV pool by **1.67x** (18.86 GiB / 802,644 tok documented; **11.29 GiB / 480,431 tok / 3.67x** measured at identical knobs). `gpu_mem_util` is a fraction of the **entire device**, so every byte a co-resident engine holds is deducted one-for-one from KV — the documented figure was measured senses-**first** into an empty device. Boot order changes the served budget, not just OOM risk.
- **#174** — `lobes init --force` **silently destroys operator-typed `.env` keys** (12 lost on a real deployment, including the cortex peer-proxy origin, flag, and credential). Contradicts `doctor --fix`, which never rewrites an existing `.env` line.
- **#175** — an operator profile **silently shadows** a newly-shipped built-in of the same name, rendering an upgrade as a no-op with no warning.
- **#173** — the field traps this work measured (`reasoning` vs `reasoning_content`; `delta.reasoning` on streaming while `delta.content` stays empty; opt-in `enable_thinking`; the SSE chunk-count throughput trap) are discoverable only in evidence transcripts — proposes `lobes help <role>` and a `GET /help` endpoint.
- **#101 correction** — the documented audio silent-drop signature ("~19 placeholder tokens") is **wrong on this build**: audio tokens scale with duration (~25 tok/s). The conclusion still holds — a discrimination control shows the model reports hearing nothing while carrying audio tokens — but the token-count test will mislead.

## [0.54.9] - 2026-07-31

### Changed

- **BREAKING (served id): `cortex` is now `unsloth/Qwen3.6-27B-NVFP4` — a MULTIMODAL primary.** Promoted after a live GB10 boot; the outgoing `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` is demoted to `candidate` and kept (it remains the only text-only 27B). The fleet's reasoning/final-authority lobe can now take image **and** video input through its own ViT — previously the role contract had *no* role that both sees an image and decides, since `senses` is forbidden `final_decision`/`repo_action` and `worker` is forbidden `final_decision`/`security_decision`. **Callers pinning the old raw id get a 404**; migrate to the stable `cortex`/`main`/`hard` aliases. MEASURED (`docs/evidence/2026-07-31-accept-multimodal-cortex-spark.txt`): boots first try at the spark-lobe shape's existing `0.44 / 262144` with no retune, KV pool 26.39 GiB / 756,642 tokens ≈ 2.89× concurrency at the full 256K window (the unquantized bf16 ViT costs ~132,300 tokens, ~15%), decode 15.0/15.5/18.4 tok/s at TTFT ~0.27 s, self-hosted MTP at 62–67% acceptance. Vision, video (reversed-motion control), thinking, `preserve_thinking` (#93) and strict tool calling with thinking on (colleague#320) all re-verified. The lane converges on the validated `worker` lane's shape: `--quantization` `modelopt`→`compressed-tensors`, spec-config `qwen3_5_mtp`/3→self-hosted `mtp`/2, and **both** `--language-model-only` and `--tokenizer=<override>` dropped — so `mtp_compose_command_items()` is now two items, not four.
- `spark.toml` / `thor.toml` take the new model, but their machine-as-brain **duo** budget (`0.30 / 131072`) is explicitly marked **inherited and unmeasured with a ViT** — it was measured for the text-only export, and this box runs `spark-lobe`, so that configuration was never booted. `thor.toml` takes the model to preserve its "cortex differs from spark's in `kv_cache_dtype` alone" contract; the checkpoint itself is unvalidated on sm_110 (#108).

- `docs/qwen3.6-27b-nvfp4-multimodal.md` — replaced the "open, unmeasured" section with **measured** live-GB10 results. Booted first try at the incumbent's own `0.44 / 262144` (no retune, unlike `thor-muse`): KV pool 26.39 GiB / 756,642 tokens ≈ 2.89× concurrency, so the unquantized bf16 ViT costs ≈132,300 tokens (~15%) of KV pool and the full 256K window survives. Decode 14.9 / 16.4 / **19.0** tok/s (TTFT ~0.27 s); self-hosted MTP engages at 62–67% draft acceptance. All behavioural gates pass: vision and **video** (directional-motion probe with the reversed clip as control), thinking, `preserve_thinking` (#93, +800-token two-turn delta), and strict tool calling with thinking on (colleague#320). The page now also states plainly that the throughput comparison against the incumbent is **not controlled** — those numbers came from a different vLLM build.

### Added

- **Pooling ROLE-IDENTITY aliases** — `embedder` / `reranker` (and the backend names `embed` / `rerank`) are now addressable on `/v1/embeddings` and `/v1/rerank`, mirroring what `cortex`/`senses` give the generate lane. `tier_aliases` is generate-only, so the pooling gears previously had **no stable address at all**: every embed consumer had to pin a concrete served id with nothing to migrate to when it changed. Same no-fallback contract as `embed-deep` — a missing gear means the alias is **absent** (honest 404), never a substitution, because an embedding served from a different model answers in the wrong vector space (the 0.6B is 1024-dim, the 4B is 2560-dim).
- **`docs/model-switch-playbook.md`** — how to swap a served checkpoint without guessing, written from the 2026-07-31 `cortex` candidate run. Records the ordering that matters (benchmark the *incumbent* first, on today's engine — that baseline is unrecoverable once the model is gone), the consumer-breakage audit (**no consumer in this mesh addresses by role name**; every one sends the raw served id, so changing `PRIMARY_SERVED_NAME` 404s them all), two measurement traps that produced wrong answers on this run, and a probe-design table with negative controls.

### Fixed

- Two measurement traps now documented rather than re-learned: counting SSE **chunks** instead of `usage.completion_tokens` under-reports decode by >2× under speculative decoding (8.2 vs the true 19.0 tok/s), and this vLLM build returns the thinking trace as **`reasoning`**, not `reasoning_content` — reading the wrong field looks exactly like a model that stopped thinking. Reconciling `completion_tokens` against visible field lengths catches both.

## [0.54.8] - 2026-07-31

### Fixed

- **`worker` peer-proxy was silently inert.** `lobes/gateway/server.py`'s `_PEER_SERVED_NAME_ENV` and `_PEER_ROLE_HINT` still carried only the five pre-`worker` roles, while `lobes/gateway/_config.py`'s `PEER_PROXY_ENV` carries eight. `_peer_served_name` therefore resolved `""` for `worker` — **even with `WORKER_SERVED_NAME` set**, because that var was never looked up — so `peer_specs_from_table` dropped the role at its `if not served_name: continue` guard. Downstream: the `ReadinessCache` never peer-probed it, `/v1/models` never advertised it, and `_proxied_owner` never matched, so `WORKER_PEER_PROXY=true` forwarded nothing and the request fell through to the referral-only `role_infeasible` 404. This hit exactly the shape that needs it — a box that only *reaches* `worker` on a peer has no `WORKER_BASE_URL`, hence no wired `Backend` to resolve off. The pre-existing `worker` proxy tests all used a **wired**-but-infeasible env, which resolves via the wired backend and never touches either table, so the gap was invisible to CI. Introduced in 0.54.6 with the `worker` role.

### Added

- `tests/test_gateway_proxy.py::test_every_proxyable_role_resolves_a_served_name` — a standing contract guard asserting every role in `PEER_PROXY_ENV` resolves a non-empty served name, so adding a proxyable role without teaching both `server.py` tables fails in CI instead of going silently inert in production. (`stt`/`tts` resolve via `_peer_served_name`'s fixed-sidecar early return rather than these tables.)
- Three unwired-`worker` proxy tests covering the paths the wired fixture skipped: `WORKER_SERVED_NAME` override, catalog `role_hint` fallback, and the data-plane forward for both the `worker` alias and the raw served id.

## [0.54.7] - 2026-07-31

### Changed

- Eidetic memory: recall bookkeeping only — refreshed `last_recall` / `recall_count` on `fleet-containers-omits-gemma-multimodal` (5 -> 6) and `multimodal-base-url-compose-default-traps-loaded` (0 -> 1) from the 2026-07-25 recall pass. No record content, no code changes.

## [0.54.6] - 2026-07-31

### Added

- The **`worker`** role — an eighth first-class Colleague role serving
  `unsloth/Qwen3.6-35B-A3B-NVFP4`, a **multimodal** (image+video, no audio)
  ~3B-active MoE with a self-hosted MTP draft at 262K native. `worker` is the
  fast ground-work **DOER** — the first role besides `cortex` permitted
  `repo_action` (it may act on the repo under cortex's direction; forbidden
  only `final_decision`/`security_decision`), served multimodal (no
  `--language-model-only`) — a "seeing doer" distinct from `senses`, which
  perceives but must not act. Wired end-to-end: catalog gear + capability tier
  (`minor < multimodal < worker < muse < main`), the eight-role registry, the
  opt-in-core gateway backend (`WORKER_BASE_URL`; unwired ⇒ infeasible, never a
  silent fallback), `OPT_IN_CORE_ROLES` + `base.toml` unknown-card veto, a
  profile-gated `vllm-worker` compose service on the Qwen nightly lane
  (compressed-tensors, `qwen3_coder`+`qwen3` parser pair, self-draft MTP), the
  `lobes up worker` verb, the `thor-worker` deployment shape, and full docs.
  **VALIDATED live on the physical Jetson AGX Thor (sm_110), 2026-07-31**
  (`docs/evidence/2026-07-31-accept-worker-thor.txt`): boots and serves at
  measured util 0.45 / full 262144 window (KV pool 41.78 GiB = 14.07×
  concurrency, weights 24.81 GiB); MTP self-draft **loads and accepts at
  89.1%**; decode 50.8 tok/s (with thinking) / 73.5 tok/s sustained, TTFT
  ~2.1 s; **image AND video intake correct** (image: red/blue with a negative
  control; video: a real Spark-webcam clip described accurately —
  subject/scene/motion, not a #101-style drop); thinking + tool + reasoning
  parsers work; `model=worker` routes through the gateway and `model=muse`
  404s `role_infeasible` with no referral. Key finding — **the vllm-worker
  lane must NOT force `--moe-backend` on sm_110**: every forced NVFP4 backend
  was refused (flashinfer_* lack sm_110 kernels; marlin/triton reject the
  mixed quantized-main/unquantized-MTP experts), so the compose omits the flag
  and vLLM auto-selects per path. Spec/plan under `docs/specs/` and
  `docs/plans/2026-07-31-thor-worker-lobe-qwen3-6-35b-a3b.md`.
- Muse goes **dormant/unhosted** mesh-wide: Thor moves off the Gemma 4 31B
  `muse` to host `worker` instead. `model=muse` now 404s `role_infeasible`
  with no `hosted_by` referral; the `muse` role, its catalog entry, and the
  `thor-muse` shape all stay in-tree (cite-don't-delete), and the tier
  vocabulary still ranks `worker < muse`.

### Fixed

- CLAUDE.md / docs: the stale "pressure policy degrades … to `minor`" wording
  now matches the shipped code — the degrade-to-`minor` substitution was
  removed; under pressure the full tiers (`cortex`/`senses`/`worker`/`muse`)
  are **shed** with HTTP 429 + `Retry-After`, and `minor` is the servable
  floor.
- **SonarCloud cleanup** (folded in from #163): cleared the 5 genuinely-open
  issues — **S8997** ×4 (`test_chatterbox_pcm16.py` and
  `test_readiness_peer_probe.py` now use the `monkeypatch` fixture instead of
  manual save/restore) and **S3776** (`tts_client.py::_synthesize_single`
  refactored, cognitive complexity 18 → 6, behavior-preserving). No
  suppressions were added — the false-positive candidates were already resolved.

## [0.54.5] - 2026-07-25

### Added

- `lobes capabilities` renders a third `loaded` state, **`by-proxy`**, for a role this box serves by forwarding to a peer — derived from the payload's existing `feasible`+`proxied` pair and shown alongside the hosting peer. Two roles in the same proxied state previously printed different values (`yes` vs `no`) purely because of leftover local `<PREFIX>_BASE_URL` wiring, while both forwarded happily and both answered 200.

### Changed

- `docs/colleague-stack.md` now states that `loaded` is a LOCAL wiring fact that does not indicate whether a proxied role works, and records why `senses` reports `loaded: true` on every fleet deployment: `multimodal` is the only optional generate backend whose compose default is non-empty (`MULTIMODAL_BASE_URL=${MULTIMODAL_BASE_URL:-http://vllm-multimodal:8000}`), and `${VAR:-default}` ignores an explicit empty value, so it cannot be unwired from `.env`.

### Fixed

- The stale comment in `lobes/cli/_commands/capabilities.py` that admitted an infeasible role's row "still shows loaded=yes" now describes the shipped three-state behaviour.

## [0.54.4] - 2026-07-25

### Changed

- `lobes.realtime.tts_client` — the TTS retry sentinel is now a dedicated `_Retry` type rather than a bare `object()`, so `_handle_tts_response()` annotates as `bytes | _Retry` and `_synthesize_single()` narrows with `isinstance` instead of suppressing the mismatch with `# type: ignore[return-value]`. The runtime contract is unchanged — callers still only ever see `bytes` — but that invariant is now statically checked rather than asserted, so a future edit cannot leak a non-bytes sentinel past the retry check unnoticed (Qodo review, PR #157).

## [0.54.3] - 2026-07-24

### Added

- tests/test_tts_pause_and_truncation.py — pause-mapping, truncation-detection and a ReDoS regression test for the S8786 rewrite (skipped offline; httpx is a realtime-extra dependency).

### Fixed

- Sonar S8786: replaced the quadratic `!{3,}$` regex in trailing_pause_ms with a linear rstrip count — a 40k-character input took ~2.6s to scan and now takes microseconds.
- Sonar S3776: extracted `_handle_tts_response` / `_is_truncated` / `_min_plausible_duration` from `_synthesize_single`, dropping its cognitive complexity from 30 (limit 15).
- Sonar S8513 (x2): collapsed chained endswith calls into single tuple-argument calls.
- Sonar S8572: the catch-all TTS retry arm now uses logging.exception so the traceback survives.
- Sonar S8410 (x3): FastAPI /v1/audio/transcriptions parameters now use Annotated type hints; `model` is correctly typed `str | None`.
- Sonar S5778 (x2): hoisted non-asserting setup calls out of pytest.raises blocks so the tests assert what they claim.

## [0.54.2] - 2026-07-24

### Added

- Spec and plan for issue #155 — STT readiness truth: device-agnostic readiness, per-lane audio gating, a completed realtime composite, and cause-naming gateway refusals.

## [0.54.1] - 2026-07-22

### Added

- `lobes.realtime._conversation.delivery_pause_ms` -- milliseconds to wait before the next audio chunk, tracking playback rather than socket drain. Runs at most `DELIVERY_LEAD_MS` (400 ms) ahead of the playhead: enough that a client buffer never runs dry, short enough that a barge-in onset lands inside a turn the server can still cancel. Returns 0 whenever delivery is already at or behind the playhead, so it only ever slows a run-ahead and never adds latency to audio the client is waiting on.

### Changed

- `secureContextHint` (Astro harness) derives a copy-pasteable remedy from the page's own origin -- the exact `ssh -L` command and URL -- instead of describing the secure-context rule and leaving the reader to apply it. A page already on localhost is told the origin is not the problem rather than sent chasing it.

### Fixed

- Realtime voice-out is now **paced to the playhead** instead of drained as fast as the socket accepts it. `_drive_response` pumped every delta in 2-4 ms for 7.5-8.5 s of audio (measured, `docs/evidence/2026-07-22-accept-realtime-voice-to-voice-spark.txt`) and then left `SPEAKING`, so a user talking over a still-playing reply was talking while the floor had already returned to `LISTENING`: a new turn opened and `response.interrupted` was never emitted. Every barge-in guarantee in `lobes/realtime/_floor.py` was correct and completely inert. It also made session history dishonest -- the floor trims an interrupted reply to the prefix plausibly heard, but with instant delivery nothing is ever undelivered, so the machine recorded the whole reply as spoken. Barge-in itself remains UNVALIDATED live (#108); this is the mechanism the next acceptance run must exercise, not a claim that it passed.
- The Astro harness's `no-device` message now names the PipeWire/PulseAudio output-only-profile trap, where a card shows a working device to `arecord` and nothing at all to the browser, and points at `pactl list sources short` plus the duplex-profile fix.
- The `unsupported` state no longer blames the origin unconditionally (Qodo #154-1). It is reached via `isSupported()`, which checks only API presence — so a browser missing `AudioWorkletNode` was being sent to set up SSH forwarding that would change nothing. `unsupportedMessage()` now reads `isSecureContext` and asserts a cause only when one is actually known: insecure origin → the forwarding remedy, secure origin → a missing-Web-Audio-APIs message, unreadable → both named and neither asserted. The pre-existing test that asserted the conflated copy is corrected and now pins both branches explicitly instead of depending on an ambient `isSecureContext`.
- `CLAUDE.md` no longer overclaims the #151 realtime work on two counts. It said "no live acceptance transcript for this work has landed under `docs/evidence/` yet", which went stale in the very squash merge that added the 2026-07-22 Spark transcript; and it stated "Speaking during playback **interrupts**" as fact, which that transcript disproves. Both are corrected to match the evidence: the block is now **PARTIALLY validated**, naming what the run proved (voice-to-voice end-to-end) and what it did not (barge-in), and flagging that the transcript predates the 0.54.1 pacing fix so it is not evidence for barge-in either way.
- `secureContextHint()` no longer invents port 4321 when `location.port` is empty (Qodo #154-2). A page served on a default port reported `""` and got a forwarding command for a service it was never served from; the real port is now derived from the protocol (80/443), and a privileged remote port maps to an unprivileged *local* one, so `ssh -L` never emits a command that needs root on the pasting side.

## [0.54.0] - 2026-07-21

### Added

- Voice-to-voice on /v1/realtime (#151): an opt-in conversation surface — a committed turn can trigger a server-side generate + Chatterbox reply streamed back over the SAME WebSocket as response.audio.delta events, interruptible by barge-in. Ears-only stays the default: a session that never sends response.create emits exactly the #149 transcription-only sequence.
- Barge-in with cancel-both semantics: speech during playback cancels the in-flight generate AND TTS, never sends the undelivered audio remainder, and emits response.interrupted — only the plausibly-heard prefix enters history. Arms the previously-dormant BARGE_IN_WINDOW_MS.
- New stdlib modules, all offline-tested: `_wire.py` (base64 event codec), `_floor.py` (floor/turn state machine with per-stage deadlines), `_turn.py` (generate payload shaping), `_conversation.py` (the wiring layer). `app.py` stays a `pragma: no cover` shell.
- Per-session conversation history + system prompt, with an operator default via DEFAULT_SYSTEM_PROMPT and a per-session connect-config override.
- site/ — a local-only Astro harness driving the realtime surface from a browser: mic capture with echoCancellation, live event stream with per-error-code distinction and VAD at_ms timing, audio-out playback, a user mute/mic-off control, and a conversation toggle. Reached through a local credential-injecting WebSocket proxy via ssh -L; never deployed. New site-build CI job.
- New env knobs threaded end-to-end (settings -> compose -> env.audio.example -> doctor --fix): BARGE_IN_WINDOW_MS, BARGE_IN_MODEL, TTS_VOICE_CONCURRENCY, DEFAULT_SYSTEM_PROMPT.
- Named error codes generate_failed, tts_failed, response_timeout and invalid_wire_event; boundary events now carry at_ms and reason, so VAD tuning is observable and a max_turn force-commit is finally distinguishable from a silence-confirmed stop.

### Changed

- BREAKING (#151, coordinated): /v1/realtime audio is now OpenAI-shaped base64 JSON events in BOTH directions — input_audio_buffer.append inbound, response.audio.delta outbound. Raw binary audio frames are gone and now yield invalid_wire_event. The deployed reachy-mini-cli speaks the old wire and cannot stream until it adapts — tracked in reachy-mini-cli#115. In-repo clients (realtime-smoke.py, realtime-voice-loop.py) are migrated.
- Per-lane TTS concurrency: a spoken reply no longer queues behind unrelated batch /v1/audio/speech work. The batch lane's observable behaviour is unchanged (verified byte-for-byte).
- Muting is a narrowed ban, not a lifted one (deviation d1): AUTOMATIC mute-during-playback stays forbidden — it is the AEC substitute that makes barge-in impossible — while user-initiated mute/mic-off is allowed, because AEC is owned at the client edge (Reachy firmware, browser echoCancellation).

### Fixed

- The segmenter's at_ms and reason were computed and then dropped before reaching the wire (pre-existing since #149), so VAD boundary timing was unobservable by any client and a max_turn force-commit looked identical to a silence-confirmed stop.
- A stock Python-template .gitignore rule (/site, meant for mkdocs output) silently swallowed the new Astro harness: git add reported nothing, with no error.
- Three env knobs the settings module read but the deployment never passed — BARGE_IN_WINDOW_MS, BARGE_IN_MODEL and TTS_CONCURRENCY — silently pinned to their in-container defaults. A new AST-based coverage test fails CI if any future Settings field is added unwired.

## [0.53.1] - 2026-07-21

### Fixed

- Six defects in the voice loop and its tests, all found by review (Qodo) and all accepted: the event reader swallowed EVERY exception from `read_frame` and retried, so an EOF after a disconnect became an endless spin and the main loop waited out its idle timeout — a dead session reading as 'nobody spoke' (timeouts now continue, anything else ends the session and says why); `speak()` ran the audio player with no timeout, which is not hypothetical — paplay was OBSERVED hanging on a sink whose ALSA device another process held, and with the mic muted for the duration that hang deafens the session permanently (now a bounded 60s per backend, falling through to the next); a mid-session mic EOF broke the feeder loop without stopping the session, faking silence again; `arecord`'s stderr was never piped, so the 'no audio' failure message could not actually quote the ALSA error it promised to; and two test weaknesses — a single `recv()` asserting an exact byte count where a stream socket may legitimately return less (now drains to the expected total, verified stable over 12 consecutive runs), and a `join(timeout=...)` that never asserted the thread finished, which is precisely where a deadlock should be reported.

## [0.53.0] - 2026-07-21

### Added

- `scripts/realtime-voice-loop.py` — talk to the machine, voice to voice, by composing the three endpoints the fleet already serves: `/v1/realtime` for ears, a generate lane for the brain, `/v1/audio/speech` for the mouth. It is also the richest live test of the realtime surface, because it holds a LONG-LIVED DUPLEX session — reading while writing, for minutes — which the one-shot `realtime-smoke.py` cannot exercise. It defaults to the Gemma 4 12B lane (`--model multimodal`, ~1s to a short reply on the DGX Spark) rather than a thinking model, since in a spoken turn latency is dead air; takes the API key from `LOBES_API_KEY` because argv is world-readable via `/proc`; supports stereo-only capture devices (`--channels 2`, downmixed); and routes playback with `--sink` for boxes where another process owns the audio device.

### Changed

- `docs/realtime-pipeline.md` documents the voice loop and three behaviours that a client of `/v1/realtime` must get right, each learned on live hardware: **answer `PING` with `PONG`** (uvicorn pings every ~20s and closes a peer that never pongs — a duplex client that ignores them dies after tens of seconds while a short smoke run never notices); **half-duplex turn-taking** (the mic is muted for the whole synthesize-and-play window, because without AEC the session transcribes the machine talking to itself — real barge-in needs AEC, tracked in #151); and picking a FAST generate lane for spoken replies.

### Fixed

- The WebSocket client in `scripts/realtime-smoke.py` bounded its reads with `settimeout`, which is a property of the SOCKET rather than of one call — so a reader handed its deadline to any thread writing on the same socket, and a write that timed out part-sent left a torn frame that desynchronised the peer's parser. Reads now wait with `select`, which observes readability without mutating socket state. Invisible to the one-shot smoke run (which streams, then reads); found by building a client that listens while it talks.

## [0.52.3] - 2026-07-21

### Changed

- The `/v1/realtime` route is split into `_open_session`, `_arm_segmenter`, `_to_pcm16k`, `_emit_turn_events`, `_transcribe_turn`, and `_pump_session` (SonarCloud S3776: cognitive complexity 30 against a limit of 15). Behaviour is unchanged and, since the route carries no unit coverage by design, the refactor was gated on a fresh live run at both wire rates rather than on the offline suite alone.

### Fixed

- The `/v1/realtime` handshake forwarded the caller's `Authorization` (and `Cookie`) to the realtime bridge. The credential is spent the moment the gateway's own inbound gate validates it and the bridge has no auth of its own, so forwarding it only widened a gateway key's blast radius to the bridge's logs and telemetry. Both headers are now dropped from the forwarded handshake (Qodo).
- A dead realtime bridge could strand a gateway handler thread. When the upstream pump ended it half-closed the client (`SHUT_WR`), which does NOT wake a `recv` blocked on that same socket — so an idle client left the thread parked until it happened to speak or hang up, leaking one thread per open session on every bridge restart. Each pump now shuts its peer down in both directions on exit. The unit test missed it because its fake socket returned EOF the moment its script ran dry; a fake that genuinely blocks is now the regression guard (Qodo).
- `VAD_MAX_TURN_MS` was read unvalidated, so `0` or a negative value made the segmenter force-commit on the first chunk of every turn and keep committing — an event storm plus one STT forward per 32 ms chunk from a single typo in `.env`. Clamped to a 1000 ms floor, matching the existing `tts_speed`/`tts_concurrency` treatment (Qodo).

## [0.52.2] - 2026-07-21

### Changed

- `/v1/realtime` is VALIDATED live on the DGX Spark GB10 (2026-07-21, `docs/evidence/2026-07-21-accept-realtime-spark.txt`): a full session ran through the gateway tunnel against the real Silero model and the real Parakeet/Chatterbox sidecars — `session.created` through transcription on one connection, at 24000 Hz and the 16000 Hz passthrough, plus the 401 on an unauthenticated handshake and the 426 on a plain GET. The #149 motivating case (a five-word question that a client-side energy threshold shattered into "Ready, she") now arrives as one whole utterance with a single speech boundary pair. Four things stay UNVALIDATED and are documented as such: a real microphone (the live runs used synthesized audio), the VAD-unavailable path, concurrent sessions, and the max-turn force-commit.

### Fixed

- The gateway tunnel sent the bridge's FIRST FRAME back upstream instead of to the client. `read_head` returns any bytes the upstream packed into the same TCP segment as its 101, and `run_tunnel` wrote them to `upstream` — so `session.created` never reached the caller, and an unmasked server frame arrived at the bridge, which RFC 6455 §5.1 requires it to close on. Every session died the instant it opened. Caught by the first live run on a DGX Spark GB10, NOT by the unit suite, whose test had asserted the wrong direction as correct — the test is now inverted and joined by a regression test naming the failure.

## [0.52.1] - 2026-07-21

### Fixed

- Pre-merge review of the realtime work caught three defects, all reproduced before fixing: the gateway's `/v1/realtime` handshake hung on every real connection (`BufferedReader.read(n)` waits to fill its buffer on a blocking socket, and a bridge that just accepted a session sends nothing more — now `read1()`, guarded by real-socketpair tests instead of a fake that returned early); Silero inference and scipy resampling ran on the bridge's single asyncio event loop, so one talking session starved every other session and every batch `/v1/audio/*` request (both now offloaded via `anyio.to_thread.run_sync`, matching the Chatterbox sidecar's convention, with the 16 kHz no-op passthrough left inline); and `scripts/realtime-smoke.py` desynced its frame parser if a read timed out between a frame's header and its payload.

## [0.52.0] - 2026-07-21

### Added

- `/v1/realtime` — the server_vad WebSocket session the realtime bridge has promised since it shipped (issue #149). One connection replaces a WS-plus-batch-POST dance: stream PCM16 mono little-endian in (24000 Hz default, 16000 Hz accepted; the server resamples to 16 kHz itself) and receive `session.created` / `input_audio_buffer.speech_started` / `...speech_stopped` / `conversation.item.input_audio_transcription.completed` / `error` events back on the SAME connection, with committed turns transcribed by Parakeet. This redeems two in-tree IOUs — `app.py`'s own "PR2 adds the /v1/realtime WebSocket route" docstring and `realtime-pipeline.md`'s "planned for a later release" boundary claim — against the #149 baseline probe, where the deployed facade served four batch routes and no WebSocket, forcing reachy-mini-cli to endpoint from a client-side energy threshold that shattered sentences at inter-word dips.
- `lobes/realtime/_segmenter.py` — the server_vad turn segmenter as a pure state machine over 512-sample / 32 ms chunks, with Silero injected as a callable so the offline suite tests segmentation with a fake VAD (no torch, no GPU). A never-silent turn force-commits at `VAD_MAX_TURN_MS` (default 30 s) with `reason="max_turn"` — a normal boundary event, never an error — so a stuck stream cannot grow bridge memory without bound.
- `lobes/realtime/_session.py` — the session event schema, config parsing, teardown bookkeeping, and session-id-scoped logging (stdlib-only, so it is unit-tested without the `[realtime]` extra). A single `error` event type discriminated by `ErrorCode` is what makes VAD-down distinguishable from silence by event type alone; credential-shaped config fields are redacted before any log line.
- `lobes/gateway/_realtime.py` — WebSocket passthrough in the stdlib gateway: a 101-upgrade and bidirectional byte relay fronting the bridge, so the session is reached through the same origin and the same opt-in `GATEWAY_API_KEY` bearer gate as every other `/v1/*` route. The handshake is relayed verbatim rather than reimplemented, both pump directions unwind on either side's close, and the session legs drop the HTTP read timeout that would otherwise kill a listening session mid-silence.
- The `stt` role advertises `realtime_vad_session` on `lobes capabilities` and gateway `GET /capabilities`, so a client discovers the session surface instead of probing for it. Advertised only when the audio overlay is wired AND the lane is feasible — a text-only fleet or a `STT_FEASIBLE=false` deployment withholds the claim.
- `scripts/realtime-smoke.py` — a live end-to-end session check (synthesize a known phrase, stream it, assert the boundary and transcript events arrive on one connection) written against a hand-rolled RFC 6455 client with no torch, no OpenAI SDK, and no WebSocket dependency, because keeping those out of a robot CLI's dependency tree is the point of server-side VAD. `docs/evidence/README-realtime-acceptance.md` documents the acceptance-evidence procedure.

### Changed

- `/v1/realtime` is DECLARED/UNVALIDATED live (#108): the offline suite proves the session, VAD, and config logic with a scripted fake VAD, but nothing has run against real Silero, real Parakeet, or hardware yet — no `docs/evidence/` transcript exists for issue #149. A plain GET to the route answers 426 Upgrade Required, and a declared-off `stt` lane answers 404 `role_infeasible` naming its peer; a session is never proxied cross-box, since the #129 proxy-lobes forwarder is POST-only.

### Fixed

- The realtime container never received its own VAD knobs: `_settings.py` read `VAD_THRESHOLD`, `VAD_SILENCE_MS`, `VAD_PREFIX_PADDING_MS`, `DEFAULT_TURN_DETECTION`, and `DEFAULT_AEC_MODE`, but neither `env.audio.example` nor the compose `environment:` block passed any of them, so every value silently pinned to its code default and no operator could tune them. All six (plus the new `VAD_MAX_TURN_MS`) are now wired with compose defaults identical to the code's, and `doctor --fix` heals a pre-existing deployment append-only, never rewriting an operator-customised line.

## [0.51.1] - 2026-07-20

### Added

- `SupportedModel.default_gpu_mem_util` — a per-model pooling budget override. The shared embed/score default (0.06) is sized for the ~0.6B gears and is SMALLER than the 4B embedder's own weights (7.56 GiB measured, vs a 0.06 x 121.69 = 7.30 GiB budget), so `lobes switch Qwen/Qwen3-Embedding-4B` previously wrote a budget the model could not load in. The 4B declares 0.11; every other model keeps the shared default.
- Served-name collision guard in the gateway: two wired backends claiming one `served_name` make `resolve_model`/`order_backends` ownership order-dependent, which on the embed lane means answering from the WRONG VECTOR SPACE. The gateway still starts (a name clash must not take the fleet down) but now warns loudly on stderr, naming the colliding backends.

### Fixed

- **The `/recall` and `/remember` wrappers forced `EIDETIC_EMBED_URL=http://localhost:8002/v1`** — a port nothing listens on — so every semantic query silently ran on eidetic's 128-dim lexical-hash fallback while the docs claimed a live embedder. Both now default to the lobes gateway (`http://localhost:8001/v1`), matching eidetic >= 0.12's own default; verified `online=True` at 1024 dim. The previous docs-only correction fixed the prose and left the scripts broken.
- Sonar `python:S1192`: extracted `_CONTEXT_32K_NATIVE` for the `"32K native"` literal, now shared by three catalog entries.

## [0.51.0] - 2026-07-20

### Added

- **`embed-deep` — an opt-in second embedding gear** (`Qwen/Qwen3-Embedding-4B`,
  2560-dim Matryoshka, MTEB multilingual 69.45 vs the 0.6B's ~64.3) beside the
  always-on hot-path embedder, addressed through the gateway as
  `model=embed-deep`. Opt-in on **both** axes — the `vllm-embed-deep` service is
  `COMPOSE_PROFILES=embed-deep`-gated and the gateway route is wired only when
  `EMBED_DEEP_BASE_URL` is set — so every existing deployment renders
  byte-identically (no shape golden changed; `vllm-embed`'s service hash is
  unchanged). Structurally it follows the `multimodal-coder` precedent: an opt-in
  backend plus a wired-only alias whose name is its backend name.
- `EMBED_DEEP_*` env block (`MODEL`, `SERVED_NAME`, `BASE_URL`, `MAX_MODEL_LEN`,
  `GPU_MEM_UTIL`, `ATTENTION_BACKEND`) and `docs/qwen3-embedding-4b.md`.
- **GB10 acceptance transcript** (`docs/evidence/2026-07-20-accept-embed-deep-gb10.txt`):
  booted live on `spark-f8a9` alongside the running spark-lobe fleet (zero fleet
  mutation) and **serves 2560 dim** — previously declared from `config.json` only.
  Matryoshka honoured at all 6 probed ladder points; paraphrase probe 0.7362 vs
  0.2818 unrelated; boots at `gpu_mem_util=0.11` (weights 7.56 GiB, KV 11.34 GiB /
  82,592 tokens, CUDA graph pool 0.84 GiB); 42.4 ms median vs the 0.6B's 11.5 ms.
- **Pooling-lane attention-backend documentation** (`docs/tuning-profiles.md`,
  `docs/machine-profiles.md`, `docs/qwen3-embedding-4b.md`): the `SM_110` trait
  keys its knobs by PROFILE ROLE name, so it **cannot reach the `embed-deep`
  GEAR** — a Thor operator must set `EMBED_DEEP_ATTENTION_BACKEND=TRITON_ATTN` by
  hand or the forward pass hangs while `/health` stays green (#105). This is the
  first place the gear-vs-role tradeoff costs something sharp.

### Changed

- The gateway's opt-in alias wiring now covers both `multimodal-coder` and
  `embed-deep` (same wired-only contract). `embed-deep` deliberately gets **no
  fallback** to the 0.6B: the two embedders occupy different vector spaces, so a
  silent downgrade would return meaningless similarity instead of an honest
  unknown-model failure. `tier_aliases` falls back *upward*, which is right for
  generation and wrong for embeddings.
- `Qwen/Qwen3-Embedding-4B` catalog status `configured` -> `load-tested` (GB10
  only; sm_110 remains UNVALIDATED per #108). `EMBED_DEEP_GPU_MEM_UTIL=0.11` is
  documented as MEASURED — with the caveat that vLLM's actual allocation (19.74
  GiB) does not reconcile with `util x total` (13.39 GiB) on this unified-memory
  card, so the knob is empirical, as it was for spark-lobe's 0.44.
- The catalog's `test_exactly_one_embed_and_one_score_model` invariant split: the
  score lane still pins exactly one model, while the embed lane now pins the
  property that actually matters — exactly one entry carries
  `role_hint="embedding"`, so the `embedder` role's reported model stays
  unambiguous under `_catalog_by_role_hint`'s first-match lookup.
- Corrected the `recall`/`remember` skill docs: the embed endpoint is the lobes
  gateway on `localhost:8001/v1` (not `:8002`), eidetic >= 0.12 sends
  `Authorization` from `EIDETIC_EMBED_API_KEY` or a borrowed
  `COLLEAGUE_API_KEY`/`CULTURE_VLLM_API_KEY`, and records live in TWO stores —
  public in the COMMITTED `<repo-root>/.eidetic/memory`, private in `$HOME` —
  which `recall.sh`'s own header already documented correctly while `SKILL.md`
  contradicted it. Filed the silent hash-fallback bug upstream as
  `agentculture/eidetic-cli#34`.

### Fixed

- **`lobes switch` on an embed-task model always named the `vllm-embed`
  service**, so switching to the 4B told operators to replace the hot-path 0.6B
  in place — silently invalidating any index built with it, precisely the hazard
  this design exists to prevent. `_pooling_notice` hardcoded one service for the
  whole embed task; it now resolves per model via `role_hint`. Found by an
  independent colleague review.

## [0.50.0] - 2026-07-17

### Added

- **`tools` on the role contract** — every role in `lobes capabilities --json`
  and gateway `GET /capabilities` now reports whether its endpoint accepts
  OpenAI `tools` on a request. Derived from the catalog's `tool_parser` (the
  same field the served `--tool-call-parser` flag is built from, so it cannot
  drift from reality without `tests/test_catalog.py`'s pairing guard failing
  first): `true` for cortex/senses/muse, `false` for the pooling roles
  (embedder/reranker serve no chat lane) and stt/tts. Previously NO role
  advertised tool support anywhere, so a Colleague could not discover it.
  Deliberately a bool, not a parser name — the served parser can diverge from
  the catalog's (`PRIMARY_TOOL_CALL_PARSER`, the `qwen3_coder_thinking`
  plugin), so naming one would be a claim `lobes.roles` cannot honestly make.
- **`tool_use` in `muse`'s declared responsibilities.** Not a widening of its
  authority: `final_decision` / `repo_action` / `security_decision` stay
  forbidden, so muse calls tools to RESEARCH a proposal, never to enact one —
  `cortex` remains the only lobe that acts.

### Changed

- **Gemma 4 lanes now serve the `gemma4` parser PAIR** (`senses`, the opt-in
  coder candidate, and `muse`): `--tool-call-parser=gemma4`
  (`Gemma4EngineToolParser`, replacing the generic `pythonic`) **plus**
  `--reasoning-parser=gemma4` (`Gemma4ParserReasoningAdapter`, previously absent
  entirely). This mirrors the cortex lane's long-standing
  `--reasoning-parser=qwen3` + `qwen3_coder` pairing. See Fixed, below, for why
  each half is load-bearing. Operators on an existing scaffold must re-run
  `lobes init` (or edit the deployed `docker-compose.yml`) to pick this up; a
  running container keeps its old flags until recreated.

### Fixed

- **Gemma 4 tool calling was silently broken on every Gemma lane.** Gemma 4 does
  not emit Python-style calls — it emits `<|tool_call>call:name{...}<tool_call|>`,
  whose delimiters are **special tokens**. The `pythonic` parser is served with
  `skip_special_tokens=True`, so those delimiters were stripped before it ran; it
  then matched nothing and vLLM relayed the model's perfectly well-formed call as
  ordinary assistant **content**, with `tool_calls: null` and
  `finish_reason: "stop"`. A caller passing `tools` got prose shaped like a tool
  call and no callable one — no error, no warning. `pythonic` was never
  evidence-backed: `runtime/_parser.py` carried its own "risk r2, pending live
  validation" caveat from the start, and that check had never run. It ran on
  2026-07-17 against the live 31B on a physical Jetson AGX Thor and disproved the
  guess. **Validated on the 31B `muse` lane only**
  (`docs/evidence/2026-07-17-accept-muse-tool-calling-thor.txt`); the 12B lanes
  inherit the family rule and remain UNVALIDATED (#108) — a strictly better
  default than a parser proven wrong for the family, not a measured claim.
- **Gemma 4 channel markers leaked into `content`.** The correct tool parser
  forces `skip_special_tokens=False` (that is how it sees `<|tool_call>`), which
  also exposes Gemma's `<|channel>thought` markers — which a *tool* parser has no
  business stripping. A plain answer came back as
  `"<|channel>thought\n<channel|>The weather in Paris is..."`. vLLM ships the
  matching half (`--reasoning-parser=gemma4`) and lobes wired it on no Gemma lane;
  it is now paired on all three. Enable both or neither: the tool parser alone
  trades a broken tool call for dirty content.
- **`lobes capabilities` no longer misreports an older gateway as unreachable.**
  Its gateway sanity-check required every current `RoleInfo` field, so a NEWER
  CLI probing an OLDER gateway (routine on a mixed-version mesh) judged a
  perfectly good response malformed, fell back to `.env` guesses, and printed
  "gateway unreachable" — false, since the gateway had answered, and the exact
  #92 dishonesty inverted. Fields added after the original #81 contract shape
  (`tools`, `feasible`) are now tolerated when absent; the stable core is already
  conclusive for that check's real job (telling a lobes gateway from a stray
  daemon on a guessed port). `_render_table` already `.get`-ed both with safe
  defaults, so an older payload renders without fabricating either.

## [0.49.0] - 2026-07-17

### Added

- First-class stt/tts (#129): the audio roles joined the feasibility + peer referral/proxy channels — `STT_/TTS_FEASIBLE` (absent = feasible, the sleeping-lobe default; every pre-#129 deployment renders byte-identically), `STT_/TTS_PEER_ORIGIN` / `_PEER_PROXY` / `_PEER_API_KEY` with the same three-condition arming, `hosted_by`/`proxied` capabilities annotations via the one shared annotator, and a capabilities-based peer readiness probe (audio roles never appear on a peer `/v1/models`).
- Per-endpoint audio routing (#129): `/v1/audio/speech` (tts) and `/v1/audio/transcriptions` (stt) route independently — a declared-off lane proxies to its peer through the same data-plane machinery as core roles (caller `Authorization` stripped + pairwise key injected, body forwarded VERBATIM, `X-Lobes-Proxied` single-hop guard with 508 `proxy_loop`, `X-Lobes-Proxied-By` attribution) or 404s `role_infeasible` with the honest referral; `AUDIO_URL` stays the local-bridge lane. The live trigger: Chatterbox on the Thor, Parakeet local on the Spark. DECLARED/UNVALIDATED until the live acceptance transcript lands (#108).

### Fixed

- Auth stragglers (#129 items 1-2): the stt/tts measure probes (`roles_measure.py`) and the minor client (`minor/_client.py`, used by `benchmark --all-lobes` and `lobes route`) now merge the same contextvar-scoped gateway auth header every assess-backed verb attaches; `lobes route` resolves and installs the deployment key like its sibling verbs. With no key configured every request is byte-identical.

## [0.48.0] - 2026-07-17

### Added

- `lobes doctor` gains two fleet checks (#119): `scaffold_files` (every expected scaffold file on disk — the 2026-07-17 Spark partial-audio incident served for hours with `/health` green and two Dockerfiles absent) and `profile_staleness` (the deployed `.env` carries the knobs the resolved machine profile requires — the 2026-07-14 Thor incident: a pre-#110 `.env` missing the SM_110 divergences hung its rerank lane silently). A key still carrying the template default where the profile requires a divergence is named; a genuine operator override only downgrades to info; shape-dropped roles are never demanded.
- `lobes doctor --fix` — the missing-only heal lane (#119): `--fix` prints the plan (still read-only), `--fix --apply` writes only ABSENT scaffold files and appends only ABSENT `.env` keys, never rewriting an existing line (compose `env_file` last-duplicate-wins would let an appended default clobber an operator value). The safe path between `lobes init` refusing (any file exists) and `--force` clobbering the whole template set, `.env` included.

### Changed

- Doctor remediation strings for stale/partial scaffolds name `doctor --fix`, never `lobes init --apply --force` (which would wipe the gateway key, peer/proxy config, and shape reclaim values).

## [0.47.1] - 2026-07-17

### Fixed

- `accept-shape.sh` and `validate-tiers.sh` now consume the compose `-f` chain from `lobes fleet files` instead of re-implementing it in bash (#138): `--restore` no longer boots the very lobe a mesh-shape backup dropped (and no longer re-introduces the gateway `depends_on` edge the shape resets), and validate-tiers gateway recreates keep the shape overlay so results describe the topology the operator actually runs. The restore path fails loudly if the chain cannot be resolved; the best-effort `_compose_down` degrades to a bare `down --remove-orphans`. A drift-proofing test pins both scripts to the CLI authority.

## [0.47.0] - 2026-07-17

### Added

- `lobes fleet files` — read-only verb printing the resolved docker compose `-f` chain (one argv token per line; empty for a plain deployment), so scripts consume the chain from the CLI instead of re-implementing it (#137)

### Changed

- One compose `-f` chain authority (#137): `compose_file_args` in `lobes/runtime/_compose.py` builds every `-f` list; `_compose_files` delegates to it, `up.py` delegates its targeted-services semantics as a parameter, and `compose_up_detached` now resolves the full chain — fixing `switch` tearing down with the full chain but bringing up with none, and `lobes serve --apply` booting a shape-dropped lobe on a shape deployment
- Deleted the dead `shape_render.py::shape_compose_files` and `ShapeRender.compose_files` (never consumed by init; a latent duplicate chain builder)

## [0.46.1] - 2026-07-17

### Fixed

- `lobes fleet up`/`down` **and `lobes up <role>`** now honour an operator-authored `docker-compose.override.yml`. `docker compose` auto-discovers that file only when it resolves the project itself; ANY explicit `-f` suppresses it. `_compose_files` returns `[]` for a plain fleet (so the override applied), but scaffolding an unrelated overlay — the `--audio` overlay or a deployment-shape override — switched it to an explicit `-f` chain that silently STOPPED applying the operator's file. Behaviour thus flipped on unrelated state, with no warning. Found live on the DGX Spark GB10, where the operator override publishes the Parakeet STT container on `127.0.0.1:9002`: a `lobes fleet up --apply` would have recreated the container without that publish and broken `reachy-mini-cli`'s default `REACHY_STT_URL`. The override is now named explicitly in the `-f` chain, LAST — after even the shape overlay — because that is what an override file means to compose: last wins. Deployments without the file are byte-identical to before.

## [0.46.0] - 2026-07-17

### Added

- **`muse` — the seventh first-class Colleague role**: the creative/ideation
  lobe, serving `nvidia/Gemma-4-31B-IT-NVFP4` (Gemma 4 31B IT, NVIDIA's
  official modelopt NVFP4 export; 256K native; declares vision+audio configs
  (plain-gemma4 line, NOT the Unified family); MTP DECLARED via
  `google/gemma-4-31B-it-assistant`, unmeasured). Addressable as `model=muse`; capability order is now
  `minor < multimodal < muse < primary`. Responsibilities:
  creative_generation / long_form_writing / ideation / style_variation /
  divergent_second_opinion; forbidden: final_decision / repo_action /
  security_decision (muse proposes, cortex decides).
- **Opt-in core roles** (`lobes.profiles.shapes.OPT_IN_CORE_ROLES`): muse
  carries the full per-machine Profile knob set (`MUSE_*` prefix, schema is
  now five core roles) but `machine-as-brain` never hosts it — a 31B cannot
  co-reside with the cortex+senses duo on a 128 GB box. The machine-as-brain
  identity set is now `DEFAULT_HOSTED_ROLES` (the six), and the
  machine-as-brain-equals-bare-card byte-identity invariant is preserved
  exactly (a non-hosted opt-in core role renders nothing).
- **`thor-muse` built-in deployment shape** (DECLARED — budget measured live;
  UNVALIDATED pending the acceptance transcript, #108 rule):
  hosts muse + embedder + reranker + audio; drops BOTH heavy default lobes
  (cortex and senses) to peer boxes. Carries the full muse declaration in its
  `[overrides.muse]` (model, `gpu_mem_util=0.55` — measured live on the
  physical Thor 2026-07-17 (26.47 GiB KV pool, 611,415 tokens, 2.33x
  concurrency; the 0.40 hypothesis was refused with 0.6 GiB KV),
  `max_model_len=262144` — the full 256K native window, `quantization=modelopt`,
  `attention_backend=TRITON_ATTN`); hosting muse renders its activation env
  (`COMPOSE_PROFILES=muse` + `MUSE_BASE_URL`). `base.toml` vetoes muse on
  unrecognised cards.
- **`vllm-muse` compose service** — profile-gated behind the `muse` Docker
  Compose profile (never started by a plain `docker compose up`), same custom
  Gemma 4 image as `vllm-multimodal` (`MUSE_IMAGE` overrides the tag).
- `scripts/accept-shape.sh`: drops reclaimable page caches before `fleet up`
  when passwordless sudo is available (the documented Thor first-boot ritual,
  now automated in the acceptance flow), and the referral phase is
  proxy-aware — a dropped role with `<PREFIX>_PEER_PROXY` armed is checked
  for a proxied answer carrying `X-Lobes-Proxied-By: <origin>` instead of
  the referral 404.
- Gateway: muse joins all four peer/feasibility env channels
  (`MUSE_FEASIBLE` / `MUSE_PEER_ORIGIN` / `MUSE_PEER_PROXY` /
  `MUSE_PEER_API_KEY`) — referral and proxy-lobes work for muse exactly like
  every core role. `lobes up muse`, `lobes measure` (llm family), pressure
  shedding (muse degrades to `minor`), and `lobes fleet status` (container
  included when activated) all cover the new role. `lobes up colleague-stack`
  deliberately stays the six default roles.

### Changed

- `OPT_IN_BACKENDS` (gateway): an unwired, unflagged muse backend is
  **infeasible by default**, so `model=muse` on every pre-muse/stale `.env`
  404s `role_infeasible` (honest, referable, proxyable) instead of silently
  upward-falling-back to cortex.
- `/capabilities` (gateway + CLI) now reports SEVEN roles; docs and the
  in-CLI explain catalog updated throughout.
- `tests/test_live_capabilities.py`: generate-role discovery now covers muse
  and the third lobe state — a proxied dropped role must answer with the
  `X-Lobes-Proxied-By` marker naming `hosted_by` (a relayed non-2xx counts as
  an honest, marked relay; referral-only drops still demand the 404).

### Fixed

- `vllm-muse` boots behind `depends_on: service_healthy` on
  `vllm-embed`/`vllm-rerank` — a concurrent cold boot at 31B scale crashed
  CUDA-graph capture (`CUBLAS_STATUS_EXECUTION_FAILED`) and every restart then
  failed vLLM's free-at-boot check against the dirtied page cache (measured
  live on the physical Thor, 2026-07-17).

## [0.45.2] - 2026-07-17

### Added

- `docs/orin-profiles.md` — Jetson AGX Orin 64GB live-validation evidence
  (2026-07-16/17, issue #127 mesh work): the operator profile serving Gemma
  `senses` at its full 128K context on Ampere sm_87 (measured `gpu_mem_util`
  0.45; 0.30 refused with 2.25 GiB KV vs the 3.08 GiB that 131072 needs; KV
  pool 802,644 tokens / 6.12x concurrency), embedder/reranker probe results,
  why `cortex` (modelopt NVFP4 W4A4) is architecturally infeasible on Ampere,
  three Jetson/sm_87 divergences found live (csv-mode GPU access needs
  `runtime: nvidia`; the Parakeet base image is Spark-only — no sm_87
  kernels; unified-memory use far exceeds the util sum), the validated #127
  cortex proxy wiring, and the gateway→gateway audio-chaining limitation
  (readiness probes `/v1/health/ready`, which a peer gateway 401s/404s —
  first-class audio referral knobs are the phase-2 candidate).

### Changed

- `docs/machine-profiles.md` cross-links the Orin worked example from the
  custom-profile section.

### Fixed

- `docs/machine-profiles.md` documented `lobes init --machine` for selecting
  a profile; `init`'s actual flag is `--profile` (`--machine` belongs to
  `switch`). The wrong flag made the documented custom-profile flow error out.

## [0.45.1] - 2026-07-17

### Changed

- CI hardening: pin `astral-sh/setup-uv`'s `version` to `0.11.29` (was `latest`) and turn on `enable-cache: true` across both workflows (`tests.yml`, `publish.yml`, 6 usages total), so a transient GitHub release-CDN outage can no longer take down every job by failing uv's "resolve latest" step; two identical CI failures on PR #132 (2026-07-16 22:39Z/22:43Z, GitHub 503 HTML from the setup-uv download) prompted the change. The action's SHA pin, tokens, and cache-dependency-glob are unchanged

## [0.45.0] - 2026-07-16

### Added

### Changed

### Fixed

## [0.44.1] - 2026-07-14

### Added

- t5 live-verification evidence (colleague#320, spark-lobe go-live): docs/evidence/2026-07-14-strict-tools-spark-lobe-spark.txt — acceptance gates PASS at util 0.44/262144, lobes assess --strict-tools 3/3 legs PASS (strict+thinking was HTTP 500), colleague captured-bytes replay returns clean read_file with thinking intact via the armed gateway knob, MTP 100% draft acceptance under the constrained grammar, and the end-to-end colleague work repro delivers changed files in 4 clean steps (was 13 steps / 0 files). The deployment under test ran lobes-cli **0.44.0** (the release carrying the fix); 0.44.1 is docs-only — it records that verification, it is not itself what was verified

## [0.44.0] - 2026-07-14

### Added

- Strict, grammar-constrained tool calls with thinking enabled (colleague#320): new lobes.vllm_plugins package ships the qwen3_coder_thinking vLLM tool-parser plugin (overrides get_structural_tag to derive reasoning from the request's effective enable_thinking, fixing the strict+thinking HTTP 500 caused by the served build's hardcoded reasoning=False); the fleet template mounts it into vllm-primary via --tool-parser-plugin and flips PRIMARY_TOOL_CALL_PARSER default to qwen3_coder_thinking; lobes init materialises the plugin file into the deployment dir; the gateway gains an opt-in GATEWAY_FORCE_STRICT_TOOLS knob injecting function.strict=true on cortex-lane tools requests with a retry-once-without-strict fallback on schema-compile failures
- spec + plan (devague /think + /spec-to-plan): docs/specs/2026-07-14-lobes-serves-strict-grammar-constrained-tool-calls.md and docs/plans/2026-07-14-… — converged frame with the proven root cause (server-side parser-salvage mangle of off-template emissions, deterministic at temp 0) and the user decisions (arm strict BOTH ways; retry-without-strict on schema-compile failure); live evidence recorded in the in-repo eidetic store

## [0.43.0] - 2026-07-14

### Added

- Mesh-brain end-state implementation (#112, t2–t6): `orin-small` built-in shape — the small-model reference shape for the Jetson AGX Orin 64GB (minor + pooling gears, BOTH heavies dropped), shipped as declared-but-UNVALIDATED data per the #108 rule, with the `minor` opt-in role added to the shape `hosts` vocabulary (`OPT_IN_ROLES`) rather than dishonestly reusing the cortex slot
- Honest cross-box referral (#112 t3, the confirmed direct+referral decision): opt-in `PRIMARY/MULTIMODAL/EMBED/RERANK_PEER_ORIGIN` env vars (operator-declared full origins, never derived — #92); with a peer declared, `lobes capabilities`, gateway `GET /capabilities`, and the 404 `role_infeasible` body name the hosting peer (`hosted_by`); annotation only — the gateway never forwards a request to a peer, and zero peer config renders byte-identical pre-referral responses (pinned byte-for-byte in tests)
- Contract-test matrix (#112 t4): data-driven per-(built-in shape, dropped role) honesty tests — capabilities flag/omit, /v1/models omission, per-alias 404 with referral, no-outbound-connection tripwire — plus pinned t1 budget regressions (spark-lobe 262144 / thor-lobe 131072 cannot be silently lowered by a golden regen)
- Acceptance evidence (#112 t5): `scripts/accept-shape.sh` gains the orin-small arm and an opt-in referral phase; live Thor transcript `docs/evidence/2026-07-14-accept-referral-thor.txt` (referral 404s with a real declared Spark origin, cross-box cortex reachability, byte-for-byte shape restore)
- Mesh-brain end-state docs (#112 t6): the four recorded decisions, the measured co-residency tax table (cortex 131072→262144, senses 32768→131072), and evidence citations in `docs/deployment-shapes.md`, `lobes explain shapes`, and CLAUDE.md

## [0.42.0] - 2026-07-14

### Added

- Deployment shapes (#113 implementation): Shape schema + three built-in shapes as TOML data (machine-as-brain, spark-lobe, thor-lobe) over the #110 Profile machinery; shape-aware budget re-derivation as declared overrides with provenance (measured live: spark-lobe cortex 0.44/262144, thor-lobe senses 0.30/131072); pure shape×card render composition with per-(shape,card) goldens; lobes init --shape behind the dry-run/--apply contract (bare init byte-identical, --single conflict); gateway dev lane: PIP_EXTRA_INDEX_URL build-arg passthrough so from-source boxes can deploy a TestPyPI .devN build without hand-edits

### Fixed

- Dropped-lobe honesty: a request for an unwired dropped role (e.g. model=senses on a spark-lobe box) now returns 404 role_infeasible on every alias instead of silently rerouting to the primary model (#92 invariant, caught by the t5 contract tests)

## [0.41.2] - 2026-07-14

### Added

- Spec + plan for the mesh-brain end-state (#112, devague /think + /spec-to-plan): one heavy lobe per box as the far end of a backward-compatible, mixable shape axis — full-native budgets (spark-lobe cortex 262144, thor-lobe senses 131072), a declared-but-unvalidated orin-small shape, and cross-box direct + honest referral (opt-in peer config; no data-plane proxying); cheap gears co-reside everywhere

## [0.41.1] - 2026-07-14

### Added

- Spec + plan: deployment shapes — machine-as-brain (default) vs per-box mesh-brain lobe profiles (spark-lobe drops Gemma senses, thor-lobe drops the Qwen cortex), flag-first selection on lobes init, per-box honesty; end-state tracked as #112 (devague frame + 8-task plan)

## [0.41.0] - 2026-07-13

### Added

- **Per-machine hardware profiles** (spec + plan shipped in 0.40.2/0.40.3; this
  is the implementation — 13 tasks, 4 waves):
  - `lobes/machines/` — per-chip **strategy registry** (one `CardStrategy`
    module per chip: spark, thor, blackwell, generic) with a shared `SM_110`
    trait; the legacy `MachineProfile`/`MACHINE_PROFILES`/`detect_machine()`
    API is derived from the registry, every pre-existing test unchanged.
    Adding a chip = one file + one registration line.
  - `lobes/profiles/` — profile schema (per-role `feasible`/`model` + seven
    machine knobs), TOML built-ins (`spark`, `thor`, `base`), loader (operator
    profile in `<deploy-dir>/profiles/<name>.toml` overrides built-ins), and
    the profile→env renderer. Thor's four divergent knobs stay single-sourced
    in the machines registry and overlay at load time.
  - `lobes/runtime/_detect.py` — host card detection (device name + compute
    capability + total memory from `/proc/meminfo`; never nvidia-smi memory
    fields — they are `[N/A]` on Thor; UNKNOWN is first-class).
  - `lobes init` detects the card and applies the resolved profile
    (`--profile` overrides with a warning); an UNKNOWN card **warns and serves
    the conservative `base` profile** (4B generate model + the two 0.6B
    pooling gears, senses disabled, no 27B) instead of refusing (#107's
    unknown-card slice).
  - `lobes doctor`/`status` report the detected card (device, compute
    capability, memory) and the chosen profile, warning on forced/unvalidated
    combinations; init persists the choice as `LOBES_PROFILE`.
  - Hardware **feasibility honoured end-to-end**: `<PREFIX>_FEASIBLE=false`
    removes the role from `lobes capabilities` / `GET /capabilities` and the
    gateway answers 404 `role_infeasible` instead of silently rerouting
    (extends the #92 invariant).
  - Per-role **correctness probes** (`lobes assess --probes [--role r]
    [--timeout s]`): cortex known-answer, embed paraphrase-beats-unrelated,
    rerank relevant-doc-first; timeout counts as FAIL (catches the sm_110
    FLASH_ATTN hang that `/health` misses).
  - **Golden rendered artifacts** per shipped profile + the template-default
    surface (`tests/goldens/`, byte-diffed, GPU-less) — a change for one
    machine cannot silently alter another's rendering (the cross-machine
    no-breakage guard).
  - **Upgrade-compat proof**: a main-scaffolded deployment keeps working with
    the new CLI (zero bytes changed by upgrade, env-name tripwire, re-init is
    diffed and `--force --apply`-gated).
  - `docs/machine-profiles.md` + `lobes explain profiles` + honest support
    tables in README/CLAUDE.md. Thor validated live 2026-07-13: 3/3
    correctness probes pass on a clean boot (rerank **correct and stable**
    under TRITON_ATTN + eager); senses unconfirmed in that run. Orin / Orin
    Nano Super named but unvalidated.
  - Fleet compose knobs are env-parameterised (per-gear kv-cache dtype,
    `--attention-config`, enforce-eager, models per role); defaults reproduce
    the shipped GB10 behavior byte-for-byte. `MULTIMODAL_ATTENTION_BACKEND`
    deliberately kept pending the GB10 check (#109).
  - Live-validation findings, recorded as plan risk r7 and in the #109 thread:
    the fp8 `k_scale` assert did **not** reproduce on the pinned nightly —
    uncalibrated fp8-KV now boots with scale-1.0 warnings (an accuracy risk,
    not a crash) — and concurrent fleet first-boot on Thor fails a **memory
    race** (each engine's profiling window sees co-resident weight loads via
    page cache) regardless of profile; sequential bring-up (primary first)
    plus `drop_caches` after teardown boots clean. Boot ordering is not
    expressible as a per-gear env knob — follow-up work.

### Changed

- `GATEWAY_DEFAULT_MODEL` default is now **empty** = follow the primary gear's
  served name (was: hardcoded 27B id). Identical behavior on spark/thor;
  correct on the `base` profile where the primary serves a 4B.

### Fixed

- The cortex correctness probe disables thinking per-request
  (`chat_template_kwargs: {"enable_thinking": false}`, the `lobes route`
  idiom) — on the thinking-mode cortex the 16-token budget was consumed
  inside the `<think>` trace and a correct model failed the probe.

## [0.40.3] - 2026-07-13

### Added

- Plan (`/spec-to-plan`, converged): **per-machine hardware profiles** —
  `docs/plans/2026-07-13-lobes-fits-the-machine-it-lands-on-one-command-det.md`.
  Thirteen tasks over four file-disjoint dependency waves (fifteen drafted, two
  rejected during convergence): the **per-chip strategy
  pattern** + profile schema + spark/thor profiles, card detection, template
  parameterisation and per-role correctness probes (wave 1); `init` applies the
  profile and role-feasibility reaches capabilities/gateway (wave 2);
  `doctor`/`status` report the profile, and the Thor profile is validated on the
  physical board (wave 3); docs + an honest support table (wave 4).

  Per-chip knowledge goes behind a **strategy pattern** — one module per chip
  (`lobes/machines/<chip>.py`) owning its own detection signature, per-role knobs
  and provenance, plus a small shared registry. Adding a chip is one new file and
  one registration line; it must not mean editing shared tables, and a change for
  one chip must not be able to break another. No existing code is deleted:
  `MachineProfile`, `MACHINE_PROFILES` and `detect_machine()` keep working,
  rebuilt *from* the registry rather than duplicated.

  On Thor the reranker stays **served and advertised** — it runs, it is simply
  not yet correct; its ordering probe is recorded as a known failure pointing at
  #105 / #106, rather than the role being hidden.

- Spec + plan rework (same frame, re-converged): per-machine knowledge is keyed
  by **causal capability trait**, not board name — of the four Thor hand-edits,
  three trace to `sm_110` (the FLASH_ATTN pooling hang, the CUDA-graph classify
  fault) and one to the checkpoint's missing KV scales; none traces to "Thor the
  board". A machine profile becomes a named validated *bundle* (detection
  signature + memory budget + model-per-role + applicable traits), so an
  unrecognised board sharing a trait inherits its fix. Three new plan tasks:
  **golden rendered compose/.env per shipped profile** byte-diffed in CI, so a
  change for one machine cannot silently alter another's rendering (t13); an
  **unknown card warns and serves a conservative small base** — no 27B — instead
  of refusing, folding the unknown-card slice of #107 in (t14); and **upgrading
  lobes-cli never breaks an existing scaffold** — zero bytes changed in the
  deployment dir, old env names honoured, re-init always diffed + `--apply`
  (t15). Two GB10 verifications are parked as tracked risks: whether the fp8-KV
  crash is checkpoint-driven (shared fix) rather than Thor-specific, and whether
  the `VLLM_ATTENTION_BACKEND` env is truly dead on the GB10's pinned image
  before t3 deletes it. Packaging per #107: profiles ship in the wheel, no pip
  extras.

### Changed

### Fixed

- Spec correction: the first cut of this spec claimed lobes had no machine-profile
  concept at all. It does — `lobes/profiles.py` already ships `MachineProfile` /
  `MACHINE_PROFILES` (spark/thor/blackwell/generic) and `detect_machine()`, wired
  to `VLLM_MACHINE` and used by `switch`/`benchmark`. The real gaps, now stated
  honestly in the spec's before-state: it is one knob-set **per machine, not per
  role**; it lacks the knobs that actually mattered on Thor (KV-cache dtype,
  enforce-eager, model-per-role, role feasibility); the **fleet** compose (the
  default path) ignores it entirely and hardcodes the Spark values; its `thor` row
  is an unvalidated guess (`status="configured"`: flashinfer / 32768 / util 0.6)
  that live Thor testing **contradicts**; and `detect_machine()` silently falls
  back to `generic` instead of admitting it does not know the card.

## [0.40.2] - 2026-07-13

### Added

- Spec (`/think`, converged): **per-machine hardware profiles** —
  `docs/specs/2026-07-13-lobes-fits-the-machine-it-lands-on-one-command-det.md`.
  lobes detects the host card and applies a profile tuned for *that* box
  (feasible roles + model per role + util / context / quantization / KV dtype /
  attention backend / enforce-eager), instead of the fleet compose's hardcoded
  GB10 values. Ships Spark (default) + Thor as supported;
  Orin / Orin Nano Super are named but unvalidated; an unrecognised card
  refuses-or-warns rather than silently applying the Spark profile.
  Every role gains a **correctness** probe, not just `/health` — a role that is
  healthy but semantically wrong must fail.

  Motivated by bringing the fleet up on a Jetson AGX Thor (sm_110): the
  Spark-tuned template scored **1 of 4 roles correct on first boot** (senses
  clean; cortex crash-looped on an fp8-KV assert; embedder accepted requests and
  never answered; reranker killed its engine with `cudaErrorLaunchFailure`), and
  reaching 3 of 4 took four hand-edits to the generated compose. Rerank is still
  wrong there — tracked in #105 / #106.

### Changed

### Fixed

## [0.40.1] - 2026-07-11

### Added

- Real-socket streaming regression test: a dribbling chunked-SSE upstream through the real `open_upstream` asserts the first relayed frame arrives while the upstream is still mid-stream — the fake-upstream tests (whose `read()` already had read1 semantics) could never catch this bug class.
- Spec + plan for the fix under `docs/specs/` and `docs/plans/` (devague frame `lobes-gateway-relays-sse-streams-frame-by-frame-th`).

### Fixed

- Gateway SSE streaming: `_Upstream.read` now uses `HTTPResponse.read1`, so `data:` frames relay to the client as the backend generates them. Previously `HTTPResponse.read(65536)` blocked until 64 KiB or EOF, releasing a whole streamed turn in one terminal burst (frames=21 first=last=3.06s through the gateway) — full-turn latency before the first visible token for any `stream: true` client (issue #103; reported by colleague, agentculture/colleague#318).

## [0.40.0] - 2026-07-09

### Security

- The `Host` request header is validated against a strict host-authority allowlist before it is echoed as an advertised origin in `GET /capabilities`. Previously the c29 origin-resolution change reflected an unsanitized, client-controlled `Host` header into every role's `endpoint` (e.g. `http://evil.example/../<script>` or a `user@host` authority), so a client scraping the contract could be handed an attacker's origin to dial. An invalid host is now treated as "no origin supplied" (empty endpoint), exactly as an absent header is. The trusted operator override `GATEWAY_PUBLIC_URL` is unaffected. (SonarCloud `pythonsecurity:S5131`.)

### Added

- `scripts/live-check.sh` + `tests/test_live_capabilities.py` — a LOCAL, single-trigger, unattended pre-PR gate that dials every advertised role endpoint+path, every id in `/v1/models`, checks CLI/gateway agreement, reproduces Colleague's role-discovery path, and fails on deployed-gateway version skew. It FAILS rather than skips when armed. A 429 (pressure shed) and a 503 carrying `Retry-After` count as reachable; only a 404, a connection failure, a `Retry-After`-less 503, or a bare 5xx are faults. that dials every advertised role endpoint+path, every id in `/v1/models`, checks CLI/gateway agreement, reproduces Colleague's role-discovery path, and fails on deployed-gateway version skew. It FAILS rather than skips when armed. A 429 (pressure shed) and a 503 carrying `Retry-After` count as reachable; only a 404, a connection failure, a `Retry-After`-less 503, or a bare 5xx are faults.
- `lobes/gateway/_readiness.py` — a bounded background probe of each backend's `/health`, mirroring `PressureCache`. Tri-state (`True` healthy / `False` reached-but-unhealthy / `None` unreachable), daemon thread, socket-free `.current()`, and a probe that degrades to `None` on `OSError`, `http.client.HTTPException` and `ValueError`.
- `GET /health` now reports `{"version": ...}`, and `lobes doctor` gains a `gateway_version_match` check that fails on skew between the deployed gateway and the CLI wheel (issue #99).
- `lobes/gateway/_routing.py::is_unknown_model` — a pure predicate separating "unknown model id" from "unspecified model".
- Ground-truth perception probes for the `senses` role: a stdlib-generated solid-colour PNG whose colour the model must name, and a Chatterbox-synthesized word the model must transcribe. Both carry negative controls; the old placeholder-media tests are relabelled as wire checks.

### Changed

- **No cross-backend failover.** `order_backends` now returns at most one backend. A request naming the cortex model can never be answered by the Gemma backend, which protects the `final_authority` role contract from #81.
- `GET /v1/models` and `GET /capabilities.ready` are backed by the live readiness cache rather than by configuration. A wired-but-dead backend is no longer advertised.
- `RoleInfo.ready` is no longer an alias of `loaded` for `cortex`/`senses`/`embedder`/`reranker`. `build_role_registry` self-enforces the invariant: a supplied `backend_ready` map is authoritative, and a present `None`, a present `False`, and a missing key all mean not-ready.
- `lobes capabilities` / `lobes endpoint` now render the running gateway's `GET /capabilities` when it is reachable, falling back to an offline `.env`-derived view with `ready=false` on every role. The CLI and the gateway can no longer disagree, because there is now one derivation instead of two. The `--json` payload is keyed strictly by role name in both modes (byte-identical to the gateway's payload in live mode); the live-vs-offline signal travels on stderr (a `source:` notice) rather than as an extra top-level key, so a strict `set(keys) == ROLES` consumer never trips (Qodo).
- A backend is wired only when its `*_BASE_URL` is set; the `or *_SERVED_NAME` clause is gone (issue #97).
- `senses` is documented as vision-only intake. The checkpoint declares audio support but vLLM's `gemma4_unified` path does not serve it (issue #101); `stt` remains the supported speech path.
- `docs/gemma4-mtp-draft.md` carries a superseded banner: DSpark does not load on vLLM 0.23 (#75). Issue #69's disabled-experiment-entry criterion is closed answered-negative.
- README's quickstart no longer claims `lobes init` scaffolds the single-model deployment; the fleet duo has been the default since #69.

### Fixed

- **#91** — a dead cortex backend no longer surfaces as a terminal `404 model does not exist`. `handle_post` rewrote the model id once, before the failover loop, then retried the same body against a backend serving a different model, which correctly 404'd; the `4xx = client error, no failover` rule relayed that as terminal and killed multi-step agent loops. A dead, unreachable or warming owner now yields **503 + `Retry-After`** with `type: backend_unavailable`.
- **#92 / #95** — the gateway never advertises an origin built from its internal listen port. Precedence is `GATEWAY_PUBLIC_URL` (an operator override for a tunnel or Host-rewriting proxy) > the request `Host` header > an empty endpoint. `GATEWAY_PUBLIC_URL` is deliberately NOT defaulted: a defaulted `public_url` outranks `Host` and would advertise loopback to every remote client.
- **#96** — `AUDIO_URL` now reaches the gateway from the base fleet compose, so `stt`/`tts` stop advertising `ready=true` on a path that 404s when the audio overlay is not composed in.
- **#97** — `GET /v1/models` no longer advertises phantom backends wired from a `*_SERVED_NAME` alone against a `default_url` naming a container that need not exist.
- `ReadinessCache` seeds its initial per-backend snapshot with `dict.fromkeys` instead of a dict comprehension (SonarCloud `python:S7519`).
- `ReadinessCache.stop()` no longer clears its thread reference while the refresh thread is still alive, so a subsequent `start()` cannot spawn a second concurrent refresh loop; the join bound now covers a full sequential refresh pass and `start()` cleanly replaces a genuinely-dead thread (Qodo reliability).
- An unknown model id returns `404 model_not_found` instead of being silently served by the default backend under a different model's weights. Unknown-ness is decided against the routing table, never against the readiness-filtered `/v1/models` list, so a wired-but-dead backend still yields 503 rather than 404.
- `test_live_main_text_returns_nonempty_content` no longer fails on a thinking model: `max_tokens=16` was consumed entirely by the reasoning trace, leaving `content=None` and `finish_reason=length` (issue #93).

## [0.39.0] - 2026-07-07

### Added

- **`preserve_thinking` on the cortex/main vLLM lane** — both compose templates
  now launch the primary/cortex service with `--default-chat-template-kwargs
  '{"preserve_thinking": true}'` next to `--reasoning-parser=qwen3`, so the
  served Qwen3.6 chat template retains **all** historical `<think>` blocks across
  a multi-turn conversation (default keeps only the reasoning after the last user
  turn). Default-on but per-request overridable; scoped to the cortex/main lane
  (embed/rerank/senses untouched, `lobes route` still forces
  `enable_thinking=false`). Closes #93.
- **Reasoning-aware `lobes.minor` client** — `assistant_turn_from_response()`
  builds an assistant history message preserving the `reasoning` /
  `reasoning_content` trace, and a new `history=` parameter on `chat_completion`
  / `chat_text` round-trips it back on subsequent turns (single-turn behaviour
  unchanged). Part of #93.
- **`lobes assess --preserve-thinking`** — a read-only two-turn token-delta
  diagnostic that proves the reasoning round-trip is live (prompt-token count
  rises when the assistant `<think>` history is preserved vs content-only).
  Part of #93.

## [0.38.0] - 2026-07-04

### Added

- `GET /capabilities` now advertises a **client-reachable** origin for every role's `endpoint` — derived from the request `Host` header, overridable with the new optional `GATEWAY_PUBLIC_URL` env (for tunnels / Host-rewriting proxies) — so a consumer can dial any role's `endpoint` directly instead of a non-routable internal host (`vllm-primary:8000`, `realtime:8080`). This covers all six roles, including `stt`/`tts`. Closes #87.
- The realtime bridge exposes `GET /v1/health/ready`, aggregating Chatterbox (TTS) + Parakeet (STT) readiness into one signal the gateway live-probes.

### Changed

- `lobes status` is now **fleet-aware**: on a fleet deployment it reports per-gear container states and points at `lobes fleet status` / `lobes capabilities`, instead of the contradictory single-model `state: model-gear-vllm — not created` line printed next to `health: ok`. A single-model deployment's output is byte-for-byte unchanged. Closes #84.
- `GET /capabilities` reports `stt`/`tts` `ready` from a **live** probe of the audio backend (not merely `AUDIO_URL` being set), so an advertised-ready audio role is genuinely consumable; `lobes capabilities --json` keeps the configured signal since the host CLI can't reach the internal backends. Closes #89 (readiness).

### Fixed

- The gateway returns a clear **503** (`Retry-After`) for `/v1/audio/transcriptions` and `/v1/audio/speech` when the audio backend is reachable but still warming (Chatterbox/Parakeet loading, or a poisoned-CUDA context now surfaced honestly by Chatterbox's `/v1/health/ready`), distinct from the **502** for a genuinely unreachable backend — closing the "advertised ready but 502s / not client-consumable" gap. Closes #89.
- The gateway's audio-readiness probe (`probe_audio_ready`) now degrades a malformed `AUDIO_URL` (a non-numeric port makes `urlsplit(...).port` raise `ValueError`) or a broken HTTP exchange (`http.client.HTTPException`) to "readiness unknown" instead of letting the exception crash the `GET /capabilities` / `POST /v1/audio/*` handler — mirroring `open_upstream`'s guard. (Qodo review, #90.)
- `build_role_registry` clamps the stt/tts `ready` signal on the audio overlay being configured, so an unconfigured overlay can never report `ready=True` with an empty endpoint even if a caller passes `audio_ready=True` — the public builder now enforces the "unconfigured ⇒ not ready" invariant its docstring already promised. (Qodo review, #90.)
- `cmd_status` is split into `_cmd_status_fleet` / `_cmd_status_single` render helpers (mirroring the existing `_cmd_status_pressure`), leaving the verb as pure dispatch — dropping its Cognitive Complexity from 16 to under the gate's 15. Each helper uses a single-`return` `if/else` (like `_cmd_status_pressure`) rather than an early return, so it renders its one exit path without tripping `python:S3516` ("always returns the same value"). Behavior-preserving; the fleet and single-model outputs are byte-for-byte unchanged. (SonarCloud `python:S3776` + `python:S3516`, #90.)

## [0.37.0] - 2026-07-03

### Changed

- Under swap/iowait pressure the gateway now **sheds** a `main`/`cortex` or `multimodal`/`senses` request with HTTP 429 + `Retry-After` (busy — retry shortly) instead of silently degrading it onto a different model. The degrade-to-minor path is removed outright (no `LOBES_PRESSURE_POLICY` toggle); an explicit `minor` request is still served as the floor. Busy is disclosed via `X-Lobes-Tier-Reason: busy` and an OpenAI-shaped `server_busy` error body, distinct from the hard 502 `upstream_unavailable`; callers (the acp `vllm-local` provider, colleague, generic OpenAI SDKs) must retry with backoff. `lobes status --pressure` and the gateway `GET /status` now report the busy-policy state (`mode`/`shed`/`retry_after`). The trigger reuses the existing swap/iowait signal; the tunable thresholds (#86) and the `/proc` sampler are untouched.

### Fixed

- Pressure no longer crosses capabilities: on a default fleet with `minor` unwired, a pressured `cortex`/`main` request previously fell through the tier upward-fallback onto the Gemma multimodal gear (a different capability), silently answering an authoritative-reasoning request with a perception model. Removing the degrade path makes that cross-capability substitution structurally impossible. Closes #85.

## [0.36.2] - 2026-07-03

### Fixed

- Fleet gateway now receives the pressure-policy thresholds (`LOBES_SWAP_DEGRADED_THRESHOLD` / `LOBES_IOWAIT_DEGRADED_THRESHOLD`) via its compose `environment`, so operators can tune the swap/iowait degrade triggers per box through `.env`. Previously those vars were only read from the gateway container's env, which the compose never populated, so the knobs silently stayed on the code defaults (75 / 50). Notably lets a box with an unreliable `/proc/pressure/io` (phantom high iowait on an idle disk, e.g. the DGX Spark GB10) raise `LOBES_IOWAIT_DEGRADED_THRESHOLD` to 100 instead of permanently degrading the generate lane. Documented in `env.example` + regression test added.

## [0.36.1] - 2026-07-03

### Fixed

- Gateway GET /capabilities (and `lobes capabilities` when read via the gateway) reported catalog-native context instead of the served `--max-model-len`, because the gateway container's environment lacked `PRIMARY_`/`MULTIMODAL_`/`EMBED_`/`RERANK_MAX_MODEL_LEN`. The fleet compose now passes those into the gateway service so the #81 served-context overlay resolves (cortex 131072, senses 32768); regression test added.

## [0.36.0] - 2026-07-03

### Added

- cortex/senses role-based Colleague contract: six first-class roles (cortex, senses, embedder, reranker, stt, tts) with responsibilities/forbidden_responsibilities (#81)
- `lobes capabilities [--json]` and `lobes endpoint <role>` — read-only role discovery over the role registry
- Gateway GET /capabilities — machine-readable {role: {endpoint, model, context, ready, responsibilities, ...}} contract for Colleague
- `lobes up <role>` and `lobes up colleague-stack` — role-based serving (dry-run by default, --apply to run; colleague-stack bundles the audio overlay)
- lobes measure [--role] [--json] — per-role runtime-only metrics (TTFT/decode-tps/prefill, docs-per-sec, RTF)
- lobes benchmark --profile {cortex-only,cortex+senses,senses-direct,qwen-nvfp4-vs-bf16,all} — comparison profiles (runtime-only)
- lobes explain roles (aliases: colleague, colleague-stack, capabilities) and docs/colleague-stack.md

### Changed

- Fleet context rebalance: cortex (Qwen 27B MTP) served at 128K, senses (Gemma 4 12B) at 32K (util 0.14, provisional); default fleet budget 0.30+0.14+0.06+0.06 = 0.56
- cortex->primary and senses->multimodal added to catalog.TIER_ROLE as the primary contract; main|multimodal|hard|normal|cheap|minor kept as back-compat aliases; brain forbidden
- lobes fleet status now reports the always-on Gemma (senses) container (FLEET_MULTIMODAL added to FLEET_CONTAINERS)

### Fixed

- Latent circular import between lobes.roles and lobes.gateway surfaced when lobes.roles was imported first

## [0.35.0] - 2026-07-02

### Added

- catalog: coolthor/gemma-4-12B-it-NVFP4A16 (Gemma 4 12B NVFP4 base it-model) as the new default multimodal gear, native MTP wired ON by default ({"method": "mtp", "model": "google/gemma-4-12B-it-assistant", "num_speculative_tokens": 1}) -- measured 28.6 tok/s decode @ 57.9% draft acceptance, the fastest Gemma config on the DGX Spark (docs/vllm-nightly-migration.md §7)
- fleet compose: opt-in vllm-multimodal-coder service (profiles: [multimodal-coder]) so the demoted coder gear stays reachable; gateway wires an opt-in MULTIMODAL_CODER_BASE_URL/MULTIMODAL_CODER_SERVED_NAME backend + a multimodal-coder alias, added only once wired

### Changed

- catalog: sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4 demoted from role_hint=multimodal to role_hint=candidate (kept, cite-don't-delete) -- native MTP measured only 30.8% draft acceptance on the coder fine-tune, not worth wiring by default
- gateway: _DEFAULT_MULTIMODAL now points at coolthor/gemma-4-12B-it-NVFP4A16; the multimodal/normal tier aliases resolve to it
- docs/gemma-4-12b-nvfp4.md, README.md, docs/gateway-fleet.md, docs/qwen3-14b-nvfp4.md updated to describe both Gemma gears (default base + opt-in coder) and cite docs/vllm-nightly-migration.md §7 for the benchmark evidence

## [0.34.1] - 2026-07-01

### Added

- docs/gemma-4-12b-nvfp4.md — first throughput/prefill benchmark for the Gemma 4 12B multimodal gear on the DGX Spark GB10: ~23 tok/s single-stream decode (23.0 sustained over 1,500 tok), prefill ~2,650 tok/s (847 tok) / ~1,954 tok/s (6,682 tok), on vLLM 0.23.1rc1.dev672 native gemma4_unified
- README acknowledgement of Mieszko Syty (FutureProofHomes; Jetson AI Lab) alongside shahizat

### Changed

- docs/gemma-4-12b-nvfp4.md notes a config-drift follow-up: on the current :nightly-audio (dev672) image the default lane util 0.12 + VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 no longer boots 8192 co-resident (cudagraph accounting changed) — benchmarked at util 0.15 with a trimmed cudagraph capture set

### Fixed

- docs/gemma-4-12b-nvfp4.md — made the #75 speculative-decoding section internally consistent: it now reads as CLOSED (route resolved, wire/measure/verdict not implemented) throughout, matching the "Resolved" bullet, instead of framing #75 as active work (Qodo); also corrected the stale claim that the 12B lane decodes slower than the primary — the benchmark shows it out-decodes the primary single-stream (~23 vs ~18–19 tok/s)

## [0.34.0] - 2026-07-01

### Added

- Gemma 4 12B multimodal gear now SERVES (text + image + audio) — live-validated on the DGX Spark GB10 via vLLM nightly's native gemma4_unified class (#71/#73); catalog status promoted configured → load-tested

### Changed

- Dockerfile.vllm-gemma4 rebased FROM vllm/vllm-openai nightly (pinned by digest) + the vllm[audio] extra (librosa==0.11.0 soundfile==0.14.0 av==17.1.0 soxr==1.1.0, pinned to the live-validated set) (was NGC 26.06 / vLLM 0.22.1 + a transformers overlay); vllm-multimodal compose/env now set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 and default MULTIMODAL_MAX_MODEL_LEN 8192 (co-resident util-0.12 KV holds ~24K tokens, not the 128K native)

### Fixed

- Gemma 4 12B serve blocker root-caused and fixed: gemma4_unified has heterogeneous per-layer head sizes (40 sliding@256 + 8 full@512) that only vLLM's native class handles — released vLLM ≤0.22.1 fell back to the transformers backend and crashed the full-attention o_proj (4096≠8192); a TRITON_ATTN backend flag does not fix it

## [0.33.1] - 2026-07-01

### Added

- Issue #75 spec + plan (Gemma 4 12B gear speculative decoding) converged via devague /think + /spec-to-plan
- docs/gemma4-mtp-draft.md — resolved spec-decode draft route (DSpark draft_model first; native google/gemma-4-12B-it-assistant recorded as escalation candidate)

### Changed

- docs/gemma-4-12b-nvfp4.md — added the Speculative decoding (#75) before-state / gap / scope-split subsection

## [0.33.0] - 2026-06-30

### Added

- Custom vLLM image Dockerfile.vllm-gemma4 (FROM nvcr.io/nvidia/vllm:26.06-py3, vLLM 0.22.1, + a pinned from-source Transformers 181beb3) so the Gemma 4 12B gemma4_unified multimodal gear loads (#71)
- MULTIMODAL_IMAGE override for the vllm-multimodal service (local build by default; optional ghcr.io/local-registry tag)
- MULTIMODAL_ATTENTION_BACKEND env (TRITON_ATTN) for Gemma 4 non-square attention

### Changed

- Multimodal gear quantization corrected to compressed-tensors (was modelopt_fp4) after live validation (#71)
- Removed the invalid gemma4_mtp speculative_config from the multimodal gear (vLLM Gemma4 MTP needs a separate gemma4_assistant draft model)

### Fixed

- docs/compose comments now name the correct base image and MULTIMODAL_IMAGE registry semantics

## [0.32.1] - 2026-06-30

### Added

- docs/specs: Gemma 4 12B multimodal-duo spec (issue #69) — /think frame for default-serving the Qwen3.6-27B-MTP + Gemma4-12B duo as main/minor/multimodal tiers (vision+audio), native-MTP on by default, DSpark draft as a disabled experiment, 14B demoted to a legacy candidate
- docs/plans: buildable plan for the Gemma duo (9 tasks across 5 dependency waves, 6 accepted-risk objects) via /spec-to-plan — covers all 26 spec targets; resolves the main/minor/multimodal pressure-ladder seam as a first-class task

### Changed

- Reworded the three Gemma-risk markers in `catalog.py` / `runtime/_parser.py` from bare `TODO(risk …)` comments to `Risk … (pending #71)` — the deferred live-validation work is tracked in issue #71 (gemma4_unified won't load on released vLLM images), so the comments now cite the tracking issue instead of an untracked TODO (clears SonarCloud `python:S1135`).

### Fixed

- Gateway: re-wire the legacy 14B `middle` generate backend from `MIDDLE_BASE_URL` / `MIDDLE_SERVED_NAME` in `build_config()`. The #69 14B demotion dropped the wiring but the compose template still ships the `vllm-middle` profile + those env vars, so enabling the profile silently fell back to the primary; the 14B is again reachable by its explicit served name (and, as intended, gets no tier alias). (Qodo)
- Gateway: a `GATEWAY_ALIASES` operator override keyed by a *legacy* tier alias (`hard`/`cheap`/`normal`) is now honoured on the pressure-aware tier path. Tier requests normalize to the new vocabulary (`hard`→`main`) before the alias lookup, which bypassed a legacy-keyed override; `build_config()` now mirrors a tier-keyed override onto its vocabulary synonyms (explicit keys still win). (Qodo)

## [0.32.0] - 2026-06-30

### Added

- Third capability tier: opt-in `vllm-middle` 14B-NVFP4 generate gear (`COMPOSE_PROFILES=middle`, GPU mem-util 0.12), inference-only (not a LoRA base).
- Gateway capability-tier aliases — callers send `model=cheap|normal|hard` and the gateway resolves to the 4B/14B/27B generate gears (same-task alias on top of task-family routing) with upward fallback when a tier is absent.
- Read-only host memory-pressure sampler (`swap_used_percent`/`iowait_percent` from /proc) and a swap/iowait pressure policy with a degraded-mode state machine (env-overridable thresholds).
- Pressure-aware tier downgrade at the gateway with an `X-Lobes-Override` bypass header; the served tier and reason cross the OpenAI boundary via `X-Lobes-Tier` / `X-Lobes-Tier-Reason` response headers.
- `lobes status --pressure` — read-only snapshot of the current tier ceiling, mode, reason, and live swap/iowait.
- `scripts/validate-tiers.sh` + `docs/validate-tiers.md` — operator-run live validation harness for the three-tier fleet on the Spark.

### Changed

- 27B primary default served context trimmed 256K→128K (`PRIMARY_MAX_MODEL_LEN=131072`) and `PRIMARY_GPU_MEM_UTIL` lowered to 0.45 so the co-resident 14B middle gear fits within the 128GB unified-memory budget (0.45 + 0.12 + 0.10 + 0.06 + 0.06 = 0.79).

## [0.31.1] - 2026-06-27

### Changed

- Mutation-safety prose in `lobes learn` and CLAUDE.md now lists the `fleet up` / `fleet down` write verbs (was only in the `--json` payload).
- CLAUDE.md documents the turn-on/turn-off lifecycle explicitly (`serve`/`stop` and `fleet up`/`down`) instead of leaving it implicit in the verb names.

## [0.31.0] - 2026-06-26

### Added

- `lobes benchmark --all-lobes --concurrency auto`: per-lobe (minor + primary) performance benchmark routed through the gateway — single-stream decode tok/s, prefill TTFT, concurrent throughput with auto-ramp to the throughput knee (req/s + p50/p95 latency + ms/token), plus the logprobs cat soft-score, rendered as one combined minor-vs-primary report with per-metric deltas.
- `lobes eval cat --score logprobs --mode open|closed`: read-only 'Where is the cat?' temporal-reasoning probe, scored by logprobs (softmax over candidate-location full-sequence echo logprobs as the headline, with a chat first-token-mass cross-check and graceful fallback when echo is unavailable).
- `lobes.bench` package: `cat_probe` (deterministic, seeded timestamped-narrative generator with exactly one unambiguous current location; open + closed modes), `cat_score` (echo-softmax headline scorer + first-token cross-check + fallback), and `report` (per-lobe markdown report renderer with minor-vs-primary deltas).
- `lobes.minor` logprobs plumbing: `chat_completion` now forwards `logprobs`/`top_logprobs`; new `completions_echo` (full-sequence `/v1/completions` echo scoring) and `gateway_supports_echo` capability probe (never raises; lets callers fall back).
- `lobes.assess` per-lobe perf engine: `measure_prefill_ttft`, `run_concurrent` (requests/sec + p50/p95 latency + ms/token), and `auto_ramp_concurrency` (1→2→4→… ramp with plateau/knee detection).

## [0.30.0] - 2026-06-26

### Added

- **The `minor` lobe — a cheap, warm co-resident Qwen3.5-4B small-brain** (issue
  #64). A new switchable catalog gear `Qwen/Qwen3.5-4B` (`role_hint="minor"`,
  served **bf16** — the first unsloth-LoRA fine-tune target; multimodal, served
  text-only via `--language-model-only`), reachable both as a switchable gear and
  as an opt-in warm co-resident backend alongside the 27B primary.
  - **New read-only verbs:** `lobes run minor "<prompt>"` (call the minor model),
    `lobes route "<text>"` (classify a task across catalog gears with an
    escalate flag + confidence), and `lobes eval minor --suite <path>` (run a
    JSONL eval suite). All three default `--base-url` to the gateway
    (`http://localhost:8000/v1`) and reuse a new stdlib-only urllib client
    (`lobes.minor`) — no new runtime dependencies.
  - **Governance + escalation** (`lobes.minor.governance`): the minor role may
    prepare/classify/format/validate/suggest/summarize/route, and escalates on
    forbidden actions (approve/finalize/delete/deploy/architectural) or any of
    five escalation conditions. Role-keyed, not model-keyed.
  - **Warm co-residency:** an opt-in `vllm-minor` fleet service (compose profile
    `minor` + `MINOR_BASE_URL`/`MINOR_SERVED_NAME` gateway env gate); the gateway
    routes the minor model id to it with failover to the primary. Default fleet
    behavior is unchanged.

### Changed

- **`runtime/_parser.py` recognizes the Qwen3.5 family** → `qwen3_coder` (it
  emits the XML function-call format, not Hermes JSON).
- **Catalog supports an unquantized bf16 generate gear** via a `quantization="none"`
  sentinel that `lobes switch` normalizes to "omit `--quantization`" and surfaces
  as a required compose edit.

### Fixed

## [0.29.0] - 2026-06-24

### Added

- **Memory-discipline "Conventions and workflow" section in `CLAUDE.md`** — a
  per-task *recall-before / remember-after* convention (scope localized to this
  repo's nick) so the vendored `remember` / `recall` skills are actually used,
  not just present: `/recall` before non-trivial work to build on prior
  decisions instead of re-deriving them, and `/remember` when a non-obvious
  decision, constraint, fix-and-why, or hard-won gotcha surfaces. The section
  documents this repo's memory as **in-repo and public** — records resolve to
  `<repo-root>/.eidetic/memory` (committed, team- and mesh-shared). Inserted
  idempotently (skipped if already present), slotted under an existing
  "Conventions and workflow" heading when one exists, else appended.

### Changed

- **Refreshed the `remember` + `recall` wrappers from eidetic-cli 0.10.0**
  (cite-don't-import) — picks up eidetic's **project-local store default**: the
  files backend now resolves per record by visibility — PUBLIC records inside a
  git repo go to `<repo-root>/.eidetic/memory` (committed, team-shared), PRIVATE
  records (or any record outside a repo) go to `$HOME/.eidetic/memory` (never
  committed), an explicit `EIDETIC_DATA_DIR` still wins, and recall reads both
  stores and merges. Also carries the 0.9.3 hardening (interactive-stdin guard,
  `help` as a search term, SIGPIPE-safe suffix parsing). **Recipe policy
  override (the wrappers here are NOT byte-verbatim):** the injected default
  visibility is flipped from eidetic's `private` to **`public`**, so a plain
  `/remember` lands the note in `./.eidetic/memory` in this repo, kept as part
  of the repo — pass `--visibility private` to route a record to `$HOME`
  instead. `remember` drives `eidetic remember` (idempotent upsert of one JSON
  record or an NDJSON batch on stdin); `recall` drives `eidetic recall` with
  four search modes (exact / approximate / keyword / hybrid). Each `SKILL.md` is
  localized only in the illustrative `--scope <nick>` examples (Provenance keeps
  "First-party to eidetic-cli"). Runtime dep: the `eidetic` CLI on PATH (else a
  local eidetic-cli checkout with `uv`) — **`eidetic >= 0.10.0`** for the
  in-repo routing; on an older CLI the public records still work but are stored
  in `$HOME/.eidetic/memory` instead of in-repo. Propagated by rollout-cli's
  `eidetic-memory` recipe.

## [0.28.1] - 2026-06-26

### Added

- **`docs/tensorrt-llm-investigation.md`** — a dated desk investigation (no live
  run) of serving the MTP 27B primary (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`)
  with **TensorRT-LLM** (`trtllm-serve`) on the DGX Spark (GB10/SM121) instead of
  vLLM. **Verdict: not yet** — TRT-LLM MTP spec-decode is DeepSeek-only in stable
  releases and the Qwen3.6 hybrid GDN/DeltaNet kernels are RC-only (both land in
  1.3.0 RC builds); serving on a stable TRT-LLM today would forfeit the ~2.4×
  decode win the checkpoint exists for. Records the engine-integration seam (the
  request path — gateway routing + `lobes assess`/`benchmark` — is already
  engine-agnostic, while the gateway `/status` `vllm:*` metrics path, `catalog.py`,
  `switch.py`, templates, and `VLLM_*` env vars are vLLM-specific), a feasibility
  table by dimension with confidence levels, a comparison against the recorded
  vLLM baseline, a minimal spike recipe, an explicit revisit trigger (TRT-LLM
  1.3.0 stable), and 11 cited sources. Linked from the README per-model notes.

### Changed

### Fixed

## [0.28.0] - 2026-06-23

### Added

- **Vendored the `remember` + `recall` memory skills from eidetic-cli**
  (cite-don't-import) — the write/read halves of eidetic's shared
  `~/.eidetic/memory` surface, so this agent (Claude and its colleague backend)
  can persist facts across sessions and recall them later, sharing one store.
  `remember` drives `eidetic remember` (idempotent upsert of one JSON record or
  an NDJSON batch on stdin, dedup by id + content hash); `recall` drives
  `eidetic recall` with four search modes — exact / approximate / keyword /
  hybrid — each hit carrying text, full provenance metadata, a relevance score,
  and a freshness signal. The `.sh` wrappers are byte-verbatim from eidetic-cli
  (their first-party origin); each `SKILL.md` is localized only in the
  illustrative `--scope <nick>` examples (Provenance keeps "First-party to
  eidetic-cli"). Both default to this agent's PRIVATE scope, reading the suffix
  from `culture.yaml`. Runtime dep: the `eidetic` CLI on PATH (else a local
  eidetic-cli checkout with `uv`). Propagated by rollout-cli's `eidetic-memory`
  recipe.

## [0.27.0] - 2026-06-22

### Changed

- **Renamed the tool from `model-gear`/`model` to `lobes`/`lobes-cli`.** The
  binary is now **`lobes`** (`lobes switch`, `lobes serve`, `lobes assess`, …),
  the import package is **`lobes`**, and the PyPI distribution is **`lobes-cli`**.
  The deployed Culture agent is renamed `model-gear` → `lobes` (`culture.yaml`,
  `AGENTS.md`).
- **Deployment dir is now `~/.lobes`** (env `$LOBES_DIR`). The legacy
  `$MODEL_GEAR_DIR` and `~/.model-gear` are still resolved as fallbacks, so a
  pre-rename deployment keeps working with the renamed CLI without redeploying.
- **Deployment-internal names are intentionally kept as `model-gear`** so a live
  fleet isn't disrupted: Docker `container_name`s (`model-gear-vllm`,
  `model-gear-gateway`, …), `mg-logwrap.sh`, `MODEL_GEAR_LOG_DIR`, and the
  served-model id are unchanged.

### Added

- **`model` is kept as a deprecated alias command** for `lobes` (same entry
  point); `--version`/help reflect whichever name was invoked.
- **`model-gear` is published on PyPI as a deprecated alias** of `lobes-cli`: a
  metadata-only shim package (`packaging/model-gear/`) that depends on
  `lobes-cli==<same version>`, plus a `publish-alias` job in `publish.yml` that
  builds and publishes it after the main release.

## [0.26.4] - 2026-06-21

### Changed

- Relicensed the project from MIT to Apache 2.0 — full Apache 2.0 LICENSE text, pyproject `license`/classifier metadata, and a new README License section. Aligns with sibling AgentCulture repos (e.g. colleague, data-refinery-cli).

## [0.26.3] - 2026-06-21

### Fixed

- Doc consistency: `docs/mistral-small-3.2-24b-nvfp4.md` and
  `docs/qwen3.6-35b-a3b-nvfp4.md` still framed Mistral as the **default** fleet
  fallback the gateway pairs with the primary. The fleet has run one *generate*
  backend by default since the single-backend default (#42); the warm fallback is
  opt-in. Reframed both docs to match — closing the drift with the `model explain`
  catalog corrected in 0.26.2.
- Corrected the Mistral doc's "How it runs in the fleet" section, which described
  a fallback wiring that no longer exists: `FALLBACK_MODEL`/`FALLBACK_MAX_MODEL_LEN`
  /`FALLBACK_GPU_MEM_UTIL`/… `.env` keys "scaffolded by `model init --fleet`" and a
  shipped `model-gear-vllm-fallback` service. The current templates ship **no**
  fallback service and the gateway reads only `FALLBACK_URL` + `FALLBACK_SERVED_NAME`
  (set after you manually add a `vllm-fallback` service). Following the old text
  produced a non-working config; rewrote it to the actual two-step opt-in, matching
  `docs/gateway-fleet.md` → "Adding a fallback".

## [0.26.2] - 2026-06-21

### Changed

- Doc alignment pass across the audio + fleet surfaces (no behavior change):
  - `docs/chatterbox-tts.md`: healthcheck shows `python3.12` (not the stale
    `python3`) and the compose snippet includes `container_name:
    model-gear-chatterbox` — matching the 0.26.1 fixes.
  - `docs/realtime-pipeline.md`: the TTS service is `chatterbox` (was `tts`); the
    overlay scaffolds a Chatterbox Dockerfile too.
  - `docs/openai-api.md`: corrected the auth caveat — the gateway is a
    pass-through and is *not* auth-aware for *any* proxied endpoint (the previous
    wording implied it gated `/v1/chat/completions`).
  - `docs/gateway-fleet.md`: endpoint list now includes `/v1/audio/*`; added an
    "Auth (known limitation)" note.
  - `model explain gateway` / `model explain tunnel` (`explain/catalog.py`): added
    the gateway-not-auth-aware caveat; fixed the Mistral entry (opt-in fallback
    candidate, not the active default pairing); listed `tunnel`/`fleet` as write
    verbs.
  - `model learn` (`learn.py`): added an "Auth / exposure" section + an
    `auth_exposure` JSON field, and `model explain tunnel`/`gateway` pointers.

## [0.26.1] - 2026-06-21

### Fixed

- `model init --fleet --audio` now scaffolds `Dockerfile.chatterbox`. The
  Chatterbox sidecar landed in 0.25 (the compose `chatterbox` service builds from
  `Dockerfile.chatterbox`), but the build file was never added to
  `AUDIO_TEMPLATES`, so the scaffold omitted it and `docker compose build
  chatterbox` failed with "Dockerfile.chatterbox: no such file". Added it to the
  audio template set (twin of the `Dockerfile.realtime` / `Dockerfile.parakeet`
  wiring) so the audio overlay can actually build and serve TTS.
- `model fleet status` now reports the TTS gear. `FLEET_TTS` still pointed at the
  old `model-gear-tts` container name, but the Chatterbox sidecar renamed the
  container to `model-gear-chatterbox` — so status listed the live TTS gear as
  "not created". Pinned `FLEET_TTS` to `model-gear-chatterbox` and added a test
  that asserts every `FLEET_AUDIO_CONTAINERS` name matches a `container_name:` in
  the packaged audio compose (catches future rename drift).
- Chatterbox container now reports healthy. Its `Dockerfile.chatterbox` installs
  the interpreter as `python3.12` (no `python3` symlink), but the compose
  healthcheck called bare `python3` — which exec-failed every interval, pinning
  the working container at "starting"/"unhealthy". Switched the healthcheck to
  `python3.12` and added a test tying the healthcheck interpreter to the one the
  Dockerfile provides.

## [0.26.0] - 2026-06-21

Documentation pass for the realtime audio overlay and the OpenAI API front: a
feature doc per audio backend, a consolidated endpoint reference, and the same
information surfaced through `model learn`, `model explain`, the README, and
`CLAUDE.md`.

### Added

- `docs/parakeet-stt.md`: per-model feature doc for the **Parakeet** STT backend
  (`nvidia/parakeet-tdt-0.6b-v2`, NeMo ASR) — the only audio model that lacked one.
  Covers the HTTP contract, the real (model-loaded + CUDA-live) readiness probe,
  fleet integration, the stale-CUDA-context runbook, and why it is not a switchable
  catalog gear.
- `docs/openai-api.md`: consolidated **OpenAI-compatible API surface** reference —
  every endpoint (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/rerank`, `/v1/score`, `/v1/audio/transcriptions`, `/v1/audio/speech`,
  `/v1/models`, `/v1/models/supported`, `/health`), routing semantics (name /
  default / failover / SSE / audio fan-out), per-endpoint `curl` examples, the
  loaded-vs-supported split, and auth/exposure.
- `model explain` topics: `realtime` / `audio` (the `/v1/audio/*` overlay),
  `transcribe` / `stt` / `parakeet` (STT), `speak` / `tts` / `chatterbox` (TTS),
  and `api` / `openai` (the endpoint surface); linked from the explain root.
- `model learn` now documents the realtime audio overlay and the OpenAI API surface
  (text + `--json` `realtime_audio` / `api_surface` fields).
- README sections for **Realtime audio (STT + TTS)** and **The OpenAI-compatible API
  surface**, plus the two audio backends added to the per-model notes.

### Changed

- `CLAUDE.md`: documents the realtime audio overlay alongside the fleet; the CLI
  package tree now lists the `gateway/`, `realtime/`, `explain/`, and `catalog.py`
  surfaces.

### Fixed

- `docs/realtime-pipeline.md`: removed the stale `NGC_API_KEY` bring-up step (a
  Magpie leftover — Chatterbox needs no NGC key) and documented the TTS → STT
  round-trip in `scripts/audio-smoke.py`.

## [0.25.0] - 2026-06-21

### Added

- **Chatterbox TTS sidecar** (`model_gear/realtime/chatterbox_server.py`): a
  FastAPI HTTP server (`GET /v1/health/ready`, `POST /v1/audio/synthesize`) that
  wraps Resemble AI's Chatterbox model and returns raw PCM16 mono 24 kHz audio.
  Supports zero-shot voice cloning via a `.wav` reference path.  Runs as the
  `chatterbox` fleet service built by the new `Dockerfile.chatterbox` (arm64
  cu128 recipe).
- `[chatterbox]` optional-deps group (`fastapi`, `uvicorn`) in `pyproject.toml`.
- `docs/chatterbox-tts.md`: bake-off numbers, arm64 install recipe, sidecar HTTP
  contract, and integration notes.
- `model_gear/templates/fleet/Dockerfile.chatterbox`: arm64 CUDA build — rebased
  on `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` (no preinstalled torch) with
  pinned `torch==2.11.0+cu128` + `torchaudio==2.11.0+cu128` + Perth, fixing the
  NGC pytorch ABI conflict with Perth observed at runtime.
- numpy fast path in `float_tensor_to_pcm16` (stdlib fallback kept for offline CI);
  parity test in `tests/test_chatterbox_pcm16.py`.
- `resolve_voice()` `.wav` check is now case-insensitive (`.WAV` / `.Wav` work).
- Dead SSML code removed from `tts_client.py` (`_insert_ssml_breaks`); stale
  Magpie references updated to Chatterbox across app, client, and tests; speed-ignored
  warning emitted when a non-default speed is passed to `synthesize()`.

### Changed

- **Replaced Magpie TTS with Chatterbox** across the realtime stack:
  `protocol.py` (`TTS_SAMPLE_RATE` 22050→24000, `resolve_voice` rewritten for
  Chatterbox — `.wav` path for cloning, `""` for default), `tts_client.py`
  (plain JSON POST, no SSML/prosody wrapping), `_settings.py` (`tts_url` default
  → `http://chatterbox:9000`, `default_voice` default → `""`),
  `docker-compose.audio.yml` (new `chatterbox:` service replaces `tts:`),
  `env.audio.example` (Magpie/NGC vars removed, `CHATTERBOX_PORT` added).

## [0.24.0] - 2026-06-20

### Added

- **`model overview --live` — a live fleet dashboard.** `overview` was a static
  description; `--live` now probes the running deployment and reports the five
  "what is it doing right now" views: **online** (per-backend health), **offered**
  (served + candidate models, task families, the endpoint list), **busy**
  (in-flight / queued requests), **usage** (cumulative prompt/generation tokens and
  finished requests by reason), and **endpoints**. It is read-only and HTTP-only —
  it works against a local deployment or a `model tunnel` hostname alike, and
  degrades gracefully when a backend or its metrics is unreachable.
- **Gateway `GET /status`** — a model-gear-native JSON aggregate. The fleet's
  backends are internal-only, so the gateway fans out to each one's `/health` +
  `/metrics` and returns `{object: "model-gear.fleet_status", default_model,
  busy: {running, waiting}, backends: [...], endpoints: [...]}`. This is the source
  `model overview --live` reads in the fleet (a bare single-model server is read
  directly from its `/metrics` + `/health`).
- **`model_gear._metrics`** — a small stdlib-only helper that parses vLLM's
  Prometheus `/metrics` (running/waiting, prompt/generation tokens,
  `request_success_total` by finish reason, KV-cache usage) and best-effort HTTP
  probes that never raise.

### Changed

### Fixed

## [0.23.0] - 2026-06-20

### Added

- **Durable vLLM logs that survive restart/recreate (#50).** When a vLLM
  container restarted, its `docker logs` — and any EngineCore crash trace — were
  lost, which blocked root-causing #50 for lack of data. `model init` now
  scaffolds `mg-logwrap.sh`, bind-mounted as each vLLM service's entrypoint: it
  tees stdout+stderr to a per-boot file `<service>-<boot>.log` under a
  host-mounted log dir (`${MODEL_GEAR_LOG_DIR:-<deploy>/logs}` → `/logs/model-gear`),
  then `exec`s the real command so vLLM stays the signal target (graceful
  shutdown) and the exit code (and `restart:` policy) are unchanged. Teeing at the
  process-I/O level captures **both** Python tracebacks and native CUDA/C++ aborts;
  if logging can't be set up it falls back to a plain `exec` and never blocks
  serving. The crash boot is preserved as its own file. Wired into the single-model
  and fleet (`primary`/`embed`/`rerank`) compose templates. See
  `docs/durable-logs.md`.
- **`model logs`** — new read-only verb to list/tail the durable logs, reading the
  host files directly so it works even after the crashed container is gone:
  `model logs` (list boots), `model logs <service>` (tail latest), and
  `model logs <service> --previous` (tail the boot that crashed, after a restart).

### Changed

- `model init` / `model serve` / `model fleet up` pre-create the host log dir
  (user-owned) before compose bind-mounts it, so logs are never root-owned.

### Fixed

## [0.22.1] - 2026-06-19

### Fixed

- **`model fleet status` now reports the embedding + reranker gears.**
  `FLEET_CONTAINERS` listed only `vllm-primary` + `gateway`, so `model fleet
  status` silently omitted the `vllm-embed` / `vllm-rerank` containers the default
  fleet (#44/#47) actually runs. Added `FLEET_EMBED` / `FLEET_RERANK` to the
  default container set — status now lists all four (the opt-in generate fallback
  stays excluded, as it is not in the default compose).

### Changed

- **Aligned the agent/human-facing prose with the co-resident gears (#44/#47).**
  `model learn`, `model overview`, `model explain` (root + fleet), `model init
  --fleet` help, the `fleet` docstring, the scaffolded `env.example` /
  `docker-compose.yml` comments, `README.md`, `CLAUDE.md`, and
  `docs/gateway-fleet.md` still described the fleet as a "2-model" /
  "two-container" / "single-backend" deployment. They now describe the default
  fleet as the generate primary plus co-resident embedding + reranker gears behind
  one gateway, routed by task family (generate / embed / score / rerank), with the
  *generate* fallback as the only opt-in backend. Added a "Task families & gears"
  section + `explain embeddings` / `explain rerank` pointers to `model learn`.

## [0.22.0] - 2026-06-19

### Added

- **Embedding + reranker gears (closes #44).** model-gear now serves two pooling
  gears alongside the chat primary, reachable through the same OpenAI-compatible
  gateway and routed by the request's `model` field:
  - `Qwen/Qwen3-Embedding-0.6B` — `POST /v1/embeddings` (vLLM
    `--runner pooling --convert embed`), native **1024-dim**, MRL-truncatable via
    the `dimensions` param (Matryoshka `--hf-overrides`).
  - `Qwen/Qwen3-Reranker-0.6B` — `POST /v1/rerank` + `/v1/score` (vLLM
    `--runner pooling --convert classify`, served via the
    `Qwen3ForSequenceClassification` `--hf-overrides`).
  - **Catalog:** `SupportedModel` gains `task` (`generate`/`embed`/`score`),
    `dimension`, and `hf_overrides`; both gears surface in `model overview --list`
    and `GET /v1/models/supported`.
  - **Fleet:** `vllm-embed` + `vllm-rerank` services in the fleet compose
    (always-warm, small `--max-model-len`/`--gpu-memory-utilization` so they
    co-reside with the 27B on a single GB10), wired as gateway backends.
  - **Gateway:** task-aware failover — an embed/score request never fails over to
    a generate backend (and vice versa); chat primary↔fallback failover preserved.
  - **CLI:** `model switch --task {generate,embed,score}` for solo serving;
    `model explain embeddings` / `rerank` / `score` document the call shapes;
    per-model docs under `docs/`.
  - **Boundary:** model-gear *serves* the gears only — no vector store, index,
    chunker, or retrieval lands here (guarded by a test); storage + retrieval are
    the consumer's half (eidetic-cli).

### Changed

### Fixed

## [0.21.1] - 2026-06-19

### Fixed

- **markdownlint:** exempt skill prompt templates (`.claude/skills/**/prompts/**`)
  from markdownlint. These are model-facing prompts fed verbatim to a backend
  (first line is `$ARGUMENTS` or a prose instruction), so MD041 (first-line H1)
  and MD032 are inapplicable — a heading would be injected into the prompt.
  `SKILL.md` is still linted; only `prompts/` is exempt. Unblocks the `lint` CI
  job after the `ask-colleague` skill was vendored in.

## [0.21.0] - 2026-06-12

### Changed

- **The fleet is now single-backend by default (Qwen primary only); the Mistral
  fallback is removed.** Live validation showed two ~30B NVFP4 models don't co-fit
  a shared GB10, so the warm dense Mistral-Small-3.2-24B fallback has been dropped
  from the default fleet and the primary restored to its **load-tested solo
  headroom**: `PRIMARY_GPU_MEM_UTIL` `0.40 → 0.6` and `PRIMARY_MAX_MODEL_LEN`
  `32768 → 262144` (full 256K). The `vllm-fallback` service is gone from
  `fleet/docker-compose.yml`, and `FLEET_CONTAINERS` no longer includes it.
- **The gateway makes the fallback optional.** `build_config` now adds a second
  backend **only** when `FALLBACK_URL` or `FALLBACK_SERVED_NAME` is set in env —
  so the default gateway serves the primary alone (no failover target), and a
  two-backend fleet still works for anyone who wires one up. Routing/failover
  primitives are unchanged; `order_backends` returns just the primary when solo.
- **Mistral stays a selectable catalog candidate** (`model overview --list`) and
  the documented opt-in fallback — only its role as the *default* fleet fallback
  is removed. README, `docs/gateway-fleet.md`, and the `model explain
  fleet/gateway` / `model init --help` text are updated to the single-backend
  default (with an "Adding a fallback" guide).

### Fixed

- `docs/gateway-fleet.md` uses `$HOME/.model-gear` instead of the non-portable
  `~/.model-gear`.

## [0.20.1] - 2026-06-12

### Fixed

Qodo review of #41:

- **`model init --fleet --audio` now scaffolds `_readiness.py`** — added
  `fleet/_readiness.py → _readiness.py` to `_compose.AUDIO_TEMPLATES`. The
  Parakeet `Dockerfile.parakeet` `COPY _readiness.py` requires it at the
  deployment-dir root, so a clean audio init previously produced a tree where
  `docker compose build stt` would fail. Covered by `test_init.py`.
- **Parakeet readiness drift guard + simplification** — removed the third
  (inline) copy of the readiness decision from `listen_server.py` (the scaffold
  now guarantees the vendored `_readiness.py` is present), and added a test
  asserting the vendored twin stays behaviourally identical to the canonical
  `model_gear/realtime/_readiness.py`.
- **CUDA readiness probe failures are now logged** — `listen_server.health()`
  emits a `logger.warning` with the exception type/message before returning
  `503`, so operators can distinguish driver-down / OOM / stale-context.
- **`scripts/audio-smoke.py` now exercises `/v1/audio/speech`** (it previously
  claimed both routes but only tested transcriptions) and wires the formerly
  unused `--stt-url` to a direct-Parakeet transcription check.
- **`docs/realtime-pipeline.md`** uses `$HOME/.model-gear` instead of the
  non-portable `~/.model-gear`.

## [0.20.0] - 2026-06-12

### Added

- **`docs/realtime-pipeline.md`** — the previously-missing runbook for the audio
  surface: that model-gear owns the live `:8080` realtime facade, the
  `model init --fleet --audio` / `model fleet up` bring-up, the topology
  (gateway path-routes `/v1/audio/*` → realtime → Parakeet/Magpie), the drift it
  fixed (#39/#40), the cheap readiness probe, and the stale-Parakeet-CUDA restart
  runbook. Resolves a doc referenced from `pyproject.toml`, the audio overlay,
  and the realtime app docstring but never written.
- **`scripts/audio-smoke.py`** — a stdlib-only live smoke test for the audio
  routes: asserts `GET :8080/openapi.json` lists both `/v1/audio/transcriptions`
  and `/v1/audio/speech`, then POSTs an in-memory 16 kHz WAV and asserts
  `200 {text: …}`. Reproduces issue #39's repro to confirm the 500→200 fix.
  Requires a running GPU box (not a CI unit test).
- **`model_gear/realtime/_readiness.py`** — a stdlib-only `evaluate_readiness()`
  helper backing the Parakeet `/v1/health/ready` cheap probe; unit-tested in CI
  without torch/nemo/GPU.

### Fixed

- **Parakeet STT healthcheck now reflects real model readiness (#39).** The
  vendored `templates/fleet/listen_server.py` `/v1/health/ready` returned
  `{"status": "ready"}` unconditionally — process liveness only — so a container
  whose CUDA context had gone stale (`CUDA error: unknown error`, every
  transcription 500ing) still reported Docker "healthy". The probe now reports
  ready **only** when the NeMo model is loaded **and** a trivial CUDA tensor op
  succeeds, returning `503` otherwise (a cheap probe, not a full transcription
  each interval). The pure decision is vendored into the Parakeet build context
  and `COPY`'d into the image so it resolves without the wheel.

## [0.19.0] - 2026-06-09

### Added

- **`scripts/gen-api-key.py`** — generate or rotate the bearer key
  (`CULTURE_VLLM_API_KEY`) that gates the served API. The secret is created with
  the stdlib `secrets` module and **never hardcoded**, so the script is safe in the
  open-source repo; the key only ever lands in the gitignored deployment `.env`
  (written `0o600`, best-effort). Hidden by default (no echo into logs/scrollback);
  `--show` prints it, `--force` rotates an existing key, and `--bytes` (min 16) is
  validated. Resolves the deployment dir like the `model` CLI (`--dir` →
  `$MODEL_GEAR_DIR` → `$HOME/.model-gear`), degrades gracefully on an unreadable or
  non-regular `.env`, and runs from a wheel install (no `model_gear` import).
  Referenced from the README "Expose the API" section.

## [0.18.0] - 2026-06-09

### Added

- **`model tunnel` — expose the local OpenAI-compatible API from anywhere via a
  Cloudflare Tunnel** (#35). Dry-run by default (prints the `cloudflared` command
  and the public `https://<host>/v1` URL); `--apply` starts a standalone
  `cloudflared tunnel run` in the background (logging to `cloudflared.log` in the
  deployment dir), and `--stop --apply` tears it down. The public hostname resolves
  `--hostname` → `$CULTURE_VLLM_PUBLIC_HOSTNAME` → `CULTURE_VLLM_PUBLIC_HOSTNAME` in
  a **gitignored** `.cf-tunnel.env`; the run-token comes from
  `CULTURE_CF_TUNNEL_TOKEN_SHUSHU` (a shushu-sealed secret name, preferred) or
  `CULTURE_CF_TUNNEL_TOKEN` (plaintext fallback). The token is **never placed on the
  process argv** (so it can't leak via `ps` or the log) — cloudflared reads it from
  the `TUNNEL_TOKEN` environment variable, which `shushu` injects (sealed mode) or
  the launcher sets directly (fallback). The resolved hostname and sealed-secret
  name are validated against a conservative charset before they reach the argv (an
  argument-injection guard). `--apply` preflights that `cloudflared` (and `shushu`)
  is on PATH, that no tunnel is already running for the deployment, and that the
  local server answers `/health`; `--stop` signals the recorded process *group* and
  confirms exit (SIGTERM → SIGKILL) before clearing a PID-reuse-safe pidfile (the
  recorded pid is identity-checked against `/proc` so a reused pid can't be killed).
  No hostname, token, or backend checkpoint id is committed. The Cloudflare side
  (tunnel + ingress + DNS) is provisioned once by `cultureflare remote-login
  --no-access`.
- **Optional bearer auth on the served API** via `CULTURE_VLLM_API_KEY`, wired into
  the single-model `docker-compose.yml` as `VLLM_API_KEY=${CULTURE_VLLM_API_KEY:-}`.
  Empty (default) leaves local dev open; set it and vLLM requires `Authorization:
  Bearer` — the gate for any public exposure. Documented in `env.example` alongside
  a note that `VLLM_SERVED_NAME` can be a generic alias to keep the checkpoint name
  out of the public `/v1/models`.
- **`cf-tunnel.env.example`** scaffolded by `model init` (single + fleet), a
  placeholder-only template the owner copies to the gitignored `.cf-tunnel.env`.
- README "Expose the API from anywhere (Cloudflare Tunnel)" section and a
  `model explain tunnel` catalog entry.

## [0.17.0] - 2026-06-03

### Changed

- **Served context raised 128K → full 256K (native) for the MTP primary on DGX
  Spark.** The `spark` machine profile's `max_model_len` default is now `262144`
  (was `131072`), with matching changes to the single-model `env.example` /
  `docker-compose.yml` defaults and the `model switch --help` / `model explain`
  text. Load-tested 2026-06-03 on the shared GB10 (util 0.6, `--max-num-seqs 2`,
  KV-FP8, MTP n=3): boots clean (CUDA-graph capture, PIECEWISE, **0.71 GiB** in 2 s
  — **no OOM**), **17.8 tok/s** decode, **74.0 %** MTP draft acceptance, both
  `model assess` probes `finish=stop`, tool-calling probe passes, and **71,601 MiB
  (~70 GiB)** resident — the *same* footprint as 32K/128K, because
  `--gpu-memory-utilization` fixes the KV-pool reservation (only the addressable
  context grows). vLLM reports **5.29× max concurrency at a full 256K request**,
  well above the `--max-num-seqs 2` decode cap, so **there is no practical
  concurrency cost** versus the 128K default. `model switch --max-model-len <N>`
  still overrides per deployment, and util stays a conservative `0.6` (shared box).
  See `docs/qwen3.6-27b-text-nvfp4-mtp.md` (new 256K benchmark) and
  `docs/tuning-profiles.md`.
- **Catalog `context` string updated.** The MTP primary now reads
  `"256K native (served at full 256K on the shared GB10)"`.
- **Scope — deliberately left at the old contexts:** fleet templates stay at 32K
  (co-residence with the 24B fallback is a different, still-unvalidated memory
  regime; `fleet/env.example` notes this), and the `thor` / `generic` machine
  profiles stay at 32K (unmeasured estimates) with `blackwell` at 64K. The
  `model switch` native-ceiling clamp (added in 0.16.0) still pins 32K-native
  candidates (`nvidia/Qwen3-32B-NVFP4`, `mmangkad/Qwen3.6-35B-A3B-NVFP4`) down to
  their own ceilings under the new 256K spark default.

### Fixed

- **`model switch` warns when an uncatalogued model would inherit an unclamped
  machine context default.** The native-ceiling clamp only protects catalogued
  models; an uncatalogued model ID (which `switch` supports) inherits the machine
  default (now spark's 262144) and would boot-fail if the checkpoint's native
  context is smaller. `switch` now emits a clear warning pointing at
  `--max-model-len` / cataloguing, rather than silently applying the high default
  (no silent clamp — an uncatalogued ceiling is unknown, so guessing one is wrong
  both ways). Addresses a Qodo reliability finding on #34.

## [0.16.0] - 2026-06-03

### Changed

- **Served context raised 32K → 128K for the MTP primary on DGX Spark.** The
  `spark` machine profile's `max_model_len` default is now `131072` (was `32768`),
  with matching changes in the single-model `env.example` /`docker-compose.yml`
  defaults. Load-tested 2026-06-03 on the shared GB10 (util 0.6, `--max-num-seqs 2`,
  KV-FP8, MTP n=3): boots clean (no CUDA-graph-capture OOM), **18.3 tok/s** decode,
  **73.3 %** MTP draft acceptance, both `model assess` probes `finish=stop`, and
  **71,963 MiB (~70 GiB)** resident — the *same* footprint as 32K, because
  `--gpu-memory-utilization` fixes the KV-pool reservation (the pool holds **9.6×**
  a full 128K request). `model switch --max-model-len <N>` still overrides per
  deployment, and util stays a conservative `0.6` (the box is shared). See
  `docs/qwen3.6-27b-text-nvfp4-mtp.md` (new 128K benchmark) and
  `docs/tuning-profiles.md`.
- **Catalog `context` strings clarified.** The MTP primary now reads
  `"256K native (served at 128K on the shared GB10)"`; the non-served candidate /
  fallback entries (`mmangkad/Qwen3.6-27B-NVFP4`, the Mistral fallback) drop the
  stale per-model "capped to 32K" note and state native context only.
- **Scope — deliberately left at the old contexts:** fleet templates stay at 32K
  (the fleet runs the primary co-resident with a 24B fallback at lower util — a
  different memory regime the single-model 128K test does not validate;
  `fleet/env.example` notes this), and the `thor` / `generic` machine profiles
  stay at 32K (unmeasured estimates) with `blackwell` at 64K.

### Fixed

- **`model switch` clamps the machine context default to a model's native ceiling.**
  Raising spark's `max_model_len` default to `131072` made it apply to *every* model
  switched to on spark — including the 32K-native catalog candidates
  (`nvidia/Qwen3-32B-NVFP4`, `mmangkad/Qwen3.6-35B-A3B-NVFP4`), where vLLM refuses a
  `--max-model-len` above the checkpoint's native limit (no YaRN) and the container
  fails to boot. `SupportedModel` now carries a numeric `native_max_model_len`, and
  `model switch` clamps the resolved context *down* to it when no explicit
  `--max-model-len` is given (an explicit value still wins, for opted-in YaRN
  configs). Fixes a Qodo correctness finding on #33.

## [0.15.0] - 2026-05-31

### Changed

- **Fleet default primary → `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`** (the MTP
  build), replacing `mmangkad/Qwen3.6-27B-NVFP4` (issue #26 follow-up). The
  tool-calling gate that kept it a candidate is now **closed**: served through the
  production compose it emits a valid `qwen3_coder` tool call, completes a full
  tool round-trip, keeps its reasoning trace, and runs MTP spec-decode at **78.6%
  draft acceptance with tool calling on** — ~2.4× single-stream decode (8 → ~19
  tok/s), ~71 GB footprint, both `model assess` probes `finish=stop`. Promoted
  across the catalog (`role_hint`), the gateway default (`_DEFAULT_PRIMARY`),
  `whoami`, both template `env.example`/`docker-compose.yml` files, and
  `culture.yaml`.
- **The MTP serve flags are now baked into the compose templates** (single-model +
  fleet `vllm-primary`): `--speculative-config`, `--trust-remote-code`,
  `--language-model-only`, the `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4` override,
  and `--max-num-seqs=2`. A fresh `model init && model serve` of the default now
  works out of the box. Quantization default is `modelopt`.
- **`model switch` notices inverted.** Because the template ships the MTP primary's
  flags, switching to a **non-MTP** model now prints "REMOVE these 4 `command:`
  lines" (was "add" for the MTP candidate); the MoE `--moe-backend` add-notice is
  unchanged. Switching to the MTP primary force-caps `--max-num-seqs` to 2.
- **`mmangkad/Qwen3.6-27B-NVFP4` archived to a candidate** — retained as the MTP
  primary's tokenizer source and the only vision-capable 27B in the catalog.

### Fixed

- **`model switch --apply` no longer takes a healthy deployment down when a manual
  compose edit is required** (Qodo review). Switching to a non-MTP model (the
  template ships the MTP primary's incompatible flags) now writes `.env` and
  **stops before the restart**, printing the lines to remove; `--force` overrides
  to recreate the container anyway.
- **MTP compose flags are a single source of truth** (`catalog.mtp_compose_command_items()`) —
  consumed by both `model switch`'s removal notice and guarded against drift from
  the packaged templates by a new test (Qodo review).
- **Security guidance for the now-default `--trust-remote-code`** added to both
  compose templates and `env.example`: HF_TOKEN is only needed for gated repos
  (defaults are public) — leave it empty or use a minimal-scope read-only token, and
  pin trusted revisions (Qodo review). Tracking the upstream tokenizer fix that would
  let us drop the override in #29.

## [0.14.0] - 2026-05-31

### Added

- **MTP (Multi-Token Prediction) candidate for the 27B** (issue #26). New catalog
  entry `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` — a text-only re-export of the
  27B primary with its **MTP draft head restored in bf16** so vLLM speculative
  decoding actually works. The lesson from the 35B MoE applied: the baseline NVFP4
  export drops the MTP head (~0 % draft acceptance), and a newer vLLM isn't
  installable on the aarch64 GB10 — so the fix is *a checkpoint that ships the MTP
  weights*, not a newer engine. Carries a catalog `speculative_config`
  (`{"method":"qwen3_5_mtp","num_speculative_tokens":3}`); quantization is
  `modelopt`. **Load-tested on the DGX Spark (GB10) 2026-05-31: 19.1 tok/s decode
  (~2.4× the baseline 27B's ~8 tok/s) at 72 % MTP draft acceptance on vLLM
  0.19.0+nv26.04** — the open risk (does the stock image accept `qwen3_5_mtp`?) is
  cleared (it resolves the `Qwen3_5MTP` draft head). One tokenizer override is
  required: the checkpoint declares the newer `TokenizersBackend` class (absent from
  nv26.04), so serve with `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4` (the cached
  sibling, same vocab); `model switch` prints it.
  - New per-model doc `docs/qwen3.6-27b-text-nvfp4-mtp.md` with the serve recipe,
    the live benchmark table (decode tok/s + acceptance vs the baseline), and the
    caveats (`--max-num-seqs 2` or it silently OOMs; the tokenizer override).

### Changed

- **`model switch` surfaces MTP serve-extras, not just MoE.** `_moe_notice` →
  `_serve_notices` (now a list): a model with a catalog `speculative_config` prints
  the exact `--speculative-config` / `--trust-remote-code` / `--language-model-only`
  compose edits (+ the `VLLM_MAX_NUM_SEQS=2` reminder), the same hand-edit pattern
  as `--moe-backend`. The `--json` dry-run replaces the `moe_notice` key with a
  `compose_edits` list. `env.example` + the `explain` catalog prose updated to match.

### Fixed

## [0.13.0] - 2026-05-31

### Added

- **Workload `purpose` + machine tuning profiles.** `model switch` now resolves
  the serve config from three layers — a **machine** profile (`--machine`,
  default auto-detected from `nvidia-smi` + hostname: GPU-memory fraction,
  context, attention backend), a **workload** profile (`--purpose`, default
  `balanced`: the batching knobs and the shape `model benchmark` exercises), and
  the model's catalog entry — with explicit `--max-model-len` / `--gpu-mem-util`
  flags overriding the machine defaults.
  - **New `model_gear/profiles.py`** (pure data module, like `catalog.py`):
    `WorkloadProfile` (`balanced` ≈1K/1K, `prompt-heavy` ≈8K/1K, `decode-heavy`
    ≈1K/8K) and `MachineProfile` (`spark` load-tested, `thor`/`blackwell`/`generic`
    configured), guarded by `tests/test_profiles.py`.
  - **Richer single-model template** — the serve command now passes
    `--attention-backend`, `--max-num-seqs`, `--max-num-batched-tokens` (env-driven),
    plus static `--enable-chunked-prefill` / `--async-scheduling`. New `.env` keys:
    `VLLM_PURPOSE`, `VLLM_MACHINE`, `VLLM_ATTENTION_BACKEND`, `VLLM_MAX_NUM_SEQS`,
    `VLLM_MAX_NUM_BATCHED_TOKENS`.
  - **Per-model MoE serve extras** — the catalog gains `moe_backend` /
    `speculative_config` (set only on the `Qwen3.6-35B-A3B` MoE candidate).
    `model switch` to the MoE prints them as a documented compose edit (they break
    the dense/hybrid models and can't be defaulted in the shared template).
  - **`model benchmark` is tied to the config** — its workload shape defaults to
    the configured `VLLM_PURPOSE` (overridable with `--purpose` / `--input-len` /
    `--output-len`).
  - `model whoami` / `model overview` surface the active `gear` (purpose/machine);
    `model explain tuning` documents the layering; `docs/tuning-profiles.md` is new.
  - Credit: the serve tuning and the three workload shapes follow **shahizat**'s
    cross-machine NVFP4 benchmark (NVIDIA Developer Forums) — see the README
    Acknowledgements and `docs/tuning-profiles.md`.
  - **Live-replicated on the shared DGX Spark (2026-05-31)** rather than trusting
    the post: with the new flags the **35B MoE candidate loads solo** (util 0.70,
    marlin) and runs **single-stream decode ~35 tok/s vs the 27B's ~7.8 — ~4.6×
    faster** (the MoE's ~3B-active advantage). Numbers + method in
    `docs/tuning-profiles.md` and `docs/qwen3.6-35b-a3b-nvfp4.md`.

### Changed

- `model switch` `--max-model-len` / `--gpu-mem-util` now default to the machine
  profile (was a fixed 32768 / 0.6); pass them explicitly to override.
- `model benchmark` replaces `--decode-tokens` with purpose-driven
  `--input-len` / `--output-len`.
- **Catalog: dropped the MTP `speculative_config` from the `mmangkad/Qwen3.6-35B-A3B-NVFP4`
  entry** (kept `--moe-backend=marlin`). Live testing showed shahizat's MTP draft
  fails to load on the `mmangkad/` copy (`qwen3_5_mtp.py` weight-shape mismatch on
  vLLM nv26.04) — it is tied to his `nvidia/` checkpoint. `model switch` no longer
  prints a recipe that wouldn't load.

### Fixed

## [0.12.0] - 2026-05-30

### Added

- **Audio I/O behind the gateway (STT + TTS) — issue #18, part 1 of 3.** model-gear
  now serves OpenAI-compatible `POST /v1/audio/transcriptions` and
  `POST /v1/audio/speech` on the same host port as the text API, fronted by the same
  stdlib gateway. The audio backends are the *same models* the standalone realtime-api
  stack ran — **NVIDIA Parakeet STT** + **Magpie TTS NIM** — consolidated into the
  fleet (no separate compose project; the realtime bridge's LLM is the fleet gateway
  itself, so there is no extra vLLM container).
  - **New `[realtime]` extra + `model_gear.realtime` package** (vendored from the
    `realtime-api` sibling, cite-don't-import): a FastAPI bridge that exposes the
    OpenAI audio surface (`/v1/audio/speech` adapts Magpie's proprietary
    `/v1/audio/synthesize`; `/v1/audio/transcriptions` forwards to Parakeet). The base
    wheel and the gateway stay stdlib-only — torch/fastapi never leak into them.
  - **Gateway audio routing** — `/v1/audio/*` is path-routed to the audio backend
    (`AUDIO_URL`) with no model rewrite and no failover; binary responses relayed
    **streamed** (chunked) so a large TTS body never buffers whole in the gateway.
    Unset `AUDIO_URL` (a text-only fleet) → those paths 404, unchanged.
  - **`model init --fleet --audio`** scaffolds the audio overlay
    (`docker-compose.audio.yml` + `Dockerfile.realtime` + a vendored
    `Dockerfile.parakeet`/`listen_server.py`) and appends the audio keys to `.env`.
    `model fleet up`/`down`/`status` auto-include the overlay when present.
  - **Co-residence caveat:** the audio services share the GPU with the LLM fleet — the
    overlay is opt-in so text-only boxes keep their GPU budget. See the per-model docs
    (PR3) for live numbers.
  - **The realtime WebSocket (`/v1/realtime`) and the `model overview`/`doctor`/`explain`
    surface land in the follow-up PRs (parts 2 and 3).**

### Changed

### Fixed

- **Audio review hardening (PR #24 review).**
  - **Gateway no longer buffers whole audio bodies** — `/v1/audio/*` responses are
    relayed chunked instead of `read_all()`'d into memory, so one large TTS WAV can't
    OOM the fleet's single front door.
  - **`TTS_CONCURRENCY` / `TTS_SPEED` clamped to ≥ 1** — `TTS_CONCURRENCY=0` previously
    seeded an `asyncio.Semaphore(0)` that hung every TTS request; a 0/negative speed
    emitted nonsensical `rate="0%"` SSML.
  - **`/v1/audio/speech` `speed` clamped to OpenAI's 0.25–4.0 range** before the Magpie
    percentage conversion, so out-of-range values no longer reach the backend as
    `rate="{huge|negative}%"` and 502.
  - **SonarCloud config** — coverage exclusions now mirror `coverage.run` `omit` (the
    `[realtime]`-extra modules can't be unit-imported offline), and the deployment
    *scaffolds* under `model_gear/templates/**` are excluded from analysis (container
    Dockerfiles + the vendored Parakeet server aren't package runtime). Added unit
    tests for `realtime.protocol`, the settings clamps, the speed clamp, and the
    streamed audio relay.

## [0.11.1] - 2026-05-30

### Added

- **`model learn --json` now includes a `models` object** (`supported_catalog` /
  `loaded_now`) — a machine-readable version of the catalog-vs-loaded explainer for
  agent consumers. (Additive field; the only observable behavior change in this
  release.)

### Changed

- **Documented "supported catalog vs. loaded now" consistently** across the README,
  `docs/gateway-fleet.md` (new "Supported catalog vs. warm backends" subsection),
  the per-model docs, and the CLI teaching surfaces (`model learn`,
  `model explain models`/`overview`/`status`/`whoami`/root, and the
  `overview`/`status`/`whoami`/`fleet status` help strings). The distinction:
  `model overview --list` / `GET /v1/models/supported` = the gears you *can* switch
  to (tagged `load-tested`/`configured`, static); the live `GET /v1/models` (which
  `model fleet status` queries) = what's actually *loaded* now. `model status` /
  `model whoami` report the *configured* served model (from `.env`) + health — not
  a live `/v1/models` query. Docs + help text (no serving/runtime behavior change).

## [0.11.0] - 2026-05-30

### Added

- **`RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4` support** — added to the
  supported-model catalog (`model overview --list`, `GET /v1/models/supported`)
  with a per-model doc, [`docs/mistral-small-3.2-24b-nvfp4.md`](docs/mistral-small-3.2-24b-nvfp4.md).
  Load-tested on the DGX Spark (GB10): ~15 GiB weights, **~14.9 tok/s** decode,
  prefill 2,009 tok in 1.49 s, tool calling ✅.
- **`mistral` tool-call parser inference** — `model_gear.runtime._parser` now maps
  Mistral-family ids (incl. the `mistralai/` org) to the `mistral` parser; `model
  switch` auto-selects it.
- **`model switch --quantization`** — the served `--quantization` is now set per
  model (read from the catalog for a known model, e.g. `compressed-tensors` for the
  RedHatAI NVFP4 Mistral vs `modelopt_fp4` for the nvidia/mmangkad checkpoints);
  `--quantization` overrides it. The single-model compose reads `VLLM_QUANTIZATION`.

### Changed

- **Fleet default fallback is now the dense Mistral-Small-3.2-24B**, replacing the
  `mmangkad/Qwen3.6-35B-A3B-NVFP4` MoE, which never loaded on the GB10 (OOM
  co-resident, stall solo — no benchmark obtained). Mistral is dense, loads
  reliably, and is smaller (~15 GiB weights). The fleet compose serves it with the
  **mistral tokenizer + images limited to 0** (required for tool-call parsing on
  the nv26.04 build; the HF tokenizer leaks `[TOOL_CALLS]` markup, and the mistral
  tokenizer alone crashes the Pixtral profiler) and **no** `--reasoning-parser`
  (instruct model). The 35B MoE is demoted to a catalogue candidate.
- `model_gear.gateway._config._DEFAULT_FALLBACK`, the fleet `docker-compose.yml` /
  `env.example` `FALLBACK_*` defaults, `docs/gateway-fleet.md`, and `README.md`
  updated for the new fallback.

### Fixed

## [0.10.1] - 2026-05-30

### Changed

- **Fleet default GPU-mem utilisations rebalanced `0.55`/`0.30` → `0.40`/`0.35`.**
  Live validation on a DGX Spark (GB10) showed `0.55`/`0.30` OOM-crash-loops the
  fallback: the 27B primary alone takes ~75 GiB at util 0.6, and `--gpu-memory-utilization`
  is fraction-of-total *per process* (the two backends don't coordinate). The new
  values are a dedicated-box estimate; the templates and docs now state plainly
  that co-residence of two ~30B models needs a dedicated box.

### Fixed

- **Docs corrected against live findings (2026-05-30):** `docs/gateway-fleet.md`
  gains a "Live validation findings" section (27B warm-up ~7 min, ~75 GiB footprint,
  8.0 tok/s decode; co-residence not viable on a shared GB10).
  `docs/qwen3.6-35b-a3b-nvfp4.md` updated from "not yet load-tested" to the actual
  result — the MoE fallback does **not** load reliably on this box (OOM co-resident;
  crash/stall even solo). `docs/qwen3.6-27b-nvfp4.md` reframed as the fleet default
  primary (was "candidate") with the warm-up measurement and a corrected
  recommendation.

## [0.10.0] - 2026-05-30

### Added

- **`GET /v1/models/supported` gateway endpoint — the "change gears" catalog.**
  Alongside the OpenAI-standard `/v1/models` (which lists only the two *loaded*
  backends), the gateway now serves the full catalog of supported models a client
  can change gears to, each flagged `loaded` (a backend serves it now) and
  `default` (the gateway routes unknown/missing names there). Non-OpenAI shape
  (`"object": "model-gear.supported_models"`) so `/v1/models` stays standard for
  existing clients. Pure `supported_models_payload()` in `gateway/_routing.py`.
- **New packaged catalog `model_gear/catalog.py`** — a dependency-free
  `SUPPORTED_MODELS` tuple (the 27B primary, the 32B dense candidate, the 35B-A3B
  MoE fallback) that is the single source of truth for both the gateway (which
  runs from a wheel and can't read `docs/`) and the CLI. `model overview --list`
  is now catalog-backed, so it is populated even in a wheel install.

### Changed

- **Fleet (and single-model) default primary → `mmangkad/Qwen3.6-27B-NVFP4`.**
  The scaffolded default served model is now the Qwen3.6 27B (hybrid
  Mamba/linear-attn + ViT, 256K native context) with `--tool-call-parser=qwen3_coder`
  — matching what runs on the DGX Spark and convertible's parent model. The dense
  `nvidia/Qwen3-32B-NVFP4` remains a supported candidate (`PRIMARY_MODEL` /
  `model switch`). Recomputed co-resident GPU memory: `PRIMARY_GPU_MEM_UTIL=0.55`
  and `FALLBACK_GPU_MEM_UTIL=0.30` (the 27B is heavier than the 32B). Updated the
  fleet + single-model templates, `gateway/_config.py`, `whoami` default,
  `culture.yaml` / `AGENTS.md` / `CLAUDE.md` (served-model coherence chain), and
  the per-model + gateway-fleet docs.

### Fixed

## [0.9.0] - 2026-05-28

### Added

- **Fallback model + single front OpenAI gateway ("fleet").** A new
  scaffold-based deployment runs **two always-warm vLLM backends behind one
  stdlib gateway** that model-gear manages as three containers
  (`model-gear-gateway`, `model-gear-vllm-primary`, `model-gear-vllm-fallback`).
  The gateway routes each request by its `model` field, defaults an
  unknown/missing name to the primary, and fails over to the other backend when
  the chosen one refuses the connection or returns a 5xx **before** the response
  body (4xx is returned verbatim; no mid-stream retry). SSE streams are relayed
  chunk-by-chunk. Default fallback: the MoE `mmangkad/Qwen3.6-35B-A3B-NVFP4`.
- **New gateway package `model_gear/gateway/`** — a pure-stdlib
  (`http.server` + `http.client`, no runtime deps) reverse proxy: `_routing.py`
  (pure name/alias/default routing + failover ordering), `_config.py` (env →
  routing table + server config), `server.py` (the `handle_post` failover seam,
  upstream client, and `ThreadingHTTPServer` handler), run as
  `python -m model_gear.gateway`.
- **`model init --fleet`** scaffolds the fleet templates
  (`docker-compose.yml` + `.env` + `Dockerfile.gateway`) and pins
  `MODEL_GEAR_VERSION` to the running release; **`model fleet up | down |
  status`** drives the deployment (`up`/`down` dry-run by default, `--apply` to
  commit; `status` is read-only and reports all three containers + the gateway
  `/health` + `/v1/models`).
- **Docs:** `docs/gateway-fleet.md` (topology, routing/failover, memory,
  verbs), `docs/qwen3.6-35b-a3b-nvfp4.md` (the MoE fallback), a README "fleet"
  section, and `model explain fleet` / `model explain gateway` entries.

### Changed

- `model_gear/runtime/_compose.py` gained a template registry
  (`SINGLE_TEMPLATES` / `FLEET_TEMPLATES`), a `templates=` argument on
  `scaffold_plan` / `write_scaffold` (single-model stays the default — existing
  callers unchanged), a `compose_up_build` helper, and `FLEET_CONTAINERS`.
- The fleet `.env` mirrors `VLLM_MODEL` / `VLLM_SERVED_NAME` /
  `VLLM_TOOL_CALL_PARSER` (= the primary) so the read-only single-model verbs
  (`status` / `whoami` / `doctor`) stay coherent on a fleet deployment.
  `model switch` remains single-model only.

### Fixed

## [0.8.1] - 2026-05-27

### Changed

- **SonarCloud cleanup (no behavior change).** Split `cmd_switch` into
  `_select_parser` / `_emit_dry_run` / `_apply_switch` helpers to bring its
  cognitive complexity under the gate, and hoisted the repeated `"(unset)"`
  literal in `model status` into a `_UNSET` constant.

## [0.8.0] - 2026-05-27

### Added

- **Per-model tool-call parser auto-selection.** New `model_gear/runtime/_parser.py`
  `infer_parser()` maps a model name to its parser (`qwen3_coder` for
  Qwen3-Coder / Qwen3.6, `hermes` for Qwen3 dense, unknown → leave untouched).
  `model switch` now picks the right parser automatically so tool calling keeps
  working across a switch without the caller remembering it; `--tool-call-parser`
  still overrides ([issue #13](https://github.com/agentculture/model-gear/issues/13)).
- **Post-switch / post-start tool-calling probe.** `model switch --apply` and
  `model serve --apply` now probe `tool_choice:"auto"` once the container is
  healthy and report PASS/FAIL (with the called tool names) — reusing the
  existing `assess` probe. `--no-probe` skips it; the probe never aborts the
  command (unreachable / HTTP 400 degrade to a FAIL result).
- **`model status` reports the active `tool_call_parser`** (`VLLM_TOOL_CALL_PARSER`),
  so "which gear am I in" is complete without `docker inspect`.

### Changed

- **`lepenseur` is retired; the deployed agent is now `model-gear`.** The tool and
  the deployed agent share one identity. Updated `culture.yaml` (`suffix: model-gear`),
  the `AGENTS.md` system prompt, `model whoami` / `learn` / `explain` output, the
  posting nick (`.claude/skills.local.yaml.example`), the compose/`.env` templates,
  `README.md`, and `CLAUDE.md` (the former "two identities" section now describes
  one).

### Fixed

## [0.7.0] - 2026-05-27

### Added

- **OpenAI tool/function calling** on the served vLLM model. The packaged compose
  template (`model_gear/templates/docker-compose.yml`) now serves with
  `--enable-auto-tool-choice` and `--tool-call-parser=${VLLM_TOOL_CALL_PARSER:-hermes}`,
  so `tool_choice:"auto"` requests return a `tool_calls` array instead of HTTP
  400. Additive — plain chat/reasoning is unaffected, no extra GPU/memory cost.
  Unblocks coder-agent harnesses that drive the model entirely through tool calls
  ([issue #9](https://github.com/agentculture/model-gear/issues/9)).
- **`VLLM_TOOL_CALL_PARSER`** env var (default `hermes`) + **`model switch
  --tool-call-parser`** — the parser is per-model: `hermes` fits Qwen3 dense
  (e.g. `Qwen3-32B`), while Qwen3-Coder / Qwen3.6 checkpoints emit the XML
  function format and need `qwen3_coder`. `switch` writes the var only when the
  flag is given, so retuning a model never clobbers its parser.
- **`model assess --tools`** — an opt-in tool-calling probe that verifies a
  `tool_choice:"auto"` request returns a `tool_calls` array naming a `finish`
  function. Degrades gracefully (a FAIL row, no abort) against a server that
  lacks the flags.

### Changed

### Fixed

## [0.6.0] - 2026-05-27

### Added

- **devague workflow trio** vendored under `.claude/skills/` (cite-don't-import):
  `think` (idea→spec), `spec-to-plan` (spec→plan), and `assign-to-workforce`
  (plan→parallel implementation) — the operator chain for the deterministic
  `devague` CLI. Authored in `agentculture/devague`, vendored via guildmaster;
  each carries `type: command` (load-bearing on the culture/agex backend, where
  a `SKILL.md` without `type:` is silently skipped). They drive the `devague`
  CLI at runtime (`uv tool install devague`), resolved portably by the wrappers.
- **`docs/skill-sources.md`** — provenance ledger recording the citation path
  and authoring origin of every vendored skill (the trio plus the six
  steward-sourced skills).

## [0.5.0] - 2026-05-27

Redesigned the repo around **running, assessing, and switching the local vLLM
model**. The model-ops logic that lived in the `model-runner` *skill* is now a
first-class CLI. lepenseur is still the deployed agent that consumes the served
model; model-gear is the tool that runs it.

### Added

- **Model-ops verbs** on the `model` CLI: `switch <model>`, `serve` (alias
  `start`) / `stop`, `status`, `assess` (correctness probes), `benchmark`
  (decode throughput + prefill), and `init` (scaffold a deployment dir). Write
  verbs (`switch`/`serve`/`stop`/`init`) are **dry-run by default** and require
  `--apply` (mutation-safety rule).
- **Scaffold-based deployment.** `docker-compose.yml` + `env.example` ship as
  packaged templates under `model_gear/templates/`; `model init` materialises
  them into `~/.model-gear` (default), a `TARGET`, or the local folder. Every
  model-ops verb resolves the deployment dir via `--compose-dir` →
  `$MODEL_GEAR_DIR` → `~/.model-gear`.
- Ported runtime modules (`model_gear/runtime/` + `model_gear/assess.py`),
  stdlib-only (`urllib`, fixed-argv `subprocess`), with full unit tests.
- `model overview` now folds in the currently-served model and the
  candidate-model list, filterable with `--current` / `--list`.

### Changed

- **PyPI distribution renamed `lepenseur` → `model-gear`; binary `lepenseur` →
  `model`; Python package `lepenseur` → `model_gear`.** Error class
  `LepenseurError` → `ModelGearError`. The `lepenseur` console script is removed.
- Agent-first verbs reframed for the tool: `whoami` reports tool/machine/served
  model/container health/agent; `learn` teaches the model-ops surface; `explain`
  catalog rewritten (`switch`/`assess`/`backend`/`models`/…).
- `doctor` is now **real** — checks docker availability, deployment scaffold,
  `.env` ↔ `culture.yaml` coherence, and `/health` reachability (a down model is
  a warning, not a failure).
- The `model-runner` skill is now a thin shim that `exec`s `model`; its
  `_assess.py` was removed (the logic lives in `model_gear/assess.py`).
- `AGENTS.md` / `culture.yaml` clarified: they describe the deployed `lepenseur`
  agent, not the repo. README + CLAUDE.md reoriented around model-gear.

### Fixed

- **BREAKING:** the vLLM container is renamed `lepenseur-vllm` → `model-gear-vllm`.
  A box running the old container must `docker compose down` under the old name,
  then `model init --apply` + `model serve --apply`.

## [0.4.0] - 2026-05-27

### Added

- `model-runner` skill (local, not vendored): `switch` the local vLLM runtime
  model and `assess`/benchmark it (stdlib `_assess.py` for correctness +
  throughput, host-side facts via the wrapper). Drives this repo's compose +
  `.env`; documented in CLAUDE.md and README. Mutating verbs (`switch`, `down`)
  are dry-run by default and require `--apply` (CLAUDE.md mutation-safety rule);
  `--port` defaults to `.env`'s `VLLM_PORT` (then 8000).

### Changed

- `docs/qwen3.6-27b-nvfp4.md`: filled with the live load-test (DGX Spark/GB10,
  2026-05-27). `mmangkad/Qwen3.6-27B-NVFP4` loads and serves under our vLLM image
  (no `--trust-remote-code`); ~7.9–8.0 tok/s decode, ~70 GB reserved, 29 GB
  weights. It is a hybrid Mamba/linear-attention vision-language model and is
  slower on decode than the 32B here — recommendation: **keep the 32B**. All
  pre-flight caveats (SGLang-only, multimodal, ModelOpt rc) validated/resolved.

## [0.3.0] - 2026-05-27

### Added

- `docs/qwen3-32b-nvfp4.md`: per-model doc for the current runtime model, with a
  live test on DGX Spark (GB10) — `nvcr.io/nvidia/vllm:26.04-py3` (engine
  `0.19.0+...nv26.04`), ~9.7 tok/s decode (batch=1), ~2,800 tok/s prefill, ~72 GB
  reserved at `gpu-memory-utilization=0.6`, correctness verified.
- `docs/qwen3.6-27b-nvfp4.md`: per-model doc for candidate
  `mmangkad/Qwen3.6-27B-NVFP4`. Its `Qwen3_5ForConditionalGeneration` arch is
  registered in the current vLLM image (so the same compose can serve it); live
  load-test/benchmark tracked by issue #6.
- README "Per-model notes" linking both docs.

### Fixed

- `docker-compose.yml`: corrected the `--reasoning-parser=qwen3` comment — on the
  nv26.04 build the `<think>` trace is returned in the `reasoning` field, not
  `reasoning_content`.

## [0.2.0] - 2026-05-27

### Added

- `docker-compose.yml` + `.env.example`: a local vLLM server (NGC
  `nvcr.io/nvidia/vllm` image) that serves the runtime model as an
  OpenAI-compatible API on `:8000` for the `acp` backend, tuned for DGX Spark
  (GB10 Blackwell, 128 GB unified memory).
- README "Running the model locally (vLLM)" section.

### Changed

- Switched lepenseur's runtime model from
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` to `nvidia/Qwen3-32B-NVFP4`
  across `culture.yaml`, `AGENTS.md`, `lepenseur/explain/catalog.py`, `README.md`,
  and `CLAUDE.md` (32B dense NVFP4 reasoning model with a thinking mode).

## [0.1.0] - 2026-05-22

### Added

- Initial CLI/PyPI sibling scaffold (copied and adapted from the `lecodeur`
  twin): top-level `lepenseur` package with the `lepenseur` console script.
- Read-only verbs: `whoami`, `learn`, `explain`, `overview`, and a `cli`
  noun with `cli overview`.
- `doctor` verb shipped as a rubric-shaped stub; real self-diagnosis semantics
  for a thinking ("non-doer") agent are deferred to a follow-up.
- Runtime identity files: `AGENTS.md` and `culture.yaml` (acp backend,
  `vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).
- CI: `tests.yml` (test + lint + `afi cli doctor . --strict` gate +
  version-check) and `publish.yml` (PyPI/TestPyPI via Trusted Publishing).
- Six vendored skills under `.claude/skills/` (cicd, communicate, version-bump,
  run-tests, sonarclaude, doc-test-alignment), provenance: steward.
