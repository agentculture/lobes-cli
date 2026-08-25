# Build Plan — cortex replica pool (#199)

slug: `cortex-replica-pool-199` · status: `exported` · from frame: `cortex-replica-pool-199`

> lobes serves one logical cortex from every box that hosts a compatible replica (Spark, Thor, Orin at 256K); each gateway routes model=cortex to the most available compatible replica and every answer says which replica served it — so when one box is overloaded, new requests land on a free one without callers changing anything

## Tasks

### t1 — Capture the PRE-POOL baseline transcript on the live Spark+Thor pair (docs/evidence/2026-08-XX-baseline-cortex-single-owner.txt): three concurrent model=cortex requests to one gateway, then the same under local pressure while the peer idles; record gateway pins and headers

- covers: c28, h20
- acceptance:
  - Transcript shows all three concurrent requests answered by the dialed box (no X-Lobes-Proxied-By) and a 429 busy under local pressure while the peer's /status shows running=0
  - Transcript records `MODEL_GEAR_VERSION` and lobes --version on both boxes and the exact curl commands

### t2 — Config: parse the plural peer family and self-origin — <PREFIX>`_PEER_ORIGINS` / <PREFIX>`_PEER_API_KEYS` (comma-separated, positional, empty key slot legal, shorter key list = startup error) and `GATEWAY_SELF_ORIGIN` — into new RoutingTable fields (`replica_origins`: Mapping\[str, tuple\[str,...\]\], `replica_api_keys`, `self_origin`) plus per-lane declared fingerprint keys (quantization, `kv_cache_dtype`, reasoning/tool parser, speculative config); existing scalar fields and `order_backends` untouched. Files: lobes/gateway/`_config.py`, lobes/gateway/`_routing.py` (fields only), tests/`test_gateway_config_replicas.py`

- covers: c2, h2, c11, h9, c18, h13, h29
- acceptance:
  - `PRIMARY_PEER_ORIGINS`=<http://a:8000,http://b:8000> with `PRIMARY_PEER_API_KEYS`=k1, yields two origins, keys (k1, '') — an empty slot is legal
  - `PRIMARY_PEER_API_KEYS` shorter than `PRIMARY_PEER_ORIGINS` raises a startup configuration error naming the prefix; a longer list is also an error
  - A trailing slash and surrounding whitespace on each origin are stripped exactly as `_peer_origins` does; an env with no \*`_PEER_ORIGINS` produces empty replica maps and a RoutingTable equal to today's
  - The parity test iterates `PEER_ORIGINS_ENV` / `PEER_API_KEYS_ENV` against `FEASIBLE_ENV` keys (extends tests/`test_gateway_config_proxy.py`'s invariant)
  - All builtin shape goldens (tests/`test_shape_goldens.py`) stay byte-identical — `shape_render` emits no \*`_PEER_ORIGINS` or `GATEWAY_SELF_ORIGIN` key
  - Every existing routing test passes unchanged; `order_backends` still returns a 0-or-1 list

### t3 — Templates + doctor: add gateway-service passthrough lines for <PREFIX>`_PEER_ORIGINS`, <PREFIX>`_PEER_API_KEYS`, `GATEWAY_SELF_ORIGIN` and the per-lane declared fingerprint keys (`PRIMARY_QUANTIZATION`, `PRIMARY_KV_CACHE_DTYPE`, `PRIMARY_REASONING_PARSER`, `PRIMARY_TOOL_PARSER`, `PRIMARY_SPECULATIVE_CONFIG` and the other prefixes' equivalents) in lobes/templates/fleet/docker-compose.yml and document them in env.example; teach lobes doctor to flag a deployed docker-compose.yml lacking any passthrough for a key set in .env. Files: lobes/templates/fleet/docker-compose.yml, lobes/templates/fleet/env.example, lobes/cli/`_commands`/doctor.py, tests

- covers: c36, h28
- acceptance:
  - A deployed compose missing the `PRIMARY_PEER_ORIGINS` passthrough while .env sets it makes lobes doctor print the missing key under a 'gateway passthrough' finding; doctor --fix --apply does not edit compose (compose is re-scaffolded, not patched)
  - The packaged template passes every key through as ${VAR:-} in the gateway service; the template test that pins the gateway environment block covers the new keys
  - env.example documents the plural block next to the singular one with the empty-slot rule and the 'peer set differs per box' note; no hostname appears

### t4 — Replica snapshot: new lobes/gateway/`_replicas.py` with ReplicaState (origin, local: bool, ready, busy, health, running, waiting, fingerprint, compatible: bool, reason, `last_seen`) and a ReplicaCache that, on the ReadinessCache daemon-thread pattern (separate peer thread, bounded timeouts, O(1) current()), probes each peer gateway's GET /status (busy, backends\[\].health, metrics.running/waiting) and GET /capabilities fingerprint with the peer key, probes the local lane's own /v1/models (id, `max_model_len`) and merges the declared lane config; computes compatible from served id + quantization + max context + runtime only, with `kv_cache_dtype`/draft/rope recorded as informational. Files: lobes/gateway/`_replicas.py`, tests/`test_gateway_replicas.py`

- covers: c5, h5, c13, h11, c33, h25, c34, h26
- acceptance:
  - current() is a dict read that opens no socket (asserted by a fake urlopen that fails the test if called from the request thread)
  - A peer whose fingerprint differs in served id, quant, `max_model_len` or runtime is marked compatible=false with a reason string naming the differing field; a peer differing only in `kv_cache_dtype` or draft mode stays compatible=true with the field recorded
  - A peer gateway that is down, times out, or reports busy:true or health!=ok for the role is marked not selectable within one refresh interval
  - For a served id absent from the catalog the fingerprint reports the live id and `max_model_len` and 'unknown' for undeclared fields — never catalog values
  - No probe dials a vLLM port directly on a peer; only the peer gateway origin is contacted

### t5 — Selection policy: pure function `select_replica`(candidates: Sequence\[ReplicaState\], \*, affinity: str | None, `local_busy`: bool) -> Selection(origin, local, reason) in new lobes/gateway/`_selection.py` — deterministic weighted least-load: filter compatible+ready+not-busy, estimated wait = (running+waiting)/`declared_weight`, local wins ties (locality), affinity (a stable hash of the key -> preferred replica) honoured only when the preferred replica is selectable and not worse than the best by more than a declared margin; reason vocabulary local-idle | peer-less-loaded | local-busy-forwarded | affinity | sole-ready | none. Files: lobes/gateway/`_selection.py`, tests/`test_gateway_selection.py`

- covers: c3, h3, c37
- acceptance:
  - Local ready and idle vs peer ready and idle -> local, reason local-idle
  - Local running=3 waiting=2 vs peer running=0 -> peer, reason peer-less-loaded; `local_busy`=True with a ready peer -> peer, reason local-busy-forwarded
  - Affinity key present, preferred replica within the margin -> preferred, reason affinity; preferred replica busy or dead -> availability wins and the reason is not affinity; absent key -> selection is purely availability-driven
  - Incompatible or unready candidates never appear in the result; all unavailable -> Selection(None, reason none)
  - Same input always yields the same output (no randomness, no clock)

### t6 — Capabilities surfaces: additive per-role 'replicas' list (origin, local, ready, busy, running, waiting, compatible, reason, fingerprint) and per-role 'fingerprint' object on the payload built by lobes/roles.py, rendered by lobes capabilities / lobes capabilities --replicas and lobes endpoint <role> --replicas including a 'would choose: <origin> (<reason>)' line; existing keys keep type and meaning; lobes route untouched. Files: lobes/roles.py, lobes/cli/`_commands`/capabilities.py, lobes/cli/`_commands`/endpoint.py, tests/`test_roles_replicas.py`, tests/`test_cli_capabilities_replicas.py`

- depends on: t2, t4, t5
- covers: c9, h7, c10, h8, c12, h10
- acceptance:
  - A payload built with no `replica_origins` has no 'replicas' key and is byte-identical to today's; tests/`test_colleague_contract.py` and tests/`test_roles_proxied.py` pass unchanged
  - With two replicas declared, `hosted_by` is still absent/str and proxied still bool; feasible stays true for the locally hosted role
  - The CLI replica view shows each candidate's ready/busy/load/compatible/reason and the would-choose line derived from `select_replica`; the offline fallback renders the declared list with ready unknown
  - lobes/cli/`_commands`/route.py has no diff

### t7 — Gateway dispatch (part 1): pool path in lobes/gateway/server.py — for a request whose model resolves (alias OR raw served id) to a role with a declared replica set, consult ReplicaCache.current() + `select_replica` before local dispatch; forward to a chosen peer origin via a generalized `_proxy_to_peer`(origin, `api_key`) that stamps X-Lobes-Proxied and returns X-Lobes-Proxied-By; an inbound X-Lobes-Proxied request is served by the local replica only (508 only when no local replica); read X-Lobes-Affinity; stamp X-Lobes-Served-By: <`GATEWAY_SELF_ORIGIN` or 'local'> on local answers and X-Lobes-Route-Reason on every pooled answer; forward X-Lobes-Affinity to the peer. Files: lobes/gateway/server.py, tests/`test_gateway_pool.py`

- depends on: t2, t4, t5
- covers: c1, h1, c4, h4, c19, h14, c31, h23, h30
- acceptance:
  - model=cortex and model=unsloth/Qwen3.8-27B-NVFP4 produce identical selection, forward target and markers in the same snapshot
  - With no replica set declared, every response, header and body is byte-identical to the pre-pool release (golden comparison over chat, models, capabilities, and error paths)
  - An inbound request carrying X-Lobes-Proxied never opens an outbound socket even when the snapshot prefers a peer; a box with no local replica still answers 508 `proxy_loop`
  - Local answers carry X-Lobes-Served-By (self origin, or 'local' when `GATEWAY_SELF_ORIGIN` is unset) and forwarded answers carry X-Lobes-Proxied-By; both carry X-Lobes-Route-Reason from the selection
  - Streaming forwarded answers relay through the existing byte tunnel; a drop mid-stream ends the client stream with no replay

### t8 — Gateway dispatch (part 2): pressure and failure semantics in lobes/gateway/server.py — under local busy pressure a pooled request is forwarded to a selectable peer instead of shed; 429 busy + Retry-After only when no replica is selectable (503 `backend_unavailable` when all are down); at most ONE forward per request (a peer's 429/4xx rides back via the existing relay and the forwarder never retries locally); pre-dispatch failure (refused/timeout/5xx before any 2xx) retries the next selectable replica at most once per replica; wire the replicas/fingerprint payload from t5 into GET /capabilities and start ReplicaCache in serve(). Files: lobes/gateway/server.py, tests/`test_gateway_pool_pressure.py`

- depends on: t7, t6
- covers: c7, h6, c15, h12, c35, h27, c37
- acceptance:
  - PressureCache busy + peer selectable -> 200 with X-Lobes-Proxied-By and reason local-busy-forwarded; busy + no selectable peer -> the existing 429 body and Retry-After
  - Both gateways busy: exactly one outbound forward and one 429 relayed to the caller, never two forwards (counted via a fake `open_upstream`)
  - First replica refuses the connection -> the next selectable replica is tried once; all refuse -> 503 `backend_unavailable` listing every attempt; a 2xx that then drops mid-stream is not retried
  - GET /capabilities on a pooled box carries the replicas list with live ready/busy from ReplicaCache; serve() starts the cache before binding so the first request has a snapshot

### t9 — N-gateway loopback integration suite: extend tests/`test_proxy_integration.py` with `_n_gateways`(n, `pool_env`) and drive the acceptance scenarios offline — spread across two replicas, local-busy forward, peer down, single hop under a marked request, mutual busy, raw-id equivalence, affinity stickiness within margin, and a no-pool byte-identical golden. Files: tests/`test_proxy_integration.py`, tests/goldens/ (no-pool golden)

- depends on: t8
- covers: c1, h1, c4, h4, c35, h27
- acceptance:
  - Three concurrent requests to gateway A with A's local replica loaded are answered by both A and B as proven by X-Lobes-Served-By / X-Lobes-Proxied-By
  - With B stopped, A keeps answering with reason sole-ready; with A marked busy and B up, new requests carry X-Lobes-Proxied-By naming B
  - A request that arrives at B already marked X-Lobes-Proxied is served locally by B and B opens no outbound socket
  - Same X-Lobes-Affinity across five sequential requests lands on the same replica while it stays selectable
  - The no-pool golden run of the suite matches the pre-pool byte-for-byte fixture

### t10 — Docs, explain and changelog: describe the replica pool (config family, empty-slot rule, `GATEWAY_SELF_ORIGIN`, selection policy, affinity header, markers, route reasons, failure table, compose passthrough + doctor check, rollback = delete the \*`_PEER_ORIGINS` line and recreate the gateway) in docs/gateway-fleet.md, docs/deployment-shapes.md, docs/colleague-stack.md (capabilities schema), docs/openai-api.md, lobes/explain/catalog.py (`_GATEWAY`/`_SHAPES`/`_API`/`_ROLES`), CLAUDE.md, CHANGELOG.md — all labelled DECLARED/UNVALIDATED until t10's transcript lands, cortex-only validation stated explicitly. Files: docs/\*.md, lobes/explain/catalog.py, CLAUDE.md, CHANGELOG.md, pyproject.toml (version bump)

- depends on: t8
- covers: c21, h15, c38, h31
- acceptance:
  - Every doc that names `hosted_by` or X-Lobes-Proxied-By also names the replicas list, X-Lobes-Served-By and X-Lobes-Route-Reason; grep finds no doc claiming the pool validated before the evidence file exists
  - lobes explain gateway / api / shapes / roles render the new text (snapshot tests updated); docs state that any non-cortex pooled role is declared/unvalidated
  - Version bumped (minor) and a CHANGELOG entry added; markdownlint passes on every touched doc

### t11 — Live acceptance on Spark+Thor: re-scaffold both deployed docker-compose.yml from the packaged template, set `GATEWAY_SELF_ORIGIN`, `PRIMARY_PEER_ORIGINS` and `PRIMARY_PEER_API_KEYS` per box (Spark lists the Thor at :8000 with an empty key slot; Thor lists the Spark at :8001 with the Spark's key), align `MODEL_GEAR_VERSION` on both, recreate only the gateway containers, and run the three scenarios — spread, Spark busy -> Thor, Thor down -> Spark — plus the affinity check; record docs/evidence/2026-XX-XX-accept-cortex-replica-pool-spark-thor.txt and flip the docs from DECLARED to VALIDATED for cortex only. Files: docs/evidence/, docs/\*.md (status flip), CLAUDE.md

- depends on: t1, t9, t10
- covers: c26, h18, c27, h19, c29, h21, c30, h22
- acceptance:
  - Both gateways list each other under cortex.replicas with compatible=true, ready=true and the live fingerprint (id unsloth/Qwen3.8-27B-NVFP4, `max_model_len` 262144), with `kv_cache_dtype` shown as fp8 vs auto
  - Three concurrent requests to the Spark gateway are served by both boxes (markers in the transcript); with the Spark saturated a new request returns 200 with X-Lobes-Proxied-By naming the Thor, not 429; with the Thor's gateway stopped the Spark keeps serving with reason sole-ready and no caller change
  - Aggregate throughput of the three concurrent requests exceeds t0's single-box baseline and the numbers are in the transcript with commands, headers and both pins
  - The culture backend was not changed; requests were sent as the deployed consumers send them (raw served id) and with X-Lobes-Affinity added by curl

## Risks

- [unknown_nonblocking] Version skew: until both gateways run the pool release, a forward to a pre-pool peer works (plain request) but its answer carries no X-Lobes-Served-By and its /capabilities has no fingerprint, so the pre-pool peer shows compatible=unknown and is not selectable — t10 aligns pins before measuring (task t11)
- [follow_up] Which client sets X-Lobes-Affinity is out of this plan: the vllm-local provider lives outside ../culture and was not examined; affinity is exercised via curl only (t8, t10) and a culture-side follow-up issue is filed from t9
- [out_of_scope] llama.cpp load/fingerprint adapter (the Orin) is out of scope — the Orin is exempt (decision c22); a pooled llama.cpp replica needs its own /status field mapping before it could ever be compatible
- [unknown_nonblocking] Declared per-replica weight: t4 takes a declared decode weight per origin (default 1.0) because no measurement of pool behaviour exists; t10 records the measured spread so a later PR can calibrate or replace it with a latency EMA (task t5)
