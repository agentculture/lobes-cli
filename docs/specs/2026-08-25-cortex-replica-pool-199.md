# cortex replica pool (#199)

> lobes serves one logical cortex from every box that hosts a compatible replica (Spark, Thor, Orin at 256K); each gateway routes model=cortex to the most available compatible replica and every answer says which replica served it — so when one box is overloaded, new requests land on a free one without callers changing anything
> instruction: Verify on the live Spark+Thor pair: send N concurrent model=cortex requests to one gateway and read the serving-replica marker on each; confirm both origins appear, then repeat with the local replica saturated and with the peer stopped.

## Audience

- Colleague parent agents and their concurrent subagents that address model=cortex through whichever lobes gateway is local to them (Spark or Thor), plus the operator who declares the replica set per box in .env

## Before → After

- Before: Each box owns cortex alone: a hosted role never consults a peer, `order_backends` returns exactly one owner (#91), local pressure sheds with 429 even while the other box idles, and three concurrent subagents all queue on the one gateway they dialed — the Spark's and Thor's capacity never merge
- After: A caller dials any one gateway with model=cortex and is served by whichever compatible replica (local or peer) is free; when the local cortex is saturated or under pressure the request goes to the free peer instead of a 429; every answer names the replica that served it; a box with no \*`_PEER_ORIGINS` declared behaves byte-identically to today

## Why it matters

- Two boxes now serve the same Qwen3.8-27B-NVFP4 cortex at 256K with very different speeds (Spark DSpark ~46 tok/s code, Thor MTP ~27 tok/s); without a pool that capacity is stranded behind per-box ownership and a busy box turns callers away while its twin idles

## Requirements

- RoutingTable gains a per-role REPLICA SET — the local Backend (if hosted) plus N operator-declared peer origins — and selection returns exactly one replica per request; a role with one owner is the degenerate one-replica case and every no-pool deployment stays byte-identical (lobes/gateway/`_routing.py`: Backend.`base_url` and RoutingTable.`peer_origins` are scalar per name, `order_backends` returns a 0-or-1 list by design)
  - honesty: The replica set is a NEW structure beside Backend/RoutingTable.`peer_origins`; `order_backends` still returns 0-or-1 for single-owner roles and every existing routing test passes unchanged
- The gateway can forward model=cortex to a peer replica even when THIS box hosts cortex — today `_proxied_owner` only engages for feasible:false roles; the pool path must sit before local dispatch and pick local-vs-peer by availability (lobes/gateway/server.py `handle_post` -> `_proxied_owner` -> `_proxy_to_peer`)
  - honesty: A hosted cortex request is dispatched to a peer only when the selection snapshot says the peer is ready AND better-placed (lower estimated load) than the local replica; a local replica that is ready and unloaded always wins the tie (locality tie-breaker)
- Single-hop stays: an inbound request carrying X-Lobes-Proxied is served by the receiving box's LOCAL replica only — never re-dispatched to a third replica — and the existing 508 `proxy_loop` refusal remains for a box with no local replica (server.py `_arriving_hop_marker` / `_PROXY_LOOP_STATUS`)
  - honesty: An inbound request carrying X-Lobes-Proxied is served by the local replica or refused — it is never forwarded again, proven by a three-gateway loopback test where hop 2 never opens an outbound socket
- Replica selection reads an O(1) cached snapshot refreshed by a background thread — the ReadinessCache pattern (local /health + peer /v1/models, 5s/3s timeouts, separate peer thread) extended with a per-replica LOAD snapshot; no probe ever runs inline on the request path (lobes/gateway/`_readiness.py` ReadinessCache.current())
  - honesty: No socket is opened on the request path for selection: the snapshot read is a dict lookup, and the load probe runs on the ReadinessCache background threads at the existing intervals (a hung peer never delays local dispatch)
- Under local swap/iowait pressure a cortex request is FORWARDED to a ready peer replica instead of shed — the 429 busy + Retry-After answer is reserved for 'no replica anywhere is available' (lobes/gateway/`_pressure_policy.py` decide, `_tier_request`.PressureCache; proxied requests already bypass local pressure per #85)
  - honesty: Under local busy pressure a cortex request is forwarded to a ready peer replica; only when no replica in the set is ready does the caller receive the existing 429 busy + Retry-After (or 503 `backend_unavailable` when all are down)
- GET /capabilities, lobes capabilities and lobes endpoint expose an additive per-role 'replicas' list (origin/node, ready, load, fingerprint, local:true|false) while feasible/proxied/`hosted_by`/ready/loaded keep their documented single-owner meaning as a summary — no existing key changes type (lobes/roles.py `annotate_peer_referrals`; docs/colleague-stack.md:334-363 contract)
  - honesty: The replicas list is ADDITIVE: every existing /capabilities key keeps its type and single-owner meaning, tests/`test_colleague_contract.py` passes unchanged, and a no-pool payload has no replicas key at all
- The control-plane 'which replica would serve this and why' view lands on the capabilities/endpoint surface (e.g. lobes endpoint cortex --replicas / lobes capabilities --replicas), NOT on lobes route — lobes route is an LLM task->tier classifier that calls the hand model (lobes/cli/`_commands`/route.py `cmd_route`) and has no --explain flag; the issue's 'lobes route --explain' wording must not be taken literally
  - honesty: lobes route is untouched; the replica view (candidates, ready, load, fingerprint, which one would be chosen and why) renders from lobes capabilities / lobes endpoint cortex and GET /capabilities
- Replica compatibility is checked against a LIVE-probed serving fingerprint per replica (served model id, runtime, quantization, max context, draft/MTP mode, tool/reasoning parser where the engine reports it), not the catalog or .env — the Spark gateway (up 4 days, pre-d4 env) still advertises cortex context=1048576 while ~/.lobes/.env says 262144, and the Orin advertises quant=modelopt/mtp=true/runtime=vllm for what is actually llama.cpp `Q4_K_M`, because RoleInfo falls back to catalog data for its unknown served id 'cortex'
  - honesty: A peer whose live-probed fingerprint (served id, quant, max context, runtime) differs from the local replica's never enters the candidate set — it is listed in replicas with compatible:false and a reason, never silently pooled
- Every replica origin carries its own outbound key that is a copy of that peer's inbound `GATEWAY_API_KEY` (O(machines) key material, never per-pair), the caller's Authorization is stripped before every forward, and a box whose inbound gate is unset gets no key — exactly the #115/#127 pairwise rules applied per origin (lobes/gateway/`_config.py` `_peer_api_keys`; docs/gateway-fleet.md proxy-lobes)
  - honesty: Each entry in <PREFIX>`_PEER_API_KEYS` is matched positionally to <PREFIX>`_PEER_ORIGINS`; a mismatch in list length is a startup configuration error, not a silent unauthenticated forward
  - honesty: An EMPTY slot in <PREFIX>`_PEER_API_KEYS` is legal and means 'this peer has no inbound gate' (the Thor sets no `GATEWAY_API_KEY` today); only a list SHORTER than the origins list is a startup error
- Every cortex answer identifies the serving replica: locally-served answers gain a marker naming this box's own origin (today only proxied answers carry X-Lobes-Proxied-By, so a caller cannot distinguish 'served here' from 'served on the peer I dialed' once the pool exists) and proxied answers keep X-Lobes-Proxied-By verbatim
  - honesty: Every cortex response carries a serving-replica marker: X-Lobes-Served-By: <this box's own declared origin> for local answers and the existing X-Lobes-Proxied-By for forwarded ones; both are testable in the loopback suite
- Docs and in-CLI text describing the single-peer contract are updated alongside the code: docs/gateway-fleet.md (proxy-lobes), docs/deployment-shapes.md (referral/proxy), docs/colleague-stack.md (capabilities schema), docs/openai-api.md (markers), lobes/explain/catalog.py `_GATEWAY`/`_SHAPES`/`_API`/`_ROLES` blocks, CLAUDE.md — and the feature is DECLARED until a live three-box acceptance transcript lands under docs/evidence/ (#108)
  - honesty: docs/gateway-fleet.md, docs/deployment-shapes.md, docs/colleague-stack.md, docs/openai-api.md, lobes/explain/catalog.py and CLAUDE.md describe the pool, and no doc or capabilities output claims it validated until the docs/evidence transcript lands
- The pool applies to requests addressed by the RAW served id (model=unsloth/Qwen3.8-27B-NVFP4) exactly as to the cortex alias — every deployed consumer pins the raw id in culture.yaml and none addresses 'cortex' (the 2026-07-31 audit finding still holds), so an alias-only pool would never see a real caller
  - honesty: A request with model=unsloth/Qwen3.8-27B-NVFP4 and one with model=cortex are placed by the same selection and carry the same markers, proven by the loopback suite
- Each gateway exposes a per-role serving fingerprint that peers read through the gateway: served id and `max_model_len` taken LIVE from the lane's own /v1/models (vLLM reports `max_model_len`=262144 on the Thor), plus the lane's DECLARED config (quantization, `kv_cache_dtype`, reasoning/tool parser, speculative config) passed into the gateway env — never the catalog fallback that today mislabels the Orin. GET /status carries none of these fields and the gateway config carries no quant/parser/kv keys at all
  - honesty: For a served id the catalog does not know, the fingerprint still reports the live id and `max_model_len` and marks undeclared fields as unknown rather than inventing quant/mtp/runtime values
- The per-replica load snapshot is read from the PEER GATEWAY's GET /status (top-level busy, backends\[\].health and metrics.running/waiting) on the readiness background thread — measured ~110 ms on the Thor — using the peer key; a replica whose /status reports busy:true or health!=ok is not selected. vLLM /metrics is not reachable cross-box (the vLLM port is unpublished on the Spark), so the local replica's own load comes from the same /status aggregation run locally
  - honesty: Selection never dials vLLM directly on a peer; with the peer gateway down the snapshot marks the replica unavailable within one refresh interval
- At most ONE forward per request: a request the local box forwards is served or refused at the receiver under the receiver's own pressure policy (its 429 rides back through the existing 4xx relay, server.py:1025) and the forwarder never retries it locally afterwards — two mutually-loaded boxes cannot ping-pong, and the caller gets an honest 429 + Retry-After
  - honesty: In a loopback test with both gateways marked busy, a request produces exactly one outbound forward and one 429, never two forwards
- The new \*`_PEER_ORIGINS` / \*`_PEER_API_KEYS` keys are added to the gateway service passthrough in lobes/templates/fleet/docker-compose.yml AND lobes doctor flags a deployed docker-compose.yml that lacks them — a .env key with no passthrough silently never reaches the container (the 2026-07-17 `MUSE_`\* trap); both the Spark's hand-edited compose and the Thor's must be re-scaffolded before the pool can arm
  - honesty: lobes doctor on a compose file missing the passthrough prints the missing keys; the loopback suite starts the gateway from the packaged template with the keys present
- Every pooled answer carries X-Lobes-Route-Reason naming why the replica was chosen (local-idle | peer-less-loaded | local-busy-forwarded | affinity | sole-ready) so a trace can distinguish a deliberate forward from a fallback without reading gateway logs; the same reason appears in the replica view's would-choose line
  - honesty: The header is present on every cortex answer when a pool is declared and absent when no pool is declared

## Honesty conditions

- A request never fails or degrades because pooling exists: with the pool disabled (no \*`_PEER_ORIGINS`) every response, header and capabilities byte is identical to the pre-pool release
- `hosted_by` remains a string and proxied a bool in every rendered payload; a pooled role reports feasible:true when hosted locally, exactly as today
- `shape_render` emits no \*`_PEER_ORIGINS` key; all builtin shape goldens stay byte-identical; the operator types the list in .env
- A mid-stream peer failure ends the client's stream with no automatic replay; a pre-dispatch failure (refused/timeout/5xx before any 2xx) retries on the next ready replica at most once per replica
- Affinity is honoured only when X-Lobes-Affinity is present and the preferred replica is ready and not worse-placed than the alternative by more than a declared margin; absent the header, selection is purely availability-driven
- Parsing `PRIMARY_PEER_ORIGINS` on the Spark yields only the Thor origin and vice versa; no hostname appears in any packaged template, profile or shape
- The culture backend's vllm-local provider needs no change to benefit: it keeps sending model=cortex to its local gateway; only X-Lobes-Affinity is new and optional
- In the acceptance run, a request sent to the Spark gateway while the Spark cortex is saturated returns 200 with X-Lobes-Proxied-By naming the Thor, not a 429
- The baseline transcript (taken before the pool lands) shows three concurrent cortex requests to one gateway all served by that box and a 429 under local pressure while the peer idles
- The pooled acceptance run's aggregate throughput for three concurrent requests exceeds the single-box baseline's — the merged capacity is measured, not asserted
- The transcript lands under docs/evidence/ with the exact commands, headers and gateway pins for all three scenarios, and the offline suite includes a byte-identical no-pool golden
- The replica view shows `kv_cache_dtype`, draft mode and rope overrides per replica so an operator can see the non-equivalence they accepted
- docs and lobes capabilities describe pooling for cortex; any other pooled role is labelled declared/unvalidated and the acceptance transcript names cortex only

## Success signals

- Live three-scenario acceptance transcript under docs/evidence/ on the real Spark+Thor pair: (1) three concurrent model=cortex requests to ONE gateway are served by both boxes (markers prove it); (2) with the Spark made busy, new requests land on the Thor; (3) with the Thor down, the Spark keeps serving with no caller change — plus a byte-identical no-pool golden and the offline test suite green

## Scope / boundaries

- The three role states awake / asleep (referral) / proxy stay exactly as documented — a pool is a property of an awake or proxied role, not a fourth state; `hosted_by` stays a string, never a list (docs/gateway-fleet.md:534-542, docs/colleague-stack.md:437-461)
- Replica origins stay operator-typed in .env, never rendered by a shape or profile and never discovered — #92 (origins are operator-declared, never derived) holds for the list exactly as for the scalar, which also keeps replica config out of the #204 force-write lifecycle gap (lobes/profiles/`shape_render.py` emits no \*`_PEER_`\* key today)
- A replica failing mid-stream ends the client's stream honestly; nothing is replayed — the relay is a one-shot byte tunnel with no buffering (server.py `_relay_streaming`), and retry-on-another-replica applies only before the upstream returned 2xx
- The plural peer family is generic across the nine prefixes but v1 VALIDATES and documents pooling for cortex only; a pooled embed/rerank/senses/worker lane is declared-unvalidated (#108) and the acceptance transcript claims nothing about them

## Non-goals

- Fan-out execution (lobes fanout), a request-trace store (lobes trace), policy plugins and learned routing stay parked in #128 — #199 defines only #127's 'runtime-aware routing' phase, which docs/specs/2026-07-16-proxy-lobes-pairwise-auth.md enumerated but never specified
- No tensor/model parallelism, no distributed execution of one request, no shared KV across boxes, no merging of vLLM processes, and no change to the model Colleague chooses — each replica is a complete independent server; the pool shares request capacity only (issue #199 non-goals)

## Assumptions

- The per-replica load signal is vLLM's own /metrics (running, waiting, `kv_cache_usage_perc`) parsed by lobes/`_metrics.py` `probe_backend` — already used by GET /status and lobes overview --live, but fanned out uncached at request time and engine-specific (llama.cpp exposes different fields) — so a llama.cpp replica needs its own load adapter or a declared static capacity
- Replica origins are declared per role as a comma-separated single-line value in the existing prefix vocabulary — <PREFIX>`_PEER_ORIGINS` beside the scalar <PREFIX>`_PEER_ORIGIN` — because .env holds one value per key (lobes/runtime/`_env.py` `set_env` rewrites every KEY= line; no repeated keys, no newlines) and every peer family is keyed by the nine PREFIXes in lobes/gateway/`_config.py` `PEER_ORIGIN_ENV` / `PEER_PROXY_ENV` / `PEER_API_KEY_ENV`
- The three deployed cortexes are NOT bit-equivalent today (probed 2026-08-25): Spark = vLLM unsloth/Qwen3.8-27B-NVFP4 + DSpark draft @262144 (46.2 code / 13.7 prose tok/s); Thor = vLLM same checkpoint + MTP @262144 with the YaRN `hf_overrides` block retained (26.8 tok/s, 0.62.0); Orin = llama.cpp unsloth Qwen3.8-27B-UD-`Q4_K_M`, -c 262144, -np 1 (ONE slot), --alias cortex (8.46 tok/s). Served ids differ too: 'unsloth/Qwen3.8-27B-NVFP4' on Spark/Thor vs 'cortex' on Orin
- Tests mirror the existing fixtures: tests/`test_proxy_integration.py` `_two_gateways` grows an N-gateway loopback variant for selection/failover, tests/`test_gateway_proxy.py` gains pool env builders, and the UNWIRED-shape guard (`test_every_proxyable_role_resolves_a_served_name` — the lesson from the 0.54.6-0.54.8 inert worker proxy) is extended to the plural family
- Spark+Thor pool membership is an explicit operator compatibility policy, not bit-equivalence: same checkpoint, served id, runtime and 262144 window, but `PRIMARY_KV_CACHE_DTYPE` differs (Spark fp8, Thor auto/bf16), the Thor keeps the YaRN `hf_overrides` block, and the drafters differ (DSpark vs MTP). These are recorded as informational fingerprint fields, not disqualifiers; only served id, quantization, max context and runtime disqualify

## Scope exploration

- `s1` — `lobes/gateway/_routing.py (Backend, RoutingTable.peer_origins, order_backends)`: One owner per served name is architected in, not incidental: `order_backends` docstring 'never a failover chain', returns a list of length 0 or 1 (#91). `peer_origins` is Mapping\[str,str\]. A pool needs a new plural structure beside it, not a reinterpretation of these fields.
  - seeds: `c2`
- `s2` — `lobes/gateway/server.py (handle_post, _proxied_owner, _proxy_to_peer, _arriving_hop_marker)`: Proxy branch is reached only when the requested role resolves to a proxied (feasible:false) backend; a hosted role never consults peers. The 508 loop guard keys on one X-Lobes-Proxied marker; with a pool the marker must mean 'local only', which also guarantees dispatch terminates in at most 2 hops.
  - seeds: `c3`, `c4`
- `s3` — `lobes/gateway/_readiness.py (ReadinessCache, _peer_loop, PeerSpec)`: Snapshot is a bare tri-state bool per name: liveness + served-id confirmation only. No load, latency or queue depth anywhere. `handle_post` does not consult it at all (a dead wired backend is still dialed live and 503s). PeerSpec has one origin. The daemon-thread/O(1)-read pattern is the right substrate for a load snapshot.
  - seeds: `c5`
- `s4` — `lobes/_metrics.py (probe_backend) and gateway GET /status`: A ready-made queue-depth signal exists (vLLM running/waiting/kv usage) but is only consumed by observability surfaces via a ThreadPoolExecutor at request time, never cached, never keyed per replica, and its field set is vLLM-specific.
  - seeds: `c6`
- `s5` — `lobes/gateway/_pressure_policy.py + lobes/gateway/_tier_request.py (PressureCache)`: Pressure is a whole-box swap/iowait signal sampled every 2s, applied only to tier aliases, and explicitly bypassed for proxied requests ('pressure describes THIS box's load'). It has no per-backend or per-peer view. This is the exact hook where 'overloaded -> send to a free machine' lands.
  - seeds: `c7`
- `s6` — `lobes/gateway/_config.py (PEER_ORIGIN_ENV, PEER_PROXY_ENV, PEER_API_KEY_ENV, _peer_origins) + lobes/runtime/_env.py + lobes/templates/fleet/{docker-compose.yml,env.example}`: Every peer family is scalar per prefix (one origin, one proxy flag, one key); compose passes each through as ${VAR:-} at docker-compose.yml:1659-1711; env.example:644-705 documents the singular block. tests/`test_gateway_config_proxy.py` pins `FEASIBLE_ENV`/peer-dict key parity, so a new family must join that check.
  - seeds: `c8`
- `s7` — `lobes/roles.py (annotate_peer_referrals, RoleInfo) + docs/colleague-stack.md capabilities schema + lobes/cli/_commands/capabilities.py _render_table`: `hosted_by` is a singular str, proxied a bool, endpoint singular; the annotator has exactly three states. RoleInfo carries model/context/quant/mtp from the STATIC catalog keyed by served id, not from a live probe. Both the live gateway and the CLI offline fallback render from this one function.
  - seeds: `c9`, `c10`
- `s8` — `lobes/profiles/ (shapes.py, shape_render.py, schema.py, builtin_shapes/*.toml, tests/test_shape_goldens.py)`: Shapes declare which roles a box HOSTS (hosts tuple -> <PREFIX>`_FEASIBLE`=false for dropped ones) and never a peer; grep for `PEER_ORIGIN` across the profiles package is empty. Goldens pin machine-as-brain byte-identity, untouched if replicas stay .env-only.
  - seeds: `c11`
- `s9` — `lobes/cli/_commands/route.py (cmd_route, _ROUTE_SYSTEM)`: route makes a live chat completion against the cheap tier to classify a task into a catalog gear and overlays lobes.minor.decide governance; it never touches RoutingTable, peer origins or `hosted_by`. Different concern, different module.
  - seeds: `c12`
- `s10` — `live probes: GET /capabilities on spark:8001, thor:8000, orin:8000; ssh orin docker inspect llamacpp-cortex; ~/.lobes/.env on all three (read-only)`: Two NVFP4 vLLM replicas with different drafters, one `Q4_K_M` llama.cpp replica with a single decode slot and a different served id. Capabilities misreport the Orin (catalog fallback) and the Spark (stale gateway env). This is issue #199 s2's exact case — the pool must not form silently from these three.
  - seeds: `c13`, `c14`
- `s11` — `lobes/gateway/server.py (_relay_streaming, _dial_owner)`: 2xx SSE is pumped chunk-by-chunk with no buffering; pre-dispatch refusal/timeout/5xx already maps to 503 `backend_unavailable`. Pre-dispatch retry across replicas is cheap to add; mid-stream replay is impossible by construction.
  - seeds: `c15`
- `s12` — `docs/specs/2026-07-16-proxy-lobes-pairwise-auth.md, docs/deliveries/2026-07-16-*.md, issue #128 (open), .devague/frames (no #199 frame existed)`: Phase 1 shipped the proxy substrate; phases 2-4 were parked as one line with #128 as the ticket. 'Runtime-aware routing' was never defined beyond that phrase, so #199 is the first definition, not a continuation of prior design text.
  - seeds: `c16`, `c17`
- `s13` — `docs/gateway-fleet.md#proxy-lobes (auth, markers, failure table) + docs/openai-api.md X-Lobes-Proxied-By`: Rules already documented per single peer: strip caller auth, replace with the peer's own inbound key, X-Lobes-Proxied-By on every proxied answer, 503 on peer down, 404 relayed terminally. A pool generalizes each rule per origin without inventing new ones.
  - seeds: `c18`
- `s14` — `docs/openai-api.md:701-708 + docs/gateway-fleet.md:600-613 (marker headers)`: X-Lobes-Proxied-By is attached only by the proxy branch; a local answer carries nothing. With a pool, honest placement needs a marker on BOTH paths.
  - seeds: `c19`
- `s15` — `tests/test_gateway_proxy.py, tests/test_proxy_integration.py, tests/test_peer_referral.py, tests/test_readiness_peer_probe.py, tests/test_gateway_config_proxy.py`: Fixture patterns are established (env-dict builders + `build_config` + `peer_specs_from_table`; two real loopback gateways for e2e). The recorded trap: a role added to `PEER_PROXY_ENV` but not to server.py's `_PEER_SERVED_NAME_ENV`/`_PEER_ROLE_HINT` is silently inert on the unwired shape.
  - seeds: `c20`
- `s16` — `lobes/explain/catalog.py (_GATEWAY:295, _SHAPES:812, _API:1158, _ROLES:1205) + docs/evidence/ proxy acceptance transcripts (2026-07-14/16/25/31)`: All five existing proxy/referral evidence files encode the single-peer contract; none exercises two candidates for one role. explain text at catalog.py:1291 hard-codes '`hosted_by`: <peer origin>'.
  - seeds: `c21`
- `s17` — `challenge pass / adjacent-systems lens: ../*/culture.yaml model pins`: Five sibling repos pin vllm-local/<raw id>; one pins unsloth/Qwen3.8-27B-NVFP4; zero pin cortex. The frame's claims all say model=cortex.
  - seeds: `c31`
- `s18` — `challenge pass / unstated-assumption lens: lobes/gateway/_config.py:150 (no self-origin exists)`: c19/h14 assume the gateway knows its own origin; nothing declares one and derivation is a recorded anti-pattern (#92). Needs a user decision (q5).
- `s19` — `challenge pass / counter-evidence lens: ~/.lobes/.env on spark and thor (PRIMARY_KV_CACHE_DTYPE, PRIMARY_HF_OVERRIDES, GATEWAY_API_KEY)`: Spark fp8 KV vs Thor auto; Thor retains YaRN factor 4; Thor sets NO inbound `GATEWAY_API_KEY` while Spark does. None of this was in the frame.
  - seeds: `c32`
- `s20` — `challenge pass / data-flow lens: vllm-primary /v1/models (thor, in-network), gateway GET /status, lobes/gateway/_config.py`: Live sources exist for id + context; quant/kv/parsers exist only in .env lane keys the gateway never receives. /status has health + running/waiting + busy but no fingerprint fields.
  - seeds: `c33`
- `s21` — `challenge pass / hidden-dependency lens: docker ps port publishing on spark (vllm-primary 8000/tcp unpublished), thor GET /status timing`: Only the gateway is reachable cross-box; /status is the load surface and already bounds per-backend probe time (server.py:1635).
  - seeds: `c34`
- `s22` — `challenge pass / concurrency lens: lobes/gateway/server.py:683-687,1025 (peer 429 relay), _pressure_policy.py`: Peer 429 relay exists; nothing in the frame said what the forwarder does with it. Bounding to one forward removes the oscillation hazard.
  - seeds: `c35`
- `s23` — `challenge pass / operations lens: ~/.lobes/docker-compose.yml on spark (8 PEER_API_KEY passthrough lines) and thor (9), eidetic record spark-proxy-golive-senses-orin-muse-thor-20260717`: Passthrough exists only for the singular keys; the plural family needs new lines on both deployed boxes.
  - seeds: `c36`
- `s24` — `challenge pass / lifecycle lens: h13 positional key rule vs the Thor's ungated inbound`: h13 as written would reject the live Thor pairing; the empty-slot rule reconciles c18 and h13.
- `s25` — `challenge pass / observability lens: docs/openai-api.md marker headers, c12 replica view`: The frame names WHICH replica served but not WHY; under a mixed policy (load + affinity + pressure) the why is what an operator debugs.
  - seeds: `c37`
- `s26` — `challenge pass / overlooked-actors lens: ../culture (vllm-local provider — NOT FOUND, unexamined)`: The affinity decision assumes a client can add a header; the client that matters was not located. Parked, non-blocking because absent-header behaviour is defined.
- `s27` — `challenge pass / scope-creep lens: lobes/gateway/_config.py nine-prefix vocabulary`: Reusing the prefix vocabulary makes every role poolable by construction; the honest claim is cortex-only validation.
  - seeds: `c38`
- `s28` — `challenge pass / failure-mode lens: GATEWAY_READ_TIMEOUT=600s / GATEWAY_CONNECT_TIMEOUT=5s (lobes/gateway/_config.py:1004-1005)`: Forwarded hops inherit the 600 s read timeout; Thor TTFT is ~300 ms short-prompt and a full 256K prefill stays well inside it. Clean pass; residual risk only if a peer wedges mid-prefill (then 503 after 600 s, same as today).
- `s29` — `challenge pass / security lens: auth stripping, X-Lobes-* headers, tailnet-only origins (docs/gateway-fleet.md proxy-lobes)`: Forwarding X-Lobes-Affinity to a peer leaks nothing; X-Lobes-Served-By exposes an internal origin exactly as X-Lobes-Proxied-By already does; the ungated Thor is a pre-existing posture, not a pool change. Clean pass.
- `s30` — `challenge pass / reversibility lens: h1 byte-identical no-pool + .env removal`: Rollback = delete the \*`_PEER_ORIGINS` line and recreate the gateway; no state, no migration. Clean pass. Version skew across gateway pins stays parked.

## Decisions

- The Orin is EXEMPT from the cortex pool for now: the first pool is the two NVFP4 vLLM replicas (Spark + Thor); the Orin's llama.cpp `Q4_K_M` cortex stays a separately-addressed candidate, and the operator may switch that box to a different lobe later
- The aligned front is EVERY box's own gateway: a caller asks any one endpoint (its local box) and the request is served by whichever compatible replica is free — the pool merges the request capacity of all servers, with no single designated dispatcher address
- Affinity key = a new X-Lobes-Affinity request header (the OpenAI 'user' field is rejected as confusing); no header means no affinity — placement is purely availability-driven. Affinity is always a preference, never ownership
- Replica config is per role, comma-separated, in the existing prefix vocabulary — <PREFIX>`_PEER_ORIGINS` with positionally matched <PREFIX>`_PEER_API_KEYS` — because the peer set differs per target box (the Spark's list names the Thor, the Thor's names the Spark); no per-node block, no box name ever hardcoded
- A box names itself in X-Lobes-Served-By via an operator-typed `GATEWAY_SELF_ORIGIN` in .env — the same never-derived rule as every peer origin (#92); with it unset, local answers carry the marker value 'local' and the pool still works

## Open parks

- [unknown_nonblocking] The load metric for a llama.cpp replica: llama-server /metrics exposes different fields than vLLM (and the Orin runs -np 1, so 'busy' is binary); whether to probe it or declare a static capacity=1 is undecided until the engine's metrics surface is read
- [unknown_nonblocking] Capacity/estimated-wait weights per replica (Spark ~46 tok/s code with DSpark, Thor ~26.8 tok/s with MTP, Orin ~8.5 tok/s) — whether declared per origin or learned from a latency EMA; the first cut can weight by declared decode speed but no measurement of pool behaviour exists yet (#108: declared until the three-box acceptance run)
- [unknown_nonblocking] Whether the fleet gateway image pin (`MODEL_GEAR_VERSION` 0.57.2 on Spark, 0.55.0 on Orin, 0.61.2 on Thor) must be aligned before a pool is honest — a pre-pool gateway on one box will not emit the local marker or the replicas list
- [unknown_nonblocking] Which client component would set X-Lobes-Affinity: the vllm-local provider code was not found under ../culture (no 'vllm-local' hit in its .py files) — it lives in the acp/colleague backend, unexamined this pass; until it can send a custom header, affinity is dormant and placement is availability-only (still correct per h16)
