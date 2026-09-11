# mesh-brain-join

> lobes mesh-brain: a home fleet of robots and boxes running lobes-cli forms one brain. Any machine with a key, or an operator approval (timed or permanent), joins the mesh at runtime, announces the roles, models and purpose it serves, and can draw on every lobe in the mesh even if it serves nothing itself. A configurable heartbeat (default once a minute) keeps it connected. Two machines serving the same role share one concurrency pool out of the box, so a second Spark joining Spark+Thor+Orin doubles the cortex pool or brings a new model.
> instruction: Run the cutover on Spark/Thor/Orin, then remove and re-add one member's join key; record the transcript under docs/evidence/ with the roster before/after.

## Audience

- Home-fleet operators and the robots/agents that consume lobes: a new box (e.g. a second DGX Spark) should join and contribute without hand-editing every peer's .env.
  - instruction: Evidence transcript includes one lobes init/serve join and one client request per role class (local, pooled, proxied, suffixed).

## Before → After

- Before: Joining today means hand-editing every member's .env with per-role peer origin, proxy and key blocks that name every other box (O(boxes x roles)), copying each peer's inbound key by hand, and recreating every gateway; a stale entry (the Orin still pooling cortex on the Thor, which no longer hosts it) is only visible from a live diff.
  - instruction: Capture with the redacting grep used in the scope survey and store under docs/evidence/ as the baseline.
- After: A new box runs 'lobes init --apply' with the mesh join key in its .env and 'lobes serve --apply'; within one heartbeat interval every other member lists its roles on /v1/models and /capabilities, same-role compatible lanes pool automatically, differing lanes appear as '{role}-{machine-name}', and roles the new box lacks are reachable on its own gateway by proxy. Removing the box (or the key) makes it vanish from every roster after missed heartbeats and the mesh keeps serving with what remains.
  - instruction: Same evidence transcript as c1; include lobes capabilities output from every member at each step.

## Why it matters

- Robots and home boxes should share cognitive load by being switched on, not by an operator threading peer blocks through N .env files: a second Spark should double the cortex pool or bring a new model the moment it joins, and losing a member should cost capacity, never correctness.
  - instruction: Evidence: git-style diff of every other member's ~/.lobes after the join shows zero changes.

## Requirements

- Mesh membership becomes RUNTIME-MUTABLE gateway state: a box admitted by key or by operator approval (timed or permanent) is added to the peer set of a RUNNING gateway with no restart. Today RoutingTable is frozen at process start (lobes/gateway/`_routing.py`:44 frozen dataclass; server.py:4230-4258 bakes it onto the handler) and ReplicaCache.`_peers` is a tuple fixed at `__init__` (`_replicas.py`:719) with no add/remove-peer method.
  - honesty: A member restarted or disconnected disappears from every other member's roster after a bounded number of missed heartbeats and requests for its roles either pool elsewhere or 404 `role_infeasible` naming no `hosted_by` — never hang.
  - honesty: The mutable routing state is a copy-on-write snapshot swapped atomically; the request path never takes a lock across a dial, and an in-flight request keeps the snapshot it started with.
- The heartbeat is one more background daemon thread started in serve() (lobes/gateway/server.py:4261-4328), following the PressureCache/ReadinessCache/ReplicaCache pattern (Event.wait(interval) loop, `_tier_request.py`:223-229), interval configurable with a 60 s default. It cannot live in the CLI: every verb is exec-once and no daemon/systemd/cron surface exists (lobes/cli/`__init__.py`:82-152; repo grep).
  - instruction: Unit test: fake opener that sleeps past the timeout on one peer; assert other peers' rosters refresh within one interval and `handle_post` latency is unaffected.
  - honesty: The heartbeat thread never blocks the request path and its interval is read from one env key with a 60 s default; a hung peer delays no other peer's probe.
- A joining box ANNOUNCES served roles, models and purpose by pushing the same payload GET /capabilities already publishes per role (RoleInfo: model/runtime/context/quant/responsibilities/`forbidden_responsibilities`/fingerprint/replicas, lobes/roles.py:471-568) to a NEW inbound mesh endpoint on the receiving gateway. Today the gateway has only `do_GET`/`do_POST` with no registration or heartbeat route (server.py:3486-3532, 3745-3811); discovery is pull-only (each box probes declared peers every 5 s, `_replicas.py`:150).
  - honesty: A keyless heartbeat or join request can never add a member to the roster: only an announcement bearing the join key enters it, so an unauthenticated box can detect the mesh and ask, but cannot inject fake capacity.
- A consumer-only box (serves nothing, draws from the mesh) ships as a built-in gateway-only shape. The schema already allows hosts=\[\] with no cardinality check (lobes/profiles/shapes.py:175, 206-284) and `shape_services` renders exactly ('gateway',) (`shape_render.py`:381-410); no built-in shape does this today, and `_routing.py`:432-436 still assumes 'a built table always has a primary backend'.
  - instruction: Ship a built-in gateway-only shape golden; boot it on one box (the Orin is a candidate) and request model=cortex through it; fix or relax the 'always has a primary backend' assumption in `_routing.py`:432-436.
  - honesty: A gateway-only member (hosts=\[\]) boots, joins, and serves every mesh role by proxy with a routing table that has no local primary backend.
- Two members serving the same role pool automatically: the pool set is derived from membership instead of the hand-typed <PREFIX>`_PEER_ORIGINS` list, while admission still passes `compare_fingerprints` (served id, quantization, `max_model_len`, runtime; unknown never pools — `_replicas.py`:196-201, 602-626) and dispatch keeps the in-flight reconciliation plus `estimated_wait`=(running+waiting)/weight ranking (`_replicas.py`:891-1013, `_selection.py`:203-217).
  - instruction: Test: two fake members announce cortex with equal fingerprints -> one pool; change one's quantization -> it is dropped from the pool and appears as a suffixed lane.
  - honesty: Pool membership is derived only from announcements that passed the fingerprint gate; a member whose lane fingerprint changes (lobes switch) leaves the pool within one interval.
- Every new gateway env key lands on the gateway service's explicit environment passthrough list (lobes/templates/fleet/docker-compose.yml:1784-2088; the silent-inert trap is documented at :2055-2058), in lobes/templates/fleet/env.example, and — for any new secret key name — in scripts/`scan_deployment_secrets.py`:37-38, which only knows the existing \*`_PEER_`\* vocabulary.
  - instruction: Extend the existing `_gateway_passthrough_check` test table with the new keys; CI secrets-scan fixture contains the join key name with a value and must fail.
  - honesty: Every new env key appears on the gateway compose passthrough, in env.example, and (for secrets) in the secrets scan; a key set in .env but missing from the list is caught by lobes doctor's passthrough check.
- Membership state is RUNTIME state, never a rendered .env key: the roster is in-memory soft state rebuilt from heartbeats, and only the approval ledger persists, in a git-ignored runtime file under the deployment dir bind-mounted into the gateway (the gateway service has no volumes today — docker-compose.yml gateway block — so this is its first). The deployment lock allowlist (lobes/runtime/`_lock.py`:44-49, 96-125) and tests/goldens (regen.py:1-24) stay untouched.
  - instruction: Test: `lock_keys`() does not contain any roster/ledger key; regen goldens unchanged; .gitignore covers the runtime file.
  - honesty: Roster and approval ledger live under the deployment dir in a git-ignored runtime file that the lock never captures and the goldens never render.
- Role-name conflict handling: a member's lane for role R joins the plain 'R' pool only on fingerprint agreement (served id, quantization, `max_model_len`, runtime — `compare_fingerprints`, `_replicas.py`:602-626); on disagreement it is exposed as 'R-<machine-name>', a new served alias listed on /v1/models and /capabilities of every member, addressable like any role and never silently substituted for plain 'R'. <machine-name> is the member's own declared name, never a hostname derived by a peer.
  - honesty: A suffixed lane '{role}-{machine}' appears on every member's /v1/models and /capabilities within one heartbeat interval and answers requests with X-Lobes-Proxied-By naming its host.
- Auto-wired proxying: for every role a member does not host, the gateway forwards model=<role> to a ready member that announced it (single hop, X-Lobes-Proxied-By preserved), replacing the <PREFIX>`_FEASIBLE`=false + `PEER_ORIGIN` + `PEER_PROXY` hand wiring. This applies to hand too — no never-proxied carve-out (`NEVER_PROXIED_BACKENDS` is already empty, `_config.py`:224).
  - instruction: Live: on the Thor (no cortex, no hand) request model=cortex and model=hand; both answer 200 with X-Lobes-Proxied-By naming the Spark.
  - honesty: Every role a member lacks is served by proxy from a ready member that announced it, chosen by the same replica selection as pooling, with hand included.
- Migration: the live Spark, Thor and Orin move off their hand-typed peer blocks onto the join key. Because the family is REPLACED, an .env still carrying <PREFIX>`_PEER_`\* keys must be refused or warned loudly by lobes doctor rather than silently ignored, and the join-key variable name is added to scripts/`scan_deployment_secrets.py` so it can never land in deployments/.
  - instruction: Add a doctor check that lists leftover \*`_PEER_`\* keys; run it on all three boxes and record 0 findings in the evidence transcript.
  - honesty: After cutover, lobes doctor on each of the three boxes reports zero \*`_PEER_`\* keys; an .env that still carries one produces a named finding rather than silence.
- Private roles (cheap, additive): a member's announcement carries an optional per-role private flag (one boolean beside the existing RoleInfo fields, lobes/roles.py:471-568); a private role is omitted from what the member broadcasts, so it never enters any peer's roster, pool, or auto-proxy, while staying addressable locally. No new endpoint, no new state beyond the announcement payload.
  - instruction: Test: member announces {cortex: private}; peer roster lacks it; local model=cortex still 200.
  - honesty: A role announced private is absent from every other member's roster, /v1/models and /capabilities, yet answers locally.
- Roster is SOFT, in-memory state rebuilt from heartbeats (no file, survives nothing, needs nothing); only the approval LEDGER (member name, approved-by, expiry) persists, in a git-ignored runtime file under the deployment dir bind-mounted into the gateway via a new compose volume — the first volume the gateway service has ever had (docker-compose.yml gateway block has none today).
  - instruction: Test: gateway starts with an empty ledger file path and an empty roster; after two fake heartbeats the roster has two members; restart -> roster empty, ledger intact.
  - honesty: After a gateway restart the roster is empty until the next heartbeats arrive, while the ledger file is unchanged; no roster content is ever written to disk.
- Bootstrap discovery: a joining box needs at least one address to detect the mesh. Introduce `LOBES_MESH_SEEDS` (comma-separated member origins, typed once on the joining box only); the roster fills in every other member after the first successful keyless detect. No mDNS/broadcast: the fleet lives on a Tailscale tailnet (tail0be7e0.ts.net) where multicast does not cross nodes.
  - instruction: Test: a box with one seed learns all N members from the seed's roster within one interval; a box with no seeds and no key behaves byte-identically to today.
  - honesty: A joining box with `LOBES_MESH_SEEDS` naming one live member learns every other member without any other box being edited; with no seeds and no key, nothing dials out.
- Member-to-member forwarding authenticates with the join key as bearer: each member's inbound gate accepts EITHER its own `GATEWAY_API_KEY` (clients) OR the mesh join key (members), replacing the per-box `PEER_API_KEY` copy (`_replicas.py`:283 attaches Authorization from the declared peer key today). The caller's own Authorization is still stripped before forwarding.
  - instruction: Test: forward from A to B carries Bearer <join key>; B with `GATEWAY_API_KEY` set accepts it; a forward with a client key or no key is 401.
  - honesty: No member ever sends a client's bearer to another member; every forwarded request carries the join key and nothing else in Authorization.
- Documentation and catalog follow the replacement in the same PR: docs/gateway-fleet.md, deployment-shapes.md, colleague-stack.md, openai-api.md, secret-rotation.md, env.example, lobes explain, and CLAUDE.md describe the join-key mesh and retire the \*`_PEER_`\* family (cite-don't-delete: the old sections move under a 'Retired' heading); deployments/jetson-agx-`thor__thor`-worker is re-captured because its verbatim compose carries the old passthrough list (no capture verb: call lobes.runtime.`_lock`.`capture_lock` directly).
  - instruction: doc-test-alignment pass: grep docs/ for `PEER_ORIGIN` outside 'Retired' headings returns nothing; secrets-scan passes on the re-captured Thor entry.
  - honesty: No doc outside a 'Retired' heading references \*`_PEER_ORIGIN`, \*`_PEER_PROXY`, \*`_PEER_API_KEY` or \*`_PEER_ORIGINS` after the change; the re-captured Thor catalog entry passes the secrets scan.
- Member identity: `LOBES_MESH_NAME`, a short operator-typed name required whenever the join key is set (no default from hostname — variation.py:1-13 records why hostname is never identity). It is the '{role}-{machine-name}' suffix source and the ledger key; a join whose name collides with a live, differently-originated member is refused with `mesh_name_conflict`.
  - instruction: Test: two fake members announce name 'spark' from different origins -> second refused; same origin re-announcing is an update.
  - honesty: A member with the join key but no `LOBES_MESH_NAME` refuses to start the mesh thread with a named error rather than deriving a name.
- Timed approval is enforced by IDENTITY, not by the key: every member's ledger (gossiped with the roster) records name -> expiry; an announcement from a lapsed or revoked name is refused by every member even when it carries the join key, and 'lobes mesh approve' on any one member propagates to all within one interval. Rotating the join key itself remains a fleet-wide restart, documented in docs/secret-rotation.md.
  - instruction: Test: approve name X for 1 s; after expiry X's announcement is refused with `mesh_approval_expired` on two members that never saw the approve call directly.
  - honesty: A lapsed name is refused by a member that never processed the approve call directly, proving the ledger gossips.
- Trust but verify: an announcement is a HINT. A receiver adds a role to a pool or to auto-proxy only after its own probe of the announcer's GET /capabilities confirms the announced fingerprint and ready — the live-probed-never-config-derived rule (#220) survives; a mismatch between announcement and probe marks the member 'unverified' in lobes mesh status and routes nothing to it.
  - instruction: Test: member announces cortex fingerprint F1 but its /capabilities probe returns F2 -> not pooled, status shows unverified.
  - honesty: A member whose probe disagrees with its announcement receives zero forwarded requests and shows 'unverified' in lobes mesh status.
- Announcements carry a schema version; a receiver ignores unknown fields from a newer member and refuses an incompatible major with `mesh_schema_incompatible` (named in lobes mesh status), so a fleet upgraded one box at a time never silently drops members.
  - instruction: Test: announcement with version+1 and an extra field is accepted; version with a different major is refused and visible in status.
  - honesty: A member two minor versions ahead, sending one unknown field, is admitted; a different major is refused with a named error visible in status.
- Fingerprint changes re-announce immediately: lobes switch / lobes up / a lane going unhealthy trigger an announcement at once, not at the next 60 s tick; receivers re-verify (F8) and move the lane between the plain pool and its suffixed name; in-flight requests complete on the lane they started on.
  - instruction: Test: fake member changes quantization -> peers reflect the suffixed lane on the next probe, well under one heartbeat interval.
  - honesty: The interval between lobes switch completing on member A and member B reflecting the new lane is bounded by B's probe refresh (5 s), not by the heartbeat interval.
- CLI surface: 'lobes mesh status' (members, last-heartbeat age, approval expiry, verified/unverified, roles incl. suffixed and private), 'lobes mesh request' (keyless ask from a box without the key), 'lobes mesh approve <name> \[--for <duration>\]' and 'lobes mesh revoke <name>' — the write verbs dry-run by default with --apply, registered like the fleet noun (lobes/cli/`_commands`/fleet.py:169-213). lobes capabilities renders members and suffixed lanes; every mesh answer carries X-Lobes-Mesh-Member naming the serving member.
  - instruction: Golden test of lobes mesh status output against a fake roster; --apply convention test as for fleet up.
  - honesty: lobes mesh approve without --apply changes nothing and prints the plan; every mesh answer carries X-Lobes-Mesh-Member.
- Raw served-id addressing (deployed consumers pin ids, e.g. culture.yaml model: vllm-local/<id>): a raw id resolves to the local lane if hosted, else to the plain pool for that id; a raw id that exists only on suffixed (fingerprint-divergent) lanes 404s `role_infeasible` with `hosted_by` listing the suffixed names rather than picking one.
  - instruction: Test: raw id hosted only as cortex-thor and cortex-orin -> 404 listing both; hosted locally -> local.
  - honesty: A raw id hosted on divergent lanes only is never answered by one of them; the 404 body lists every suffixed name.
- The keyless detect and join-request endpoints are bounded: at most N pending join requests (default 8) with a TTL, one request per origin, rate-limited, and rejections logged through the collapsed `_authlog` pattern so a scanner cannot flood logs or memory. Detect returns only 'a mesh exists, name of this member, schema version' — never the roster.
  - instruction: Test: 100 join requests from one fake origin -> one pending entry; detect body contains no member list.
  - honesty: Under a flood of keyless join requests memory stays bounded at N pending entries and the log shows one collapsed line, not one per request.
- Cutover rollback: before touching a box, its ~/.lobes is copied to ~/.lobes.pre-mesh-<UTC timestamp> and the previous `MODEL_GEAR_VERSION` recorded; rollback = copy back + recreate the gateway with the four -f compose files (Spark) / env -u `GATEWAY_API_KEY` (Thor). The evidence transcript names the backup path per box.
  - instruction: Evidence: ls of the backup dir on all three boxes before the first change.
  - honesty: The backup directory exists on every box before its .env is modified, and the transcript names it.
- Containment of a misbehaving member: announced capacity passes the existing `resolve_capacity` clamp (`CAPACITY_CLAMP_MAX`=64, `_replicas.py`:455-497); a flapping member (joins/leaves > 3 times in 10 intervals) is held out for one interval with reason `mesh_flapping` in status; nothing a member announces can remove another member.
  - instruction: Test: clamp applied to a fake announcement of capacity 10^6; flapping fake member held out.
  - honesty: A fake announcement with capacity 10^6 ranks with weight <= 64; a flapping member is held out for one interval and shown as `mesh_flapping`.

## Honesty conditions

- The announcement holds on real hardware: a fourth box (or one of the three re-joined) forms the mesh by key alone, with no peer block typed on any other member.
- Both audiences are exercised: an operator joins a box with one env var, and a robot client (Qwen Code or curl with model=<role>) reaches every mesh role through its local gateway.
- With no join key set, no heartbeat thread starts and no mesh route answers anything but 404; the four byte-identical tests keep passing unchanged.
- A proxied request that would need a second hop is refused 508 exactly as today, including when the roster (not a typed origin) chose the peer.
- No request is ever answered by a different served id than the one addressed; a suffixed lane is never used to satisfy a plain-role request.
- X-Lobes-Route-Reason values remain the closed set in `_selection.py`:110-125; mesh provenance travels in new X-Lobes-Mesh-\* headers.
- render.py, `shape_render.py` and schema.py contain no mesh, roster, peer or self-origin logic after the change.
- No committed file under deployments/, docs/evidence/ or tests/goldens/ contains a join key value; the scan fails on a fixture that does.
- Delivering the join key to an approved requestor travels over the same tailnet HTTP the gateways already trust (no TLS at this layer per docs/gateway-fleet.md); the approval flow must state this rather than imply a secure channel.
- 'Approve' on ANY member is enough: the approval and the delivered key reach the requestor once, and every other member learns the approval by gossip — the operator never repeats it per box.
- The keyless detect endpoint never returns the roster or any member origin; the keyed roster endpoint returns it in full.
- Observed end-to-end on the live fleet, with the roster listing captured before and after the join and after the removal.
- The pre-cutover state is captured verbatim before any change (the three .env peer blocks, redacted keys, and the stale Orin pool entry).
- Adding a member never requires editing any other member's files.
- Each numbered signal is measured on the live fleet, with the number recorded, and any that fails is recorded as failed rather than dropped.
- GET /v1/realtime on a member that lacks stt returns 404 `role_infeasible` even while another member announces stt.
- doctor reports a named finding when the shell's `LOBES_MESH_KEY` differs from the .env value.

## Success signals

- Measured on the live Spark+Thor+Orin fleet after cutover: (1) a member joining with the key is listed by all other members within <= 2 heartbeat intervals (<= 120 s at the 60 s default); (2) zero \*`_PEER_`\* keys remain in any member's .env and lobes doctor reports 0 findings; (3) with 2 members serving a compatible cortex, 4 concurrent requests to one member's gateway are served by both (X-Lobes-Served-By and X-Lobes-Proxied-By both observed); (4) stopping one member's gateway: the others answer 200 for the roles they still hold within <= 2 intervals, and 404 `role_infeasible` (never a hang > 10 s) for a role only it had; (5) with no join key set, the gateway's responses are byte-identical to 0.75.x (existing byte-identical tests still pass).
  - instruction: Evidence transcript has one section per signal (1)-(5) with the measured value and PASS/FAIL.

## Scope / boundaries

- Mesh-brain is a NEW opt-in axis: with nothing declared, every response stays byte-identical to today and no heartbeat runs. Pinned by tests/`test_gateway_pool.py`:269 (`test_no_pool_declared_is_byte_identical`), tests/`test_gateway_config_proxy.py`:402, tests/`test_gateway_proxy.py`:674 and :718.
  - instruction: Run tests/`test_gateway_pool.py`::`test_no_pool_declared_is_byte_identical` and the three siblings; add one asserting no mesh thread is started when the key is unset.
- Forwarding stays single-hop: a request arriving with X-Lobes-Proxied that would depart again is refused 508 `proxy_loop` (server.py:762-780, 874). Membership never makes forwarding transitive.
  - instruction: Test: member A lacks role R, roster says B hosts it, B also lacks it and its roster points to C; A->B returns 508 `proxy_loop`, never reaches C.
- No cross-role or cross-model substitution (issue #91; `order_backends` returns one owner, never a chain, `_routing.py`:398-437) and no pooling without fingerprint agreement — a self-declared 'I serve cortex' is admitted to a pool only when its live fingerprint matches.
  - instruction: Test: model=cortex with only cortex-thor available (fingerprint mismatch) returns 404 `role_infeasible` with `hosted_by` empty, not a cortex-thor answer.
- The X-Lobes-Route-Reason vocabulary is closed (lobes/gateway/`_selection.py`:110-125; burned once by a meaning change). Mesh routing information travels in NEW headers, following the X-Lobes-Route-Load precedent (server.py:846-855).
  - instruction: Test asserting the reason vocabulary set is unchanged; new header names documented in docs/openai-api.md.
- lobes/profiles/render.py, `shape_render.py` and schema.py never render peer identity, capacity or self-origin — none of PEER/`MAX_ACTIVE`/`SELF_ORIGIN` exist in those modules today and the #92 rule is restated at 30+ sites (e.g. `_config.py`:264, 450, 608; orin.toml:525-533). Membership feeds the gateway at runtime, not the renderer.
  - instruction: grep gate in tests: none of MESH/PEER/`SELF_ORIGIN` symbols appear in lobes/profiles/; shape goldens unchanged except the new gateway-only shape.
- No mesh credential ever enters deployment.lock.toml or deployments/: keys stay in .env / .secrets.env / runtime state, and the secrets-scan CI job (.github/workflows/tests.yml:83-114) learns any new key name.
  - instruction: secrets-scan fixture test plus a grep of the evidence transcript before commit.
- Auto-wired proxying covers POST lanes only; GET /v1/realtime (WebSocket tunnel) stays local-only exactly as today — the proxy-lobes forwarder is POST-only (lobes/gateway/`_realtime.py`:19, docs/realtime-pipeline.md:688).
  - instruction: Test: a member without stt answers 404 `role_infeasible` on /v1/realtime even when another member announces stt.
- The join key is read from .env / .secrets.env via the compose passthrough only; lobes doctor warns when the invoking shell exports a different value (the Thor ~/.bashrc trap, issue #209), since compose interpolates shell env ahead of .env.
  - instruction: doctor test: shell env `LOBES_MESH_KEY` != .env value -> named finding.

## Non-goals

- Not reusing or depending on culture's mesh link/trust model. Culture's trust is a static full/restricted flag on a manually configured server link with a shared plaintext password, N^2 links, no timed trust and no self-enrollment (`culture_core`/`mesh_config.py`:28; docs/shared/concepts/federation.md:30-41, 58-62); its PRESENCE heartbeat is per-agent liveness, not membership (docs/resident-presence.md:63-92). lobes' mesh stays HTTP gateway-to-gateway with zero code dependency on culture; culture's model is a pattern to mirror only.
- No throughput claim for a heterogeneous pool. Pooling balances by slot count (<PREFIX>`_MAX_ACTIVE` is concurrency, not service rate); the Orin peer-only pool measured 51% slower at 4 concurrent than pinning on a 4.4x speed-mismatched pair (docs/gateway-fleet.md:860-880, issue #232). PeerReplica has a weight field (`_replicas.py`:440) but the only constructor call never sets it (server.py:4122). A service-rate weight is a separate piece of work.

## Assumptions

- Live baseline 2026-09-11 (ssh, read-only): Spark hosts cortex+rerank and proxies worker/embed->Thor, associate->Orin; Thor hosts worker/embed/rerank and proxies cortex/hand->Spark; Orin hosts associate and pools cortex over Spark+Thor — but the Thor no longer hosts cortex, so that replica reads compatible:false ('quantization: unknown; runtime: unknown'): a static peer list already went stale. Orin .env pins `MODEL_GEAR_VERSION`=0.63.1 while its gateway serves 0.70.0, lacks `GATEWAY_SELF_ORIGIN`, and neither Thor nor Orin has the lobes CLI on PATH. All three are on the tailnet tail0be7e0.ts.net (Spark :8001, Thor :8000, Orin :8000).
- Thor and Orin are available for live adaptation and validation via ssh thor@thor and ssh orin@orin; on the Thor every compose invocation must run under env -u `GATEWAY_API_KEY` because ~/.bashrc exports the key and would arm its inbound gate.
- A joining box's origin is still typed by an operator ONCE, on that box, as its own `GATEWAY_SELF_ORIGIN` (lobes/gateway/`_config.py`:604-615) and carried in its announcement; the receiving box never derives, resolves or guesses a peer URL. This keeps the #92 lesson ('never fabricate an absolute URL') while removing the O(boxes x roles) hand-copying of peer blocks.

## Scope exploration

- `s1` — `lobes/gateway/_config.py + _routing.py + server.py (peer declaration, frozen RoutingTable)`: All peer/replica declaration is static-at-boot env parsed once by `build_config` (`_config.py`:1056-1370); RoutingTable is frozen (`_routing.py`:44) and baked onto the handler (server.py:4230-4258); no add/remove-peer path anywhere. Dynamic join needs a runtime-mutable membership layer feeding the existing caches.
  - seeds: `c3`, `q1` (question, resolved), `q2` (question, resolved)
- `s2` — `lobes/gateway/_replicas.py + _selection.py (ReplicaCache, capacity, fingerprint)`: Pull-only 5 s probe of declared peers' /status + /capabilities (:150, :187-192); compatibility via `compare_fingerprints` (:602-626); in-flight reconciliation (:891-1013); `select_replica` is a pure ranking function (`estimated_wait`=(running+waiting)/weight, `_selection.py`:203-217) and its reason vocabulary is closed (:110-125). `_peers` frozen at `__init__` (:719). PeerReplica.weight exists (:440) but server.py:4122 never sets it.
  - seeds: `c7`, `c16`, `c20`
- `s3` — `lobes/gateway/server.py routes + serve() threads (do_GET/do_POST, /capabilities, /status, auth split)`: Only /health, /status, /capabilities are keyless; every /v1/\* GET and every POST is bearer-gated (server.py:195-238, 3486-3532, 3745-3811). No registration/heartbeat route exists. serve() already runs Pressure/Readiness/Replica daemon threads (server.py:4261-4328), so a heartbeat thread fits the pattern. RoleInfo already carries responsibilities/`forbidden_responsibilities` as the 'purpose' field (lobes/roles.py:471-568).
  - seeds: `c4`, `c5`, `q4` (question, resolved)
- `s4` — `lobes/gateway/_pressure_policy.py + concurrency accounting`: Outcomes are dispatch / forward / 429 shed; no gateway-level wait queue — only vLLM's engine queue (:98-111; server.py:1513-1521). <PREFIX>`_MAX_ACTIVE` is a measured knee written only by lobes calibrate --apply (calibrate.py:347-349), unrelated to --max-num-seqs (`WORKER_MAX_NUM_SEQS`=1 on the Thor is an OOM cap the gateway never reads).
  - seeds: `q5` (question, resolved)
- `s5` — `lobes/profiles/render.py, shape_render.py, schema.py, builtin_shapes/*.toml`: No PEER/`MAX_ACTIVE`/`SELF_ORIGIN` symbol exists in the renderer; #92 restated at 30+ sites. hosts=\[\] is valid (shapes.py:175, 206-284) and renders ('gateway',) (`shape_render.py`:381-410) but no built-in gateway-only shape exists; `_routing.py`:432-436 assumes a primary backend.
  - seeds: `c6`, `c17`
- `s6` — `lobes/templates/fleet/docker-compose.yml gateway environment passthrough + env.example`: Gateway uses an explicit environment: list, never `env_file` (:1756-1764); every \*`_PEER_ORIGIN`(S)/`_PEER_PROXY`/`_PEER_API_KEY`(S)/`_MAX_ACTIVE`/`GATEWAY_SELF_ORIGIN` is listed (:1932-2088) and an unlisted key is silently inert (:2055-2058). env.example documents the manual join steps (:763-937).
  - seeds: `c8`
- `s7` — `lobes/cli (verb registry, capabilities, status, doctor, calibrate, runtime/_env.py)`: Every verb is exec-once; no daemon home. Write verbs are dry-run by default with --apply. `set_env` rewrites existing lines (`_env.py`:78-108) while doctor --fix hand-rolls append-only (doctor.py:927-935) — an auto-join must never clobber operator-typed peer keys. lobes capabilities renders from the gateway payload with an offline .env fallback (capabilities.py:165-283). A 'lobes mesh' verb family registers like fleet (fleet.py:169-213).
  - seeds: `c4`, `q3` (question, resolved)
- `s8` — `tests/, tests/goldens/, lobes/runtime/_lock.py, deployments/, scripts/scan_deployment_secrets.py, CI`: Byte-identical-with-no-config is pinned by four tests; peers are faked with env dicts + injected openers (`test_gateway_proxy.py`:198-211, `test_gateway_peer_advert.py`:33-49). Goldens capture only profile->env; lock allowlist excludes \*`_PEER_`\* (`_lock.py`:44-49); no runtime state dir exists; the secrets scan knows only the existing key names (:37-38). CI: test, lint (afi rubric), secrets-scan, site-build, version-check.
  - seeds: `c13`, `c18`, `c9`
- `s9` — `docs/gateway-fleet.md, deployment-shapes.md, colleague-stack.md, openai-api.md, secret-rotation.md`: Documented contracts the idea must extend, not break: #92 operator-typed origins; zero-config byte-identical; single-hop; O(machines) key copy; tailnet-only transport with no TLS at this layer; hand never proxied; feasible:false means not hosted; proxied ready/context come from the peer's advert (#220); peers must agree on fingerprint. A shared cross-box concurrency pool is NOT documented as existing; joining today is a manual per-box .env edit with hand-copied keys.
  - seeds: `c14`, `c15`, `c12`
- `s10` — `docs/specs 2026-07-14 (#112), 2026-08-25 (#199), 2026-08-27 (capacity), 2026-08-30 (peer-only pools); issues #92 #128 #199 #209 #215 #232`: Both capacity-relative routing (#221, 0.67.0) and peer-only pools (#233, 0.70.0) shipped, so 'draw from lobes while serving nothing' already exists for operator-typed peers. Recorded non-goals the idea reverses: replica origins never discovered (#199 spec :82), \*`_PEER_`\* excluded from the lock (peer-only spec s16), capacity is pulled never pushed (2026-08-27 Decisions). Nothing in the tracker covers dynamic join, approval, heartbeat or a consumer-only shape; #232 (service-rate weight) and #215 (raw-id pressure gate) are the adjacent open gaps.
  - seeds: `q2` (question, resolved), `c20`
- `s11` — `Live fleet: Spark (local), Thor (ssh thor@thor), Orin (ssh orin@orin) ~/.lobes/.env, docker ps, /health, /status, /capabilities`: Three-box tailnet mesh, every peer relationship hand-typed per role per box. Orin's cortex pool still lists the Thor, which no longer hosts cortex (compatible:false) — a static list already stale. Orin pins 0.63.1 but serves 0.70.0, lacks `GATEWAY_SELF_ORIGIN`; lobes CLI absent on Thor and Orin. /capabilities answered 200 without a key on all three, which is the keyless design, not a gate failure.
  - seeds: `c10`, `c11`
- `s12` — `../culture (culture_core/mesh_config.py, cli/server.py, docs/shared/concepts/federation.md, docs/resident-presence.md)`: Culture's mesh: server-to-server links with a shared password and a static full/restricted trust flag, N^2 links, no timed trust, no self-enrollment; PRESENCE heartbeat (30 s default) is per-agent liveness. No 'mesh brain' concept, no capability broadcast, and no code path in culture knows about lobes gateways (vllm-local is static host:port + key on the lobes side).
  - seeds: `c19`
- `s13` — `user decisions q1-q6 (2026-09-11)`: Every member is the brain (no hub); dynamic membership REPLACES the typed peer family; one mesh-wide join key with request/approve delivery; roster keyed, detect/join keyless; queue = today's load balancing; auto-wire proxying with '{role}-{machine-name}' suffix on fingerprint conflict.
  - seeds: `c21`, `c22`, `c23`, `c24`, `c25`, `c26`, `c27`, `c28`, `c29`
- `s14` — `challenge pass / adjacent-systems lens: lobes/templates/fleet/docker-compose.yml gateway block, _realtime.py:19, deployments/jetson-agx-thor__thor-worker, docs/*`: Gateway has NO volumes (contradicts c9's file roster); realtime forwarder is POST-only; the Thor catalog entry carries the old passthrough verbatim; six docs plus CLAUDE.md describe the retired family.
  - seeds: `c37`, `c40`, `c41`
- `s15` — `challenge pass / hidden-dependency lens: tailnet transport (tail0be7e0.ts.net), no multicast; _replicas.py:283 outbound bearer`: Every member is the brain, but the FIRST contact needs a typed seed — no broadcast on a tailnet; member-to-member auth must move from per-box peer keys to the join key.
  - seeds: `c38`, `c39`
- `s16` — `challenge pass / unstated-assumptions lens: lobes/variation.py:1-13 (hostname never identity), announcement trust, version skew`: No member-name concept exists; suffix and ledger need one. Announced fingerprints are self-declared — the #220 live-probe rule must survive. Mixed lobes versions across members were unaddressed.
  - seeds: `c42`, `c43`, `c44`, `c45`
- `s17` — `challenge pass / lifecycle-and-actors lens: lobes switch/up, lobes/cli verb registry, culture.yaml raw-id consumers`: A fingerprint change mid-run, the absent CLI verb family, and raw-id addressing on divergent lanes were all unspecified.
  - seeds: `c46`, `c47`, `c48`
- `s18` — `challenge pass / security lens: keyless endpoints, _authlog.py, issue #209 shell-env trap`: Keyless detect/join are unauthenticated surfaces needing bounds; the join key inherits the Thor ~/.bashrc interpolation trap.
  - seeds: `c49`, `c50`
- `s19` — `challenge pass / concurrency-and-distributed-state lens: frozen RoutingTable (_routing.py:44), roster agreement`: Copy-on-write snapshot swap needed; no consensus by design, staleness bounded by missed-heartbeat count.
  - seeds: `c51`
- `s20` — `challenge pass / observability-rollback-containment lens: capacity clamp _replicas.py:455-497, cutover procedure`: Rollback path and backup naming were missing from the cutover decision; capacity clamp already contains bogus announcements; flapping needed a rule.
  - seeds: `c52`, `c53`
- `s21` — `challenge pass / hardware lens: Spark GB10, Thor sm_110, Orin sm_87`: Clean: no lane budget, kernel, or engine change is implied — the mesh layer touches only the gateway container. Residual: a gateway-only shape on the Orin has never been booted (#108).
- `s22` — `challenge pass / data-loss lens: .env, ~/.lobes, eidetic stores`: Clean for data: the cutover rewrites .env only, with a full-dir backup required (F16); downtime is approved. Residual: consumers mid-request during the gateway recreate get connection errors, not corruption.

## Decisions

- Topology: EVERY member is the mesh brain. Each box holds its own replicated roster view; any member can be disconnected or restarted and the rest of the mesh continues with fewer resources. No hub, no single point of failure.
- Dynamic membership REPLACES the operator-typed <PREFIX>`_PEER_ORIGIN`(S)/`_PEER_PROXY`/`_PEER_API_KEY`(S) family. Each machine is responsible for its own declaration only (its self origin, what it serves) and needs no cross-box coordination, yet enjoys every other member's offers. This supersedes the recorded 'never discovered' decisions of #199 (spec :82) and the peer-only-pools spec s16.
- Credential: ONE mesh-wide shared join key, an environment variable. A box that lacks it may REQUEST to join; an operator on any member approves and the key is delivered to the requestor.
- Endpoint auth: the roster (who serves what) is readable only with the join key; the heartbeat/detect and join-request endpoints are keyless so a new box can discover that a mesh exists and ask to join.
- 'Queue' means today's load balancing: same-role lanes on different members share the load through the existing dispatch / forward / shed pool mechanics. No gateway-level wait queue is introduced.
- Auto-wire proxying for every role a member lacks. When two members serve the same role with DIFFERENT config or capabilities (fingerprint mismatch), the conflicting lanes are exposed under the suffixed name '{role}-{machine-name}' so every role name means exactly one thing; matching fingerprints pool under the plain role name.
- Cutover approved: the Spark, Thor and Orin migrate off the typed peer family onto the join key in one move, implemented over ssh thor@thor / ssh orin@orin, with downtime as needed.
- hand is proxied like any other role a member lacks; no never-proxied carve-out. Whether a member can mark a role PRIVATE (served locally, withheld from the mesh) is a follow-up unless it is cheap.
- No consensus protocol: each member routes on its OWN roster view; two members may briefly disagree about a third. A member is dropped from a view after 3 missed heartbeats (default, configurable), so staleness is bounded at ~3 intervals and the mesh never blocks waiting for agreement.
  - instruction: Test: stop a fake member's heartbeats -> removed after the 3rd missed tick, not before; requests for its roles pool elsewhere or 404.

## Hard questions

- What is 'the mesh brain' a box connects to: ONE designated hub box holding the roster (hub-and-spoke), or EVERY member (gossip / full mesh)? Hub is simpler and matches 'auto-connects to the mesh brain'; full mesh matches 'any mesh machine can draw' without a single point of failure. The design differs materially. (resolved: decided by the user 2026-09-11 -> c21)
- Does dynamic membership REPLACE or SUPPLEMENT the operator-typed <PREFIX>`_PEER_ORIGIN`(S) family? Supplement (static declarations remain a supported fallback and win on conflict) preserves every existing deployment; replace simplifies the code but reverses recorded decisions in docs/specs/2026-08-25-cortex-replica-pool-199.md:82 and docs/specs/2026-08-30-peer-only-replica-pools.md s16. (resolved: decided by the user 2026-09-11 -> c22)
- Approval model: who approves, where, and how does a TIMED approval expire? Proposed: 'lobes mesh approve <box> \[--for 24h\]' on the brain box writes the ledger; the joining box's heartbeat is refused with a named error once the grant lapses. And does 'with a key' mean a mesh-wide shared join key (one secret admits any box, permanent) as distinct from today's per-box inbound `GATEWAY_API_KEY` copy (O(machines))? (resolved: decided by the user 2026-09-11 -> c23)
- probe (resolved: stray probe entry made while checking the JSON output shape; ignore)
- Auth posture of the new join/heartbeat endpoint: /capabilities and /status are keyless by design (server.py:195-238) while every /v1/\* and POST is bearer-gated. Should announcements be accepted only with the join key or an approved identity (recommended), and should the roster itself be readable keylessly like /capabilities? (resolved: decided by the user 2026-09-11 -> c24)
- 'Queue mechanism works out of the box' — today the outcomes are dispatch, forward to a less-loaded compatible replica, or 429 shed; only vLLM's own engine queue waits (`_pressure_policy.py`:98-111, server.py:1513-1521). Is auto-formed pooling with today's dispatch/forward/shed enough, or is a gateway-level bounded WAIT (hold the request until a slot frees anywhere in the pool) in scope? (resolved: decided by the user 2026-09-11 -> c25)
- A member that serves a role NOBODY else serves (a new model, e.g. the second Spark brings senses) must be reachable from every other member. Today that is the singular proxy path (feasible:false + `PEER_ORIGIN` + `PEER_PROXY`). Should membership auto-wire proxying for roles a box lacks, or only pooling for roles it already hosts? (resolved: decided by the user 2026-09-11 -> c26)
- contradiction with The gateway service in lobes/templates/fleet/docker-compose.yml declares NO volumes (probe: awk over the gateway block finds no volumes: key), so a roster file 'under the deployment dir' is invisible to the container as written.? (resolved: Resolved by the user confirming amended c9 (2026-09-11): roster in-memory, approval ledger persisted via a new gateway bind mount.)

## Open parks

- [unknown_nonblocking] Service-rate weighting so a heterogeneous pool does not slow down under load (issue #232): PeerReplica.weight exists but no channel populates it. Not needed to form the pool; needed before any throughput claim.
- [unknown_nonblocking] Join-key rotation without a fleet-wide restart (e.g. accepting old+new for one interval).
- [follow_up] Fleet hygiene before live validation: Orin `MODEL_GEAR_VERSION` pin drift (0.63.1 vs 0.70.0 served), missing `GATEWAY_SELF_ORIGIN` on Orin, lobes CLI absent on Thor and Orin, stale Thor entry in Orin's cortex pool.
- [follow_up] Raw-id requests under local pressure are not forwarded (#215, pressure gate is tier-alias-only); an auto-formed pool inherits this divergence until #215 lands. (Re-parked: v4 was resolved by mistake in place of the hygiene park v3.)
- [follow_up] LAN-local zero-config discovery (mDNS) for robots on the same Wi-Fi with no tailnet — seeds suffice for this fleet; revisit when a box joins without a tailnet address.

## Resolved vagueness

- [unknown_nonblocking] Whether hub-held membership state belongs in the deployment lock / variation catalog at all, or stays an explicit runtime-only exclusion like \*`_PEER_`\* keys today. — resolved: Moot: the peer family is replaced, so no membership state is a rendered key; roster stays runtime-only and outside the lock.
- [follow_up] Raw-id requests under local pressure are not forwarded (#215, pressure gate is tier-alias-only); an auto-formed pool inherits this divergence until #215 lands. — resolved: superseded by cutover decision: hygiene folds into the migration work
