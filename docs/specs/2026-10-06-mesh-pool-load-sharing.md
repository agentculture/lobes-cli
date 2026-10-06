# mesh pool load sharing

> Every verified mesh member of a role is addressable by name from ANY mesh gateway: `model=cortex-spark2` always lands on spark2's engine and `cortex-spark` on spark's, never balanced — while plain `cortex` stays the pool
> instruction: offline test: forward cortex-spark2 from a fake spark gateway; assert X-Lobes-Mesh-Member hop header on the outbound request and that the destination's `_pool_selection` returns sole-ready

## Audience

- Operators and clients (Qwen Code first) that want to pick a specific box for a role, and the mesh gateways that must route that choice.
  - instruction: after rollout, ~/.qwen/settings.json has cortex-spark and cortex-spark2 entries both at <http://localhost:8001/v1>; qwen -m each answers

## Before → After

- Before: A member is addressable by name only when its fingerprint DISAGREES (`find_suffixed_lane` walks placement.suffixed only, lobes/gateway/`_mesh_routing.py`:857). spark and spark2 agree, so neither has a name; today the only way to pick spark2 is pointing the client at spark2's own gateway with spark2's own key (Option 1, set up and verified 2026-10-06: qwen -m main moved spark2's `request_success_total` 2181->2184).
  - instruction: cite this session's Option 1 check (spark2 `request_success_total` 2180->2181 via curl, 2181->2184 via qwen -m main) and today's live pinning observation in memory mesh-pool-never-spreads-load
- After: From any mesh gateway, `{role}-{member}` resolves for every verified member of that role, agreeing or not; the box's OWN name is served by its local lane with no forward. One base URL and one key in Qwen Code reach every box.
  - instruction: offline test matrix over {hosting box, non-hosting box} x {self name, agreeing peer, disagreeing peer, unknown member}

## Why it matters

- A caller can pin work to a box (warm cache, isolation, A/B) without per-box URLs and keys, and the pin survives a future pooling fix because a forward carries the hop marker and the destination serves a marked arrival locally (sole-ready, server.py `_pool_selection`).
  - instruction: test enables a populated replica pool on the destination and asserts a hop-marked member-lane arrival is still served locally

## Requirements

- `find_suffixed_lane` (or a sibling member-lane resolver) matches `{role}-{member}` for EVERY verified member of the role, not only fingerprint-disagreeing ones; the forward path at server.py:3253-3300 is reused unchanged (`served_name` rewritten to the destination's backend name, join-key signed, X-Lobes-Proxied-By + mesh markers).
  - instruction: unit test on `find_suffixed_lane`/member resolver + server forward test asserting body model == primary and Authorization == join key
  - honesty: Agreeing members resolve by name and the outbound body carries the destination's backend name, not the alias
- `{role}-{self}` (self = this box's `LOBES_MESH_NAME`) on a box that hosts the role is served by the local lane: no pool selection, no forward, X-Lobes-Served-By set. On a box that does not host the role it 404s `role_infeasible`, never forwards to some other member.
  - instruction: test with an injected `open_upstream` that fails if called; assert 200 + X-Lobes-Served-By; separate test asserts 404 `role_infeasible`
  - honesty: Self-name is served locally with no outbound connection; a non-hosting box 404s `role_infeasible` for its own name
- /capabilities gains an additive per-role `member_lanes` list naming every addressable `{role}-{member}` (self included when hosted); `suffixed_lanes` keeps its meaning (disagreeing members only). With mesh disabled the payload stays byte-identical.
  - instruction: golden/byte-identity test on /capabilities with mesh disabled; mesh-enabled test lists cortex-spark and cortex-spark2
  - honesty: `member_lanes` is additive and absent with mesh disabled
- GET /v1/models lists every addressable `{role}-{member}` id (self included when hosted) alongside the existing entries; with mesh disabled the listing is byte-identical.
  - instruction: test `list_models_payload` with a two-member snapshot + byte-identity test with `mesh_snapshot`=None
  - honesty: /v1/models lists cortex-spark and cortex-spark2 on a mesh member and is byte-identical with mesh disabled
- `{role}-{member}` for a member that announced the role but is not yet probed returns 503 `role_unverified` with Retry-After: 5, error.`hosted_by` and the X-Lobes-Mesh-Member/-Unverified headers, reusing the plain-role boot-window body.
  - instruction: test with MemberInfo.probed=False: assert 503, Retry-After: 5, `hosted_by`, and that `open_upstream` is not called
  - honesty: An unprobed member's name gets 503 `role_unverified`, never 404 or a forward
- The self lane (`{role}-{self}`) applies THIS box's pressure policy exactly like the plain role (429 `server_busy` + Retry-After when shed) and never forwards to a peer under pressure — a pin is a pin. Source: `handle_post` pressure branch (server.py:3116-3160).
  - instruction: test with pressure=busy and a mesh peer present: model=cortex-<self> returns 429 `server_busy`, `open_upstream` not called
  - honesty: A pressured box sheds its own self-lane request with 429 and dials nothing
- A hop-marked arrival naming this box's OWN member lane is served locally, not answered with the 508 proxy-loop body the peer-lane branch (server.py:3256) returns, because no second forward happens.
  - instruction: test: hop marker + model=cortex-<self> on a hosting box returns 200 from the local lane
  - honesty: A hop-marked self-lane arrival is served locally (200), not 508

## Honesty conditions

- A member-named request is never re-balanced: the destination receives it with the hop marker and serves it locally
- Selection policy file is not modified
- Single-hop rule still holds for member lanes
- innereye-<member> and member names for stt/tts never forward; hand-<member> forwards like cortex
- No new peer env keys are parsed
- Qwen Code can address each box with a distinct model id from one provider baseUrl
- The before-state is measured, not recalled
- Every verified member of a role resolves by name on every mesh gateway, including the box's own name
- The pin survives a pooling fix
- No plain-pool behaviour changes
- A private-announced role yields no member lane
- Measured by engine counters on both boxes, not by gateway headers
- Rollout order is explicit

## Success signals

- Live on spark+spark2 (+Thor as a non-hosting gateway): 10 requests with model=cortex-spark2 through spark's gateway raise spark2's vllm:`request_success_total` by 10 and spark's by 0; the mirror (cortex-spark through spark2's gateway, and cortex-spark locally on spark) holds; `qwen -m cortex-spark2` answers via localhost:8001. Transcript under docs/evidence/.
  - instruction: script reads vllm:`request_success_total` inside each model-gear-vllm-primary before/after each 10-request batch; commit transcript to docs/evidence/2026-10-XX-accept-mesh-member-lanes.txt

## Scope / boundaries

- The selection policy in lobes/gateway/`_selection.py` is not changed. It was validated live on real load inputs in #199 t10 (docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt); the bug is its inputs, not its ranking.
  - instruction: git diff main -- lobes/gateway/`_selection.py` is empty at PR time
- The single-hop rule stays: a request arriving with the hop marker is served locally or refused, never re-selected (`_pool_selection`'s `_arriving_hop_marker` branch).
  - instruction: test: a hop-marked arrival naming a PEER's member lane returns the 508 proxy-loop body (existing server.py:3256 branch), never a second forward
- Member lanes exist only for mesh-forwardable, model-addressable roles: innereye (`MESH_UNFORWARDABLE_ROLES`, lobes/roles.py:259) and /v1/realtime get none; stt/tts are reached via /v1/audio/\* not model ids, so get no member lane. hand IS mesh-forwarded today (`NEVER_PROXIED_BACKENDS` = frozenset(), lobes/gateway/`_config.py`:207) and gets member lanes like any other role.
  - instruction: test: model=innereye-spark2 opens no upstream; model=hand-spark2 forwards to spark2 with `served_name`=hand backend
- Plain-role routing, pool candidate selection (`_pool_selection`, `_selection.py`) and the fingerprint plain/suffixed split are unchanged; a member lane never joins or leaves the plain pool.
  - instruction: existing tests/`test_gateway_pool`\*.py, `test_gateway_selection.py` and `test_mesh_routing`\*.py pass unmodified
- A role a member announced private gets no member lane anywhere (Announcement.public() strips it before `verified_roles`), and an innereye-<member> name never resolves to a peer.
  - instruction: test with Announcement.public() stripping cortex: cortex-<member> 404s and `member_lanes` omits it

## Non-goals

- No revival of the retired operator-typed <PREFIX>`_PEER_ORIGINS` / `_PEER_API_KEYS` family (retired in code at t14, see CLAUDE.md Retired section). The mesh roster is the only candidate source.
  - instruction: grep the diff for `_PEER_ORIGIN`; none added to lobes/gateway/`_config.py`
- No throughput-gain claim for heterogeneous pairs: the peer-only pool on Spark+Thor measured 51% slower at 4 concurrent (docs/evidence/2026-08-30-accept-peer-only-pool-orin.txt). The claim is scoped to identical replicas, which spark and spark2 are (same util 0.58 / 262144 / DSpark lane).
- Member suffixes on tier aliases (`main-spark2`, `hard-spark2`) and on raw checkpoint ids are not added; only Colleague role names take a member suffix.

## Assumptions

- ROOT CAUSE 1: no ReplicaCache exists for any mesh pool. `build_replica_caches` (lobes/gateway/server.py:5814) returns {} when table.`replica_origins` is empty, which every real deployment has been since t14 deleted <PREFIX>`_PEER_ORIGINS` parsing (comment at server.py:5653-5657).
- ROOT CAUSE 2: mesh candidates carry no load. `_merge_mesh_candidates` (server.py:2121) synthesizes every mesh peer with running=0, waiting=0, weight=8, calibrated=True; `_maybe_synth_local_candidate` (server.py:2157) synthesizes the local lane with running=0 too. Load is never observed for either side.
- ROOT CAUSE 3: with every wait at 0, `select_replica`'s `_rank_key` (lobes/gateway/`_selection.py`: wait, not-local, origin) sends everything to the local lane on a hosting box (reason local-idle) and to the lexically-smallest origin (<http://spark>...) on a non-hosting box, never spark2.
- ROOT CAUSE 4: in-flight dispatch accounting (ReplicaCache.`begin_dispatch`/`end_dispatch`, `_replicas.py`:895-950) is reached via caches.get(backend) (server.py:5898, 5934). With no cache there is no accounting, so even a burst through ONE gateway is not spread.
- A member name resolves only on a gateway running the new code: spark2/Thor/Orin gateways on today's image keep returning 404 `model_not_found` for an agreeing member's name (probed live 2026-10-06 on spark: cortex-spark2 and cortex-spark both 404). The Qwen Code use case needs only spark's gateway re-imaged; the c26 Thor leg needs Thor's too.
  - instruction: plan task: re-image spark's gateway first (Qwen use case), then Thor/spark2/Orin before the c26 cross-box legs; record versions in the evidence transcript

## Scope exploration

- `s1` — `lobes/gateway/server.py build_replica_caches (5814-5880)`: Cache construction is gated on table.`replica_origins`, which nothing populates since t14; mesh-enabled boxes therefore run with caches={} and no probe threads
  - seeds: `c2`, `c6` (rejected)
- `s2` — `lobes/gateway/server.py _merge_mesh_candidates / _maybe_synth_local_candidate / _pool_selection (2121-2275)`: Mesh and local candidates are synthesized with frozen zero load; mesh candidates are appended even if a cache already holds that origin
  - seeds: `c3`, `c8` (rejected)
- `s3` — `lobes/gateway/_selection.py select_replica / _rank_key`: Ranking is correct for real inputs; with all waits equal it resolves local-first then origin-ascending, which is exactly the observed pinning
  - seeds: `c4`, `c11`
- `s4` — `lobes/gateway/_replicas.py ReplicaCache (685-1474)`: Load probing, in-flight reconciliation and capacity resolution all exist and were live-validated in #199; the peer tuple is frozen at `__init__`, so it cannot follow a dynamic roster
  - seeds: `c5`, `c7` (rejected), `q2` (question)
- `s5` — `lobes/gateway/server.py /status handler (3870-3890, 4720-4745)`: /status publishes per-backend running/waiting and is outside the /v1 auth gate, so mesh peers can be probed without the mesh key
  - seeds: `c6` (rejected)
- `s6` — `lobes/gateway/server.py _local_backend_fingerprint (4063)`: Returns None with no cache; a live cache turns it into the placement reference, which can reclassify a peer as divergent
  - seeds: `q1` (question)
- `s7` — `memory: mesh-pool-never-spreads-load (measured 2026-10-06)`: Live: hosting box always local-idle at 8 concurrent with `max_num_seqs`=2; Thor sent a whole burst to spark, none to spark2; verify via vllm:`request_success_total` deltas
  - seeds: `c4`, `c10` (rejected)
- `s8` — `docs/evidence 2026-08-25 replica-pool + 2026-08-30 peer-only-pool transcripts`: The env pool spread load on real probes (validated); heterogeneous pairs lose throughput, so claims stay scoped to identical replicas
  - seeds: `c11`, `c15`
- `s9` — `CLAUDE.md Retired peer family + NEVER_PROXIED_BACKENDS`: Env peer family is retired in code. CORRECTED by the challenge pass (s11): hand IS mesh-forwarded (`NEVER_PROXIED_BACKENDS` is empty); only innereye and /v1/realtime are never mesh-forwarded.
  - seeds: `c13`, `c14`
- `s10` — `_pool_selection hop-marker branch`: Single-hop arrivals are sole-ready locally; unaffected by giving candidates real load
  - seeds: `c12`
- `s11` — `challenge pass / unstated-assumptions lens: lobes/gateway/_config.py:207 + lobes/roles.py:259 + live /mesh/roster`: hand is mesh-forwarded (`NEVER_PROXIED_BACKENDS` empty; spark2 announces hand plain); the only unforwardable role is innereye. Corrected c13/c24.
  - seeds: `c13`, `c24`
- `s12` — `challenge pass / adjacent-systems lens: live /mesh/roster on spark`: spark is not in its own roster (members: thor, orin, spark2), so the self lane must key on `LOBES_MESH_NAME`, not the snapshot
  - seeds: `c21`
- `s13` — `challenge pass / failure-modes lens: handle_post pressure + hop-marker branches (server.py:3116-3160, 3256)`: self lane needs the local pressure shed and must not 508 on a hop-marked arrival
  - seeds: `c32`, `c33`
- `s14` — `challenge pass / cheap-probe lens: spark gateway, 2026-10-06`: cortex-spark2 and cortex-spark both 404 `model_not_found` today; confirms before-state and that un-upgraded gateways will keep 404ing
  - seeds: `c34`
- `s15` — `challenge pass / operations + reversibility lens: LOBES_MESH_NAME, gateway re-image path`: renames and rollbacks silently break pinned clients; parked v4, no mitigation in scope
- `s16` — `challenge pass / security lens: suffixed forward path (server.py:3253-3300)`: clean: caller bearer is stripped and the forward is signed with the join key; member names are already exposed in X-Lobes-Mesh-Member headers, so listing them in /v1/models discloses nothing new to a key holder
- `s17` — `challenge pass / naming-collision lens: SUPPORTED_MODELS + ROLES`: clean: no catalog id starts with a role name plus hyphen; no role name is a prefix of another; hand LoRA uses ':' not '-'
- `s18` — `challenge pass / concurrency lens`: clean: resolution is a pure read of the immutable RoutingSnapshot per request; residual risk only if a member is dropped between resolve and forward (existing forward error path applies)
- `s19` — `challenge pass / unexamined: lobes capabilities CLI renderer, colleague/qwen clients' model-list parsing`: not read this pass; parked v5

## Decisions

- Per-member names appear in both GET /v1/models and /capabilities `member_lanes`.
- An announced-but-unprobed member's `{role}-{member}` answers 503 `role_unverified` + Retry-After, not 404.
- The plain-pool load-sharing fix (c6-c10) is out of this spec, parked as follow-up v3.

## Open parks

- [unknown_nonblocking] Neither Spark declares `PRIMARY_MAX_ACTIVE`, so both rank at the neutral capacity (raw queue depth). Fine for identical replicas, but a calibrated knee (lobes calibrate) would let `is_full` forward on saturation instead of queueing. Calibration is a follow-up, not this fix.
- [unknown_nonblocking] Probed load is up to one refresh interval (5 s) stale for traffic entering the OTHER gateway; only this gateway's own dispatches are counted first-hand. Whether that staleness matters for agentic multi-turn load on spark/spark2 is unmeasured.
- [unknown_nonblocking] Clients pinned to `cortex-spark2` break if a member is renamed (`LOBES_MESH_NAME` is operator-typed) or the gateway is rolled back to a pre-feature image — both answer 404. No alias/rename path is planned.
- [unknown_nonblocking] lobes capabilities CLI renderer and any third-party /capabilities consumer were NOT read in this pass; `member_lanes` is additive and assumed tolerated, unverified.
- [follow_up] Fix the plain-pool load sharing (root causes c2-c5: no ReplicaCache for mesh pools, synthetic zero-load candidates). Optional follow-up to this spec; the scoped requirements c6-c10 move there.
