# Build Plan — mesh pool load sharing

slug: `mesh-pool-load-sharing` · status: `exported` · from frame: `mesh-pool-load-sharing`

> Every verified mesh member of a role is addressable by name from ANY mesh gateway: `model=cortex-spark2` always lands on spark2's engine and `cortex-spark` on spark's, never balanced — while plain `cortex` stays the pool

## Tasks

### t1 — Member-lane resolver in lobes/gateway/`_mesh_routing.py`

- instruction: Add MemberLane + `member_lanes` + `find_member_lane` beside SuffixedLane/`find_suffixed_lane` (~line 645-880); reuse `suffixed_lane_name` for the name format and `_pending_origins_for_role` for pending. Do NOT touch `compute_role_placement` or the plain/suffixed split. Pure functions over RoutingSnapshot, no I/O. Self is NOT in the roster (live 2026-10-06), so self comes only from the `self_name` argument.
- covers: c20, h6, c13, h19, c24, h20
- acceptance:
  - tests/`test_mesh_member_lanes.py`: `member_lanes`(snapshot, role, `self_name`=..., `self_hosts`=...) returns one MemberLane(name='{role}-{member}', role, member, origin, `is_self`, pending) per verified member of the role, AGREEING AND DISAGREEING alike, plus `is_self` for `self_name` when `self_hosts`, plus pending=True for members whose probed is False and who announce the role
  - `find_member_lane`(snapshot, requested, roles, `self_name`=..., `hosted_roles`=...) resolves cortex-spark2 (agreeing peer), cortex-thor (disagreeing peer, same result as today's `find_suffixed_lane`), cortex-<self> (`is_self`), and returns None for an unknown member, for innereye-<member> (`MESH_UNFORWARDABLE_ROLES`) and for a role the member announced private
  - hand-<member> resolves like cortex (`NEVER_PROXIED_BACKENDS` is empty); `find_suffixed_lane` and `compute_role_placement` are unchanged and their existing tests pass unmodified

### t2 — /v1/models accepts extra member-lane ids in lobes/gateway/`_routing.py`

- instruction: `list_models_payload` is at lobes/gateway/`_routing.py`:617. Add a keyword-only `member_lane_ids`: Sequence\[str\] = () parameter; no other behaviour change. Do not touch server.py — the caller is wired in the dispatch task.
- covers: c30, h17
- acceptance:
  - tests/`test_routing_models_member_lanes.py`: `list_models_payload`(..., `member_lane_ids`=('cortex-spark','cortex-spark2')) appends one OpenAI model object per id (object='model', `owned_by` matching the existing convention), de-duplicated against ids already listed
  - with `member_lane_ids` omitted or empty the payload is byte-identical to today's (golden comparison)

### t3 — /capabilities `member_lanes` in lobes/roles.py `annotate_mesh_naming`

- instruction: `annotate_mesh_naming` is at lobes/roles.py ~1905. Call t1's `member_lanes`; `self_name` comes from a new optional parameter (the caller passes `LOBES_MESH_NAME`). Additive key only.
- depends on: t1
- covers: c22, h8
- acceptance:
  - tests/`test_capabilities_member_lanes.py`: with a mesh snapshot holding spark2 (agreeing) and `self_name`=spark hosting cortex, the cortex entry carries `member_lanes` == \['cortex-spark','cortex-spark2'\] (sorted); `suffixed_lanes` keeps today's meaning (disagreeing only)
  - `mesh_snapshot`=None leaves the payload byte-identical; roles with no addressable member get no `member_lanes` key

### t4 — Member-lane dispatch + /v1/models wiring in lobes/gateway/server.py

- instruction: Replace the `find_suffixed_lane` call at server.py:3253 with t1's `find_member_lane`, keeping the existing peer-forward body (`served_name`=`ROLE_BACKEND`, `join_key`, rewrite=True) for non-self lanes. For `is_self`: fall through to the normal local backend path with the requested model rewritten to the plain role, so the existing pressure branch applies, but skip `_pool_selection` (call `order_backends` for the owned backend directly). Reuse `_role_unverified_body` for pending. Self name = `_build_mesh_config`().name. Wire `member_lane_ids` into `_get_v1_models` (~5174) via t2's parameter.
- depends on: t1, t2
- covers: c1, h1, c18, h4, c19, h5, c21, h7, c31, h18, c32, h21, c33, h22, c12, h13
- acceptance:
  - tests/`test_gateway_member_lanes.py`: model=cortex-spark2 on a fake spark gateway forwards once to spark2's origin with body model rewritten to the backend name, Authorization = join key, hop marker set, X-Lobes-Proxied-By + X-Lobes-Mesh-Member on the answer; the destination's `_pool_selection` for a hop-marked arrival returns sole-ready even with a populated replica pool
  - model=cortex-<self> on a hosting box is served by the local lane with an `open_upstream` that fails if a peer is dialed; X-Lobes-Served-By set; under pressure it returns 429 `server_busy` and dials nothing; with a hop marker it is still served locally (200, not 508); on a non-hosting box it returns 404 `role_infeasible`
  - a pending (unprobed) member's name returns 503 `role_unverified`, Retry-After: 5, error.`hosted_by` and X-Lobes-Mesh-Member/-Unverified headers, no upstream; a hop-marked arrival naming a PEER lane still returns the existing 508 proxy-loop body
  - GET /v1/models on a mesh member lists every member lane id from t1; with mesh disabled /v1/models and the POST surface are byte-identical (existing tests pass unmodified)

### t5 — Docs, CLAUDE.md hand correction, explain text, version bump

- instruction: bump.py reads a changelog JSON on stdin with lowercase keys (added/changed/fixed). Keep CLAUDE.md edits minimal: fix the hand sentence and add one paragraph on member lanes under the mesh-brain join section.
- depends on: t3, t4
- covers: c11, h12, c23, h9
- acceptance:
  - docs/gateway-fleet.md mesh section documents {role}-{member} member lanes (self, agreeing, disagreeing, pending 503, pressure 429, exclusions innereye + /v1/realtime + private roles); lobes explain mesh mentions them; CLAUDE.md's hand paragraph no longer claims hand is never proxied
  - git diff main -- lobes/gateway/`_selection.py` is empty; uv run pytest -n auto passes with tests/`test_gateway_pool`\*.py, `test_gateway_selection.py` and `test_mesh_routing`\*.py unmodified; no `_PEER_ORIGIN` key added to lobes/gateway/`_config.py`; black/isort/flake8/bandit and afi cli doctor --strict clean
  - version bumped minor via .claude/skills/version-bump/scripts/bump.py with a CHANGELOG entry (uv.lock re-pin committed)

### t6 — Live rollout + acceptance transcript on spark, spark2, Thor; switch Qwen Code to member lanes

- instruction: Use the lobes-deploy skill's lobes-compose.sh; never run compose in lobes/templates/. Read memories peer-gateway-repin-dev-index-trap and lobes-up-gateway-drops-audio-overlay before re-imaging. Count engine metrics inside each model-gear-vllm-primary container, not via gateway headers.
- depends on: t5
- covers: c26, h11, c16, h2, c17, h3
- acceptance:
  - spark's gateway re-imaged first, then spark2, Thor, Orin; each box's gateway version recorded in the transcript
  - docs/evidence/<date>-accept-mesh-member-lanes.txt: 10 x model=cortex-spark2 via spark's gateway raises spark2's vllm:`request_success_total` by 10 and spark's by 0; mirror legs (cortex-spark via spark2's gateway, cortex-spark locally on spark, cortex-spark2 via Thor's gateway) hold; before-state cites the 2026-10-06 404 probe and Option 1 counters
  - ~/.qwen/settings.json (backed up first) has cortex-spark and cortex-spark2 entries at <http://localhost:8001/v1>; qwen -m each answers and moves the right engine counter; the interim direct spark2 entry (id main) is removed

## Risks

- [unknown_nonblocking] If the live fingerprints of spark and spark2 ever diverge, plain cortex stops pooling them but member lanes keep working; the transcript should note the current placement (plain vs suffixed) for context
