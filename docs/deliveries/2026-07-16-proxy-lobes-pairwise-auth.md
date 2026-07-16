# Delivery Summary — proxy-lobes + pairwise auth

plan: `proxy-lobes-pairwise-auth` · run: `partial` · date: `2026-07-16`
baseline: `devague summary skeleton`

## Intent

Deliver issue #127 phase 1 — the heterogeneous multi-node routing substrate —
by landing #115 proxy-lobes (a gateway serves a dropped role by following its
own referral to the hosting peer) plus pairwise gateway auth (inbound
`GATEWAY_API_KEY` bearer gate; per-peer outbound credentials that never
propagate a caller's key). Executed as the confirmed 10-task plan
(`docs/plans/2026-07-16-proxy-lobes-pairwise-auth.md`) fanned out by
/assign-to-workforce in waves `t1 | t2 t3 t4 t5 t7 | t6 | t8 t9`, with `t10`
(live cross-box acceptance) operator-led post-merge. The run is `partial`
solely because `t10` cannot run until PR #130 merges and publishes.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — t1 Config plumbing: \<PREFIX\>_PEER_PROXY + \<PREFIX\>_PEER_API_KEY env channels and GATEWAY_API_KEY (fallback CULTURE_VLLM_API_KEY) in lobes/gateway/_config.py; RoutingTable gains peer_proxied + peer_api_keys, ServerConfig gains api_key. Files: lobes/gateway/_config.py, lobes/gateway/_routing.py (fields only), tests/test_gateway_config_proxy.py
- `t2` — t2 Inbound auth gate in lobes/gateway/server.py: when cfg.api_key is set every data-plane route 401s missing/wrong Authorization: Bearer via hmac.compare_digest BEFORE body parse; /health + probes stay keyless. Files: lobes/gateway/server.py (auth helpers, handle_post entry, GET dispatch), tests/test_gateway_auth.py
- `t3` — t3 CLI attaches the key: gateway-dialing paths in lobes/assess.py + lobes/cli/_runtime_ops.py read GATEWAY_API_KEY/CULTURE_VLLM_API_KEY from the deployment .env and send Authorization. Files: lobes/assess.py, lobes/cli/_runtime_ops.py, tests/test_cli_gateway_auth.py
- `t4` — t4 Peer-readiness probe: lobes/gateway/_readiness.py learns to probe a proxied role's peer origin (/v1/models with the per-peer key) so ready/advertised reflects the peer's live state; bounded timeout, never blocks local probes. Files: lobes/gateway/_readiness.py, tests/test_readiness_peer_probe.py
- `t5` — t5 PROXIED capabilities state: lobes/roles.py advertises a proxied role as state=proxied with the operator-declared origin (hosted_by retained) — never as locally served. Files: lobes/roles.py, tests/test_roles_proxied.py
- `t6` — t6 Proxy data-plane branch in lobes/gateway/server.py: follow-the-referral forwarding for dropped+proxied roles — replace inbound Authorization with the per-peer key, add hop-marker request header + X-Lobes-Proxied-By response header, single-hop loop guard, peer-down 503 Retry-After, peer role_infeasible relay, /v1/models peer-gating; reuse the existing streaming relay. Files: lobes/gateway/server.py, lobes/gateway/_routing.py (list_models_payload peer gating), tests/test_gateway_proxy.py
- `t7` — t7 Templates + shape render: fleet compose gateway env block gains GATEWAY_API_KEY + per-role PEER_PROXY/PEER_API_KEY as scoped environment: entries (never env_file); env.example documents them; shape render/accept round-trips the new knobs byte-for-byte. Files: lobes/templates/fleet/docker-compose.yml, lobes/templates/fleet/env.example, lobes/profiles/shape_render.py (if knobs flow through shapes), tests/goldens updates via regen
- `t8` — t8 Integration suite + goldens: a two-gateway mock-pair test (proxying box + peer box in-process) exercising the full after-state — proxied senses answer, 401 gate, loop guard, peer-down 503, marker headers, no-key-leak grep across captured logs/bodies — plus the byte-identical no-config golden assertions. Files: tests/test_proxy_integration.py (new), tests/goldens/ additions
- `t9` — t9 Docs: rewrite docs/gateway-fleet.md auth section (limitation lifted), extend docs/deployment-shapes.md + docs/colleague-stack.md + docs/openai-api.md with proxy-lobes states, pairwise-key contract (one inbound key per box, one outbound key per peer), marker headers; update lobes/explain catalog + CLAUDE.md; re-verify before-state citations against main. Files: docs/*.md, lobes/explain/, CLAUDE.md
- `t10` — t10 Live cross-box acceptance (operator-led, post-merge/publish): re-pin Thor off 0.43.0.dev239, enable MULTIMODAL_PEER_PROXY + pairwise keys on Spark, run a senses chat request against Spark's gateway, verify Thor-produced Gemma completion + honest capabilities on both boxes, commit the transcript under docs/evidence/

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `PEER_PROXY_ENV`/`PEER_API_KEY_ENV`/`_gateway_api_key` in `_config.py`; `RoutingTable.peer_proxied`/`peer_api_keys` (repr-hidden) + `ServerConfig.api_key`; 34 tests. Commit `f2cca14`. |
| `t2` | delivered | Opt-in bearer gate: 401 before body parse on every POST + GET `/v1/*`; `/health`/`/capabilities`/`/status` keyless by documented policy; `hmac.compare_digest`; 35 tests. Commit `0c53082`. |
| `t3` | delivered | `gateway_auth_headers` + `friendly_unauthorized_errors` in `_runtime_ops.py`; contextvar `auth_headers()` in `assess.py` reaching capabilities/assess/benchmark/measure/endpoint; 21 tests. Commit `09137b3`. Known gaps documented (see Remaining Work). |
| `t4` | delivered | `PeerSpec` + `probe_peer_ready` + isolated peer-probe thread in `_readiness.py` (peer must list the exact served id); 30 tests. Commit `773170b`. |
| `t5` | delivered | `proxied: true` + `hosted_by` key-presence contract in `roles.py`; byte-identity proven against an independent pre-change oracle; 20 tests. Commit `73158a4`. |
| `t6` | delivered | Proxy branch in `handle_post` (Authorization replaced with pairwise key, `X-Lobes-Proxied` hop marker, 508 `proxy_loop`, peer-down 503 + Retry-After, peer-declines relay, pressure bypass), `/v1/models` peer-gating, capabilities `peer_ready` channel, `serve()` wiring; 41 tests. Commit `dd39a8c`. |
| `t7` | delivered | 10 scoped `${VAR:-}` gateway env entries (never `env_file`); env.example knob docs + worked example; goldens regen = exactly the 10 inert lines; shape_render confirmed no-change-needed. Commit `9f1178a`. |
| `t8` | delivered | Two-real-gateway socket harness: proxied chat + SSE incrementality, 401 isolation, loop guard, peer-down, peer-declines, credential-hygiene grep, byte-for-byte no-config golden wire bytes; 15 tests, zero product bugs found. Commit `bce12ec`. |
| `t9` | delivered | gateway-fleet auth section rewritten; deployment-shapes three-state table + pairwise contract; colleague-stack/openai-api/realtime/explain/CLAUDE.md updated; 7 stale-citation clusters fixed in docs+code comments. Commit `5ef548e` (+ `812461d` for the 3 out-of-scope sites it flagged). |
| `t10` | blocked | Cannot run until PR #130 merges and PyPI publishes 0.45.0; Thor must be re-pinned off `0.43.0.dev239` first (plan risk r1, challenge park v2). Operator-led; `ssh thor@thor` access approved. |

## Mid-work Decisions

No `/deviate` records exist for this plan — no execution step departed from a
confirmed task's contract far enough to require the deviation gate. Decisions
made within task boundaries, captured directly:

- `t2` kept `/status` keyless alongside `/capabilities` (the spec listed only
  `/health` + readiness) — it is the control-plane observability aggregate,
  serves no inference; documented in the route-policy comment.
- `t2` gates the whole GET `/v1/*` namespace so a 401 outranks the 404 —
  unauthenticated callers cannot enumerate routes.
- `t3` used a `contextvars.ContextVar` context manager instead of threading a
  `headers=` kwarg through ~10 public entry points — existing test fakes
  monkeypatch `_post`/`_get` with narrower signatures, and `roles_measure.py`
  calls them by name; the contextvar reaches both without a second plumbing
  pass.
- `t3` discovered `status --pressure` makes no network call at all (pure
  `/proc` sampling) — the plan's assumption it needed wiring was wrong;
  nothing to wire.
- `t5` chose key-presence (never a `proxied: false` sentinel) to keep hosted
  roles byte-identical; `t6` then threaded a NEW `peer_ready` signal rather
  than relaxing t5's feasible clamp, keeping t5's pinned tests unmodified.
- `t6` chose 508 for the loop refusal and resolves an unwired proxied role's
  served id as wired-backend → `<PREFIX>_SERVED_NAME` env → catalog hint,
  with the peer probe supplying the honesty.
- `t7` recomputed a hash-pin guard test (`test_tool_parser_plugin.py`) that
  pins the gateway compose subtree — a necessary consequence of editing that
  block, outside its declared file list; only the gateway hash changed.
- Main-agent close-out: colleague review prompted one defensive fix (an
  unresolvable served id builds no peer spec — degrades to referral-only) and
  a method-policy comment; the review's HEAD/OPTIONS "bypass" was refuted
  (unimplemented methods 501 before any routing) and two repr-safety items
  were already satisfied. The live tailnet hostname was scrubbed from the new
  spec/frame artifacts to placeholders before publishing the PR. SonarCloud
  S8508 (mutable ContextVar default) fixed with an immutable `None` default.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t3` | Two gateway-dialing stragglers were NOT wired (audio-overlay stt/tts probes in `roles_measure.py` build their own requests; `lobes/minor/_client.py` is a separate client) — both degrade gracefully rather than traceback; filed as #129 | needs-follow-up |
| `t7` | Touched `tests/test_tool_parser_plugin.py` (hash-pin recompute) beyond its declared file list — required by the compose edit, verified only the gateway hash moved | acceptable |
| `t10` | Not executed in this run: sequenced after merge + publish by design (plan risk r1) | needs-follow-up |

All other tasks delivered to their acceptance criteria as confirmed — backed
by the task-by-task accounting above and the per-task test suites.

## Evidence

- tests: full suite `uv run pytest -n auto -q` — **1919 passed, 14 skipped**
  (baseline before the run: 1726 passed; +193, zero regressions at each of the
  9 TDD merge gates — before AND after every merge)
- integration: `tests/test_proxy_integration.py` — 15/15 pass (two real
  gateways over live sockets)
- lint: `black --check` / `isort --check-only` / `flake8` / `bandit -c
  pyproject.toml -r lobes` — all clean; `uv run afi cli doctor . --strict` —
  exit 0; `markdownlint-cli2` on touched docs — 0 errors
- commits: `6b1a0b0..005c885` (26 commits on `feat/127-proxy-lobes-pairwise-auth`)
- PR: #130 (all CI checks pass incl. SonarCloud quality gate after the S8508
  fix; Qodo: 0 bugs / 0 rule violations / 0 requirement gaps, 0 inline
  threads)
- issues: #127 (parent), #115 (proxy-lobes), #128 + #129 (follow-ups filed)
- colleague review: run artifact under `.colleague/` (graded 4/5; one finding
  fixed in commit `812461d`'s follow-on `fix:` commit)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| A dropped+proxied role's request forwards to the peer and relays JSON and SSE answers with the pairwise key, never the caller's | high | test `tests/test_proxy_integration.py` (credential-hygiene + SSE tests) · commit `dd39a8c` |
| Inbound auth 401s missing/wrong keys before any backend work; unset = byte-identical | high | test `tests/test_gateway_auth.py` (35 tests) · byte-identical goldens in `tests/test_proxy_integration.py` |
| Misconfigured proxy loops fail fast (single hop, 508 `proxy_loop`) | high | tests in `tests/test_gateway_proxy.py` + `tests/test_proxy_integration.py` |
| Capabilities/`/v1/models` stay honest: proxied state named, ids listed only on a live peer verdict | high | tests `tests/test_roles_proxied.py`, `tests/test_gateway_proxy.py` |
| No-config deployments are wire-byte-identical to before the feature | high | golden byte-equality tests in `tests/test_proxy_integration.py` |
| The lobes CLI works unchanged when auth is enabled | medium | test `tests/test_cli_gateway_auth.py` (21 tests; known stragglers in #129) |
| Spark serves senses by proxy from Thor on the live pair (the announcement's end-to-end claim) | unverified | t10 not yet run — not claimed done; requires merge + publish + Thor re-pin |

## Remaining Work / Follow-up

- `t10` — live cross-box acceptance: after PR #130 merges and 0.45.0
  publishes, re-pin Thor off `0.43.0.dev239`, arm `MULTIMODAL_PEER_PROXY` +
  pairwise keys on Spark (+ `GATEWAY_API_KEY` on Thor), run the senses
  request against Spark's gateway, commit the transcript under
  `docs/evidence/`. Owner: operator + main agent (`ssh thor@thor` approved).
- #128 — #127 phases 2–4: fan-out execution, `lobes trace`, runtime-aware
  routing, policy plugins.
- #129 — auth-key stragglers (stt/tts measure probes, `lobes/minor` client)
  and the missing peer-proxy channel for audio roles.
- Human gate 3 — PR #130 review + merge (this artifact is the review map).
