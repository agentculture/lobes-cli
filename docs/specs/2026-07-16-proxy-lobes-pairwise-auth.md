# proxy-lobes + pairwise auth

> lobes serves the whole mesh brain from any box: a gateway can serve a dropped role by following its own referral — proxying the request to the peer that hosts the lobe (proxy-lobes, #115) — so a local app on Spark asking for senses seamlessly gets Thor's answer; the exposed gateway enforces inbound auth on its own key, and each box reaches its peers with pairwise operator-declared credentials that are never propagated forward from the caller (#127)
> instruction: Verify via the live cross-box acceptance run (senses request to Spark returns a Thor-produced Gemma completion; capabilities honest on both boxes) plus the offline unit suite (routing, auth, loop guard, peer-down 503, byte-identical no-config goldens).

## Audience

- Local apps and mesh agents on any box (Culture agents, eidetic, colleague CLI) that dial their box's lobes gateway; operators running lobes init --shape across Spark/Thor/Orin.

## Before → After

- Before: Today a dropped role 404s role_infeasible with a hosted_by referral the CLIENT must follow — every local app needs per-box endpoint config + the peer's key propagated to it; the fleet gateway enforces no inbound auth at all (docs/gateway-fleet.md known limitation).
- After: An app pointed at Spark's gateway asks for senses and gets Thor's Gemma answer over the spark->thor pairwise credential, without knowing Thor exists; the exposed Spark key authenticates callers inbound; lobes capabilities shows senses as PROXIED with the peer origin; misconfig loops fail fast; no-peer deployments byte-identical.

## Why it matters

- One brain across the mesh (#112 end-state) only works if consumers can stay single-endpoint: apps configured once against their local box, placement changed by operators in config — and key distribution stays O(machines) pairwise instead of O(apps x machines).

## Requirements

- Proxy-lobe data plane: when a role is dropped (<PREFIX>_FEASIBLE=false) AND a peer origin is declared AND proxying is opted in for that role, lobes/gateway/server.py handle_post forwards the request to the peer origin instead of returning the 404 role_infeasible — the third lobe state from issue #115 (awake / asleep / proxy). Builds on RoutingTable.peer_origins (lobes/gateway/_routing.py:59) which today is annotation-only ('it is NEVER dialed').
  - honesty: A senses request against Spark with proxy enabled returns the PEER's model output (Gemma served_name in the response body) — never a locally-substituted model (#91); verified by a unit test with a mock peer and live on the physical pair.
- Honesty surfaces follow #115's table: a proxied role is advertised as PROXIED (origin = the peer) on lobes capabilities and GET /capabilities (lobes/roles.py hosted_by annotation site) — never as locally served; per the #92 lesson (gotcha gateway-public-url-must-not-default-2026-07-09) proxy origins are operator-declared, never derived.
  - honesty: lobes capabilities and GET /capabilities on a proxying box mark the role with an explicit proxied state naming the peer origin (hosted_by retained), and ready reflects a live proxied-path probe or is honestly unknown — never a hardcoded true.
- Inbound gateway auth: the fleet gateway grows opt-in Authorization: Bearer enforcement against a configured key (docs/gateway-fleet.md:376 records today's known limitation — the gateway is pass-through, not auth-aware; CULTURE_VLLM_API_KEY only protects the legacy single-model path via VLLM_API_KEY in templates/docker-compose.yml:51).
  - honesty: With GATEWAY_API_KEY set, every data-plane route (chat, embeddings, rerank, audio) rejects a missing/wrong Authorization: Bearer with 401 before touching any backend; health/readiness endpoints stay unauthenticated for probes; with the knob unset behavior is byte-identical to today.
- Pairwise outbound credentials: when proxying to a peer the gateway REPLACES the inbound Authorization header with an operator-declared per-peer key (e.g. <PREFIX>_PEER_API_KEY next to <PREFIX>_PEER_ORIGIN in _config.PEER_ORIGIN_ENV's convention) — the caller's key authenticates only to the box it dialed; keys are between machines and never propagated forward.
  - honesty: The inbound Authorization header is NEVER forwarded to a peer: the proxy branch strips it and attaches the operator-declared per-peer key; a test asserts the caller's key does not appear in the outbound request; peer keys never appear in logs, traces, capabilities output, or error bodies.
- Proxied answers are visible: a proxied response carries an explicit marker header (e.g. X-Lobes-Proxied-By: <peer origin>) so callers always see which box actually answered — #127's stated design principle ('without hiding which lobe actually handled the request'); the request-side hop marker (loop guard, h5) and this response-side marker are one header family.
  - honesty: A unit test asserts the marker header is present on proxied responses, absent on locally-served ones, and carries the operator-declared peer origin verbatim (never a derived URL, #92).
- Turning on inbound auth must not break the box's own tooling: the lobes CLI verbs that dial the local gateway (capabilities, assess, status --pressure, measure, benchmark) read the deployment's key from .env and send Authorization when GATEWAY_API_KEY resolves — provenance: lobes/assess.py + cli/_runtime_ops.py dial the gateway with stdlib urllib today, keyless.
  - honesty: With GATEWAY_API_KEY set in a test deployment .env, lobes capabilities / assess against the local gateway succeed keylessly-from-the-user's-view (the CLI attaches the key itself); with the wrong key in .env they fail with a clear 401 message, not a traceback.
- Peer-declines honesty: when the proxied-to peer itself answers 404 role_infeasible (the peer also dropped the role — a misdeclared peer origin), the proxying box relays an honest terminal failure naming the peer, and never chains a second hop (single-hop rule, h5).
  - honesty: A mock-peer test where the peer returns role_infeasible asserts the proxying box's response is a terminal error naming the peer and that no second outbound request is attempted.
- Key handling hygiene: bearer comparison is timing-safe (hmac.compare_digest); GATEWAY_API_KEY and *_PEER_API_KEY reach the gateway container as scoped environment: entries in the compose template — never via env_file (the gateway env block deliberately avoids inheriting .env secrets; PR #117 Qodo finding).
  - honesty: Code review + a grep-test assert compare_digest is used for the bearer check and that no key value appears in any log line, error body, or capabilities payload the test suite captures.

## Honesty conditions

- The shipped feature matches the announcement end-to-end on the live pair: Spark serves senses by proxy from Thor under pairwise keys, and the exported spec's success signal (c17) is demonstrated with committed evidence.
- A request carrying the proxy-hop marker that resolves to another proxy departure is refused with an explicit error (not forwarded), covered by a two-box misconfiguration unit test.
- Peer connect-refused/timeout yields 503 with Retry-After from the proxying box (no cross-model fallback); the proxying box's own pressure policy does NOT gate proxied requests — the peer's does (its gateway applies its own policy on arrival).
- A golden/regression test renders a no-proxy, no-auth deployment before and after this change and asserts byte-identical gateway responses on /capabilities, /v1/models, and the role_infeasible 404 body.
- The delivered surfaces are exercised by the named consumers' path: an OpenAI-compatible client pointed at the box gateway (no client-side changes needed) and lobes init/--shape for operators.
- Each element of the after-state maps to a shipped, tested behavior: proxied senses answer, inbound 401 gate, PROXIED capabilities state, loop-guard error, byte-identical no-config goldens.
- The before-state is cited from code/docs as they exist today (docs/gateway-fleet.md:376 known limitation; _routing.py peer_origins 'NEVER dialed' comment) — re-verified against main at implementation start.
- Key distribution stays O(machines): the delivered config needs exactly one inbound key per box plus one outbound key per (box, peer role source) pair — no per-app peer keys anywhere in the shipped docs or templates.
- The success signal is only claimed once the live cross-box run's transcript is committed under docs/evidence/ and the offline suite passes in CI.

## Success signals

- Live cross-box proof on the physical boxes (like docs/evidence/2026-07-14-accept-referral-thor.txt): a chat request naming senses against Spark's gateway returns a Gemma completion produced on Thor, with capabilities honest on both boxes; plus offline unit coverage for routing, auth, loop-guard, and peer-down 503.

## Scope / boundaries

- Proxying is single-hop with a loop guard: a gateway never re-proxies a request that already arrived proxied (e.g. an X-Lobes-Proxied marker refused at the second hop) — two boxes misconfigured to point at each other must fail fast, not ping-pong.
- Failure modes carried from #115: hosting peer down => 503 with Retry-After (single-owner rule, no cross-model failover per #91); a proxied request counts against the PEER's pressure policy, never masked by the proxying box.
- Default-off, byte-identical: with no proxy opt-in config, every surface (capabilities, 404 role_infeasible + hosted_by referral, /v1/models) stays byte-identical to today's contract — referral-only deployments (Spark/Thor live today) must not silently start proxying or enforcing auth.

## Non-goals

- Out of scope from #127's own non-goals + phases 2-4: fan-out execution (lobes fanout), request tracing store (lobes trace), queue-depth/latency-aware routing, policy plugins, learned routing, KV/tensor sharing, replacing inference runtimes, Colleague-level reasoning policy. This delivery is the phase-1 substrate: config-driven placement, capability addressing via the existing role/alias vocabulary, proxy data plane, auth, explainable routing.
- No parallel lobe/node YAML vocabulary: #127's 'lobes:'/'nodes:' config maps onto the EXISTING six-role contract (cortex/senses/embedder/reranker/stt/tts), tier aliases, machine profiles, and deployment shapes + *_PEER_ORIGIN env convention — lobes-cli already models 'named lobes'; we extend that, not fork a second config schema.

## Assumptions

- Peers are reachable at operator-declared origins on the tailnet (e.g. <http://thor.<tailnet>.ts.net:8000>); transport security between boxes is the tailnet's/tunnel's job at this layer — pairwise keys authenticate, TLS termination is out of scope (existing practice for CULTURE_VLLM_API_KEY over cloudflared).
- stt/tts (audio overlay) have no peer-origin channel today (outside the Profile schema, lobes/gateway/_config.py PEER_ORIGIN_ENV covers primary/multimodal/embed/rerank only) — proxy-lobes covers the four core roles first; audio-role proxying is a follow-up.
- The forwarded body carries the RESOLVED served model id and the peer serves that same id — verified at runtime by the peer-readiness probe against the peer's /v1/models (c19), so a proxied request never asks the peer for a model it does not serve.

## Scope exploration

- `s1` — `lobes/gateway/_routing.py (RoutingTable.peer_origins, infeasible_owner, order_backends)`: peer_origins exists since mesh-brain t3 but is control-plane only ('NEVER dialed — the gateway does no data-plane proxying to peers (proxy-lobes is deferred, issue #115)'); order_backends enforces single-owner no-failover (#91) — proxying must preserve that: a proxied senses request is still answered by senses-on-the-peer, never substituted
  - seeds: `c2`
- `s2` — `lobes/gateway/server.py (_role_infeasible_body:430, handle_post:~830, _HOP_BY_HOP:86)`: the 404 role_infeasible + hosted_by body is built at _role_infeasible_body; handle_post gates infeasible BEFORE pressure-shedding; the relay already strips hop-by-hop headers incl. proxy-authorization — the natural seam for a follow-the-referral proxy branch and for Authorization replacement
  - seeds: `c2`, `c5`
- `s3` — `lobes/roles.py:529 (_annotate hosted_by)`: capabilities hosted_by annotation for unhosted roles lands here; a proxied role needs a distinct advertised state (proxied, origin=peer) per #115's three-state table
  - seeds: `c3`
- `s4` — `docs/gateway-fleet.md:376 + templates (docker-compose.yml:51, fleet/docker-compose.yml gateway env)`: auth today: CULTURE_VLLM_API_KEY→VLLM_API_KEY only on the LEGACY single-model vLLM service; the fleet gateway is documented not-auth-aware and its compose env block passes no key; inbound gateway auth is new work, per-endpoint auth 'planned for a later release'
  - seeds: `c4`
- `s5` — `issue #115 body (proxy-lobes, deferred from #112)`: the three-state lobe table (awake/asleep/proxy), honesty constraint (advertise as PROXIED, never local), no-silent-rerouting (#91), peer-down=>503 Retry-After, and pressure-counting-on-the-peer are already decided there — #127's phase-1 routing substrate is this issue plus auth
  - seeds: `c6`, `c7`
- `s6` — `issue #127 body (agentculture/lobes-cli)`: the umbrella issue proposes lobe/node YAML config, lobes run/fanout/trace verbs, and 4 phases; its own non-goals exclude KV sharing, merged servers, reasoning policy; phases 2-4 (fanout, runtime-aware routing, policy plugins) are explicitly later phases — this delivery targets phase 1 mapped onto the existing role/shape/peer-origin machinery
  - seeds: `c9`, `c10`
- `s7` — `~/.lobes/.env (live Spark spark-lobe deployment)`: MULTIMODAL_PEER_ORIGIN=<http://thor.<tailnet>.ts.net:8000> and CULTURE_VLLM_API_KEY are already set live; Thor's gateway is port 8000, Spark's is 8001; the senses->Thor proxy is exactly the first real deployment of this feature
  - seeds: `c11`
- `s8` — `lobes/profiles/shapes.py + shape_render.py (deployment shapes t1-t3)`: shapes are pure data (hosts + overrides) rendered to env; the proxy opt-in and per-peer key knobs must render through the same shape/env pipeline so lobes init --shape stays the single entry point and restore stays byte-for-byte
  - seeds: `c8`
- `s9` — `challenge pass / adjacent-systems lens: lobes/templates/fleet/docker-compose.yml ports + eidetic/colleague consumers`: only the gateway publishes a host port (${VLLM_PORT:-8000}:8000) — backends are compose-network-internal, so inbound gateway auth covers the box's entire published surface; local consumers (colleague CLI :8001, eidetic embed) dial the gateway and are covered by c21
  - seeds: `c21`
- `s10` — `challenge pass / concurrency lens: lobes/gateway/server.py ThreadingHTTPServer + per-request upstream connections`: clean pass — the proxy branch reuses the existing per-request connection + read1 streaming relay; no shared mutable state is added; the readiness cache is the only cross-thread structure and already exists
- `s11` — `challenge pass / lifecycle lens: accept-shape backup/restore + image rollback`: new env keys are inert to an old gateway image (unknown env ignored) so rollback is safe; the shape re-render must round-trip the new knobs byte-for-byte like *_PEER_ORIGIN does (c8 goldens)
  - seeds: `c8`
- `s12` — `challenge pass / cheap probe: live Thor gateway from Spark (2026-07-16)`: <http://thor.<tailnet>.ts.net:8000/health> ok + /capabilities 200 — peer reachability for the acceptance run is real today; Thor still pinned to 0.43.0.dev239 (parked v2)
- `s13` — `challenge pass / security lens: bearer handling + secret scoping`: timing-safe compare + scoped environment: entries captured as c23; replay/rate-limit/TLS parked as v1 (tailnet transport assumption c11)
  - seeds: `c23`
- `s14` — `challenge pass / observability lens: issue #127 design principle`: the spec lacked any caller-visible signal of WHERE a proxied answer came from — c20 adds the proxied-by marker family
  - seeds: `c20`

## Decisions

- Implementation lands on the existing seams: RoutingTable.peer_origins gains a proxy flag + per-peer credential; handle_post grows a follow-the-referral branch reusing the existing streaming relay; capabilities/roles.py annotates the proxied state; env knobs render through shapes like *_PEER_ORIGIN does.
- Answer to the /v1/models hard question: a proxied role's served_name IS listed on the proxying box — but only when a live peer-readiness probe passes (the readiness cache learns to probe the peer origin for proxied roles), preserving 'advertised implies reachable' (#92); peer down => the id drops from /v1/models exactly like a dead local backend.

## Hard questions

- What does /v1/models list on the proxying box — does a proxied role's served_name appear (it IS servable there now) and how does that stay honest with 'advertised implies reachable' when the peer is down?
