# Build Plan — proxy-lobes + pairwise auth

slug: `proxy-lobes-pairwise-auth` · status: `exported` · from frame: `proxy-lobes-pairwise-auth`

> lobes serves the whole mesh brain from any box: a gateway can serve a dropped role by following its own referral — proxying the request to the peer that hosts the lobe (proxy-lobes, #115) — so a local app on Spark asking for senses seamlessly gets Thor's answer; the exposed gateway enforces inbound auth on its own key, and each box reaches its peers with pairwise operator-declared credentials that are never propagated forward from the caller (#127)

## Tasks

### t1 — t1 Config plumbing: <PREFIX>_PEER_PROXY + <PREFIX>_PEER_API_KEY env channels and GATEWAY_API_KEY (fallback CULTURE_VLLM_API_KEY) in lobes/gateway/_config.py; RoutingTable gains peer_proxied + peer_api_keys, ServerConfig gains api_key. Files: lobes/gateway/_config.py, lobes/gateway/_routing.py (fields only), tests/test_gateway_config_proxy.py

- covers: c5, h12, c16
- acceptance:
  - build_config maps <PREFIX>_PEER_PROXY=true into RoutingTable.peer_proxied ONLY when that prefix also has a peer origin and is infeasible; origin without the knob stays referral-only (q1)
  - GATEWAY_API_KEY resolution: explicit value wins, else CULTURE_VLLM_API_KEY, else None => auth disabled (q2); channels scoped to the four core roles
  - no-knob env yields config objects equal to today's on every existing field; peer keys never appear in repr/str

### t2 — t2 Inbound auth gate in lobes/gateway/server.py: when cfg.api_key is set every data-plane route 401s missing/wrong Authorization: Bearer via hmac.compare_digest BEFORE body parse; /health + probes stay keyless. Files: lobes/gateway/server.py (auth helpers, handle_post entry, GET dispatch), tests/test_gateway_auth.py

- depends on: t1
- covers: c4, h3, c23, h17
- acceptance:
  - no header / malformed / wrong key each 401 with an OpenAI-shaped error and zero upstream connections (h3)
  - /health answers keyless; api_key unset => byte-identical on every route
  - comparison is hmac.compare_digest; 401 body never echoes any key material

### t3 — t3 CLI attaches the key: gateway-dialing paths in lobes/assess.py + lobes/cli/_runtime_ops.py read GATEWAY_API_KEY/CULTURE_VLLM_API_KEY from the deployment .env and send Authorization. Files: lobes/assess.py, lobes/cli/_runtime_ops.py, tests/test_cli_gateway_auth.py

- depends on: t1
- covers: c21, h15, c13, h9
- acceptance:
  - lobes capabilities/assess succeed against an auth-enabled mock gateway using the .env key; wrong key => clear 401 message, not a traceback (h15)
  - keyless deployment sends no Authorization header (byte-identical)

### t4 — t4 Peer-readiness probe: lobes/gateway/_readiness.py learns to probe a proxied role's peer origin (/v1/models with the per-peer key) so ready/advertised reflects the peer's live state; bounded timeout, never blocks local probes. Files: lobes/gateway/_readiness.py, tests/test_readiness_peer_probe.py

- depends on: t1
- covers: h2
- acceptance:
  - a proxied role reports ready=True only when the peer answers and lists the resolved served id (c19/c24); peer down => ready None/False and the id drops from /v1/models
  - peer probe failures never delay or fail local-backend probing (isolated timeout + thread-safety preserved)

### t5 — t5 PROXIED capabilities state: lobes/roles.py advertises a proxied role as state=proxied with the operator-declared origin (hosted_by retained) — never as locally served. Files: lobes/roles.py, tests/test_roles_proxied.py

- depends on: t1
- covers: c3
- acceptance:
  - capabilities payload marks proxied roles distinctly from hosted and from referral-only-dropped; origin string is the operator's verbatim (#92)
  - referral-only and no-peer deployments render byte-identically to today

### t6 — t6 Proxy data-plane branch in lobes/gateway/server.py: follow-the-referral forwarding for dropped+proxied roles — replace inbound Authorization with the per-peer key, add hop-marker request header + X-Lobes-Proxied-By response header, single-hop loop guard, peer-down 503 Retry-After, peer role_infeasible relay, /v1/models peer-gating; reuse the existing streaming relay. Files: lobes/gateway/server.py, lobes/gateway/_routing.py (list_models_payload peer gating), tests/test_gateway_proxy.py

- depends on: t2, t4
- covers: c2, h1, h4, c6, h5, c7, h6, c20, h14, c22, h16
- acceptance:
  - a dropped+proxied role's request forwards to the peer origin and streams the peer's SSE/JSON answer back unchanged, model id = the peer's served id (h1)
  - outbound request carries the per-peer key and NOT the caller's Authorization; a test greps the captured outbound headers for the caller's token and finds nothing (h4)
  - a request already carrying the hop marker that would depart again is refused with an explicit error and no outbound attempt (h5)
  - peer connect-refused/timeout => 503 with Retry-After; local pressure policy is not applied to proxied requests (h6)
  - peer answers role_infeasible => terminal error naming the peer, no second hop (h16)
  - proxied responses carry X-Lobes-Proxied-By: <peer origin> verbatim; locally-served responses do not (h14)

### t7 — t7 Templates + shape render: fleet compose gateway env block gains GATEWAY_API_KEY + per-role PEER_PROXY/PEER_API_KEY as scoped environment: entries (never env_file); env.example documents them; shape render/accept round-trips the new knobs byte-for-byte. Files: lobes/templates/fleet/docker-compose.yml, lobes/templates/fleet/env.example, lobes/profiles/shape_render.py (if knobs flow through shapes), tests/goldens updates via regen

- depends on: t1
- covers: c8
- acceptance:
  - default-rendered deployment (no new vars set) produces byte-identical compose/env to today (goldens regen shows no diff for machine-as-brain/spark-lobe/thor-lobe/orin-small without the new knobs)
  - keys reach the gateway service as scoped environment: entries; env_file is not introduced (c23)

### t8 — t8 Integration suite + goldens: a two-gateway mock-pair test (proxying box + peer box in-process) exercising the full after-state — proxied senses answer, 401 gate, loop guard, peer-down 503, marker headers, no-key-leak grep across captured logs/bodies — plus the byte-identical no-config golden assertions. Files: tests/test_proxy_integration.py (new), tests/goldens/ additions

- depends on: t6, t5, t7, t3
- covers: h7, c14, h10
- acceptance:
  - every after-state element of c14 maps to a passing test in this suite (h10)
  - before/after golden responses for a no-proxy no-auth deployment are byte-identical on /capabilities, /v1/models, and the role_infeasible 404 (h7)

### t9 — t9 Docs: rewrite docs/gateway-fleet.md auth section (limitation lifted), extend docs/deployment-shapes.md + docs/colleague-stack.md + docs/openai-api.md with proxy-lobes states, pairwise-key contract (one inbound key per box, one outbound key per peer), marker headers; update lobes/explain catalog + CLAUDE.md; re-verify before-state citations against main. Files: docs/*.md, lobes/explain/, CLAUDE.md

- depends on: t6, t5
- covers: c15, h11
- acceptance:
  - the before-state citations (gateway-fleet.md:376 known limitation, _routing.py NEVER-dialed comment) are updated where the code changed and confirmed-still-true where it did not (h11)
  - docs state the O(machines) key layout and the tailnet transport assumption; no doc claims Orin validation (#108 rule)

### t10 — t10 Live cross-box acceptance (operator-led, post-merge/publish): re-pin Thor off 0.43.0.dev239, enable MULTIMODAL_PEER_PROXY + pairwise keys on Spark, run a senses chat request against Spark's gateway, verify Thor-produced Gemma completion + honest capabilities on both boxes, commit the transcript under docs/evidence/

- depends on: t8, t9
- covers: c1, h8, c17, h13
- acceptance:
  - committed docs/evidence/ transcript shows: request to Spark, X-Lobes-Proxied-By Thor origin, Gemma served id in the response, capabilities honest on both boxes (h13)
  - the offline suite passes in CI on the release commit (h13)

## Risks

- [unknown_nonblocking] Live acceptance (t10) depends on a PyPI release carrying this feature and re-pinning Thor's gateway off the TestPyPI 0.43.0.dev239 pin (challenge park v2) — it runs post-merge, outside the workforce waves
- [unknown_nonblocking] Cross-box peer probes from the readiness cache add WAN-ish latency into a loop that today only probes localhost containers — t4 must isolate cadence/timeouts so a flapping tailnet link cannot stall local readiness (task t4)
- [unknown_nonblocking] t8's integration suite may need to touch shared test fixtures other tasks also edit — merge it last in its wave and re-run the full suite after merge (TDD gate) (task t8)
