# lobes tells clients the truth about a fleet: a consumer can resolve any role from /capabilities, dial the one client-reachable URL the contract names, and get a working answer -- for chat AND audio -- while 'lobes status' reports the fleet that is actually running instead of a phantom single-model container.

> lobes tells clients the truth about a fleet: a consumer can resolve any role from /capabilities, dial the one client-reachable URL the contract names, and get a working answer -- for chat AND audio -- while 'lobes status' reports the fleet that is actually running instead of a phantom single-model container.

## Audience

- The fleet's machine consumers -- chiefly colleague, which resolves cortex/senses/stt/tts from GET /capabilities and dials them to run the model -- plus human operators running 'lobes status' on a fleet deployment.

## Before → After

- Before: Today on a fleet: (a) 'lobes status' shows the single-model view and prints 'state: model-gear-vllm -- not created' next to 'health: ok', contradicting itself and never mentioning senses/embedder/reranker; (b) each role's /capabilities 'endpoint' points at a non-routable host (http://localhost:8000 for cortex/senses, http://realtime:8080 for stt/tts) that 404s, so a consumer must discover-by-probing that only the gateway origin :8001 answers; (c) even the gateway origin 502s for /v1/audio/transcriptions and /v1/audio/speech, so stt/tts advertise ready:true but are not client-consumable through any reachable URL.
- After: On a fleet, 'lobes status' detects the fleet and reports it AS a fleet: per-gear container states via the existing _compose.fleet_containers(deploy_dir) + inspect_state(name) loop (the same building block lobes fleet status uses), the gateway health, and a pointer to 'lobes fleet status' / 'lobes capabilities'. It no longer prints 'model-gear-vllm -- not created' next to 'health: ok'. A genuine single-model deployment is byte-for-byte unchanged. Both human and --json outputs reflect the fleet shape (e.g. a deployment: fleet|single field).

## Why it matters

- /capabilities is a machine-readable contract whose entire purpose is discovery: a consumer resolves a role and dials it without human help. A field that 404s, a status line that contradicts /health, and an advertised-ready audio role that 502s each force every consumer into the same live-probe-then-hardcode-a-workaround cycle -- exactly what the contract exists to prevent. It is the concrete blocker for colleague#286 (senses voice arc).

## Requirements

- Status fleet-detection reuses existing primitives -- no new detection heuristic invented ad hoc: _compose already exposes FLEET_CONTAINERS (now including FLEET_MULTIMODAL), fleet_containers(deploy_dir), audio_overlay_present, and the fleet-only Dockerfile.gateway marker. The fix delegates to the same fleet_containers + inspect_state(name) loop cmd_fleet_status already uses.
  - honesty: The status fleet path calls _compose.fleet_containers(deploy_dir) + inspect_state(name) (the primitives cmd_fleet_status already uses) with a fleet-vs-single detector built from an existing on-disk marker (e.g. Dockerfile.gateway presence / fleet-container existence) -- no parallel ad-hoc heuristic.

## Honesty conditions

- A consumer resolving any of the six roles from /capabilities and dialing the one documented field reaches a working OpenAI endpoint for that role (chat / embeddings / rerank / audio), and 'lobes status' on a fleet names the running gears -- with no field a reasonable consumer would dial pointing at a 404 or unreachable host.
- The fix is verified from the consumers' vantage -- colleague resolving cortex/senses/stt/tts, and an operator running 'lobes status' on a fleet -- not only via unit internals.
- Each of the three documented pre-fix failures (phantom status container; endpoint on a non-routable host; audio 502) is pinned by a regression test asserting the pre-fix wrongness, so the fix demonstrably closes a real gap.
- After the fix a consumer needs no live-probe-then-hardcode workaround: the contract alone yields the reachable URL and truthful readiness, unblocking colleague#286's voice round-trip to move from SKIP toward PASS.
- The diff touches only reporting/reachability/readiness surfaces (status.py, roles.py, the gateway /capabilities route + audio proxy, compose env) and leaves served models, the cortex/senses context/util rebalance, the role set, and the audio backend images unchanged.
- On a live fleet, 'lobes status' reports deployment=fleet with per-gear states + gateway health and never prints 'model-gear-vllm -- not created'; a single-model deployment's output is byte-for-byte unchanged -- both asserted by tests.
- Each listed test exists and passes in CI, and the live GB10 chat+audio confirmation is actually run, not asserted on faith.

## Success signals

- All three issues close with tests + one live confirmation: (a) #84 -- a fleet-shaped 'lobes status' test asserts no phantom 'model-gear-vllm -- not created' line and shows per-gear states, and a single-model test asserts byte-identical legacy output; (b) #87 -- a /capabilities test asserts every role's endpoint is the reachable gateway origin (Host-derived, GATEWAY_PUBLIC_URL-overridable), with none pointing at :8000-when-VLLM_PORT-differs or realtime:8080; (c) #89 -- a readiness test asserts stt/tts 'ready' follows the backend health probe (ready:false while a backend 503s) and the audio proxy returns a clear 503 for a warming backend. Live on the GB10: colleague resolves the roles from /capabilities and round-trips both chat AND audio through the gateway origin.

## Scope / boundaries

- Scope is reachability/reporting truth only: fix 'lobes status' fleet-awareness (#84), make the /capabilities contract name a client-reachable URL and stop advertising non-routable per-role endpoints (#87), and make the advertised-ready audio routes actually round-trip through the reachable origin OR stop advertising them ready (#89). It is NOT a change to the served models, the cortex/senses context/util rebalance, the role set, or the audio backends (Parakeet/Chatterbox); it does not add new external per-role host ports or a new service-discovery mechanism.

## Non-goals

- Not giving each role its own externally-published host:port; routing stays centralized through the single gateway origin. Not adding auth/tunnel changes. Not re-benchmarking or re-tuning the fleet.

## Decisions

- #87 fix: 'endpoint' becomes the ACTUAL client-reachable gateway origin for all SIX roles (it was already meant to be the client-facing gateway address for cortex/senses/embedder/reranker -- just computed with the wrong port; audio leaked the internal realtime:8080). Mechanism (user pick): the gateway's GET /capabilities derives its reachable origin from the request Host header + scheme by default, honoring an explicit GATEWAY_PUBLIC_URL override (tunnels / Host-rewriting proxies). This closes the one call site (server.py:559 capabilities_payload) that omits gateway_url. stt/tts 'endpoint' is routed through the gateway origin (+ /v1/audio/* path), NOT realtime:8080. 'endpoint' stays a real, dial-able field documented as the client base URL; 'path' stays the sub-path; internal upstream URLs (vllm-primary:8000, realtime:8080) are never leaked. The CLI 'lobes capabilities' keeps resolving VLLM_PORT as today; both transports stay shape-identical via the single build_role_registry.
- #89 readiness fix (user pick): redefine stt/tts 'ready'/'loaded' to reflect a LIVE probe of the audio backends' /v1/health/ready (realtime / chatterbox / stt), replacing the bool(AUDIO_URL) config check. ready:true then means a client can actually round-trip audio through the gateway origin. Colleague keys off 'ready' today, so it degrades honestly with zero consumer change. The gateway-fronted roles (cortex/senses/embedder/reranker) keep their current configured-readiness (no issue filed on them; scope not extended to all six).
- #89 proxy hardening (user pick, scope C): beyond honest readiness, add audio-proxy robustness in this PR -- distinguish a warming/not-yet-ready backend (surface a clear 503, not a bare relayed 502) from a genuinely unreachable one, and handle the stale-CUDA-context Chatterbox failure mode (the documented 'TTS backend returned no audio' / poisoned CUDA context that a 'docker compose restart chatterbox' clears). The gateway->realtime->chatterbox/parakeet path itself is structurally correct (verbatim forward, relay-without-failover), so this is robustness/observability, not a re-architecture.
