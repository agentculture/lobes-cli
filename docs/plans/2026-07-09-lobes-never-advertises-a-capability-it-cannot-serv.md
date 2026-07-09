# Build Plan — lobes never advertises a capability it cannot serve: every endpoint, model and role in the machine-readable contract is proven reachable, and a fleet fault degrades to a retryable 503 instead of a terminal 404

slug: `lobes-never-advertises-a-capability-it-cannot-serv` · status: `exported` · from frame: `lobes-never-advertises-a-capability-it-cannot-serv`

> lobes never advertises a capability it cannot serve: every endpoint, model and role in the machine-readable contract is proven reachable, and a fleet fault degrades to a retryable 503 instead of a terminal 404

## Tasks

### t1 — Backend wiring gate: _optional_backend wires a backend ONLY when its *_BASE_URL is set (drop the 'or *_SERVED_NAME' clause), so no phantom backend is ever invented from a default_url

- covers: c19, h24
- acceptance:
  - With MULTIMODAL_CODER_SERVED_NAME and MIDDLE_SERVED_NAME set but their *_BASE_URL empty and no container, build_config wires NEITHER backend
  - resolve_model('nvidia/Qwen3-14B-NVFP4') on such a deployment does not resolve to the primary's weights
  - The packaged fleet env.example (which sets both vars when a profile is on) is unaffected: enabling COMPOSE_PROFILES=middle still wires the middle backend
  - Files touched: lobes/gateway/_config.py, tests/test_gateway_config_wiring.py (new)

### t2 — No cross-backend failover: order_backends returns exactly ONE backend for every input, and the test that currently asserts cross-model failover is inverted

- covers: c23, h21
- acceptance:
  - order_backends(table, served) returns a list of length <= 1 for every input, including tier-alias-resolved names
  - tests/test_gateway_routing.py::test_order_backends_generate_still_failovers_between_generate_backends is INVERTED to assert the new contract, not deleted
  - The static tier_aliases upward fallback (an unwired tier maps to a higher rung at table-build time) is preserved and still tested
  - Files touched: lobes/gateway/_routing.py, tests/test_gateway_routing.py

### t3 — Backend readiness cache: a bounded BACKGROUND probe of each backend's /health, mirroring PressureCache, exposing a cached tri-state per backend and never probing on the request hot path

- covers: c8, h8
- acceptance:
  - A new lobes/gateway/_readiness.py exposes a cache with .current() returning per-backend readiness, refreshed off the request path on an interval
  - A unit test proves .current() opens no socket (injected probe callable, call count asserted zero across N reads)
  - The probe degrades to 'unknown' (never raises) on OSError, http.client.HTTPException and ValueError — the malformed-URL lesson from PR #90
  - The background thread is a daemon and stops cleanly on server shutdown
  - Files touched: lobes/gateway/_readiness.py (new), tests/test_gateway_readiness.py (new)

### t4 — Fleet template truth: inject GATEWAY_PUBLIC_URL (derived from the published VLLM_PORT) and AUDIO_URL into the gateway container from the BASE fleet compose, so the advertised origin is configured truth and stt/tts stop advertising a 404 path (issue #96)

- covers: c11, c18
- acceptance:
  - lobes/templates/fleet/docker-compose.yml gateway service passes GATEWAY_PUBLIC_URL defaulted from VLLM_PORT, and AUDIO_URL, without requiring the -f docker-compose.audio.yml overlay
  - A test parses the packaged fleet compose and asserts the gateway environment contains both keys
  - env.example documents GATEWAY_PUBLIC_URL as the tunnel/proxy override and notes it defaults to the published port
  - An audio-less deployment still yields audio_url unset in the gateway (AUDIO_URL empty), so stt/tts report loaded=false rather than a 404ing ready=true
  - Files touched: lobes/templates/fleet/docker-compose.yml, lobes/templates/fleet/env.example, tests/test_fleet_template_gateway_env.py (new)

### t5 — roles.py: RoleInfo.ready stops being an alias of loaded for the four gateway-fronted roles, and the endpoint is never built from the gateway's internal listen port

- depends on: t3
- covers: c15, c3
- acceptance:
  - build_role_registry accepts a per-backend readiness signal; ready reflects it while loaded stays the config fact — the stt/tts separation from #89, generalised to cortex/senses/embedder/reranker
  - _gateway_base_url no longer returns an absolute URL when only an internal listen port is known: with no gateway_url and no public_url the endpoint is empty, never http://localhost:<GATEWAY_PORT>
  - An unconfigured/unready role is still returned (never omitted, never raises), matching the existing six-roles-always-present contract
  - Files touched: lobes/roles.py, tests/test_roles.py

### t6 — Gateway core: dead owner yields a retryable 503, the readiness cache feeds /v1/models + /capabilities.ready, and reachable_origin prefers configured truth over Host-header inference

- depends on: t1, t2, t3, t5
- covers: c4, h4, c5, h5, c12, h23, c14, h13, h10, h14, h19
- acceptance:
  - handle_post: an owner that refuses, times out, or returns >=500 yields HTTP 503 with a Retry-After header and an OpenAI-shaped error whose type (e.g. backend_unavailable) is distinguishable from both 'model unknown' and the all-backends-down 502
  - A fake-fleet test proves a request naming the cortex model with the primary dead NEVER opens a connection to the multimodal backend (upstream-opener call sites asserted)
  - Race test: a fake fleet lists model M, the owner is then killed, and a completion naming M returns 503 + Retry-After — never 404
  - Converse test: an id that was never in /v1/models is not silently served by the default backend under a different model's weights
  - GET /v1/models is filtered by the cached readiness signal; GET /capabilities .ready reflects it for all six roles
  - reachable_origin(None, None) never fabricates an absolute URL from GATEWAY_PORT; GATEWAY_PUBLIC_URL > Host header > empty
  - The POST hot path opens no probe connection (asserted by call count against an injected opener)
  - Files touched: lobes/gateway/server.py, tests/test_gateway_server.py, tests/test_gateway_capabilities.py

### t7 — CLI truth: lobes capabilities agrees with GET /capabilities byte-for-byte, and lobes doctor detects deployed-gateway version skew

- depends on: t5, t6
- covers: h3, c27, h22
- acceptance:
  - A test asserts GET /capabilities and 'lobes capabilities --json' return identical endpoint/ready/loaded for all six roles on the same deployment config
  - The CLI never reports stt/tts ready=true purely because AUDIO_URL is a string in .env
  - lobes doctor gains a check comparing the running gateway container's lobes.__version__ against the CLI wheel's, failing with severity=error on mismatch and remediation naming the rebuild command
  - Run against the rig as it stands (gateway 0.36.0, CLI 0.39.0) the version-skew check FAILS; after a rebuild it passes
  - Files touched: lobes/cli/_commands/capabilities.py, lobes/cli/_commands/doctor.py, tests/test_cli_capabilities.py, tests/test_doctor.py

### t8 — The senses perception probe: prove coolthor actually PERCEIVES, not merely that the wire accepts content-parts — ground-truth image and ground-truth audio

- depends on: t4, t6
- covers: c7, h7, c17
- acceptance:
  - The image probe generates a solid-colour PNG in-process and asserts the model NAMES that colour (red -> 'red', blue -> 'blue'), not merely that content is non-empty
  - The audio probe synthesizes a known word via the rig's own /v1/audio/speech and asserts the transcription contains it — this only passes once t4 wires AUDIO_URL (issue #96)
  - The existing 1x1-placeholder assertions are replaced or explicitly relabelled as wire-liveness checks, so 'image+text confirmed' never again means 'HTTP 200 with non-empty content'
  - Files touched: tests/test_smoke_duo.py

### t9 — The pre-PR live gate: ONE local command, unattended, that dials every advertised role endpoint+path and the deployed gateway's version, and FAILS rather than skips

- depends on: t4, t6, t7
- covers: c1, h1, c2, h2, c6, h6, c10, h9, c22, h17, h18
- acceptance:
  - A single command runs the whole capabilities live-test to a pass/fail exit code with no prompts and no manual steps
  - When the operator asked for the live gate (the env var / flag is set) an unreachable deployment FAILS the run; it never degrades to pytest-skip
  - For every role in GET /capabilities the gate dials endpoint+path and asserts a non-404 status; for every id in GET /v1/models it asserts a completion never returns 404
  - It reproduces Colleague's discovery path: given ONLY the gateway origin and no COLLEAGUE_*_BASE_URL override, resolve cortex and senses from the contract and get an answer
  - It fails on deployed-gateway version skew
  - Run against today's rig it exits non-zero naming the :8000 endpoint, the 404ing audio path, and the 0.36.0-vs-0.39.0 skew; after redeploy it exits zero
  - The repo's pre-PR convention (CLAUDE.md / the run-tests skill) names this command so it is not silently never run
  - Files touched: tests/test_live_capabilities.py (new), scripts or Makefile target for the single trigger, CLAUDE.md

### t10 — Documentary repairs: record the perception evidence, retire the DSpark route, and stop README claiming lobes init is single-model

- depends on: t9, t8
- covers: h16
- acceptance:
  - docs/gemma-4-12b-nvfp4.md no longer contains the line-279 admission that content-correctness checks were 'not independently re-run against the base checkpoint specifically' — because they were, and the coolthor results are recorded there
  - docs/gemma4-mtp-draft.md carries a superseded banner: DSpark (deepseek-ai/dspark_gemma4_12b_block7) does NOT load on vLLM 0.23 (Gemma4DSparkModel unsupported), per #75 — it must no longer read as 'the ONE route task t3 should wire next'
  - README.md quickstart no longer documents 'lobes init' as scaffolding the single-model deployment on :8000; the duo is the default and --single opts out
  - Files touched: docs/gemma-4-12b-nvfp4.md, docs/gemma4-mtp-draft.md, README.md

## Risks

- [unknown_nonblocking] Requiring *_BASE_URL to wire a backend could break a hand-edited .env that set only *_SERVED_NAME. The packaged fleet template always sets both when a profile is enabled, but an operator's local file may not — needs a release note and possibly a warning path. (task t1)
- [unknown_nonblocking] The background readiness thread lives inside a stdlib ThreadingHTTPServer. Daemon-thread lifecycle, clean shutdown, and behaviour under 'docker compose down' need care — a probe thread that outlives the server or blocks shutdown is a regression. (task t3)
- [unknown_nonblocking] The audio perception probe depends on Chatterbox TTS, which has a recorded history of a poisoned CUDA context (500s cleared only by restarting the container). The probe may need a readiness precondition or a retry, or it will flake for reasons unrelated to senses. (task t9)
- [follow_up] The live gate cannot run in CI (no GPU, no fleet), so nothing structurally forces it to run. If it is only a convention it will be skipped exactly when it matters — which is precisely how #87's fix shipped in 0.38.0 while the rig ran 0.36.0. Consider a pre-push hook or a PR-template checkbox. (task t8)
