# Build Plan — lobes tells clients the truth about a fleet: a consumer can resolve any role from /capabilities, dial the one client-reachable URL the contract names, and get a working answer -- for chat AND audio -- while 'lobes status' reports the fleet that is actually running instead of a phantom single-model container.

slug: `lobes-tells-clients-the-truth-about-a-fleet-a-cons` · status: `exported` · from frame: `lobes-tells-clients-the-truth-about-a-fleet-a-cons`

> lobes tells clients the truth about a fleet: a consumer can resolve any role from /capabilities, dial the one client-reachable URL the contract names, and get a working answer -- for chat AND audio -- while 'lobes status' reports the fleet that is actually running instead of a phantom single-model container.

## Tasks

### t1 — #84: make 'lobes status' fleet-aware (status.py + a _compose is_fleet detector)

- covers: c3, c7, c8, h7, h8, h4, c2
- acceptance:
  - On a fleet-deployment fixture, 'lobes status' (human + --json) reports deployment=fleet with per-gear states from _compose.fleet_containers(deploy_dir)+inspect_state(name), shows gateway health, points at 'lobes fleet status'/'lobes capabilities', and contains NO 'model-gear-vllm -- not created' line.
  - On a single-model fixture, 'lobes status' output is byte-for-byte identical to pre-change (regression test guards the legacy path).
  - Fleet detection is a new _compose helper (is_fleet(deploy_dir)) built from an existing on-disk marker (Dockerfile.gateway presence / fleet-container existence) -- no new compose-file parser.

### t2 — #87+#89: roles.py -- audio endpoint via gateway origin + optional live-readiness signal

- covers: h2, h4
- acceptance:
  - _audio_role builds endpoint = <gateway origin> + role path; tests/test_roles.py asserts stt/tts endpoint is the gateway origin (NOT http://realtime:8080), replacing the two verbatim :8080 assertions at test_roles.py:219,295.
  - build_role_registry accepts an optional audio-readiness signal; when supplied, stt/tts ready/loaded reflect it (test: ready=False passed -> ready:false even though audio_url is set); when omitted, falls back to configured bool(audio_url) for back-compat.
  - All six roles keep an identical JSON key set (dataclasses.asdict shape unchanged).

### t3 — #87: wire GATEWAY_PUBLIC_URL through the fleet gateway service (templates/env.example)

- covers: h6
- acceptance:
  - The fleet gateway service passes GATEWAY_PUBLIC_URL through to the gateway container (default derived from VLLM_PORT, overridable); env.example documents it.
  - A template/compose test asserts GATEWAY_PUBLIC_URL is wired on the fleet gateway service.

### t4 — #89: chatterbox /v1/health/ready reports 503 on a poisoned-CUDA / non-synthesizing state

- covers: h4
- acceptance:
  - chatterbox_server /v1/health/ready returns 503 (not 200) when the CUDA context is poisoned / the model cannot synthesize, so a live readiness probe surfaces the stale-CUDA failure honestly; a test stubs a poisoned-context state and asserts 503.

### t5 — #87+#89: gateway server.py -- Host-derived reachable origin, GATEWAY_PUBLIC_URL override, live audio-readiness probe, 503-for-warming

- depends on: t2
- covers: c1, h2, c4, h5, h4
- acceptance:
  - capabilities_payload derives the reachable origin from the request Host header + scheme and passes it as gateway_url to build_role_registry; a gateway test with Host: localhost:8001 asserts every role endpoint origin is http://localhost:8001, not :8000.
  - ServerConfig reads GATEWAY_PUBLIC_URL; when set it overrides the Host-derived origin (test asserts a tunnel URL wins over Host).
  - The gateway live-probes the audio backends' /v1/health/ready and passes the signal into build_role_registry; /capabilities reports stt/tts ready:false while a stubbed backend 503s.
  - handle_audio_post returns 503 ('backend warming/not ready') when the audio backend is up-but-not-ready, distinct from 502 when unreachable; both asserted by tests.
  - GET /capabilities and 'lobes capabilities --json' stay shape-identical (same per-role keys).

### t6 — integration: live GB10 chat+audio round-trip via gateway origin, docs, version bump

- depends on: t1, t3, t4, t5
- covers: c5, h6, c13, h9, h3, c2, h5
- acceptance:
  - Version bumped via the version-bump skill (pyproject.toml + CHANGELOG) so version-check CI passes.
  - Docs updated (docs/colleague-stack.md + docs/openai-api.md) to state endpoint is the client-reachable gateway origin and stt/tts ready is a live health signal.
  - Live on the GB10: colleague resolves cortex/senses/stt/tts from GET /capabilities and round-trips BOTH a chat completion AND audio transcription+speech through the gateway origin (or, if a backend is warming, /capabilities honestly reports ready:false and the round-trip is retried after warmup) -- result recorded, not asserted on faith.
  - The full diff touches only status.py, _compose.py, roles.py, gateway/{server,_config}.py, realtime/chatterbox_server.py, templates/, docs/ and tests -- no served-model/catalog/util change (scope check).

## Risks

- [unknown_nonblocking] Where the live audio-readiness probe lives: gateway-computed (server.py probes the backends' /v1/health/ready and passes a signal into build_role_registry, mirroring fleet_status's existing _metrics.probe_backend pattern) vs roles.py self-probing. RECOMMENDED: gateway-computed -- keeps roles.py pure (it currently does no network I/O; the code comment says liveness is not the dataclass's job). (task t5)
- [unknown_nonblocking] CLI 'lobes capabilities' readiness divergence: the host CLI generally cannot reach the compose-internal realtime:8080 to probe, so it reports configured-readiness while the gateway reports live. Both transports keep an identical KEY shape (the honesty condition), only the 'ready' VALUE may differ by vantage. RECOMMENDED: gateway is colleague's source of truth; document that the CLI reports configured-readiness. (task t2)
- [unknown_nonblocking] Stale-CUDA Chatterbox remedy in this PR: report 503 from /v1/health/ready on a poisoned context (so the gateway probe surfaces it honestly) vs an active self-heal restart. RECOMMENDED: report 503 now (honest readiness); auto-restart is a follow-up issue, not this PR. (task t4)
- [unknown_nonblocking] Scheme detection for the Host-derived origin (http vs https behind a tunnel): the request may arrive http even when the public URL is https. RECOMMENDED: default http from the Host header; rely on the GATEWAY_PUBLIC_URL override for https tunnels / reverse proxies that rewrite scheme. (task t5)
