# Build Plan — stt readiness truth 155

slug: `stt-readiness-truth-155` · status: `exported` · from frame: `stt-readiness-truth-155`

> lobes never reports a lobe unhealthy while it is correctly serving — STT readiness tells the truth about CPU fallback

## Tasks

### t1 — Make evaluate_readiness device-agnostic in BOTH _readiness.py copies

- instruction: FILES (do not touch others): lobes/realtime/_readiness.py, lobes/templates/fleet/_readiness.py, tests/test_realtime_readiness.py. Keep both copies in sync — cite-don't-import; a drift test is part of the acceptance.
- covers: c19, h2, c20, h7
- acceptance:
  - evaluate_readiness(model_loaded=True, cuda_ok=False) returns (200, body) and the body reports the serving device (e.g. {'status':'ready','device':'cpu'})
  - evaluate_readiness(model_loaded=False, ...) still returns 503 regardless of cuda_ok — the loaded-model gate is intact
  - a test asserts lobes/templates/fleet/_readiness.py and lobes/realtime/_readiness.py expose the same evaluate_readiness contract, so the vendored copy cannot drift
  - the module docstring no longer states 'ready only when model loaded AND CUDA live'; it states the model-loaded gate and the device-reporting rule

### t2 — Report the NeMo model's real device from the Parakeet /v1/health/ready handler

- instruction: FILES: lobes/templates/fleet/listen_server.py only. Derive the device from the loaded model (next(model.parameters()).device); do NOT probe torch.zeros(1, device='cuda') — that reports the process, not the model.
- depends on: t1
- covers: c1, h1
- acceptance:
  - the handler derives the device the model actually loaded onto (e.g. next(model.parameters()).device) rather than probing torch.zeros(1, device='cuda')
  - with CUDA_VISIBLE_DEVICES='' the container answers 200 with device='cpu' and docker reports it healthy
  - a failed/unloaded model still answers 503 and docker reports it unhealthy

### t3 — Split aggregate_audio_ready into per-lane signals while KEEPING a composite

- instruction: FILES: lobes/realtime/audio_facade.py, tests/test_realtime_audio_facade.py. ADD per-lane answers; do NOT delete the composite — /v1/realtime depends on it (boundary c21).
- covers: c21, h14
- acceptance:
  - audio_facade exposes a per-lane readiness answer for stt and for tts independently
  - the composite all-of-them verdict still exists and is what /v1/realtime consumes — the split ADDS, it does not replace
  - unit test: tts not ready + stt ready => stt lane ready, tts lane not ready, composite not ready

### t4 — Complete the realtime composite: probe VAD and the generate lane, not just tts+stt

- instruction: FILES: lobes/realtime/app.py, lobes/realtime/_settings.py. app.py is a 'pragma: no cover' shell — keep decision logic in stdlib modules the offline suite covers. Heed risk r1: bound the peer probe, prefer a cached background probe over a synchronous cross-box dial on a health path.
- depends on: t3
- covers: c22, h15
- acceptance:
  - lobes/realtime/app.py ready() probes Silero VAD availability and the voice generate lane in addition to tts_url and stt_url
  - the generate-lane probe honours a PROXIED peer (MULTIMODAL_PEER_ORIGIN/_PROXY) rather than assuming the lane is local
  - when any one capability is missing the response is 503 and NAMES which one (not_ready lists e.g. ['vad'] or ['generate'])
  - per-lane routes remain available so a caller can ask about stt alone without the composite verdict

### t5 — Gateway: route audio per-lane and make every refusal name its real cause

- instruction: FILES: lobes/gateway/server.py, lobes/gateway/_readiness.py, tests/test_gateway_audio_routing.py. Preserve the existing tri-state (True/False/None) discipline — 'reachable but warming' and 'cannot reach' must stay distinct.
- depends on: t4
- covers: c5, h4, c6, h5
- acceptance:
  - POST /v1/audio/transcriptions is gated on the stt lane alone; POST /v1/audio/speech on the tts lane alone
  - a non-200 body states the actual cause (e.g. stt lane unavailable / model not loaded) and is determinable without docker logs
  - the body never says 'warming up' / 'retry shortly' for a state that has persisted past the container start_period
  - regression test: a backend that answers 200 container-direct is never refused by the gateway for the same payload

### t6 — Give lobes capabilities a LIVE audio probe so it stops contradicting fleet status

- instruction: FILES: lobes/roles.py, lobes/cli/_commands/capabilities.py. Preserve the #81 clamp: an unconfigured overlay is never ready and never has an endpoint, no matter what probe signal is passed in.
- depends on: t4
- covers: c4, h3
- acceptance:
  - the CLI supplies a real audio_ready signal to build_role_registry instead of passing None and falling back to the config fact
  - lobes capabilities and lobes fleet status report the same ready verdict for stt when run against the same box
  - capabilities' verdict CHANGES when the container's real state changes, proving it dials rather than echoing .env
  - with the audio overlay unconfigured, capabilities still reports not-ready with an empty endpoint (the existing #81 clamp is preserved)

### t7 — Ship a scripted degraded-STT live probe and capture the acceptance transcript

- instruction: FILES: scripts/ (new probe script) only — no docs, no evidence. Script must be runnable without a real cgroup regression (simulate via CUDA_VISIBLE_DEVICES=''). The live run is t10.
- depends on: t2, t5, t6
- covers: c16, h13, c13, h10, c14, h11, c12, h9
- acceptance:
  - a script SIMULATES the CPU-fallback condition (CUDA_VISIBLE_DEVICES='' on the stt container) so the fault needs no real cgroup regression to reproduce
  - it captures all four signals in ONE transcript: fleet status, capabilities, container-direct POST, gateway POST
  - it asserts the known-content probe end-to-end: Front_Center.wav through the GATEWAY returns 200 with a transcript containing 'front' and 'center'
  - it asserts the consumer path specifically — the gateway route, not container-direct — since that is what webcam-cli and reachy-mini-cli use
  - it asserts lane independence: with only chatterbox stopped, transcriptions still return 200
  - the transcript lands under docs/evidence/ per the #108 evidence rule

### t8 — Add the after-state test matrix: one failing-before/passing-after test per clause

- instruction: FILES: a NEW test module (e.g. tests/test_stt_readiness_truth.py) only — do not edit test files owned by t1/t3/t5. Record the fail-before/pass-after demonstration in the PR body.
- depends on: t5, t6
- covers: c15, h12
- acceptance:
  - a dedicated test module covers all four after-state clauses: device-agnostic ready, per-lane independence, CLI/gateway agreement, cause-naming refusal body
  - each test is demonstrated to FAIL against the pre-change code and PASS after — recorded in the PR description
  - the module lives in its own test file so it does not collide with the per-task test files edited in earlier waves

### t9 — Update every doc and comment that asserts 'ready implies CUDA live'

- instruction: FILES: docs/parakeet-stt.md, docs/gateway-fleet.md, CLAUDE.md, and the _readiness.py docstring is owned by t1 — coordinate, do not duplicate. Finish with a repo-wide grep proving no 'ready implies CUDA live' claim survives.
- depends on: t1
- covers: c8, h8
- acceptance:
  - docs/parakeet-stt.md, docs/gateway-fleet.md and CLAUDE.md describe the model-loaded gate plus device reporting, not the CUDA gate
  - the docs state that lobes never REPAIRS GPU visibility — no auto-restart, no cgroup manipulation, no device re-attach — and that recovery is operator-initiated
  - the docs record the fresh-container diagnostic (a new --gpus all container seeing the GPU proves the loss is per-container, not host-wide)
  - no doc, comment or test asserting 'ready implies CUDA live' survives a repo-wide grep

### t10 — Run the degraded-STT probe on the live box and commit the acceptance transcript

- instruction: OPERATOR-ONLY — needs the live box; cannot run in a throwaway worktree. Run t7's script on spark, commit the transcript under docs/evidence/. Per #108, no doc may claim validation until this lands.
- depends on: t7
- covers: c13, h10, c14, h11
- acceptance:
  - the probe is executed against a real fleet (host spark) with the stt container forced to CPU, not a mock
  - the run captures all four signals in one transcript: fleet status, capabilities, container-direct POST, gateway POST
  - the transcript demonstrates the consumer path is unblocked — a gateway POST returns 200 with a correct transcript while the gear serves on CPU
  - the transcript is committed under docs/evidence/ per the #108 rule, and no doc claims validation until it lands

## Risks

- [unknown_nonblocking] Probing a PROXIED generate peer inside the realtime readiness endpoint puts a cross-box call on a health path — a slow or flapping peer could stall or flap /v1/health/ready. Needs a bounded timeout and probably a cached background probe (the ReadinessCache/PressureCache pattern already in lobes/gateway/_readiness.py), not a synchronous dial. (task t4)
- [follow_up] Device-agnostic readiness means a silently-CPU-serving container now reads HEALTHY. The GPU regression becomes invisible to anyone who only watches docker health — the reported device is the ONLY remaining signal, so it must surface somewhere an operator actually looks (lobes status / capabilities), or this fix trades a false alarm for a silent degradation. (task t1)
- [follow_up] The readiness code ships INSIDE the stt image, so none of this reaches a live box without an image rebuild and a MODEL_GEAR_VERSION bump. This deployment already skews three ways (gateway 0.52.3, realtime 0.54.0, repo 0.54.1), so a fix can appear merged yet be absent in production.
