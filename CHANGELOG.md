# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.40.3] - 2026-07-13

### Added

- Plan (`/spec-to-plan`, converged): **per-machine hardware profiles** —
  `docs/plans/2026-07-13-lobes-fits-the-machine-it-lands-on-one-command-det.md`.
  Thirteen tasks over four file-disjoint dependency waves (fifteen drafted, two
  rejected during convergence): the **per-chip strategy
  pattern** + profile schema + spark/thor profiles, card detection, template
  parameterisation and per-role correctness probes (wave 1); `init` applies the
  profile and role-feasibility reaches capabilities/gateway (wave 2);
  `doctor`/`status` report the profile, and the Thor profile is validated on the
  physical board (wave 3); docs + an honest support table (wave 4).

  Per-chip knowledge goes behind a **strategy pattern** — one module per chip
  (`lobes/machines/<chip>.py`) owning its own detection signature, per-role knobs
  and provenance, plus a small shared registry. Adding a chip is one new file and
  one registration line; it must not mean editing shared tables, and a change for
  one chip must not be able to break another. No existing code is deleted:
  `MachineProfile`, `MACHINE_PROFILES` and `detect_machine()` keep working,
  rebuilt *from* the registry rather than duplicated.

  On Thor the reranker stays **served and advertised** — it runs, it is simply
  not yet correct; its ordering probe is recorded as a known failure pointing at
  #105 / #106, rather than the role being hidden.

- Spec + plan rework (same frame, re-converged): per-machine knowledge is keyed
  by **causal capability trait**, not board name — of the four Thor hand-edits,
  three trace to `sm_110` (the FLASH_ATTN pooling hang, the CUDA-graph classify
  fault) and one to the checkpoint's missing KV scales; none traces to "Thor the
  board". A machine profile becomes a named validated *bundle* (detection
  signature + memory budget + model-per-role + applicable traits), so an
  unrecognised board sharing a trait inherits its fix. Three new plan tasks:
  **golden rendered compose/.env per shipped profile** byte-diffed in CI, so a
  change for one machine cannot silently alter another's rendering (t13); an
  **unknown card warns and serves a conservative small base** — no 27B — instead
  of refusing, folding the unknown-card slice of #107 in (t14); and **upgrading
  lobes-cli never breaks an existing scaffold** — zero bytes changed in the
  deployment dir, old env names honoured, re-init always diffed + `--apply`
  (t15). Two GB10 verifications are parked as tracked risks: whether the fp8-KV
  crash is checkpoint-driven (shared fix) rather than Thor-specific, and whether
  the `VLLM_ATTENTION_BACKEND` env is truly dead on the GB10's pinned image
  before t3 deletes it. Packaging per #107: profiles ship in the wheel, no pip
  extras.

### Changed

### Fixed

- Spec correction: the first cut of this spec claimed lobes had no machine-profile
  concept at all. It does — `lobes/profiles.py` already ships `MachineProfile` /
  `MACHINE_PROFILES` (spark/thor/blackwell/generic) and `detect_machine()`, wired
  to `VLLM_MACHINE` and used by `switch`/`benchmark`. The real gaps, now stated
  honestly in the spec's before-state: it is one knob-set **per machine, not per
  role**; it lacks the knobs that actually mattered on Thor (KV-cache dtype,
  enforce-eager, model-per-role, role feasibility); the **fleet** compose (the
  default path) ignores it entirely and hardcodes the Spark values; its `thor` row
  is an unvalidated guess (`status="configured"`: flashinfer / 32768 / util 0.6)
  that live Thor testing **contradicts**; and `detect_machine()` silently falls
  back to `generic` instead of admitting it does not know the card.

## [0.40.2] - 2026-07-13

### Added

- Spec (`/think`, converged): **per-machine hardware profiles** —
  `docs/specs/2026-07-13-lobes-fits-the-machine-it-lands-on-one-command-det.md`.
  lobes detects the host card and applies a profile tuned for *that* box
  (feasible roles + model per role + util / context / quantization / KV dtype /
  attention backend / enforce-eager), instead of the fleet compose's hardcoded
  GB10 values. Ships Spark (default) + Thor as supported;
  Orin / Orin Nano Super are named but unvalidated; an unrecognised card
  refuses-or-warns rather than silently applying the Spark profile.
  Every role gains a **correctness** probe, not just `/health` — a role that is
  healthy but semantically wrong must fail.

  Motivated by bringing the fleet up on a Jetson AGX Thor (sm_110): the
  Spark-tuned template scored **1 of 4 roles correct on first boot** (senses
  clean; cortex crash-looped on an fp8-KV assert; embedder accepted requests and
  never answered; reranker killed its engine with `cudaErrorLaunchFailure`), and
  reaching 3 of 4 took four hand-edits to the generated compose. Rerank is still
  wrong there — tracked in #105 / #106.

### Changed

### Fixed

## [0.40.1] - 2026-07-11

### Added

- Real-socket streaming regression test: a dribbling chunked-SSE upstream through the real `open_upstream` asserts the first relayed frame arrives while the upstream is still mid-stream — the fake-upstream tests (whose `read()` already had read1 semantics) could never catch this bug class.
- Spec + plan for the fix under `docs/specs/` and `docs/plans/` (devague frame `lobes-gateway-relays-sse-streams-frame-by-frame-th`).

### Fixed

- Gateway SSE streaming: `_Upstream.read` now uses `HTTPResponse.read1`, so `data:` frames relay to the client as the backend generates them. Previously `HTTPResponse.read(65536)` blocked until 64 KiB or EOF, releasing a whole streamed turn in one terminal burst (frames=21 first=last=3.06s through the gateway) — full-turn latency before the first visible token for any `stream: true` client (issue #103; reported by colleague, agentculture/colleague#318).

## [0.40.0] - 2026-07-09

### Security

- The `Host` request header is validated against a strict host-authority allowlist before it is echoed as an advertised origin in `GET /capabilities`. Previously the c29 origin-resolution change reflected an unsanitized, client-controlled `Host` header into every role's `endpoint` (e.g. `http://evil.example/../<script>` or a `user@host` authority), so a client scraping the contract could be handed an attacker's origin to dial. An invalid host is now treated as "no origin supplied" (empty endpoint), exactly as an absent header is. The trusted operator override `GATEWAY_PUBLIC_URL` is unaffected. (SonarCloud `pythonsecurity:S5131`.)

### Added

- `scripts/live-check.sh` + `tests/test_live_capabilities.py` — a LOCAL, single-trigger, unattended pre-PR gate that dials every advertised role endpoint+path, every id in `/v1/models`, checks CLI/gateway agreement, reproduces Colleague's role-discovery path, and fails on deployed-gateway version skew. It FAILS rather than skips when armed. A 429 (pressure shed) and a 503 carrying `Retry-After` count as reachable; only a 404, a connection failure, a `Retry-After`-less 503, or a bare 5xx are faults. that dials every advertised role endpoint+path, every id in `/v1/models`, checks CLI/gateway agreement, reproduces Colleague's role-discovery path, and fails on deployed-gateway version skew. It FAILS rather than skips when armed. A 429 (pressure shed) and a 503 carrying `Retry-After` count as reachable; only a 404, a connection failure, a `Retry-After`-less 503, or a bare 5xx are faults.
- `lobes/gateway/_readiness.py` — a bounded background probe of each backend's `/health`, mirroring `PressureCache`. Tri-state (`True` healthy / `False` reached-but-unhealthy / `None` unreachable), daemon thread, socket-free `.current()`, and a probe that degrades to `None` on `OSError`, `http.client.HTTPException` and `ValueError`.
- `GET /health` now reports `{"version": ...}`, and `lobes doctor` gains a `gateway_version_match` check that fails on skew between the deployed gateway and the CLI wheel (issue #99).
- `lobes/gateway/_routing.py::is_unknown_model` — a pure predicate separating "unknown model id" from "unspecified model".
- Ground-truth perception probes for the `senses` role: a stdlib-generated solid-colour PNG whose colour the model must name, and a Chatterbox-synthesized word the model must transcribe. Both carry negative controls; the old placeholder-media tests are relabelled as wire checks.

### Changed

- **No cross-backend failover.** `order_backends` now returns at most one backend. A request naming the cortex model can never be answered by the Gemma backend, which protects the `final_authority` role contract from #81.
- `GET /v1/models` and `GET /capabilities.ready` are backed by the live readiness cache rather than by configuration. A wired-but-dead backend is no longer advertised.
- `RoleInfo.ready` is no longer an alias of `loaded` for `cortex`/`senses`/`embedder`/`reranker`. `build_role_registry` self-enforces the invariant: a supplied `backend_ready` map is authoritative, and a present `None`, a present `False`, and a missing key all mean not-ready.
- `lobes capabilities` / `lobes endpoint` now render the running gateway's `GET /capabilities` when it is reachable, falling back to an offline `.env`-derived view with `ready=false` on every role. The CLI and the gateway can no longer disagree, because there is now one derivation instead of two. The `--json` payload is keyed strictly by role name in both modes (byte-identical to the gateway's payload in live mode); the live-vs-offline signal travels on stderr (a `source:` notice) rather than as an extra top-level key, so a strict `set(keys) == ROLES` consumer never trips (Qodo).
- A backend is wired only when its `*_BASE_URL` is set; the `or *_SERVED_NAME` clause is gone (issue #97).
- `senses` is documented as vision-only intake. The checkpoint declares audio support but vLLM's `gemma4_unified` path does not serve it (issue #101); `stt` remains the supported speech path.
- `docs/gemma4-mtp-draft.md` carries a superseded banner: DSpark does not load on vLLM 0.23 (#75). Issue #69's disabled-experiment-entry criterion is closed answered-negative.
- README's quickstart no longer claims `lobes init` scaffolds the single-model deployment; the fleet duo has been the default since #69.

### Fixed

- **#91** — a dead cortex backend no longer surfaces as a terminal `404 model does not exist`. `handle_post` rewrote the model id once, before the failover loop, then retried the same body against a backend serving a different model, which correctly 404'd; the `4xx = client error, no failover` rule relayed that as terminal and killed multi-step agent loops. A dead, unreachable or warming owner now yields **503 + `Retry-After`** with `type: backend_unavailable`.
- **#92 / #95** — the gateway never advertises an origin built from its internal listen port. Precedence is `GATEWAY_PUBLIC_URL` (an operator override for a tunnel or Host-rewriting proxy) > the request `Host` header > an empty endpoint. `GATEWAY_PUBLIC_URL` is deliberately NOT defaulted: a defaulted `public_url` outranks `Host` and would advertise loopback to every remote client.
- **#96** — `AUDIO_URL` now reaches the gateway from the base fleet compose, so `stt`/`tts` stop advertising `ready=true` on a path that 404s when the audio overlay is not composed in.
- **#97** — `GET /v1/models` no longer advertises phantom backends wired from a `*_SERVED_NAME` alone against a `default_url` naming a container that need not exist.
- `ReadinessCache` seeds its initial per-backend snapshot with `dict.fromkeys` instead of a dict comprehension (SonarCloud `python:S7519`).
- `ReadinessCache.stop()` no longer clears its thread reference while the refresh thread is still alive, so a subsequent `start()` cannot spawn a second concurrent refresh loop; the join bound now covers a full sequential refresh pass and `start()` cleanly replaces a genuinely-dead thread (Qodo reliability).
- An unknown model id returns `404 model_not_found` instead of being silently served by the default backend under a different model's weights. Unknown-ness is decided against the routing table, never against the readiness-filtered `/v1/models` list, so a wired-but-dead backend still yields 503 rather than 404.
- `test_live_main_text_returns_nonempty_content` no longer fails on a thinking model: `max_tokens=16` was consumed entirely by the reasoning trace, leaving `content=None` and `finish_reason=length` (issue #93).

## [0.39.0] - 2026-07-07

### Added

- **`preserve_thinking` on the cortex/main vLLM lane** — both compose templates
  now launch the primary/cortex service with `--default-chat-template-kwargs
  '{"preserve_thinking": true}'` next to `--reasoning-parser=qwen3`, so the
  served Qwen3.6 chat template retains **all** historical `<think>` blocks across
  a multi-turn conversation (default keeps only the reasoning after the last user
  turn). Default-on but per-request overridable; scoped to the cortex/main lane
  (embed/rerank/senses untouched, `lobes route` still forces
  `enable_thinking=false`). Closes #93.
- **Reasoning-aware `lobes.minor` client** — `assistant_turn_from_response()`
  builds an assistant history message preserving the `reasoning` /
  `reasoning_content` trace, and a new `history=` parameter on `chat_completion`
  / `chat_text` round-trips it back on subsequent turns (single-turn behaviour
  unchanged). Part of #93.
- **`lobes assess --preserve-thinking`** — a read-only two-turn token-delta
  diagnostic that proves the reasoning round-trip is live (prompt-token count
  rises when the assistant `<think>` history is preserved vs content-only).
  Part of #93.

## [0.38.0] - 2026-07-04

### Added

- `GET /capabilities` now advertises a **client-reachable** origin for every role's `endpoint` — derived from the request `Host` header, overridable with the new optional `GATEWAY_PUBLIC_URL` env (for tunnels / Host-rewriting proxies) — so a consumer can dial any role's `endpoint` directly instead of a non-routable internal host (`vllm-primary:8000`, `realtime:8080`). This covers all six roles, including `stt`/`tts`. Closes #87.
- The realtime bridge exposes `GET /v1/health/ready`, aggregating Chatterbox (TTS) + Parakeet (STT) readiness into one signal the gateway live-probes.

### Changed

- `lobes status` is now **fleet-aware**: on a fleet deployment it reports per-gear container states and points at `lobes fleet status` / `lobes capabilities`, instead of the contradictory single-model `state: model-gear-vllm — not created` line printed next to `health: ok`. A single-model deployment's output is byte-for-byte unchanged. Closes #84.
- `GET /capabilities` reports `stt`/`tts` `ready` from a **live** probe of the audio backend (not merely `AUDIO_URL` being set), so an advertised-ready audio role is genuinely consumable; `lobes capabilities --json` keeps the configured signal since the host CLI can't reach the internal backends. Closes #89 (readiness).

### Fixed

- The gateway returns a clear **503** (`Retry-After`) for `/v1/audio/transcriptions` and `/v1/audio/speech` when the audio backend is reachable but still warming (Chatterbox/Parakeet loading, or a poisoned-CUDA context now surfaced honestly by Chatterbox's `/v1/health/ready`), distinct from the **502** for a genuinely unreachable backend — closing the "advertised ready but 502s / not client-consumable" gap. Closes #89.
- The gateway's audio-readiness probe (`probe_audio_ready`) now degrades a malformed `AUDIO_URL` (a non-numeric port makes `urlsplit(...).port` raise `ValueError`) or a broken HTTP exchange (`http.client.HTTPException`) to "readiness unknown" instead of letting the exception crash the `GET /capabilities` / `POST /v1/audio/*` handler — mirroring `open_upstream`'s guard. (Qodo review, #90.)
- `build_role_registry` clamps the stt/tts `ready` signal on the audio overlay being configured, so an unconfigured overlay can never report `ready=True` with an empty endpoint even if a caller passes `audio_ready=True` — the public builder now enforces the "unconfigured ⇒ not ready" invariant its docstring already promised. (Qodo review, #90.)
- `cmd_status` is split into `_cmd_status_fleet` / `_cmd_status_single` render helpers (mirroring the existing `_cmd_status_pressure`), leaving the verb as pure dispatch — dropping its Cognitive Complexity from 16 to under the gate's 15. Each helper uses a single-`return` `if/else` (like `_cmd_status_pressure`) rather than an early return, so it renders its one exit path without tripping `python:S3516` ("always returns the same value"). Behavior-preserving; the fleet and single-model outputs are byte-for-byte unchanged. (SonarCloud `python:S3776` + `python:S3516`, #90.)

## [0.37.0] - 2026-07-03

### Changed

- Under swap/iowait pressure the gateway now **sheds** a `main`/`cortex` or `multimodal`/`senses` request with HTTP 429 + `Retry-After` (busy — retry shortly) instead of silently degrading it onto a different model. The degrade-to-minor path is removed outright (no `LOBES_PRESSURE_POLICY` toggle); an explicit `minor` request is still served as the floor. Busy is disclosed via `X-Lobes-Tier-Reason: busy` and an OpenAI-shaped `server_busy` error body, distinct from the hard 502 `upstream_unavailable`; callers (the acp `vllm-local` provider, colleague, generic OpenAI SDKs) must retry with backoff. `lobes status --pressure` and the gateway `GET /status` now report the busy-policy state (`mode`/`shed`/`retry_after`). The trigger reuses the existing swap/iowait signal; the tunable thresholds (#86) and the `/proc` sampler are untouched.

### Fixed

- Pressure no longer crosses capabilities: on a default fleet with `minor` unwired, a pressured `cortex`/`main` request previously fell through the tier upward-fallback onto the Gemma multimodal gear (a different capability), silently answering an authoritative-reasoning request with a perception model. Removing the degrade path makes that cross-capability substitution structurally impossible. Closes #85.

## [0.36.2] - 2026-07-03

### Fixed

- Fleet gateway now receives the pressure-policy thresholds (`LOBES_SWAP_DEGRADED_THRESHOLD` / `LOBES_IOWAIT_DEGRADED_THRESHOLD`) via its compose `environment`, so operators can tune the swap/iowait degrade triggers per box through `.env`. Previously those vars were only read from the gateway container's env, which the compose never populated, so the knobs silently stayed on the code defaults (75 / 50). Notably lets a box with an unreliable `/proc/pressure/io` (phantom high iowait on an idle disk, e.g. the DGX Spark GB10) raise `LOBES_IOWAIT_DEGRADED_THRESHOLD` to 100 instead of permanently degrading the generate lane. Documented in `env.example` + regression test added.

## [0.36.1] - 2026-07-03

### Fixed

- Gateway GET /capabilities (and `lobes capabilities` when read via the gateway) reported catalog-native context instead of the served `--max-model-len`, because the gateway container's environment lacked `PRIMARY_`/`MULTIMODAL_`/`EMBED_`/`RERANK_MAX_MODEL_LEN`. The fleet compose now passes those into the gateway service so the #81 served-context overlay resolves (cortex 131072, senses 32768); regression test added.

## [0.36.0] - 2026-07-03

### Added

- cortex/senses role-based Colleague contract: six first-class roles (cortex, senses, embedder, reranker, stt, tts) with responsibilities/forbidden_responsibilities (#81)
- `lobes capabilities [--json]` and `lobes endpoint <role>` — read-only role discovery over the role registry
- Gateway GET /capabilities — machine-readable {role: {endpoint, model, context, ready, responsibilities, ...}} contract for Colleague
- `lobes up <role>` and `lobes up colleague-stack` — role-based serving (dry-run by default, --apply to run; colleague-stack bundles the audio overlay)
- lobes measure [--role] [--json] — per-role runtime-only metrics (TTFT/decode-tps/prefill, docs-per-sec, RTF)
- lobes benchmark --profile {cortex-only,cortex+senses,senses-direct,qwen-nvfp4-vs-bf16,all} — comparison profiles (runtime-only)
- lobes explain roles (aliases: colleague, colleague-stack, capabilities) and docs/colleague-stack.md

### Changed

- Fleet context rebalance: cortex (Qwen 27B MTP) served at 128K, senses (Gemma 4 12B) at 32K (util 0.14, provisional); default fleet budget 0.30+0.14+0.06+0.06 = 0.56
- cortex->primary and senses->multimodal added to catalog.TIER_ROLE as the primary contract; main|multimodal|hard|normal|cheap|minor kept as back-compat aliases; brain forbidden
- lobes fleet status now reports the always-on Gemma (senses) container (FLEET_MULTIMODAL added to FLEET_CONTAINERS)

### Fixed

- Latent circular import between lobes.roles and lobes.gateway surfaced when lobes.roles was imported first

## [0.35.0] - 2026-07-02

### Added

- catalog: coolthor/gemma-4-12B-it-NVFP4A16 (Gemma 4 12B NVFP4 base it-model) as the new default multimodal gear, native MTP wired ON by default ({"method": "mtp", "model": "google/gemma-4-12B-it-assistant", "num_speculative_tokens": 1}) -- measured 28.6 tok/s decode @ 57.9% draft acceptance, the fastest Gemma config on the DGX Spark (docs/vllm-nightly-migration.md §7)
- fleet compose: opt-in vllm-multimodal-coder service (profiles: [multimodal-coder]) so the demoted coder gear stays reachable; gateway wires an opt-in MULTIMODAL_CODER_BASE_URL/MULTIMODAL_CODER_SERVED_NAME backend + a multimodal-coder alias, added only once wired

### Changed

- catalog: sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4 demoted from role_hint=multimodal to role_hint=candidate (kept, cite-don't-delete) -- native MTP measured only 30.8% draft acceptance on the coder fine-tune, not worth wiring by default
- gateway: _DEFAULT_MULTIMODAL now points at coolthor/gemma-4-12B-it-NVFP4A16; the multimodal/normal tier aliases resolve to it
- docs/gemma-4-12b-nvfp4.md, README.md, docs/gateway-fleet.md, docs/qwen3-14b-nvfp4.md updated to describe both Gemma gears (default base + opt-in coder) and cite docs/vllm-nightly-migration.md §7 for the benchmark evidence

## [0.34.1] - 2026-07-01

### Added

- docs/gemma-4-12b-nvfp4.md — first throughput/prefill benchmark for the Gemma 4 12B multimodal gear on the DGX Spark GB10: ~23 tok/s single-stream decode (23.0 sustained over 1,500 tok), prefill ~2,650 tok/s (847 tok) / ~1,954 tok/s (6,682 tok), on vLLM 0.23.1rc1.dev672 native gemma4_unified
- README acknowledgement of Mieszko Syty (FutureProofHomes; Jetson AI Lab) alongside shahizat

### Changed

- docs/gemma-4-12b-nvfp4.md notes a config-drift follow-up: on the current :nightly-audio (dev672) image the default lane util 0.12 + VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 no longer boots 8192 co-resident (cudagraph accounting changed) — benchmarked at util 0.15 with a trimmed cudagraph capture set

### Fixed

- docs/gemma-4-12b-nvfp4.md — made the #75 speculative-decoding section internally consistent: it now reads as CLOSED (route resolved, wire/measure/verdict not implemented) throughout, matching the "Resolved" bullet, instead of framing #75 as active work (Qodo); also corrected the stale claim that the 12B lane decodes slower than the primary — the benchmark shows it out-decodes the primary single-stream (~23 vs ~18–19 tok/s)

## [0.34.0] - 2026-07-01

### Added

- Gemma 4 12B multimodal gear now SERVES (text + image + audio) — live-validated on the DGX Spark GB10 via vLLM nightly's native gemma4_unified class (#71/#73); catalog status promoted configured → load-tested

### Changed

- Dockerfile.vllm-gemma4 rebased FROM vllm/vllm-openai nightly (pinned by digest) + the vllm[audio] extra (librosa==0.11.0 soundfile==0.14.0 av==17.1.0 soxr==1.1.0, pinned to the live-validated set) (was NGC 26.06 / vLLM 0.22.1 + a transformers overlay); vllm-multimodal compose/env now set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 and default MULTIMODAL_MAX_MODEL_LEN 8192 (co-resident util-0.12 KV holds ~24K tokens, not the 128K native)

### Fixed

- Gemma 4 12B serve blocker root-caused and fixed: gemma4_unified has heterogeneous per-layer head sizes (40 sliding@256 + 8 full@512) that only vLLM's native class handles — released vLLM ≤0.22.1 fell back to the transformers backend and crashed the full-attention o_proj (4096≠8192); a TRITON_ATTN backend flag does not fix it

## [0.33.1] - 2026-07-01

### Added

- Issue #75 spec + plan (Gemma 4 12B gear speculative decoding) converged via devague /think + /spec-to-plan
- docs/gemma4-mtp-draft.md — resolved spec-decode draft route (DSpark draft_model first; native google/gemma-4-12B-it-assistant recorded as escalation candidate)

### Changed

- docs/gemma-4-12b-nvfp4.md — added the Speculative decoding (#75) before-state / gap / scope-split subsection

## [0.33.0] - 2026-06-30

### Added

- Custom vLLM image Dockerfile.vllm-gemma4 (FROM nvcr.io/nvidia/vllm:26.06-py3, vLLM 0.22.1, + a pinned from-source Transformers 181beb3) so the Gemma 4 12B gemma4_unified multimodal gear loads (#71)
- MULTIMODAL_IMAGE override for the vllm-multimodal service (local build by default; optional ghcr.io/local-registry tag)
- MULTIMODAL_ATTENTION_BACKEND env (TRITON_ATTN) for Gemma 4 non-square attention

### Changed

- Multimodal gear quantization corrected to compressed-tensors (was modelopt_fp4) after live validation (#71)
- Removed the invalid gemma4_mtp speculative_config from the multimodal gear (vLLM Gemma4 MTP needs a separate gemma4_assistant draft model)

### Fixed

- docs/compose comments now name the correct base image and MULTIMODAL_IMAGE registry semantics

## [0.32.1] - 2026-06-30

### Added

- docs/specs: Gemma 4 12B multimodal-duo spec (issue #69) — /think frame for default-serving the Qwen3.6-27B-MTP + Gemma4-12B duo as main/minor/multimodal tiers (vision+audio), native-MTP on by default, DSpark draft as a disabled experiment, 14B demoted to a legacy candidate
- docs/plans: buildable plan for the Gemma duo (9 tasks across 5 dependency waves, 6 accepted-risk objects) via /spec-to-plan — covers all 26 spec targets; resolves the main/minor/multimodal pressure-ladder seam as a first-class task

### Changed

- Reworded the three Gemma-risk markers in `catalog.py` / `runtime/_parser.py` from bare `TODO(risk …)` comments to `Risk … (pending #71)` — the deferred live-validation work is tracked in issue #71 (gemma4_unified won't load on released vLLM images), so the comments now cite the tracking issue instead of an untracked TODO (clears SonarCloud `python:S1135`).

### Fixed

- Gateway: re-wire the legacy 14B `middle` generate backend from `MIDDLE_BASE_URL` / `MIDDLE_SERVED_NAME` in `build_config()`. The #69 14B demotion dropped the wiring but the compose template still ships the `vllm-middle` profile + those env vars, so enabling the profile silently fell back to the primary; the 14B is again reachable by its explicit served name (and, as intended, gets no tier alias). (Qodo)
- Gateway: a `GATEWAY_ALIASES` operator override keyed by a *legacy* tier alias (`hard`/`cheap`/`normal`) is now honoured on the pressure-aware tier path. Tier requests normalize to the new vocabulary (`hard`→`main`) before the alias lookup, which bypassed a legacy-keyed override; `build_config()` now mirrors a tier-keyed override onto its vocabulary synonyms (explicit keys still win). (Qodo)

## [0.32.0] - 2026-06-30

### Added

- Third capability tier: opt-in `vllm-middle` 14B-NVFP4 generate gear (`COMPOSE_PROFILES=middle`, GPU mem-util 0.12), inference-only (not a LoRA base).
- Gateway capability-tier aliases — callers send `model=cheap|normal|hard` and the gateway resolves to the 4B/14B/27B generate gears (same-task alias on top of task-family routing) with upward fallback when a tier is absent.
- Read-only host memory-pressure sampler (`swap_used_percent`/`iowait_percent` from /proc) and a swap/iowait pressure policy with a degraded-mode state machine (env-overridable thresholds).
- Pressure-aware tier downgrade at the gateway with an `X-Lobes-Override` bypass header; the served tier and reason cross the OpenAI boundary via `X-Lobes-Tier` / `X-Lobes-Tier-Reason` response headers.
- `lobes status --pressure` — read-only snapshot of the current tier ceiling, mode, reason, and live swap/iowait.
- `scripts/validate-tiers.sh` + `docs/validate-tiers.md` — operator-run live validation harness for the three-tier fleet on the Spark.

### Changed

- 27B primary default served context trimmed 256K→128K (`PRIMARY_MAX_MODEL_LEN=131072`) and `PRIMARY_GPU_MEM_UTIL` lowered to 0.45 so the co-resident 14B middle gear fits within the 128GB unified-memory budget (0.45 + 0.12 + 0.10 + 0.06 + 0.06 = 0.79).

## [0.31.1] - 2026-06-27

### Changed

- Mutation-safety prose in `lobes learn` and CLAUDE.md now lists the `fleet up` / `fleet down` write verbs (was only in the `--json` payload).
- CLAUDE.md documents the turn-on/turn-off lifecycle explicitly (`serve`/`stop` and `fleet up`/`down`) instead of leaving it implicit in the verb names.

## [0.31.0] - 2026-06-26

### Added

- `lobes benchmark --all-lobes --concurrency auto`: per-lobe (minor + primary) performance benchmark routed through the gateway — single-stream decode tok/s, prefill TTFT, concurrent throughput with auto-ramp to the throughput knee (req/s + p50/p95 latency + ms/token), plus the logprobs cat soft-score, rendered as one combined minor-vs-primary report with per-metric deltas.
- `lobes eval cat --score logprobs --mode open|closed`: read-only 'Where is the cat?' temporal-reasoning probe, scored by logprobs (softmax over candidate-location full-sequence echo logprobs as the headline, with a chat first-token-mass cross-check and graceful fallback when echo is unavailable).
- `lobes.bench` package: `cat_probe` (deterministic, seeded timestamped-narrative generator with exactly one unambiguous current location; open + closed modes), `cat_score` (echo-softmax headline scorer + first-token cross-check + fallback), and `report` (per-lobe markdown report renderer with minor-vs-primary deltas).
- `lobes.minor` logprobs plumbing: `chat_completion` now forwards `logprobs`/`top_logprobs`; new `completions_echo` (full-sequence `/v1/completions` echo scoring) and `gateway_supports_echo` capability probe (never raises; lets callers fall back).
- `lobes.assess` per-lobe perf engine: `measure_prefill_ttft`, `run_concurrent` (requests/sec + p50/p95 latency + ms/token), and `auto_ramp_concurrency` (1→2→4→… ramp with plateau/knee detection).

## [0.30.0] - 2026-06-26

### Added

- **The `minor` lobe — a cheap, warm co-resident Qwen3.5-4B small-brain** (issue
  #64). A new switchable catalog gear `Qwen/Qwen3.5-4B` (`role_hint="minor"`,
  served **bf16** — the first unsloth-LoRA fine-tune target; multimodal, served
  text-only via `--language-model-only`), reachable both as a switchable gear and
  as an opt-in warm co-resident backend alongside the 27B primary.
  - **New read-only verbs:** `lobes run minor "<prompt>"` (call the minor model),
    `lobes route "<text>"` (classify a task across catalog gears with an
    escalate flag + confidence), and `lobes eval minor --suite <path>` (run a
    JSONL eval suite). All three default `--base-url` to the gateway
    (`http://localhost:8000/v1`) and reuse a new stdlib-only urllib client
    (`lobes.minor`) — no new runtime dependencies.
  - **Governance + escalation** (`lobes.minor.governance`): the minor role may
    prepare/classify/format/validate/suggest/summarize/route, and escalates on
    forbidden actions (approve/finalize/delete/deploy/architectural) or any of
    five escalation conditions. Role-keyed, not model-keyed.
  - **Warm co-residency:** an opt-in `vllm-minor` fleet service (compose profile
    `minor` + `MINOR_BASE_URL`/`MINOR_SERVED_NAME` gateway env gate); the gateway
    routes the minor model id to it with failover to the primary. Default fleet
    behavior is unchanged.

### Changed

- **`runtime/_parser.py` recognizes the Qwen3.5 family** → `qwen3_coder` (it
  emits the XML function-call format, not Hermes JSON).
- **Catalog supports an unquantized bf16 generate gear** via a `quantization="none"`
  sentinel that `lobes switch` normalizes to "omit `--quantization`" and surfaces
  as a required compose edit.

### Fixed

## [0.29.0] - 2026-06-24

### Added

- **Memory-discipline "Conventions and workflow" section in `CLAUDE.md`** — a
  per-task *recall-before / remember-after* convention (scope localized to this
  repo's nick) so the vendored `remember` / `recall` skills are actually used,
  not just present: `/recall` before non-trivial work to build on prior
  decisions instead of re-deriving them, and `/remember` when a non-obvious
  decision, constraint, fix-and-why, or hard-won gotcha surfaces. The section
  documents this repo's memory as **in-repo and public** — records resolve to
  `<repo-root>/.eidetic/memory` (committed, team- and mesh-shared). Inserted
  idempotently (skipped if already present), slotted under an existing
  "Conventions and workflow" heading when one exists, else appended.

### Changed

- **Refreshed the `remember` + `recall` wrappers from eidetic-cli 0.10.0**
  (cite-don't-import) — picks up eidetic's **project-local store default**: the
  files backend now resolves per record by visibility — PUBLIC records inside a
  git repo go to `<repo-root>/.eidetic/memory` (committed, team-shared), PRIVATE
  records (or any record outside a repo) go to `$HOME/.eidetic/memory` (never
  committed), an explicit `EIDETIC_DATA_DIR` still wins, and recall reads both
  stores and merges. Also carries the 0.9.3 hardening (interactive-stdin guard,
  `help` as a search term, SIGPIPE-safe suffix parsing). **Recipe policy
  override (the wrappers here are NOT byte-verbatim):** the injected default
  visibility is flipped from eidetic's `private` to **`public`**, so a plain
  `/remember` lands the note in `./.eidetic/memory` in this repo, kept as part
  of the repo — pass `--visibility private` to route a record to `$HOME`
  instead. `remember` drives `eidetic remember` (idempotent upsert of one JSON
  record or an NDJSON batch on stdin); `recall` drives `eidetic recall` with
  four search modes (exact / approximate / keyword / hybrid). Each `SKILL.md` is
  localized only in the illustrative `--scope <nick>` examples (Provenance keeps
  "First-party to eidetic-cli"). Runtime dep: the `eidetic` CLI on PATH (else a
  local eidetic-cli checkout with `uv`) — **`eidetic >= 0.10.0`** for the
  in-repo routing; on an older CLI the public records still work but are stored
  in `$HOME/.eidetic/memory` instead of in-repo. Propagated by rollout-cli's
  `eidetic-memory` recipe.

## [0.28.1] - 2026-06-26

### Added

- **`docs/tensorrt-llm-investigation.md`** — a dated desk investigation (no live
  run) of serving the MTP 27B primary (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`)
  with **TensorRT-LLM** (`trtllm-serve`) on the DGX Spark (GB10/SM121) instead of
  vLLM. **Verdict: not yet** — TRT-LLM MTP spec-decode is DeepSeek-only in stable
  releases and the Qwen3.6 hybrid GDN/DeltaNet kernels are RC-only (both land in
  1.3.0 RC builds); serving on a stable TRT-LLM today would forfeit the ~2.4×
  decode win the checkpoint exists for. Records the engine-integration seam (the
  request path — gateway routing + `lobes assess`/`benchmark` — is already
  engine-agnostic, while the gateway `/status` `vllm:*` metrics path, `catalog.py`,
  `switch.py`, templates, and `VLLM_*` env vars are vLLM-specific), a feasibility
  table by dimension with confidence levels, a comparison against the recorded
  vLLM baseline, a minimal spike recipe, an explicit revisit trigger (TRT-LLM
  1.3.0 stable), and 11 cited sources. Linked from the README per-model notes.

### Changed

### Fixed

## [0.28.0] - 2026-06-23

### Added

- **Vendored the `remember` + `recall` memory skills from eidetic-cli**
  (cite-don't-import) — the write/read halves of eidetic's shared
  `~/.eidetic/memory` surface, so this agent (Claude and its colleague backend)
  can persist facts across sessions and recall them later, sharing one store.
  `remember` drives `eidetic remember` (idempotent upsert of one JSON record or
  an NDJSON batch on stdin, dedup by id + content hash); `recall` drives
  `eidetic recall` with four search modes — exact / approximate / keyword /
  hybrid — each hit carrying text, full provenance metadata, a relevance score,
  and a freshness signal. The `.sh` wrappers are byte-verbatim from eidetic-cli
  (their first-party origin); each `SKILL.md` is localized only in the
  illustrative `--scope <nick>` examples (Provenance keeps "First-party to
  eidetic-cli"). Both default to this agent's PRIVATE scope, reading the suffix
  from `culture.yaml`. Runtime dep: the `eidetic` CLI on PATH (else a local
  eidetic-cli checkout with `uv`). Propagated by rollout-cli's `eidetic-memory`
  recipe.

## [0.27.0] - 2026-06-22

### Changed

- **Renamed the tool from `model-gear`/`model` to `lobes`/`lobes-cli`.** The
  binary is now **`lobes`** (`lobes switch`, `lobes serve`, `lobes assess`, …),
  the import package is **`lobes`**, and the PyPI distribution is **`lobes-cli`**.
  The deployed Culture agent is renamed `model-gear` → `lobes` (`culture.yaml`,
  `AGENTS.md`).
- **Deployment dir is now `~/.lobes`** (env `$LOBES_DIR`). The legacy
  `$MODEL_GEAR_DIR` and `~/.model-gear` are still resolved as fallbacks, so a
  pre-rename deployment keeps working with the renamed CLI without redeploying.
- **Deployment-internal names are intentionally kept as `model-gear`** so a live
  fleet isn't disrupted: Docker `container_name`s (`model-gear-vllm`,
  `model-gear-gateway`, …), `mg-logwrap.sh`, `MODEL_GEAR_LOG_DIR`, and the
  served-model id are unchanged.

### Added

- **`model` is kept as a deprecated alias command** for `lobes` (same entry
  point); `--version`/help reflect whichever name was invoked.
- **`model-gear` is published on PyPI as a deprecated alias** of `lobes-cli`: a
  metadata-only shim package (`packaging/model-gear/`) that depends on
  `lobes-cli==<same version>`, plus a `publish-alias` job in `publish.yml` that
  builds and publishes it after the main release.

## [0.26.4] - 2026-06-21

### Changed

- Relicensed the project from MIT to Apache 2.0 — full Apache 2.0 LICENSE text, pyproject `license`/classifier metadata, and a new README License section. Aligns with sibling AgentCulture repos (e.g. colleague, data-refinery-cli).

## [0.26.3] - 2026-06-21

### Fixed

- Doc consistency: `docs/mistral-small-3.2-24b-nvfp4.md` and
  `docs/qwen3.6-35b-a3b-nvfp4.md` still framed Mistral as the **default** fleet
  fallback the gateway pairs with the primary. The fleet has run one *generate*
  backend by default since the single-backend default (#42); the warm fallback is
  opt-in. Reframed both docs to match — closing the drift with the `model explain`
  catalog corrected in 0.26.2.
- Corrected the Mistral doc's "How it runs in the fleet" section, which described
  a fallback wiring that no longer exists: `FALLBACK_MODEL`/`FALLBACK_MAX_MODEL_LEN`
  /`FALLBACK_GPU_MEM_UTIL`/… `.env` keys "scaffolded by `model init --fleet`" and a
  shipped `model-gear-vllm-fallback` service. The current templates ship **no**
  fallback service and the gateway reads only `FALLBACK_URL` + `FALLBACK_SERVED_NAME`
  (set after you manually add a `vllm-fallback` service). Following the old text
  produced a non-working config; rewrote it to the actual two-step opt-in, matching
  `docs/gateway-fleet.md` → "Adding a fallback".

## [0.26.2] - 2026-06-21

### Changed

- Doc alignment pass across the audio + fleet surfaces (no behavior change):
  - `docs/chatterbox-tts.md`: healthcheck shows `python3.12` (not the stale
    `python3`) and the compose snippet includes `container_name:
    model-gear-chatterbox` — matching the 0.26.1 fixes.
  - `docs/realtime-pipeline.md`: the TTS service is `chatterbox` (was `tts`); the
    overlay scaffolds a Chatterbox Dockerfile too.
  - `docs/openai-api.md`: corrected the auth caveat — the gateway is a
    pass-through and is *not* auth-aware for *any* proxied endpoint (the previous
    wording implied it gated `/v1/chat/completions`).
  - `docs/gateway-fleet.md`: endpoint list now includes `/v1/audio/*`; added an
    "Auth (known limitation)" note.
  - `model explain gateway` / `model explain tunnel` (`explain/catalog.py`): added
    the gateway-not-auth-aware caveat; fixed the Mistral entry (opt-in fallback
    candidate, not the active default pairing); listed `tunnel`/`fleet` as write
    verbs.
  - `model learn` (`learn.py`): added an "Auth / exposure" section + an
    `auth_exposure` JSON field, and `model explain tunnel`/`gateway` pointers.

## [0.26.1] - 2026-06-21

### Fixed

- `model init --fleet --audio` now scaffolds `Dockerfile.chatterbox`. The
  Chatterbox sidecar landed in 0.25 (the compose `chatterbox` service builds from
  `Dockerfile.chatterbox`), but the build file was never added to
  `AUDIO_TEMPLATES`, so the scaffold omitted it and `docker compose build
  chatterbox` failed with "Dockerfile.chatterbox: no such file". Added it to the
  audio template set (twin of the `Dockerfile.realtime` / `Dockerfile.parakeet`
  wiring) so the audio overlay can actually build and serve TTS.
- `model fleet status` now reports the TTS gear. `FLEET_TTS` still pointed at the
  old `model-gear-tts` container name, but the Chatterbox sidecar renamed the
  container to `model-gear-chatterbox` — so status listed the live TTS gear as
  "not created". Pinned `FLEET_TTS` to `model-gear-chatterbox` and added a test
  that asserts every `FLEET_AUDIO_CONTAINERS` name matches a `container_name:` in
  the packaged audio compose (catches future rename drift).
- Chatterbox container now reports healthy. Its `Dockerfile.chatterbox` installs
  the interpreter as `python3.12` (no `python3` symlink), but the compose
  healthcheck called bare `python3` — which exec-failed every interval, pinning
  the working container at "starting"/"unhealthy". Switched the healthcheck to
  `python3.12` and added a test tying the healthcheck interpreter to the one the
  Dockerfile provides.

## [0.26.0] - 2026-06-21

Documentation pass for the realtime audio overlay and the OpenAI API front: a
feature doc per audio backend, a consolidated endpoint reference, and the same
information surfaced through `model learn`, `model explain`, the README, and
`CLAUDE.md`.

### Added

- `docs/parakeet-stt.md`: per-model feature doc for the **Parakeet** STT backend
  (`nvidia/parakeet-tdt-0.6b-v2`, NeMo ASR) — the only audio model that lacked one.
  Covers the HTTP contract, the real (model-loaded + CUDA-live) readiness probe,
  fleet integration, the stale-CUDA-context runbook, and why it is not a switchable
  catalog gear.
- `docs/openai-api.md`: consolidated **OpenAI-compatible API surface** reference —
  every endpoint (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/rerank`, `/v1/score`, `/v1/audio/transcriptions`, `/v1/audio/speech`,
  `/v1/models`, `/v1/models/supported`, `/health`), routing semantics (name /
  default / failover / SSE / audio fan-out), per-endpoint `curl` examples, the
  loaded-vs-supported split, and auth/exposure.
- `model explain` topics: `realtime` / `audio` (the `/v1/audio/*` overlay),
  `transcribe` / `stt` / `parakeet` (STT), `speak` / `tts` / `chatterbox` (TTS),
  and `api` / `openai` (the endpoint surface); linked from the explain root.
- `model learn` now documents the realtime audio overlay and the OpenAI API surface
  (text + `--json` `realtime_audio` / `api_surface` fields).
- README sections for **Realtime audio (STT + TTS)** and **The OpenAI-compatible API
  surface**, plus the two audio backends added to the per-model notes.

### Changed

- `CLAUDE.md`: documents the realtime audio overlay alongside the fleet; the CLI
  package tree now lists the `gateway/`, `realtime/`, `explain/`, and `catalog.py`
  surfaces.

### Fixed

- `docs/realtime-pipeline.md`: removed the stale `NGC_API_KEY` bring-up step (a
  Magpie leftover — Chatterbox needs no NGC key) and documented the TTS → STT
  round-trip in `scripts/audio-smoke.py`.

## [0.25.0] - 2026-06-21

### Added

- **Chatterbox TTS sidecar** (`model_gear/realtime/chatterbox_server.py`): a
  FastAPI HTTP server (`GET /v1/health/ready`, `POST /v1/audio/synthesize`) that
  wraps Resemble AI's Chatterbox model and returns raw PCM16 mono 24 kHz audio.
  Supports zero-shot voice cloning via a `.wav` reference path.  Runs as the
  `chatterbox` fleet service built by the new `Dockerfile.chatterbox` (arm64
  cu128 recipe).
- `[chatterbox]` optional-deps group (`fastapi`, `uvicorn`) in `pyproject.toml`.
- `docs/chatterbox-tts.md`: bake-off numbers, arm64 install recipe, sidecar HTTP
  contract, and integration notes.
- `model_gear/templates/fleet/Dockerfile.chatterbox`: arm64 CUDA build — rebased
  on `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` (no preinstalled torch) with
  pinned `torch==2.11.0+cu128` + `torchaudio==2.11.0+cu128` + Perth, fixing the
  NGC pytorch ABI conflict with Perth observed at runtime.
- numpy fast path in `float_tensor_to_pcm16` (stdlib fallback kept for offline CI);
  parity test in `tests/test_chatterbox_pcm16.py`.
- `resolve_voice()` `.wav` check is now case-insensitive (`.WAV` / `.Wav` work).
- Dead SSML code removed from `tts_client.py` (`_insert_ssml_breaks`); stale
  Magpie references updated to Chatterbox across app, client, and tests; speed-ignored
  warning emitted when a non-default speed is passed to `synthesize()`.

### Changed

- **Replaced Magpie TTS with Chatterbox** across the realtime stack:
  `protocol.py` (`TTS_SAMPLE_RATE` 22050→24000, `resolve_voice` rewritten for
  Chatterbox — `.wav` path for cloning, `""` for default), `tts_client.py`
  (plain JSON POST, no SSML/prosody wrapping), `_settings.py` (`tts_url` default
  → `http://chatterbox:9000`, `default_voice` default → `""`),
  `docker-compose.audio.yml` (new `chatterbox:` service replaces `tts:`),
  `env.audio.example` (Magpie/NGC vars removed, `CHATTERBOX_PORT` added).

## [0.24.0] - 2026-06-20

### Added

- **`model overview --live` — a live fleet dashboard.** `overview` was a static
  description; `--live` now probes the running deployment and reports the five
  "what is it doing right now" views: **online** (per-backend health), **offered**
  (served + candidate models, task families, the endpoint list), **busy**
  (in-flight / queued requests), **usage** (cumulative prompt/generation tokens and
  finished requests by reason), and **endpoints**. It is read-only and HTTP-only —
  it works against a local deployment or a `model tunnel` hostname alike, and
  degrades gracefully when a backend or its metrics is unreachable.
- **Gateway `GET /status`** — a model-gear-native JSON aggregate. The fleet's
  backends are internal-only, so the gateway fans out to each one's `/health` +
  `/metrics` and returns `{object: "model-gear.fleet_status", default_model,
  busy: {running, waiting}, backends: [...], endpoints: [...]}`. This is the source
  `model overview --live` reads in the fleet (a bare single-model server is read
  directly from its `/metrics` + `/health`).
- **`model_gear._metrics`** — a small stdlib-only helper that parses vLLM's
  Prometheus `/metrics` (running/waiting, prompt/generation tokens,
  `request_success_total` by finish reason, KV-cache usage) and best-effort HTTP
  probes that never raise.

### Changed

### Fixed

## [0.23.0] - 2026-06-20

### Added

- **Durable vLLM logs that survive restart/recreate (#50).** When a vLLM
  container restarted, its `docker logs` — and any EngineCore crash trace — were
  lost, which blocked root-causing #50 for lack of data. `model init` now
  scaffolds `mg-logwrap.sh`, bind-mounted as each vLLM service's entrypoint: it
  tees stdout+stderr to a per-boot file `<service>-<boot>.log` under a
  host-mounted log dir (`${MODEL_GEAR_LOG_DIR:-<deploy>/logs}` → `/logs/model-gear`),
  then `exec`s the real command so vLLM stays the signal target (graceful
  shutdown) and the exit code (and `restart:` policy) are unchanged. Teeing at the
  process-I/O level captures **both** Python tracebacks and native CUDA/C++ aborts;
  if logging can't be set up it falls back to a plain `exec` and never blocks
  serving. The crash boot is preserved as its own file. Wired into the single-model
  and fleet (`primary`/`embed`/`rerank`) compose templates. See
  `docs/durable-logs.md`.
- **`model logs`** — new read-only verb to list/tail the durable logs, reading the
  host files directly so it works even after the crashed container is gone:
  `model logs` (list boots), `model logs <service>` (tail latest), and
  `model logs <service> --previous` (tail the boot that crashed, after a restart).

### Changed

- `model init` / `model serve` / `model fleet up` pre-create the host log dir
  (user-owned) before compose bind-mounts it, so logs are never root-owned.

### Fixed

## [0.22.1] - 2026-06-19

### Fixed

- **`model fleet status` now reports the embedding + reranker gears.**
  `FLEET_CONTAINERS` listed only `vllm-primary` + `gateway`, so `model fleet
  status` silently omitted the `vllm-embed` / `vllm-rerank` containers the default
  fleet (#44/#47) actually runs. Added `FLEET_EMBED` / `FLEET_RERANK` to the
  default container set — status now lists all four (the opt-in generate fallback
  stays excluded, as it is not in the default compose).

### Changed

- **Aligned the agent/human-facing prose with the co-resident gears (#44/#47).**
  `model learn`, `model overview`, `model explain` (root + fleet), `model init
  --fleet` help, the `fleet` docstring, the scaffolded `env.example` /
  `docker-compose.yml` comments, `README.md`, `CLAUDE.md`, and
  `docs/gateway-fleet.md` still described the fleet as a "2-model" /
  "two-container" / "single-backend" deployment. They now describe the default
  fleet as the generate primary plus co-resident embedding + reranker gears behind
  one gateway, routed by task family (generate / embed / score / rerank), with the
  *generate* fallback as the only opt-in backend. Added a "Task families & gears"
  section + `explain embeddings` / `explain rerank` pointers to `model learn`.

## [0.22.0] - 2026-06-19

### Added

- **Embedding + reranker gears (closes #44).** model-gear now serves two pooling
  gears alongside the chat primary, reachable through the same OpenAI-compatible
  gateway and routed by the request's `model` field:
  - `Qwen/Qwen3-Embedding-0.6B` — `POST /v1/embeddings` (vLLM
    `--runner pooling --convert embed`), native **1024-dim**, MRL-truncatable via
    the `dimensions` param (Matryoshka `--hf-overrides`).
  - `Qwen/Qwen3-Reranker-0.6B` — `POST /v1/rerank` + `/v1/score` (vLLM
    `--runner pooling --convert classify`, served via the
    `Qwen3ForSequenceClassification` `--hf-overrides`).
  - **Catalog:** `SupportedModel` gains `task` (`generate`/`embed`/`score`),
    `dimension`, and `hf_overrides`; both gears surface in `model overview --list`
    and `GET /v1/models/supported`.
  - **Fleet:** `vllm-embed` + `vllm-rerank` services in the fleet compose
    (always-warm, small `--max-model-len`/`--gpu-memory-utilization` so they
    co-reside with the 27B on a single GB10), wired as gateway backends.
  - **Gateway:** task-aware failover — an embed/score request never fails over to
    a generate backend (and vice versa); chat primary↔fallback failover preserved.
  - **CLI:** `model switch --task {generate,embed,score}` for solo serving;
    `model explain embeddings` / `rerank` / `score` document the call shapes;
    per-model docs under `docs/`.
  - **Boundary:** model-gear *serves* the gears only — no vector store, index,
    chunker, or retrieval lands here (guarded by a test); storage + retrieval are
    the consumer's half (eidetic-cli).

### Changed

### Fixed

## [0.21.1] - 2026-06-19

### Fixed

- **markdownlint:** exempt skill prompt templates (`.claude/skills/**/prompts/**`)
  from markdownlint. These are model-facing prompts fed verbatim to a backend
  (first line is `$ARGUMENTS` or a prose instruction), so MD041 (first-line H1)
  and MD032 are inapplicable — a heading would be injected into the prompt.
  `SKILL.md` is still linted; only `prompts/` is exempt. Unblocks the `lint` CI
  job after the `ask-colleague` skill was vendored in.

## [0.21.0] - 2026-06-12

### Changed

- **The fleet is now single-backend by default (Qwen primary only); the Mistral
  fallback is removed.** Live validation showed two ~30B NVFP4 models don't co-fit
  a shared GB10, so the warm dense Mistral-Small-3.2-24B fallback has been dropped
  from the default fleet and the primary restored to its **load-tested solo
  headroom**: `PRIMARY_GPU_MEM_UTIL` `0.40 → 0.6` and `PRIMARY_MAX_MODEL_LEN`
  `32768 → 262144` (full 256K). The `vllm-fallback` service is gone from
  `fleet/docker-compose.yml`, and `FLEET_CONTAINERS` no longer includes it.
- **The gateway makes the fallback optional.** `build_config` now adds a second
  backend **only** when `FALLBACK_URL` or `FALLBACK_SERVED_NAME` is set in env —
  so the default gateway serves the primary alone (no failover target), and a
  two-backend fleet still works for anyone who wires one up. Routing/failover
  primitives are unchanged; `order_backends` returns just the primary when solo.
- **Mistral stays a selectable catalog candidate** (`model overview --list`) and
  the documented opt-in fallback — only its role as the *default* fleet fallback
  is removed. README, `docs/gateway-fleet.md`, and the `model explain
  fleet/gateway` / `model init --help` text are updated to the single-backend
  default (with an "Adding a fallback" guide).

### Fixed

- `docs/gateway-fleet.md` uses `$HOME/.model-gear` instead of the non-portable
  `~/.model-gear`.

## [0.20.1] - 2026-06-12

### Fixed

Qodo review of #41:

- **`model init --fleet --audio` now scaffolds `_readiness.py`** — added
  `fleet/_readiness.py → _readiness.py` to `_compose.AUDIO_TEMPLATES`. The
  Parakeet `Dockerfile.parakeet` `COPY _readiness.py` requires it at the
  deployment-dir root, so a clean audio init previously produced a tree where
  `docker compose build stt` would fail. Covered by `test_init.py`.
- **Parakeet readiness drift guard + simplification** — removed the third
  (inline) copy of the readiness decision from `listen_server.py` (the scaffold
  now guarantees the vendored `_readiness.py` is present), and added a test
  asserting the vendored twin stays behaviourally identical to the canonical
  `model_gear/realtime/_readiness.py`.
- **CUDA readiness probe failures are now logged** — `listen_server.health()`
  emits a `logger.warning` with the exception type/message before returning
  `503`, so operators can distinguish driver-down / OOM / stale-context.
- **`scripts/audio-smoke.py` now exercises `/v1/audio/speech`** (it previously
  claimed both routes but only tested transcriptions) and wires the formerly
  unused `--stt-url` to a direct-Parakeet transcription check.
- **`docs/realtime-pipeline.md`** uses `$HOME/.model-gear` instead of the
  non-portable `~/.model-gear`.

## [0.20.0] - 2026-06-12

### Added

- **`docs/realtime-pipeline.md`** — the previously-missing runbook for the audio
  surface: that model-gear owns the live `:8080` realtime facade, the
  `model init --fleet --audio` / `model fleet up` bring-up, the topology
  (gateway path-routes `/v1/audio/*` → realtime → Parakeet/Magpie), the drift it
  fixed (#39/#40), the cheap readiness probe, and the stale-Parakeet-CUDA restart
  runbook. Resolves a doc referenced from `pyproject.toml`, the audio overlay,
  and the realtime app docstring but never written.
- **`scripts/audio-smoke.py`** — a stdlib-only live smoke test for the audio
  routes: asserts `GET :8080/openapi.json` lists both `/v1/audio/transcriptions`
  and `/v1/audio/speech`, then POSTs an in-memory 16 kHz WAV and asserts
  `200 {text: …}`. Reproduces issue #39's repro to confirm the 500→200 fix.
  Requires a running GPU box (not a CI unit test).
- **`model_gear/realtime/_readiness.py`** — a stdlib-only `evaluate_readiness()`
  helper backing the Parakeet `/v1/health/ready` cheap probe; unit-tested in CI
  without torch/nemo/GPU.

### Fixed

- **Parakeet STT healthcheck now reflects real model readiness (#39).** The
  vendored `templates/fleet/listen_server.py` `/v1/health/ready` returned
  `{"status": "ready"}` unconditionally — process liveness only — so a container
  whose CUDA context had gone stale (`CUDA error: unknown error`, every
  transcription 500ing) still reported Docker "healthy". The probe now reports
  ready **only** when the NeMo model is loaded **and** a trivial CUDA tensor op
  succeeds, returning `503` otherwise (a cheap probe, not a full transcription
  each interval). The pure decision is vendored into the Parakeet build context
  and `COPY`'d into the image so it resolves without the wheel.

## [0.19.0] - 2026-06-09

### Added

- **`scripts/gen-api-key.py`** — generate or rotate the bearer key
  (`CULTURE_VLLM_API_KEY`) that gates the served API. The secret is created with
  the stdlib `secrets` module and **never hardcoded**, so the script is safe in the
  open-source repo; the key only ever lands in the gitignored deployment `.env`
  (written `0o600`, best-effort). Hidden by default (no echo into logs/scrollback);
  `--show` prints it, `--force` rotates an existing key, and `--bytes` (min 16) is
  validated. Resolves the deployment dir like the `model` CLI (`--dir` →
  `$MODEL_GEAR_DIR` → `$HOME/.model-gear`), degrades gracefully on an unreadable or
  non-regular `.env`, and runs from a wheel install (no `model_gear` import).
  Referenced from the README "Expose the API" section.

## [0.18.0] - 2026-06-09

### Added

- **`model tunnel` — expose the local OpenAI-compatible API from anywhere via a
  Cloudflare Tunnel** (#35). Dry-run by default (prints the `cloudflared` command
  and the public `https://<host>/v1` URL); `--apply` starts a standalone
  `cloudflared tunnel run` in the background (logging to `cloudflared.log` in the
  deployment dir), and `--stop --apply` tears it down. The public hostname resolves
  `--hostname` → `$CULTURE_VLLM_PUBLIC_HOSTNAME` → `CULTURE_VLLM_PUBLIC_HOSTNAME` in
  a **gitignored** `.cf-tunnel.env`; the run-token comes from
  `CULTURE_CF_TUNNEL_TOKEN_SHUSHU` (a shushu-sealed secret name, preferred) or
  `CULTURE_CF_TUNNEL_TOKEN` (plaintext fallback). The token is **never placed on the
  process argv** (so it can't leak via `ps` or the log) — cloudflared reads it from
  the `TUNNEL_TOKEN` environment variable, which `shushu` injects (sealed mode) or
  the launcher sets directly (fallback). The resolved hostname and sealed-secret
  name are validated against a conservative charset before they reach the argv (an
  argument-injection guard). `--apply` preflights that `cloudflared` (and `shushu`)
  is on PATH, that no tunnel is already running for the deployment, and that the
  local server answers `/health`; `--stop` signals the recorded process *group* and
  confirms exit (SIGTERM → SIGKILL) before clearing a PID-reuse-safe pidfile (the
  recorded pid is identity-checked against `/proc` so a reused pid can't be killed).
  No hostname, token, or backend checkpoint id is committed. The Cloudflare side
  (tunnel + ingress + DNS) is provisioned once by `cultureflare remote-login
  --no-access`.
- **Optional bearer auth on the served API** via `CULTURE_VLLM_API_KEY`, wired into
  the single-model `docker-compose.yml` as `VLLM_API_KEY=${CULTURE_VLLM_API_KEY:-}`.
  Empty (default) leaves local dev open; set it and vLLM requires `Authorization:
  Bearer` — the gate for any public exposure. Documented in `env.example` alongside
  a note that `VLLM_SERVED_NAME` can be a generic alias to keep the checkpoint name
  out of the public `/v1/models`.
- **`cf-tunnel.env.example`** scaffolded by `model init` (single + fleet), a
  placeholder-only template the owner copies to the gitignored `.cf-tunnel.env`.
- README "Expose the API from anywhere (Cloudflare Tunnel)" section and a
  `model explain tunnel` catalog entry.

## [0.17.0] - 2026-06-03

### Changed

- **Served context raised 128K → full 256K (native) for the MTP primary on DGX
  Spark.** The `spark` machine profile's `max_model_len` default is now `262144`
  (was `131072`), with matching changes to the single-model `env.example` /
  `docker-compose.yml` defaults and the `model switch --help` / `model explain`
  text. Load-tested 2026-06-03 on the shared GB10 (util 0.6, `--max-num-seqs 2`,
  KV-FP8, MTP n=3): boots clean (CUDA-graph capture, PIECEWISE, **0.71 GiB** in 2 s
  — **no OOM**), **17.8 tok/s** decode, **74.0 %** MTP draft acceptance, both
  `model assess` probes `finish=stop`, tool-calling probe passes, and **71,601 MiB
  (~70 GiB)** resident — the *same* footprint as 32K/128K, because
  `--gpu-memory-utilization` fixes the KV-pool reservation (only the addressable
  context grows). vLLM reports **5.29× max concurrency at a full 256K request**,
  well above the `--max-num-seqs 2` decode cap, so **there is no practical
  concurrency cost** versus the 128K default. `model switch --max-model-len <N>`
  still overrides per deployment, and util stays a conservative `0.6` (shared box).
  See `docs/qwen3.6-27b-text-nvfp4-mtp.md` (new 256K benchmark) and
  `docs/tuning-profiles.md`.
- **Catalog `context` string updated.** The MTP primary now reads
  `"256K native (served at full 256K on the shared GB10)"`.
- **Scope — deliberately left at the old contexts:** fleet templates stay at 32K
  (co-residence with the 24B fallback is a different, still-unvalidated memory
  regime; `fleet/env.example` notes this), and the `thor` / `generic` machine
  profiles stay at 32K (unmeasured estimates) with `blackwell` at 64K. The
  `model switch` native-ceiling clamp (added in 0.16.0) still pins 32K-native
  candidates (`nvidia/Qwen3-32B-NVFP4`, `mmangkad/Qwen3.6-35B-A3B-NVFP4`) down to
  their own ceilings under the new 256K spark default.

### Fixed

- **`model switch` warns when an uncatalogued model would inherit an unclamped
  machine context default.** The native-ceiling clamp only protects catalogued
  models; an uncatalogued model ID (which `switch` supports) inherits the machine
  default (now spark's 262144) and would boot-fail if the checkpoint's native
  context is smaller. `switch` now emits a clear warning pointing at
  `--max-model-len` / cataloguing, rather than silently applying the high default
  (no silent clamp — an uncatalogued ceiling is unknown, so guessing one is wrong
  both ways). Addresses a Qodo reliability finding on #34.

## [0.16.0] - 2026-06-03

### Changed

- **Served context raised 32K → 128K for the MTP primary on DGX Spark.** The
  `spark` machine profile's `max_model_len` default is now `131072` (was `32768`),
  with matching changes in the single-model `env.example` /`docker-compose.yml`
  defaults. Load-tested 2026-06-03 on the shared GB10 (util 0.6, `--max-num-seqs 2`,
  KV-FP8, MTP n=3): boots clean (no CUDA-graph-capture OOM), **18.3 tok/s** decode,
  **73.3 %** MTP draft acceptance, both `model assess` probes `finish=stop`, and
  **71,963 MiB (~70 GiB)** resident — the *same* footprint as 32K, because
  `--gpu-memory-utilization` fixes the KV-pool reservation (the pool holds **9.6×**
  a full 128K request). `model switch --max-model-len <N>` still overrides per
  deployment, and util stays a conservative `0.6` (the box is shared). See
  `docs/qwen3.6-27b-text-nvfp4-mtp.md` (new 128K benchmark) and
  `docs/tuning-profiles.md`.
- **Catalog `context` strings clarified.** The MTP primary now reads
  `"256K native (served at 128K on the shared GB10)"`; the non-served candidate /
  fallback entries (`mmangkad/Qwen3.6-27B-NVFP4`, the Mistral fallback) drop the
  stale per-model "capped to 32K" note and state native context only.
- **Scope — deliberately left at the old contexts:** fleet templates stay at 32K
  (the fleet runs the primary co-resident with a 24B fallback at lower util — a
  different memory regime the single-model 128K test does not validate;
  `fleet/env.example` notes this), and the `thor` / `generic` machine profiles
  stay at 32K (unmeasured estimates) with `blackwell` at 64K.

### Fixed

- **`model switch` clamps the machine context default to a model's native ceiling.**
  Raising spark's `max_model_len` default to `131072` made it apply to *every* model
  switched to on spark — including the 32K-native catalog candidates
  (`nvidia/Qwen3-32B-NVFP4`, `mmangkad/Qwen3.6-35B-A3B-NVFP4`), where vLLM refuses a
  `--max-model-len` above the checkpoint's native limit (no YaRN) and the container
  fails to boot. `SupportedModel` now carries a numeric `native_max_model_len`, and
  `model switch` clamps the resolved context *down* to it when no explicit
  `--max-model-len` is given (an explicit value still wins, for opted-in YaRN
  configs). Fixes a Qodo correctness finding on #33.

## [0.15.0] - 2026-05-31

### Changed

- **Fleet default primary → `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`** (the MTP
  build), replacing `mmangkad/Qwen3.6-27B-NVFP4` (issue #26 follow-up). The
  tool-calling gate that kept it a candidate is now **closed**: served through the
  production compose it emits a valid `qwen3_coder` tool call, completes a full
  tool round-trip, keeps its reasoning trace, and runs MTP spec-decode at **78.6%
  draft acceptance with tool calling on** — ~2.4× single-stream decode (8 → ~19
  tok/s), ~71 GB footprint, both `model assess` probes `finish=stop`. Promoted
  across the catalog (`role_hint`), the gateway default (`_DEFAULT_PRIMARY`),
  `whoami`, both template `env.example`/`docker-compose.yml` files, and
  `culture.yaml`.
- **The MTP serve flags are now baked into the compose templates** (single-model +
  fleet `vllm-primary`): `--speculative-config`, `--trust-remote-code`,
  `--language-model-only`, the `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4` override,
  and `--max-num-seqs=2`. A fresh `model init && model serve` of the default now
  works out of the box. Quantization default is `modelopt`.
- **`model switch` notices inverted.** Because the template ships the MTP primary's
  flags, switching to a **non-MTP** model now prints "REMOVE these 4 `command:`
  lines" (was "add" for the MTP candidate); the MoE `--moe-backend` add-notice is
  unchanged. Switching to the MTP primary force-caps `--max-num-seqs` to 2.
- **`mmangkad/Qwen3.6-27B-NVFP4` archived to a candidate** — retained as the MTP
  primary's tokenizer source and the only vision-capable 27B in the catalog.

### Fixed

- **`model switch --apply` no longer takes a healthy deployment down when a manual
  compose edit is required** (Qodo review). Switching to a non-MTP model (the
  template ships the MTP primary's incompatible flags) now writes `.env` and
  **stops before the restart**, printing the lines to remove; `--force` overrides
  to recreate the container anyway.
- **MTP compose flags are a single source of truth** (`catalog.mtp_compose_command_items()`) —
  consumed by both `model switch`'s removal notice and guarded against drift from
  the packaged templates by a new test (Qodo review).
- **Security guidance for the now-default `--trust-remote-code`** added to both
  compose templates and `env.example`: HF_TOKEN is only needed for gated repos
  (defaults are public) — leave it empty or use a minimal-scope read-only token, and
  pin trusted revisions (Qodo review). Tracking the upstream tokenizer fix that would
  let us drop the override in #29.

## [0.14.0] - 2026-05-31

### Added

- **MTP (Multi-Token Prediction) candidate for the 27B** (issue #26). New catalog
  entry `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` — a text-only re-export of the
  27B primary with its **MTP draft head restored in bf16** so vLLM speculative
  decoding actually works. The lesson from the 35B MoE applied: the baseline NVFP4
  export drops the MTP head (~0 % draft acceptance), and a newer vLLM isn't
  installable on the aarch64 GB10 — so the fix is *a checkpoint that ships the MTP
  weights*, not a newer engine. Carries a catalog `speculative_config`
  (`{"method":"qwen3_5_mtp","num_speculative_tokens":3}`); quantization is
  `modelopt`. **Load-tested on the DGX Spark (GB10) 2026-05-31: 19.1 tok/s decode
  (~2.4× the baseline 27B's ~8 tok/s) at 72 % MTP draft acceptance on vLLM
  0.19.0+nv26.04** — the open risk (does the stock image accept `qwen3_5_mtp`?) is
  cleared (it resolves the `Qwen3_5MTP` draft head). One tokenizer override is
  required: the checkpoint declares the newer `TokenizersBackend` class (absent from
  nv26.04), so serve with `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4` (the cached
  sibling, same vocab); `model switch` prints it.
  - New per-model doc `docs/qwen3.6-27b-text-nvfp4-mtp.md` with the serve recipe,
    the live benchmark table (decode tok/s + acceptance vs the baseline), and the
    caveats (`--max-num-seqs 2` or it silently OOMs; the tokenizer override).

### Changed

- **`model switch` surfaces MTP serve-extras, not just MoE.** `_moe_notice` →
  `_serve_notices` (now a list): a model with a catalog `speculative_config` prints
  the exact `--speculative-config` / `--trust-remote-code` / `--language-model-only`
  compose edits (+ the `VLLM_MAX_NUM_SEQS=2` reminder), the same hand-edit pattern
  as `--moe-backend`. The `--json` dry-run replaces the `moe_notice` key with a
  `compose_edits` list. `env.example` + the `explain` catalog prose updated to match.

### Fixed

## [0.13.0] - 2026-05-31

### Added

- **Workload `purpose` + machine tuning profiles.** `model switch` now resolves
  the serve config from three layers — a **machine** profile (`--machine`,
  default auto-detected from `nvidia-smi` + hostname: GPU-memory fraction,
  context, attention backend), a **workload** profile (`--purpose`, default
  `balanced`: the batching knobs and the shape `model benchmark` exercises), and
  the model's catalog entry — with explicit `--max-model-len` / `--gpu-mem-util`
  flags overriding the machine defaults.
  - **New `model_gear/profiles.py`** (pure data module, like `catalog.py`):
    `WorkloadProfile` (`balanced` ≈1K/1K, `prompt-heavy` ≈8K/1K, `decode-heavy`
    ≈1K/8K) and `MachineProfile` (`spark` load-tested, `thor`/`blackwell`/`generic`
    configured), guarded by `tests/test_profiles.py`.
  - **Richer single-model template** — the serve command now passes
    `--attention-backend`, `--max-num-seqs`, `--max-num-batched-tokens` (env-driven),
    plus static `--enable-chunked-prefill` / `--async-scheduling`. New `.env` keys:
    `VLLM_PURPOSE`, `VLLM_MACHINE`, `VLLM_ATTENTION_BACKEND`, `VLLM_MAX_NUM_SEQS`,
    `VLLM_MAX_NUM_BATCHED_TOKENS`.
  - **Per-model MoE serve extras** — the catalog gains `moe_backend` /
    `speculative_config` (set only on the `Qwen3.6-35B-A3B` MoE candidate).
    `model switch` to the MoE prints them as a documented compose edit (they break
    the dense/hybrid models and can't be defaulted in the shared template).
  - **`model benchmark` is tied to the config** — its workload shape defaults to
    the configured `VLLM_PURPOSE` (overridable with `--purpose` / `--input-len` /
    `--output-len`).
  - `model whoami` / `model overview` surface the active `gear` (purpose/machine);
    `model explain tuning` documents the layering; `docs/tuning-profiles.md` is new.
  - Credit: the serve tuning and the three workload shapes follow **shahizat**'s
    cross-machine NVFP4 benchmark (NVIDIA Developer Forums) — see the README
    Acknowledgements and `docs/tuning-profiles.md`.
  - **Live-replicated on the shared DGX Spark (2026-05-31)** rather than trusting
    the post: with the new flags the **35B MoE candidate loads solo** (util 0.70,
    marlin) and runs **single-stream decode ~35 tok/s vs the 27B's ~7.8 — ~4.6×
    faster** (the MoE's ~3B-active advantage). Numbers + method in
    `docs/tuning-profiles.md` and `docs/qwen3.6-35b-a3b-nvfp4.md`.

### Changed

- `model switch` `--max-model-len` / `--gpu-mem-util` now default to the machine
  profile (was a fixed 32768 / 0.6); pass them explicitly to override.
- `model benchmark` replaces `--decode-tokens` with purpose-driven
  `--input-len` / `--output-len`.
- **Catalog: dropped the MTP `speculative_config` from the `mmangkad/Qwen3.6-35B-A3B-NVFP4`
  entry** (kept `--moe-backend=marlin`). Live testing showed shahizat's MTP draft
  fails to load on the `mmangkad/` copy (`qwen3_5_mtp.py` weight-shape mismatch on
  vLLM nv26.04) — it is tied to his `nvidia/` checkpoint. `model switch` no longer
  prints a recipe that wouldn't load.

### Fixed

## [0.12.0] - 2026-05-30

### Added

- **Audio I/O behind the gateway (STT + TTS) — issue #18, part 1 of 3.** model-gear
  now serves OpenAI-compatible `POST /v1/audio/transcriptions` and
  `POST /v1/audio/speech` on the same host port as the text API, fronted by the same
  stdlib gateway. The audio backends are the *same models* the standalone realtime-api
  stack ran — **NVIDIA Parakeet STT** + **Magpie TTS NIM** — consolidated into the
  fleet (no separate compose project; the realtime bridge's LLM is the fleet gateway
  itself, so there is no extra vLLM container).
  - **New `[realtime]` extra + `model_gear.realtime` package** (vendored from the
    `realtime-api` sibling, cite-don't-import): a FastAPI bridge that exposes the
    OpenAI audio surface (`/v1/audio/speech` adapts Magpie's proprietary
    `/v1/audio/synthesize`; `/v1/audio/transcriptions` forwards to Parakeet). The base
    wheel and the gateway stay stdlib-only — torch/fastapi never leak into them.
  - **Gateway audio routing** — `/v1/audio/*` is path-routed to the audio backend
    (`AUDIO_URL`) with no model rewrite and no failover; binary responses relayed
    **streamed** (chunked) so a large TTS body never buffers whole in the gateway.
    Unset `AUDIO_URL` (a text-only fleet) → those paths 404, unchanged.
  - **`model init --fleet --audio`** scaffolds the audio overlay
    (`docker-compose.audio.yml` + `Dockerfile.realtime` + a vendored
    `Dockerfile.parakeet`/`listen_server.py`) and appends the audio keys to `.env`.
    `model fleet up`/`down`/`status` auto-include the overlay when present.
  - **Co-residence caveat:** the audio services share the GPU with the LLM fleet — the
    overlay is opt-in so text-only boxes keep their GPU budget. See the per-model docs
    (PR3) for live numbers.
  - **The realtime WebSocket (`/v1/realtime`) and the `model overview`/`doctor`/`explain`
    surface land in the follow-up PRs (parts 2 and 3).**

### Changed

### Fixed

- **Audio review hardening (PR #24 review).**
  - **Gateway no longer buffers whole audio bodies** — `/v1/audio/*` responses are
    relayed chunked instead of `read_all()`'d into memory, so one large TTS WAV can't
    OOM the fleet's single front door.
  - **`TTS_CONCURRENCY` / `TTS_SPEED` clamped to ≥ 1** — `TTS_CONCURRENCY=0` previously
    seeded an `asyncio.Semaphore(0)` that hung every TTS request; a 0/negative speed
    emitted nonsensical `rate="0%"` SSML.
  - **`/v1/audio/speech` `speed` clamped to OpenAI's 0.25–4.0 range** before the Magpie
    percentage conversion, so out-of-range values no longer reach the backend as
    `rate="{huge|negative}%"` and 502.
  - **SonarCloud config** — coverage exclusions now mirror `coverage.run` `omit` (the
    `[realtime]`-extra modules can't be unit-imported offline), and the deployment
    *scaffolds* under `model_gear/templates/**` are excluded from analysis (container
    Dockerfiles + the vendored Parakeet server aren't package runtime). Added unit
    tests for `realtime.protocol`, the settings clamps, the speed clamp, and the
    streamed audio relay.

## [0.11.1] - 2026-05-30

### Added

- **`model learn --json` now includes a `models` object** (`supported_catalog` /
  `loaded_now`) — a machine-readable version of the catalog-vs-loaded explainer for
  agent consumers. (Additive field; the only observable behavior change in this
  release.)

### Changed

- **Documented "supported catalog vs. loaded now" consistently** across the README,
  `docs/gateway-fleet.md` (new "Supported catalog vs. warm backends" subsection),
  the per-model docs, and the CLI teaching surfaces (`model learn`,
  `model explain models`/`overview`/`status`/`whoami`/root, and the
  `overview`/`status`/`whoami`/`fleet status` help strings). The distinction:
  `model overview --list` / `GET /v1/models/supported` = the gears you *can* switch
  to (tagged `load-tested`/`configured`, static); the live `GET /v1/models` (which
  `model fleet status` queries) = what's actually *loaded* now. `model status` /
  `model whoami` report the *configured* served model (from `.env`) + health — not
  a live `/v1/models` query. Docs + help text (no serving/runtime behavior change).

## [0.11.0] - 2026-05-30

### Added

- **`RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4` support** — added to the
  supported-model catalog (`model overview --list`, `GET /v1/models/supported`)
  with a per-model doc, [`docs/mistral-small-3.2-24b-nvfp4.md`](docs/mistral-small-3.2-24b-nvfp4.md).
  Load-tested on the DGX Spark (GB10): ~15 GiB weights, **~14.9 tok/s** decode,
  prefill 2,009 tok in 1.49 s, tool calling ✅.
- **`mistral` tool-call parser inference** — `model_gear.runtime._parser` now maps
  Mistral-family ids (incl. the `mistralai/` org) to the `mistral` parser; `model
  switch` auto-selects it.
- **`model switch --quantization`** — the served `--quantization` is now set per
  model (read from the catalog for a known model, e.g. `compressed-tensors` for the
  RedHatAI NVFP4 Mistral vs `modelopt_fp4` for the nvidia/mmangkad checkpoints);
  `--quantization` overrides it. The single-model compose reads `VLLM_QUANTIZATION`.

### Changed

- **Fleet default fallback is now the dense Mistral-Small-3.2-24B**, replacing the
  `mmangkad/Qwen3.6-35B-A3B-NVFP4` MoE, which never loaded on the GB10 (OOM
  co-resident, stall solo — no benchmark obtained). Mistral is dense, loads
  reliably, and is smaller (~15 GiB weights). The fleet compose serves it with the
  **mistral tokenizer + images limited to 0** (required for tool-call parsing on
  the nv26.04 build; the HF tokenizer leaks `[TOOL_CALLS]` markup, and the mistral
  tokenizer alone crashes the Pixtral profiler) and **no** `--reasoning-parser`
  (instruct model). The 35B MoE is demoted to a catalogue candidate.
- `model_gear.gateway._config._DEFAULT_FALLBACK`, the fleet `docker-compose.yml` /
  `env.example` `FALLBACK_*` defaults, `docs/gateway-fleet.md`, and `README.md`
  updated for the new fallback.

### Fixed

## [0.10.1] - 2026-05-30

### Changed

- **Fleet default GPU-mem utilisations rebalanced `0.55`/`0.30` → `0.40`/`0.35`.**
  Live validation on a DGX Spark (GB10) showed `0.55`/`0.30` OOM-crash-loops the
  fallback: the 27B primary alone takes ~75 GiB at util 0.6, and `--gpu-memory-utilization`
  is fraction-of-total *per process* (the two backends don't coordinate). The new
  values are a dedicated-box estimate; the templates and docs now state plainly
  that co-residence of two ~30B models needs a dedicated box.

### Fixed

- **Docs corrected against live findings (2026-05-30):** `docs/gateway-fleet.md`
  gains a "Live validation findings" section (27B warm-up ~7 min, ~75 GiB footprint,
  8.0 tok/s decode; co-residence not viable on a shared GB10).
  `docs/qwen3.6-35b-a3b-nvfp4.md` updated from "not yet load-tested" to the actual
  result — the MoE fallback does **not** load reliably on this box (OOM co-resident;
  crash/stall even solo). `docs/qwen3.6-27b-nvfp4.md` reframed as the fleet default
  primary (was "candidate") with the warm-up measurement and a corrected
  recommendation.

## [0.10.0] - 2026-05-30

### Added

- **`GET /v1/models/supported` gateway endpoint — the "change gears" catalog.**
  Alongside the OpenAI-standard `/v1/models` (which lists only the two *loaded*
  backends), the gateway now serves the full catalog of supported models a client
  can change gears to, each flagged `loaded` (a backend serves it now) and
  `default` (the gateway routes unknown/missing names there). Non-OpenAI shape
  (`"object": "model-gear.supported_models"`) so `/v1/models` stays standard for
  existing clients. Pure `supported_models_payload()` in `gateway/_routing.py`.
- **New packaged catalog `model_gear/catalog.py`** — a dependency-free
  `SUPPORTED_MODELS` tuple (the 27B primary, the 32B dense candidate, the 35B-A3B
  MoE fallback) that is the single source of truth for both the gateway (which
  runs from a wheel and can't read `docs/`) and the CLI. `model overview --list`
  is now catalog-backed, so it is populated even in a wheel install.

### Changed

- **Fleet (and single-model) default primary → `mmangkad/Qwen3.6-27B-NVFP4`.**
  The scaffolded default served model is now the Qwen3.6 27B (hybrid
  Mamba/linear-attn + ViT, 256K native context) with `--tool-call-parser=qwen3_coder`
  — matching what runs on the DGX Spark and convertible's parent model. The dense
  `nvidia/Qwen3-32B-NVFP4` remains a supported candidate (`PRIMARY_MODEL` /
  `model switch`). Recomputed co-resident GPU memory: `PRIMARY_GPU_MEM_UTIL=0.55`
  and `FALLBACK_GPU_MEM_UTIL=0.30` (the 27B is heavier than the 32B). Updated the
  fleet + single-model templates, `gateway/_config.py`, `whoami` default,
  `culture.yaml` / `AGENTS.md` / `CLAUDE.md` (served-model coherence chain), and
  the per-model + gateway-fleet docs.

### Fixed

## [0.9.0] - 2026-05-28

### Added

- **Fallback model + single front OpenAI gateway ("fleet").** A new
  scaffold-based deployment runs **two always-warm vLLM backends behind one
  stdlib gateway** that model-gear manages as three containers
  (`model-gear-gateway`, `model-gear-vllm-primary`, `model-gear-vllm-fallback`).
  The gateway routes each request by its `model` field, defaults an
  unknown/missing name to the primary, and fails over to the other backend when
  the chosen one refuses the connection or returns a 5xx **before** the response
  body (4xx is returned verbatim; no mid-stream retry). SSE streams are relayed
  chunk-by-chunk. Default fallback: the MoE `mmangkad/Qwen3.6-35B-A3B-NVFP4`.
- **New gateway package `model_gear/gateway/`** — a pure-stdlib
  (`http.server` + `http.client`, no runtime deps) reverse proxy: `_routing.py`
  (pure name/alias/default routing + failover ordering), `_config.py` (env →
  routing table + server config), `server.py` (the `handle_post` failover seam,
  upstream client, and `ThreadingHTTPServer` handler), run as
  `python -m model_gear.gateway`.
- **`model init --fleet`** scaffolds the fleet templates
  (`docker-compose.yml` + `.env` + `Dockerfile.gateway`) and pins
  `MODEL_GEAR_VERSION` to the running release; **`model fleet up | down |
  status`** drives the deployment (`up`/`down` dry-run by default, `--apply` to
  commit; `status` is read-only and reports all three containers + the gateway
  `/health` + `/v1/models`).
- **Docs:** `docs/gateway-fleet.md` (topology, routing/failover, memory,
  verbs), `docs/qwen3.6-35b-a3b-nvfp4.md` (the MoE fallback), a README "fleet"
  section, and `model explain fleet` / `model explain gateway` entries.

### Changed

- `model_gear/runtime/_compose.py` gained a template registry
  (`SINGLE_TEMPLATES` / `FLEET_TEMPLATES`), a `templates=` argument on
  `scaffold_plan` / `write_scaffold` (single-model stays the default — existing
  callers unchanged), a `compose_up_build` helper, and `FLEET_CONTAINERS`.
- The fleet `.env` mirrors `VLLM_MODEL` / `VLLM_SERVED_NAME` /
  `VLLM_TOOL_CALL_PARSER` (= the primary) so the read-only single-model verbs
  (`status` / `whoami` / `doctor`) stay coherent on a fleet deployment.
  `model switch` remains single-model only.

### Fixed

## [0.8.1] - 2026-05-27

### Changed

- **SonarCloud cleanup (no behavior change).** Split `cmd_switch` into
  `_select_parser` / `_emit_dry_run` / `_apply_switch` helpers to bring its
  cognitive complexity under the gate, and hoisted the repeated `"(unset)"`
  literal in `model status` into a `_UNSET` constant.

## [0.8.0] - 2026-05-27

### Added

- **Per-model tool-call parser auto-selection.** New `model_gear/runtime/_parser.py`
  `infer_parser()` maps a model name to its parser (`qwen3_coder` for
  Qwen3-Coder / Qwen3.6, `hermes` for Qwen3 dense, unknown → leave untouched).
  `model switch` now picks the right parser automatically so tool calling keeps
  working across a switch without the caller remembering it; `--tool-call-parser`
  still overrides ([issue #13](https://github.com/agentculture/model-gear/issues/13)).
- **Post-switch / post-start tool-calling probe.** `model switch --apply` and
  `model serve --apply` now probe `tool_choice:"auto"` once the container is
  healthy and report PASS/FAIL (with the called tool names) — reusing the
  existing `assess` probe. `--no-probe` skips it; the probe never aborts the
  command (unreachable / HTTP 400 degrade to a FAIL result).
- **`model status` reports the active `tool_call_parser`** (`VLLM_TOOL_CALL_PARSER`),
  so "which gear am I in" is complete without `docker inspect`.

### Changed

- **`lepenseur` is retired; the deployed agent is now `model-gear`.** The tool and
  the deployed agent share one identity. Updated `culture.yaml` (`suffix: model-gear`),
  the `AGENTS.md` system prompt, `model whoami` / `learn` / `explain` output, the
  posting nick (`.claude/skills.local.yaml.example`), the compose/`.env` templates,
  `README.md`, and `CLAUDE.md` (the former "two identities" section now describes
  one).

### Fixed

## [0.7.0] - 2026-05-27

### Added

- **OpenAI tool/function calling** on the served vLLM model. The packaged compose
  template (`model_gear/templates/docker-compose.yml`) now serves with
  `--enable-auto-tool-choice` and `--tool-call-parser=${VLLM_TOOL_CALL_PARSER:-hermes}`,
  so `tool_choice:"auto"` requests return a `tool_calls` array instead of HTTP
  400. Additive — plain chat/reasoning is unaffected, no extra GPU/memory cost.
  Unblocks coder-agent harnesses that drive the model entirely through tool calls
  ([issue #9](https://github.com/agentculture/model-gear/issues/9)).
- **`VLLM_TOOL_CALL_PARSER`** env var (default `hermes`) + **`model switch
  --tool-call-parser`** — the parser is per-model: `hermes` fits Qwen3 dense
  (e.g. `Qwen3-32B`), while Qwen3-Coder / Qwen3.6 checkpoints emit the XML
  function format and need `qwen3_coder`. `switch` writes the var only when the
  flag is given, so retuning a model never clobbers its parser.
- **`model assess --tools`** — an opt-in tool-calling probe that verifies a
  `tool_choice:"auto"` request returns a `tool_calls` array naming a `finish`
  function. Degrades gracefully (a FAIL row, no abort) against a server that
  lacks the flags.

### Changed

### Fixed

## [0.6.0] - 2026-05-27

### Added

- **devague workflow trio** vendored under `.claude/skills/` (cite-don't-import):
  `think` (idea→spec), `spec-to-plan` (spec→plan), and `assign-to-workforce`
  (plan→parallel implementation) — the operator chain for the deterministic
  `devague` CLI. Authored in `agentculture/devague`, vendored via guildmaster;
  each carries `type: command` (load-bearing on the culture/agex backend, where
  a `SKILL.md` without `type:` is silently skipped). They drive the `devague`
  CLI at runtime (`uv tool install devague`), resolved portably by the wrappers.
- **`docs/skill-sources.md`** — provenance ledger recording the citation path
  and authoring origin of every vendored skill (the trio plus the six
  steward-sourced skills).

## [0.5.0] - 2026-05-27

Redesigned the repo around **running, assessing, and switching the local vLLM
model**. The model-ops logic that lived in the `model-runner` *skill* is now a
first-class CLI. lepenseur is still the deployed agent that consumes the served
model; model-gear is the tool that runs it.

### Added

- **Model-ops verbs** on the `model` CLI: `switch <model>`, `serve` (alias
  `start`) / `stop`, `status`, `assess` (correctness probes), `benchmark`
  (decode throughput + prefill), and `init` (scaffold a deployment dir). Write
  verbs (`switch`/`serve`/`stop`/`init`) are **dry-run by default** and require
  `--apply` (mutation-safety rule).
- **Scaffold-based deployment.** `docker-compose.yml` + `env.example` ship as
  packaged templates under `model_gear/templates/`; `model init` materialises
  them into `~/.model-gear` (default), a `TARGET`, or the local folder. Every
  model-ops verb resolves the deployment dir via `--compose-dir` →
  `$MODEL_GEAR_DIR` → `~/.model-gear`.
- Ported runtime modules (`model_gear/runtime/` + `model_gear/assess.py`),
  stdlib-only (`urllib`, fixed-argv `subprocess`), with full unit tests.
- `model overview` now folds in the currently-served model and the
  candidate-model list, filterable with `--current` / `--list`.

### Changed

- **PyPI distribution renamed `lepenseur` → `model-gear`; binary `lepenseur` →
  `model`; Python package `lepenseur` → `model_gear`.** Error class
  `LepenseurError` → `ModelGearError`. The `lepenseur` console script is removed.
- Agent-first verbs reframed for the tool: `whoami` reports tool/machine/served
  model/container health/agent; `learn` teaches the model-ops surface; `explain`
  catalog rewritten (`switch`/`assess`/`backend`/`models`/…).
- `doctor` is now **real** — checks docker availability, deployment scaffold,
  `.env` ↔ `culture.yaml` coherence, and `/health` reachability (a down model is
  a warning, not a failure).
- The `model-runner` skill is now a thin shim that `exec`s `model`; its
  `_assess.py` was removed (the logic lives in `model_gear/assess.py`).
- `AGENTS.md` / `culture.yaml` clarified: they describe the deployed `lepenseur`
  agent, not the repo. README + CLAUDE.md reoriented around model-gear.

### Fixed

- **BREAKING:** the vLLM container is renamed `lepenseur-vllm` → `model-gear-vllm`.
  A box running the old container must `docker compose down` under the old name,
  then `model init --apply` + `model serve --apply`.

## [0.4.0] - 2026-05-27

### Added

- `model-runner` skill (local, not vendored): `switch` the local vLLM runtime
  model and `assess`/benchmark it (stdlib `_assess.py` for correctness +
  throughput, host-side facts via the wrapper). Drives this repo's compose +
  `.env`; documented in CLAUDE.md and README. Mutating verbs (`switch`, `down`)
  are dry-run by default and require `--apply` (CLAUDE.md mutation-safety rule);
  `--port` defaults to `.env`'s `VLLM_PORT` (then 8000).

### Changed

- `docs/qwen3.6-27b-nvfp4.md`: filled with the live load-test (DGX Spark/GB10,
  2026-05-27). `mmangkad/Qwen3.6-27B-NVFP4` loads and serves under our vLLM image
  (no `--trust-remote-code`); ~7.9–8.0 tok/s decode, ~70 GB reserved, 29 GB
  weights. It is a hybrid Mamba/linear-attention vision-language model and is
  slower on decode than the 32B here — recommendation: **keep the 32B**. All
  pre-flight caveats (SGLang-only, multimodal, ModelOpt rc) validated/resolved.

## [0.3.0] - 2026-05-27

### Added

- `docs/qwen3-32b-nvfp4.md`: per-model doc for the current runtime model, with a
  live test on DGX Spark (GB10) — `nvcr.io/nvidia/vllm:26.04-py3` (engine
  `0.19.0+...nv26.04`), ~9.7 tok/s decode (batch=1), ~2,800 tok/s prefill, ~72 GB
  reserved at `gpu-memory-utilization=0.6`, correctness verified.
- `docs/qwen3.6-27b-nvfp4.md`: per-model doc for candidate
  `mmangkad/Qwen3.6-27B-NVFP4`. Its `Qwen3_5ForConditionalGeneration` arch is
  registered in the current vLLM image (so the same compose can serve it); live
  load-test/benchmark tracked by issue #6.
- README "Per-model notes" linking both docs.

### Fixed

- `docker-compose.yml`: corrected the `--reasoning-parser=qwen3` comment — on the
  nv26.04 build the `<think>` trace is returned in the `reasoning` field, not
  `reasoning_content`.

## [0.2.0] - 2026-05-27

### Added

- `docker-compose.yml` + `.env.example`: a local vLLM server (NGC
  `nvcr.io/nvidia/vllm` image) that serves the runtime model as an
  OpenAI-compatible API on `:8000` for the `acp` backend, tuned for DGX Spark
  (GB10 Blackwell, 128 GB unified memory).
- README "Running the model locally (vLLM)" section.

### Changed

- Switched lepenseur's runtime model from
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` to `nvidia/Qwen3-32B-NVFP4`
  across `culture.yaml`, `AGENTS.md`, `lepenseur/explain/catalog.py`, `README.md`,
  and `CLAUDE.md` (32B dense NVFP4 reasoning model with a thinking mode).

## [0.1.0] - 2026-05-22

### Added

- Initial CLI/PyPI sibling scaffold (copied and adapted from the `lecodeur`
  twin): top-level `lepenseur` package with the `lepenseur` console script.
- Read-only verbs: `whoami`, `learn`, `explain`, `overview`, and a `cli`
  noun with `cli overview`.
- `doctor` verb shipped as a rubric-shaped stub; real self-diagnosis semantics
  for a thinking ("non-doer") agent are deferred to a follow-up.
- Runtime identity files: `AGENTS.md` and `culture.yaml` (acp backend,
  `vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).
- CI: `tests.yml` (test + lint + `afi cli doctor . --strict` gate +
  version-check) and `publish.yml` (PyPI/TestPyPI via Trusted Publishing).
- Six vendored skills under `.claude/skills/` (cicd, communicate, version-bump,
  run-tests, sonarclaude, doc-test-alignment), provenance: steward.
