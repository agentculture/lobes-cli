# stt readiness truth 155

> lobes never reports a lobe unhealthy while it is correctly serving — STT readiness tells the truth about CPU fallback
> instruction: Change evaluate_readiness to be device-agnostic, split the audio aggregate per-lane, give the CLI a live audio probe so capabilities matches fleet status, and make the gateway 503 body name its real cause. Verify with the Front_Center.wav known-content probe through the gateway.

## Audience

- Operators of a lobes fleet, and downstream consumers of the OpenAI-compatible audio surface — webcam-cli, reachy-mini-cli, and any caller of POST /v1/audio/transcriptions or GET /v1/realtime — plus the agent that reads lobes fleet status / lobes capabilities to decide whether STT is usable at all.

## Before → After

- Before: The same gear answers four surfaces four different ways at the same instant (verified live on spark, 2026-07-24): lobes fleet status says unhealthy, lobes capabilities says loaded yes, a container-direct POST returns 200 with a CORRECT transcript, and the gateway returns 503 'audio backend not ready yet (warming up) — retry shortly'. An operator cannot tell which to believe, and the one surface that is right (the 200) is the one nothing routes to.
- After: A lobe that is correctly serving is never reported unhealthy nor refused at the gateway. STT readiness is device-agnostic (model loaded on ANY device reports ready, and every audio lane routes), stt and tts fail independently, lobes fleet status and lobes capabilities give the same verdict for the same gear, and any refusal that does happen names its real cause at the gateway boundary instead of a generic warming-up message.

## Why it matters

- Consumers are blocked by a gear that works. webcam-cli's audio half could not be demonstrated through the documented gateway path while Parakeet was transcribing correctly the entire time; and 'retry shortly' sends a caller into an unbounded retry against a state that never clears on its own — the STT container had been serving on CPU for five hours.

## Requirements

- lobes fleet status and lobes capabilities stop disagreeing about the same gear. Verified live: fleet status shows model-gear-stt unhealthy (docker health from the 503 probe) while uv run lobes capabilities prints stt ... yes <http://localhost:8001> — because lobes/roles.py build_role_registry is called with audio_ready=None from the CLI, and audio_ready_signal then falls back to the CONFIG fact (audio_configured), never dialing anything.
  - honesty: Run against the same box in the same second, lobes fleet status and lobes capabilities give the SAME ready verdict for stt — and capabilities' verdict changes when the container's real state changes, proving it is a live probe rather than an echo of .env config.
- the gateway 503 body names the real reason at the boundary. Measured live: POST :8001/v1/audio/transcriptions returns 503 {"message":"audio backend not ready yet (warming up) — retry shortly","type":"upstream_unavailable"} — but the gear is not warming up, it has been serving on CPU for 5 hours. 'retry shortly' is actively wrong advice for a persistent degraded state.
  - honesty: A caller receiving a non-200 from /v1/audio/* can determine the cause from the response body alone, with no access to docker logs; and the body never says 'warming up / retry shortly' for a state that has already persisted past the container's start_period.
- the 200-while-503 combination is eliminated: either the gateway routes transcriptions to the working endpoint, or the endpoint stops answering 200. Both halves are observable today at the same instant (container-direct 200 with a correct transcript, gateway 503).
  - honesty: There is no instant at which a container-direct POST /v1/audio/transcriptions returns 200 with a correct transcript while the gateway returns non-200 for that same payload.
- REPLACES c3 under decision c17: evaluate_readiness stops gating on CUDA at all — model_loaded alone determines ready, so a model loaded on CPU reports 200 ready exactly like one on GPU. The cuda_ok probe is demoted from a gate to reported detail (the response names the device it actually loaded onto) so an operator can still see a GPU regression without it blocking traffic. Both copies change together: lobes/realtime/_readiness.py and the vendored lobes/templates/fleet/_readiness.py.
  - honesty: A stt container with CUDA unavailable but a loaded NeMo model returns 200 from /v1/health/ready, docker marks it healthy, and the response body still reports device=cpu so the GPU regression stays visible without gating traffic.
- The realtime composite is today INCOMPLETE and the split is the moment to fix it: lobes/realtime/app.py's ready() probes ONLY settings.tts_url and settings.stt_url, so it reports a session consumable while (a) Silero VAD may be unavailable and (b) the generate lane may be unreachable. On this very box the voice lane's generate target is multimodal, which is MULTIMODAL_FEASIBLE=false and PROXIED to <http://orin.tail0be7e0.ts.net:8000> — so a realtime session here depends on a REMOTE peer that the readiness aggregate never dials.
  - honesty: With Silero unavailable, or with the voice generate lane unreachable (including a declared-but-down proxied peer), the realtime readiness reports not-ready and names which capability is missing — instead of today's 200 based on tts+stt alone.

## Honesty conditions

- A container that transcribes a known-content WAV correctly is reported ready by /v1/health/ready, healthy by docker, ready by lobes capabilities AND routable by the gateway — all four surfaces agreeing — regardless of whether the model sits on GPU or CPU.
- lobes ships no code that attempts to REPAIR container GPU visibility — no auto-restart, no cgroup manipulation, no device re-attach. The change is confined to reporting and routing paths; a container that has lost its device cgroup still requires an operator-initiated restart to regain the GPU.
- Each named consumer actually exercises the surface this spec changes: webcam-cli and reachy-mini-cli both reach POST /v1/audio/transcriptions through the gateway rather than container-direct, and at least one agent surface (lobes capabilities or fleet status) is read programmatically to decide whether STT is usable.
- All four contradictory signals are reproducible on demand rather than being a one-off observation: with the stt container's GPU visibility removed, fleet status reports unhealthy, capabilities reports loaded yes, a container-direct POST returns 200 with a correct transcript, and the gateway returns 503 — all captured in one evidence transcript.
- The blockage is real, not hypothetical: a consumer following only the documented gateway path cannot obtain a transcript while the gear is transcribing correctly — demonstrated for webcam-cli on host spark, 2026-07-24.
- Every clause of the after-state is independently checkable: device-agnostic ready, per-lane independence, CLI/gateway agreement, and a cause-naming refusal body each have a test that FAILS against today's code and PASSES after the change.
- The CPU-fallback condition can be SIMULATED (e.g. CUDA_VISIBLE_DEVICES='' on the stt container so NeMo loads on CPU) rather than requiring a real cgroup regression to occur — so the three success checks are runnable on demand as a scripted live probe, not only by waiting for the fault to recur.
- With chatterbox stopped and stt healthy, POST /v1/audio/transcriptions returns 200 while POST /v1/audio/speech returns a non-200 that names tts as the unavailable lane.
- With the NeMo model deliberately unloaded or failed, /v1/health/ready still returns 503 and docker still marks the container unhealthy — the loaded-model gate is intact — while a CUDA-less but loaded container returns 200.
- After the change, GET /v1/realtime still refuses when any capability the session needs is missing, while POST /v1/audio/transcriptions succeeds with tts down — proving the composite and the per-lane signals coexist rather than one replacing the other.

## Success signals

- Reproducible on a box whose stt container has lost GPU visibility but still transcribes: (1) POST /v1/audio/transcriptions through the gateway returns 200 with a correct transcript for the known-content probe /usr/share/sounds/alsa/Front_Center.wav -> 'Front, center.'; (2) lobes fleet status and lobes capabilities report the same verdict for stt; (3) stopping ONLY chatterbox leaves transcriptions still answering 200.

## Scope / boundaries

- the GPU-visibility loss itself is a HOST/runtime regression, not lobes code, and lobes does not own fixing it. Proved live: a fresh docker run --rm --gpus all nvidia/cuda container sees 'GPU 0: NVIDIA GB10', while the 5-hour-old model-gear-stt and model-gear-chatterbox both report 'Failed to initialize NVML: Unknown Error' with torch.cuda.device_count()==0 and CUDA_VISIBLE_DEVICES unset — the running containers lost their device cgroup mid-life (systemd Reloading events at 16:52 and 20:32 in journalctl). vLLM containers survive because they already hold a CUDA context. lobes owns REPORTING this honestly, not repairing the container runtime.
- REPLACES c7 under decision c17: the #39 / c16 intent survives only IN PART. /v1/health/ready must still refuse with 503 when the model is NOT loaded — it never returns to the unconditional 200 of the pre-#39 liveness-only handler, which remains the bug #39 fixed. But the CUDA half of the gate is deliberately DELETED, not softened: cuda_ok stops being a gate and becomes reported detail. Any doc, comment or test asserting 'ready implies CUDA live' must therefore be UPDATED rather than preserved — including _readiness.py's own docstring, which currently states the model-loaded AND cuda-live rule as the contract.
- USER CORRECTION: the per-lane split (c18) applies to the BATCH facade ONLY. The realtime bridge deliberately WRAPS four capabilities — stt, tts, VAD and the multimodal/generate lane — and a voice-to-voice session genuinely needs all of them at once, so GET /v1/realtime KEEPS a composite readiness verdict. Splitting aggregate_audio_ready must ADD per-lane answers for /v1/audio/transcriptions and /v1/audio/speech without DELETING the all-of-them signal the realtime session depends on.

## Non-goals

- the bare python3 in the stt healthcheck is NOT a bug and needs no change — the reporter's secondary observation is a false alarm. Dockerfile.parakeet is FROM scitrera/dgx-spark-vllm:0.16.0-t4, which ships /usr/bin/python3 (Python 3.12.3, verified in-container); the adjacent chatterbox comment warns about the DIFFERENT nvidia/cuda base, where python3 genuinely is absent (docker exec model-gear-chatterbox python3 fails with 'executable file not found in $PATH').
- GET /v1/realtime returning 404 is deployment version skew, not a route fault, and is out of scope here. The deployed gateway image reports lobes 0.52.3 (docker exec model-gear-gateway python3 -c 'import lobes'), and the /v1/realtime tunnel shipped in 0.53.0 — so is_realtime_path does not exist in the running gateway. In current code lobes/gateway/_realtime.py refuses a plain GET with 426 not_an_upgrade, never 404.

## Assumptions

- SETTLED EMPIRICALLY on host spark 2026-07-24: reading 1 in issue #155 is correct — Parakeet serves correct transcripts on CPU. A known-content speech WAV (Front_Center.wav, from the stt container's own /usr/share/sounds/alsa) POSTed to 127.0.0.1:9002 returned {"text":"Front, center."} in 1.68s, and nvidia-smi shows NO stt process holding GPU memory (only 4 VLLM EngineCore + reachy-mini-cli). The 200s are genuine, not hollow.
- any fix to the STT probe reaches a live box only via an image REBUILD plus a MODEL_GEAR_VERSION bump — the readiness code runs inside the stt container, which pins lobes-cli==${MODEL_GEAR_VERSION}. This deployment is already skewed three ways: gateway 0.52.3, realtime 0.54.0, CLI/repo 0.54.1.

## Scope exploration

- `s1` — `live fleet on host spark (docker ps / nvidia-smi / container-direct POST :9002)`: Settles issue #155's central unknown: a known-content speech WAV transcribes correctly ('Front, center.') while no stt process holds GPU memory — reading 1 (CPU-serving), not reading 2 (hollow 200s)
  - seeds: `c2`, `c6`
- `s2` — `lobes/realtime/_readiness.py (+ vendored lobes/templates/fleet/_readiness.py)`: evaluate_readiness(model_loaded, cuda_ok) is two-state by construction — the CPU-serving case has no representable value, so it collapses onto the same 503 as a dead model. Both copies must change together (cite-don't-import)
  - seeds: `c3`, `c7`
- `s3` — `lobes/templates/fleet/listen_server.py`: get_model() calls ASRModel.from_pretrained() and never pins a device, so NeMo silently chooses CPU when CUDA is absent at load time; the /v1/health/ready handler then probes torch.zeros(1, device='cuda') — a signal about the PROCESS, not about the loaded model's actual device
  - seeds: `c3`
- `s4` — `lobes/roles.py build_role_registry / _resolve_audio_role`: the CLI passes audio_ready=None, so audio_ready_signal falls back to the config fact audio_configured — this is the structural reason lobes capabilities says yes while lobes fleet status says unhealthy
  - seeds: `c4`
- `s5` — `lobes/gateway/server.py probe_audio_ready + the /v1/audio/* 503 body`: the gateway's only audio signal is the bridge aggregate, and its refusal text is hardcoded 'warming up — retry shortly' with no room to name a persistent degraded cause
  - seeds: `c5`
- `s6` — `lobes/realtime/audio_facade.py aggregate_audio_ready`: STT and TTS readiness are ANDed into one verdict for the whole /v1/audio/* namespace, so the two lanes cannot fail independently
  - seeds: `c6`
- `s7` — `container GPU visibility (docker inspect DeviceRequests, in-container nvidia-smi, fresh --gpus all container, journalctl)`: stt/chatterbox both hold a valid nvidia DeviceRequest and NVIDIA_VISIBLE_DEVICES=all yet see zero GPUs, while a fresh container sees GPU 0: NVIDIA GB10 — the loss is a runtime/cgroup regression on already-running containers, outside lobes' repair scope but inside its reporting scope
  - seeds: `c8`
- `s8` — `lobes/templates/fleet/Dockerfile.parakeet + docker-compose.audio.yml healthchecks`: the stt healthcheck's bare python3 is correct — the parakeet base (scitrera/dgx-spark-vllm) ships /usr/bin/python3; only the chatterbox nvidia/cuda base lacks it, which is what the adjacent comment warns about. Reporter's secondary observation is a false alarm
  - seeds: `c9`
- `s9` — `lobes/gateway/_realtime.py + deployed container lobes versions`: a plain GET /v1/realtime refuses with 426 not_an_upgrade in current code; the observed 404 comes from the deployed gateway being lobes 0.52.3, which predates the 0.53.0 route entirely — version skew, not a route fault
  - seeds: `c10`, `c11`
- `s10` — `MODEL_GEAR_VERSION image pin (deployed .env vs pyproject.toml)`: the readiness fix ships INSIDE the stt image, so it needs a rebuild plus a version bump to reach a box; this deployment already runs gateway 0.52.3 / realtime 0.54.0 against a 0.54.1 repo
  - seeds: `c11`
- `s11` — `live recovery check (docker restart model-gear-stt model-gear-chatterbox)`: CONFIRMS the runtime-regression diagnosis: after restart both containers see GPU 0 again, stt /v1/health/ready returns 200 in ~20s, both report healthy, and the gateway POST /v1/audio/transcriptions now returns 200 {"text":"Front, center."} — nvidia-smi shows stt (2894 MiB) and chatterbox (3292 MiB) back on the GPU. The lobes-side defects (two-state readiness, CLI/gateway disagreement, misleading 503 text, coupled audio lane) are NOT fixed by this and remain in scope
  - seeds: `c8`
- `s12` — `lobes/realtime/app.py ready() + this box's .env voice-lane wiring`: USER-FLAGGED: realtime wraps stt+tts+VAD+multimodal, but ready() probes only tts_url and stt_url — VAD and the generate lane are absent from the aggregate, and on this box that generate lane is MULTIMODAL_FEASIBLE=false proxied to orin, so realtime readiness never dials the peer it actually depends on
  - seeds: `c21`, `c22`

## Decisions

- USER DECISION (q1): readiness is DEVICE-AGNOSTIC. A loaded model reports 200 ready regardless of the device it landed on, and the gateway routes every audio lane including /v1/realtime. Rejected alternatives: a separate degraded rung, and batch-routes-but-realtime-refuses. Residual risk accepted: a CPU-serving stt makes the server_vad realtime session ~1.1x realtime, so a realtime caller may fall behind rather than receive a clean refusal.
- USER DECISION (q2): the audio readiness aggregate SPLITS PER-LANE. /v1/audio/transcriptions gates on stt alone and /v1/audio/speech gates on tts alone, replacing aggregate_audio_ready's single ANDed verdict, so neither lane is hostage to the other's health.

## Hard questions

- contradiction with c17? RESOLVED: c3 was rejected and replaced by c19 (device-agnostic readiness) under user decision c17. Resolved by frame edit — devague has no move to resolve a hard question, and a blocking hard question on a REJECTED claim still blocks converge. (blocking)
