# Build Plan — realtime voice-to-voice + astro test site (#151)

slug: `realtime-voice-to-voice-astro-test-site-151` · status: `exported` · from frame: `realtime-voice-to-voice-astro-test-site-151`

> lobes serves full realtime voice-to-voice: a spoken turn over /v1/realtime gets a spoken, interruptible reply back on the same WebSocket, and an Astro browser test site in the org site's visual language drives it live — mic in, animated event stream, audio out

## Tasks

### t1 — Wire codec module: lobes/realtime/_wire.py + tests — base64 event codecs for input_audio_buffer.append (parse) and response.audio.delta (serialize), delta chunk sizing, malformed-input rejection; stdlib-only

- acceptance:
  - append event with valid base64 decodes to exact PCM bytes; malformed base64/JSON yields a named error value, never an escaping exception
  - delta serialization round-trips PCM byte-exact; tests pass with no [realtime] extra installed

### t2 — Floor/turn state machine: lobes/realtime/_floor.py + tests — pure state class (listening/transcribing/responding/speaking), injected callbacks + clock, cancel-both on interrupt, truncation of undelivered chunks, per-stage deadlines

- covers: c17, h9, h5, c35, h26, c31, h23
- acceptance:
  - offline tests walk the full state graph including an interrupt arriving in every responding/speaking state
  - timeout expiry in each stage returns the floor to listening with the named error within the configured bound (injected clock)
  - an interrupt mid-delivery stops the undelivered remainder (fake sink) and emits one interruption event with the truncated marker
  - a SpeechStarted arriving while speaking is consumed as the barge-in trigger; the segmenter module is not modified

### t3 — Session schema + history: extend lobes/realtime/_session.py + tests — new EventType/ErrorCode/SessionState members (response lifecycle, interruption, generate/tts/timeout errors), per-session in-memory history + system prompt, teardown drops all

- covers: c5, c34, h25
- acceptance:
  - every new event dataclass is frozen and event_to_dict-serializable; the floor holder is explicit in the schema
  - a two-turn offline conversation asserts the second generate request carries the first exchange; history lives only on the Session object
  - teardown from every new state releases history and bookkeeping (idempotent, any state)

### t4 — Ears-only wire migration: app.py input path parses base64 append events via _wire (binary path removed); the #149 offline event-sequence tests are UPDATED to the event format, not deleted

- depends on: t1
- covers: c39, h29, h30
- acceptance:
  - updated #149 tests assert the transcription-only event sequence 1:1 over base64 append input
  - grep gate: no code path accepts or emits raw binary audio frames after the change

### t5 — Turn request shaping: lobes/realtime/_turn.py + tests — stdlib builder for the chat/completions payload (history, env model default multimodal, enable_thinking=false), response parsing, 404 role_infeasible mapped to the named error with hosted_by

- covers: c4, h4
- acceptance:
  - payload builder emits chat_template_kwargs enable_thinking=false and the env-resolved model; empty OPENAI_MODEL means gateway default-routing
  - a fake 404 role_infeasible body maps to the named error event carrying the hosted_by peer hint; no silent fallback to another lane exists

### t6 — Conversation route wiring: app.py response.create trigger wires _floor + _turn + _wire + tts_client (cancel_event armed), per-stage timeouts from settings, delta emission; TTS-out path calls no resampler; gateway diff zero lines

- depends on: t2, t3, t4, t5
- covers: c2, h30, c3, h3, c6, h6, c31, c35
- acceptance:
  - the route shell stays thin pragma-no-cover wiring — every decision lives in the stdlib modules
  - no resample call exists in the TTS-out path and the lobes/gateway diff is zero lines
  - a non-opt-in session still emits the transcription-only sequence (updated tests unmodified by this task)

### t7 — TTS gate re-scope: tts_client.py/_settings.py — per-lane semaphores (voice lane vs batch) or a raised default, chosen and documented with rationale

- covers: c33, h24
- acceptance:
  - an offline test proves a voice-lane synthesis does not queue behind a saturated batch lane at the shipped default
  - env.audio.example states the chosen default and its rationale

### t8 — Deployment passthrough + doctor heal: docker-compose.audio.yml environment + env.audio.example gain BARGE_IN_WINDOW_MS/BARGE_IN_MODEL, system-prompt and every new voice-turn key; doctor heal list + tests extended

- depends on: t7
- covers: c16, h8
- acceptance:
  - a test asserts every key _settings.py reads appears in BOTH the compose environment block and env.audio.example
  - doctor --fix heals each absent key without rewriting existing .env lines

### t9 — In-repo client migration: scripts/realtime-smoke.py + scripts/realtime-voice-loop.py speak the base64 event wire (append in, delta out); duplex rules (pong pings, select reads, lock-guarded writes) carry over; helper tests updated

- depends on: t1
- covers: c42, h31
- acceptance:
  - updated offline helper tests cover the append/delta framing arithmetic
  - grep gate: no in-repo client sends raw binary audio frames; the smoke script passes live against the deployed base64 wire (deferred to the acceptance task)

### t10 — Site scaffold + design system: site/ Astro 7 project — org tokens/fonts/keyframes/data-reveal ported from ../org/site-astro global.css, layout + shell, reduced-motion kill switch, AA pairs both themes, fonts self-hosted

- covers: c8
- acceptance:
  - npm run build succeeds on Node 22; the global CSS carries the prefers-reduced-motion kill and the documented AA token pairs in both themes
  - the built output makes zero external network requests (fonts bundled)
  - package.json is pinned COMPLETE in this task — every dependency wave-2 site tasks (t11/t12/t13) will need is declared up front, so no wave-2 task edits package.json

### t11 — Mic + playback island: site client JS — getUserMedia({echoCancellation:true}) + AudioWorklet capture to 24 kHz PCM16 base64 append events; delta player; playback stops on interruption event; start-control gesture gating; permission-denied/not-found as distinct states

- depends on: t10
- covers: c37, h28, c6, h6, c31, h23
- acceptance:
  - mic and playback arm only from the start control; NotAllowedError/NotFoundError render their own distinct states
  - local playback stops on the interruption event; grep gate: no mic-mute logic anywhere in site code
  - the worklet emits well-formed append events (fixture-tested encode path)

### t12 — Event stream UI: site components rendering the live event log — every event type with timestamps, VAD boundary timing visibility for knob tuning, each named error code visually distinct from silence and from disconnect

- depends on: t10
- covers: c8, h7, c27, h19
- acceptance:
  - a fixture replay of every event type renders each distinctly; silence renders nothing; disconnect renders its own state
  - boundary events display at_ms timing so VAD_THRESHOLD/VAD_SILENCE_MS/VAD_PREFIX_PADDING_MS effects are observable

### t13 — Local proxy + dev flow: Vite server.proxy ws upgrade-header injection (standalone tiny proxy as fallback), site README documenting ssh -L as the primary secure-context flow and the local-only stance

- depends on: t10
- covers: c22, h12, c36, h27, c24, h18
- acceptance:
  - served assets and config responses contain no Authorization value or key material (grep of built output + devtools check)
  - with the proxy down the site cannot connect at all; the key lives only in the proxy process env
  - README documents ssh -L as primary (mkcert as alternative) and states local-only: no deploy workflow exists under .github/workflows

### t14 — Site CI job: tests.yml gains a site-build job (npm ci && npm run build, Node 22); sonar.sources, version-check path filters, and hatch packaging untouched

- depends on: t10
- covers: c19, h11
- acceptance:
  - a deliberately broken site file fails the job (verified once on the PR branch)
  - the built wheel is byte-identical with and without site/ present

### t15 — Boundary guard sweep: assert the untouched surfaces — batch /v1/audio/* handlers, catalog.py, gateway do_GET dispatcher and tunnel, header-only auth tests, import-isolation tests — all pass unmodified; PR checklist records the zero-diff surfaces

- depends on: t6
- covers: c11, h14, c12, h15, c20, h16, c23, h17
- acceptance:
  - the tunnel refusal test for an armed STT_PEER_PROXY, the header-only auth tests, and the import-isolation tests pass unmodified
  - the PR description carries the zero-diff checklist: batch handlers, catalog.py, do_GET dispatcher, no query-param/subprotocol auth anywhere in lobes/gateway

### t16 — Docs flip: realtime-pipeline.md (drop half-duplex IOU, document conversation opt-in + base64 wire + new events), openai-api.md endpoint table, gateway-fleet.md, explain catalog _REALTIME — same PR; before-state citations kept as history; doc-test-alignment run

- depends on: t6
- covers: c18, h10, c28, h20
- acceptance:
  - grep gate: no doc or explain entry claims the session is audio-in-only/half-duplex except as explicit history
  - all four surfaces change in the same PR; the before-state remains cited from f1e6ffa, the voice-loop docstring, and #149 non-goal c15
  - the reachy coordinated wire break (c40) is documented with the migration note

### t17 — Live acceptance + evidence: the Spark run through the site via ssh -L — speak, hear the reply, interrupt mid-playback; speakers-at-volume AEC test; one A/B audio exchange against the OpenAI realtime service (or record precisely why not); transcript under docs/evidence/ BEFORE any validated wording

- depends on: t6, t9, t11, t12, t13
- covers: c1, h1, c29, h21, c41, h32
- acceptance:
  - the transcript records a full spoken exchange, an interruption mid-playback visible in the event stream, and the speakers-at-volume AEC result
  - the run goes through the documented ssh -L flow from the operator laptop with a real microphone
  - the transcript lands under docs/evidence/ before any doc, README, or CLAUDE.md says validated (#108)

### t18 — User mute / mic-off control: an explicit operator affordance on the site (mute mic, mic off/release device), distinct from and never triggered by playback; the no-mic-mute gate narrows to forbidding AUTOMATIC mute-during-playback (deviation d1)

- acceptance:
  - a user can mute the mic and turn it off from the UI; state is visible and distinguishable by more than colour, and the device is genuinely released on mic-off
  - the narrowed gate still fails if any code path mutes the mic in response to a playback/response event — asserted by test, with the deviation d1 rationale cited in the test
  - muted state is honest in the event stream: the operator can tell "muted" from "silence" from "disconnected"

### t19 — Site conversation arming + vocabulary sync: a conversation toggle that sends response.create (default OFF so ears-only stays the default), and sync the site error vocabulary with the servers new invalid_wire_event code

- acceptance:
  - with the toggle off the session is byte-identical to ears-only; with it on, a committed turn produces a spoken reply over the same socket
  - the site error vocabulary matches _session.ErrorCode exactly, with a fixture per code, and a test fails if the two drift

## Risks

- [unknown_nonblocking] barge_in_model semantics are undefined in-tree — window-only barge-in ships; the knob stays declared/unconsumed-beyond-threading until the live run shows window-only is insufficient (frame park v3) (task t2)
- [unknown_nonblocking] client-AEC quality at speakers-at-volume is unproven — if browser echoCancellation underperforms, the session transcribes the machine during playback; the acceptance run includes the named test and barge_in_model is the recorded mitigation (frame park v4) (task t17)
- [unknown_nonblocking] Vite ws upgrade-header injection is unverified on this Vite major — the standalone tiny proxy is the pinned fallback; discovering this mid-build must not stall the wave (c22 instruction) (task t13)
- [unknown_nonblocking] the reachy-mini-cli adaptation is an external-repo commit outside this plan — until it lands, the deployed robot cannot stream (coordinated break accepted, frame decision c40); sequencing with the lobes deploy is the operator call (task t4)
- [unknown_nonblocking] full-read TTS synthesis delays the first delta for long replies — if live latency disappoints, sentence-split pipelining (synthesize chunk 1 while speaking it) is the lever; not built preemptively (task t6)
- [follow_up] concurrent sessions stay unvalidated by this acceptance run (single-operator site) — the #149 debt item is retired only for the real microphone; concurrency validation folds into the gateway-hardening sibling issue (frame parks v2/v5 lineage)
- [follow_up] full OpenAI Realtime API parity is a named follow-up: this plan adopts the audio-path event shapes only (frame park v5)
- [unknown_nonblocking] a user-initiated mute during an active response is a new interaction the floor machine never modelled: muting is not barge-in, so it must NOT interrupt the reply — verify the two paths stay distinct (deviation d1) (task t18)
