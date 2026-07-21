# realtime voice-to-voice + astro test site (#151)

> lobes serves full realtime voice-to-voice: a spoken turn over /v1/realtime gets a spoken, interruptible reply back on the same WebSocket, and an Astro browser test site in the org site's visual language drives it live — mic in, animated event stream, audio out
> instruction: acceptance procedure mirrors docs/evidence/README-realtime-acceptance.md: fleet up with --audio, drive the session from the Astro site with a real microphone, save the event log as the transcript

## Audience

- voice-capable clients of the fleet and the humans behind them: reachy-mini-cli (firmware-AEC robot, ears-only consumer that must not regress), native/terminal clients like the f1e6ffa voice loop (header-auth capable), the developer at the local Astro site experiencing the surface, and the operator reading the live event stream to tune VAD knobs

## Before → After

- Before: a voice conversation lives in the client: scripts/realtime-voice-loop.py stitches ears/brain/mouth across three endpoints, half-duplex by design (mic muted through the whole synthesize-and-play window), interruption is impossible, and the only observability is terminal prints — VAD tuning is guessed at, not observed
- After: a spoken turn over ONE WebSocket gets a spoken reply on the same connection — commit, generate, synthesize all server-side — and speaking during playback interrupts it; the local Astro site renders the whole exchange live (boundaries, transcripts, reply text, audio out, interruptions, named errors) in the org visual language, so the surface is experienced in a browser instead of inferred from prints

## Requirements

- a committed turn optionally triggers a server-side generate + TTS reply streamed back over the SAME WebSocket — the extension point exists: app.py _pump_session currently IGNORES text/control frames, citing the #149 spec non-goal (no response.create, no mid-session session.update), and protocol.py already ships dormant gen_response_id/gen_content_part_id for exactly this
  - instruction: the turn machinery lands in stdlib-only modules beside _session.py/_segmenter.py; app.py stays a pragma-no-cover shell wiring them to real httpx/TTS
  - honesty: a session that never opts into conversation emits the transcription-only event sequence on the new base64 wire (offline-asserted with the UPDATED #149 tests), and the opt-in turn flow commit-generate-TTS-out is a scripted offline test against fake backends
- audio-out over the session needs NO resample and NO gateway change: protocol.py pins TTS_SAMPLE_RATE=24000 == CLIENT_SAMPLE_RATE ("matches, no resample"), tts_client.synthesize already runs in the bridge container, and the gateway tunnel is already full-duplex (run_tunnel pumps both directions in parallel threads)
  - instruction: assert by unit test that the TTS-out path calls no resampler — audio-out frames reuse tts_client.synthesize PCM verbatim; the gateway tunnel diff must be zero lines
  - honesty: no resample call exists in the TTS-out path (24 kHz end to end) and the gateway diff for audio-out is zero lines — the tunnel already relays both directions
- the server-side brain call is already configured but unused: _settings.py ships openai_base_url (default <http://gateway:8000>), openai_api_key, openai_model ("" = gateway default-routes) in the bridge env — the voice lane defaults to model=multimodal with chat_template_kwargs {"enable_thinking": false} per issue #151 (measured ~1 s to a short reply vs cortex reasoning latency), operator-overridable via OPENAI_MODEL
  - instruction: read OPENAI_MODEL/OPENAI_BASE_URL from settings at turn time; default the voice lane to multimodal with enable_thinking=false; map a gateway 404 role_infeasible to the named error event, passing the hosted_by peer hint through
  - honesty: the voice-lane default is env-tunable (OPENAI_MODEL) and, with the default lane infeasible on the deployed shape, a voice turn yields a NAMED error event derived from the gateway 404 — never a silent fallback to another lane
- turn-taking state must be explicit in the event schema (who holds the floor), not implied by event ordering: SessionState today is IDLE/SPEECH/TRANSCRIBING/CLOSED — no responding/speaking state exists — and the event schema is deliberately small (6 event types, 3 error codes), so voice-to-voice adds states + response/audio-out events to _session.py
  - instruction: extend SessionState + EventType + ErrorCode in _session.py, the module that owns the schema; keep every new dataclass frozen and event_to_dict-serializable
  - honesty: the floor holder is explicit in the event schema (a state field or event family), and the offline suite walks the full state graph — including an interrupt arriving in every responding/speaking state
- barge-in arms the shipped-but-dormant knobs: _settings.py has barge_in_window_ms (default 750) and barge_in_model, read from env and never consumed anywhere; the voice-loop script proves the alternative — its muted-mic half-duplex is explicitly "what barge-in cannot do" (its own docstring: without AEC the session transcribes the machine talking to itself)
  - instruction: arm the shipped barge_in_window_ms/barge_in_model knobs; thread them through compose env per c16
  - honesty: with audio-out playing, injected speech stops playback within barge_in_window_ms and emits the interruption event; the browser site does barge-in with echoCancellation only — no mic muting anywhere in the site code
- the Astro test site reuses the org site VISUAL LANGUAGE, not its architecture: design tokens (dawn palette, --mesh-*/--accent, Fraunces + Albert Sans variable fonts), the node-breathe/halo-breathe keyframes, the data-reveal IntersectionObserver system, and terminal-pane chrome all port directly — but org has NO live-network pattern anywhere (zero fetch/WebSocket/EventSource under src/, philosophy "replay captured data, never dial a live machine"), so the mic + WS event-stream island is genuinely new client JS
  - instruction: port the design tokens + keyframes + data-reveal system from ../org/site-astro global.css; the live WS island is new client JS bundled by Astro, not is:inline
  - honesty: the site meets the org bar it borrows: all animation dies under prefers-reduced-motion, text/background pairs hold WCAG AA in both themes, and each named error code renders visually distinct from silence and from disconnect
- every new knob must be threaded through the deployment, not just read: docker-compose.audio.yml already passes OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL (currently dead config — nothing in app.py consumes them) but NO BARGE_IN_* keys — the #149 s4 lesson applies verbatim: a settings key the compose env never passes silently pins to its default inside the container; compose environment block + env.audio.example + doctor-heal coverage move together
  - instruction: add BARGE_IN_WINDOW_MS/BARGE_IN_MODEL plus every new voice-turn key to docker-compose.audio.yml environment AND env.audio.example; extend the doctor-heal key list and its test in the same change
  - honesty: every key _settings.py reads exists in docker-compose.audio.yml environment AND env.audio.example, and doctor --fix heals absent ones — asserted by a test, not by inspection
- offline tests for the turn-taking state machine per the established split: all 188 realtime-adjacent tests are stdlib-only or loopback-only (no fastapi/torch in CI), app.py route shells are pragma-no-cover and coverage-omitted in pyproject — the new floor/response/interrupt logic must land in stdlib modules the offline suite fully covers, with the route shell live-tested (issue AC5)
  - instruction: model the floor/turn machine as a pure state class in a new stdlib module (like _segmenter.py: injected callables, no I/O); test every transition offline, including an interrupt from each responding/speaking state
  - honesty: CI with no [realtime] extra installed runs the new turn-taking tests green; the new logic lives in stdlib-only modules under offline coverage
- the docs flip in the same PR: realtime-pipeline.md explicitly IOUs this issue ("It is half-duplex — no barge-in... Real barge-in needs AEC and is tracked in #151") and records the voice-loop script as the client-side stitch this work moves server-side; openai-api.md endpoint table, gateway-fleet.md, and the lobes explain realtime catalog entry all carry the audio-in-only framing and must change together
  - instruction: update realtime-pipeline.md (drop the half-duplex IOU, document the conversation opt-in + new events), openai-api.md endpoint table, gateway-fleet.md, and the explain catalog _REALTIME entry in the same PR; run doc-test-alignment before opening it
  - honesty: after the PR, no doc or explain entry claims the session is audio-in-only/half-duplex except as history; realtime-pipeline.md, openai-api.md, gateway-fleet.md, and lobes explain realtime flip in the same PR
- the site needs its own CI lane: Node 20 is already installed in the lint job (for markdownlint) but nothing runs npm ci/build anywhere; sonar.sources=lobes would not scan a site dir; version-check and publish path filters would not gate it; hatch packages only lobes/ so the site never ships in the wheel — a site-build job (the org repo pattern: npm ci && npm run build failing CI on a broken build) is new work
  - instruction: add a site-build job to tests.yml (npm ci && npm run build in the site dir, Node 22 per the org engines pin); leave sonar.sources, version-check path filters, and hatch packaging untouched
  - honesty: a broken Astro build fails PR CI, and the built wheel is byte-identical with and without the site directory present
- the local Astro test site connects through a LOCAL server-side WebSocket proxy that injects the Authorization: Bearer credential before forwarding to the gateway — the browser never holds or sends the key; the concrete mechanism (Astro/Vite dev-server proxy with an upgrade-header hook vs a tiny standalone local proxy) is a plan-time choice
  - instruction: prefer the Astro/Vite dev-server proxy (server.proxy, ws:true, upgrade-header hook) so no extra process is needed; a tiny standalone local proxy is the fallback if Vite cannot inject upgrade headers
  - honesty: the API key never reaches the browser: not in served JS/HTML or any config response; the proxy injects Authorization server-side; with the proxy down the site cannot connect at all
- the audio-out DELIVERY model is pinned: synthesize() is full-read (its own docstring; the Chatterbox sidecar has no streaming route), so the bridge holds the complete reply PCM — audio-out is sent as sequential chunked WS frames, an interruption stops the UNDELIVERED remainder server-side, and the client stops LOCAL playback of already-delivered frames on the interruption event; both halves are required for barge-in to feel instant
  - honesty: an offline test interrupts mid-delivery and asserts the undelivered remainder is never sent and the truncation event follows in-order; the site stops local playback on the interruption event in the acceptance run
- the TTS concurrency gate is sized for conversation: _tts_semaphore is module-GLOBAL with default TTS_CONCURRENCY=1, shared by every session AND the batch /v1/audio/speech route — a voice reply serializes behind any unrelated TTS work, which is dead air; the spec must either raise the default or scope the gate per-lane, and say which
  - honesty: the chosen concurrency default is stated in env.audio.example with its rationale, and an offline test proves a voice turn does not queue behind the batch lane at that default
- the server keeps in-session conversation history: the voice loop proves a coherent conversation needs history + a system prompt (its client-side history list + SYSTEM_PROMPT), so the bridge holds per-session, in-memory history that dies with the session (consistent with the ephemeral non-goal), with an operator-set default system prompt via env and a per-session override in the connect config
  - honesty: history lives only on the Session object — no disk, no module state; teardown drops it; an offline test drives a two-turn conversation and asserts the second generate request carries the first exchange
- every response stage has a timeout with a named error event: generate and TTS forwards get bounded waits (precedents: app.py _STT_FORWARD_TIMEOUT=60, the voice loop PLAYBACK_TIMEOUT_S=60 whose comment records a wedged backend stranding the conversation, tts_client httpx 120/read-60) — on expiry the floor RETURNS TO THE USER with a named error, never a session stuck in a responding state
  - honesty: offline tests expire each stage against a hanging fake backend and assert the named error event arrives and the floor returns to listening within the configured bound
- the site dev flow survives a headless Spark: the browser runs on the operator laptop, and getUserMedia requires a secure context (HTTPS or localhost) — plain http://<box>:<port> has NO microphone; the documented flow is ssh -L port-forwarding (site + gateway both reached as localhost on the laptop), with mkcert HTTPS as the alternative if forwarding is unacceptable
  - honesty: the site README documents the ssh -L flow as the primary path, and the live acceptance run is itself performed through it (laptop browser, forwarded localhost)
- the site gates audio behind a user gesture and renders permission failure as a first-class state: AudioContext starts suspended until a gesture and mic access prompts — a start button arms mic + playback together, and NotAllowedError/NotFoundError render visually distinct from silence, disconnect, and the named server error codes
  - honesty: mic + playback arm only from the start control; a denied mic permission renders its own distinct state, verified manually during the acceptance run (permission prompts are not automatable offline)
- every in-repo client flips to the base64 event wire in the same PR series: scripts/realtime-smoke.py and scripts/realtime-voice-loop.py speak raw binary frames today and are the acceptance tooling — they migrate to input_audio_buffer.append/response.audio.delta (the f1e6ffa duplex rules — pong the pings, select-based reads, lock-guarded writes — carry over unchanged), and the #149 binary-input offline tests are UPDATED to the event format, not deleted
  - honesty: after the PR series, grep finds no in-repo client sending raw binary audio frames; the smoke script passes live against the deployed base64 wire, and the updated offline helper tests cover the append/delta framing arithmetic

## Honesty conditions

- a live Spark GB10 run through the production gateway: speak, hear the reply, interrupt it mid-playback, and the recorded event transcript shows the interruption — committed under docs/evidence/ before any doc says validated
- the tunnel refusal test for an armed STT_PEER_PROXY still asserts refusal after the change; no new WS egress to peers exists anywhere in the diff
- the batch /v1/audio/* handlers and catalog.py show a zero diff in the PR
- the gateway do_GET dispatcher gains no static/file-serving branch in the PR diff
- after the PR, lobes/gateway contains no query-param or Sec-WebSocket-Protocol token handling, and the auth tests still assert header-only Bearer
- no deploy workflow for the site exists under .github/workflows; the site README states local-only and documents the dev flow (npm run dev + local proxy + local gateway + key)
- each named audience has a concrete shipped surface: reachy an unchanged ears-only contract, the terminal client a still-working voice loop, the site developer a running site, the operator a live event stream that shows VAD boundaries and timings
- the before-state is cited from in-tree evidence, not memory: the f1e6ffa commit message, the voice-loop docstring (half-duplex, muted mic), and #149 spec non-goal c15
- the live acceptance run demonstrates every element of the after-state: a spoken reply on one connection, an interruption mid-playback, and the site rendering boundaries, transcripts, reply text, audio and named errors
- an offline test asserts the non-opt-in session emits the transcription-only event sequence 1:1 in the new wire format, and no code path accepts or emits raw binary audio frames after the change
- each acceptance criterion maps to a named offline test or to the evidence transcript, the mapping is explicit in the exported spec, and the A/B criterion is demonstrated by pointing the same client at the OpenAI realtime service for one audio exchange (or recording precisely why not in the transcript)

## Success signals

- the #151 acceptance criteria hold on the new wire: a spoken reply with no second HTTP call; barge-in visible in the event stream; ears-only consumers keep the transcription-only event sequence (OpenAI-shaped, base64); the site drives a full conversation rendering every event type including each named error code distinctly; the turn-taking machine is fully offline-tested; a live transcript lands under docs/evidence/ before any validated wording (#108); an A/B smoke of the same client against the OpenAI realtime service is possible for the audio path

## Scope / boundaries

- no cross-box WebSocket, still: the #129 proxy-lobes forwarder is POST-only and the tunnel refuses WS proxying even when STT_PEER_PROXY is armed (spec boundary c13, restated in _realtime.py docstring) — voice-to-voice adds traffic ON the session, not new session transport
- the batch routes /v1/audio/speech and /v1/audio/transcriptions stay byte-identical (the #149 boundary carries forward — reachy measured-working links must not regress), and Parakeet/Chatterbox/Silero stay hardcoded sidecars outside catalog.py (the established audio-overlay rule)
- the gateway does not become a static file host: do_GET is a fixed dispatcher (health/status/capabilities/models/realtime, else 404) — the site is served by Astro dev/preview (or Pages, per the deployment question), never by the gateway process
- query-parameter and WebSocket-subprotocol authentication are OUT of scope: public realtime clients are robots and native applications capable of setting the existing Authorization: Bearer header on the handshake — the public gateway remains header-authenticated
- no public site: the Astro site is local-only — testing, experiencing, and exemplifying the realtime surface against a local gateway; no Cloudflare Pages lane, no deploy workflow (the site-build CI job keeps it compiling, nothing publishes it)
- ears-only stays the DEFAULT MODE, on a new wire: a session that never sends response.create still gets exactly the transcription-only event sequence (created, boundaries, transcripts, named errors) and is never forced into conversation — but the wire format is now OpenAI-shaped base64 JSON events in BOTH directions (input_audio_buffer.append in, response.audio.delta out); the #149 raw-binary input contract is superseded, its removal coordinated with reachy-mini-cli

## Non-goals

- sessions stay ephemeral: no resume across bridge/gateway restarts, no session state on disk, an interrupted response is simply gone — the #149 lifecycle boundary (s20) is not reopened by voice-to-voice

## Assumptions

- browser-side AEC is free and changes the barge-in calculus: getUserMedia({audio:{echoCancellation:true}}) gives the site echo-cancelled mic input natively, and reachy-mini-cli cancels echo in firmware (#149) — the two known consumers cover AEC client-side, so server-side AEC can stay a declared-off AECMode passthrough rather than new DSP machinery
- no new Python dependencies: the [realtime] extra + Dockerfile.realtime already ship fastapi/uvicorn/httpx/numpy/scipy/silero-vad/torch, and the brain+mouth calls reuse in-container httpx + tts_client — voice-to-voice is wiring, not new wheels
- the #108 evidence rule governs every validated claim: a live acceptance transcript under docs/evidence/ must land before any doc claims voice-to-voice or the site validated — and #149 left four items explicitly UNVALIDATED (real microphone, vad_unavailable path, concurrent sessions, max-turn cap); the browser site IS the real-microphone test vehicle, so its acceptance run can retire that debt
- scripts/realtime-voice-loop.py stays the terminal-side reference client: scripts/ is repo-only (hatch packages only lobes/), the script calls itself a scratch tool, and its client-side three-endpoint stitch becomes obsolete-by-default once the loop moves server-side — it is not deleted, it is the pre-#151 fallback and the non-browser live test
- AEC ownership sits at the client edge, whoever that is: Reachy Mini cancels echo in firmware, the browser site gets it from getUserMedia echoCancellation, and a mic-speaker unit may do it in hardware — server-side AEC stays the declared-off AECMode passthrough with no DSP machinery
- the cancellation plumbing partially exists: tts_client.synthesize and _synthesize_single already thread a cancel_event asyncio.Event (checked before each request) — barge-in truncation arms an existing hook rather than inventing one
- the segmenter needs ZERO changes for barge-in: it is a floor-agnostic pure state machine that keeps segmenting whatever audio arrives — a SpeechStarted while the machine speaks IS the barge-in trigger, consumed by the new floor machine sitting above it; per-session isolation is already documented and tested

## Scope exploration

- `s1` — `lobes/realtime/app.py (_pump_session, realtime route)`: text/control frames are deliberately ignored today with a comment citing the #149 non-goals — response.create lands here; the route is a thin pragma-no-cover shell over stdlib-tested modules, and the established split (stdlib logic offline-tested, route shells live-tested) governs any new turn machinery
  - seeds: `c2`
- `s2` — `lobes/realtime/protocol.py + tts_client.py + lobes/gateway/_realtime.py`: Chatterbox emits 24 kHz PCM16 and the client wire format is 24 kHz PCM16 — TTS-out is a passthrough; synthesize() is in-container; the byte tunnel relays opaque bytes both ways concurrently, so server-to-client audio frames traverse it with zero gateway work
  - seeds: `c3`
- `s3` — `lobes/realtime/_settings.py + scripts/realtime-voice-loop.py`: Settings already carries the LLM backend block (openai_base_url/api_key/model) and the voice-loop script records the measured model choice in-tree: multimodal ~1 s to first reply, "a thinking model spends that on a trace nobody hears" — but a spark-lobe shape drops senses, so the default must tolerate an infeasible multimodal lane (gateway 404s role_infeasible, never silently falls back)
  - seeds: `c4`
- `s4` — `lobes/realtime/_session.py (SessionState, EventType, ErrorCode)`: the session engine is stdlib-only and fully offline-tested; its docstring scopes it to "audio-in, boundaries, transcription" per the #149 spec — voice-to-voice extends SessionState (floor holder), EventType (response/audio-out/interruption events), and ErrorCode (generate/TTS forward failures) in the module that owns the schema
  - seeds: `c5`
- `s5` — `lobes/realtime/_settings.py (barge_in_*) + scripts/realtime-voice-loop.py (muted-mic pattern) + commit f1e6ffa`: barge_in_window_ms=750 / barge_in_model ship unused ("the in-tree barge_in_/response-id machinery stays dormant" — #149 non-goal); the f1e6ffa voice loop is half-duplex BY DESIGN (mic muted during playback) because server AEC is absent — barge-in is precisely the capability that muting forecloses
  - seeds: `c6`
- `s6` — `lobes/gateway/server.py (inbound bearer gate + _handle_realtime)`: auth precedes the realtime tunnel by design ("a rejected handshake costs zero planning, zero upstream sockets"); the gate is header-only Bearer with hmac.compare_digest — no query-param or subprotocol path exists; session config already travels as query params on the connect URL, so a token query param would reach the bridge (and its logs) unless the gateway strips it — the mechanism choice is a real security decision
  - seeds: `c7`
- `s7` — `../org/site-astro (global.css tokens, HeroMesh/LobesDiagram/LobesTerminal, Layout.astro, astro.config.mjs)`: Astro 7, output static, zero integrations, zero UI frameworks; animations are CSS-only (compositor-friendly scale/opacity keyframes, per-instance --bd delays, global prefers-reduced-motion kill switch, WCAG AA both themes documented in global.css); terminals are build-time replay engines over typed capture data (CaptureLine kind cmd/cont/out) — the aesthetic and a11y bar to match; deploys to Cloudflare Pages via wrangler in org CI
  - seeds: `c8`
- `s8` — `lobes/realtime/protocol.py (AECMode) + _session.py (parse_session_config) + issue #151 note`: AECMode {none, aec} and default_aec_mode="none" are shipped and validated in config parsing but drive nothing; the issue REQUIRES server AEC stay optional/default-off for the firmware-AEC consumer; the browser gets AEC from getUserMedia constraints — client-side AEC covers both known consumers
  - seeds: `c9`
- `s9` — `issue #151 Notes + docs/specs/2026-07-21-realtime-ws-server-vad-149.md (Non-goals, c15)`: the #149 spec deliberately recorded no-response.create/no-LLM/no-TTS-out as non-goal c15 with the barge-in machinery "dormant" — #151 is the planned lift of exactly that non-goal, while its Notes pin the ears-only default for reachy-mini-cli
  - seeds: `c10`
- `s10` — `lobes/gateway/_realtime.py (docstring: "No cross-box WebSocket", spec c13)`: the tunnel docstring hard-codes the rationale: a half-served WS crossing a box boundary would break the loop-guard and attribution guarantees the POST proxy lane provides — lifting this is its own future issue, not #151
  - seeds: `c11`
- `s11` — `docs/evidence/2026-07-21-accept-realtime-spark.txt + README-realtime-acceptance.md + CLAUDE.md realtime section`: the #149 acceptance run used synthesized audio through the gateway tunnel at both rates and caught the tunnel leftover-direction bug — but a real microphone, the VAD-unavailable path, concurrent sessions, and the max-turn cap are recorded as still-unvalidated; the site closes the real-mic gap by construction
  - seeds: `c15`
- `s12` — `lobes/templates/fleet/docker-compose.audio.yml (realtime service env) + env.audio.example`: the bridge service already receives STT_URL/TTS_URL/OPENAI_* plus the VAD block and DEFAULT_AEC_MODE; BARGE_IN_WINDOW_MS/BARGE_IN_MODEL are read by _settings.py but absent from the compose environment — wired-but-dead on one side, read-but-never-passed on the other
  - seeds: `c4`, `c16`
- `s13` — `tests/test_realtime_*.py + test_gateway_realtime_ws.py + pyproject.toml coverage omit list`: 188 tests across ten realtime files, every one offline (stdlib or 127.0.0.1 loopback); the f1e6ffa duplex-client regressions live in test_realtime_smoke_helpers.py; the [realtime] extra (fastapi/torch) is never installed in CI, so route shells stay pragma-no-cover — the split is policy, not accident
  - seeds: `c17`
- `s14` — `docs/realtime-pipeline.md + docs/openai-api.md + docs/gateway-fleet.md + lobes/explain/catalog.py (_REALTIME)`: realtime-pipeline.md names #151 by number as the tracked barge-in/AEC follow-up — an in-tree IOU exactly like the one #149 redeemed; the explain catalog mirrors the pipeline doc including the "audio-in only" framing and the still-unvalidated list
  - seeds: `c18`
- `s15` — `.github/workflows/tests.yml + publish.yml + sonar-project.properties + pyproject.toml [tool.hatch.build.targets.wheel]`: CI is uv/Python-shaped end to end; the one Node use is a global markdownlint install; org CI has the reusable pattern (site-build job + Cloudflare Pages deploy via wrangler) if the site lives here and deploys at all
  - seeds: `c19`
- `s16` — `lobes/gateway/server.py (do_GET dispatcher — no static serving)`: grep confirms no static/text-html/SimpleHTTPRequestHandler path exists; hosting the site in-gateway would be new machinery with no precedent — Astro tooling serves the site
  - seeds: `c20`
- `s17` — `scripts/realtime-voice-loop.py + scripts/realtime-smoke.py (the shared duplex WS client) + pyproject packaging`: the duplex-safe WebSocket client lives in realtime-smoke.py and is reused by the voice loop; its hard-won rules (pong uvicorn pings within ~20s, select()-based read deadlines, lock-guarded writes) are the survival contract for ANY non-browser client of the richer session — the browser WebSocket gets ping/pong for free
  - seeds: `c21`
- `s18` — `challenge pass / process lens: frame state (.devague questions q3)`: q3 text asked barge-in cancellation semantics but was resolved with the AEC-ownership decision — a real undecided decision was wearing a resolved flag; re-raised as q5
- `s19` — `challenge pass / adjacent-systems + concurrency lens: lobes/realtime/tts_client.py`: synthesize threads a cancel_event (existing cancellation hook) BUT _tts_semaphore is module-global, default TTS_CONCURRENCY=1, shared by all sessions and the batch speech route — one hook to arm, one contention default to fix
  - seeds: `c32`, `c33`
- `s20` — `challenge pass / failure-mode lens: tts_client.py (full-read) + chatterbox_server.py (no streaming route)`: the sidecar synthesizes whole requests (raw PCM16 response, no streaming endpoint) — the bridge holds the complete reply before the first out-frame, so chunked send + server-side truncation of the undelivered remainder is the only workable interruption model; also grounds per-stage timeouts
  - seeds: `c31`, `c35`
- `s21` — `challenge pass / data-flow lens: scripts/realtime-voice-loop.py (history + SYSTEM_PROMPT client-side)`: the working conversation keeps history and the system prompt in the CLIENT; moving the loop server-side moves both — unstated in the pre-challenge spec, now an explicit requirement
  - seeds: `c34`
- `s22` — `challenge pass / overlooked-actors lens: the operator laptop browser vs the headless DGX Spark`: the browser is NOT on the box: getUserMedia demands a secure context, so <http://spark:port> has no mic at all — the ssh -L localhost flow (or mkcert) is a shipping requirement, not a nice-to-have; autoplay/permission gating is the same actor overlooked
  - seeds: `c36`, `c37`
- `s23` — `challenge pass / concurrency lens: lobes/realtime/_segmenter.py`: floor-agnostic pure state machine, per-session isolation documented and tested, keeps segmenting during playback — SpeechStarted during a response IS the barge-in trigger; no segmenter changes needed. Clean pass on the segmenter itself
  - seeds: `c38`
- `s24` — `challenge pass / security lens: gateway auth strip + local proxy key path`: clean pass: the caller credential is stripped before the bridge (gateway _DROP_FROM_HANDSHAKE), the local proxy holds the only key and h12/h17 already pin never-in-browser and header-only; residual exposure is the operator laptop itself, out of scope
- `s25` — `challenge pass / migration + reversibility lens: event schema evolution`: clean pass: the conversation surface is additive and opt-in per session — a #149-era client never opts in and h13 asserts byte-identity by unmodified tests; rollback is not sending response.create; no store, schema version, or on-disk state exists to migrate (ephemeral non-goal)
- `s26` — `challenge pass / observability + containment lens: session logging + event stream`: the event stream is the observability surface (h19) and get_session_logger stamps ids into every record; the gap found was containment on wedged backends — routed as the per-stage-timeout requirement, floor returns to user
  - seeds: `c35`
- `s27` — `challenge pass / adjacent-systems lens (reopened by q6): the wire-format decision vs reachy-mini-cli + in-repo clients`: strict base64 both ways supersedes the live-validated #149 binary input: the deployed reachy client and BOTH in-repo scripts (smoke, voice loop) speak binary today — the break is recorded as a coordinated decision (c40), the in-repo migration as a requirement (c42), and full OpenAI parity is parked (v5), scoped to the audio-path shapes
  - seeds: `c39`, `c40`, `c42`

## Decisions

- the wire break is coordinated, not hidden: reachy-mini-cli adapts to the base64 event format in its next commit; the window where an un-updated deployed reachy cannot stream is accepted by the operator — a deliberate, recorded break, not a regression

## Open / follow-up

- gateway realtime hardening — per-client session limits, credential isolation, timeouts, resource-budget enforcement for the internet-exposed service — ships as a sibling issue the exported spec cites by name; the #149 session-cap park folds into that issue
- full OpenAI Realtime API parity — session.update semantics, conversation.item.* schemas, the full response lifecycle, tool calls over the session, ephemeral tokens — stays a named follow-up: this work adopts the AUDIO-PATH event shapes only (append, audio.delta, response.create, transcription events); claiming more would over-advertise
