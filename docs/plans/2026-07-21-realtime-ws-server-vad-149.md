# Build Plan — realtime WS server_vad (#149)

slug: `realtime-ws-server-vad-149` · status: `exported` · from frame: `realtime-ws-server-vad-149`

> lobes serves /v1/realtime WebSocket sessions on the audio overlay: a client streams PCM over one connection, server-side Silero VAD (turn_detection: server_vad) emits speech-start/speech-stop boundary events, committed turns return as Parakeet transcription events on the same session, VAD failure is a named session error distinct from silence, and the stt role advertises the realtime capability — reachy-mini-cli deletes its client-side energy-threshold endpointing (issue #149, the PR2 app.py already promises)

## Tasks

### t1 — VAD segmenter: pure state machine in a new stdlib module lobes/realtime/_segmenter.py over 512-sample/32ms PCM chunks, Silero injected as a callable

- covers: c3, h3, c27, h19, c29, h21
- acceptance:
  - tests/test_realtime_segmenter.py passes offline with no torch and no [realtime] extra installed
  - a scripted fake-VAD silence-speech-silence sequence yields speech_started at the padded onset and speech_stopped after vad_silence_ms
  - a never-silent stream hits the documented max-turn cap and receives the documented force-commit-or-error outcome
  - two interleaved segmenter instances with different scripted patterns each produce exactly their own boundary events

### t2 — Session engine: stdlib module lobes/realtime/_session.py — event schema, session config parsing (rate/AEC/turn_detection), teardown bookkeeping, session-id logging

- covers: c5, h5, c14, h12, c26, h18, c28, h20, c31, h23
- acceptance:
  - config parse accepts PCM16 mono at 24000 (default) and 16000, rejecting others with a named error
  - AEC defaults to none and stays off unless the session explicitly enables it
  - VAD-unavailable yields the documented named error event and no boundary events; a silent healthy session yields neither — distinguishable by event type alone
  - teardown from every state (idle, mid-speech, mid-transcription) releases all session bookkeeping, asserted by test
  - every event and log record carries the session id; no key material appears in any log line; no code path persists session state

### t3 — Gateway WS tunnel: 101-upgrade + bidirectional socket relay in gateway/server.py, /v1/realtime routing branch, bearer gate on handshake, role_infeasible 404 for declared-off stt

- covers: c8, h8, c13, h11, c12, h10, c26, h18
- acceptance:
  - with GATEWAY_API_KEY set, an upgrade request with a missing or wrong bearer is rejected before any tunnel or session allocation (offline fake-socket test)
  - the 101 handshake is relayed and bytes pump both directions until either side closes; both pump threads unwind on either close (offline fake-socket test)
  - with STT_FEASIBLE=false the handshake receives the role_infeasible 404, carrying hosted_by when a peer origin is declared
  - the session socket is exempt from the HTTP read timeout; existing gateway audio routing and peer tests pass unchanged

### t4 — Capabilities: stt role advertises realtime via ROLE_RESPONSIBILITIES in lobes/roles.py; CLI + gateway surfaces + contract doc follow

- covers: c6, h6
- acceptance:
  - on an audio-enabled registry, lobes capabilities and GET /capabilities both show the realtime capability under stt (asserted in test_cli_capabilities.py)
  - a text-only fleet shows no realtime claim; test_colleague_contract.py and docs/colleague-stack.md updated and green

### t5 — Templates + env passthrough: VAD_/AEC_/TURN_ keys (plus the max-turn cap key) into docker-compose.audio.yml environment block and env.audio.example; doctor heal covers them

- covers: c7, h7
- acceptance:
  - a fresh scaffold's realtime service receives VAD_THRESHOLD, VAD_SILENCE_MS, VAD_PREFIX_PADDING_MS, DEFAULT_TURN_DETECTION, DEFAULT_AEC_MODE and the max-turn cap; env.audio.example documents each
  - doctor --fix on a pre-PR2 scaffold heals the missing keys append-only, never rewriting an existing .env line (test)

### t6 — The /v1/realtime route: thin WS shell in app.py wiring real Silero + scipy resample (24k to 16k) into the stdlib session engine; committed turns forward to Parakeet

- depends on: t1, t2
- covers: c2, h2, c4, h4
- acceptance:
  - the route registers on the same FastAPI app as the batch routes, stays a thin pragma-no-cover shell, and imports the stdlib session/segmenter modules for all logic
  - a committed turn forwards to settings.stt_url and emits a transcription event on the SAME connection; an STT failure emits the named error event, never a silent drop
  - session config declares input rate: 24000 default, 16000 accepted, PCM16 mono little-endian stated in the route docs

### t7 — Docs flip: realtime-pipeline.md boundary section, openai-api.md endpoint table, explain catalog — wire format, cap, restart contract, baseline citation

- depends on: t3, t6
- covers: c9, h9, c19, h14, c21, h16
- acceptance:
  - no doc records the WS as future work; openai-api.md lists /v1/realtime with PCM16 mono LE, 24000 default / 16000 accepted stated explicitly; lobes explain realtime names the session surface
  - the ephemeral-session restart contract and the max-turn cap are documented; the #149 baseline probe and the in-tree IOUs are cited

### t8 — Live smoke + acceptance evidence procedure: a plain-websocket-client script drives connect, stream, boundaries, transcript over ONE connection against a deployed overlay

- depends on: t3, t6
- covers: c1, h1, c18, h13, c20, h15, c22, h17
- acceptance:
  - the script uses plain websocket-client-level code (no torch, no OpenAI SDK) and prints PASS/FAIL per step like scripts/audio-smoke.py
  - the script documents the acceptance-evidence procedure: transcript lands under docs/evidence/ before any validated wording (#108)

### t9 — Release chores: minor version bump + CHANGELOG entry; full offline suite green

- depends on: t7, t8
- acceptance:
  - version bumped minor via the bump script with a CHANGELOG entry; uv run pytest -n auto green with no [realtime] extra installed

## Risks

- [unknown_nonblocking] concurrent-session cap undecided (frame park v2): ThreadingHTTPServer is thread-per-session and the expected consumer population is about one robot — implement unbounded-but-documented unless review demands a cap (task t3)
- [unknown_nonblocking] silero-vad floats from >=5.1 to 6.x at image build time; if the VADIterator API drifts across the major, the image build pins tighter — never vendors the model (task t6)
- [unknown_nonblocking] live acceptance is post-merge sequencing: the deployed facade installs lobes-cli[realtime]==MODEL_GEAR_VERSION from PyPI (CDN-lag gotcha), so the docs/evidence/ transcript lands after release + rebuild on an audio-hosting box (task t8)
