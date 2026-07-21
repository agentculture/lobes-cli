# realtime WS server_vad (#149)

> lobes serves /v1/realtime WebSocket sessions on the audio overlay: a client streams PCM over one connection, server-side Silero VAD (turn_detection: server_vad) emits speech-start/speech-stop boundary events, committed turns return as Parakeet transcription events on the same session, VAD failure is a named session error distinct from silence, and the stt role advertises the realtime capability — reachy-mini-cli deletes its client-side energy-threshold endpointing (issue #149, the PR2 app.py already promises)
> instruction: implement as ONE PR (PR2) on lobes/realtime + gateway + templates + docs; verify with the offline suite plus scripts-level live WS smoke against a deployed audio overlay

## Audience

- reachy-mini-cli (the robot hearing path measured in #149) first; any OpenAI-Realtime-style consumer of the lobes audio overlay second

## Before → After

- Before: the realtime container serves four batch routes and no WebSocket; consumers endpoint client-side — reachy's energy threshold shatters sentences at inter-word dips (measured: a five-word question arrives as 'Ready, she')
- After: a consumer opens ONE WebSocket to /v1/realtime, streams PCM, and receives server-side speech-start/speech-stop boundaries plus Parakeet transcripts for committed turns — client-side endpointing deleted

## Why it matters

- a batch-only endpoint named realtime misleads every next consumer; server-side VAD is lobes' own recorded design (server_vad in protocol.py, the app.py PR2 IOU), and consuming it over WS keeps torch out of robot CLIs and their CI

## Requirements

- lobes/realtime/app.py gains the /v1/realtime WebSocket route its own docstring (line 11) promises; per the established split the route stays a thin shell and the session/VAD logic lands in a stdlib-only module mirroring audio_facade.py, importable without the [realtime] extra
  - honesty: the offline unit suite imports and tests the session/VAD logic module with the [realtime] extra ABSENT; app.py stays a thin pragma-no-cover shell like the existing routes
- the server_vad segmenter is a pure state machine over 512-sample/32ms PCM chunks (protocol.py's declared framing) with the Silero model injected as a callable, so the offline suite unit-tests segmentation with a fake VAD — no torch, no GPU (issue criterion 7)
  - honesty: tests drive the segmenter with a scripted fake VAD through silence-speech-silence and assert boundary events, prefix padding, and silence-ms commit timing — no torch import anywhere in the test path
- committed turns are transcribed over the same session by forwarding to settings.stt_url (Parakeet), reusing the batch transcriptions forward pattern — one connection replaces WS-plus-separate-batch-POST (criterion 4)
  - honesty: a committed turn yields a transcription event on the SAME connection carrying Parakeet's text; an STT backend failure surfaces as a named error event on the session, never a silently dropped turn
- a session with server_vad armed emits a NAMED error event when the Silero model fails to load or is unavailable, so a consumer can tell no-speech from VAD-down (criterion 5); the /v1/health/ready aggregate stays a probe of the two HTTP backends
  - honesty: with Silero load forced to fail, a server_vad session receives a documented, named error event and no boundary events; a healthy-VAD silent session emits neither — the two are distinguishable by event type alone
- the stt role advertises the realtime/VAD capability on both lobes capabilities and gateway GET /capabilities (criterion 6) — additive via ROLE_RESPONSIBILITIES beats a new RoleInfo schema field; either way test_cli_capabilities.py / test_colleague_contract.py and docs/colleague-stack.md move with it
  - honesty: on an audio-enabled deployment both lobes capabilities and GET /capabilities show the realtime capability under stt; on a text-only fleet neither claims it
- the VAD knobs _settings.py already reads (VAD_THRESHOLD, VAD_SILENCE_MS, VAD_PREFIX_PADDING_MS, DEFAULT_TURN_DETECTION, DEFAULT_AEC_MODE) get wired through docker-compose.audio.yml's environment block and env.audio.example — today NEITHER carries them, so container settings silently pin to defaults; doctor --fix missing-only heal covers existing deployments
  - honesty: a fresh scaffold's compose passes every VAD_/AEC_/TURN_ key and env.audio.example documents each; doctor --fix on a pre-PR2 deployment heals the missing keys append-only, never rewriting an existing .env line
- the WS handshake honors the same opt-in GATEWAY_API_KEY bearer gate as every /v1/* data-plane route, whichever reachability path is chosen
  - honesty: with GATEWAY_API_KEY set an unauthenticated /v1/realtime handshake is rejected before session allocation; with it unset behavior is byte-identical to the keyless contract
- docs flip together: realtime-pipeline.md's Boundary section (which today records the WS as planned-for-a-later-release), openai-api.md's endpoint table, and the lobes explain realtime entry all gain /v1/realtime with the explicit dtype/channel/framing statement criterion 2 demands
  - honesty: after PR2 no doc records the WS as future work, openai-api.md lists /v1/realtime with dtype/channels/rate stated explicitly, and lobes explain realtime names the session surface — all flipped in the same PR
- session teardown is leak-free: a client disconnect in ANY state (idle, mid-speech, mid-transcription) closes the upstream leg, unwinds the gateway tunnel threads, and frees session buffers; close propagates in both directions — a dropped robot never strands a bridge session or a gateway thread
  - honesty: an offline test drives a session through connect-stream-disconnect at each state and asserts no leaked task/thread/buffer bookkeeping; the gateway tunnel unwinds both pump directions on either side's close
- per-turn audio buffering is bounded: a documented max-turn length force-commits (or errors) with a named event when exceeded — a never-silent stream cannot grow bridge memory without limit
  - honesty: the max-turn cap is a documented, env-tunable value; an offline test streams past it with a never-silent fake VAD and receives the documented commit-or-error event
- session lifecycle is observable: session open/close, VAD arm/fail, and STT forward failures are logged with session ids in the bridge, and tunnel open/close in the gateway — lobes logs tells a session's story
  - honesty: grepping bridge logs for one session id reconstructs its lifecycle: open, VAD state, commits, errors, close
- VAD state is per-session: concurrent sessions get isolated segmentation state (shared model, per-session iterator state or equivalent) — interleaved sessions never corrupt each other's boundaries
  - honesty: an offline test interleaves two sessions with different scripted fake-VAD patterns and asserts each gets exactly its own boundary events

## Honesty conditions

- every announced behavior is demonstrated: WS session + boundary events + in-session transcripts by committed offline tests, the deployed-robot claim by a live acceptance transcript under docs/evidence/ before any validated wording (#108 rule)
- the existing batch-route tests (gateway audio routing/peer, facade parsers) pass unchanged and no PR2 hunk touches the speech/transcriptions request or response shapes
- with the stt lane declared off, the /v1/realtime handshake gets the role_infeasible 404 naming hosted_by; no code path forwards a WebSocket across boxes
- a session that omits AEC config runs with aec none; enabling AEC is an explicit per-session opt-in and its absence changes nothing
- reachy-mini-cli can consume the session with plain websocket-client-level code — no torch, no OpenAI SDK requirement on the robot
- the spec cites the #149 live probe (four batch routes, no WS) and the measured fragment transcripts as the baseline, not a generic problem statement
- one connection suffices end-to-end — audio in, boundary events, transcript events — demonstrated by the live smoke script against a deployed overlay
- the spec cites the in-tree IOUs (app.py PR2 docstring, realtime-pipeline.md boundary section) rather than generic motivation
- the offline suite runs green with no GPU and no [realtime] extra installed, and any validated wording lands only with the docs/evidence/ transcript (#108 rule)
- the deployed container starts a server_vad session with egress blocked; if the >=5.1 float ever breaks the VADIterator API the image build pins tighter, never vendors the model
- restart behavior is documented; no code path persists or restores session state, and reconnect-and-restart is stated as the client contract

## Success signals

- offline: the suite exercises VAD segmentation and session events with a fake VAD, no GPU; live: reachy's live-mic five-word question transcribes whole on the deployed fleet, recorded as an acceptance transcript under docs/evidence/ (#108)

## Scope / boundaries

- the batch routes /v1/audio/speech and /v1/audio/transcriptions stay byte-identical — reachy's measured-working links (gateway TTS, WAV to Parakeet) must not regress
- no WebSocket peer-proxying in PR2: the #129 stt-lane proxy-lobes forwarder is POST-only; a declared-off stt lane 404s the /v1/realtime handshake role_infeasible with referral annotation (hosted_by) only
- server-side AEC stays optional and default-off in the session handshake — Reachy Mini's mic array cancels echo in firmware (issue note); AECMode plus default_aec_mode=none already encode this
- sessions are ephemeral: no resume across bridge/gateway restarts and no session state on disk — a dropped connection means the client reconnects and starts fresh

## Non-goals

- no full OpenAI Realtime conversation surface: no response.create, no LLM turns, no TTS-out over the session — PR2 scope is audio-in, server_vad boundaries, transcription events (the issue's criteria stop there); the in-tree barge_in_/response-id machinery stays dormant
- no reachy-mini-cli changes from here — their endpointing root-cause investigation (reachy-mini-cli#108) is theirs, and the issue states this ask does not depend on its outcome
- silero-vad is not a switchable gear: catalog.py untouched; Parakeet and Chatterbox stay hardcoded sidecars (the established audio-overlay rule)

## Assumptions

- no new dependencies: the [realtime] extra and Dockerfile.realtime already ship fastapi, uvicorn, httpx, numpy, scipy, silero-vad, torch — the deployed image contains everything PR2 needs
- PR1 already landed PR2's config surface: _settings.py's VAD/turn-detection block (default_turn_detection=server_vad, default_aec_mode=none — AEC off by default, matching the issue's firmware-AEC note) is consumed by PR2, not invented
- PROBED 2026-07-21: the silero-vad wheel bundles its model weights (silero_vad/data/silero_vad.jit + onnx variants, inspected in silero_vad-6.2.1-py3-none-any.whl) — VAD arms with no network egress at boot or session start

## Scope exploration

- `s1` — `lobes/realtime/app.py`: docstring line 11 already promises PR2 adds the /v1/realtime WebSocket route to this same app; routes are thin pragma-no-cover shells over stdlib-tested helpers, and the batch transcriptions handler is the existing Parakeet forward to reuse for committed turns
  - seeds: `c2`, `c4`
- `s2` — `lobes/realtime/protocol.py + tests/test_realtime_protocol.py`: TurnDetectionType.SERVER_VAD, AECMode, and the Silero framing contract (16 kHz, 512 samples, 32 ms) are shipped and unit-tested — but CLIENT_SAMPLE_RATE=24000 is also pinned and test-asserted as the OpenAI Realtime wire format, so the issue's 16 kHz streaming ask conflicts with the in-tree wire contract
  - seeds: `c3`
- `s3` — `lobes/realtime/_settings.py`: the full VAD/turn-detection settings block already exists (vad_threshold 0.5, vad_silence_ms 600, vad_prefix_padding_ms 300, default_turn_detection server_vad, default_aec_mode none, barge_in_*) with a docstring saying it is for the realtime WS pipeline — PR1 landed PR2's config surface
  - seeds: `c11`, `c14`
- `s4` — `lobes/templates/fleet/env.audio.example + docker-compose.audio.yml environment block`: grep finds NO VAD_/AEC_/TURN_ keys in either file — the settings module reads knobs the deployment never passes, so they silently pin to defaults inside the container; the passthrough plus doctor-heal coverage is real PR2 work
  - seeds: `c7`
- `s5` — `pyproject.toml [realtime] extra + lobes/templates/fleet/Dockerfile.realtime`: fastapi, uvicorn, httpx, numpy, scipy, python-multipart, silero-vad, torch are all in the extra the image installs (lobes-cli[realtime]==MODEL_GEAR_VERSION from PyPI) — no dependency work, but live validation requires a published release first
  - seeds: `c10`
- `s6` — `lobes/gateway/server.py`: stdlib ThreadingHTTPServer reverse proxy: _HOP_BY_HOP strips Upgrade, do_GET serves only health/status/capabilities/models, relay is buffered-or-SSE-rechunked — NO WebSocket passthrough exists, so criterion 1 (reachable through the gateway) is new tunnel machinery or a published-port decision; the opt-in GATEWAY_API_KEY bearer gate covers every /v1/* data-plane route and must cover the WS handshake too
  - seeds: `c8`
- `s7` — `lobes/gateway/_routing.py + _config.py`: _config.py:328 already reserves audio_url for /v1/audio/* plus /v1/realtime in PR2; but is_audio_path matches only /v1/audio/* and the per-role map (#129) is POST-path-keyed, so /v1/realtime needs its own explicit routing branch; the stt-lane peer-proxy forwarder is POST-only — WS proxying would be new, unscoped machinery
  - seeds: `c12`, `c13`
- `s8` — `lobes/roles.py`: stt role: ROLE_RESPONSIBILITIES ('transcribe', 'audio_input_to_text') is an additive tuple, while _audio_role builds a fixed RoleInfo schema (role/model/runtime/endpoint/path/context/quant/mtp/tools/...) — advertising realtime via a responsibility string is contract-compatible; a new RoleInfo field is a schema bump rippling into CLI, gateway, tests, and docs/colleague-stack.md
  - seeds: `c6`
- `s9` — `docs/realtime-pipeline.md + docs/openai-api.md + lobes/explain/catalog.py`: realtime-pipeline.md's Boundary section explicitly records the /v1/realtime WebSocket protocol as planned for a later release — PR2 redeems an in-tree IOU; openai-api.md's endpoint table and the explain catalog name the audio surface and must flip in the same PR
  - seeds: `c9`, `c15`
- `s10` — `pyproject.toml coverage omit list`: app.py/__main__.py/tts_client.py/chatterbox_server.py are excluded because the offline CI env installs no [realtime] extra — the established policy is stdlib-only logic modules fully tested, route shells live-tested; criterion 7's offline WS smoke test either respects that split or changes the dev-env dependency policy
  - seeds: `c3`
- `s11` — `lobes/catalog.py (rule, via CLAUDE.md + realtime-pipeline.md)`: Parakeet and Chatterbox are hardcoded sidecars, deliberately NOT in the switchable catalog — Silero VAD follows the same rule; no catalog entry
  - seeds: `c17`
- `s12` — `eidetic recall: spark-lobe go-live + audio-overlay repair + stt/tts referral (#129)`: the Spark box dropped the audio overlay on 2026-07-14 but it was repaired/live by 2026-07-17 and the issue probed it live on 2026-07-21; stt/tts joined the referral/proxy contract in #129 — cross-box audio is declared/unvalidated, reinforcing that WS peer-proxying stays out
  - seeds: `c13`, `c16`
- `s13` — `challenge pass / adjacent-systems lens: gateway ThreadingHTTPServer threading + timeouts`: a WS tunnel holds its handler thread for the session lifetime and must exempt the session socket from the request/read timeouts the relay applies to HTTP; thread-per-session is why the session-cap question exists — seeded the teardown requirement and the cap park
  - seeds: `c26`
- `s14` — `challenge pass / concurrency lens: silero VAD iterator state x concurrent sessions`: the Silero model is shareable but segmentation state is per-stream; nothing in the spec said so until now — seeded the per-session-state requirement
  - seeds: `c29`
- `s15` — `challenge pass / failure-mode lens: never-silent stream, unbounded buffer`: with VAD armed but silence never detected, per-turn PCM accumulates without bound in the bridge — seeded the max-turn cap requirement
  - seeds: `c27`
- `s16` — `challenge pass / observability lens: bridge + gateway logging`: the batch routes log per-request via uvicorn; nothing specified session-scoped logging for long-lived WS state — seeded the session-id logging requirement
  - seeds: `c28`
- `s17` — `challenge pass / hidden-dependency lens: silero-vad model packaging (PROBED)`: pip download silero-vad + wheel inspection in scratch: model weights ship inside the wheel (silero_vad/data/*.jit/.onnx) — no torch.hub or network fetch at load; NOTE pyproject floats silero-vad>=5.1 while PyPI is at 6.2.1, an unpinned major the image build absorbs
  - seeds: `c30`
- `s18` — `challenge pass / security lens: WS handshake auth + post-upgrade relay`: CLEAN: the bearer gate covers the handshake (c8); post-upgrade bytes are opaque payload the gateway relays without interpretation; the facade stays authless inside the compose network — the existing trust model, unchanged; residual: key material never appears in session logs (folded into the logging requirement)
  - seeds: `c28`
- `s19` — `challenge pass / reversibility + migration lens: additive surface`: CLEAN: no store or schema migration; rollback is the standing ops path (re-pin MODEL_GEAR_VERSION + rebuild); the batch routes are untouched (c12) so reverting PR2 restores the exact pre-PR2 surface
- `s20` — `challenge pass / lifecycle lens: restart behavior`: bridge/gateway restarts drop live sessions; nothing resumes them — made explicit as the ephemeral-sessions boundary instead of staying unstated
  - seeds: `c31`

## Decisions

- reachability (q1): /v1/realtime is served THROUGH the gateway — server.py gains WebSocket passthrough (101 upgrade + bidirectional socket relay); the bearer gate covers the handshake; the facade port stays unpublished
- wire format (q2): the session accepts PCM16 mono little-endian at a session-declared input rate — 24000 Hz default per protocol.py, 16000 Hz accepted; the server owns all resampling (scipy is already in the [realtime] extra)
- offline test depth (q3): keep the established split — stdlib-only session/VAD logic fully unit-tested offline with a fake VAD, route shells pragma-no-cover, live WS smoke via scripts/ — no fastapi in the dev env
