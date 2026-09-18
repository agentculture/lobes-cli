# Build Plan — hebrew-realtime

slug: `hebrew-realtime` · status: `exported` · from frame: `hebrew-realtime`

> lobes ships a Hebrew realtime duplex voice session on the DGX Spark: speak Hebrew into GET /v1/realtime, an ivrit.ai Whisper model transcribes it, associate answers in Hebrew, phonikud vocalizes the reply before a Hebrew-capable TTS speaks it back — and when associate decides to call a tool, the call is handed to the connected client over the same socket and the client's result is spoken back, so an agent the client owns can operate this machine by voice

## Tasks

### t1 — Live proof 1/3 — back up ~/.lobes, stop ComfyUI, record the freed budget

- instruction: Use the lobes-deploy skill. Operator-approved drop (frame c5). Do not re-scaffold the Spark: its compose is hand-kept (memory: spark-compose-lacks-fingerprint-passthroughs). Evidence file is owned by this task only.
- covers: c7, h6
- acceptance:
  - A timestamped copy of ~/.lobes exists before any edit, and its path is written in docs/evidence/2026-09-hebrew-realtime-spike-spark.txt
  - ComfyUI is stopped via the lobes-deploy skill's lobes-compose.sh against ~/.lobes (`COMPOSE_PROFILES` no longer lists innereye); the evidence file shows free -g before and after and GET /capabilities reporting innereye feasible:false
  - The evidence file states the restore command and no docker compose command was run inside lobes/templates/

### t2 — Live proof 2/3 — Hebrew STT spike: ivrit-ai/whisper-large-v3-turbo under transformers+PyTorch vs vLLM Whisper on the GB10

- instruction: Scratch containers only, each with --memory capped. CTranslate2/faster-whisper is out (no aarch64 CUDA wheel, speaches#620). Force language=he. Record on card 1 (arecord -D plughw:1,0 -f `S16_LE` -r 16000 -c 1). Resolves plan risks r1 and parked v3.
- depends on: t1
- acceptance:
  - docs/evidence/2026-09-hebrew-stt-spike-spark.txt records, for each runtime that boots: image, load time, resident memory, and wall latency for 3 s / 8 s / 20 s Hebrew clips recorded on the reSpeaker XVF3800
  - Each runtime is scored on at least 10 Hebrew utterances against a typed reference with a negative control (a different utterance must NOT match) and a silence clip (hallucinated text is reported verbatim, not hidden)
  - The file ends with one chosen runtime and the reason; a runtime that failed to boot is recorded with its error, not dropped

### t3 — Live proof 3/3 — Hebrew TTS A/B: Chatterbox Multilingual (he) with and without phonikud niqqud vs phonikud-tts, plus phonikud CPU latency

- instruction: Scratch containers, --memory capped, `HF_HUB_DISABLE_XET`=1. Play WAVs to the pipewire sink of the XVF3800 so 24k->16k resampling is handled. Resolves plan risk r2 and feeds the deferred q4/v6 licence decision — do not decide q4 here.
- depends on: t1
- covers: h29
- acceptance:
  - docs/evidence/2026-09-hebrew-tts-ab-spark.txt has one row per engine arm (chatterbox-plain, chatterbox-niqqud, phonikud-tts) for: a clean Hebrew sentence, a mixed Hebrew+Latin-path+digits sentence, a question; with synth wall time, audio duration, sample rate and the WAV path
  - phonikud `add_diacritics` + phonemize latency on this box's CPU is measured per sentence over at least 20 sentences (median and max)
  - The operator (a Hebrew speaker) has listened on the reSpeaker speaker and their per-arm verdict is recorded verbatim; the agent does not grade Hebrew audio quality itself (lapse l2)
  - Licences of every weight file pulled are listed with their source URL, including phonikud's model weights (parked v5)

### t4 — Pin the wire contract: session.update, tool events, language and stage-timing fields in `_session.py` and `_wire.py`

- instruction: This task is the CONTRACT every later task builds against (memory: workforce-file-disjoint-hides-contract-conflicts) — land it alone, first, and export the dataclass/field names in the PR description. Owns lobes/realtime/`_session.py`, `_wire.py` and their two test files. No behaviour yet: parsing, shapes, serialisation only.
- covers: c32, h31, c24, h17
- acceptance:
  - `_session.py` gains SessionConfig.tools/`tool_choice`/language, EventType members session.updated, response.`function_call_arguments`.done and a conversation.item.create(`function_call_output`) inbound shape, all named exactly as OpenAI Realtime names them; response.done gains an optional timings mapping
  - A pure function translates OpenAI Realtime's flat tool shape to chat-completions' nested shape and is covered for a stock OpenAI payload; session.updated echoes only fields that took effect and a malformed session.update yields a named error without closing the session
  - The audio codec half of `_wire.py`, `_segmenter.py` and `_pcm.py` are byte-identical to main, and every pre-existing test in tests/`test_realtime_wire.py` and tests/`test_realtime_session.py` passes unmodified

### t5 — Generate payload and reply parsing for tools in `_turn.py`

- instruction: Owns lobes/realtime/`_turn.py` and tests/`test_realtime_turn.py` only. Keep `enable_thinking`:false — measured honored by associate with tools on 2026-09-18. Remove or align the dead English `DEFAULT_SYSTEM_PROMPT` copy here (the session's copy wins).
- depends on: t4
- covers: c9
- acceptance:
  - `build_turn_payload` includes nested tools and `tool_choice` only when the session declared tools; with none declared the payload is byte-identical to today's (existing tests unmodified)
  - `parse_turn_response` returns a distinct tool-call result (call id, name, arguments verbatim) when message.`tool_calls` is present, and never treats it as text to synthesize; history helpers serialise assistant `tool_calls` and role:tool entries

### t6 — Tool-wait state in the floor state machine

- instruction: Owns lobes/realtime/`_floor.py` and tests/`test_realtime_floor.py` only. Add Stage.TOOL, a FailureReason, and entries in `MACHINE_HELD_STATES` / `_STAGE_OF_STATE` / `_TIMEOUT_REASON`. Default deadline value is plan risk r5 — make it a constructor argument, do not hardcode policy.
- depends on: t4
- covers: c10, h21
- acceptance:
  - A tool-call reply moves RESPONDING to a new machine-held tool-wait state that arms no TTS deadline; a tool result moves it back to RESPONDING; speech onset past the barge-in window interrupts it and emits the interrupted event with stage=tool
  - The tool wait has its own configurable deadline (`TOOL_WAIT_TIMEOUT_MS`) that expires into a named failure reason and returns the floor to LISTENING; tests drive the clock, no sleeps

### t7 — Conversation bridge: tool dispatch, call-id bookkeeping, event order, synthetic cancel

- instruction: Owns lobes/realtime/`_conversation.py` and tests/`test_realtime_conversation.py` only. `on_control_event` is the single dispatch point (currently drops everything but response.create). PROBED 2026-09-18: associate repeats an orphan tool result as fact, so nothing downstream will catch a bookkeeping miss.
- depends on: t5, t6
- covers: h20, c28, h28, c31, h30, c4, h4
- acceptance:
  - A session that never declares tools emits exactly the pre-change event sequence — the existing tests/`test_realtime_conversation.py` cases pass unmodified
  - For a tool turn the transcription.completed event is emitted before response.`function_call_arguments`.done, asserted by an ordering test
  - `function_call_output` is accepted only for the single outstanding `call_id`; orphan, late (after timeout or barge-in) and duplicate outputs each yield a named error with history unchanged; an accepted output waits for response.create before the next generate
  - An interrupted or expired tool wait closes the assistant `tool_calls` entry with a synthetic cancelled tool output before the next generate request

### t8 — Settings and defaults: session language, Hebrew default prompt, tool-wait deadline knob

- instruction: Owns lobes/realtime/`_settings.py`, tests/`test_realtime_settings.py`, and CREATES lobes/templates/fleet/docker-compose.audio-he.yml + env.audio-he.example with the realtime service env only (t10/t11 add their services to that overlay later, in dependency order). The Hebrew deployment sets `OPENAI_MODEL`=associate in env.audio-he.example.
- depends on: t4
- covers: c30, c3, h3
- acceptance:
  - `_settings.py` reads `REALTIME_LANGUAGE` (default en), `TOOL_WAIT_TIMEOUT_MS` and keeps `OPENAI_MODEL`; a Hebrew `DEFAULT_SYSTEM_PROMPT` variant is selected when language is he and tells the model to answer in short spoken Hebrew and to describe rather than recite paths, identifiers and long numbers
  - Every new settings field is present in the realtime service env of the Hebrew overlay file and its env example, satisfying tests/`test_realtime_audio_env_coverage.py`; docker-compose.audio.yml and env.audio.example are byte-identical to main
  - A test shows a generate failure for model=associate (404 `role_infeasible` with `hosted_by`, 429, 503 `role_unverified`) surfaces as `generate_failed` carrying `hosted_by` and status, never as text from another model

### t9 — Vocalization hook: phonikud in the TTS text path

- instruction: Owns lobes/realtime/`_vocalize.py` (new), lobes/realtime/`tts_client.py` and new tests/`test_realtime_vocalize.py` + tests/`test_realtime_tts_text.py`. `tts_client` imports httpx at top and is coverage-omitted — move the pure text helpers into `_vocalize.py` or a sibling so they are testable offline. phonikud itself is imported lazily inside the realtime image only.
- covers: c12, h23, h5
- acceptance:
  - A new stdlib-importable module lobes/realtime/`_vocalize.py` takes an injected diacritizer callable; on its exception or timeout it returns the input text and reports a warning naming the cause; Latin and digit spans come back unchanged; fully unit-tested offline
  - `tts_client`.synthesize calls the hook after `_clean_for_tts` and before chunking when language is he, and chunk sizing counts base characters, not niqqud combining marks; `_clean_for_tts`/`_split_for_tts`/`trailing_pause_ms` gain their first tests, including gershayim and a mixed-script sentence
  - When a debug flag is set the vocalized text is observable (log line or debug event) so an acceptance run can show niqqud really reached TTS

### t10 — Hebrew STT sidecar: Whisper server honoring the `listen_server` contract

- instruction: Owns the two new template files plus the stt service block of docker-compose.audio-he.yml. Model id comes from `STT_MODEL` (default ivrit-ai/whisper-large-v3-turbo). Reuse lobes/templates/fleet/`_readiness.py`.
- depends on: t2, t8
- covers: c13, h24, c6
- acceptance:
  - lobes/templates/fleet/`listen_server_whisper.py` serves POST /v1/audio/transcriptions with the same response JSON shape as `listen_server.py` and honors the language form field; GET /v1/health/ready reports `model_loaded`, `cuda_ok` and the loaded model id and is non-200 until a transcription can succeed
  - Dockerfile.whisper-stt builds on aarch64 with the runtime t2 chose, installs with uv pip install --system, and the stt-he service is added to docker-compose.audio-he.yml under the same service name and port the bridge already targets
  - An offline test imports the server's pure request/response helpers and asserts shape parity with `listen_server.py`

### t11 — Hebrew TTS sidecar: the engine t3 selected behind the /v1/audio/synthesize contract

- instruction: If t3 picks Chatterbox Multilingual, add lobes/realtime/`chatterbox_multilingual_server.py` + Dockerfile.chatterbox-ml rather than editing `chatterbox_server.py`. If the operator's verdict favours phonikud-tts, STOP and raise it: q4 is unresolved and this task then needs a /deviate.
- depends on: t3, t10
- covers: c6
- acceptance:
  - The sidecar answers POST /v1/audio/synthesize {text, voice} with raw PCM16 mono 24 kHz and GET /v1/health/ready exactly as `chatterbox_server.py` does, so `tts_client`'s HTTP code is unchanged
  - Language is a server setting (`TTS_LANGUAGE`=he) passed to the engine; with it unset the existing `chatterbox_server.py` behaviour and file are byte-identical to main
  - The tts service block is added to docker-compose.audio-he.yml; no non-commercially-licensed weight is referenced by any shipped template (q4 deferred, plan risk r3)

### t12 — Wire it in app.py: session language to STT, tool branch in the response driver, stage timings

- instruction: Owns lobes/realtime/app.py only (a pragma:no-cover shell) — push every decision into the stdlib modules t4-t9 own and keep app.py to wiring. Extract the STT request builder into a testable pure function in `_turn.py`'s sibling or `audio_facade.py` if needed, but touch no file another wave-mate owns.
- depends on: t7, t8, t9
- covers: c11, h22, c33, h32
- acceptance:
  - `_forward_turn_to_stt` sends the session language (per-session override, else `REALTIME_LANGUAGE`, else en); a test over the extracted pure request builder shows the multipart fields are byte-identical to today's when nothing is configured
  - The response driver handles a tool-call result without synthesizing speech, and response.done carries per-stage milliseconds (stt, generate, `tool_wait`, phonikud, tts, `first_delta`) with absent stages omitted

### t13 — Capabilities advert names the served audio engines

- instruction: Owns lobes/roles.py and tests/`test_roles`\*.py. Check first how the gateway process obtains env for roles.py (it may already pass os.environ through) — that determines whether h25 holds.
- covers: c27, h27, c14, h25
- acceptance:
  - lobes/roles.py takes stt/tts model, runtime and language from the deployment's declaration (`STT_MODEL` / `STT_RUNTIME` / `STT_LANGUAGE` and the TTS equivalents), defaulting to today's Parakeet/Chatterbox constants; with nothing declared lobes capabilities and GET /capabilities output is byte-identical to main
  - git diff main -- lobes/gateway/ is empty for the whole branch; if the advert cannot be made honest without a gateway change, the task stops and raises it (plan risk r4)

### t14 — Acceptance client: reSpeaker/Reachy capture-and-play script with a demo tool

- instruction: Owns scripts/realtime-he-accept.py and tests/`test_realtime_he_accept_helpers.py`. Stdlib + the websocket client already used by scripts/realtime-smoke.py. reSpeaker XVF3800 first (card 1, unclaimed); Reachy Mini second and only if reachy-mini-daemon releases the card. Memory: silent-failure-antipattern-voice-tools.
- depends on: t4
- covers: c22, h16, c2, h2
- acceptance:
  - scripts/realtime-he-accept.py captures mono 16 kHz from a chosen ALSA/pipewire device, plays 24 kHz deltas to the SAME device's sink, answers WebSocket PING, declares one harmless read-only tool via session.update, executes it client-side and returns `function_call_output` then response.create
  - It logs every wire event with a timestamp, prints transcript-then-call order, and every failure path speaks (named error + exit code) rather than idling to a timeout
  - grep shows no tool implementation, tool schema or agent-framework import under lobes/ — the only tool lives in scripts/ and tests fixtures

### t15 — First-class selection: lobes init --audio --audio-lang he, overlay materialisation, new goldens

- instruction: Owns lobes/cli/`_commands`/init.py, lobes/runtime/`_compose.py`, lobes/profiles/`shape_render.py` (only if needed) and tests/goldens/\*\*. Language is an overlay choice, not a shape and not a variation (frame c16).
- depends on: t8, t10, t11, t13
- covers: c20, h14, c25, h18
- acceptance:
  - lobes init --fleet --audio --audio-lang he (dry-run by default) materialises docker-compose.audio-he.yml, the two new Dockerfiles and env.audio-he.example, and lobes fleet up includes the overlay when present; without the flag the scaffold is byte-identical to main
  - Every existing file under tests/goldens/ is byte-identical; new goldens are added only for the Hebrew selection and tests/goldens/README.md's regen note is followed
  - The full offline suite (uv run pytest -n auto) passes with no pre-existing test modified except where a task above explicitly added tool or language cases

### t16 — Docs, explain catalog, site harness events, version bump

- instruction: Owns docs/\*\*, lobes/explain/catalog.py, site/\*\*, CLAUDE.md, pyproject.toml, CHANGELOG.md, uv.lock. Memory: version-bump-stdin-changelog (pipe lowercase-key JSON, commit the uv.lock re-pin).
- depends on: t12, t15
- covers: c26, h19
- acceptance:
  - New docs/hebrew-realtime.md; docs/realtime-pipeline.md, docs/openai-api.md, docs/gateway-fleet.md, lobes/explain/catalog.py and CLAUDE.md stop saying tool calls are a follow-up and that STT/TTS are fixed English engines — every new statement marked DECLARED/UNVALIDATED until t17's transcript lands
  - Each file:line cited by frame claim c26 is re-read at the branch point and the docs quote what it says now; site/src/scripts/realtime-events.ts lists the new event types and the site still builds
  - Version bumped with the version-bump skill and a CHANGELOG entry; markdownlint passes on touched docs

### t17 — Live acceptance on the Spark: one Hebrew session with a tool round-trip and a barge-in, on the reSpeaker

- instruction: Deploy from the PR's TestPyPI dev wheel via lobes-compose.sh (memory: testpypi-dev-wheel-propagation-race — wait 60 s and retry). The operator speaks; the agent does not synthesize the Hebrew input.
- depends on: t12, t14, t15, t16
- covers: c1, h1, c21, h15, c15, h26
- acceptance:
  - docs/evidence/2026-09-accept-hebrew-realtime-spark.txt shows ONE session through the gateway tunnel: a Hebrew utterance from the real XVF3800 microphone transcribed correctly with a negative control, a Hebrew reply from model=associate heard on its speaker, a second utterance triggering the demo tool whose client-supplied result is spoken, and a barge-in attempted while the speaker is playing
  - The latency table is built from response.done timings alone; the X-Lobes-Mesh-Member header on the generate hop and the absence of any other cross-box traffic during the session are shown
  - Anything that failed is reported as failed with its output; docs flip from DECLARED/UNVALIDATED to VALIDATED only for what the transcript actually shows

## Risks

- [unknown_nonblocking] Which runtime serves the ivrit.ai Whisper checkpoint on the GB10 (transformers+PyTorch vs vLLM's Whisper path) is unmeasured; t10's Dockerfile cannot be written until t2 picks one (task t2)
- [unknown_nonblocking] Which Hebrew TTS engine wins is unmeasured and depends on a human listening verdict; if phonikud-tts wins, t11 cannot ship it without the deferred q4 licence decision and needs a /deviate (task t3)
- [follow_up] q4 deferred by the operator: whether a non-commercial phonikud-tts may appear in Apache-2.0 fleet templates is decided after the A/B (frame v6) (task t11)
- [unknown_nonblocking] h25 says the branch touches nothing under lobes/gateway/, but an honest stt/tts advert (c27) may need the gateway to pass new env to roles.py; if so one of the two confirmed conditions must give and the operator decides which (task t13)
- [unknown_nonblocking] The tool-wait deadline has no natural default and what the user hears while waiting is undesigned (frame v2); t6 ships it as a knob, the default value is chosen from t17's observations (task t6)
- [follow_up] First-audio latency is unbudgeted: generate and TTS are whole-reply stages inside a turn (frame v1). Sentence-level streaming of generate->phonikud->TTS is NOT in this plan; t17's timings decide whether it becomes the next plan (task t17)
- [unknown_nonblocking] associate is a mesh-shared lane on the Orin (`max_num_seqs`=2); a voice turn behind a long-context prefill has no priority and no local fallback (frame v4) (task t17)
