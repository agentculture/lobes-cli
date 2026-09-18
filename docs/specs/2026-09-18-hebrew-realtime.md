# hebrew-realtime

> lobes ships a Hebrew realtime duplex voice session on the DGX Spark: speak Hebrew into GET /v1/realtime, an ivrit.ai Whisper model transcribes it, associate answers in Hebrew, phonikud vocalizes the reply before a Hebrew-capable TTS speaks it back — and when associate decides to call a tool, the call is handed to the connected client over the same socket and the client's result is spoken back, so an agent the client owns can operate this machine by voice

## Audience

- The builder of a voice agent that does operations on this machine. That agent, its tools and their execution are the AUDIENCE, not part of this deployment — lobes ships only the Hebrew realtime duplex session and the tool-call wire

## Before → After

- Before: No realtime is running on the Spark today: ~/.lobes/.env has `STT_FEASIBLE`=false and `TTS_FEASIBLE`=false, only model-gear-gateway and model-gear-comfyui are up, and docker-compose.audio.yml + Dockerfile.parakeet/chatterbox there date from 2026-07-17..22 (pre-0.54.1). 'Replace existing realtime' is therefore a bring-up beside a stale overlay, not a swap of running sidecars
- Before: Duplex streaming in both directions and barge-in are already built (#149/#151, paced delivery in 0.54.1): audio streams in as `input_audio_buffer`.append and out as response.audio.delta on one socket. hebrew-realtime reuses that session as-is and does not rebuild it
- After: A Hebrew speaker talks to the Spark through a microphone and hears a Hebrew answer from associate on the same socket, can interrupt it, and can ask for something that needs a tool: the client receives the function call, runs it under its own authority, returns the output, and hears the result spoken. English realtime still renders and behaves exactly as before for any box that does not select Hebrew

## Why it matters

- Today the realtime lane is English-only by construction (Parakeet ignores language, app.py:208 hardcodes 'en') and cannot call tools at all, so a Hebrew-speaking operator cannot drive a machine agent by voice on local models; docs/realtime-pipeline.md:703-709 has carried tool calls as a named follow-up since #151

## Requirements

- associate is the voice lane's speaker: the realtime bridge generates through model=associate (the fast model), not the current multimodal default
  - honesty: The bridge sends model=associate on every voice turn of the Hebrew deployment, and when associate is unreachable the session emits `generate_failed` naming `hosted_by` — never a silent answer from another model
- associate's tool calling is connected to the realtime session: tools the client declares reach the generate call, and a tool call associate makes is delivered to the client over the socket, whose result continues the turn
  - honesty: A session that declares no tools emits exactly the pre-change event sequence (the additive-only contract reachy-mini-cli depends on), and a declared tool call is never synthesized as speech
- STT is an ivrit.ai Hebrew Whisper model and the TTS path runs through phonikud (Hebrew diacritization/G2P) — replacing the English Parakeet + Chatterbox pair for this deployment
  - honesty: The STT sidecar really loads an ivrit-ai checkpoint (its readiness probe reports the model id) and the text reaching TTS has really passed through phonikud (observable as niqqud in a debug event or log), not a pass-through stub
- The realtime turn grows a tool-calling path that does not exist today: lobes/realtime never references tools/`tool_choice` (`_turn.py`:115-144 builds model/messages/`max_tokens`/temperature/`chat_template_kwargs` only; `_turn.py`:321-334 reads message.content and silently ignores `tool_calls`); SessionConfig has no tools field (`_session.py`:225-239); every client event except `input_audio_buffer`.append and response.create is dropped (`_conversation.py`:397-406); config arrives as WS query params, there is no session.update
  - honesty: Tool events use OpenAI Realtime's names and payload shapes verbatim (session.update tools, response.`function_call_arguments`.done, conversation.item.create with `function_call_output`, response.create to continue), so a stock OpenAI Realtime client's tool loop works unmodified
- The floor state machine gains a waiting-on-the-client's-tool state between RESPONDING and SPEAKING (`_floor.py`:190-246): machine-held so barge-in still interrupts it, no TTS deadline armed, and its own deadline — the existing 60 s per-stage timeouts are for backend HTTP calls the floor controls, not an external agent running a command
  - honesty: Speaking while the floor waits on a tool interrupts it exactly as it interrupts speech, and a tool result that never arrives ends in a named timeout error with the session still usable — never a wedged floor
- The live session's STT language stops being hardcoded English: app.py:208 sends data={'language':'en'} on every WS turn (the batch route's form default at app.py:131 is also 'en'), and both `DEFAULT_SYSTEM_PROMPT` copies (`_session.py`:564-569, `_turn.py`:102-107) are English — a Hebrew session needs a language knob and a Hebrew default prompt
  - honesty: Language is a per-deployment default AND a per-session override, and with neither set the STT request still says 'en' byte-for-byte
- phonikud sits in the TTS text path, after `_clean_for_tts` and before synthesis: `tts_client.py`'s cleaning/splitting (160-177, 64-97, 185-218) is Latin-punctuation-shaped, entirely untested (module is coverage-omitted, pyproject.toml:70-86), counts niqqud combining marks against its 600-char ceiling, and passes gershayim through raw
  - honesty: phonikud failing or timing out degrades to un-vocalized text with a logged warning rather than a silent turn, and its per-sentence latency on this box's CPU is measured and written down
- Hebrew STT is a new sidecar, not a `PARAKEET_MODEL` override: `listen_server.py`:53 loads via `nemo_asr` ASRModel.`from_pretrained`, which cannot load a Whisper checkpoint, and Dockerfile.parakeet:20 bakes the Parakeet download. The sidecar keeps the existing contract (POST /v1/audio/transcriptions, /v1/health/ready with `model_loaded`+`cuda_ok` per `_readiness.py`) so the bridge and gateway do not change
  - honesty: The Hebrew STT sidecar answers the same two routes with the same response shapes as `listen_server.py`, so neither lobes/gateway nor the bridge's STT client changes
- Live validation uses real hardware on the Spark: a Reachy Mini Lite on USB supplies a microphone with AEC and direction-of-arrival plus a speaker, and a second reSpeaker will get its own speaker later — so the acceptance run is a real microphone with client-edge AEC, not synthesized audio
  - honesty: The acceptance client captures from the Reachy Mini card with its AEC active, and barge-in is attempted while the speaker is actually playing — the case synthesized audio could never test
- The stt/tts capabilities advert names the engine actually served: lobes/roles.py:265-268 hardcodes `_STT_MODEL`=nvidia/parakeet-tdt-0.6b-v2 and `_TTS_MODEL`=ResembleAI/chatterbox (this box advertises exactly those today while serving neither), so a Hebrew lane would announce an English model to every mesh member and client. model, runtime and a language field come from the deployment's declaration
  - honesty: GET /capabilities on the Hebrew Spark shows the ivrit-ai id and language he for stt, and an English box's advert is byte-identical to today's
- The bridge, not the model lane, owns tool-call bookkeeping: a `function_call_output` is accepted only when its `call_id` matches the one outstanding call of the current response; an orphan, late (after timeout or barge-in) or duplicate output gets a named error and never enters history. PROBED 2026-09-18: associate accepted a role:tool message with an unknown `tool_call_id` and spoke its content as fact (0.52 s), so nothing downstream will catch it
  - honesty: Offline tests cover orphan, late and duplicate outputs each producing the named error with history unchanged
- Mixed-script replies are in scope for the TTS A/B and the acceptance run: a tool-result reply is mostly Latin paths, dated filenames and digits inside a Hebrew sentence (probe A: 'בתיקייה /home/spark/git/lobes-cli/docs/specs נמצאים שלושה קבצים: 2026-09-18-hebrew-realtime.md ...'). The Hebrew default system prompt tells the model to describe rather than recite paths and identifiers, and each TTS candidate is scored on a mixed sentence, not only on clean Hebrew
  - honesty: The A/B table has a row per engine for a mixed Hebrew+Latin+digits sentence, judged by a Hebrew speaker, and the default prompt's effect is shown with a before/after reply
- Within a turn the server always emits conversation.item.`input_audio_transcription`.completed BEFORE any function-call event, so the client that owns execution can show, log or confirm what was heard before acting. Whisper-family models hallucinate fluent text on silence and noise (parked v3), and with tools declared a phantom transcript becomes a phantom command — lobes cannot judge intent, but it must never hide the transcript that caused a call
  - honesty: An offline test asserts event order for a tool turn, and the acceptance client logs transcript then call
- session.update is accepted with OpenAI Realtime's flat tool shape ({type:function, name, description, parameters}) and translated to chat-completions' nested {type:function, function:{...}} for the generate call; `tool_choice` passes through. Every session.update field the server does not act on is answered explicitly (session.updated echoes only what took effect), never silently swallowed — the silent-failure rule learned on PR #150/#152
  - honesty: A stock OpenAI Realtime session.update payload with tools round-trips in an offline test, and an unsupported field is visibly absent from session.updated
- Stage timings are observable on the wire: response.done carries additive per-stage milliseconds (stt, generate, `tool_wait`, phonikud, tts, `first_delta`). The success signal demands per-stage first-audio latency and today only boundary events carry `at_ms`, so without this the acceptance numbers would come from log scraping
  - honesty: The acceptance transcript's latency table is built from response.done fields alone

## Honesty conditions

- All four legs are exercised in ONE session on the Spark, not four separate demos: Hebrew speech in, Hebrew transcript, Hebrew reply audio out, and one tool round-trip whose result is spoken
- No file under lobes/ executes a tool, imports an agent framework, or ships a tool definition; the only tool schemas in-tree are test fixtures and the acceptance client's
- The live proof backs up ~/.lobes before touching it and never runs docker compose inside lobes/templates/ — the lobes-deploy skill's rule
- The PR's diff touches no file under lobes/gateway/
- STT, TTS and the bridge for the Hebrew session all run on the Spark; only POST /v1/chat/completions crosses to the Orin, and it carries X-Lobes-Mesh-Member
- Every existing golden under tests/goldens/ is byte-identical after the change; only new goldens are added
- The transcript includes a negative control for STT (a different utterance must NOT match), per-stage latency numbers for first audio, and uses the real Reachy Mini microphone rather than synthesized audio
- No change to `_segmenter.py`, `_pcm.py` or the audio half of `_wire.py`; the existing realtime test files pass unmodified except where they gain tool cases
- English-default behaviour is proven by the unchanged offline suite plus unchanged goldens, not asserted
- The cited lines still say what is claimed at the commit the work branches from

## Success signals

- An acceptance transcript under docs/evidence/ from the Spark: a synthesized-or-recorded Hebrew utterance through the gateway tunnel is transcribed correctly (with a negative control), answered by associate in Hebrew, spoken back as 24 kHz deltas, and a second utterance triggers a declared tool whose client-supplied result is spoken — with first-audio latency measured per stage. Until it lands every surface says DECLARED/UNVALIDATED (#108)

## Scope / boundaries

- The gateway is not touched: /v1/realtime is an opaque byte tunnel gated on stt feasibility (gateway/`_realtime.py`:37-163) so new tool events pass unparsed; /v1/audio/\* is path-routed with no model id or language; tools/`tool_choice`/streamed `tool_calls` pass byte-identical for associate and `GATEWAY_FORCE_STRICT_TOOLS` never reaches it (`_STRICT_TOOL_LANES`={'primary'}, server.py:399)
- The WebSocket session never crosses a box: the realtime tunnel refuses a peer-hosted stt lane (gateway/`_realtime.py`:18-26, server.py:5065-5150), so Hebrew STT + TTS + the bridge are co-located on the Spark with the gateway that terminates the socket, while associate stays on the Orin and is reached per-turn through the mesh (POST, single hop)
- The English Parakeet + Chatterbox pair stays in-tree and stays the default (cite-don't-delete): Hebrew is a selectable alternative for the stt/tts roles, and a box that never asks for it renders byte-identical goldens

## Non-goals

- hebrew-realtime is not a 'variation' in this repo's vocabulary and gets no deployments/<id>/ entry of that kind: lobes/variation.py:1-51 defines a variation as machine-type identity only, never a feature or language flavor. The closest existing axis is the deployment shape — and shapes today only turn roles on/off, with `ROLE_SERVICE` a flat stt->stt / tts->chatterbox map (`shape_render.py`:84-104) and no \[roles.stt\]/\[roles.tts\] tables
- No agent, no tool implementations and no tool execution ship in lobes: the bridge never runs a tool, it relays the model's call to the client and waits. This keeps the role contract intact — associate is forbidden `repo_action`, and it stays a proposer; whatever acts on the machine is the client's agent under the client's authority

## Assumptions

- associate speaks usable Hebrew even though its model card does not list it: MEASURED live 2026-09-18 from the Spark gateway through the mesh to the Orin — two Hebrew prompts answered in fluent Hebrew in 1.59 s / 1.95 s (56/60 tokens, `enable_thinking`:false honored, no reasoning leak), with one wrong word choice in the first reply. Card lists only English, Spanish, French, German, Italian, Japanese
- The ivrit.ai STT candidate is ivrit-ai/whisper-large-v3-turbo (0.8B, Apache-2.0) served from transformers+PyTorch or vLLM's native Whisper path; the ivrit.ai page the user linked is a dataset/org landing page naming no model, and the faster-whisper/-ct2 variants ivrit.ai recommends are blocked on this box — CTranslate2 ships no aarch64 CUDA wheel (speaches-ai/speaches#620, filed against a DGX Spark)
- phonikud runs on CPU inside the bridge or TTS sidecar: it is an int8 ONNX diacritizer (phonikud-onnx, phonikud-1.0.int8.onnx) plus a rule-based phonemize() to IPA, code CC BY 4.0 — and onnxruntime-gpu has no official aarch64 wheel, so CPU is the only no-build path. Its latency per sentence is unstated upstream and unmeasured here
- The hardware client is not ready-made: both USB devices capture `S16_LE` 2-channel 16 kHz only (the session wants mono, 16 kHz is an accepted rate), reachy-mini-daemon (pid 3192) and pipewire already hold the Reachy card, and reachy-mini-cli still speaks the pre-#151 binary wire (reachy-mini-cli#115) — so the live proof needs a small capture/playback client of its own, or #115 landed first
- An interrupted or timed-out tool wait may leave the assistant's `tool_calls` entry in history without a result: the associate lane tolerates it (probe B). This is measured for associate only — a deployment overriding `OPENAI_MODEL` to another lane may get a template error, so the bridge closes the entry with a synthetic cancelled output rather than relying on lane tolerance
- The reSpeaker XVF3800 is the first acceptance device, the Reachy Mini the second: the reSpeaker is unclaimed by any process and has mic + speaker on one USB device (so its hardware AEC sees the playback reference), whereas the Reachy card is held by reachy-mini-daemon and reachy-mini-cli still speaks the pre-#151 wire (#115)

## Scope exploration

- `s1` — `live deployment ~/.lobes on the Spark (docker ps, .env, docker-compose.audio.yml)`: `STT_FEASIBLE`=false / `TTS_FEASIBLE`=false / `PRIMARY_FEASIBLE`=false / `COMPOSE_PROFILES`=innereye; containers up = gateway + comfyui only; audio overlay file is a stale Jul-22 copy building Dockerfile.realtime.local from a vendored `lobes_cli`-0.54.0 wheel
  - seeds: `c7`
- `s2` — `live memory on the GB10 (ps rss, free -g)`: ComfyUI main.py holds 34.1 GiB RSS of 121 GiB (87 GiB used, 33 available); `INNEREYE_DECLARED_PEAK_GIB`=31.42. Dropping it is what funds a ~1.6 GiB Whisper-large-v3 + TTS pair; dropping it is pure deployment config (unset `COMPOSE_PROFILES`=innereye), no repo code change
  - seeds: `c5`
- `s3` — `associate live probe via http://localhost:8001/v1/chat/completions (model=associate, proxied to orin.tail0be7e0.ts.net:8000)`: Hebrew replies 1.59 s and 1.95 s wall for 56/60 completion tokens; `chat_template_kwargs` `enable_thinking`:false produced direct content with no reasoning trace; a Hebrew 'list my home dir' request with one tool returned `tool_calls`=\[`list_directory` {path:/home/spark}\] in 0.48 s. Hebrew quality is unofficial: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B model card omits Hebrew
  - seeds: `c8`, `c3`
- `s4` — `lobes/realtime/_turn.py, _conversation.py, _session.py, _wire.py`: No tools anywhere in the package; generate is one blocking non-streaming POST (app.py:583-587); conversation.item.create is silently ignored today; docs/realtime-pipeline.md:703-709 names tool calls over the session an explicit follow-up, not a near-done gap. New surface needed: session tools declaration, function-call out event, `function_call_output` in event, history entries with tool role
  - seeds: `c9`, `c4`
- `s5` — `lobes/realtime/_floor.py`: States LISTENING/TRANSCRIBING/RESPONDING/SPEAKING/CLOSED; `on_reply_text` (515-532) moves RESPONDING->SPEAKING unconditionally; `MACHINE_HELD_STATES`, `_STAGE_OF_STATE`, Stage and FailureReason all need a tool member; after a tool result the turn loops back to RESPONDING (generate again with the result in history)
  - seeds: `c10`
- `s6` — `lobes/realtime/app.py STT forward + system prompts`: `_forward_turn_to_stt` hardcodes language=en (app.py:208); no model name is sent to STT; VAD/segmenter/`_pcm` are language-agnostic and untouched; system prompt is overridable today via `DEFAULT_SYSTEM_PROMPT` env and per-session query param
  - seeds: `c11`
- `s7` — `lobes/realtime/tts_client.py + chatterbox_server.py`: synthesize() POSTs {text,voice} to `TTS_URL`/v1/audio/synthesize and reads whole PCM16 24 kHz (no streaming); `chatterbox_server.py` calls ChatterboxTTS.`from_pretrained` + generate(text, exaggeration, `cfg_weight`, `audio_prompt_path`) — English single-language class, no `language_id`; voice lane is a separate semaphore pool
  - seeds: `c12`, `c6`
- `s8` — `lobes/templates/fleet/{docker-compose.audio.yml,Dockerfile.parakeet,listen_server.py,env.audio.example}`: stt service builds Dockerfile.parakeet (NeMo, scitrera/dgx-spark-vllm base); `PARAKEET_MODEL` is runtime-read but NeMo-only; docs/parakeet-stt.md:38 says Parakeet is English-only and ignores language; the overlay is switched on by file presence, not a compose profile
  - seeds: `c13`
- `s9` — `lobes/gateway/_realtime.py, server.py audio fan-out + rewrite_model, _config.py`: Gateway holds no STT/TTS model id and no language assumption; `rewrite_model` (server.py:234-245) touches only the model key; associate lane parsers are `qwen3_coder` + `nemotron_v3` (fleet docker-compose.yml:1754-1755), tool calls VALIDATED live 2026-08-20, strict-tools UNPROBED
  - seeds: `c14`
- `s10` — `gateway realtime tunnel vs mesh forwarder`: Mesh/proxy forwarding is POST-only; only the generate call leaves the box. The bridge already calls `OPENAI_BASE_URL`=<http://gateway:8000>, so model=associate resolves via the mesh with no bridge change beyond `OPENAI_MODEL`. Every voice turn therefore pays a Spark->Orin tailnet hop (measured 1.6-2.0 s for ~58 tokens non-streamed)
  - seeds: `c15`, `c3`
- `s11` — `lobes/variation.py, variation_catalog.py, profiles/shape_render.py, shapes.py, init.py --audio`: --audio is a monolithic boolean (init.py:1962-1965); `AUDIO_ROLES`=('stt','tts') fixed; no language/engine axis exists anywhere in the repo (grep hebrew/phonikud/ivrit = 0 hits). A new shape means 4 new goldens per tests/goldens/README.md:49-54
  - seeds: `c16`
- `s12` — `CLAUDE.md associate role contract + user's stated audience`: associate responsibilities include `tool_use` but forbid `final_decision`/`security_decision`/`code_authoring`/`repo_action`; relaying calls to a client-side executor is consistent with that, executing them server-side would not be
  - seeds: `c17`, `c4`
- `s13` — `https://www.ivrit.ai/en/ivrit-ai-2/ + https://huggingface.co/ivrit-ai + speaches#620`: Landing page: 22,000 h dataset, no model named. HF org: whisper-large-v3, -turbo, -ct2, -turbo-ct2 (Oct 2025), whisper-large-v3-turbo-onnx (Dec 2025), Yiddish siblings, pyannote diarization; no TTS, no LLM, nothing streaming. Cards Apache-2.0; WER lives on an external leaderboard, not fetched. vLLM-loads-this-checkpoint is architecturally plausible, untested
  - seeds: `c18`, `c13`
- `s14` — `https://github.com/phonikud/phonikud README + org`: Three stages: add niqqud -> add stress/vocal-shva marks (invented Unicode chars) -> IPA. API: Phonikud(path).`add_diacritics`(text); phonemize(vocalized). Designed to feed thewh1teagle/phonikud-tts (StyleTTS2/Piper, onnxruntime) whose LICENSE is non-commercial; ILSpeech/Saspeech datasets are non-commercial too
  - seeds: `c19`, `c12`
- `s15` — `tests/goldens contract + repo cite-don't-delete rule`: Chatterbox appears in ~85 files and Parakeet's id in 12 (roles.py, explain/catalog.py, learn.py, docs, tests); every shape x card has a byte-for-byte golden .env. Replacing rather than adding would move every golden and break reachy-mini-cli's English path
  - seeds: `c20`
- `s16` — `USB audio on the Spark (lsusb, arecord -l, aplay -l)`: 38fb:1001 Pollen Robotics Reachy Mini Audio = ALSA card 2 (capture + playback); 2886:001a Seeed reSpeaker XVF3800 4-Mic Array = card 1 (capture + playback); /dev/ttyACM0 present; reachy-mini-cli installed at ~/.local/bin. Client-edge AEC is exactly where deviation d1 of #151 says AEC belongs, and it retires two standing UNVALIDATED items: real microphone, and barge-in under real echo
  - seeds: `c22`
- `s17` — `ALSA hw params + device holders (arecord --dump-hw-params, fuser, pgrep)`: hw:1,0 and hw:2,0 both report FORMAT `S16_LE`, CHANNELS 2, RATE 16000; pcmC2D0p is held by pipewire; reachy-mini-daemon is running. Recalled gotcha (PR #150/#152): paplay hung on a sink the daemon held exclusively, and a /v1/realtime client must answer WebSocket PING or uvicorn drops it at ~20 s
  - seeds: `c23`
- `s18` — `lobes/realtime/app.py:583-587 _post_generate + tts_client synthesize + docs/evidence/`: The WIRE streams both ways; the two stages INSIDE a turn do not — `_post_generate` is one awaited httpx POST returning resp.content (no stream/aiter anywhere in `_turn.py`, app.py, `tts_client.py`, `_conversation.py`), and TTS is read whole before the first delta. docs/evidence/ holds no realtime transcript newer than 2026-07-22, which predates the 0.54.1 pacing fix, so barge-in is built but has no in-repo live transcript
  - seeds: `c24`
- `s19` — `challenge pass / adjacent-systems lens: lobes/roles.py audio advert + live GET /capabilities`: roles.py:265-268 are module constants fed to `_audio_role`; live advert on the Spark reads stt=nvidia/parakeet-tdt-0.6b-v2 / tts=ResembleAI/chatterbox with feasible:false. Scoping had marked roles.py 'not touched' — that was wrong
  - seeds: `c27`
- `s20` — `challenge pass / failure-mode lens: associate lane with malformed tool history (scratch probe, 3 requests)`: A: well-formed call+result -> correct Hebrew summary in 0.92 s. B: dangling assistant `tool_calls` followed by a new user turn -> 200, normal reply, no 400 (so an interrupted tool wait may leave the call in history on THIS lane). C: orphan tool result with no matching call -> 200 and the content is repeated as truth
  - seeds: `c28`
- `s21` — `challenge pass / unstated-assumptions lens: what a tool-using voice reply actually contains (probe A output)`: Everyone in the frame assumed replies are Hebrew prose; with tools they are file paths, dates and numbers. phonikud documents an English fallback; Chatterbox Multilingual's behaviour on Latin tokens under language=he is unknown
  - seeds: `c30`
- `s22` — `challenge pass / security lens: voice -> transcript -> tool call -> machine operation; live bearer gate`: `GATEWAY_API_KEY` is set in the Spark's .env so /v1/realtime is bearer-gated on this box (the gate is opt-in in general). Tools are declared and executed by the connecting client, so an unauthenticated peer gains no execution on this machine through lobes — the residual hazard is misheard or hallucinated speech, which only the client can gate
  - seeds: `c31`
- `s23` — `challenge pass / adjacent-systems lens: OpenAI Realtime wire vs chat-completions tool schema; recalled silent-failure memory`: Realtime tools are flat, chat-completions tools are nested; `_session.py` has no session.update handler and `_conversation`.`on_control_event` drops unknown events without a word (397-406)
  - seeds: `c32`
- `s24` — `challenge pass / observability lens: _session.py event dataclasses vs c21's latency demand`: Boundary events carry `at_ms` and reason (#151); response.\* events carry no timing. Additive fields keep the OpenAI names verbatim (h20)
  - seeds: `c33`
- `s25` — `challenge pass / overlooked-actors lens: innereye client after ComfyUI is dropped (c5)`: Live advert today: innereye feasible:true ready:true on the Spark, hosted nowhere else. After the drop it flips infeasible and every mesh peer 404s `role_infeasible` with no `hosted_by`. ~/.lobes holds docker-compose.\*.bak-20260918-\*-pre-innereye files, so rollback is a restore + `COMPOSE_PROFILES`=innereye. The user decided this explicitly; recorded as a consequence, not a finding
  - seeds: `c5`
- `s26` — `challenge pass / operations lens: mesh routing of audio roles (grep audio|stt|tts|realtime in gateway/_mesh_*.py = 0 hits)`: No mesh member hosts stt/tts today and the mesh routing modules never name the audio lanes; clean for this idea — a Hebrew stt is not auto-wired to English callers elsewhere. Not examined: whether the mesh announce payload includes audio roles via the generic capabilities copy
- `s27` — `challenge pass / reversibility lens: ~/.lobes backups + h6`: Backup-first and never-compose-in-templates are already honesty conditions; clean. Not examined: disk headroom for a Whisper + TTS image build and HF cache on this box
- `s28` — `USB audio, second look (aplay --dump-hw-params hw:1,0, fuser, pactl sinks)`: reSpeaker XVF3800 (card 1) now has a speaker on its own output: playback is `S16_LE` 2ch 16000 Hz ONLY, no process holds its capture or playback nodes, its pipewire sink is SUSPENDED (free) while the Reachy Mini sink is RUNNING under reachy-mini-daemon. Two consequences: the session's 24 kHz mono deltas must be resampled to 16 kHz stereo at the client (pipewire does this if the client plays to the sink rather than hw:), and the XVF3800's AEC needs the far-end reference, which it only has when playback goes through its OWN USB output — so mic and speaker must be the same device for barge-in to be a fair test
  - seeds: `c22`, `c23`

## Decisions

- ComfyUI/innereye is dropped from this box to make room for the Hebrew audio stack

## Open parks

- [unknown_nonblocking] First-audio latency is unbudgeted: generate is one non-streaming POST (1.6-2.0 s measured Spark->Orin for ~58 Hebrew tokens), Whisper is batch per committed turn, phonikud latency is unstated, and TTS is full-read before the first delta. Whether sentence-level streaming of generate->phonikud->TTS is in scope for v1 or a follow-up is undecided
- [unknown_nonblocking] The tool-wait deadline has no natural value: an external agent running a shell command is open-ended, the floor's 60 s stage timeouts are not, and what the user hears while waiting (silence, a spoken filler, nothing) is undesigned
- [unknown_nonblocking] Whisper-large-v3-turbo hallucinates on silence/short clips and Silero's 600 ms `VAD_SILENCE_MS` was tuned on English; Hebrew WER for the chosen checkpoint on this box, and whether vLLM's Whisper path or a plain transformers sidecar serves it, are both unmeasured
- [unknown_nonblocking] associate is a shared lane: the Orin runs `max_num_seqs`=2 for the whole mesh and a cold long-context prefill there is measured in tens of minutes (2,390 s at 1M). A voice turn arriving behind one has no priority and no local fallback — hand is the only role this box could host as a floor, and its Hebrew is untested
- [unknown_nonblocking] phonikud's model-weight licence and CC BY 4.0 attribution duties for a redistributed container image were not read; only the repo README was
- [follow_up] q4 deferred: whether a non-commercial phonikud-tts may appear in Apache-2.0 fleet templates as an opt-in engine is decided after the A/B listening test
