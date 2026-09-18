# hebrew-realtime — handoff (written 2026-09-18, end of day 1)

Read this first when resuming. Everything below is committed on the local,
UNPUSHED branch `spec/hebrew-realtime` (53 commits ahead of `main`, tip
`5ec198d` + this file). Full offline suite: **5780 passed, 15 skipped**.

## What this is

A Hebrew realtime duplex voice session on the DGX Spark, with tool calling
relayed to the client (OpenAI Realtime event names). Method artifacts:

- Spec: `docs/specs/2026-09-18-hebrew-realtime.md` (frame `hebrew-realtime`, 35 claims, 18 obligations)
- Plan: `docs/plans/2026-09-18-hebrew-realtime.md` + `...-split.md` (17 tasks)
- Deviations d1–d9: `devague deviate --list` — **the plan text is stale wherever a deviation applies**
- Evidence: `docs/evidence/2026-09-hebrew-*.txt` (7 files; every number below comes from them)

## Decisions that changed the plan (deviations, all operator-approved)

| id | decision |
|---|---|
| d1 | Voice lane = Gemma 4 `senses` (`model=multimodal`) hosted LOCALLY on the Spark, not `associate` on the Orin |
| d2 | Measure 12B first, then larger. Result: **`nvidia/Gemma-4-26B-A4B-NVFP4` promoted live** (32–44 tok/s, clean tool calls). 31B dropped |
| d3 | h25 relaxed: additive gateway changes allowed; advert gets a nullable `language` field (NOT yet built) |
| d4 | conikud replaces phonikud — **superseded by d8** (BlueTTS needs no separate G2P) |
| d5 | Working loop before the TTS A/B; Chatterbox Multilingual was the stand-in voice |
| d6 | Mixed-language speech ("Evidense") is out of the acceptance bar |
| d7 | **Sentence-level streaming** pulled into the plan — IN PROGRESS (agent) |
| d8 | **BlueTTS is the Hebrew voice** (160–325 ms/sentence on CPU, 0% round-trip WER, operator: "Sounds perfect!") — sidecar NOT yet built |
| d9 | **Speculative turn-taking**: (A) hidden speculation at ~250 ms silence, (B) early commit ~350–400 ms + continuation merge. Build AFTER d7 merges. Classified risky |

Operator's latency target: first audio **~200–500 ms** (ideally 200–400). Measured today: 2.6 s (one sentence, Chatterbox). Projection: ~550–750 ms with d7 + BlueTTS; ~400–500 ms perceived with d9.

## Task status

Merged: t1 t2 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 (+ t3 measured, superseded by d8).
Open: **t15** (`lobes init --audio --audio-lang he` + new goldens), **t16** (docs, catalog entry for the 26B, explain, version bump, CLAUDE.md), **t17** (acceptance transcript).
New work from deviations: d7 streaming (running), web harness (running), BlueTTS sidecar, d9 speculation, d3 advert `language` field + gateway image.

## Background agents that were RUNNING at handoff

Their reports arrive as messages; if the session was compacted they may already be finished — check the branches.

| branch / worktree | task | model |
|---|---|---|
| `agent/he-stream` — `/home/spark/git/.worktrees.lobes-cli/agent-he-stream` | d7 sentence-level streaming in `lobes/realtime/**` (`_turn` SSE accumulator, new `_sentences.py`, floor segment queue, bridge streaming surface, `app.py`, `GENERATE_STREAM` knob) | opus |
| ~~`agent/he-site`~~ | web harness — **MERGED at handoff** (site tests 245/245, build OK). Untested in a real browser: the operator still has to try it (steps in `site/README.md`) | sonnet |

To integrate each: run its tests on the integration branch (baseline) → `git merge --no-ff agent/<name>` → run tests again → `git worktree remove` → full suite. They touch disjoint trees. **Verify every claim in a subagent report** — today two reports contained false provenance/claims that only a live run exposed.

## Live state of the Spark (`~/.lobes`) — CHANGED TODAY

Running: `gateway` (RELEASED image 0.79.0 → its `/capabilities` still advertises Parakeet/Chatterbox and quant `compressed-tensors`), `vllm-multimodal` (26B-A4B, util 0.28, MTP draft `google/gemma-4-26B-A4B-it-assistant`), `stt` (ivrit-ai Whisper turbo, transformers, confidence gate −0.35), `chatterbox` (Chatterbox **Multilingual** he, phonikud niqqud, T=0.3, retuned runaway guard), `realtime` (bridge built from this branch's wheel `lobes_cli-0.80.1`).
Stopped: `comfyui` (innereye infeasible mesh-wide; container kept, not removed).

- Full backup before anything: `~/.lobes.pre-hebrew-realtime-20260918T121206Z`; plus 8 in-place `*.bak-20260918-*` files.
- Hebrew deltas live in **`~/.lobes/docker-compose.override.yml`** (hand-carried copy of `docker-compose.audio-he.yml`, because the installed CLI does not know that overlay until t15) and two local Dockerfiles: `Dockerfile.realtime.local` (arg `LOCAL_WHEEL`) and `Dockerfile.chatterbox-ml.local`.
- `.env` keys set today: `MULTIMODAL_*` (26B), `MULTIMODAL_FEASIBLE=true`, `STT_FEASIBLE/TTS_FEASIBLE=true`, `LOBES_IOWAIT_DEGRADED_THRESHOLD=100`, `VAD_SILENCE_MS=1000`, `OPENAI_API_KEY` (= the gateway key, for the bridge), `REALTIME_LANGUAGE=he`, `OPENAI_MODEL=multimodal`, `STT_*`, `TTS_*`, `PHONIKUD_MODEL_PATH`, `COMPOSE_PROFILES=` (empty).
- To redeploy bridge/TTS after code changes: `uv build --wheel`, copy the wheel to `~/.lobes/`, then `lobes-compose.sh --apply build <svc>` and `--apply up -d --no-deps <svc>`.
- reSpeaker mixer: card 1 `'PCM',1` raised 40→50 (−10 dB). Revert: `amixer -c 1 sset 'PCM',1 40`.
- Scratch images left on the box: `he-spike-whisper-stt:test`, `he-spike-tts-ml:test`. BlueTTS checkout + venv: scratchpad `bluetts/` (session-specific path — re-clone if gone: `github.com/maxmelichov/BlueTTS`, weights `notmax123/BlueTTS2.5-onnx`).

## Traps learned today (do not re-learn)

1. `lobes-compose.sh` wants **`--apply` BEFORE the subcommand**; after it the run is silently a dry-run.
2. **No inline `# comments` on `.env` values** the CLI reads (#274) — I repeated this and had to hoist seven.
3. This GB10 reports a chronic ~98% `/proc/pressure/io` (artifact) → the gateway shed every LOCAL full-tier request 429 until the iowait threshold was set to 100.
4. `coolthor/gemma-4-12B-it-NVFP4A16` **damages Hebrew**; bf16 and Google's QAT `w4a16-ct` do not. vLLM 0.23.1rc1 refuses `draft_model` speculation for Gemma 4 (mixed KV-cache groups).
5. Whisper hallucinates plain text on non-speech (`'תודה רבה'`); `<|nospeech|>` prob reads 0.000 on this fine-tune; **avg token logprob separates cleanly** (speech ≥ −0.12, hallucination ≤ −0.58) → gate at −0.35. Also strip bidi controls.
6. Chatterbox Multilingual needs niqqud for Hebrew (92% → ~9% WER), samples non-deterministically (run-ons up to 24 s; T=0.3 best), and a Latin word derails it.
7. **The reSpeaker XVF3800's AEC reports `converged: false`** (Reachy Mini: `true`) via the sibling `~/git/microphone-cli` (`uv run microphone array aec get XVF3800`). On the reSpeaker the session barges in on its own voice (6 self-interruptions in 7 turns); on the Reachy Mini it does not. UNRESOLVED — ask the operator how the reSpeaker's speaker is wired. Use the **Reachy Mini** for live sessions until fixed.
8. Conversation on `/v1/realtime` is **opt-in**: the client must send `response.create` to arm. A client must stream the mic for the WHOLE session (own thread) or turns never end.
9. `VAD_SILENCE_MS=600` split a deliberate Hebrew sentence at an ~850 ms pause → raised to 1000 on this box (d9 will replace this trade-off).
10. Silero stays as the VAD (operator decision); PulseVAD false-triggered on all four non-speech clips.
11. Compose tags the multilingual TTS build `lobes-chatterbox:latest`, which is also `Dockerfile.chatterbox-ml.local`'s base → rebuilds stack on themselves. Harmless so far; clean up when BlueTTS replaces it.

## How to talk to it right now

```bash
cd ~/.lobes && set -a && . ./.env; set +a
cd ~/git/lobes-cli && LOBES_API_KEY="$GATEWAY_API_KEY" python3 -u scripts/realtime-he-accept.py \
  --base-url http://localhost:${VLLM_PORT:-8001} --language he --backend pipewire \
  --source alsa_input.usb-Pollen_Robotics_Reachy_Mini_Audio_202000386253800193-00.analog-stereo \
  --sink   alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio_202000386253800193-00.analog-stereo \
  --capture-channel 0 --converse 120 --tool-root ~/git/lobes-cli/docs --timeout 150
```

## Next steps, in order

1. Integrate `agent/he-stream` (TDD-gated, verify claims). `agent/he-site` is already merged.
2. **BlueTTS sidecar** behind the existing `/v1/audio/synthesize` contract (raw PCM16 mono **24 kHz** — BlueTTS outputs 44.1 kHz, resample; CPU onnxruntime; voice `noa`; RenikudPlus G2P downloads on first use — pre-warm in readiness). Keep Chatterbox Multilingual in-tree as the alternative. Open: the HF weights repo declares **no licence** (code is MIT) — operator was asked whether to contact the author (Max Melichov); unanswered.
3. Rebuild wheel → redeploy `realtime` + the new TTS; measure first audio from `response.done.timings` on the **Reachy Mini**; test a human barge-in (still UNTESTED).
4. **d9 speculation**: layer A (hidden), then layer B (early commit + continuation merge), tuned with the operator speaking.
5. Operator tests the web harness in the browser (`site/README.md` will carry the steps).
6. t15, d3 advert field, t16 (incl. catalog entry for the 26B; issue **#276** tracks testing it on the Orin), version bump, then t17 acceptance transcript on real hardware → `/validate-delivery` → `/summarize-delivery` → PR via the `cicd` skill (push needs `dangerouslyDisableSandbox`).

## Operator answers (2026-09-18, end of day)

- **BlueTTS weights licence:** the operator will ask the author himself. Do NOT contact the author or draft an issue. Until a licence is declared, BlueTTS may run on this box but its weights repo must not be referenced by a shipped template as a default.
- **reSpeaker wiring:** USB to the Spark, speaker plugged DIRECTLY into the reSpeaker's own 3.5 mm output — i.e. the correct wiring for its echo canceller (the chip's reference is what it plays over USB). So `converged: false` is NOT a wiring mistake. Still unexplained; next things to read with `~/git/microphone-cli`: firmware version, the reference/far-end gain and any `TEST_AEC_DISABLE_CONTROL`-style parameter, and whether the 3.5 mm output level (`'PCM',1`) is too low for the filter to see an echo at all. The operator asked whether "a screen" (monitor speakers) would work: NO for echo cancellation — audio played through HDMI never passes through the reSpeaker, so its canceller has no reference for it and the echo would be worse, not better. **Fallback the operator accepted: use the Reachy Mini.**
- **Reachy session silence:** the operator simply stopped talking. So a HUMAN BARGE-IN IS STILL UNTESTED on any device.

## Update — d7 streaming MERGED (2026-09-18, after the handoff)

- `agent/he-stream` merged `--no-ff`; worktree and branch removed. Full suite **5882 passed, 15 skipped**; black/isort/flake8 clean. No pre-existing test file was modified (checked from the diff stat: six NEW test files only). Templates touched: the two `audio-he` files only.
- Probe run by the main agent: the chunker keeps `14:30` whole and merges a short `שלום!` into the next sentence.
- **NOT deployed.** The live `realtime` container still runs the pre-streaming wheel. Next-step 3 (rebuild wheel → redeploy) now carries streaming with it.
- `GENERATE_STREAM` defaults **true for every language**, English included — a default-behaviour change that no live run has exercised yet. `GENERATE_STREAM=false` is the exact rollback.
- Left out by the agent because `_session.py` was frozen for it: `response.text.delta` events and the `first_sentence` / `first_audio_ready` timing keys (`STAGE_TIMING_KEYS` is pinned to six by a test). Under streaming, `response.text.done` can arrive AFTER the first audio deltas. `docs/realtime-pipeline.md` not updated (fold into t16).
- Lapse **l3** filed (proposed): the agent never saw `_sentences.py`'s tests fail before implementing.
- Next steps list: item 1 is DONE; start at item 2 (BlueTTS sidecar).

## Update 2 — BlueTTS + streaming LIVE, reSpeaker fixed (2026-09-18, evening)

Evidence: `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt`. Suite **5929 passed, 15 skipped**.

- **BlueTTS sidecar built and deployed** (`lobes/realtime/bluetts_server.py`, `Dockerfile.bluetts`, extra `bluetts`). Live as a NEW service `bluetts` (`model-gear-bluetts`, image `lobes-bluetts:local`, `Dockerfile.bluetts.local`) in `~/.lobes/docker-compose.override.yml`; weights copied to `~/.cache/bluetts/onnx_models` (durable; the scratchpad copy can go). The bridge points at it via `TTS_URL=${REALTIME_TTS_URL:-http://bluetts:9000}` and its phonikud hook is off (`PHONIKUD_MODEL_PATH=${REALTIME_PHONIKUD_MODEL_PATH:-}`). **Rollback to Chatterbox-ML:** set `REALTIME_TTS_URL=http://chatterbox:9000` and `REALTIME_PHONIKUD_MODEL_PATH` to the `PHONIKUD_MODEL_PATH` value, recreate `realtime`. `chatterbox` is still running (GPU memory not yet reclaimed). Backups: `*.bak-20260918-*-bluetts`.
- No shipped template names the BlueTTS weights; no `audio-he` overlay wiring for BlueTTS exists yet (licence pending — operator is asking the author). The Dockerfile must install from the engine's own `uv.lock` (renikud-plus 0.5.0 breaks the G2P).
- **Streaming deployed.** New knob `REPLY_FIRST_CLAUSE_MIN_CHARS` (default 24; **5 on this box**): first audio after a tool result 1.1 s → ~0.6 s. Live on the reSpeaker: 671–1020 ms after end of speech (plus the 1000 ms VAD wait).
- **reSpeaker FIXED — trap 7 is resolved:** its pipewire card profile was on the DIGITAL output at 34 % volume. `pactl set-card-profile <card> output:analog-stereo+input:analog-stereo` + sink volume 100 % → AEC `converged: true`, zero self-interruptions in 3 turns. If it misbehaves again, check the profile first (it can flip when the device re-enumerates; the source/sink numeric suffixes change too).
- **Human barge-in: WORKS** (first recorded, reSpeaker, 806 ms into a reply). n=1.
- Accept client fix: playback is drained before terminate (replies were cut at session end).
- Trap 12: never `pkill -f realtime-he-accept.py` from a shell whose own command line contains that string.
- BlueTTS speed: steps 5→2 buys only ~20–25 %; TensorRT (`blue_trt`) is the real lever, unbuilt. Operator asked whether sub-100 ms TTS would help: marginal for first audio, useful for the seam after the short first clause.
- **Open questions to the operator:** does the short first piece + pause sound natural? try `BLUETTS_STEPS=3`?
- **Next:** d9 speculative turn-taking (the 1000 ms VAD wait is now the largest single cost), then stop `chatterbox`, then t15/d3/t16/t17.

## Update 3 — d9 speculative turn-taking BUILT and live (2026-09-18, night)

Suite **5967 passed, 15 skipped**. Evidence sections 6–8 of `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt`.

- **Layer A, hidden speculation** (`_speculation.py`, `Segmenter(eager_silence_ms=)` → `SpeechPaused`/`SpeechResumed`, `ConversationBridge.build_speculative_request`, `VAD_EAGER_MS`, default 0). Adopted only when the real request is byte-identical (`can_adopt`). LIVE with a human on the reSpeaker: first audio **1–113 ms after the commit** (was 671–1020), 3 barge-ins honoured, 0 errors.
- **Layer B, continuation merge** (`CONTINUATION_WINDOW_MS`, `CONTINUATION_TOOL_HOLD_MS`, both default off/inert; `Floor.on_continuation_onset`, `Session.pop_history_if_last`, `ConversationBridge.tool_call_hold_ms`, `on_speech_started` now returns whether it was a continuation). Proven on a RECORDING only. **Not yet tried live.**
- **This box now runs:** `VAD_SILENCE_MS=500` (.env), `VAD_EAGER_MS=160`, `CONTINUATION_WINDOW_MS=1200`, `REPLY_FIRST_CLAUSE_MIN_CHARS=5` (override defaults). Safe fallback if live feels wrong: `VAD_SILENCE_MS=1000`, `CONTINUATION_WINDOW_MS=0`, `VAD_EAGER_MS=250` (the configuration the operator confirmed "works perfectly").
- **Honest latency floor:** perceived delay ≈ max(VAD_SILENCE_MS, VAD_EAGER_MS + ~600 ms pipeline [STT ~170 + first clause ~300 + TTS ~120]). At the settings above that is ~760 ms from end of speech. Going lower needs a faster pipeline (TTS streaming/TensorRT, smaller first clause), not a shorter silence.
- **Open problem exposed by the short commit:** a noise onset with a BLANK transcript still counts as a barge-in and kills the reply / closes an outstanding tool call. Pre-existing; design question for the operator.
- Wire: a merged turn produces two `transcription.completed` items (half, then whole); nothing marks the supersession. The web harness will show both.
- PocketTTS: operator says a Hebrew build exists (private HF Space `thewh1teeagle/pockettts`, 401 from here; no public Hebrew checkpoint found). A/B against BlueTTS when the operator can share it.
- Lapses filed (proposed): l3, l4. **Next:** live-test layer B with the operator, decide the blank-onset question, stop `chatterbox`, then t15/d3/t16/t17.

## Update 4 — live session with layers A+B (2026-09-18, night)

Operator: "speed is great". 12 turns, 0 errors; plain replies first audio 328–370 ms after the commit; tool turns 1.16–1.64 s (the 500 ms tool hold is a fixed cost on every tool turn — a candidate to tune down, e.g. 300). **Zero continuation merges occurred**, so layer B's take-back is proven on a recording only. Mixed Hebrew/English (folder names) → issue **#277**. Evidence section 9.
