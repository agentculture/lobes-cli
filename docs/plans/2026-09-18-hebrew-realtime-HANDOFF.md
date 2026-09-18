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
