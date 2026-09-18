# Delivery Summary — hebrew-realtime

plan: `hebrew-realtime` · run: `partial` · date: `2026-09-18`
baseline: `devague summary skeleton`

## Intent

Replace the DGX Spark's realtime overlay with a Hebrew duplex voice session — Hebrew speech in, Hebrew speech out, with the model's tool calls relayed to a client that owns the tools (OpenAI Realtime event names) — as an opt-in overlay that leaves the English overlay byte-identical. The run executed the 17-task plan `docs/plans/2026-09-18-hebrew-realtime.md` through `/assign-to-workforce`, amended in flight by ten approved deviations, and ended with PR #279 **open, not merged**. It is `partial` because one plan task (`t17`, the acceptance transcript) was not run, and one obligation (`o18`) is filed as failing.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Live proof 1/3 — back up ~/.lobes, stop ComfyUI, record the freed budget
- `t2` — Live proof 2/3 — Hebrew STT spike: ivrit-ai/whisper-large-v3-turbo under transformers+PyTorch vs vLLM Whisper on the GB10
- `t3` — Live proof 3/3 — Hebrew TTS A/B: Chatterbox Multilingual (he) with and without phonikud niqqud vs phonikud-tts, plus phonikud CPU latency
- `t4` — Pin the wire contract: session.update, tool events, language and stage-timing fields in `_session.py` and `_wire.py`
- `t5` — Generate payload and reply parsing for tools in `_turn.py`
- `t6` — Tool-wait state in the floor state machine
- `t7` — Conversation bridge: tool dispatch, call-id bookkeeping, event order, synthetic cancel
- `t8` — Settings and defaults: session language, Hebrew default prompt, tool-wait deadline knob
- `t9` — Vocalization hook: phonikud in the TTS text path
- `t10` — Hebrew STT sidecar: Whisper server honoring the `listen_server` contract
- `t11` — Hebrew TTS sidecar: the engine t3 selected behind the /v1/audio/synthesize contract
- `t12` — Wire it in app.py: session language to STT, tool branch in the response driver, stage timings
- `t13` — Capabilities advert names the served audio engines
- `t14` — Acceptance client: reSpeaker/Reachy capture-and-play script with a demo tool
- `t15` — First-class selection: lobes init --audio --audio-lang he, overlay materialisation, new goldens
- `t16` — Docs, explain catalog, site harness events, version bump
- `t17` — Live acceptance on the Spark: one Hebrew session with a tool round-trip and a barge-in, on the reSpeaker

## Actual Delivery

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `~/.lobes` backed up to `~/.lobes.pre-hebrew-realtime-20260918T121206Z`, ComfyUI stopped; `docs/evidence/2026-09-hebrew-realtime-spike-spark.txt` |
| `t2` | delivered | transformers chosen over vLLM Whisper; sidecar proof + avg-logprob hallucination gate; `docs/evidence/2026-09-hebrew-stt-spike-spark.txt` |
| `t3` | partial | Chatterbox Multilingual arms measured with/without niqqud (`docs/evidence/2026-09-hebrew-tts-ab-spark.txt`); the phonikud-tts arm was superseded by BlueTTS (`d8`) |
| `t4` | delivered | `lobes/realtime/_session.py`, `_wire.py` — session.update, tool events, language, `StageTimings` |
| `t5` | delivered | `lobes/realtime/_turn.py` — tools in the payload, `ToolCallResult`; later + SSE `StreamAccumulator`, `parallel_tool_calls: false` |
| `t6` | delivered | `FloorState.TOOL_WAIT`, timeout, interrupt; later + segment queue and `on_continuation_onset` |
| `t7` | delivered | `lobes/realtime/_conversation.py` — dispatch, call-id bookkeeping, named errors, synthetic cancel |
| `t8` | delivered | `Settings.language`, Hebrew default prompt, `TOOL_WAIT_TIMEOUT_MS` (120 s); the model default changed by `d1` |
| `t9` | delivered | `lobes/realtime/_vocalize.py` + hook in `tts_client.py`; timeout and event-loop defects fixed in review |
| `t10` | delivered | `Dockerfile.whisper-stt`, `listen_server_whisper.py`; upload cap added in review |
| `t11` | delivered | `chatterbox_multilingual_server.py`, `Dockerfile.chatterbox-ml` (the `d5` stand-in, kept as the shipped default) |
| `t12` | delivered | `lobes/realtime/app.py` wiring; cannot be executed offline — exercised live on the Spark only |
| `t13` | delivered | `_declared_audio_engine`; the `language` half landed later via `d3` (`lobes.roles.role_payload`) |
| `t14` | delivered | `scripts/realtime-he-accept.py` (+ five fixes found by using it live) |
| `t15` | delivered | `lobes init --fleet --audio --audio-lang he`, compose chain, doctor heal, `tests/goldens/overlays/audio-he-defaults.env` |
| `t16` | delivered | `docs/hebrew-realtime.md`, `docs/contracts/`, realtime-pipeline boundaries, explain catalog, CLAUDE.md, site harness, version 0.81.0 |
| `t17` | blocked | NOT run. Needs the PR's TestPyPI dev wheel deployed and the operator speaking; the live box still runs a hand-carried local wheel |

## Mid-work Decisions

Approved deviations, quoted from the delivery store:

- `d1` — The Hebrew voice lane is Gemma 4 senses (12B, model=multimodal) hosted LOCALLY on the Spark, replacing associate on the Orin from the start; 26B-A4B MoE and 31B stay later candidates. t8's Hebrew default becomes `OPENAI_MODEL`=multimodal (the bridge's existing default), t17 runs acceptance against senses, and the live proof brings the senses lane up on the Spark after ComfyUI is dropped. Nothing crosses the box during a voice turn any more — Operator decision 2026-09-18: associate's model card omits Hebrew and its Hebrew was only agent-judged on n=5 probes (lapse l2); Gemma 4 is a supported multilingual lane, the Spark is empty once ComfyUI is dropped and its profile already tunes senses at util 0.14, and hosting locally removes the per-turn Spark->Orin hop (measured 1.6-2.0 s). Known cost: Gemma tool calling is validated on the 31B only — the 12B's gemma4 parser pair is UNVALIDATED (#108), so t17's tool round-trip is its first real test
- `d2` — The live proof measures the voice lane in order: Gemma 4 12B senses first (the d1 lane), then larger Gemma 4 checkpoints on the Spark — the 26B-A4B MoE first, the 31B after — each on decode tok/s, a Hebrew tool-call probe and the operator's verdict on its Hebrew; the lane moves up only if a larger model wins — Operator decision 2026-09-18: start with the 12B because it is already tuned for this box, then measure higher. The 26B-A4B has no catalog entry and no vetted NVFP4 export here, so it is a new-checkpoint bring-up inside the live proof, not a knob flip
- `d3` — h25 is relaxed from 'the branch touches nothing under lobes/gateway/' to 'additive gateway changes only — the /v1/realtime tunnel and request routing stay untouched'. The capabilities advert gains a NULLABLE language field on audio roles (null when undeclared, so the no-declaration advert is no longer byte-identical: it gains "language": null), and the fleet compose gateway service passes `STT_MODEL`/`STT_RUNTIME`/`STT_LANGUAGE` and the TTS equivalents through — t13 proved plan risk r4: both advert serialisers use dataclasses.asdict (gateway/server.py:4224, cli/`_commands`/capabilities.py:260) so a conditional key is impossible without touching the gateway, and the gateway container never receives the new env keys, leaving live GET /capabilities advertising Parakeet on a Hebrew box. Operator chose an honest advert (c27) over a frozen gateway (h25) and accepted a nullable field, 2026-09-18
- `d4` — After the current work is done, migrate the Hebrew text-preparation stage from phonikud to conikud (<https://huggingface.co/conikud/conikud-onnx>, MIT, `conikud_int8`.onnx 669 MB, pip install git+<https://github.com/conikud/conikud-onnx>, API G2P().phonemize(text) / .alternatives(text,k)) and test it on this box. It joins the TTS A/B as an arm first; it replaces phonikud only where the A/B shows it works — Operator direction 2026-09-18 (/deviate): conikud replaces phonikud. Read before building: conikud is a G2P that outputs IPA WITH STRESS MARKS from unvocalized Hebrew (e.g. 'קניתי ספר חדש' -> 'kanˈiti sˈefeʁ χadˈaʃ'); it does NOT output niqqud-vocalized Hebrew text the way phonikud's `add_diacritics` does. So it is a drop-in only for a PHONEME-input TTS (the phonikud-tts / StyleTTS2 / Piper family), not for a TEXT-input engine such as Chatterbox Multilingual, which would read IPA as Latin letters. The merged t9 hook (`vocalize_hebrew` with an injected str->str callable over Hebrew spans, `PHONIKUD_MODEL_PATH`, the 'phonikud' timing key) stays structurally valid — a conikud factory is one more injected callable — but whether it is USED depends on which TTS engine t3 selects. Published accuracy: 9.97% WER internal, 16.56% WER on MILIM-Bench; int8 is the CPU default (no aarch64 onnxruntime-gpu wheel anyway)
- `d5` — The working loop comes before the TTS A/B: t11 builds a Chatterbox MULTILINGUAL (`language_id`=he) sidecar first as the stand-in Hebrew voice, the stack is brought up live on the Spark, and the TTS comparison then happens INSIDE the loop — engines swapped behind the same /v1/audio/synthesize contract and scored by an objective round trip (device A speaks a known sentence -> device B's microphone -> Whisper -> WER) plus the operator's ear. The two USB audio sets (reSpeaker XVF3800, Reachy Mini) are also run as two sessions talking to each other to soak turn-taking and barge-in. The conikud arm (d4) joins once a phoneme-input engine is wired — Operator decision 2026-09-18 ('working loop is great ... have the 2 respeaker sets talk to each other'). d4 already narrowed the A/B to one question (text-input Chatterbox vs conikud -> phoneme-input engine), a live loop makes every later comparison easier to judge, and lobes-chatterbox:latest already ships chatterbox-tts 0.1.7 with ChatterboxMultilingualTTS and 'he' among its 23 languages — no new engine is needed for a stand-in
- `d6` — Mixed-language transcription and speech (Latin words, paths, identifiers inside Hebrew — e.g. 'evidence' heard as 'Evidense', or a Latin word derailing the Hebrew voice) is NOT a concern for this product and is dropped from the acceptance bar: no mixed-script row is required in the TTS comparison or the acceptance transcript. The runaway guard stays as the generic protection. — Operator decision 2026-09-18: 'Don't worry about issues like Evidense - it won't be an issue for us.' The intended use is Hebrew speech; tool arguments will not depend on spoken Latin identifiers
- `d7` — Sentence-level STREAMING of the reply moves INTO this plan: the generate call streams (SSE), a pure sentence chunker releases each completed sentence to TTS while generation continues, and audio deltas start as soon as the first sentence is synthesized. Tool calls are assembled from streamed `tool_calls` deltas and never spoken. Barge-in cancels the stream, pending syntheses and undelivered audio. A non-streaming fallback stays for backends that cannot stream — Operator decision 2026-09-18 ('We need realtime stream'). Measured live: first audio 2.6 s for a one-sentence reply and 7.3 s for a longer one, of which whole-reply TTS was 1.9-5.9 s; the engine synthesizes ~1.5x faster than real time, so per-sentence pipelining keeps first audio near ~2 s independent of reply length. This was plan risk r6 (parked as a follow-up); the operator pulled it forward
- `d8` — The Hebrew voice is BlueTTS (github.com/maxmelichov/BlueTTS, Supertonic architecture, ONNX on CPU, code MIT, RenikudPlus G2P built in, plain Hebrew text in) — replacing Chatterbox Multilingual as the voice-lane TTS and superseding the conikud migration (d4): with BlueTTS no separate diacritizer/G2P stage is needed. Chatterbox Multilingual + phonikud stays in-tree as the alternative engine — Operator proposal and verdict 2026-09-18 ('What about using bluetts.com?' ... 'Sounds perfect!'), after asking for ~200-400 ms first audio. Measured on the GB10 CPU: 160-325 ms per sentence (RTF ~0.08-0.10) vs Chatterbox's 1.7-1.9 s, 0% round-trip WER vs 8.8%, no run-on syntheses. It also removes the non-commercial phonikud-tts question (q4). Open: the Hugging Face weights repo declares no licence; output is 44.1 kHz and needs resampling to the 24 kHz wire
- `d9` — Speculative ('spendy') turn-taking joins the plan, built after sentence streaming (d7) lands: (A) HIDDEN SPECULATION — at a short provisional silence (`VAD_EAGER_MS` ~250 ms) the server snapshots the turn audio and runs STT -> streamed generate -> first-clause TTS with NOTHING emitted and NO history written; resumed speech cancels it without trace, a confirmed end of turn promotes it and releases the ready audio at once. (B) EARLY COMMIT + CONTINUATION MERGE — the confirming silence drops to ~350-400 ms; when the user resumes shortly after a commit, the existing barge-in stops the reply, the aborted response is dropped from history, and the previous + new audio are re-transcribed as ONE turn. Both knobs are env-tunable and default OFF for the English overlay (byte-identical behaviour) — Operator decision 2026-09-18: 'Do we still wait that 600ms-1000ms? We can short it by being spendy and cancelling if not relevant.' Target first audio 200-500 ms. Measured: the end-of-turn silence (1000 ms on this box after 600 ms split a deliberate sentence) is the largest single term of perceived latency; STT 150-215 ms, first clause ~250-350 ms, BlueTTS 160-325 ms all fit inside that wait. Wasted GPU work is free on a single-user local box. c24/h17 ('no change to `_segmenter.py`') is knowingly relaxed: the segmenter gains a provisional-silence event
- `d10` — Input-level gate `VAD_MIN_LEVEL_PCT`: speech whose held peak level is below a configurable % of full scale is ignored (opens no turn, interrupts no reply). Default off; 4 on the Spark. — operator request during live testing, 2026-09-18: 'We need an ignore on low voice, like 3-5%. Even 10% or make it configurable' — Silero cannot tell a far voice from a near one

Decisions no deviation record covers:

- `tool_wait` defaults to 120 s, not the 60 s of the model-run stages — it waits on the client's own tool; defended in review (Qodo thread on `_floor.py`).
- The one-tool-call-per-step contract was made explicit: requests with tools send `parallel_tool_calls: false`, and surplus calls are logged by name (review finding).
- Re-running `lobes init --audio` now appends only ABSENT env keys — a defect that pre-dates this run in the English lane, fixed for both (review finding).
- `docs/contracts/` was created as the home for client-facing wire contracts; the remaining ones are issue #278.
- The continuation merge was limited to one take-back per utterance after a real browser session re-answered the same sentence three times.
- BlueTTS ships wired into NO compose file and names no weights repo — its weights declare no licence; the operator is asking the author.
- The English overlay's files were kept byte-identical, so `VAD_MIN_LEVEL_PCT` and the other new knobs are passed through the Hebrew overlay only.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t8` (`d1`) | The Hebrew voice lane is Gemma 4 senses (`model=multimodal`) hosted LOCALLY on the Spark, replacing associate on the Orin | risky |
| `t3` (`d2`) | The live proof measured the 12B first, then the 26B-A4B, which was promoted; the 31B was dropped | acceptable |
| `t13` (`d3`) | h25 relaxed to additive gateway changes; the advert gained a nullable `language` | acceptable |
| `t9` (`d4`) | conikud migration — NEVER DONE; superseded by `d8` (BlueTTS has its own G2P) | needs-follow-up |
| `t11` (`d5`) | Working loop before the TTS A/B, with Chatterbox Multilingual as the stand-in voice | acceptable |
| `t3` (`d6`) | Mixed-language speech is outside the acceptance bar (now issue #277) | acceptable |
| `t12` (`d7`) | Sentence-level streaming pulled into the plan | acceptable |
| `t11` (`d8`) | BlueTTS is the Hebrew voice on the Spark; not the shipped default (licence) | acceptable |
| `t6` (`d9`) | Speculative turn-taking (hidden speculation + continuation merge) | risky |
| `t8` (`d10`) | Input-level gate `VAD_MIN_LEVEL_PCT` | acceptable |
| `t17` | The acceptance transcript was not produced; its content exists only scattered across several live sessions from a non-released wheel — no record covers this | needs-follow-up |
| `t16` | The plan said docs mark every new statement DECLARED/UNVALIDATED "until t17's transcript lands"; t17 did not land, so they still say so — consistent, but the flip to VALIDATED is owed | needs-follow-up |
| `t12` | `GENERATE_STREAM` defaults ON for every language: an English default-behaviour change no live English session has exercised — no record classifies this beyond `d7` | risky |

## Evidence

- tests: `uv run pytest -n auto` — **6043 passed, 16 skipped** at `576b7e5`; per-obligation subsets re-run for `/validate-delivery` at `9455942`-era head (all green, counts in records `e1`–`e14`, `e16`)
- tests (httpx-gated, run with httpx): `tests/test_tts_client_hebrew_diacritizer.py`, `tests/test_tts_pause_and_truncation.py` — 23 passed
- site: `vitest` 249/249, `astro build` clean
- lint: black / isort / flake8 / bandit clean; `afi cli doctor . --strict` exit 0; `scripts/scan_deployment_secrets.py` clean; live secret values searched for in the branch: 0 hits
- SonarCloud: quality gate **OK** on `9455942` (reliability 3 → 1, new coverage 84.0 %); 2 late minor issues fixed in `576b7e5`, not yet re-analysed
- CI on PR #279: lint, test, site-build, secrets-scan, version-check, test-publish, GitGuardian — pass
- live: `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt` (sections 1–10) and the six other `docs/evidence/2026-09-hebrew-*.txt` files — DGX Spark GB10, 2026-09-18, hand-carried wheel
- devague records: evidence `e1`–`e17` and deltas `b1`–`b8` — filed `llm`-origin, **approved by the operator on 2026-09-18** ("Confirm all"); `e17` is approved AS A FAILING record
- commits: `main..2cc0da1` (97 commits) · PR: #279 (open) · issues: #276, #277, #278 · consumer note: agentculture/shabbos-goy#1

## Delivery Claims

Approved lapses `l1` (grader-unverified: a scope verdict taken from a subagent without reading the file) and `l2` (n-below-claim: Hebrew quality judged on n=5 by the agent) cap the claims they touch. `l3` and `l4` (both control-absent: tests written without the red step being observed — the sentence chunker, and the tool-call hold) were approved by the operator on 2026-09-18 and cap what rests on those tests: the chunker's behaviour is claimed from the live sessions, not from its unit tests alone, and the tool hold stays at `low` with the merge it belongs to.

| Claim | Confidence | Evidence |
|-------|------------|----------|
| Tool calls are relayed over `/v1/realtime` with OpenAI Realtime event names, and lobes executes none | high | `e3`, `e4`, `e5` (proposed) · `tests/test_realtime_conversation.py` · 7 live tool turns, `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt` s.9 |
| An unarmed / tool-less session is unchanged | medium | `e1` — offline only; the route shell is never imported offline |
| The Hebrew overlay is opt-in and every pre-existing golden is byte-identical | high | `e13` · `git diff main -- tests/goldens` (additions only) · real `docker compose config` of the merged chain |
| `stt`/`tts` adverts name the declared engine and language; nothing declared = identical to `main` | medium | `e12` · byte comparison against `main`'s code (4918 bytes equal) — NOT observed on the live gateway (image still 0.79.0) |
| Hebrew STT is accurate enough to converse, with non-speech hallucinations gated | medium | `docs/evidence/2026-09-hebrew-stt-spike-spark.txt` (4/4 noise dropped, 22/22 speech kept); capped by `l2`-style judgement: live slips observed ('שומע'→'שונה') |
| First audio went from 2.6 s to 1–113 ms after the commit | high (for one box, one speaker) | `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt` s.2, s.4, s.7 — live, human, reSpeaker; n = 7 turns |
| Human barge-in stops the reply | medium | `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt` s.4, s.7, s.9 — n = 6 live interruptions, one device |
| Hidden speculation is traceless when discarded and adopted only on a byte-identical request | medium | `tests/test_realtime_speculation.py` · `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt` s.6–7; the discard count in the live session was not separated from scripted runs |
| The continuation merge re-joins a paused sentence, and a tool call is held while it still can | low | `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt` s.8 — a doctored RECORDING only; zero live merges observed (s.9); capped by approved lapse `l4` (the hold's tests never seen red) — the 499–500 ms hold itself WAS observed live on 7 tool turns |
| The input-level gate ignores quiet voices | low | `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt` s.10 — a scaled recording stands in for a far voice; no real background speaker tried |
| BlueTTS serves Hebrew at 93–305 ms per sentence behind the existing contract | high | `docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt` s.1 · `tests/test_bluetts_server.py` · operator: "Sounds perfect!" |
| The web harness drives the session in a real browser | medium | driven in Chrome on the Spark (connect, arm, tools declared, transcript, reply, latency table); the stutter fix is NOT confirmed by ear and the browser tool-call path is untested |
| `lobes init --fleet --audio --audio-lang he` scaffolds a bootable Hebrew deployment | unverified | files and merged config verified; NO box has been brought up from this path (#108) |
| Nothing leaves the Spark during a Hebrew session | unverified | inferred from the wiring after `d1`; no traffic capture — obligation `o15` has no evidence record |
| A restored backup returns the box to its prior state | unverified | the backup exists (`e15`); the restore was never exercised |
| The acceptance transcript exists | unverified — **failing** | `e17` outcome `fail`: `docs/evidence/2026-09-accept-hebrew-realtime-spark.txt` does not exist |

## Remaining Work / Follow-up

- `t17` — deploy the PR's TestPyPI dev wheel on the Spark (update the override's `LOCAL_WHEEL`/pin; wait out the TestPyPI propagation race), run ONE session with the operator speaking: Hebrew in/out, a tool round trip, a barge-in, an STT negative control, the latency table. Then flip the docs from DECLARED/UNVALIDATED only for what it shows. Owner: operator + agent.
- PR #279 — awaiting the human merge decision; SonarCloud to re-analyse `576b7e5`.
- `d4` (conikud) — decide whether it is withdrawn now that BlueTTS owns G2P, or still wanted for the Chatterbox path.
- `o15` — capture traffic during a session if the "nothing leaves the box" claim matters.
- Continuation merge and level gate — prove each with a live voice (a real mid-sentence pause; a real TV/background speaker).
- A noise onset with a blank transcript still counts as a barge-in (design question, unaddressed). A merged turn emits two transcript events with no supersession marker.
- English deployments: `GENERATE_STREAM` on by default is unexercised live; the new knobs are not passed through the English overlay.
- The live gateway image is still 0.79.0 (stale advert); `TTS_DEBUG_TEXT=1` is still on in `~/.lobes/.env`.
- BlueTTS weights licence (operator is asking the author); PocketTTS Hebrew A/B if a checkpoint becomes available.
- Issues: #276 (26B on the Orin), #277 (mixed Hebrew/English), #278 (the remaining wire contracts).
