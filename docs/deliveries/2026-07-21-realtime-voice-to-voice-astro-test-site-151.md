# Delivery Summary — realtime voice-to-voice + astro test site (#151)

plan: `realtime-voice-to-voice-astro-test-site-151` · run: `partial` · date: `2026-07-21`
baseline: `devague summary skeleton`

## Intent

Lift the #149 non-goal: make `/v1/realtime` a surface you can *talk to*. A committed
turn optionally triggers a server-side generate + TTS reply streamed back over the
**same** WebSocket, interruptible by barge-in — with ears-only preserved as the
default so reachy-mini-cli is not forced into a conversation surface it does not
want. Alongside it, a local-only Astro harness so the surface can be experienced in
a browser (real mic, live event stream, audio out) instead of inferred from terminal
prints. Executed as 19 tasks fanned out across four dependency waves by
`/assign-to-workforce`, each in an isolated worktree behind a TDD merge gate.

## Planned Work

Quoted verbatim from the `devague summary` skeleton:

- `t1` — Wire codec module: lobes/realtime/_wire.py + tests — base64 event codecs for input_audio_buffer.append (parse) and response.audio.delta (serialize), delta chunk sizing, malformed-input rejection; stdlib-only
- `t2` — Floor/turn state machine: lobes/realtime/_floor.py + tests — pure state class (listening/transcribing/responding/speaking), injected callbacks + clock, cancel-both on interrupt, truncation of undelivered chunks, per-stage deadlines
- `t3` — Session schema + history: extend lobes/realtime/_session.py + tests — new EventType/ErrorCode/SessionState members (response lifecycle, interruption, generate/tts/timeout errors), per-session in-memory history + system prompt, teardown drops all
- `t4` — Ears-only wire migration: app.py input path parses base64 append events via _wire (binary path removed); the #149 offline event-sequence tests are UPDATED to the event format, not deleted
- `t5` — Turn request shaping: lobes/realtime/_turn.py + tests — stdlib builder for the chat/completions payload (history, env model default multimodal, enable_thinking=false), response parsing, 404 role_infeasible mapped to the named error with hosted_by
- `t6` — Conversation route wiring: app.py response.create trigger wires `_floor` + `_turn` + `_wire` + tts_client (cancel_event armed), per-stage timeouts from settings, delta emission; TTS-out path calls no resampler; gateway diff zero lines
- `t7` — TTS gate re-scope: tts_client.py/_settings.py — per-lane semaphores (voice lane vs batch) or a raised default, chosen and documented with rationale
- `t8` — Deployment passthrough + doctor heal: docker-compose.audio.yml environment + env.audio.example gain BARGE_IN_WINDOW_MS/BARGE_IN_MODEL, system-prompt and every new voice-turn key; doctor heal list + tests extended
- `t9` — In-repo client migration: scripts/realtime-smoke.py + scripts/realtime-voice-loop.py speak the base64 event wire (append in, delta out); duplex rules (pong pings, select reads, lock-guarded writes) carry over; helper tests updated
- `t10` — Site scaffold + design system: site/ Astro 7 project — org tokens/fonts/keyframes/data-reveal ported from ../org/site-astro global.css, layout + shell, reduced-motion kill switch, AA pairs both themes, fonts self-hosted
- `t11` — Mic + playback island: site client JS — getUserMedia({echoCancellation:true}) + AudioWorklet capture to 24 kHz PCM16 base64 append events; delta player; playback stops on interruption event; start-control gesture gating; permission-denied/not-found as distinct states
- `t12` — Event stream UI: site components rendering the live event log — every event type with timestamps, VAD boundary timing visibility for knob tuning, each named error code visually distinct from silence and from disconnect
- `t13` — Local proxy + dev flow: Vite server.proxy ws upgrade-header injection (standalone tiny proxy as fallback), site README documenting ssh -L as the primary secure-context flow and the local-only stance
- `t14` — Site CI job: tests.yml gains a site-build job (npm ci && npm run build, Node 22); sonar.sources, version-check path filters, and hatch packaging untouched
- `t15` — Boundary guard sweep: assert the untouched surfaces — batch /v1/audio/* handlers, catalog.py, gateway do_GET dispatcher and tunnel, header-only auth tests, import-isolation tests — all pass unmodified; PR checklist records the zero-diff surfaces
- `t16` — Docs flip: realtime-pipeline.md (drop half-duplex IOU, document conversation opt-in + base64 wire + new events), openai-api.md endpoint table, gateway-fleet.md, explain catalog _REALTIME — same PR; before-state citations kept as history; doc-test-alignment run
- `t17` — Live acceptance + evidence: the Spark run through the site via ssh -L — speak, hear the reply, interrupt mid-playback; speakers-at-volume AEC test; one A/B audio exchange against the OpenAI realtime service (or record precisely why not); transcript under docs/evidence/ BEFORE any validated wording
- `t18` — User mute / mic-off control: an explicit operator affordance on the site (mute mic, mic off/release device), distinct from and never triggered by playback; the no-mic-mute gate narrows to forbidding AUTOMATIC mute-during-playback (deviation d1)
- `t19` — Site conversation arming + vocabulary sync: a conversation toggle that sends response.create (default OFF so ears-only stays the default), and sync the site error vocabulary with the servers new invalid_wire_event code

## Actual Delivery

18 of 19 tasks merged. `t17` is the sole undelivered task and is blocked on physical
hardware, not on code.

| Plan task | Status | What actually landed |
|-----------|--------|----------------------|
| `t1` | delivered | `lobes/realtime/_wire.py` + 31 tests; named `WireFormatError`/`WireErrorCode`; chunk size derived from `protocol.py`. Merge `1a0e918` |
| `t2` | delivered | `lobes/realtime/_floor.py` + 62 tests, 100 % module coverage; cancel-both barge-in, three per-stage deadlines, turn tokens. Merge `2acd37d` |
| `t3` | delivered | `_session.py` extended: 5 response events, 3 error codes, RESPONDING/SPEAKING states, per-session history + system prompt. Merge `3ef71d7` |
| `t4` | delivered | inbound path migrated to base64 append events; binary frames now a named `unsupported_frame_type` error, not a tolerated input. Merge `7ecb02c` |
| `t5` | delivered | `lobes/realtime/_turn.py` + 40 tests; deliberately policy-free on model; real gateway 404 shape used as the test fixture. Merge `13eaffe` |
| `t6` | delivered | `lobes/realtime/_conversation.py` (the convergence); four cross-task reconciliations; pump/watchdog asserted structurally. Merge `28da1bb` |
| `t7` | delivered | per-lane semaphores **and** per-lane httpx clients (preserving the reset-cannot-race invariant). Merge `54fe902` |
| `t8` | delivered | 5 env keys threaded end to end + an AST coverage test that fails CI on any future unwired `Settings` field. Merge `83dbd2f` |
| `t9` | delivered | both in-repo clients migrated; base64 codec inlined (not imported) to preserve the scripts' zero-dependency property, cross-checked against the real server codec. Merge `366f9cf` |
| `t10` | delivered | `site/` Astro 7 scaffold, org design system ported, `package.json` pinned complete. Merge `ab0fe2b` |
| `t11` | delivered | mic capture + playback island; muting made impossible by construction (throwing test double). Merge `9692a4e` |
| `t12` | delivered | live event log; every event type and error code distinct by icon + label + badge, never colour alone. Merge `35680db` |
| `t13` | delivered | Vite ws proxy with credential injection; `ssh -L` dev flow documented; plan risk `r3` settled empirically. Merge `7898667` |
| `t14` | delivered | `site-build` CI job (Node 22, npm cache); wheel verified unaffected. Merge `a03341e` |
| `t15` | delivered | boundary sweep: no violation found; batch handlers verified byte-for-byte; 2 new standing guards. Merge `af721e0` |
| `t16` | delivered | 5 doc surfaces flipped; status recorded as DECLARED/not-validated per #108. Merge `b9dc85d` |
| `t17` | **blocked** | **not delivered.** Requires a physical microphone, speakers at volume, and an operator at a browser. No transcript exists under `docs/evidence/` for this work |
| `t18` | delivered | user mute / mic-off; gate narrowed to forbid only *automatic* mute. Merge `cf8410c` |
| `t19` | delivered | conversation toggle (default OFF) + error-vocabulary drift guard reading `_session.py` off disk. Merge `ddf8ac9` |

## Mid-work Decisions

- `d1` — allow user-initiated mute and mic-off as an explicit control on the site and future clients; the blanket "no mic muting anywhere in site code" gate narrows to forbidding AUTOMATIC mute-during-playback only. Reason, from the record: real hardware now exists — the Reachy Mini mic cancels echo in firmware and audio plays out via Reachy or the HDMI monitor. The blanket ban was protecting against muting used as an AEC *substitute* (the f1e6ffa half-duplex hack that makes barge-in impossible). With AEC owned at the client edge, a user-initiated mute is an orthogonal privacy affordance that does not reintroduce that failure mode.
- **`t19` did not exist in the confirmed plan and was added mid-run.** `t11`/`t12`/`t13` covered mic, events and connection, but no task covered *sending* `response.create` — so the headline acceptance criterion ("a spoken turn produces a spoken reply") could not have been demonstrated live. This was a plan-authoring gap, not a departure from a confirmed decision, so it was added as a task rather than recorded as a deviation. Commit `0482910`.
- **Chunk size reconciled to one owner.** `_wire` (100 ms) and `_floor` (40 ms) shipped disagreeing defaults. `t6` deleted `_floor`'s and made `chunk_bytes` a **required** constructor argument, so a second default cannot drift back. The value stays revisitable at the live run; the duplication does not.
- **Three site islands, built blind to each other, invented three different DOM event vocabularies.** Rather than edit three merged, independently-tested islands to agree after the fact, one bridge in `index.astro` translates. Each island keeps the API its own tests pin. Commit `73427b4`.
- **`index.astro` was reserved for coordinator wiring, and `vitest.config.ts` was committed up front** (`f760c7d`) — both to stop three parallel wave-2 agents colliding on shared files the dependency graph does not protect.
- **The Qodo `OPENAI_MODEL` alias-validation finding was rejected**, with reasoning posted to the PR: it is operator env config rather than a caller parameter, the default already *is* an alias, validating it would make the bridge stricter than the gateway it calls, and a wrong value already surfaces as a named `role_infeasible` error rather than a silent fallback.

## Drift From Plan

| Plan item | Reason for divergence | Classification |
|-----------|-----------------------|----------------|
| `t11` (`d1`) | real hardware now exists (Reachy firmware AEC, Reachy/HDMI playout), so muting is no longer an AEC substitute; a user-initiated mute is an orthogonal control affordance and does not reintroduce the half-duplex failure mode barge-in cannot coexist with | acceptable |
| `t17` | not executed — needs a physical microphone, speakers at volume, and an operator at a browser. Additionally blocked twice over: the deployed bridge is 0.52.3 with no `_wire`/`_conversation` (it cannot speak this wire), and this box runs the `spark-lobe` shape with `MULTIMODAL_FEASIBLE=false`, so the default voice lane would 404 `role_infeasible` | needs-follow-up |
| `t2` / `t3` | the two tasks minted different failure vocabularies (`_floor` distinguishes three timeout reasons; `_session` collapses them into one `response_timeout`). `t6` wrote the mapping table and requires the stage be named in the message text | acceptable |
| `t4` | put a `WireErrorCode` into `ErrorEvent.code`, a field documented as always carrying a `_session.ErrorCode`, because `_session.py` was outside `t4`'s file scope. `t6` collapsed the two enums by adding `INVALID_WIRE_EVENT` | acceptable |

## Evidence

- tests (Python): `uv run pytest -n auto -q` — **2577 passed, 14 skipped** (skips are live-deployment-gated)
- tests (site): `npx vitest run` in `site/` — **179 passed** across 9 files
- build: `npm run build` in `site/` — pass (4 pages); `npm run check` — 0 errors / 0 warnings / 0 hints
- lint: `black --check`, `isort --check-only`, `flake8`, `bandit -c pyproject.toml -r lobes` — all clean
- rubric: `uv run afi cli doctor . --strict` — exit 0
- markdown: `markdownlint-cli2 "**/*.md" "#node_modules" "#.local"` — 0 errors (59 files)
- zero-diff boundaries: `git diff main -- lobes/gateway/` — **0 lines**; `git diff main -- lobes/catalog.py` — **0 lines**
- commits: `f1e6ffa..363bad5` (52 commits, 18 task merges)
- PRs / issues: lobes-cli `#153`; issue `#151`; reachy-mini-cli `#115` (the coordinated wire-break migration)
- spec / plan: `docs/specs/2026-07-21-realtime-voice-to-voice-astro-test-site-151.md`, `docs/plans/2026-07-21-realtime-voice-to-voice-astro-test-site-151.md`
- deviation record: `.devague/deliveries/realtime-voice-to-voice-astro-test-site-151.json` (`d1`, approved, `acceptable`)

## Delivery Claims

| Claim | Confidence | Evidence |
|-------|------------|----------|
| The conversation surface is implemented and offline-proven end to end | high | file `lobes/realtime/_conversation.py` · merge `28da1bb` · `uv run pytest -n auto -q` (2577 passed) |
| A session that never arms is byte-identical to the #149 ears-only contract | high | `tests/test_realtime_conversation.py::test_a_session_that_never_opts_in_emits_the_transcription_only_sequence` — structural property of `_conversation.py` (every floor call behind `if self.armed`) |
| Barge-in cancels both generate and TTS and truncates undelivered audio | high | file `lobes/realtime/_floor.py` · merge `2acd37d` · 62 offline tests incl. interrupt-from-every-state |
| The gateway is untouched by this work | high | `git diff main -- lobes/gateway/` — 0 lines · merge `af721e0` |
| The batch `/v1/audio/*` routes are unchanged | high | handler bodies compared byte-for-byte: 1903/1903 and 1094/1094 identical (`t15`) |
| The Astro harness builds and is local-only | high | `npm run build` pass · no deploy workflow under `.github/workflows` · merge `ab0fe2b` |
| The site drives a full conversation against a live fleet | **unverified** | `t17` not run — no live transcript exists |
| Browser `echoCancellation` keeps the speakers out of the mic at volume | **unverified** | asserted as a `getUserMedia` constraint; its *effect* is unmeasured. This is the risk `barge_in_model` is parked against |
| Barge-in interrupts within `barge_in_window_ms` as a latency bound | **unverified** | the guard reading shipped; which of the two readings holds live is exactly what `t17` must record |
| VAD `at_ms` timing is observable live | medium | `at_ms`/`reason` now reach the wire (`t6`); rendering verified in fixture replay only, never against a live session |
| Concurrent sessions, the VAD-unavailable path, and the max-turn cap | **unverified** | carried forward from #149; **not** retired by this work |

## Remaining Work / Follow-up

- **`t17` — the live acceptance run.** Blocking for any "validated" wording anywhere (#108). Needs, in order: (1) rebuild the realtime bridge from this branch — the deployed container is **0.52.3** and has neither `_wire` nor `_conversation`, so it cannot speak this wire (`Dockerfile.realtime` installs `lobes-cli==${VERSION}` from PyPI, and 0.54.0 is unpublished; the PR's TestPyPI `.dev` build or a local wheel are the two routes); (2) set `OPENAI_MODEL` to a lane this box hosts — `MULTIMODAL_FEASIBLE=false` under the `spark-lobe` shape, so the default voice lane would 404 `role_infeasible`; (3) drive the run from a laptop browser over `ssh -L`, including the speakers-at-volume AEC test and an A/B exchange against the OpenAI realtime service.
- **reachy-mini-cli must adapt to the base64 wire** — filed as reachy-mini-cli#115. Until it lands, the deployed robot cannot stream. Operator-accepted coordinated break (`c40`), but the window should be short.
- **`barge_in_model` is threaded but unconsumed** — window-only barge-in ships. It is the recorded mitigation if the speakers-at-volume AEC test disappoints.
- **`barge_in_window_ms` carries two readings** in our own documents — a guard window (shipped) versus a latency bound (honesty condition `h6`). `t17`'s transcript must record which it demonstrates.
- **`ws` / `@types/ws` are now dead dependencies** in `site/package.json` — the contingency for a risk (`r3`) that did not materialise, retained only because the file was pinned during the parallel wave. Safe to remove.
- **Full OpenAI Realtime parity is a named non-goal** (frame park `v5`): this work adopts the audio-path event shapes only.
- **Gateway realtime hardening** — per-client limits, credential isolation, timeouts, resource budgets — parked to a sibling issue (`v2`), with #149's session-cap park folding into it.
