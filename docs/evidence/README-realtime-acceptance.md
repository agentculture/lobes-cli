# Acceptance-evidence procedure: `/v1/realtime` (issue #149)

This file documents how a live acceptance run of the `/v1/realtime`
WebSocket session surface becomes evidence in this repo — the same #108 rule
every other entry under `docs/evidence/` already follows (see e.g.
`2026-07-17-accept-muse-tool-calling-thor.txt`,
`2026-07-14-accept-thor-lobe-thor.txt`): **a transcript lands here BEFORE any
doc, README, or CLAUDE.md may describe the surface as "validated" or
"measured live."** The transcript is what makes the claim true, not the
other way around.

As of the PR that added this file (issue #149 task t8), the
`/v1/realtime` route (`lobes/realtime/app.py`), the session engine
(`lobes/realtime/_session.py`), the VAD segmenter
(`lobes/realtime/_segmenter.py`), and the gateway tunnel
(`lobes/gateway/_realtime.py`) have all been unit-tested offline, but the
whole stack has **never been exercised against real hardware**. Nothing in
this repository should claim otherwise until the procedure below has
actually been run and its output committed.

## The tool

`scripts/realtime-smoke.py` — a stdlib-only smoke test, modelled on the
existing `scripts/audio-smoke.py`. It opens ONE `/v1/realtime` WebSocket
through the fleet gateway, synthesizes a known phrase via
`/v1/audio/speech` (Chatterbox), streams that phrase's PCM16 audio into the
session in real-time-ish 32 ms chunks followed by trailing silence, and
asserts the full event sequence arrives on that SAME connection:
`session.created` → `input_audio_buffer.speech_started` →
`input_audio_buffer.speech_stopped` →
`conversation.item.input_audio_transcription.completed` with the known
phrase's keywords present in the transcript. It prints `PASS`/`FAIL` per
step and exits non-zero on any failure — see the script's own module
docstring for the full step-by-step contract, including how it distinguishes
an explicit `error`/`vad_unavailable` event from a bare timeout (never
treating "no events arrived" as success).

It deliberately does not depend on `websocket-client`, `websockets`, an
OpenAI SDK, numpy, or torch — see the script's docstring for why (the whole
point of issue #149 is to keep those OFF the `reachy-mini-cli` robot client
that consumes this surface; a smoke test with a heavyweight dependency would
undercut its own motivation).

## Procedure

1. **Bring up a real, audio-enabled fleet** on the target box:

   ```bash
   lobes init --fleet --audio --apply
   lobes fleet up --apply
   ```

   Confirm both STT (Parakeet) and TTS (Chatterbox) report ready — either
   `lobes status` or `GET /v1/health/ready` on the realtime bridge directly.

2. **Run the smoke script** against that deployment:

   ```bash
   python3 scripts/realtime-smoke.py --base-url http://localhost:8000
   ```

   Add `--api-key <key>` if the gateway has `GATEWAY_API_KEY` set. Add
   `--phrase` to override the known phrase (the default,
   `"The quick brown fox jumps over the lazy dog."`, uses a keyword list
   already proven to survive Parakeet's normalizations — a custom phrase may
   not).

3. **Capture the complete stdout** — every `PASS`/`FAIL` line plus the final
   `Results: n/m checks passed` summary — into a new file:

   ```text
   docs/evidence/<date>-accept-realtime-<box>.txt
   ```

   `<date>` is `YYYY-MM-DD`; `<box>` is the short hostname the run happened
   on (e.g. `spark`, `thor`), mirroring every other filename already under
   this directory. Include the command line invoked and the repo revision
   (`git rev-parse --short HEAD`) at the top of the transcript, the way the
   existing evidence files do.

4. **Commit that file.** Only once it exists in the repo may a doc describe
   `/v1/realtime` as validated or measured live — and that doc should cite
   the evidence file by path, the way `docs/gemma-4-31b-nvfp4.md` and
   CLAUDE.md's own "Colleague roles" section cite their evidence files
   today. A doc edit that claims validation without a corresponding file
   here is exactly the failure mode issue #108 exists to prevent.

## What a passing run does and does not prove

A passing run proves the full round trip works end-to-end on ONE connection,
on the box it was run on, with the configuration active at run time (the
VAD thresholds, the Parakeet/Chatterbox versions, the machine profile in
effect). It does **not** by itself validate:

- concurrent-session behavior (this script opens exactly one session);
- the `vad_max_turn_ms` force-commit path (the phrase used here is short —
  see `lobes/realtime/_segmenter.py`'s module docstring for that behavior);
- barge-in or any surface this script's default arguments don't exercise.

A failing run is equally informative — the honest `FAIL` lines (especially
the criterion-5 distinction between a named `error` event and a bare
timeout) are meant to localize exactly which stage broke, not just report a
binary yes/no.
