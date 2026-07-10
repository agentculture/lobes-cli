# lobes gateway relays SSE streams frame-by-frame: the first token through the gateway arrives at backend speed, not at turn end

> lobes gateway relays SSE streams frame-by-frame: the first token through the gateway arrives at backend speed, not at turn end
> instruction: one-line core change: _Upstream.read returns self._resp.read1(n) instead of self._resp.read(n) (lobes/gateway/server.py:205-206); everything else in the relay (per-chunk framing + flush) already streams correctly

## Audience

- streaming OpenAI clients behind the gateway: colleague's token-streaming lane (agentculture/colleague#318 vLLM SSE consumption + live cockpit generation tail), and any OpenAI SDK caller with stream=True
  - instruction: acceptance probe is the issue #103 curl command verbatim: time each data: frame's arrival; healthy = first well below last

## Before → After

- Before: the gateway's _Upstream.read(n) delegates to http.client.HTTPResponse.read(n), which blocks until n bytes (64 KiB, _CHUNK) accumulate or EOF; a whole SSE turn is a few KB, so every data: frame is released in one terminal burst at EOF — issue #103 probe: frames=21 first=3.06s last=3.06s through the gateway
  - instruction: repro script: dribble N small chunked SSE frames with sleeps from a BaseHTTPRequestHandler; time each read()/read1() return through http.client — buffered signature is 1 return at EOF
- Before: the relay loop itself (_relay_streaming, server.py) is already frame-correct — per-read chunked framing plus flush; only the upstream read blocks. The issue's suggested fix (FastAPI/httpx StreamingResponse) does not apply: the gateway is a stdlib ThreadingHTTPServer + http.client proxy
  - instruction: leave _relay_streaming (per-chunk frame_chunk + flush) untouched; the diff is confined to _Upstream.read
- Before: test gap: tests/test_gateway_server.py _FakeUpstream.read() returns one chunk per call — read1 semantics — so the suite green-lights the relay loop while the real HTTPResponse.read blocks until buffer-full/EOF; the fake models the fix, not the bug
  - instruction: keep _FakeUpstream as-is for routing/logic tests; the new dribble test (c9) is the timing guard — verify it fails on the reverted read before merging
- After: with stream:true through the gateway, the first data: frame reaches the client at backend time-to-first-token; the issue #103 curl probe shows first well below last, and no client changes anything
  - instruction: after merge + deploy: re-run the issue probe on the rig; the local curl probe is the acceptance gate (user decision on q1) — the tunnel path is a parked follow-up

## Why it matters

- time-to-first-token is the entire point of streaming; through the gateway it silently degrades to full-turn latency, colleague's feels-alive lane renders as one terminal burst, and its livecheck grades the rig SKIP (an intermediary buffers SSE)
  - instruction: cite issue #103 and agentculture/colleague#318 in the PR description and CHANGELOG entry

## Requirements

- the streaming relay reads the upstream with read1 semantics — HTTPResponse.read1(n), which returns as soon as any bytes are available — instead of the buffer-filling read(n); loop termination on empty bytes is unchanged
  - instruction: change _Upstream.read (lobes/gateway/server.py:205-206) to return self._resp.read1(n); keep the duck-type method name read() so _FakeUpstream and handle_post callers are untouched
  - honesty: HTTPResponse.read1(n) returns empty bytes only at EOF on a blocking socket — never mid-stream — so the existing 'if not chunk: break' termination is unchanged; the one behavioral delta is a TRUNCATED upstream: read() raises IncompleteRead where read1 ends the relay at the chunked terminator cleanly, which is acceptable (the client sees a terminated stream either way)
- an in-repo regression test that FAILS on read-until-full semantics: a real-socket upstream that dribbles SSE frames with delays, asserting the client receives the first relayed frame while the upstream is still mid-stream (arrival timing, not just chunked framing)
  - instruction: fixture: real ThreadingHTTPServer upstream dribbling ~5 chunked SSE frames at ~100 ms gaps, wired through the real open_upstream; client asserts the first relayed frame arrives before the upstream has written its last frame (e.g. first-arrival < half the dribble span)
  - honesty: the regression test uses a REAL http.client upstream (not _FakeUpstream, whose read() already has read1 semantics and cannot catch this bug) and FAILS when _Upstream.read is reverted to self._resp.read(n)

## Honesty conditions

- with the fix deployed, the exact curl probe from issue #103 run against the gateway shows the first data: frame arriving near backend time-to-first-token and well before the last frame — no client-side change involved
- the fix is client-agnostic: a plain curl SSE probe (no SDK, no colleague machinery) through the gateway observes incremental frame arrival, so ANY OpenAI-compatible streaming client benefits with zero client change
- a local reproduction (stdlib dribble server, 10 SSE frames at 100 ms intervals) shows HTTPResponse.read(65536) returning ONCE at EOF (first=last=1.0s — the exact issue signature) while read1(65536) returns each frame as it arrives (first=0.0s)
- the write side is proven non-buffering: the local repro's dribble server uses the same BaseHTTPRequestHandler + per-frame flush machinery the gateway relay uses, and its frames arrived on schedule at a read1 client — so fixing the upstream READ is sufficient, no relay-loop change needed
- reverting _Upstream.read to self._resp.read(n) keeps the ENTIRE existing suite green — demonstrating the fake-upstream tests cannot catch this bug class and the new real-socket timing test is the only guard
- the impact is externally attested: issue #103 quotes colleague's livecheck verdict ('stream delivered as one terminal burst — an intermediary (gateway proxy) buffers SSE; rig-side, not a colleague regression') and the 220-delta/32-34s probe signature
- shown twice: the in-repo regression test proves first-frame relay while the upstream is mid-stream (CI-durable), and the deployed-rig curl probe (h5) shows the same first-well-below-last signature live
- the diff surface confirms the boundary: only _Upstream.read's delegate changes plus new tests; no new dependencies in pyproject.toml; _relay_buffered/read_all byte-identical; the audio relay inherits earlier arrival but its contract (whole-file relay, chunked) is unchanged
- verified on the rig by re-running the exact issue #103 curl probe after deploy (frames=N first<<last), then confirmed by colleague's offered re-probe of tests/test_vllm_live_streaming.py

## Success signals

- the issue #103 curl probe through the gateway shows the first data: frame well before the last (healthy stream signature), and colleague's gated live proof tests/test_vllm_live_streaming.py flips SKIP to PASS with no client change
  - instruction: deploy the fixed gateway (lobes init scaffold re-pins lobes-cli== the released version — the gateway container does NOT auto-update, see issue #99 skew note), re-run the curl probe from the issue verbatim, then comment on #103 asking colleague to re-probe

## Scope / boundaries

- no client-side changes; the gateway stays a stdlib proxy (no FastAPI/httpx/asyncio rewrite); the buffered JSON path (read_all) is untouched; the audio relay keeps its contract (it shares _relay_streaming and simply inherits the earlier-arrival behavior)
  - instruction: review the PR diff against this list; anything beyond server.py's _Upstream.read + tests is out of scope

## Hard questions

- colleague's own probe may traverse the host cloudflared tunnel rather than localhost:8001 — if the tunnel adds its OWN SSE buffering, the gateway fix alone flips the local curl probe but not colleague's livecheck; is the local probe the acceptance gate, with the tunnel path a follow-up?
  - **resolved (user, 2026-07-11): the local curl probe is the acceptance gate.** The issue's evidence is the localhost:8001 probe, which isolates the gateway; tunnel behavior stays a parked non-blocking unknown, re-probed after deploy (see Accepted plan risks).

## Accepted plan risks

<!-- parked unknown_nonblocking items; the spec_md exporter drops these, appended by hand (frame JSON retains them) -->

- whether the remote path colleague dials (host cloudflared tunnel in front of gateway port 8001) adds its own SSE buffering on top of the gateway's — re-probe the tunnel origin after the gateway fix lands; cloudflared generally passes SSE through, and the issue's localhost curl probe already isolates the gateway as the local culprit

## Open / follow-up

- colleague re-probe of tests/test_vllm_live_streaming.py after the fixed gateway is deployed on the rig (colleague offered on the issue; needs the deploy first — the gateway container pins lobes-cli at scaffold time and does not auto-update)
