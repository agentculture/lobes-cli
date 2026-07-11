# Build Plan — lobes gateway relays SSE streams frame-by-frame: the first token through the gateway arrives at backend speed, not at turn end

slug: `lobes-gateway-relays-sse-streams-frame-by-frame-th` · status: `exported` · from frame: `lobes-gateway-relays-sse-streams-frame-by-frame-th`

> lobes gateway relays SSE streams frame-by-frame: the first token through the gateway arrives at backend speed, not at turn end

## Tasks

### t1 — Fix: _Upstream.read delegates to HTTPResponse.read1 (frame-by-frame upstream reads)

- instruction: Edit lobes/gateway/server.py _Upstream.read (line ~205): return self._resp.read1(n). Do not touch _relay_streaming, _relay_buffered, open_upstream, or the audio path. Run uv run pytest -n auto to prove the old suite green.
- covers: c1, c4, h7, c8, h2
- acceptance:
  - lobes/gateway/server.py _Upstream.read returns self._resp.read1(n) instead of self._resp.read(n); the duck-type method name read() is unchanged
  - _relay_streaming (per-chunk frame_chunk + flush) and _relay_buffered/read_all are byte-identical to before — the diff inside lobes/ is confined to _Upstream.read
  - the entire existing test suite passes unmodified (uv run pytest -n auto)

### t2 — Regression test: real-socket dribble upstream proves frame-by-frame relay timing

- instruction: Add test to tests/test_gateway_server.py: fixture starts a real ThreadingHTTPServer whose do_POST dribbles ~5 chunked SSE frames at ~100 ms gaps (write chunked framing + flush + sleep); point the gateway fixture's backend base_url at it (real open_upstream, no monkeypatched fake); raw-socket client records arrival time of each relayed chunk; assert first-arrival < half the dribble span. Dev-verify it fails with read1 reverted to read.
- covers: c3, h1, c5, h8, c9, h3, c7
- acceptance:
  - new test in tests/test_gateway_server.py: a REAL ThreadingHTTPServer upstream dribbles ~5 chunked SSE frames at ~100 ms gaps, wired through the real open_upstream and the real handler relay (no _FakeUpstream)
  - the test asserts the client receives the first relayed frame while the upstream is still mid-stream (first-arrival < half the dribble span), not just chunked framing
  - verified during development: the new test FAILS with _Upstream.read reverted to self._resp.read(n) while the rest of the suite stays green (proving the fake-upstream tests cannot catch this bug class)

### t3 — Release hygiene: version bump, CHANGELOG + PR citing #103 and colleague#318, boundary audit

- instruction: python3 .claude/skills/version-bump/scripts/bump.py patch (pipe changelog JSON on stdin per memory note; commit the uv.lock re-pin). CHANGELOG cites #103 + agentculture/colleague#318 and the frames=21 first=last=3.06s signature. Audit git diff main: only server.py one-liner + tests + docs/specs + docs/plans + CHANGELOG + pyproject/uv.lock.
- depends on: t1, t2
- covers: c6, h9, c10, h11
- acceptance:
  - patch version bump via .claude/skills/version-bump/scripts/bump.py; CHANGELOG entry cites issue #103 and agentculture/colleague#318 and quotes the buffered-burst signature (frames=21 first=last)
  - PR diff audited against the boundary: only lobes/gateway/server.py (_Upstream.read) + tests + docs/CHANGELOG/pyproject; no new dependencies in pyproject.toml; no FastAPI/httpx/asyncio; audio relay contract untouched

### t4 — Deploy + live acceptance: rig curl probe first<<last, comment on #103 for colleague re-probe

- instruction: On the rig: re-scaffold/re-pin the gateway container to the released lobes-cli (container pins at lobes init time, does not auto-update — #99), docker compose up -d --build gateway; re-run the issue #103 curl probe verbatim against localhost:8001; post probe numbers to #103 asking colleague to re-probe, signed '- lobes (Claude)'.
- depends on: t3
- covers: h5, c2, h6, c11, h4, h10
- acceptance:
  - after merge + PyPI release, the rig gateway container is re-pinned/re-scaffolded to the fixed lobes-cli (the container does NOT auto-update — issue #99 skew) and restarted
  - the exact issue #103 curl probe re-run against localhost:8001 shows the first data: frame well below the last (the user-decided acceptance gate; client-agnostic — no SDK involved)
  - a comment is posted on #103 reporting the probe numbers and asking colleague to re-probe tests/test_vllm_live_streaming.py (expected SKIP to PASS), signed per convention

## Risks

- [unknown_nonblocking] cloudflared tunnel in front of gateway 8001 may add its own SSE buffering for remote clients — re-probe the tunnel origin after deploy; local curl probe is the acceptance gate (frame q1 resolution) (task t4)
- [follow_up] colleague livecheck flip depends on colleague re-running its probe after deploy — external follow-up, not gating this repo's merge (task t4)
