# Build Plan — Under pressure, lobes tells you the cortex model is busy and to retry shortly — it never silently degrades your request onto a weaker or different-capability model.

slug: `under-pressure-lobes-tells-you-the-cortex-model-is` · status: `exported` · from frame: `under-pressure-lobes-tells-you-the-cortex-model-is`

> Under pressure, lobes tells you the cortex model is busy and to retry shortly — it never silently degrades your request onto a weaker or different-capability model.

## Tasks

### t1 — Pressure policy: replace degrade-to-minor with a busy verdict (lobes/gateway/_pressure_policy.py)

- covers: c4
- acceptance:
  - decide() under swap>SWAP_DEGRADED_THRESHOLD or iowait>IOWAIT_DEGRADED_THRESHOLD returns mode=busy (shed) and NO downgraded served tier (no max_allowed_tier=minor)
  - decide() at or below both thresholds returns serve-as-requested (mode=warm) for main, senses and minor, unchanged from today
  - an explicit minor request is never marked busy (minor is the floor; served as requested)
  - the _DEGRADED_FLOOR degrade path is gone and there is NO LOBES_PRESSURE_POLICY toggle to restore it: no code path rewrites a request onto a cheaper tier under pressure

### t2 — Tier request: return a busy signal instead of substituting a model under pressure (lobes/gateway/_tier_request.py)

- depends on: t1
- covers: h7
- acceptance:
  - resolve_tier_request() under a busy verdict returns a busy marker (busy=True, reason=pressure) and NO served_name: it never resolves the degraded tier through routing
  - regression for issue #85: add a fleet fixture to tests/test_tier_request.py with primary+multimodal wired and minor UNWIRED, and assert a cortex request under swap=80 returns busy, never served_name=gemma
  - a senses/multimodal request under pressure also returns busy; an explicit minor request resolves to the minor served_name; an unpressured request resolves to its own served_name unchanged

### t3 — Gateway: emit HTTP 429 busy response on the shed path (lobes/gateway/server.py)

- depends on: t2
- covers: c15, c16, c10, c13, h2, h10
- acceptance:
  - when resolve_tier_request returns busy, handle_post responds 429 with a Retry-After header, an X-Lobes-Tier-Reason: busy header, and an OpenAI-shaped error object (error.message/error.type/error.code), parallel to the existing 502 _error_body path
  - on the shed path NO upstream backend is dialed: assert via a stub that records upstream calls that primary/multimodal/minor are never reached
  - the 429 busy is distinguishable from the existing 502 all-backends-unavailable by status code plus header alone; on the streaming path the 429 status is emitted before any SSE body bytes
  - fleet_status_payload (GET status) exposes the current busy policy state (mode=busy plus the pressure reason) so operators can see requests are being shed

### t4 — lobes status --pressure reports mode=busy without a live request (lobes/cli/_commands/status.py)

- depends on: t1
- covers: c13, h3
- acceptance:
  - _cmd_status_pressure reports mode=busy and the current tier ceiling (which tiers would be shed) computed from decide(), WITHOUT issuing a live generate request
  - the reported ceiling matches the admission decision handle_post would make for the same pressure sample (main and senses shed, minor served)
  - --json output carries the busy mode/reason fields; human output labels the busy state clearly

### t5 — End-to-end test: pressured request gets 429, retry after pressure clears gets a real answer (tests/test_gateway_busy_e2e.py)

- depends on: t3
- covers: c1, c21, h4, h11
- acceptance:
  - with an injected high-pressure sample, a cortex request over the gateway returns 429+Retry-After (never a substitute model), and a senses request likewise returns 429
  - with pressure cleared, the same cortex request returns 200 from the real cortex model, proving the busy signal is transient and a retrying client eventually succeeds
  - the assertion runs end-to-end through handle_post, not as a unit test of decide() alone

### t6 — Document the busy-backpressure contract; remove degrade-to-minor from the docs (docs/gateway-fleet.md, docs/openai-api.md, lobes/explain)

- depends on: t3
- covers: c2, c5, c6, h5, h6, h8, h9
- acceptance:
  - docs/gateway-fleet.md and lobes explain gateway state: degrade-to-minor is removed; under pressure the gateway returns 429+Retry-After busy for cortex AND senses; explicit minor is still served
  - docs/openai-api.md documents the 429+Retry-After contract and states that callers (acp vllm-local provider, colleague, generic OpenAI SDKs) MUST retry with backoff
  - the rationale (a wrong-capability answer is worse than a retry delay) and the boundary (only the response changes; lobes/runtime/_pressure.py and the #86 threshold env vars are untouched) are both stated
  - grep confirms no doc still claims pressure degrades cortex/main onto minor or Gemma

### t7 — Version bump + CHANGELOG for the busy-backpressure change (pyproject.toml, CHANGELOG.md)

- depends on: t3, t4, t5, t6
- acceptance:
  - the version-bump script bumps pyproject.toml above main and prepends a CHANGELOG entry describing the removal of pressure-degrade and the new 429 busy contract, noting it closes #85
  - the uv.lock re-pin is committed if the bump changes it

## Risks

- [follow_up] Non-retrying callers regress: a client that treats 429 as fatal becomes LESS available under pressure than the old always-answer-with-something behavior. Client retry handling (honesty h5) is cross-repo (culture acp vllm-local backend + colleague) and cannot be built in this repo. Audit those callers and, if needed, file a change against them before relying on busy in production. Track against #85. (task t6)
- [out_of_scope] Queue-depth admission control (shed on vLLM busy.waiting rather than swap/iowait) was considered and deferred (decision c14). Revisit as a follow-up if the swap/iowait signal proves too coarse a trigger for real saturation.
