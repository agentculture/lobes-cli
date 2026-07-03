# Under pressure, lobes tells you the cortex model is busy and to retry shortly — it never silently degrades your request onto a weaker or different-capability model.

> Under pressure, lobes tells you the cortex model is busy and to retry shortly — it never silently degrades your request onto a weaker or different-capability model.

## Audience

- Callers of the lobes gateway generate lane — the Culture mesh acp vllm-local provider and colleague — plus operators running the memory/swap-pressured GB10 fleet.

## Before → After

- Before: Under pressure the policy caps the tier to minor; with minor unwired the router's upward-fallback silently serves the request from Gemma (multimodal) — a different capability — disclosed only via X-Lobes-Tier headers (bug #85).
- After: A cortex OR senses request under swap/iowait pressure gets an explicit, retryable busy response (HTTP 429 + Retry-After) instead of a silent 2xx answer from a substituted cheaper or different-capability model.

## Why it matters

- A caller asking cortex for authoritative reasoning must never silently receive a weaker or wrong-capability answer; an honest 'busy, retry' preserves the capability contract (#81) and lets clients back off instead of acting on a degraded result.

## Requirements

- The busy response carries a disclosure header (X-Lobes-Tier-Reason: busy) plus Retry-After so a client can distinguish transient busy (retry) from the existing hard 502 'all backends unavailable' (don't retry).
  - honesty: A client can reliably distinguish busy (retry) from the existing 502 all-backends-unavailable (don't retry) from status+header alone, and on the streaming path the busy status is emitted before any SSE body bytes.
- lobes status --pressure and the gateway /capabilities/GET status surface the busy policy state (mode=busy, current tier ceiling, whether requests are being shed) so operators can see WHY a client is being told to wait.
  - honesty: lobes status --pressure and the gateway status endpoint report mode=busy and the current tier ceiling WITHOUT issuing a live generate request, and the reported ceiling matches the admission decision handle_post would actually make.
- Under pressure, the gateway returns HTTP 429 with a Retry-After header and an OpenAI-shaped error body for a generate/multimodal request it would otherwise have substituted onto a different model, instead of rewriting it onto that model.
  - honesty: The acp vllm-local provider, colleague, and generic OpenAI SDKs treat 429 + Retry-After on POST /v1/chat/completions as a retryable transient with backoff, not a fatal error surfaced to the end user.

## Honesty conditions

- On a live pressured fleet, a cortex request observably returns a retryable busy signal (not a Gemma/other-model completion), and a retrying client eventually receives a genuine cortex answer once pressure clears.
- The acp vllm-local provider and colleague are the actual pressured-fleet callers of the generate lane (they dial the gateway origin per #87), so THEIR retry behavior is what determines whether busy backpressure is safe in practice.
- The silent Gemma substitution is the real current code path, not a hypothetical: resolve_tier_request('cortex', swap=80%) on a default fleet with minor unwired returns served=Gemma, tier=minor, reason=pressure (reproduced against the live code).
- For a cortex caller, a wrong-capability/degraded answer is materially worse than a retry delay — callers would rather wait and retry than silently act on a Gemma answer to an authoritative-reasoning request.
- The change is confined to the gateway response/decision path (handle_post + pressure-policy decision); the swap/iowait sampler (_pressure.py) and the threshold env vars (#86) are untouched.
- On the shed path the gateway issues NO upstream model request — it returns 429 itself — so a busy cortex/senses request provably never reaches a substitute model.
- Success is observable end-to-end (not just a unit-tested decision()): a scripted pressured request shows 429+Retry-After for cortex, and after pressure clears a 200 from the real cortex model.

## Success signals

- On a pressured fleet with no minor gear, a cortex request returns 429 busy+Retry-After (never a Gemma completion), and a retrying client eventually gets a real cortex answer once pressure clears.

## Scope / boundaries

- Not a request queue/scheduler or admission-control rewrite, and not a change to vLLM's internal batching; the swap/iowait sampling and tunable thresholds (#86) stay as the pressure signal — only the RESPONSE to pressure changes.

## Assumptions

- The mesh acp vllm-local provider and colleague treat a 429 + Retry-After on /v1/chat/completions as a transient/retryable signal and back off, rather than surfacing it as a hard failure to the end user.

## Decisions

- The busy trigger reuses the EXISTING swap/iowait pressure signal (mode=degraded becomes mode=busy): when the policy would have degraded, it sheds instead. It does NOT introduce vLLM queue-depth (waiting-count) admission control.
- The silent degrade-to-minor path is REMOVED outright: no 'degrade' mode and no LOBES_PRESSURE_POLICY toggle to restore it. Under pressure the only behavior is busy backpressure — which makes the cross-capability substitution of #85 structurally impossible.
- Busy applies to ANY request pressure would otherwise substitute — both cortex (main/hard) and senses (multimodal/normal). The gateway never serves a different model than requested under pressure; a request whose own gear is healthy and served-as-requested is unaffected.
- An explicit model=minor request is still served under pressure (served as requested, not a substitution). Busy is returned only for a request whose requested tier/capability pressure would otherwise have swapped for a different model.

## Resolved forks (decided during /think)

These were open when the frame started; the confirmed Decisions above record the choices.

- HTTP status for busy: 503 (overloaded) vs 429 (rate-limited) → **429 + Retry-After**.
- Remove the silent degrade path outright vs keep it as an opt-in mode → **removed outright** (no `LOBES_PRESSURE_POLICY` toggle).
- Trigger: swap/iowait pressure vs vLLM queue depth → **reuse the existing swap/iowait pressure signal** (queue-depth admission control is a possible follow-up, out of scope here).
- Scope: cortex/generate lane only vs any cross-capability substitution → **any substitution** (cortex and senses).

## Open risk (carry into the plan)

- If a caller does NOT honor 429 / Retry-After (treats it as fatal), removing degrade makes lobes strictly **less available** under pressure than today's always-answer-with-something behavior — a regression for non-retrying callers. This is exactly why client retry handling is a **honesty condition to verify** (h5), not a silent assumption. Mitigation to confirm during build: audit the acp `vllm-local` provider and colleague retry behavior before shipping, and file/track it against #85 if a client needs a change.
