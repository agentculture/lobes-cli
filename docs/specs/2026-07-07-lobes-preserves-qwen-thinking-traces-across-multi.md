# lobes preserves Qwen thinking traces across multi-turn agent loops on the vLLM cortex, so historical <think> reasoning is carried back into context instead of discarded

> lobes preserves Qwen thinking traces across multi-turn agent loops on the vLLM cortex, so historical <think> reasoning is carried back into context instead of discarded

## Audience

- the lobes cortex agent (and any Colleague/mesh caller running multi-turn loops against the vLLM 27B primary) whose reasoning quality benefits from seeing its own prior <think> traces

## Before → After

- Before: --reasoning-parser=qwen3 already exposes the trace as message.reasoning, but the Qwen chat template strips historical <think> blocks and lobes' client keeps only 'content', so every turn discards the model's prior reasoning
- After: vLLM launches with preserve_thinking=true and lobes' generate client round-trips the assistant 'reasoning' field back into history, so a multi-turn call re-feeds prior <think> traces and continuity across planning/validation/agent loops improves

## Why it matters

- Qwen3.6 emits reasoning designed for extended agent chains; preserving historical thinking should improve continuity across multi-turn repo work, planning, validation and agent loops instead of re-deriving it each turn

## Requirements

- vLLM primary/cortex launch adds --default-chat-template-kwargs '{"preserve_thinking": true}' alongside the existing --reasoning-parser=qwen3, in both templates/docker-compose.yml and templates/fleet/docker-compose.yml
  - honesty: the deployed vLLM build boots with --default-chat-template-kwargs '{"preserve_thinking": true}' (no unknown-arg crash), and the rendered prompt retains historical <think> blocks when that kwarg is set
- the generate client captures assistant message.reasoning alongside message.content, and re-sends both on subsequent turns using the 'reasoning' field so vLLM re-renders the historical <think>
  - honesty: the served model returns a populated message.reasoning, and re-sending an assistant turn carrying that 'reasoning' key causes vLLM to re-inject it as <think> in the next render rather than silently dropping it
- a read-only diagnostic verb (e.g. lobes assess/doctor extension) runs a two-turn probe and reports prompt-token delta between content-only and content+reasoning history, proving preservation is live
  - honesty: the two-turn probe shows a measurable, repeatable prompt-token increase attributable to re-fed reasoning (signal above run-to-run noise)

## Honesty conditions

- end-to-end: with preserve_thinking on and reasoning re-sent, a later turn's rendered prompt actually contains the earlier <think> (provable by the token-delta diagnostic)
- a real caller runs multi-turn loops against cortex where its own prior reasoning is useful (the lobes agent / Colleague mesh loops)
- after the change a two-turn call re-feeds prior <think> without regressing existing single-turn behavior or route's terse path
- verified true by template inspection: without preserve_thinking the served mmangkad/Qwen3.6 template strips historical <think> for turns before the last user query, and lobes' client keeps only content
- if preserved reasoning yields no measurable continuity benefit, the feature still stays opt-in and costs only prompt tokens — no correctness regression
- the change touches only the cortex/main generate lane and its client; route, minor/pressure lanes, embedder/reranker/senses stay untouched
- the diagnostic's token-delta is reproducible and clearly attributable to reasoning history, not run-to-run noise

## Success signals

- a diagnostic command shows prompt-token count rises on a two-turn call when reasoning history is preserved vs content-only, proving the historical <think> is actually re-fed

## Scope / boundaries

- not renaming the cortex role, not mandating preserve-thinking globally (lobes route keeps enable_thinking=false), not building an external memory system

## Decisions

- lobes route stays enable_thinking=false and the pressure-policy degraded 'minor' lane is unaffected — preserve_thinking is opt-in to the cortex/main generate lane, not global
- delivery is BOTH: launch default --default-chat-template-kwargs '{"preserve_thinking": true}' on the cortex/main lane in templates/docker-compose.yml AND templates/fleet/docker-compose.yml, AND the generate client is chat_template_kwargs-aware so it can override per-request (request-level kwargs override the server default; e.g. lobes route keeps its terse path)
- VERIFIED (not assumed): --default-chat-template-kwargs is a stable vLLM serve flag since 0.9.0 and both pinned images (nvcr.io/nvidia/vllm:26.04-py3, 2026 vllm/vllm-openai nightly) are newer; preserve_thinking is a real variable in the served mmangkad/Qwen3.6-27B-NVFP4 chat template (gates historical <think> retention). Residual boot-check lives in c8's honesty condition h1.

## Hard questions

- Does the pinned nv26.04 vLLM image's CLI expose --default-chat-template-kwargs, or is it newer-vLLM-only? If absent the container fails to boot (hard failure, not graceful degrade).  [RESOLVED: verified — flag stable since vLLM 0.9.0, both pinned images newer; preserve_thinking present in served mmangkad/Qwen3.6 template. Residual boot-check = c8/h1.]
- risk: If --default-chat-template-kwargs or the preserve_thinking template var is unsupported on the pinned image, c8 as written is blocked; fallback is per-request extra_body chat_template_kwargs instead of a launch default
