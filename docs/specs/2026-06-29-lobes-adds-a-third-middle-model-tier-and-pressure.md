# lobes adds a third (middle) model tier and pressure-aware tier routing

> Three capability tiers — **cheap** (4B bf16) / **normal** (14B NVFP4) / **hard** (27B @ 128K) — stay co-resident on the Spark behind the gateway. Downstream callers request a tier *alias* (`model=cheap|normal|hard`) instead of a concrete model name; the 27B served context is trimmed from 256K to 128K to make room; a full swap/iowait **pressure policy** (with a degraded mode) governs tier resolution. LoRA training stays on the existing 4B bf16 `minor` (the 14B is inference-only — *not* a LoRA base).
>
> _Implements [issue #68](https://github.com/agentculture/lobes-cli/issues/68). Spec produced via `/think` (devague); frame `lobes-adds-a-third-middle-model-tier-and-pressure`._

## Audience

- Downstream Culture mesh agents and tools that today hardcode a model name or backend, plus the operator running the lobes fleet on the shared DGX Spark

## Before → After

- Before: Two generate gears exist (minor Qwen3.5-4B + 27B primary) but there is no middle tier, no capability-tier contract (callers hardcode model names), and no pressure-aware routing on swap/iowait
- After: Three generate tiers (cheap 4B / middle 8B-14B / hard 27B) are co-resident behind the gateway; a caller asks lobes for a capability tier (cheap/normal/hard) and gets back the chosen model + mode + reason, with pressure-aware fallback when unified memory is under stress

## Why it matters

- DGX Spark uses unified memory: CPU and GPU share one physical pool, so running the 27B as the default path causes swap thrash. Tiering lets the box behave like a memory-budgeted lab device — small models carry routine work, the large model is the escalation/authority path

## Requirements

- All three generate gears (4B bf16 + 14B NVFP4 + 27B@128K) plus the embed and rerank pooling gears are simultaneously resident and healthy within the Spark 128GB unified-memory budget
  - honesty: The GPU mem-util budget is shown to sum under 1.0 (27B@128K + 14B-NVFP4 + 4B + embed + rerank) AND lobes fleet status reports all five gears healthy after bring-up on the Spark
- The gateway resolves the model aliases cheap/normal/hard to the 4B/14B/27B generate gears respectively, as a same-task (generate) alias layer on top of the existing task-family routing
  - honesty: An OpenAI request with model=normal returns a chat completion whose served model is the 14B gear; an embed request still never fails over to a generate gear (same-task constraint preserved)
- lobes samples host memory pressure read-only (swap_used_percent + iowait_percent, from /proc) without mutating the system
  - honesty: Pressure fields are populated from real /proc readings on the Spark and the sampler is side-effect-free (no writes, no container restarts)
- Pressure thresholds from issue #68 drive tier resolution and a degraded-mode state machine (swap>50 avoid new hard jobs; swap>65 prefer cheap/middle; swap>75 degraded=cheap-only; iowait>25 no new hard; iowait>50 emergency degraded); under degraded mode a model=hard request resolves down to a permitted tier and the response reports reason=pressure
  - honesty: An injected/simulated swap>75% causes a model=hard request to be served by the cheap tier with reason=pressure; every threshold is config-driven, not hardcoded magic numbers
- A manual override lets a caller force the requested tier despite pressure (bypassing the automatic downgrade); the override and the downgrade reason cross the OpenAI-compatible boundary via HTTP headers, and the response model field reflects the actually-served model
  - honesty: With the override header set, model=hard is served by the 27B even under simulated degraded pressure; without it, the same request downgrades
- A read-only observability surface reports the active selection as {tier, model, mode(warm|lazy|disabled|degraded), reason(default|escalation|pressure|manual_override), pressure{swap_used_percent, iowait_percent}} — e.g. via lobes status --pressure
  - honesty: lobes status --pressure emits exactly {tier, model, mode, reason, pressure{swap_used_percent, iowait_percent}} and mutates nothing (read-only, like the existing status verb)

## Honesty conditions

- After bring-up, all three generate gears (4B + middle + 27B) plus embed/rerank are simultaneously resident and healthy, verified by lobes fleet status / nvidia-smi, within the 128GB unified-memory budget
- A caller names only a capability tier (cheap/normal/hard) and the gateway/route resolves it to the correct co-resident gear — no concrete model id crosses the contract boundary
- A concrete downstream caller (e.g. a Culture mesh agent or daria) can switch from a hardcoded model id to a tier alias with no other code change and keep working
- On the Spark, running the 27B as the default path is shown to drive swap/iowait up, and routing routine work to the 4B/14B measurably lowers that pressure
- Today, with only minor(4B) + primary(27B), there is no catalog gear whose role_hint marks it as the middle tier, so route/select cannot offer a normal/middle option
- After bring-up a caller issues cheap/normal/hard and receives a completion from the matching gear, with mode and reason surfaced
- The shipped scope contains no lobes train / fine-tuning code path; LoRA training appears only as a referenced follow-up
- The memory math is shown: trimmed-27B@128K KV cache + 14B-NVFP4 weights+KV + 4B + embed + rerank (+optional audio) sums under the GPU mem-util budget that fits 128GB

## Success signals

- A downstream caller selects a capability tier by sending model=cheap|normal|hard to the OpenAI-compatible gateway and receives a completion from the right gear without ever naming a concrete model; all three generate gears plus embed/rerank fit within the Spark 128GB unified-memory budget

## Scope / boundaries

- This spec delivers the third (middle) tier plus the capability-tier routing contract and the full pressure policy; it does NOT implement LoRA fine-tuning itself (lobes train stays a follow-up)

## Non-goals

- Not dynamically unloading/reloading the 27B per request; the three generate gears stay co-resident — memory headroom comes from trimming the 27B served context, not from swapping models in and out

## Decisions

- Middle tier = a 14B-class NVFP4 checkpoint (memory-lean ~8GB weights, inference-only escalation reviewer/coder). Exact HF checkpoint id verified separately (candidates: nvidia/Qwen3-14B-NVFP4 or equivalent), consistent with the existing nvidia/Qwen3-32B-NVFP4 candidate
- 27B primary served context is trimmed from full 256K (262144) to 128K (131072) via PRIMARY_MAX_MODEL_LEN, freeing roughly half the KV cache to make room for the co-resident middle gear
- Tier-request surface is the gateway model-alias ONLY: a caller sends model=cheap|normal|hard to the OpenAI-compatible endpoint and the gateway resolves it to the 4B/14B/27B generate gear. No new 'lobes select --tier' verb and no 'route --tier' field in this spec
- Full pressure policy is in scope: lobes senses host memory pressure (swap%, iowait%) and applies the issue #68 thresholds, including a degraded-mode state machine and pressure-aware downgrade
- LoRA training target is the existing 4B bf16 minor lobe; the 14B NVFP4 middle tier is inference-only and is NOT a LoRA base. The 'three tiers to scale work and train loras' goal = scale inference across 3 tiers + keep the 4B as the trainable base

## Hard questions

- risk: A 14B NVFP4 checkpoint must actually load and produce non-gibberish output on the fleet's Blackwell vLLM image; older dense Qwen3 NVFP4 (cf. the supported 32B candidate) is lower-risk than a Qwen3.5/3.6 hybrid which hit a known FLA bug pre-vLLM-0.23

## Open questions (parked, non-blocking)

- Exact 14B NVFP4 HF checkpoint id — does a same-generation Qwen3.6-14B NVFP4 exist, or do we fall back to a dense Qwen3-14B NVFP4 (consistent with the existing `nvidia/Qwen3-32B-NVFP4` candidate)? Resolve by checking HF + a load-test on the fleet image before promoting.
- Exact HTTP header names for the tier override + the downgrade reason, and whether a *streaming* response can carry the reason (response header vs first-chunk metadata).
