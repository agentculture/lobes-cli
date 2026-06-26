# lobes ships a 'minor' lobe: a cheap, warm co-resident Qwen3.5-4B small-brain served bf16 alongside the 27B primary, driven by lobes run/route/eval with escalation + confidence governance, and chosen as the first unsloth-LoRA fine-tune target (training itself comes later)

> lobes ships a 'minor' lobe: a cheap, warm co-resident Qwen3.5-4B small-brain served bf16 alongside the 27B primary, driven by lobes run/route/eval with escalation + confidence governance, and chosen as the first unsloth-LoRA fine-tune target (training itself comes later)

## Audience

- lobes maintainers and Culture-mesh agents; agent loops that need a cheap local model for high-frequency low-risk calls (classify, route, format, validate, summarize) instead of spending 27B-primary tokens

## Before → After

- Before: lobes only switches/serves/assesses ONE served model; it has no notion of a cheap secondary worker, no co-resident small-brain, and no verb that invokes a model with a prompt (only assess.py's internal urllib correctness probes hit the model today)
- After: Qwen3.5-4B is a catalog gear with role_hint=minor, served WARM CO-RESIDENT behind the gateway; lobes overview --list and GET /v1/models/supported list it
- After: governance: minor MAY prepare/classify/format/validate/suggest but is config-blocked from approve/finalize/delete/deploy/architectural decisions; escalation conditions (needs_codebase_context, security_sensitive, architectural_decision, write_or_delete_operation, final_review_required) route work upward
- After: new verbs (read-only): lobes run minor "<prompt>" calls the co-resident 4B; lobes route "<text>" picks which GEAR handles a task (minor vs 27B primary vs candidate) AND whether to escalate, with a confidence score; lobes eval minor --suite ... runs an eval; v1 routes ONLY across lobes gears (not tools, not mesh agents)

## Why it matters

- the architectural primitive becomes 'minor' (a role), not 'qwen4b' (a model): lobes stays model-agnostic while AgentCulture gets a practical cheap-local default and its first LoRA target, and reflex work stops burning the 27B

## Requirements

- lobes/runtime/_parser.py learns the Qwen3.5 family so infer_parser(id) == the catalog tool_parser (the catalog test asserts equality), set to whatever tool format Qwen3.5-4B actually emits
  - honesty: the parser is set from Qwen3.5-4B's REAL emitted tool format (verified, not assumed), and a _parser unit test pins infer_parser('Qwen3.5-4B') to that value
- the catalog can represent an UNQUANTIZED bf16 generate gear: a quantization-field convention (a sentinel that lobes switch/compose translate to omitting --quantization) plus updated catalog tests, since today every generate entry requires a non-empty quantization
  - honesty: a bf16 entry boots under the GB10 vLLM image with the chosen quantization convention (—quantization omitted), and the updated catalog tests still pin every other invariant for the other gears
- lobes run/route/eval reuse the existing stdlib-only urllib client pattern (assess.py) to call the OpenAI-compatible endpoint — no new heavy deps — and stay read-only per the mutation-safety contract
  - honesty: run/route/eval add zero runtime deps beyond stdlib (matching assess.py) and pass the read-only / mutation-safety checks in doctor
- warm co-residency: the fleet compose gains an opt-in 'minor' generate backend and the gateway can target minor vs primary for generate traffic
  - honesty: the co-resident minor backend boots alongside the 27B on the GB10 within memory budget (modest max_model_len + bounded gpu-mem) without OOMing the primary

## Honesty conditions

- every piece lands behind an EXISTING safety contract — read-only verbs (run/route/eval) and dry-run-by-default mutations (compose/switch) — not a new unguarded surface
- the named reflex tasks (classify/route/format/validate/summarize) are within a 4B's reliable range; anything needing codebase context or judgment ESCALATES rather than silently degrading
- confirmed against the CLI surface: no existing verb invokes the model with a prompt (assess.py probes are internal), and the catalog has no role_hint=minor / co-resident generate gear today
- the role abstraction holds — swapping the 4B for another small model later is a catalog edit only; the verbs target the 'minor' ROLE, not a model id
- the new entry passes every tests/test_catalog.py invariant and appears in BOTH lobes overview --list and GET /v1/models/supported
- the allowed/forbidden duty split and escalation conditions are ENFORCED in config (minor refuses/escalates a forbidden action), not merely documented prose
- the bf16-base decision is sufficient to LoRA-finetune later with unsloth without re-exporting the served checkpoint (bf16 base is the adapter target)
- every listed signal is OBSERVABLE in CI/local (run returns a completion, route returns decision+confidence, catalog/parser/doctor tests pass), not subjective
- lobes route emits a structured, machine-parseable decision (chosen gear + escalate flag + confidence) validated by a test, so agent loops can consume it

## Success signals

- lobes run minor returns a completion from the co-resident 4B; lobes route returns a structured decision with confidence; escalation thresholds are representable and tested; catalog + parser + afi cli doctor --strict tests pass; docs state allowed/forbidden minor duties and name Qwen3.5-4B as the first fine-tune target

## Scope / boundaries

- NOT actually fine-tuning yet — no training runs, datasets, or shipped adapters; bf16 base is chosen to ENABLE later unsloth LoRA. NOT making minor a reviewer/approver. NOT displacing the 27B primary. The model is an implementation detail of the 'minor' role

## Decisions

- served checkpoint = Qwen/Qwen3.5-4B (bf16), chosen because unsloth LoRA needs the bf16 base; community cosmicproc/Qwen3.5-4B-NVFP4 is documented as an UNTESTED config alternative, not the default
- role_hint=minor — a new behavior-free catalog label (only 'primary' is load-bearing in switch.py/MTP; overview prints role_hint, gateway routes by task not role)
- warm co-resident (opt-in always-on second generate backend), not switch-only — served at a modest max_model_len and bounded gpu-mem so it does not crowd the 27B
- VERIFIED model facts (HF config.json): Qwen3.5-4B is hybrid linear-attn (Gated Delta/SSM) + full-attention (NOT MoE, so no moe_backend), multimodal (ViT image+video), built-in MTP draft head, 256K native (max_position_embeddings 262144), bf16, Apache-2.0, public; tool format = qwen3_coder (XML); served as a text-only reflex brain via --language-model-only to drop the vision tower

## Hard questions

- risk: lobes gaining run/route/eval shifts it from model-ops toward an agent-runtime — scope creep vs its stated identity ('runs, assesses, switches'); mitigate by keeping verbs THIN (prompt in / decision out), no agent loop, no persistent state
- what does 'route' decide BETWEEN — lobes gears, tools, or other agents/lobes? the routing-target taxonomy must be defined or the verb is underspecified (blocking)

## Open / follow-up

- lobes train minor (unsloth LoRA/QLoRA): real training runs, datasets, adapter artifacts, adapter hot-swap/serving — deferred ('not yet'); the bf16-base decision exists to enable it
- eval-suite CONTENTS (evals/tool-routing, formatting, contract-following datasets) — the 'lobes eval' verb lands now; the curated suites come later
- Qwen3.5-4B ships a built-in MTP draft head (mtp_num_hidden_layers=1) — native speculative decoding for the minor with no separate draft checkpoint; deferred because the catalog test gates speculative_config on an 'MTP' id substring
