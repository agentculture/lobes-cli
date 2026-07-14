# lobes serves strict, grammar-constrained tool calls with thinking enabled: the cortex ships a think-aware tool-parser plugin so structural-tag decoding survives </think>, and every OpenAI tools caller gets well-formed tool calls instead of parser-salvage mangles — Colleague #320's read_file" + empty-arguments failure is impossible by construction

> lobes serves strict, grammar-constrained tool calls with thinking enabled: the cortex ships a think-aware tool-parser plugin so structural-tag decoding survives </think>, and every OpenAI tools caller gets well-formed tool calls instead of parser-salvage mangles — Colleague #320's read_file" + empty-arguments failure is impossible by construction
> instruction: Implement as: (1) lobes/vllm_plugins/qwen3_thinking_tool_parser.py — subclass the image's Qwen3EngineToolParser, override get_structural_tag to derive reasoning from the request's effective enable_thinking (default on), register as qwen3_coder_thinking, assert the import surface at load; (2) fleet compose template mounts the file into vllm-primary and adds --tool-parser-plugin + PRIMARY_TOOL_CALL_PARSER=qwen3_coder_thinking; (3) gateway: opt-in GATEWAY_FORCE_STRICT_TOOLS knob injecting function.strict=true on cortex-lane tools requests; (4) verify with the #320 replay matrix + a real colleague work item; (5) reply on colleague#320 with root cause + fix, file the reasoning=False hardcode upstream on vLLM

## Audience

- OpenAI-tools callers of the cortex generate lane — Colleague's work/drive loop first (issue #320), plus any mesh agent sending tools with tool_choice=auto through the lobes gateway

## Before → After

- Before: Unconstrained thinking-mode generation drifts off the qwen3_coder XML template (hallucinated '<tool_call>\n\n[tool_id]{json}' emissions); vLLM parser salvage returns name='read_file"' with empty arguments; every Colleague work item ends incomplete with zero changed files. strict:true (the OpenAI knob that should fix it) returns HTTP 500 because the served vLLM build hardcodes reasoning=False when building the structural-tag grammar, so the FSM cannot accept the </think> token
- After: With thinking ON and strict armed, the cortex returns well-formed tool calls (valid name, schema-valid JSON arguments) — proven by replaying the captured Colleague #320 request: it yields a clean read_file {"path": "calc.py"} instead of a mangle or a 500; a real 'colleague work' item lands its change

## Why it matters

- Cortex-driven agentic work is the fleet's reason to exist: a cortex that cannot deliver a single tool call reliably makes every downstream consumer (Colleague work/drive, mesh agents) a no-deliverable loop. Constrained decoding makes malformed tool calls impossible by construction rather than merely less likely

## Requirements

- lobes ships a tool-parser plugin (a small Python file in the lobes package, mounted into the vllm-primary container and loaded via vLLM's --tool-parser-plugin flag) that registers a think-aware parser: it subclasses the image's Qwen3EngineToolParser and overrides get_structural_tag to pass reasoning=True when the request has thinking enabled, so the structural-tag grammar accepts 'any_text </think> \n\n <tool_call>…' — strict + thinking stops 500ing
  - honesty: The plugin loads in the pinned image and registers under a distinct name (e.g. qwen3_coder_thinking): 'vllm serve --tool-parser-plugin /path/plugin.py --tool-call-parser qwen3_coder_thinking' boots healthy, and a strict+thinking replay of the captured #320 request returns a clean structured call instead of HTTP 500
- The plugin is thinking-aware per request, not hardcoded: when a request disables thinking (e.g. lobes route's enable_thinking=false, or any caller passing chat_template_kwargs), the plugin passes reasoning=False — a reasoning=True grammar REQUIRES a closing </think>, so forcing it on a no-think request would break generation. The decision input is the request's effective enable_thinking, with the server default (thinking on) as the fallback
  - honesty: get_structural_tag receives the full ChatCompletionRequest, and the request's effective thinking state (chat_template_kwargs.enable_thinking, defaulting to on) is readable there — verified by a no-think strict request still returning a clean call through the plugin (a reasoning=True grammar would hang/never satisfy on a no-think generation)
- The fix applies to the cortex/main generate lane only (vllm-primary command args + a mounted plugin file); the senses/embed/rerank/minor lanes are untouched, mirroring the preserve_thinking (#93) scoping precedent
  - honesty: senses/embed/rerank/minor service definitions in the rendered compose are byte-identical before/after the change (template diff), and their lanes' probes still pass
- Who arms strict is config-gated in the lobes gateway: an opt-in env knob (e.g. GATEWAY_FORCE_STRICT_TOOLS=1) that injects strict:true into every function tool on cortex-lane chat requests that carry tools, so existing callers (Colleague today) get constrained tool calls without a client change; with the knob off, behavior is byte-identical to today and callers may still send strict:true themselves
  - honesty: With the knob unset, gateway request bytes to the primary are identical to today (goldens/diff test); with the knob set, only chat-completions requests carrying tools on the cortex lane gain function.strict=true, and the #320 replay through the gateway returns the clean call

## Honesty conditions

- Post-deploy, a strict+thinking request to the live cortex returns a well-formed tool call (valid name, schema-valid JSON args) — the captured #320 replay passes against the deployed fleet, not just in theory
- The fix covers Colleague's unmodified request shape exactly as captured (14 tools, tool_choice=auto, non-streaming, temperature 0.0) — the replay uses the captured bytes, not a synthetic approximation
- The before-state is reproducible on demand: replaying the captured request against the unfixed fleet deterministically returns name='read_file"' args={} (proven live 2026-07-14, eidetic record colleague-320-tool-call-mangle-root-cause)
- Multi-turn holds, not just turn one: in a real tool loop (assistant→tool→assistant), every thinking-on generation emits </think> before any tool call, so the reasoning=True grammar is satisfiable on every turn — proven by the colleague work item completing its full loop
- The same 'colleague work' repro that today reports write-no-changes/incomplete delivers changed files after the fix — the value claim is measured by the consumer's own outcome, not by lobes-side metrics alone
- The shipped diff touches only the lobes repo: vLLM image digest unchanged, zero colleague-repo commits; the entire surface is the plugin file, the vllm-primary compose args, and the gateway knob
- The four-way matrix runs against the live re-deployed fleet and every leg passes, including a real 'colleague work' item that lands changed files on a throwaway repo — the same repro that today reports write-no-changes
- The plugin import of Qwen3EngineToolParser + the get_structural_tag(request, reasoning=...) signature is asserted at container start; a mismatched image fails the primary's boot health check loudly rather than serving unconstrained calls silently

## Success signals

- The captured #320 replay matrix passes against the live fleet: (1) baseline non-strict request still answers (no regression), (2) strict + thinking-on returns a clean structured read_file call with valid JSON args (today: HTTP 500), (3) strict + enable_thinking=false still returns a clean call, (4) 'lobes assess' correctness probes and a real 'colleague work' item on a throwaway repo deliver changed files. MTP spec-decode stays active throughout (SpecDecoding metrics present in the primary log)

## Scope / boundaries

- Not an upstream vLLM patch or image rebuild: the pinned vllm/vllm-openai image stays byte-identical; the fix rides the image's own extension surface (--tool-parser-plugin + ToolParserManager). Filing the reasoning=False hardcode upstream is a follow-up courtesy, not this fix. Not a Colleague code change either: colleague#320 gets a reply with findings, but lobes does not edit the colleague repo. Not a fix for the model's template drift itself (that's the checkpoint's behavior; we constrain around it)

## Assumptions

- The vLLM image pin stays on 0.23.1rc1.dev672+g93d8f834d (or any image whose Qwen3EngineToolParser + structural-tag registry expose the same override surface); if the pin moves, the plugin's subclass target and get_structural_tag signature must be re-verified — the plugin should fail loudly at container start if the import surface changed

## Decisions

- Chosen mechanism is xgrammar structural-tag constrained decoding (strict:true), NOT prompt engineering, NOT retry-on-mangle in the gateway, NOT switching tool-call parser names (qwen3_coder and qwen3_xml alias the same class in this build) — because constrained decoding makes the failure impossible rather than less frequent, and it is already proven live on this box with MTP active
- Arm-strict ownership (user decision): BOTH — the lobes gateway ships the opt-in GATEWAY_FORCE_STRICT_TOOLS knob (fixes Colleague on this rig with zero client change), and the colleague#320 reply recommends Colleague also send strict:true client-side for rigs without the knob
- Compile-failure behavior under force-strict (user decision, resolves q2): the gateway catches a schema-compile failure and RETRIES the request once without strict — the caller degrades to today's unconstrained behavior instead of breaking; the gateway logs the offending schema for follow-up

## Hard questions

- Does ToolParserManager.import_tool_parser in this image accept a mounted file path and register before serving starts, and does the compose template have a clean way to mount one file into vllm-primary?
- When force-strict is on and a caller's tool schema fails xgrammar compilation (all 14 Colleague schemas compiled fine, but arbitrary mesh callers are unvetted), what does the caller see — a 400 with a clear error, or should the gateway retry that request without strict as a fallback? [RESOLVED by user decision 2026-07-14, recorded as confirmed claim c15: gateway retries once without strict on schema-compile failure; offending schema logged] (blocking)

## Open / follow-up

- Should GATEWAY_FORCE_STRICT_TOOLS eventually default ON for the cortex lane (making constrained tool calls the fleet default), and should Colleague ALSO send strict:true client-side for rigs without the gateway knob? Both are post-fix policy calls
- Why the model drifts to the hallucinated '[tool_id]{json}' format on some prompts at temp 0 (checkpoint training artifact; out of lobes' control — we constrain around it, we don't retrain it)
