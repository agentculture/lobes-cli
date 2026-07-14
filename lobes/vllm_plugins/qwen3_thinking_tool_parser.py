"""PLACEHOLDER — overwritten by task t1 of the same devague plan (issue #93-
adjacent, the qwen3_coder_thinking tool-parser plugin). Task t2 (this task's
own scope) only wires FLEET/INIT MATERIALISATION of whatever file lands at
this exact path (`lobes/vllm_plugins/qwen3_thinking_tool_parser.py`) into a
deployment dir — it does not implement the parser itself.

t1's real module registers a vLLM ``ToolParser`` named ``qwen3_coder_thinking``
— a reasoning-aware variant of vLLM's built-in ``qwen3_coder`` parser. Upstream
hardcodes ``reasoning=False`` on every emitted tool call, which breaks strict
structural tags for a thinking model that needs to stay reasoning-aware across
a tool-call turn. Loaded by vLLM via ``--tool-parser-plugin`` naming this file
by PATH (mounted read-only into the ``vllm-primary`` container by
``lobes/templates/fleet/docker-compose.yml``) — never imported by the `lobes`
package itself, so this placeholder has zero runtime effect on the CLI.

Do not ship this placeholder to a real deployment: merging t1's branch
overwrites this file with the actual parser implementation. It exists purely
so t2's `lobes init` fleet-materialisation step (``lobes.runtime._compose``:
``PLUGIN_PACKAGE`` / ``write_plugin_file``) has a real file at the pinned
package path to read and copy while t1 develops in parallel, in an isolated
worktree, on a different branch.
"""
