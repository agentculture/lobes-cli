"""lobes.vllm_plugins — vLLM tool-parser plugins for the served Qwen3.6 model.

This package is **not** imported by the `lobes` CLI/gateway at runtime; it is
data shipped in the wheel that the vLLM *container* points at via
``--tool-parser-plugin lobes/vllm_plugins/qwen3_thinking_tool_parser.py`` (vLLM
``exec``s that one file directly — it does not `import lobes`). Kept as a
package (rather than a loose script) so the pure logic lives in an
independently testable, vllm-free sibling module.

Public modules:

- :mod:`lobes.vllm_plugins._thinking` — pure helper (zero vllm imports,
  unit-tested offline). Computes the effective ``reasoning`` state for a
  chat-completion request from its ``chat_template_kwargs``.
- :mod:`lobes.vllm_plugins.qwen3_thinking_tool_parser` — the vLLM plugin
  itself. Imports vllm at module scope, so it can only be *exercised* offline
  via injected ``sys.modules`` stubs (see ``tests/test_vllm_plugin_thinking.py``);
  it is genuinely loaded only inside the served vLLM container.

See ``docs/`` for the full story on why the served vLLM build's hardcoded
``reasoning=False`` breaks strict tool calling for a thinking model.
"""

from __future__ import annotations
