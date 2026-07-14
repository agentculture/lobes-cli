"""lobes.vllm_plugins — vLLM tool-parser plugins for the served Qwen3.6 model.

This package is **not** imported by the `lobes` CLI/gateway at runtime; it is
data shipped in the wheel that the vLLM *container* points at via
``--tool-parser-plugin`` (a flag that names a FILE PATH, never a Python import
path — vLLM ``exec``s that one file directly; it does not `import lobes`).
`lobes init`'s fleet scaffold reads each module's packaged SOURCE TEXT
(``importlib.resources`` — single source of truth in the package, not a
``lobes/templates/`` copy) and writes it verbatim into the deployment dir next
to ``docker-compose.yml``, where the relevant vLLM service mounts it read-only.
See ``lobes.runtime._compose`` (``PLUGIN_PACKAGE`` / ``write_plugin_file``) for
the materialisation mechanism, and ``lobes/templates/fleet/docker-compose.yml``
(the ``vllm-primary`` service) for the mount + ``--tool-parser-plugin`` wiring.

Keep plugin modules syntactically self-contained: they must be valid standalone
Python files vLLM can load by path inside ITS OWN container image (which has
vllm/torch installed), not necessarily importable from the base lobes wheel's
(vllm-free) environment. The package split exists so the pure logic lives in an
independently testable, vllm-free sibling module.

Public modules:

- :mod:`lobes.vllm_plugins._thinking` — pure helper (zero vllm imports,
  unit-tested offline). Computes the effective ``reasoning`` state for a
  chat-completion request from its ``chat_template_kwargs``.
- :mod:`lobes.vllm_plugins.qwen3_thinking_tool_parser` — the vLLM plugin
  itself: registers the ``qwen3_coder_thinking`` tool-call parser (a
  reasoning-aware variant of the upstream ``qwen3_coder`` parser) for the
  cortex/main generate lane. Imports vllm at module scope, so it can only be
  *exercised* offline via injected ``sys.modules`` stubs (see
  ``tests/test_vllm_plugin_thinking.py``); it is genuinely loaded only inside
  the served vLLM container.

See ``docs/`` for the full story on why the served vLLM build's hardcoded
``reasoning=False`` breaks strict tool calling for a thinking model.
"""

from __future__ import annotations

__all__: list[str] = []
