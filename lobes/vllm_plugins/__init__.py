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

HARD CONSTRAINT — plugin modules must be fully self-contained: inside the vLLM
container the mounted plugin file is the ONLY lobes-authored file that exists
(the upstream ``vllm/vllm-openai`` image has no ``lobes`` package installed),
so a plugin may import ``vllm`` and the stdlib and NOTHING under ``lobes.*``.
``tests/test_vllm_plugin_thinking.py`` pins this with an AST check over the
materialised source. It follows that plugins are only *exercised* offline via
injected ``sys.modules`` stubs; they genuinely load only inside the served
vLLM container.

Public modules:

- :mod:`lobes.vllm_plugins.qwen3_thinking_tool_parser` — registers the
  ``qwen3_coder_thinking`` tool-call parser (a reasoning-aware variant of the
  upstream ``qwen3_coder`` parser) for the cortex/main generate lane, deriving
  the structural-tag grammar's ``reasoning`` flag from the request's own
  effective ``enable_thinking`` state.

See ``docs/`` for the full story on why the served vLLM build's hardcoded
``reasoning=False`` breaks strict tool calling for a thinking model.
"""

from __future__ import annotations

__all__: list[str] = []
