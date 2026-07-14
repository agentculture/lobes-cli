"""lobes vllm_plugins — custom vLLM engine plugin modules shipped with the wheel.

Every module here runs INSIDE a vLLM server container, loaded by vLLM via a
``--*-plugin`` flag (e.g. ``--tool-parser-plugin``) that names a FILE PATH,
never a Python import path — lobes itself never imports these modules. `lobes
init`'s fleet scaffold instead reads each module's packaged SOURCE TEXT
(``importlib.resources`` — single source of truth in the package, not a
``lobes/templates/`` copy) and writes it verbatim into the deployment dir next
to ``docker-compose.yml``, where the relevant vLLM service mounts it read-only.
See ``lobes.runtime._compose`` (``PLUGIN_PACKAGE`` / ``write_plugin_file``) for
the materialisation mechanism, and ``lobes/templates/fleet/docker-compose.yml``
(the ``vllm-primary`` service) for the mount + ``--tool-parser-plugin`` wiring.

Keep modules here syntactically self-contained: they must be valid standalone
Python files vLLM can load by path inside ITS OWN container image (which has
vllm/torch installed), not necessarily importable from the base lobes wheel's
(vllm-free) environment.

Public surface:

- :mod:`lobes.vllm_plugins.qwen3_thinking_tool_parser` — registers the
  ``qwen3_coder_thinking`` tool-call parser (a reasoning-aware variant of the
  upstream ``qwen3_coder`` parser) for the cortex/main generate lane.
"""

from __future__ import annotations

__all__: list[str] = []
