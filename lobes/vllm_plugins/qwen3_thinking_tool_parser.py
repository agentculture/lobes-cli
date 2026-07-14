"""Think-aware Qwen3 tool-call parser plugin, loaded by vLLM itself.

vLLM's plugin loader ``exec``s this file directly (per
``--tool-parser-plugin /opt/lobes/qwen3_thinking_tool_parser.py``); it is
never imported by the ``lobes`` CLI/gateway package. HARD CONSTRAINT: inside
the vllm-primary container this is the ONLY file that exists — the upstream
``vllm/vllm-openai`` image has no ``lobes`` package installed and the compose
template mounts nothing else — so this module must stay fully
self-contained: it may import ``vllm`` and the stdlib, and nothing under
``lobes.*`` (``tests/test_vllm_plugin_thinking.py`` pins this with an AST
check over the materialised source).

Why this override exists: the served vLLM build (0.23.1rc1.dev672) hardcodes
``reasoning=False`` when building the structural-tag grammar for strict tool
calling (``vllm/parser/abstract_parser.py: _apply_structural_tag`` calls
``self._tool_parser.get_structural_tag(request, reasoning=False)``). For the
Qwen3.6 thinking model that produces a grammar that cannot accept the
``</think>`` special token, so strict tool-call requests 500. Passing
``reasoning=True`` fixes the grammar, but then REQUIRES a closing
``</think>`` in the output — so ``reasoning`` must track whether the request
actually has thinking enabled, not just be flipped on unconditionally.
"""

from __future__ import annotations

import inspect
from typing import Any

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers import qwen3_engine_tool_parser as _qwen3_engine_tool_parser_module

#: The server default is thinking ON — a request only turns it off by
#: explicitly setting ``chat_template_kwargs={"enable_thinking": False}``.
_DEFAULT_REASONING = True


def effective_reasoning(request: Any) -> bool:
    """Return whether ``request`` has thinking mode effectively enabled.

    ``request`` may be an attribute-style object (vLLM's
    ``ChatCompletionRequest``) or a plain ``dict`` — either way it is expected
    to carry a ``chat_template_kwargs`` field (``dict | None``).

    - ``chat_template_kwargs`` absent or ``None`` -> ``True`` (server default:
      thinking on, nothing overrides it).
    - ``chat_template_kwargs`` present but without an ``enable_thinking`` key
      -> ``True`` (same reasoning: no override present).
    - ``enable_thinking`` present -> its boolean value, verbatim.
    """
    if isinstance(request, dict):
        chat_template_kwargs = request.get("chat_template_kwargs")
    else:
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None)

    if not chat_template_kwargs:
        return _DEFAULT_REASONING

    if "enable_thinking" not in chat_template_kwargs:
        return _DEFAULT_REASONING

    return bool(chat_template_kwargs["enable_thinking"])


# --- Loud import-surface assert --------------------------------------------
# Pinned to the served image, vLLM 0.23.1rc1.dev672's
# vllm/tool_parsers/{abstract_tool_parser,qwen3_engine_tool_parser}.py surface.
# If Qwen3EngineToolParser disappears or its get_structural_tag loses the
# keyword-only `reasoning` param upstream, this override would silently stop
# doing anything (the subclass would still "work" but no longer correct the
# hardcoded reasoning=False) — so boot must fail loudly here instead.
Qwen3EngineToolParser = getattr(_qwen3_engine_tool_parser_module, "Qwen3EngineToolParser", None)

if Qwen3EngineToolParser is None:
    raise RuntimeError(
        "qwen3_thinking_tool_parser: vLLM import-surface mismatch — "
        "vllm.tool_parsers.qwen3_engine_tool_parser.Qwen3EngineToolParser does "
        "not exist. This plugin is pinned to the served vLLM image "
        "(0.23.1rc1.dev672); refusing to load rather than silently serving "
        "non-think-aware structural tags for strict tool calling."
    )

_structural_tag_sig = inspect.signature(Qwen3EngineToolParser.get_structural_tag)
_reasoning_param = _structural_tag_sig.parameters.get("reasoning")

if _reasoning_param is None or _reasoning_param.kind != inspect.Parameter.KEYWORD_ONLY:
    raise RuntimeError(
        "qwen3_thinking_tool_parser: vLLM import-surface mismatch — "
        "Qwen3EngineToolParser.get_structural_tag no longer has a keyword-only "
        "`reasoning` parameter. This plugin is pinned to the served vLLM image "
        "(0.23.1rc1.dev672); refusing to load rather than silently serving "
        "non-think-aware structural tags for strict tool calling."
    )


@ToolParserManager.register_module("qwen3_coder_thinking")
class Qwen3ThinkingToolParser(Qwen3EngineToolParser):
    """``Qwen3EngineToolParser`` with a request-aware ``reasoning`` flag.

    vLLM always calls ``get_structural_tag(request, reasoning=False)``
    (abstract_parser._apply_structural_tag), which breaks strict tool calling
    for a thinking model. Override so ``reasoning`` tracks the request's own
    thinking state instead of the caller's hardcoded ``False``.
    """

    def get_structural_tag(self, request, *, reasoning: bool = False):
        return super().get_structural_tag(request, reasoning=effective_reasoning(request))
