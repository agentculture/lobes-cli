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

Why this override exists: the served vLLM builds (the 0.26.1rc1 nightly and
0.31.0; first observed on 0.23.1rc1.dev672) hardcode ``reasoning=False`` when
building the structural-tag grammar for strict tool calling
(``vllm/parser/abstract_parser.py`` calls
``self._tool_parser.get_structural_tag(request, reasoning=False)``; 0.31.0 adds
``strict_level=self.tool_strict_level`` to that call). For the
Qwen3.6 thinking model that produces a grammar that cannot accept the
``</think>`` special token, so strict tool-call requests 500. Passing
``reasoning=True`` fixes the grammar, but then REQUIRES a closing
``</think>`` in the output — so ``reasoning`` must track whether the request
actually has thinking enabled, not just be flipped on unconditionally.

That per-request derivation is right only on an engine that applies the
grammar from the first generated token. vLLM 0.30.0 and later (measured on
0.30.0, 0.31.0 and 0.32.0rc1, #310 deviation d5) start it only AFTER reasoning
ends, so a thinking-prefixed grammar sits inside its thinking tag forever:
every tool call is accepted as thinking text, EOS stays masked, and a strict
request repeats the call until ``max_tokens``. There the plugin passes
``reasoning=False`` for every request. The boundary is the measured one, not a
proven upstream change point: 0.27-0.29 are unmeasured and keep the
derivation, and the vLLM change behind it is not isolated.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any

import vllm
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


#: First engine measured to start the structured-output grammar after reasoning.
_POST_REASONING_GRAMMAR_FROM = (0, 30)


def grammar_starts_after_reasoning(version: str) -> bool:
    """Return whether vLLM ``version`` applies the grammar only after reasoning.

    Parses the leading ``major.minor`` (``"0.31.0+cu130"``, ``"0.31+cu130"``,
    ``"0.30rc1"``, ``"0.32.0rc1.dev301+g..."`` and ``"0.26.1rc1.dev942+g..."``
    all parse) and raises ``RuntimeError`` on anything else, so an engine whose
    version cannot be read fails at boot instead of silently getting the wrong
    grammar. A dev build is classified by its base release: a ``0.29.x`` nightly
    keeps the per-request derivation even if it already carries the (not
    isolated) upstream change, which is why the chosen mode is logged at import.
    """
    match = re.match(r"(\d+)\.(\d+)(?!\d)", version or "")
    if match is None:
        raise RuntimeError(
            "qwen3_thinking_tool_parser: cannot read the vLLM version "
            f"({version!r}), so it cannot tell whether the engine applies the "
            "strict-tool grammar before or after reasoning; refusing to load."
        )
    return (int(match.group(1)), int(match.group(2))) >= _POST_REASONING_GRAMMAR_FROM


_VLLM_VERSION = getattr(vllm, "__version__", "")
_GRAMMAR_AFTER_REASONING = grammar_starts_after_reasoning(_VLLM_VERSION)

logging.getLogger("vllm.lobes.qwen3_coder_thinking").info(
    "qwen3_coder_thinking: vLLM %s -> structural-tag reasoning flag %s",
    _VLLM_VERSION,
    (
        "passed through from vLLM (grammar starts after reasoning)"
        if _GRAMMAR_AFTER_REASONING
        else "derived per request from enable_thinking"
    ),
)


# --- Loud import-surface assert --------------------------------------------
# Pinned to the get_structural_tag surface of the nightly (0.26.1rc1:
# `reasoning` only) and vLLM 0.31.0 (`reasoning` + `strict_level`), both in
# vllm/tool_parsers/{abstract_tool_parser,qwen3_engine_tool_parser}.py.
# If Qwen3EngineToolParser disappears, `reasoning` stops being keyword-only,
# `strict_level` stops being keyword-only, or a new REQUIRED parameter appears,
# this override would silently stop correcting the hardcoded reasoning=False
# (or crash at the first strict call) — so boot must fail loudly here instead.
Qwen3EngineToolParser = getattr(_qwen3_engine_tool_parser_module, "Qwen3EngineToolParser", None)

if Qwen3EngineToolParser is None:
    raise RuntimeError(
        "qwen3_thinking_tool_parser: vLLM import-surface mismatch — "
        "vllm.tool_parsers.qwen3_engine_tool_parser.Qwen3EngineToolParser does "
        "not exist. This plugin supports vLLM 0.26.1rc1 (nightly) and 0.31.0; "
        "refusing to load rather than silently serving "
        "non-think-aware structural tags for strict tool calling."
    )

_structural_tag_sig = inspect.signature(Qwen3EngineToolParser.get_structural_tag)
_reasoning_param = _structural_tag_sig.parameters.get("reasoning")

if _reasoning_param is None or _reasoning_param.kind != inspect.Parameter.KEYWORD_ONLY:
    raise RuntimeError(
        "qwen3_thinking_tool_parser: vLLM import-surface mismatch — "
        "Qwen3EngineToolParser.get_structural_tag no longer has a keyword-only "
        "`reasoning` parameter. This plugin supports vLLM 0.26.1rc1 (nightly) "
        "and 0.31.0; refusing to load rather than silently serving "
        "non-think-aware structural tags for strict tool calling."
    )

_strict_level_param = _structural_tag_sig.parameters.get("strict_level")

if _strict_level_param is not None and _strict_level_param.kind != inspect.Parameter.KEYWORD_ONLY:
    raise RuntimeError(
        "qwen3_thinking_tool_parser: vLLM import-surface mismatch — "
        "Qwen3EngineToolParser.get_structural_tag has a `strict_level` "
        "parameter that is not keyword-only, so it cannot be forwarded safely; "
        "refusing to load."
    )

_unknown_required = [
    name
    for name, param in list(_structural_tag_sig.parameters.items())[2:]  # past self, request
    if name not in ("reasoning", "strict_level")
    and param.default is inspect.Parameter.empty
    and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
]

if _unknown_required:
    raise RuntimeError(
        "qwen3_thinking_tool_parser: vLLM import-surface mismatch — "
        "Qwen3EngineToolParser.get_structural_tag gained required parameter(s) "
        f"{_unknown_required} this override does not forward; refusing to load "
        "rather than failing at the first strict tool call."
    )

#: Sentinel: `strict_level` is forwarded only when the caller supplied it, since
#: the nightly's parent has no such parameter (and 0.31.0's own default applies
#: when it is omitted, so we never have to import ToolStrictLevel).
_UNSET: Any = object()


@ToolParserManager.register_module("qwen3_coder_thinking")
class Qwen3ThinkingToolParser(Qwen3EngineToolParser):
    """``Qwen3EngineToolParser`` with an engine- and request-aware ``reasoning`` flag.

    vLLM always calls ``get_structural_tag(request, reasoning=False)``
    (abstract_parser._apply_structural_tag). On an engine that applies the
    grammar from the first token that breaks strict tool calling for a
    thinking model, so ``reasoning`` tracks the request's own thinking state.
    On an engine that starts the grammar after reasoning (vLLM >= 0.30.0) the
    grammar must not include the thinking block, so the caller's own
    ``reasoning`` (``False`` from every engine measured) is passed through.
    0.31.0 also passes ``strict_level``; it is forwarded untouched when present
    and left out when absent (nightly parent has no such parameter).
    """

    def get_structural_tag(self, request, *, reasoning: bool = False, strict_level=_UNSET):
        extra = {} if strict_level is _UNSET else {"strict_level": strict_level}
        derived = reasoning if _GRAMMAR_AFTER_REASONING else effective_reasoning(request)
        return super().get_structural_tag(request, reasoning=derived, **extra)
