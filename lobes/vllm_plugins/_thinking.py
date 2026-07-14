"""Pure helper: the effective ``reasoning`` state of a chat-completion request.

Zero vllm imports — this module must import and unit-test cleanly in the
offline CI environment (no vllm installed there). See
:mod:`lobes.vllm_plugins.qwen3_thinking_tool_parser` for the plugin that
consumes it.
"""

from __future__ import annotations

from typing import Any

#: The server default is thinking ON — a request only turns it off by
#: explicitly setting ``chat_template_kwargs={"enable_thinking": False}``.
_DEFAULT_REASONING = True


def effective_reasoning(request: Any) -> bool:
    """Return whether ``request`` has thinking mode effectively enabled.

    ``request`` may be an attribute-style object (e.g. vLLM's
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
