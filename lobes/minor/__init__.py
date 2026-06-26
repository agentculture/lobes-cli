"""lobes.minor — stdlib urllib OpenAI chat-completions client.

Exposes the two public helpers so callers can simply write::

    from lobes.minor import chat_completion, chat_text
"""

from lobes.minor._client import chat_completion, chat_text

__all__ = ["chat_completion", "chat_text"]
