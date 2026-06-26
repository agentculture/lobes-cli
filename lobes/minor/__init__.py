"""lobes.minor — stdlib urllib OpenAI chat-completions client and governance.

Exposes the two public helpers so callers can simply write::

    from lobes.minor import chat_completion, chat_text

Also re-exports the governance API::

    from lobes.minor import decide, Decision, ROLE, ALLOWED, FORBIDDEN, ESCALATION_CONDITIONS
"""

from lobes.minor._client import (
    chat_completion,
    chat_text,
    completions_echo,
    gateway_supports_echo,
)
from lobes.minor.governance import (
    ALLOWED,
    ESCALATION_CONDITIONS,
    FORBIDDEN,
    ROLE,
    Decision,
    decide,
)

__all__ = [
    "chat_completion",
    "chat_text",
    "completions_echo",
    "gateway_supports_echo",
    # governance
    "ROLE",
    "ALLOWED",
    "FORBIDDEN",
    "ESCALATION_CONDITIONS",
    "Decision",
    "decide",
]
