"""Stdlib-only OpenAI chat-completions client for the lobes minor backend.

Mirrors the urllib idiom used in :mod:`lobes.assess` — no third-party
dependencies, no reads or writes to .env / docker-compose / any repo file.

Public API
----------
chat_completion(prompt, *, base_url, model, system=None,
                timeout=60, max_tokens=None, temperature=None) -> dict
    POST an OpenAI chat-completions request and return the parsed JSON dict.

chat_text(...) -> str
    Thin wrapper: return just the assistant message content string.
"""

from __future__ import annotations

import json
import urllib.request


def chat_completion(
    prompt: str,
    *,
    base_url: str,
    model: str,
    system: str | None = None,
    timeout: int = 60,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict:
    """POST an OpenAI chat-completions request; return the parsed JSON dict.

    Parameters
    ----------
    prompt:
        The user message content.
    base_url:
        OpenAI-compatible base URL, e.g. ``"http://localhost:8000/v1"``.
        A trailing slash is tolerated and stripped.
    model:
        The model identifier to pass in the request body.
    system:
        Optional system message prepended before the user turn.
    timeout:
        Socket timeout in seconds (default 60).
    max_tokens:
        If set, forwarded as ``max_tokens`` in the request body.
    temperature:
        If set, forwarded as ``temperature`` in the request body.

    Returns
    -------
    dict
        The full parsed JSON response from the server.
    """
    url = base_url.rstrip("/") + "/chat/completions"

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {"model": model, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # local endpoint only
        return json.load(resp)


def chat_text(
    prompt: str,
    *,
    base_url: str,
    model: str,
    system: str | None = None,
    timeout: int = 60,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Return just the assistant message content from a chat-completions call.

    All parameters are forwarded to :func:`chat_completion`; see that function
    for full documentation.
    """
    result = chat_completion(
        prompt,
        base_url=base_url,
        model=model,
        system=system,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return result["choices"][0]["message"]["content"]
