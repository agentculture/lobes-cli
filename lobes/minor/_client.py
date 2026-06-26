"""Stdlib-only OpenAI chat-completions client for the lobes minor backend.

Mirrors the urllib idiom used in :mod:`lobes.assess` — no third-party
dependencies, no reads or writes to .env / docker-compose / any repo file.

Public API
----------
chat_completion(prompt, *, base_url, model, system=None,
                timeout=60, max_tokens=None, temperature=None,
                logprobs=None, top_logprobs=None) -> dict
    POST an OpenAI chat-completions request and return the parsed JSON dict.

chat_text(...) -> str
    Thin wrapper: return just the assistant message content string.

completions_echo(prompt, continuation, *, base_url, model,
                 logprobs=1, timeout=60) -> dict
    POST to /v1/completions with echo=true for full-sequence token scoring.

gateway_supports_echo(*, base_url, model, timeout=10) -> bool
    Capability probe: returns True only when the gateway routes /v1/completions
    and responds with valid per-token logprobs. Never raises.
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
    extra_body: dict | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
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
    extra_body:
        Optional extra top-level request fields merged into the body — e.g.
        ``{"chat_template_kwargs": {"enable_thinking": False}}`` to turn off a
        thinking-mode model's ``<think>`` trace. Explicit ``max_tokens`` /
        ``temperature`` args take precedence over the same keys here.
    logprobs:
        If set, forwarded as ``logprobs`` in the request body. Enables
        per-token log-probability output in the response.
    top_logprobs:
        If set, forwarded as ``top_logprobs`` in the request body. Specifies
        how many of the most likely tokens to return at each position (0–20).
        Requires ``logprobs=True``.

    Returns
    -------
    dict
        The full parsed JSON response from the server.  When logprobs are
        requested, ``choices[0]["logprobs"]["content"][i]["top_logprobs"]``
        contains the ranked tokens at each output position.
    """
    url = base_url.rstrip("/") + "/chat/completions"

    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict = {"model": model, "messages": messages}
    if extra_body:
        payload.update(extra_body)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if top_logprobs is not None:
        payload["top_logprobs"] = top_logprobs

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


def completions_echo(
    prompt: str,
    continuation: str,
    *,
    base_url: str,
    model: str,
    logprobs: int = 1,
    timeout: int = 60,
) -> dict:
    """POST to ``/v1/completions`` with ``echo=true`` for full-sequence token scoring.

    This uses the OpenAI legacy completions endpoint that vLLM exposes.  By
    concatenating *prompt* and *continuation* and setting ``echo=true`` with
    ``max_tokens=0``, the server scores every token in the combined string
    without generating new ones.

    Parameters
    ----------
    prompt:
        The prefix string (e.g. a question or context).
    continuation:
        The string to score, appended to *prompt* before sending.
    base_url:
        OpenAI-compatible base URL, e.g. ``"http://localhost:8000/v1"``.
        A trailing slash is tolerated and stripped.
    model:
        The model identifier to pass in the request body.
    logprobs:
        Number of top log-probability candidates to return per token
        (default ``1``).
    timeout:
        Socket timeout in seconds (default 60).

    Returns
    -------
    dict
        The full parsed JSON response.  Callers read
        ``choices[0]["logprobs"]["token_logprobs"]`` (list of floats) and
        ``choices[0]["logprobs"]["tokens"]`` (list of strings).
    """
    url = base_url.rstrip("/") + "/completions"
    payload: dict = {
        "model": model,
        "prompt": prompt + continuation,
        "echo": True,
        "logprobs": logprobs,
        "max_tokens": 0,
        "temperature": 0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # local endpoint only
        return json.load(resp)


def gateway_supports_echo(
    *,
    base_url: str,
    model: str,
    timeout: int = 10,
) -> bool:
    """Probe whether the gateway routes ``/v1/completions`` with echo support.

    Issues a tiny :func:`completions_echo` call and returns ``True`` only when
    the response carries a non-empty ``token_logprobs`` list.  Returns ``False``
    — and never raises — on HTTP errors (e.g. 404/501), connection failures,
    JSON decode errors, or a missing / empty logprobs structure.

    Parameters
    ----------
    base_url:
        OpenAI-compatible base URL, e.g. ``"http://localhost:8000/v1"``.
    model:
        The model identifier to pass in the probe request.
    timeout:
        Socket timeout in seconds (default 10).

    Returns
    -------
    bool
        ``True`` if echo is available and functional; ``False`` otherwise.
    """
    try:
        result = completions_echo(
            "Ping",
            " pong",
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        token_logprobs = result["choices"][0]["logprobs"]["token_logprobs"]
        return isinstance(token_logprobs, list) and len(token_logprobs) > 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, IndexError):
        return False
