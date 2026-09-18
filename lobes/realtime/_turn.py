"""Generate-turn request shaping — a pure builder/parser for the voice lane's
``/v1/chat/completions`` call to the fleet gateway (issue #151 t5).

A committed voice turn (issue #151's extension of the #149 ears-only session)
optionally triggers a server-side "think" step: history + a system prompt go
out as one chat/completions request, and the assistant's reply text comes
back to be spoken. This module owns the SHAPE of that round trip — the
request body, the request's URL/headers, and how a response (success or
failure) turns into either a reply string, a distinct tool-call result, or a
named exception. It is a pure, stdlib-only module (``dataclasses``, ``json``,
``typing`` only — never ``httpx``, ``urllib``, or a socket) so it imports and
is fully unit-tested without the ``[realtime]`` extra, exactly like its
siblings :mod:`lobes.realtime._segmenter` (VAD state machine) and
:mod:`lobes.realtime._settings` (env parsing). The actual HTTP call —
opening a connection, awaiting a response, retrying on timeout — belongs to
the route layer (``app.py``, task #151 t6), which is a thin, ``pragma: no
cover`` shell that calls into this module on both ends: build a
:class:`TurnRequest` before the call, hand the raw response to
:func:`parse_turn_response` after it.

Config values this module needs (model, base URL, API key, max_tokens,
temperature, system prompt) are all **explicit parameters** — this module
never imports :mod:`lobes.realtime._settings` and never reads ``os.environ``
itself. The caller (task t6, wiring the live
:class:`~lobes.realtime._settings.Settings` and a
:class:`~lobes.realtime._session.Session`'s history) resolves those values
and passes them through. Conversation history arrives as a plain
``list[dict]`` of ``{"role": ..., "content": ...}`` messages — this module
places no other requirement on where that list came from.

The one exception to "stdlib-only, no sibling imports" is
:mod:`lobes.realtime._session` itself, for two narrow, `pure` reasons (task
#151 t5): the canonical :data:`DEFAULT_SYSTEM_PROMPT` text (aliased, not
duplicated — see "Default system prompt" below) and the FLAT-to-NESTED tool
translators (:func:`~lobes.realtime._session.realtime_tools_to_chat_completions`,
:func:`~lobes.realtime._session.realtime_tool_choice_to_chat_completions` —
see "Tools" below). ``_session`` is itself stdlib-only (no ``httpx``, no
socket), so this does not pull the ``[realtime]`` extra into this module's
import graph; it stays fully unit-testable without it.

Default system prompt — the session's copy wins
--------------------------------------------------------------------------
This module used to carry its own near-duplicate English prompt text, which
had already drifted from :mod:`lobes.realtime._session`'s
``DEFAULT_SYSTEM_PROMPT`` — the copy a live :class:`~lobes.realtime._session.Session`
actually falls back to. :data:`DEFAULT_SYSTEM_PROMPT` here is now a plain
alias of that name, not a second copy, so the two can never disagree again.

Tools — FLAT Realtime shape in, NESTED chat-completions shape on the wire
--------------------------------------------------------------------------
:func:`build_turn_payload` accepts ``tools`` in the same FLAT
``{"type": "function", "name", "description", "parameters"}`` shape
:class:`~lobes.realtime._session.SessionConfig.tools` holds, and nests them
via :func:`~lobes.realtime._session.realtime_tools_to_chat_completions`
before they go on the wire — this module is not a second place that shape
translation is encoded. "The session declared tools" means a non-empty
``tools`` sequence; ``None`` (not declared) and ``()`` (declared empty) both
leave the payload BYTE-IDENTICAL to a call with no ``tools=`` argument at
all — no ``"tools"`` key, no ``"tool_choice"`` key either, even if
``tool_choice`` was passed. A reply's ``tool_calls`` come back through
:func:`parse_turn_response` as a distinct :class:`ToolCallResult` — never a
string, never synthesized as if it were spoken text.

The measured shape this formalizes lives in ``scripts/realtime-voice-loop.py``'s
``think()``: ``model="multimodal"`` (the Gemma 4 12B lane — measured ~1s to a
short reply on this box, versus the 27B ``cortex`` lane's reasoning-trace
latency that a spoken turn cannot afford), ``max_tokens=160``,
``temperature=0.7``, ``chat_template_kwargs={"enable_thinking": False}`` (no
reasoning trace — dead air the speaker never hears), and a spoken-style
system prompt. This module keeps those exact numbers as defaults
(:data:`DEFAULT_MAX_TOKENS`, :data:`DEFAULT_TEMPERATURE`,
:data:`DEFAULT_SYSTEM_PROMPT`) but never hardcodes a default *model* — see
"Model resolution" below.

Model resolution — empty means gateway default-routing, never a fallback lane
--------------------------------------------------------------------------
:mod:`lobes.realtime._settings`'s ``openai_model`` field is ``""`` when
``OPENAI_MODEL`` is unset, and docs/openai-api.md documents the gateway's own
contract for that: "Supply the served model name in `model`, or omit it to
hit the primary." :func:`build_turn_payload` mirrors this exactly — a falsy
``model`` (``""`` or ``None``, including simply not passing the argument)
OMITS the ``"model"`` key from the payload entirely, rather than sending
``"model": ""``. Both are treated identically by the gateway's own
``extract_model()`` (``lobes/gateway/server.py``, which folds an empty string
to ``None`` exactly like a missing key), so this is a style choice, not a
correctness one — omitting matches the documented usage precedent. This
module has NO opinion on which model is "the voice lane's default"; that
policy decision (e.g. "multimodal") belongs one layer up, in whatever code
resolves ``Settings.openai_model`` and decides what to pass as
``build_turn_payload(..., model=...)``.

Failure mapping — role_infeasible is a NAMED, distinct error; never a fallback
--------------------------------------------------------------------------
A ``spark-lobe`` (or similar mesh-brain) deployment shape can drop the
``multimodal``/``senses`` lane entirely — the gateway then answers a chat/
completions request for it with ``404 role_infeasible``
(``lobes/gateway/server.py``'s ``_role_infeasible_body``), optionally naming
the peer that hosts the role via ``hosted_by`` (the opt-in honest-referral
contract, issue #112 t3). :func:`parse_turn_response` detects EXACTLY this
shape (status 404 AND the error object's ``code``/``type`` is
``"role_infeasible"``) and raises :class:`RoleInfeasibleError` carrying
``hosted_by`` — never a placeholder reply string, never a second attempt
against a different model. A caller that does not explicitly catch
:class:`RoleInfeasibleError` sees it propagate: there is no code path in this
module, anywhere, that swallows it. Any OTHER failure (a different 404 like
``model_not_found``, a 5xx, a malformed body) raises the plainer
:class:`TurnResponseError` sibling instead — the two are deliberately
distinguishable so a caller can react to "the lane doesn't exist here"
(``RoleInfeasibleError`` — dial ``hosted_by``, or surface a named session
error) differently from "the call to an existing lane failed"
(``TurnResponseError`` — retry/timeout policy, a t6 concern).

Never logged here
-------------------
This module performs no logging at all — it has no I/O to log around. When a
caller (the session engine, mirroring its own convention) does log around a
call built here, the reply text :func:`parse_turn_response` returns must
never be logged verbatim, only its length; that discipline lives with the
caller, not here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NoReturn

from ._session import DEFAULT_SYSTEM_PROMPT as _SESSION_DEFAULT_SYSTEM_PROMPT
from ._session import realtime_tool_choice_to_chat_completions, realtime_tools_to_chat_completions

# --- defaults, mirroring scripts/realtime-voice-loop.py's think() ----------

# Generous enough for a short spoken reply, small enough to keep the reply
# latency low — a voice turn is dead air until the reply starts.
DEFAULT_MAX_TOKENS = 160
DEFAULT_TEMPERATURE = 0.7

# Alias, not a second copy (task #151 t5): this module previously carried its
# own near-duplicate English prompt text, which had already drifted from
# :data:`lobes.realtime._session.DEFAULT_SYSTEM_PROMPT` (the one a live
# Session actually falls back to — see that module's own docstring). The
# session's copy wins; this name stays exported, unchanged in value, so
# every existing caller/import of ``_turn.DEFAULT_SYSTEM_PROMPT`` (this
# module's own defaults below, and ``tests/test_realtime_turn.py``) keeps
# working without having to know the canonical text now lives one module
# over.
DEFAULT_SYSTEM_PROMPT = _SESSION_DEFAULT_SYSTEM_PROMPT

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


# --- payload builder ---------------------------------------------------------


def build_turn_payload(
    history: list[dict],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Sequence[Mapping[str, object]] | None = None,
    tool_choice: str | Mapping[str, object] | None = None,
) -> dict:
    """Assemble the ``/v1/chat/completions`` JSON body for one voice turn.

    ``history`` is prepended with ``{"role": "system", "content":
    system_prompt}`` and copied into a NEW list — the caller's own list is
    never mutated (the session's history is the caller's to own; this
    function only reads it). No thinking trace: ``chat_template_kwargs``
    always forces ``enable_thinking: False`` — a spoken turn cannot afford
    the latency of a reasoning trace nobody hears (measured honored by
    ``associate`` with tools on 2026-09-18: tool calling and
    ``enable_thinking: false`` are not in tension).

    ``model`` falsy (``""``, ``None``, or simply omitted) OMITS the
    ``"model"`` key entirely rather than sending an empty string — see the
    module docstring's "Model resolution" section for why, and what that
    means for the caller (the gateway default-routes).

    ``tools`` is the SAME FLAT Realtime shape
    :class:`~lobes.realtime._session.SessionConfig.tools` holds
    (``{"type": "function", "name", "description", "parameters"}``) — this
    function nests it via
    :func:`~lobes.realtime._session.realtime_tools_to_chat_completions`
    before it goes in the payload, so no second place encodes that
    translation. "Declared" means a non-empty sequence: ``None`` (not
    declared) and ``()``/``[]`` (declared empty) are both treated as NOT
    declared — the payload gets no ``"tools"`` key and no ``"tool_choice"``
    key either, byte-identical to a call that never mentions tools at all,
    even when a ``tool_choice`` argument was passed alongside them. When
    tools ARE declared, ``tool_choice`` (if not ``None``) is nested via
    :func:`~lobes.realtime._session.realtime_tool_choice_to_chat_completions`
    and included too.
    """
    payload: dict = {}
    if model:
        payload["model"] = model
    payload["messages"] = [{"role": "system", "content": system_prompt}, *history]
    payload["max_tokens"] = max_tokens
    payload["temperature"] = temperature
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        payload["tools"] = realtime_tools_to_chat_completions(tools)
        if tool_choice is not None:
            payload["tool_choice"] = realtime_tool_choice_to_chat_completions(tool_choice)
    return payload


# --- request assembly: URL + headers + body ---------------------------------


def turn_endpoint_url(base_url: str) -> str:
    """``base_url`` (e.g. ``Settings.openai_base_url``) → the full chat/completions URL."""
    return base_url.rstrip("/") + _CHAT_COMPLETIONS_PATH


def turn_request_headers(api_key: str | None) -> dict[str, str]:
    """Request headers for the turn call. ``api_key`` falsy omits ``Authorization``
    entirely (mirrors ``scripts/realtime-voice-loop.py``'s ``_post_json`` helper).
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


@dataclass(frozen=True)
class TurnRequest:
    """A fully-assembled ``/v1/chat/completions`` call: url + headers + JSON body.

    Bundles :func:`turn_endpoint_url`, :func:`turn_request_headers`, and
    :func:`build_turn_payload` so the route layer (task t6) makes one call
    here and then exactly one ``httpx.post(req.url, headers=req.headers,
    json=req.body)`` — still pure: nothing on this object has touched a
    socket.
    """

    url: str
    headers: dict[str, str]
    body: dict


def build_turn_request(
    history: list[dict],
    *,
    base_url: str,
    api_key: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Sequence[Mapping[str, object]] | None = None,
    tool_choice: str | Mapping[str, object] | None = None,
) -> TurnRequest:
    """Convenience wrapper: the complete :class:`TurnRequest` in one call."""
    return TurnRequest(
        url=turn_endpoint_url(base_url),
        headers=turn_request_headers(api_key),
        body=build_turn_payload(
            history,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        ),
    )


# --- named failures -----------------------------------------------------------


class TurnRequestError(RuntimeError):
    """Base for every named failure a generate turn can raise.

    The route layer (task t6) catches this hierarchy and turns each concrete
    subclass into ONE of the session's named ``error`` events — never a
    silent fallback to a placeholder reply or a different lane. Catching
    just this base class still sees every failure mode below.
    """


class RoleInfeasibleError(TurnRequestError):
    """The gateway answered ``404 role_infeasible`` — the requested generate
    lane is not hosted on this deployment shape (e.g. a ``spark-lobe`` shape
    that dropped ``senses``/``multimodal``).

    ``hosted_by`` carries the operator-declared peer origin from the
    gateway's honest-referral body (``lobes.gateway.server._role_infeasible_body``),
    or ``None`` when no peer origin was declared. This is a deliberately
    NAMED, distinct exception from :class:`TurnResponseError` — the
    dedicated type is what lets a caller distinguish "this lane does not
    exist here" from "the call to an existing lane failed", and there is no
    other code path in this module that could turn a role_infeasible 404
    into anything other than this exception (see the module docstring's
    "Failure mapping" section — this is the invariant task #151 t5's
    acceptance criteria pin down: no silent fallback to another lane).
    """

    def __init__(self, message: str, *, hosted_by: str | None = None) -> None:
        super().__init__(message)
        self.hosted_by = hosted_by


class TurnResponseError(TurnRequestError):
    """Any other non-2xx or malformed ``/v1/chat/completions`` response —
    a different 404 (e.g. ``model_not_found``), a 5xx, or a body that is not
    valid JSON / not shaped like a chat/completions response.

    ``hosted_by`` carries the gateway's own peer hint when the error body
    declared one. The ``503 role_unverified`` shape (the mesh boot window,
    ``lobes.gateway.server._role_unverified_body``) does exactly that, and
    before this field a caller could only recover the pending peer's origin
    by pattern-matching the gateway's English message text. It stays a plain
    attribute on this generic type rather than a second
    :class:`RoleInfeasibleError`-style subclass: ``role_unverified`` means
    "not yet", which a caller handles like any other transient failure, while
    ``role_infeasible``'s terminal "never" is the one that earns its own
    type.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, hosted_by: str | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.hosted_by = hosted_by


# --- the tool-call result: a distinct type, never confusable with text -------


@dataclass(frozen=True)
class ToolCallResult:
    """The generate backend asked the client to run a tool — never text to speak.

    :func:`parse_turn_response` returns this INSTEAD of a ``str`` when
    ``message.tool_calls`` is present in the reply — a separate dataclass, not
    a string carrying some sentinel, so a caller cannot accidentally hand this
    to a TTS stage and speak it: the two return shapes are structurally
    distinguishable, e.g. ``isinstance(result, ToolCallResult)``.

    ``call_id``, ``name`` and ``arguments`` are the FIRST tool call of the
    reply, carried through byte-for-byte (``arguments`` stays the raw JSON
    *string* the backend sent — this module never parses it, exactly like
    :class:`~lobes.realtime._session.FunctionCallOutput` never parses a
    client's tool result). A reply's bridge tracks one outstanding call at a
    time, so only the first is surfaced this way; ``tool_call_count`` is the
    number of tool calls the reply actually carried (``1`` for the common
    case), so a caller that wants to log/react to "the model asked for N
    things at once" still can without this module surfacing more than one.
    """

    call_id: str
    name: str
    arguments: str
    tool_call_count: int = 1


def assistant_tool_call_message(result: ToolCallResult) -> dict:
    """The ``{"role": "assistant", ...}`` history entry for a surfaced tool call.

    Chat-completions shape: ``content`` is ``None`` (an assistant turn that
    calls a tool carries no spoken text) and ``tool_calls`` holds exactly the
    one call :class:`ToolCallResult` surfaced, with ``arguments`` passed
    through as the same JSON string — never re-serialized, never
    re-interpreted.
    """
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": result.call_id,
                "type": "function",
                "function": {"name": result.name, "arguments": result.arguments},
            }
        ],
    }


def tool_result_message(call_id: str, content: str) -> dict:
    """The ``{"role": "tool", ...}`` history entry for a tool's result.

    ``call_id`` is the :class:`ToolCallResult`/``FunctionCallOutput`` handle
    this result answers; ``content`` is the caller's result text verbatim —
    this module has no opinion on its shape (a JSON string by convention,
    same as :class:`~lobes.realtime._session.FunctionCallOutput.output`, but
    never parsed here).
    """
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# --- response parsing + failure mapping --------------------------------------


def _load_json_object(body: bytes) -> dict | None:
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _error_object(data: dict | None) -> dict | None:
    error = data.get("error") if data else None
    return error if isinstance(error, dict) else None


def parse_turn_response(status_code: int, body: bytes) -> str | ToolCallResult:
    """Turn a ``/v1/chat/completions`` HTTP response into a reply or a tool call.

    On any non-200 status: raises :class:`RoleInfeasibleError` when the body
    is the gateway's ``role_infeasible`` shape (status 404 AND the error
    object's ``code`` or ``type`` is ``"role_infeasible"``), carrying
    ``hosted_by`` through unchanged; raises the plainer
    :class:`TurnResponseError` for every other non-200 status (a different
    404, a 5xx, or a non-JSON body).

    On status 200: raises :class:`TurnResponseError` if the body is not
    valid JSON or is not shaped like a chat/completions response
    (``choices[0].message`` missing, ``content`` present but not a string, or
    a malformed ``tool_calls``).

    When ``message.tool_calls`` is present and non-empty, returns a
    :class:`ToolCallResult` for the FIRST call — this is checked BEFORE
    ``content`` is even looked at, so a reply that carries both (some
    backends echo empty/partial text alongside a tool call) is never
    mistaken for text to synthesize.

    Otherwise (no ``tool_calls``): a ``null``/absent ``content`` is NOT an
    error — it returns ``""``, mirroring
    ``scripts/realtime-voice-loop.py``'s ``think()``
    (``(msg.get("content") or "").strip()``). Otherwise returns the reply
    text with surrounding whitespace stripped.
    """
    data = _load_json_object(body)

    if status_code != 200:
        _raise_for_error_status(status_code, data)

    if data is None:
        raise TurnResponseError(
            "generate backend returned a non-JSON response", status_code=status_code
        )
    return _extract_reply(data)


def _raise_for_error_status(status_code: int, data: dict | None) -> NoReturn:
    """Always raises — the named error a non-200 generate response deserves.

    Split out of :func:`parse_turn_response` so the happy path reads as three
    lines: the branching that distinguishes a dropped LANE from an ordinary
    backend failure is the only genuinely intricate part, and it is all here.
    """
    error = _error_object(data)
    code = error.get("code") if error else None
    kind = error.get("type") if error else None
    message = (error.get("message") if error else None) or (
        f"generate backend returned HTTP {status_code}"
    )
    hosted_by = error.get("hosted_by") if error else None
    hosted_by = hosted_by if isinstance(hosted_by, str) and hosted_by else None
    # A 404 whose error object names role_infeasible means this box does not
    # host the lane — a different fact from "the backend broke", and the one
    # that must carry `hosted_by` so the caller can name the peer instead of
    # silently falling back to another lane.
    if status_code == 404 and "role_infeasible" in (code, kind):
        raise RoleInfeasibleError(message, hosted_by=hosted_by)
    # Every OTHER shape keeps the hint too when the gateway sent one — a 503
    # `role_unverified` names the mesh member that announced the lane, and
    # that origin is a field here, not something to grep out of the message.
    raise TurnResponseError(message, status_code=status_code, hosted_by=hosted_by)


def _extract_reply(data: dict) -> str | ToolCallResult:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TurnResponseError("generate response missing 'choices'")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise TurnResponseError("generate response missing choices[0].message")
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return _extract_tool_call(tool_calls)
    content = message.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise TurnResponseError("generate response 'content' is not a string")
    return content.strip()


def _extract_tool_call(tool_calls: object) -> ToolCallResult:
    """The FIRST call of ``message.tool_calls``, as a :class:`ToolCallResult`.

    Raises :class:`TurnResponseError` on any shape defect — a tool call this
    module cannot name (missing ``id``/``function.name``, or a
    non-string ``arguments``) is a malformed backend response, exactly like a
    missing ``choices[0].message`` is; it is never silently dropped in favor
    of falling through to the text path.
    """
    if not isinstance(tool_calls, list) or not tool_calls:
        raise TurnResponseError("generate response 'tool_calls' is not a non-empty list")
    first = tool_calls[0]
    if not isinstance(first, dict):
        raise TurnResponseError("generate response tool_calls[0] is not an object")
    call_id = first.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise TurnResponseError("generate response tool_calls[0] missing 'id'")
    function = first.get("function")
    if not isinstance(function, dict):
        raise TurnResponseError("generate response tool_calls[0] missing 'function'")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise TurnResponseError("generate response tool_calls[0].function missing 'name'")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise TurnResponseError(
            "generate response tool_calls[0].function.arguments is not a string"
        )
    return ToolCallResult(
        call_id=call_id, name=name, arguments=arguments, tool_call_count=len(tool_calls)
    )
