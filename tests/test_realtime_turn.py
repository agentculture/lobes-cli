"""Tests for the generate-turn payload builder (stdlib-only; no [realtime] extra).

Mirrors :mod:`tests.test_realtime_segmenter`'s house style: every case drives
:mod:`lobes.realtime._turn` directly with plain Python values — no httpx, no
FastAPI, no real gateway. The ``TestRoleInfeasibleMapping`` class is the one
gating acceptance criterion 2 (task #151 t5): a fake 404 ``role_infeasible``
body must map to a named, distinct exception carrying ``hosted_by``, and nothing
in this module may swallow it into a fallback reply string.
"""

from __future__ import annotations

import json

import pytest

from lobes.realtime._turn import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    RoleInfeasibleError,
    ToolCallResult,
    TurnRequest,
    TurnRequestError,
    TurnResponseError,
    assistant_tool_call_message,
    build_turn_payload,
    build_turn_request,
    parse_turn_response,
    tool_result_message,
    turn_endpoint_url,
    turn_request_headers,
)


def _role_infeasible_body(hosted_by: str | None = None) -> bytes:
    """A fake gateway 404 body, shaped exactly like
    :func:`lobes.gateway.server._role_infeasible_body` — the real function
    this module's mapping logic must stay compatible with.
    """
    error = {
        "message": "The model `multimodal` is not feasible on this machine.",
        "type": "role_infeasible",
        "code": "role_infeasible",
    }
    if hosted_by:
        error["hosted_by"] = hosted_by
    return json.dumps({"error": error}).encode("utf-8")


def _chat_completion_body(content: str | None = "hello there") -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode("utf-8")


# --- build_turn_payload: acceptance criterion 1 ----------------------------


class TestBuildTurnPayload:
    def test_emits_enable_thinking_false(self) -> None:
        payload = build_turn_payload([], model="multimodal")
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_emits_the_resolved_model(self) -> None:
        payload = build_turn_payload([], model="multimodal")
        assert payload["model"] == "multimodal"

    def test_empty_model_omits_the_model_key_entirely(self) -> None:
        # An empty OPENAI_MODEL means the gateway default-routes — the OpenAI
        # docs (docs/openai-api.md) say "Supply the served model name in
        # `model`, or omit it to hit the primary", and the gateway's own
        # extract_model() treats "" exactly like a missing key. Sending
        # "model": "" would also work (extract_model() folds both to None)
        # but omitting the key is what the documented contract describes.
        payload = build_turn_payload([], model="")
        assert "model" not in payload

    def test_none_model_omits_the_model_key_entirely(self) -> None:
        payload = build_turn_payload([], model=None)
        assert "model" not in payload

    def test_no_model_argument_also_omits_the_key(self) -> None:
        # No hidden default lane (e.g. no baked-in "multimodal") — an
        # unspecified model is exactly as unspecified as an explicit "".
        payload = build_turn_payload([])
        assert "model" not in payload

    def test_system_prompt_is_the_first_message(self) -> None:
        payload = build_turn_payload(
            [{"role": "user", "content": "hi"}],
            model="multimodal",
            system_prompt="You are terse.",
        )
        assert payload["messages"][0] == {"role": "system", "content": "You are terse."}

    def test_history_is_appended_after_the_system_prompt_in_order(self) -> None:
        history = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "and 3+3"},
        ]
        payload = build_turn_payload(history, model="multimodal")
        assert payload["messages"][1:] == history

    def test_does_not_mutate_the_caller_history_list(self) -> None:
        history = [{"role": "user", "content": "hi"}]
        original = list(history)
        build_turn_payload(history, model="multimodal")
        assert history == original

    def test_default_max_tokens_and_temperature_match_the_measured_voice_loop_values(
        self,
    ) -> None:
        # scripts/realtime-voice-loop.py's think() measured max_tokens=160,
        # temperature=0.7 on this box; this module's defaults mirror that.
        payload = build_turn_payload([], model="multimodal")
        assert payload["max_tokens"] == DEFAULT_MAX_TOKENS == 160
        assert payload["temperature"] == DEFAULT_TEMPERATURE == 0.7

    def test_max_tokens_and_temperature_are_overridable(self) -> None:
        payload = build_turn_payload([], model="multimodal", max_tokens=32, temperature=0.1)
        assert payload["max_tokens"] == 32
        assert payload["temperature"] == 0.1

    def test_default_system_prompt_is_spoken_style(self) -> None:
        # Not asserting the exact wording (an implementation detail) — only
        # the contract callers rely on: short, spoken replies, no markdown.
        assert "markdown" in DEFAULT_SYSTEM_PROMPT.lower()
        payload = build_turn_payload([], model="multimodal")
        assert payload["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_payload_is_json_serializable(self) -> None:
        payload = build_turn_payload(
            [{"role": "user", "content": "hi"}], model="multimodal", max_tokens=10
        )
        json.dumps(payload)  # must not raise


# --- turn_endpoint_url / turn_request_headers / build_turn_request ---------


class TestRequestAssembly:
    def test_endpoint_url_appends_the_chat_completions_path(self) -> None:
        assert turn_endpoint_url("http://gateway:8000") == "http://gateway:8000/v1/chat/completions"

    def test_endpoint_url_strips_a_trailing_slash(self) -> None:
        assert (
            turn_endpoint_url("http://gateway:8000/") == "http://gateway:8000/v1/chat/completions"
        )

    def test_headers_include_bearer_when_api_key_set(self) -> None:
        assert turn_request_headers("EMPTY") == {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        }

    def test_headers_omit_authorization_when_api_key_empty_or_none(self) -> None:
        assert turn_request_headers("") == {"Content-Type": "application/json"}
        assert turn_request_headers(None) == {"Content-Type": "application/json"}

    def test_build_turn_request_bundles_url_headers_and_payload(self) -> None:
        req = build_turn_request(
            [{"role": "user", "content": "hi"}],
            base_url="http://gateway:8000",
            api_key="secret",
            model="multimodal",
        )
        assert isinstance(req, TurnRequest)
        assert req.url == "http://gateway:8000/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer secret"
        assert req.body["model"] == "multimodal"
        assert req.body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_build_turn_request_with_empty_model_omits_the_model_key(self) -> None:
        req = build_turn_request([], base_url="http://gateway:8000", api_key="EMPTY", model="")
        assert "model" not in req.body


# --- parse_turn_response: success path --------------------------------------


class TestParseTurnResponseSuccess:
    def test_extracts_the_reply_text(self) -> None:
        body = _chat_completion_body("hello there")
        assert parse_turn_response(200, body) == "hello there"

    def test_strips_surrounding_whitespace(self) -> None:
        body = _chat_completion_body("  hello there  \n")
        assert parse_turn_response(200, body) == "hello there"

    def test_null_content_returns_empty_string_rather_than_raising(self) -> None:
        body = _chat_completion_body(None)
        assert parse_turn_response(200, body) == ""

    def test_raises_turn_response_error_on_non_json_200_body(self) -> None:
        with pytest.raises(TurnResponseError):
            parse_turn_response(200, b"not json")

    def test_raises_turn_response_error_when_choices_missing(self) -> None:
        body = json.dumps({}).encode("utf-8")
        with pytest.raises(TurnResponseError):
            parse_turn_response(200, body)

    def test_raises_turn_response_error_when_choices_empty(self) -> None:
        body = json.dumps({"choices": []}).encode("utf-8")
        with pytest.raises(TurnResponseError):
            parse_turn_response(200, body)

    def test_raises_turn_response_error_when_message_missing(self) -> None:
        body = json.dumps({"choices": [{"index": 0}]}).encode("utf-8")
        with pytest.raises(TurnResponseError):
            parse_turn_response(200, body)

    def test_raises_turn_response_error_when_content_not_a_string(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": 5}}]}).encode("utf-8")
        with pytest.raises(TurnResponseError):
            parse_turn_response(200, body)


# --- parse_turn_response: generic failure path ------------------------------


class TestParseTurnResponseGenericFailure:
    def test_raises_turn_response_error_on_500(self) -> None:
        body = json.dumps({"error": {"message": "boom", "type": "server_error"}}).encode("utf-8")
        with pytest.raises(TurnResponseError) as exc_info:
            parse_turn_response(500, body)
        assert not isinstance(exc_info.value, RoleInfeasibleError)

    def test_500_error_is_not_mistaken_for_role_infeasible(self) -> None:
        # A 500 can never carry the role_infeasible mapping — only a 404 can.
        body = _role_infeasible_body(hosted_by="http://thor.example.ts.net:8000")
        with pytest.raises(TurnResponseError) as exc_info:
            parse_turn_response(500, body)
        assert not isinstance(exc_info.value, RoleInfeasibleError)

    def test_404_model_not_found_is_a_generic_error_not_role_infeasible(self) -> None:
        # A different 404 shape (model_not_found) must not be swallowed into
        # the role_infeasible path — the two are namespaced by code/type.
        body = json.dumps(
            {
                "error": {
                    "message": "The model `bogus` does not exist.",
                    "type": "model_not_found",
                    "code": "model_not_found",
                }
            }
        ).encode("utf-8")
        with pytest.raises(TurnResponseError) as exc_info:
            parse_turn_response(404, body)
        assert not isinstance(exc_info.value, RoleInfeasibleError)

    def test_raises_on_non_json_error_body(self) -> None:
        with pytest.raises(TurnResponseError):
            parse_turn_response(502, b"<html>bad gateway</html>")


# --- parse_turn_response: role_infeasible mapping (acceptance criterion 2) --


class TestRoleInfeasibleMapping:
    def test_404_role_infeasible_raises_role_infeasible_error(self) -> None:
        body = _role_infeasible_body()
        with pytest.raises(RoleInfeasibleError):
            parse_turn_response(404, body)

    def test_role_infeasible_error_is_a_turn_request_error(self) -> None:
        # So a caller that only catches the base class still sees it — it is
        # never invisible to a generic error handler.
        assert issubclass(RoleInfeasibleError, TurnRequestError)

    def test_hosted_by_peer_hint_is_carried_through(self) -> None:
        body = _role_infeasible_body(hosted_by="http://thor.example.ts.net:8000")
        with pytest.raises(RoleInfeasibleError) as exc_info:
            parse_turn_response(404, body)
        assert exc_info.value.hosted_by == "http://thor.example.ts.net:8000"

    def test_hosted_by_is_none_when_no_peer_origin_was_declared(self) -> None:
        body = _role_infeasible_body(hosted_by=None)
        with pytest.raises(RoleInfeasibleError) as exc_info:
            parse_turn_response(404, body)
        assert exc_info.value.hosted_by is None

    def test_message_is_preserved_from_the_gateway_body(self) -> None:
        body = _role_infeasible_body(hosted_by="http://thor.example.ts.net:8000")
        with pytest.raises(RoleInfeasibleError) as exc_info:
            parse_turn_response(404, body)
        assert "not feasible on this machine" in str(exc_info.value)

    def test_detected_via_the_real_gateway_body_shape(self) -> None:
        # Ground the fixture in the ACTUAL gateway function so this test
        # breaks (loudly) if the real 404 body shape ever drifts, instead of
        # silently testing against a fixture nobody keeps in sync.
        from lobes.gateway.server import _role_infeasible_body as real_role_infeasible_body

        body = real_role_infeasible_body(
            "multimodal", "multimodal", "http://thor.example.ts.net:8000"
        )
        with pytest.raises(RoleInfeasibleError) as exc_info:
            parse_turn_response(404, body)
        assert exc_info.value.hosted_by == "http://thor.example.ts.net:8000"

    def test_detected_via_the_real_gateway_body_shape_with_no_peer(self) -> None:
        from lobes.gateway.server import _role_infeasible_body as real_role_infeasible_body

        body = real_role_infeasible_body("multimodal", "multimodal", None)
        with pytest.raises(RoleInfeasibleError) as exc_info:
            parse_turn_response(404, body)
        assert exc_info.value.hosted_by is None

    def test_no_silent_fallback_the_exception_propagates_uncaught(self) -> None:
        # The core "no silent fallback to another lane" invariant: nothing in
        # this module catches RoleInfeasibleError internally and substitutes
        # a placeholder reply or a different model — a caller that does not
        # explicitly handle it sees the exception, full stop.
        body = _role_infeasible_body(hosted_by="http://thor.example.ts.net:8000")

        def _naive_caller() -> str:
            # Simulates a route-layer caller that does NOT special-case
            # RoleInfeasibleError — if this module ever silently produced a
            # fallback string instead of raising, this function would return
            # something instead of raising, and the test would fail below.
            return parse_turn_response(404, body)

        with pytest.raises(RoleInfeasibleError):
            _naive_caller()

    def test_module_never_re_resolves_or_rewrites_the_model_on_failure(self) -> None:
        # build_turn_payload has no knowledge of any prior failure and no
        # retry/fallback parameter to opt into one — the ONLY way a
        # different model gets used is a caller explicitly building a new
        # payload with a different `model=` argument (an operator/session
        # decision one layer up, never automatic here).
        import inspect

        sig = inspect.signature(build_turn_payload)
        assert "fallback" not in sig.parameters
        assert "retry" not in sig.parameters
        sig2 = inspect.signature(parse_turn_response)
        assert "fallback" not in sig2.parameters


def test_module_exports_a_stable_public_surface() -> None:
    import lobes.realtime._turn as turn_mod

    for name in (
        "build_turn_payload",
        "build_turn_request",
        "parse_turn_response",
        "turn_endpoint_url",
        "turn_request_headers",
        "TurnRequest",
        "TurnRequestError",
        "RoleInfeasibleError",
        "TurnResponseError",
        "ToolCallResult",
        "assistant_tool_call_message",
        "tool_result_message",
    ):
        assert hasattr(turn_mod, name), f"missing expected export: {name}"


# --- build_turn_payload with tools: acceptance criterion 1 (task #151 t5) --

_FLAT_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Look up the current weather.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


class TestBuildTurnPayloadTools:
    def test_no_tools_argument_is_byte_identical_to_no_tools_call(self) -> None:
        with_arg = build_turn_payload([], model="multimodal")
        without_arg = build_turn_payload([], model="multimodal", tools=None)
        assert with_arg == without_arg
        assert "tools" not in with_arg
        assert "tool_choice" not in with_arg

    def test_none_tools_omits_tools_and_tool_choice_keys(self) -> None:
        payload = build_turn_payload([], model="multimodal", tools=None, tool_choice="auto")
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_empty_tuple_tools_omits_tools_and_tool_choice_keys(self) -> None:
        payload = build_turn_payload([], model="multimodal", tools=(), tool_choice="auto")
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_declared_tools_are_nested_into_chat_completions_shape(self) -> None:
        payload = build_turn_payload([], model="multimodal", tools=(_FLAT_TOOL,))
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the current weather.",
                    "parameters": _FLAT_TOOL["parameters"],
                },
            }
        ]

    def test_declared_tools_with_no_tool_choice_omit_tool_choice_key(self) -> None:
        payload = build_turn_payload([], model="multimodal", tools=(_FLAT_TOOL,))
        assert "tool_choice" not in payload

    def test_declared_tools_with_string_tool_choice_pass_through(self) -> None:
        payload = build_turn_payload(
            [], model="multimodal", tools=(_FLAT_TOOL,), tool_choice="required"
        )
        assert payload["tool_choice"] == "required"

    def test_declared_tools_with_forced_tool_choice_are_nested(self) -> None:
        payload = build_turn_payload(
            [],
            model="multimodal",
            tools=(_FLAT_TOOL,),
            tool_choice={"type": "function", "name": "get_weather"},
        )
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_payload_with_tools_is_still_json_serializable(self) -> None:
        payload = build_turn_payload(
            [], model="multimodal", tools=(_FLAT_TOOL,), tool_choice="auto"
        )
        json.dumps(payload)  # must not raise

    def test_enable_thinking_false_is_kept_alongside_declared_tools(self) -> None:
        # Measured honored by associate with tools on 2026-09-18 (spec claim):
        # declaring tools must not disturb the no-reasoning-trace default.
        payload = build_turn_payload([], model="multimodal", tools=(_FLAT_TOOL,))
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_build_turn_request_forwards_tools_and_tool_choice(self) -> None:
        req = build_turn_request(
            [],
            base_url="http://gateway:8000",
            model="multimodal",
            tools=(_FLAT_TOOL,),
            tool_choice="auto",
        )
        assert req.body["tools"][0]["function"]["name"] == "get_weather"
        assert req.body["tool_choice"] == "auto"


# --- parse_turn_response with tool_calls: acceptance criterion 2 -----------


def _tool_call_body(
    call_id: str = "call_abc123",
    name: str = "get_weather",
    arguments: str = '{"city": "Tel Aviv"}',
    content: str | None = None,
    extra_calls: int = 0,
) -> bytes:
    tool_calls = [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
    ]
    for i in range(extra_calls):
        tool_calls.append(
            {
                "id": f"call_extra{i}",
                "type": "function",
                "function": {"name": "other_tool", "arguments": "{}"},
            }
        )
    return json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    ).encode("utf-8")


class TestParseTurnResponseToolCalls:
    def test_tool_calls_returns_a_tool_call_result_not_a_string(self) -> None:
        result = parse_turn_response(200, _tool_call_body())
        assert isinstance(result, ToolCallResult)
        assert not isinstance(result, str)

    def test_tool_call_result_carries_call_id_name_and_arguments_verbatim(self) -> None:
        result = parse_turn_response(
            200,
            _tool_call_body(call_id="call_xyz", name="get_weather", arguments='{"city": "Haifa"}'),
        )
        assert result.call_id == "call_xyz"
        assert result.name == "get_weather"
        assert result.arguments == '{"city": "Haifa"}'

    def test_arguments_are_never_parsed_as_json_by_this_module(self) -> None:
        # Malformed/partial JSON arguments must still come through untouched —
        # this module never interprets them.
        result = parse_turn_response(200, _tool_call_body(arguments="{not valid json"))
        assert result.arguments == "{not valid json"

    def test_only_the_first_tool_call_is_surfaced(self) -> None:
        result = parse_turn_response(200, _tool_call_body(call_id="call_first", extra_calls=2))
        assert result.call_id == "call_first"

    def test_tool_call_count_reflects_how_many_the_reply_carried(self) -> None:
        result = parse_turn_response(200, _tool_call_body(extra_calls=2))
        assert result.tool_call_count == 3

    def test_single_tool_call_count_defaults_to_one(self) -> None:
        result = parse_turn_response(200, _tool_call_body())
        assert result.tool_call_count == 1

    def test_tool_calls_present_is_never_treated_as_text_even_with_content(self) -> None:
        # Some backends echo partial/empty text alongside a tool call — this
        # must never be synthesized as the spoken reply.
        result = parse_turn_response(200, _tool_call_body(content="I'll check that for you"))
        assert isinstance(result, ToolCallResult)

    def test_empty_tool_calls_list_falls_through_to_text(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": []}}]}
        ).encode("utf-8")
        assert parse_turn_response(200, body) == "hi"

    def test_malformed_tool_call_missing_id_raises_turn_response_error(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"type": "function", "function": {"name": "x", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")
        with pytest.raises(TurnResponseError):
            parse_turn_response(200, body)

    def test_malformed_tool_call_missing_function_raises_turn_response_error(self) -> None:
        body = json.dumps({"choices": [{"message": {"tool_calls": [{"id": "call_1"}]}}]}).encode(
            "utf-8"
        )
        with pytest.raises(TurnResponseError):
            parse_turn_response(200, body)

    def test_malformed_tool_call_non_string_arguments_raises_turn_response_error(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"id": "call_1", "function": {"name": "x", "arguments": 5}}
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")
        with pytest.raises(TurnResponseError):
            parse_turn_response(200, body)


# --- history helpers: assistant tool_calls + role:tool entries -------------


class TestHistoryHelpers:
    def test_assistant_tool_call_message_shape(self) -> None:
        result = ToolCallResult(call_id="call_1", name="get_weather", arguments='{"city": "X"}')
        entry = assistant_tool_call_message(result)
        assert entry == {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "X"}'},
                }
            ],
        }

    def test_assistant_tool_call_message_is_json_serializable(self) -> None:
        result = ToolCallResult(call_id="call_1", name="get_weather", arguments="{}")
        json.dumps(assistant_tool_call_message(result))  # must not raise

    def test_tool_result_message_shape(self) -> None:
        entry = tool_result_message("call_1", '{"temp_c": 21}')
        assert entry == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"temp_c": 21}',
        }

    def test_tool_result_message_is_json_serializable(self) -> None:
        json.dumps(tool_result_message("call_1", "done"))  # must not raise

    def test_history_round_trip_can_be_fed_back_into_build_turn_payload(self) -> None:
        # The whole point: these entries slot straight into the next turn's
        # history without build_turn_payload needing any tool-specific
        # handling of history items themselves.
        result = ToolCallResult(call_id="call_1", name="get_weather", arguments='{"city": "X"}')
        history = [
            {"role": "user", "content": "what's the weather in X?"},
            assistant_tool_call_message(result),
            tool_result_message("call_1", '{"temp_c": 21}'),
        ]
        payload = build_turn_payload(history, model="multimodal")
        assert payload["messages"][1:] == history
        json.dumps(payload)  # must not raise


# --- DEFAULT_SYSTEM_PROMPT aligned with _session's copy ---------------------


def test_default_system_prompt_is_the_session_default_system_prompt() -> None:
    # Task #151 t5: the dead English copy is gone; this is now an alias, not
    # a second source of truth, so the two can never disagree.
    from lobes.realtime._session import DEFAULT_SYSTEM_PROMPT as SESSION_DEFAULT_SYSTEM_PROMPT

    assert DEFAULT_SYSTEM_PROMPT is SESSION_DEFAULT_SYSTEM_PROMPT


# --- a non-404 error body's peer hint (hebrew-realtime t7) ------------------


def _role_unverified_body(hosted_by: str) -> bytes:
    # Mirrors lobes/gateway/server.py's _role_unverified_body: the mesh
    # boot-window "not yet", which names the pending member.
    return json.dumps(
        {
            "error": {
                "message": f"announced by the mesh member at `{hosted_by}`, unverified",
                "type": "role_unverified",
                "code": "role_unverified",
                "hosted_by": hosted_by,
            }
        }
    ).encode("utf-8")


def test_a_503_role_unverified_carries_hosted_by_as_a_field() -> None:
    with pytest.raises(TurnResponseError) as exc_info:
        parse_turn_response(503, _role_unverified_body("http://spark:8000"))
    # Deliberately NOT a RoleInfeasibleError: "not yet" is a transient
    # failure, "never" is the one that earns its own type.
    assert not isinstance(exc_info.value, RoleInfeasibleError)
    assert exc_info.value.hosted_by == "http://spark:8000"
    assert exc_info.value.status_code == 503


def test_an_ordinary_failure_body_carries_no_peer_hint() -> None:
    with pytest.raises(TurnResponseError) as exc_info:
        parse_turn_response(500, b'{"error": {"message": "boom"}}')
    assert exc_info.value.hosted_by is None
