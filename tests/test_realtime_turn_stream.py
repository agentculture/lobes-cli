"""Streamed generation — the payload flag and the incremental SSE consumer.

``_turn.py`` gained two additive surfaces for sentence-level streaming
(approved deviation d7, 2026-09-18):

- ``build_turn_payload(..., stream=True)`` — the ONE key that turns the
  generate call into a stream. ``stream=False`` (the default) leaves the
  payload byte-identical to before this change, which is what makes
  ``GENERATE_STREAM=false`` an exact rollback.
- :class:`~lobes.realtime._turn.StreamAccumulator` — a pure consumer of
  ``chat.completion.chunk`` SSE lines that yields text deltas as they arrive
  and ends with the SAME result shape :func:`parse_turn_response` produces:
  a ``str`` reply, or a :class:`~lobes.realtime._turn.ToolCallResult`.

The invariant that matters most here is the text-then-tool-call policy: once
any ``tool_calls`` delta is seen the accumulator STOPS releasing text for
speech and the final result is the tool call. Speaking a prefix the model
abandoned in favour of calling a tool would be a lie the client cannot undo.
"""

from __future__ import annotations

import json

import pytest

import lobes.realtime._turn as T


def chunk(**delta: object) -> str:
    """One OpenAI ``chat.completion.chunk`` SSE data line."""
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


def tool_chunk(index: int = 0, **function: object) -> str:
    call: dict[str, object] = {"index": index}
    if "id" in function:
        call["id"] = function.pop("id")
    if function:
        call["function"] = function
    return chunk(tool_calls=[call])


def feed_all(acc: T.StreamAccumulator, lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        out.extend(item.text for item in acc.feed_line(line))
    return out


# --- the payload flag --------------------------------------------------------


def test_stream_false_leaves_the_payload_byte_identical() -> None:
    history = [{"role": "user", "content": "מה השעה"}]
    assert T.build_turn_payload(history) == T.build_turn_payload(history, stream=False)
    assert "stream" not in T.build_turn_payload(history)


def test_stream_true_adds_exactly_one_key() -> None:
    history = [{"role": "user", "content": "מה השעה"}]
    plain = T.build_turn_payload(history)
    streamed = T.build_turn_payload(history, stream=True)
    assert streamed.pop("stream") is True
    assert streamed == plain


def test_build_turn_request_threads_the_stream_flag_through() -> None:
    request = T.build_turn_request([], base_url="http://gw:8000", stream=True)
    assert request.body["stream"] is True
    assert T.build_turn_request([], base_url="http://gw:8000").body.get("stream") is None


# --- the consumer: text ------------------------------------------------------


def test_text_deltas_are_released_as_they_arrive_and_accumulate_into_the_result() -> None:
    acc = T.StreamAccumulator()
    released = feed_all(
        acc,
        [chunk(role="assistant"), chunk(content="השעה "), chunk(content="ארבע"), "data: [DONE]"],
    )
    assert released == ["השעה ", "ארבע"]
    assert acc.result() == "השעה ארבע"
    assert acc.done is True


def test_comments_blank_lines_and_non_data_fields_are_ignored() -> None:
    acc = T.StreamAccumulator()
    assert acc.feed_line("") == []
    assert acc.feed_line(": keep-alive") == []
    assert acc.feed_line("event: message") == []
    assert acc.feed_line("\n") == []
    assert feed_all(acc, [chunk(content="שלום")]) == ["שלום"]


def test_bytes_lines_are_accepted_exactly_like_str_lines() -> None:
    acc = T.StreamAccumulator()
    assert [item.text for item in acc.feed_line(chunk(content="שלום").encode("utf-8"))] == ["שלום"]


def test_a_usage_only_chunk_with_no_choices_is_not_an_error() -> None:
    acc = T.StreamAccumulator()
    assert acc.feed_line('data: {"object":"chat.completion.chunk","choices":[]}') == []
    assert acc.result() == ""


def test_an_empty_stream_yields_an_empty_reply_not_an_exception() -> None:
    # The floor names an empty reply `generate_failed`; that decision stays
    # there, so this layer must hand it a plain empty string.
    acc = T.StreamAccumulator()
    acc.feed_line("data: [DONE]")
    assert acc.result() == ""


@pytest.mark.parametrize("line", ("data: not json", "data: [1, 2, 3]"))
def test_a_malformed_data_line_raises_the_named_turn_error(line: str) -> None:
    acc = T.StreamAccumulator()
    with pytest.raises(T.TurnResponseError):
        acc.feed_line(line)


# --- the consumer: tool calls -----------------------------------------------


def test_tool_call_fragments_are_reassembled_across_chunks() -> None:
    acc = T.StreamAccumulator()
    feed_all(
        acc,
        [
            tool_chunk(id="call_1", name="get_time"),
            tool_chunk(arguments='{"tz":'),
            tool_chunk(arguments='"UTC"}'),
            "data: [DONE]",
        ],
    )
    result = acc.result()
    assert isinstance(result, T.ToolCallResult)
    assert (result.call_id, result.name, result.arguments) == (
        "call_1",
        "get_time",
        '{"tz":"UTC"}',
    )
    assert result.tool_call_count == 1


def test_several_parallel_calls_surface_the_first_and_report_the_count() -> None:
    acc = T.StreamAccumulator()
    feed_all(
        acc,
        [
            tool_chunk(0, id="call_a", name="first"),
            tool_chunk(1, id="call_b", name="second"),
            tool_chunk(0, arguments="{}"),
            tool_chunk(1, arguments="{}"),
        ],
    )
    result = acc.result()
    assert isinstance(result, T.ToolCallResult)
    assert result.call_id == "call_a"
    assert result.tool_call_count == 2


def test_a_tool_call_wins_over_text_and_stops_releasing_it() -> None:
    # THE policy: a reply that starts as text and then calls a tool must not
    # be spoken. Text released BEFORE the tool call is the bridge's to cancel
    # (it does); text after it never leaves this module at all.
    acc = T.StreamAccumulator()
    released = feed_all(
        acc,
        [
            chunk(content="רגע, "),
            tool_chunk(id="call_1", name="get_time"),
            chunk(content="אני בודק"),
            tool_chunk(arguments="{}"),
        ],
    )
    assert released == ["רגע, "]  # the post-tool-call text is never released
    assert acc.saw_tool_call is True
    result = acc.result()
    assert isinstance(result, T.ToolCallResult)
    assert result.name == "get_time"


def test_a_tool_call_missing_its_id_is_a_malformed_response() -> None:
    acc = T.StreamAccumulator()
    acc.feed_line(tool_chunk(name="get_time"))
    with pytest.raises(T.TurnResponseError):
        acc.result()


def test_a_tool_call_missing_its_name_is_a_malformed_response() -> None:
    acc = T.StreamAccumulator()
    acc.feed_line(tool_chunk(id="call_1"))
    with pytest.raises(T.TurnResponseError):
        acc.result()


def test_absent_arguments_default_to_an_empty_json_object_string() -> None:
    # A no-argument tool streams no `arguments` fragment at all; that is a
    # well-formed call, not a defect.
    acc = T.StreamAccumulator()
    acc.feed_line(tool_chunk(id="call_1", name="now"))
    result = acc.result()
    assert isinstance(result, T.ToolCallResult)
    assert result.arguments == ""


def test_result_matches_parse_turn_response_for_the_same_reply() -> None:
    # The two paths must be interchangeable: GENERATE_STREAM is a transport
    # switch, never a behaviour switch.
    acc = T.StreamAccumulator()
    feed_all(acc, [chunk(content="  השעה ארבע  ")])
    non_streamed = T.parse_turn_response(
        200,
        json.dumps({"choices": [{"message": {"content": "  השעה ארבע  "}}]}).encode(),
    )
    assert acc.result() == non_streamed == "השעה ארבע"


def test_an_empty_arguments_string_is_legal_in_both_paths() -> None:
    # The non-streamed parser accepts `""` (it is a string), so the stream's
    # empty accumulator must too — same reply, same result.
    acc = T.StreamAccumulator()
    feed_all(acc, [tool_chunk(id="call_1", name="now", arguments="")])
    non_streamed = T.parse_turn_response(
        200,
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "now", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode(),
    )
    assert acc.result() == non_streamed


def test_a_non_string_arguments_fragment_is_a_malformed_response() -> None:
    # The non-streamed parser raises on a non-string `arguments`; ignoring it
    # here would hand the client tool whatever happened to accumulate.
    acc = T.StreamAccumulator()
    line = tool_chunk(id="call_1", name="now", arguments={"tz": "UTC"})
    with pytest.raises(T.TurnResponseError):
        acc.feed_line(line)


def test_a_non_string_arguments_fragment_after_good_ones_still_raises() -> None:
    acc = T.StreamAccumulator()
    acc.feed_line(tool_chunk(id="call_1", name="now"))
    acc.feed_line(tool_chunk(arguments='{"tz":'))
    line = tool_chunk(arguments=17)
    with pytest.raises(T.TurnResponseError):
        acc.feed_line(line)


def test_parallel_stream_calls_name_the_dropped_ones() -> None:
    acc = T.StreamAccumulator()
    feed_all(
        acc,
        [
            tool_chunk(0, id="call_a", name="first"),
            tool_chunk(1, id="call_b", name="second"),
        ],
    )
    result = acc.result()
    assert isinstance(result, T.ToolCallResult)
    assert result.dropped_names == ("second",)
