"""One tool call per model step — made honest on the wire, and never silent.

The voice session's bookkeeping carries exactly ONE outstanding tool call
(``OutstandingToolCall``), so a reply asking for several can only ever have
its first surfaced. Two halves keep that honest:

- the request ASKS for one — ``parallel_tool_calls: false`` travels with
  every payload that declares tools (and with no payload that does not, so a
  tools-free call stays byte-identical to one made before this existed);
- a backend that answers with several anyway is logged at WARNING, naming
  the count and the tool names that were dropped. A client that sees one
  call answered out of three has no other way to learn that happened.
"""

from __future__ import annotations

import json
import logging

import lobes.realtime._session as S
from lobes.realtime._conversation import ConversationBridge, GenerateConfig
from lobes.realtime._speculation import can_adopt
from lobes.realtime._turn import build_turn_payload

_FLAT_TOOL = {
    "type": "function",
    "name": "get_time",
    "description": "The current time.",
    "parameters": {"type": "object", "properties": {}},
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms


def make_bridge(config: dict | None = None, **kwargs) -> tuple[ConversationBridge, FakeClock]:
    clock = FakeClock()
    session, _created = S.Session.create(S.parse_session_config(config or {}))
    bridge = ConversationBridge(
        session,
        cancel_generate=lambda: None,
        cancel_tts=lambda: None,
        generate=GenerateConfig(base_url="http://gw:8000", **kwargs),
        clock=clock,
        chunk_bytes=4800,
    )
    return bridge, clock


def open_turn(bridge: ConversationBridge, clock: FakeClock) -> int:
    bridge.arm()
    bridge.on_speech_started()
    clock.advance(1000)
    bridge.on_speech_stopped()
    bridge.on_transcript("מה השעה")
    turn_id = bridge.take_pending_response()
    assert turn_id is not None
    return turn_id


def _tool_call_body(*names: str) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {"name": name, "arguments": "{}"},
                            }
                            for i, name in enumerate(names)
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    ).encode("utf-8")


# --- the request asks for one ------------------------------------------------


def test_declared_tools_ask_the_backend_for_one_call_at_a_time() -> None:
    payload = build_turn_payload([], model="multimodal", tools=(_FLAT_TOOL,))
    assert payload["parallel_tool_calls"] is False


def test_no_declared_tools_means_no_parallel_tool_calls_key() -> None:
    # The byte-identical guarantee: a tools-free payload gains nothing.
    for tools in (None, (), []):
        payload = build_turn_payload([], model="multimodal", tools=tools, tool_choice="auto")
        assert "parallel_tool_calls" not in payload
    assert build_turn_payload([], model="multimodal") == build_turn_payload(
        [], model="multimodal", tools=None
    )


def test_the_flag_is_json_serializable_beside_the_nested_tools() -> None:
    json.dumps(build_turn_payload([], model="multimodal", tools=(_FLAT_TOOL,)))


# --- speculation still adopts a turn that declares tools ---------------------


def test_a_session_with_tools_still_yields_an_adoptable_speculation() -> None:
    # can_adopt compares the two bodies for equality and both are built by
    # build_turn_request — the new key must land in both, or hidden
    # speculation would silently stop adopting for every tool-using session.
    bridge, _clock = make_bridge({"tools": [_FLAT_TOOL]}, stream=True)
    bridge.arm()
    bridge.on_speech_started()
    speculative = bridge.build_speculative_request("מה השעה")
    assert speculative is not None
    assert speculative.body["parallel_tool_calls"] is False

    bridge.on_speech_stopped()
    bridge.on_transcript("מה השעה")
    real = bridge.build_generate_request(bridge.take_pending_response())
    assert can_adopt(speculative, real)


# --- a backend that answers with several anyway is loud ----------------------


def test_extra_tool_calls_are_dropped_with_a_warning_naming_them(caplog) -> None:
    bridge, clock = make_bridge()
    turn_id = open_turn(bridge, clock)
    with caplog.at_level(logging.WARNING, logger="lobes.realtime._session"):
        assert bridge.on_generate_response(
            200, _tool_call_body("get_time", "get_weather", "get_news"), turn_id=turn_id
        )
    dropped = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(dropped) == 1
    message = dropped[0].getMessage()
    assert "3" in message
    assert "get_weather" in message
    assert "get_news" in message
    assert "get_time" not in message  # the surfaced call is not a dropped one


def test_a_single_tool_call_logs_no_warning(caplog) -> None:
    bridge, clock = make_bridge()
    turn_id = open_turn(bridge, clock)
    with caplog.at_level(logging.WARNING, logger="lobes.realtime._session"):
        assert bridge.on_generate_response(200, _tool_call_body("get_time"), turn_id=turn_id)
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
