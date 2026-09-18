"""``app.py``'s STREAMING obligations, asserted structurally against its source.

``app.py`` imports fastapi/httpx/torch and is never imported by the offline
suite (no CI lane installs the ``[realtime]`` extra), so — exactly as
``tests/test_realtime_conversation.py`` already does for the delivery pump and
the watchdog — the runtime contracts that CANNOT be unit-tested are asserted
against its AST instead. Wire any of these differently and every unit test
above still passes while the live bridge is wrong.

Four things are load-bearing here (approved deviation d7, 2026-09-18):

1. there IS a streaming path, and it consumes an SSE body incrementally;
2. a non-200 streamed response goes through the EXISTING error path WITH its
   headers — that is the only route a shed lane's ``Retry-After`` has;
3. the ``awaiting_tool_result`` guard precedes any ``synthesize`` call, so a
   tool turn is never spoken;
4. the streamed delivery pump awaits between chunks, or barge-in is inert for
   exactly the reason ``delivery_pause_ms`` exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

import lobes.realtime._wire as W

APP = Path(W.__file__).parent / "app.py"


def _source() -> str:
    return APP.read_text(encoding="utf-8")


def _function(name: str):
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"app.py has no function named {name!r}")


def _called_attrs(node: ast.AST) -> set[str]:
    return {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }


# --- 1: the streaming path exists and is incremental ------------------------


def test_the_stream_path_consumes_the_body_line_by_line() -> None:
    node = _function("_stream_generate")
    calls = _called_attrs(node)
    assert "stream" in calls, "the streamed generate must use httpx's stream() context"
    assert "aiter_lines" in calls, "an SSE body must be consumed incrementally, never read whole"
    assert "feed_line" in calls
    assert {"on_generate_delta", "on_generate_stream_end"} <= calls


def test_the_route_dispatches_on_the_operator_knob() -> None:
    node = _function("_drive_response")
    names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    assert "streaming_enabled" in names, "GENERATE_STREAM must select the path"
    # …and the non-streaming body is still right here, unchanged: the
    # pre-existing structural tests assert the pump and the voice lane
    # against THIS function.
    assert "deliver_next" in _called_attrs(node)


def test_the_bridge_is_built_with_the_settings_stream_flag() -> None:
    node = _function("_build_bridge")
    segment = ast.get_source_segment(_source(), node) or ""
    assert "stream=settings.generate_stream" in segment


# --- 2: a non-200 stream keeps the existing, headered error path ------------


def test_the_streamed_error_path_forwards_headers_to_the_bridge() -> None:
    node = _function("_drive_streamed_response")
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "on_generate_response"
    ]
    assert calls, "a non-200 streamed response must reuse on_generate_response"
    for call in calls:
        assert "headers" in {kw.arg for kw in call.keywords}, (
            "a shed lane's Retry-After has nowhere else to travel — the "
            "headers must reach the bridge"
        )


def test_a_non_200_stream_reads_the_body_whole_instead_of_iterating_it() -> None:
    node = _function("_stream_generate")
    assert "aread" in _called_attrs(node)


def test_every_named_generate_failure_is_still_surfaced() -> None:
    node = _function("_drive_streamed_response")
    handlers = {
        ast.unparse(handler.type) if handler.type else ""
        for handler in ast.walk(node)
        if isinstance(handler, ast.ExceptHandler)
    }
    assert "httpx.TimeoutException" in handlers
    assert "httpx.HTTPError" in handlers
    # A malformed chunk mid-stream must be a NAMED failure, not a truncated
    # reply presented as a whole one.
    assert "TurnRequestError" in handlers
    assert "fail_generate" in _called_attrs(node)


# --- 3: a tool turn is never spoken -----------------------------------------


def test_the_awaiting_tool_result_guard_precedes_every_synthesize_call() -> None:
    node = _function("_synth_worker")
    guards = [
        n.lineno
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and n.attr == "awaiting_tool_result"
    ]
    synths = [
        n.lineno
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "synthesize"
    ]
    assert guards and synths
    assert min(guards) < min(synths), (
        "the tool-turn guard must come BEFORE the synthesis, or a tool call's "
        "abandoned text prefix gets spoken"
    )


def test_the_synth_worker_uses_the_voice_lane_and_the_cancel_event() -> None:
    node = _function("_synth_worker")
    segment = ast.get_source_segment(_source(), node) or ""
    assert "lane=VOICE_LANE" in segment
    assert "cancel_event=active.tts_cancel" in segment
    assert "timings_out=tts_timings" in segment


def test_the_synth_worker_is_sequential_not_a_gather() -> None:
    # TTS_VOICE_CONCURRENCY is 1 and the floor delivers in index order, so a
    # parallel worker would buy nothing and could starve segment N.
    node = _function("_synth_worker")
    assert "gather" not in _called_attrs(node)


# --- 4: the streamed pump still paces ---------------------------------------


def test_the_streamed_pump_awaits_between_chunks_and_paces_delivery() -> None:
    node = _function("_pump_delivery")
    loops = [inner for inner in ast.walk(node) if isinstance(inner, ast.While)]
    assert loops
    for loop in loops:
        assert any(
            isinstance(inner, ast.Await)
            for inner in ast.walk(ast.Module(body=loop.body, type_ignores=[]))
        ), "the delivery pump must await between chunks or a barge-in can never land"
    calls = _called_attrs(node) | {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    assert "delivery_pause_ms" in calls
    assert "mark_first_delta" in calls
    assert "response_in_progress" in calls, "the loop condition is the bridge's decision, not ours"


def test_cancellation_stops_all_three_concurrent_pieces() -> None:
    node = _function("_drive_streamed_response")
    assert "cancel" in _called_attrs(node)
    finallys = [inner for inner in ast.walk(node) if isinstance(inner, ast.Try) and inner.finalbody]
    assert finallys, "the streamed driver must clean up its workers in a finally"
