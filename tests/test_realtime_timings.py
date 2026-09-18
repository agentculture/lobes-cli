"""Offline tests for the per-stage stopwatch — hebrew-realtime t12 (c33).

``response.done`` carries per-stage milliseconds. The BOUNDARIES are
``app.py``'s (only the route knows when a POST left), but the accumulation is
a decision, so it lives in :mod:`lobes.realtime._timings` and is proven here
against an injected clock. The bridge half — how a measured turn's timings
reach ``response.done`` at all — is proven against a real
:class:`~lobes.realtime._conversation.ConversationBridge`, since a stopwatch
nothing reads would be a silently inert feature.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

import lobes.realtime._conversation as C
import lobes.realtime._session as S
import lobes.realtime._timings as T

from .test_realtime_conversation import CHUNK, chat_body, make_bridge, pcm, pump, run_to_speaking


class StepClock:
    """A monotonic-ms clock a test moves by hand."""

    def __init__(self, start_ms: int = 5_000) -> None:
        self.now_ms = start_ms

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, ms: int) -> int:
        self.now_ms += ms
        return self.now_ms


def test_module_imports_without_the_realtime_extra() -> None:
    importlib.import_module("lobes.realtime._timings")


def test_module_source_never_imports_forbidden_deps() -> None:
    tree = ast.parse(Path(T.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    forbidden = {"fastapi", "httpx", "numpy", "scipy", "torch", "uvicorn", "anyio", "silero_vad"}
    assert not (imported & forbidden), f"_timings.py must stay stdlib-only: {imported}"


# ---------------------------------------------------------------------------
# a stage never started stays ABSENT — never zero
# ---------------------------------------------------------------------------


def test_a_fresh_clock_measures_nothing() -> None:
    clock = StepClock()
    assert T.StageClock(clock).snapshot().as_dict() == {}


def test_an_unstarted_stage_is_absent_not_zero() -> None:
    clock = StepClock()
    stage_clock = T.StageClock(clock)
    stage_clock.start("stt")
    clock.advance(40)
    stage_clock.stop("stt")

    timings = stage_clock.snapshot()
    assert timings.as_dict() == {"stt": 40}
    # Every other stage is absent from the wire, not reported as instant.
    assert timings.tool_wait is None
    assert timings.phonikud is None


def test_a_started_but_unstopped_stage_stays_absent() -> None:
    # An unfinished span has no honest duration; reporting the partial one
    # would read as a completed stage.
    clock = StepClock()
    stage_clock = T.StageClock(clock)
    stage_clock.start("generate")
    clock.advance(900)
    assert stage_clock.snapshot().as_dict() == {}


def test_stopping_a_stage_that_never_started_is_a_no_op() -> None:
    stage_clock = T.StageClock(StepClock())
    stage_clock.stop("tts")
    assert stage_clock.snapshot().as_dict() == {}


# ---------------------------------------------------------------------------
# a stage entered twice ACCUMULATES — the tool turn's two generate legs
# ---------------------------------------------------------------------------


def test_generate_accumulates_across_a_tool_turns_two_legs() -> None:
    clock = StepClock()
    stage_clock = T.StageClock(clock)

    stage_clock.start("generate")
    clock.advance(300)
    stage_clock.stop("generate")
    stage_clock.start("tool_wait")
    clock.advance(1_500)
    stage_clock.stop("tool_wait")
    stage_clock.start("generate")
    clock.advance(700)
    stage_clock.stop("generate")

    assert stage_clock.snapshot().as_dict() == {"generate": 1_000, "tool_wait": 1_500}


def test_restarting_a_running_stage_keeps_the_earlier_boundary() -> None:
    clock = StepClock()
    stage_clock = T.StageClock(clock)
    stage_clock.start("tts")
    clock.advance(100)
    stage_clock.start("tts")  # a second open — the span stays the wider one
    clock.advance(100)
    stage_clock.stop("tts")
    assert stage_clock.snapshot().as_dict() == {"tts": 200}


def test_recorded_spans_accumulate_with_measured_ones() -> None:
    # phonikud is measured INSIDE tts_client.synthesize and handed back.
    clock = StepClock()
    stage_clock = T.StageClock(clock)
    stage_clock.record("phonikud", 120)
    stage_clock.record("phonikud", 30)
    assert stage_clock.snapshot().as_dict() == {"phonikud": 150}


def test_a_negative_span_never_reports_a_negative_millisecond() -> None:
    stage_clock = T.StageClock(StepClock())
    stage_clock.record("tts", -5)
    assert stage_clock.snapshot().as_dict() == {"tts": 0}


# ---------------------------------------------------------------------------
# first_delta — the FIRST chunk only
# ---------------------------------------------------------------------------


def test_first_delta_measures_commit_to_the_first_chunk_only() -> None:
    clock = StepClock()
    stage_clock = T.StageClock(clock)
    stage_clock.start(T.FIRST_DELTA_STAGE)
    clock.advance(2_400)
    stage_clock.mark_first_delta()
    clock.advance(9_000)
    stage_clock.mark_first_delta()  # every later chunk — ignored
    assert stage_clock.snapshot().as_dict() == {"first_delta": 2_400}


def test_mark_first_delta_without_a_start_measures_nothing() -> None:
    stage_clock = T.StageClock(StepClock())
    stage_clock.mark_first_delta()
    assert stage_clock.snapshot().as_dict() == {}


def test_reset_forgets_the_previous_turn() -> None:
    clock = StepClock()
    stage_clock = T.StageClock(clock)
    stage_clock.start("stt")
    clock.advance(50)
    stage_clock.stop("stt")
    stage_clock.start(T.FIRST_DELTA_STAGE)
    stage_clock.mark_first_delta()

    stage_clock.reset()
    assert stage_clock.snapshot().as_dict() == {}
    # …including the once-only first_delta latch.
    stage_clock.start(T.FIRST_DELTA_STAGE)
    clock.advance(7)
    stage_clock.mark_first_delta()
    assert stage_clock.snapshot().as_dict() == {"first_delta": 7}


def test_an_unknown_stage_is_refused_loudly() -> None:
    stage_clock = T.StageClock(StepClock())
    for call in (stage_clock.start, stage_clock.stop):
        with pytest.raises(ValueError):
            call("thinking")


def test_the_stage_names_are_the_wire_contracts_own() -> None:
    clock = StepClock()
    stage_clock = T.StageClock(clock)
    for stage in S.STAGE_TIMING_KEYS:
        stage_clock.start(stage)
        clock.advance(1)
        stage_clock.stop(stage)
    assert set(stage_clock.snapshot().as_dict()) == set(S.STAGE_TIMING_KEYS)


# ---------------------------------------------------------------------------
# the bridge half — measured stages reach response.done, absent ones do not
# ---------------------------------------------------------------------------


def finish_a_spoken_turn(bridge, clock) -> list[dict]:
    turn_id = run_to_speaking(bridge, clock)
    bridge.on_tts_audio(pcm(CHUNK * 2), turn_id=turn_id)
    pump(bridge, turn_id)
    return bridge.drain()


def done_payload(payloads: list[dict]) -> dict:
    return next(p for p in payloads if p["type"] == S.EventType.RESPONSE_DONE)


def test_a_bridge_with_no_provider_serializes_response_done_unchanged() -> None:
    # The pre-existing contract: a route that measures nothing (and every
    # offline test) still produces a response.done with no timings key.
    bridge, _cancels, clock = make_bridge()
    bridge.arm()
    assert "timings" not in done_payload(finish_a_spoken_turn(bridge, clock))


def test_a_provider_puts_the_measured_stages_on_response_done() -> None:
    bridge, _cancels, clock = make_bridge()
    bridge.arm()
    stage_clock = T.StageClock(StepClock())
    stage_clock.record("stt", 310)
    stage_clock.record("generate", 820)
    stage_clock.record("first_delta", 1_400)
    bridge.set_timings_provider(stage_clock.snapshot)

    done = done_payload(finish_a_spoken_turn(bridge, clock))
    assert done["timings"] == {"stt": 310, "generate": 820, "first_delta": 1_400}


def test_a_provider_that_measured_nothing_adds_no_key() -> None:
    # An all-absent StageTimings is falsy by contract — it must serialize away
    # entirely rather than as an empty mapping.
    bridge, _cancels, clock = make_bridge()
    bridge.arm()
    bridge.set_timings_provider(T.StageClock(StepClock()).snapshot)
    assert "timings" not in done_payload(finish_a_spoken_turn(bridge, clock))


def test_the_provider_is_read_at_response_done_not_at_construction() -> None:
    # The route measures the LAST stages (tts, first_delta) after the bridge
    # exists; reading the provider early would drop every one of them.
    bridge, _cancels, clock = make_bridge()
    bridge.arm()
    stage_clock = T.StageClock(StepClock())
    bridge.set_timings_provider(stage_clock.snapshot)
    stage_clock.record("tts", 640)

    done = done_payload(finish_a_spoken_turn(bridge, clock))
    assert done["timings"] == {"tts": 640}


def test_a_tool_turns_done_carries_both_generate_legs() -> None:
    # The whole reason the provider is read at ResponseDone: a tool turn's
    # second generate is measured long after the first.
    bridge, _cancels, clock = make_bridge()
    bridge.arm()
    stage_clock = T.StageClock(StepClock())
    bridge.set_timings_provider(stage_clock.snapshot)

    bridge.on_speech_started(at_ms=0)
    bridge.on_speech_stopped(at_ms=1_000, reason="silence")
    bridge.on_transcript("what time is it")
    turn_id = bridge.take_pending_response()
    assert turn_id is not None
    stage_clock.record("generate", 200)
    tool_body = _tool_call_body("call-1", "clock_now", "{}")
    bridge.on_generate_response(200, tool_body, turn_id=turn_id)
    assert bridge.awaiting_tool_result
    stage_clock.record("tool_wait", 4_000)
    bridge.on_function_call_output(
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call-1", "output": "16:30"},
        }
    )
    bridge.arm()
    assert bridge.take_pending_response() == turn_id
    stage_clock.record("generate", 300)
    bridge.on_generate_response(200, chat_body("half past four"), turn_id=turn_id)
    assert bridge.take_pending_synthesis() == (turn_id, "half past four")
    bridge.on_tts_audio(pcm(CHUNK), turn_id=turn_id)
    pump(bridge, turn_id)

    done = done_payload(bridge.drain())
    assert done["timings"] == {"generate": 500, "tool_wait": 4_000}


def _tool_call_body(call_id: str, name: str, arguments: str) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ],
                    }
                }
            ]
        }
    ).encode()


def test_set_timings_provider_is_additive_and_clearable() -> None:
    bridge, _cancels, clock = make_bridge()
    bridge.arm()
    bridge.set_timings_provider(lambda: S.StageTimings(tts=9))
    bridge.set_timings_provider(None)
    assert "timings" not in done_payload(finish_a_spoken_turn(bridge, clock))


def test_the_provider_hook_stays_on_the_stdlib_bridge() -> None:
    # The route is `pragma: no cover` and unimportable offline, so the hook it
    # calls has to live here, where CI can prove it exists.
    assert callable(C.ConversationBridge.set_timings_provider)


# ---------------------------------------------------------------------------
# the route's own wiring — app.py is fastapi/torch-only and never imported by
# this suite, so its obligations are asserted against its SOURCE, structurally
# (the same idiom test_realtime_conversation.py uses for the pump/watchdog).
# ---------------------------------------------------------------------------

APP_SOURCE = Path(C.__file__).with_name("app.py").read_text(encoding="utf-8")
APP_TREE = ast.parse(APP_SOURCE)


def app_function(name: str):
    for node in ast.walk(APP_TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"app.py has no function named {name!r}")


def called_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def keyword_names(node: ast.AST, call_name: str) -> set[str]:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == call_name:
                return {kw.arg for kw in child.keywords if kw.arg}
    raise AssertionError(f"no call to {call_name!r} in this node")


def stage_literals(node: ast.AST, method: str) -> list[str]:
    """Every literal stage name passed to ``clock.<method>(...)`` in *node*."""
    stages = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == method
            and child.args
        ):
            arg = child.args[0]
            if isinstance(arg, ast.Constant):
                stages.append(arg.value)
            elif isinstance(arg, ast.Name) and arg.id == "FIRST_DELTA_STAGE":
                stages.append(T.FIRST_DELTA_STAGE)
    return stages


def test_the_route_attaches_the_stopwatch_to_the_bridge() -> None:
    assert "set_timings_provider" in called_names(app_function("_build_bridge"))


def test_the_route_passes_the_operators_tool_wait_timeout() -> None:
    assert "tool_wait_timeout_ms" in keyword_names(
        app_function("_build_bridge"), "ConversationBridge"
    )


def test_the_route_passes_the_deployment_language_as_the_session_default() -> None:
    assert "default_language" in keyword_names(
        app_function("_open_session"), "parse_session_config"
    )


def test_the_route_passes_the_generate_headers_back_to_the_bridge() -> None:
    # A shed lane's Retry-After has nowhere else to travel.
    keywords = keyword_names(app_function("_drive_response"), "on_generate_response")
    assert "headers" in keywords


def test_the_route_measures_the_stages_it_alone_can_see() -> None:
    driver = app_function("_drive_response")
    assert set(stage_literals(driver, "start")) == {"generate", "tool_wait", "tts"}
    assert set(stage_literals(driver, "stop")) == {"generate", "tts"}
    assert "mark_first_delta" in called_names(driver)
    assert "record" in called_names(driver)  # phonikud, measured inside synthesize

    committed = app_function("_emit_turn_events")
    assert set(stage_literals(committed, "start")) == {"stt", T.FIRST_DELTA_STAGE}
    assert "reset" in called_names(committed)
    assert stage_literals(app_function("_transcribe_turn"), "stop") == ["stt"]
    assert stage_literals(app_function("_pump_session"), "stop") == ["tool_wait"]


def test_the_driver_returns_on_a_tool_call_before_it_synthesizes() -> None:
    # The acceptance criterion: a tool-call result must not be spoken, and the
    # response must not end. Asserted positionally — the guard has to come
    # BEFORE the synthesis, or it guards nothing.
    driver = app_function("_drive_response")
    guards = [
        node.lineno
        for node in ast.walk(driver)
        if isinstance(node, ast.Attribute) and node.attr == "awaiting_tool_result"
    ]
    synthesis = [
        node.lineno
        for node in ast.walk(driver)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "synthesize"
    ]
    assert guards and synthesis
    assert min(guards) < min(synthesis)


def test_the_tool_call_guard_returns_rather_than_falling_through() -> None:
    driver = app_function("_drive_response")
    guard_ifs = [
        node
        for node in ast.walk(driver)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "awaiting_tool_result"
            for child in ast.walk(node.test)
        )
    ]
    assert guard_ifs, "the driver must branch on bridge.awaiting_tool_result"
    assert any(
        isinstance(stmt, ast.Return) for node in guard_ifs for stmt in ast.walk(node)
    ), "the tool branch must end the driver, not fall through to TTS"


def test_the_spoken_reply_is_synthesized_in_the_sessions_language() -> None:
    keywords = keyword_names(app_function("_drive_response"), "synthesize")
    assert {"language", "timings_out"} <= keywords


def test_every_internal_helper_call_in_app_py_matches_its_definition() -> None:
    # app.py is never imported by CI, so a call site left behind at an old
    # arity (this task changed six of them) would only surface on a live box,
    # mid-session. This walks the module's own helpers and checks each call
    # against the definition's positional/keyword shape.
    definitions = {
        node.name: node
        for node in ast.walk(APP_TREE)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for call in [node for node in ast.walk(APP_TREE) if isinstance(node, ast.Call)]:
        func = call.func
        if not isinstance(func, ast.Name) or func.id not in definitions:
            continue
        spec = definitions[func.id].args
        names = [arg.arg for arg in spec.posonlyargs + spec.args]
        required = len(names) - len(spec.defaults)
        supplied = len(call.args) + len({kw.arg for kw in call.keywords if kw.arg})
        assert required <= supplied <= len(names) or spec.vararg or spec.kwarg, (
            f"app.py calls {func.id}() with {supplied} argument(s); "
            f"its definition takes {required}-{len(names)}"
        )
        for keyword in call.keywords:
            if keyword.arg:
                assert keyword.arg in names + [
                    arg.arg for arg in spec.kwonlyargs
                ], f"app.py calls {func.id}(..., {keyword.arg}=...), which it does not accept"
