"""Offline tests for the STT forward's request builder — hebrew-realtime t12 (c11).

A ``/v1/realtime`` session that speaks Hebrew has to TELL Parakeet so; before
this task ``app.py`` hardcoded ``data={"language": "en"}`` on every forward,
so a Hebrew session was transcribed as English. The fix has to clear a second
bar: an unconfigured English deployment must send byte-for-byte what it sent
before. ``app.py`` is fastapi/httpx/torch-only and is never imported by this
suite, so the builder is a pure function HERE and the route's use of it is
asserted against its source with :mod:`ast` — the same idiom
``test_realtime_conversation.py`` uses for the route's pump and watchdog.
"""

from __future__ import annotations

import ast
from pathlib import Path

import lobes.realtime.audio_facade as A
from lobes.realtime._session import DEFAULT_LANGUAGE

APP_SOURCE = Path(A.__file__).with_name("app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# byte-identity when nothing is configured
# ---------------------------------------------------------------------------


def test_an_unconfigured_deployment_sends_exactly_todays_fields() -> None:
    # Verbatim what app.py posted before this task: one field, language=en.
    assert A.build_stt_forward_fields(None) == {"language": "en"}
    assert A.build_stt_forward_fields("") == {"language": "en"}
    assert DEFAULT_LANGUAGE == "en"


def test_the_file_part_is_unchanged_too() -> None:
    assert A.build_stt_forward_files(b"RIFFbytes") == {
        "file": ("turn.wav", b"RIFFbytes", "audio/wav")
    }


# ---------------------------------------------------------------------------
# the two-level resolution: session language > deployment default > "en"
# ---------------------------------------------------------------------------


def test_a_session_language_reaches_the_backend() -> None:
    assert A.build_stt_forward_fields("he") == {"language": "he"}


def test_a_deployment_default_applies_when_the_session_says_nothing() -> None:
    assert A.build_stt_forward_fields(None, "he") == {"language": "he"}


def test_a_session_language_beats_the_deployment_default() -> None:
    assert A.build_stt_forward_fields("en", "he") == {"language": "en"}


def test_a_blank_request_is_not_a_language() -> None:
    # Declaring "" to Parakeet is a worse answer than falling through.
    assert A.resolve_stt_language("   ", "he") == "he"
    assert A.resolve_stt_language(None, "") == "en"


# ---------------------------------------------------------------------------
# the route actually uses it (app.py cannot be imported; assert its source)
# ---------------------------------------------------------------------------


def _function(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(APP_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"app.py has no function named {name!r}")


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_the_route_builds_its_stt_fields_with_the_shared_builder() -> None:
    called = _called_names(_function("_forward_turn_to_stt"))
    assert "build_stt_forward_fields" in called
    assert "build_stt_forward_files" in called


def test_the_route_no_longer_hardcodes_a_language_anywhere() -> None:
    # The dict literal this task removed — asserted over the CODE, so the
    # docstring that quotes it for the reader does not satisfy the check.
    hardcoded = [
        node
        for node in ast.walk(ast.parse(APP_SOURCE))
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == "language" for key in node.keys)
    ]
    assert not hardcoded, "the STT language must come from the session, not a literal"


def test_the_forward_takes_the_sessions_language() -> None:
    node = _function("_forward_turn_to_stt")
    args = [arg.arg for arg in node.args.args]
    assert "language" in args, "the STT forward must be told which language the session speaks"


def test_the_batch_route_resolves_its_form_default_too() -> None:
    node = _function("transcriptions")
    assert "build_stt_forward_fields" in _called_names(node)
