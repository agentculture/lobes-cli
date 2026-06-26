"""Tests for ``lobes route`` — task-to-gear routing decision verb.

Uses a throwaway argparse parser so the test is independent of
lobes/cli/__init__.py wiring (which is done in a separate task, t8).
No real HTTP ever happens: the minor client is monkeypatched in every test.
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import pytest

from lobes.cli._commands import route

# ---------------------------------------------------------------------------
# Helper to make a canned chat_completion response
# ---------------------------------------------------------------------------


def _make_completion(
    chosen_gear: str = "primary",
    confidence: float = 0.85,
    reason: str = "Test reason",
    conditions: list[str] | None = None,
) -> dict:
    """Build a canned chat_completion response dict."""
    if conditions is None:
        conditions = []
    content = json.dumps(
        {
            "chosen_gear": chosen_gear,
            "confidence": confidence,
            "reason": reason,
            "conditions": conditions,
        }
    )
    return {
        "id": "chatcmpl-route-001",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


# ---------------------------------------------------------------------------
# Helper: build a throwaway parser and register the route verb
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    """Return a fresh ArgumentParser with the ``route`` subcommand registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    route.register(sub)
    return p


# ---------------------------------------------------------------------------
# Importability / registration
# ---------------------------------------------------------------------------


def test_register_is_importable_standalone() -> None:
    """route.register can be called on a plain argparse parser with no CLI wiring."""
    p = _make_parser()
    args = p.parse_args(["route", "summarize this PR", "--model", "test-model"])
    assert args.text == "summarize this PR"
    assert args.model == "test-model"


def test_register_exposes_func_default() -> None:
    """Parsed args carry a ``func`` attribute (the handler)."""
    p = _make_parser()
    args = p.parse_args(["route", "hello", "--model", "m"])
    assert callable(args.func)


# ---------------------------------------------------------------------------
# (a) Decision structure — chosen_gear + escalate + confidence∈[0,1] + reason
# ---------------------------------------------------------------------------


def test_decision_has_required_fields(capsys: pytest.CaptureFixture[str]) -> None:
    """The routing decision includes chosen_gear, escalate, confidence∈[0,1], reason."""
    p = _make_parser()
    args = p.parse_args(
        [
            "route",
            "summarize this PR",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )

    canned = _make_completion(
        chosen_gear="minor", confidence=0.9, reason="Simple summarization task"
    )

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)

    assert "chosen_gear" in decision
    assert "escalate" in decision
    assert "confidence" in decision
    assert "reason" in decision

    assert isinstance(decision["chosen_gear"], str)
    assert isinstance(decision["escalate"], bool)
    assert isinstance(decision["reason"], str)
    assert (
        0.0 <= decision["confidence"] <= 1.0
    ), f"confidence out of range: {decision['confidence']}"


def test_decision_chosen_gear_matches_model_suggestion(capsys: pytest.CaptureFixture[str]) -> None:
    """chosen_gear in the output matches the model's gear suggestion."""
    p = _make_parser()
    args = p.parse_args(
        [
            "route",
            "classify this text",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )

    canned = _make_completion(
        chosen_gear="minor", confidence=0.95, reason="Quick classification task"
    )

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert decision["chosen_gear"] == "minor"
    assert not decision["escalate"]


def test_decision_no_escalation_without_conditions(capsys: pytest.CaptureFixture[str]) -> None:
    """No escalation conditions → escalate is False for an allowed routing duty."""
    p = _make_parser()
    args = p.parse_args(
        ["route", "format this JSON", "--model", "m", "--base-url", "http://localhost/v1", "--json"]
    )

    canned = _make_completion(chosen_gear="minor", confidence=0.92, conditions=[])

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert decision["escalate"] is False


# ---------------------------------------------------------------------------
# (b) Escalation conditions force escalate=True regardless of model suggestion
# ---------------------------------------------------------------------------


def test_security_sensitive_forces_escalate(capsys: pytest.CaptureFixture[str]) -> None:
    """security_sensitive condition forces escalate=True even if model suggests minor."""
    p = _make_parser()
    args = p.parse_args(
        [
            "route",
            "review auth tokens",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )

    canned = _make_completion(
        chosen_gear="minor",
        confidence=0.8,
        reason="Could handle locally but security flag applies",
        conditions=["security_sensitive"],
    )

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert decision["escalate"] is True, "security_sensitive must force escalate=True"


def test_write_or_delete_condition_forces_escalate(capsys: pytest.CaptureFixture[str]) -> None:
    """write_or_delete_operation condition forces escalate=True."""
    p = _make_parser()
    args = p.parse_args(
        [
            "route",
            "delete all logs",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )

    canned = _make_completion(
        chosen_gear="primary",
        confidence=0.7,
        reason="Write operation",
        conditions=["write_or_delete_operation"],
    )

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert decision["escalate"] is True


def test_architectural_decision_forces_escalate(capsys: pytest.CaptureFixture[str]) -> None:
    """architectural_decision condition forces escalate=True."""
    p = _make_parser()
    args = p.parse_args(
        [
            "route",
            "choose the new database engine",
            "--model",
            "m",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )

    canned = _make_completion(
        chosen_gear="primary",
        confidence=0.6,
        reason="Architecture choice",
        conditions=["architectural_decision"],
    )

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert decision["escalate"] is True


# ---------------------------------------------------------------------------
# (c) Confidence clamped to [0, 1]
# ---------------------------------------------------------------------------


def test_confidence_clamped_above_one(capsys: pytest.CaptureFixture[str]) -> None:
    """confidence > 1.0 from the model is clamped to 1.0."""
    p = _make_parser()
    args = p.parse_args(
        ["route", "hello", "--model", "m", "--base-url", "http://localhost/v1", "--json"]
    )

    canned = _make_completion(confidence=1.5)

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert decision["confidence"] == 1.0, f"expected 1.0, got {decision['confidence']}"


def test_confidence_clamped_below_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """confidence < 0.0 from the model is clamped to 0.0."""
    p = _make_parser()
    args = p.parse_args(
        ["route", "hello", "--model", "m", "--base-url", "http://localhost/v1", "--json"]
    )

    canned = _make_completion(confidence=-0.3)

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert decision["confidence"] == 0.0, f"expected 0.0, got {decision['confidence']}"


def test_confidence_in_range_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    """Confidence already in [0,1] is passed through unchanged."""
    p = _make_parser()
    args = p.parse_args(
        ["route", "hello", "--model", "m", "--base-url", "http://localhost/v1", "--json"]
    )

    canned = _make_completion(confidence=0.73)

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert (
        abs(decision["confidence"] - 0.73) < 1e-6
    ), f"confidence changed: {decision['confidence']}"


def test_route_disables_thinking_and_caps_tokens() -> None:
    """route must cap max_tokens and disable thinking so a thinking-mode model
    returns a terse parseable decision without running past the client timeout
    (regression for the live TimeoutError on the 4B's <think> trace)."""
    p = _make_parser()
    args = p.parse_args(
        ["route", "hello", "--model", "m", "--base-url", "http://localhost/v1", "--json"]
    )
    canned = _make_completion()
    with patch("lobes.cli._commands.route.chat_completion", return_value=canned) as mock_cc:
        route.cmd_route(args)

    kwargs = mock_cc.call_args.kwargs
    assert kwargs.get("max_tokens"), "route must pass a max_tokens cap"
    assert (
        kwargs.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False
    ), "route must disable thinking via chat_template_kwargs"


# ---------------------------------------------------------------------------
# Plain-text output mode
# ---------------------------------------------------------------------------


def test_text_mode_contains_all_field_names(capsys: pytest.CaptureFixture[str]) -> None:
    """Plain-text (non-JSON) mode prints chosen_gear, escalate, confidence, reason labels."""
    p = _make_parser()
    args = p.parse_args(["route", "summarize", "--model", "m", "--base-url", "http://localhost/v1"])

    canned = _make_completion(chosen_gear="primary", confidence=0.8, reason="Complex task")

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        rc = route.cmd_route(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "chosen_gear" in out
    assert "escalate" in out
    assert "confidence" in out
    assert "reason" in out


def test_text_mode_is_not_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Plain-text mode does NOT emit a top-level JSON object to stdout."""
    p = _make_parser()
    args = p.parse_args(["route", "hello", "--model", "m", "--base-url", "http://localhost/v1"])

    canned = _make_completion()

    with patch("lobes.cli._commands.route.chat_completion", return_value=canned):
        route.cmd_route(args)

    out = capsys.readouterr().out.strip()
    try:
        json.loads(out)
        is_json = True
    except json.JSONDecodeError:
        is_json = False
    assert not is_json, "plain-text mode must not emit a raw JSON object to stdout"


# ---------------------------------------------------------------------------
# Default base URL
# ---------------------------------------------------------------------------


def test_default_base_url_is_gateway() -> None:
    """The default base-url is http://localhost:8000/v1 (the fleet gateway)."""
    p = _make_parser()
    args = p.parse_args(["route", "hello", "--model", "m"])
    assert args.base_url == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# Read-only contract
# ---------------------------------------------------------------------------


def test_route_no_file_writes(
    tmp_path: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    """Handler never touches the filesystem (read-only)."""
    p = _make_parser()
    args = p.parse_args(["route", "hello", "--model", "m", "--base-url", "http://localhost/v1"])

    files_before = set(tmp_path.iterdir())
    with patch("lobes.cli._commands.route.chat_completion", return_value=_make_completion()):
        route.cmd_route(args)
    files_after = set(tmp_path.iterdir())

    assert files_before == files_after, "read-only verb must not create files"
