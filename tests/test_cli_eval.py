"""Tests for ``lobes eval minor`` — JSONL eval-suite runner (read-only verb).

The handler is tested directly (without wiring through main()) because t8
handles CLI registration; this module only needs the verb to be importable
and runnable as a standalone handler.

Monkeypatching strategy: the handler calls ``lobes.minor.chat_text`` (imported
into the eval module's namespace).  The tests patch the reference in the eval
module (``eval_cmd.chat_text``) so responses are deterministic and no HTTP
traffic is made.
"""

from __future__ import annotations

import argparse
import json
import types

import pytest

from lobes.cli._commands import eval as eval_cmd
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> types.SimpleNamespace:
    """Build an argparse Namespace matching what register() produces for 'eval minor'."""
    defaults: dict = {
        "json": False,
        "base_url": "http://localhost:8001/v1",
        "model": "test-model",
        "timeout": 10,
        "suite": None,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Core pass/fail test (the main acceptance criterion)
# ---------------------------------------------------------------------------


def test_eval_minor_one_pass_one_fail(tmp_path, monkeypatch, capsys) -> None:
    """A 2-case suite with one passing and one failing expectation produces 1/2."""
    suite = tmp_path / "suite.jsonl"
    # Case 0 expects "world" in the response → pass (response contains it).
    # Case 1 expects "goodbye" in the response → fail (response does not contain it).
    suite.write_text(
        '{"prompt": "say hello", "expect_substring": "world"}\n'
        '{"prompt": "say farewell", "expect_substring": "goodbye"}\n',
        encoding="utf-8",
    )

    responses = iter(["hello, world!", "see ya"])
    monkeypatch.setattr(eval_cmd, "chat_text", lambda *a, **kw: next(responses))

    args = _make_args(suite=str(suite))
    rc = eval_cmd.cmd_eval_minor(args)

    assert rc == 0  # read-only verb — always returns 0 (pass/fail is in the report)
    out = capsys.readouterr().out
    assert "1/2" in out  # aggregate: 1 passed out of 2 total
    assert "PASS" in out
    assert "FAIL" in out


# ---------------------------------------------------------------------------
# Missing-suite path
# ---------------------------------------------------------------------------


def test_eval_minor_missing_suite(tmp_path) -> None:
    """A nonexistent suite path raises ModelGearError with EXIT_USER_ERROR."""
    args = _make_args(suite=str(tmp_path / "does_not_exist.jsonl"))
    with pytest.raises(ModelGearError) as exc:
        eval_cmd.cmd_eval_minor(args)
    assert exc.value.code == EXIT_USER_ERROR
    assert "not found" in exc.value.message.lower()


# ---------------------------------------------------------------------------
# Empty-suite path
# ---------------------------------------------------------------------------


def test_eval_minor_empty_suite(tmp_path, capsys) -> None:
    """An empty (or all-blank/comment) suite reports 0/0 without crashing."""
    suite = tmp_path / "empty.jsonl"
    suite.write_text("", encoding="utf-8")

    args = _make_args(suite=str(suite))
    rc = eval_cmd.cmd_eval_minor(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "0/0" in out


def test_eval_minor_comment_only_suite(tmp_path, capsys) -> None:
    """Lines starting with '#' are skipped; suite with only comments is 0/0."""
    suite = tmp_path / "comments.jsonl"
    suite.write_text(
        "# this is a comment\n\n# another comment\n",
        encoding="utf-8",
    )

    args = _make_args(suite=str(suite))
    rc = eval_cmd.cmd_eval_minor(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "0/0" in out


# ---------------------------------------------------------------------------
# JSON output mode
# ---------------------------------------------------------------------------


def test_eval_minor_json_output(tmp_path, monkeypatch, capsys) -> None:
    """--json emits a structured dict with 'passed', 'total', and 'cases'."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        '{"prompt": "q1", "expect_substring": "yes"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_cmd, "chat_text", lambda *a, **kw: "yes, definitely")

    args = _make_args(suite=str(suite), json=True)
    rc = eval_cmd.cmd_eval_minor(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] == 1
    assert payload["total"] == 1
    assert isinstance(payload["cases"], list)
    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert case["pass"] is True
    assert case["prompt"] == "q1"


def test_eval_minor_json_output_with_fail(tmp_path, monkeypatch, capsys) -> None:
    """JSON report correctly marks a case as pass=False when the expectation fails."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        '{"prompt": "q1", "expect_regex": "^exact$"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_cmd, "chat_text", lambda *a, **kw: "not an exact match")

    args = _make_args(suite=str(suite), json=True)
    eval_cmd.cmd_eval_minor(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] == 0
    assert payload["total"] == 1
    assert payload["cases"][0]["pass"] is False


# ---------------------------------------------------------------------------
# Regex expectation
# ---------------------------------------------------------------------------


def test_eval_minor_expect_regex_pass(tmp_path, monkeypatch, capsys) -> None:
    """expect_regex uses re.search; a matching response passes."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        '{"prompt": "p", "expect_regex": "\\\\d+"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_cmd, "chat_text", lambda *a, **kw: "the answer is 42")

    args = _make_args(suite=str(suite), json=True)
    eval_cmd.cmd_eval_minor(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["cases"][0]["pass"] is True


def test_eval_minor_expect_regex_fail(tmp_path, monkeypatch, capsys) -> None:
    """expect_regex that does not match the response produces pass=False."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        '{"prompt": "p", "expect_regex": "^\\\\d+$"}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_cmd, "chat_text", lambda *a, **kw: "no digits here")

    args = _make_args(suite=str(suite), json=True)
    eval_cmd.cmd_eval_minor(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["cases"][0]["pass"] is False


# ---------------------------------------------------------------------------
# register() creates a usable argparse sub-parser (importable standalone)
# ---------------------------------------------------------------------------


def test_register_creates_eval_parser() -> None:
    """register() attaches an 'eval' sub-parser with a working 'minor' sub-command."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    eval_cmd.register(sub)

    args = parser.parse_args(["eval", "minor", "--suite", "my.jsonl"])
    assert args.suite == "my.jsonl"
    assert args.func is eval_cmd.cmd_eval_minor


def test_register_default_base_url() -> None:
    """The default --base-url is set and non-empty."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    eval_cmd.register(sub)

    args = parser.parse_args(["eval", "minor", "--suite", "s.jsonl"])
    assert args.base_url  # non-empty
    assert "localhost" in args.base_url  # local endpoint by default
