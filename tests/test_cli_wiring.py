"""Tests for CLI top-level wiring — run, route, eval registered in __init__.py.

These tests call ``_build_parser()`` the same way ``main()`` does and verify
that the three new verbs (``run``, ``route``, ``eval``) are reachable via the
top-level dispatcher.  No real HTTP is made: handlers are not invoked, only
the parsed ``args.func`` attribute is inspected.
"""

from __future__ import annotations

import pytest

from lobes.cli import _build_parser
from lobes.cli._commands import eval as eval_cmd
from lobes.cli._commands import route as route_cmd
from lobes.cli._commands import run as run_cmd

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse(argv: list[str]):
    """Build the top-level parser (same as main()) and parse *argv*."""
    parser = _build_parser()
    return parser, parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Verb registration — each verb appears in --help output
# ---------------------------------------------------------------------------


def test_help_contains_run(capsys: pytest.CaptureFixture[str]) -> None:
    """'run' appears in the top-level --help output."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "run" in out, "'run' not found in --help output"


def test_help_contains_route(capsys: pytest.CaptureFixture[str]) -> None:
    """'route' appears in the top-level --help output."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "route" in out, "'route' not found in --help output"


def test_help_contains_eval(capsys: pytest.CaptureFixture[str]) -> None:
    """'eval' appears in the top-level --help output."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "eval" in out, "'eval' not found in --help output"


# ---------------------------------------------------------------------------
# Dispatch — args.func maps to the correct handler
# ---------------------------------------------------------------------------


def test_run_dispatches_to_cmd_run_minor() -> None:
    """``lobes run minor <prompt> --model m`` sets args.func to run's handler."""
    _, args = _parse(["run", "minor", "hi", "--model", "m"])
    assert args.func is run_cmd.cmd_run_minor


def test_route_dispatches_to_cmd_route() -> None:
    """``lobes route <text> --model m`` sets args.func to route's handler."""
    _, args = _parse(["route", "hello", "--model", "m"])
    assert args.func is route_cmd.cmd_route


def test_eval_dispatches_to_cmd_eval_minor() -> None:
    """``lobes eval minor --suite s.jsonl`` sets args.func to eval's handler."""
    _, args = _parse(["eval", "minor", "--suite", "suite.jsonl"])
    assert args.func is eval_cmd.cmd_eval_minor


# ---------------------------------------------------------------------------
# Eval default endpoint is the gateway (port 8000, not 8001)
# ---------------------------------------------------------------------------


def test_eval_default_base_url_is_gateway() -> None:
    """The eval verb default --base-url is http://localhost:8000/v1 (gateway port)."""
    _, args = _parse(["eval", "minor", "--suite", "s.jsonl"])
    assert (
        args.base_url == "http://localhost:8000/v1"
    ), f"expected http://localhost:8000/v1, got {args.base_url!r}"
