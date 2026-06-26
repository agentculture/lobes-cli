"""End-to-end integration tests: minor-lobe success signals and read-only safety.

Covers spec announcement c1, safety h1, success signals c9, observability h12.

Differences from per-verb unit tests (test_cli_run.py, test_cli_route.py,
test_cli_eval.py):

- Exercises the WIRED top-level parser (_build_parser / _dispatch / main())
  exactly the way main() does — not a throwaway per-verb parser.
- Covers run, route, and eval together in one end-to-end suite.
- Asserts read-only safety for all three verbs (no --apply, no file writes,
  no .env or docker-compose created).
- Confirms catalog+parser agreement for the minor gear (Qwen/Qwen3.5-4B).

All tests are fully offline: lobes.minor._client functions are monkeypatched at
the handler-module path (e.g. ``lobes.cli._commands.run.chat_text``) so no real
HTTP ever happens.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from lobes.catalog import supported_models
from lobes.cli import _build_parser, _dispatch, main
from lobes.runtime._parser import infer_parser

# ---------------------------------------------------------------------------
# Canned stubs
# ---------------------------------------------------------------------------

_CANNED_TEXT = "The answer is forty-two."

_CANNED_COMPLETION = {
    "id": "chatcmpl-integration-001",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": _CANNED_TEXT},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
}

# Routing response: model suggests "minor" gear with no escalation conditions.
_ROUTE_CONTENT = json.dumps(
    {
        "chosen_gear": "minor",
        "confidence": 0.88,
        "reason": "Simple formatting task; minor is appropriate.",
        "conditions": [],
    }
)

_CANNED_ROUTE_COMPLETION = {
    "id": "chatcmpl-route-integration-001",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": _ROUTE_CONTENT},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse(argv: list[str]):
    """Build the full top-level parser (same as main()) and parse *argv*."""
    parser = _build_parser()
    return parser, parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 1. Catalog + parser agreement for the minor gear (c1)
# ---------------------------------------------------------------------------


def test_catalog_has_minor_gear_qwen3_5_4b() -> None:
    """The catalog has at least one role_hint='minor' entry with id='Qwen/Qwen3.5-4B'."""
    minor_models = [m for m in supported_models() if m.role_hint == "minor"]
    assert minor_models, "catalog must have at least one role_hint='minor' entry"
    ids = [m.id for m in minor_models]
    assert "Qwen/Qwen3.5-4B" in ids, f"expected 'Qwen/Qwen3.5-4B' in minor-role models; got {ids}"


def test_infer_parser_minor_gear_is_qwen3_coder() -> None:
    """infer_parser for 'Qwen/Qwen3.5-4B' resolves to 'qwen3_coder'."""
    assert infer_parser("Qwen/Qwen3.5-4B") == "qwen3_coder"


def test_catalog_and_infer_parser_agree_for_minor_gear() -> None:
    """The catalog's tool_parser field matches infer_parser for the minor gear id."""
    minor = next(
        (m for m in supported_models() if m.id == "Qwen/Qwen3.5-4B"),
        None,
    )
    assert minor is not None, "Qwen/Qwen3.5-4B not found in catalog"
    inferred = infer_parser(minor.id)
    assert inferred == minor.tool_parser, (
        f"infer_parser({minor.id!r})={inferred!r} does not match "
        f"catalog tool_parser={minor.tool_parser!r}"
    )


# ---------------------------------------------------------------------------
# 2. lobes run minor — wired top-level parser → exit 0, text to stdout (c9)
# ---------------------------------------------------------------------------


def test_run_minor_wired_parser_text_exit0(capsys: pytest.CaptureFixture[str]) -> None:
    """'lobes run minor <prompt>' via the wired parser: exit 0, stub text on stdout."""
    _, args = _parse(
        [
            "run",
            "minor",
            "hello world",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
        ]
    )

    with patch("lobes.cli._commands.run.chat_text", return_value=_CANNED_TEXT):
        rc = _dispatch(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert _CANNED_TEXT in out


def test_run_minor_via_main_exit0(capsys: pytest.CaptureFixture[str]) -> None:
    """main(['run', 'minor', ...]) returns 0 with the stub text on stdout."""
    with patch("lobes.cli._commands.run.chat_text", return_value=_CANNED_TEXT):
        rc = main(
            [
                "run",
                "minor",
                "hello",
                "--model",
                "test-model",
                "--base-url",
                "http://localhost/v1",
            ]
        )

    assert rc == 0
    assert _CANNED_TEXT in capsys.readouterr().out


def test_run_minor_json_mode_via_wired_parser(capsys: pytest.CaptureFixture[str]) -> None:
    """'lobes run minor --json' via the wired parser emits a chat-completion JSON object."""
    _, args = _parse(
        [
            "run",
            "minor",
            "hello",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )

    with patch("lobes.cli._commands.run.chat_completion", return_value=_CANNED_COMPLETION):
        rc = _dispatch(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["choices"][0]["message"]["content"] == _CANNED_TEXT


# ---------------------------------------------------------------------------
# 3. lobes route — wired top-level parser → structured decision (c9)
# ---------------------------------------------------------------------------


def test_route_wired_parser_json_decision_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """'lobes route <text> --json' returns chosen_gear, escalate, confidence∈[0,1]."""
    _, args = _parse(
        [
            "route",
            "format this PR title",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )

    with patch(
        "lobes.cli._commands.route.chat_completion",
        return_value=_CANNED_ROUTE_COMPLETION,
    ):
        rc = _dispatch(args)

    assert rc == 0
    decision = json.loads(capsys.readouterr().out)
    assert "chosen_gear" in decision
    assert "escalate" in decision
    assert "confidence" in decision
    assert isinstance(decision["escalate"], bool)
    assert 0.0 <= decision["confidence"] <= 1.0


def test_route_benign_task_escalate_false(capsys: pytest.CaptureFixture[str]) -> None:
    """Benign task with no escalation conditions → escalate=False."""
    _, args = _parse(
        [
            "route",
            "summarize this paragraph",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )

    with patch(
        "lobes.cli._commands.route.chat_completion",
        return_value=_CANNED_ROUTE_COMPLETION,
    ):
        rc = _dispatch(args)

    assert rc == 0
    decision = json.loads(capsys.readouterr().out)
    assert decision["escalate"] is False


def test_route_via_main_exit0(capsys: pytest.CaptureFixture[str]) -> None:
    """main(['route', ...]) returns 0."""
    with patch(
        "lobes.cli._commands.route.chat_completion",
        return_value=_CANNED_ROUTE_COMPLETION,
    ):
        rc = main(
            [
                "route",
                "format this text",
                "--model",
                "test-model",
                "--base-url",
                "http://localhost/v1",
                "--json",
            ]
        )

    assert rc == 0


# ---------------------------------------------------------------------------
# 4. lobes eval minor — wired top-level parser → exit 0, report on stdout (c9)
# ---------------------------------------------------------------------------


def test_eval_minor_wired_parser_exit0_report(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """'lobes eval minor --suite ...' via the wired parser: exit 0, pass report on stdout."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"prompt": "hello", "expect_substring": "world"}\n', encoding="utf-8")

    _, args = _parse(
        [
            "eval",
            "minor",
            "--suite",
            str(suite),
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
        ]
    )

    with patch("lobes.cli._commands.eval.chat_text", return_value="hello world"):
        rc = _dispatch(args)

    assert rc == 0
    out = capsys.readouterr().out
    # Text mode must mention pass count or "passed".
    assert "1/1" in out or "passed" in out.lower()


def test_eval_minor_json_via_main(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """main(['eval', 'minor', '--suite', ..., '--json']) emits structured JSON with passed/total."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"prompt": "test", "expect_substring": "ok"}\n', encoding="utf-8")

    with patch("lobes.cli._commands.eval.chat_text", return_value="ok"):
        rc = main(
            [
                "eval",
                "minor",
                "--suite",
                str(suite),
                "--model",
                "test-model",
                "--base-url",
                "http://localhost/v1",
                "--json",
            ]
        )

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] == 1
    assert report["total"] == 1


# ---------------------------------------------------------------------------
# 5. Read-only safety — verbs expose no --apply (h1)
# ---------------------------------------------------------------------------


def test_run_verb_has_no_apply() -> None:
    """'lobes run minor' argparse namespace has no 'apply' attribute."""
    _, args = _parse(["run", "minor", "hello", "--model", "m"])
    assert not hasattr(args, "apply"), "run must not expose --apply (it is read-only)"


def test_route_verb_has_no_apply() -> None:
    """'lobes route' argparse namespace has no 'apply' attribute."""
    _, args = _parse(["route", "hello", "--model", "m"])
    assert not hasattr(args, "apply"), "route must not expose --apply (it is read-only)"


def test_eval_verb_has_no_apply(tmp_path) -> None:
    """'lobes eval minor' argparse namespace has no 'apply' attribute."""
    _, args = _parse(["eval", "minor", "--suite", str(tmp_path / "s.jsonl")])
    assert not hasattr(args, "apply"), "eval must not expose --apply (it is read-only)"


# ---------------------------------------------------------------------------
# 6. Read-only safety — no file writes during run / route / eval (h1)
# ---------------------------------------------------------------------------


def test_run_minor_does_not_create_files(tmp_path, capsys) -> None:
    """'lobes run minor' via the wired parser must not create any files."""
    _, args = _parse(
        [
            "run",
            "minor",
            "hello",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
        ]
    )
    before = set(tmp_path.iterdir())
    with patch("lobes.cli._commands.run.chat_text", return_value=_CANNED_TEXT):
        _dispatch(args)
    assert set(tmp_path.iterdir()) == before, "run must not create files"


def test_route_does_not_create_files(tmp_path, capsys) -> None:
    """'lobes route' via the wired parser must not create any files."""
    _, args = _parse(
        [
            "route",
            "hello",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )
    before = set(tmp_path.iterdir())
    with patch(
        "lobes.cli._commands.route.chat_completion",
        return_value=_CANNED_ROUTE_COMPLETION,
    ):
        _dispatch(args)
    assert set(tmp_path.iterdir()) == before, "route must not create files"


def test_eval_minor_does_not_create_extra_files(tmp_path, capsys) -> None:
    """'lobes eval minor' must not create files beyond the suite it was given."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"prompt": "hi", "expect_substring": "hi"}\n', encoding="utf-8")
    before = set(tmp_path.iterdir())

    _, args = _parse(
        [
            "eval",
            "minor",
            "--suite",
            str(suite),
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
        ]
    )
    with patch("lobes.cli._commands.eval.chat_text", return_value="hi"):
        _dispatch(args)

    assert set(tmp_path.iterdir()) == before, "eval must not create new files"


# ---------------------------------------------------------------------------
# 7. Read-only safety — no .env or docker-compose written (h1, explicit names)
# ---------------------------------------------------------------------------


def test_run_does_not_write_env_or_compose(tmp_path, monkeypatch, capsys) -> None:
    """run verb must not write .env or docker-compose to the working directory."""
    monkeypatch.chdir(tmp_path)
    _, args = _parse(
        [
            "run",
            "minor",
            "hello",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
        ]
    )
    with patch("lobes.cli._commands.run.chat_text", return_value=_CANNED_TEXT):
        _dispatch(args)

    assert not (tmp_path / ".env").exists(), ".env must not be created by run"
    assert not (
        tmp_path / "docker-compose.yml"
    ).exists(), "docker-compose.yml must not be created by run"


def test_route_does_not_write_env_or_compose(tmp_path, monkeypatch, capsys) -> None:
    """route verb must not write .env or docker-compose to the working directory."""
    monkeypatch.chdir(tmp_path)
    _, args = _parse(
        [
            "route",
            "hello",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--json",
        ]
    )
    with patch(
        "lobes.cli._commands.route.chat_completion",
        return_value=_CANNED_ROUTE_COMPLETION,
    ):
        _dispatch(args)

    assert not (tmp_path / ".env").exists(), ".env must not be created by route"
    assert not (
        tmp_path / "docker-compose.yml"
    ).exists(), "docker-compose.yml must not be created by route"


def test_eval_does_not_write_env_or_compose(tmp_path, monkeypatch, capsys) -> None:
    """eval verb must not write .env or docker-compose to the working directory."""
    monkeypatch.chdir(tmp_path)
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"prompt": "hi", "expect_substring": "hi"}\n', encoding="utf-8")
    _, args = _parse(
        [
            "eval",
            "minor",
            "--suite",
            str(suite),
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
        ]
    )
    with patch("lobes.cli._commands.eval.chat_text", return_value="hi"):
        _dispatch(args)

    assert not (tmp_path / ".env").exists(), ".env must not be created by eval"
    assert not (
        tmp_path / "docker-compose.yml"
    ).exists(), "docker-compose.yml must not be created by eval"
