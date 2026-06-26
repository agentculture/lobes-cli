"""Tests for ``lobes run minor`` — read-only prompt-to-lobe verb.

Uses a throwaway argparse parser so the test is independent of
lobes/cli/__init__.py wiring (which is done in a separate task, t8).
No real HTTP ever happens: the client is monkeypatched in every test.
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import pytest

from lobes.cli._commands import run
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError

# ---------------------------------------------------------------------------
# Canned fixtures
# ---------------------------------------------------------------------------

_CANNED_TEXT = "Forty-two."
_CANNED_COMPLETION = {
    "id": "chatcmpl-t5-001",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": _CANNED_TEXT},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}

# ---------------------------------------------------------------------------
# Helper: build a throwaway parser and register the run verb
# ---------------------------------------------------------------------------


def _make_parser() -> argparse.ArgumentParser:
    """Return a fresh ArgumentParser with the ``run`` subcommand registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    run.register(sub)
    return p


# ---------------------------------------------------------------------------
# Importability / registration
# ---------------------------------------------------------------------------


def test_register_is_importable_standalone() -> None:
    """run.register can be called on a plain argparse parser with no CLI wiring."""
    p = _make_parser()
    # Parse a full argv so we confirm the subparser was wired correctly.
    args = p.parse_args(["run", "minor", "hello world", "--model", "test-model"])
    assert args.lobe == "minor"
    assert args.prompt == "hello world"
    assert args.model == "test-model"


def test_register_exposes_func_default() -> None:
    """Parsed args carry a ``func`` attribute (the handler)."""
    p = _make_parser()
    args = p.parse_args(["run", "minor", "hi", "--model", "m"])
    assert callable(args.func)


# ---------------------------------------------------------------------------
# Text mode (default)
# ---------------------------------------------------------------------------


def test_run_minor_prints_text(capsys: pytest.CaptureFixture[str]) -> None:
    """Default (non-JSON) mode prints the assistant text to stdout."""
    p = _make_parser()
    args = p.parse_args(
        ["run", "minor", "hello", "--model", "test-model", "--base-url", "http://localhost/v1"]
    )

    with patch("lobes.cli._commands.run.chat_text", return_value=_CANNED_TEXT) as mock_ct:
        rc = run.cmd_run_minor(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert _CANNED_TEXT in out
    mock_ct.assert_called_once()


def test_run_minor_text_no_json_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Text mode does NOT write JSON to stdout."""
    p = _make_parser()
    args = p.parse_args(["run", "minor", "hi", "--model", "m", "--base-url", "http://h/v1"])

    with patch("lobes.cli._commands.run.chat_text", return_value="plain text"):
        run.cmd_run_minor(args)

    out = capsys.readouterr().out.strip()
    # Must not be valid JSON (it's just the plain assistant text).
    try:
        json.loads(out)
        is_json = True
    except json.JSONDecodeError:
        is_json = False
    assert not is_json, "text mode must not emit JSON to stdout"


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------


def test_run_minor_json_emits_structured(capsys: pytest.CaptureFixture[str]) -> None:
    """``--json`` mode emits a structured chat-completion object to stdout."""
    p = _make_parser()
    args = p.parse_args(
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
        rc = run.cmd_run_minor(args)

    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["id"] == "chatcmpl-t5-001"
    assert payload["choices"][0]["message"]["content"] == _CANNED_TEXT


def test_run_minor_json_nothing_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """JSON mode must not write diagnostics to stdout (strict stdout/stderr split)."""
    p = _make_parser()
    args = p.parse_args(
        ["run", "minor", "hi", "--model", "m", "--base-url", "http://h/v1", "--json"]
    )

    with patch("lobes.cli._commands.run.chat_completion", return_value=_CANNED_COMPLETION):
        run.cmd_run_minor(args)

    err = capsys.readouterr().err
    assert err == "", f"unexpected stderr in JSON mode: {err!r}"


# ---------------------------------------------------------------------------
# Read-only contract
# ---------------------------------------------------------------------------


def test_run_minor_no_file_writes(tmp_path, capsys) -> None:
    """Handler never touches the filesystem (read-only contract: no .env, no compose)."""
    p = _make_parser()
    args = p.parse_args(
        ["run", "minor", "hello", "--model", "test-model", "--base-url", "http://localhost/v1"]
    )

    files_before = set(tmp_path.iterdir())
    with patch("lobes.cli._commands.run.chat_text", return_value=_CANNED_TEXT):
        run.cmd_run_minor(args)
    files_after = set(tmp_path.iterdir())

    assert files_before == files_after, "read-only verb must not create files"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_run_wrong_lobe_raises_error() -> None:
    """Passing a lobe name other than 'minor' raises ModelGearError (EXIT_USER_ERROR)."""
    p = _make_parser()
    args = p.parse_args(["run", "minor", "hello", "--model", "m"])
    args.lobe = "primary"  # force invalid lobe name past argparse

    with pytest.raises(ModelGearError) as exc:
        run.cmd_run_minor(args)
    assert exc.value.code == EXIT_USER_ERROR
    assert "minor" in exc.value.message.lower() or "unsupported" in exc.value.message.lower()


def test_run_no_catalog_minor_and_no_model_flag_raises() -> None:
    """When the catalog has no 'minor' model and --model isn't supplied, raise ModelGearError."""
    p = _make_parser()
    # --model is NOT passed; catalog currently has no role_hint=='minor' entry.
    args = p.parse_args(["run", "minor", "hello", "--base-url", "http://localhost/v1"])
    # Clear the model attr to simulate no --model flag.
    args.model = None

    with pytest.raises(ModelGearError) as exc:
        run.cmd_run_minor(args)
    assert exc.value.code == EXIT_USER_ERROR
    assert "minor" in exc.value.message.lower() or "catalog" in exc.value.message.lower()


# ---------------------------------------------------------------------------
# Optional flags forwarded to the client
# ---------------------------------------------------------------------------


def test_run_system_flag_forwarded(capsys) -> None:
    """``--system`` is passed through to the underlying client call."""
    p = _make_parser()
    args = p.parse_args(
        [
            "run",
            "minor",
            "hello",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--system",
            "Be concise.",
        ]
    )

    with patch("lobes.cli._commands.run.chat_text", return_value=_CANNED_TEXT) as mock_ct:
        run.cmd_run_minor(args)

    call_kwargs = mock_ct.call_args.kwargs
    assert call_kwargs.get("system") == "Be concise."


def test_run_max_tokens_flag_forwarded(capsys) -> None:
    """``--max-tokens`` is parsed as int and forwarded to the client."""
    p = _make_parser()
    args = p.parse_args(
        [
            "run",
            "minor",
            "hello",
            "--model",
            "test-model",
            "--base-url",
            "http://localhost/v1",
            "--max-tokens",
            "128",
        ]
    )

    with patch("lobes.cli._commands.run.chat_text", return_value=_CANNED_TEXT) as mock_ct:
        run.cmd_run_minor(args)

    call_kwargs = mock_ct.call_args.kwargs
    assert call_kwargs.get("max_tokens") == 128


def test_run_model_and_base_url_forwarded(capsys) -> None:
    """``--model`` and ``--base-url`` reach the client unchanged."""
    p = _make_parser()
    args = p.parse_args(
        [
            "run",
            "minor",
            "hi",
            "--model",
            "my-special-model",
            "--base-url",
            "http://myhost:9999/v1",
        ]
    )

    with patch("lobes.cli._commands.run.chat_text", return_value="ok") as mock_ct:
        run.cmd_run_minor(args)

    call_kwargs = mock_ct.call_args.kwargs
    assert call_kwargs.get("model") == "my-special-model"
    assert call_kwargs.get("base_url") == "http://myhost:9999/v1"
