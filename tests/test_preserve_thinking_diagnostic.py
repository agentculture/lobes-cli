"""Tests for the two-turn preserve_thinking token-delta diagnostic (issue #93, t3).

All hermetic — no live server. ``_post`` / ``_get`` are monkeypatched. The fake
server models the observable fact the diagnostic proves: a turn-2 request's
``usage.prompt_tokens`` *rises* when the assistant turn in history carries a
``<think>`` / ``reasoning`` trace (the chat template re-renders it), and stays at
baseline when the assistant turn is content-only.
"""

from __future__ import annotations

import builtins
import json

import pytest

import lobes.assess as A
from lobes.cli import main

# The fake token accounting: a two-turn conversation costs _BASE_PROMPT_TOKENS,
# plus _REASONING_TOKENS extra when a reasoning trace survives into history.
_BASE_PROMPT_TOKENS = 40
_REASONING_TOKENS = 25


def _fake_get(url, path, timeout=10):
    if path == "/health":
        return 200, {"status": "ok"}
    if path == "/v1/models":
        return 200, {"data": [{"id": "foo/bar", "max_model_len": 32768}]}
    return 200, {}


def _reasoning_in_history(messages: list[dict]) -> bool:
    """True if any assistant message carries a reasoning trace (field or <think>)."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for key in ("reasoning", "reasoning_content"):
            val = m.get(key)
            if isinstance(val, str) and val:
                return True
        if "<think>" in (m.get("content") or ""):
            return True
    return False


def _make_fake_post(
    *, reasoning_key: str = "reasoning", turn1_reasoning: str | None = "step1 step2"
):
    """Fake ``_post``: turn-1 emits a trace; turn-2 prompt_tokens rise iff it survives.

    ``turn1_reasoning=None`` simulates a server that returned no trace on turn 1,
    so the preserved probe has nothing to re-send and the delta collapses to ~0.
    """

    def _post(url, payload, timeout=300):
        messages = payload["messages"]
        # Turn 1 is a single user message → return content + (maybe) a trace.
        if len(messages) == 1 and messages[0].get("role") == "user":
            msg: dict = {"content": "The ball costs $0.05."}
            if turn1_reasoning:
                msg[reasoning_key] = turn1_reasoning
            return {
                "choices": [{"message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": _BASE_PROMPT_TOKENS, "completion_tokens": 12},
            }
        # Turn 2: measure prompt_tokens; the trace re-rendering costs extra.
        pt = _BASE_PROMPT_TOKENS
        if _reasoning_in_history(messages):
            pt += _REASONING_TOKENS
        return {
            "choices": [{"message": {"content": "0.10"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": pt, "completion_tokens": 4},
        }

    return _post


# ---------------------------------------------------------------------------
# Acceptance criterion 1 + 3 — the probe function
# ---------------------------------------------------------------------------


def test_probe_returns_counts_and_positive_delta(monkeypatch) -> None:
    """Both prompt-token counts are reported and the delta is positive (preserved)."""
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _make_fake_post())
    r = A.run_preserve_thinking_probe("http://localhost:8000", "foo/bar")
    assert r["with_reasoning"] == _BASE_PROMPT_TOKENS + _REASONING_TOKENS
    assert r["content_only"] == _BASE_PROMPT_TOKENS
    assert r["delta"] == _REASONING_TOKENS
    assert r["delta"] > 0
    assert r["trace_field"] == "reasoning"
    assert r["model"] == "foo/bar"


def test_probe_detects_reasoning_content_key(monkeypatch) -> None:
    """The alternate ``reasoning_content`` trace field is also detected + preserved."""
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _make_fake_post(reasoning_key="reasoning_content"))
    r = A.run_preserve_thinking_probe("http://localhost:8000", "foo/bar")
    assert r["trace_field"] == "reasoning_content"
    assert r["delta"] > 0


def test_probe_delta_zero_without_reasoning(monkeypatch) -> None:
    """When turn 1 yields no trace, nothing is preserved and the delta is ~0."""
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _make_fake_post(turn1_reasoning=None))
    r = A.run_preserve_thinking_probe("http://localhost:8000", "foo/bar")
    assert r["with_reasoning"] == r["content_only"]
    assert r["delta"] == 0
    assert r["trace_field"] == "(none)"


def test_probe_holds_preserve_thinking_constant_across_requests(monkeypatch) -> None:
    """Confound guard (Qodo #94): both turn-2 requests set the SAME
    ``preserve_thinking`` flag, so the only variable is whether the assistant
    history carries the reasoning trace — the delta is attributable solely to the
    reasoning replay, not to the flag.
    """
    monkeypatch.setattr(A, "_get", _fake_get)
    captured: list[dict] = []
    base_post = _make_fake_post()

    def _capturing_post(url, payload, timeout=300):
        captured.append(payload)
        return base_post(url, payload, timeout)

    monkeypatch.setattr(A, "_post", _capturing_post)
    A.run_preserve_thinking_probe("http://localhost:8000", "foo/bar")

    turn2 = [p for p in captured if len(p["messages"]) > 1]
    assert len(turn2) == 2, "expected exactly two turn-2 requests (A and B)"
    # Both requests carry identical chat_template_kwargs — the flag is held constant.
    for p in turn2:
        assert p.get("chat_template_kwargs") == {"preserve_thinking": True}
    # Exactly one request re-sends reasoning in history (A); the other is content-only (B).
    with_reasoning = [p for p in turn2 if _reasoning_in_history(p["messages"])]
    assert len(with_reasoning) == 1, "the reasoning history must be the ONLY variable"


def test_probe_performs_no_file_writes(monkeypatch) -> None:
    """Read-only contract: the probe must never open a file for writing."""
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _make_fake_post())
    real_open = builtins.open

    def _guard_open(file, mode="r", *a, **k):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"probe attempted a file write: {file!r} (mode={mode!r})")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", _guard_open)
    r = A.run_preserve_thinking_probe("http://localhost:8000", "foo/bar")
    assert r["delta"] > 0


def test_render_preserve_thinking_shows_counts_and_delta(monkeypatch) -> None:
    """The render helper surfaces both counts and the delta in its text."""
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _make_fake_post())
    r = A.run_preserve_thinking_probe("http://localhost:8000", "foo/bar")
    text = A.render_preserve_thinking(r)
    assert str(r["with_reasoning"]) in text
    assert str(r["content_only"]) in text
    assert "delta" in text.lower()


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — the read-only CLI surface (--preserve-thinking flag)
# ---------------------------------------------------------------------------


def test_cli_preserve_thinking_text(monkeypatch, capsys) -> None:
    """`lobes assess --preserve-thinking` prints both counts + delta, exits 0."""
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _make_fake_post())
    rc = main(["assess", "--port", "8000", "--preserve-thinking"])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(_BASE_PROMPT_TOKENS + _REASONING_TOKENS) in out  # with_reasoning
    assert str(_BASE_PROMPT_TOKENS) in out  # content_only
    assert "delta" in out.lower()


def test_cli_preserve_thinking_json(monkeypatch, capsys) -> None:
    """`--preserve-thinking --json` emits a structured dict with the three fields."""
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _make_fake_post())
    rc = main(["assess", "--port", "8000", "--preserve-thinking", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["with_reasoning"] == _BASE_PROMPT_TOKENS + _REASONING_TOKENS
    assert payload["content_only"] == _BASE_PROMPT_TOKENS
    assert payload["delta"] == _REASONING_TOKENS


def test_cli_preserve_thinking_exit_zero_when_not_preserved(monkeypatch, capsys) -> None:
    """A ~0 delta still exits 0 — it's a diagnostic, the verdict lives in output."""
    monkeypatch.setattr(A, "_get", _fake_get)
    monkeypatch.setattr(A, "_post", _make_fake_post(turn1_reasoning=None))
    rc = main(["assess", "--port", "8000", "--preserve-thinking", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["delta"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
