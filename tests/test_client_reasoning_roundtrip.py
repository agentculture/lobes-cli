"""Tests for reasoning-aware round-tripping in lobes.minor._client (#93).

Preserves a Qwen thinking-mode ``reasoning``/``reasoning_content`` trace across
multi-turn calls by (a) extracting it into an assistant history turn via
``assistant_turn_from_response`` and (b) forwarding an optional ``history``
list through ``chat_completion`` unchanged when absent.

Mocks ``urllib.request.urlopen`` directly (mirrors the idiom in
tests/test_overview_live.py) so the outbound POST body can be inspected
without a real socket.
"""

from __future__ import annotations

import json

from lobes.minor._client import assistant_turn_from_response, chat_completion

# ---------------------------------------------------------------------------
# Fake urlopen plumbing
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, data: bytes, status: int = 200) -> None:
        self._data, self.status = data, status

    def read(self, n: int = -1) -> bytes:
        return self._data[:n] if n and n > 0 else self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen_factory(response_body: dict, captured: list):
    """Return a urlopen stand-in that records the decoded request body."""

    def _fake_urlopen(req, timeout=60):
        captured.append(json.loads(req.data.decode()))
        return _FakeResp(json.dumps(response_body).encode())

    return _fake_urlopen


_RESPONSE_WITH_REASONING = {
    "id": "chatcmpl-reasoning-001",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 4.",
                "reasoning": "2 + 2 = 4, a basic arithmetic fact.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17},
}

_RESPONSE_NO_REASONING = {
    "id": "chatcmpl-plain-001",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Just an answer."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
}


# ---------------------------------------------------------------------------
# assistant_turn_from_response
# ---------------------------------------------------------------------------


def test_assistant_turn_from_response_carries_reasoning() -> None:
    turn = assistant_turn_from_response(_RESPONSE_WITH_REASONING)
    assert turn["role"] == "assistant"
    assert turn["content"] == "The answer is 4."
    assert turn["reasoning"] == "2 + 2 = 4, a basic arithmetic fact."


def test_assistant_turn_from_response_no_reasoning_key_when_absent() -> None:
    turn = assistant_turn_from_response(_RESPONSE_NO_REASONING)
    assert turn == {"role": "assistant", "content": "Just an answer."}
    assert "reasoning" not in turn
    assert "reasoning_content" not in turn


def test_assistant_turn_from_response_handles_reasoning_content_key() -> None:
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Done.",
                    "reasoning_content": "Older vLLM build's trace field.",
                }
            }
        ]
    }
    turn = assistant_turn_from_response(resp)
    assert turn["reasoning_content"] == "Older vLLM build's trace field."
    assert "reasoning" not in turn


# ---------------------------------------------------------------------------
# chat_completion(history=...) round-trip
# ---------------------------------------------------------------------------


def test_two_turn_history_includes_prior_reasoning(monkeypatch) -> None:
    """A second-turn call must forward the first turn's assistant reasoning."""
    captured: list = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(_RESPONSE_NO_REASONING, captured),
    )

    first_response = _RESPONSE_WITH_REASONING
    assistant_turn = assistant_turn_from_response(first_response)

    history = [
        {"role": "user", "content": "What is 2 + 2?"},
        assistant_turn,
    ]
    chat_completion(
        "And what is 3 + 3?",
        base_url="http://x/v1",
        model="test-model",
        history=history,
    )

    assert len(captured) == 1
    outbound_messages = captured[0]["messages"]

    # The prior assistant turn's reasoning must be present in the outbound body.
    assistant_messages = [m for m in outbound_messages if m.get("role") == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["reasoning"] == "2 + 2 = 4, a basic arithmetic fact."
    assert assistant_messages[0]["content"] == "The answer is 4."

    # Full outbound shape: history entries then the new user turn appended last.
    assert outbound_messages[0] == {"role": "user", "content": "What is 2 + 2?"}
    assert outbound_messages[1] == assistant_turn
    assert outbound_messages[-1] == {"role": "user", "content": "And what is 3 + 3?"}


def test_single_turn_unchanged_when_no_history(monkeypatch) -> None:
    """No history= arg: message shape is identical to pre-#93 behavior (no reasoning key)."""
    captured: list = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(_RESPONSE_NO_REASONING, captured),
    )

    chat_completion("Hello", base_url="http://x/v1", model="test-model")

    assert len(captured) == 1
    messages = captured[0]["messages"]
    assert messages == [{"role": "user", "content": "Hello"}]
    for m in messages:
        assert "reasoning" not in m
        assert "reasoning_content" not in m


def test_enable_thinking_false_path_unchanged_without_history(monkeypatch) -> None:
    """extra_body enable_thinking=false still merges through when history is absent."""
    captured: list = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(_RESPONSE_NO_REASONING, captured),
    )

    chat_completion(
        "Hello",
        base_url="http://x/v1",
        model="test-model",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert len(captured) == 1
    body = captured[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    for m in body["messages"]:
        assert "reasoning" not in m
