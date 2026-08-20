"""Tests for the preserve_thinking server flag on the vLLM primary/cortex
service (issue #93 — preserve Qwen thinking traces across multi-turn calls).

Asserts that:
  - lobes/templates/docker-compose.yml (legacy single-model scaffold) carries
    `--default-chat-template-kwargs '{"preserve_thinking": true}'` on the
    primary service's command:, adjacent to --reasoning-parser=qwen3.
  - lobes/templates/fleet/docker-compose.yml carries the same flag on
    vllm-primary (the cortex role).
  - The fleet's embed/rerank pooling services (vllm-embed, vllm-rerank) do
    NOT gain --default-chat-template-kwargs — they run --runner=pooling and
    have no reasoning/chat-template surface at all.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_SINGLE_COMPOSE = _TEMPLATES / "docker-compose.yml"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"

_FLAG = "--default-chat-template-kwargs"
_KWARGS_VALUE = '{"preserve_thinking": true}'


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _command(compose: dict, service: str) -> list[str]:
    return compose["services"][service]["command"]


def _command_tokens(compose: dict, service: str) -> list[str]:
    """Like ``_command``, but always a token LIST.

    ``vllm-primary``'s ``command:`` is now a shell-lexed STRING (spec-knobs
    task, the same off-switch mechanism as the senses lane — see
    ``tests/test_senses_speculative_config.py``), not a YAML list. Tokenizing
    the raw (unsubstituted) text with ``shlex`` is safe here: none of ``$``,
    ``{``, ``}`` are shell metacharacters, so every bare ``${VAR:-default}``
    stays one token, and the only quoted segment (the preserve_thinking JSON)
    round-trips because its wrapping single quotes are real quote characters
    in the source.
    """
    command = compose["services"][service]["command"]
    if isinstance(command, str):
        return shlex.split(command)
    return list(command)


class TestSingleScaffoldPrimaryHasPreserveThinking:
    def test_default_chat_template_kwargs_flag_present(self) -> None:
        command = _command(_load(_SINGLE_COMPOSE), "vllm")
        assert _FLAG in command, f"{_FLAG} not found on vllm service command"

    def test_preserve_thinking_value_present(self) -> None:
        command = _command(_load(_SINGLE_COMPOSE), "vllm")
        assert _KWARGS_VALUE in command, f"{_KWARGS_VALUE!r} not found on vllm service command"

    def test_flag_is_adjacent_to_reasoning_parser(self) -> None:
        command = _command(_load(_SINGLE_COMPOSE), "vllm")
        reasoning_idx = next(
            i for i, item in enumerate(command) if item.startswith("--reasoning-parser")
        )
        assert command[reasoning_idx + 1 : reasoning_idx + 3] == [
            _FLAG,
            _KWARGS_VALUE,
        ], "preserve_thinking flag must sit directly after --reasoning-parser=qwen3"


class TestFleetPrimaryHasPreserveThinking:
    def test_default_chat_template_kwargs_flag_present(self) -> None:
        command = _command(_load(_FLEET_COMPOSE), "vllm-primary")
        assert _FLAG in command, f"{_FLAG} not found on vllm-primary command"

    def test_preserve_thinking_value_present(self) -> None:
        command = _command(_load(_FLEET_COMPOSE), "vllm-primary")
        assert _KWARGS_VALUE in command, f"{_KWARGS_VALUE!r} not found on vllm-primary command"

    def test_flag_is_adjacent_to_reasoning_parser(self) -> None:
        command = _command_tokens(_load(_FLEET_COMPOSE), "vllm-primary")
        reasoning_idx = next(
            i for i, item in enumerate(command) if item.startswith("--reasoning-parser")
        )
        assert command[reasoning_idx + 1 : reasoning_idx + 3] == [
            _FLAG,
            _KWARGS_VALUE,
        ], "preserve_thinking flag must sit directly after --reasoning-parser=qwen3"


class TestFleetPoolingServicesUnaffected:
    """vllm-embed and vllm-rerank are pooling runners (--runner=pooling) with
    no reasoning-parser surface at all; they must not gain the new flag."""

    def test_vllm_embed_has_no_chat_template_kwargs(self) -> None:
        command = _command(_load(_FLEET_COMPOSE), "vllm-embed")
        assert _FLAG not in command, "vllm-embed must NOT gain --default-chat-template-kwargs"

    def test_vllm_rerank_has_no_chat_template_kwargs(self) -> None:
        command = _command(_load(_FLEET_COMPOSE), "vllm-rerank")
        assert _FLAG not in command, "vllm-rerank must NOT gain --default-chat-template-kwargs"
