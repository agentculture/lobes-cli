"""Tests for per-model tool-call parser inference (``lobes.runtime._parser``)."""

from __future__ import annotations

import pytest

from lobes.runtime import _parser


@pytest.mark.parametrize(
    "model, expected",
    [
        # Qwen3-Coder / Qwen3.5 / Qwen3.6 emit the XML function format → qwen3_coder
        ("mmangkad/Qwen3.6-27B-NVFP4", "qwen3_coder"),
        ("sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP", "qwen3_coder"),
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder"),
        ("some/qwen3_6-foo", "qwen3_coder"),
        # Qwen3.5 also emits the XML function-call format → qwen3_coder
        ("Qwen/Qwen3.5-4B", "qwen3_coder"),
        ("cosmicproc/Qwen3.5-4B-NVFP4", "qwen3_coder"),
        ("some/qwen3-5-foo", "qwen3_coder"),
        ("some/qwen3_5-foo", "qwen3_coder"),
        # Qwen3 dense → hermes
        ("nvidia/Qwen3-32B-NVFP4", "hermes"),
        ("Qwen/Qwen3-8B", "hermes"),
        # Mistral → mistral (emits the [TOOL_CALLS] format)
        ("RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4", "mistral"),
        ("mistralai/Mistral-7B", "mistral"),
        # Unknown family → None (leave the configured parser untouched)
        ("meta-llama/Llama-3-8B", None),
        ("", None),
        # A bare "coder" must NOT trigger qwen3_coder on unrelated checkpoints —
        # the coder marker is Qwen3-scoped (regression guard for the false positive).
        ("deepseek-ai/deepseek-coder-6.7b-instruct", None),
        ("codellama/CodeLlama-13b", None),
        ("Qwen/Qwen2.5-Coder-7B-Instruct", None),
        # mistralai/* are mistral-family — the org prefix contains "mistral", so
        # they resolve to the mistral parser too (they share [TOOL_CALLS]).
        ("mistralai/Ministral-8B-Instruct-2410", "mistral"),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mistral"),
        # A bare mixtral/ministral basename (no mistralai prefix) stays unknown.
        ("Mixtral-8x7B", None),
        ("some/ministral-8b", None),
        # Gemma 4 → the purpose-built "gemma4" parser (Gemma4EngineToolParser).
        # NOT "pythonic": risk r2 is now CLOSED against the served checkpoint
        # (2026-07-17, live 31B on Thor) and the pythonic guess was disproven —
        # Gemma 4 emits `<|tool_call>call:name{...}<tool_call|>` with special-token
        # delimiters that pythonic (skip_special_tokens=True) never sees, so it
        # parsed nothing and the call leaked out as assistant content.
        ("sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4", "gemma4"),
        # NVFP4 base gear (§7 "support both" — the new default multimodal gear).
        ("coolthor/gemma-4-12B-it-NVFP4A16", "gemma4"),
        ("some/gemma-4-27b-it", "gemma4"),
        ("some/gemma4-9b", "gemma4"),
    ],
)
def test_infer_parser(model, expected) -> None:
    assert _parser.infer_parser(model) == expected


def test_infer_parser_is_case_insensitive() -> None:
    assert _parser.infer_parser("NVIDIA/QWEN3-32B-NVFP4") == "hermes"
    assert _parser.infer_parser("MMANGKAD/QWEN3-CODER") == "qwen3_coder"
