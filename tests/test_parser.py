"""Tests for per-model tool-call parser inference (``model_gear.runtime._parser``)."""

from __future__ import annotations

import pytest

from model_gear.runtime import _parser


@pytest.mark.parametrize(
    "model, expected",
    [
        # Qwen3-Coder / Qwen3.6 emit the XML function format → qwen3_coder
        ("mmangkad/Qwen3.6-27B-NVFP4", "qwen3_coder"),
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder"),
        ("some/qwen3_6-foo", "qwen3_coder"),
        # Qwen3 dense → hermes
        ("nvidia/Qwen3-32B-NVFP4", "hermes"),
        ("Qwen/Qwen3-8B", "hermes"),
        # Unknown family → None (leave the configured parser untouched)
        ("meta-llama/Llama-3-8B", None),
        ("mistralai/Mistral-7B", None),
        ("", None),
        # A bare "coder" must NOT trigger qwen3_coder on unrelated checkpoints —
        # the coder marker is Qwen3-scoped (regression guard for the false positive).
        ("deepseek-ai/deepseek-coder-6.7b-instruct", None),
        ("codellama/CodeLlama-13b", None),
        ("Qwen/Qwen2.5-Coder-7B-Instruct", None),
    ],
)
def test_infer_parser(model, expected) -> None:
    assert _parser.infer_parser(model) == expected


def test_infer_parser_is_case_insensitive() -> None:
    assert _parser.infer_parser("NVIDIA/QWEN3-32B-NVFP4") == "hermes"
    assert _parser.infer_parser("MMANGKAD/QWEN3-CODER") == "qwen3_coder"
