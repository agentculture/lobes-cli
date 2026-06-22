"""Tests for per-model tool-call parser inference (``lobes.runtime._parser``)."""

from __future__ import annotations

import pytest

from lobes.runtime import _parser


@pytest.mark.parametrize(
    "model, expected",
    [
        # Qwen3-Coder / Qwen3.6 emit the XML function format → qwen3_coder
        ("mmangkad/Qwen3.6-27B-NVFP4", "qwen3_coder"),
        ("sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP", "qwen3_coder"),
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder"),
        ("some/qwen3_6-foo", "qwen3_coder"),
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
    ],
)
def test_infer_parser(model, expected) -> None:
    assert _parser.infer_parser(model) == expected


def test_infer_parser_is_case_insensitive() -> None:
    assert _parser.infer_parser("NVIDIA/QWEN3-32B-NVFP4") == "hermes"
    assert _parser.infer_parser("MMANGKAD/QWEN3-CODER") == "qwen3_coder"
