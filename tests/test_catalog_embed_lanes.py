"""Catalog vector-space identity fields and the oes-t4 embedding entries."""

from __future__ import annotations

import dataclasses

import pytest

from lobes.catalog import (
    ENGINE_SENTENCE_TRANSFORMERS,
    ENGINE_VLLM,
    ENGINES,
    SUPPORTED_MODELS,
    SupportedModel,
    serves_with_vllm,
)

_NEW_IDS = (
    "google/embeddinggemma-2",
    "Qwen/Qwen3-VL-Embedding-8B",
    "Qwen/Qwen3-VL-Reranker-8B",
    "nvidia/Nemotron-3-Embed-8B-BF16",
    "nomic-ai/nomic-embed-code",
)


def _by_id(model_id: str) -> SupportedModel:
    return next(m for m in SUPPORTED_MODELS if m.id == model_id)


def test_new_fields_default_empty() -> None:
    fields = {f.name: f for f in dataclasses.fields(SupportedModel)}
    assert fields["modalities"].default == ()
    assert fields["mrl_dims"].default == ()
    assert fields["normalization"].default == ""


def test_sentence_transformers_engine_registered() -> None:
    assert ENGINE_SENTENCE_TRANSFORMERS == "sentence-transformers"
    assert ENGINE_SENTENCE_TRANSFORMERS in ENGINES
    assert ENGINES[-1] == ENGINE_SENTENCE_TRANSFORMERS


def test_existing_entries_untouched() -> None:
    old = [m for m in SUPPORTED_MODELS if m.id not in _NEW_IDS]
    assert old
    for m in old:
        assert m.modalities == () and m.mrl_dims == () and m.normalization == ""


@pytest.mark.parametrize("model_id", _NEW_IDS)
def test_new_entries_are_inert_candidates(model_id: str) -> None:
    m = _by_id(model_id)
    assert m.role_hint == "candidate"
    # "load-tested" only for what was actually booted on the Orin (2026-10-07);
    # the Qwen3-VL reranker never served, so it stays "configured".
    assert m.status == ("configured" if model_id == "Qwen/Qwen3-VL-Reranker-8B" else "load-tested")
    assert m.doc.endswith(".md")


def test_ids_unique() -> None:
    ids = [m.id for m in SUPPORTED_MODELS]
    assert len(ids) == len(set(ids))


def test_embeddinggemma() -> None:
    m = _by_id("google/embeddinggemma-2")
    assert m.task == "embed"
    assert m.engine == ENGINE_SENTENCE_TRANSFORMERS
    assert not serves_with_vllm(m)
    assert m.dimension == 768
    assert m.mrl_dims == (128, 256, 512, 768)
    assert m.modalities == ("text", "code", "image", "video", "audio")
    assert m.normalization == "l2"
    assert m.native_max_model_len == 8192


def test_qwen3_vl_pair() -> None:
    e = _by_id("Qwen/Qwen3-VL-Embedding-8B")
    r = _by_id("Qwen/Qwen3-VL-Reranker-8B")
    assert (e.task, e.engine) == ("embed", ENGINE_VLLM)
    assert (r.task, r.engine) == ("score", ENGINE_VLLM)
    assert e.modalities == r.modalities == ("text", "image", "video")


def test_nemotron_and_nomic() -> None:
    n = _by_id("nvidia/Nemotron-3-Embed-8B-BF16")
    assert (n.task, n.engine, n.dimension) == ("embed", ENGINE_VLLM, 4096)
    assert n.modalities == ("text",)
    c = _by_id("nomic-ai/nomic-embed-code")
    assert (c.task, c.engine) == ("embed", ENGINE_VLLM)
    assert c.modalities == ("text", "code")


def test_no_jina_code() -> None:
    assert not any("jina" in m.id.lower() for m in SUPPORTED_MODELS)
