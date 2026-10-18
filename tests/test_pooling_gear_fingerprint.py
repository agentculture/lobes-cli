"""Pooling gears declare a real quantization so identical members can pool (#297).

The embedder and reranker lanes pass no ``--quantization`` flag, so the engine
loads the checkpoint as-is — exactly what the catalog's ``"none"`` sentinel
means. Without a declared value both lanes advertised ``quantization: unknown``,
and the strict pool rule never pools an unknown, so a mesh member that does not
host them (the DGX Spark) answered plain ``embedder`` / ``reranker`` with 404 even
though spark2, Thor and Orin all serve identical lanes.
"""

from __future__ import annotations

import pytest

from lobes.gateway._config import _lane_fingerprints
from lobes.gateway._replicas import Fingerprint, compare_fingerprints
from lobes.profiles.loader import resolve_profile
from lobes.profiles.render import profile_env

CARDS = ("spark", "thor", "orin", "base")


@pytest.mark.parametrize("card", CARDS)
def test_every_card_declares_no_quantization_for_the_pooling_gears(card: str) -> None:
    env = profile_env(resolve_profile(card))
    assert env.get("EMBED_QUANTIZATION") == "none"
    assert env.get("RERANK_QUANTIZATION") == "none"


def _fingerprint(env: dict[str, str], backend: str, served_id: str) -> Fingerprint:
    declared = _lane_fingerprints(env).get(backend, {})
    return Fingerprint(
        served_id=served_id,
        max_model_len=8192,
        runtime="vllm",
        quantization=declared.get("QUANTIZATION", "unknown"),
        kv_cache_dtype=declared.get("KV_CACHE_DTYPE", "unknown"),
        reasoning_parser=declared.get("REASONING_PARSER", "unknown"),
        tool_parser=declared.get("TOOL_CALL_PARSER", "unknown"),
        speculative_config=declared.get("SPECULATIVE_CONFIG", "unknown"),
    )


@pytest.mark.parametrize(
    ("backend", "served_id"),
    [("embed", "Qwen/Qwen3-Embedding-0.6B"), ("rerank", "Qwen/Qwen3-Reranker-0.6B")],
)
def test_identical_rendered_pooling_gears_pool(backend: str, served_id: str) -> None:
    a = _fingerprint(profile_env(resolve_profile("spark")), backend, served_id)
    b = _fingerprint(profile_env(resolve_profile("orin")), backend, served_id)
    assert compare_fingerprints(a, b) == (True, "")


def test_an_undeclared_quantization_still_never_pools() -> None:
    # The strict rule itself is unchanged: an unknown on either side still refuses.
    a = _fingerprint({}, "embed", "Qwen/Qwen3-Embedding-0.6B")
    b = _fingerprint({}, "embed", "Qwen/Qwen3-Embedding-0.6B")
    assert compare_fingerprints(a, b) == (False, "quantization: unknown")
