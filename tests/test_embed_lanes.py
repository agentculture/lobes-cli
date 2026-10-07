"""Specialist embed/rerank lane registry (orin-embedding-specialist, t3)."""

from __future__ import annotations

import dataclasses

import pytest

from lobes import embed_lanes
from lobes.catalog import TIER_ROLE
from lobes.embed_lanes import EMBED_LANES, EmbedLane, validate_lane_name
from lobes.roles import ROLES

_MODALITIES = {"text", "code", "image", "video", "audio"}

_EXPECTED_ROLES = (
    "cortex",
    "senses",
    "muse",
    "worker",
    "associate",
    "hand",
    "embedder",
    "reranker",
    "stt",
    "tts",
    "innereye",
)


def test_roles_unchanged():
    assert ROLES == _EXPECTED_ROLES


def test_lane_names_and_ids():
    got = {lane.name: lane.catalog_id for lane in EMBED_LANES}
    assert got == {
        "gemma2-embed": "google/embeddinggemma-2",
        "qwen3vl-embed": "Qwen/Qwen3-VL-Embedding-8B",
        "qwen3vl-rerank": "Qwen/Qwen3-VL-Reranker-8B",
        "nemotron-embed": "nvidia/Nemotron-3-Embed-8B-BF16",
        "nomic-code-embed": "nomic-ai/nomic-embed-code",
    }


def test_lanes_are_frozen_and_well_formed():
    assert isinstance(EMBED_LANES, tuple)
    lane_copy = dataclasses.replace(EMBED_LANES[0])  # never mutate the shared registry
    with pytest.raises(dataclasses.FrozenInstanceError):
        lane_copy.name = "x"  # type: ignore[misc]
    for lane in EMBED_LANES:
        assert isinstance(lane, EmbedLane)
        assert lane.task in {"embed", "score"}
        assert lane.engine in {"vllm", "sentence-transformers"}
        assert lane.normalization in {"l2", "none", ""}  # "" = not declared by the card
        assert lane.modalities
        assert set(lane.modalities) <= _MODALITIES
        assert lane.base_url_env == lane.name.upper().replace("-", "_") + "_BASE_URL"
        validate_lane_name(lane.name)


def test_gemma2_declaration():
    lane = next(x for x in EMBED_LANES if x.name == "gemma2-embed")
    assert lane.engine == "sentence-transformers"
    assert lane.dim == 768
    assert lane.mrl_dims == (128, 256, 512, 768)
    assert lane.normalization == "l2"
    assert lane.modalities == ("text", "code", "image", "video", "audio")
    assert lane.base_url_env == "GEMMA2_EMBED_BASE_URL"


def test_names_unique():
    names = [lane.name for lane in EMBED_LANES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("role", ROLES)
def test_refuses_role_name(role):
    with pytest.raises(ValueError):
        validate_lane_name(role)


@pytest.mark.parametrize("alias", sorted(TIER_ROLE))
def test_refuses_tier_alias(alias):
    with pytest.raises(ValueError):
        validate_lane_name(alias)


@pytest.mark.parametrize("alias", ["embed", "rerank", "primary", "multimodal-coder", "embed-deep"])
def test_refuses_backend_and_self_named_aliases(alias):
    with pytest.raises(ValueError):
        validate_lane_name(alias)


@pytest.mark.parametrize(
    "name", ["embedder-spark", "reranker-thor", "cortex-spark2", "worker-x", "embed-foo"]
)
def test_refuses_member_lane_pattern(name):
    with pytest.raises(ValueError):
        validate_lane_name(name)


def test_refuses_empty_and_bad_charset():
    for bad in ("", "Gemma2", "has space"):
        with pytest.raises(ValueError):
            validate_lane_name(bad)


def test_no_registered_lane_collides():
    assert embed_lanes.EMBED_LANES  # sanity
    for lane in EMBED_LANES:
        assert not lane.name.startswith(("embedder-", "reranker-"))


def test_every_lane_agrees_with_its_catalog_entry():
    """The catalog owns vector-space identity; the registry must never drift from it."""
    from lobes.catalog import SUPPORTED_MODELS

    by_id = {m.id: m for m in SUPPORTED_MODELS}
    for lane in EMBED_LANES:
        entry = by_id[lane.catalog_id]
        assert lane.engine == entry.engine, lane.name
        assert lane.task == entry.task, lane.name
        assert lane.dim == entry.dimension, lane.name
        assert lane.modalities == entry.modalities, lane.name
        assert lane.mrl_dims == entry.mrl_dims, lane.name
        assert lane.normalization == entry.normalization, lane.name
