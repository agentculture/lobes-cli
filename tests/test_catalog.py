"""The supported-model catalog is a single source of truth — guard it against
drift from the docs, the parser inference, and the gateway's default primary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_gear.catalog import (
    SUPPORTED_MODELS,
    as_dicts,
    mtp_compose_command_items,
    supported_models,
)
from model_gear.gateway import _config
from model_gear.runtime._parser import infer_parser

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_TEMPLATES = Path(__file__).resolve().parents[1] / "model_gear" / "templates"

_FIELDS = ("id", "role_hint", "shape", "context", "tool_parser", "quantization", "status", "doc")


def test_catalog_is_nonempty_and_accessors_agree() -> None:
    assert len(SUPPORTED_MODELS) >= 3
    assert supported_models() is SUPPORTED_MODELS
    dicts = as_dicts()
    assert len(dicts) == len(SUPPORTED_MODELS)
    assert {d["id"] for d in dicts} == {m.id for m in SUPPORTED_MODELS}


def test_every_entry_has_all_fields_nonempty() -> None:
    for entry in as_dicts():
        for field in _FIELDS:
            assert entry.get(field), f"{entry.get('id')}: empty/missing {field}"


def test_catalog_ids_are_unique() -> None:
    ids = [m.id for m in SUPPORTED_MODELS]
    assert len(ids) == len(set(ids))


def test_status_values_are_known() -> None:
    assert {m.status for m in SUPPORTED_MODELS} <= {"load-tested", "configured"}


@pytest.mark.skipif(not _DOCS.is_dir(), reason="docs/ not shipped (wheel install)")
def test_every_doc_file_exists() -> None:
    # The machine catalog and the human prose must not silently diverge.
    for model in SUPPORTED_MODELS:
        assert (_DOCS / model.doc).is_file(), f"{model.id}: missing docs/{model.doc}"


def test_tool_parser_matches_infer_parser() -> None:
    # The catalog must agree with the runtime's parser inference (the source of
    # truth model switch uses), or a fleet backend would be misconfigured.
    for model in SUPPORTED_MODELS:
        assert infer_parser(model.id) == model.tool_parser, model.id


def test_gateway_default_primary_and_fallback_are_in_catalog() -> None:
    ids = {m.id for m in SUPPORTED_MODELS}
    assert _config._DEFAULT_PRIMARY in ids
    assert _config._DEFAULT_FALLBACK in ids


def test_moe_backend_aligns_with_shape() -> None:
    # --moe-backend belongs to MoE checkpoints alone — it breaks the dense/hybrid
    # models. Tie the invariant to the architecture phrase so the two can't drift.
    for model in SUPPORTED_MODELS:
        is_moe = model.shape.lower().startswith("moe")
        assert bool(model.moe_backend) == is_moe, f"{model.id}: moe_backend vs shape"
    moe = next(m for m in SUPPORTED_MODELS if m.shape.lower().startswith("moe"))
    assert moe.moe_backend == "marlin"
    # the 35B MoE candidate does NOT carry the MTP speculative-config — it fails on
    # the mmangkad checkpoint (verified live 2026-05-31); see its doc.
    assert moe.speculative_config == ""


def test_speculative_config_only_on_mtp_checkpoints() -> None:
    # --speculative-config (MTP draft) is carried only by a checkpoint that ships
    # MTP draft weights — flagged by the -MTP suffix in its id. A baseline NVFP4
    # export drops the draft head (0% acceptance), so it must stay empty elsewhere.
    for model in SUPPORTED_MODELS:
        if model.speculative_config:
            assert "MTP" in model.id.upper(), f"{model.id}: speculative_config on a non-MTP id"
            method = json.loads(model.speculative_config).get("method")
            assert method, f"{model.id}: speculative_config missing 'method'"
    # the MTP-grafted 27B primary (issue #26) carries the qwen3_5_mtp draft config.
    sak = next(m for m in SUPPORTED_MODELS if m.id == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP")
    cfg = json.loads(sak.speculative_config)
    assert cfg["method"] == "qwen3_5_mtp"
    assert cfg["num_speculative_tokens"] == 3


def test_mtp_command_items_match_packaged_templates() -> None:
    # The MTP primary's extra command items are baked into the compose templates AND
    # named by `model switch` as the lines to remove for a non-MTP model. The catalog
    # helper is the single source of truth — guard it against drift from the packaged
    # templates (both single-model and the fleet vllm-primary service must ship them).
    items = mtp_compose_command_items()
    assert items[0].startswith("--speculative-config=")
    for template in ("docker-compose.yml", "fleet/docker-compose.yml"):
        text = (_TEMPLATES / template).read_text(encoding="utf-8")
        for item in items:
            assert item in text, f"{item!r} missing from templates/{template} (drift)"
