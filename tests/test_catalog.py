"""The supported-model catalog is a single source of truth — guard it against
drift from the docs, the parser inference, and the gateway's default primary."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_gear.catalog import SUPPORTED_MODELS, as_dicts, supported_models
from model_gear.gateway import _config
from model_gear.runtime._parser import infer_parser

_DOCS = Path(__file__).resolve().parents[1] / "docs"

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


def test_moe_serve_extras_align_with_shape() -> None:
    # The MoE-only serve flags (--moe-backend / --speculative-config MTP) belong
    # to MoE checkpoints alone — they break the dense/hybrid models and must stay
    # off them. Tie the invariant to the architecture phrase so the two can't drift.
    for model in SUPPORTED_MODELS:
        is_moe = model.shape.lower().startswith("moe")
        assert bool(model.moe_backend) == is_moe, f"{model.id}: moe_backend vs shape"
        if not is_moe:
            assert model.speculative_config == "", f"{model.id}: stray speculative_config"
    # and the 35B MoE candidate actually carries shahizat's flags
    moe = next(m for m in SUPPORTED_MODELS if m.shape.lower().startswith("moe"))
    assert moe.moe_backend == "marlin"
    assert "mtp" in moe.speculative_config
