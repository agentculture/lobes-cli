"""The supported-model catalog is a single source of truth — guard it against
drift from the docs, the parser inference, and the gateway's default primary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lobes.catalog import (
    SUPPORTED_MODELS,
    TIER_ROLE,
    SupportedModel,
    as_dicts,
    mtp_compose_command_items,
    resolve_tier,
    supported_models,
)
from lobes.gateway import _config
from lobes.runtime._parser import infer_parser

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"

# Fields required non-empty for ALL models (task-agnostic).
_FIELDS_ALL = ("id", "role_hint", "shape", "context", "status", "doc")
# Fields required non-empty ONLY for generate (chat/completion) models.
# Embedding and reranker gears have no tool parser and no quantization flag.
_FIELDS_GENERATE = ("tool_parser", "quantization")


def test_catalog_is_nonempty_and_accessors_agree() -> None:
    assert len(SUPPORTED_MODELS) >= 3
    assert supported_models() is SUPPORTED_MODELS
    dicts = as_dicts()
    assert len(dicts) == len(SUPPORTED_MODELS)
    assert {d["id"] for d in dicts} == {m.id for m in SUPPORTED_MODELS}


def test_every_entry_has_all_fields_nonempty() -> None:
    for entry in as_dicts():
        for field in _FIELDS_ALL:
            assert entry.get(field), f"{entry.get('id')}: empty/missing {field}"
        if entry.get("task", "generate") == "generate":
            for field in _FIELDS_GENERATE:
                assert entry.get(field), f"{entry.get('id')}: empty/missing {field}"


def test_catalog_ids_are_unique() -> None:
    ids = [m.id for m in SUPPORTED_MODELS]
    assert len(ids) == len(set(ids))


def test_status_values_are_known() -> None:
    assert {m.status for m in SUPPORTED_MODELS} <= {"load-tested", "configured"}


def test_native_max_model_len_is_a_positive_int() -> None:
    # The clamp `lobes switch` applies relies on a real, positive ceiling per model;
    # a missing/zero value would silently disable the boot-safety clamp.
    for model in SUPPORTED_MODELS:
        assert isinstance(model.native_max_model_len, int), model.id
        assert model.native_max_model_len > 0, model.id


@pytest.mark.skipif(not _DOCS.is_dir(), reason="docs/ not shipped (wheel install)")
def test_every_doc_file_exists() -> None:
    # The machine catalog and the human prose must not silently diverge.
    for model in SUPPORTED_MODELS:
        assert (_DOCS / model.doc).is_file(), f"{model.id}: missing docs/{model.doc}"


def test_tool_parser_matches_infer_parser() -> None:
    # The catalog must agree with the runtime's parser inference (the source of
    # truth lobes switch uses), or a fleet backend would be misconfigured.
    # Restrict to generate (chat/completion) models: embed/score gears have no
    # tool parser (tool_parser="") but infer_parser would return "hermes" for
    # any Qwen3 id — those models don't do tool calling, so the field is empty.
    for model in SUPPORTED_MODELS:
        if model.task == "generate":
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
    # named by `lobes switch` as the lines to remove for a non-MTP model. The catalog
    # helper is the single source of truth — guard it against drift from the packaged
    # templates (both single-model and the fleet vllm-primary service must ship them).
    items = mtp_compose_command_items()
    assert items[0].startswith("--speculative-config=")
    for template in ("docker-compose.yml", "fleet/docker-compose.yml"):
        text = (_TEMPLATES / template).read_text(encoding="utf-8")
        for item in items:
            assert item in text, f"{item!r} missing from templates/{template} (drift)"


# ---------------------------------------------------------------------------
# New-field contract: task / dimension / hf_overrides (issue #44)
# ---------------------------------------------------------------------------

_VALID_TASKS = {"generate", "embed", "score"}
_EMBEDDER_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANKER_ID = "Qwen/Qwen3-Reranker-0.6B"


def test_task_values_are_valid() -> None:
    # Every catalog entry must declare a known task; unknown strings would silently
    # mis-route the model to the wrong vLLM endpoint.
    for model in SUPPORTED_MODELS:
        assert model.task in _VALID_TASKS, f"{model.id}: unknown task {model.task!r}"


def test_exactly_one_embed_and_one_score_model() -> None:
    # The catalog must contain exactly one embedder and exactly one reranker so that
    # fleet routing is unambiguous — two embed/score entries would require a tiebreaker
    # that doesn't exist yet.
    embed_ids = [m.id for m in SUPPORTED_MODELS if m.task == "embed"]
    score_ids = [m.id for m in SUPPORTED_MODELS if m.task == "score"]
    assert embed_ids == [_EMBEDDER_ID], f"embed models: {embed_ids}"
    assert score_ids == [_RERANKER_ID], f"score models: {score_ids}"


def test_embed_models_have_positive_dimension() -> None:
    # A zero dimension would make the Matryoshka truncation range meaningless and
    # would break any client that reads the dimension to allocate buffers.
    for model in SUPPORTED_MODELS:
        if model.task == "embed":
            assert model.dimension > 0, f"{model.id}: embed model must have dimension > 0"


def test_embedder_dimension_is_1024() -> None:
    embedder = next(m for m in SUPPORTED_MODELS if m.id == _EMBEDDER_ID)
    assert embedder.dimension == 1024, f"{_EMBEDDER_ID}: expected dimension 1024"


@pytest.mark.skipif(not _DOCS.is_dir(), reason="docs/ not shipped (wheel install)")
def test_embedder_doc_file_exists() -> None:
    embedder = next(m for m in SUPPORTED_MODELS if m.id == _EMBEDDER_ID)
    assert (_DOCS / embedder.doc).is_file(), f"{_EMBEDDER_ID}: missing docs/{embedder.doc}"


def test_embed_models_have_valid_hf_overrides() -> None:
    # The Matryoshka override must be present and must be valid JSON — vLLM parses it
    # at serve time, and a malformed override silently disables truncation.
    for model in SUPPORTED_MODELS:
        if model.task == "embed":
            assert model.hf_overrides, f"{model.id}: embed model must have non-empty hf_overrides"
            parsed = json.loads(model.hf_overrides)
            assert isinstance(parsed, dict), f"{model.id}: hf_overrides is not a JSON object"


def test_score_models_have_valid_hf_overrides_with_architecture() -> None:
    # The reranker's hf_overrides must name Qwen3ForSequenceClassification in its
    # "architectures" list — vLLM uses this to pick the correct model class. A missing
    # or mis-spelled entry causes a load-time failure that can't be caught until serve.
    for model in SUPPORTED_MODELS:
        if model.task == "score":
            assert model.hf_overrides, f"{model.id}: score model must have non-empty hf_overrides"
            parsed = json.loads(model.hf_overrides)
            assert isinstance(parsed, dict), f"{model.id}: hf_overrides is not a JSON object"
            archs = parsed.get("architectures", [])
            assert (
                "Qwen3ForSequenceClassification" in archs
            ), f"{model.id}: hf_overrides missing 'Qwen3ForSequenceClassification' in architectures"


def test_generate_models_have_zero_dimension_and_empty_hf_overrides() -> None:
    # Chat/completion models have no embedding dimension and need no hf_overrides —
    # a non-zero dimension or a stray override would confuse the fleet routing logic.
    for model in SUPPORTED_MODELS:
        if model.task == "generate":
            assert model.dimension == 0, f"{model.id}: generate model must have dimension == 0"
            assert (
                model.hf_overrides == ""
            ), f"{model.id}: generate model must have empty hf_overrides"


def test_as_dicts_includes_new_fields() -> None:
    # as_dicts() is the JSON-serialisation path used by the gateway and the CLI's
    # --json output. All three new fields must appear in every dict so downstream
    # consumers can rely on them without hasattr/KeyError guards.
    required_new_fields = {"task", "dimension", "hf_overrides"}
    for entry in as_dicts():
        missing = required_new_fields - entry.keys()
        assert not missing, f"{entry.get('id')}: as_dicts() missing fields {missing}"


def test_embed_score_hf_overrides_match_fleet_template() -> None:
    # The fleet compose hardcodes --hf-overrides for the vllm-embed / vllm-rerank
    # services; the catalog stores the same JSON in each gear's hf_overrides field.
    # Guard against the two drifting (mirrors test_mtp_command_items_match_packaged_
    # templates): every embed/score gear's hf_overrides must appear *verbatim* in the
    # fleet template, so a catalog edit that forgets the compose (or vice versa) fails
    # the build instead of silently serving with stale overrides.
    fleet = (_TEMPLATES / "fleet" / "docker-compose.yml").read_text(encoding="utf-8")
    pooling = [m for m in SUPPORTED_MODELS if m.task in ("embed", "score")]
    assert pooling, "expected at least one embed/score gear in the catalog"
    for model in pooling:
        assert model.hf_overrides, f"{model.id}: embed/score gear has empty hf_overrides"
        assert model.hf_overrides in fleet, (
            f"{model.id}: hf_overrides not found verbatim in "
            "templates/fleet/docker-compose.yml (catalog<->compose drift)"
        )


# ---------------------------------------------------------------------------
# Qwen3.5-4B "minor" gear + bf16 "none" quantization sentinel (issue #64)
# ---------------------------------------------------------------------------

_MINOR_ID = "Qwen/Qwen3.5-4B"


def test_minor_gear_exists_with_correct_fields() -> None:
    # The minor gear must be present in the catalog with exactly the fields the
    # acceptance criteria specify — any field drift is a misconfiguration bug.
    minor = next((m for m in SUPPORTED_MODELS if m.id == _MINOR_ID), None)
    assert minor is not None, f"{_MINOR_ID} not found in catalog"
    assert minor.role_hint == "minor"
    assert minor.shape == "hybrid linear-attn + ViT (multimodal)"
    assert minor.context == "256K native"
    assert minor.native_max_model_len == 262144
    assert minor.tool_parser == "qwen3_coder"
    assert minor.quantization == "none"
    assert minor.status == "configured"
    assert minor.doc == "qwen3.5-4b-minor.md"
    assert minor.task == "generate"
    assert minor.dimension == 0
    assert minor.hf_overrides == ""
    assert minor.moe_backend == ""
    assert minor.speculative_config == ""


def test_minor_gear_shape_is_not_moe_so_no_moe_backend() -> None:
    # The 4B minor gear is a hybrid model (not MoE) — moe_backend must be empty
    # (the moe_backend_aligns_with_shape invariant also enforces this).
    minor = next(m for m in SUPPORTED_MODELS if m.id == _MINOR_ID)
    assert not minor.shape.lower().startswith(
        "moe"
    ), f"{_MINOR_ID}: shape must not be MoE — got {minor.shape!r}"
    assert minor.moe_backend == "", f"{_MINOR_ID}: moe_backend must be empty for a non-MoE model"


def test_minor_gear_quantization_is_none_sentinel() -> None:
    # quantization="none" is the bf16/unquantized sentinel — the value must be the
    # literal string "none" (not an empty string, not None) so switch can distinguish
    # "unquantized" from "uncatalogued" (which uses empty/absent).
    minor = next(m for m in SUPPORTED_MODELS if m.id == _MINOR_ID)
    assert (
        minor.quantization == "none"
    ), f"{_MINOR_ID}: expected quantization='none', got {minor.quantization!r}"


# ---------------------------------------------------------------------------
# Gemma 4 12B "multimodal" gear + main/minor/multimodal tier reframe (t2)
# ---------------------------------------------------------------------------
# The "normal" tier is reframed from the 14B "middle" gear to the Gemma 4 12B
# unified-multimodal gear; the 14B is demoted (KEPT) to role_hint="candidate".

_GEMMA_ID = "sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4"
_14B_ID = "nvidia/Qwen3-14B-NVFP4"
_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"


def test_gemma_multimodal_gear_exists_with_correct_fields() -> None:
    # The Gemma multimodal gear must be present with exactly the fields the
    # acceptance criteria specify — any field drift is a misconfiguration bug.
    gemma = next((m for m in SUPPORTED_MODELS if m.id == _GEMMA_ID), None)
    assert gemma is not None, f"{_GEMMA_ID} not found in catalog"
    assert gemma.role_hint == "multimodal"
    assert gemma.task == "generate"
    assert gemma.tool_parser == "pythonic"
    # NVFP4 in compressed-tensors format (config.json quant_method), NOT modelopt —
    # verified live on the Spark (#71); modelopt_fp4 fails with a method mismatch.
    assert gemma.quantization == "compressed-tensors"
    assert gemma.status == "load-tested"  # GB10 2026-07-01: text+image+audio ✓ on nightly (#71/#73)
    assert gemma.doc == "gemma-4-12b-nvfp4.md"
    assert gemma.native_max_model_len == 131072
    assert gemma.dimension == 0
    assert gemma.hf_overrides == ""
    assert gemma.moe_backend == ""  # not MoE
    # The shape phrase must advertise the unified multimodal (text+image+audio) nature.
    shape = gemma.shape.lower()
    assert "multimodal" in shape
    for modality in ("text", "image", "audio"):
        assert modality in shape, f"{_GEMMA_ID}: shape must mention {modality!r}"


def test_gemma_tool_parser_matches_infer_parser() -> None:
    # The catalog's pythonic parser must agree with the runtime's inference (t1).
    gemma = next(m for m in SUPPORTED_MODELS if m.id == _GEMMA_ID)
    assert infer_parser(gemma.id) == "pythonic"
    assert gemma.tool_parser == infer_parser(gemma.id)


def test_gemma_has_no_speculative_config() -> None:
    # Despite the "-MTP" name, this unified checkpoint exposes no gemma4_assistant
    # draft, and vLLM 0.21/0.22 enable Gemma4 MTP only via a SEPARATE gemma4_assistant
    # draft model — not self-speculation from the unified target. Passing
    # {"method": "gemma4_mtp"} is rejected ("Unsupported speculative method"), verified
    # live on the Spark (#71). So the gear carries NO speculative_config until a draft
    # model is sourced (tracked follow-up); it serves without spec-decode.
    gemma = next(m for m in SUPPORTED_MODELS if m.id == _GEMMA_ID)
    assert gemma.speculative_config == "", (
        f"{_GEMMA_ID}: speculative_config must be empty — Gemma4 native MTP needs a "
        "separate gemma4_assistant draft model (see docs/gemma-4-12b-nvfp4.md)"
    )


def test_14b_is_demoted_to_candidate() -> None:
    # The 14B is KEPT but demoted from the "middle" tier to a legacy candidate;
    # it must remain in the catalog and now carry role_hint="candidate".
    middle = next((m for m in SUPPORTED_MODELS if m.id == _14B_ID), None)
    assert middle is not None, f"{_14B_ID} must be KEPT in the catalog (demoted, not deleted)"
    assert middle.role_hint == "candidate", f"{_14B_ID}: expected demotion to 'candidate'"


def test_tier_role_map_uses_new_vocabulary() -> None:
    # Primary vocabulary: main/minor/multimodal. Back-compat aliases retained.
    assert TIER_ROLE["main"] == "primary"
    assert TIER_ROLE["minor"] == "minor"
    assert TIER_ROLE["multimodal"] == "multimodal"
    assert TIER_ROLE["cheap"] == "minor"
    assert TIER_ROLE["normal"] == "multimodal"
    assert TIER_ROLE["hard"] == "primary"


def test_resolve_tier_multimodal_and_normal_return_gemma() -> None:
    # Both the new "multimodal" alias and the back-compat "normal" alias resolve
    # to the Gemma 4 multimodal gear (the reframed normal tier).
    for tier in ("multimodal", "normal"):
        model = resolve_tier(tier)
        assert isinstance(model, SupportedModel)
        assert model.id == _GEMMA_ID, f"resolve_tier({tier!r}) -> {model.id} (expected Gemma)"
        assert model.role_hint == "multimodal"


def test_resolve_tier_main_and_hard_return_primary() -> None:
    for tier in ("main", "hard"):
        model = resolve_tier(tier)
        assert model.role_hint == "primary"
        assert model.id == _PRIMARY_ID
        assert model.task == "generate"


def test_resolve_tier_minor_and_cheap_return_4b_minor() -> None:
    for tier in ("minor", "cheap"):
        model = resolve_tier(tier)
        assert model.role_hint == "minor"
        assert model.id == _MINOR_ID


def test_resolve_tier_unknown_still_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        resolve_tier("ultra")
