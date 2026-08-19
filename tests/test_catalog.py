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
    speculative_config_item,
    supported_models,
)
from lobes.gateway import _config
from lobes.runtime._parser import infer_parser

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_WORKER_ID = "unsloth/Qwen3.6-35B-A3B-NVFP4"

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
        # A forced --moe-backend belongs to MoE checkpoints alone (it breaks the
        # dense/hybrid models) — but a MoE gear MAY leave it empty to let vLLM
        # auto-select. worker does: every forced NVFP4 MoE backend was refused on
        # Thor sm_110 (docs/evidence/2026-07-31-accept-worker-thor.txt). So the
        # invariant is one-directional: a moe_backend implies MoE, not vice-versa.
        if model.moe_backend:
            assert is_moe, f"{model.id}: moe_backend on a non-MoE shape"
    moe = next(m for m in SUPPORTED_MODELS if m.shape.lower().startswith("moe"))
    assert moe.moe_backend == "marlin"  # the mmangkad candidate (first MoE gear)
    # the 35B MoE candidate does NOT carry the MTP speculative-config — it fails on
    # the mmangkad checkpoint (verified live 2026-05-31); see its doc.
    assert moe.speculative_config == ""


# Checkpoints that carry a *self-hosted* MTP draft (own baked-in draft weights,
# vLLM's generic "mtp" method, no external "model"/"draft_model_id" key) but
# whose id does NOT happen to contain an "-MTP" suffix. The 27B primary's id
# carries "-MTP" because IT is a re-export that GRAFTED a draft head onto a
# baseline that otherwise lacked one; this unsloth export never dropped its
# draft in the first place, so its id has no reason to say so. Verified
# against the checkpoint's actual config.json (fetched 2026-07-31), not card
# prose: text_config.mtp_num_hidden_layers=1, and quantization_config.ignore
# carries a "re:^mtp.*" pattern — i.e. real MTP weight tensors exist in the
# checkpoint and are deliberately left unquantized. See the worker catalog
# entry's own comment for the full citation.
# unsloth/Qwen3.6-27B-NVFP4 (now a demoted CANDIDATE) is the same publisher's
# same-recipe 27B sibling and qualifies for the same reason, verified the same
# way against its own published config (fetched 2026-07-31): mtp_num_hidden_layers=1,
# a top-level "unsloth_fixed_mtp" flag, and 15 real `mtp.*` tensors in
# model.safetensors.index.json. Like the worker gear it never lost its draft
# head, so its upstream id never carried an "-MTP" marker.
# unsloth/Qwen3.8-27B-NVFP4 (the current PRIMARY, since 2026-08-19) qualifies
# for the identical reason, verified against its own published config (fetched
# 2026-08-19): text_config.mtp_num_hidden_layers=1 — a self-hosted MTP draft
# module, no external draft repo, and no "-MTP" marker in its upstream id.
_SELF_HOSTED_MTP_WITHOUT_ID_MARKER = frozenset(
    {_WORKER_ID, "unsloth/Qwen3.6-27B-NVFP4", "unsloth/Qwen3.8-27B-NVFP4"}
)


def test_speculative_config_only_on_mtp_or_external_draft_checkpoints() -> None:
    # --speculative-config (MTP draft) is carried by a checkpoint for one of THREE
    # reasons: (1) it ships its OWN MTP draft weights, flagged by an "-MTP" suffix
    # in its id (e.g. the 27B primary's MTP-grafted re-export — a baseline NVFP4
    # export drops the draft head, 0% acceptance); (2) it wires a SEPARATE,
    # external draft model via the "model"/"draft_model_id" key (Gemma4 native
    # MTP — the draft is a public HF checkpoint, not baked into the served id;
    # see docs/vllm-nightly-migration.md §7 — coolthor/gemma-4-12B-it-NVFP4A16
    # carries no "-MTP" in its id but is wired to the external
    # google/gemma-4-12B-it-assistant draft); or (3) it is explicitly allow-listed
    # in _SELF_HOSTED_MTP_WITHOUT_ID_MARKER — a checkpoint that ships its own
    # draft weights (like case 1) but whose upstream id never carried an "-MTP"
    # marker because it never lost the draft to begin with (the worker gear).
    for model in SUPPORTED_MODELS:
        if not model.speculative_config:
            continue
        cfg = json.loads(model.speculative_config)
        method = cfg.get("method")
        assert method, f"{model.id}: speculative_config missing 'method'"
        has_mtp_in_id = "MTP" in model.id.upper()
        has_external_draft = bool(cfg.get("model") or cfg.get("draft_model_id"))
        is_allow_listed_self_hosted = model.id in _SELF_HOSTED_MTP_WITHOUT_ID_MARKER
        assert has_mtp_in_id or has_external_draft or is_allow_listed_self_hosted, (
            f"{model.id}: speculative_config carried but neither an -MTP id, an "
            "external draft 'model'/'draft_model_id' key, nor an explicit "
            "_SELF_HOSTED_MTP_WITHOUT_ID_MARKER allow-listing is present"
        )
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


def test_exactly_one_score_model() -> None:
    # One reranker keeps score routing unambiguous — a second entry would need a
    # tiebreaker that doesn't exist for the score lane.
    score_ids = [m.id for m in SUPPORTED_MODELS if m.task == "score"]
    assert score_ids == [_RERANKER_ID], f"score models: {score_ids}"


def test_exactly_one_embed_model_carries_the_embedding_role_hint() -> None:
    # The embed lane now holds MORE than one entry (the 0.6B hot-path gear plus the
    # opt-in `embed-deep` slot), which is fine: they are wired as separate backends
    # with distinct served names and disambiguated by the "embed-deep" alias.
    #
    # What must stay singular is the ROLE DEFAULT. roles.ROLE_ROLE_HINT maps the
    # `embedder` role to role_hint "embedding" and _catalog_by_role_hint takes the
    # FIRST match, so a second "embedding" entry would silently hijack which model
    # `lobes capabilities` reports for the role. Additional embed gears must carry a
    # non-"embedding" role_hint (e.g. "candidate").
    embedding_hinted = [m.id for m in SUPPORTED_MODELS if m.role_hint == "embedding"]
    assert embedding_hinted == [_EMBEDDER_ID], f"role_hint='embedding' models: {embedding_hinted}"


def test_every_embed_model_has_a_distinct_id() -> None:
    # Served-name routing (resolve_model / order_backends) matches on the served
    # name, which defaults to the catalog id — duplicates would make the owning
    # backend for a request ambiguous.
    embed_ids = [m.id for m in SUPPORTED_MODELS if m.task == "embed"]
    assert len(embed_ids) == len(set(embed_ids)), f"duplicate embed ids: {embed_ids}"
    assert _EMBEDDER_ID in embed_ids


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
    # The 4B gear must STILL be present in the catalog with exactly the fields
    # the original acceptance criteria specify — any field drift is a
    # misconfiguration bug. Only its role_hint moved: the `hand` lobe took over
    # the cheap-tier slot and the 4B was demoted to a plain candidate
    # (cite-don't-delete), so the checkpoint stays selectable via `lobes switch`.
    minor = next((m for m in SUPPORTED_MODELS if m.id == _MINOR_ID), None)
    assert minor is not None, f"{_MINOR_ID} not found in catalog"
    assert minor.role_hint == "candidate"
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
#
# "Support both" (docs/vllm-nightly-migration.md §7, 2026-07-02): the catalog now
# carries TWO Gemma 4 12B gears — the NVFP4 BASE it-model (coolthor/…, DEFAULT
# "multimodal" gear, native MTP wired: 28.6 tok/s @ 57.9% draft acceptance, the
# fastest Gemma config measured) and the CODER fine-tune (sakamakismile/…, KEPT
# but DEMOTED to role_hint="candidate": coding-strong, but native MTP only reaches
# 30.8% acceptance on it — not worth wiring). Callers pick coding-strength
# (explicit id or the opt-in "multimodal-coder" alias) vs MTP-throughput (the
# default "multimodal"/"normal" tier).

_GEMMA_BASE_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_GEMMA_CODER_ID = "sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4"
_14B_ID = "nvidia/Qwen3-14B-NVFP4"
# Fleet default primary since 2026-08-19 (qwen3.8-cortex-upgrade plan t3);
# unsloth/Qwen3.6-27B-NVFP4 is demoted to role_hint="candidate" and kept
# (cite-don't-delete) — see its catalog entry's demotion comment.
_PRIMARY_ID = "unsloth/Qwen3.8-27B-NVFP4"

# The exact native-MTP speculative_config §7 measured on the NVFP4 base gear —
# 28.6 tok/s decode at 57.9% draft acceptance (vs the coder's 30.8%/~6% win, and
# the bf16 base's 93.9% accept but only 14.6 tok/s — bf16 is a speed trap). Note
# the "model" key, NOT "draft_model_id" — vLLM 0.23's SpeculativeConfig rejects
# that outdated key (verified live, §6).
_GEMMA_BASE_MTP_SPECULATIVE_CONFIG = (
    '{"method": "mtp", "model": "google/gemma-4-12B-it-assistant",' ' "num_speculative_tokens": 1}'
)


def test_gemma_multimodal_gear_exists_with_correct_fields() -> None:
    # The Gemma multimodal gear must be present with exactly the fields the
    # acceptance criteria specify — any field drift is a misconfiguration bug.
    gemma = next((m for m in SUPPORTED_MODELS if m.id == _GEMMA_BASE_ID), None)
    assert gemma is not None, f"{_GEMMA_BASE_ID} not found in catalog"
    assert gemma.role_hint == "multimodal"
    assert gemma.task == "generate"
    assert gemma.tool_parser == "gemma4"  # not "pythonic" — see test_parser.py
    # NVFP4 in compressed-tensors format (config.json quant_method), NOT modelopt —
    # matches the coder entry's quantization path; verified live on the Spark
    # (#71) for the same checkpoint family; modelopt_fp4 fails with a method
    # mismatch.
    assert gemma.quantization == "compressed-tensors"
    assert gemma.status == "load-tested"  # GB10 2026-07-02: 28.6 tok/s + MTP (§7)
    assert gemma.doc == "gemma-4-12b-nvfp4.md"
    assert gemma.native_max_model_len == 131072
    assert gemma.dimension == 0
    assert gemma.hf_overrides == ""
    assert gemma.moe_backend == ""  # not MoE
    # The shape phrase must advertise the unified multimodal (text+image+audio) nature.
    shape = gemma.shape.lower()
    assert "multimodal" in shape
    for modality in ("text", "image", "audio"):
        assert modality in shape, f"{_GEMMA_BASE_ID}: shape must mention {modality!r}"


def test_gemma_tool_parser_matches_infer_parser() -> None:
    # The catalog's parser must agree with the runtime's inference (t1), for BOTH
    # Gemma gears (base default + demoted coder candidate). "gemma4" — Gemma 4's
    # native tool-call syntax needs Gemma4EngineToolParser, not the generic
    # pythonic parser (disproven live 2026-07-17; see test_parser.py).
    for gemma_id in (_GEMMA_BASE_ID, _GEMMA_CODER_ID):
        gemma = next(m for m in SUPPORTED_MODELS if m.id == gemma_id)
        assert infer_parser(gemma.id) == "gemma4"
        assert gemma.tool_parser == infer_parser(gemma.id)


def test_gemma_base_has_native_mtp_speculative_config() -> None:
    # The NVFP4 base gear is the new default "multimodal" tier — native MTP is
    # wired ON by default (§7: 28.6 tok/s decode, 57.9% draft acceptance, the
    # fastest Gemma config measured). Exact-string match: this value is also
    # baked into lobes/templates/fleet/docker-compose.yml's vllm-multimodal
    # command (see test_gemma_base_mtp_speculative_config_round_trips_through_helper
    # + test_fleet_compose_multimodal_vision_active_has_native_mtp_spec_decode
    # for the drift guards).
    gemma = next(m for m in SUPPORTED_MODELS if m.id == _GEMMA_BASE_ID)
    assert gemma.speculative_config == _GEMMA_BASE_MTP_SPECULATIVE_CONFIG
    cfg = json.loads(gemma.speculative_config)
    assert cfg["method"] == "mtp"
    assert cfg["model"] == "google/gemma-4-12B-it-assistant"
    assert cfg["num_speculative_tokens"] == 1
    assert "draft_model_id" not in cfg, (
        "vLLM 0.23's SpeculativeConfig rejects the outdated 'draft_model_id' key "
        "— the draft id must be under 'model' (verified live, §6)"
    )


def test_gemma_coder_is_demoted_candidate_with_no_speculative_config() -> None:
    # "Support both" (§7, 2026-07-02): the coder fine-tune is KEPT (cite-don't-
    # delete) but DEMOTED from the default "multimodal" gear to role_hint=
    # "candidate" — coding-strong, but native MTP only reaches 30.8% draft
    # acceptance on it (a marginal ~6% decode win), so it stays selectable by id
    # (or the opt-in "multimodal-coder" alias) without carrying a
    # speculative_config.
    coder = next(m for m in SUPPORTED_MODELS if m.id == _GEMMA_CODER_ID)
    assert coder.role_hint == "candidate", f"{_GEMMA_CODER_ID}: expected demotion to 'candidate'"
    assert coder.speculative_config == "", (
        f"{_GEMMA_CODER_ID}: speculative_config must be empty — native MTP measured "
        "only 30.8% draft acceptance on this fine-tune (see docs/vllm-nightly-"
        "migration.md §6/§7), not worth wiring by default"
    )


def test_exactly_one_gemma_multimodal_gear() -> None:
    # The tier resolver depends on there being exactly one role_hint="multimodal"
    # gear — two would make resolve_tier("multimodal") ambiguous (first-match).
    multimodal_gears = [m for m in SUPPORTED_MODELS if m.role_hint == "multimodal"]
    assert [m.id for m in multimodal_gears] == [_GEMMA_BASE_ID]


def test_gemma_base_mtp_speculative_config_round_trips_through_helper() -> None:
    # Mirrors test_speculative_config_item_matches_27b_primarys_mtp_item's coverage
    # of the 27B primary's qwen3_5_mtp route, but for the Gemma NVFP4 base's native
    # MTP route (§7) — against the LIVE catalog entry (not a throwaway replace()),
    # since this route is the real, wired default now. Proves speculative_config_
    # item() is generic — not hardcoded to the 27B primary search
    # mtp_compose_command_items() does — by building the exact --speculative-config
    # compose item and parsing it back to the same JSON dict the catalog carries.
    #
    # Historical note: the DSpark draft-model route (deepseek-ai/dspark_gemma4_12b_
    # block7) investigated for this gear in #75 is INVALID on vLLM 0.23 — its
    # custom Gemma4DSparkModel architecture is not in the supported speculative-
    # draft set (§6, "DSpark MTP route for Gemma: INVALID on vLLM 0.23") — so it is
    # not wired here or anywhere in the catalog.
    live_gemma = next(m for m in SUPPORTED_MODELS if m.id == _GEMMA_BASE_ID)

    item = speculative_config_item(live_gemma)

    assert item == f"--speculative-config={_GEMMA_BASE_MTP_SPECULATIVE_CONFIG}"
    # round-trip: strip the flag prefix and parse the JSON back — must equal the
    # source config exactly (add/remove through `lobes switch` relies on this).
    parsed = json.loads(item.removeprefix("--speculative-config="))
    assert parsed == json.loads(_GEMMA_BASE_MTP_SPECULATIVE_CONFIG)
    assert parsed["method"] == "mtp"
    assert parsed["model"] == "google/gemma-4-12B-it-assistant"
    assert parsed["num_speculative_tokens"] == 1

    # This item must ALSO appear verbatim in the fleet compose template — the
    # single source of truth (catalog) must not drift from the packaged YAML.
    #
    # Since t5 (the senses MTP off-switch) the flag is env-parameterized, so the
    # guard pins it as the knob's DEFAULT rather than as a bare list item: the
    # whole `--speculative-config=<json>` token is the default of
    # MULTIMODAL_SPECULATIVE_CONFIG, single-quoted so the JSON's spaces survive
    # compose's shell-lexing as one argv token. Matching the full substitution —
    # not just the item text — is what keeps this a real drift guard: the item
    # string now also appears in that lane's prose comments, which a substring
    # check alone would happily match after someone deleted the actual flag.
    fleet_text = (_TEMPLATES / "fleet" / "docker-compose.yml").read_text(encoding="utf-8")
    parameterized = "${MULTIMODAL_SPECULATIVE_CONFIG-'" + item + "'}"
    assert parameterized in fleet_text, (
        f"{parameterized!r} missing from templates/fleet/docker-compose.yml "
        "(drift between the catalog entry and the senses lane's shipped default)"
    )


def test_speculative_config_item_matches_27b_primarys_mtp_item() -> None:
    # speculative_config_item() must be the SAME formatting mtp_compose_command_items()
    # uses for the 27B primary — proving the extraction is byte-identical, not a
    # parallel/duplicated implementation that could drift.
    sak = next(m for m in SUPPORTED_MODELS if m.id == _PRIMARY_ID)
    assert speculative_config_item(sak) == mtp_compose_command_items()[0]


def test_14b_is_demoted_to_candidate() -> None:
    # The 14B is KEPT but demoted from the "middle" tier to a legacy candidate;
    # it must remain in the catalog and now carry role_hint="candidate".
    middle = next((m for m in SUPPORTED_MODELS if m.id == _14B_ID), None)
    assert middle is not None, f"{_14B_ID} must be KEPT in the catalog (demoted, not deleted)"
    assert middle.role_hint == "candidate", f"{_14B_ID}: expected demotion to 'candidate'"


# ---------------------------------------------------------------------------
# qwen3.8-cortex-upgrade (plan t3): 3.8 promoted to primary, 3.6 demoted
# ---------------------------------------------------------------------------

_OUTGOING_PRIMARY_ID = "unsloth/Qwen3.6-27B-NVFP4"
_INCOMING_PRIMARY_ID = "unsloth/Qwen3.8-27B-NVFP4"


def test_qwen38_primary_gear_exists_with_correct_fields() -> None:
    # The config-verified 3.8 entry (fetched 2026-08-19) must be present and
    # carry role_hint="primary" plus the fields config files establish.
    gear = next((m for m in SUPPORTED_MODELS if m.id == _INCOMING_PRIMARY_ID), None)
    assert gear is not None, f"{_INCOMING_PRIMARY_ID} not found in catalog"
    assert gear.role_hint == "primary"
    assert gear.task == "generate"
    assert gear.quantization == "compressed-tensors"
    assert gear.native_max_model_len == 262144
    assert gear.tool_parser == "qwen3_coder"
    assert gear.speculative_config


def test_qwen38_is_the_only_primary_gear() -> None:
    # resolve_tier depends on exactly one role_hint="primary" generate gear —
    # a second entry would make "main"/"hard"/"cortex" resolution ambiguous by
    # first-match, exactly like the multimodal/worker/hand single-gear guards
    # elsewhere in this file.
    primaries = [m.id for m in SUPPORTED_MODELS if m.role_hint == "primary"]
    assert primaries == [_INCOMING_PRIMARY_ID], f"role_hint='primary' models: {primaries}"


def test_qwen36_is_demoted_to_candidate_and_kept() -> None:
    # The outgoing 3.6 primary is KEPT (cite-don't-delete) and demoted to a
    # legacy candidate; it must remain in the catalog and stay selectable.
    outgoing = next((m for m in SUPPORTED_MODELS if m.id == _OUTGOING_PRIMARY_ID), None)
    assert outgoing is not None, f"{_OUTGOING_PRIMARY_ID} must be KEPT (demoted, not deleted)"
    assert outgoing.role_hint == "candidate", f"{_OUTGOING_PRIMARY_ID}: expected demotion"
    assert outgoing.task == "generate"


def test_resolve_tier_main_and_hard_return_qwen38_primary() -> None:
    for tier in ("main", "hard", "cortex"):
        model = resolve_tier(tier)
        assert model.id == _INCOMING_PRIMARY_ID, f"resolve_tier({tier!r}) -> {model.id}"


def test_tier_role_map_uses_new_vocabulary() -> None:
    # Primary vocabulary: main/minor/multimodal. Back-compat aliases retained.
    # `minor`/`cheap` point at the `hand` BACKEND since the hand lobe replaced
    # Qwen3.5-4B in that slot — the tier NAMES survive for back-compat, the
    # `minor` backend role does not.
    assert TIER_ROLE["main"] == "primary"
    assert TIER_ROLE["minor"] == "hand"
    assert TIER_ROLE["multimodal"] == "multimodal"
    assert TIER_ROLE["cheap"] == "hand"
    assert TIER_ROLE["normal"] == "multimodal"
    assert TIER_ROLE["hard"] == "primary"
    assert TIER_ROLE["hand"] == "hand"
    # No tier resolves to a "minor" backend role any more.
    assert "minor" not in set(TIER_ROLE.values())


def test_resolve_tier_multimodal_and_normal_return_gemma() -> None:
    # Both the new "multimodal" alias and the back-compat "normal" alias resolve
    # to the Gemma 4 multimodal gear (the reframed normal tier).
    for tier in ("multimodal", "normal"):
        model = resolve_tier(tier)
        assert isinstance(model, SupportedModel)
        assert model.id == _GEMMA_BASE_ID, f"resolve_tier({tier!r}) -> {model.id} (expected Gemma)"
        assert model.role_hint == "multimodal"


def test_resolve_tier_main_and_hard_return_primary() -> None:
    for tier in ("main", "hard"):
        model = resolve_tier(tier)
        assert model.role_hint == "primary"
        assert model.id == _PRIMARY_ID
        assert model.task == "generate"


def test_resolve_tier_minor_cheap_and_hand_all_return_the_hand_gear() -> None:
    # The hand lobe REPLACED Qwen3.5-4B in the cheap-tier slot, so all three
    # spellings resolve to the same lane. `minor`/`cheap` are back-compat names
    # for it, not a second gear.
    for tier in ("minor", "cheap", "hand"):
        model = resolve_tier(tier)
        assert model.role_hint == "hand", f"resolve_tier({tier!r}) -> {model.role_hint}"
        assert model.id == _HAND_ID, f"resolve_tier({tier!r}) -> {model.id}"


def test_demoted_4b_remains_in_catalog_after_the_hand_repoint() -> None:
    # cite-don't-delete: repointing the tier must not remove the checkpoint.
    # It stays selectable via `lobes switch` even though no tier resolves to it.
    assert any(m.id == _MINOR_ID for m in SUPPORTED_MODELS), f"{_MINOR_ID} was deleted, not demoted"


def test_resolve_tier_unknown_still_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        resolve_tier("ultra")


# ---------------------------------------------------------------------------
# `hand` gear: LiquidAI/LFM2.5-1.2B-Instruct — the ninth Colleague role and the
# fleet's designated fine-tuning base (hand-lobe plan, t1).
# ---------------------------------------------------------------------------

_HAND_ID = "LiquidAI/LFM2.5-1.2B-Instruct"


def test_hand_gear_exists_with_correct_fields() -> None:
    hand = next((m for m in SUPPORTED_MODELS if m.id == _HAND_ID), None)
    assert hand is not None, f"{_HAND_ID} not found in catalog"
    assert hand.role_hint == "hand"
    assert hand.native_max_model_len == 32768
    # bf16 sentinel: the vllm-hand lane omits --quantization entirely rather
    # than passing it empty, which would corrupt the weights.
    assert hand.quantization == "none"
    assert hand.tool_parser == "lfm2"
    assert hand.task == "generate"
    assert hand.doc == "lfm2.5-1.2b-hand.md"
    # Text-only: no ViT, no MoE, no speculative draft head in v1.
    assert hand.dimension == 0
    assert hand.moe_backend == ""
    assert hand.speculative_config == ""
    assert hand.hf_overrides == ""


def test_hand_is_the_only_gear_carrying_the_hand_role_hint() -> None:
    # `resolve_tier` returns the FIRST generate model matching the role, so a
    # second hand-hinted entry would silently decide the tier by list order.
    hand_hinted = [m.id for m in SUPPORTED_MODELS if m.role_hint == "hand"]
    assert hand_hinted == [_HAND_ID], f"role_hint='hand' models: {hand_hinted}"


def test_tier_aliases_capability_order_is_hand_multimodal_worker_muse_primary() -> None:
    """Assert the ORDER, not just the mapping.

    ``tier_aliases`` derives ascending capability order from each role's *last*
    occurrence position in ``TIER_ROLE``'s values sequence, so the dict's key
    ORDER is load-bearing: it decides which lane an unwired tier falls back
    upward to. A reorder that still maps every key correctly can silently
    invert the ladder, which is why this asserts the sequence.
    """
    last_pos: dict[str, int] = {}
    for index, role in enumerate(TIER_ROLE.values()):
        last_pos[role] = index
    ascending = sorted(last_pos, key=lambda role: last_pos[role])
    assert ascending == ["hand", "multimodal", "worker", "muse", "primary"]


# ---------------------------------------------------------------------------
# `worker` gear: unsloth/Qwen3.6-35B-A3B-NVFP4 (thor-worker-lobe plan, t1)
# ---------------------------------------------------------------------------
# A DISTINCT catalog entry from the existing mmangkad/Qwen3.6-35B-A3B-NVFP4
# "candidate" above — same architecture family, different org/export. Facts
# below are verified against the checkpoint's ACTUAL config files (fetched
# 2026-07-31), not card prose:
#   https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/resolve/main/config.json
#     - text_config.max_position_embeddings = 262144
#     - quantization_config.quant_method = "compressed-tensors"
#   https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/resolve/main/hf_quant_config.json
#     - 404 Not Found (compressed-tensors checkpoints carry their quant config
#       inline in config.json; there is no separate hf_quant_config.json here,
#       unlike the nvidia/modelopt-style exports elsewhere in this catalog).
#   The README's own vLLM MTP serve command:
#     vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4 \
#         --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'


def test_worker_gear_exists_with_correct_fields() -> None:
    worker = next((m for m in SUPPORTED_MODELS if m.id == _WORKER_ID), None)
    assert worker is not None, f"{_WORKER_ID} not found in catalog"
    assert worker.role_hint == "worker"
    assert worker.native_max_model_len == 262144  # config.json max_position_embeddings
    assert worker.quantization == "compressed-tensors"  # config.json quant_method
    assert worker.tool_parser == "qwen3_coder"
    assert worker.status == "load-tested"  # Thor sm_110 2026-07-31: boots+serves
    assert worker.task == "generate"
    assert worker.dimension == 0
    assert worker.hf_overrides == ""
    assert worker.doc, f"{_WORKER_ID}: doc must be non-empty"


def test_worker_gear_speculative_config_matches_checkpoint_card() -> None:
    worker = next(m for m in SUPPORTED_MODELS if m.id == _WORKER_ID)
    assert worker.speculative_config == '{"method": "mtp", "num_speculative_tokens": 2}'
    cfg = json.loads(worker.speculative_config)
    assert cfg["method"] == "mtp"
    assert cfg["num_speculative_tokens"] == 2
    # Self-hosted draft: no external "model"/"draft_model_id" key.
    assert "model" not in cfg
    assert "draft_model_id" not in cfg


def test_worker_gear_is_moe_with_auto_selected_backend() -> None:
    worker = next(m for m in SUPPORTED_MODELS if m.id == _WORKER_ID)
    assert worker.shape.lower().startswith("moe")
    # NO forced moe_backend: measured on Thor sm_110 (2026-07-31), every forced
    # NVFP4 MoE backend was refused (flashinfer_* lack sm_110 kernels;
    # marlin/triton reject the mixed quantized-main/unquantized-MTP experts), so
    # the gear leaves --moe-backend off and vLLM auto-selects per path.
    assert worker.moe_backend == ""


def test_worker_gear_tool_parser_matches_infer_parser() -> None:
    worker = next(m for m in SUPPORTED_MODELS if m.id == _WORKER_ID)
    assert infer_parser(worker.id) == worker.tool_parser


def test_worker_gear_does_not_modify_the_mmangkad_candidate_sibling() -> None:
    # The pre-existing mmangkad/ candidate entry must stay untouched: same
    # role_hint="candidate", 32K native, marlin moe_backend, no speculative_config.
    mmangkad = next(m for m in SUPPORTED_MODELS if m.id == "mmangkad/Qwen3.6-35B-A3B-NVFP4")
    assert mmangkad.role_hint == "candidate"
    assert mmangkad.native_max_model_len == 32768
    assert mmangkad.moe_backend == "marlin"
    assert mmangkad.speculative_config == ""


def test_exactly_one_worker_gear() -> None:
    workers = [m.id for m in SUPPORTED_MODELS if m.role_hint == "worker"]
    assert workers == [_WORKER_ID], f"role_hint='worker' models: {workers}"


def test_resolve_tier_worker_returns_the_worker_gear() -> None:
    model = resolve_tier("worker")
    assert isinstance(model, SupportedModel)
    assert model.id == _WORKER_ID
    assert model.role_hint == "worker"
    assert model.task == "generate"


# ---------------------------------------------------------------------------
# `unsloth/gemma-4-12B-it-qat-w4a16` — QAT int4 W4A16 candidate for the Gemma
# 4 12B unified family (unsloth-qat-senses-first-class-orin-variation plan, t1)
# ---------------------------------------------------------------------------
# A SECOND candidate senses/multimodal checkpoint, mirroring the coolthor gear
# in architecture (same Gemma4UnifiedForConditionalGeneration class, same
# gemma4 tool-call parser, same compressed-tensors quantization flag) but a
# DIFFERENT quant scheme (int4 pack-quantized weight-only, not FP4) and DOUBLE
# the native context. Verified against the checkpoint's own config.json
# (fetched unauthenticated 2026-08-04), not card prose.

_QAT_W4A16_ID = "unsloth/gemma-4-12B-it-qat-w4a16"


def test_qat_w4a16_gear_exists_with_correct_fields() -> None:
    gear = next((m for m in SUPPORTED_MODELS if m.id == _QAT_W4A16_ID), None)
    assert gear is not None, f"{_QAT_W4A16_ID} not found in catalog"
    assert gear.tool_parser == "gemma4"
    assert gear.quantization == "compressed-tensors"
    assert gear.native_max_model_len == 262144  # config.json max_position_embeddings
    assert gear.status in {"configured", "load-tested"}
    assert gear.status == "configured", (
        f"{_QAT_W4A16_ID}: nothing is live-probed yet — status must not claim " "load-tested"
    )
    assert gear.doc == "gemma-4-12b-qat-w4a16.md"
    assert gear.task == "generate"
    assert gear.dimension == 0
    assert gear.hf_overrides == ""
    assert gear.moe_backend == ""
    # No MTP/draft head ships in this checkpoint (config.json carries no
    # mtp_num_hidden_layers or draft-head fields) — unlike the coolthor gear's
    # wired google/gemma-4-12B-it-assistant draft.
    assert gear.speculative_config == ""


def test_qat_w4a16_shape_mentions_every_declared_modality() -> None:
    # config.json declares vision_config, audio_config, AND a video_token_id —
    # video is a genuinely new declaration this checkpoint's export makes that
    # the shared _SHAPE_GEMMA4_UNIFIED literal (text+image+audio) predates.
    gear = next(m for m in SUPPORTED_MODELS if m.id == _QAT_W4A16_ID)
    shape = gear.shape.lower()
    for modality in ("text", "image", "audio", "video"):
        assert modality in shape, f"{_QAT_W4A16_ID}: shape must mention {modality!r}"


def test_qat_w4a16_tool_parser_matches_infer_parser() -> None:
    gear = next(m for m in SUPPORTED_MODELS if m.id == _QAT_W4A16_ID)
    assert infer_parser(gear.id) == "gemma4"
    assert gear.tool_parser == infer_parser(gear.id)


def test_qat_w4a16_context_is_double_the_coolthor_incumbent() -> None:
    # config.json max_position_embeddings=262144 vs the coolthor gear's
    # measured 131072 — the QAT export doubles the native window.
    gear = next(m for m in SUPPORTED_MODELS if m.id == _QAT_W4A16_ID)
    coolthor = next(m for m in SUPPORTED_MODELS if m.id == _GEMMA_BASE_ID)
    assert gear.native_max_model_len == 2 * coolthor.native_max_model_len


def test_qat_w4a16_does_not_hijack_the_multimodal_tier_default() -> None:
    # DELIBERATE deviation from the covering plan's "role_hint=multimodal"
    # wording: tests/test_catalog.py's test_exactly_one_gemma_multimodal_gear
    # pins role_hint="multimodal" to EXACTLY [coolthor] — a second entry with
    # that role_hint would make resolve_tier("multimodal")/("senses")/("normal")
    # ambiguous by first-match. Mirrors the same reasoning the sakamakismile
    # coder entry already uses (role_hint="candidate", not "multimodal"). This
    # keeps the fleet-wide tier default, and every thor/spark profile pinning
    # the raw coolthor id, untouched — a per-box profile is how a deployment
    # actually selects this checkpoint for its senses role, not this field.
    gear = next(m for m in SUPPORTED_MODELS if m.id == _QAT_W4A16_ID)
    assert gear.role_hint == "candidate"
    assert resolve_tier("multimodal").id == _GEMMA_BASE_ID
    assert resolve_tier("senses").id == _GEMMA_BASE_ID
    assert resolve_tier("normal").id == _GEMMA_BASE_ID


@pytest.mark.skipif(not _DOCS.is_dir(), reason="docs/ not shipped (wheel install)")
def test_qat_w4a16_doc_records_the_honesty_bar() -> None:
    # Structural guard (mirrors the byte-guard tests above, but for prose
    # honesty): the per-model doc must actually record the facts that make
    # this entry NOT a copy-paste of the coolthor gear — the int4-vs-FP4 quant
    # difference, the doubled context, and the known #101 audio-drop risk —
    # rather than silently inheriting coolthor's capability claims.
    text = (_DOCS / "gemma-4-12b-qat-w4a16.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "int4" in lowered
    assert "262144" in text
    assert "pending-live-probe" in lowered or "pending live probe" in lowered
    assert "#101" in text
