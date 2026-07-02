"""The supported-model catalog — the "gears" lobes can change to.

A pure, dependency-free data module: the single source of truth for the models
lobes knows how to serve (each one load-tested or configured on the DGX
Spark and documented under ``docs/``). It ships *in the wheel* so both runtimes
can read it:

* the CLI (``lobes overview --list``) — which would otherwise scan ``docs/`` and
  find nothing in a wheel install (``docs/`` is not packaged), and
* the gateway (``GET /v1/models/supported``) — which runs from a pip-installed
  wheel inside its container and has no source tree to scan.

The per-model ``docs/`` files remain the *human* prose; this module is the
*machine* catalog. ``tests/test_catalog.py`` asserts the two cannot silently
diverge (every ``doc`` file exists; every parser matches ``infer_parser``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Shared ``context`` literal — three catalog entries (the archived Mistral
# fallback and both Gemma 4 12B unified entries) share this exact native
# context window; a single constant keeps them from drifting independently
# (SonarCloud: duplicated string literal).
_CONTEXT_128K_NATIVE = "128K native"


@dataclass(frozen=True)
class SupportedModel:
    """One model the fleet/CLI can serve — a gear you can change to."""

    id: str  # OpenAI model id (== the vLLM --served-model-name)
    # The fleet's default role for this gear. One of:
    # "primary" | "fallback" | "candidate" | "minor" | "multimodal" | "embedding" | "reranker".
    # The generate-lane tier aliases (main/minor/multimodal + back-compat
    # cheap/normal/hard) resolve to a gear by this field — see TIER_ROLE / resolve_tier.
    role_hint: str
    shape: str  # architecture in a phrase, e.g. "dense" / "MoE (~3B active)"
    context: str  # native context window, human-readable
    # The largest --max-model-len this checkpoint serves with vLLM's *default* rope
    # (no YaRN/rope-scaling override) — a hard ceiling: vLLM refuses a larger value
    # and the container fails to boot. `lobes switch` clamps the machine-profile
    # context default DOWN to this, so a high machine default (e.g. spark's 256K)
    # can't silently boot-fail a 32K-native model. An explicit --max-model-len wins.
    native_max_model_len: int
    tool_parser: str  # vLLM --tool-call-parser (must match runtime._parser.infer_parser)
    quantization: str  # vLLM --quantization
    status: str  # "load-tested" (measured on this hardware) | "configured" (not yet)
    doc: str  # per-model markdown under docs/ (filename only)
    # Per-model serve extras for MoE checkpoints. Empty for dense/hybrid models;
    # set only where the architecture needs them. These are NOT in the default
    # single-model template (docker compose can't conditionally omit a flag, and
    # an empty `--moe-backend=` token breaks vLLM) — `lobes switch` surfaces them
    # as a documented compose edit. See docs/qwen3.6-35b-a3b-nvfp4.md.
    moe_backend: str = ""  # vLLM --moe-backend (e.g. "marlin") for MoE models
    speculative_config: str = ""  # vLLM --speculative-config JSON (e.g. MTP draft)
    task: str = "generate"  # "generate" | "embed" | "score"
    dimension: int = 0  # embedding output dimension; 0 for non-embedding models
    hf_overrides: str = ""  # vLLM --hf-overrides JSON string


SUPPORTED_MODELS: tuple[SupportedModel, ...] = (
    SupportedModel(
        id="mmangkad/Qwen3.6-27B-NVFP4",
        # Archived former primary (superseded 2026-05-31 by the MTP build below).
        # Kept in the catalog for two reasons: (1) it is the tokenizer source the
        # MTP primary serves with (--tokenizer=mmangkad/Qwen3.6-27B-NVFP4), and
        # (2) it is the only *vision-capable* 27B — the MTP primary is text-only,
        # so this is the fallback when an image path is needed.
        role_hint="candidate",
        shape="hybrid Mamba/linear-attn + ViT (multimodal)",
        context="256K native",
        native_max_model_len=262144,
        tool_parser="qwen3_coder",
        quantization="modelopt_fp4",
        status="load-tested",
        doc="qwen3.6-27b-nvfp4.md",
    ),
    SupportedModel(
        id="RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4",
        role_hint="fallback",
        shape="dense (vision-capable)",
        context=_CONTEXT_128K_NATIVE,
        native_max_model_len=131072,
        tool_parser="mistral",
        quantization="compressed-tensors",
        status="load-tested",
        doc="mistral-small-3.2-24b-nvfp4.md",
    ),
    SupportedModel(
        id="nvidia/Qwen3-32B-NVFP4",
        role_hint="candidate",
        shape="dense",
        context="32K (→131K via YaRN)",
        # 32K native: 131K needs an explicit YaRN --rope-scaling override (pass
        # --max-model-len 131072 with it). Without that, 32768 is the boot ceiling.
        native_max_model_len=32768,
        tool_parser="hermes",
        quantization="modelopt_fp4",
        status="load-tested",
        doc="qwen3-32b-nvfp4.md",
    ),
    SupportedModel(
        id="sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
        # Fleet default primary since 2026-05-31 (promoted from candidate after the
        # tool-calling gate passed: a valid qwen3_coder tool call + full tool
        # round-trip + reasoning trace, all under the production compose, with MTP
        # spec-decode active at 78.6% draft acceptance and 18.7 tok/s decode —
        # ~2.4x the archived baseline 27B). Replaces mmangkad/Qwen3.6-27B-NVFP4.
        role_hint="primary",
        shape="hybrid Mamba/linear-attn (text-only, MTP draft head)",
        context="256K native (served at full 256K on the shared GB10)",
        native_max_model_len=262144,
        tool_parser="qwen3_coder",
        quantization="modelopt",
        status="load-tested",
        doc="qwen3.6-27b-text-nvfp4-mtp.md",
        # MTP primary (issue #26): an MTP-grafted re-export of the archived 27B —
        # the baseline NVFP4 export drops the MTP draft head (0% draft acceptance),
        # so this repo restores it in bf16 for vLLM speculative decoding. The
        # --speculative-config is catalog data (like moe_backend): compose can't omit
        # an empty flag, so `lobes switch` surfaces it as a hand edit. Load-tested on
        # the GB10 2026-05-31: 19.1 tok/s decode (~2.4x the baseline 27B) at 72% MTP
        # acceptance on vLLM 0.19.0+nv26.04. Also needs --trust-remote-code +
        # --language-model-only, VLLM_MAX_NUM_SEQS=2 (4 OOMs at n=3/256K), and a
        # tokenizer override (--tokenizer=mmangkad/Qwen3.6-27B-NVFP4 — the checkpoint's
        # tokenizer_config declares TokenizersBackend, absent from the nv26.04 image).
        # Quantization `modelopt` resolves to modelopt_fp4. See the doc.
        speculative_config='{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}',
    ),
    SupportedModel(
        id="mmangkad/Qwen3.6-35B-A3B-NVFP4",
        role_hint="candidate",
        shape="MoE (~3B active per token)",
        context="32K",
        native_max_model_len=32768,
        tool_parser="qwen3_coder",
        quantization="modelopt_fp4",
        status="configured",
        doc="qwen3.6-35b-a3b-nvfp4.md",
        # MoE-only serve extra: the marlin MoE kernel — verified to load this
        # checkpoint *solo* on the GB10 (2026-05-31, util 0.70). lobes switch
        # surfaces it as a compose edit; it must not land on the dense/hybrid models.
        # shahizat's MTP --speculative-config is intentionally NOT carried: it is
        # tied to the nvidia/ checkpoint and FAILS to load on this mmangkad copy
        # (qwen3_5_mtp.py weight-shape mismatch on vLLM nv26.04). See the doc.
        moe_backend="marlin",
    ),
    SupportedModel(
        id="Qwen/Qwen3-Embedding-0.6B",
        # Embedding gear (issue #44): 1024-dim dense text embeddings with Matryoshka
        # nesting (32/64/128/256/512/768/1024). Zero tool-parser and quantization —
        # this is a pooling model, not a chat/completion model. Served via vLLM's
        # embedding endpoint (/v1/embeddings). The hf_overrides enables Matryoshka
        # truncation so consumers can request sub-1024 dimensions without re-serving.
        role_hint="embedding",
        shape="dense embedding (text)",
        context="32K native",
        native_max_model_len=32768,
        tool_parser="",
        quantization="",
        status="load-tested",  # GB10 2026-06-19: dim 1024, MRL 256 ✓, ~28ms warm, co-resident
        doc="qwen3-embedding-0.6b.md",
        task="embed",
        dimension=1024,
        hf_overrides=(
            '{"is_matryoshka": true,'
            ' "matryoshka_dimensions": [32, 64, 128, 256, 512, 768, 1024]}'
        ),
    ),
    SupportedModel(
        id="nvidia/Qwen3-14B-NVFP4",
        # 14B dense NVFP4 — a LEGACY CANDIDATE, KEPT but DEMOTED. It was the
        # fleet's "middle"/normal tier between the 4B minor and the 27B primary;
        # the normal tier is now served by the Gemma 4 12B unified-multimodal gear
        # (role_hint="multimodal"), so this 14B is demoted to role_hint="candidate"
        # and is no longer the normal tier (no tier alias resolves to it). It stays
        # in the catalog as a supported candidate you can switch to explicitly by id.
        # Not load-tested on the DGX Spark (status="configured"). 32K native context
        # (→131K via YaRN, same as the 32B entry). Dense architecture like
        # Qwen3-32B-NVFP4 — no MoE, no MTP draft head, no hf_overrides. Consistent
        # with the nvidia/ Qwen3-32B-NVFP4 entry (same org, NVFP4, hermes tool-call
        # format, modelopt_fp4 quantization). The exact HF checkpoint id is an
        # accepted plan risk (issue #68): verify on the Spark before any promotion.
        # See docs/qwen3-14b-nvfp4.md.
        role_hint="candidate",
        shape="dense",
        context="32K (→131K via YaRN)",
        native_max_model_len=32768,
        tool_parser="hermes",
        quantization="modelopt_fp4",
        status="configured",
        doc="qwen3-14b-nvfp4.md",
        task="generate",
    ),
    SupportedModel(
        id="Qwen/Qwen3.5-4B",
        # bf16 base (the unsloth-LoRA fine-tune target): the fleet's first LoRA
        # target and "minor" small-brain companion to the 27B primary. Multimodal
        # (hybrid linear-attn + ViT) — serve text-only via --language-model-only.
        # Built-in MTP head not used in v1 (no speculative_config carried).
        # quantization="none" is the bf16/unquantized sentinel — VLLM_QUANTIZATION
        # is NOT written on switch; the operator must REMOVE the --quantization
        # flag from the compose command: by hand (the single-model template defaults
        # to --quantization=modelopt when VLLM_QUANTIZATION is absent, which would
        # corrupt bf16 weights). See docs/qwen3.5-4b-minor.md.
        role_hint="minor",
        shape="hybrid linear-attn + ViT (multimodal)",
        context="256K native",
        native_max_model_len=262144,
        tool_parser="qwen3_coder",
        quantization="none",
        status="configured",
        doc="qwen3.5-4b-minor.md",
        task="generate",
    ),
    SupportedModel(
        id="coolthor/gemma-4-12B-it-NVFP4A16",
        # Gemma 4 12B (Google DeepMind) BASE it-model, NVFP4 — the fleet's DEFAULT
        # "multimodal" generate gear (and the "normal" tier) as of the "support both"
        # decision (docs/vllm-nightly-migration.md §7, 2026-07-02). Same UNIFIED
        # architecture as the coder entry below (Gemma4UnifiedForConditionalGeneration:
        # text + image + AUDIO in one checkpoint, no separate sidecars). Promoted over
        # the coder because it is the exact target the public
        # google/gemma-4-12B-it-assistant MTP draft was trained for: measured **28.6
        # tok/s decode at 57.9% draft acceptance** with native MTP on — the FASTEST
        # Gemma config measured (beats the coder's 24 tok/s no-spec/+MTP, and the bf16
        # base+MTP's 14.6 tok/s — bf16 has higher 93.9% acceptance but a much slower
        # no-spec floor). "Less coder, more MTP" — see §7 for the full comparison
        # table. Tool calls use the Python-style "pythonic" parser (matches
        # runtime._parser.infer_parser, which returns "pythonic" for gemma-4* ids).
        role_hint="multimodal",
        shape="unified multimodal (text+image+audio)",
        # Same base-model family as the coder entry — text_config.max_position_
        # embeddings=131072 confirmed for the Unified 12B IT line (#71); not
        # independently re-measured for this exact NVFP4A16 export.
        context=_CONTEXT_128K_NATIVE,
        native_max_model_len=131072,
        tool_parser="pythonic",
        # quantization matches the coder entry's compressed-tensors NVFP4 path
        # (config.json quant_method="compressed-tensors"); modelopt_fp4 fails with a
        # quant-method mismatch on this checkpoint family (verified #71).
        quantization="compressed-tensors",
        status="load-tested",  # GB10 2026-07-02: 19.8 tok/s no-spec, 28.6 tok/s +MTP (§7)
        doc="gemma-4-12b-nvfp4.md",
        task="generate",
        # Native MTP, default-on (§7, measured 2026-07-02): the public assistant
        # draft, wired with the "model" key (NOT "draft_model_id" — vLLM 0.23's
        # SpeculativeConfig rejects that outdated key; verified live). 57.9% draft
        # acceptance, ~1.45x decode speedup (19.8 -> 28.6 tok/s).
        speculative_config=(
            '{"method": "mtp", "model": "google/gemma-4-12B-it-assistant",'
            ' "num_speculative_tokens": 1}'
        ),
    ),
    SupportedModel(
        id="sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4",
        # Gemma 4 12B (Google DeepMind) CODER fine-tune — KEPT as an opt-in
        # candidate (cite-don't-delete), DEMOTED from the default "multimodal" gear
        # by the "support both" decision (docs/vllm-nightly-migration.md §7,
        # 2026-07-02): coding-strong, but native MTP is only 30.8% draft acceptance
        # here (the coder fine-tune's output distribution has shifted away from what
        # the assistant draft — trained against the base it-model — expects), a
        # marginal ~6% decode win not worth wiring by default. The NVFP4 base entry
        # above is the new default "multimodal"/"normal" tier gear. This entry stays
        # selectable by id (`lobes switch coolthor/... ` is the default; this coder
        # checkpoint remains a supported candidate for coding-heavy workloads).
        #
        # A UNIFIED multimodal model: a single Gemma4UnifiedForConditionalGeneration
        # serves text + image + AUDIO in one checkpoint (no separate sidecars). Tool
        # calls use the Python-style "pythonic" parser (matches runtime._parser.
        # infer_parser, which returns "pythonic" for gemma-4* ids — set in t1).
        #
        # status="load-tested". Serve-enablement RESOLVED on the Spark GB10 (#71/#73,
        # 2026-07-01): the gear SERVES on the custom image (Dockerfile.vllm-gemma4 =
        # vllm/vllm-openai nightly, vLLM 0.23.1rc1 + the vllm[audio] extra) via vLLM's
        # NATIVE Gemma4UnifiedForConditionalGeneration class, which handles the
        # heterogeneous per-layer head sizes (40 sliding@256 + 8 full@512) that broke
        # released vLLM <=0.22.1 (transformers-backend fallback → o_proj marlin_gemm
        # 4096≠8192; a backend flag does NOT fix it — the native class does). Validated
        # live: text ✓, image+text ✓, audio+text ✓ (transcribed a TTS clip verbatim);
        # ~15.7 GiB footprint ≈ 0.12 budget. See docs/gemma-4-12b-nvfp4.md and #71.
        role_hint="candidate",
        shape="unified multimodal (text+image+audio)",
        # Native context confirmed 128K (text_config.max_position_embeddings=131072,
        # read from the checkpoint config during #71 live validation).
        context=_CONTEXT_128K_NATIVE,
        native_max_model_len=131072,
        tool_parser="pythonic",
        # This checkpoint is NVFP4 in compressed-tensors format (config.json
        # quant_method="compressed-tensors", format "nvfp4-pack-quantized") — NOT
        # nvidia modelopt. vLLM must be told --quantization=compressed-tensors;
        # passing modelopt_fp4 fails with a quant-method-mismatch (verified #71).
        quantization="compressed-tensors",
        status="load-tested",  # GB10 2026-07-01: text+image+audio ✓ on vLLM nightly (#71/#73)
        doc="gemma-4-12b-nvfp4.md",
        task="generate",
        # No speculative_config: native MTP was measured on this checkpoint (§6/§7)
        # but only reaches 30.8% draft acceptance (~6% decode win) — the coder
        # fine-tune's distribution has shifted too far from what the assistant draft
        # (trained against the base it-model) expects. Not worth wiring by default;
        # the NVFP4 base entry above carries the wired MTP config instead. See
        # docs/vllm-nightly-migration.md §7.
    ),
    SupportedModel(
        id="Qwen/Qwen3-Reranker-0.6B",
        # Reranker gear (issue #44): cross-encoder that scores (query, passage) pairs.
        # Built on Qwen3ForSequenceClassification with a binary yes/no logit head;
        # served via vLLM's score endpoint (/v1/score). The hf_overrides declare the
        # non-standard architecture class and the two classifier tokens so vLLM can
        # load the head correctly. Zero tool-parser and quantization (score-only model).
        role_hint="reranker",
        shape="dense cross-encoder (Qwen3ForSequenceClassification)",
        context="32K native",
        native_max_model_len=32768,
        tool_parser="",
        quantization="",
        status="load-tested",  # GB10 2026-06-19: /v1/rerank+/v1/score ✓, ~25ms warm, co-resident
        doc="qwen3-reranker-0.6b.md",
        task="score",
        dimension=0,
        hf_overrides=(
            '{"architectures": ["Qwen3ForSequenceClassification"],'
            ' "classifier_from_token": ["no", "yes"],'
            ' "is_original_qwen3_reranker": true}'
        ),
    ),
)


def supported_models() -> tuple[SupportedModel, ...]:
    """The full supported-model catalog (the gears you can change to)."""
    return SUPPORTED_MODELS


def as_dicts() -> list[dict[str, str]]:
    """The catalog as plain dicts — for JSON emission without importing the dataclass."""
    return [asdict(model) for model in SUPPORTED_MODELS]


# The tokenizer the MTP primary serves with — a base-checkpoint override (the MTP
# checkpoint's tokenizer_config declares a class absent from the nv26.04 image; see
# docs/qwen3.6-27b-text-nvfp4-mtp.md caveat 1). Drop once fixed upstream (issue #29).
MTP_TOKENIZER_OVERRIDE = "mmangkad/Qwen3.6-27B-NVFP4"


# ---------------------------------------------------------------------------
# Tier → role_hint map — the generate-lane capability tiers
# ---------------------------------------------------------------------------
# Vocabulary reframed to main / minor / multimodal (the prior cheap/normal/hard
# tier names are retained as back-compat aliases). The "normal" tier is now the
# Gemma 4 12B unified-multimodal gear (role_hint="multimodal"); it replaced the
# 14B "middle" gear, which is demoted to a legacy candidate (no tier resolves
# to it any more).

#: Maps a tier alias to the ``role_hint`` of the gear that serves it.
#:
#: Primary vocabulary:
#:   main       → primary    (27B MTP primary — full capability, the "hard" tier)
#:   minor      → minor      (4B bf16 small-brain companion — fast, low memory)
#:   multimodal → multimodal (Gemma 4 12B unified text+image+audio gear)
#:
#: Back-compat aliases (the prior cheap/normal/hard tier names):
#:   cheap  → minor      (== minor)
#:   normal → multimodal (was the 14B "middle"; reframed to the Gemma gear)
#:   hard   → primary    (== main)
TIER_ROLE: dict[str, str] = {
    # Primary vocabulary.
    "main": "primary",
    "minor": "minor",
    "multimodal": "multimodal",
    # Back-compat aliases.
    "cheap": "minor",
    "normal": "multimodal",
    "hard": "primary",
}


def resolve_tier(tier: str) -> "SupportedModel":
    """Return the *first* generate-task ``SupportedModel`` whose ``role_hint``
    matches ``TIER_ROLE[tier]``.

    :param tier: A tier alias — one of the :data:`TIER_ROLE` keys. The primary
        vocabulary is ``"main"`` / ``"minor"`` / ``"multimodal"``; the legacy
        ``"cheap"`` / ``"normal"`` / ``"hard"`` names are retained as aliases.
        ``"main"`` and ``"hard"`` resolve to the primary; ``"minor"`` and
        ``"cheap"`` to the 4B minor; ``"multimodal"`` and ``"normal"`` to the
        Gemma 4 multimodal gear.
    :raises ValueError: If *tier* is not a known key in :data:`TIER_ROLE`.
    """
    role = TIER_ROLE.get(tier)
    if role is None:
        known = ", ".join(sorted(TIER_ROLE))
        raise ValueError(f"unknown tier {tier!r} — must be one of: {known}")
    for model in SUPPORTED_MODELS:
        if model.role_hint == role and model.task == "generate":
            return model
    # Should never happen if the catalog is internally consistent.
    raise LookupError(  # pragma: no cover
        f"no generate-task model with role_hint={role!r} found in catalog "
        f"(tier={tier!r}); catalog may be incomplete"
    )


def speculative_config_item(model: SupportedModel) -> str:
    """The ``--speculative-config=<json>`` compose item for a model's speculative
    decoding config.

    Generic across *any* gear carrying a non-empty ``speculative_config`` — not
    hardcoded to the 27B primary. ``mtp_compose_command_items()`` below calls this to
    build the primary's item; a future gear with its own draft-model route (e.g. a
    Gemma DSpark ``draft_model`` config — see ``tests/test_catalog.py``'s
    ``test_gemma_dspark_speculative_config_round_trips_through_helper``, issue #75)
    can call it directly with its own catalog entry (or a throwaway copy of one)
    without duplicating the JSON-embedding logic, and without the 27B-specific
    ``--trust-remote-code`` / ``--language-model-only`` / ``--tokenizer=`` extras that
    ``mtp_compose_command_items()`` also emits.

    :raises ValueError: if ``model.speculative_config`` is empty — there is nothing
        to format.
    """
    if not model.speculative_config:
        raise ValueError(f"{model.id}: speculative_config is empty — nothing to format")
    return f"--speculative-config={model.speculative_config}"


def mtp_compose_command_items() -> list[str]:
    """The extra compose ``command:`` items the MTP default primary needs.

    These four flags are baked into the packaged compose templates *and* named by
    ``lobes switch`` as the lines to remove when switching to a non-MTP model. This
    is the single source of truth so the two cannot drift — ``tests/test_catalog.py``
    asserts the packaged templates contain exactly these items, and the speculative
    config is pulled from the primary catalog entry rather than re-typed.

    Returns argv tokens (no YAML quoting) in compose ``command:`` order.
    """
    primary = next(
        (m for m in SUPPORTED_MODELS if m.role_hint == "primary" and m.speculative_config),
        None,
    )
    spec_item = (
        speculative_config_item(primary) if primary else '--speculative-config={"method": "..."}'
    )
    return [
        spec_item,
        "--trust-remote-code",
        "--language-model-only",
        f"--tokenizer={MTP_TOKENIZER_OVERRIDE}",
    ]
