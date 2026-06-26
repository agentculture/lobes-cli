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


@dataclass(frozen=True)
class SupportedModel:
    """One model the fleet/CLI can serve — a gear you can change to."""

    id: str  # OpenAI model id (== the vLLM --served-model-name)
    role_hint: str  # "primary" | "fallback" | "candidate" (the fleet's default role)
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
        context="128K native",
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
    spec = primary.speculative_config if primary else '{"method": "..."}'
    return [
        f"--speculative-config={spec}",
        "--trust-remote-code",
        "--language-model-only",
        f"--tokenizer={MTP_TOKENIZER_OVERRIDE}",
    ]
