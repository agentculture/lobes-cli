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

# Shared ``context`` literals — several catalog entries share these exact
# native context windows; a single constant keeps them from drifting
# independently (SonarCloud: duplicated string literal). Same rationale for
# the Gemma 4 unified-multimodal ``shape`` literal shared by the 12B pair and
# the 31B muse gear.
_CONTEXT_32K_NATIVE = "32K native"
_CONTEXT_128K_NATIVE = "128K native"
_CONTEXT_256K_NATIVE = "256K native"

# Self-hosted MTP draft via vLLM's generic "mtp" method (no external draft
# repo) — shared by every checkpoint that bakes its own mtp.* module in.
_MTP_SELF_HOSTED_N2 = '{"method": "mtp", "num_speculative_tokens": 2}'
_SHAPE_GEMMA4_UNIFIED = "unified multimodal (text+image+audio)"


@dataclass(frozen=True)
class SupportedModel:
    """One model the fleet/CLI can serve — a gear you can change to."""

    id: str  # OpenAI model id (== the vLLM --served-model-name)
    # The fleet's default role for this gear. One of:
    # "primary" | "fallback" | "candidate" | "minor" | "multimodal" | "muse" |
    # "embedding" | "reranker".
    # The generate-lane tier aliases (main/minor/multimodal/muse + back-compat
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
    # Per-model pooling budget override. 0.0 = "use the shared embed/score default"
    # (0.06, sized for the ~0.6B gears). A LARGER pooling model must declare its own,
    # or `lobes switch` would hand it a budget its weights do not fit in — measured:
    # the 4B's weights alone are 7.56 GiB, while 0.06 x 121.69 GiB = 7.30 GiB, so the
    # shared default cannot load it at all. See docs/qwen3-embedding-4b.md.
    default_gpu_mem_util: float = 0.0


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
        context=_CONTEXT_256K_NATIVE,
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
        # DEMOTED from fleet default primary 2026-07-31 (operator-confirmed),
        # replaced by unsloth/Qwen3.6-27B-NVFP4 above — same 27B family and the
        # same Qwen3_5ForConditionalGeneration arch, but MULTIMODAL (its export
        # keeps the ViT this one dropped) and with a self-hosted MTP draft
        # instead of a grafted one. Kept, not deleted (cite-don't-delete): it is
        # the last text-only 27B in the catalog, so it remains the pick for a
        # deployment that wants the smaller weight footprint and no vision, and
        # it is the checkpoint every pre-0.54.9 evidence transcript was measured
        # against. Selectable via `lobes switch`.
        role_hint="candidate",
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
        id="unsloth/Qwen3.6-27B-NVFP4",
        # CANDIDATE for a MULTIMODAL cortex — the same-family 27B sibling of the
        # `worker` gear below (unsloth/Qwen3.6-35B-A3B-NVFP4), from the SAME
        # publisher with the SAME export recipe. If promoted it would replace
        # the text-only sakamakismile/…-Text-NVFP4-MTP primary above and give
        # the fleet's reasoning/final-authority lobe image AND video intake for
        # the first time.
        #
        # STATUS: untested. Everything below is read off the checkpoint's own
        # published config files (fetched 2026-07-31) — NOT from a boot. No
        # gpu_mem_util or max_model_len is declared anywhere for it, because on
        # a unified-memory card those are MEASURED truths, not arithmetic (the
        # rule thor-muse's refused 0.40 and thor-worker's accepted 0.45 both
        # established). Promotion to role_hint="primary" is gated on a live
        # GB10 boot + an evidence transcript under docs/evidence/.
        #
        # Verified against the ACTUAL config files, not card prose:
        #   https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4/resolve/main/config.json
        #     - architectures: ["Qwen3_5ForConditionalGeneration"], model_type
        #       "qwen3_5" — the SAME arch the current text-only primary serves,
        #       so this is a checkpoint swap, not an engine-support question.
        #     - text_config.max_position_embeddings = 262144 (native 256K),
        #       num_hidden_layers = 64, hybrid linear-attn (linear_key_head_dim
        #       etc.) — matching the incumbent primary.
        #     - language_model_only = FALSE, with a vision_config (27-layer ViT,
        #       hidden 1152), image_token_id=248056, video_token_id=248057 and
        #       vision_start/end (248053/248054). NO audio_config. So a promoted
        #       lane must NOT pass --language-model-only (unlike the incumbent,
        #       whose export physically removed the ViT).
        #     - mtp_num_hidden_layers = 1 AND a top-level "unsloth_fixed_mtp"
        #       flag; the safetensors index carries 15 real `mtp.*` tensors
        #       (mtp.fc.weight, mtp.layers.0.*), i.e. the draft module is
        #       physically present, self-hosted, no external draft repo. The
        #       incumbent primary needed its draft GRAFTED back on because the
        #       baseline NVFP4 export dropped it (0% acceptance); this export
        #       never lost it.
        #     - quantization_config.format "mixed-precision": 8-bit
        #       float-quantized for attention/lm_head/upper-8 MLP layers, 4-bit
        #       nvfp4-pack-quantized for the MLP gate/up/down, with the ViT
        #       (model.visual.*) and every linear_attn left UNQUANTIZED — 303
        #       ignore patterns. quant_method is compressed-tensors, NOT nvidia
        #       modelopt, so a promoted lane needs
        #       --quantization=compressed-tensors, not the incumbent's
        #       `modelopt`.
        #   model.safetensors.index.json: 1968 tensors / 23.42 GB across 5 shards.
        #   chat_template.jinja: CONTAINS the `preserve_thinking` variable, so
        #     issue #93's --default-chat-template-kwargs flag keeps working
        #     across the swap. It also ships its own tokenizer, so the
        #     incumbent's --tokenizer=mmangkad/Qwen3.6-27B-NVFP4 override (a
        #     workaround for a TokenizersBackend declaration absent from the
        #     image) would be DROPPED.
        #   License apache-2.0.
        # PROMOTED to fleet default primary 2026-07-31 (operator-confirmed), after
        # the live GB10 boot below. Replaces sakamakismile/…-Text-NVFP4-MTP, which
        # is demoted to `candidate` and kept (cite-don't-delete).
        #
        # DEMOTED from fleet default primary 2026-08-19 (operator-confirmed,
        # qwen3.8-cortex-upgrade plan t3), replaced by unsloth/Qwen3.8-27B-NVFP4
        # below — the next-generation checkpoint from the SAME publisher and
        # export recipe (Qwen3_5ForConditionalGeneration → the 3.8 line), with
        # the same self-hosted MTP draft and multimodal ViT this entry has, plus
        # a config-verified 1M-token YaRN reach this entry's card never claimed.
        # Kept, not deleted (cite-don't-delete): it is the checkpoint every
        # 2026-07-31..2026-08-19 evidence transcript was measured against, and it
        # remains selectable via `lobes switch`.
        role_hint="candidate",
        shape="hybrid Mamba/linear-attn + ViT (text+image+video, self-hosted MTP draft)",
        context="256K native (served at full 256K on the spark-lobe shape)",
        native_max_model_len=262144,
        tool_parser="qwen3_coder",
        quantization="compressed-tensors",
        # MEASURED live on the DGX Spark GB10, 2026-07-31 — see
        # docs/evidence/2026-07-31-accept-multimodal-cortex-spark.txt.
        # Booted FIRST TRY at the spark-lobe shape's own 0.44 / 262144 (no retune,
        # unlike thor-muse's refused 0.40): KV pool 26.39 GiB / 756,642 tokens at
        # the full 256K window, vs the outgoing text-only primary's 888,946 at the
        # same knobs — so the unquantized bf16 ViT costs ~132,300 tokens (~15%) of
        # KV pool and the full window survives with embedder/reranker/embed-deep
        # still co-resident. (KV pool / max_model_len = a ~2.89x CEILING vs the
        # outgoing 3.39x — that is how many full-context requests the cache could
        # HOLD, not measured concurrency. Real saturation is unmeasured for this
        # lane; on the worker lane two consumers measured saturation near width
        # 8-9 against a 14.07x ceiling. Never multiply a single-stream tok/s by a
        # ceiling.)
        # Decode 14.9 / 16.4 / 19.0 tok/s SINGLE-STREAM (short/medium/512-tok gen) at TTFT
        # ~0.27s; self-hosted MTP engages at 62-67% draft acceptance, mean
        # acceptance length 2.24-2.35.
        # Gates all pass: image (colour + negative control), VIDEO (directional
        # motion with the reversed clip as control), thinking (4,195-char trace —
        # note the field is `reasoning`, NOT `reasoning_content` on this build),
        # preserve_thinking #93 (+800-token two-turn delta), and strict tool
        # calling with thinking ON (colleague#320) — clean structured call.
        # CAVEAT: the throughput comparison against the outgoing primary is NOT
        # controlled (its numbers came from a different vLLM build). See
        # docs/model-switch-playbook.md.
        status="load-tested",
        doc="qwen3.6-27b-nvfp4-multimodal.md",
        # Self-hosted draft (no external "model" key), mirroring the 35B-A3B
        # worker's own README serve command. UNMEASURED on this checkpoint:
        # the 35B twin reached 89.1% acceptance at 2 tokens, but that is the
        # sibling's number, not this one's.
        speculative_config=_MTP_SELF_HOSTED_N2,
        task="generate",
    ),
    SupportedModel(
        id="unsloth/Qwen3.8-27B-NVFP4",
        # Fleet default PRIMARY since 2026-08-19 (operator-confirmed,
        # qwen3.8-cortex-upgrade plan t3), replacing unsloth/Qwen3.6-27B-NVFP4
        # above (demoted to `candidate`, kept per cite-don't-delete).
        #
        # Every fact below is read off the checkpoint's own published config
        # files (fetched 2026-08-19) — NOT from a boot. Per the thor-muse /
        # thor-worker rule, gpu_mem_util and max_model_len for the co-resident
        # fleet shapes are MEASURED truths, not arithmetic, so this entry does
        # not declare them beyond the checkpoint's own native ceiling; a live
        # GB10 boot + evidence transcript (plan tasks t7/t8/t9) supplies the
        # rest, and this comment should be re-verified against that transcript
        # once it lands.
        #
        # Verified against the ACTUAL config files, not card prose:
        #   https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4/resolve/main/config.json
        #     - architectures: ["Qwen3_5ForConditionalGeneration"], model_type
        #       "qwen3_5" — the SAME arch the outgoing 3.6 primary serves, so
        #       this is a checkpoint swap within the same engine-support
        #       family, not a new-architecture bring-up.
        #     - text_config.max_position_embeddings = 262144 (native 256K),
        #       num_hidden_layers = 64 — matching the outgoing primary's
        #       native window; the checkpoint card's 1M-token reach is via a
        #       YaRN --hf-overrides on text_config.rope_parameters, not a
        #       larger native ceiling, so native_max_model_len stays 262144
        #       here and the 1M knobs are wired at the profile/template layer
        #       (plan task t5), not in this catalog entry.
        #     - vision_config is present (ViT) with image_token_id and
        #       video_token_id fields — MULTIMODAL, so a served lane must NOT
        #       pass --language-model-only, same as the outgoing primary.
        #     - mtp_num_hidden_layers = 1 — a self-hosted MTP draft module is
        #       physically present in the checkpoint (own baked-in draft
        #       weights), no external draft repo needed, mirroring the
        #       outgoing primary's own self-hosted draft.
        #     - quantization_config.quant_method "compressed-tensors",
        #       format "mixed-precision" — fp8 attention/lm_head + nvfp4 MLP,
        #       ViT left unquantized. Same quantization family the outgoing
        #       primary uses, so --quantization=compressed-tensors carries
        #       forward unchanged.
        #   ~23.4 GB checkpoint size. License apache-2.0.
        #   tokenizer.json: "truncation": null on the fixed revision fetched
        #     2026-08-19 — an EARLIER revision of this checkpoint hardcoded
        #     "max_length": 2048 in its tokenizer.json, which would silently
        #     truncate every long-context request; re-verify this field after
        #     any re-download, it is not guaranteed to stay null across
        #     revisions.
        #   chat_template.jinja: CONTAINS the `preserve_thinking` variable, so
        #     issue #93's --default-chat-template-kwargs flag keeps working
        #     across the swap. It ships its own tokenizer and its own ViT —
        #     no --tokenizer override, no --language-model-only, same as the
        #     outgoing primary.
        role_hint="primary",
        shape="hybrid Mamba/linear-attn + ViT (text+image+video, self-hosted MTP draft)",
        context=_CONTEXT_256K_NATIVE,
        native_max_model_len=262144,
        tool_parser="qwen3_coder",
        quantization="compressed-tensors",
        # MEASURED live on the DGX Spark GB10, 2026-08-19 — see
        # docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt: boots at
        # 0.58/1048576 (1M via YaRN hf-overrides; 0.60 refused twice, the
        # opt-in embed-deep gear reclaimed), KV pool 42.07 GiB = 1,271,476
        # tokens = 1.21x ceiling at full 1M, decode 19.9-24.0 tok/s
        # single-stream, MTP 54-61% acceptance at n=2, 328K-token needle
        # retrieval beyond the native ceiling, image + strict-tools gates
        # pass on vLLM 0.26.1rc1.dev942.
        status="load-tested",
        doc="qwen3.8-27b-nvfp4.md",
        # Self-hosted draft (no external "model" key) via the generic "mtp"
        # method — validated live 2026-08-19 (54-61% draft acceptance at
        # n=2 on the GB10; the outgoing 3.6 needed the qwen3_5_mtp key on
        # the 0.23 engine, the 0.26 engine takes "mtp" directly).
        speculative_config=_MTP_SELF_HOSTED_N2,
        task="generate",
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
        context=_CONTEXT_32K_NATIVE,
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
        id="Qwen/Qwen3-Embedding-4B",
        # The "deep" embedding slot: the higher-fidelity companion to the 0.6B hot-path
        # gear above, wired as the opt-in `embed-deep` backend (gateway alias
        # "embed-deep", COMPOSE_PROFILES=embed-deep). 2560-dim Matryoshka, MTEB
        # multilingual mean 69.45 vs the 0.6B's ~64.3 — bought with ~8 GiB of weights
        # and a much slower forward pass, which is why it is opt-in and NOT the
        # embedder role's default.
        #
        # role_hint is "candidate", NOT "embedding": roles.ROLE_ROLE_HINT maps the
        # `embedder` role to role_hint "embedding" and _catalog_by_role_hint takes the
        # FIRST match, so a second "embedding" entry would silently hijack the role's
        # reported model. The deep slot is a switchable gear, not a role default.
        #
        # NON-INTEROPERABLE with the 0.6B: embeddings from the two models live in
        # different vector spaces, so a corpus indexed by one can only be queried by
        # the same one. Truncating this model to 1024 dims via Matryoshka does NOT
        # make it compatible with the 0.6B's 1024. See docs/qwen3-embedding-4b.md.
        role_hint="candidate",
        shape="dense embedding (text)",
        context=_CONTEXT_32K_NATIVE,
        native_max_model_len=32768,
        tool_parser="",
        quantization="",
        # GB10 2026-07-20: serves 2560 dim, matryoshka ladder honoured at all 6 probed
        # points, paraphrase probe 0.74 vs 0.28 unrelated, boots at util 0.11 co-resident
        # with the full spark-lobe fleet (weights 7.56 GiB, KV 11.34 GiB / 82,592 tokens),
        # 42.4 ms median vs the 0.6B's 11.5 ms. sm_110 remains UNVALIDATED — see the doc.
        status="load-tested",
        doc="qwen3-embedding-4b.md",
        task="embed",
        dimension=2560,
        # MEASURED on the GB10 2026-07-20 — the shared 0.06 pooling default is
        # SMALLER than this model's weights (7.56 GiB vs a 7.30 GiB budget).
        default_gpu_mem_util=0.11,
        hf_overrides=(
            '{"is_matryoshka": true,'
            ' "matryoshka_dimensions": [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560]}'
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
        id="LiquidAI/LFM2.5-1.2B-Instruct",
        # The `hand` lobe — the fleet's NINTH Colleague role and its designated
        # fine-tuning base (issue #81 role set; the hand-lobe spec/plan under
        # docs/specs + docs/plans). "Muscle memory": one cheap base, many LoRA
        # adapters, each mastering a domain. This is the gear the `minor`/`cheap`
        # tier now resolves to — it REPLACES Qwen/Qwen3.5-4B below in that slot
        # (which stays in the catalog as a plain candidate, cite-don't-delete).
        #
        # Architecture: LFM2.5 is a short-conv + GQA hybrid (Lfm2ForCausalLM),
        # registered in vLLM >= 0.23.0. Verified live against the exact pinned
        # nightly digest on a physical Jetson AGX Thor (2026-08-10):
        # VLLM_VERSION 0.23.1rc1.dev672+g93d8f834d, LFM2_REGISTERED True.
        #
        # quantization="none" is the bf16/unquantized sentinel — VLLM_QUANTIZATION
        # is NOT written on switch, and the `vllm-hand` fleet lane omits the
        # --quantization flag entirely rather than passing it empty. At ~1.2B
        # params (~2.4 GiB bf16) that is the point: cheap enough to co-reside on
        # every card in the mesh.
        #
        # TEXT-ONLY: no ViT, so the lane carries no --language-model-only (there is
        # nothing to switch off) and `hand` advertises neither image_understanding
        # nor video_understanding. No thinking mode either — LiquidAI ships
        # LFM2.5-1.2B-Thinking as a SEPARATE checkpoint — so unlike the cortex
        # (qwen3) and Gemma 4 (gemma4) lanes this one needs NO --reasoning-parser.
        #
        # Tool calls use LFM2's own <|tool_call_start|>/<|tool_call_end|> delimiters,
        # which are SPECIAL TOKENS — the same trap that made `pythonic` silently
        # wrong for Gemma 4. vLLM ships a purpose-built "lfm2" parser
        # (vllm/tool_parsers/lfm2_tool_parser.py, registered as "lfm2"); its
        # __init__ resolves both delimiters via self.vocab.get() and RAISES if
        # either is missing, so a tokenizer revision without them fails loudly at
        # startup instead of degrading to prose. See docs/lfm2.5-1.2b-hand.md.
        role_hint="hand",
        shape="hybrid short-conv + GQA (text-only)",
        context="32K native",
        native_max_model_len=32768,
        tool_parser="lfm2",
        quantization="none",
        status="configured",
        doc="lfm2.5-1.2b-hand.md",
        task="generate",
    ),
    SupportedModel(
        id="Qwen/Qwen3.5-4B",
        # DEMOTED to a plain candidate (cite-don't-delete) when the `hand` lobe
        # above took over the minor/cheap tier and the LoRA-base duty. It was the
        # fleet's first LoRA target and "minor" small-brain companion to the 27B
        # primary; nothing about the checkpoint changed, only which gear the tier
        # resolves to. Still selectable via `lobes switch`. Multimodal
        # (hybrid linear-attn + ViT) — serve text-only via --language-model-only.
        # Built-in MTP head not used in v1 (no speculative_config carried).
        # quantization="none" is the bf16/unquantized sentinel — VLLM_QUANTIZATION
        # is NOT written on switch; the operator must REMOVE the --quantization
        # flag from the compose command: by hand (the single-model template defaults
        # to --quantization=modelopt when VLLM_QUANTIZATION is absent, which would
        # corrupt bf16 weights). See docs/qwen3.5-4b-minor.md.
        role_hint="candidate",
        shape="hybrid linear-attn + ViT (multimodal)",
        context=_CONTEXT_256K_NATIVE,
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
        # table. Tool calls use Gemma 4's native `<|tool_call>call:name{...}` syntax
        # via the purpose-built "gemma4" parser (matches runtime._parser.infer_parser,
        # which returns "gemma4" for gemma-4* ids). This was "pythonic" until the
        # 2026-07-17 live check proved that parser cannot see Gemma 4's
        # special-token delimiters; see the _parser.py rule for the evidence.
        # UNVALIDATED on THIS 12B checkpoint (#108) — it inherits the family rule.
        #
        # Content-correctness, live on THIS checkpoint via model=multimodal: image+text
        # VERIFIED against ground truth with a negative control (replies "Red"/"Blue"
        # for red/blue test images; a blue image correctly fails a "red" assertion).
        # audio+text is NOT served — vLLM's gemma4_unified drops the input_audio
        # content part (adds ~19 placeholder tokens, no content) instead of rejecting
        # it, so a caller gets 200 OK and a fluent answer that ignored the audio. This
        # is a vLLM gap, not a checkpoint gap (config.json declares audio_config /
        # audio_token_id). Tracked as #101. See docs/gemma-4-12b-nvfp4.md
        # #live-validation-status-71 for the full evidence table.
        role_hint="multimodal",
        shape=_SHAPE_GEMMA4_UNIFIED,
        # Same base-model family as the coder entry — text_config.max_position_
        # embeddings=131072 confirmed for the Unified 12B IT line (#71); not
        # independently re-measured for this exact NVFP4A16 export.
        context=_CONTEXT_128K_NATIVE,
        native_max_model_len=131072,
        tool_parser="gemma4",
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
        # calls use Gemma 4's native `<|tool_call>call:name{...}` syntax via the
        # purpose-built "gemma4" parser (matches runtime._parser.infer_parser, which
        # returns "gemma4" for gemma-4* ids). Was "pythonic" until the 2026-07-17
        # live check disproved it; see the _parser.py rule. UNVALIDATED on THIS
        # coder checkpoint (#108) — it inherits the family rule.
        #
        # status="load-tested". Serve-enablement RESOLVED on the Spark GB10 (#71/#73,
        # 2026-07-01): the gear SERVES on the custom image (Dockerfile.vllm-gemma4 =
        # vllm/vllm-openai nightly, vLLM 0.23.1rc1 + the vllm[audio] extra) via vLLM's
        # NATIVE Gemma4UnifiedForConditionalGeneration class, which handles the
        # heterogeneous per-layer head sizes (40 sliding@256 + 8 full@512) that broke
        # released vLLM <=0.22.1 (transformers-backend fallback → o_proj marlin_gemm
        # 4096≠8192; a backend flag does NOT fix it — the native class does). Validated
        # live: text ✓, image+text ✓. The original "audio+text ✓ (transcribed a TTS
        # clip verbatim)" note was never actually verified against ground truth — the
        # check behind it asserted only HTTP 200 + non-empty content against a
        # placeholder clip. When tested properly against the base checkpoint (see the
        # coolthor entry below), audio+text did NOT hold: vLLM's gemma4_unified drops
        # the input_audio content part rather than serving it. Tracked as #101.
        # ~15.7 GiB footprint ≈ 0.12 budget. See docs/gemma-4-12b-nvfp4.md and #71.
        role_hint="candidate",
        shape=_SHAPE_GEMMA4_UNIFIED,
        # Native context confirmed 128K (text_config.max_position_embeddings=131072,
        # read from the checkpoint config during #71 live validation).
        context=_CONTEXT_128K_NATIVE,
        native_max_model_len=131072,
        tool_parser="gemma4",
        # This checkpoint is NVFP4 in compressed-tensors format (config.json
        # quant_method="compressed-tensors", format "nvfp4-pack-quantized") — NOT
        # nvidia modelopt. vLLM must be told --quantization=compressed-tensors;
        # passing modelopt_fp4 fails with a quant-method-mismatch (verified #71).
        quantization="compressed-tensors",
        status="load-tested",  # GB10 2026-07-01: text+image ✓; audio+text NOT served (#101)
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
        id="unsloth/gemma-4-12B-it-qat-w4a16",
        # Second candidate senses/multimodal checkpoint for the Gemma 4 12B
        # UNIFIED family (unsloth-qat-senses-first-class-orin-variation plan,
        # t1) — QAT (quantization-aware trained), finetuned from
        # google/gemma-4-12B-it, compressed-tensors serialized explicitly for
        # vLLM. A later task in the same plan wires a first-class Orin card
        # profile/shape to boot this as its `senses` gear; here it is only
        # cataloged as a switchable gear, NOT yet booted.
        #
        # Verified against the checkpoint's ACTUAL config.json (fetched
        # unauthenticated 2026-08-04 — the repo is not gated), NOT card prose:
        #   https://huggingface.co/unsloth/gemma-4-12B-it-qat-w4a16/resolve/main/config.json
        #     - architectures: ["Gemma4UnifiedForConditionalGeneration"],
        #       model_type "gemma4_unified" — the SAME class the coolthor
        #       gear above serves (this repo's existing
        #       Dockerfile.vllm-gemma4 image already handles it).
        #     - text_config.max_position_embeddings = 262144 (256K) — DOUBLE
        #       the coolthor incumbent's 131072. The HF card summary also
        #       claims 256K, but per this repo's #108 discipline the card is
        #       the ATTEMPT target, not the evidence — config.json is what is
        #       cited here, and the number still needs a live boot to prove
        #       it actually serves at that window (a later plan task does
        #       that boot).
        #     - quantization_config: quant_method "compressed-tensors",
        #       format "pack-quantized", num_bits=4, strategy "group",
        #       group_size=32, symmetric=true — i.e. INT4 weight-only
        #       (W4A16), NOT the coolthor/coder gears' FP4
        #       ("nvfp4-pack-quantized"). Both resolve to the same vLLM
        #       --quantization=compressed-tensors flag (vLLM reads the exact
        #       scheme off the checkpoint's own config), but the KERNEL PATH
        #       differs — int4 pack-quantized vs NVFP4 — and is UNPROVEN on
        #       any of this fleet's hardware until a live boot exercises it.
        #       (int4-weight-only is what makes an Ampere sm_87 target
        #       plausible at all — no Blackwell FP4 tensor cores needed for
        #       16-bit activations — but plausible is not proven.)
        #     - vision_config (gemma4_unified_vision) AND audio_config
        #       (gemma4_unified_audio) both present; image_token_id=258880,
        #       audio_token_id=258881, video_token_id=258884 — VIDEO is
        #       natively declared here, which this repo's existing
        #       "text+image+audio" Gemma capability shape
        #       (_SHAPE_GEMMA4_UNIFIED) predates and does not mention.
        #     - no mtp_num_hidden_layers / draft-head field anywhere in the
        #       config — unlike the coolthor gear's wired native MTP
        #       (google/gemma-4-12B-it-assistant draft), this checkpoint
        #       ships no draft head, so speculative_config stays empty.
        #
        # role_hint is DELIBERATELY "candidate", not the literal
        # role_hint="multimodal" the covering plan's requirement text names:
        # test_exactly_one_gemma_multimodal_gear (tests/test_catalog.py) pins
        # role_hint="multimodal" to EXACTLY [coolthor] — a second entry with
        # that role_hint would make resolve_tier("multimodal")/("senses")/
        # ("normal") ambiguous by first-match. This mirrors the exact
        # reasoning the sakamakismile coder entry above already uses
        # (role_hint="candidate", not "multimodal", for the identical
        # singular-tier-owner reason). Keeping this entry a candidate leaves
        # the fleet-wide tier default — and every thor/spark profile that
        # pins the raw coolthor id — untouched, consistent with the covering
        # plan's own "thor/spark render byte-identical" requirement. A
        # per-box operator profile (not this catalog field) is how a
        # deployment actually selects this checkpoint for its `senses` role.
        #
        # STATUS: "configured" — nothing below is live-probed. No boot, no
        # gpu_mem_util, no measured KV pool; image/video/audio/reasoning/
        # tool-calling are all PENDING LIVE PROBE (a later plan task runs
        # them against real hardware; see docs/gemma-4-12b-qat-w4a16.md). The
        # one already-filed risk worth flagging up front: issue #101 found
        # vLLM's gemma4_unified silently DROPS input_audio content (200 OK,
        # fluent reply that ignored the audio) on the coolthor checkpoint —
        # config.json declaring audio_config here is NOT evidence this
        # checkpoint's audio will be served; #101 is a vLLM-path gap, not a
        # per-checkpoint one, so it is expected to reproduce until re-probed.
        role_hint="candidate",
        shape="unified multimodal (text+image+audio+video declared, QAT int4 W4A16)",
        context=_CONTEXT_256K_NATIVE,
        native_max_model_len=262144,
        tool_parser="gemma4",
        quantization="compressed-tensors",
        status="configured",  # un-booted candidate — see docs/gemma-4-12b-qat-w4a16.md
        doc="gemma-4-12b-qat-w4a16.md",
        task="generate",
    ),
    SupportedModel(
        id="nvidia/Gemma-4-31B-IT-NVFP4",
        # Gemma 4 31B IT (Google DeepMind), NVIDIA's official NVFP4 export — the
        # `muse` gear: the fleet's OPT-IN creative/ideation generate lobe (the
        # seventh Colleague role). NVIDIA ships only the 31B + 26B-A4B Gemma 4
        # sizes in NVFP4 (the 12B `senses` gear is a community export) — this is
        # the 31B. PLAIN gemma4 line (model_type "gemma4",
        # Gemma4ForConditionalGeneration), NOT the Unified 12B family — but the
        # checkpoint still declares vision_config + audio_config with
        # image/audio token ids, i.e. multimodal intake like `senses`; the same
        # vLLM audio gap (#101) is assumed to apply until measured. Weights are
        # 30.4 GiB across 4 safetensors shards; config.json quant_method is
        # "modelopt" (hf_quant_config.json: NVFP4, FP8 KV-cache scheme with
        # calibrated scales — unlike the Qwen MTP re-export on Thor, #109).
        # Tool calls use the "gemma4" parser (infer_parser: gemma-4* ids) — Gemma 4
        # emits native `<|tool_call>call:name{...}<tool_call|>`, whose delimiters are
        # SPECIAL TOKENS that only Gemma4EngineToolParser (skip_special_tokens=False)
        # can see. VALIDATED live on this checkpoint, 2026-07-17 on a physical Thor:
        # under the old "pythonic" value the delimiters were stripped and the call
        # was relayed as content (tool_calls=null); under "gemma4" it parses into a
        # real tool_calls array. Evidence:
        # docs/evidence/2026-07-17-accept-muse-tool-calling-thor.txt.
        #
        # Too heavy to co-reside with the cortex+senses duo on a 128 GB box —
        # machine-as-brain never hosts it; a muse-hosting deployment shape
        # (`lobes init --shape thor-muse`) is the only built-in way to serve it.
        role_hint="muse",
        shape=_SHAPE_GEMMA4_UNIFIED,
        # text_config.max_position_embeddings=262144 (read from the checkpoint
        # config, 2026-07-17); the thor-muse shape serves the FULL native
        # window (262144 — operator decision, no box-budget trim).
        context=_CONTEXT_256K_NATIVE,
        native_max_model_len=262144,
        tool_parser="gemma4",
        # NVIDIA modelopt NVFP4 (config.json quant_method="modelopt" — resolves
        # to modelopt_fp4), NOT compressed-tensors like the community 12B export.
        quantization="modelopt",
        status="configured",  # declared 2026-07-17; first live boot pending (Thor)
        doc="gemma-4-31b-nvfp4.md",
        task="generate",
        # Native MTP via the public plain-line assistant draft
        # (gemma4_assistant family — vLLM's hf_config_override normalizes it to
        # gemma4_mtp with forced n_predict=1; see docs/gemma4-mtp-draft.md's
        # family table). Same "model" key shape as the 12B entry. DECLARED, not
        # yet measured on this 31B target — the first acceptance run gates it.
        speculative_config=(
            '{"method": "mtp", "model": "google/gemma-4-31B-it-assistant",'
            ' "num_speculative_tokens": 1}'
        ),
    ),
    SupportedModel(
        id="unsloth/Qwen3.6-35B-A3B-NVFP4",
        # Former `worker` gear (thor-worker-lobe plan, t1; VALIDATED live on the
        # physical Jetson AGX Thor sm_110, 2026-07-31 — see
        # docs/evidence/2026-07-31-accept-worker-thor.txt). DEMOTED from
        # role_hint="worker" to a kept candidate (nemotron-lightning-worker
        # plan, #187, t3): the worker seat moves to
        # nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 below — a
        # text-only, non-coding doer with a 1M native ceiling — per the
        # operator decision recorded in
        # docs/plans/2026-08-20-nemotron-lightning-worker.md. Kept, not
        # deleted (cite-don't-delete): this is the checkpoint every
        # 2026-07-31..2026-08-20 worker-lane evidence transcript was measured
        # against, and it stays selectable via `lobes switch`. Nothing below
        # this comment changed — same fields, same facts, only role_hint moved.
        #
        # A DISTINCT entry from the mmangkad/Qwen3.6-35B-A3B-NVFP4 "candidate"
        # above (same architecture family, different org/export). Unlike that
        # copy, this unsloth export ships its OWN MTP draft module baked into
        # the checkpoint (never dropped one to begin with), so it "can act as
        # its own speculative draft for faster decoding" (some throughput
        # tradeoff vs plain inference — the card's own words).
        #
        # Verified against the checkpoint's ACTUAL config files (fetched
        # 2026-07-31), not card prose:
        #   https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/resolve/main/config.json
        #     - architectures: ["Qwen3_5MoeForConditionalGeneration"],
        #       model_type "qwen3_5_moe"; text_config.max_position_embeddings
        #       = 262144 (native); num_experts=256, num_experts_per_tok=8
        #       (MoE, ~3B active/token); text_config.mtp_num_hidden_layers=1,
        #       and quantization_config.ignore carries a "re:^mtp.*" pattern —
        #       i.e. the checkpoint's own MTP weight tensors physically exist
        #       and are deliberately left UNQUANTIZED, confirming the
        #       self-hosted draft module the card describes.
        #     - quantization_config.quant_method="compressed-tensors" (mixed
        #       precision: 8-bit float-quantized for attention/lm_head/the
        #       upper 8 MLP layers, 4-bit nvfp4-pack-quantized for the MoE
        #       experts) — NOT nvidia modelopt, unlike the mmangkad/ candidate
        #       above; resolves directly to quantization="compressed-tensors"
        #       (same convention as the Gemma 4 12B gears' compressed-tensors
        #       entries — no modelopt_fp4 translation needed).
        #   https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4/resolve/main/hf_quant_config.json
        #     - 404 Not Found: compressed-tensors checkpoints carry their
        #       quant config inline in config.json; there is no separate
        #       hf_quant_config.json here (that file is a modelopt/TensorRT
        #       export convention, absent from this compressed-tensors one).
        #   The README's own vLLM MTP serve command matches the card claim
        #   verbatim: `--speculative-config '{"method": "mtp",
        #   "num_speculative_tokens": 2}'` — no external "model"/
        #   "draft_model_id" key, because the draft lives IN this checkpoint
        #   (see the config facts above), unlike the Gemma gears' native MTP
        #   (a separate public HF draft id). This id carries no "-MTP" suffix
        #   despite shipping its own draft weights — see
        #   tests/test_catalog.py's _SELF_HOSTED_MTP_WITHOUT_ID_MARKER for why
        #   the generic id/external-draft guard still allows it.
        #   README DGX Spark serving note (--moe-backend flashinfer_b12x under
        #   CUTE_DSL_ARCH=sm_121a; explicitly recommends AGAINST marlin here —
        #   "2x slower"): carried below as the best-cited default, but sm_121a
        #   is Spark's arch, not Thor's (sm_110 — see
        #   docs/machine-profiles.md); UNCONFIRMED on Thor until the live
        #   boot (plan task t7).
        role_hint="candidate",
        # MULTIMODAL (image+video) — config.json (fetched 2026-07-31) carries a
        # vision_config (27-layer ViT), image_token_id=248056,
        # video_token_id=248057, vision_start/end tokens (248053/248054), and
        # NO audio_config. Operator decision (2026-07-31): worker is served
        # MULTIMODAL — a "seeing doer" (image+video intake + repo_action) — so
        # the compose lane does NOT pass --language-model-only (unlike the 27B
        # cortex MTP primary, whose export dropped its ViT). VALIDATED live on
        # the physical Jetson AGX Thor (sm_110), 2026-07-31: vLLM serves
        # Qwen3_5MoeForConditionalGeneration + MTP together (MTP draft acceptance
        # 89.1%, ~50.8 tok/s decode), image AND video intake correct (image:
        # red/blue + negative control; video: a real webcam clip described
        # accurately), thinking/tool parsers work. See
        # docs/evidence/2026-07-31-accept-worker-thor.txt.
        shape="MoE (~3B active) + ViT (text+image+video)",
        context="256K native (→~1.01M via YaRN)",
        native_max_model_len=262144,
        tool_parser="qwen3_coder",
        quantization="compressed-tensors",
        status="load-tested",  # Thor sm_110 2026-07-31: boots+serves, MTP 89.1%, vision ✓
        doc="qwen3.6-35b-a3b-nvfp4.md",
        # moe_backend="" (auto-select) — NOT forced. Measured on Thor sm_110:
        # every forced NVFP4 MoE backend was refused (flashinfer_* lack sm_110
        # kernels; marlin/triton reject the mixed quantized-main/unquantized-MTP
        # experts). vLLM auto-selects a working kernel per path. `lobes switch`
        # therefore adds NO --moe-backend flag for this gear.
        moe_backend="",
        speculative_config=_MTP_SELF_HOSTED_N2,
        task="generate",
    ),
    SupportedModel(
        id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        # The NEW `worker` gear (nemotron-lightning-worker plan, #187, t3),
        # replacing unsloth/Qwen3.6-35B-A3B-NVFP4 above (demoted to
        # role_hint="candidate", kept — cite-don't-delete). A fast, TEXT-ONLY,
        # non-coding doer: action selection, tool loops, RAG, digestion, repo
        # inspection/navigation — never code authoring or the final decision
        # (roles.py's ROLE_RESPONSIBILITIES for `worker` is redefined to match
        # in a sibling task, t4; this entry only changes the catalog).
        #
        # UNGATED checkpoint (no HF license wall). Verified against the
        # checkpoint's ACTUAL config files, fetched 2026-08-20 — NOT card
        # prose:
        #   https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/resolve/main/config.json
        #     - architectures: ["NemotronHForCausalLM"], model_type
        #       "nemotron_h" — a DIFFERENT engine-support family from the
        #       outgoing Qwen3_5MoeForConditionalGeneration worker; nemotron_h
        #       serving on this repo's pinned vLLM nightly is UNPROVEN until
        #       the covering plan's t1/t2 spikes land (sm_110 SASS/PTX +
        #       standalone serve probe).
        #     - max_position_embeddings = 1048576 (1M native ceiling) — a
        #       config-verified YaRN reach, not card-prose. `lobes switch`
        #       clamps a machine default DOWN to this only if it is smaller;
        #       nothing in this catalog or any shape allocates the full 1M by
        #       default (that is a live-boot decision, see the plan's t2/t8).
        #     - 52 hidden layers; hybrid Mamba-2 state-space + sparse-MoE +
        #       selective-attention layers (128 routed experts, 1 shared
        #       expert, 6 experts/token — ~3B active of 30B total, matching
        #       the card's own "30B/3B active" framing).
        #     - NO vision_config anywhere in the file — this checkpoint is
        #       TEXT-ONLY, unlike the outgoing worker's ViT (image+video)
        #       intake. `worker` therefore LOSES image_understanding/
        #       video_understanding on this swap (roles.py t4 tracks the
        #       responsibilities-vocabulary change).
        #     - No mtp_num_hidden_layers / draft-head field and no
        #       speculative-decoding field in config.json — so
        #       speculative_config stays empty (the honest sentinel) even
        #       though the model card separately advertises MTP/DSpark
        #       support; that support is DECLARED by the card, UNMEASURED by
        #       us (plan t2 evaluates MTP/DSpark separately from plain
        #       decode).
        #   https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/resolve/main/hf_quant_config.json
        #     - producer.name="modelopt" (version 0.44.0rc5); quant_algo
        #       "MIXED_PRECISION": FP8 on attention/lm_head-style projections,
        #       W4A16_NVFP4 (group_size=16) on the routed-expert up/down
        #       projections; kv_cache_quant_algo="FP8". This is the SAME
        #       nvidia-modelopt family as the muse 31B gear and the outgoing
        #       27B MTP primary (`quantization="modelopt"`), NOT the
        #       compressed-tensors format the outgoing worker used — do not
        #       carry that forward.
        #   Model card (fetched 2026-08-20, prose — cited separately from the
        #   config facts above): license OpenMDW-1.1; the card's OWN example
        #   vLLM serve command passes `--reasoning-parser nemotron_v3` (no
        #   catalog field for this — it is a serve-time flag, tracked in the
        #   covering plan/docs, not this dataclass) and
        #   `--tool-call-parser qwen3_coder` — i.e. the publisher itself
        #   asserts this checkpoint emits Qwen3-Coder-shaped tool calls
        #   despite not being a Qwen model. That claim is UNVALIDATED on our
        #   engine (this repo has been burned by silently-wrong parser pairs
        #   before — see the gemma4 `pythonic` history in CLAUDE.md) until the
        #   plan's t2 structured-tool_calls probe confirms it; carried here as
        #   the best-cited default because the alternative (an empty
        #   tool_parser) would fail this repo's own
        #   test_tool_parser_matches_infer_parser invariant. The card also
        #   suggests `--moe-backend marlin`, which is NOT carried here: the
        #   outgoing worker's own sm_110 history (`moe_backend=""` below)
        #   showed every forced NVFP4 MoE backend refused on this exact
        #   hardware family, so a forced marlin pick would be an unverified
        #   guess in the opposite direction from that lesson.
        #
        # STATUS (2026-08-20, deviation d1): VALIDATED live on the DGX Spark
        # GB10 — NOT on the Thor the covering plan originally targeted. On the
        # Spark (util 0.30, 65536 window, MTP off): weights 17.85 GiB, KV pool
        # 3,560,789 tokens (54.33x at 65K), 75.1 tok/s single-stream, ~75 ms
        # median streaming TTFT, structured tool_calls PASS through the
        # nemotron_v3 + qwen3_coder parser pair (which validates the card's
        # cross-family parser claim below). On the Thor (sm_110) it is a
        # NO-GO: the engine wedges indefinitely in "Warming up Mamba2 SSD
        # Triton kernels" on BOTH the fleet 8bd082 nightly and the upstream
        # v0.27.1 release image (idle-CPU hang, not slow JIT) — so `worker`
        # serves on the Spark and other boxes refer/proxy to it. Evidence:
        # docs/evidence/2026-08-20-accept-worker-hand-spark.txt,
        # docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt. See
        # docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md and
        # docs/plans/2026-08-20-nemotron-lightning-worker.md.
        role_hint="worker",
        shape="hybrid Mamba-2 + sparse-MoE (~3B active per token, text-only)",
        context="1M native (1,048,576 max_position_embeddings)",
        native_max_model_len=1048576,
        # VALIDATED live on the Spark 2026-08-20: structured tool_calls parse
        # through nemotron_v3 (reasoning) + qwen3_coder (tools), nothing
        # leaked into content. The card's cross-family parser claim held.
        tool_parser="qwen3_coder",
        quantization="modelopt",
        # Spark GB10 2026-08-20: 75.1 tok/s, tools ✓; Thor sm_110 NO-GO
        # (Mamba2 warmup wedge — see the STATUS comment above).
        status="load-tested",
        doc="nemotron-3.5-lightning-30b-a3b-nvfp4.md",
        # moe_backend="" (auto-select), mirroring the outgoing worker's own
        # hard-won sm_110 lesson (forced NVFP4 MoE backends were refused on
        # this exact hardware family) rather than carrying the card's
        # untested "marlin" suggestion forward.
        moe_backend="",
        # No speculative_config: config.json carries no MTP/draft-head field
        # (see the long comment above) — the card's MTP/DSpark claim is
        # declared, UNMEASURED, and evaluated separately by plan task t2.
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
        context=_CONTEXT_32K_NATIVE,
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
#:
#: Capability-ROLE names (new vocabulary layered over the EXISTING backend roles
#: — no internal service/env/container is renamed):
#:   cortex → primary    (== main — the "thinking" primary backend)
#:   senses → multimodal (== multimodal — the vision+audio backend)
#:   worker → worker     (unsloth Qwen3.6-35B-A3B-NVFP4 MoE lobe — role IS the
#:                        backend name; opt-in, hosted only by a
#:                        worker-hosting deployment shape — thor-worker-lobe
#:                        plan)
#:   muse   → muse       (Gemma 4 31B creative/ideation lobe — role IS the
#:                        backend name; opt-in, hosted only by a muse-hosting
#:                        deployment shape)
#:   hand   → hand       (LiquidAI LFM2.5-1.2B fine-tuning base — role IS the
#:                        backend name; default-hosted on every card, and the
#:                        gear the minor/cheap tier resolves to since the hand
#:                        lobe replaced Qwen3.5-4B in that slot)
TIER_ROLE: dict[str, str] = {
    # Primary vocabulary. ``minor`` (and its ``cheap`` alias) point at the
    # ``hand`` backend: hand REPLACED the 4B in the cheap-tier slot, so the two
    # spellings are the same lane, not two lanes. There is no ``minor`` backend
    # role any more — the tier name survives for back-compat, the role does not.
    "main": "primary",
    "minor": "hand",
    "multimodal": "multimodal",
    # Back-compat aliases.
    "cheap": "hand",
    "normal": "multimodal",
    "hard": "primary",
    # Capability-ROLE names (alias the same backends as main / multimodal;
    # hand/muse/worker are their own backends). Order matters: ``tier_aliases``
    # derives ascending capability order from each role's *last* occurrence
    # position here, so the hand-role alias must appear before a multimodal one,
    # multimodal before the worker one, worker before the muse one, and muse
    # before a primary-role one (hand < senses < worker < muse < cortex) to keep
    # the last-occurrence sequence ascending
    # (hand < multimodal < worker < muse < primary).
    "hand": "hand",
    "senses": "multimodal",
    "worker": "worker",
    "muse": "muse",
    "cortex": "primary",
}


def resolve_tier(tier: str) -> "SupportedModel":
    """Return the *first* generate-task ``SupportedModel`` whose ``role_hint``
    matches ``TIER_ROLE[tier]``.

    :param tier: A tier alias — one of the :data:`TIER_ROLE` keys. The primary
        vocabulary is ``"main"`` / ``"minor"`` / ``"multimodal"``; the legacy
        ``"cheap"`` / ``"normal"`` / ``"hard"`` names are retained as aliases.
        ``"main"`` and ``"hard"`` resolve to the primary; ``"minor"``,
        ``"cheap"`` and ``"hand"`` all resolve to the 1.2B ``hand`` gear (which
        replaced the 4B in that slot — the 4B is still in the catalog as a
        candidate, but no tier resolves to it); ``"multimodal"`` and ``"normal"``
        to the Gemma 4 multimodal gear.
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

    These flags are baked into the packaged compose templates *and* named by
    ``lobes switch`` as the lines to remove when switching to a non-MTP model. This
    is the single source of truth so the two cannot drift — ``tests/test_catalog.py``
    asserts the packaged templates contain exactly these items, and the speculative
    config is pulled from the primary catalog entry rather than re-typed.

    **TWO items since 2026-07-31, not four.** The outgoing text-only primary
    (``sakamakismile/…-Text-NVFP4-MTP``) additionally needed
    ``--language-model-only`` (its export had the ViT stripped) and
    ``--tokenizer=<MTP_TOKENIZER_OVERRIDE>`` (its ``tokenizer_config`` declared a
    ``TokenizersBackend`` absent from the image). The promoted multimodal primary
    (``unsloth/Qwen3.6-27B-NVFP4``) needs NEITHER: it declares
    ``language_model_only=false`` and keeping its ViT is the whole point, and it
    ships its own tokenizer. ``MTP_TOKENIZER_OVERRIDE`` is retained for the
    demoted checkpoint, which is still selectable via ``lobes switch``.

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
    ]
