# The Gemma 4 12B gear gets speculative decoding: lobes sources or builds a gemma4_assistant draft model, wires it via --speculative-config, measures draft acceptance and decode speedup, and restores it on the catalog gemma entry + compose flag when it beats the no-spec baseline

> The Gemma 4 12B gear gets speculative decoding: lobes sources or builds a gemma4_assistant draft model, wires it via --speculative-config, measures draft acceptance and decode speedup, and restores it on the catalog gemma entry + compose flag when it beats the no-spec baseline

## Audience

- lobes operators/maintainers and the Culture mesh that consumes the multimodal (Gemma 4 12B) generate lane

## Before → After

- Before: the Gemma gear carries no --speculative-config: its -MTP checkpoint exposes no gemma4_assistant draft and vLLM 0.21/0.22 reject the gemma4_mtp method, so its single-stream decode lacks any speculative boost
- After: the Gemma gear has a measured verdict on speculative decoding: either a validated draft wired via --speculative-config with recorded acceptance and decode speedup and restored on the catalog gemma entry + compose flag, OR a documented negative recording the numbers that rule it out

## Why it matters

- the 27B MTP primary gets ~2.4x single-stream decode from speculative decoding; the Gemma multimodal/normal lane has none, so per-stream it is comparatively slow — spec-decode closes that gap or we learn it cannot and stop chasing it

## Requirements

- any draft model must share the served checkpoint tokenizer and vocab
  - honesty: a candidate draft loads in vLLM alongside the target with no vocab-size/tokenizer mismatch and yields non-zero draft acceptance
- wiring flows through the catalog: the gemma entry speculative_config drives the compose --speculative-config items (the same catalog-data pattern the 27B MTP primary uses via mtp_compose_command_items)
  - honesty: lobes switch adds/removes the gemma --speculative-config items the same way it does for the 27B MTP primary, and the catalog compose-items round-trip test still passes

## Honesty conditions

- the 'sources or builds' promise resolves to ONE concrete route — a measured DSpark draft_model, a sourced gemma4_assistant draft, or a documented 'no compatible draft available' — never left ambiguous
- the multimodal/normal generate lane (model=multimodal|normal) has real mesh traffic worth speeding up — not a dormant tier
- the verdict is backed by recorded numbers (acceptance percent, baseline vs spec tok/s) from a live co-resident serve, not an estimate
- verified in-repo: the catalog gemma entry carries no speculative_config and the compose has no gemma --speculative-config items (true as of this frame)
- the 27B primary ~2.4x MTP decode gain is documented (qwen3.6-27b-text-nvfp4-mtp.md) and the Gemma gear has no equivalent boost — the gap is real, not assumed
- the scope split holds: serve-enablement stays in #71, draft-training stays a separate follow-up, and #75 delivers route + wiring + measurement + keep/drop decision only
- the recorded numbers come from the same fleet/compose the gear deploys under, co-resident with the running 27B primary
- DSpark draft_model wiring actually loads on this checkpoint and yields >0 percent acceptance; if it crashes or hits 0 percent the cheap path is invalid

## Success signals

- a committed verdict: either catalog gemma speculative_config + the compose --speculative-config items restored with recorded acceptance percent and tok/s speedup beating the no-spec baseline, or a doc section recording the measured numbers that rule it out

## Scope / boundaries

- out of scope: training a new Gemma BASE model, the serve-enablement fix itself (owned by issue #71), and the realtime /v1/audio overlay; in scope is the draft model + wiring + measurement + the keep/drop decision

## Non-goals

- not committing to BUILDING (training/distilling) a gemma4_assistant draft head within this issue unless sourcing fails AND DSpark proves the win — a build is its own follow-up

## Assumptions

- the gear serve-enablement follow-up (issue #71: force TRITON_ATTN on the transformers backend) lands before MTP decode speedup can be measured end-to-end; until then measurement is blocked

## Decisions

- measure the existing DSpark draft (deepseek-ai/dspark_gemma4_12b_block7) via the draft_model method FIRST as the cheap path to prove whether spec-decode is worthwhile; only source/build a native gemma4_assistant draft if DSpark shows the win is real but the route is insufficient
- wire the assistant draft with num_speculative_tokens=1 (n_predict=1), per the issue, when using the native gemma4_mtp method
- issue #75 is GATED on issue #71 serve-enablement: the gear must SERVE (TRITON_ATTN honored on the transformers backend) before any draft can be loaded, so #75 does not proceed to its measure-and-decide leg until #71 lands — desk-sourcing a candidate draft is the only sub-task that can start earlier

## Hard questions

- Is the issue specifically about the NATIVE gemma4_mtp path (a model_type==gemma4_assistant draft), or about any speculative-decoding speedup for the Gemma gear (which DSpark draft_model could satisfy)?
  - **RESOLVED (user, 2026-07-01):** any speculative-decoding speedup counts. The
    existing DSpark `draft_model` route is measured FIRST; a native
    `gemma4_assistant` draft is pursued only if DSpark proves the win is real but
    insufficient (see Decisions).
- risk: serve-enablement (#71) may not land, blocking end-to-end MTP measurement indefinitely; the draft + wiring can still ship, but the speedup number cannot
  - **RESOLVED (user, 2026-07-01):** #75 is BLOCKED on #71 — it does not ship draft
    wiring ahead of serve. Only desk-sourcing a candidate draft can start before
    #71 lands; everything measurable waits for the gear to serve (see Decisions).

## Open / follow-up

- the recipe/cost/data to BUILD (distill/train) a gemma4_assistant draft head if sourcing fails — its own follow-up, not this issue
- gear serve-enablement (issue #71: force TRITON_ATTN on the transformers backend so the gear actually serves) — a dependency owned by #71; MTP measurement is gated on it

## Accepted unknown (non-blocking — exporter drops these; restored by hand)

- whether a `model_type==gemma4_assistant` draft for this exact checkpoint/vocab
  exists to SOURCE on HF is unknown — the #75 spike determines it. If none
  exists, building one is the separate follow-up above (not this issue). This is
  the frame's `unknown_nonblocking` v1; it lives in the frame JSON for
  /spec-to-plan but the spec_md exporter omits it, so it is restored here.

## Grounding (verified facts, hand-appended for the plan leg)

- The `gemma4_mtp` self-speculation path does **not** exist: vLLM 0.21/0.22 enable
  Gemma 4 MTP only when a *separate* draft has `model_type == "gemma4_assistant"`
  (see `vllm/config/speculative.py` `hf_config_override` / `use_gemma4_mtp`).
  Unlike DeepSeek (`deepseek_v3 → deepseek_mtp` auto-derivation), the unified
  target cannot self-speculate; `{"method": "gemma4_mtp"}` is rejected with
  `Unsupported speculative method`. The served checkpoint
  (`sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`) carries no
  mtp/assistant/nextn keys despite its `-MTP` name.
- The DSpark candidate is `deepseek-ai/dspark_gemma4_12b_block7`, wired today via
  `MULTIMODAL_SPECULATIVE_CONFIG={"method": "draft_model", "draft_model_id":
  "deepseek-ai/dspark_gemma4_12b_block7", "num_speculative_tokens": 3}` —
  documented as a disabled, unvalidated experiment in `docs/gemma-4-12b-nvfp4.md`.
- Catalog→compose wiring pattern: the 27B primary's `speculative_config` field on
  its catalog entry drives `mtp_compose_command_items()` (`lobes/catalog.py`),
  which `lobes switch` adds/removes. A restored gemma speculative_config follows
  the same path; a catalog test asserts the compose items round-trip.
- Hard dependency (#71): the gear LOADS but does not yet SERVE — Gemma 4's
  non-square attention (`global_head_dim 512 ≠ head_dim 256`) needs
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, not honored on vLLM's transformers-modeling
  backend (o_proj GEMM shape mismatch 4096≠8192). No draft can be measured until
  this is fixed.
