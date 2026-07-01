# Build Plan — The Gemma 4 12B gear gets speculative decoding: lobes sources or builds a gemma4_assistant draft model, wires it via --speculative-config, measures draft acceptance and decode speedup, and restores it on the catalog gemma entry + compose flag when it beats the no-spec baseline

slug: `the-gemma-4-12b-gear-gets-speculative-decoding-lob` · status: `exported` · from frame: `the-gemma-4-12b-gear-gets-speculative-decoding-lob`

> The Gemma 4 12B gear gets speculative decoding: lobes sources or builds a gemma4_assistant draft model, wires it via --speculative-config, measures draft acceptance and decode speedup, and restores it on the catalog gemma entry + compose flag when it beats the no-spec baseline

## Tasks

### t1 — Resolve the draft route: desk-source a gemma4_assistant draft and confirm the DSpark candidate

- covers: c1, h6, c10
- acceptance:
  - docs/gemma4-mtp-draft.md exists and names exactly ONE resolved route: a DSpark draft_model id, a sourced model_type==gemma4_assistant id, or 'no compatible draft available'
  - the doc records whether any model_type==gemma4_assistant draft exists on HF for this checkpoint and the tokenizer/vocab match (same vocab size) vs the served checkpoint

### t2 — Document the before-state, the gap, the scope split, and the audience in docs/gemma-4-12b-nvfp4.md

- covers: c2, c3, c4, c5, c6, h7, h8, h9, h10
- acceptance:
  - verified in-repo: the doc states the gemma gear carries no speculative_config / --speculative-config today and the gemma4_mtp method is rejected
  - the doc cites the 27B ~2.4x MTP decode gain (qwen3.6-27b-text-nvfp4-mtp.md) and frames the per-stream gap for the multimodal/normal lane the mesh consumes
  - the doc records the scope split: serve-enablement -> #71, draft-training -> separate follow-up, #75 delivers route+wiring+measure+decide, done = a measured verdict

### t3 — Make the gemma speculative wiring catalog-driven (catalog entry -> compose --speculative-config), like the 27B MTP primary

- depends on: t1
- covers: c11, h2, c10
- acceptance:
  - the gemma catalog entry carries a speculative_config field (the route resolved in t1) and lobes switch adds/removes the gemma --speculative-config compose items the same way it does for the 27B MTP primary
  - a catalog test asserts the gemma --speculative-config items round-trip through switch / the compose-items helper, mirroring the MTP primary test, and the draft shares the served checkpoint tokenizer/vocab

### t4 — Measure draft acceptance and decode speedup on the live co-resident serve (gated on #71)

- depends on: t3
- covers: c3, c7, h1, h3, h4
- acceptance:
  - with the gear serving (post-#71), the draft loads with no vocab/tokenizer mismatch and yields non-zero draft acceptance
  - records draft acceptance percent and baseline-vs-spec tok/s from a live serve on the same fleet/compose the gear deploys under, co-resident with the running 27B primary

### t5 — Commit the verdict: restore speculative_config if it beats baseline, else document the negative

- depends on: t4
- covers: c7, c3, h6
- acceptance:
  - if acceptance/speedup beats the no-spec baseline: the gemma catalog speculative_config + compose items are restored as default and lobes status/switch reflect it; otherwise the gear keeps no spec-decode and docs/gemma4-mtp-draft.md records the measured numbers that rule it out
  - exactly one concrete outcome is committed (route resolved to restore-or-document), satisfying the 'one concrete route' promise

## Risks

- [unknown_nonblocking] whether any native model_type==gemma4_assistant draft exists to source on HF is unknown — DSpark draft_model may be the only viable path (task t1)
- [follow_up] serve-enablement (#71) gates measurement: t4/t5 cannot run until the gemma gear actually serves (TRITON_ATTN honored on the transformers backend) (task t4)
- [unknown_nonblocking] DSpark draft_model may fail to load or yield 0 percent acceptance on this checkpoint, invalidating the cheap path and forcing a native-draft or stop decision (task t4)

## Execution waves (from `devague plan waves` — scheduling metadata, not orchestration)

- **wave 0:** t1, t2 — parallel; file-disjoint (t1 → new `docs/gemma4-mtp-draft.md`, t2 → existing `docs/gemma-4-12b-nvfp4.md`) — **DONE** (merged)
- **wave 1:** t3 — **re-binned behind #71** (see correction below)
- **wave 2:** t4 — depends on t3; **blocked on #71** (gear must serve before any draft can be measured)
- **wave 3:** t5 — depends on t4 (decide once measured)

### Planning correction (2026-07-01): t3 is gated on #71, not buildable-now

The original split binned t3 ("make the gemma wiring catalog-driven") as
buildable now. On implementation it proved otherwise: the repo **deliberately
guards** the current no-spec-decode invariant with three tests that name #75 as
the follow-up —

- `test_gemma_has_no_speculative_config` (asserts `gemma.speculative_config == ""`),
- `test_fleet_compose_multimodal_vision_active_no_spec_decode` (asserts the
  `vllm-multimodal` service carries **no** `--speculative-config`),
- the MTP-items drift guard.

Adding the DSpark `speculative_config` and flipping those guards **is** the
restore action — which this spec gates behind measurement (t5) and #71. Enabling
it now would bake an unvalidated draft into a gear that cannot yet serve, on zero
evidence. So **t3 is re-binned behind #71 alongside t4/t5**: the honest
buildable-now work was t1 + t2 (both merged). t3 → t5 resume when #71
serve-enablement lands.

- [follow_up] **serve-enablement (#71) also gates t3**: enabling the gemma
  speculative_config + flipping the three guard tests is the restore action, only
  valid after t4's measurement on a serving gear (task t3).
