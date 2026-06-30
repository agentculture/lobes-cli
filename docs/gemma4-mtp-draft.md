# Gemma 4 12B gear — speculative-decoding draft route (issue #75, task t1)

> Resolves the `unknown_nonblocking` risk in
> [`docs/specs/2026-07-01-gemma-4-12b-gear-gets-speculative-decoding-draft.md`](specs/2026-07-01-gemma-4-12b-gear-gets-speculative-decoding-draft.md)
> ("whether a `model_type==gemma4_assistant` draft for this exact
> checkpoint/vocab exists to SOURCE on HF is unknown — the #75 spike
> determines it"). Desk research only — no live serve, no live measurement
> (the gear does not yet serve; see [#71](gemma-4-12b-nvfp4.md#live-validation-status-71)).
> Researched and written 2026-07-01.

## Resolved route

**Route: `deepseek-ai/dspark_gemma4_12b_block7` via the `draft_model` method
(DSpark)** — this is the ONE route task t3 should wire next.

This is the route to wire and measure **first**, per the user's explicit
sequencing decision recorded in the spec (`Decisions`, resolved 2026-07-01):
*measure DSpark first as the cheap path; only source/build a native
`gemma4_assistant`-family draft if DSpark proves the win real but
insufficient.* That decision is **not superseded** by anything found below —
it was a sequencing choice independent of whether a native draft exists — but
the finding below changes what "escalate" means in practice: it is no longer
a build-from-scratch problem.

Already-documented wiring shape (`docs/gemma-4-12b-nvfp4.md`, currently
disabled by default):

```env
MULTIMODAL_SPECULATIVE_CONFIG={"method": "draft_model", "draft_model_id": "deepseek-ai/dspark_gemma4_12b_block7", "num_speculative_tokens": 3}
```

**Headline update to the spec's open question:** a native
`model_type==gemma4_assistant`-*family* draft for this exact checkpoint
**does exist** and is publicly available on HF —
[`google/gemma-4-12B-it-assistant`](https://huggingface.co/google/gemma-4-12B-it-assistant)
(`model_type: gemma4_unified_assistant`). This was previously recorded as
unknown; it is now resolved. It is documented below as the **escalation
candidate** for t3/t4 if/when DSpark proves insufficient — not the route
chosen to wire first.

## HF sourcing search

### 1. Native `gemma4_assistant`-family draft — exists, for the Unified (multimodal) line

vLLM's `gemma4_mtp` support (`vllm/config/speculative.py`,
`hf_config_override` / `use_gemma4_mtp`) recognizes **two** assistant
`model_type` strings, not one:

```python
if hf_config.model_type in ("gemma4_assistant", "gemma4_unified_assistant"):
    hf_config.model_type = "gemma4_mtp"
    ...
    hf_config.update({"n_predict": 1, "architectures": ["Gemma4MTPModel"]})
```

(source: <https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py>,
fetched 2026-07-01.) `gemma4_assistant` is the plain-text-model family
(E2B/E4B/26B-A4B/31B); `gemma4_unified_assistant` is the dedicated family for
the **Unified** (multimodal) line — which is what our served checkpoint is
(`model_type: gemma4_unified`, `Gemma4UnifiedForConditionalGeneration`).

Google publishes one assistant per Gemma 4 size, confirmed via the HF API
(`https://huggingface.co/api/models/google/gemma-4-12B-it-assistant`, fetched
2026-07-01) and the `vllm-project/recipes` Gemma4 doc
(<https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md>):

| Target | Assistant draft | `model_type` |
|---|---|---|
| E2B IT | `google/gemma-4-E2B-it-assistant` | `gemma4_assistant` (centroids masking) |
| E4B IT | `google/gemma-4-E4B-it-assistant` | `gemma4_assistant` (centroids masking) |
| **12B IT (Unified)** | **`google/gemma-4-12B-it-assistant`** | **`gemma4_unified_assistant`** |
| 26B-A4B IT | `google/gemma-4-26B-A4B-it-assistant` | `gemma4_assistant` |
| 31B IT | `google/gemma-4-31B-it-assistant` | `gemma4_assistant` |

`google/gemma-4-12B-it-assistant` model card
(<https://huggingface.co/google/gemma-4-12B-it-assistant>): "This model card
is for the Multi-Token Prediction (MTP) drafters for the Gemma 4 models,"
Apache-2.0, authored by Google DeepMind, intended target `google/gemma-4-12B-it`
(the IT 12B Unified model — text+image+audio, matching our checkpoint's
audio-capable family). 0.42B params (BF16, `safetensors`,
`Gemma4UnifiedAssistantForCausalLM`).

**Wiring shape (for reference / t3, not chosen as the first route):**

```json
{"method": "mtp", "model": "google/gemma-4-12B-it-assistant", "num_speculative_tokens": 1}
```

`num_speculative_tokens: 1` is not a guess — it mirrors the plan's existing
Decision ("wire the assistant draft with num_speculative_tokens=1 (n_predict=1)
... when using the native gemma4_mtp method") and is *forced* by
`hf_config_override` setting `n_predict: 1` on the assistant's config
unconditionally — the assistant runs all its decoder layers in one forward
pass to produce a single draft token. A community report
(linked from vLLM issue #42005 discussion) confirms this method string and
shape working for the E4B target on vLLM v0.21.0:
`--speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E4B-it-assistant","num_speculative_tokens":4}'`.
**Note `num_speculative_tokens` in that example is 4 for E4B** — the
n_predict=1-per-round constraint is about the assistant's single-forward-pass
draft-token production, not necessarily a hard cap on how many speculative
positions the scheduler proposes per step; treat the 12B's exact
`num_speculative_tokens` as **to verify at serve time** (t3/t4), not assumed
from this E4B data point.

Also confirmed (vLLM issue #42005, "Gemma 4 assistant speculative decoding
docs do not match actual behavior on vLLM 0.20.1", closed by docs-only
PR #42180): an *earlier* vLLM (0.20.1) raised
`NotImplementedError: Speculative Decoding with draft models or parallel
drafting does not support multimodal models yet` when an assistant checkpoint
was passed **without** `"method": "mtp"` (i.e., treated as a generic external
draft model rather than routed through the dedicated MTP/KV-sharing proposer
path). The fix was procedural (use `"method": "mtp"` explicitly), not a code
change — PR #42180 only touched docs. This is consistent with, and does not
contradict, the spec's existing finding that `{"method": "gemma4_mtp"}` is
rejected: the correct method literal is `"mtp"`, not `"gemma4_mtp"` (the
`gemma4_mtp` string is vLLM's *internal* normalized `model_type`, written by
`hf_config_override` from the on-disk assistant config — it is never the
caller-facing method name).

**Open / not directly confirmed:** the E4B↔v0.21.0 report is the only live
confirmation of the `"method":"mtp"` shape found in this search. No equivalent
public report was found specifically for the **12B Unified** target (our
checkpoint's family) or for vLLM 0.22.1 (our pinned version). The presence of
a dedicated `gemma4_unified_assistant` branch in `hf_config_override` is
strong circumstantial evidence the Unified/multimodal target is intentionally
supported via this path — but it is unverified end-to-end for our checkpoint.
**To verify at serve time (t4), gated on #71.**

### 2. DSpark — confirmed purpose-built for this exact target family

[`deepseek-ai/dspark_gemma4_12b_block7`](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7)
(part of DeepSeek's [DeepSpec](https://github.com/deepseek-ai/DeepSpec)
collection of speculative-decoding drafts). `config.json` fetched directly
(2026-07-01):

```json
{
  "architectures": ["Gemma4DSparkModel"],
  "model_type": "gemma4_text",
  "block_size": 7,
  "vocab_size": 262144,
  "num_target_layers": 48,
  "target_layer_ids": [5, 17, 29, 41, 46],
  "target_model_type": "gemma4_unified",
  "target_text_model_type": "gemma4_unified_text"
}
```

`target_model_type: "gemma4_unified"` / `target_text_model_type:
"gemma4_unified_text"` / `num_target_layers: 48` are an explicit,
machine-readable declaration that this draft was trained against the **same
target family and layer count** as our served checkpoint (whose own
`text_config.num_hidden_layers` is 48 — confirmed below). DSpark is *not* a
`model_type==gemma4_assistant`/`gemma4_unified_assistant` checkpoint — it is
a different draft architecture (`Gemma4DSparkModel`, a Markov/confidence-head
multi-block drafter, `block_size=7` meaning up to 7 candidate tokens per
draft block), wired through vLLM's generic `draft_model` method, not the
native MTP path. 3.4B params, BF16, 6.86 GiB on disk.

No README/model card text was published for this checkpoint (HF returns
"Entry not found" for `README.md`) — provenance here rests on the `config.json`
fields above, which is sufficient to confirm target-family compatibility but
not sufficient to know training data, recommended `num_speculative_tokens`,
or license. **To verify at serve time:** whether vLLM 0.22.1's generic
`draft_model` method carries any residual restriction against multimodal/
unified targets (the same class of issue that hit assistant-style checkpoints
on vLLM 0.20.1 when *not* routed through the MTP path, per #42005 above) —
this has not been confirmed lifted for `draft_model` specifically, only
inferred from `vllm/config/speculative.py` containing no explicit
multimodal-target gate. If DSpark's `draft_model` wiring turns out to be
blocked on 0.22.1 for the same reason, that is itself useful t4 signal in
favor of escalating to the native `mtp` route documented above.

## Tokenizer / vocab compatibility vs the served checkpoint

Served checkpoint:
`sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`
(`config.json` fetched directly 2026-07-01):

```json
{
  "model_type": "gemma4_unified",
  "text_config": {
    "model_type": "gemma4_unified_text",
    "vocab_size": 262144,
    "num_hidden_layers": 48
  }
}
```

Per the served checkpoint's own HF card (AI-summarized fetch, not raw JSON —
flagged accordingly), it is an NVFP4 quantization of
`yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1`, itself a fine-tune of
`google/gemma-4-12B-it` — the same base `google/gemma-4-12B-it-assistant` was
built to draft for, and the same target family DSpark's `config.json`
declares (`target_model_type: gemma4_unified`).

| Checkpoint | role | `vocab_size` | tokenizer | layers |
|---|---|---|---|---|
| `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4` | served target | **262144** | `GemmaTokenizer` | 48 |
| `deepseek-ai/dspark_gemma4_12b_block7` (DSpark) | draft candidate (route chosen) | **262144** | (uses target's; declares `target_model_type=gemma4_unified`, `num_target_layers=48`) | 5 (drafter), targets 48 |
| `google/gemma-4-12B-it-assistant` (native) | draft candidate (escalation) | **262144** | `GemmaTokenizer` | 4 (drafter, KV-shared with target) |

**Conclusion: vocab size matches across all three (262144), and both
candidates' configs explicitly target the served checkpoint's model family
(`gemma4_unified` / `gemma4_unified_text`, 48 target layers).** Tokenizer
*class* matches (`GemmaTokenizer`) between the served checkpoint and the
native assistant; DSpark does not ship its own tokenizer files and is
expected to reuse the target's. This is necessary-but-not-sufficient
evidence: vocab/tokenizer compatibility removes the most common hard-failure
mode (shape mismatch / garbage output from a vocab mismatch), but it does
**not** by itself guarantee a non-zero, useful acceptance rate against this
specific **fine-tuned** target (`...coder-fable5-composer2.5...`) rather than
the base `google/gemma-4-12B-it` the drafts were presumably trained/evaluated
against — fine-tuning can shift the target's output distribution enough to
measurably change (usually reduce, rarely break) draft acceptance even with
identical architecture and vocab. Actual acceptance percent and tok/s are
explicitly **not** claimed here — that is t4's job, live, post-#71.

## Next

- **t3** (catalog-driven wiring): wire the gemma catalog entry's
  `speculative_config` field to the DSpark `draft_model` route above
  (mirroring `mtp_compose_command_items()` / the 27B primary's
  `speculative_config` pattern in `lobes/catalog.py`), with a compose-items
  round-trip test. Keep the native `mtp` route (`google/gemma-4-12B-it-assistant`)
  documented as the escalation path, not wired by default.
- **#71** (serve-enablement) is the hard gate: the gear LOADS but does not yet
  SERVE (`VLLM_ATTENTION_BACKEND=TRITON_ATTN` not yet honored on the
  transformers-modeling backend — see
  [`gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md#live-validation-status-71)).
  No draft — DSpark or native — can be loaded or measured until #71 lands;
  t4/t5 stay blocked until then.
