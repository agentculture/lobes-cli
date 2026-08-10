# LiquidAI/LFM2.5-1.2B-Instruct — the `hand` lobe

`hand` is the fleet's **ninth Colleague role** and its **designated fine-tuning
base**. The metaphor is muscle memory: **one cheap base, many LoRA adapters,
each mastering a domain**. Where `worker` is an untrained generalist doer,
`hand` is a trained specialist — it knows a few things extremely well because
someone taught it, and nothing else.

At ~1.2B parameters (~2.4 GiB in bf16) it is cheap enough to co-reside on
**every card in the mesh**, which is the whole point: a specialist you cannot
afford to keep resident is not muscle memory, it is a trip to another box.

| | |
|---|---|
| checkpoint | `LiquidAI/LFM2.5-1.2B-Instruct` |
| role | `hand` (ninth Colleague role, default-hosted) |
| tier aliases | `model=hand`, and `model=minor` / `model=cheap` (repointed) |
| architecture | `Lfm2ForCausalLM` — hybrid short-conv + GQA, **text-only** |
| context | 32768 native |
| quantization | none (bf16) |
| tool parser | `lfm2` |
| reasoning parser | **none** — no thinking mode |
| LoRA | served with `--enable-lora` |

## Why this checkpoint

The spec's own answer (`docs/specs/2026-08-09-hand-lobe-lfm2-5-1-2b.md`) is
that the fleet had **no LoRA surface in-tree at all** and its nominal
fine-tuning base — `Qwen/Qwen3.5-4B` — was still broken on its pinned image.
LFM2.5-1.2B is small enough to train cheaply, small enough to keep resident
everywhere, and — unlike the 4B it replaces — actually boots on the pinned
nightly.

`Qwen/Qwen3.5-4B` stays in the catalog as a plain candidate
(cite-don't-delete): nothing about that checkpoint changed, only which gear the
`minor` tier resolves to. It is still selectable via `lobes switch`.

## Architecture and vLLM support

`Lfm2ForCausalLM` requires **vLLM ≥ 0.23.0**. Verified against the exact pinned
nightly digest — not inferred from the version string — on a physical Jetson
AGX Thor:

```text
VLLM_VERSION: 0.23.1rc1.dev672+g93d8f834d
LFM2_REGISTERED: True
archs: ['Lfm2ForCausalLM', 'Lfm2MoeForCausalLM',
        'ColBERTLfm2Model', 'Lfm2VlForConditionalGeneration']
```

Both Thor and Orin already carry digest `7c5a10e9a8b3`, so the `hand` lane needs
**no new image pull**.

Note the fourth architecture in that list: `Lfm2VlForConditionalGeneration` is
LiquidAI's *vision* variant. **This checkpoint is not it.** `hand` is text-only,
carries no ViT, and advertises neither `image_understanding` nor
`video_understanding`. That also means the lane carries **no
`--language-model-only`** — there is no vision tower to switch off.

## Tool calling — the `lfm2` parser

LFM2 emits its own syntax, whose delimiters are **special tokens**:

```text
<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>
```

This is the same shape of trap that made `pythonic` silently wrong for Gemma 4
— a parser running with `skip_special_tokens=True` never sees the delimiters,
matches nothing, and vLLM relays a perfectly well-formed call as ordinary
assistant **content** with `tool_calls: null`. vLLM ships a purpose-built parser
registered as **`lfm2`**
(`vllm/tool_parsers/lfm2_tool_parser.py` → `Lfm2ToolParser`), and
`lobes.runtime._parser.infer_parser` returns it for any `lfm2`-family id.

**This one fails loudly rather than silently.** `Lfm2ToolParser.__init__`
resolves both delimiters through `self.vocab.get()` and **raises** when either
is missing — so a tokenizer revision that dropped them kills server startup
instead of degrading to prose. That is strictly better than the Gemma 4 failure
mode, and it is why the acceptance criterion for live validation is phrased as
"confirm the actually-pulled tokenizer revision carries
`<|tool_call_start|>`/`<|tool_call_end|>`".

### No reasoning parser

Unlike the cortex lane (`--reasoning-parser=qwen3`) and all three Gemma 4 lanes
(which need `--tool-call-parser=gemma4` and `--reasoning-parser=gemma4` as a
**matched pair**), `hand` needs **no** `--reasoning-parser`.
`LFM2.5-1.2B-Instruct` has no thinking mode; LiquidAI ships
`LFM2.5-1.2B-Thinking` as a separate checkpoint. Enabling a reasoning parser
here would be arming a half of a pair that has no other half.

## LoRA — the muscle-memory surface

The lane is served with `--enable-lora` **armed and the inventory empty**. That
is a deliberate v1 shape: the *serving* half of muscle memory ships now, the
*training* half is cross-repo (see below), and an armed-but-empty lane is
honest in a way a promised-later one is not.

Adapters are declared as a fixed list at boot:

```bash
HAND_LORA_MODULES="legal=/models/adapters/legal,sql=/models/adapters/sql"
```

| knob | default | meaning |
|---|---|---|
| `HAND_LORA_MODULES` | *(empty)* | `name=path` list, comma-separated, fixed at boot |
| `HAND_MAX_LORAS` | `4` | concurrently-resident adapters |
| `HAND_MAX_LORA_RANK` | `32` | maximum adapter rank |

**There is no runtime hot-load.** Adding an adapter is a lane restart. This is a
recorded decision, not an omission: a hot-load surface is a mutable-state API on
a lane whose whole value is being cheap and predictable.

### Addressing an adapter

| request | serves |
|---|---|
| `model=hand` | the **base** checkpoint |
| `model=minor` / `model=cheap` | the base (back-compat tier spellings) |
| `model=hand:<domain>` | the adapter named `<domain>` |

The bare name **never 404s** — `hand` with no adapters declared is a working
lane, not a broken one. A `hand:<domain>` naming an adapter that is not in the
inventory is **refused with a clear error**; it never silently falls back to the
base or to another lane, because a caller who asked for the legal specialist and
got the generalist has been lied to.

### `--max-lora-rank` and honest failure

`HAND_MAX_LORA_RANK` defaults to **32**, which covers the ranks unsloth's
defaults produce for a model this size. An adapter whose rank exceeds the
configured maximum **fails at load** — vLLM refuses it, and the honesty rule
(#92) then applies: an adapter that did not load is absent from `GET /v1/models`
and from `/capabilities`. A declared-but-unloaded adapter must never read as
usable. If you train at a higher rank, raise the knob and restart the lane.

## Pressure policy — the servable floor

`hand` is the **servable floor**. Under the swap > 75 % / iowait > 50 % pressure
policy, `cortex`, `senses`, `worker` and `muse` each shed with **HTTP 429 +
`Retry-After`**; `hand` is served regardless. It inherits this from the `minor`
tier it replaced, and it is the reason the floor is worth keeping: something has
to answer when the box is under pressure.

## Responsibilities

| | |
|---|---|
| **responsibilities** | `domain_mastery`, `learned_skill`, `specialized_task`, `tool_use` |
| **forbidden** | `final_decision`, `repo_action`, `security_decision` |

`hand` proposes and executes within its domain; `cortex` decides. The forbidden
list is deliberately conservative for v1 — **adding** a responsibility later is
contract-compatible, **removing** one is a break, so v1 withholds `repo_action`
rather than granting it speculatively. Granting it once adapters actually exist
is tracked as **agentculture/lobes-cli#180**.

## Training — a one-directional boundary

Adapters are produced by **`unsloth-cli`**, out of tree. lobes **serves**
adapters; it does not train them, and nothing under `lobes/` imports, shells out
to, or depends on unsloth — the only mentions are in comments and docs like this
one. The dependency runs one way.

`unsloth-cli` 0.5.0 ships only scaffold verbs, so the muscle-memory loop cannot
yet be exercised end to end. Verifying/adding LFM2.5 support there is tracked as
**agentculture/unsloth-cli#16**.

## Validation status

Per the #108 rule, `hand` is **DECLARED** on every card and becomes **VALIDATED**
only on the cards whose acceptance transcript has landed under `docs/evidence/`.
One box's successful boot never promotes another card.

**No card is validated.** `hand` is DECLARED everywhere.

| card | status | evidence |
|---|---|---|
| Jetson AGX Orin (sm_87, 64 GB) | DECLARED — *served once, budget not reproducible* | `docs/evidence/2026-08-10-partial-hand-orin.txt` |
| Jetson AGX Thor (sm_110, 128 GB) | DECLARED — boot failed | [#181](https://github.com/agentculture/lobes-cli/issues/181) |
| DGX Spark GB10 (128 GB) | DECLARED — *functionally sound, budget not reproducible* | `docs/evidence/2026-08-10-partial-hand-spark.txt` |
| `base` (unrecognised card) | DECLARED | untestable by construction |

The Orin and the Spark both land in the same split state, and for the same
reason — see **"Why no budget reproduces"** below, which is the finding that
matters more than either card's numbers.

**What the Orin runs established — and what they did not.** The lane *served*
once, and everything observed on that live engine holds: the bf16 sentinel and
the text-only / no-reasoning-parser flags plumbed correctly, a correct
known-answer completion, an unknown model id refused with 404, and — the check
this lane exists to pass — a **tool call returning a structured `tool_calls`
array**, with the tokenizer's `<|tool_call_start|>` / `<|tool_call_end|>`
confirmed present as special tokens (ids 10 and 11).

**The Spark reproduced every one of those functional results** on a third card
and added two: `HAND_ATTENTION_BACKEND=auto` resolving through the
`--attention-config` flag (the first live confirmation that the `d8` dead-knob
fix actually works, rather than merely not crashing), and `GET /v1/models`
returning exactly one entry — the base — so an empty adapter inventory
advertises no phantom adapter. The container also reached compose `healthy`.
Full transcript: `docs/evidence/2026-08-10-partial-hand-spark.txt`.

### Why no budget reproduces

Both cards that served `hand` produced a **different KV pool on every boot** at
identical settings. On the Spark, three boots minutes apart:

| run | free RAM before | available KV | KV tokens |
|---|---|---|---|
| 1 | ~28 GiB | 6.21 GiB | 541,886 |
| 2 | ~18 GiB | 3.34 GiB | 291,970 |
| 3 | ~18 GiB | 3.54 GiB | 308,754 |

Runs 2 and 3 agree within 6% and were taken at the same free-memory level; run
1 was taken minutes after 31.7 GiB was freed on the box. The Orin's spread
(2.7 GiB, then 0.14 GiB, then 0.09 GiB) is the same effect on a smaller card
with a tighter margin. The mechanism:

> On a unified-memory card with co-resident tenants, `gpu_mem_util` does not
> name a stable budget. vLLM profiles against memory that is **free at that
> instant**, so the same util yields a different KV pool depending on what else
> is resident. A single boot's KV number measures the box's state, not the
> card's capacity for the role.

This does not make `0.06` wrong — `hand` booted and served at `0.06` on every
run of both cards. It makes any **measured** pool or concurrency figure for
this role unsupportable, which is why all four cards stay DECLARED.

The practical reading for an operator: raise `HAND_GPU_MEM_UTIL` until the
profile reports comfortable KV on *your* box under *your* steady-state load, or
trim `HAND_MAX_MODEL_LEN` (vLLM names the length that would fit — 7984 in the
Orin's tightest run). Do not treat a number from one boot as a budget. Tracked
in [#183](https://github.com/agentculture/lobes-cli/issues/183).

One operational note the Spark runs paid for: four earlier attempts on that box
failed to boot at all, and the cause was **memory exhaustion on a shared
workstation**, not the lane. The box was in its own pressure-policy `busy`
state (swap 100% full, 16 GiB available, a browser holding 31.7 GiB); a
`docker compose up -d` took over ten minutes and the engine never finished
profiling. With the browser closed, the identical command returned in ten
seconds and the engine was serving in 71.75 s.

Two things these runs deliberately do **not** establish: adapter serving end to end
(v1 ships zero adapters, so `hand:<domain>` resolution and the adapter-honesty
filter are covered by offline tests only), and anything reached through the
lobes gateway rather than the engine directly. Both are tracked in
[#183](https://github.com/agentculture/lobes-cli/issues/183).

**A budget was refuted here.** The Orin was declared at `0.10` and the box
refused it twice — once because the lane was missing the cudagraph-estimate
off-switch (available KV profiled to **-9.25 GiB**, negative), and once because
`0.10` simply does not fit beside the orin-lobe shape's committed lanes. Both
runs are recorded in
`docs/evidence/2026-08-10-hand-lobe-budget-derivation.txt`. The reasoning
behind `0.10` was plausible and wrong, which is the argument for measuring
rather than computing a budget on this class of board.

See `docs/machine-profiles.md` and `docs/deployment-shapes.md` for the per-card
picture.

## See also

- `docs/colleague-stack.md` — the nine-role contract
- `docs/gateway-fleet.md` — the `hand` lane and its adapter aliases
- `docs/qwen3.5-4b-minor.md` — the demoted 4B this replaced in the tier
- `lobes explain roles` / `lobes capabilities`
