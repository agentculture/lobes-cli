# lobes serves the brain shape you choose: machine-as-brain keeps the whole brain on one box; mesh-brain gives each box the lobes it is best at — the Spark drops Gemma, Thor drops Qwen — one safe command either way

> lobes serves the brain shape you choose: machine-as-brain keeps the whole brain on one box; mesh-brain gives each box the lobes it is best at — the Spark drops Gemma, Thor drops Qwen — one safe command either way
> instruction: Ship the shape axis end-to-end: lobes init offers machine-as-brain (default) and per-box mesh-lobe shapes; verify by running the success-signal commands on the GB10 and the Thor.

## Audience

- Operators of one-or-many NVIDIA boxes running the Culture mesh local brain — the single-box user (whole brain on one machine) and the multi-box fleet operator (DGX Spark GB10 + Jetson AGX Thor 128GB today, Jetson AGX Orin 64GB next) — plus the agents consuming the roles.
  - instruction: Validate each audience leg: goldens for the one-box path; physical-box boots for the Spark and Thor shapes; unchanged per-box endpoints for mesh consumers.

## Before → After

- Before: Fleet composition is fixed, not chosen: the four core roles (cortex, senses, embedder, reranker) are unconditional in the fleet template — only minor/middle/multimodal-coder sit behind COMPOSE_PROFILES — so Spark-drops-Gemma or Thor-drops-Qwen today means hand-editing docker-compose.yml, exactly the drift the #108 goldens exist to prevent. #108 profiles say what a card CAN serve and how it is tuned; nothing says what a box SHOULD host in a mesh. The substrate is verified: #109 closed on GB10 evidence (fp8-KV is an sm_110-conditional trait; the attention-backend env is dead on the shared pinned nightly).
  - instruction: Cite the evidence in the spec: the template lines showing no profiles: stanza on the four core roles, and closed #109 with the GB10 findings. No code change needed for this claim.
- After: An operator picks a deployment shape safely: machine-as-brain (default — one box hosts every lobe it can, today's behaviour) or a per-box mesh-brain lobe profile. Near-term shapes: the Spark keeps the Qwen cortex + embedder + reranker + audio and drops Gemma senses; Thor keeps Gemma senses + the rest and drops the Qwen cortex.
  - instruction: Implement the spark and thor mesh-lobe shapes as data over the #108 Profile schema; verify via h5: zero-hand-edit boots on both physical boxes with every remaining role passing its correctness probe.

## Why it matters

- Not everyone has many machines — the whole-brain-on-one-box shape must stay first-class and zero-decision; mesh-brain specialization is opt-in for operators who have a fleet.
  - instruction: Keep the bare lobes init path byte-identical on recognised cards (machine-as-brain goldens unchanged); the default path gains zero new required decisions.
- Co-residency taxes every lobe: senses is trimmed to 32K, cortex to 128K instead of its 256K native, inside a 0.56 GPU budget on the GB10 — a box that hosts only the lobes it is best at gives each lobe its machine back.
  - instruction: Quote the shipped env values in the spec (senses 32K, cortex 128K vs 256K solo-native, 0.56 budget); the reclaim requirement c12 is where the tax gets repaid.

## Requirements

- Deployment shape is a first-class axis ORTHOGONAL to the hardware profile: shape (which lobes this box hosts) x card (how each lobe is tuned on this silicon) compose as data, without a named-profile matrix explosion.
  - instruction: Model shape as a role-subset overlay composed with the #108 card profile at render time; prove with per-(shape,card) goldens.
  - honesty: The spark and thor mesh-lobe shapes are expressible purely as data over the #108 Profile schema (role subset + model per role + knobs) — no per-shape code fork — proven by rendering both shapes into byte-diffed goldens like t13.
- Selecting the shape is easy and safe: a flag, a wizard, or a profile set on lobes init — something easy to set up without doing damage. Dry-run by default, --apply to commit, recoverable by re-running init with the previous shape.
  - instruction: Implement shape selection on lobes init behind the existing dry-run/--apply mutation-safety contract; verify via h2: diff shown on dry-run, zero bytes changed, --apply commits, re-init with the previous shape restores it byte-for-byte.
  - honesty: A dry-run shape change prints the full compose/.env diff and changes zero bytes on disk; --apply is required to commit; re-running init with the previous shape restores the previous rendering byte-for-byte.
- A dropped lobe is dropped honestly end-to-end: the box capabilities omit it and the gateway 4xxes requests for it (reusing the #92 invariant and #108-t6 machinery) — never half-served, never silently rerouted to a different model.
  - instruction: Wire dropped-role honesty through the existing #92 / #108-t6 path: capabilities omission plus gateway 4xx, with a contract test per dropped role.
  - honesty: With senses dropped on the Spark, GET /capabilities omits senses and a POST with model=senses or model=multimodal returns 4xx — verified live on the box and encoded as a contract test.
- Freed budget is actually reclaimed: when senses leaves the Spark, the remaining lobes budgets are re-derived (cortex util/context can rise) rather than staying at co-residency values — otherwise specialization buys nothing.
  - instruction: Re-derive budgets in the shape overlay (spark-lobe raises cortex util/context above 0.30/131072); measure on the GB10 and record the numbers in the docs.
  - honesty: The spark mesh-lobe rendered .env gives cortex a strictly larger budget than the co-resident values (util 0.30 / 131072), and the box boots it healthily — measured on the GB10, not asserted.

## Honesty conditions

- Both commands exist and are safe: on a supported box, bare lobes init yields machine-as-brain and lobes init --shape <mesh-lobe> yields that box's mesh shape — both dry-run by default, proven live on the GB10 and the Thor plus goldens in CI.
- Every named audience is actually served: the one-box default path stays zero-decision (goldens byte-identical), the Spark and Thor shapes are validated on their physical boxes, and consuming agents keep addressing each box exactly as the Culture mesh does today.
- The before-state is checkable in the shipped tree: the four core roles carry no profiles: stanza in lobes/templates/fleet/docker-compose.yml (only minor/middle/multimodal-coder do), and #109 is closed with the GB10 evidence recorded on the issue.
- Both near-term shapes boot on their physical boxes with zero hand-edits: spark mesh-lobe (no Gemma) on the GB10 and thor mesh-lobe (no Qwen) on the Thor, every remaining role passing its correctness probe.
- The bare-init machine-as-brain path is regression-proof: rendering on a recognised card is byte-identical before and after this change (goldens unchanged).
- The tax numbers are the shipped values, not estimates: MULTIMODAL_MAX_MODEL_LEN=32768, PRIMARY_MAX_MODEL_LEN=131072 vs 262144 solo-native, and the 0.30+0.14+0.06+0.06=0.56 GPU budget — all readable in the deployed env.
- Scope holds in the diff: the PR ships shape selection plus the two mesh-lobe shapes only — no one-lobe-per-box rendering, no Orin profile, no re-doing the #108 substrate — anything beyond routes to #112.
- The signal is runnable, not prose: a documented command sequence (init --shape, dry-run diff, --apply, correctness probes, capabilities check) executes on both boxes and its transcript is recorded in the PR.

## Success signals

- On the Spark, one safe command (dry-run shows the full diff, --apply commits) moves the box from machine-as-brain to its mesh-lobe shape: Gemma is gone, every remaining role is up and passes its correctness probe, and capabilities are honest — senses is absent and a request for it gets a 4xx (#92 invariant). The mirror move on Thor drops the Qwen cortex. A one-box user runs bare lobes init and gets machine-as-brain with zero extra decisions.
  - instruction: Automate the success signal as the acceptance script run on both boxes; attach the transcripts to the PR.

## Scope / boundaries

- The one-lobe-per-box end-state (Qwen-only Spark, Gemma-only Thor, small models on Orin 64GB) is tracked as #112 and stays a future PR — this change only makes deployment shape a safe first-class choice and ships the first two mesh-lobe shapes. The #108 knob/feasibility/golden machinery is the substrate, not re-done here. Orin ships nothing until someone boots one.
  - instruction: Enforce at PR review: the diff touches shape selection plus the two shapes only; anything one-lobe-per-box or Orin routes to #112.

## Assumptions

- The #108 implementation (t1 profile schema, t6 feasibility honesty, t13 goldens) lands before or with this work; this spec composes on top of it rather than duplicating it.

## Decisions

- machine-as-brain stays the default: bare lobes init on any box yields the whole brain that box can serve; mesh-brain shapes are opt-in per box.
- Cross-box story is per-box honesty only for this change: each box advertises only the lobes it hosts and consumers address each box directly (how the Culture mesh already connects per machine); gateway proxying of absent roles and a brain-level capabilities view are deferred to the #112 design work.
- Shape selection ships as a flag first: lobes init --shape <machine-as-brain|spark-lobe|thor-lobe>, dry-run by default, --apply to commit; an interactive wizard is a parked follow-up, not part of this change.

## Hard questions

- RESOLVED (user decision, see Decisions): when a box drops a lobe, is the cross-box story per-box honesty only, or does the gateway proxy absent roles to the peer box that hosts them? → Per-box honesty only for this change; proxying and the brain-level capabilities view are deferred to the #112 design work.

## Open / follow-up

- One-lobe-per-box end-state (Qwen-only Spark, Gemma-only Thor 128GB, small models on Orin 64GB) — tracked as agentculture/lobes-cli#112, a future PR.
- Jetson AGX Orin 64GB small-model lobe — named target, unvalidated card until someone boots one (per the #108 rule).
- Interactive shape-selection wizard on lobes init (UX sugar over the --shape flag).
