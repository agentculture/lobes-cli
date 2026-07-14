# Build Plan — lobes serves the brain shape you choose: machine-as-brain keeps the whole brain on one box; mesh-brain gives each box the lobes it is best at — the Spark drops Gemma, Thor drops Qwen — one safe command either way

slug: `lobes-serves-the-brain-shape-you-choose-machine-as` · status: `exported` · from frame: `lobes-serves-the-brain-shape-you-choose-machine-as`

> lobes serves the brain shape you choose: machine-as-brain keeps the whole brain on one box; mesh-brain gives each box the lobes it is best at — the Spark drops Gemma, Thor drops Qwen — one safe command either way

## Tasks

### t1 — Shape schema + built-in shapes: a Shape declares the role subset a box hosts (machine-as-brain = every role the card can serve; spark-lobe = cortex+embedder+reranker+audio, no senses; thor-lobe = senses+embedder+reranker+audio, no cortex) as pure data composed over the #108 Profile schema

- instruction: OWNS: lobes/profiles/ shape schema + builtin shape data files (machine-as-brain, spark-lobe, thor-lobe) + tests/test_shapes.py. Touch NOTHING else. Compose over the LANDED #110 substrate: lobes/profiles/schema.py (Profile/RoleProfile, ROLES, KNOB_NAMES) + loader.py (resolve_profile, operator TOML overrides). Shape data files are TOML, matching lobes/profiles/builtin/. Per user decision: shapes own all SIX Colleague roles — the four core roles map onto Profile machinery; stt/tts are first-class shape members mapping onto the audio overlay (lobes/templates/fleet/docker-compose.audio.yml surfaces).
- covers: c4, c9, h1
- acceptance:
  - A Shape round-trips load -> serialise -> identical; an unknown role in a shape is a load error, not a silent drop
  - All three built-in shapes are expressible as data files with zero per-shape python forks; spark-lobe and thor-lobe differ from machine-as-brain only by role subset and budget overrides
  - stt/tts are expressible shape members: spark-lobe and thor-lobe both declare them hosted, and the schema maps the four core roles to Profile entries and the audio pair to the overlay — an unknown role in a shape is still a load error

### t2 — Shape-aware budget re-derivation: a shape that drops a lobe re-derives the remaining lobes' budgets (spark-lobe raises cortex util/context above the co-resident 0.30/131072) as declared shape overrides, never runtime mutation

- instruction: OWNS: the budget-override fields inside t1's shape data files + their tests. Sequenced after t1 (same files — never same-wave). Every derived value carries a provenance comment naming its cause; machine-as-brain values stay byte-identical to shipped.
- depends on: t1
- covers: c12, h4
- acceptance:
  - The spark-lobe rendered .env gives cortex a strictly larger budget than the co-resident values (util > 0.30 or max-model-len > 131072), stated as data in the shape with provenance
  - machine-as-brain budgets are byte-identical to today's shipped values

### t3 — Render composition shape x card: rendering a (shape, card-profile) pair yields compose/.env; per-(shape,card) goldens checked into tests, rendering a pure function of (shape, profile, template) with no host state

- instruction: OWNS: the shape-x-card render glue + tests/goldens/<shape>-<card>/ + tests/test_shape_goldens.py. Touch NO CLI. Build on the LANDED #110 renderer (lobes/profiles/render.py:profile_env; infeasible role renders <PREFIX>_FEASIBLE=false) and follow the existing tests/goldens/*.env + regen.py convention — the nested per-(shape,card) goldens are additive, do not break test_profile_goldens.py. Rendering maps the four core roles via profile_env and stt/tts via the audio overlay env/compose include; stays a pure function of (shape, profile, template) — no GPU, no host reads.
- depends on: t1
- covers: c9, h1, c5, h6, c2
- acceptance:
  - The spark-lobe golden contains no senses service; the thor-lobe golden contains no cortex service; the machine-as-brain golden is byte-identical to the pre-change rendering on a recognised card
  - A change to one shape's data leaves every other (shape,card) golden byte-identical, enforced by CI on a GPU-less runner

### t4 — lobes init --shape <machine-as-brain|spark-lobe|thor-lobe>: flag-first selection behind the existing dry-run/--apply mutation-safety contract; bare init stays machine-as-brain with zero new decisions

- instruction: OWNS: lobes/cli/_commands/init.py + its tests. Keep the existing dry-run-by-default/--apply mutation-safety contract; consume t3's render API, do not re-implement rendering; bare init path byte-identical.
- depends on: t3
- covers: c1, h7, c10, h2
- acceptance:
  - Dry-run prints the full compose/.env diff and changes zero bytes on disk; --apply is required to commit; re-running init with the previous shape restores the previous rendering byte-for-byte
  - Bare lobes init renders machine-as-brain exactly as before this change (golden unchanged); --shape with an unknown value is a user error naming the valid shapes

### t5 — Dropped-lobe honesty end-to-end: a role the shape drops is omitted from lobes capabilities and GET /capabilities, and the gateway returns 4xx for requests addressing it (model=senses/multimodal on spark-lobe; model=cortex/main on thor-lobe) — reusing the #92 invariant and #108-t6 machinery, one contract test per dropped role

- instruction: OWNS: lobes/gateway/ + the capabilities command + contract tests. Target the SHIPPED #110 machinery exactly: _config.py (FEASIBLE_ENV), _routing.py (RoutingTable.infeasible, list_models_payload), server.py (_feasibility_response -> 404 role_infeasible), roles.py (RoleInfo.feasible), cli/_commands/capabilities.py. Shipped contract: capabilities FLAG a dropped role (feasible:false, annotated), /v1/models omits it, generate lane returns 404 role_infeasible. The corrective acceptance criterion below supersedes the earlier 'omitted from capabilities / generic 4xx' phrasing.
- depends on: t1
- covers: c11, h3, c17, h13
- acceptance:
  - With spark-lobe rendered, capabilities omit senses and a POST with model=senses or model=multimodal returns 4xx (contract test); mirror assertions for thor-lobe and cortex/main
  - No dropped role is ever silently rerouted to a different model — the 4xx names the role as not hosted on this box
  - SUPERSEDES the omit-wording above: with spark-lobe rendered, capabilities show senses feasible:false, /v1/models omits senses, and POST model=senses or model=multimodal returns 404 role_infeasible (contract test); mirror assertions for thor-lobe and cortex/main

### t6 — Acceptance script + GB10 validation: a documented, runnable command sequence (init --shape spark-lobe, dry-run diff, --apply, correctness probes, capabilities check) executed on the physical Spark; senses gone, every remaining role passes its probe, reclaimed cortex budget measured and recorded

- instruction: Validation ON the GB10 (this box). Author the acceptance script (init --shape spark-lobe, dry-run diff, --apply, probes, capabilities check); move ~/.lobes aside first so the run starts from a clean deployment dir; record the transcript and the measured reclaimed cortex budget in the PR.
- depends on: t2, t4, t5
- covers: c8, h12, h5, h4, h8, c1, c4
- acceptance:
  - The script runs end-to-end on the GB10 with zero hand-edits to the generated compose; its transcript is attached to the PR
  - Live on the box: GET /capabilities omits senses and model=senses returns 4xx; every hosted role passes its correctness probe
  - The measured reclaimed cortex budget (util/context) is recorded with numbers, not asserted
  - Live on the box, per the shipped #110 surface (supersedes the omit/4xx phrasing above): GET /capabilities shows senses feasible:false, GET /v1/models omits it, POST model=senses returns 404 role_infeasible

### t7 — Thor validation: the same acceptance script with --shape thor-lobe on the physical Thor; the Qwen cortex is gone, senses + the pooling gears are up and pass their probes

- instruction: Validation ON the physical Thor — cannot run from the Spark; coordinate with the Thor operator. Re-run t6's script with --shape thor-lobe from a clean deployment dir; transcript to the PR; rerank ordering remains an expected failure only while #105/#106 stay open.
- depends on: t4, t5
- covers: h5, h12, c4, h8
- acceptance:
  - The script runs end-to-end on the Thor with zero hand-edits; transcript attached to the PR; capabilities omit cortex and model=cortex/main returns 4xx
  - Senses and the pooling gears pass their correctness probes (rerank ordering stays an expected failure only if #105/#106 are still open, recorded as such)
  - Live on the Thor, per the shipped #110 surface (supersedes the omit/4xx phrasing above): capabilities show cortex feasible:false, /v1/models omits it, POST model=cortex or model=main returns 404 role_infeasible

### t8 — Docs + evidence: docs/deployment-shapes.md (the shape reference, support table, co-residency tax numbers, before-state citation of the unconditional core roles and closed #109), lobes explain shapes, CLAUDE.md update, and the PR scope checklist enforcing the boundary (shape selection + two shapes only; anything one-lobe-per-box or Orin routes to #112)

- instruction: OWNS: docs/deployment-shapes.md (new) + lobes/explain/ + CLAUDE.md + the PR scope checklist. Docs only, no source. Quote shipped tax values; cite the template lines and closed #109 as before-state evidence; route anything one-lobe-per-box or Orin to #112.
- depends on: t6, t7
- covers: c2, h8, c3, h9, c6, h10, c7, h11
- acceptance:
  - docs quote the shipped tax values (senses 32K, cortex 128K vs 256K solo-native, 0.56 budget) and cite the template lines + closed #109 as before-state evidence
  - The PR description carries the scope checklist; review rejects any diff beyond shape selection + the two shapes
  - lobes explain shapes renders the same reference in-CLI

## Risks

- [unknown_nonblocking] The #108 implementation (t1 profile schema, t6 feasibility honesty, t13 goldens) has not landed yet — this plan's t1/t3/t5 compose on that machinery; build alongside or after it and re-check the composition surface when it lands
- [unknown_nonblocking] t7 needs hands on the physical Thor box — it cannot run from the Spark; schedule with the Thor operator
- [unknown_nonblocking] Exact reclaimed cortex numbers on spark-lobe (256K native? higher util?) are measured in t6, not specced
- [follow_up] Interactive shape wizard and gateway proxying of absent roles are parked follow-ups (frame v4, #112)
