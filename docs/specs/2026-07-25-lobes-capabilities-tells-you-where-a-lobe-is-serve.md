# lobes capabilities tells you WHERE a lobe is served — a proxied role reads by-proxy, never a bare loaded:no

> lobes capabilities tells you WHERE a lobe is served — a proxied role reads by-proxy, never a bare loaded:no
> instruction: Render a third state in lobes/cli/_commands/capabilities.py _render_table: a role with feasible=false AND proxied=true prints 'by-proxy' in the loaded column with its hosted_by peer, instead of yes/no. Derive it from the payload's existing proxied + hosted_by keys — do NOT change the gateway JSON. Redeem the stale comment at capabilities.py:289-292 while you are there.

## Audience

- Operators reading 'lobes capabilities' / 'lobes fleet status' on a mesh-brain box, and programmatic consumers of gateway GET /capabilities (Colleague, webcam-cli, reachy-mini-cli) that decide from the payload whether a role is usable and where it lives.

## Before → After

- Before: Verified live on spark 2026-07-25: senses and muse are BOTH dropped-and-proxied and BOTH serve 200 through this gateway (muse proved end-to-end: X-Lobes-Proxied-By thor, plus a clean tool_calls response). Yet 'lobes capabilities' prints senses loaded=yes and muse loaded=no. The difference is not health, reachability, or the peer — it is only whether <PREFIX>_BASE_URL happens to be set in this box's local .env.
- After: A proxied role reads 'by-proxy' in the capabilities table instead of a bare yes/no, and names its hosting peer. loaded stops being the field an operator consults to answer 'is this usable' for a role this box does not host. Both senses and muse read identically for identical situations — the display no longer depends on leftover local wiring.

## Why it matters

- 'loaded' is the ONLY yes/no column in the capabilities table, so operators read it as 'is this role usable'. For a proxied role it answers a different question (did this box wire a local backend), so it is both wrong-looking (muse:no while muse serves) and misleading (senses:yes while spark runs no senses container at all). An operator cannot tell a working proxied lobe from a dead one.

## Requirements

- A role this box serves via a peer reads 'by-proxy' (not yes, not no) in the capabilities table, and names the hosting peer alongside it.
  - instruction: Widen the loaded column to fit 'by-proxy', keep 'yes'/'no' for local/absent roles, and cover all three states in tests/ against a fixture payload. The offline (non-gateway) path has no proxied key — default it so an older or hand-built payload never raises, matching the existing .get(feasible, True) convention two lines below.
  - honesty: A programmatic consumer of GET /capabilities that today branches on loaded as a JSON boolean keeps working across the upgrade, OR the break is deliberate, named, and coordinated with the known consumers (Colleague, webcam-cli, reachy-mini-cli) the way the #151 wire break was.
  - honesty: The by-proxy state is derived from the SAME signals the gateway already sends (proxied + hosted_by), so the CLI and the gateway can never disagree about whether a role is proxied.
- Spark's stale local wiring is corrected so senses' displayed state no longer depends on a leftover MULTIMODAL_BASE_URL pointing at a compose service this box does not run.
  - instruction: On spark ~/.lobes/.env comment out MULTIMODAL_BASE_URL (keep MULTIMODAL_SERVED_NAME — the peer probe falls back to it), then recreate ONLY the gateway with: docker compose -f docker-compose.yml -f docker-compose.shape.yml up -d --no-deps gateway. Verify with a live model=senses POST expecting 200 + X-Lobes-Proxied-By orin, and confirm vllm-multimodal did not boot and cortex was not recreated.
  - honesty: After the wiring correction, a live POST for model=senses through spark still returns 200 with X-Lobes-Proxied-By orin — proving the peer probe still resolves the same served id and the proxy is still armed.
  - honesty: The correction is applied without booting the dropped vllm-multimodal container or recreating cortex — i.e. the gateway restart passes -f docker-compose.yml -f docker-compose.shape.yml, per the recorded shape-override trap.

## Honesty conditions

- The table distinguishes THREE readings an operator can act on — locally served, served by a named peer, and genuinely absent — visible in one command's output, rather than collapsing the last two onto the same 'no'.
- Both audiences are actually reached: the CLI table is what an operator runs, and at least one named programmatic consumer genuinely reads GET /capabilities to decide role usability (checkable in that consumer's source) — so a CLI-only change is verified sufficient for the operator audience and explicitly NOT claimed to change anything for the payload audience.
- The asymmetry is reproducible on demand, not a one-off: toggling <PREFIX>_BASE_URL alone flips the affected role's loaded value with no change on the peer side, proving the display tracks local wiring rather than the peer's real state.
- Demonstrable today: two roles in the SAME proxied state, both returning 200 through this gateway, print DIFFERENT loaded values — and no other column in that table distinguishes a working proxied lobe from a dead one.
- After the change a role's displayed state is a function of (feasible, proxied, hosted_by) ONLY and never of whether <PREFIX>_BASE_URL is set — provable by toggling that var and observing the displayed state does not move.
- No file on a routing or auth path is modified — the diff is confined to the CLI renderer (and its tests/docs) — and a live model=muse and model=senses POST return byte-identical results before and after, headers included.
- The success check runs as ONE scripted probe on spark that emits both the capabilities table and the live proxied 200s into a single transcript, so it is re-runnable evidence rather than a claim.

## Success signals

- On spark, after the change: senses and muse BOTH read by-proxy with their hosting peer named (orin / thor), while cortex/embedder/reranker still read as locally served; and a live POST for model=muse still returns 200 with X-Lobes-Proxied-By thor, proving the display change moved no traffic.

## Scope / boundaries

- Scoped to how a role's serving LOCATION is reported (capabilities table + payload) and to correcting spark's stale local wiring. NOT a change to routing, proxy behaviour, auth, or which box hosts what — every request that works today keeps working byte-identically.

## Non-goals

- Not redefining 'ready', and not touching the audio-role readiness probe — that is issue #155's converged-but-unmerged spec on branch spec/stt-readiness-truth-155, which targets the SAME file (lobes/roles.py) and the SAME CLI renderer but a different field. This work must compose with it, not pre-empt or duplicate it.
- Not upgrading Thor. Thor runs gateway 0.46.0 vs repo 0.54.1 and muse has served 8 days with 0 restarts; the version drift is real but is a separate operational decision, not part of this contract change.

## Assumptions

- ROOT CAUSE, read from source, not inferred: loaded = (backend is not None) at lobes/roles.py:455 — a pure LOCAL wiring fact that never consults the peer. Both roles reach _optional_backend (_config.py:514/530), which returns None when <PREFIX>_BASE_URL is empty. Spark's .env still sets MULTIMODAL_BASE_URL (leftover from when spark hosted senses) but leaves MUSE_BASE_URL empty, so senses gets a Backend object and muse does not. The peer's real state rides on 'ready', which is true for both.
- The two roles were dropped by DIFFERENT mechanisms, which is why their loaded values differ: senses was dropped by the explicit veto MULTIMODAL_FEASIBLE=false, which sets infeasible but leaves the wiring intact; muse was never wired at all, and because muse is in OPT_IN_BACKENDS (_config.py:90) an unwired opt-in lobe defaults to infeasible with no FEASIBLE var needed (_config.py:315-316).
- The gateway payload ALREADY carries proxied:true and hosted_by for exactly these roles (verified live in spark's GET /capabilities for both senses and muse), so a by-proxy DISPLAY state is derivable today with no change to the wire contract at all.

## Decisions

- Correcting spark's wiring is SAFE for the senses proxy, verified by reading the resolution order at server.py:726-763: with MULTIMODAL_BASE_URL removed the peer served-name falls to step 2 (MULTIMODAL_SERVED_NAME, which IS set to coolthor/gemma-4-12B-it-NVFP4A16) and step 3 (catalog canonical) resolves to the SAME id — and orin's /v1/models serves exactly that id. So the peer probe keeps passing and the proxy stays armed.
- USER DECISION (q1): by-proxy is a CLI DISPLAY state only — zero wire change. The capabilities table renders 'by-proxy' derived from the proxied + hosted_by fields the gateway payload already carries; GET /capabilities JSON keeps loaded as a bool and gains no new key. Rationale: it delivers the operator clarity asked for without breaking any consumer that branches on loaded as a boolean, and the wiring fix (c14) independently makes the payload self-consistent, since both proxied roles then report loaded:false + proxied:true.
- USER DECISION: spark's stale wiring is corrected NOW, applied shape-override-safe — the gateway container is recreated with -f docker-compose.yml -f docker-compose.shape.yml so the dropped vllm-multimodal never boots and the 27B cortex is never recreated. Verified afterwards by a live model=senses POST returning 200 with X-Lobes-Proxied-By orin.
