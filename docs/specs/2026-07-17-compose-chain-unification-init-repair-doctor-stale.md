# compose-chain unification, init repair, doctor staleness, audio peer-proxy (#137/#138/#119/#129)

> lobes closes the four post-#127 gaps in one sweep: every compose invocation — CLI verb or acceptance script — resolves the same overlay chain from one shared builder, so switch/serve/restore can never boot a shape-dropped lobe; lobes init heals a partial scaffold missing-only without touching operator state; lobes doctor flags a deployment predating its card profile's required knobs; and stt/tts join the honest-referral/proxy contract, so the Spark serves tts by proxying to the Thor Chatterbox under the same auth, loop-guard, and attribution guarantees as every other lobe

## Audience

- operators of the live mesh boxes (Spark GB10, Thor, Orin) running lobes init/doctor/fleet/accept-shape; agents (Claude and the colleague backend) that call the lobes CLI in loops and rely on dry-run safety; OpenAI-compatible clients of the gateway audio lanes (reachy-mini-cli, Culture agents)

## Before → After

- After: one builder decides the compose chain everywhere — CLI verbs and acceptance scripts agree by construction; the recovery paths are shape-faithful; a partial or stale scaffold is detected AND healed by doctor (--fix, missing-only, never touching operator state); stt/tts are first-class roles with honest referral and proxy, so the Spark serves tts from the Thor with attribution, loop guard, and pairwise auth; every lobes-shipped client attaches the gateway bearer when inbound auth is on

## Why it matters

- three of the four gaps were found live on the deployed pair: restore — the recovery path — boots the very lobe a shape dropped; a partial scaffold serves for hours with /health green and no heal path short of clobbering the operator .env (gateway key, peer config, reclaim values); and pointing AUDIO_URL at a peer leaks the caller credential — the trust (#127) and honesty (#92/#112) properties already shipped for the five core roles do not yet cover audio or the operational tooling

## Requirements

- #137: one parameterised -f chain builder — runtime/_compose.py::_compose_files stays the single source of truth (base + audio + shape + operator override; [] for a plain fleet), and up.py::_compose_file_args folds in as a parameter for its genuinely different semantics (needs_audio = the TARGETED services need audio, not merely overlay-exists) instead of remaining a parallel copy
  - instruction: parameterise _compose_files with an audio-inclusion mode (overlay-present vs targeted-services-need-audio) and have up.py::_compose_file_args delegate; keep the shape-last + operator-override-last ordering documented in the builder docstring
  - honesty: after the change no second -f chain construction exists in Python — up.py delegates to the parameterised builder and its targeted-services semantics survive its existing #136 tests
- #137: every mutating caller resolves through the shared builder — compose_up_detached (hardcoded bare docker compose up -d, runtime/_compose.py:532) goes through it, fixing switch.py:412-413 (down with the full chain, up with none) AND serve.py:48, which since #69 runs fleet deployments, so lobes serve --apply on a shape deployment boots the dropped lobe today (new finding, not in the #137 inventory)
  - instruction: route compose_up_detached through _compose_files (a plain fleet resolves [] so the bare-argv path is byte-identical); add a regression test for serve-on-shape and note the serve.py:48 finding on issue #137
  - honesty: compose_up_detached resolves the same file set as compose_down for every overlay combination, and a shape deployment brought up via lobes serve --apply starts no shape-dropped service
- #137/#138: scripts consume the chain from the CLI — a read-only helper emits the resolved -f list so accept-shape.sh (_compose_down line 106, --restore line 161) and validate-tiers.sh (_compose_args line 94) cannot drift again by construction
  - instruction: add a read-only CLI surface that prints the resolved -f args one per line for a deployment dir (e.g. a fleet files/--compose-files flag on an existing read verb); scripts mapfile it; document it in docs/gateway-fleet.md or docs/deployment-shapes.md
  - honesty: accept-shape.sh and validate-tiers.sh contain no hand-rolled -f list: they consume the helper output and fail loudly when it is unavailable
- #138: accept-shape.sh --restore includes docker-compose.shape.yml, so restoring a spark-lobe/thor-lobe/thor-muse backup no longer boots the very lobe the shape dropped (restore is the recovery path); validate-tiers.sh _recreate_gateway likewise recreates the gateway with the shape overlay so results describe the topology the operator actually runs
  - instruction: both scripts consume the c4 helper; the stopgap hand-append is acceptable only if the helper cannot land in the same PR (it can — same repo)
  - honesty: a restored spark-lobe/thor-lobe/thor-muse backup starts no dropped service and the recreated gateway carries no dangling depends_on — asserted by a test on the emitted chain and verified in the next live accept-shape run
- #137 (the #136 lesson): one test asserts that every caller — _compose_files, the up.py parameterisation, compose_up_detached / compose_up_build / compose_down, and the script-facing helper — resolves the identical file set for the same deployment; _compose_file_args drifted precisely because it had no tests
  - instruction: parametrised pytest over tmp_path deployments with the overlay files present/absent; compare resolved lists, not behaviors
  - honesty: one test enumerates every caller — _compose_files, the up.py parameterisation, compose_up_detached/compose_up_build/compose_down argv, the script-facing helper — and asserts identical file sets across the overlay matrix: none, audio, shape, audio+shape, each with and without the operator override
- #119: the repair env append skips keys already present — docker compose env_file semantics let the LAST duplicate key win, so appending a blank key (e.g. HF_CACHE=) over a set one silently clobbers it (the 2026-07-17 Spark repair gotcha)
  - instruction: parse existing keys first (reuse lobes.runtime._env reading); append only absent keys from the template; never reorder or rewrite existing lines
  - honesty: re-running the heal on an already-complete .env is a byte-identical no-op, and a set key (e.g. HF_CACHE) survives a heal that appends other keys
- #119: lobes doctor gains a profile-staleness check — resolves the machine profile exactly as lobes init does, renders it pure, and diffs required divergence knobs AND missing scaffold files against the deployment (the Spark failure mode was files silently absent with /health green); remediation names the repair mode, never lobes init --apply --force; operator-set lines downgrade to info; read-only like the existing docker/compose_present/env_coherence/health/version_skew checks (doctor.py:229-247)
  - instruction: new doctor check(s) using resolve_init_profile + render_shape + scaffold_plan against the deployed dir; severity warning for missing keys/files, info for value differences; remediation text names doctor --fix, never init --force
  - honesty: the check resolves the profile via the SAME code path init uses (no forked render); it flags missing required knobs and missing scaffold files; an operator-set differing value downgrades to info; the 2026-07-14 Thor stale .env and the 2026-07-17 Spark partial scaffold are reproduced as fixtures and both caught
- #129 items 1-2: the auth stragglers route through the shared header helper (cli/_runtime_ops.py:80 gateway_auth_headers) — roles_measure.py audio probes (lines 277, 357) and minor/_client.py (lines 119, 242) build keyless urllib requests today and degrade opaquely when inbound gateway auth is enabled
  - instruction: thread the gateway_auth_headers result (or the resolved header dict) into roles_measure audio probes and minor/_client request builders; cover with one auth-on test each
  - honesty: with an inbound gateway key set, the stt/tts measure probes and the minor-client calls succeed with the bearer attached; with no key configured their requests are byte-identical to today
- #129 item 3: stt/tts gain peer channels — STT_/TTS_PEER_ORIGIN / _PEER_PROXY / _PEER_API_KEY with the same three-condition arming (truthy knob + declared origin + not locally served) and hosted_by annotation on lobes capabilities, GET /capabilities, and the audio 404 body; today PEER_*_ENV (gateway/_config.py:113-171) covers only the five core backends, ROLE_BACKEND (roles.py:65) has no stt/tts, and _audio_role (roles.py:401) takes no peer argument
  - instruction: extend PEER_ORIGIN_ENV/PEER_PROXY_ENV/PEER_API_KEY_ENV and the routing-table composition; add stt/tts to ROLE_BACKEND and give _audio_role the peer argument so annotate_peer_referrals covers all seven roles
  - honesty: STT_/TTS_ peer channels behave identically to the five core prefixes — same _as_bool parsing, same three-condition arming, same repr=False secrecy — and capabilities plus the audio-role 404 body name hosted_by for a declared audio peer
- #129: audio peer routing is per-endpoint — /v1/audio/speech (tts) and /v1/audio/transcriptions (stt) move independently, unlike AUDIO_URL which path-routes the whole /v1/audio/* namespace to one backend (_routing.py:98-105); the live ask is exactly the split case: Chatterbox on the Thor, Parakeet staying local on the Spark
  - instruction: route the two audio endpoints to per-role backends instead of the single AUDIO_URL namespace match; AUDIO_URL keeps serving as the local default for both (c14)
  - honesty: tts-remote + stt-local is expressible and tested: /v1/audio/speech forwards to the declared tts peer while /v1/audio/transcriptions serves locally in the same deployment, and vice versa
- #129: audio forwards reuse the proxy-lobes data-plane guarantees — strip the caller Authorization and inject the peer key (server.py:906 already does this for core roles; handle_audio_post filter_headers at server.py:1467 leaks it today), single-hop X-Lobes-Proxied guard with 508 proxy_loop, and X-Lobes-Proxied-By attribution — so the four AUDIO_URL contract violations recorded on #129 become impossible on the new lane
  - instruction: reuse the existing _proxy_to_peer machinery (strip + inject + guard + attribution) for the audio endpoints rather than extending handle_audio_post filter_headers forwarding
  - honesty: on the audio forward path the caller Authorization never reaches the peer; a request already marked X-Lobes-Proxied is refused 508 proxy_loop; every proxied audio answer carries X-Lobes-Proxied-By — each asserted by a test
- q3 decision: the heal path belongs to doctor, not init — no init --repair. lobes doctor gains --fix: missing-only repair that writes only absent scaffold files and appends only absent .env keys, never rewriting an existing line. Per the repo mutation-safety rule the fix lane is dry-run by default — doctor --fix prints the fix plan, doctor --fix --apply commits it (agents call CLIs in loops; safe-by-default is mandatory)
  - instruction: fix plan derives from the same findings the read-only check reports; write missing template files via the existing template readers; append missing env keys per c8; refuse to touch a file that exists
  - honesty: doctor --fix without --apply mutates nothing (asserted by test); --fix --apply writes only files and keys absent beforehand; the reproduced Spark case (missing Dockerfile.chatterbox + Dockerfile.realtime + all audio env keys) heals to a working overlay with every pre-existing .env line untouched
- q2 decision: delete shape_render.py::shape_compose_files and the ShapeRender.compose_files field (dead — computed by render_shape but read only by tests); the shared builder in runtime/_compose.py is the only chain authority, and the tests that read the dead field are updated or removed with it
  - instruction: delete shape_compose_files + the field + its tests in the #137 PR
  - honesty: shape_render exports no compose-files API afterwards; the ShapeRender dataclass loses the field; grep finds no remaining reference outside git history
- q1 decision: stt/tts are promoted to FIRST-CLASS backends — they join the Profile/RoutingTable backend schema alongside primary/multimodal/muse/embed/rerank, so feasibility, the STT_/TTS_PEER_ORIGIN / _PEER_PROXY / _PEER_API_KEY channels, hosted_by annotation, and /capabilities honesty all ride the one existing mechanism (ROLE_BACKEND gains stt/tts; _audio_role gains the peer argument)
  - instruction: first-class means the ROUTING/peer/capabilities mechanism, not new tuning knobs: no per-card audio entries in profile TOMLs (c25); goldens prove the no-op
  - honesty: stt/tts appear in the routing/peer schema with feasibility and peer channels, while machine-as-brain and every existing no-audio deployment render byte-identically — the shape goldens and capabilities output for existing deployments are unchanged
- q4 decision: delivery is four sequential PRs in dependency order #137 -> #138 -> #119 -> #129; for each PR Claude tracks Qodo, Copilot, and SonarCloud, fixes valid findings, pushes back on invalid ones with reasoning, resolves every thread, and merges before the next PR branch starts (every PR version-bumped per the every-PR-bumps rule)
  - instruction: order: PR1 #137+#138 builder/helper/scripts (or #137 then #138 per plan), then #119 doctor, then #129 audio; track reviews via the cicd skill await/status
  - honesty: PR N+1 branches only after PR N is merged; every Qodo/Copilot/Sonar thread on each PR is resolved before its merge; every PR bumps the version (CI version-check green)

## Honesty conditions

- every announced behavior is demonstrated by a committed test or live transcript: the equivalence test, the shape-faithful restore, the doctor --fix heal of a partial scaffold, and the Spark-to-Thor TTS round-trip under docs/evidence/ — and all four issues close with merged PRs
- the no-overlay argv is byte-identical before and after the change for every verb, preserving compose auto-discovery of the operator override
- plain doctor performs zero writes even when findings exist, and the --fix dry-run names every file and key it would touch before --apply commits
- the named consumers exercise the delivered surfaces directly: operators via init/doctor/fleet/accept-shape, agents via the dry-run-default verbs, clients via an unchanged OpenAI audio API
- each after-state clause maps to a merged PR in the four-PR sequence; none is aspirational at close
- the three live incidents cited are real and reproduced as fixtures/tests before being fixed (Thor rerank hang 2026-07-14; Spark partial scaffold 2026-07-17; AUDIO_URL credential-leak mechanics confirmed in code)
- the success signals are observable artifacts (tests, transcripts, merged PRs with resolved threads), not self-assessments

## Success signals

- the cross-caller equivalence test is green across the overlay matrix; accept-shape.sh --restore on a mesh-shape backup boots zero dropped services; lobes doctor detects and --fix --apply heals a reproduction of the 2026-07-17 Spark partial scaffold; a live Spark-to-Thor /v1/audio/speech round-trip returns audio with X-Lobes-Proxied-By naming the Thor while /v1/audio/transcriptions stays local (evidence under docs/evidence/); four sequential PRs merged with every review thread resolved

## Scope / boundaries

- a plain fleet (no lobes overlay) keeps resolving [] — compose resolves the project itself and its own convention layers base + docker-compose.override.yml (the documented _compose_files contract, runtime/_compose.py:464-501); machine-as-brain stays a zero-new-decisions byte-identical no-op
  - instruction: assert [] from the builder for a plain fleet and bare argv from compose_up_detached/compose_down in the equivalence test
- plain lobes doctor and the chain-emitting helper stay read-only; doctor --fix is the single write lane on doctor and follows the repo write-verb convention (dry-run by default, --apply to commit); lobes init keeps exactly its current refuse/--force contract — no new init modes
  - instruction: keep the check/fix split in doctor.py: checks return findings; --fix consumes findings; --apply gates the writes

## Non-goals

- #119 own out-of-scope note stands: no continuous ready-time ordering probe for pooling lanes here (its own issue if wanted); #107 tuned-small-model work untouched
- no validated claim for cross-box audio without a live acceptance transcript under docs/evidence/ (the #108 rule) — the Spark-to-Thor TTS run is the acceptance vehicle; docs say declared/unvalidated until it lands
- stt/tts stay out of the switchable catalog (lobes/catalog.py) and keep their fixed checkpoints — Parakeet and Chatterbox, no per-card tuning knobs — even as they become first-class roles in the routing/peer schema

## Assumptions

- AUDIO_URL remains the local-bridge lane (the gateway fans /v1/audio/* to the realtime bridge); an armed per-role audio peer channel takes precedence for its endpoint only; with no STT_/TTS_PEER_* knob set anywhere, every response stays byte-identical to today (the pre-#115 contract pattern)
- audio peer keys follow the O(machines) convention — the outbound <PREFIX>_PEER_API_KEY is a copy of the peer box own inbound GATEWAY_API_KEY, never minted per pairing (the #127 pairwise-auth rule, gateway/_config.py:153-171)

## Scope exploration

- `s1` — `lobes/runtime/_compose.py::_compose_files (464-501)`: the post-#135 source of truth: base + audio overlay + shape overlay + operator override, in that order, with [] for a plain fleet so compose auto-discovery keeps working; compose_down (528) and compose_up_build (536) resolve through it
  - seeds: `c2`, `c16`
- `s2` — `lobes/runtime/_compose.py::compose_up_detached (532) + its callers`: hardcoded bare docker compose up -d, never calls _compose_files; called by switch.py:413 AND serve.py:48 — and serve is fleet-by-default since #69 (its docstring says it brings up the duo), so the single-model-only safety note in the #137 inventory does not hold: lobes serve --apply on a shape deployment boots the dropped lobe, the same failure class as the #138 restore path
  - seeds: `c3`
- `s3` — `lobes/cli/_commands/up.py::_compose_file_args (157-183)`: the parallel builder with genuinely different semantics — needs_audio means the TARGETED services need the audio overlay, not overlay-exists — so consolidation must parameterise _compose_files, not blindly delegate; it gained tests only in #136 (test_cli_up.py, 6 references), which is why it drifted unnoticed before
  - seeds: `c2`, `c6`
- `s4` — `lobes/cli/_commands/switch.py::_apply_switch (412-413)`: tears down with the full chain (compose_down resolves via _compose_files) and brings up with none (compose_up_detached) — the same command disagrees with itself about which files the deployment is made of
  - seeds: `c3`
- `s5` — `lobes/profiles/shape_render.py::shape_compose_files (232-242)`: computed by render_shape (312) into ShapeRender.compose_files but never read by init.py (only rendered.env is consumed; verified — init.py has no compose_files reference) — a latent duplicate builder that ships the drift the day someone wires it up; it also omits the shape overlay and operator override
  - seeds: `c2`
- `s6` — `scripts/accept-shape.sh (_compose_down 106-117; --restore 153-169)`: both hand-rolled chains are base + audio + operator override with NO shape overlay; --restore then runs up -d, so a restored spark-lobe/thor-lobe/thor-muse backup boots the very lobe its shape dropped and re-introduces the gateway depends_on edge the shape !resets — the recovery path is the broken one; _compose_down is benign (down --remove-orphans removes more, not less)
  - seeds: `c4`, `c5`
- `s7` — `scripts/validate-tiers.sh::_compose_args (94-102) + _recreate_gateway (104-109)`: the header comment claims to mirror lobes.runtime._compose._compose_files but the chain omits the shape overlay, so _recreate_gateway force-recreates the gateway WITHOUT it on a shape deployment — the validation results describe a topology the operator does not run
  - seeds: `c4`, `c5`
- `s8` — `lobes/cli/_commands/init.py (_emit_apply 330-416, cmd_init 419-458) + runtime/_compose.py::write_scaffold (316-349)`: --force is a blanket flag over the whole template set, and .env is in every set (FLEET_TEMPLATES maps fleet/env.example to .env), so the only paths today are refuse-any-existing-file or clobber-everything; the deployed .env carries CULTURE_VLLM_API_KEY, the *_PEER_* proxy config, and the shape reclaim values — exactly the operator state #119 says to respect
  - seeds: `c21` (originally seeded `c7`, the init --repair proposal, rejected in favour of doctor --fix)
- `s9` — `lobes/runtime/_compose.py::append_audio_env (352-363)`: a blind append of env.audio.example onto .env with no already-present check; combined with docker compose env_file last-duplicate-wins semantics, a repair that re-appends would silently clobber set values (the HF_CACHE gotcha from the 2026-07-17 Spark repair)
  - seeds: `c8`
- `s10` — `lobes/cli/_commands/doctor.py (44-247)`: five checks today — docker, compose_present, env_coherence, health, version_skew — none profile-aware, none checks for missing scaffold files; matches the #119 incident where a pre-#110 .env served for weeks missing the thor profile required knobs with /health green, and the 2026-07-17 Spark case where scaffold files were silently absent
  - seeds: `c9`
- `s11` — `lobes/gateway/_config.py (100-171, 174-231)`: PEER_ORIGIN_ENV / PEER_PROXY_ENV / PEER_API_KEY_ENV cover exactly primary/multimodal/muse/embed/rerank; module comments explicitly scope audio out (outside the Profile schema and carries no referral channel) — extending here is the targeted option in the mechanism question
  - seeds: `c11`
- `s12` — `lobes/gateway/_routing.py::is_audio_path (98-105) + gateway/server.py::handle_audio_post (1426-1467)`: the whole /v1/audio/* namespace is path-routed to the single AUDIO_URL backend, and handle_audio_post forwards via filter_headers which drops only hop-by-hop headers — the caller Authorization reaches the peer verbatim; confirms all four AUDIO_URL contract violations recorded on #129 (all-or-nothing, credential leak, no loop guard/attribution, dishonest capabilities)
  - seeds: `c12`, `c13`, `c14`
- `s13` — `lobes/gateway/server.py proxy branch (628-653, 790-906)`: the core-role forward already implements every guarantee audio needs: Authorization stripped before forward + peer key injected (906), X-Lobes-Proxied single-hop guard with 508 proxy_loop naming both hops (790-813), X-Lobes-Proxied-By attribution (652-653) — the audio lane should reuse this machinery, not reinvent it
  - seeds: `c13`, `c15`
- `s14` — `lobes/roles.py (65, 401, 593-602, 672)`: ROLE_BACKEND has no stt/tts entries, _audio_role takes no peer argument, and annotate_peer_referrals iterates ROLE_BACKEND — so audio roles structurally cannot carry hosted_by today; the registry builds stt/tts via _audio_role at 593-596
  - seeds: `c11`
- `s15` — `lobes/roles_measure.py (277, 357) + lobes/minor/_client.py (119, 242) + lobes/cli/_runtime_ops.py::gateway_auth_headers (80)`: the two keyless stdlib clients (audio measure probes and the minor client used by benchmark --all-lobes and lobes route) versus the shared auth-header helper already used by the assess/benchmark/capabilities/measure CLI layers — the #129 items 1-2 stragglers, a contained fix
  - seeds: `c10`
- `s16` — `issues #108/#127 conventions + docs/evidence/ + the 2026-07-17 session memory`: the #108 rule (no validated claim without a committed transcript) and the #127 O(machines) key rule govern the audio extension; the Spark-to-Thor TTS request recorded on #129 (2026-07-17) is the real-deployment trigger item 3 was explicitly waiting on; the live Thor runs hand-tuned muse budget values that diverge from the shipped thor-muse.toml, which constrains the acceptance choreography
  - seeds: `c15`, `c20`
