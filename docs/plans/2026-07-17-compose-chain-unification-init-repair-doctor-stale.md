# Build Plan — compose-chain unification, init repair, doctor staleness, audio peer-proxy (#137/#138/#119/#129)

slug: `compose-chain-unification-init-repair-doctor-stale` · status: `exported` · from frame: `compose-chain-unification-init-repair-doctor-stale`

> lobes closes the four post-#127 gaps in one sweep: every compose invocation — CLI verb or acceptance script — resolves the same overlay chain from one shared builder, so switch/serve/restore can never boot a shape-dropped lobe; lobes init heals a partial scaffold missing-only without touching operator state; lobes doctor flags a deployment predating its card profile's required knobs; and stt/tts join the honest-referral/proxy contract, so the Spark serves tts by proxying to the Thor Chatterbox under the same auth, loop-guard, and attribution guarantees as every other lobe

## Tasks

### t1 — PR#137 t1: parameterise _compose_files as the single chain builder and route every runtime caller (compose_up_detached / compose_up_build / compose_down) through it

- covers: c2, c3, h7
- acceptance:
  - _compose_files gains an audio-inclusion mode (overlay-present vs targeted-services-need-audio); shape-last then operator-override-last ordering preserved and documented in the builder docstring
  - compose_up_detached resolves via the builder; a plain no-overlay fleet keeps its bare docker compose up -d argv byte-identical
  - regression test: lobes serve --apply and lobes switch --apply on a shape deployment resolve the same file set as compose_down and start no shape-dropped service

### t2 — PR#137 t2: up.py::_compose_file_args delegates to the parameterised builder

- depends on: t1
- covers: h6
- acceptance:
  - no second -f chain construction remains in up.py; the existing #136 targeted-services tests in test_cli_up.py stay green unchanged

### t3 — PR#137 t3: read-only CLI surface that emits the resolved -f list for a deployment dir

- depends on: t1
- covers: c4
- acceptance:
  - prints one -f arg per line, resolved by the shared builder; a read verb that mutates nothing; documented in docs/deployment-shapes.md or docs/gateway-fleet.md and in lobes explain

### t4 — PR#137 t4: cross-caller equivalence test over the overlay matrix

- depends on: t1, t2, t3
- covers: c6, h10, c16, h11
- acceptance:
  - one parametrised test asserts _compose_files, the up.py delegation, compose_up_detached / compose_up_build / compose_down argv, and the script-facing helper resolve identical file sets for: none, audio, shape, audio+shape — each with and without docker-compose.override.yml
  - asserts [] for the plain fleet, preserving compose auto-discovery of the operator override

### t5 — PR#137 t5: delete shape_render.py::shape_compose_files and the ShapeRender.compose_files field

- covers: c23, h12
- acceptance:
  - the function, the dataclass field, and the tests that read them are gone; grep finds no remaining reference outside git history

### t6 — PR#137 t6 wrap: version bump, changelog, serve.py:48 finding noted on issue #137, PR opened and driven through Qodo/Copilot/SonarCloud to merge

- depends on: t1, t2, t3, t4, t5
- acceptance:
  - version bumped per the every-PR-bumps rule; every review thread resolved; PR merged to main before any #138 work starts

### t7 — PR#138 t7: accept-shape.sh consumes the CLI-emitted chain (_compose_down and --restore)

- depends on: t6
- covers: c5, c4
- acceptance:
  - no hand-rolled -f list remains in the script; --restore on a mesh-shape backup includes docker-compose.shape.yml so no dropped service boots and the gateway keeps no dangling depends_on edge

### t8 — PR#138 t8: validate-tiers.sh consumes the CLI-emitted chain (run-local threshold override still merged last)

- depends on: t6
- covers: h8
- acceptance:
  - _compose_args builds from the helper output; _recreate_gateway recreates the gateway WITH the shape overlay on a shape deployment; the temporary threshold override stays last in the chain

### t9 — PR#138 t9 wrap: version bump, PR through review to merge

- depends on: t7, t8
- acceptance:
  - version bumped; every review thread resolved; PR merged before any #119 work starts

### t10 — PR#119 t10: doctor profile-staleness + missing-scaffold checks (read-only)

- depends on: t9
- covers: c9, h14, c29, h4
- acceptance:
  - resolves the profile via the SAME resolve_init_profile / render_shape path init uses (no forked render); flags missing required knobs and missing scaffold files as warnings; an operator-set differing value downgrades to info; remediation text names doctor --fix, never init --force
  - fixtures reproduce the 2026-07-14 Thor stale .env and the 2026-07-17 Spark partial scaffold; both are caught

### t11 — PR#119 t11: doctor --fix heal lane (missing-only; dry-run by default, --apply commits)

- depends on: t10
- covers: c21, h15, c22, h16, c8, h13
- acceptance:
  - --fix without --apply mutates nothing (test-asserted) and names every file and key it would touch; --fix --apply writes only absent files and appends only absent env keys — the reproduced Spark case heals with every pre-existing .env line untouched
  - the append parses existing keys first; a set HF_CACHE survives a heal that appends other keys; re-running the heal on a complete deployment is a byte-identical no-op

### t12 — PR#119 t12 wrap: version bump, PR through review to merge

- depends on: t10, t11
- acceptance:
  - version bumped; every review thread resolved; PR merged before any #129 work starts

### t13 — PR#129 t13: auth stragglers through gateway_auth_headers (roles_measure audio probes + minor client)

- depends on: t12
- covers: c10, h17
- acceptance:
  - with an inbound gateway key set, the stt/tts measure probes and the minor-client calls attach the bearer (one auth-on test each); with no key configured their requests are byte-identical to today

### t14 — PR#129 t14: stt/tts first-class in the routing/peer schema (ROLE_BACKEND entries, _audio_role peer argument, STT_/TTS_ peer env channels)

- depends on: t12
- covers: c11, h18, c24, h21
- acceptance:
  - STT_/TTS_PEER_ORIGIN / _PEER_PROXY / _PEER_API_KEY parse with the same _as_bool, three-condition arming, and repr=False secrecy as the five core prefixes; lobes capabilities, GET /capabilities, and the audio 404 body carry hosted_by for a declared audio peer
  - no catalog change and no per-card audio knobs in profile TOMLs; shape goldens and existing no-audio capabilities output stay byte-identical

### t15 — PR#129 t15: per-endpoint audio forwarding through the proxy-lobes machinery

- depends on: t14
- covers: c12, h19, c13, h20
- acceptance:
  - /v1/audio/speech and /v1/audio/transcriptions route per-role: tts-remote + stt-local is expressible and tested, and vice versa; AUDIO_URL stays the local default for both
  - on the forward path the caller Authorization never reaches the peer; a request arriving with X-Lobes-Proxied is refused 508 proxy_loop; every proxied answer carries X-Lobes-Proxied-By — each test-asserted

### t16 — PR#129 t16: live cross-box acceptance — the Spark serves tts from the Thor; evidence committed

- depends on: t15, t13
- covers: c1, h1, c30, h5, c27, h2, c28, h3, h9
- acceptance:
  - a live Spark-to-Thor /v1/audio/speech round-trip returns audio with X-Lobes-Proxied-By naming the Thor while /v1/audio/transcriptions serves locally; the transcript lands under docs/evidence/ and docs claim validated only after it lands (#108)
  - the same live session verifies accept-shape.sh --restore boots zero dropped services on a mesh-shape backup (the live half of the restore honesty condition)

### t17 — PR#129 t17 wrap + sequence close: version bump, review-to-merge, four-PR sequence proven

- depends on: t16
- covers: c26, h22
- acceptance:
  - PR N+1 branched only after PR N merged across all four PRs; every Qodo/Copilot/SonarCloud thread resolved pre-merge; every PR version-bumped (CI version-check green)

## Risks

- [unknown_nonblocking] physical validation depends on the live pair: the Thor currently runs hand-tuned muse values (util 0.55 / 262144) diverging from the shipped thor-muse.toml, and hosting Chatterbox alongside muse changes its memory budget — reconcile the Thor shape/budget during t16 (task t16)
- [unknown_nonblocking] first-class stt/tts touches the Profile/RoutingTable schema whose goldens must stay byte-identical for existing deployments; if the schema resists a no-op extension, fall back to peer channels outside the Profile schema and surface the deviation before building t14 (task t14)
- [unknown_nonblocking] live boxes pin the gateway image to a released lobes-cli via MODEL_GEAR_VERSION, so #129 gateway behavior reaches the pair only after a PyPI release + image rebuild; PyPI CDN propagation lag bit the 0.44.0 acceptance (retry minutes later) (task t16)
