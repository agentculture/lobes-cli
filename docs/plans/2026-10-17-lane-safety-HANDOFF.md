# Lane safety (#304 #307 #308 #309) — handoff

Written 2026-10-17 for a context compaction. Read this first, then
`docs/deliveries/2026-10-17-lane-safety-and-quality-304-307-308-309.md`.

## Where things stand

| Item | State |
|---|---|
| Spec / plan | `docs/specs/2026-10-17-lane-safety-and-quality-304-307-308-309.md`, `docs/plans/2026-10-17-lane-safety-and-quality-304-307-308-309.md`, the split record `…-split.md`. devague frame/plan slug: `lane-safety-and-quality-304-307-308-309`. |
| **PR-A** lobes-cli#314 | Branch `lane-safety-304-307-308-309`, version 0.87.0. Content: boot hold + stagger, every service wrapped, per-lane compile cache, profiles `[host_env]`, shape `[host_env]`, stt NGC entrypoint chained, orin-associate associate-first. CI green, review rounds 1–2 fixed. **Merge first.** |
| **PR-B** lobes-cli#315 | Branch `lane-safety-bundle`, version 0.88.0, base retargeted to `main`, so its diff includes PR-A until PR-A merges. Content: crash backoff + memory gate, `.held` markers, page-cache clear, `lobes watcher`, thermal shedding, opt-in `benchmark --lock`, doctor findings, FlashInfer cache, machine-CLI extras pinned (`dgx-spark-cli>=0.9.1`, `jetson-thor-cli>=0.6.1`, `jetson-orin-cli>=0.7.1`). CI green at `ad3e129`. |
| Machine CLIs | **Released and merged**: dgx-spark-cli#26 (0.9.1), jetson-thor-cli#8 (0.6.1), jetson-orin-cli#5 (0.7.1). GitGuardian incidents 38315781 and 38315782 are false positives (`NOPASSWD: ALL` docstring text). The operator must close them in the dashboard. |
| **In flight** | Round-3 review fixes: a background agent working in worktree `/home/spark/git/.worktrees.lobes-cli/review3`, branch `fix/review3`, off `lane-safety-bundle`. Eight findings: compose-project candidates (symlinked deploy dir), `name: ${…}` interpolation, the grant hint naming a missing CLI, bounded detection cache with stable UNKNOWN, one lane-state helper, `_env_float` parity, de-duplicated I/O, doctor staleness against the deployed shape. **Operator decision: after these merge, stop the review loop.** |

## Finishing the in-flight work

1. Check that the round-3 agent committed on `fix/review3`. Inspect it with `git -C /home/spark/git/.worktrees.lobes-cli/review3 log --oneline lane-safety-bundle..`.
2. Merge it into the bundle, then run every gate and check the **real exit codes**. A `| tail` swallows them, which already let one bad commit through:

   ```bash
   cd /home/spark/git/lobes-cli && git switch lane-safety-bundle
   git merge --no-ff fix/review3 -m "merge round-3 review fixes"
   uv run --python 3.12 pytest -n auto -q; echo rc=$?
   for c in "black --check -q lobes tests" "isort --check-only -q lobes tests" "flake8 lobes tests" \
            "bandit -q -c pyproject.toml -r lobes" "afi cli doctor . --strict"; do uv run $c >/dev/null 2>&1; echo "$c rc=$?"; done
   markdownlint-cli2 "**/*.md" "#node_modules" "#.local"; echo rc=$?
   git push origin lane-safety-bundle
   git worktree remove /home/spark/git/.worktrees.lobes-cli/review3
   ```

3. Wait for CI on #314 and #315 to go green. Read Sonar with `SONAR_TOKEN` set: `curl -s -u $SONAR_TOKEN: "https://sonarcloud.io/api/qualitygates/project_status?projectKey=agentculture_model-gear&pullRequest=315"`. The project key is the legacy `agentculture_model-gear`.
4. Tell the operator both PRs are ready for review and merge, in order: PR-A, then PR-B.

## Rules learned this session (do not relearn)

- **Python 3.12 for gates.** Worktree `uv run` can pick 3.13, where `test_gateway_pool_saturation::test_the_engine_plumbing_is_gone_from_the_server_module` fails because 3.13 dedents docstrings. That failure predates this work.
- **devague state is tracked per branch.** Run devague moves on `lane-safety-bundle` only, and commit `.devague` after every move. Filing on PR-A's stale copy minted a duplicate id once.
- **Stacked PRs:** fix PR-A's content on PR-A, then merge PR-A into the bundle. Never the reverse.
- **Thor is shared with mjolnir-agent** (cross-session peer `mjolnir-agent-d8`). Coordinate every Thor window over SendMessage before touching the box. Its window 3 started about 12:30Z and runs to about 14:30Z with the worker stopped. It messages "restored" when done.
- **Never run `docker compose` in `lobes/templates/`.** Use the deployment (`~/.lobes`) or the `lobes-deploy` skill.
- **Sonar S2083 on the CLIs' state file was a false positive.** The taint was the CLI's own JSON in `~/.local/state`. It was marked FALSE-POSITIVE through the API with a rationale comment, using `SONAR_TOKEN`, with the operator's approval.

## Live changes left on boxes

- **Orin** `~/.lobes/docker-compose.override.yml`: `vllm-embed` has the compile-cache mount (`VLLM_CACHE_ROOT=/cache/vllm`, `./compile-cache/embed`). Backup: `docker-compose.override.yml.bak-lane-safety-20261017T082531Z`. Orin's clocks are pinned by an operator `jetson_clocks` run; that is not ours, so leave it.
- **Scratch venvs** `~/lane-safety-hw` on Spark, Orin and Thor (branch lobes + machine CLIs), and a user-local `uv` on Orin. They are inert, and can be deleted.
- **After PR-B ships, every box will show `logwrap_stale`**, because the wrapper changed. Fix it with `lobes doctor --refresh-logwrap --apply`, then recreate the lanes. Boxes whose `lane-state/` is root-owned self-heal on the next lane start.

## Open follow-ups (issues)

- **#310** Fleet move to vLLM v0.31.0. New data from mjolnir-agent: the official 0.31.0 worker config hit Xid 31 (MMU fault) at 131K depth × 4 concurrent on Thor. Worth a comment on #310 once mjolnir-agent's repro (window 3) reports.
- **#312** `measure` and `benchmark --profile` record no host conditions.
- **#313** The ≥ 30 min Spark thermal soak, never run.
- **#316** Switching shapes leaves role keys in `.env`. A strict xfail test pins it. Fold in the general "rendered-key ownership" design.
- **#317** Health-gated boot release (`LOBES_BOOT_AFTER`), replacing orin-associate's fixed 400 s timer. That timer has about 40 s of slack and holds the gateway dark for about 9 min after a reboot.
- **Open operator idea:** run drop_caches between crash restarts even when memory looks sufficient. Warm-cache boots still crash about 1 in 4 on Thor (a #309 comment from mjolnir-agent).
- **Operator adjudication pending in devague:** lapses l3–l8, evidence e1–e56, deltas b1–b5. Deviations d1–d8 are approved.

## Evidence

- `docs/evidence/2026-10-17-accept-lane-safety-hardware.txt`: the real-hardware pass on Spark, Orin and Thor. Three bugs were found and fixed there: wrapper SIGTERM, `clocks_locked`, and Thor fan and clock parsing. The compile cache measured 31.78 s cold against 5.33 s warm.
