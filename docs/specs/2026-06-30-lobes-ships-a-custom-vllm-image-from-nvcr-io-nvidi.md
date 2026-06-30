# lobes ships a custom vLLM image (FROM nvcr.io/nvidia/vllm:26.05.post1-py3 + a Transformers build that registers gemma4_unified) so the Gemma 4 12B multimodal gear finally loads on the DGX Spark, is live-validated, and is promoted from configured to load-tested

> lobes ships a custom vLLM image (FROM nvcr.io/nvidia/vllm:26.05.post1-py3 + a Transformers build that registers gemma4_unified) so the Gemma 4 12B multimodal gear finally loads on the DGX Spark, is live-validated, and is promoted from configured to load-tested

## Audience

- lobes operators on the DGX Spark, and the Culture-mesh callers who address the generate lane as model=multimodal (back-compat normal)

## Before → After

- Before: gemma4_unified registers on no released NGC vLLM image (26.04/vLLM0.19 and 26.05.post1/vLLM0.21 both crash at config load), so the gear ships status=configured and the vllm-multimodal container never starts
- After: the vllm-multimodal gear boots on the custom image (gemma4_unified loads), answers an image+text and an audio+text request, and the catalog status reads load-tested

## Why it matters

- the main+multimodal duo (#69) promises native vision+audio on the generate lane without the realtime overlay — that headline is vapor until the gear actually loads

## Requirements

- ship templates/fleet/Dockerfile.vllm-gemma4 (FROM nvcr.io/nvidia/vllm:26.05.post1-py3; uv pip install --system a PINNED transformers ref that registers gemma4_unified) and a compose build: directive on vllm-multimodal ONLY — primary/embed/rerank keep image: nvcr.io/nvidia/vllm:26.04-py3
  - honesty: inside the built image: AutoConfig.from_pretrained(gemma checkpoint, trust_remote_code=True) succeeds, ModelRegistry shows the Gemma4 unified arch, AND vLLM still imports and 'vllm serve' boots (the transformers bump did not break vLLM 0.21.0)
- make the vllm-multimodal image overridable via a MULTIMODAL_IMAGE env var (default = the local build) so an operator can point at a tag pushed to ghcr.io/agentculture or a local registry, without a registry being required for the default path
  - honesty: with MULTIMODAL_IMAGE set, docker compose uses that image and skips the local build; unset, it builds Dockerfile.vllm-gemma4
- build the image and live-validate co-resident with ZERO disruption to the running fleet (multimodal on a free host port alongside the live primary/embed/rerank; no recreate/restart of those containers)
  - honesty: during build+validate, vllm-primary/embed/rerank keep their container IDs and uptime (not recreated/restarted) and the gateway keeps serving
- IF and only if validation passes (boots, image+text, audio+text, MTP draft acceptance > 0 or corrected method string, util within the 0.69 budget), flip the catalog multimodal status configured -> load-tested and record the measured numbers in docs/gemma-4-12b-nvfp4.md
  - honesty: the catalog status edit is conditional: a passing run sets load-tested with measured numbers in the doc; a failing run leaves status=configured and records the residual

## Honesty conditions

- the custom image recipe lands in the repo (Dockerfile + compose wiring) and the gear either reaches load-tested or stays configured with a documented residual — the announcement holds on either branch
- a model=multimodal (and back-compat model=normal) request resolves to the Gemma gear via the gateway once vllm-multimodal is up and healthy
- after the work, docker ps shows model-gear-vllm-multimodal healthy and both an image+text and an audio+text request return valid assistant output
- today, starting vllm-multimodal on 26.04 or 26.05.post1 crashes at config load with the gemma4_unified 'not recognized / install Transformers from source' error (the recorded t7 finding)
- with the gear loaded, the generate lane serves vision+audio natively for chat input — no realtime /v1/audio overlay needed for image/audio-in
- the diff touches only vllm-multimodal's image/build plus the multimodal catalog status and its doc; primary/embed/rerank image lines and the realtime overlay are untouched
- tests/test_smoke_duo.py Layer B passes against the live gateway with LOBES_SMOKE_BASE_URL set

## Success signals

- model=multimodal answers an image+text and an audio+text request on the Spark, the gated tests/test_smoke_duo.py Layer B passes with LOBES_SMOKE_BASE_URL set, and the catalog reads status=load-tested (pass) OR a documented parked residual with the gear left configured (fail) — recipe merged either way

## Scope / boundaries

- not in scope: migrating primary/embed/rerank to the 26.05 base (3 follow-up issues, only if multimodal works well); changing tier vocabulary or gateway routing; touching the realtime /v1/audio overlay (Parakeet/Chatterbox); making a container registry a hard dependency

## Non-goals

- no separate scaffold-vs-validate split: the recipe PR merges regardless, but live validation + the status flip happen in the same effort because we are already on the Spark

## Decisions

- scope: only vllm-multimodal moves to the custom 26.05 image now; if it works well, 3 follow-up issues migrate primary, embed, and rerank one-by-one (Q1 confirmed)
- delivery: local compose build: is the default (matches gateway/chatterbox/parakeet/realtime); optional MULTIMODAL_IMAGE override allows a ghcr.io/agentculture or local-registry tag (Q2 confirmed + registry feasible)
- tooling: the Dockerfile installs transformers with uv (uv pip install --system), not pip (user preference)
- pin policy: bake a PINNED transformers ref into the Dockerfile for reproducible rebuilds; the exact ref is discovered during validation (proposed default, Q3 unselected)

## Hard questions

- risk: no transformers ref both registers gemma4_unified AND keeps vLLM 0.21.0 importing/serving -> gear stays configured, recipe still merges, and we await an NGC/vLLM release that bundles gemma4_unified
- risk: gemma4_mtp may be the wrong native-MTP method string for vLLM 0.21.0's Gemma4MTPModel — confirm against the served checkpoint (r4)
- risk: vision+audio KV may push measured util past 0.12 / the 0.69 default-fleet budget (r5)

## Parked unknowns (accepted plan risks)

> Resolved during live validation on the Spark. These gate the `load-tested`
> promotion, **not** the recipe PR — the deliverable merges the image recipe
> regardless and leaves the gear `configured` if any of these can't be cleared.
> (Rendered by hand: the devague `spec_md` exporter drops `unknown_nonblocking`
> vagueness; the frame JSON retains v1–v4 so `/spec-to-plan` still reads them.)

- v1 — exact pinned transformers ref that registers `gemma4_unified` AND keeps vLLM 0.21.0 working
- v2 — correct Gemma 4 native-MTP `--speculative-config` method string on vLLM 0.21.0 (resolves r4)
- v3 — measured multimodal GPU util (vision+audio KV) vs the 0.12 budget (resolves r5)
- v4 — confirmed Gemma 4 12B native context window (131072 default until measured)
