# Build Plan — lobes ships a custom vLLM image (FROM nvcr.io/nvidia/vllm:26.05.post1-py3 + a Transformers build that registers gemma4_unified) so the Gemma 4 12B multimodal gear finally loads on the DGX Spark, is live-validated, and is promoted from configured to load-tested

slug: `lobes-ships-a-custom-vllm-image-from-nvcr-io-nvidi` · status: `exported` · from frame: `lobes-ships-a-custom-vllm-image-from-nvcr-io-nvidi`

> lobes ships a custom vLLM image (FROM nvcr.io/nvidia/vllm:26.05.post1-py3 + a Transformers build that registers gemma4_unified) so the Gemma 4 12B multimodal gear finally loads on the DGX Spark, is live-validated, and is promoted from configured to load-tested

## Tasks

### t1 — Author lobes/templates/fleet/Dockerfile.vllm-gemma4

- covers: c9
- acceptance:
  - new file FROM nvcr.io/nvidia/vllm:26.05.post1-py3
  - transformers installed via uv pip install --system (not bare pip install transformers); a TRANSFORMERS_REF build ARG parameterizes the pinned ref
  - a build-stage verification RUN asserts AutoConfig loads the gemma checkpoint and ModelRegistry shows the Gemma4 unified arch
  - tests/test_gemma4_dockerfile.py asserts the FROM base, uv usage, the ARG, and the verification step (static file assertions, no docker daemon)

### t2 — Wire vllm-multimodal to the custom image with a MULTIMODAL_IMAGE override; leave primary/embed/rerank on 26.04

- covers: c9, c10, c6, h2, h10
- acceptance:
  - docker-compose.yml: vllm-multimodal gains a build: block (dockerfile Dockerfile.vllm-gemma4) plus image: ${MULTIMODAL_IMAGE:-local-tag}; primary/embed/rerank keep image: nvcr.io/nvidia/vllm:26.04-py3 unchanged
  - env.example documents MULTIMODAL_IMAGE (unset = local build; set = a ghcr.io/agentculture or local-registry tag)
  - with MULTIMODAL_IMAGE set the fleet uses that tag and skips the build; unset it builds locally (the lobes fleet up --build interaction is documented)
  - tests/test_gemma4_compose.py asserts the compose structure: multimodal build+override present; other vLLM services still pinned to 26.04; realtime overlay untouched

### t3 — Build the image on the Spark and verify gemma4_unified registers without breaking vLLM 0.21.0; pin the working transformers ref

- depends on: t1
- covers: h1
- acceptance:
  - image builds from Dockerfile.vllm-gemma4
  - inside the image: AutoConfig.from_pretrained(checkpoint, trust_remote_code=True) succeeds AND ModelRegistry lists the Gemma4 unified arch AND import vllm + vllm serve --help work
  - the verified transformers ref is pinned as the TRANSFORMERS_REF default in the Dockerfile
  - if no ref satisfies both, record the residual and stop (gear stays configured) per the spec fallback

### t4 — Co-resident live serve + functional validation (image+text, audio+text) with zero fleet disruption

- depends on: t2, t3
- covers: c2, c3, c11, h3, h6, h7, h9
- acceptance:
  - vllm-multimodal boots healthy on a free host port alongside the running fleet; vllm-primary/embed/rerank keep their container IDs and uptime (not recreated) and the gateway keeps serving
  - an image+text and an audio+text chat request both return valid assistant output (vision + audio active, no realtime overlay)
  - model=multimodal (and back-compat normal) routes to the Gemma gear via the gateway
  - record MTP draft acceptance (confirm the correct gemma4_mtp method string), measured GPU util vs 0.12/0.69, and the native context window

### t5 — Run the gated smoke Layer B against the live gateway

- depends on: t4
- covers: c8, h11
- acceptance:
  - LOBES_SMOKE_BASE_URL=<gateway> uv run pytest tests/test_smoke_duo.py passes (Layer B image+text and audio+text via model=multimodal)
  - no changes to Layer A; any needed test fixes are minimal and documented

### t6 — Conditional promotion: flip catalog status + record numbers, or document the residual

- depends on: t4, t5
- covers: c1, c4, c5, c12, h4, h5, h8
- acceptance:
  - IF all validation passed: catalog.py multimodal status configured->load-tested and docs/gemma-4-12b-nvfp4.md records measured MTP/util/context numbers
  - IF any check failed: status stays configured and the doc records the parked residual + next step (await NGC release)
  - tests/test_catalog.py passes; the diff still touches only the multimodal entry and its doc (announcement holds on either branch)

## Risks

- [unknown_nonblocking] no transformers ref both registers gemma4_unified AND keeps vLLM 0.21.0 importing/serving -> gear stays configured, recipe still merges, await an NGC release (v1) (task t3)
- [unknown_nonblocking] correct Gemma4 native-MTP --speculative-config method string on vLLM 0.21.0 (gemma4_mtp unconfirmed) — r4/v2 (task t4)
- [unknown_nonblocking] measured vision+audio GPU util vs the 0.12 / 0.69 default-fleet budget — r5/v3 (task t4)
- [unknown_nonblocking] confirmed Gemma 4 12B native context window (131072 default until measured) — v4 (task t4)
- [unknown_nonblocking] lobes fleet up --build always rebuilds; the MULTIMODAL_IMAGE skip-build path may need a no-build invocation or pull step (task t2)
- [follow_up] if multimodal works well, open 3 follow-up issues to migrate primary, embed, and rerank to the 26.05 base one-by-one
