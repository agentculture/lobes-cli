# Qwen3-VL-Embedding-8B — the `qwen3vl-embed` measured, not-carried lane (4096-d)

> One entry in lobes's supported catalog. **Status: `load-tested` on the Jetson
> AGX Orin (2026-10-07); measured, NOT carried: it cannot co-reside with the
> standard lane.** Transcripts:
> [vLLM spike](evidence/2026-10-07-spike-orin-embed-vllm-8b.txt),
> [code head-to-head](evidence/2026-10-07-h2h-code-retrieval-orin.txt).

## Catalog entry

`Qwen/Qwen3-VL-Embedding-8B`, lane name `qwen3vl-embed`: multimodal embedding
(text, image, video), up to **4096-d** with user-defined output dimensions 64 to
4096 (a continuous range, so no MRL ladder is declared), 32768 context.
`role_hint="candidate"`. A different vector space from the Qwen3-Embedding gears.

## License

Apache-2.0, per the Hugging Face card read 2026-10-07 (PUBLISHED-ELSEWHERE).

## Measured on the Orin (2026-10-07)

- Serves **only at util 0.52** (`--memory 44g`): 4096-d, unit norm, container
  30.74 GiB, KV 5.26 GiB = 38,320 tokens, minimum host available during boot
  7,064 MiB. Util 0.35 and 0.45 and 0.35 with `--skip-mm-profiling` were all
  refused: the multimodal encoder profile costs about 9 to 11 GiB above the
  15.49 GiB weights.
- Both Qwen3-VL engines were OOM-killed at `--memory 30g` mid weight-load:
  on Jetson, GPU memory counts against the container's cgroup limit.
- With the EG2 sidecar resident it refused at 0.52; it booted only with the
  sidecar and Nemotron stopped (31.59 GiB, KV 3.14 GiB in the head-to-head).
  It is **not co-residable**.
- Negative control (single probe): cos 0.7689 relevant against 0.3114
  irrelevant, weaker separation.
- Code retrieval, nDCG@10: `nl_to_code` 0.8831 (n=48), `issue_to_source`
  0.5713 (n=24). Margin over EmbeddingGemma 2: -5.93 and +4.85 points.
  Caveat: its instruction is a system prompt on its card; the harness sent the
  Qwen3 `Instruct:`/`Query:` prefix as text (marked `verified=false`), so its
  score may understate it.

## Status: measured, not carried

The `orin-embed` shape does not host it. The shipped compose service defaults
(util 0.35, `mem_limit` 24g) are DECLARED and are refuted by the measurements
above; running it needs util 0.52 and a `mem_limit` well above 30g, with the
other lanes stopped. Image and video embedding through this model was not
measured.
