# nomic-embed-code — the `nomic-code-embed` measured, not-carried lane (3584-d)

> One entry in lobes's supported catalog. **Status: `load-tested` on the Jetson
> AGX Orin (2026-10-07); measured, NOT carried by any shape.** Transcripts:
> [vLLM spike](evidence/2026-10-07-spike-orin-embed-vllm-8b.txt),
> [code head-to-head](evidence/2026-10-07-h2h-code-retrieval-orin.txt).

## Catalog entry

`nomic-ai/nomic-embed-code`, lane name `nomic-code-embed`: dense code embedding
on a Qwen2 backbone, hidden size **3584** (the output dimension), 32768
`max_position_embeddings`, text and code. `role_hint="candidate"`.

## License

Apache-2.0, per the Hugging Face card read 2026-10-07 (PUBLISHED-ELSEWHERE).

## Measured on the Orin (2026-10-07)

- Serves on the pinned vLLM engine at **util 0.35**, `max_model_len=8192`: KV
  3.04 GiB = 56,896 tokens, container 19.18 GiB in the spike (19.17 GiB in the
  head-to-head; minimum host available during scoring 18,970 MiB). Boot 147 s.
- Negative control: cos(query, relevant) 0.6043 against -0.0241; 3584-d, unit
  norm.
- Code retrieval, nDCG@10: `nl_to_code` 0.9520 (n=48), `issue_to_source`
  0.5931 (n=24). Margin over EmbeddingGemma 2: +0.95 and +7.02 points; it did
  not clear decision c26's 5-point bar on both families.

## Status: measured, not carried

It stays a declared candidate: the `orin-embed` shape does not host it. To run
it, add its compose profile (`nomic-code-embed`) and knobs yourself. The
shipped compose defaults for this service (`NOMIC_CODE_EMBED_GPU_MEM_UTIL`
0.35, `mem_limit` 24g) are DECLARED defaults; only util 0.35 and the figures
above are MEASURED. Its vectors are a different space from every other lane.

## Not validated

Quality beyond the code queries (#296); mesh forwarding of this lane.
