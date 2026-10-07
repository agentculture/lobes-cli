# Nemotron-3-Embed-8B — the `nemotron-embed` opt-in lane (4096-d text)

> One entry in lobes's supported catalog. **Status: `load-tested` on the Jetson
> AGX Orin (2026-10-07). It is an OPT-IN candidate, NOT a standard lane**
> (deviation d1). Transcripts:
> [vLLM spike](evidence/2026-10-07-spike-orin-embed-vllm-8b.txt),
> [code head-to-head](evidence/2026-10-07-h2h-code-retrieval-orin.txt).

## Catalog entry

`nvidia/Nemotron-3-Embed-8B-BF16`, lane name `nemotron-embed`: dense text
embedding on a Ministral3 backbone with mean pooling, **4096-d**, L2-normalized,
text only. The catalog context ceiling is 262144 (`config.json`
`max_position_embeddings`; the card evaluates at 4096). `role_hint="candidate"`.

## License

OpenMDW-1.1 (not Apache-2.0), per the Hugging Face card read 2026-10-07
(PUBLISHED-ELSEWHERE).

## Measured on the Orin (2026-10-07)

- Serves on the pinned vLLM engine (`--runner pooling --convert embed`,
  TRITON_ATTN) at **util 0.35**, `max_model_len=8192`: weights 14.94 GiB,
  container 19.26 GiB in the spike, **20.56 GiB** as served in the head-to-head
  (`mem_limit` 26g). Boot 171 s.
- Negative control: cos(query, relevant) 0.60 against cos(query, irrelevant)
  0.0017; 4096-d, unit norm.
- Code retrieval, nDCG@10: `nl_to_code` 0.9447 (n=48), `issue_to_source`
  0.6065 (n=24). Margin over EmbeddingGemma 2: +0.23 and **+8.36** points. It
  did not clear decision c26's 5-point bar on both families, so it did not take
  the code slot.
- With it added beside `gemma2-embed` and the 0.6B pair, 18.7 GiB stayed
  "available" through a full head-to-head pass: the opt-in fits.

## Why it is opt-in, not standard

It beat EmbeddingGemma 2 only on 24 `issue_to_source` queries (no confidence
interval) at about 4.4x the memory (20.6 vs 4.7 GiB), and it is text only.
"Bigger is not always better." The broader evaluation that could justify an 8B
lane is issue #296; until it lands this stays a declared opt-in.

## Opting in

Add `nemotron-embed` to `COMPOSE_PROFILES` and the measured knobs in `.env`
(see [`orin-embed-deployment.md`](orin-embed-deployment.md#opting-in-nemotron)).
It is not rendered by the shape. Without `NEMOTRON_EMBED_BASE_URL`,
`model=nemotron-embed` is a 404 on that box. Its vectors are a different space
from every other lane.

## Not validated

Quality beyond the 24+48 code queries (#296); co-residency under concurrent
load; mesh forwarding of this lane (the acceptance run exercised only
`gemma2-embed`).
