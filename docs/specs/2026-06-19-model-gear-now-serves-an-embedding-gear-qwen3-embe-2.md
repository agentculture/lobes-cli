# model-gear now serves an embedding gear (Qwen3-Embedding-0.6B, 1024-dim) and a reranker gear (Qwen3-Reranker-0.6B) as always-warm fleet backends co-resident with the Qwen3.6-27B primary on the shared DGX Spark, and every consumer reaches all three through one OpenAI-compatible gateway routed by the request model field.

> model-gear now serves an embedding gear (Qwen3-Embedding-0.6B, 1024-dim) and a reranker gear (Qwen3-Reranker-0.6B) as always-warm fleet backends co-resident with the Qwen3.6-27B primary on the shared DGX Spark, and every consumer reaches all three through one OpenAI-compatible gateway routed by the request model field.

## Audience

- jetson-ai-lab-cli (primary; filed #44, building its Discord+docs index) and eidetic-cli (storage+retrieval, second consumer); both call it as an ordinary OpenAI-compatible provider (base_url + api_key + model name), no model-gear-specific SDK.

## Before → After

- Before: the gateway already proxies /v1/embeddings but nothing warm sits behind it; the catalog holds only generative chat gears; there is no reranker anywhere in model-gear.
- After: an indexing client POSTs /v1/embeddings {model,input:[...]} and gets 1024-dim vectors, and POSTs /v1/rerank {model,query,documents:[...]} and gets ranked scores, both served warm by model-gear through the same gateway port the chat primary already uses.

## Why it matters

- this unblocks the indexing half of jetson-ai-lab-cli and the retrieval half of eidetic: model-gear vectorizes, eidetic stores and retrieves, the two compose cleanly.

## Requirements

- The catalog SupportedModel must express task (generate|embed|score), embedding dimension, and per-model hf-overrides / vLLM serve-extras, so embed and score gears are first-class catalog rows alongside chat gears, with the wheel-shipped catalog the single source of truth.
  - honesty: SupportedModel carries task + dimension + hf_overrides; overview --list and /v1/models/supported show the embed+score gears with task+dim; tests assert catalog<->template<->doc consistency for the new fields.
- The gateway must route embedding + score/rerank requests to the right backend by model name and list the warm embed+score backends in /v1/models; embed/score/rerank bodies all carry a model field so handle_post model-routing should cover them (verify, do not assume).
  - honesty: with the fleet up, embeddings + chat are both answered warm via one gateway; any handle_post change is justified by a failing case; /v1/models lists the embed+score served names.
- model explain must document the embeddings, rerank, and score call shapes (request + response + dimension + served model names) so a consumer wires it from the docs alone; model overview --list and /v1/models/supported must surface the new gears.
  - honesty: a consumer wires the integration using only model explain embeddings + rerank (request+response+dim+served names); documented served names equal the catalog ids.
- Both gears are proven WARM on the DGX Spark before the PR lands: curl /v1/embeddings returns a 1024-len vector and curl /v1/rerank returns ranked scores while the 27B primary still answers chat (all three simultaneously healthy); per-model docs carry real measured numbers, not estimates.
  - honesty: the measured curl results (1024-len vector; ranked scores; 27B chat OK simultaneously) are captured from the actual DGX Spark deployment and pasted into the per-model docs.

## Honesty conditions

- All three backends (27B chat, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B) are simultaneously healthy on the GB10, and one gateway port serves chat + /v1/embeddings + /v1/rerank, each routed by the request model field.
- A consumer using only an OpenAI-style client (base_url + api_key + model name) can call all three gears with no model-gear-specific code.
- curl /v1/embeddings {model,input:[hi]} returns data[0].embedding of length 1024, and curl /v1/rerank {model,query,documents} returns results sorted by relevance_score, on the same gateway port as chat.
- On main today, /v1/embeddings is forwarded by the gateway but yields no vector because no embed backend is in the routing table, and grep finds no rerank/score code.
- jetson-ai-lab-cli and eidetic-cli can each point an embed/rerank client at this endpoint and index/retrieve without model-gear owning any storage.
- No vector store, chunker, or retrieval code is added to model-gear; the diff only adds serving (catalog + compose + gateway routing + docs + tests).
- model overview --list shows both gears; the curl checks pass on the box; per-model docs contain measured numbers; model explain embeddings/rerank/score all resolve.

## Success signals

- model overview --list shows the embedding + reranker gears; curl /v1/embeddings returns a 1024-len vector; curl /v1/rerank returns ranked scores; the 27B still answers chat with all three warm; per-model docs record measured dimension + throughput + co-residency; model explain embeddings/rerank/score document the call shapes.

## Scope / boundaries

- model-gear only SERVES the gears (embeddings + scores over HTTP); it does NOT store vectors, build or own an index, chunk text, or do retrieval (that is eidetic). No vector DB lands here.

## Non-goals

- Not adding a vector database, not changing any consumer code, not building a generic multi-task autoscaler, not removing or demoting the 27B chat primary.

## Assumptions

- Two 0.6B gears (~1.2GB each) co-fit easily alongside the 27B primary (~70GB at util 0.6) within the GB10 128GB unified memory; lowering the primary gpu-mem-util to make room is acceptable if measurement shows it is needed.

## Decisions

- Topology: embedding + reranker run as ADDITIONAL always-warm fleet backends (extra vllm services + gateway routing by model name), not via single-model swaps. model switch also gains embed/score (--task) support so a gear can be served solo for isolated load-testing.
- Models: Qwen3-Embedding-0.6B at its native 1024-dim, and Qwen3-Reranker-0.6B (smallest, leaves the most 27B headroom). Qwen3-Embedding-4B / Reranker-4B are documented upgrade paths, not warmed.
