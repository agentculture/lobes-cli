# Contracts

One file per **way of talking to a lobes box**, written for whoever is on the
other end of the wire — a harness, an agent, another service. A contract says
what the client sends, what comes back, in what order, and what the server
guarantees when things go wrong. It is not a description of how the server is
built; that lives in the rest of `docs/`.

## What makes a file a contract

- **The client's point of view.** Every section answers "what do I send, what
  do I get". Server internals appear only when a client must plan around them.
- **Examples copied from a real exchange**, shortened only where marked `…`,
  with the box and date they came from. No invented payloads.
- **Failure behaviour stated as a table** — the situation, and exactly what the
  client receives. "Never a silent fallback" is a rule of this project; a
  contract is where a client learns what it gets instead.
- **What the server does NOT do** is said out loud (it runs no tools, it does
  not roll back a side effect, it does not resume a session).
- **Known limits carry a date** and, where one exists, an issue number.
- Changing anything a contract promises is a **wire change**: say so in the
  CHANGELOG, and prefer adding over renaming or removing.

## Index

| contract | covers | status |
|---|---|---|
| [realtime-tool-calling.md](realtime-tool-calling.md) | `GET /v1/realtime` — the duplex voice session: audio in and out, turn events, the tool-call round trip, interruption, the latency features as a client sees them | written 2026-09-18 from live session logs (DGX Spark, Hebrew stack) |

## Not written yet

These surfaces exist and are described elsewhere, but have no contract in this
form. The pointer is where the current description lives.

| surface | endpoints | described today in |
|---|---|---|
| LLM generation | `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses`; role and tier aliases; tool calling; streaming; the 404 `role_infeasible` / 429 shed / 503 `role_unverified` answers | `docs/openai-api.md`, `docs/colleague-stack.md`, `docs/gateway-fleet.md` |
| Image and video input | image/video parts on the chat endpoints (`cortex`, `senses`, `worker`) | the per-model docs |
| Embeddings and reranking | `POST /v1/embeddings`, `/v1/rerank`, `/v1/score` | `docs/qwen3-embedding-0.6b.md`, `docs/qwen3-reranker-0.6b.md` |
| Batch audio | `POST /v1/audio/transcriptions`, `/v1/audio/speech` | `docs/realtime-pipeline.md`, `docs/openai-api.md` |
| Discovery | `GET /capabilities`, `/v1/models`, `/mesh/roster` | `docs/colleague-stack.md`, `docs/gateway-fleet.md` |
