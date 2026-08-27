# docs/experiments — checkpoints and engines we evaluated

An **experiment** is a checkpoint, engine, or runtime this fleet seriously
considered and did not (or has not yet) put into service. Each file answers the
same three questions: **what it is, why we wanted it, and why it is not
running.**

## What belongs here

- a checkpoint evaluated and **not adopted**, or adopted-later — the record of
  the evaluation itself
- an **engine or runtime** weighed against the fleet's own (llama.cpp, SGLang,
  a plugin) for a specific model
- work that was **deferred rather than finished** — so resuming needs no
  re-derivation

## What does not

| it is | it goes |
|---|---|
| a gear the catalog declares and a lane serves | `docs/<model>.md`, beside its siblings |
| a raw measurement transcript from a physical box | `docs/evidence/` |
| a converged spec or plan | `docs/specs/`, `docs/plans/` |
| a container image's digest, version, and arch validation | `docs/image-ledger.md` |

An experiment doc **cites** those; it does not replace them.

## The rule that matters

`#108` applies here more sharply than anywhere else in `docs/`, because these
files describe things that were never served. **Every number states where it
came from** — a published card, or a measurement on a *different* model — and
nothing may read as validated without a transcript under `docs/evidence/`.

An experiment that graduates keeps its file. The evaluation is why the decision
was made, and deleting it loses that.

## Index

| doc | checkpoint | status |
|---|---|---|
| [`qwen3.8-flash-next-gguf-llamacpp-vllm.md`](qwen3.8-flash-next-gguf-llamacpp-vllm.md) | `unsloth/Qwen3.8-Flash-Next-GGUF` — 125B MoE + 51B n-gram, 6B active | **NOT SERVED** — deferred 2026-08-27; nothing native fits 122 GiB, llama.cpp costs ~25x on prefill, vLLM route has four open unknowns |
