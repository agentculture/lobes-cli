# decider — schema-scored decisions on POST /v1/systemone

`decider` is the Colleague role for **typed decisions**: a caller sends a state
and a set of typed questions, and gets back a probability for every option of
every question, not prose. It is served by **decision models** (a backbone plus
a scoring head, no free-form generation) through the **decider sidecar**
(`lobes/decider_sidecar/`), and reached through the gateway on
`POST /v1/systemone`.

Responsibilities: `typed_decision`, `classification`, `routing_choice`,
`triage`, `guardrail_check`, `final_decision`. Forbidden: `repo_action`,
`security_decision`. It is the first non-`cortex` role allowed to make a final
decision, because its answer is a calibrated distribution over options the
caller defined, not an open-ended action.

## The checkpoints

| Checkpoint | Quant | Where | Status |
|---|---|---|---|
| `culture-dev/clef-27b-GPTQ-W4A16-g64` (the default `decider`) | INT4 weight-only, GPTQ group 64 (W4A16) | Jetson AGX Orin, `clef-vllm` backend | MEASURED (below) |
| `culture-dev/clef-27b-NVFP4` | NVFP4 (W4A4) | Blackwell (DGX Spark) | MEASURED; mis-calibrated on the Jetson AGX Thor |
| `Cloudflare/clef` | none (bf16) | DGX Spark, `clef` backend | MEASURED (the interim, and the reference) |
| `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | none (bf16) | any box with ~8 GB | DECLARED/UNVALIDATED (#108) |

All three Clef builds are made from `Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c`
(Apache-2.0, a Qwen3.8-27B post-train with a joint schema head). The two
quantized builds are our own (llm-compressor 0.14.0), published by the
`culture-dev` org; no third-party Clef quant is served (deviation d4).

### The quant answer

- **Clef on the Orin (Ampere, sm_87): weight-only 4-bit.** Ampere has no FP4
  tensor cores, so a W4A4 (NVFP4) checkpoint is out. A W4A16 build runs on
  vLLM's Marlin kernel with bf16 activations. Ours keeps `lm_head`,
  `embed_tokens`, the ViT and `linear_attn.in_proj_a/b` in bf16; 18 GB.
- **Clef on Blackwell: NVFP4.** FP4 weights and activations, every
  `linear_attn` projection kept bf16 (deviation d5); 27 GB.
- **Strands Decider 2B: bf16, no quant.** It is ~2B parameters (a rank-48 LoRA
  and a ~3.1M-parameter fp32 pointer head on `Qwen/Qwen3.5-2B-Base`, read from
  the release's own config) and is the fine-tune student; quantizing it saves
  little memory and would have to be redone after every fine-tune.

### Measured (418 labelled questions, `culture-dev/clef-probe-set`)

| Run | Top-1 accuracy | Agreement with bf16 | Median ms (<= 1,024 tokens) |
|---|---|---|---|
| bf16 / DGX Spark | 0.9354 | — | 441 |
| W4A16 / Jetson AGX Orin (plain `v0.31.0`, 2026-10-14) | 0.9378 | 0.988 | 505 |
| NVFP4 / DGX Spark | 0.9330 | 0.9785 | 181 |

The 0.988 row was measured on plain `v0.31.0`, before the sidecar moved to the
`-ubuntu2404` fleet pin. It was re-measured on `-ubuntu2404` (decision c53), same probe set, same bf16 Spark reference, in
`docs/evidence/2026-10-17-accept-vllm-031-orin.txt`:

| Run (Jetson AGX Orin, W4A16) | Top-1 accuracy | Agreement with bf16 | KL mean | Median ms |
|---|---|---|---|---|
| plain `v0.31.0` (2026-10-14) | 0.9354 | 0.9904 | 0.00308 | 506 |
| `v0.31.0-ubuntu2404` (the fleet pin) | 0.9378 | **0.9928** (415/418) | 0.00295 | 516 |

The `-ubuntu2404` base is the one the sidecar now builds from, so 0.9928 is the
current MEASURED value and 0.988 is its plain-`v0.31.0` predecessor (kept, not
deleted). The `orin-decider` shape's render remains DECLARED/UNVALIDATED
(#108): the re-measure above ran the lane on the Orin, not a boot from that
render.

On the Jetson AGX Thor (sm_110) the NVFP4 files give flattened probabilities
(mean top-1 probability 0.31 vs 0.92; agreement 0.957–0.964); the cause is not
identified, and attention is ruled out. On the Orin, the `clef-vllm` path beat
the transformers NF4 path: 0.9437 vs 0.9276 accuracy on the questions both
answered, 2.3–3.3x faster, and up to 16,384 tokens where the NF4 runtime
refuses anything over 8,192. Sources:
`docs/evidence/2026-10-14-verify-clef-quants.txt`,
`docs/evidence/2026-10-14-ab-clef-vllm-vs-sidecar-orin.txt`.

## Calling it

```http
POST /v1/systemone
Authorization: Bearer <GATEWAY_API_KEY>
X-Lobes-Model: decider:general
Content-Type: application/json
```

- **The domain comes from the `X-Lobes-Model` header**, `decider:<domain>`.
  The `/v1/systemone` body carries no routable `model`, so the gateway never
  reads it: the body is forwarded **byte-identical** and the answer relayed
  unchanged. An absent header means `decider:general`.
- **Each domain is one sidecar** (`decider-<domain>`), declared by the shape
  (`[decider_domains.<name>]`). A domain this box does not serve is forwarded
  once to the verified mesh member that announces it, with
  `X-Lobes-Mesh-Member` on the reply; two members serving one domain with
  different checkpoints is a `409`, never a silent pick.
- **Errors:** `404 model_not_found` for an undeclared domain (never a fall-back
  to `general`), `404 role_infeasible` when no decider is wired, and
  `503 backend_unavailable` with `Retry-After` when the sidecar does not answer.
- **Never shed.** The decider path reads no pressure signal, so swap or iowait
  never turn a decision into a `429` (`NEVER_SHED_ROLES`).
- **Response formats pass through.** Clef answers in Cloudflare's shape (below);
  a Strands domain answers in strands-decider's own shape. lobes reshapes
  neither.

## Worked example: `lobes route`'s gear choice as a decision

`lobes route` asks the `hand` model, in prose, which gear should take a task.
The same choice as a `/v1/systemone` question, sent to the Orin's W4A16 Clef on
2026-10-14 (measured, one request, 299 input tokens, 1,066 ms):

```json
{
  "model": "clef",
  "state": "Task: refactor the gateway's retry loop across three modules and update their tests; production is not affected.",
  "questions": {
    "gear": {
      "type": "choice",
      "instructions": "Which lobe should handle this task?",
      "criteria": {
        "hand": "a tiny, cheap specialist for short routine tasks",
        "worker": "a fast doer for bulk code edits and ground work",
        "cortex": "the large reasoning model for hard, high-stakes decisions",
        "embedder": "a text embedding model for retrieval"
      }
    },
    "urgent": {"type": "noul", "instructions": "Does the task need to be handled immediately?"}
  }
}
```

```json
{
  "model": "clef",
  "answers": {
    "gear": {
      "type": "choice",
      "choice": "worker",
      "confidence": 0.9412,
      "probabilities": {"hand": 0.0258, "worker": 0.9412, "cortex": 0.0266, "embedder": 0.0064}
    },
    "urgent": {"type": "noul", "noul": 0.019}
  },
  "usage": {"input_tokens": 299, "output_tokens": 0}
}
```

A `choice` question returns the pick, its confidence and the full
distribution; a `noul` question returns the probability of "true". The caller
can apply its own threshold, which a prose answer does not allow. `lobes route`
itself is unchanged; this shows the decider as the consumer the role exists
for.

## Deploying it on the Orin

`lobes init --shape orin-decider --profile orin --apply` renders `orin-embed`
plus `decider:general` (W4A16, `clef-vllm`, `gpu_mem_util 0.40`,
`max_model_len 16384`, `mem_limit 32g`). The decider is in the Orin's
one-heavy-lobe group: it never co-resides with `cortex`, `senses` or
`associate`. The budget was measured with the same image outside this render;
the render itself is DECLARED/UNVALIDATED until its live acceptance lands (#108).

The two optional engine knobs, `DECIDER_ATTENTION_BACKEND` and
`DECIDER_KV_CACHE_MEMORY_BYTES`, are documented in
`lobes/decider_sidecar/server.py`. A fixed KV size skips vLLM's free-memory
profile, whose assert failed on the Thor when page cache was reclaimed mid-boot.

### Rollback

Back up `~/.lobes` **before** switching shapes (`cp -a ~/.lobes
~/.lobes-bak-<date>-pre-orin-decider`); the backup is the rollback. A
re-render to `orin-embed` alone is **not** a complete rollback, for the reason
the `orin-embed` rollback rehearsal recorded
(`docs/evidence/2026-10-07-rollback-orin-embed-to-orin-associate.txt`): `.env`
is merge-only and `lobes init` never deletes files, so `DECIDER_DOMAINS`,
`DECIDER_DOMAIN_MODELS`, `DECIDER_DOMAIN_CONTEXTS` and `decider` in
`COMPOSE_PROFILES` would stay behind
and the gateway would keep advertising a domain nothing starts. In order:

1. Stop and remove the sidecar: `stop decider-general`, then `rm -f
   decider-general`, through the deployment's own compose chain
   (`lobes-compose.sh`, never `lobes/templates/`).
2. Restore the backed-up `~/.lobes` over the live one.
3. Recreate the gateway (`up -d --no-deps gateway`) so it re-reads `.env`.

## See also

- `docs/colleague-stack.md` — why `decider` is a role
- `lobes/decider_sidecar/` — the sidecar and its backends
- `lobes explain decider`
