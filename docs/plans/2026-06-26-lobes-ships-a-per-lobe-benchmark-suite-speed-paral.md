# Build Plan — lobes ships a per-lobe benchmark suite: speed, parallel throughput, prefill (read) and generation latency for the minor (Qwen3.5-4B) and primary (27B) lobes through the gateway, plus a logprobs-scored 'Where is the cat?' temporal-reasoning probe that grades soft-correctness over candidate locations

slug: `lobes-ships-a-per-lobe-benchmark-suite-speed-paral` · status: `exported` · from frame: `lobes-ships-a-per-lobe-benchmark-suite-speed-paral`

> lobes ships a per-lobe benchmark suite: speed, parallel throughput, prefill (read) and generation latency for the minor (Qwen3.5-4B) and primary (27B) lobes through the gateway, plus a logprobs-scored 'Where is the cat?' temporal-reasoning probe that grades soft-correctness over candidate locations

## Tasks

### t1 — Logprobs client plumbing: chat top_logprobs + /v1/completions echo + gateway-echo capability probe

- covers: c12, h5, h11
- acceptance:
  - chat_completion forwards logprobs+top_logprobs and returns per-token top_logprobs including the answer-position token
  - a completions-echo call (echo=true, logprobs) returns per-token logprobs for an appended continuation string
  - a gateway-echo capability probe returns False when /v1/completions echo is unavailable so callers can fall back

### t2 — Cat-probe generator: timestamped narrative with one unambiguous CURRENT location, open + closed modes, candidate set

- covers: c4, h3, c16, h10
- acceptance:
  - a generated case has exactly one CURRENT location entailed by the timestamps, verified by a deterministic solver in the test
  - closed mode lists the candidate locations in the prompt; open mode omits them
  - the candidate-location set is exposed on the case and recoverable from the narrative text for open-mode scoring

### t3 — Per-lobe perf engine: decode tok/s, prefill TTFT, concurrent req/s + p50/p95 + per-token decode, auto-ramp to plateau

- covers: c2, h1
- acceptance:
  - single-stream decode reports tokens/sec and prefill reports TTFT in ms
  - a concurrent driver reports requests/sec + p50/p95 latency and per-token decode (ms/token, total s)
  - concurrency auto-ramps 1->2->4->8->16... and stops when req/s gain between steps < threshold, reporting the knee + per-step rows

### t4 — Combined per-lobe report renderer: markdown table, per-metric minor-vs-primary deltas + cat soft-score delta

- covers: c1, c6, h8, h14
- acceptance:
  - render takes a per-lobe results dict and emits a markdown table with one row per lobe and explicit per-metric deltas (minor vs primary)
  - the cat soft-score and its minor-vs-primary delta appear alongside the perf deltas in the same table

### t5 — Logprobs cat scorer: softmax over full-sequence echo logprobs (headline) + first-token mass cross-check, renormalised, with fallback

- depends on: t1, t2
- covers: c5, h4, c17
- acceptance:
  - headline soft-score = softmax over candidates' full-sequence echo logprobs and equals 1.0 iff all mass is on the correct location
  - first-token probability mass from chat top_logprobs is computed and reported as a cross-check
  - score renormalises over exactly the candidate set, lands in [0,1]; when echo is unavailable the headline records 'unavailable' and falls back to first-token mass

### t6 — lobes eval cat --score logprobs verb wiring generator+scorer, read-only, --mode open|closed

- depends on: t5
- covers: c16
- acceptance:
  - lobes eval cat --suite <jsonl> --score logprobs runs read-only against a backend and emits per-case soft-score + headline/cross-check + the correct location
  - --mode {open,closed} selects the probe mode and both run end to end

### t7 — lobes benchmark --all-lobes --concurrency auto: per-lobe perf + cat scorer through the gateway, read-only, combined report

- depends on: t3, t5, t4
- covers: c3, h2, c7, c8, c9, h7, h9, h12, h13, c15
- acceptance:
  - lobes benchmark --all-lobes runs the perf engine + cat scorer against each lobe (minor, primary) through the gateway with every number labelled by lobe
  - a single invocation produces both the four perf metrics and the cat soft-score per lobe, rendered together via the report renderer
  - the suite is read-only (no --apply, no writes, no external datasets) and the output answers 'is the minor fast enough and good-enough to stay co-resident?'
  - the soft-score delta is shown so a reader can judge it against run-to-run noise

## Risks

- [follow_up] /v1/completions echo may not be gateway-routed to backends; first build step verifies it, else the scorer falls back to first-token mass (task t1)
