# embed_h2h

Code-retrieval head-to-head for the Orin embedding specialist slot (#291).

- `corpus.jsonl`: function/method-level chunks of `.py` files at pinned SHAs
  (lobes-cli `971117b1cb79a8f242bdb056ae0601f9d8a9a348`, culture
  `ff2581578b4614a4b0b2023bbc4b1a5c41630710`). Chunks under 4 lines are skipped, text is
  truncated to 2400 chars, and each repo is capped at 1500 chunks (the cap is not reached).
  Rebuild with `build_corpus.py`.
- `queries.jsonl`: hand-written labelled queries, 48 `nl_to_code` and 24 `issue_to_source`;
  `relevant` maps chunk id to graded relevance (2 primary, 1 related). Regenerate and validate
  with `build_queries.py`.
- `prompts.json`: per-model query/document prompts, each with `source`/`verified`.
- `h2h.py`: scorer and HTTP harness (stdlib only). `THRESHOLD_POINTS = 5.0` is the committed
  decision margin (nDCG@10 points, on both families).
