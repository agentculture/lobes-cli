# DSpark spike — raw arm transcripts, and the window each was taken at

The `arm-*.json` files committed alongside
`2026-08-24-spike-dspark-cortex-spark.txt` each record **`max_model_len: null`**.

That is a real gap, not a formatting quirk, and it was raised in review (Qodo
review of PR #200, finding 5): the plan's own honesty condition `h18` requires
every reported measurement to name the window in force, and a raw transcript
that cannot say which window it belongs to forces the reader to trust a
filename.

**Why it is null.** `scripts/spec-arms.py` discovers the window by asking the
lane's `/v1/models`. The gateway in front of this deployment does not expose
`max_model_len` on that endpoint, so the script recorded `null` — correctly
reporting "unknown" rather than guessing. The window was recorded in prose in
the transcript instead.

**The mapping, from the transcript's own section headers:**

| file | target | window | arm |
|---|---|---|---|
| `arm-mtp-n2.json` | `unsloth/Qwen3.8-27B-NVFP4` | 1048576 | mtp-n2 (incumbent, pre-stop) |
| `arm-mtp-n2-768k.json` | `unsloth/Qwen3.8-27B-NVFP4` | 786432 | mtp-n2 |
| `arm-dspark-768k.json` | `unsloth/Qwen3.8-27B-NVFP4` | 786432 | dspark (block 7) |
| `arm-none-768k.json` | `unsloth/Qwen3.8-27B-NVFP4` | 786432 | none |
| `arm-mtp-n2-262k.json` | `unsloth/Qwen3.8-27B-NVFP4` | 262144 | mtp-n2 |
| `arm-dspark-262k.json` | `unsloth/Qwen3.8-27B-NVFP4` | 262144 | dspark (block 7) |
| `arm-none-262k.json` | `unsloth/Qwen3.8-27B-NVFP4` | 262144 | none |
| `arm-a16-none-262k.json` | `huginnfork/Qwen3.8-27B-NVFP4A16` | 262144 | none |
| `arm-a16-dspark-262k.json` | `huginnfork/Qwen3.8-27B-NVFP4A16` | 262144 | dspark (block 7) |

The JSON files are **not** edited to add the window after the fact — they are a
historical record of what the harness actually observed, and back-filling a
value the run did not capture is exactly the fabrication this repo's evidence
convention forbids. This table is the mapping; the transcript is the authority.

**Forward fix:** the window should come from the deployment's own
`PRIMARY_MAX_MODEL_LEN` (or an explicit flag) rather than from an endpoint that
may not carry it, so a future run records it in the JSON itself.
