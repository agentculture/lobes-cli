"""``model learn`` — the learnability affordance.

Prints a structured self-teaching prompt with enough shape that an agent can
author its own usage skill without scraping ``--help``. Supports ``--json``.
"""

from __future__ import annotations

import argparse

from model_gear import __version__
from model_gear.cli._output import emit_result

_TEXT = """\
model-gear — run, assess, and switch the local vLLM model.

Purpose
-------
model-gear is the tooling that runs the local, OpenAI-compatible vLLM model the
Culture mesh consumes (the model-gear agent connects to it over the acp
`vllm-local` provider). It scaffolds a deployment, starts/stops the server,
switches which model is served, and assesses/benchmarks whatever is running —
producing the numbers in the per-model docs under docs/.

Commands
--------
  model init [TARGET]     Scaffold a deployment dir (default ~/.model-gear).
                          Dry-run by default; --apply to write, --force to overwrite.
  model serve             Start the vLLM server (alias: start). Dry-run; --apply.
  model stop              Stop the vLLM server. Dry-run; --apply.
  model switch <model>    Switch the served model. Dry-run; --apply recreates the
                          container and waits for /health.
  model fleet up|down|status
                          Drive the gateway fleet: the generate primary plus
                          co-resident embedding + reranker gears behind one OpenAI
                          port, routed by task family (a generate fallback is opt-in).
                          Scaffold with 'model init --fleet'. up/down dry-run; --apply.
  model tunnel            Expose the local API at a public hostname via a Cloudflare
                          Tunnel (--stop to tear down). Dry-run; --apply.
  model status            Read-only: the configured served model (.env), container
                          state, /health. (Catalog to switch to: overview --list.)
  model assess            Read-only: correctness probes + reasoning-trace field.
  model benchmark         Read-only: decode throughput + prefill latency.
  model overview          Snapshot of the tool, the served model, and the supported
                          catalog (--current = configured model; --list = catalog).
  model whoami            Tool + machine + served model + container health.
  model explain <path>... Markdown docs for a topic (e.g. 'model explain switch').
  model doctor            Diagnose docker / compose / .env / health.

Mutation safety
---------------
Write verbs default to DRY RUN and require --apply to commit: `switch`, `serve`,
`stop`, `init`, `tunnel`. Agents call CLIs in loops, so safe-by-default is mandatory. The
read-only verbs (`status`, `assess`, `benchmark`, `overview`, `whoami`,
`explain`, `doctor`) never change the world.

Models: supported vs. warm
--------------------------
Two different questions. The SUPPORTED CATALOG — the gears you can switch to, each
tagged load-tested or configured — is `model overview --list` (and the gateway's
GET /v1/models/supported); it is static, defined in model_gear/catalog.py and
shipped in the wheel. What's LOADED right now (actually in GPU memory) is the live
GET /v1/models (which `model fleet status` queries). `model status` / `model whoami`
instead report the model the deployment is configured to serve (from .env) + health
— config, not a live list. Mnemonic: the catalog is the menu; /v1/models is what's
hot now.

Task families & gears
---------------------
The fleet serves four task families behind the one gateway, routed by the
request's `model` field: `generate` (the always-warm Qwen primary), `embed`
(Qwen3-Embedding-0.6B → POST /v1/embeddings), and `score` / `rerank`
(Qwen3-Reranker-0.6B → POST /v1/rerank + /v1/score). The embedding and reranker
gears are tiny (~0.6B, util 0.06 each) and co-reside with the 27B primary on one
GB10; a second *generate* backend (warm fallback) is the only opt-in piece.
`model switch` can also serve a single embed/score gear solo (auto-detected from
the catalog, or forced with `--task embed|score`).

Machine-readable output
-----------------------
Every command supports --json. Errors in JSON mode emit
{"code", "message", "remediation"} to stderr. Stdout and stderr are never mixed.

Exit-code policy
----------------
  0 success
  1 user-input error (bad flag, bad path, missing arg)
  2 environment / setup error (docker missing, .env unreadable, endpoint down)
  3+ reserved

More detail
-----------
  model explain model-gear
  model explain switch
  model explain backend
  model explain embeddings   (POST /v1/embeddings — the embedding gear)
  model explain rerank       (POST /v1/rerank + /v1/score — the reranker gear)

Homepage: https://github.com/agentculture/model-gear
"""


def _as_json_payload() -> dict[str, object]:
    return {
        "tool": "model-gear",
        "version": __version__,
        "purpose": (
            "Tooling to run, assess, and switch the local vLLM model the Culture "
            "mesh consumes (the model-gear agent connects over the acp vllm-local provider)."
        ),
        "serves": "model-gear",
        "commands": [
            {"path": ["init"], "summary": "Scaffold a deployment dir (dry-run; --apply)."},
            {
                "path": ["serve"],
                "summary": "Start the vLLM server (alias: start; dry-run; --apply).",
            },
            {"path": ["stop"], "summary": "Stop the vLLM server (dry-run; --apply)."},
            {"path": ["switch"], "summary": "Switch the served model (dry-run; --apply)."},
            {
                "path": ["fleet"],
                "summary": (
                    "Drive the gateway fleet: generate primary + co-resident embedding "
                    "and reranker gears, routed by task family (up/down/status; --apply)."
                ),
            },
            {
                "path": ["tunnel"],
                "summary": "Expose the local API at a public hostname via a Cloudflare Tunnel "
                "(--stop; --apply).",
            },
            {
                "path": ["status"],
                "summary": "Configured served model, state, /health; catalog: overview --list.",
            },
            {"path": ["assess"], "summary": "Correctness probes + reasoning-trace field."},
            {"path": ["benchmark"], "summary": "Decode throughput + prefill latency."},
            {
                "path": ["overview"],
                "summary": "Tool snapshot + served model + supported catalog (--current/--list).",
            },
            {"path": ["whoami"], "summary": "Tool, machine, served model, container health."},
            {"path": ["explain"], "summary": "Markdown docs by topic path."},
            {"path": ["doctor"], "summary": "Diagnose docker/compose/.env/health."},
        ],
        "mutation_safety": {
            "write_verbs": [
                "switch",
                "serve",
                "stop",
                "init",
                "fleet up",
                "fleet down",
                "tunnel",
            ],
            "rule": "dry-run by default; require --apply to commit",
        },
        "exit_codes": {
            "0": "success",
            "1": "user-input error",
            "2": "environment/setup error",
        },
        "json_support": True,
        "models": {
            "supported_catalog": (
                "The gears you can switch to, each tagged load-tested/configured. Static "
                "(model_gear/catalog.py, shipped in the wheel). Read: 'model overview --list' "
                "or gateway GET /v1/models/supported."
            ),
            "loaded_now": (
                "Model(s) actually in GPU memory right now. Live source: GET /v1/models "
                "(which 'model fleet status' queries). 'model status'/'model whoami' report the "
                "configured served model (from .env) + health, not a live list."
            ),
            "task_families": (
                "The fleet routes by task family on one gateway port: 'generate' (the "
                "Qwen primary), 'embed' (Qwen3-Embedding-0.6B → /v1/embeddings), and "
                "'score'/'rerank' (Qwen3-Reranker-0.6B → /v1/rerank + /v1/score). The "
                "embedding + reranker gears co-reside with the primary (util 0.06 each); "
                "a second generate backend (warm fallback) is opt-in."
            ),
        },
        "explain_pointer": "model explain <path> (e.g. 'model explain switch')",
    }


def cmd_learn(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result(_as_json_payload(), json_mode=True)
    else:
        emit_result(_TEXT, json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "learn",
        help="Print a structured self-teaching prompt for agent consumers.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_learn)
