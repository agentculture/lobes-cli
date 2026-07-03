"""``lobes learn`` — the learnability affordance.

Prints a structured self-teaching prompt with enough shape that an agent can
author its own usage skill without scraping ``--help``. Supports ``--json``.
"""

from __future__ import annotations

import argparse

from lobes import __version__
from lobes.cli._output import emit_result

_TEXT = """\
lobes — run, assess, and switch the local vLLM model.

Purpose
-------
lobes is the tooling that runs the local, OpenAI-compatible vLLM model the
Culture mesh consumes (the lobes agent connects to it over the acp
`vllm-local` provider). It scaffolds a deployment, starts/stops the server,
switches which model is served, and assesses/benchmarks whatever is running —
producing the numbers in the per-model docs under docs/.

Commands
--------
  lobes init [TARGET]     Scaffold a deployment dir (default ~/.lobes).
                          Dry-run by default; --apply to write, --force to overwrite.
  lobes serve             Start the vLLM server (alias: start). Dry-run; --apply.
  lobes stop              Stop the vLLM server. Dry-run; --apply.
  lobes switch <model>    Switch the served model. Dry-run; --apply recreates the
                          container and waits for /health.
  lobes up <role>         Start ONE Colleague role's gear (cortex/senses/embedder/
                          reranker/stt/tts) or the full 'colleague-stack' (all six).
                          --down stops just that role; dry-run; --apply.
  lobes fleet up|down|status
                          Drive the gateway fleet: the generate primary plus
                          co-resident embedding + reranker gears behind one OpenAI
                          port, routed by task family (a generate fallback is opt-in).
                          Scaffold with 'lobes init --fleet'. up/down dry-run; --apply.
  lobes tunnel            Expose the local API at a public hostname via a Cloudflare
                          Tunnel (--stop to tear down). Dry-run; --apply.
  lobes status            Read-only: the configured served model (.env), container
                          state, /health. (Catalog to switch to: overview --list.)
  lobes assess            Read-only: correctness probes + reasoning-trace field.
  lobes benchmark         Read-only: decode throughput + prefill latency.
  lobes overview          Snapshot of the tool, the served model, and the supported
                          catalog (--current = configured model; --list = catalog).
  lobes whoami            Tool + machine + served model + container health.
  lobes explain <path>... Markdown docs for a topic (e.g. 'lobes explain switch').
  lobes doctor            Diagnose docker / compose / .env / health.

Mutation safety
---------------
Write verbs default to DRY RUN and require --apply to commit: `switch`, `serve`,
`stop`, `up`, `init`, `fleet up`, `fleet down`, `tunnel`. Agents call CLIs in loops, so
safe-by-default is mandatory. The read-only verbs (`status`, `assess`, `benchmark`,
`capabilities`, `endpoint`, `measure`, `overview`, `whoami`, `explain`, `doctor`)
never change the world.

Models: supported vs. warm
--------------------------
Two different questions. The SUPPORTED CATALOG — the gears you can switch to, each
tagged load-tested or configured — is `lobes overview --list` (and the gateway's
GET /v1/models/supported); it is static, defined in lobes/catalog.py and
shipped in the wheel. What's LOADED right now (actually in GPU memory) is the live
GET /v1/models (which `lobes fleet status` queries). `lobes status` / `lobes whoami`
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
`lobes switch` can also serve a single embed/score gear solo (auto-detected from
the catalog, or forced with `--task embed|score`).

Realtime audio (opt-in overlay)
-------------------------------
`lobes init --fleet --audio` adds an OpenAI /v1/audio/* facade (a `realtime` bridge
container) backed by two fixed GPU sidecars: Parakeet STT (nvidia/parakeet-tdt-0.6b-v2,
POST /v1/audio/transcriptions) and Chatterbox TTS (Resemble AI, 0.5B, POST
/v1/audio/speech, 24 kHz, zero-shot voice cloning). Both are open-weights — no NGC
key. These backends are hardcoded (NOT in the switchable catalog). `lobes fleet up`
builds and starts them with the rest of the fleet. See `lobes explain realtime`.

OpenAI API surface
------------------
One port (default :8000), routed by the request's `model` field: /v1/chat/completions,
/v1/completions, /v1/embeddings, /v1/rerank, /v1/score, /v1/audio/transcriptions,
/v1/audio/speech, /v1/models (loaded now), /v1/models/supported (the catalog), /health.
See `lobes explain api` and docs/openai-api.md.

Auth / exposure
---------------
Set CULTURE_VLLM_API_KEY in the deployment .env before exposing the API. vLLM
enforces it as a bearer token on the single-model `lobes serve` path. The fleet
gateway is a pass-through and is NOT auth-aware (a known limitation) — bearer keys
do not protect its proxied endpoints, so add Cloudflare Access or an IP allowlist
when tunnelling the fleet. Use `lobes tunnel` to expose the local API via a
Cloudflare Tunnel. See `lobes explain tunnel`.

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
  lobes explain lobes
  lobes explain switch
  lobes explain backend
  lobes explain embeddings   (POST /v1/embeddings — the embedding gear)
  lobes explain rerank       (POST /v1/rerank + /v1/score — the reranker gear)
  lobes explain realtime     (the /v1/audio/* overlay — Parakeet STT + Chatterbox TTS)
  lobes explain api          (the full OpenAI-compatible endpoint surface)
  lobes explain roles        (the six-role Colleague contract: cortex/senses + services)
  lobes explain gateway      (the fleet front — routing, /status, auth limitation)
  lobes explain tunnel       (expose the local API anywhere via Cloudflare Tunnel)

Homepage: https://github.com/agentculture/lobes-cli
"""


def _as_json_payload() -> dict[str, object]:
    return {
        "tool": "lobes",
        "version": __version__,
        "purpose": (
            "Tooling to run, assess, and switch the local vLLM model the Culture "
            "mesh consumes (the lobes agent connects over the acp vllm-local provider)."
        ),
        "serves": "lobes",
        "commands": [
            {"path": ["init"], "summary": "Scaffold a deployment dir (dry-run; --apply)."},
            {
                "path": ["serve"],
                "summary": "Start the vLLM server (alias: start; dry-run; --apply).",
            },
            {"path": ["stop"], "summary": "Stop the vLLM server (dry-run; --apply)."},
            {"path": ["switch"], "summary": "Switch the served model (dry-run; --apply)."},
            {
                "path": ["up"],
                "summary": (
                    "Start ONE Colleague role (cortex/senses/embedder/reranker/stt/tts) "
                    "or the full 'colleague-stack'; --down to stop (dry-run; --apply)."
                ),
            },
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
                "path": ["capabilities"],
                "summary": (
                    "Resolve the six Colleague roles (cortex/senses/embedder/reranker/"
                    "stt/tts) to endpoint + metadata (--json)."
                ),
            },
            {
                "path": ["endpoint"],
                "summary": "Print one role's base URL (cortex/senses/embedder/reranker/stt/tts).",
            },
            {
                "path": ["measure"],
                "summary": (
                    "Per-role runtime metrics — TTFT/decode-tps, docs/sec, RTF " "(--role; --json)."
                ),
            },
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
                "up",
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
                "(lobes/catalog.py, shipped in the wheel). Read: 'lobes overview --list' "
                "or gateway GET /v1/models/supported."
            ),
            "loaded_now": (
                "Model(s) actually in GPU memory right now. Live source: GET /v1/models "
                "(which 'lobes fleet status' queries). 'lobes status'/'lobes whoami' report the "
                "configured served model (from .env) + health, not a live list."
            ),
            "task_families": (
                "The fleet routes by task family on one gateway port: 'generate' (the "
                "Qwen primary), 'embed' (Qwen3-Embedding-0.6B → /v1/embeddings), and "
                "'score'/'rerank' (Qwen3-Reranker-0.6B → /v1/rerank + /v1/score). The "
                "embedding + reranker gears co-reside with the primary (util 0.06 each); "
                "a second generate backend (warm fallback) is opt-in."
            ),
            "realtime_audio": (
                "Opt-in overlay ('lobes init --fleet --audio'): an OpenAI /v1/audio/* "
                "facade (a 'realtime' bridge) backed by two fixed sidecars — Parakeet STT "
                "(/v1/audio/transcriptions) and Chatterbox TTS (/v1/audio/speech, 24 kHz, "
                "voice cloning). Both open-weights (no NGC key); hardcoded, not switchable. "
                "See 'lobes explain realtime'."
            ),
            "api_surface": (
                "One OpenAI-compatible port (default :8000), routed by the request's "
                "'model' field: /v1/chat/completions, /v1/completions, /v1/embeddings, "
                "/v1/rerank, /v1/score, /v1/audio/transcriptions, /v1/audio/speech, "
                "/v1/models (loaded), /v1/models/supported (catalog), /health. See "
                "'lobes explain api' and docs/openai-api.md."
            ),
            "auth_exposure": (
                "Set CULTURE_VLLM_API_KEY in .env before exposing the API; vLLM enforces "
                "it as a bearer token on the single-model 'lobes serve' path. The fleet "
                "gateway is a pass-through and is NOT auth-aware (known limitation) — bearer "
                "keys don't protect its proxied endpoints, so add Cloudflare Access or an IP "
                "allowlist when tunnelling the fleet. Expose via 'lobes tunnel'. See "
                "'lobes explain tunnel' / 'lobes explain gateway'."
            ),
        },
        "explain_pointer": "lobes explain <path> (e.g. 'lobes explain switch')",
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
