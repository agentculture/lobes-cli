"""``model overview`` — read-only descriptive snapshot of the tool.

Describes model-gear to an agent reader: identity (tool/version/machine), the
verb surface, capabilities, the currently-served model, and the supported-model
catalog (the gears you can change to). ``--current`` shows only the served-model
block; ``--list`` shows only the supported-model catalog.

The shared section/render helpers here are reused by the ``cli`` noun's
``overview`` (see :mod:`model_gear.cli._commands.cli`). Descriptive verbs never
hard-fail on a missing target path — an optional positional ``target`` is
accepted and ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_gear.catalog import supported_models
from model_gear.cli import _live, _runtime_ops
from model_gear.cli._commands.whoami import report
from model_gear.cli._errors import ModelGearError
from model_gear.cli._output import emit_result
from model_gear.runtime import _compose, _env

_VERBS = [
    "init [TARGET] — scaffold a deployment dir (--fleet for the gateway; dry-run; --apply)",
    "serve / stop — start / stop the vLLM server (dry-run; --apply)",
    "switch <model> — switch the served model (dry-run; --apply)",
    "fleet up / down / status — drive the gateway fleet: generate primary + co-resident "
    "embedding + reranker gears, routed by task family (dry-run; --apply)",
    "tunnel — expose the local API at a public hostname via a Cloudflare Tunnel "
    "(--stop; dry-run; --apply)",
    "status — current model, container state, /health",
    "logs — read-only: list/tail the durable vLLM logs that survive restart (issue #50)",
    "assess — correctness probes against the served model",
    "benchmark — decode throughput + prefill latency",
    "overview — this snapshot (--current / --list to filter; --live for the running fleet)",
    "whoami — tool, machine, served model, container health",
    "explain <path> — markdown docs for a topic",
    "doctor — diagnose docker / compose / .env / health",
]

_CAPABILITIES = [
    "run — init / serve / stop the local vLLM deployment",
    "assess — correctness probes against the served model",
    "switch — change the served model (dry-run by default)",
    "benchmark — decode throughput + prefill latency",
    "fleet — front the Qwen primary plus co-resident embedding + reranker gears with one "
    "OpenAI gateway, routed by task family (a generate fallback is opt-in)",
]


def _docs_dir() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        d = parent / "docs"
        if d.is_dir():
            return d
    return None


def _first_heading(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except OSError:
        pass
    return path.stem


def candidate_models() -> list[dict[str, str]]:
    """The supported-model catalog — the gears you can change to.

    Sourced from the packaged :mod:`model_gear.catalog` so it is populated even in
    a wheel install (where ``docs/`` is not shipped). When a source-tree ``docs/``
    is present, each title is enriched from the per-model doc's first heading.
    """
    docs = _docs_dir()
    out: list[dict[str, str]] = []
    for model in supported_models():
        doc_path = docs / model.doc if docs is not None else None
        if doc_path is not None and doc_path.is_file():
            title = _first_heading(doc_path)
        else:
            title = f"{model.id} — {model.role_hint}, {model.shape}"
        out.append({"doc": model.doc, "title": title})
    return out


def _served_section(ident: dict) -> dict[str, object]:
    gear = ident.get("gear", {})
    return {
        "title": "Currently served",
        "items": [
            f"model: {ident['served_model']}",
            f"port: {ident['port']}",
            f"gear: {gear.get('purpose', 'balanced')} / {gear.get('machine', 'spark')}",
            f"container health: {ident['container_health']}",
        ],
    }


def _candidates_section(ident: dict, candidates: list[dict[str, str]]) -> dict[str, object]:
    base = str(ident["served_model"]).split("/")[-1].lower()
    items = []
    for c in candidates:
        stem = Path(c["doc"]).stem.lower()
        served = base and (base in stem or stem in base)
        items.append(f"{c['doc']} — {c['title']}" + (" (served)" if served else ""))
    if not items:
        items = ["(no supported models in the catalog)"]
    return {"title": "Supported models", "items": items}


def tool_sections(*, current: bool = False, listing: bool = False) -> list[dict[str, object]]:
    """Sections describing model-gear the tool (used by the global verb)."""
    ident = report()
    if current:
        return [_served_section(ident)]
    candidates = candidate_models()
    if listing:
        return [_candidates_section(ident, candidates)]
    machine = ident["machine"]
    return [
        {
            "title": "Identity",
            "items": [
                f"tool: {ident['tool']}",
                f"version: {ident['version']}",
                f"machine: {machine['host']} ({machine['gpu']})",
                f"agent served: {ident['agent']}",
            ],
        },
        {"title": "Verbs", "items": list(_VERBS)},
        {"title": "Capabilities", "items": list(_CAPABILITIES)},
        _served_section(ident),
        _candidates_section(ident, candidates),
    ]


def cli_sections() -> list[dict[str, object]]:
    """Sections describing the CLI surface itself (used by `cli overview`)."""
    return [
        {
            "title": "Verbs",
            "items": list(_VERBS) + ["cli overview — describe the CLI surface (this command)"],
        },
        {
            "title": "Conventions",
            "items": [
                "every command supports --json",
                "write verbs (switch/serve/stop/init/tunnel) are dry-run by default; "
                "--apply to commit",
                "results to stdout, errors/diagnostics to stderr (never mixed)",
                "exit codes: 0 success, 1 user error, 2 environment error, 3+ reserved",
            ],
        },
    ]


def render_text(subject: str, sections: list[dict[str, object]]) -> str:
    lines = [f"# {subject}", ""]
    for section in sections:
        lines.append(f"## {section['title']}")
        for item in section["items"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).rstrip()


def emit_overview(subject: str, sections: list[dict[str, object]], *, json_mode: bool) -> None:
    if json_mode:
        emit_result({"subject": subject, "sections": sections}, json_mode=True)
    else:
        emit_result(render_text(subject, sections), json_mode=False)


def _served_name(args: argparse.Namespace) -> str | None:
    """Configured served-model name from ``.env`` (best-effort; None if unscaffolded).

    Resolved independently of the probed port so ``overview --live --port N`` still
    labels the model from the deployment's ``.env``.
    """
    try:
        deploy_dir = _compose.resolve_deployment_dir(getattr(args, "compose_dir", None))
    except ModelGearError:
        return None
    env_path = deploy_dir / _compose.ENV_FILE
    return (
        _env.read_env(env_path, "VLLM_SERVED_NAME") or _env.read_env(env_path, "VLLM_MODEL") or None
    )


def cmd_overview(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    if getattr(args, "live", False):
        # Live dashboard: probe the running deployment (gateway /status or a single
        # vLLM /metrics) for online/offered/busy/usage/endpoints. HTTP-only.
        port, _ = _runtime_ops.resolve_port_soft(args)
        sections = _live.live_sections(port, _served_name(args))
        emit_overview("model-gear (live)", sections, json_mode=json_mode)
        return 0
    sections = tool_sections(
        current=bool(getattr(args, "current", False)),
        listing=bool(getattr(args, "list", False)),
    )
    emit_overview("model-gear", sections, json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "overview",
        help="Read-only snapshot of model-gear (identity, verbs, served model, candidates).",
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Ignored — overview always describes model-gear itself.",
    )
    p.add_argument(
        "--current", action="store_true", help="Show only the configured served model (from .env)."
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Show only the supported-model catalog (the gears you can switch to).",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Live dashboard: what is online / offered / busy + usage + endpoints "
        "(probes the running deployment).",
    )
    p.add_argument("--port", type=int, help="Host port to probe with --live (default: VLLM_PORT).")
    p.add_argument(
        "--compose-dir", help="Deployment dir (default: $MODEL_GEAR_DIR or ~/.model-gear)."
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_overview)
