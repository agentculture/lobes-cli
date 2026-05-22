"""``lepenseur learn`` — the learnability affordance.

Prints a structured self-teaching prompt with enough shape that an agent can
author its own usage skill without scraping ``--help``. Supports ``--json`` for
agents that prefer structure to prose.
"""

from __future__ import annotations

import argparse

from lepenseur import __version__
from lepenseur.cli._output import emit_result

_TEXT = """\
lepenseur — the local coding agent of the Culture mesh.

Purpose
-------
lepenseur ("le codeur" — the coder) implements, edits, and tests code. It is the
*doer* of a matched pair: lepenseur ("le penseur" — the thinker) reasons and
plans; lepenseur executes. daria (awareness) is the next-closest sibling. At
runtime lepenseur is served by a local vLLM code model over the acp backend (not
Claude-backed) — see 'lepenseur explain backend'.

Commands
--------
  lepenseur whoami        Smallest identity probe: nick, version, backend,
                         served model (read from culture.yaml). Supports --json.
  lepenseur learn         Print this self-teaching prompt. Supports --json.
  lepenseur explain <path>...
                         Print markdown docs for a topic (e.g.
                         'lepenseur explain backend'). Supports --json.

Mutation safety
---------------
Any future verb that writes defaults to dry-run; pass --apply to commit. The
verbs above are read-only.

Machine-readable output
-----------------------
Every command supports --json. Errors in JSON mode emit
{"code", "message", "remediation"} to stderr. Stdout and stderr are never mixed.

Exit-code policy
----------------
  0 success
  1 user-input error (bad flag, bad path, missing arg)
  2 environment / setup error (tool not installed, unreadable file)
  3+ reserved

More detail
-----------
  lepenseur explain lepenseur
  lepenseur explain backend
  lepenseur explain whoami

Homepage: https://github.com/agentculture/lepenseur
"""


def _as_json_payload() -> dict[str, object]:
    return {
        "tool": "lepenseur",
        "version": __version__,
        "purpose": (
            "The local coding agent of the Culture mesh: implements, edits, and "
            "tests code. The 'doer' to lepenseur's 'thinker'."
        ),
        "siblings": {"closest": "lepenseur", "next": "daria"},
        "commands": [
            {"path": ["whoami"], "summary": "Identity probe (nick, version, backend, model)."},
            {"path": ["learn"], "summary": "Self-teaching prompt."},
            {"path": ["explain"], "summary": "Markdown docs by topic path."},
        ],
        "exit_codes": {
            "0": "success",
            "1": "user-input error",
            "2": "environment/setup error",
        },
        "json_support": True,
        "explain_pointer": "lepenseur explain <path> (e.g. 'lepenseur explain backend')",
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
