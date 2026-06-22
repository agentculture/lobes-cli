"""``model cli`` — noun grouping CLI-surface introspection.

Exists to satisfy the agent-first rubric's ``overview_cli_noun_exists`` check.
``model cli overview`` describes the CLI surface itself (distinct from the global
``overview``, which describes the tool and the served model).
"""

from __future__ import annotations

import argparse

from lobes.cli._commands.overview import cli_sections, emit_overview


def cmd_cli_overview(args: argparse.Namespace) -> int:
    emit_overview("model cli", cli_sections(), json_mode=bool(getattr(args, "json", False)))
    return 0


def _no_verb(args: argparse.Namespace) -> int:
    # `model cli` with no sub-verb prints the noun's overview.
    return cmd_cli_overview(args)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "cli",
        help="CLI-surface introspection (see 'model cli overview').",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=_no_verb, json=False)
    # `p` is a _ModelGearArgumentParser (the top-level subparsers were built with
    # that parser_class); propagate it so `cli overview` parse errors route through
    # the structured error contract instead of argparse's default stderr/exit 2.
    noun_sub = p.add_subparsers(dest="cli_command", parser_class=type(p))
    ov = noun_sub.add_parser("overview", help="Describe the lobes CLI surface.")
    ov.add_argument("--json", action="store_true", help="Emit structured JSON.")
    ov.set_defaults(func=cmd_cli_overview)
