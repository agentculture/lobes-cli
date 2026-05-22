"""``lepenseur explain <path>...`` — global markdown catalog lookup.

Takes zero or more path tokens and resolves them via :mod:`lepenseur.explain`.
Unknown paths raise :class:`LepenseurError` with a remediation pointing at
``lepenseur explain lepenseur``.
"""

from __future__ import annotations

import argparse

from lepenseur.cli._output import emit_result
from lepenseur.explain import resolve


def cmd_explain(args: argparse.Namespace) -> int:
    path = tuple(args.path) if args.path else ()
    markdown = resolve(path)  # raises LepenseurError on miss → caught in _dispatch
    json_mode = bool(getattr(args, "json", False))
    if json_mode:
        emit_result({"path": list(path), "markdown": markdown}, json_mode=True)
    else:
        emit_result(markdown, json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "explain",
        help="Print markdown docs for a topic (e.g. 'lepenseur explain backend').",
    )
    p.add_argument(
        "path",
        nargs="*",
        help="Topic path tokens; empty = root (same as 'lepenseur').",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_explain)
