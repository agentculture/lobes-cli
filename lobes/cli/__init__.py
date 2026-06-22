"""Unified CLI entry point for lobes (binary: ``lobes``; ``model`` is a deprecated alias).

The model-ops verbs (``switch``, ``serve``/``stop``, ``status``, ``assess``,
``benchmark``, ``init``, ``tunnel``) are the heart of the tool; the agent-first verbs
(``whoami``, ``learn``, ``explain``, ``overview``, ``doctor``, ``cli``) keep the
sibling rubric satisfied. Each verb module exposes ``register(sub)`` following
the same pattern.

Error propagation contract
--------------------------
Every handler raises :class:`lobes.cli._errors.ModelGearError` on failure;
``main()`` catches it via :func:`_dispatch` and routes through
:mod:`lobes.cli._output`. Unknown exceptions are wrapped into a
``ModelGearError`` so no Python traceback leaks to stderr.

Argparse errors (unknown verb, missing arg) also route through the structured
format — ``_ModelGearArgumentParser`` overrides ``.error()`` and the subparsers
are built with ``parser_class=_ModelGearArgumentParser``. Whether errors render
as text or JSON depends on whether ``--json`` appears in the raw argv
(:func:`main` sets ``_json_hint`` before ``parse_args``).
"""

from __future__ import annotations

import argparse
import sys

from lobes import __version__
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_error


class _ModelGearArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that routes errors through :func:`emit_error`.

    Argparse's default error handler writes ``prog: error: <msg>`` to stderr
    and exits 2, skipping the ModelGearError plumbing (and the ``hint:`` line
    agents look for). This subclass emits the structured format and exits with
    :attr:`EXIT_USER_ERROR`.

    JSON mode: parse-time errors happen before ``args.json`` exists, so we rely
    on a class-level ``_json_hint`` that :func:`main` pre-populates by scanning
    raw argv for ``--json``. Shared across all subparser instances.
    """

    _json_hint: bool = False

    def error(self, message: str) -> None:  # type: ignore[override]
        err = ModelGearError(
            code=EXIT_USER_ERROR,
            message=message,
            remediation=f"run '{self.prog} --help' to see valid arguments",
        )
        emit_error(err, json_mode=type(self)._json_hint)
        raise SystemExit(err.code)


def _argv_has_json(argv: list[str] | None) -> bool:
    tokens = argv if argv is not None else sys.argv[1:]
    return any(t == "--json" or t.startswith("--json=") for t in tokens)


def _detect_prog() -> str:
    """Return the invocation name so ``--version`` and help text match the binary.

    When invoked as ``lobes`` → ``"lobes"``; as ``model`` (deprecated alias) → ``"model"``;
    as ``python -m lobes`` → ``"lobes"``.
    """
    import os

    argv0 = os.path.basename(sys.argv[0]) if sys.argv else "lobes"
    # Strip .py suffix (python -m lobes → __main__.py on some Python versions)
    name = argv0.removesuffix(".py").removesuffix("__main__")
    return name if name in ("lobes", "model") else "lobes"


def _build_parser() -> argparse.ArgumentParser:
    from lobes.cli._commands import assess as _assess_cmd
    from lobes.cli._commands import benchmark as _benchmark_cmd
    from lobes.cli._commands import cli as _cli_group
    from lobes.cli._commands import doctor as _doctor_cmd
    from lobes.cli._commands import explain as _explain_cmd
    from lobes.cli._commands import fleet as _fleet_cmd
    from lobes.cli._commands import init as _init_cmd
    from lobes.cli._commands import learn as _learn_cmd
    from lobes.cli._commands import logs as _logs_cmd
    from lobes.cli._commands import overview as _overview_cmd
    from lobes.cli._commands import serve as _serve_cmd
    from lobes.cli._commands import status as _status_cmd
    from lobes.cli._commands import stop as _stop_cmd
    from lobes.cli._commands import switch as _switch_cmd
    from lobes.cli._commands import tunnel as _tunnel_cmd
    from lobes.cli._commands import whoami as _whoami_cmd

    parser = _ModelGearArgumentParser(
        prog=_detect_prog(),
        description="lobes — run, assess, and switch the local vLLM model",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    # parser_class propagates to every subparser so their .error() routes
    # through _ModelGearArgumentParser too.
    sub = parser.add_subparsers(dest="command", parser_class=_ModelGearArgumentParser)

    # Model-ops verbs (the heart of the tool).
    _switch_cmd.register(sub)
    _serve_cmd.register(sub)
    _stop_cmd.register(sub)
    _status_cmd.register(sub)
    _assess_cmd.register(sub)
    _benchmark_cmd.register(sub)
    _init_cmd.register(sub)
    _fleet_cmd.register(sub)
    _logs_cmd.register(sub)
    _tunnel_cmd.register(sub)

    # Agent-first / introspection verbs (sibling rubric).
    _whoami_cmd.register(sub)
    _learn_cmd.register(sub)
    _explain_cmd.register(sub)
    _overview_cmd.register(sub)
    _doctor_cmd.register(sub)
    _cli_group.register(sub)

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    """Invoke the registered handler and translate exceptions to exit codes.

    A handler may return ``None`` (success, exit 0) or an ``int`` exit code.
    Failures MUST raise :class:`ModelGearError`; any other exception is wrapped
    into one so no Python traceback leaks.
    """
    json_mode = bool(getattr(args, "json", False))
    try:
        rc = args.func(args)
    except ModelGearError as err:
        emit_error(err, json_mode=json_mode)
        return err.code
    except Exception as err:  # noqa: BLE001 - last-resort; wrap and route cleanly
        wrapped = ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"unexpected: {err.__class__.__name__}: {err}",
            remediation="file a bug at https://github.com/agentculture/lobes-cli/issues",
        )
        emit_error(wrapped, json_mode=json_mode)
        return wrapped.code
    return rc if rc is not None else 0


def main(argv: list[str] | None = None) -> int:
    # Pre-parse peek so argparse-level errors honour --json.
    _ModelGearArgumentParser._json_hint = _argv_has_json(argv)
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return _dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
