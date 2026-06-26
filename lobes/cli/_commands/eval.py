"""``lobes eval`` — read-only evaluation harness for lobes backends.

Suite file format (JSONL)
-------------------------
A **suite** is a UTF-8 text file where every non-blank, non-comment line
is a JSON object::

    {"prompt": "What is 2+2?",        "expect_substring": "4"}
    {"prompt": "Is the sky blue?",    "expect_regex": "yes|true|blue"}
    # This line is a comment and is ignored.

Fields per case object:

``prompt`` (str, **required**)
    The user message sent to the backend via ``lobes.minor.chat_text``.

Exactly one expectation field (also **required**):

``expect_substring`` (str)
    The model response must contain this string (case-sensitive).

``expect_regex`` (str)
    The model response must match this ``re.search`` pattern.  The pattern is
    applied to the full response string with no flags.

Blank lines and lines whose first non-whitespace character is ``#`` are
silently skipped.  Any other line that is not valid JSON raises
:class:`~lobes.cli._errors.ModelGearError` with :attr:`EXIT_USER_ERROR`.

Handler
-------
``lobes eval minor --suite <path> [--base-url URL] [--model NAME]
                                  [--timeout SECS] [--json]``

* **Read-only** — never changes the world; ``--apply`` is not needed.
* Reports per-case ``PASS`` / ``FAIL`` and an aggregate ``passed/total``.
* On ``--json``: emits one structured dict with ``passed``, ``total``, and
  ``cases`` (each case carries ``index``, ``prompt``, ``response``, ``pass``,
  and ``expectation``).
* Exit code is always ``0`` — pass/fail lives in the report, not the exit code.
  A missing suite file is the only error that exits non-zero (via
  :class:`ModelGearError`).

Empty suite (zero cases after skipping blanks/comments): reports ``0/0``, no
HTTP calls made, exit ``0``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.minor import chat_text  # patched in tests via eval_cmd.chat_text

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------


def _load_suite(path: Path) -> list[dict]:
    """Load and parse a JSONL suite file; return a list of case dicts.

    Raises :class:`ModelGearError` if the file is missing or a data line is
    malformed / missing required fields.
    """
    if not path.exists():
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"suite file not found: {path}",
            remediation="pass an existing .jsonl file path via --suite",
        )

    cases: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"suite parse error at line {lineno}: {exc}",
                remediation="every non-blank, non-comment line must be valid JSON",
            ) from exc
        if "prompt" not in obj:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"suite line {lineno}: missing required field 'prompt'",
                remediation="each case object must have a 'prompt' string field",
            )
        if "expect_substring" not in obj and "expect_regex" not in obj:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=(
                    f"suite line {lineno}: no expectation field "
                    "(need 'expect_substring' or 'expect_regex')"
                ),
                remediation=(
                    "add 'expect_substring' (substring check) "
                    "or 'expect_regex' (re.search pattern) to the case"
                ),
            )
        cases.append(obj)
    return cases


# ---------------------------------------------------------------------------
# Expectation checking
# ---------------------------------------------------------------------------


def _check(response: str, case: dict) -> bool:
    """Return ``True`` if *response* satisfies the case expectation."""
    if "expect_substring" in case:
        return case["expect_substring"] in response
    if "expect_regex" in case:
        return bool(re.search(case["expect_regex"], response))
    return False  # unreachable after _load_suite validation


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_eval_minor(args: argparse.Namespace) -> int:
    """Handler for ``lobes eval minor``."""
    json_mode = bool(getattr(args, "json", False))
    suite_path = Path(str(args.suite))
    base_url: str = getattr(args, "base_url", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL
    model: str = getattr(args, "model", None) or ""
    timeout: int = int(getattr(args, "timeout", _DEFAULT_TIMEOUT))

    cases = _load_suite(suite_path)

    results: list[dict] = []
    for idx, case in enumerate(cases):
        response = chat_text(
            case["prompt"],
            base_url=base_url,
            model=model,
            timeout=timeout,
        )
        passed = _check(response, case)
        expectation: dict = {}
        if "expect_substring" in case:
            expectation = {"expect_substring": case["expect_substring"]}
        elif "expect_regex" in case:
            expectation = {"expect_regex": case["expect_regex"]}
        results.append(
            {
                "index": idx,
                "prompt": case["prompt"],
                "response": response,
                "pass": passed,
                "expectation": expectation,
            }
        )

    passed_count = sum(1 for r in results if r["pass"])
    total = len(results)

    report: dict = {
        "passed": passed_count,
        "total": total,
        "cases": results,
    }

    if json_mode:
        emit_result(report, json_mode=True)
    else:
        lines: list[str] = []
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            lines.append(f"  [{status}] case {r['index']}: {r['prompt']!r}")
        lines.append(f"result: {passed_count}/{total} passed")
        emit_result("\n".join(lines), json_mode=False)

    return 0


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``eval`` noun group and its ``minor`` sub-verb."""
    p = sub.add_parser(
        "eval",
        help="Read-only: run a JSONL eval suite against a lobes backend.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    # When called as bare ``lobes eval`` with no sub-verb, print help.
    p.set_defaults(func=lambda a: (p.print_help(), 0)[1], json=False)

    # Propagate the custom parser class (if this parser has one) so sub-verb
    # parse errors also route through the structured error contract.
    noun_sub = p.add_subparsers(dest="eval_command", parser_class=type(p))

    minor_p = noun_sub.add_parser(
        "minor",
        help="Evaluate using the lobes.minor stdlib client (no third-party deps).",
    )
    minor_p.add_argument(
        "--suite",
        required=True,
        metavar="PATH",
        help="Path to a JSONL eval suite file (see 'lobes eval --help' for format).",
    )
    minor_p.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        metavar="URL",
        help=f"OpenAI-compatible base URL (default: {_DEFAULT_BASE_URL!r}).",
    )
    minor_p.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Model identifier forwarded in each request (default: empty string).",
    )
    minor_p.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT,
        metavar="SECS",
        help=f"Socket timeout in seconds (default: {_DEFAULT_TIMEOUT}).",
    )
    minor_p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    minor_p.set_defaults(func=cmd_eval_minor)
