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

from lobes.bench.cat_probe import generate_case  # noqa: F401 (re-exported for tests)
from lobes.bench.cat_score import score_case  # patched in tests via eval_cmd.score_case
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.minor import chat_text  # patched in tests via eval_cmd.chat_text

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_TIMEOUT = 60
_JSON_HELP = "Emit structured JSON."
_CAT_VALID_MODES = ("open", "closed")


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
# Cat-suite loading
# ---------------------------------------------------------------------------


def _load_cat_suite(path: Path) -> list[dict]:
    """Load and parse a cat-probe JSONL suite; return a list of case dicts.

    Each non-blank, non-comment line must be a JSON object with at least a
    ``seed`` (int) field.  Optional keys: ``mode`` (``"open"``/``"closed"``)
    and ``n_characters`` (int).

    Raises :class:`ModelGearError` if the file is missing, a data line is
    malformed, or the ``seed`` field is absent.
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
        if "seed" not in obj:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=f"suite line {lineno}: missing required field 'seed'",
                remediation="each cat-suite case object must have an integer 'seed' field",
            )
        try:
            int(obj["seed"])
        except (TypeError, ValueError):
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=(
                    f"suite line {lineno}: 'seed' must be an integer, " f"got {obj['seed']!r}"
                ),
                remediation="set 'seed' to an integer value (e.g. 42)",
            )
        if "mode" in obj and obj["mode"] not in _CAT_VALID_MODES:
            raise ModelGearError(
                code=EXIT_USER_ERROR,
                message=(
                    f"suite line {lineno}: invalid 'mode' {obj['mode']!r} "
                    f"(must be one of {_CAT_VALID_MODES})"
                ),
                remediation="set 'mode' to 'open' or 'closed'",
            )
        if "n_characters" in obj:
            try:
                n = int(obj["n_characters"])
                if n < 1:
                    raise ValueError("non-positive")
            except (TypeError, ValueError):
                raise ModelGearError(
                    code=EXIT_USER_ERROR,
                    message=(
                        f"suite line {lineno}: 'n_characters' must be a positive integer, "
                        f"got {obj['n_characters']!r}"
                    ),
                    remediation="set 'n_characters' to a positive integer (e.g. 4)",
                )
        cases.append(obj)
    return cases


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


def cmd_eval_cat(args: argparse.Namespace) -> int:
    """Handler for ``lobes eval cat``."""
    json_mode = bool(getattr(args, "json", False))
    suite_path = Path(str(args.suite))
    base_url: str = getattr(args, "base_url", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL
    model: str = getattr(args, "model", None) or ""
    timeout: int = int(getattr(args, "timeout", _DEFAULT_TIMEOUT))
    cli_mode: str = getattr(args, "mode", "closed") or "closed"

    suite_cases = _load_cat_suite(suite_path)

    results: list[dict] = []
    for entry in suite_cases:
        seed: int = int(entry["seed"])
        case_mode: str = entry.get("mode", cli_mode)
        n_characters: int = int(entry.get("n_characters", 4))

        case = generate_case(seed=seed, mode=case_mode, n_characters=n_characters)
        scored = score_case(case, base_url=base_url, model=model, timeout=timeout)

        results.append(
            {
                "seed": seed,
                "mode": case_mode,
                "answer": scored["answer"],
                "soft_score": scored["soft_score"],
                "headline": scored["headline"],
                "first_token_mass": scored["first_token_mass"],
                "echo_available": scored["echo_available"],
            }
        )

    mean_soft: float = sum(r["soft_score"] for r in results) / len(results) if results else 0.0

    report: dict = {
        "mode": cli_mode,
        "score": "logprobs",
        "mean_soft_score": mean_soft,
        "cases": results,
    }

    if json_mode:
        emit_result(report, json_mode=True)
    else:
        lines: list[str] = []
        for r in results:
            lines.append(
                f"  seed={r['seed']} mode={r['mode']}"
                f" answer={r['answer']!r}"
                f" soft_score={r['soft_score']:.4f}"
                f" headline={r['headline']}"
                f" first_token_mass={r['first_token_mass']:.4f}"
            )
        lines.append(f"mean soft-score: {mean_soft:.4f}")
        emit_result("\n".join(lines), json_mode=False)

    return 0


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``eval`` noun group and its ``minor`` and ``cat`` sub-verbs."""
    p = sub.add_parser(
        "eval",
        help="Read-only: run a JSONL eval suite against a lobes backend.",
    )
    p.add_argument("--json", action="store_true", help=_JSON_HELP)
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
    minor_p.add_argument("--json", action="store_true", help=_JSON_HELP)
    minor_p.set_defaults(func=cmd_eval_minor)

    cat_p = noun_sub.add_parser(
        "cat",
        help=(
            "Evaluate the cat-probe (temporal-reasoning) using logprobs scoring. "
            "Read-only; exit code is always 0."
        ),
    )
    cat_p.add_argument(
        "--suite",
        required=True,
        metavar="PATH",
        help=(
            "Path to a cat-probe JSONL suite file. "
            "Each non-blank, non-comment line must be a JSON object with an integer "
            "'seed' field, and optional 'mode' and 'n_characters' overrides."
        ),
    )
    cat_p.add_argument(
        "--score",
        default="logprobs",
        choices=["logprobs"],
        metavar="SCORER",
        help="Scoring method (default: 'logprobs'; only logprobs is supported).",
    )
    cat_p.add_argument(
        "--mode",
        default="closed",
        choices=["open", "closed"],
        metavar="MODE",
        help=(
            "Probe mode applied to cases whose suite line does not override it "
            "(default: 'closed'). 'closed' enumerates candidate locations in the "
            "prompt; 'open' omits the options list."
        ),
    )
    cat_p.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        metavar="URL",
        help=f"OpenAI-compatible base URL (default: {_DEFAULT_BASE_URL!r}).",
    )
    cat_p.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Model identifier forwarded in each request (default: empty string).",
    )
    cat_p.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT,
        metavar="SECS",
        help=f"Socket timeout in seconds (default: {_DEFAULT_TIMEOUT}).",
    )
    cat_p.add_argument("--json", action="store_true", help=_JSON_HELP)
    cat_p.set_defaults(func=cmd_eval_cat)
