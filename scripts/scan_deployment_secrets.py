#!/usr/bin/env python3
"""Secret gate over every committed deployment artifact (t3 of the
deployment-lock-per-box plan, docs/plans/2026-08-29-deployment-lock-per-box.md).

No CI job scanned committed files for secrets before this script existed —
"secrets stay in .env" was enforced only by the .gitignore pattern and
reviewer attention. The deployment-lock practice deliberately widens what
gets committed (rendered compose files, an operator-authored
docker-compose.override.yml, Dockerfiles, and a generated
deployment.lock.toml), so this script is the mechanical gate that keeps
that claim honest.

Scope, deliberately narrower than a generic secret scanner: this only
flags KNOWN deployment-secret key names carrying a non-empty, non-template
value. It is not a general entropy/regex secret scanner (gitleaks etc.);
see the plan's t3 rationale for why a small, unit-testable, stdlib script
is preferred over depending solely on a third-party action.

Crucially, the path list below is NOT "the repository's default file set" —
it names the lock (``deployment.lock.toml``) and every verbatim-committed
compose/override/Dockerfile under ``deployments/<box>/`` explicitly, because
the lock writer's own allowlist (t6) protects only the generated lock and
provides zero protection to a hand-authored docker-compose.override.yml
lobes never writes and cannot vouch for.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

# Secret-shaped key names (see CLAUDE.md's "Machine profiles" section and
# lobes/templates/fleet/env.example for the full vocabulary). Suffix
# patterns cover every role prefix (PRIMARY_, MULTIMODAL_, MUSE_, WORKER_,
# HAND_, ASSOCIATE_, EMBED_, RERANK_, STT_, TTS_, ...) without enumerating
# roles, so a future role's key is covered by construction.
_EXACT_SECRET_KEYS = frozenset({"GATEWAY_API_KEY", "CULTURE_VLLM_API_KEY", "HF_TOKEN"})
_SECRET_KEY_SUFFIXES = ("_PEER_API_KEY", "_PEER_API_KEYS", "_PEER_ORIGIN", "_PEER_ORIGINS")

# Glob patterns, relative to the scan root, naming every committed
# deployment artifact this gate must cover. Deliberately explicit rather
# than "scan everything" — see the module docstring.
DEFAULT_SCAN_GLOBS = (
    "deployments/**/deployment.lock.toml",
    "deployments/**/docker-compose*.yml",
    "deployments/**/docker-compose*.yaml",
    "deployments/**/Dockerfile*",
)

# A value that is purely a shell/compose variable reference is a template
# placeholder, not a committed secret — it carries no value at rest.
#
# ONLY three spellings qualify, and the narrowness is the point (PR #223
# review, defect 1). Docker Compose's ``${VAR:-default}`` / ``${VAR-default}``
# interpolation SUBSTITUTES the literal default when the variable is absent
# (``:-`` also when it is set-but-empty), so a non-empty default really is a
# credential committed at rest — the file works, with that value, on a box
# that never exports the variable. An earlier ``(:?-[^}]*)?`` permitted
# arbitrary fallback content and let exactly that through.
#
#   ${VAR}    bare reference, no default            -> safe
#   ${VAR:-}  explicitly empty default (unset/empty) -> safe
#   ${VAR-}   explicitly empty default (unset)       -> safe
#
# Anything else — including other expansion operators such as ``${VAR:?err}``
# or ``${VAR:+x}`` — is flagged, deliberately erring toward a noisy gate over
# a silent one.
_PURE_VAR_REF_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(?::?-)?\}$")

# KEY=value (.env) and KEY: value / "- KEY=value" (compose YAML) assignment
# shapes. Captures the key name and the remainder of the line as the raw
# value. Dockerfile ENV/ARG is a genuinely different grammar and gets its own
# parser below — see ``_scan_dockerfile``.
_ASSIGNMENT_RE = re.compile(
    r"""
    ^\s*
    (?:-\s+)?                  # optional YAML list-item dash (environment: lists)
    ["']?([A-Za-z_][A-Za-z0-9_]*)["']?   # key name, optionally quoted
    \s*[:=]\s*
    (.*)$
    """,
    re.VERBOSE,
)

# Dockerfile ENV/ARG instruction. Directives are case-insensitive in
# practice, and BOTH assignment forms are valid:
#
#   ENV KEY value            (whitespace form, single variable, value = rest)
#   ENV KEY=v1 KEY2=v2       (equals form, multiple variables per instruction)
#   ARG KEY[=default]
#
# The whitespace form has no ``=`` at all, so the assignment regex above
# never matched it and a committed ``ENV HF_TOKEN <secret>`` sailed straight
# through the gate (PR #223 review, defect 2).
_DOCKERFILE_INSTR_RE = re.compile(r"^\s*(ENV|ARG)\s+(.+)$", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_secret_key(name: str) -> bool:
    if name in _EXACT_SECRET_KEYS:
        return True
    return any(name.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)


def _clean_value(raw: str) -> str:
    value = raw.strip()
    # Strip a trailing unquoted-string comment (# ...), but never touch a
    # '#' that lives inside a quoted value.
    if value and value[0] not in "\"'":
        hash_idx = value.find("#")
        if hash_idx != -1:
            value = value[:hash_idx].strip()
    # Strip a single layer of matching quotes.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip()


@dataclass(frozen=True)
class Finding:
    path: Path
    line_no: int
    key: str
    value: str

    def render(self, root: Path | None = None) -> str:
        shown = self.path.relative_to(root) if root else self.path
        return f"{shown}:{self.line_no}: {self.key} carries a non-empty value"


def is_dockerfile(path: Path) -> bool:
    """True for the ``Dockerfile*`` artifacts the glob list names.

    Dockerfiles are parsed with the ENV/ARG instruction grammar rather than
    the .env/YAML assignment grammar; the two genuinely differ.
    """
    name = path.name.lower()
    return name.startswith("dockerfile") or name.endswith(".dockerfile")


def _iter_dockerfile_instructions(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line_no, joined_instruction)``, honouring ``\\`` continuations.

    ``line_no`` is the line the instruction STARTS on, so a finding on a
    continued ENV points at the instruction a reviewer has to read.
    """
    buffer: list[str] = []
    start = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            # A comment line is skipped whether or not it sits inside a
            # continuation — Docker strips it before joining.
            continue
        if not buffer:
            if not stripped:
                continue
            start = line_no
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1])
            continue
        buffer.append(stripped)
        yield start, " ".join(part for part in buffer if part)
        buffer = []
    if buffer:
        yield start, " ".join(part for part in buffer if part)


def _dockerfile_assignments(remainder: str) -> list[tuple[str, str]]:
    """Split an ENV/ARG remainder into ``(key, value)`` pairs.

    Two forms, distinguished exactly as Docker does — by whether the FIRST
    token carries an ``=``:

    * ``KEY value with spaces`` — one variable, value is the whole remainder.
    * ``KEY1=a KEY2=b`` — any number of variables.
    """
    try:
        tokens = shlex.split(remainder, comments=False, posix=True)
    except ValueError:  # unbalanced quotes — fall back to naive splitting
        tokens = remainder.split()
    if not tokens:
        return []

    if "=" not in tokens[0]:
        key = tokens[0]
        return [(key, " ".join(tokens[1:]))] if _IDENTIFIER_RE.match(key) else []

    pairs: list[tuple[str, str]] = []
    for token in tokens:
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if _IDENTIFIER_RE.match(key):
            pairs.append((key, value))
    return pairs


def _scan_dockerfile(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_no, instruction in _iter_dockerfile_instructions(text):
        match = _DOCKERFILE_INSTR_RE.match(instruction)
        if not match:
            continue
        for key, value in _dockerfile_assignments(match.group(2)):
            if not _is_secret_key(key):
                continue
            # shlex has already removed one layer of quoting; a '#' inside a
            # Dockerfile instruction is literal, so no comment stripping.
            value = value.strip()
            if not value or _PURE_VAR_REF_RE.match(value):
                continue
            findings.append(Finding(path=path, line_no=line_no, key=key, value=value))
    return findings


def _scan_assignments(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT_RE.match(line)
        if not match:
            continue
        key, raw_value = match.group(1), match.group(2)
        if not _is_secret_key(key):
            continue
        value = _clean_value(raw_value)
        if not value:
            continue
        if _PURE_VAR_REF_RE.match(value):
            continue
        findings.append(Finding(path=path, line_no=line_no, key=key, value=value))
    return findings


def scan_text(path: Path, text: str) -> list[Finding]:
    if is_dockerfile(path):
        return _scan_dockerfile(path, text)
    return _scan_assignments(path, text)


def scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return scan_text(path, text)


def iter_scan_paths(root: Path, globs: Iterable[str] = DEFAULT_SCAN_GLOBS) -> Iterator[Path]:
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def scan_paths(root: Path, globs: Iterable[str] = DEFAULT_SCAN_GLOBS) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_scan_paths(root, globs):
        findings.extend(scan_file(path))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repo root to scan from (default: cwd)",
    )
    parser.add_argument(
        "--glob",
        action="append",
        dest="globs",
        default=None,
        help="override the default glob list (repeatable)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    globs = args.globs if args.globs else DEFAULT_SCAN_GLOBS

    findings = scan_paths(root, globs)

    if findings:
        print("Secret-shaped values found in committed deployment artifacts:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding.render(root)}", file=sys.stderr)
        print(
            f"\n{len(findings)} finding(s). Secrets belong only in a gitignored "
            "*.env file — see CLAUDE.md and docs/specs/2026-08-29-deployment-lock-per-box.md.",
            file=sys.stderr,
        )
        return 1

    scanned = list(iter_scan_paths(root, globs))
    print(f"scan-deployment-secrets: clean ({len(scanned)} file(s) scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
