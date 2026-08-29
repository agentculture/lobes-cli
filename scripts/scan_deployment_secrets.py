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

# A value that is purely a shell/compose variable reference (e.g.
# ``${GATEWAY_API_KEY}`` or ``${GATEWAY_API_KEY:-}``) is a template
# placeholder, not a committed secret — it carries no value at rest in the
# file. Anything else non-empty is flagged.
_PURE_VAR_REF_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(:?-[^}]*)?\}$")

# KEY=value (.env / Dockerfile ENV|ARG) and KEY: value / "- KEY=value"
# (compose YAML) assignment shapes. Captures the key name and the
# remainder of the line as the raw value.
_ASSIGNMENT_RE = re.compile(
    r"""
    ^\s*
    (?:-\s+)?                  # optional YAML list-item dash (environment: lists)
    (?:ENV\s+|ARG\s+)?         # optional Dockerfile directive
    ["']?([A-Za-z_][A-Za-z0-9_]*)["']?   # key name, optionally quoted
    \s*[:=]\s*
    (.*)$
    """,
    re.VERBOSE,
)


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


def scan_text(path: Path, text: str) -> list[Finding]:
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
