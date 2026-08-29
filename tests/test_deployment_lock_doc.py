"""Guard for docs/deployment-lock.md (deployment-lock-per-box plan, t12).

t12 is docs-only, so this test does NOT restate the doc's claims — it
cross-checks them against the REAL tree, so a future drift (a renamed flag, a
moved evidence transcript, a capture verb finally landing) fails loudly here
instead of silently rotting the page.

Two things it is deliberately strict about, because they are the honesty
requirements the covering plan cares about more than completeness:

* every ``docs/evidence/`` path the doc cites must resolve — the motivating
  2026-08-25 Spark/Thor divergence has to stay *cited*, never remembered;
* the page must keep saying what is NOT validated, and each of those
  statements must still be TRUE of the tree (no captured variation, no
  capture verb, an offline warn-only buildability preflight).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "deployment-lock.md"
CATALOG_ROOT = REPO_ROOT / "deployments"
CLI_COMMANDS = REPO_ROOT / "lobes" / "cli" / "_commands"

#: The transcript the doc cites for the before-state (the Spark's hand-edited
#: compose, found during #199 prep).
MOTIVATING_TRANSCRIPT = "docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt"

_EVIDENCE_RE = re.compile(r"docs/evidence/[A-Za-z0-9][A-Za-z0-9._+-]*")


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _imported_names(path: Path) -> set[str]:
    """Every name *path* actually imports (docstring mentions do not count)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }


# --- the page exists and is reachable ---------------------------------------


def test_the_deep_reference_exists() -> None:
    assert DOC.is_file()


def test_the_explain_page_points_at_the_deep_reference() -> None:
    from lobes.explain.catalog import ENTRIES

    page = ENTRIES[("lock",)]
    assert "docs/deployment-lock.md" in page
    # The three aliases resolve to one body — a reader who guesses any of them
    # lands on the same page.
    assert ENTRIES[("deployment-lock",)] is page
    assert ENTRIES[("variations",)] is page


# --- citations resolve ------------------------------------------------------


def test_the_motivating_incident_is_cited_to_a_committed_transcript() -> None:
    """Acceptance criterion 3: cited, not remembered."""
    text = _doc()
    assert MOTIVATING_TRANSCRIPT in text
    assert (REPO_ROOT / MOTIVATING_TRANSCRIPT).is_file()
    assert "#214" in text


def test_every_cited_evidence_path_resolves() -> None:
    missing = [
        cited
        for cited in sorted(set(_EVIDENCE_RE.findall(_doc())))
        if not (REPO_ROOT / cited).is_file()
    ]
    assert not missing, f"docs/deployment-lock.md cites missing transcripts: {missing}"


# --- each audience's mechanism actually exists ------------------------------


def test_each_audience_mechanism_names_something_real() -> None:
    """Acceptance criterion 1: no audience is served only by prose."""
    text = _doc()
    # operator + third party: the restore flag
    assert "--from-lock" in text
    assert "--from-lock" in (CLI_COMMANDS / "init.py").read_text(encoding="utf-8")
    # doctor: the finding id
    assert "lock_drift" in text
    assert '"lock_drift"' in (CLI_COMMANDS / "doctor.py").read_text(encoding="utf-8")
    # CI: the required job and the scanner it runs
    assert "secrets-scan" in text
    workflow = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "secrets-scan:" in workflow
    assert (REPO_ROOT / "scripts" / "scan_deployment_secrets.py").is_file()
    # third party: the catalog they read from a checkout alone
    assert (CATALOG_ROOT / "README.md").is_file()


# --- the honesty statements are still true of the tree ----------------------


def test_the_doc_declares_an_unvalidated_section() -> None:
    text = _doc()
    assert "## What is not validated" in text
    for marker in ("No real box has been captured", "Serve-after-restore is unmeasured"):
        assert marker in text


def test_no_captured_variation_exists_so_the_empty_catalog_claim_holds() -> None:
    """The doc says the catalog ships ZERO variations. If a real one ever
    lands, this fails and the claim must be rewritten rather than left to rot."""
    assert "zero variations" in _doc().lower()
    entries = [
        path
        for path in CATALOG_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    ]
    assert entries == [], f"a variation landed; docs/deployment-lock.md must be updated: {entries}"


def test_no_capture_verb_exists_so_the_library_only_claim_holds() -> None:
    """The doc says lock capture has no CLI caller. This fails the day one
    lands — which is exactly when the sentence stops being true."""
    assert "no capture verb" in _doc().lower()
    writers = {"capture_lock", "write_lock", "build_lock"}
    callers = sorted(
        path.name for path in CLI_COMMANDS.glob("*.py") if _imported_names(path) & writers
    )
    assert callers == [], f"a capture caller landed in {callers}; update docs/deployment-lock.md"


def test_the_buildability_preflight_is_still_offline_and_warn_only() -> None:
    """The doc says the restore's preflight never queries an index and never
    raises. Both halves are checked against the wiring, not trusted."""
    text = _doc()
    assert "offline and warn-only" in text
    imported = _imported_names(CLI_COMMANDS / "init.py")
    assert "check_lock_buildability" in imported
    # No index query and no raising path is wired into the restore — the
    # module docstring MENTIONS both, which is why this inspects imports
    # rather than grepping the source text.
    assert "default_pypi_index_query" not in imported
    assert "assert_buildable" not in imported
