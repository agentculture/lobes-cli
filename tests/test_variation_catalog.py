"""Tests for the variation catalog — ``deployments/<id>/`` layout + the
info-file contract (t9, ``docs/plans/2026-08-29-deployment-lock-per-box.md``).

**Nothing here is a real capture.** Capturing a genuine variation requires
running ``lobes`` on real hardware (a DGX Spark, a Jetson AGX Thor, a Jetson
AGX Orin) and no such box has been captured into ``deployments/`` yet. Every
variation these tests exercise lives under ``tests/fixtures/deployments/`` and
is labelled a fixture in its own info file. The published catalog is checked
too — but by a rule that is *vacuously* satisfied while it holds no
variations, which is exactly today's honest state.

The heart of the task is the honesty requirement. CLAUDE.md's #108 rule makes
most shipped shapes DECLARED, not measured (all four Orin shapes and
``thor-muse`` have no acceptance transcript), so a catalog of variations will
show "no measured result" for most of its entries on day one. These tests
assert that showing it is *explicit*: an info file must either cite an
existing ``docs/evidence/`` transcript or say, in the exact words the contract
fixes, that there is no measured result. A blank a reader could mistake for a
measurement is a failure, and the negative tests below prove the validator
actually fails on one.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lobes import variation_catalog as vc
from lobes.runtime._lock import LOCK_FILENAME, file_digest, load_lock

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
FIXTURE_CATALOG = FIXTURE_ROOT / "deployments"
PUBLISHED_CATALOG = REPO_ROOT / "deployments"

# The two committed fixture variations: one that CITES a measured result, one
# that honestly declares it has none. Both directory-name forms are covered —
# a bare variation id, and the `<id>__<shape>` form.
FIXTURE_MEASURED = "fixture-card"
FIXTURE_UNMEASURED = "fixture-card__fixture-shape"

_SCANNER_PATH = REPO_ROOT / "scripts" / "scan_deployment_secrets.py"


def _load_scanner():
    spec = importlib.util.spec_from_file_location("scan_deployment_secrets_t9", _SCANNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _fixture_dirs() -> list[Path]:
    return sorted(p for p in FIXTURE_CATALOG.iterdir() if p.is_dir())


# --- the layout itself -------------------------------------------------------


def test_fixture_catalog_holds_both_evidence_states() -> None:
    names = {p.name for p in _fixture_dirs()}
    assert names == {FIXTURE_MEASURED, FIXTURE_UNMEASURED}


@pytest.mark.parametrize("name", [FIXTURE_MEASURED, FIXTURE_UNMEASURED])
def test_fixture_variation_validates_clean(name: str) -> None:
    problems = vc.validate_variation(FIXTURE_CATALOG / name, repo_root=REPO_ROOT)
    assert problems == []


def test_published_catalog_validates() -> None:
    """Every variation actually published under ``deployments/`` is well-formed.

    Vacuously true today: no real box has been captured, so the catalog holds
    only its README and the info-file template. The rule is written so that
    the first genuine capture is checked the moment it lands.
    """
    assert PUBLISHED_CATALOG.is_dir()
    problems = vc.catalog_problems(PUBLISHED_CATALOG, repo_root=REPO_ROOT)
    assert problems == {}


def test_published_catalog_readme_states_no_box_captured_yet() -> None:
    readme = (PUBLISHED_CATALOG / "README.md").read_text(encoding="utf-8")
    assert readme.count(vc.NO_MEASURED_RESULT) >= 1
    assert readme.lower().count("no real box has been captured") >= 1


def test_info_template_is_not_itself_a_variation() -> None:
    """The template must not be mistaken for a published variation."""
    assert (PUBLISHED_CATALOG / vc.INFO_TEMPLATE_FILENAME).is_file()
    assert vc.variation_dirs(PUBLISHED_CATALOG) == []


# --- acceptance criterion 1: complete enough for a restore -------------------


@pytest.mark.parametrize("name", [FIXTURE_MEASURED, FIXTURE_UNMEASURED])
def test_variation_lock_loads_and_names_every_file_it_ships(name: str) -> None:
    """A variation carries everything a ``--from-lock`` restore would need.

    Scope note: ``lobes init --from-lock`` is t7's verb and is not imported
    here. What is tested is the property this task owns — that a variation
    directory is *loadable and complete*: its lock parses via
    :func:`load_lock`, its ``[files]`` table is non-empty, every file it names
    is present, and every digest matches the bytes on disk.
    """
    directory = FIXTURE_CATALOG / name
    lock = load_lock(directory / LOCK_FILENAME)

    assert lock.files, "a variation with no [files] table cannot be restored"
    for filename, digest in lock.files.items():
        target = directory / filename
        assert target.is_file(), f"{name}: lock names {filename}, which is absent"
        assert file_digest(target) == digest, f"{name}: {filename} does not match its digest"


@pytest.mark.parametrize("name", [FIXTURE_MEASURED, FIXTURE_UNMEASURED])
def test_variation_records_every_deployment_file_it_ships(name: str) -> None:
    """No unrecorded compose/Dockerfile hides in a variation directory."""
    directory = FIXTURE_CATALOG / name
    lock = load_lock(directory / LOCK_FILENAME)
    on_disk = {p.name for p in directory.iterdir() if p.is_file()}
    bookkeeping = {LOCK_FILENAME, vc.INFO_FILENAME}
    assert on_disk - bookkeeping == set(lock.files)


@pytest.mark.parametrize("name", [FIXTURE_MEASURED, FIXTURE_UNMEASURED])
def test_variation_materialises_on_a_machine_that_never_served_it(name: str, tmp_path) -> None:
    """The bytes a restore would write land byte-identically elsewhere.

    This models the third-party adoption case — a machine that has never
    served the variation — at the level this task owns: copying the
    lock-named files into an empty deployment dir reproduces them exactly,
    with no re-render and no host state consulted. t7 owns the verb that does
    this from the CLI; the property it depends on is asserted here.
    """
    source = FIXTURE_CATALOG / name
    lock = load_lock(source / LOCK_FILENAME)

    target = tmp_path / "never-served-this"
    target.mkdir()
    for filename in lock.files:
        shutil.copyfile(source / filename, target / filename)

    for filename, digest in lock.files.items():
        assert (target / filename).read_bytes() == (source / filename).read_bytes()
        assert file_digest(target / filename) == digest
    # A restore must not need — or invent — a .env: secrets stay out of the
    # catalog entirely.
    assert not (target / ".env").exists()


# --- acceptance criterion 2: cited evidence must exist -----------------------


def test_every_cited_evidence_path_exists() -> None:
    """Across both catalogs: every ``docs/evidence/`` citation resolves."""
    citations: list[tuple[str, str]] = []
    for catalog in (FIXTURE_CATALOG, PUBLISHED_CATALOG):
        for directory in vc.variation_dirs(catalog):
            info = vc.parse_info((directory / vc.INFO_FILENAME).read_text(encoding="utf-8"))
            citations.extend((directory.name, path) for path in info.evidence_paths)

    assert citations, "the fixture catalog must exercise at least one citation"
    for name, cited in citations:
        assert (REPO_ROOT / cited).is_file(), f"{name} cites missing evidence {cited}"


def test_measured_fixture_cites_a_real_transcript() -> None:
    directory = FIXTURE_CATALOG / FIXTURE_MEASURED
    lock = load_lock(directory / LOCK_FILENAME)
    info = vc.parse_info((directory / vc.INFO_FILENAME).read_text(encoding="utf-8"))

    assert lock.evidence is not None
    assert (REPO_ROOT / lock.evidence).is_file()
    assert lock.evidence in info.measured_evidence_paths
    assert not info.declares_no_measured_result


def test_a_citation_that_does_not_resolve_is_a_finding(tmp_path) -> None:
    directory = _copy_fixture(FIXTURE_MEASURED, tmp_path)
    info_path = directory / vc.INFO_FILENAME
    info_path.write_text(
        info_path.read_text(encoding="utf-8").replace(
            _measured_citation(), "docs/evidence/2999-01-01-not-a-real-transcript.txt"
        ),
        encoding="utf-8",
    )

    problems = vc.validate_variation(directory, repo_root=REPO_ROOT)
    assert any("2999-01-01-not-a-real-transcript.txt" in p for p in problems)


# --- acceptance criterion 3: never a blank readable as a measurement ---------


def test_unmeasured_fixture_says_so_explicitly() -> None:
    directory = FIXTURE_CATALOG / FIXTURE_UNMEASURED
    lock = load_lock(directory / LOCK_FILENAME)
    info = vc.parse_info((directory / vc.INFO_FILENAME).read_text(encoding="utf-8"))

    assert lock.evidence is None
    assert info.declares_no_measured_result
    assert info.measured_evidence_paths == ()
    # The words are fixed by the contract, not left to each author's phrasing.
    assert vc.NO_MEASURED_RESULT in info.measured_result_body


def test_a_blank_measured_result_is_rejected(tmp_path) -> None:
    """The failure mode this criterion exists to prevent."""
    directory = _copy_fixture(FIXTURE_UNMEASURED, tmp_path)
    info_path = directory / vc.INFO_FILENAME
    info_path.write_text(
        info_path.read_text(encoding="utf-8").replace(vc.NO_MEASURED_RESULT, ""),
        encoding="utf-8",
    )

    problems = vc.validate_variation(directory, repo_root=REPO_ROOT)
    assert any("no measured result" in p.lower() for p in problems)


def test_a_missing_measured_result_section_is_rejected(tmp_path) -> None:
    directory = _copy_fixture(FIXTURE_UNMEASURED, tmp_path)
    info_path = directory / vc.INFO_FILENAME
    kept = info_path.read_text(encoding="utf-8").split(f"## {vc.MEASURED_RESULT_HEADING}")[0]
    info_path.write_text(kept, encoding="utf-8")

    problems = vc.validate_variation(directory, repo_root=REPO_ROOT)
    assert any(vc.MEASURED_RESULT_HEADING in p for p in problems)


def test_claiming_no_measured_result_while_the_lock_cites_one_is_rejected(tmp_path) -> None:
    """The two halves of a variation cannot disagree about evidence."""
    directory = _copy_fixture(FIXTURE_MEASURED, tmp_path)
    info_path = directory / vc.INFO_FILENAME
    info_path.write_text(
        info_path.read_text(encoding="utf-8").replace(_measured_citation(), vc.NO_MEASURED_RESULT),
        encoding="utf-8",
    )

    problems = vc.validate_variation(directory, repo_root=REPO_ROOT)
    assert any("evidence" in p.lower() for p in problems)


def test_missing_info_file_is_rejected(tmp_path) -> None:
    directory = _copy_fixture(FIXTURE_UNMEASURED, tmp_path)
    (directory / vc.INFO_FILENAME).unlink()

    problems = vc.validate_variation(directory, repo_root=REPO_ROOT)
    assert any(vc.INFO_FILENAME in p for p in problems)


def test_missing_lock_is_rejected(tmp_path) -> None:
    directory = _copy_fixture(FIXTURE_UNMEASURED, tmp_path)
    (directory / LOCK_FILENAME).unlink()

    problems = vc.validate_variation(directory, repo_root=REPO_ROOT)
    assert any(LOCK_FILENAME in p for p in problems)


def test_a_tampered_file_breaks_its_digest(tmp_path) -> None:
    directory = _copy_fixture(FIXTURE_UNMEASURED, tmp_path)
    override = directory / "docker-compose.override.yml"
    override.write_text(override.read_text(encoding="utf-8") + "\n# hand edit\n", encoding="utf-8")

    problems = vc.validate_variation(directory, repo_root=REPO_ROOT)
    assert any("docker-compose.override.yml" in p for p in problems)


def test_directory_name_must_agree_with_the_lock(tmp_path) -> None:
    directory = _copy_fixture(FIXTURE_UNMEASURED, tmp_path, dest_name="some-other-name")

    problems = vc.validate_variation(directory, repo_root=REPO_ROOT)
    assert any("some-other-name" in p for p in problems)


# --- parser unit tests -------------------------------------------------------


def test_parse_info_reads_title_sections_and_citations() -> None:
    info = vc.parse_info(
        "\n".join(
            [
                "# fixture-card — a fixture",
                "",
                "## What this variation is",
                "",
                "Prose.",
                "",
                "## Measured result",
                "",
                "Measured live: `docs/evidence/2026-08-20-accept-hand-spark.txt`.",
                "",
                "## Notes",
                "",
                "See also docs/evidence/2026-08-20-accept-cortex-local-thor.txt.",
                "",
            ]
        )
    )

    assert info.title == "fixture-card — a fixture"
    assert vc.MEASURED_RESULT_HEADING in info.sections
    assert info.measured_evidence_paths == ("docs/evidence/2026-08-20-accept-hand-spark.txt",)
    assert len(info.evidence_paths) == 2
    assert not info.declares_no_measured_result


def test_parse_info_strips_trailing_atx_hashes_and_whitespace_from_headings() -> None:
    """python:S8786: ``_HEADING_RE`` was rewritten to avoid catastrophic
    backtracking (``^(#{1,6})\\s+(.*?)\\s*#*\\s*$`` -> a greedy full-line
    capture plus a plain-Python strip in ``_heading_text``). Same behaviour:
    a closing ATX ``##`` and trailing whitespace are both stripped, and a
    ``#`` that is genuinely part of the heading text is not."""
    info = vc.parse_info(
        "\n".join(
            [
                "# Title with trailing hashes ##",
                "",
                "## What this variation is   ",
                "",
                "Prose.",
                "",
                "## Measured result",
                "",
                vc.NO_MEASURED_RESULT,
                "",
                "## mentions #1 not a closing hash",
                "",
            ]
        )
    )
    assert info.title == "Title with trailing hashes"
    assert info.sections.count(vc.DESCRIPTION_HEADING) == 1
    assert info.sections.count("mentions #1 not a closing hash") == 1


def test_parse_info_ignores_non_heading_lines() -> None:
    """A line that is not ``#{1,6}`` followed by a space is not a heading —
    a bare ``#`` with no following text, or a ``#`` mid-sentence."""
    info = vc.parse_info(
        "\n".join(
            [
                "not a heading # at all",
                "#no-space-after-hash",
                "# ",
                "## real section",
                "",
                "body",
            ]
        )
    )
    assert info.title == ""
    assert info.sections == ("real section",)


def test_parse_info_detects_the_no_measured_result_marker() -> None:
    info = vc.parse_info(
        "\n".join(
            [
                "# t",
                "",
                "## Measured result",
                "",
                vc.NO_MEASURED_RESULT,
                "",
            ]
        )
    )
    assert info.declares_no_measured_result
    assert info.measured_evidence_paths == ()


# --- the committed tree must actually be committable -------------------------


def test_catalog_files_are_stageable() -> None:
    """A silently-ignored catalog file would be a nasty failure.

    ``.gitignore``'s positional rule ignores a ``.env`` SUFFIX, so nothing
    here may be ``.env``-suffixed. Proven with git itself rather than by
    reading the rule.
    """
    tracked_candidates = [
        PUBLISHED_CATALOG / "README.md",
        PUBLISHED_CATALOG / vc.INFO_TEMPLATE_FILENAME,
    ]
    for directory in _fixture_dirs():
        tracked_candidates.extend(sorted(p for p in directory.iterdir() if p.is_file()))

    for path in tracked_candidates:
        assert not path.name.endswith(".env")
        result = subprocess.run(  # nosec B603 B607 - fixed argv, repo-local
            ["git", "check-ignore", "-q", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        assert result.returncode == 1, f"{path} is gitignored and would never be committed"


# --- the CI secret gate covers the fixture tree ------------------------------


def test_secret_scanner_is_clean_over_the_catalogs() -> None:
    scanner = _load_scanner()
    for root in (FIXTURE_ROOT, REPO_ROOT):
        assert scanner.scan_paths(root) == []


def test_secret_scanner_would_catch_a_planted_token_in_a_variation(tmp_path) -> None:
    """The gate can fail — proven against this task's own layout."""
    scanner = _load_scanner()
    catalog = tmp_path / "deployments"
    catalog.mkdir()
    directory = _copy_fixture(FIXTURE_UNMEASURED, catalog)
    override = directory / "docker-compose.override.yml"
    override.write_text(
        override.read_text(encoding="utf-8")
        + "\n      - GATEWAY_API_KEY=not-a-real-key-planted-by-a-test\n",
        encoding="utf-8",
    )

    assert scanner.scan_paths(tmp_path)


# --- helpers -----------------------------------------------------------------


def _copy_fixture(name: str, parent: Path, *, dest_name: str | None = None) -> Path:
    target = parent / (dest_name or name)
    shutil.copytree(FIXTURE_CATALOG / name, target)
    return target


def _measured_citation() -> str:
    lock = load_lock(FIXTURE_CATALOG / FIXTURE_MEASURED / LOCK_FILENAME)
    assert lock.evidence is not None
    return lock.evidence


def test_the_heading_separator_is_unquantified_so_it_cannot_be_ambiguous() -> None:
    """`[ \\t]+` would let the separator and `(.*)` both claim a space (S8786).

    Measured, Python's engine does not degrade on the quantified form — but
    the ambiguity is real in the pattern, so the shipped regex removes it
    rather than depending on `(.*)$` never being backtracked into.
    """
    assert vc._HEADING_RE.pattern == r"^(#{1,6})[ \t](.*)$"


@pytest.mark.parametrize(
    "line, expected",
    [
        ("# One space", ("#", "One space")),
        ("##  Two spaces", ("##", "Two spaces")),
        ("###\t\t Tabs then space", ("###", "Tabs then space")),
        ("#### Closed ####", ("####", "Closed")),
        ("##   ", ("##", "")),
    ],
)
def test_extra_separators_are_stripped_not_captured(line, expected) -> None:
    """Dropping the `+` moves surplus separators into group 2; `_heading_text`
    strips them, so every previously-accepted spelling parses identically."""
    match = vc._HEADING_RE.search(line)
    assert match is not None
    assert (match.group(1), vc._heading_text(match)) == expected


@pytest.mark.parametrize("line", ["#NoSpace", "####### Seven hashes", "plain text"])
def test_non_headings_stay_unmatched(line) -> None:
    assert vc._HEADING_RE.search(line) is None
