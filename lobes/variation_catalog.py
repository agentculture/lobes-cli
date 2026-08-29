"""The variation catalog — ``deployments/<id>/`` layout and its info-file contract.

The repo publishes **all** docker variations, not only the boxes this
operator happens to run: a third party picks one from ``deployments/`` and
adopts it with ``lobes init --from-lock`` (t7). That widens the
deployment-lock practice from a per-box backup into a chooser's catalog, and a
chooser needs two things from every entry — the files to run it, and an honest
statement of what running it actually produced.

**The layout.** One directory per variation, directly under the catalog root:

.. code-block:: text

   deployments/
     README.md                    # the catalog's own front matter
     VARIATION.template.md        # the info-file template (not a variation)
     <variation-id>/              # e.g. `spark`, or `spark__spark-lobe`
       deployment.lock.toml       # the committed lock (lobes.runtime._lock)
       VARIATION.md               # the info file — this module's contract
       docker-compose.yml         # …and every other file the lock names,
       docker-compose.override.yml#    committed verbatim
       Dockerfile.gateway

The directory NAME is the variation id (:mod:`lobes.variation` — machine type
or setup, never a hostname), optionally suffixed ``__<shape>`` when a
deployment shape is applied. Both halves are cross-checked against the lock's
own ``variation`` / ``shape`` fields, so a directory cannot quietly describe a
different box than its lock does. (The ``__`` separator mirrors
``tests/goldens/shapes/<shape>__<card>.env``; the order is reversed here so a
listing sorts by variation id.)

**The honesty requirement.** CLAUDE.md's #108 rule means most shipped shapes
are DECLARED, not measured — all four Orin shapes and ``thor-muse`` have no
acceptance transcript — so "choose a variation and see the result" will show
no measured result for most of the catalog. Publishing it must not manufacture
the appearance of evidence. Hence the contract enforced by
:func:`validate_variation`:

* an info file's ``## Measured result`` section either CITES one or more
  existing ``docs/evidence/`` transcripts, or contains the exact sentence
  :data:`NO_MEASURED_RESULT`. Never both, and never neither — a blank a reader
  could mistake for a measurement is a finding, not a default;
* every cited path must resolve against the repo root, so a citation cannot
  rot into a claim about a file nobody can read;
* the info file and the lock must agree: ``lock.evidence`` is ``None`` exactly
  when the info file declares no measured result, and is cited there
  otherwise.

An info file CITES rather than restates its numbers, deliberately: the
transcripts under ``docs/evidence/`` are the measurement, and a second copy of
the figures would drift from them.

**Completeness.** A variation must carry what a restore needs: its lock's
``[files]`` table is non-empty, every file it names is present, every digest
matches the bytes on disk, and no unrecorded compose file or Dockerfile is
lying around beside them. Secrets are never in the catalog at all — the lock
is an allowlist of rendered knobs by construction
(:mod:`lobes.runtime._lock`), and a restore leaves ``.env`` alone.

Stdlib only, and no filesystem writes: this module reads and reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lobes.runtime._lock import LOCK_FILENAME, DeploymentLock, file_digest, load_lock

#: The catalog root's directory name, relative to the repo root.
CATALOG_DIRNAME = "deployments"

#: Every variation directory carries one of these beside its lock.
INFO_FILENAME = "VARIATION.md"

#: The blank template authors copy. Lives at the catalog root and is NOT a
#: variation — :func:`variation_dirs` only ever returns directories.
INFO_TEMPLATE_FILENAME = "VARIATION.template.md"

#: The heading whose section carries the evidence claim.
MEASURED_RESULT_HEADING = "Measured result"

#: The heading whose section says what the variation *is*.
DESCRIPTION_HEADING = "What this variation is"

#: The exact sentence a variation with no acceptance transcript must carry.
#: Fixed by the contract rather than left to each author's phrasing, so both a
#: reader and a test can recognise it without interpretation.
NO_MEASURED_RESULT = "No measured result."

#: Where measured results live. Citations are repo-root-relative paths.
EVIDENCE_DIRNAME = "docs/evidence"

#: Separates a variation id from an applied shape in a directory name.
SHAPE_SEPARATOR = "__"

# Deliberately NOT `^(#{1,6})\s+(.*?)\s*#*\s*$` (python:S8786): `.*?` lazily
# overlapping the trailing `\s*#*\s*` gives the engine exponentially many
# ways to partition a long run of whitespace/`#`, so a heading-shaped line
# with no terminating `$` on this same logical run (e.g. deep inside a
# larger malformed match attempt) is catastrophically slow to reject.
# Instead this captures the whole rest of the line with a single greedy
# group — linear, no ambiguity — and :func:`_heading_text` strips the
# optional trailing ATX ``#`` marker(s) and any extra leading space
# afterward, in plain Python.
#
# The separator is `[ \t]` and NOT `[ \t]+`: quantifying it leaves `[ \t]+`
# and `(.*)` both able to match a space or tab, so a long run of them has
# many valid partitions. Measured, Python's engine does not actually
# degrade there (400k spaces rejects in ~0.5 ms, scaling linearly) because
# `(.*)$` cannot fail in MULTILINE and so is never backtracked into — but
# an unquantified separator removes the ambiguity outright rather than
# relying on that property of one engine. Behaviour is unchanged: the
# extra separators fall into group 2 and `_heading_text` strips them.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t](.*)$", re.MULTILINE)
_EVIDENCE_RE = re.compile(rf"{re.escape(EVIDENCE_DIRNAME)}/[A-Za-z0-9][A-Za-z0-9._+-]*")


def _heading_text(match: re.Match) -> str:
    """The heading name from a :data:`_HEADING_RE` match, trailing ATX
    ``#`` marker(s) and surrounding whitespace stripped — same result the
    old ``\\s*#*\\s*$`` suffix produced, computed without backtracking."""
    return match.group(2).strip().rstrip("#").rstrip()


@dataclass(frozen=True)
class VariationInfo:
    """A parsed ``VARIATION.md``.

    :param title: The level-1 heading, or ``""`` when the file has none.
    :param sections: Every level-2 heading, in document order.
    :param measured_result_body: The raw text of the ``## Measured result``
        section (``""`` when the section is absent — indistinguishable from an
        empty one on purpose: both are "says nothing", and both are findings).
    :param evidence_paths: Every ``docs/evidence/`` citation anywhere in the
        file, de-duplicated, in first-appearance order.
    :param measured_evidence_paths: The citations inside the measured-result
        section — the ones that constitute the evidence CLAIM. A transcript
        mentioned in a "Notes" aside is not a claim of measurement.
    :param declares_no_measured_result: Whether the measured-result section
        carries :data:`NO_MEASURED_RESULT` verbatim.
    """

    title: str
    sections: tuple[str, ...]
    measured_result_body: str
    evidence_paths: tuple[str, ...]
    measured_evidence_paths: tuple[str, ...]
    declares_no_measured_result: bool

    @property
    def has_measured_result_section(self) -> bool:
        return MEASURED_RESULT_HEADING in self.sections


def _dedup(paths: list[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for path in paths:
        seen.setdefault(path, None)
    return tuple(seen)


def _section_body(text: str, heading: str) -> str:
    """The text under the level-2 *heading*, up to the next heading of any level."""
    for match in _HEADING_RE.finditer(text):
        if len(match.group(1)) != 2 or _heading_text(match) != heading:
            continue
        start = match.end()
        following = _HEADING_RE.search(text, start)
        return text[start : following.start() if following else len(text)]
    return ""


def parse_info(text: str) -> VariationInfo:
    """Parse a ``VARIATION.md``'s *text* into a :class:`VariationInfo`.

    A pure string function — it never touches the filesystem, so the contract
    can be exercised on candidate text that was never written to disk.
    """
    headings = [(len(m.group(1)), _heading_text(m)) for m in _HEADING_RE.finditer(text)]
    title = next((name for level, name in headings if level == 1), "")
    sections = tuple(name for level, name in headings if level == 2)
    measured = _section_body(text, MEASURED_RESULT_HEADING)
    return VariationInfo(
        title=title,
        sections=sections,
        measured_result_body=measured,
        evidence_paths=_dedup(_EVIDENCE_RE.findall(text)),
        measured_evidence_paths=_dedup(_EVIDENCE_RE.findall(measured)),
        declares_no_measured_result=NO_MEASURED_RESULT in measured,
    )


def split_directory_name(name: str) -> tuple[str, str | None]:
    """Split a directory name into ``(variation id, shape or None)``.

    ``"spark__spark-lobe"`` -> ``("spark", "spark-lobe")``;
    ``"spark"`` -> ``("spark", None)``.
    """
    variation_id, separator, shape = name.partition(SHAPE_SEPARATOR)
    return (variation_id, shape if separator else None)


def variation_dirs(catalog_root: Path) -> list[Path]:
    """The variation directories under *catalog_root*, sorted by name.

    A directory is a variation by POSITION, not by content — anything directly
    under the catalog root that is not hidden and does not start with ``_``.
    A malformed entry is therefore reported by :func:`validate_variation`
    rather than silently skipped, which is the whole point: a variation
    missing its lock must be a finding, not an absence.
    """
    if not catalog_root.is_dir():
        return []
    return sorted(
        path
        for path in catalog_root.iterdir()
        if path.is_dir() and not path.name.startswith((".", "_"))
    )


def _validate_structure(name: str, info: VariationInfo, problems: list[str]) -> None:
    """Title + required sections — the shape checks that don't need the lock."""
    if not info.title:
        problems.append(f"{name}/{INFO_FILENAME}: no level-1 title naming the variation")
    if DESCRIPTION_HEADING not in info.sections:
        problems.append(f"{name}/{INFO_FILENAME}: missing '## {DESCRIPTION_HEADING}' section")
    if not info.has_measured_result_section:
        problems.append(f"{name}/{INFO_FILENAME}: missing '## {MEASURED_RESULT_HEADING}' section")


def _validate_evidence_citations(
    name: str, info: VariationInfo, repo_root: Path, problems: list[str]
) -> None:
    """Every ``docs/evidence/`` citation anywhere in the file resolves on disk."""
    for cited in info.evidence_paths:
        if not (repo_root / cited).is_file():
            problems.append(f"{name}/{INFO_FILENAME}: cites missing evidence {cited}")


def _validate_measured_result_claim(name: str, info: VariationInfo, problems: list[str]) -> None:
    """The #108 honesty rule: exactly one of "cites a transcript" or the
    fixed :data:`NO_MEASURED_RESULT` sentence — never both, never neither."""
    if info.declares_no_measured_result and info.measured_evidence_paths:
        problems.append(
            f"{name}/{INFO_FILENAME}: '{MEASURED_RESULT_HEADING}' both declares "
            f"'{NO_MEASURED_RESULT}' and cites a transcript — it must do exactly one"
        )
    elif not info.declares_no_measured_result and not info.measured_evidence_paths:
        problems.append(
            f"{name}/{INFO_FILENAME}: '{MEASURED_RESULT_HEADING}' cites no "
            f"{EVIDENCE_DIRNAME}/ transcript and does not state '{NO_MEASURED_RESULT}' "
            "— a blank here reads as a measurement"
        )


def _validate_lock_agreement(
    name: str, info: VariationInfo, lock: DeploymentLock, problems: list[str]
) -> None:
    """The info file's claim and the lock's own ``evidence`` field must agree."""
    if lock.evidence is None and not info.declares_no_measured_result:
        problems.append(
            f"{name}: the lock records no evidence but {INFO_FILENAME} claims a measured result"
        )
        return
    if lock.evidence is None:
        return
    if info.declares_no_measured_result:
        problems.append(
            f"{name}: the lock cites evidence {lock.evidence} but {INFO_FILENAME} "
            f"states '{NO_MEASURED_RESULT}'"
        )
    elif lock.evidence not in info.measured_evidence_paths:
        problems.append(
            f"{name}/{INFO_FILENAME}: does not cite the lock's own evidence {lock.evidence}"
        )


def _validate_info(
    directory: Path, lock: DeploymentLock | None, repo_root: Path, problems: list[str]
) -> None:
    name = directory.name
    info_path = directory / INFO_FILENAME
    if not info_path.is_file():
        problems.append(f"{name}: missing {INFO_FILENAME}")
        return

    info = parse_info(info_path.read_text(encoding="utf-8"))
    _validate_structure(name, info, problems)
    if not info.has_measured_result_section:
        return

    _validate_evidence_citations(name, info, repo_root, problems)
    _validate_measured_result_claim(name, info, problems)

    if lock is not None:
        _validate_lock_agreement(name, info, lock, problems)


def _validate_files(directory: Path, lock: DeploymentLock, problems: list[str]) -> None:
    name = directory.name
    if not lock.files:
        problems.append(f"{name}: the lock's [files] table is empty — nothing to restore")

    for filename, digest in sorted(lock.files.items()):
        target = directory / filename
        if not target.is_file():
            problems.append(f"{name}: the lock names {filename}, which is not in the directory")
        elif file_digest(target) != digest:
            problems.append(f"{name}: {filename} does not match the digest the lock records")

    bookkeeping = {LOCK_FILENAME, INFO_FILENAME}
    unrecorded = (
        {p.name for p in directory.iterdir() if p.is_file()} - bookkeeping - set(lock.files)
    )
    for filename in sorted(unrecorded):
        problems.append(f"{name}: {filename} is present but not recorded in the lock's [files]")


def validate_variation(directory: Path, *, repo_root: Path) -> list[str]:
    """Check one variation directory against the contract; returns the problems.

    An empty list means the variation is well-formed: its lock loads, its
    directory name agrees with the lock, every file the lock names is present
    and unmodified, nothing unrecorded sits beside them, and its info file
    makes an honest, resolvable evidence claim.

    Reporting a LIST rather than raising is deliberate — a chooser wants every
    problem with a variation at once, and a CI gate wants to print them all.
    """
    problems: list[str] = []
    name = directory.name

    if not directory.is_dir():
        return [f"{name}: not a directory"]

    lock: DeploymentLock | None = None
    lock_file = directory / LOCK_FILENAME
    if not lock_file.is_file():
        problems.append(f"{name}: missing {LOCK_FILENAME}")
    else:
        try:
            lock = load_lock(lock_file)
        except Exception as exc:  # noqa: BLE001 - any parse failure is one finding
            problems.append(f"{name}/{LOCK_FILENAME}: unreadable ({exc})")

    if lock is not None:
        expected_id, expected_shape = split_directory_name(name)
        if lock.variation != expected_id:
            problems.append(
                f"{name}: directory name implies variation id {expected_id!r} "
                f"but the lock records {lock.variation!r}"
            )
        if (lock.shape or None) != expected_shape:
            problems.append(
                f"{name}: directory name implies shape {expected_shape!r} "
                f"but the lock records {lock.shape!r}"
            )
        _validate_files(directory, lock, problems)

    _validate_info(directory, lock, repo_root, problems)
    return problems


def catalog_problems(catalog_root: Path, *, repo_root: Path) -> dict[str, list[str]]:
    """Every variation's problems under *catalog_root*, keyed by directory name.

    An empty mapping means the whole catalog is well-formed — including the
    honest case where it holds no variations at all.
    """
    found = {
        directory.name: validate_variation(directory, repo_root=repo_root)
        for directory in variation_dirs(catalog_root)
    }
    return {name: problems for name, problems in found.items() if problems}
