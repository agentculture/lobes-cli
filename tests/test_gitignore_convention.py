"""Verify the repo's positional `.gitignore` convention for env files.

The convention (see `.gitignore`) is positional, not a per-file list:

- any name ENDING in `.env` (a `*.env` suffix) is a secret dotfile and is
  ignored by construction — `.env`, `.cf-tunnel.env`, `.secrets.env`, and any
  future secret dotfile need no separate entry.
- any name STARTING with `.env.` (a `.env.` prefix) is a template/sample and
  stays tracked (`.env.example`, `.env.sample`, `.env.lock`).
- `tests/goldens/**/*.env` is a deliberate negation: those are committed
  fixture snapshots, not secrets, so a newly generated golden must stay
  stageable despite matching the `*.env` suffix rule.

These tests build a throwaway git repo seeded with the REAL, shipped
`.gitignore` (never a copy of the rule text) so they test what actually
ships, not a restatement of it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE_PATH = REPO_ROOT / ".gitignore"


def _run_git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path, gitignore_text: str) -> Path:
    """Create a scratch git repo seeded with the given .gitignore content."""
    repo = tmp_path / "scratch-repo"
    repo.mkdir()
    _run_git("init", "-q", cwd=repo)
    _run_git("config", "user.email", "test@example.com", cwd=repo)
    _run_git("config", "user.name", "Test", cwd=repo)
    (repo / ".gitignore").write_text(gitignore_text)
    (repo / "tests" / "goldens" / "shapes").mkdir(parents=True)
    _run_git("add", ".gitignore", cwd=repo)
    _run_git("commit", "-q", "-m", "seed gitignore", cwd=repo)
    return repo


def _is_ignored(repo: Path, relpath: str) -> bool:
    """True if `git check-ignore` reports relpath as ignored."""
    result = _run_git("check-ignore", "-q", relpath, cwd=repo)
    return result.returncode == 0


def _is_stageable(repo: Path, relpath: str) -> bool:
    """Create relpath (if absent) and confirm `git add` actually stages it."""
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("PLACEHOLDER=1\n")
    add_result = _run_git("add", relpath, cwd=repo)
    if add_result.returncode != 0:
        return False
    status = _run_git("status", "--porcelain", cwd=repo)
    # A staged new file shows as "A  <path>" (added, not ignored).
    return any(line[:2] == "A " and line[3:] == relpath for line in status.stdout.splitlines())


@pytest.fixture
def real_gitignore_text() -> str:
    assert GITIGNORE_PATH.exists(), f"missing {GITIGNORE_PATH}"
    return GITIGNORE_PATH.read_text()


@pytest.fixture
def repo(tmp_path: Path, real_gitignore_text: str) -> Path:
    return _init_repo(tmp_path, real_gitignore_text)


class TestPositionalEnvRule:
    """Criterion 1: *.env suffix ignored, .env. prefix not."""

    @pytest.mark.parametrize(
        "relpath",
        [".env", ".cf-tunnel.env", ".secrets.env", "nested/dir/.env"],
    )
    def test_env_suffix_is_ignored(self, repo: Path, relpath: str) -> None:
        (repo / relpath).parent.mkdir(parents=True, exist_ok=True)
        (repo / relpath).write_text("SECRET=1\n")
        assert _is_ignored(
            repo, relpath
        ), f"expected {relpath!r} to be ignored by the *.env suffix rule"

    @pytest.mark.parametrize("relpath", [".env.example", ".env.sample"])
    def test_env_prefix_is_not_ignored(self, repo: Path, relpath: str) -> None:
        (repo / relpath).write_text("EXAMPLE=1\n")
        assert not _is_ignored(
            repo, relpath
        ), f"expected {relpath!r} to remain tracked (.env. prefix)"


class TestGoldensNegation:
    """Criterion 2: newly created goldens *.env files are stageable."""

    def test_new_root_golden_is_stageable(self, repo: Path) -> None:
        assert _is_stageable(repo, "tests/goldens/new-profile.env")

    def test_new_shape_golden_is_stageable(self, repo: Path) -> None:
        assert _is_stageable(repo, "tests/goldens/shapes/new-shape.env")


class TestNegationIsLoadBearing:
    """Criterion 3: the test must fail if the negation line is stripped."""

    def test_negation_removal_breaks_goldens_staging(
        self, tmp_path: Path, real_gitignore_text: str
    ) -> None:
        stripped_lines = [
            line
            for line in real_gitignore_text.splitlines(keepends=True)
            if line.strip() != "!tests/goldens/**/*.env"
        ]
        assert len(stripped_lines) < len(
            real_gitignore_text.splitlines(keepends=True)
        ), "expected to find and strip the goldens negation line"
        stripped_text = "".join(stripped_lines)

        broken_repo = _init_repo(tmp_path, stripped_text)

        # Without the negation, a new goldens .env file should be IGNORED,
        # not stageable — proving the negation line is what makes criterion
        # 2 pass on the real .gitignore.
        assert _is_ignored(broken_repo, "tests/goldens/new-profile.env")
        assert _is_ignored(broken_repo, "tests/goldens/shapes/new-shape.env")
        assert not _is_stageable(broken_repo, "tests/goldens/new-profile.env")
        assert not _is_stageable(broken_repo, "tests/goldens/shapes/new-shape.env")
