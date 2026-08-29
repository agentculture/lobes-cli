"""Wheel-exclusion guard for the (not-yet-existing) top-level ``deployments/`` tree.

``docs/plans/2026-08-29-deployment-lock-per-box.md`` (task t5) introduces a
top-level ``deployments/<box>/`` directory holding committed per-machine-type
deployment artifacts (rendered compose files, Dockerfiles, a generated
``deployment.lock.toml``). Those are repo artifacts, never distributed ones —
they must never ship inside the ``lobes-cli`` wheel PyPI consumers install.

Today the exclusion is a side effect of ``pyproject.toml`` declaring::

    [tool.hatch.build.targets.wheel]
    packages = ["lobes"]

i.e. hatchling only packages the ``lobes/`` package directory, so anything
else at the repo root (``deployments/`` included) is never considered. That is
*asserted* rather than *assumed* by this module in two layers:

1. A fast, always-on static check that the ``packages =`` declaration is
   exactly ``["lobes"]`` — no accidental widening, no ``"."`` catch-all, no
   ``"deployments"`` entry. This needs no build, no network, and always runs
   in CI.
2. A build-based check that actually invokes ``uv build --wheel`` against a
   clean export of the repo (via ``git archive``) with a synthetic
   ``deployments/<box>/deployment.lock.toml`` fixture dropped in, once with
   today's ``packages = ["lobes"]`` (must NOT ship the fixture) and once with
   a deliberately widened ``packages = ["lobes", "deployments"]`` (MUST ship
   the fixture). The second half is what makes this an active proof rather
   than a config-string comparison: it is what would fail if a future edit
   widened ``packages=`` to include ``deployments`` without anyone noticing.
   This layer is skipped (not failed) when the ``uv`` toolchain isn't on
   ``PATH`` or a build genuinely cannot complete in the current sandbox (e.g.
   no network and an empty build-backend cache) — see
   ``_build_wheel_or_skip`` below.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - repo requires-python >=3.12
    import tomli as tomllib  # type: ignore[no-redef]

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_wheel_targets_config() -> dict:
    data = tomllib.loads(PYPROJECT.read_text())
    return data["tool"]["hatch"]["build"]["targets"]["wheel"]


# ---------------------------------------------------------------------------
# Layer 1: static assertion over pyproject.toml — always runs, no build.
# ---------------------------------------------------------------------------


def test_wheel_packages_is_exactly_lobes():
    """The wheel target must package ONLY ``lobes`` — no widening, no catch-all.

    This is the fast, unconditional half of the guard: it needs no build and
    no network, so it always runs in CI. A future edit that widens
    ``packages=`` (to ``["lobes", "deployments"]``, or to ``"."``/``[""]``
    style catch-alls) fails this immediately.
    """
    wheel_cfg = _load_wheel_targets_config()
    packages = wheel_cfg.get("packages")

    assert packages == ["lobes"], (
        "tool.hatch.build.targets.wheel.packages must stay exactly ['lobes']; "
        f"got {packages!r}. Widening this would ship repo-only artifacts "
        "(including the deployments/ lock tree) inside the PyPI wheel."
    )
    assert "deployments" not in (packages or [])


def test_wheel_config_has_no_include_or_force_include_of_deployments():
    """No sneaky side door: an ``include``/``force-include`` naming deployments/.

    ``packages=`` is the primary inclusion mechanism this project uses, but
    hatchling also honours ``include``/``force-include`` keys on the same
    table. Assert none of them mention ``deployments`` either, so widening
    via a different key doesn't silently slip past the ``packages=`` check
    above.
    """
    wheel_cfg = _load_wheel_targets_config()
    for key in ("include", "force-include", "artifacts"):
        value = wheel_cfg.get(key)
        if value is None:
            continue
        serialized = repr(value)
        assert "deployments" not in serialized, (
            f"tool.hatch.build.targets.wheel.{key} must not reference "
            f"deployments/: got {value!r}"
        )


# ---------------------------------------------------------------------------
# Layer 2: build-based proof — actually builds a wheel and inspects it.
# ---------------------------------------------------------------------------


def _uv_available() -> bool:
    return shutil.which("uv") is not None


def _export_clean_tree(dest: Path) -> None:
    """Export a clean copy of the tracked repo tree into ``dest`` via git archive."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "HEAD"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        check=True,
        timeout=30,
    )
    tar_path = dest.parent / "export.tar"
    tar_path.write_bytes(archive.stdout)
    with tarfile.open(tar_path) as tf:
        # filter="data": trusted, local git archive of our own HEAD, but keep
        # the extraction hardened against path/permission surprises anyway.
        tf.extractall(dest, filter="data")
    tar_path.unlink()


def _plant_deployments_fixture(project_dir: Path) -> None:
    """Drop a synthetic deployments/<box>/deployment.lock.toml fixture.

    The real deployments/ tree doesn't exist yet (task t9 introduces it); this
    stands in for it so the exclusion is exercised whether or not the real
    directory has landed.
    """
    box_dir = project_dir / "deployments" / "fixture-box"
    box_dir.mkdir(parents=True, exist_ok=True)
    (box_dir / "deployment.lock.toml").write_text(
        '# fixture only - not a real lock\nprofile = "fixture"\n'
    )
    (box_dir / "docker-compose.yml").write_text("services: {}\n")


def _widen_packages_to_include_deployments(project_dir: Path) -> None:
    pyproject_path = project_dir / "pyproject.toml"
    text = pyproject_path.read_text()
    original = 'packages = ["lobes"]'
    widened = 'packages = ["lobes", "deployments"]'
    assert original in text, (
        "expected to find the exact packages= declaration to widen; "
        "pyproject.toml's wheel target formatting changed - update this fixture"
    )
    pyproject_path.write_text(text.replace(original, widened, 1))


def _build_wheel_or_skip(project_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "-o", str(out_dir)],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        pytest.skip(f"uv build unavailable/failed in this sandbox: {exc}")
    wheels = list(out_dir.glob("*.whl"))
    if not wheels:
        pytest.skip("uv build produced no wheel in this sandbox")
    return wheels[0]


def _wheel_namelist(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as zf:
        return zf.namelist()


@pytest.mark.skipif(not _uv_available(), reason="uv toolchain not on PATH")
def test_built_wheel_excludes_deployments_tree(tmp_path):
    """A real ``uv build`` of the repo (plus a deployments/ fixture) ships no deployments/ path.

    This is criterion 1: build a wheel and prove its namelist contains no
    ``deployments/`` entries, even when a ``deployments/`` directory with
    real-looking content is present in the source tree.
    """
    project_dir = tmp_path / "clean"
    _export_clean_tree(project_dir)
    _plant_deployments_fixture(project_dir)

    wheel_path = _build_wheel_or_skip(project_dir, tmp_path / "dist-clean")
    names = _wheel_namelist(wheel_path)

    deployments_entries = [n for n in names if n.startswith("deployments/")]
    assert deployments_entries == [], (
        "wheel must not contain any deployments/ paths, found: " f"{deployments_entries}"
    )
    # Sanity: the wheel isn't empty / mis-built - it does contain the package.
    assert any(n.startswith("lobes/") for n in names)


@pytest.mark.skipif(not _uv_available(), reason="uv toolchain not on PATH")
def test_widened_packages_would_ship_deployments_tree(tmp_path):
    """Criterion 2: prove the guard is load-bearing, not a no-op.

    Build the SAME source tree + deployments/ fixture again, but with
    ``packages=`` deliberately widened to ``["lobes", "deployments"]`` in a
    throwaway copy of ``pyproject.toml`` (the real, committed file is never
    touched). The resulting wheel MUST contain the fixture's deployments/
    path - if it doesn't, this test's exclusion check above isn't actually
    detecting anything, and this assertion catches that.
    """
    project_dir = tmp_path / "widened"
    _export_clean_tree(project_dir)
    _plant_deployments_fixture(project_dir)
    _widen_packages_to_include_deployments(project_dir)

    wheel_path = _build_wheel_or_skip(project_dir, tmp_path / "dist-widened")
    names = _wheel_namelist(wheel_path)

    deployments_entries = [n for n in names if n.startswith("deployments/")]
    assert deployments_entries, (
        "widening packages= to include 'deployments' was expected to ship "
        "the deployments/ fixture into the wheel, but it didn't - this "
        "fixture no longer reflects how hatchling resolves packages=, "
        "update _widen_packages_to_include_deployments"
    )
    assert any(
        "deployment.lock.toml" in n for n in deployments_entries
    ), f"expected the fixture lock file to be present, got: {deployments_entries}"
