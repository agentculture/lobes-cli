"""Upgrade-compatibility tests (t15, per-machine-profiles plan).

Invariant under test: ``pip install -U lobes-cli`` changes zero bytes in an
existing deployment dir, and every read/dry-run verb keeps operating a
deployment scaffolded by the PREVIOUS version. Adopting a new
template/profile is an explicit, diffed, ``--apply``'d re-init — never a side
effect of upgrading.

The "previous version" scaffold is built from ``tests/fixtures/upgrade_compat/``
— verbatim copies of what ``main``'s ``lobes init`` (single + fleet) produce;
see ``tests/fixtures/upgrade_compat/README.md`` for provenance (source commit
sha). ``lobes init`` copies a template's content byte-for-byte
(``env.example`` -> ``.env``, ``fleet/docker-compose.yml`` ->
``docker-compose.yml``), so a plain copy + rename of the vendored fixtures IS
what an old scaffold's files contain — no further materialisation needed.
"""

from __future__ import annotations

import hashlib
import re
from importlib.resources import files
from pathlib import Path

from lobes.cli import main
from lobes.runtime import _detect

FIXTURES = Path(__file__).parent / "fixtures" / "upgrade_compat"

_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)")
_ENV_KEY_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)=", re.MULTILINE)


def _fake_card(resolved: str) -> _detect.DetectedCard:
    """A known, injected card — avoids touching real hardware in these tests
    (same idiom as tests/test_init_profile.py)."""
    return _detect.DetectedCard(
        resolved=resolved,
        device_name="NVIDIA Thor",
        compute_capability="sm_110",
        total_memory_gb=125.9,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


def _materialize_old_single_scaffold(target: Path) -> None:
    """Build an old (main-scaffolded) single-model deployment dir."""
    target.mkdir(parents=True, exist_ok=True)
    (target / "docker-compose.yml").write_text(
        (FIXTURES / "single" / "docker-compose.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (target / ".env").write_text(
        (FIXTURES / "single" / "env.example").read_text(encoding="utf-8"), encoding="utf-8"
    )


def _materialize_old_fleet_scaffold(target: Path) -> None:
    """Build an old (main-scaffolded) fleet deployment dir.

    ``_compose.is_fleet`` keys off ``Dockerfile.gateway``'s presence. That file
    is byte-identical between ``main`` and the current tree (verified via
    ``git show main:... | diff`` during fixture vendoring), so copying the
    packaged (current) template for it is faithful to an old fleet scaffold —
    only ``docker-compose.yml``/``.env`` (the files under test) are vendored
    from ``main`` verbatim.
    """
    target.mkdir(parents=True, exist_ok=True)
    (target / "docker-compose.yml").write_text(
        (FIXTURES / "fleet" / "docker-compose.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (target / ".env").write_text(
        (FIXTURES / "fleet" / "env.example").read_text(encoding="utf-8"), encoding="utf-8"
    )
    gateway_dockerfile = (files("lobes.templates") / "fleet" / "Dockerfile.gateway").read_text(
        encoding="utf-8"
    )
    (target / "Dockerfile.gateway").write_text(gateway_dockerfile, encoding="utf-8")


def _hash_tree(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root``, keyed by path relative to ``root``."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


_SCAFFOLDS = {
    "single": (_materialize_old_single_scaffold, ["--single"]),
    "fleet": (_materialize_old_fleet_scaffold, []),
}


# --- 1 & 3: old-scaffold operability + zero-byte upgrade -------------------


def test_old_single_scaffold_operability_and_zero_byte(tmp_path) -> None:
    _old_scaffold_operability_and_zero_byte(tmp_path, "single")


def test_old_fleet_scaffold_operability_and_zero_byte(tmp_path, monkeypatch) -> None:
    # init's fleet dry-run resolves a per-machine profile even with no --apply
    # (the plan must be honest about what --apply would do); inject a known
    # card so this never depends on the real host's hardware.
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card("thor"))
    _old_scaffold_operability_and_zero_byte(tmp_path, "fleet")


def _old_scaffold_operability_and_zero_byte(tmp_path: Path, kind: str) -> None:
    materialize, init_topology_args = _SCAFFOLDS[kind]
    deploy = tmp_path / "deploy"
    materialize(deploy)
    before = _hash_tree(deploy)
    assert before, "fixture materialisation produced no files"

    # status: text + --json (read-only; must never write or raise)
    rc = main(["status", "--compose-dir", str(deploy)])
    assert rc == 0, f"status (text) failed on an old {kind} scaffold"

    rc = main(["status", "--compose-dir", str(deploy), "--json"])
    assert rc == 0, f"status --json failed on an old {kind} scaffold"

    # serve: dry-run planning (no --apply => no docker call, no write)
    rc = main(["serve", "--compose-dir", str(deploy)])
    assert rc == 0, f"serve dry-run failed on an old {kind} scaffold"

    # stop: dry-run planning
    rc = main(["stop", "--compose-dir", str(deploy)])
    assert rc == 0, f"stop dry-run failed on an old {kind} scaffold"

    # init-over-existing: dry-run only (no --apply) — must plan, not write.
    rc = main(["init", str(deploy), *init_topology_args])
    assert rc == 0, f"init dry-run over an existing {kind} scaffold failed"

    after = _hash_tree(deploy)
    assert after == before, (
        f"a read-only/dry-run verb modified the old {kind} scaffold on disk "
        "(zero-byte-upgrade invariant broken)"
    )


# --- 2: env-var name compatibility (the rename tripwire) -------------------


def _extract_vars(compose_text: str, env_text: str) -> set[str]:
    """Every var name the compose file *reads* (``${VAR...}``) union every var
    name the env file *declares* (``VAR=...`` at line start)."""
    referenced = set(_VAR_RE.findall(compose_text))
    declared = set(_ENV_KEY_RE.findall(env_text))
    return referenced | declared


def test_single_template_env_vars_still_honoured() -> None:
    main_vars = _extract_vars(
        (FIXTURES / "single" / "docker-compose.yml").read_text(encoding="utf-8"),
        (FIXTURES / "single" / "env.example").read_text(encoding="utf-8"),
    )
    current_vars = _extract_vars(
        (files("lobes.templates") / "docker-compose.yml").read_text(encoding="utf-8"),
        (files("lobes.templates") / "env.example").read_text(encoding="utf-8"),
    )
    missing = main_vars - current_vars
    assert not missing, f"single template dropped/renamed env var(s): {sorted(missing)}"


def test_fleet_template_env_vars_still_honoured() -> None:
    main_vars = _extract_vars(
        (FIXTURES / "fleet" / "docker-compose.yml").read_text(encoding="utf-8"),
        (FIXTURES / "fleet" / "env.example").read_text(encoding="utf-8"),
    )
    current_vars = _extract_vars(
        (files("lobes.templates") / "fleet" / "docker-compose.yml").read_text(encoding="utf-8"),
        (files("lobes.templates") / "fleet" / "env.example").read_text(encoding="utf-8"),
    )
    missing = main_vars - current_vars
    assert not missing, f"fleet template dropped/renamed env var(s): {sorted(missing)}"


# --- 4: init-over-existing requires --apply; no silent rewrite -------------


def test_init_over_old_single_scaffold_requires_apply(tmp_path) -> None:
    deploy = tmp_path / "deploy"
    _materialize_old_single_scaffold(deploy)
    before = _hash_tree(deploy)

    # No --apply: prints a plan, changes nothing (already covered by the
    # operability test above; re-asserted here as the acceptance-criteria-4
    # focused case).
    rc = main(["init", str(deploy), "--single"])
    assert rc == 0
    assert _hash_tree(deploy) == before

    # --apply without --force: init's documented behaviour is to REFUSE an
    # existing file (lobes/runtime/_compose.py write_scaffold), not overwrite
    # it — and it checks EVERY dest file before writing ANY, so a refusal
    # never leaves a half-rewritten scaffold.
    rc = main(["init", str(deploy), "--single", "--apply"])
    assert rc == 1  # EXIT_USER_ERROR — "already exists ... --force to overwrite"
    assert _hash_tree(deploy) == before, "a refused --apply must not write anything"


def test_init_over_old_fleet_scaffold_requires_apply(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card("thor"))
    deploy = tmp_path / "deploy"
    _materialize_old_fleet_scaffold(deploy)
    before = _hash_tree(deploy)

    rc = main(["init", str(deploy)])
    assert rc == 0
    assert _hash_tree(deploy) == before

    rc = main(["init", str(deploy), "--apply"])
    assert rc == 1
    assert _hash_tree(deploy) == before, "a refused --apply must not write anything"


def test_init_over_old_scaffold_force_apply_is_the_only_write_path(tmp_path, monkeypatch) -> None:
    """Documents the escape hatch: --force --apply is the explicit, deliberate
    re-init the invariant carves out — it's opt-in, never a side effect."""
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card("thor"))
    deploy = tmp_path / "deploy"
    _materialize_old_fleet_scaffold(deploy)
    before = _hash_tree(deploy)

    rc = main(["init", str(deploy), "--apply", "--force"])
    assert rc == 0
    after = _hash_tree(deploy)
    assert after != before, "--force --apply is expected to rewrite the scaffold"


# --- fixture sanity ---------------------------------------------------------


def test_fixtures_are_nonempty_and_distinct_from_current_templates() -> None:
    """Guards against an accidental no-op vendoring (fixture == current
    template would make every test above vacuous)."""
    for kind, compose_rel, env_rel in (
        ("single", "docker-compose.yml", "env.example"),
        ("fleet", "fleet/docker-compose.yml", "fleet/env.example"),
    ):
        old_compose = (FIXTURES / kind / "docker-compose.yml").read_text(encoding="utf-8")
        old_env = (FIXTURES / kind / "env.example").read_text(encoding="utf-8")
        assert old_compose.strip()
        assert old_env.strip()
        current_compose = (files("lobes.templates") / compose_rel).read_text(encoding="utf-8")
        current_env = (files("lobes.templates") / env_rel).read_text(encoding="utf-8")
        # Not a hard requirement that they differ forever, but at vendoring
        # time (t3/t4/t6/t13 already landed on this branch) they do; a
        # same-content fixture would silently defeat the rename tripwire.
        assert (old_compose, old_env) != (current_compose, current_env), (
            f"{kind} fixture is byte-identical to the current template — "
            "re-vendor from a later main commit or this test proves nothing"
        )
