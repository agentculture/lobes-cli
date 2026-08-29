"""``lobes doctor`` ``lock_drift`` + ``lobes switch``'s lock-staleness warning
(deployment-lock-per-box plan, t8).

The committed lock (t6, ``lobes/runtime/_lock.py``) records an allowlisted
snapshot of a deployment's rendered ``.env`` knobs plus a ``[files]`` table of
verbatim-committed file digests. This module gives ``lobes doctor`` a
``lock_drift`` finding that diffs BOTH tables against what is actually on
disk, naming the SPECIFIC differing files/keys (never merely "drift exists"),
and gives ``lobes switch`` a heads-up: it writes straight into ``.env``
(``switch.py`` around ``_apply_switch``/``_apply_env_only``), a first-class,
documented, dry-run-guarded verb — not a hand edit — so it can make a
committed lock stale just as surely as one can.

Two live incidents motivate the two pieces:

* 2026-08-25 (#199 t11 prep): the Spark's ``docker-compose.yml`` was hand-
  edited with a baked ``--speculative-config`` while the Thor's happened to
  equal the packaged template — a difference only a live diff revealed. A
  committed lock plus ``lock_drift`` turns that into a mechanical check.
* the frame's own honesty condition: the drift story assumed hand edits were
  the only source of divergence, and ``lobes switch`` proves they are not —
  see ``docs/specs/2026-08-29-deployment-lock-per-box.md``'s "lock is
  invalidated by a FIRST-CLASS VERB" assumption.
"""

from __future__ import annotations

import json
import types

from lobes.cli import main
from lobes.cli._commands.init import DEFAULT_SHAPE, _apply_profile_env
from lobes.cli._runtime_ops import resolve_init_profile
from lobes.profiles.shape_render import render_shape
from lobes.profiles.shapes import resolve_shape
from lobes.runtime import _compose, _detect, _env, _health
from lobes.runtime._lock import LOCK_FILENAME, build_lock, file_digest, lock_toml, write_lock


def _card(resolved: str = "spark", name: str = "NVIDIA GB10", cc: str = "sm_121") -> object:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name=name,
        compute_capability=cc,
        total_memory_gb=128.0,
        hostname="testbox",
        device_tree_model=None,
        sources={
            "device_name": "nvidia-smi",
            "compute_capability": "nvidia-smi",
            "total_memory_gb": "/proc/meminfo",
            "hostname": "socket.gethostname",
            "device_tree_model": "unavailable",
        },
    )


def _scaffold_fleet(path, *, profile: str = "spark"):
    """A complete fleet deployment with a profile render applied — mirrors
    ``lobes init --apply --profile <profile>``."""
    _compose.write_scaffold(path, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    _compose.write_plugin_file(path, force=True)
    profile_obj, _card_, _warn = resolve_init_profile(profile, path)
    _apply_profile_env(
        path / ".env", dict(render_shape(resolve_shape(DEFAULT_SHAPE), profile_obj).env)
    )
    _env.set_env(path / ".env", "LOBES_PROFILE", profile)
    return path


def _tracked_files(deploy_dir, names: tuple[str, ...] = ("docker-compose.yml", ".env")) -> dict:
    """A plausible lock ``[files]`` table: compose file(s) plus ``.env``'s own
    digest. Tracking ``.env``'s digest is legitimate under t6's generic
    ``name -> "sha256:<hex>"`` mapping — it is a HASH, never the file's
    content, so it never enters the secret-free contract's crosshairs, and it
    is exactly what lets ``lock_drift`` notice a plain ``.env`` write (the
    kind ``lobes switch`` makes) the same way it notices a hand-edited
    compose file."""
    return {name: file_digest(deploy_dir / name) for name in names if (deploy_dir / name).is_file()}


def _capture_and_write_lock(deploy_dir, *, variation: str = "spark-test", profile: str = "spark"):
    env = _env.read_env_file(deploy_dir / ".env")
    lock = build_lock(
        variation=variation,
        env=env,
        profile=profile,
        shape=DEFAULT_SHAPE,
        files=_tracked_files(deploy_dir),
    )
    write_lock(deploy_dir, lock)
    return lock


def _doctor_json(capsys, *args: str) -> dict:
    main(["doctor", "--json", *args])
    return json.loads(capsys.readouterr().out)


def _find(checks: list[dict], id_: str) -> dict | None:
    return next((c for c in checks if c["id"] == id_), None)


# --- absence of a lock is not drift -----------------------------------------


def test_no_lock_present_emits_no_finding_at_all(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    assert not (tmp_path / LOCK_FILENAME).exists()
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card())

    payload = _doctor_json(capsys)
    assert _find(payload["checks"], "lock_drift") is None


# --- the passing case ---------------------------------------------------


def test_lock_drift_passes_when_nothing_diverges(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _capture_and_write_lock(tmp_path)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card())

    payload = _doctor_json(capsys)
    check = _find(payload["checks"], "lock_drift")
    assert check is not None
    assert check["passed"] is True
    assert check["severity"] == "info"


# --- naming the SPECIFIC differing files/keys -------------------------------


def test_lock_drift_names_the_specific_differing_file(tmp_path, monkeypatch, capsys) -> None:
    """A hand-edited compose file is named by exact filename — never a
    generic 'drift exists' — and an untouched tracked file is NOT named."""
    _scaffold_fleet(tmp_path)
    _capture_and_write_lock(tmp_path)
    # Hand-edit docker-compose.yml, mirroring the 2026-08-25 Spark incident
    # (a baked --speculative-config nobody captured back into the lock).
    compose_path = tmp_path / "docker-compose.yml"
    compose_path.write_text(
        compose_path.read_text(encoding="utf-8") + "\n# hand-edited, not re-captured\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card())

    payload = _doctor_json(capsys)
    check = _find(payload["checks"], "lock_drift")
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "docker-compose.yml" in check["message"]
    assert ".env" not in check["message"]  # tracked but untouched — not named
    assert LOCK_FILENAME in check["remediation"]


def test_lock_drift_names_the_specific_differing_locked_key(tmp_path, monkeypatch, capsys) -> None:
    """A hand-edited PRIMARY_* knob is named by exact key — an untouched
    locked key is not swept up with it."""
    _scaffold_fleet(tmp_path)
    _capture_and_write_lock(tmp_path)
    _env.set_env(tmp_path / ".env", "PRIMARY_GPU_MEM_UTIL", "0.99")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card())

    payload = _doctor_json(capsys)
    check = _find(payload["checks"], "lock_drift")
    assert check["passed"] is False
    assert "PRIMARY_GPU_MEM_UTIL" in check["message"]
    assert "PRIMARY_MAX_MODEL_LEN" not in check["message"]


def test_lock_drift_names_a_missing_tracked_file(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    lock = build_lock(
        variation="spark-test",
        env=_env.read_env_file(tmp_path / ".env"),
        profile="spark",
        shape=DEFAULT_SHAPE,
        files={
            **_tracked_files(tmp_path),
            "docker-compose.override.yml": "sha256:" + "0" * 64,
        },
    )
    write_lock(tmp_path, lock)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card())

    payload = _doctor_json(capsys)
    check = _find(payload["checks"], "lock_drift")
    assert check["passed"] is False
    assert "docker-compose.override.yml (missing)" in check["message"]


# --- doctor --fix never touches the lock or an existing .env line ----------


def test_fix_apply_ignores_lock_drift_and_never_rewrites_an_existing_env_line(
    tmp_path, monkeypatch, capsys
) -> None:
    """``lock_drift`` is read-only: --fix's missing-only heal must not try to
    'fix' it, and — the pre-existing convention this must not break — an
    existing .env line is never rewritten even when a locked key differs."""
    _scaffold_fleet(tmp_path)
    _capture_and_write_lock(tmp_path)
    _env.set_env(tmp_path / ".env", "PRIMARY_GPU_MEM_UTIL", "0.99")  # now drifts from the lock
    env_before = (tmp_path / ".env").read_text(encoding="utf-8")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card())

    payload = _doctor_json(capsys, "--fix", "--apply")
    assert not any("PRIMARY_GPU_MEM_UTIL" in a for a in payload["fix_applied"])
    env_after = (tmp_path / ".env").read_text(encoding="utf-8")
    assert env_after == env_before  # untouched: the drifted value survives verbatim
    assert "PRIMARY_GPU_MEM_UTIL=0.99" in env_after

    # The report describes the AFTER state — lock_drift still fails, honestly.
    check = _find(payload["checks"], "lock_drift")
    assert check["passed"] is False


# --- switch's own lock-staleness warning ------------------------------------


def test_switch_dry_run_warns_when_a_lock_is_present(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _capture_and_write_lock(tmp_path)

    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert LOCK_FILENAME in out
    assert "re-capture" in out or "recapture" in out.lower()


def test_switch_dry_run_is_silent_about_the_lock_when_none_is_present(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    assert not (tmp_path / LOCK_FILENAME).exists()

    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert LOCK_FILENAME not in out


def test_switch_dry_run_still_leaves_the_env_untouched(tmp_path, capsys) -> None:
    """The lock warning must not change dry-run's core promise: nothing is written."""
    _scaffold_fleet(tmp_path)
    lock = _capture_and_write_lock(tmp_path)
    env_before = (tmp_path / ".env").read_text(encoding="utf-8")

    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / ".env").read_text(encoding="utf-8") == env_before
    assert (tmp_path / LOCK_FILENAME).read_text(encoding="utf-8") == lock_toml(lock)


# --- the end-to-end story: switch --apply -> doctor sees lock_drift --------


def test_switch_apply_makes_the_env_tracked_lock_stale_and_doctor_reports_it(
    tmp_path, monkeypatch, capsys
) -> None:
    """The honesty condition this task exists to close: running a normal,
    documented, dry-run-guarded verb — not a hand edit — leaves a committed
    lock describing a box that no longer exists. ``switch --apply`` writes
    .env (``VLLM_MODEL`` et al); the lock here tracks .env's own digest (a
    hash, never its content — see ``_tracked_files``), so the write is
    exactly what ``lock_drift`` catches, naming ``.env`` specifically."""
    _scaffold_fleet(tmp_path)
    _capture_and_write_lock(tmp_path)

    def _ok() -> types.SimpleNamespace:
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_compose, "compose_down", lambda d: _ok())
    monkeypatch.setattr(_compose, "compose_up_detached", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)

    rc = main(["switch", "foo/bar", "--compose-dir", str(tmp_path), "--apply", "--no-probe"])
    assert rc == 0
    err = capsys.readouterr().err
    assert LOCK_FILENAME in err  # switch itself warned before/while writing .env

    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card())
    payload = _doctor_json(capsys, "--compose-dir", str(tmp_path))
    check = _find(payload["checks"], "lock_drift")
    assert check is not None
    assert check["passed"] is False
    assert ".env" in check["message"]
