"""``lobes doctor`` lane-safety findings + the targeted wrapper refresh (t16).

Contract: ``docs/contracts/lane-safety-runtime.md``. Every finding must fire
when its condition holds and skip cleanly (``None``, never an error) when it
does not apply. The refresh writes exactly two paths — the backup and the
wrapper — and is its own named flag, never part of the missing-only ``--fix``
(the #174 lesson: the heal lane is append-only, full stop).
"""

from __future__ import annotations

import fcntl
import json
import os
from importlib.resources import files
from pathlib import Path

import pytest

from lobes.cli import main
from lobes.cli._commands import _doctor_lane_safety as ls
from lobes.cli._commands import doctor
from lobes.runtime import _compose, _detect, _env, _machine, _window
from lobes.watcher import _install

_VLLM_ONLY_COMPOSE = """\
services:
  vllm-primary:
    image: vllm/vllm-openai
    environment:
      - MG_LOG_NAME=primary
    volumes:
      - ./mg-logwrap.sh:/usr/local/bin/mg-logwrap:ro
    command: >-
      vllm serve some/model --port=8000
  vllm-embed:
    image: vllm/vllm-openai
    environment:
      - MG_LOG_NAME=embed
      - VLLM_CACHE_ROOT=${LOBES_VLLM_CACHE_ROOT:-/cache/vllm}
      - FLASHINFER_WORKSPACE_BASE=${LOBES_FLASHINFER_WORKSPACE_BASE:-/cache/flashinfer}
    volumes:
      - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/embed:/cache/vllm
      - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/embed-flashinfer:/cache/flashinfer
    command:
      - vllm
      - serve
      - Qwen/Qwen3-Embedding-0.6B
  gateway:
    image: lobes-gateway
    # a comment mentioning vllm serve must not count
    command: ["python3", "-m", "lobes.gateway"]
"""


def _packaged_logwrap() -> str:
    return _compose._read_template(files("lobes.templates"), _compose.LOG_WRAPPER)


def _card(resolved: str) -> _detect.DetectedCard:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name=None,
        compute_capability=None,
        total_memory_gb=None,
        hostname=None,
        device_tree_model=None,
        sources={},
    )


def _pin_card(monkeypatch, resolved: str) -> None:
    monkeypatch.setattr(_detect, "detect_card", lambda *a, **k: _card(resolved))


def _fleet(tmp_path: Path) -> Path:
    d = tmp_path / "deploy"
    _compose.write_scaffold(d, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    _compose.write_plugin_file(d, force=True)
    return d


def _bare(tmp_path: Path, compose: str = _VLLM_ONLY_COMPOSE) -> Path:
    d = tmp_path / "bare"
    d.mkdir()
    (d / _compose.COMPOSE_FILE).write_text(compose, encoding="utf-8")
    (d / _compose.ENV_FILE).write_text("VLLM_PORT=8000\n", encoding="utf-8")
    return d


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


# --- logwrap_stale ----------------------------------------------------------


def test_logwrap_stale_fires_on_a_hand_edited_wrapper(tmp_path: Path) -> None:
    d = _bare(tmp_path)
    (d / _compose.LOG_WRAPPER).write_text('#!/bin/bash\nexec "$@"\n', encoding="utf-8")
    check = ls.logwrap_stale_check(d)
    assert check["id"] == "logwrap_stale"
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "--refresh-logwrap" in check["remediation"]


def test_logwrap_stale_passes_on_the_packaged_wrapper(tmp_path: Path) -> None:
    d = _bare(tmp_path)
    (d / _compose.LOG_WRAPPER).write_text(_packaged_logwrap(), encoding="utf-8")
    check = ls.logwrap_stale_check(d)
    assert check["passed"] is True


def test_logwrap_stale_skips_when_the_wrapper_is_absent(tmp_path: Path) -> None:
    assert ls.logwrap_stale_check(_bare(tmp_path)) is None


# --- compile_cache_unmounted ------------------------------------------------


def test_compile_cache_unmounted_names_the_unmounted_vllm_service(tmp_path: Path) -> None:
    check = ls.compile_cache_unmounted_check(_bare(tmp_path))
    assert check["id"] == "compile_cache_unmounted"
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "vllm-primary" in check["message"]
    assert "vllm-embed" not in check["message"]
    assert "gateway" not in check["message"]
    rem = check["remediation"]
    assert "docker-compose.override.yml" in rem
    assert "${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/primary:/cache/vllm" in rem
    assert "VLLM_CACHE_ROOT=${LOBES_VLLM_CACHE_ROOT:-/cache/vllm}" in rem
    assert "${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/primary-flashinfer:/cache/flashinfer" in rem
    assert "FLASHINFER_WORKSPACE_BASE=${LOBES_FLASHINFER_WORKSPACE_BASE:-/cache/flashinfer}" in rem
    assert "--force" not in rem


def test_compile_cache_unmounted_fires_on_mount_without_env(tmp_path: Path) -> None:
    compose = _VLLM_ONLY_COMPOSE.replace(
        "      - VLLM_CACHE_ROOT=${LOBES_VLLM_CACHE_ROOT:-/cache/vllm}\n", ""
    )
    check = ls.compile_cache_unmounted_check(_bare(tmp_path, compose))
    assert check["passed"] is False
    assert "vllm-embed" in check["message"]


def test_compile_cache_unmounted_honours_an_override_file(tmp_path: Path) -> None:
    d = _bare(tmp_path)
    (d / _compose.LOCAL_OVERRIDE).write_text(
        "services:\n"
        "  vllm-primary:\n"
        "    environment:\n"
        "      - VLLM_CACHE_ROOT=${LOBES_VLLM_CACHE_ROOT:-/cache/vllm}\n"
        "      - FLASHINFER_WORKSPACE_BASE=${LOBES_FLASHINFER_WORKSPACE_BASE:-/cache/flashinfer}\n"
        "    volumes:\n"
        "      - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/primary:/cache/vllm\n"
        "      - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/primary-flashinfer"
        ":/cache/flashinfer\n",
        encoding="utf-8",
    )
    assert ls.compile_cache_unmounted_check(d)["passed"] is True


def test_compile_cache_unmounted_fires_on_vllm_cache_without_flashinfer(tmp_path: Path) -> None:
    """d7: a lane with t8's /cache/vllm bind but no FlashInfer bind is flagged."""
    compose = _VLLM_ONLY_COMPOSE.replace(
        "      - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/embed-flashinfer:/cache/flashinfer\n",
        "",
    )
    check = ls.compile_cache_unmounted_check(_bare(tmp_path, compose))
    assert check["passed"] is False
    assert "vllm-embed (no FlashInfer mount)" in check["message"]
    rem = check["remediation"]
    assert "/embed-flashinfer:/cache/flashinfer" in rem
    assert "/embed:/cache/vllm" in rem


def test_compile_cache_unmounted_fires_on_flashinfer_mount_without_env(tmp_path: Path) -> None:
    compose = _VLLM_ONLY_COMPOSE.replace(
        "      - FLASHINFER_WORKSPACE_BASE=${LOBES_FLASHINFER_WORKSPACE_BASE:-/cache/flashinfer}\n",
        "",
    )
    check = ls.compile_cache_unmounted_check(_bare(tmp_path, compose))
    assert check["passed"] is False
    assert "vllm-embed (no FLASHINFER_WORKSPACE_BASE)" in check["message"]


def test_compile_cache_unmounted_names_t8_gap_first(tmp_path: Path) -> None:
    """A lane lacking everything reports the vLLM cache gap, the original finding."""
    check = ls.compile_cache_unmounted_check(_bare(tmp_path))
    assert "vllm-primary (no mount)" in check["message"]


def test_compile_cache_unmounted_passes_on_a_fresh_fleet_scaffold(tmp_path: Path) -> None:
    check = ls.compile_cache_unmounted_check(_fleet(tmp_path))
    assert check["passed"] is True, check["message"]


def test_compile_cache_unmounted_skips_without_vllm_services(tmp_path: Path) -> None:
    d = _bare(tmp_path, "services:\n  gateway:\n    image: x\n")
    assert ls.compile_cache_unmounted_check(d) is None


def test_compile_cache_unmounted_ignores_parked_services(tmp_path: Path) -> None:
    d = _bare(tmp_path)
    (d / _compose.SHAPE_OVERLAY).write_text(
        'services:\n  vllm-primary:\n    profiles: ["shape-dropped"]\n', encoding="utf-8"
    )
    assert _compose.shape_parked_service_keys(
        (d / _compose.SHAPE_OVERLAY).read_text(encoding="utf-8")
    ) == {"vllm-primary"}
    assert ls.compile_cache_unmounted_check(d)["passed"] is True


# --- compile_cache_size -----------------------------------------------------


def test_compile_cache_size_reports_total_bytes(tmp_path: Path) -> None:
    d = _bare(tmp_path)
    lane = d / "compile-cache" / "primary"
    lane.mkdir(parents=True)
    (lane / "a.bin").write_bytes(b"x" * 100)
    (lane / "b.bin").write_bytes(b"y" * 23)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"z" * 5000)
    os.symlink(outside, lane / "link")  # never followed
    check = ls.compile_cache_size_check(d)
    assert check["id"] == "compile_cache_size"
    assert check["severity"] == "info"
    assert check["passed"] is True
    assert "123 bytes" in check["message"]


def test_compile_cache_size_reads_the_env_knob(tmp_path: Path) -> None:
    d = _bare(tmp_path)
    cache = tmp_path / "elsewhere"
    cache.mkdir()
    (cache / "f").write_bytes(b"q" * 7)
    _env.set_env(d / _compose.ENV_FILE, "LOBES_COMPILE_CACHE_DIR", str(cache))
    check = ls.compile_cache_size_check(d)
    assert "7 bytes" in check["message"]
    assert str(cache) in check["message"]


def test_compile_cache_size_skips_when_absent(tmp_path: Path) -> None:
    assert ls.compile_cache_size_check(_bare(tmp_path)) is None


# --- jetson_clocks_locked ---------------------------------------------------


def _show(monkeypatch, result: _machine.MachineResult) -> None:
    monkeypatch.setattr(_machine, "window", lambda action: result)


def test_clocks_locked_fires_outside_a_benchmark_window(monkeypatch) -> None:
    _show(monkeypatch, _machine.MachineResult(True, None, {"active": True}, "machine-cli"))
    check = ls.jetson_clocks_locked_check()
    assert check["id"] == "jetson_clocks_locked"
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "window restore" in check["remediation"]


def test_clocks_locked_passes_when_not_active(monkeypatch) -> None:
    _show(monkeypatch, _machine.MachineResult(True, None, {"active": False}, "machine-cli"))
    assert ls.jetson_clocks_locked_check()["passed"] is True


def test_clocks_locked_skips_when_the_tool_is_missing(monkeypatch) -> None:
    _show(monkeypatch, _machine.MachineResult(False, "tool_missing"))
    assert ls.jetson_clocks_locked_check() is None


def test_clocks_locked_does_not_warn_during_a_benchmark_window(monkeypatch) -> None:
    _show(monkeypatch, _machine.MachineResult(True, None, {"active": True}, "machine-cli"))
    fd = os.open(_window.lock_path(), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check = ls.jetson_clocks_locked_check()
    finally:
        os.close(fd)
    assert check["passed"] is True
    assert "benchmark window" in check["message"]


# clocks_locked (contract addition 2026-10-17): the CLI reads the devfreq
# min == max state directly, so a lock made OUTSIDE any lobes window (operator
# jetson_clocks, a crashed run) is visible even though `active` is false. The
# Orin on 2026-10-17 read gpu cur=min=max=1300500000 with `active: false`.


def _held_window():
    fd = os.open(_window.lock_path(), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_clocks_locked_outside_any_window_warns_with_jetson_clocks_restore(monkeypatch) -> None:
    payload = {"active": False, "stored": False, "clocks_locked": True}
    _show(monkeypatch, _machine.MachineResult(True, None, payload, "orin"))
    check = ls.jetson_clocks_locked_check()
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "outside any lobes window" in check["message"]
    assert "operator jetson_clocks or a crashed run" in check["message"]
    assert "on purpose" in check["message"]
    assert "sudo jetson_clocks --restore" in check["remediation"]
    assert "reboot" in check["remediation"]
    assert "window restore" not in check["remediation"]


def test_clocks_locked_with_a_stored_window_points_at_window_restore(monkeypatch) -> None:
    payload = {"active": True, "stored": True, "clocks_locked": True}
    _show(monkeypatch, _machine.MachineResult(True, None, payload, "orin"))
    check = ls.jetson_clocks_locked_check()
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "locked by a lobes window" in check["message"]
    assert "window restore" in check["remediation"]
    assert "jetson_clocks --restore" not in check["remediation"]


def test_clocks_locked_during_a_benchmark_window_is_info(monkeypatch) -> None:
    payload = {"active": True, "stored": True, "clocks_locked": True}
    _show(monkeypatch, _machine.MachineResult(True, None, payload, "orin"))
    fd = _held_window()
    try:
        check = ls.jetson_clocks_locked_check()
    finally:
        os.close(fd)
    assert check["passed"] is True
    assert check["severity"] == "info"
    assert "benchmark window" in check["message"]


def test_clocks_not_locked_passes_even_if_active_is_stale(monkeypatch) -> None:
    payload = {"active": True, "stored": True, "clocks_locked": False}
    _show(monkeypatch, _machine.MachineResult(True, None, payload, "orin"))
    check = ls.jetson_clocks_locked_check()
    assert check["passed"] is True
    assert check["severity"] == "info"


@pytest.mark.parametrize("payload", [{"active": True, "clocks_locked": None}, {"active": True}])
def test_clocks_locked_null_or_missing_falls_back_to_active(monkeypatch, payload) -> None:
    _show(monkeypatch, _machine.MachineResult(True, None, payload, "spark"))
    check = ls.jetson_clocks_locked_check()
    assert check["passed"] is False  # active and no benchmark lock: the old rule
    assert "window restore" in check["remediation"]


@pytest.mark.parametrize("payload", [{"active": False, "clocks_locked": None}, {"active": False}])
def test_clocks_locked_null_or_missing_and_inactive_passes(monkeypatch, payload) -> None:
    _show(monkeypatch, _machine.MachineResult(True, None, payload, "spark"))
    assert ls.jetson_clocks_locked_check()["passed"] is True


# --- drop_caches_grant_missing ----------------------------------------------


def test_grant_missing_fires_on_a_known_card(monkeypatch) -> None:
    _pin_card(monkeypatch, "thor")
    monkeypatch.setattr(_machine, "grant_status", lambda: _machine.GrantStatus(False))
    monkeypatch.setattr(_machine, "machine_cli_for", lambda card: "thor")
    check = ls.drop_caches_grant_missing_check()
    assert check["id"] == "drop_caches_grant_missing"
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "thor grant-permissions" in check["remediation"]


def test_grant_missing_without_the_cli_names_the_manual_rule(monkeypatch) -> None:
    _pin_card(monkeypatch, "orin")
    monkeypatch.setattr(_machine, "grant_status", lambda: _machine.GrantStatus(False))
    monkeypatch.setattr(_machine, "machine_cli_for", lambda card: None)
    check = ls.drop_caches_grant_missing_check()
    assert check["passed"] is False
    assert "lobes-cli[orin]" in check["remediation"]
    assert "vm.drop_caches=3" in check["remediation"]


def test_grant_missing_passes_when_granted(monkeypatch) -> None:
    _pin_card(monkeypatch, "spark")
    monkeypatch.setattr(_machine, "grant_status", lambda: _machine.GrantStatus(True, "sysctl"))
    assert ls.drop_caches_grant_missing_check()["passed"] is True


def test_grant_missing_skips_on_an_unknown_card(monkeypatch) -> None:
    def _boom():  # pragma: no cover - must never run
        raise AssertionError("grant_status probed on an unknown card")

    monkeypatch.setattr(_machine, "grant_status", _boom)
    assert ls.drop_caches_grant_missing_check() is None


# --- watcher_inactive -------------------------------------------------------


def _manifest(d: Path, unit_type: str = "user") -> None:
    (d / _install.MANIFEST).write_text(
        json.dumps({"unit": _install.UNIT_NAME, "unit_type": unit_type, "files": []}),
        encoding="utf-8",
    )


def test_watcher_inactive_fires_when_installed_but_not_active(tmp_path, monkeypatch) -> None:
    d = _bare(tmp_path)
    _manifest(d)
    monkeypatch.setattr(ls, "_watcher_active_state", lambda system: "inactive")
    check = ls.watcher_inactive_check(d)
    assert check["id"] == "watcher_inactive"
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "systemctl --user" in check["remediation"]


def test_watcher_inactive_system_unit_remediation(tmp_path, monkeypatch) -> None:
    d = _bare(tmp_path)
    _manifest(d, "system")
    monkeypatch.setattr(ls, "_watcher_active_state", lambda system: "failed" if system else "x")
    check = ls.watcher_inactive_check(d)
    assert check["passed"] is False
    assert "sudo systemctl" in check["remediation"]


def test_watcher_inactive_passes_when_active(tmp_path, monkeypatch) -> None:
    d = _bare(tmp_path)
    _manifest(d)
    monkeypatch.setattr(ls, "_watcher_active_state", lambda system: "active")
    assert ls.watcher_inactive_check(d)["passed"] is True


def test_watcher_not_installed_is_informational(tmp_path, monkeypatch) -> None:
    def _boom(system):  # pragma: no cover - must never run
        raise AssertionError("systemctl probed without a manifest")

    monkeypatch.setattr(ls, "_watcher_active_state", _boom)
    check = ls.watcher_inactive_check(_bare(tmp_path))
    assert check["passed"] is True
    assert check["severity"] == "info"
    assert "optional" in check["message"]


# --- registration -----------------------------------------------------------


def test_doctor_registers_the_lane_safety_checks(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    (d / _compose.LOG_WRAPPER).write_text("stale\n", encoding="utf-8")
    main(["doctor", "--json", "--compose-dir", str(d)])
    ids = {c["id"]: c for c in json.loads(capsys.readouterr().out)["checks"]}
    assert ids["logwrap_stale"]["passed"] is False
    assert "compile_cache_unmounted" in ids
    assert "watcher_inactive" in ids
    # unknown card (conftest): the machine findings skip cleanly
    assert "drop_caches_grant_missing" not in ids
    assert "jetson_clocks_locked" not in ids


def test_lane_safety_findings_never_make_doctor_unhealthy(tmp_path) -> None:
    d = _bare(tmp_path)
    (d / _compose.LOG_WRAPPER).write_text("stale\n", encoding="utf-8")
    checks = ls.lane_safety_checks(d)
    assert checks
    assert all(c["severity"] in ("warn", "info") for c in checks)


# --- the targeted refresh ---------------------------------------------------


def test_refresh_apply_writes_exactly_backup_and_wrapper(tmp_path: Path) -> None:
    d = _fleet(tmp_path)
    old = '#!/bin/bash\n# hand-edited\nexec "$@"\n'
    (d / _compose.LOG_WRAPPER).write_text(old, encoding="utf-8")
    before = _snapshot(d)
    actions = ls.apply_refresh_logwrap(d)
    after = _snapshot(d)
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed == {"mg-logwrap.sh", "mg-logwrap.sh.bak"}
    assert (d / "mg-logwrap.sh.bak").read_text(encoding="utf-8") == old
    assert (d / "mg-logwrap.sh").read_text(encoding="utf-8") == _packaged_logwrap()
    assert actions


def test_refresh_is_a_noop_when_current(tmp_path: Path) -> None:
    d = _fleet(tmp_path)
    before = _snapshot(d)
    assert ls.apply_refresh_logwrap(d) == []
    assert _snapshot(d) == before


def test_refresh_writes_nothing_when_the_wrapper_is_absent(tmp_path: Path) -> None:
    d = _fleet(tmp_path)
    (d / _compose.LOG_WRAPPER).unlink()
    before = _snapshot(d)
    assert ls.apply_refresh_logwrap(d) == []
    assert _snapshot(d) == before
    assert "--fix" in " ".join(ls.refresh_logwrap_plan(d))


def test_refresh_cli_dry_run_writes_nothing(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    (d / _compose.LOG_WRAPPER).write_text("stale\n", encoding="utf-8")
    before = _snapshot(d)
    main(["doctor", "--refresh-logwrap", "--compose-dir", str(d)])
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "mg-logwrap.sh.bak" in out
    assert _snapshot(d) == before


def test_refresh_cli_apply(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    (d / _compose.LOG_WRAPPER).write_text("stale\n", encoding="utf-8")
    main(["doctor", "--refresh-logwrap", "--apply", "--json", "--compose-dir", str(d)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["refresh_logwrap_applied"]
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["logwrap_stale"]["passed"] is True
    assert (d / "mg-logwrap.sh.bak").read_text(encoding="utf-8") == "stale\n"


def test_plain_fix_apply_never_refreshes_a_stale_wrapper(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    (d / _compose.LOG_WRAPPER).write_text("stale\n", encoding="utf-8")
    main(["doctor", "--fix", "--apply", "--json", "--compose-dir", str(d)])
    capsys.readouterr()
    assert (d / _compose.LOG_WRAPPER).read_text(encoding="utf-8") == "stale\n"
    assert not (d / "mg-logwrap.sh.bak").exists()


def test_refresh_refuses_with_role(tmp_path) -> None:
    from lobes.cli._errors import ModelGearError

    with pytest.raises(ModelGearError):
        doctor._validate_doctor_flags(
            "cortex", fix=False, apply=False, repin=False, refresh_logwrap=True
        )


# --- compile-cache dir creation ---------------------------------------------


def test_ensure_compile_cache_dirs_creates_every_lane(tmp_path: Path) -> None:
    d = _fleet(tmp_path)
    created = _compose.ensure_compile_cache_dirs(d)
    assert (d / "compile-cache").is_dir()
    assert (d / "compile-cache" / "primary").is_dir()
    assert (d / "compile-cache" / "embed").is_dir()
    assert (d / "compile-cache" / "primary-flashinfer").is_dir()
    assert (d / "compile-cache" / "embed-flashinfer").is_dir()
    assert d / "compile-cache" / "primary" in created
    assert d / "compile-cache" / "primary-flashinfer" in created
    assert _compose.ensure_compile_cache_dirs(d) == []  # idempotent


def test_ensure_compile_cache_dirs_dry_run_creates_nothing(tmp_path: Path) -> None:
    d = _fleet(tmp_path)
    planned = _compose.ensure_compile_cache_dirs(d, dry_run=True)
    assert d / "compile-cache" / "primary" in planned
    assert d / "compile-cache" / "primary-flashinfer" in planned
    assert not (d / "compile-cache").exists()


def test_compile_cache_lanes_parses_both_mounts(tmp_path: Path) -> None:
    """d7: the parser picks up ``<lane>`` (/cache/vllm) and ``<lane>-flashinfer``
    (/cache/flashinfer) binds, and nothing bound elsewhere."""
    d = _bare(tmp_path)
    (d / _compose.LOCAL_OVERRIDE).write_text(
        "services:\n"
        "  x:\n"
        "    volumes:\n"
        "      - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/odd:/somewhere/else\n"
        "      # - ${LOBES_COMPILE_CACHE_DIR:-./compile-cache}/gone-flashinfer:/cache/flashinfer\n",
        encoding="utf-8",
    )
    assert _compose.compile_cache_lanes(d) == ["embed", "embed-flashinfer"]


def test_ensure_compile_cache_dirs_honours_the_env_knob(tmp_path: Path) -> None:
    d = _fleet(tmp_path)
    _env.set_env(d / _compose.ENV_FILE, "LOBES_COMPILE_CACHE_DIR", "./cc")
    _compose.ensure_compile_cache_dirs(d)
    assert (d / "cc" / "primary").is_dir()
    assert not (d / "compile-cache").exists()


def test_ensure_compile_cache_dirs_chowns_to_the_sudo_user(tmp_path, monkeypatch) -> None:
    d = _fleet(tmp_path)
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_GID", "5678")
    monkeypatch.setattr(os, "chown", lambda p, u, g: calls.append((str(p), u, g)))
    created = _compose.ensure_compile_cache_dirs(d)
    assert created
    assert {(u, g) for _, u, g in calls} == {(1234, 5678)}
    assert str(d / "compile-cache") in {p for p, _, _ in calls}


def test_ensure_compile_cache_dirs_never_chowns_as_a_user(tmp_path, monkeypatch) -> None:
    d = _fleet(tmp_path)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(os, "chown", lambda *a: pytest.fail("chown as non-root"))
    assert _compose.ensure_compile_cache_dirs(d)


def test_init_apply_creates_the_compile_cache(tmp_path, capsys) -> None:
    target = tmp_path / "fresh"
    main(["init", str(target), "--apply"])
    capsys.readouterr()
    assert (target / "compile-cache" / "primary").is_dir()


def test_init_single_creates_the_compile_cache(tmp_path, capsys) -> None:
    target = tmp_path / "single"
    main(["init", str(target), "--single", "--apply"])
    capsys.readouterr()
    assert (target / "compile-cache" / "vllm").is_dir()


def test_init_dry_run_creates_no_compile_cache(tmp_path, capsys) -> None:
    target = tmp_path / "dry"
    main(["init", str(target)])
    capsys.readouterr()
    assert not (target / "compile-cache").exists()


def test_doctor_fix_plans_then_creates_missing_cache_dirs(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    main(["doctor", "--fix", "--json", "--compose-dir", str(d)])
    payload = json.loads(capsys.readouterr().out)
    assert any(p.endswith("compile-cache/primary") for p in payload["fix_plan"]["dirs"])
    assert not (d / "compile-cache").exists()
    main(["doctor", "--fix", "--apply", "--json", "--compose-dir", str(d)])
    payload = json.loads(capsys.readouterr().out)
    assert (d / "compile-cache" / "primary").is_dir()
    assert any("compile-cache/primary" in a for a in payload["fix_applied"])
    assert payload["fix_plan"]["dirs"] == []
