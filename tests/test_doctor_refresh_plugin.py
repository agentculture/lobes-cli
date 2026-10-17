"""``lobes doctor --refresh-plugin`` and the ``plugin_stale`` finding (vllm-0-31 t2).

Mirrors the ``--refresh-logwrap`` lane: its own named flag (never part of the
missing-only ``--fix``), dry-run by default, and ``--apply`` writes exactly two
paths — the ``.bak`` of the current copy, then the packaged plugin.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lobes.cli import main
from lobes.cli._commands import _doctor_plugin as dp
from lobes.cli._commands import doctor
from lobes.cli._errors import ModelGearError
from lobes.runtime import _compose

PLUGIN = _compose.PLUGIN_DEST_NAME
BAK = PLUGIN + ".bak"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fleet(tmp_path: Path) -> Path:
    d = tmp_path / "deploy"
    _compose.write_scaffold(d, force=True, templates=dict(_compose.FLEET_TEMPLATES))
    _compose.write_plugin_file(d, force=True)
    return d


def _stale(d: Path) -> None:
    (d / PLUGIN).write_text("# stale plugin\n", encoding="utf-8")


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_dry_run_plan_names_path_and_both_hashes(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    _stale(d)
    before = _snapshot(d)
    main(["doctor", "--refresh-plugin", "--compose-dir", str(d)])
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert str(d / PLUGIN) in out
    assert _sha("# stale plugin\n") in out
    assert _sha(_compose.plugin_source()) in out
    assert BAK in out
    assert _snapshot(d) == before


def test_apply_writes_exactly_backup_and_packaged_copy(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    _stale(d)
    before = _snapshot(d)
    main(["doctor", "--refresh-plugin", "--apply", "--json", "--compose-dir", str(d)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["refresh_plugin_applied"]
    after = _snapshot(d)
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed == {PLUGIN, BAK}
    assert (d / BAK).read_text(encoding="utf-8") == "# stale plugin\n"
    assert (d / PLUGIN).read_text(encoding="utf-8") == _compose.plugin_source()
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["plugin_stale"]["passed"] is True


def test_up_to_date_reports_and_writes_nothing(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    before = _snapshot(d)
    main(["doctor", "--refresh-plugin", "--compose-dir", str(d)])
    assert "up to date" in capsys.readouterr().out
    main(["doctor", "--refresh-plugin", "--apply", "--compose-dir", str(d)])
    assert "up to date" in capsys.readouterr().out
    assert _snapshot(d) == before
    assert dp.apply_refresh_plugin(d) == []


def test_absent_plugin_is_not_refreshed(tmp_path) -> None:
    d = _fleet(tmp_path)
    (d / PLUGIN).unlink()
    assert dp.apply_refresh_plugin(d) == []
    assert "--fix" in " ".join(dp.refresh_plugin_plan(d))
    assert not (d / PLUGIN).exists()
    assert dp.plugin_stale_check(d) is None


def test_finding_flags_stale_copy_with_remediation(tmp_path) -> None:
    d = _fleet(tmp_path)
    _stale(d)
    check = dp.plugin_stale_check(d)
    assert check["id"] == "plugin_stale"
    assert check["passed"] is False
    assert check["severity"] == "warn"
    assert "--refresh-plugin" in check["remediation"]


def test_finding_passes_when_current(tmp_path) -> None:
    check = dp.plugin_stale_check(_fleet(tmp_path))
    assert check["passed"] is True


def test_doctor_report_carries_the_finding(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    _stale(d)
    main(["doctor", "--json", "--compose-dir", str(d)])
    payload = json.loads(capsys.readouterr().out)
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["plugin_stale"]["passed"] is False


def test_plain_fix_apply_never_refreshes_a_stale_plugin(tmp_path, capsys) -> None:
    d = _fleet(tmp_path)
    _stale(d)
    main(["doctor", "--fix", "--apply", "--json", "--compose-dir", str(d)])
    capsys.readouterr()
    assert (d / PLUGIN).read_text(encoding="utf-8") == "# stale plugin\n"
    assert not (d / BAK).exists()


def test_refuses_with_role() -> None:
    with pytest.raises(ModelGearError):
        doctor._validate_doctor_flags(
            "cortex", fix=False, apply=False, repin=False, refresh_plugin=True
        )


def test_apply_alone_message_names_the_flag() -> None:
    with pytest.raises(ModelGearError) as exc:
        doctor._validate_doctor_flags(None, fix=False, apply=True, repin=False)
    assert "--refresh-plugin" in exc.value.message


def test_combines_with_refresh_logwrap(tmp_path, capsys) -> None:
    """Both are independent write lanes (like --repin-version + --refresh-logwrap)."""
    d = _fleet(tmp_path)
    _stale(d)
    (d / _compose.LOG_WRAPPER).write_text("stale\n", encoding="utf-8")
    main(
        [
            "doctor",
            "--refresh-plugin",
            "--refresh-logwrap",
            "--apply",
            "--json",
            "--compose-dir",
            str(d),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["refresh_plugin_applied"]
    assert payload["refresh_logwrap_applied"]
    assert (d / BAK).exists()
    assert (d / (_compose.LOG_WRAPPER + ".bak")).exists()
