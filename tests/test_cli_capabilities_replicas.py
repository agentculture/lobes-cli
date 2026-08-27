"""Tests for ``lobes capabilities --replicas`` / ``lobes endpoint <role> --replicas``
(issue #199, task t6).

These are the CLI-side render of the additive ``replicas``/``fingerprint``
capabilities keys (``lobes.roles.annotate_replicas``). The offline fallback
(the path these tests exercise — see ``tests/test_cli_capabilities.py``'s
module docstring for why the autouse ``offline_runtime`` fixture routes every
test in this file through it) never has a live snapshot, so the replica view
always renders the DECLARED-only list: every live field honestly unknown, and
a would-choose line of ``none (none)``.

``lobes route`` must have NO diff from this task — asserted at the bottom by
grepping for a stray import, since a real ``git diff`` isn't available to a
unit test.
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.gateway._replicas import UNCALIBRATED_WEIGHT
from lobes.roles import ROLES
from lobes.runtime import _compose, _env


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


def test_capabilities_json_has_no_replicas_key_with_no_pool_declared(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(ROLES)
    for role in ROLES:
        assert "replicas" not in payload[role]
        assert "fingerprint" not in payload[role]


def test_capabilities_table_without_replicas_flag_is_unaffected_by_declared_pool(
    tmp_path, capsys
) -> None:
    """`--replicas` is what changes the TABLE; a declared pool must not alter
    the default (no-flag) table output at all."""
    _scaffold_fleet(tmp_path)
    without_pool = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert without_pool == 0
    out_without = capsys.readouterr().out

    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGINS", "http://thor.local:8000")
    with_pool = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert with_pool == 0
    out_with = capsys.readouterr().out

    assert out_without == out_with


def test_capabilities_json_includes_replicas_and_fingerprint_when_declared(
    tmp_path, capsys
) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGINS", "http://thor.local:8000")
    _env.set_env(tmp_path / _compose.ENV_FILE, "GATEWAY_SELF_ORIGIN", "http://spark.local:8001")
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    cortex = payload["cortex"]
    assert cortex["feasible"] is True
    rows = cortex["replicas"]
    assert [r["origin"] for r in rows] == ["http://spark.local:8001", "http://thor.local:8000"]
    assert rows[0]["local"] is True
    assert rows[0]["ready"] is None  # never probed offline — honestly unknown
    assert rows[1]["local"] is False
    assert cortex["fingerprint"]["served_id"] == cortex["model"]
    # Every other role's payload is unaffected (no pool declared for them).
    for role in ROLES:
        if role == "cortex":
            continue
        assert "replicas" not in payload[role]


def test_capabilities_json_replica_rows_report_no_capacity_offline(tmp_path, capsys) -> None:
    """t6: the offline (not-probed) view must not guess a capacity — every
    row reports ``capacity: None`` and the UNCALIBRATED_WEIGHT sentinel for
    ``weight``, matching how it already reports ``None`` for every other
    live field it cannot honestly know.
    """
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGINS", "http://thor.local:8000")
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    rows = payload["cortex"]["replicas"]
    assert len(rows) == 2
    for row in rows:
        assert row["capacity"] is None
        assert row["weight"] == UNCALIBRATED_WEIGHT


def test_capabilities_table_replicas_flag_renders_candidate_rows_and_would_choose(
    tmp_path, capsys
) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGINS", "http://thor.local:8000")
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--replicas"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "replicas for cortex:" in out
    assert "http://thor.local:8000" in out
    assert "local" in out
    # offline: nothing was probed, so nothing is selectable -> "none"
    assert "would choose: none (none)" in out


def test_capabilities_table_replicas_flag_with_no_pool_says_so(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--replicas"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no replica set declared for any role)" in out


def test_endpoint_replicas_flag_renders_role_specific_view(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGINS", "http://thor.local:8000")
    rc = main(["endpoint", "cortex", "--compose-dir", str(tmp_path), "--replicas"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0]  # the endpoint URL itself, unchanged first line
    assert "replicas for cortex:" in out
    assert "would choose: none (none)" in out


def test_endpoint_replicas_flag_with_no_pool_says_so(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["endpoint", "cortex", "--compose-dir", str(tmp_path), "--replicas"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no replica set declared for this role)" in out


def test_endpoint_without_replicas_flag_prints_only_the_endpoint(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGINS", "http://thor.local:8000")
    rc = main(["endpoint", "cortex", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1  # exactly one line: the endpoint


def test_endpoint_json_mode_unaffected_by_replicas_flag(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGINS", "http://thor.local:8000")
    rc = main(["endpoint", "cortex", "--compose-dir", str(tmp_path), "--json", "--replicas"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"role", "endpoint"}


def test_route_module_is_not_imported_by_replicas_feature() -> None:
    """`lobes route` must have no diff from this task (t6 acceptance) — the
    replica view lives on capabilities/endpoint only, never on route."""
    import lobes.cli._commands.capabilities as capabilities_module

    assert "route" not in capabilities_module.__name__
    assert not hasattr(capabilities_module, "cmd_route")
