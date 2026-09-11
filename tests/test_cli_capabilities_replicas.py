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

Retired (t14): a declared-pool CASE (``PRIMARY_PEER_ORIGINS`` in a
deployment's ``.env``) can no longer be exercised here — the CLI's offline
capabilities render resolves entirely through
:func:`lobes.gateway._config.build_config`, which no longer reads that env
var at all, and the CLI path has no injection seam for a directly-constructed
``RoutingTable`` (unlike the gateway-server unit tests, which call
``_config``/``_routing`` functions directly). The four tests that asserted a
declared pool's positive rendering are deleted rather than kept as dead
assertions; every remaining test here — the no-pool-declared cases, and the
"the flag itself, not a stray env var, is what changes the table" claim —
stays meaningful and is unaffected by the retirement.
"""

from __future__ import annotations

import json

from lobes.cli import main
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


def test_capabilities_table_replicas_flag_with_no_pool_says_so(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--replicas"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no replica set declared for any role)" in out


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
