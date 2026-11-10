"""This box's mesh name: running gateway over .env, never a roster peer."""

import subprocess

import pytest

from lobes.rollout import _identity as ident

# The autouse conftest fixture stubs the module attribute; keep the real one.
REAL = ident.running_gateway_mesh_name


def _run(stdout="", rc=0):
    def run(argv, **kw):
        assert argv[:3] == ["docker", "inspect", ident.GATEWAY_CONTAINER]
        return subprocess.CompletedProcess(argv, rc, stdout, "")

    return run


def test_reads_only_the_mesh_name_from_the_container_env():
    out = "PATH=/usr/bin\nGATEWAY_API_KEY=sk-secret\nLOBES_MESH_NAME=core\n"
    assert REAL(run=_run(out)) == "core"


def test_no_container_or_no_key_is_none():
    assert REAL(run=_run("", rc=1)) is None
    assert REAL(run=_run("PATH=/x\n")) is None


def test_docker_missing_is_none():
    def run(*a, **k):
        raise OSError("no docker")

    assert REAL(run=run) is None


def test_running_gateway_wins_over_a_stale_env_and_warns():
    # 2026-11-10: core's .env said spark (another member's name)
    name, warning = ident.resolve_local_name("spark", ["spark", "thor"], inspect=lambda: "core")
    assert name == "core"
    assert "spark" in warning and "core" in warning


def test_env_used_when_no_container():
    assert ident.resolve_local_name("core", ["spark"], inspect=lambda: None) == ("core", None)


def test_a_roster_peer_is_never_this_box():
    with pytest.raises(ident.IdentityError, match="peer"):
        ident.resolve_local_name("spark", ["spark", "thor"], inspect=lambda: None)


def test_explicit_local_still_refused_when_it_is_a_peer():
    assert (
        ident.resolve_local_name("x", ["spark"], explicit="core", inspect=lambda: "y")[0] == "core"
    )
    with pytest.raises(ident.IdentityError):
        ident.resolve_local_name(None, ["spark"], explicit="spark", inspect=lambda: None)
