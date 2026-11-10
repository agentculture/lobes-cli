"""Tests for ``lobes mesh resident add <name>`` (gradual-fleet-rollout, t14)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lobes.cli import main
from lobes.rollout import _plan, _residents
from lobes.rollout._residents import Field, Residents
from lobes.runtime import _compose, _env

ROSTER = {
    "members": [
        {"name": "spark-f8a9", "origin": "http://100.64.0.1:8000", "verified": True},
        {"name": "thor", "origin": "http://thor.lan", "verified": True},
        {"name": "pending-box", "origin": "http://x:8000", "verified": False},
    ]
}


@pytest.fixture
def fake_roster():
    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps(ROSTER).encode()
            self.send_response(200 if self.path == "/mesh/roster" else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def deploy(tmp_path):
    _compose.write_scaffold(tmp_path, force=True)
    _env.set_env(tmp_path / _compose.ENV_FILE, "LOBES_MESH_KEY", "k")
    return tmp_path


def run(deploy, port, *extra):
    return main(
        ["mesh", "resident", "add", *extra, "--compose-dir", str(deploy), "--port", str(port)]
    )


def seed(deploy, residents=None, guests=None):
    _residents.save(Residents(residents or {}, guests or {}), deploy)


def test_dry_run_writes_nothing(deploy, fake_roster, capsys):
    rc = run(deploy, fake_roster, "spark-f8a9")
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "spark-f8a9" in out and "--apply" in out
    assert not (deploy / "residents.toml").exists()


def test_apply_adds_resident_with_needs_you_fields(deploy, fake_roster):
    assert run(deploy, fake_roster, "spark-f8a9", "--apply") == 0
    res = _residents.load(deploy)
    e = res.residents["spark-f8a9"]
    assert e["port"] == Field(8000, "detected")  # from the roster origin
    for f in ("ssh", "user", "lobes_path", "gateway_key"):
        assert e[f].state == "needs_you" and e[f].value is None  # never guessed


def test_name_not_derived_from_hostname(deploy, fake_roster):
    run(deploy, fake_roster, "thor", "--apply")
    e = _residents.load(deploy).residents["thor"]
    assert e["ssh"].state == "needs_you"  # the mesh name is not an ssh target


def test_promotes_guest_and_keeps_others(deploy, fake_roster):
    seed(
        deploy,
        residents={"thor": _residents._entry({"ssh": "t"})},
        guests={"spark-f8a9": _residents._entry(None)},
    )
    assert run(deploy, fake_roster, "spark-f8a9", "--apply") == 0
    res = _residents.load(deploy)
    assert "spark-f8a9" in res.residents and "spark-f8a9" not in res.guests
    assert res.residents["thor"]["ssh"].value == "t"


def test_json_dry_run(deploy, fake_roster, capsys):
    assert run(deploy, fake_roster, "thor", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dry_run"] is True and data["name"] == "thor"
    assert "needs_you" in data["entry"]["ssh"]["state"]


def test_unknown_name_refused(deploy, fake_roster, capsys):
    assert run(deploy, fake_roster, "nope", "--apply") == 1
    assert "not a mesh member" in capsys.readouterr().err
    assert not (deploy / "residents.toml").exists()


def test_unverified_member_refused(deploy, fake_roster, capsys):
    assert run(deploy, fake_roster, "pending-box", "--apply") == 1
    assert "verified" in capsys.readouterr().err


def test_already_resident_refused(deploy, fake_roster, capsys):
    seed(deploy, residents={"thor": _residents._entry(None)})
    before = (deploy / "residents.toml").read_text()
    assert run(deploy, fake_roster, "thor", "--apply") == 1
    assert "already a resident" in capsys.readouterr().err
    assert (deploy / "residents.toml").read_text() == before


def test_roster_unreachable_is_env_error(deploy):
    assert run(deploy, 1, "thor", "--apply") == 2


def test_rollouts_skip_guests(deploy):
    seed(deploy, guests={"thor": _residents._entry(None)})
    res = _residents.load(deploy)
    plan = _plan.build_plan(res, {}, {}, "2.0.0", "zed")
    assert plan.member("thor").status == _plan.SKIPPED_GUEST
    assert plan.waves == []
