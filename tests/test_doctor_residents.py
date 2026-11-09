from lobes.cli._commands._doctor_residents import residents_checks
from lobes.rollout import _residents as R

FULL = {"ssh": "h", "user": "u", "port": 22, "lobes_path": "/x", "gateway_key": True}


def _save(tmp_path, probes, roster_names):
    roster = {"members": [{"name": n, "verified": True} for n in roster_names]}
    R.save(R.generate(roster, probes), tmp_path)


def _env(tmp_path):
    (tmp_path / ".env").write_text("LOBES_MESH_KEY=sekret\n")


def _ids(checks):
    return sorted(c["id"] for c in checks)


def test_absent_file_no_findings(tmp_path):
    assert residents_checks(tmp_path, fetch=lambda p, k: {"members": []}) == []


def test_new_mesh_member(tmp_path):
    _save(tmp_path, {"a": FULL}, ["a"])
    _env(tmp_path)
    roster = {
        "members": [
            {"name": "a", "verified": True},
            {"name": "b", "verified": True},
            {"name": "c", "verified": False},
        ]
    }
    seen = {}

    def fetch(port, key):
        seen["key"] = key
        return roster

    out = residents_checks(tmp_path, fetch=fetch)
    assert _ids(out) == ["new_mesh_member"]
    assert "lobes mesh resident add b" in out[0]["remediation"]
    assert "sekret" not in str(out)


def test_field_missing_and_gateway_open(tmp_path):
    _save(tmp_path, {"a": {**FULL, "user": None, "gateway_key": False}}, ["a"])
    out = residents_checks(tmp_path, fetch=lambda p, k: None)
    assert _ids(out) == ["gateway_open", "resident_field_missing"]
    miss = next(c for c in out if c["id"] == "resident_field_missing")
    assert "a" in miss["message"] and "user" in miss["message"]
    assert "residents.toml" in miss["remediation"]
    assert "GATEWAY_API_KEY" in next(c for c in out if c["id"] == "gateway_open")["remediation"]


def test_unreachable_gateway_skips_member_finding(tmp_path):
    _save(tmp_path, {"a": FULL}, ["a"])
    _env(tmp_path)

    def boom(p, k):
        raise OSError

    assert residents_checks(tmp_path, fetch=boom) == []


def test_read_only(tmp_path):
    _save(tmp_path, {"a": {**FULL, "gateway_key": False}}, ["a"])
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    residents_checks(tmp_path, fetch=lambda p, k: None)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
